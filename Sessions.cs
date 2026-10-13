using System.Text.Json.Nodes;
using Microsoft.Data.Sqlite;

namespace LlmGateway;

/// <summary>
/// Chat storage, modelled as conversations containing turns.
///
/// A turn is one model invocation and carries everything needed to judge it
/// without joining anything: when it ran, which model actually served it (not
/// the alias that was requested), the user messages it introduced, its reply,
/// the token/cost accounting, and -- crucially -- `reaction`: the user's next
/// message in response to that output.
///
/// That last column is the reason for this shape. "Did the user accept or
/// swipe at this answer?" is the single most useful quality signal available,
/// and storing it next to the turn makes it a column read instead of a window
/// function over a flat message log. It is filled in when the following turn
/// arrives, so the newest turn is simply NULL until someone replies.
///
/// Writes are append-only: each turn is a new row, so an N-turn thread costs
/// O(N) rather than the O(N^2) of rewriting one JSON blob per turn. Size is
/// bounded by a rolling buffer that drops the oldest conversations.
/// </summary>
public static class Sessions
{
    static readonly object Lock = new();

    /// <summary>Serialises database work. The pruner takes the same gate to
    /// VACUUM, which needs exclusive access -- so it must wait for in-flight
    /// writers rather than racing them.</summary>
    static readonly SemaphoreSlim Gate = new(1, 1);

    /// <summary>Writes since the pruner last looked, and when the last one
    /// happened. VACUUM only runs once traffic has gone quiet, otherwise a
    /// busy gateway would be compacting continuously.</summary>
    static long _writes;
    static long _lastWriteTicks;

    static bool Debug => Environment.GetEnvironmentVariable("SESSION_DEBUG") == "1";

    /// <summary>Number of writers that have completed since the last prune.</summary>
    public static long WritesSincePrune => Interlocked.Read(ref _writes);

    /// <summary>Seconds since the last write, or long.MaxValue if none yet.</summary>
    public static double IdleSeconds =>
        Interlocked.Read(ref _lastWriteTicks) == 0 ? double.MaxValue
        : (DateTime.UtcNow - new DateTime(Interlocked.Read(ref _lastWriteTicks))).TotalSeconds;

    public static (long Conversations, long Turns, long Bytes) Size()
    {
        lock (Lock)
        {
            try
            {
                using var conn = Open();
                using var c = conn.CreateCommand();
                c.CommandText = "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()";
                return (0, 0, Convert.ToInt64(c.ExecuteScalar()));
            }
            catch { return (0, 0, 0); }
        }
    }

    public sealed record Usage(long Prompt, long Completion, long Reasoning, long Cached);

    static SqliteConnection Open()
    {
        var conn = new SqliteConnection($"Data Source={Env.SessionDb}");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = "PRAGMA journal_mode=WAL";
        cmd.ExecuteNonQuery();
        return conn;
    }

    public static void Init()
    {
        var dir = Path.GetDirectoryName(Env.SessionDb);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        lock (Lock)
        {
            using var conn = Open();

            // The previous blob-per-session schema is dropped outright. It grew
            // to 3.4 GB and its history has no ongoing value; keeping a
            // compatibility path for it would be carrying a shape we have
            // deliberately moved away from.
            using (var probe = conn.CreateCommand())
            {
                probe.CommandText = "DROP TABLE IF EXISTS sessions";
                probe.ExecuteNonQuery();
            }
            using (var probe = conn.CreateCommand())
            {
                probe.CommandText = "DROP TABLE IF EXISTS messages";
                probe.ExecuteNonQuery();
            }

            using var cmd = conn.CreateCommand();
            cmd.CommandText = """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    model TEXT NOT NULL DEFAULT 'auto',
                    turns INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    ts REAL NOT NULL,
                    provider TEXT NOT NULL DEFAULT 'ocg',
                    served_model TEXT NOT NULL,
                    requested_model TEXT,
                    prompt TEXT NOT NULL,          -- JSON array: new user msgs
                    reply TEXT,                    -- assistant reply, NULL if none
                    status INTEGER NOT NULL DEFAULT 0,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_tokens INTEGER NOT NULL DEFAULT 0,
                    usd REAL NOT NULL DEFAULT 0,
                    compacted INTEGER NOT NULL DEFAULT 0,
                    reaction TEXT,                 -- next user msg; NULL if unanswered
                    reaction_ts REAL
                );
                CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conv_id, seq);
                CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at);
                """;
            cmd.ExecuteNonQuery();
        }
    }

    /// <summary>New user messages this request introduced, found by comparing
    /// against what is already stored rather than by counting.
    ///
    /// Counting is wrong the moment a client compacts: compaction replaces
    /// history with a summary, so the incoming array is *shorter* than what we
    /// hold, and a count-based slice returns nothing and silently drops the
    /// turn. Comparing prefixes instead makes compaction a divergence at
    /// offset 0, so the post-compaction content is simply stored as new.
    ///
    /// Returns the delta and whether a divergence was seen.</summary>
    static (JsonArray Delta, bool Diverged) PromptDelta(SqliteConnection conn, string sid,
        JsonArray? full)
    {
        var delta = new JsonArray();
        if (full is null) return (delta, false);

        var stored = new List<string>();
        using (var q = conn.CreateCommand())
        {
            q.CommandText = "SELECT prompt FROM turns WHERE conv_id = @id ORDER BY seq";
            q.Parameters.AddWithValue("@id", sid);
            using var r = q.ExecuteReader();
            while (r.Read())
            {
                try
                {
                    if (JsonNode.Parse(r.GetString(0)) is JsonArray pa)
                        foreach (var m in pa) stored.Add(Key(m));
                }
                catch { }
            }
        }
        if (stored.Count == 0) return (CloneUserMessages(full), false);

        // Longest common prefix between what we hold and what arrived.
        int lcp = 0;
        int n = Math.Min(stored.Count, full.Count);
        while (lcp < n && stored[lcp] == Key(full[lcp])) lcp++;
        bool diverged = lcp < stored.Count || lcp == 0 && full.Count > 0 && stored.Count > 0;

        for (int i = lcp; i < full.Count; i++)
        {
            var m = full[i];
            if (m is JsonObject o && o["role"]?.GetValue<string>() == "assistant") continue;
            delta.Add(m?.DeepClone());
        }
        return (delta, diverged);
    }

    static string Key(JsonNode? m)
    {
        if (m is not JsonObject o) return m?.ToJsonString() ?? "";
        var role = o["role"]?.GetValue<string>() ?? "";
        var content = o["content"]?.ToJsonString() ?? "";
        return role + "\u0001" + content;
    }

    static JsonArray CloneUserMessages(JsonArray full)
    {
        var d = new JsonArray();
        foreach (var m in full)
        {
            if (m is JsonObject o && o["role"]?.GetValue<string>() == "assistant") continue;
            d.Add(m?.DeepClone());
        }
        return d;
    }

    public static void RecordTurn(string sid, string requestedModel, string servedModel,
        string provider, JsonArray? messages, string? reply, int status, long latencyMs,
        Usage usage, double usd)
    {
        if (string.IsNullOrEmpty(sid)) return;
        try
        {
            lock (Lock)
            {
                using var conn = Open();
                using var tx = conn.BeginTransaction();

                int seq;
                using (var q = conn.CreateCommand())
                {
                    q.CommandText = "SELECT COALESCE(MAX(seq), -1) + 1 FROM turns WHERE conv_id = @id";
                    q.Parameters.AddWithValue("@id", sid);
                    seq = Convert.ToInt32(q.ExecuteScalar());
                }

                var (promptDelta, diverged) = PromptDelta(conn, sid, messages);

                // What the user just sent is their reaction to the PREVIOUS output. Note this
                // is the current turn's prompt delta, not the previous turn's
                // prompt -- that would be the question that produced the
                // previous answer rather than the response to it.
                string? reaction = null;
                if (promptDelta.Count > 0)
                    reaction = promptDelta[0]?["content"]?.ToJsonString();

                var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                if (reaction is not null)
                {
                    using var react = conn.CreateCommand();
                    react.CommandText = """
                        UPDATE turns SET reaction = @r, reaction_ts = @t
                        WHERE id = (SELECT id FROM turns WHERE conv_id = @id ORDER BY seq DESC LIMIT 1)
                        """;
                    react.Parameters.AddWithValue("@r", reaction);
                    react.Parameters.AddWithValue("@t", now);
                    react.Parameters.AddWithValue("@id", sid);
                    react.ExecuteNonQuery();
                }

                string promptJson = promptDelta?.ToJsonString() ?? "[]";
                using (var ins = conn.CreateCommand())
                {
                    ins.CommandText = """
                        INSERT INTO turns (conv_id, seq, ts, provider, served_model, requested_model,
                                           prompt, reply, status, latency_ms,
                                           prompt_tokens, completion_tokens, reasoning_tokens,
                                           cached_tokens, usd)
                        VALUES (@conv, @seq, @ts, @prov, @served, @req, @prompt, @reply, @status,
                                @lat, @pt, @ct, @rt, @ctok, @usd)
                        """;
                    ins.Parameters.AddWithValue("@conv", sid);
                    ins.Parameters.AddWithValue("@seq", seq);
                    ins.Parameters.AddWithValue("@ts", now);
                    ins.Parameters.AddWithValue("@prov", provider);
                    ins.Parameters.AddWithValue("@served", servedModel);
                    ins.Parameters.AddWithValue("@req", requestedModel);
                    ins.Parameters.AddWithValue("@prompt", promptJson);
                    ins.Parameters.AddWithValue("@reply", (object?)reply ?? DBNull.Value);
                    ins.Parameters.AddWithValue("@status", status);
                    ins.Parameters.AddWithValue("@lat", latencyMs);
                    ins.Parameters.AddWithValue("@pt", usage.Prompt);
                    ins.Parameters.AddWithValue("@ct", usage.Completion);
                    ins.Parameters.AddWithValue("@rt", usage.Reasoning);
                    ins.Parameters.AddWithValue("@ctok", usage.Cached);
                    ins.Parameters.AddWithValue("@usd", usd);
                    ins.ExecuteNonQuery();
                }

                using (var up = conn.CreateCommand())
                {
                    up.CommandText = """
                        INSERT INTO conversations (id, model, turns, created_at, updated_at)
                        VALUES (@id, @model, 1, @ts, @ts)
                        ON CONFLICT(id) DO UPDATE SET
                            turns = turns + 1,
                            updated_at = excluded.updated_at
                        """;
                    up.Parameters.AddWithValue("@id", sid);
                    up.Parameters.AddWithValue("@model", requestedModel);
                    up.Parameters.AddWithValue("@ts", now);
                    up.ExecuteNonQuery();
                }

                tx.Commit();
            }
            Interlocked.Increment(ref _writes);
            Interlocked.Exchange(ref _lastWriteTicks, DateTime.UtcNow.Ticks);
        }
        catch (Exception e)
        {
            // Fail-open: persistence never breaks the request path. Logged only
            // on request, because silently swallowing these is exactly what
            // made an earlier regression invisible.
            if (Debug) Console.Error.WriteLine($"[sessions] {e.GetType().Name}: {e.Message}");
        }
    }

    /// <summary>Rolling buffer: drop the oldest conversations once the file
    /// crosses the cap, then VACUUM to actually return the space. Called only
    /// by <see cref="SessionPrunerService"/>, never from the request path.</summary>
    public static (long Dropped, long BytesBefore, long BytesAfter) Prune()
    {
        long before = 0, after = 0, dropped = 0;
        var cap = (long)Env.SessionDbMaxMb * 1024 * 1024;
        if (cap <= 0) return (0, 0, 0);
        Gate.Wait();
        try
        {
            lock (Lock)
            {
                using var conn = Open();
                before = Convert.ToInt64(Scalar(conn, "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"));
                if (before <= cap) return (0, before, before);
                using (var c = conn.CreateCommand())
                {
                    c.CommandText = "SELECT COUNT(*) FROM conversations";
                    dropped = Convert.ToInt64(c.ExecuteScalar());
                }
                using var tx = conn.BeginTransaction();
                foreach (var sql in new[]
                {
                    "DELETE FROM turns WHERE conv_id IN (SELECT id FROM conversations ORDER BY updated_at ASC LIMIT max(1,(SELECT COUNT(*) FROM conversations)/5))",
                    "DELETE FROM conversations WHERE id IN (SELECT id FROM conversations ORDER BY updated_at ASC LIMIT max(1,(SELECT COUNT(*) FROM conversations)/5))",
                })
                {
                    using var c = conn.CreateCommand();
                    c.CommandText = sql;
                    c.ExecuteNonQuery();
                }
                tx.Commit();
            }
            // VACUUM needs exclusive access, hence the gate and the idle wait.
            // It rewrites the whole file, so it must not run on a live gateway
            // mid-request -- that is the entire point of the hosted service.
            lock (Lock)
            {
                using var conn = Open();
                using var v = conn.CreateCommand();
                v.CommandText = "VACUUM";
                v.ExecuteNonQuery();
                after = Convert.ToInt64(Scalar(conn, "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"));
            }
        }
        catch (Exception e)
        {
            if (Debug) Console.Error.WriteLine($"[sessions/prune] {e.GetType().Name}: {e.Message}");
        }
        finally
        {
            Interlocked.Exchange(ref _writes, 0);
            Gate.Release();
        }
        return (dropped, before, after);
    }

    static long Scalar(SqliteConnection conn, string sql)
    {
        using var c = conn.CreateCommand();
        c.CommandText = sql;
        return Convert.ToInt64(c.ExecuteScalar());
    }

    /// <summary>Reconstruct the OpenAI-style message list for /v1/session.</summary>
    public static (JsonArray Messages, string Model, string? Served)? Load(string sid)
    {
        if (string.IsNullOrEmpty(sid)) return null;
        try
        {
            lock (Lock)
            {
                using var conn = Open();
                var msgs = new JsonArray();
                string model = "auto";
                string? served = null;
                using var c = conn.CreateCommand();
                c.CommandText = "SELECT model FROM conversations WHERE id = @id";
                c.Parameters.AddWithValue("@id", sid);
                if (c.ExecuteScalar() is string m) model = m;
                using var sv = conn.CreateCommand();
                sv.CommandText = "SELECT served_model FROM turns WHERE conv_id = @id ORDER BY seq DESC LIMIT 1";
                sv.Parameters.AddWithValue("@id", sid);
                if (sv.ExecuteScalar() is string sv2) served = sv2;
                using var t = conn.CreateCommand();
                t.CommandText = "SELECT prompt, reply FROM turns WHERE conv_id = @id ORDER BY seq";
                t.Parameters.AddWithValue("@id", sid);
                using var tr = t.ExecuteReader();
                while (tr.Read())
                {
                    try
                    {
                        if (JsonNode.Parse(tr.GetString(0)) is JsonArray pa)
                            foreach (var x in pa) msgs.Add(x?.DeepClone());
                    }
                    catch { }
                    if (!tr.IsDBNull(1))
                        // reply is stored as plain text, not JSON -- parsing it
                        // as JSON throws on any reply that does not happen to
                        // start with { or [.
                        msgs.Add(new JsonObject
                        {
                            ["role"] = "assistant",
                            ["content"] = tr.GetString(1),
                        });
                }
                return (msgs, model, served);
            }
        }
        catch (Exception e)
        {
            if (Debug) Console.Error.WriteLine($"[sessions/load] {e.GetType().Name}: {e.Message}");
            return null;
        }
    }

    /// <summary>Turns with their user reactions — the shape rule 4 will read.</summary>
    public static List<Dictionary<string, object?>> TurnsWithReactions(int limit = 500)
    {
        var outp = new List<Dictionary<string, object?>>();
        try
        {
            lock (Lock)
            {
                using var conn = Open();
                using var c = conn.CreateCommand();
                c.CommandText = """
                    SELECT t.conv_id, t.seq, t.ts, t.served_model, t.reply, t.reaction,
                           t.prompt_tokens, t.completion_tokens, t.usd, t.status
                    FROM turns t ORDER BY t.id DESC LIMIT @n
                    """;
                c.Parameters.AddWithValue("@n", limit);
                using var r = c.ExecuteReader();
                while (r.Read())
                    outp.Add(new Dictionary<string, object?>
                    {
                        ["conv"] = r.GetString(0),
                        ["seq"] = r.GetInt64(1),
                        ["ts"] = r.GetInt64(2),
                        ["served_model"] = r.GetString(3),
                        ["reply"] = r.IsDBNull(4) ? null : r.GetString(4),
                        ["reaction"] = r.IsDBNull(5) ? null : r.GetString(5),
                        ["prompt_tokens"] = r.GetInt64(6),
                        ["completion_tokens"] = r.GetInt64(7),
                        ["usd"] = r.GetDouble(8),
                        ["status"] = r.GetInt32(9),
                    });
            }
        }
        catch { }
        return outp;
    }

    public static void Delete(string sid)
    {
        if (string.IsNullOrEmpty(sid)) return;
        try
        {
            lock (Lock)
            {
                using var conn = Open();
                using var tx = conn.BeginTransaction();
                foreach (var sql in new[]
                {
                    "DELETE FROM turns WHERE conv_id = @id",
                    "DELETE FROM conversations WHERE id = @id",
                })
                {
                    using var c = conn.CreateCommand();
                    c.CommandText = sql;
                    c.Parameters.AddWithValue("@id", sid);
                    c.ExecuteNonQuery();
                }
                tx.Commit();
            }
        }
        catch { }
    }

    public static (long Conversations, long Turns, long Bytes, long Reacted) Stats()
    {
        try
        {
            lock (Lock)
            {
                using var conn = Open();
                using var c = conn.CreateCommand();
                c.CommandText = """
                    SELECT (SELECT COUNT(*) FROM conversations), (SELECT COUNT(*) FROM turns),
                           (SELECT COUNT(*) FROM turns WHERE reaction IS NOT NULL),
                           (SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size())
                    """;
                using var r = c.ExecuteReader();
                if (!r.Read()) return (0, 0, 0, 0);
                return (r.GetInt64(0), r.GetInt64(1), r.GetInt64(2), r.GetInt64(3));
            }
        }
        catch { return (0, 0, 0, 0); }
    }
}