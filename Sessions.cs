using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Data.Sqlite;

namespace LlmGateway;

/// <summary>
/// Server-side chat session persistence (v1.7.0): SQLite keyed by cookie.
/// Thread-safe per-call connections, WAL. Same schema as the Python version —
/// the PVC database carries over unmodified across the rewrite.
/// </summary>
public static class Sessions
{
    static readonly object Lock = new();

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
            using var cmd = conn.CreateCommand();
            cmd.CommandText = """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    messages TEXT NOT NULL,      -- JSON array of {role, content}
                    model TEXT NOT NULL DEFAULT 'auto',
                    updated_at REAL NOT NULL
                )
                """;
            cmd.ExecuteNonQuery();
        }
    }

    public static (JsonArray? Messages, string Model)? Load(string sid)
    {
        if (string.IsNullOrEmpty(sid)) return null;
        try
        {
            lock (Lock)
            {
                using var conn = Open();
                using var cmd = conn.CreateCommand();
                cmd.CommandText = "SELECT messages, model FROM sessions WHERE id = @id";
                cmd.Parameters.AddWithValue("@id", sid);
                using var r = cmd.ExecuteReader();
                if (!r.Read()) return null;
                var msgs = JsonNode.Parse(r.GetString(0)) as JsonArray;
                return (msgs, r.GetString(1));
            }
        }
        catch { return null; }
    }

    public static void Save(string sid, JsonArray? messages, string model)
    {
        try
        {
            lock (Lock)
            {
                using var conn = Open();
                using var cmd = conn.CreateCommand();
                cmd.CommandText = """
                    INSERT INTO sessions (id, messages, model, updated_at)
                    VALUES (@id, @messages, @model, @ts)
                    ON CONFLICT(id) DO UPDATE SET
                        messages = excluded.messages,
                        model = excluded.model,
                        updated_at = excluded.updated_at
                    """;
                cmd.Parameters.AddWithValue("@id", sid);
                cmd.Parameters.AddWithValue("@messages", messages?.ToJsonString() ?? "[]");
                cmd.Parameters.AddWithValue("@model", model);
                cmd.Parameters.AddWithValue("@ts", DateTimeOffset.UtcNow.ToUnixTimeSeconds());
                cmd.ExecuteNonQuery();
            }
        }
        catch { /* fail-open: persistence never breaks the request path */ }
    }

    public static void Delete(string sid)
    {
        try
        {
            lock (Lock)
            {
                using var conn = Open();
                using var cmd = conn.CreateCommand();
                cmd.CommandText = "DELETE FROM sessions WHERE id = @id";
                cmd.Parameters.AddWithValue("@id", sid);
                cmd.ExecuteNonQuery();
            }
        }
        catch { }
    }
}
