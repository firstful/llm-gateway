using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Channels;

namespace LlmGateway;

/// <summary>
/// RAG injection (v1.2.0) + live session ingestion (v1.3.0).
/// Header-driven control plane, fail-open everywhere: any search error leaves
/// the request untouched. The embeddings service is at RAG_BASE_URL (/search,
/// /ingest/sessions).
/// </summary>
public static class Rag
{
    static readonly HttpClient Http = BuildClient();

    static HttpClient BuildClient()
    {
        var handler = new SocketsHttpHandler { ConnectTimeout = TimeSpan.FromSeconds(5) };
        return new HttpClient(handler) { Timeout = TimeSpan.FromSeconds(10) };
    }

    public static (string? Consumer, bool DocsEnabled, bool Optin) RequestHeaders(HttpContext ctx)
    {
        var consumer = (ctx.Request.Headers["X-Consumer"].ToString() ?? "").Trim();
        var docsRaw = (ctx.Request.Headers["X-RAG-Docs"].ToString() ?? "").Trim().ToLowerInvariant();
        var optinRaw = (ctx.Request.Headers["X-RAG-Optin"].ToString() ?? "").Trim().ToLowerInvariant();
        return (
            consumer.Length > 0 ? consumer : null,
            docsRaw is not ("0" or "false" or "no"),
            optinRaw is not ("0" or "false" or "no"));
    }

    /// <summary>Extract injected-RAG metadata from the (possibly RAG-prepended) message list.</summary>
    public static (bool Injected, int Chunks, int Bytes) RagMeta(JsonArray? messages)
    {
        if (messages is null || messages.Count == 0 || messages[0] is not JsonObject first) return (false, 0, 0);
        var content = first["content"]?.GetValue<string>() ?? "";
        if (first["role"]?.GetValue<string>() != "system" || !content.StartsWith(Env.RagPrefix)) return (false, 0, 0);
        var n = content.Split("--- [", StringSplitOptions.None).Length - 1;
        return (true, n, Encoding.UTF8.GetByteCount(content));
    }

    static async Task<List<JsonObject>> SearchAsync(JsonObject payload)
    {
        try
        {
            using var req = new HttpRequestMessage(HttpMethod.Post, $"{Env.RagBaseUrl}/search")
            {
                Content = new StringContent(payload.ToJsonString(), Encoding.UTF8, "application/json"),
            };
            using var resp = await Http.SendAsync(req);
            resp.EnsureSuccessStatusCode();
            var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
            var outList = new List<JsonObject>();
            if (doc.RootElement.TryGetProperty("results", out var results) && results.ValueKind == JsonValueKind.Array)
                foreach (var r in results.EnumerateArray())
                    if (r.ValueKind == JsonValueKind.Object) outList.Add(JsonNode.Parse(r.GetRawText())!.AsObject());
            return outList;
        }
        catch { return new List<JsonObject>(); }
    }

    /// <summary>Balanced retrieval: query docs and sessions SEPARATELY (top_k//2 each)
    /// and interleave, so session narratives don't starve the docs answer.</summary>
    public static async Task<List<JsonObject>> RetrieveAsync(string query, string? consumer, bool docsEnabled)
    {
        if (!Env.RagEnabled) return new List<JsonObject>();
        var perSource = Math.Max(1, (Env.RagTopK + 1) / 2);

        var merged = new List<JsonObject>();
        var sess = new JsonObject
        {
            ["query"] = query,
            ["top_k"] = perSource,
            ["min_score"] = Env.RagMinScore,
            ["sources"] = new JsonArray("session"),
        };
        if (consumer is not null) sess["consumers"] = new JsonArray(consumer);
        merged.AddRange(await SearchAsync(sess));

        if (docsEnabled)
        {
            var docs = new JsonObject
            {
                ["query"] = query,
                ["top_k"] = perSource,
                ["min_score"] = Env.RagMinScore,
                ["sources"] = new JsonArray("docs"),
            };
            merged.AddRange(await SearchAsync(docs));
        }

        var sessions = merged.Where(h => (h["source"]?.GetValue<string>()) == "session").ToList();
        var docsList = merged.Where(h => (h["source"]?.GetValue<string>()) == "docs").ToList();
        var outList = new List<JsonObject>();
        for (int i = 0; i < Math.Max(sessions.Count, docsList.Count); i++)
        {
            if (i < sessions.Count) outList.Add(sessions[i]);
            if (i < docsList.Count) outList.Add(docsList[i]);
        }
        return outList.Take(Env.RagTopK).ToList();
    }

    public static string SystemMessage(string query, List<JsonObject> hits)
    {
        var parts = new List<string>();
        foreach (var h in hits.Take(Env.RagTopK))
        {
            var src = h["source"]?.GetValue<string>() ?? "?";
            var consumer = h["consumer"]?.GetValue<string>() ?? "";
            var title = h["title"]?.GetValue<string>() ?? "";
            var text = h["text"]?.GetValue<string>() ?? "";
            if (text.Length > Env.RagMaxChars) text = text[..Env.RagMaxChars];
            var score = h["score"]?.GetValue<double>() ?? 0;
            var who = consumer.Length > 0 ? $" [{consumer}]" : "";
            parts.Add($"--- [{src}{who}] {title} (score {score:0.###})\n{text}");
        }
        return Env.RagPrefix + "\n\n" + string.Join("\n\n", parts);
    }

    static string? LastUserMessage(JsonArray messages)
    {
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            var m = messages[i] as JsonObject;
            if (m is null) continue;
            if (m["role"]?.GetValue<string>() != "user") continue;
            var content = m["content"];
            if (content is null || content.GetValueKind() != JsonValueKind.String) continue;
            var text = content.GetValue<string>().Trim();
            if (text.Length > 0) return text;
        }
        return null;
    }

    /// <summary>Return messages with a RAG context system message injected (fail-open: unchanged).</summary>
    public static async Task<JsonArray> InjectAsync(JsonArray messages, string? consumer, bool docsEnabled)
    {
        var query = LastUserMessage(messages);
        if (query is null) return messages;
        var hits = await RetrieveAsync(query, consumer, docsEnabled);
        if (hits.Count == 0) return messages;
        var ctx = SystemMessage(query, hits);
        // prepend as the FIRST system message (standard RAG pattern; keeps the
        // client's own system prompt intact as a separate message).
        var copy = new JsonArray();
        copy.Add(new JsonObject { ["role"] = "system", ["content"] = ctx });
        foreach (var m in messages) copy.Add(m?.DeepClone());
        return copy;
    }

    // ---------------- live session ingestion (deferred queue) ----------------

    public sealed record LiveChunk(string Consumer, string UserMsg, string Reply, long Ts);

    // bounded + DropWrite: the queue never blocks the response path (fail-open)
    public static readonly Channel<LiveChunk> LiveQueue = Channel.CreateBounded<LiveChunk>(
        new BoundedChannelOptions(10000) { SingleReader = true, SingleWriter = false, FullMode = BoundedChannelFullMode.DropWrite });

    /// <summary>Enqueue one turn pair for background embedding. Never raises, never blocks.</summary>
    public static void TeeLiveTurn(string? consumer, string userMsg, string reply)
    {
        if (!Env.LiveIngestEnabled || string.IsNullOrEmpty(consumer) || string.IsNullOrWhiteSpace(userMsg)) return;
        try
        {
            LiveQueue.Writer.TryWrite(new LiveChunk(
                consumer, userMsg.Trim(), reply.Trim(), DateTimeOffset.UtcNow.ToUnixTimeSeconds()));
        }
        catch { /* fail-open */ }
    }

    public static async Task FlushLiveAsync()
    {
        var batch = new List<LiveChunk>();
        while (batch.Count < Env.LiveFlushBatch && LiveQueue.Reader.TryRead(out var item)) batch.Add(item);
        if (batch.Count == 0) return;

        // group by consumer, one POST per consumer
        foreach (var group in batch.GroupBy(i => i.Consumer))
        {
            var chunks = new JsonArray();
            foreach (var item in group)
            {
                var text = $"user: {item.UserMsg}";
                if (item.Reply.Length > 0) text += $"\nassistant: {item.Reply}";
                var dt = DateTimeOffset.FromUnixTimeSeconds(item.Ts).ToLocalTime();
                chunks.Add(new JsonObject
                {
                    ["id"] = $"live:{item.Consumer}:{item.Ts}",
                    ["title"] = $"live {item.Consumer} {dt:yyyy-MM-dd HH:mm}",
                    ["text"] = text,
                    ["meta"] = new JsonObject { ["consumer"] = item.Consumer, ["live"] = true, ["ts"] = item.Ts },
                });
            }
            try
            {
                var payload = new JsonObject { ["consumer"] = group.Key, ["chunks"] = chunks };
                using var req = new HttpRequestMessage(HttpMethod.Post, $"{Env.RagBaseUrl}/ingest/sessions")
                {
                    Content = new StringContent(payload.ToJsonString(), Encoding.UTF8, "application/json"),
                };
                using var resp = await Http.SendAsync(req);
                resp.EnsureSuccessStatusCode();
            }
            catch { /* fail-open: NFS snapshot catchup covers it */ }
        }
    }
}

/// <summary>Background flusher: drain the live-ingest queue every LIVE_FLUSH_SECONDS.</summary>
public class LiveFlusherService : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        if (!Env.LiveIngestEnabled) return;
        while (!ct.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(TimeSpan.FromSeconds(Env.LiveFlushSeconds), ct);
                await Rag.FlushLiveAsync();
            }
            catch (OperationCanceledException) { break; }
            catch { /* keep draining */ }
        }
    }
}
