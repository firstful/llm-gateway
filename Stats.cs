using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace LlmGateway;

/// <summary>
/// Per-request statistics: in-memory rolling window for GET /stats + optional
/// telemetry flush (raw-text POST, &lt;=TELEMETRY_MAX_BYTES). Reed 2026-08-07:
/// track stats over time so real-world RAG overhead is measurable, not guessed.
/// Fail-open everywhere: stats must NEVER break the request path.
/// </summary>
public static class Stats
{
    public sealed record Usage(int Prompt, int Completion, int Reasoning);
    public sealed record RagMeta(bool Injected, int Chunks, int Bytes);
    public sealed record Rec(long Ts, string? Consumer, string Endpoint, bool Stream, int Status,
        long LatencyMs, RagMeta Rag, Usage Usage);

    static readonly object Lock = new();
    static readonly List<Rec> Window = new();
    static readonly List<Rec> Pending = new();

    public static long Requests, WithRag, RagBytes, RagChunks, PromptTokens, CompletionTokens,
        ReasoningTokens, LatencyMs, Errors, DroppedTelemetry;

    public static void Record(Rec rec)
    {
        lock (Lock)
        {
            Window.Add(rec);
            while (Window.Count > Env.StatsWindow) Window.RemoveAt(0);
            Pending.Add(rec);
            Requests++;
            if (rec.Rag.Injected) { WithRag++; RagBytes += rec.Rag.Bytes; RagChunks += rec.Rag.Chunks; }
            PromptTokens += rec.Usage.Prompt; CompletionTokens += rec.Usage.Completion;
            ReasoningTokens += rec.Usage.Reasoning; LatencyMs += rec.LatencyMs;
            if (rec.Status >= 400) Errors++;
        }
    }

    static string Line(Rec r)
    {
        // one compact JSON line for telemetry (~150B); short keys to stay far
        // under the telemetry body cap when batched
        var o = new JsonObject
        {
            ["ts"] = r.Ts,
            ["c"] = (r.Consumer ?? "-")[..Math.Min((r.Consumer ?? "-").Length, 24)],
            ["ep"] = r.Endpoint,
            ["s"] = r.Stream ? 1 : 0,
            ["st"] = r.Status,
            ["ms"] = r.LatencyMs,
            ["ri"] = r.Rag.Injected ? 1 : 0,
            ["rc"] = r.Rag.Chunks,
            ["rb"] = r.Rag.Bytes,
            ["pt"] = r.Usage.Prompt,
            ["ct"] = r.Usage.Completion,
            ["rt"] = r.Usage.Reasoning,
        };
        return o.ToJsonString();
    }

    public static async Task FlushAsync()
    {
        if (string.IsNullOrEmpty(Env.TelemetryUrl)) return;
        List<Rec> batch;
        lock (Lock)
        {
            if (Pending.Count == 0) return;
            batch = new List<Rec>();
            var size = 0;
            while (Pending.Count > 0)
            {
                var line = Line(Pending[0]);
                if (size + Encoding.UTF8.GetByteCount(line) + 1 > Env.TelemetryMaxBytes) break;
                size += Encoding.UTF8.GetByteCount(line) + 1;
                batch.Add(Pending[0]);
                Pending.RemoveAt(0);
            }
        }
        if (batch.Count == 0) return;
        try
        {
            var body = string.Join('\n', batch.Select(Line));
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            var resp = await client.PostAsync(Env.TelemetryUrl, new StringContent(body, Encoding.UTF8));
            if ((int)resp.StatusCode >= 400)
                throw new InvalidOperationException($"telemetry HTTP {(int)resp.StatusCode}");
        }
        catch
        {
            lock (Lock) DroppedTelemetry += batch.Count;
        }
    }

    static double Pct(List<long> arr, double p)
    {
        if (arr.Count == 0) return 0;
        arr.Sort();
        return arr[Math.Min(arr.Count - 1, (int)(arr.Count * p))];
    }

    static double Avg(IEnumerable<long> arr)
        => arr.Any() ? Math.Round(arr.Average(), 1) : 0;

    /// <summary>GET /stats body — mirrors the Python shape exactly.</summary>
    public static JsonObject GetStats(int limit)
    {
        List<Rec> win;
        long totalsRequests, totalsLatency, totalsRagBytes;
        lock (Lock)
        {
            win = new List<Rec>(Window);
            totalsRequests = Requests; totalsLatency = LatencyMs; totalsRagBytes = RagBytes;
        }
        var n = win.Count;
        var ragRecs = win.Where(r => r.Rag.Injected).ToList();
        var withRag = ragRecs.Count;

        var agg = new JsonObject
        {
            ["window_requests"] = n,
            ["window_with_rag"] = withRag,
            ["rag_inject_rate_pct"] = n > 0 ? Math.Round(100.0 * withRag / n, 1) : 0,
            ["rag_bytes"] = new JsonObject
            {
                ["total"] = ragRecs.Sum(r => (long)r.Rag.Bytes),
                ["avg"] = withRag > 0 ? Math.Round(ragRecs.Average(r => (double)r.Rag.Bytes), 1) : 0,
                ["p50"] = Pct(ragRecs.Select(r => (long)r.Rag.Bytes).ToList(), 0.5),
                ["p95"] = Pct(ragRecs.Select(r => (long)r.Rag.Bytes).ToList(), 0.95),
            },
            ["rag_tokens_est"] = new JsonObject
            {
                ["avg"] = withRag > 0 ? Math.Round(ragRecs.Average(r => (double)r.Rag.Bytes) / 4 / withRag, 1) : 0,
            },
            ["prompt_tokens"] = new JsonObject
            {
                ["avg"] = Avg(win.Select(r => (long)r.Usage.Prompt)),
                ["p95"] = Pct(win.Select(r => (long)r.Usage.Prompt).ToList(), 0.95),
            },
            ["completion_tokens"] = new JsonObject { ["avg"] = Avg(win.Select(r => (long)r.Usage.Completion)) },
            ["reasoning_tokens"] = new JsonObject { ["avg"] = Avg(win.Select(r => (long)r.Usage.Reasoning)) },
            ["latency_ms"] = new JsonObject
            {
                ["avg"] = Avg(win.Select(r => r.LatencyMs)),
                ["p95"] = Pct(win.Select(r => r.LatencyMs).ToList(), 0.95),
            },
            ["errors"] = win.Count(r => r.Status >= 400),
            ["by_consumer"] = new JsonObject(),
        };

        var byConsumer = agg["by_consumer"]!.AsObject();
        foreach (var r in win)
        {
            var c = r.Consumer ?? "-";
            if (byConsumer[c] is not JsonObject b)
            {
                b = new JsonObject { ["requests"] = 0, ["with_rag"] = 0, ["rag_bytes"] = 0, ["prompt_tokens"] = 0 };
                byConsumer[c] = b;
            }
            b["requests"] = (int)b["requests"]! + 1;
            if (r.Rag.Injected)
            {
                b["with_rag"] = (int)b["with_rag"]! + 1;
                b["rag_bytes"] = (int)b["rag_bytes"]! + r.Rag.Bytes;
            }
            b["prompt_tokens"] = (int)b["prompt_tokens"]! + r.Usage.Prompt;
        }

        var recent = new JsonArray();
        foreach (var r in win.Skip(Math.Max(0, win.Count - Math.Max(1, Math.Min(limit, 100)))))
        {
            recent.Add(new JsonObject
            {
                ["ts"] = r.Ts,
                ["consumer"] = r.Consumer,
                ["endpoint"] = r.Endpoint,
                ["stream"] = r.Stream,
                ["status"] = r.Status,
                ["latency_ms"] = r.LatencyMs,
                ["rag"] = new JsonObject { ["injected"] = r.Rag.Injected, ["chunks"] = r.Rag.Chunks, ["bytes"] = r.Rag.Bytes },
                ["usage"] = new JsonObject { ["prompt"] = r.Usage.Prompt, ["completion"] = r.Usage.Completion, ["reasoning"] = r.Usage.Reasoning },
            });
        }

        var totals = new JsonObject
        {
            ["requests"] = Requests, ["with_rag"] = WithRag, ["rag_bytes"] = RagBytes, ["rag_chunks"] = RagChunks,
            ["prompt_tokens"] = PromptTokens, ["completion_tokens"] = CompletionTokens,
            ["reasoning_tokens"] = ReasoningTokens, ["latency_ms"] = LatencyMs, ["errors"] = Errors,
            ["dropped_telemetry"] = DroppedTelemetry,
            ["avg_latency_ms"] = totalsRequests > 0 ? Math.Round((double)totalsLatency / totalsRequests, 1) : 0,
            ["rag_bytes_per_request"] = totalsRequests > 0 ? Math.Round((double)totalsRagBytes / totalsRequests, 1) : 0,
        };

        return new JsonObject
        {
            ["app"] = Env.AppName,
            ["version"] = Env.AppVersion,
            ["window"] = Env.StatsWindow,
            ["aggregates_window"] = agg,
            ["totals_since_start"] = totals,
            ["telemetry"] = new JsonObject { ["url"] = Env.TelemetryUrl, ["flush_seconds"] = Env.StatsFlushSeconds },
            ["recent"] = recent,
        };
    }
}

/// <summary>Background flusher: drain pending stats to telemetry in bounded batches.</summary>
public class StatsFlusherService : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(TimeSpan.FromSeconds(Env.StatsFlushSeconds), ct);
                await Stats.FlushAsync();
            }
            catch (OperationCanceledException) { break; }
            catch { /* fail-open */ }
        }
    }
}
