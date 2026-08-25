using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace LlmGateway;

/// <summary>
/// Rolling JSON-lines request log (v2.2.0) — one line per client request under
/// REQ_LOG_DIR (the /data PVC), capturing EVERY provider attempt with full
/// error detail so fallback chains are debuggable after the fact. Cached
/// prompt tokens are reported separately and EXCLUDED from input_tokens.
/// Fail-open everywhere: logging must NEVER break the request path.
/// </summary>
public static class ReqLog
{
    public sealed record Attempt(
        string Provider, string Upstream, string Outcome, int Status, long Ms, string? Detail);

    static readonly object Lock = new();
    static readonly string Dir = Env.ReqLogDir;
    static readonly long MaxBytes = (long)Env.ReqLogMaxMb * 1024 * 1024;
    static readonly int Keep = Math.Max(1, Env.ReqLogKeepFiles);
    const string Prefix = "req-";
    static readonly JsonSerializerOptions JsonOpts =
        new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    public static void Write(HttpContext ctx, string endpoint, string? model, bool stream,
        int status, long latencyMs, IReadOnlyList<Attempt> attempts,
        Stats.RagMeta rag, Stats.Usage usage, int cachedInput, string? error)
    {
        try
        {
            var (consumer, _, _) = Rag.RequestHeaders(ctx);
            var arr = new JsonArray();
            foreach (var a in attempts)
                arr.Add(new JsonObject
                {
                    ["provider"] = a.Provider,
                    ["upstream"] = a.Upstream,
                    ["outcome"] = a.Outcome,
                    ["status"] = a.Status,
                    ["ms"] = a.Ms,
                    ["detail"] = Trunc(a.Detail),
                });
            var line = new JsonObject
            {
                ["ts"] = DateTimeOffset.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss.fff'Z'"),
                ["endpoint"] = endpoint,
                ["consumer"] = consumer ?? "-",
                ["model"] = string.IsNullOrEmpty(model) ? "auto" : model,
                ["stream"] = stream,
                ["status"] = status,
                ["ms"] = latencyMs,
                ["attempts"] = arr,
                ["rag"] = new JsonObject
                {
                    ["injected"] = rag.Injected,
                    ["chunks"] = rag.Chunks,
                    ["bytes"] = rag.Bytes,
                },
                ["usage"] = new JsonObject
                {
                    // cached prompt tokens excluded from input; reported alongside
                    ["input_tokens"] = Math.Max(0, usage.Prompt - cachedInput),
                    ["cached_input_tokens"] = cachedInput,
                    ["output_tokens"] = usage.Completion,
                    ["reasoning_tokens"] = usage.Reasoning,
                },
                ["error"] = Trunc(error),
            };
            Append(line.ToJsonString(JsonOpts) + "\n");
        }
        catch { /* fail-open */ }
    }

    static string? Trunc(string? s)
        => string.IsNullOrEmpty(s) ? s : s.Length <= 400 ? s : s[..400];

    static void Append(string line)
    {
        lock (Lock)
        {
            Directory.CreateDirectory(Dir);
            var (path, idx, size) = CurrentFile();
            if (size >= MaxBytes)
            {
                idx++;
                path = Path.Combine(Dir, $"{Prefix}{idx:D4}.jsonl");
            }
            File.AppendAllText(path, line);
            Prune(idx);
        }
    }

    static (string Path, int Idx, long Size) CurrentFile()
    {
        var idx = 0;
        foreach (var f in Directory.EnumerateFiles(Dir, $"{Prefix}*.jsonl"))
        {
            var name = Path.GetFileNameWithoutExtension(f);
            if (name.Length == Prefix.Length + 4
                && int.TryParse(name[Prefix.Length..], out var n) && n > idx)
                idx = n;
        }
        var p = Path.Combine(Dir, $"{Prefix}{idx:D4}.jsonl");
        return (p, idx, File.Exists(p) ? new FileInfo(p).Length : 0);
    }

    static void Prune(int cur)
    {
        var curPath = Path.Combine(Dir, $"{Prefix}{cur:D4}.jsonl");
        var all = Directory.EnumerateFiles(Dir, $"{Prefix}*.jsonl")
            .Select(f => (Path: f, N: int.TryParse(Path.GetFileNameWithoutExtension(f)[Prefix.Length..], out var x) ? x : int.MinValue))
            .OrderBy(t => t.N)
            .Select(t => t.Path)
            .ToList();
        for (var i = 0; i < all.Count - Keep; i++)
        {
            if (all[i] == curPath && all.Count == 1) break;
            try { File.Delete(all[i]); } catch { break; }
        }
    }
}
