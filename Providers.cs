using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace LlmGateway;

/// <summary>
/// llm-gateway — OpenAI-compatible LLM gateway for the Sutliff homelab.
/// C# port (v2.0.0) of the Python 1.7.4 service. Serves /v1/chat/completions +
/// /v1/responses, aggregating multiple upstream LLM providers. Model names are
/// prefixed by provider (case-insensitive):
///
///   OCZ-&lt;model&gt;    -> opencode zen  (https://opencode.ai/zen/v1) — main API,
///                     free models live here under their own ids
///   OCG-&lt;model&gt;    -> opencode-go   (https://opencode.ai/zen/go/v1) — weekly-capped
///   LOCAL-&lt;model&gt;  -> local GPU     (llama-ctrl; ONLY when LOCAL_ENABLED=1)
///   auto           -> explicit ordered fallback chain — see AUTO_CHAIN
///                     (comma-separated "provider:model" legs, tried in
///                     sequence), then LOCAL-&lt;LOCAL_MODEL&gt; if enabled
///
/// Chat sessions persist server-side (SQLite + cookie), the UI shows live
/// provider + local-LLM status.
/// LAN-only service (llm.homelab.internal) + user-facing UI at llm.ffilt.us.
/// </summary>
public static class Env
{
    public const string AppName = "llm-gateway";
    public const string AppVersion = "2.1.0";

    // Only used as a fallback advertisement when a provider /models listing
    // can't be verified. The actual auto chain is explicit — see AUTO_CHAIN.
    public static string AutoModel { get; } = EnvOr("AUTO_MODEL", "mimo-v2.5");
    // Explicit ordered fallback chain for `auto`: comma-separated "provider:model"
    // legs, tried in sequence. Each leg is plainly named — no implicit AGNO rules.
    public static string AutoChainRaw { get; } = EnvOr(
        "AUTO_CHAIN", "ocg:mimo-v2.5,or:openrouter/free,or:deepseek/deepseek-v4-flash");
    public static string LocalModel { get; } = EnvOr("LOCAL_MODEL", "lfm2.5-2.6b");
    public static int ModelsCacheTtl { get; } = IntOr("MODELS_CACHE_TTL", 300);

    // RAG injection (v1.2.0) — header-driven control plane:
    //   X-Consumer: <id>       scope session embeddings to this consumer
    //   X-RAG-Docs: 1|0        include docs source (default on)
    //   X-RAG-Optin: 1|0       master switch (default on)
    // Fail-open: any search error leaves the request untouched.
    public static bool RagEnabled { get; } = EnvOr("RAG_ENABLED", "1") is not ("0" or "false" or "False");
    public static string RagBaseUrl { get; } = EnvOr("RAG_BASE_URL", "http://embeddings.homelab.internal").TrimEnd('/');
    public static int RagTopK { get; } = IntOr("RAG_TOP_K", 3);
    public static double RagMinScore { get; } = DoubleOr("RAG_MIN_SCORE", 0.15);
    public static int RagMaxChars { get; } = IntOr("RAG_MAX_CHARS", 1800);

    // Live session ingestion (v1.3.0) — tee every turn through a deferred queue
    // into the embeddings service (/ingest/sessions). PRIMARY session-memory path.
    public static bool LiveIngestEnabled { get; } = EnvOr("LIVE_INGEST_ENABLED", "1") is not ("0" or "false" or "False");
    public static double LiveFlushSeconds { get; } = DoubleOr("LIVE_FLUSH_SECONDS", 3);
    public static int LiveFlushBatch { get; } = IntOr("LIVE_FLUSH_BATCH", 20);

    // Chat session persistence (v1.7.0)
    public static string SessionDb { get; } = EnvOr("SESSION_DB", "/data/sessions.db");
    public const string SessionCookie = "llm_session";
    public static int SessionTtlDays { get; } = IntOr("SESSION_TTL_DAYS", 30);

    // Live local-LLM status cache (v1.7.0) — proxy llama-ctrl /status, cached 5s.
    public static string LocalCtrlUrl { get; } = EnvOr("LOCAL_CTRL_URL", "http://llama-ctrl.llama.svc.cluster.local").TrimEnd('/');

    // Request statistics (rolling + telemetry) — v1.6/2026-08-07.
    public static int StatsWindow { get; } = IntOr("STATS_WINDOW", 5000);

    // Rolling JSON-lines request log (v2.2.0) — lives on the /data PVC so it
    // survives restarts. Each file caps at REQ_LOG_MAX_MB, oldest rolled off
    // past REQ_LOG_KEEP_FILES. Fail-open: logging must never break a request.
    public static string ReqLogDir { get; } = EnvOr("REQ_LOG_DIR", "/data/logs");
    public static int ReqLogMaxMb { get; } = IntOr("REQ_LOG_MAX_MB", 100);
    public static int ReqLogKeepFiles { get; } = IntOr("REQ_LOG_KEEP_FILES", 10);
    public static int ReqLogWindowSeconds { get; } = IntOr("REQ_LOG_WINDOW_SECONDS", 600);
    public static int StatsFlushSeconds { get; } = IntOr("STATS_FLUSH_SECONDS", 30);
    public static string TelemetryUrl { get; } = EnvOr("TELEMETRY_URL", "").TrimEnd('/');
    public static int TelemetryMaxBytes { get; } = IntOr("TELEMETRY_MAX_BYTES", 3800);
    public static int RecoveryKeepAliveSeconds { get; } = IntOr("RECOVERY_KEEPALIVE_SECONDS", 5);
    public static int RecoveryMaxWaitSeconds { get; } = IntOr("RECOVERY_MAX_WAIT_SECONDS", 600);

    public const string RagPrefix =
        "The following is authoritative context from the homelab operator's docs and " +
        "session history (retrieved via RAG). It is FACT, not a hint — prefer it over " +
        "guessing or tool checks when it answers the user's question:";

    public static string EnvOr(string key, string def)
        => Environment.GetEnvironmentVariable(key) is { Length: > 0 } v ? v : def;

    public static int IntOr(string key, int def)
        => int.TryParse(Environment.GetEnvironmentVariable(key), out var v) ? v : def;

    public static double DoubleOr(string key, double def)
        => double.TryParse(Environment.GetEnvironmentVariable(key), System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : def;
}

/// <summary>API error carrying an HTTP status — rendered as the OpenAI error shape.</summary>
public class ApiError : Exception
{
    public int Status { get; }
    public ApiError(int status, string detail) : base(detail) => Status = status;
}

public class Provider
{
    public required string Key { get; init; }
    public required string Label { get; init; }
    public required string BaseUrl { get; init; }
    public string ApiKey { get; init; } = "";

    public HttpClient Client { get; } = BuildClient();

    static HttpClient BuildClient()
    {
        var handler = new SocketsHttpHandler
        {
            ConnectTimeout = TimeSpan.FromSeconds(10),
            PooledConnectionLifetime = TimeSpan.FromMinutes(5),
        };
        // read timeout mirrors httpx read=600 (generous for long generations)
        return new HttpClient(handler) { Timeout = TimeSpan.FromSeconds(600) };
    }
}

/// <summary>Provider registry, model availability cache, routing + reasoning echo.</summary>
public static class Providers
{
    // LOCAL is opt-in (LOCAL_ENABLED=1). Python hardcoded it and advertised a
    // dead LOCAL-* ad in /v1/models after the RX 470 shutdown — fixed in the port.
    public static bool LocalEnabled { get; } = Env.EnvOr("LOCAL_ENABLED", "0") is not ("0" or "false" or "False");

    public static readonly List<Provider> All = Build();

    // Explicit ordered fallback chain for `auto` (from AUTO_CHAIN). Each leg is
    // (provider_key, upstream_model); only providers actually in the registry are
    // kept, so a leg referencing a removed provider is silently dropped.
    public static readonly List<(string Key, string Model)> AutoChain =
        Env.AutoChainRaw
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(ParseLeg)
            .Where(l => l.Key is not null)
            .Select(l => (l.Key!, l.Model!))
            .ToList();

    static (string? Key, string? Model) ParseLeg(string leg)
    {
        var i = leg.IndexOf(':');
        if (i <= 0) return (null, null);
        var key = leg[..i].Trim().ToLowerInvariant();
        var model = leg[(i + 1)..].Trim();
        if (model.Length == 0 || !All.Any(p => p.Key == key)) return (null, null);
        return (key, model);
    }

    // fallback on these upstream outcomes for auto chains
    public static readonly HashSet<int> FallbackStatus = new() { 401, 402, 403, 408, 429, 500, 502, 503, 504 };

    static readonly ConcurrentDictionary<string, (DateTime At, HashSet<string>? Ids)> ModelsCache = new();

    // OCZ (opencode.ai/zen/v1) serves some models under ids that differ from
    // OCG's /go endpoint. Known renames that don't fit the -free suffix rule.
    static readonly Dictionary<string, string> OczAliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["ox-alpha-free"] = "x-preview-f-free",
    };

    /// <summary>
    /// Automatic model resolution for auto chain legs: map the desired
    /// model name to an id the provider actually serves, or null when a
    /// verifiable /models listing shows no match (the leg is skipped so the
    /// chain falls through instead of dying on an upstream 404, which is NOT
    /// in FallbackStatus). Only OCZ resolves today — zen's free-model ids
    /// differ from the /go endpoint's (exact -> alias -> +free -> -free).
    /// </summary>
    public static async Task<string?> ResolveAutoAsync(string key, string want)
    {
        if (key != "ocz") return want;
        var ids = await ProviderModelsAsync(key);
        if (ids is null || ids.Count == 0)
        {
            // can't verify — best static guess (upstream errors fall through)
            return OczAliases.TryGetValue(want, out var blind) ? blind : want;
        }
        if (ids.Contains(want)) return want;
        if (OczAliases.TryGetValue(want, out var alias) && ids.Contains(alias)) return alias;
        if (ids.Contains(want + "-free")) return want + "-free";
        if (want.EndsWith("-free") && ids.Contains(want[..^5])) return want[..^5];
        return null; // verifiably not served -> skip this leg
    }

    // DeepSeek thinking-mode echo cache: keyed by tool_call id, capped.
    static readonly object EchoLock = new();
    static readonly Dictionary<string, string> ReasoningEcho = new();
    static readonly List<string> EchoOrder = new();
    const int ReasoningEchoMax = 4096;

    static List<Provider> Build()
    {
        var list = new List<Provider>
        {
            new() { Key = "or", Label = "OR", BaseUrl = Env.EnvOr("OR_BASE_URL", "https://openrouter.ai/api/v1").TrimEnd('/'), ApiKey = Env.EnvOr("OR_API_KEY", "") },
            new() { Key = "ocz", Label = "OCZ", BaseUrl = Env.EnvOr("OCZ_BASE_URL", "https://opencode.ai/zen/v1").TrimEnd('/'),
                // same OpenCode workspace key serves both zen endpoints — default to the OCG key
                ApiKey = Env.EnvOr("OCZ_API_KEY", Env.EnvOr("OCG_API_KEY", "")) },
            new() { Key = "ocg", Label = "OCG", BaseUrl = Env.EnvOr("OCG_BASE_URL", "").TrimEnd('/'), ApiKey = Env.EnvOr("OCG_API_KEY", "") },
        };
        if (LocalEnabled)
        {
            list.Add(new Provider
            {
                Key = "local", Label = "LOCAL",
                BaseUrl = Env.EnvOr("LOCAL_BASE_URL", Env.LocalCtrlUrl).TrimEnd('/'),
                ApiKey = Env.EnvOr("LOCAL_API_KEY", ""), // no auth — LAN trust domain
            });
        }
        return list;
    }

    public static Provider? ByKey(string key) => All.FirstOrDefault(p => p.Key == key);

    public static Dictionary<string, string> Headers(Provider p) => new()
    {
        ["Content-Type"] = "application/json",
        ["Authorization"] = $"Bearer {p.ApiKey}",
    };

    /// <summary>Cached set of model ids a provider serves (null = couldn't verify).</summary>
    public static async Task<HashSet<string>?> ProviderModelsAsync(string key)
    {
        var now = DateTime.UtcNow;
        if (ModelsCache.TryGetValue(key, out var hit) && now - hit.At < TimeSpan.FromSeconds(Env.ModelsCacheTtl))
            return hit.Ids;

        var p = ByKey(key);
        HashSet<string>? ids = null;
        if (p is not null)
        {
            try
            {
                using var req = new HttpRequestMessage(HttpMethod.Get, $"{p.BaseUrl}/models");
                if (!string.IsNullOrEmpty(p.ApiKey)) req.Headers.TryAddWithoutValidation("Authorization", $"Bearer {p.ApiKey}");
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
                using var resp = await p.Client.SendAsync(req, cts.Token);
                resp.EnsureSuccessStatusCode();
                var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(cts.Token));
                ids = new HashSet<string>();
                if (doc.RootElement.TryGetProperty("data", out var data) && data.ValueKind == JsonValueKind.Array)
                    foreach (var m in data.EnumerateArray())
                        if (m.TryGetProperty("id", out var id) && id.ValueKind == JsonValueKind.String)
                            ids.Add(id.GetString()!);
            }
            catch
            {
                ids = null;
            }
        }
        ModelsCache[key] = (now, ids);
        return ids;
    }

    /// <summary>Recent reachability + latency probe per provider (30s-cached — for the status UI).</summary>
    public sealed record ProvProbe(bool Reachable, long LatencyMs, int ModelCount, DateTime At);

    static readonly ConcurrentDictionary<string, ProvProbe> ProbeCache = new();

    public static async Task<ProvProbe?> ProbeAsync(string key)
    {
        var now = DateTime.UtcNow;
        if (ProbeCache.TryGetValue(key, out var hit) && now - hit.At < TimeSpan.FromSeconds(30))
            return hit;

        var p = ByKey(key);
        ProvProbe? probe = null;
        if (p is not null)
        {
            var at = DateTime.UtcNow;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                using var req = new HttpRequestMessage(HttpMethod.Get, $"{p.BaseUrl}/models");
                if (!string.IsNullOrEmpty(p.ApiKey)) req.Headers.TryAddWithoutValidation("Authorization", $"Bearer {p.ApiKey}");
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
                using var resp = await p.Client.SendAsync(req, cts.Token);
                sw.Stop();
                if (resp.IsSuccessStatusCode)
                {
                    var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(cts.Token));
                    var n = 0;
                    if (doc.RootElement.TryGetProperty("data", out var data) && data.ValueKind == JsonValueKind.Array)
                        n = data.GetArrayLength();
                    probe = new ProvProbe(true, sw.ElapsedMilliseconds, n, at);
                }
                else
                {
                    probe = new ProvProbe(false, sw.ElapsedMilliseconds, 0, at);
                }
            }
            catch
            {
                sw.Stop();
                probe = new ProvProbe(false, sw.ElapsedMilliseconds, 0, at);
            }
            ProbeCache[key] = probe;
        }
        return probe;
    }

    /// <summary>Ordered (provider_key, upstream_model) chain for a model name.</summary>
    public static async Task<List<(string Key, string Upstream)>> ResolveAsync(string? model)
    {
        if (string.IsNullOrWhiteSpace(model))
            throw new ApiError(400, "model is required");
        var m = model!.Trim();
        var low = m.ToLowerInvariant();

        if (low == "auto")
        {
            // Explicit ordered fallback chain from AUTO_CHAIN (comma-separated
            // "provider:model" legs, tried in sequence), then the local GPU as
            // FINAL fallback (if enabled; serves LOCAL_MODEL, not a chain model).
            // Each leg goes through ResolveAutoAsync: OCZ's ids differ from OCG's
            // (e.g. ox-alpha-free -> x-preview-f-free), and a leg whose provider
            // verifiably doesn't serve its model is skipped so the chain falls
            // through instead of dying on an upstream 404 (not in FallbackStatus).
            var chain = new List<(string Key, string Upstream)>();
            foreach (var (k, want) in AutoChain)
            {
                var up = await ResolveAutoAsync(k, want);
                if (up is not null) chain.Add((k, up));
            }
            // Last resort: if every explicit leg was skipped, relay the raw chain
            // so the request still attempts them (upstream errors surface clearly).
            if (chain.Count == 0)
                chain = AutoChain.Select(l => (l.Key, l.Model)).ToList();
            if (LocalEnabled) chain.Add(("local", Env.LocalModel));
            return chain;
        }

        foreach (var key in new[] { "or", "ocz", "ocg", "local" })
        {
            if (!low.StartsWith(key + "-")) continue;
            var upstream = m[(key.Length + 1)..].Trim();
            if (upstream.Length == 0)
                throw new ApiError(400, $"model '{m}': missing upstream model after '{key}-'");
            return new List<(string, string)> { (key, upstream) };
        }

        throw new ApiError(404,
            $"unknown model '{m}'. Use a provider prefix (OR-/OCZ-/OCG-/LOCAL-) or 'auto'.");
    }

    /// <summary>Fire one request at one provider. stream=true -> SSE relay path.</summary>
    public static async Task<(HttpResponseMessage Resp, HttpClient Client)> UpstreamAsync(
        string key, string path, JsonObject body, bool stream)
    {
        var p = ByKey(key) ?? throw new ApiError(502, $"provider '{key}' not configured");
        var req = new HttpRequestMessage(HttpMethod.Post, $"{p.BaseUrl}{path}")
        {
            Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"),
        };
        if (!string.IsNullOrEmpty(p.ApiKey))
            req.Headers.TryAddWithoutValidation("Authorization", $"Bearer {p.ApiKey}");
        var resp = await p.Client.SendAsync(req, stream ? HttpCompletionOption.ResponseHeadersRead : HttpCompletionOption.ResponseContentRead);
        return (resp, p.Client);
    }

    public static bool IsThinkingModel(string name)
    {
        var low = (name ?? "").ToLowerInvariant();
        return low.Contains("deepseek") || low.Contains("glm") || low.Contains("reasoner") || low.Contains("think");
    }

    /// <summary>Cache reasoning_content from a chat.completions JSON body, keyed by tool_call ids.</summary>
    public static void CacheReasoningEcho(JsonNode? content)
    {
        if (content is not JsonObject obj) return;
        if (obj["choices"] is not JsonArray choices) return;
        foreach (var choice in choices)
        {
            var msg = choice?["message"];
            var rc = msg?["reasoning_content"]?.GetValue<string>();
            if (string.IsNullOrEmpty(rc)) continue;
            if (msg!["tool_calls"] is JsonArray tcs)
                foreach (var tc in tcs)
                {
                    var cid = tc?["id"]?.GetValue<string>();
                    if (!string.IsNullOrEmpty(cid)) CacheEcho(cid, rc);
                }
        }
    }

        static JsonObject? FirstChoice(JsonObject? obj)
    {
        if (obj?["choices"] is JsonArray arr && arr.Count > 0)
            return arr[0] as JsonObject;
        return null;
    }

    /// <summary>Harvest reasoning_content + tool_call ids from a streaming SSE body.</summary>
    public static void CacheReasoningEchoFromSse(byte[] raw)
    {
        string pending = "";
        foreach (var line in Encoding.UTF8.GetString(raw).Split('\n'))
        {
            var trimmed = line.Trim();
            if (!trimmed.StartsWith("data:")) continue;
            var data = trimmed[5..].Trim();
            if (data.Length == 0 || data == "[DONE]") continue;
            JsonObject? obj;
            try { obj = JsonNode.Parse(data) as JsonObject; } catch { continue; }
            if (obj is null) continue;
            var delta = FirstChoice(obj)?["delta"];
            var rc = delta?["reasoning_content"]?.GetValue<string>();
            if (!string.IsNullOrEmpty(rc)) pending += rc;
            if (delta?["tool_calls"] is JsonArray tcs)
                foreach (var tc in tcs)
                {
                    var cid = tc?["id"]?.GetValue<string>();
                    if (!string.IsNullOrEmpty(cid) && pending.Length > 0) CacheEcho(cid, pending);
                }
            if (FirstChoice(obj)?["finish_reason"] is not null) pending = "";
        }
    }

    static void CacheEcho(string cid, string rc)
    {
        lock (EchoLock)
        {
            if (!ReasoningEcho.ContainsKey(cid)) EchoOrder.Add(cid);
            ReasoningEcho[cid] = rc;
            while (ReasoningEcho.Count > ReasoningEchoMax && EchoOrder.Count > 0)
            {
                var oldest = EchoOrder[0];
                EchoOrder.RemoveAt(0);
                ReasoningEcho.Remove(oldest);
            }
        }
    }

    /// <summary>Guarantee assistant tool_call messages carry reasoning_content when the
    /// upstream model is a thinking model. Injects cached reasoning or an empty echo.</summary>
    public static void EnsureReasoningEcho(JsonArray? messages, bool thinking)
    {
        if (!thinking || messages is null) return;
        foreach (var node in messages)
        {
            var msg = node as JsonObject;
            if (msg is null) continue;
            if (msg["role"]?.GetValue<string>() != "assistant") continue;
            if (msg["tool_calls"] is not JsonArray) continue;
            if (msg["reasoning_content"] is not null) continue; // client already echoed it
            var injected = "";
            foreach (var tc in msg["tool_calls"] as JsonArray ?? [])
            {
                var cid = tc?["id"]?.GetValue<string>();
                if (cid is null) continue;
                lock (EchoLock)
                {
                    if (ReasoningEcho.TryGetValue(cid, out var rc)) { injected = rc; break; }
                }
            }
            msg["reasoning_content"] = injected;
        }
    }
}
