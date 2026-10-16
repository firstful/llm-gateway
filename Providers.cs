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
///   auto / free   -> the scorer-ranked list (free tier first, gated on
///                     reliability, speed and the OpenCode Go quota windows);
///                     the serving path just walks it top to bottom
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
    // can't be verified. `auto`/`free` routing is owned entirely by
    // ModelScorerService — there is no hand-maintained chain any more.
    public static string AutoModel { get; } = EnvOr("AUTO_MODEL", "mimo-v2.5");
    public static string LocalModel { get; } = EnvOr("LOCAL_MODEL", "lfm2.5-2.6b");
    public static int ModelsCacheTtl { get; } = IntOr("MODELS_CACHE_TTL", 300);

    // Operator model pin (see ModelPin). The pin file sits beside the session
    // store so a pin survives a service restart but still expires on its TTL.
    public static string PinFile { get; } = EnvOr("PIN_FILE", "/data/model-pin.json");
    // Guards the admin endpoints that change routing. Deliberately NOT optional:
    // if it is unset those endpoints refuse to run rather than defaulting open,
    // because an unauthenticated endpoint that repoints every LLM request on
    // the LAN is a real privilege, not a convenience.
    public static string AdminToken { get; } = EnvOr("ADMIN_TOKEN", "");

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
    // Rolling-buffer cap on the session store. The blob-per-session layout grew
    // to 3.4 GB on a box with 3.4 GiB RAM; oldest sessions are dropped wholesale
    // once the file crosses this, and space is returned to the OS.
    public static int SessionDbMaxMb { get; } = IntOr("SESSION_DB_MAX_MB", 1024);

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

    /// <summary>Half-life, in hours, of a model's observed ok/fail history. After
    /// this long a past outcome counts half as much as it did, so a model that
    /// has genuinely recovered is not still disqualified by last month's 429s.
    /// 0 disables decay and restores strictly cumulative counters.</summary>
    public static TimeSpan MetricsHalfLife { get; } =
        TimeSpan.FromHours(Math.Max(0, IntOr("METRICS_HALF_LIFE_HOURS", 24)));
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

    // fallback on these upstream outcomes when walking the scored ranking
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

    /// <summary>The scorer's ranked chain for an alias, best first. Empty when
    /// the background scorer has not completed its first pass yet.</summary>
    static List<(string Key, string Upstream)> Ranked(string low)
    {
        var ranked = ModelScorerService.Current;
        if (ranked.Length == 0) return new List<(string, string)>();
        var picks = low == "free" ? Array.FindAll(ranked, ModelScorerService.IsFree) : ranked;
        return picks.Select(m => (Key: "ocg", Upstream: m)).ToList();
    }

    /// <summary>Ordered (provider_key, upstream_model) route list for a model name.
    /// Synchronous: the only thing it used to await was the removed chain's
    /// per-leg upstream resolution, and nothing since depends on that being async.
    /// </summary>
    public static Task<List<(string Key, string Upstream)>> ResolveAsync(string? model)
    {
        if (string.IsNullOrWhiteSpace(model))
            throw new ApiError(400, "model is required");
        var m = model!.Trim();
        var low = m.ToLowerInvariant();

        if (low == "auto" || low == "free")
        {
            // An operator pin wins over the scorer, and is strict by default so
            // that a failure surfaces instead of being papered over by a
            // fall-through to the model we were trying not to use.
            var pin = ModelPin.Active;
            if (pin is not null)
            {
                var pinned = new List<(string, string)> { ("ocg", pin.Model) };
                if (pin.Fallback)
                {
                    foreach (var r in Ranked(low))
                        if (!pinned.Any(c => c.Item2 == r.Upstream)) pinned.Add(r);
                }
                return Task.FromResult<List<(string, string)>>(pinned);
            }

            // The scorer owns the ranked list; the serving path is a dumb
            // executor that walks it top-to-bottom until one model works.
            // OpenCode Go only -- OpenRouter and OpenCode Zen were dropped.
            var ranked = Ranked(low);
            if (ranked.Count > 0) return Task.FromResult<List<(string, string)>>(ranked);

            // No ranked list yet: the scorer has not completed its first pass.
            // There is no chain to fall back on any more -- the scorer retries
            // its first pass at startup precisely so this window stays tiny --
            // so fail loudly rather than serving an empty route list.
            throw new ApiError(503,
                $"model '{m}': the scorer's first ranking pass has not completed yet; retry in a few seconds");
        }

        foreach (var key in new[] { "or", "ocz", "ocg", "local" })
        {
            if (!low.StartsWith(key + "-")) continue;
            var upstream = m[(key.Length + 1)..].Trim();
            if (upstream.Length == 0)
                throw new ApiError(400, $"model '{m}': missing upstream model after '{key}-'");
            return Task.FromResult<List<(string, string)>>(new() { (key, upstream) });
        }

        throw new ApiError(404,
            $"unknown model '{m}'. Use a provider prefix (OR-/OCZ-/OCG-/LOCAL-), 'auto', or 'free'.");
    }

    /// <summary>Fire one request at one provider. stream=true -> SSE relay path.</summary>
    public static async Task<(HttpResponseMessage Resp, HttpClient Client)> UpstreamAsync(
        string key, string path, JsonObject body, bool stream, string? sessionId = null)
    {
        var p = ByKey(key) ?? throw new ApiError(502, $"provider '{key}' not configured");
        var req = new HttpRequestMessage(HttpMethod.Post, $"{p.BaseUrl}{path}")
        {
            Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"),
        };
        if (!string.IsNullOrEmpty(p.ApiKey))
            req.Headers.TryAddWithoutValidation("Authorization", $"Bearer {p.ApiKey}");
        // Forward session as x-opencode-session for upstream routing (OpenCode Go requires it)
        if (!string.IsNullOrEmpty(sessionId))
            req.Headers.TryAddWithoutValidation("x-opencode-session", sessionId);
        var resp = await p.Client.SendAsync(req, stream ? HttpCompletionOption.ResponseHeadersRead : HttpCompletionOption.ResponseContentRead);
        return (resp, p.Client);
    }

    /// <summary>
    /// OpenCode Go thinking models ([1210]) accept ONLY low|high|max.
    /// Hermes sends provider-agnostic values (none|minimal|low|medium|high|
    /// xhigh|max|ultra) — clamp to the nearest accepted level per leg.
    /// Fail-open: unknown/missing values are left untouched.
    /// </summary>
    public static void NormalizeReasoningEffort(JsonObject body, string providerKey)
    {
        if (providerKey is not ("ocg" or "ocz")) return;

        // OCG rejects message-level "name" fields ([invalid_request_error]
        // "name is not supported by this endpoint"). Strip them — the field is
        // optional decoration in chat.completions and OCG never uses it.
        if (body["messages"] is JsonArray msgs)
            foreach (var node in msgs)
                if (node is JsonObject msg && msg.ContainsKey("name"))
                    msg.Remove("name");

        // top-level reasoning_effort (chat.completions form)
        if (body["reasoning_effort"] is JsonValue v && v.TryGetValue<string>(out var e))
        {
            var clamped = e.Trim().ToLowerInvariant() switch
            {
                "none" or "minimal" or "medium" => "low",
                "xhigh" => "high",
                "ultra" => "max",
                var other when other is not ("low" or "high" or "max") => "low",
                _ => null, // already valid
            };
            if (clamped is not null) body["reasoning_effort"] = clamped;
        }
        // Responses-API form: "reasoning": {"effort": "..."}
        if (body["reasoning"] is JsonObject r && r["effort"] is JsonValue rv && rv.TryGetValue<string>(out var re))
        {
            var clamped = re.Trim().ToLowerInvariant() switch
            {
                "none" or "minimal" or "medium" => "low",
                "xhigh" => "high",
                "ultra" => "max",
                var other when other is not ("low" or "high" or "max") => "low",
                _ => null,
            };
            if (clamped is not null) r["effort"] = clamped;
        }
    }

    public static bool IsThinkingModel(string name)
    {
        if (string.IsNullOrEmpty(name)) return false;
        // Authoritative: api.json declares reasoning per model. 37 of 37 OpenCode
        // Go models are reasoning models, and 20 additionally require the
        // reasoning field replayed on earlier assistant turns.
        if (ModelScorerService.NeedsReasoningEcho(name)) return true;
        var low = name.ToLowerInvariant();
        return low.Contains("deepseek") || low.Contains("glm") || low.Contains("reasoner") || low.Contains("think");
    }

    /// <summary>Field this model requires on replayed assistant turns, or null.</summary>
    public static string? ReasoningField(string? name) =>
        string.IsNullOrEmpty(name) ? null : ModelScorerService.ReasoningField(name);

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
            if (msg["reasoning_content"] is not null) continue; // client already echoed it
            var toolCalls = msg["tool_calls"] as JsonArray;
            // Only tool-call turns can be matched to a cached reasoning block by
            // call id. For a plain assistant turn we have nothing keyed to it,
            // so leave the message alone rather than fabricate a field.
            if (toolCalls is null || toolCalls.Count == 0) continue;
            var injected = (string?)null;
            foreach (var tc in toolCalls)
            {
                var cid = tc?["id"]?.GetValue<string>();
                if (cid is null) continue;
                lock (EchoLock)
                {
                    if (ReasoningEcho.TryGetValue(cid, out var rc)) { injected = rc; break; }
                }
            }
            // Omit the key entirely when nothing was cached: a fabricated empty
            // string is a different claim from "no reasoning", and models that
            // validate the field reject the wrong one.
            if (injected is null) continue;
            msg["reasoning_content"] = injected;
        }
    }
}
