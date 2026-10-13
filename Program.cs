using System.Text.Json.Nodes;
using LlmGateway;

var builder = WebApplication.CreateBuilder(args);
builder.Logging.SetMinimumLevel(LogLevel.Warning);
builder.Services.AddHostedService<LiveFlusherService>();
builder.Services.AddHostedService<StatsFlusherService>();
builder.Services.AddHostedService<ModelScorerService>();

// PORT env parity with the Python version (ASPNETCORE_URLS wins if set)
var port = Environment.GetEnvironmentVariable("PORT");
if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ASPNETCORE_URLS")))
{
    builder.WebHost.UseUrls($"http://+:{port ?? "8080"}");
}

var app = builder.Build();

// ---------------- error handling ----------------
app.Use(async (ctx, next) =>
{
    try { await next(); }
    catch (ApiError e)
    {
        ctx.Response.StatusCode = e.Status;
        ctx.Response.ContentType = "application/json";
        var body = new JsonObject
        {
            ["error"] = new JsonObject
            {
                ["message"] = e.Message,
                ["type"] = "invalid_request_error",
                ["code"] = null,
            },
        };
        await ctx.Response.WriteAsync(body.ToJsonString());
    }
});

Sessions.Init();
var uiPath = Path.Combine(AppContext.BaseDirectory, "ui.html");
var statusPath = Path.Combine(AppContext.BaseDirectory, "status.html");

// ---------------- chat UI ----------------
app.MapGet("/", () => Results.File(uiPath, "text/html"));
app.MapGet("/ui", () => Results.File(uiPath, "text/html"));
app.MapGet("/status", () => Results.File(statusPath, "text/html"));

// ---------------- health ----------------
app.MapGet("/health", () => new JsonObject
{
    ["status"] = "ok",
    ["app"] = Env.AppName,
    ["providers"] = new JsonArray(Providers.All.Select(p => (JsonNode)p.Key).ToArray()),
    ["rag"] = new JsonObject { ["enabled"] = Env.RagEnabled, ["base_url"] = Env.RagBaseUrl },
    ["live_ingest"] = new JsonObject { ["enabled"] = Env.LiveIngestEnabled, ["queued"] = Rag.LiveQueue.Reader.Count },
});

// ---------------- request statistics ----------------
app.MapGet("/stats", (int limit = 20) => Results.Json(Stats.GetStats(limit)));

// ---------------- live local-LLM status (cached 5s) ----------------
var localStatus = new LocalStatusCache();
app.MapGet("/local/status", () => localStatus.Get());

// ---------------- provider availability for the UI status bar ----------------
app.MapGet("/local/providers", async () =>
{
    var outObj = new JsonObject();
    foreach (var p in Providers.All)
    {
        var served = await Providers.ProviderModelsAsync(p.Key);
        outObj[p.Key] = new JsonObject
        {
            ["label"] = p.Label,
            ["reachable"] = served is not null,
            ["models"] = served is null
                ? new JsonArray()
                : new JsonArray(served.OrderBy(x => x, StringComparer.Ordinal).Select(s => (JsonNode)s).ToArray()),
        };
    }
    return Results.Json(outObj);
});

// ---------------- provider status (JSON feed for /status) ----------------
app.MapGet("/status.json", async () =>
{
    var prov = new JsonObject();
    foreach (var p in Providers.All)
    {
        var pr = await Providers.ProbeAsync(p.Key);
        var models = p.Key == "local" ? null : await Providers.ProviderModelsAsync(p.Key);
        prov[p.Key] = new JsonObject
        {
            ["key"] = p.Key,
            ["label"] = p.Label,
            ["reachable"] = pr?.Reachable ?? false,
            ["latency_ms"] = pr?.LatencyMs ?? 0,
            ["model_count"] = pr?.ModelCount ?? 0,
            ["models"] = models is null
                ? new JsonArray()
                : new JsonArray(models.OrderBy(x => x, StringComparer.Ordinal).Select(s => (JsonNode)s).ToArray()),
        };
    }

    var recent = ReqLog.Recent(Env.ReqLogWindowSeconds);
    var provRec = new JsonObject();
    foreach (var (k, v) in recent.Providers.OrderBy(kv => kv.Key))
        provRec[k] = new JsonObject
        {
            ["ok"] = v.Ok,
            ["fail"] = v.Fail,
            ["last_error"] = ReqLog.Trunc(v.LastError),
            ["last_upstream"] = v.LastUpstream,
        };

    return Results.Json(new JsonObject
    {
        ["ts"] = DateTimeOffset.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'"),
        ["app"] = Env.AppName,
        ["version"] = Env.AppVersion,
        ["providers"] = prov,
        ["scorer"] = new JsonObject
        {
            ["ready"] = ModelScorerService.Current.Length > 0,
            ["last_refresh"] = ModelScorerService.LastOk == DateTimeOffset.MinValue
                ? null : ModelScorerService.LastOk.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'"),
            ["last_error"] = ModelScorerService.LastError == "" ? null : ModelScorerService.LastError,
            ["cost_weight"] = ModelScorerService.CostWeight,
            ["quota"] = ModelScorerService.Quota?["usage"]?.DeepClone(),
            ["order"] = new JsonArray(ModelScorerService.Current.Select(m => (JsonNode)m!).ToArray()),
            ["models"] = ModelScorerService.Describe(),
        },
        ["recent"] = new JsonObject
        {
            ["window_s"] = Env.ReqLogWindowSeconds,
            ["requests"] = recent.Requests,
            ["errors"] = recent.Errors,
            ["providers"] = provRec,
        },
    });
});

// ---------------- chat sessions (cookie-keyed, v1.7.0) ----------------
app.MapGet("/v1/session", (HttpContext ctx) =>
{
    var sid = ctx.Request.Cookies[Env.SessionCookie];
    if (string.IsNullOrEmpty(sid))
    {
        sid = Guid.NewGuid().ToString("N");
        var data = new JsonObject { ["id"] = sid, ["messages"] = new JsonArray(), ["model"] = "auto", ["new"] = true };
        ctx.Response.Cookies.Append(Env.SessionCookie, sid, new CookieOptions
        {
            MaxAge = TimeSpan.FromDays(Env.SessionTtlDays),
            HttpOnly = true,
            SameSite = SameSiteMode.Lax,
            Path = "/",
        });
        return Results.Json(data);
    }
    var loaded = Sessions.Load(sid);
    var (messages, model) = loaded is { } t ? (t.Messages ?? new JsonArray(), t.Model) : (new JsonArray(), "auto");
    return Results.Json(new JsonObject
    {
        ["id"] = sid,
        ["messages"] = messages,
        ["model"] = model,
        ["new"] = false,
    });
});

app.MapPost("/v1/session/clear", (HttpContext ctx) =>
{
    var sid = ctx.Request.Cookies[Env.SessionCookie];
    if (!string.IsNullOrEmpty(sid)) Sessions.Delete(sid);
    return Results.Json(new JsonObject { ["ok"] = true });
});

// ---------------- /v1/models ----------------
app.MapGet("/v1/models", async () =>
{
    var ids = new List<string> { "auto", "free" };
    foreach (var p in Providers.All)
    {
        var served = await Providers.ProviderModelsAsync(p.Key);
        if (served is null) served = new HashSet<string> { Env.AutoModel }; // can't verify -> advertise default only
        foreach (var mid in served.OrderBy(x => x, StringComparer.Ordinal))
            ids.Add($"{p.Label}-{mid}");
    }
    // de-dup preserving order
    var seen = new HashSet<string>();
    var data = new JsonArray();
    foreach (var mid in ids)
    {
        if (!seen.Add(mid)) continue;
        data.Add(new JsonObject
        {
            ["id"] = mid,
            ["object"] = "model",
            ["created"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            ["owned_by"] = "llm-gateway",
        });
    }
    return Results.Json(new JsonObject { ["object"] = "list", ["data"] = data });
});

// ---------------- chat + responses ----------------
app.MapPost("/v1/chat/completions", Chat.ChatCompletions);
app.MapPost("/v1/responses", Chat.Responses);

app.Run();

/// <summary>Best-effort snapshot of llama-ctrl /status (cached 5s; null = unreachable).</summary>
sealed class LocalStatusCache
{
    static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(3) };
    JsonNode? _cache;
    DateTime _at;

    public JsonObject Get()
    {
        var now = DateTime.UtcNow;
        if (_cache is null || now - _at > TimeSpan.FromSeconds(5))
        {
            try
            {
                using var resp = Http.GetAsync($"{Env.LocalCtrlUrl}/status").GetAwaiter().GetResult();
                resp.EnsureSuccessStatusCode();
                _cache = JsonNode.Parse(resp.Content.ReadAsStringAsync().GetAwaiter().GetResult());
            }
            catch { _cache = null; }
            _at = now;
        }
        if (_cache is null)
            return new JsonObject { ["reachable"] = false, ["detail"] = "ctrl unreachable" };
        var obj = (_cache as JsonObject)?.DeepClone() as JsonObject ?? new JsonObject();
        obj["reachable"] = true;
        return obj;
    }
}
