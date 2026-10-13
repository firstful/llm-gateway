using System.Collections.Concurrent;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace LlmGateway;

/// <summary>
/// Per-model outcome counters, accumulated in-process as requests complete.
/// Deliberately in-memory rather than derived from the request log: the
/// scorer and the request path share a process, so there is nothing to
/// re-read and nothing to rotate.
/// </summary>
public static class ModelMetrics
{
    sealed class Counter
    {
        public long Ok, Fail, RateLimited, Reqs;
        public double Usd;
        public long Prompt, Completion, Cached;
    }

    static readonly ConcurrentDictionary<string, Counter> Map = new();

    static Counter Get(string m) =>
        Map.GetOrAdd(m, _ => new Counter());

    public static void Success(string model, long prompt, long completion, long cached, double usd)
    {
        var c = Get(model);
        Interlocked.Increment(ref c.Ok);
        Interlocked.Increment(ref c.Reqs);
        Interlocked.Add(ref c.Prompt, prompt);
        Interlocked.Add(ref c.Completion, completion);
        Interlocked.Add(ref c.Cached, cached);
        lock (c) { c.Usd += usd; }
    }

    public static void Failure(string model, int status)
    {
        var c = Get(model);
        Interlocked.Increment(ref c.Fail);
        if (status == 429) Interlocked.Increment(ref c.RateLimited);
    }

    public sealed record Snapshot(long Ok, long Fail, long RateLimited, long Reqs,
        double Usd, long Prompt, long Completion, long Cached);

    public static IReadOnlyDictionary<string, Snapshot> All() =>
        Map.ToDictionary(kv => kv.Key, kv =>
        {
            var c = kv.Value;
            double u; lock (c) { u = c.Usd; }
            return new Snapshot(c.Ok, c.Fail, c.RateLimited, c.Reqs, u,
                                c.Prompt, c.Completion, c.Cached);
        });
}

/// <summary>
/// Ranked model list for OpenCode Go, recomputed in the background.
///
/// The serving path is deliberately dumb: it walks <see cref="Current"/>
/// top-to-bottom and uses the first model that works. All intelligence —
/// roster, pricing, quota pressure, observed reliability — lives here.
///
/// Sources are authoritative JSON only. Nothing is scraped from HTML or
/// markdown: the docs scraper could not tell a base rate card from a
/// long-context tier or a peak multiplier, and silently reported the wrong
/// one. models.opencode.ai/api.json publishes cost including context tiers.
/// </summary>
public sealed class ModelScorerService : BackgroundService
{
    const string ApiJson = "https://models.opencode.ai/api.json";
    static readonly Regex Blacklist =
        new(@"muse.*spark.*contributor", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    // tunables
    const int RefreshMinutes = 5;
    const int MinReliabilitySample = 40;
    const double ReliabilityPrior = 0.90;
    const double MinReliability = 0.75;   // below this a leg wastes a round trip
    const double CostWeightBase = 0.25;   // quota comfortable
    const double CostWeightMax = 0.75;    // quota nearly exhausted

    sealed class Rate
    {
        public string Model = "";
        public double Input, Output, CacheRead;
        public List<(long Size, double In, double Out, double Cache)> Tiers = new();
        public bool IsFree => Input == 0 && Output == 0;
    }

    sealed class Scored
    {
        public string Model = "";
        public bool Free;
        public long N, Ok, Fail;
        public double Reliability, UsdPerReq, Score;
    }

    /// <summary>Immutable scorer state, published as a single reference so
    /// readers never observe a half-updated ranking (a torn order array would
    /// route a request to a model the scorer just demoted).</summary>
    sealed class State
    {
        public Rate[] Rates = Array.Empty<Rate>();
        public string[] Order = Array.Empty<string>();
        public JsonNode? Quota;
        public double CostWeight;
        public DateTimeOffset LastOk = DateTimeOffset.MinValue;
        public string LastError = "";
    }

    static volatile State _state = new();
    static State S => _state;

    public static string[] Current => _state.Order;
    public static JsonNode? Quota => _state.Quota;
    public static double CostWeight => _state.CostWeight;
    public static DateTimeOffset LastOk => _state.LastOk;
    public static string LastError => _state.LastError;

    static readonly HttpClient Http = Providers.ByKey("ocg")?.Client
        ?? new HttpClient();

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        // First pass immediately so the list exists before traffic arrives.
        while (!ct.IsCancellationRequested)
        {
            try { await Refresh(ct); }
            catch (Exception e)
            {
                var cur = _state; cur.LastError = e.GetType().Name + ": " + e.Message;
            }
            try { await Task.Delay(TimeSpan.FromMinutes(RefreshMinutes), ct); }
            catch (OperationCanceledException) { return; }
        }
    }

    static async Task<JsonNode?> GetJson(string url, string? bearer)
    {
        using var req = new HttpRequestMessage(HttpMethod.Get, url);
        req.Headers.TryAddWithoutValidation("User-Agent", "llm-gateway-scorer/1.0");
        if (!string.IsNullOrEmpty(bearer))
            req.Headers.TryAddWithoutValidation("Authorization", "Bearer " + bearer);
        using var resp = await Http.SendAsync(req);
        if (!resp.IsSuccessStatusCode) return null;
        return JsonNode.Parse(await resp.Content.ReadAsStringAsync());
    }

    /// <summary>True when the rate card prices this model at zero. The free
    /// tier is a separate pool (flat daily cap, no USD windows), so `free`
    /// selects from it without needing its own chain.</summary>
    public static bool IsFree(string model)
    {
        var r = _state.Rates.FirstOrDefault(x => x.Model == model);
        return r is not null && r.IsFree;
    }

    /// <summary>Tier-aware cost. Picks the highest context threshold the request
    /// actually meets, matching how the vendor bills.</summary>
    public static double Cost(string model, long prompt, long completion, long cached)
    {
        var r = _state.Rates.FirstOrDefault(x => x.Model == model);
        if (r is null) return 0;
        double cin = r.Input, cout = r.Output, cc = r.CacheRead;
        foreach (var t in r.Tiers.OrderByDescending(t => t.Size))
            if (prompt >= t.Size) { cin = t.In; cout = t.Out; cc = t.Cache; break; }
        return (prompt * cin + cached * cc + completion * cout) / 1e6;
    }

    static (Rate Rate, string Name)[] ParseRoster(JsonNode? root)
    {
        var go = root?["opencode-go"]?["models"] as JsonObject;
        if (go is null) return Array.Empty<(Rate, string)>();
        var outp = new List<(Rate, string)>();
        foreach (var (id, node) in go)
        {
            var c = node?["cost"] as JsonObject;
            if (c is null) continue;
            var r = new Rate
            {
                Model = id,
                Input = c["input"]?.GetValue<double>() ?? 0,
                Output = c["output"]?.GetValue<double>() ?? 0,
                CacheRead = c["cache_read"]?.GetValue<double>() ?? 0,
            };
            if (c["tiers"] is JsonArray tiers)
                foreach (var t in tiers)
                {
                    var size = t?["tier"]?["size"]?.GetValue<long>() ?? 0;
                    if (size <= 0) continue;
                    r.Tiers.Add((size,
                        t?["input"]?.GetValue<double>() ?? r.Input,
                        t?["output"]?.GetValue<double>() ?? r.Output,
                        t?["cache_read"]?.GetValue<double>() ?? r.CacheRead));
                }
            outp.Add((r, id));
        }
        return outp.ToArray();
    }

    async Task Refresh(CancellationToken ct)
    {
        var roster = ParseRoster(await GetJson(ApiJson, null));
        if (roster.Length == 0)
            throw new InvalidOperationException("api.json returned no opencode-go models");
        var st = new State { Rates = roster.Select(r => r.Rate).ToArray() };

        var ocg = Providers.ByKey("ocg");
        var served = await Providers.ProviderModelsAsync("ocg");
        var live = new HashSet<string>(
            (served ?? new HashSet<string>()).Select(m =>
                m.StartsWith("OCG-") ? m[4..] : m));

        st.Quota = await GetJson($"{ocg?.BaseUrl}/usage", ocg?.ApiKey);

        // worst window dominates; all denominated in USD
        var pcts = new List<double>();
        if (st.Quota?["usage"] is JsonObject u)
            foreach (var (_, w) in u)
            {
                var p = w?["percent"]?.GetValue<double>();
                if (p is not null) pcts.Add(p.Value);
            }
        double pressure = pcts.Count > 0 ? pcts.Max() / 100.0 : 0;
        st.CostWeight = CostWeightBase +
            (CostWeightMax - CostWeightBase) * Math.Min(1.0, pressure / 0.8);

        var obs = ModelMetrics.All();
        var rows = new List<Scored>();
        foreach (var rate in _state.Rates)
        {
            if (live.Count > 0 && !live.Contains(rate.Model)) continue;
            if (Blacklist.IsMatch(rate.Model)) continue;
            obs.TryGetValue(rate.Model, out var s);
            long ok = s?.Ok ?? 0, fail = s?.Fail ?? 0, n = ok + fail;
            double rel = n >= MinReliabilitySample ? (double)ok / n : ReliabilityPrior;

            // Cold start: price a never-called model off the rate card using the
            // median observed token shape. Without this every unobserved model
            // ties and the ranking collapses to alphabetical order.
            double perReq;
            if ((s?.Reqs ?? 0) >= 10) perReq = s!.Usd / s.Reqs;
            else
            {
                long mp = Median(obs.Values.Select(v => v.Prompt));
                long mc = Median(obs.Values.Select(v => v.Completion));
                long mk = Median(obs.Values.Select(v => v.Cached));
                perReq = (mp * rate.Input + mk * rate.CacheRead + mc * rate.Output) / 1e6;
            }

            rows.Add(new Scored
            {
                Model = rate.Model,
                Free = rate.IsFree,
                N = n,
                Ok = ok,
                Fail = fail,
                Reliability = rel,
                UsdPerReq = perReq,
            });
        }

        var viable = rows.Where(r => r.Reliability >= MinReliability).ToList();
        if (viable.Count == 0) viable = rows;   // never emit an empty list

        double refCost = viable.Where(r => !r.Free && r.UsdPerReq > 0)
                               .Select(r => r.UsdPerReq).DefaultIfEmpty(0).Max();
        if (refCost <= 0) refCost = 1e-9;
        foreach (var r in viable)
        {
            double cheap = r.Free ? 1.0 : Math.Max(0.0, 1.0 - r.UsdPerReq / refCost);
            r.Score = (1 - st.CostWeight) * r.Reliability + st.CostWeight * cheap;
        }

        // free tier is a separate pool (flat daily cap, no USD windows) so it
        // leads; within each pool, score order.
        viable.Sort((a, b) => a.Free != b.Free
            ? (a.Free ? -1 : 1)
            : b.Score.CompareTo(a.Score));

        st.Order = viable.Select(r => r.Model).ToArray();
        st.LastOk = DateTimeOffset.UtcNow;
        st.LastError = "";
        _state = st;
    }

    static long Median(IEnumerable<long> xs)
    {
        var a = xs.Where(x => x > 0).OrderBy(x => x).ToArray();
        return a.Length == 0 ? 0 : a[a.Length / 2];
    }

    /// <summary>Score table for /status.json.</summary>
    public static JsonArray Describe()
    {
        var arr = new JsonArray();
        var obs = ModelMetrics.All();
        var byModel = _state.Rates.ToDictionary(r => r.Model, r => r);
        foreach (var m in _state.Order)
        {
            obs.TryGetValue(m, out var s);
            long ok = s?.Ok ?? 0, fail = s?.Fail ?? 0, n = ok + fail;
            double rel = n >= MinReliabilitySample ? (double)ok / n : ReliabilityPrior;
            byModel.TryGetValue(m, out var rate);
            arr.Add(new JsonObject
            {
                ["model"] = m,
                ["free"] = rate?.IsFree ?? false,
                ["observed"] = n,
                ["ok"] = ok,
                ["fail"] = fail,
                ["rate_limited"] = s?.RateLimited ?? 0,
                ["reliability"] = Math.Round(rel, 4),
                ["est_usd_per_req"] = rate is null
                    ? null
                    : Math.Round((s?.Reqs ?? 0) >= 10 ? s!.Usd / s.Reqs : Cost(m, 0, 0, 0), 8),
            });
        }
        return arr;
    }
}
