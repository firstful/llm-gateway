using System.Text.Json.Nodes;

namespace LlmGateway;

/// <summary>
/// External quality signals for the cold-start prior, from three public JSON
/// APIs. These answer "which model is any good" for models we have never
/// called; our own counters answer "does it actually work" for models we have.
///
/// SCALES ARE NOT COMPARABLE. OpenRouter's intelligence_index and publicai's
/// coding index are 0-100 standardised indices; kuryzhev's swe_bench is a
/// benchmark score that runs high across the board. deepseek-v4-flash reads
/// 24.4 / 55.2 / 62.0 on the three -- averaging those raw produces a confident,
/// meaningless number. Every source is therefore robustly normalised within its
/// own cohort before blending, so only *relative standing within that source*
/// is ever used.
///
/// COVERAGE DIFFERS PER MODEL. Only sources that actually carry a model
/// contribute to it, and their weights are renormalised; nothing is imputed.
/// Models seen by one source are shrunk toward the neutral midpoint in
/// proportion to how little evidence backs them.
/// </summary>
public static class QualitySources
{
    static readonly HttpClient Http = new(new SocketsHttpHandler
    {
        ConnectTimeout = TimeSpan.FromSeconds(10),
    })
    {
        Timeout = TimeSpan.FromSeconds(90),
        DefaultRequestHeaders =
        {
            { "User-Agent", "llm-gateway-quality/1.0" },
        },
    };

    // OR: best coverage and a documented field. PA: lowest coverage and
    // structurally capped at 100 rows, so it is only a third opinion.
    const double W_OR = 0.45;
    const double W_KQ = 0.35;   // swe_bench: most task-relevant for coding agents
    const double W_PA = 0.20;
    const double W_TTFT = 0.25; // UX term; renormalised separately

    public sealed class Entry
    {
        public string Model = "";
        public double Quality;          // 0..1, normalised
        public double Confidence;        // 0..1, share of total weight backing it
        public double TtftMs = -1;      // -1 = unknown
        public double Ux = 0.5;         // 0..1 normalised TTFT, higher = faster
        public int Sources;
        public JsonObject? RawDebug;
    }

    /// <summary>Strip org prefix, version stamps and preview/free noise, then
    /// remove all separators so 'glm-5.3' matches 'glm-5-3'. Every source uses a
    /// different slug convention; without this the join silently misses models
    /// and understates coverage.</summary>
    public static string Canon(string s)
    {
        s = s.ToLowerInvariant();
        var slash = s.LastIndexOf('/');
        if (slash >= 0) s = s[(slash + 1)..];
        s = System.Text.RegularExpressions.Regex.Replace(s, @"[-_]?\d{6,8}$", "");
        s = System.Text.RegularExpressions.Regex.Replace(
            s, @"[-_.]?(preview\d*|exp|experimental|free|beta)$", "");
        return System.Text.RegularExpressions.Regex.Replace(s, @"[^a-z0-9]", "");
    }

    static async Task<JsonNode?> Get(string url, CancellationToken ct)
    {
        try
        {
            using var resp = await Http.GetAsync(url, ct);
            if (!resp.IsSuccessStatusCode) return null;
            return JsonNode.Parse(await resp.Content.ReadAsStringAsync(ct));
        }
        catch { return null; }
    }

    /// <summary>
    /// Median/MAD standardisation. Chosen over mean/stdev because these cohorts
    /// are small (15-27 models) and skewed, and a couple of frontier outliers
    /// would otherwise dominate the scale for everyone else.
    /// </summary>
    static Dictionary<string, double> Normalise(Dictionary<string, (double V, bool Inv)> raw)
    {
        var outp = new Dictionary<string, double>();
        if (raw.Count == 0) return outp;

        // invert so that "higher is always better" before computing location
        var vals = raw.ToDictionary(k => k.Key, k => k.Value.Inv ? -k.Value.V : k.Value.V);
        var xs = vals.Values.OrderBy(v => v).ToArray();
        double Median(double[] a) => a[a.Length / 2];

        var med = Median(xs);
        var devs = xs.Select(v => Math.Abs(v - med)).OrderBy(v => v).ToArray();
        var mad = devs.Length == 0 ? 0 : Median(devs);
        // 1.4826 scales MAD to a stdev-equivalent for normal data.
        var denom = 1.4826 * mad;
        if (denom <= 1e-12)
        {
            // Degenerate spread (all equal): everything sits at neutral.
            foreach (var k in vals.Keys) outp[k] = 0.5;
            return outp;
        }
        foreach (var (k, v) in vals)
        {
            var z = Math.Clamp((v - med) / denom, -3.0, 3.0);
            outp[k] = (z + 3.0) / 6.0;   // -3..3 -> 0..1
        }
        return outp;
    }

    public static async Task<IReadOnlyDictionary<string, Entry>> Fetch(
        IReadOnlyCollection<string> roster, CancellationToken ct)
    {
        // raw[model] -> list of (normalisedValue, weight)
        var qual = new Dictionary<string, List<(double V, double W)>>();
        var ttftRaw = new Dictionary<string, double>();
        var debug = new Dictionary<string, JsonObject?>();

        void Add(string canon, double v, double w, JsonObject? d)
        {
            if (!qual.TryGetValue(canon, out var l)) qual[canon] = l = new();
            l.Add((v, w));
            if (d is not null) debug[canon] = d;
        }

        // ---- OpenRouter: benchmarks.artificial_analysis.intelligence_index
        // `artificial_analysis` is live but UNDOCUMENTED (docs only describe
        // design_arena), so every access is defensive and the whole source
        // disappears cleanly if the shape ever changes.
        var orBuf = new Dictionary<string, (double V, bool Inv)>();
        var orNode = await Get("https://openrouter.ai/api/v1/models", ct);
        if (orNode?["data"] is JsonArray orArr)
            foreach (var m in orArr)
            {
                var id = m?["id"]?.GetValue<string>();
                if (id is null) continue;
                var aa = m?["benchmarks"]?["artificial_analysis"] as JsonObject;
                var idx = aa?["intelligence_index"];
                if (idx is null) continue;
                double v;
                try { v = idx.GetValue<double>(); } catch { continue; }
                orBuf[Canon(id)] = (v, false);
            }
        var orNorm = Normalise(orBuf);
        foreach (var (k, v) in orNorm) Add(k, v, W_OR, null);

        // ---- kuryzhev: swe_bench (quality) + time_to_first_token_ms (UX)
        var kqBuf = new Dictionary<string, (double V, bool Inv)>();
        var kBench = await Get("https://llm.kuryzhev.cloud/api/benchmarks", ct);
        if (kBench?["benchmarks"] is JsonArray kArr)
            foreach (var b in kArr)
            {
                var id = b?["model_id"]?.GetValue<string>();
                var sw = b?["swe_bench"];
                if (id is null || sw is null) continue;
                double v;
                try { v = sw.GetValue<double>(); } catch { continue; }
                kqBuf[Canon(id)] = (v, false);
            }
        var kqNorm = Normalise(kqBuf);
        foreach (var (k, v) in kqNorm) Add(k, v, W_KQ, null);

        var speed = await Get("https://llm.kuryzhev.cloud/api/speed-metrics", ct);
        if (speed?["metrics"] is JsonArray sArr)
            foreach (var s in sArr)
            {
                var id = s?["model_id"]?.GetValue<string>();
                var tt = s?["time_to_first_token_ms"];
                if (id is null || tt is null) continue;
                double v;
                try { v = tt.GetValue<double>(); } catch { continue; }
                ttftRaw[Canon(id)] = v;   // lower is better
            }

        // ---- publicai: coding-scope index, lowest weight (41% coverage, and
        // the endpoint hard-caps at 100 rows so it is structurally blind to the
        // tail of the index).
        var paBuf = new Dictionary<string, (double V, bool Inv)>();
        try
        {
            var path = Environment.GetEnvironmentVariable("QUALITY_PA_URL");
            var paNode = path is null ? null : await Get(path, ct);
            if (paNode?["models"] is JsonArray paArr)
                foreach (var e in paArr)
                {
                    var id = e?["id"]?.GetValue<string>();
                    var ix = e?["index"];
                    if (id is null || ix is null) continue;
                    double v;
                    try { v = ix.GetValue<double>(); } catch { continue; }
                    paBuf[Canon(id)] = (v, false);
                }
        }
        catch { }
        var paNorm = Normalise(paBuf);
        foreach (var (k, v) in paNorm) Add(k, v, W_PA, null);

        // ---- blend
        // lower TTFT is better, so it is inverted before normalising
        var ttftInv = ttftRaw.ToDictionary(k => k.Key, k => (k.Value, Inv: true));
        var ttftNorm = Normalise(ttftInv);
        var totalQ = W_OR + W_KQ + W_PA;
        var outp = new Dictionary<string, Entry>();
        foreach (var model in roster)
        {
            var c = Canon(model);
            var e = new Entry { Model = model, RawDebug = debug.GetValueOrDefault(c) };

            if (qual.TryGetValue(c, out var parts) && parts.Count > 0)
            {
                double num = 0, den = 0;
                foreach (var (v, w) in parts) { num += v * w; den += w; }
                var mean = den > 0 ? num / den : 0.5;
                e.Confidence = Math.Min(1.0, den / totalQ);
                e.Sources = parts.Count;
                // Shrink toward neutral in proportion to how few sources back
                // this model. One weak source should not read as certain as
                // three corroborating ones.
                e.Quality = 0.5 + (mean - 0.5) * e.Confidence;
            }
            if (ttftNorm.TryGetValue(c, out var tv)) { e.TtftMs = ttftRaw[c]; e.Ux = tv; }
            outp[model] = e;
        }
        return outp;
    }
}
