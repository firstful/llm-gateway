using System.Text.Json.Nodes;

namespace LlmGateway;

/// <summary>Operator model override for `auto` / `free` routing.
///
/// Two properties matter more than the routing itself:
///
/// <para><b>It always carries a TTL.</b> A pin is a diagnostic instrument --
/// "does this model do the thing I think it does?" -- and an instrument left
/// switched on after the experiment quietly becomes the permanent
/// configuration, which is exactly the failure this endpoint exists to avoid.
/// The expiry is absolute and persisted, so a pin survives a restart of the
/// service but still releases on its own afterwards.</para>
///
/// <para><b>It is strict by default.</b> A pinned model becomes the entire
/// route list, so a failure surfaces as an error rather than falling through to
/// whatever the scorer would have picked next. A silent fallback would mask the
/// behaviour under test, which defeats the point. Pass <c>fallback: true</c> to
/// opt into appending the normal ranked chain behind it.</para>
///
/// <para>Explicitly requested models are never touched: pinning only
/// reinterprets <c>auto</c> and <c>free</c>.</para>
/// </summary>
public static class ModelPin
{
    public sealed record Pin(string Model, bool Fallback, DateTimeOffset Expires);

    static readonly object Lock = new();

    /// <summary>Last-known pin, re-evaluated for expiry on every read. A pin
    /// that has lapsed reads as absent and is cleared on the next touch, so
    /// nothing has to run a timer to honour it.</summary>
    static Pin? _pin;

    public const int DefaultTtlSeconds = 1800;
    public const int MaxTtlSeconds = 86400;

    /// <summary>The live pin, or null when unpinned or expired.</summary>
    public static Pin? Active
    {
        get
        {
            lock (Lock)
            {
                if (_pin is not null && DateTimeOffset.UtcNow >= _pin.Expires)
                {
                    _pin = null;
                    TrySave();
                }
                return _pin;
            }
        }
    }

    public static void Load()
    {
        try
        {
            if (!File.Exists(Env.PinFile)) return;
            var d = JsonNode.Parse(File.ReadAllText(Env.PinFile)) as JsonObject;
            if (d is null) return;
            var model = d["model"]?.GetValue<string>() ?? "";
            if (model.Length == 0) return;
            var expires = DateTimeOffset.Parse(d["expires"]?.GetValue<string>()
                ?? DateTimeOffset.MinValue.ToString("O"));
            if (DateTimeOffset.UtcNow >= expires) return;   // already lapsed
            _pin = new Pin(model, d["fallback"]?.GetValue<bool>() ?? false, expires);
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"[pin] ignoring unreadable pin file: {e.Message}");
        }
    }

    /// <summary>Normalise an operator-supplied model id to the bare upstream
    /// name the scorer works in. Accepts <c>OCG-cat-jam-free</c>,
    /// <c>ocg:cat-jam-free</c> and <c>cat-jam-free</c> alike, because all three
    /// are things a person would reasonably type.</summary>
    public static string Normalise(string model)
    {
        var m = model.Trim();
        foreach (var sep in new[] { "-", ":" })
        {
            if (m.StartsWith("ocg" + sep, StringComparison.OrdinalIgnoreCase))
                m = m[(sep.Length + 3)..];
        }
        return m.Trim().ToLowerInvariant();
    }

    /// <summary>Pin a model. Rejects one that is not in the roster so a typo
    /// fails here, loudly, rather than 404ing every subsequent request.</summary>
    public static (bool Ok, string Message, JsonObject State) Set(string model, bool fallback, int ttlSeconds)
    {
        var name = Normalise(model);
        if (name.Length == 0) return (false, "model is required", Describe());

        var roster = ModelScorerService.Current;
        if (roster.Length > 0 && Array.IndexOf(roster, name) < 0)
            return (false,
                $"'{name}' is not in the model roster. The scorer has not loaded a roster yet, " +
                $"or the id is wrong -- check GET /admin/model for known ids.", Describe());

        if (ttlSeconds <= 0) ttlSeconds = DefaultTtlSeconds;
        if (ttlSeconds > MaxTtlSeconds) ttlSeconds = MaxTtlSeconds;

        lock (Lock)
        {
            _pin = new Pin(name, fallback, DateTimeOffset.UtcNow.AddSeconds(ttlSeconds));
            TrySave();
        }
        return (true, $"pinned '{name}' for {ttlSeconds}s", Describe());
    }

    public static (bool Ok, string Message, JsonObject State) Clear()
    {
        string? was;
        lock (Lock)
        {
            was = _pin?.Model;
            _pin = null;
            TrySave();
        }
        return (true, was is null ? "was not pinned" : $"unpinned '{was}'", Describe());
    }

    public static JsonObject Describe()
    {
        lock (Lock)
        {
            var expired = _pin is not null && DateTimeOffset.UtcNow >= _pin.Expires;
            var p = expired ? null : _pin;
            var o = new JsonObject
            {
                ["pinned"] = p is not null,
                ["model"] = p?.Model,
                ["fallback"] = p?.Fallback,
                ["expires"] = p?.Expires.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'"),
                ["expires_in_s"] = p is null ? null
                    : (int)Math.Max(0, (p.Expires - DateTimeOffset.UtcNow).TotalSeconds),
                ["ttl_max_s"] = MaxTtlSeconds,
            };
            return o;
        }
    }

    static void TrySave()
    {
        try
        {
            if (_pin is null)
            {
                if (File.Exists(Env.PinFile)) File.Delete(Env.PinFile);
                return;
            }
            var dir = Path.GetDirectoryName(Env.PinFile);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            File.WriteAllText(Env.PinFile, new JsonObject
            {
                ["model"] = _pin.Model,
                ["fallback"] = _pin.Fallback,
                ["expires"] = _pin.Expires.ToString("O"),
            }.ToJsonString());
        }
        catch (Exception e)
        {
            // A pin that cannot be persisted still applies to this process, but
            // it would not survive a restart -- say so rather than pretending.
            Console.Error.WriteLine($"[pin] could not persist pin: {e.Message}");
        }
    }
}