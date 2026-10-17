using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace LlmGateway;

/// <summary>
/// /v1/chat/completions + /v1/responses handlers. Port of the Python 1.7.4
/// logic: resolve model -> ordered provider chain, try each in turn, relay or
/// convert the upstream response, tee live session ingest + persist chat
/// sessions + record stats. Streaming relays SSE incrementally.
/// </summary>
public static class Chat
{
    // ---------------- shared helpers ----------------

    static async Task<JsonObject?> ParseBody(HttpContext ctx)
    {
        try
        {
            return await JsonNode.ParseAsync(ctx.Request.Body) as JsonObject;
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"ParseBody failed: {e.Message} ({e.GetType().Name})");
            return null;
        }
    }

    static string? SessionId(HttpContext ctx, JsonArray? messages = null)
    {
        var sid = ctx.Request.Cookies[Env.SessionCookie];
        if (string.IsNullOrEmpty(sid))
        {
            // No cookie: this is a headless client or a fresh agent session.
            // A running session only appends, so the messages already stored
            // for it are an exact prefix of what just arrived. Recover that
            // conversation before falling back to a new id -- otherwise every
            // request invents a fresh GUID and one long conversation fragments
            // into dozens of single-turn ones with no way to relate them.
            sid = Sessions.FindByPrefix(messages);
            if (sid is null)
                sid = Guid.NewGuid().ToString("N");
            // Set cookie before response body is written
            ctx.Response.Cookies.Append(Env.SessionCookie, sid, new CookieOptions
            {
                MaxAge = TimeSpan.FromDays(Env.SessionTtlDays),
                HttpOnly = true,
                SameSite = SameSiteMode.Lax
            });
        }
        return sid;
    }

    /// <summary>Persist one turn: what was asked, what answered, which model
    /// actually served it, and when. Only the user messages this turn
    /// introduced are stored, so the write is proportional to the delta rather
    /// than to the whole conversation.</summary>
    static void SaveSessionFrom(HttpContext ctx, Sessions.Turn turn, string? sid = null)
    {
        sid ??= SessionId(ctx, turn.Messages);
        if (string.IsNullOrEmpty(sid) || string.IsNullOrWhiteSpace(turn.Reply)) return;
        Sessions.RecordTurn(sid, turn);
    }

    static void TeeTurn(HttpContext ctx, JsonArray? messages, string reply)
    {
        var (consumer, _, _) = Rag.RequestHeaders(ctx);
        Rag.TeeLiveTurn(consumer, LastUserMessage(messages) ?? "", reply);
    }
    static string? LastUserMessage(JsonArray? messages)
    {
        if (messages is null) return null;
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            var m = messages[i] as JsonObject;
            if (m is null) continue;
            if (m["role"]?.GetValue<string>() != "user") continue;
            if (m["content"] is JsonValue v && v.TryGetValue<string>(out var text) && !string.IsNullOrWhiteSpace(text))
                return text.Trim();
        }
        return null;
    }

    static (Stats.Usage U, int Cached) UsageFrom(JsonNode? content)
    {
        var u = content?["usage"];
        if (u is null) return (new Stats.Usage(0, 0, 0), 0);
        var reasoning = u["completion_tokens_details"]?["reasoning_tokens"]?.GetValue<int>() ?? 0;
        var cached = u["prompt_tokens_details"]?["cached_tokens"]?.GetValue<int>() ?? 0;
        return (new Stats.Usage(
            u["prompt_tokens"]?.GetValue<int>() ?? 0,
            u["completion_tokens"]?.GetValue<int>() ?? 0,
            reasoning), cached);
    }

    /// <summary>Safe access to choices[0]: the final SSE chunk is often
    /// {"choices":[],...} — JsonArray indexer throws on out-of-range.</summary>
    static JsonObject? FirstChoice(JsonObject? obj)
    {
        if (obj?["choices"] is JsonArray arr && arr.Count > 0)
            return arr[0] as JsonObject;
        return null;
    }

    /// <summary>Whether any message in the conversation carries an image
    /// (OpenAI chat-style `image_url` parts or Responses-style `input_image`
    /// items). The gateway is text-only for now: image requests cost upstream
    /// credits, so we reject them up front before resolving a provider.</summary>
    static bool HasImageContent(JsonArray? messages)
    {
        if (messages is null) return false;
        foreach (var m in messages)
        {
            var msg = m as JsonObject;
            if (msg is null) continue;
            if (msg["content"] is JsonArray parts)
            {
                foreach (var part in parts)
                {
                    var obj = part as JsonObject;
                    var t = obj?["type"]?.GetValue<string>()?.ToLowerInvariant();
                    if (t is "image_url" or "image" or "input_image" or "image_file")
                        return true;
                    if (obj?["image_url"] is not null || obj?["input_image"] is not null)
                        return true;
                }
            }
        }
        return false;
    }

    /// <summary>Reconstruct assistant content from OpenAI SSE stream bytes.</summary>
    static string SseReplyText(byte[] raw)
    {
        var parts = new StringBuilder();
        foreach (var line in Encoding.UTF8.GetString(raw).Split('\n'))
        {
            var t = line.Trim();
            if (!t.StartsWith("data:")) continue;
            var payload = t[5..].Trim();
            if (payload.Length == 0 || payload == "[DONE]") continue;
            JsonObject? obj;
            try { obj = JsonNode.Parse(payload) as JsonObject; } catch { continue; }
            if (obj is null) continue;
            var delta = FirstChoice(obj)?["delta"]?["content"];
            if (delta is JsonValue v && v.TryGetValue<string>(out var s)) parts.Append(s);
        }
        return parts.ToString();
    }

    /// <summary>
    /// Replace the "model" field in every <c>data:</c> SSE payload with
    /// <paramref name="servedModel"/>. Lines that aren't parseable JSON (comments,
    /// <c>[DONE]</c>, blank keep-alives) are passed through untouched.
    ///
    /// The rewrite is a targeted string splice of just the <c>"model":"..."</c>
    /// span, not a JSON re-serialization: a proxy must not reformat or re-escape
    /// upstream payloads (field order, unicode escaping, float formatting).
    /// </summary>
    static byte[] RewriteSseModel(string block, string servedModel)
    {
        var enc = System.Text.Encoding.UTF8;
        var sb = new StringBuilder(block.Length + 64);
        var lines = block.Split('\n');
        for (int i = 0; i < lines.Length; i++)
        {
            // Split on '\n' yields a trailing "" for a block that ended in a
            // newline. Re-emit that element verbatim with no added newline so
            // the output is byte-identical in framing to the input (no blank
            // lines accumulate across chunks).
            bool isTail = i == lines.Length - 1;
            var line = lines[i];
            var t = line.TrimEnd('\r');
            var trailing = line.Length > t.Length ? "\r" : "";
            if (isTail) { sb.Append(line); continue; }
            if (!t.StartsWith("data:")) { sb.Append(line).Append('\n'); continue; }
            var payload = t[5..].Trim();
            if (payload.Length == 0 || payload == "[DONE]") { sb.Append(line).Append('\n'); continue; }
            var spliced = SpliceModelField(payload, servedModel);
            sb.Append("data: ").Append(spliced ?? payload).Append(trailing).Append('\n');
        }
        return enc.GetBytes(sb.ToString());

        // Returns the payload with its "model" value replaced, or null to leave
        // the line alone (no "model" key, or unparseable / ambiguous JSON).
        static string? SpliceModelField(string payload, string model)
        {
            JsonNode? probe;
            try { probe = JsonNode.Parse(payload); } catch { return null; }
            if (probe is not JsonObject obj || obj["model"] is null) return null;
            // Locate the exact "model":"<old>" span within the raw payload text.
            // Searching from the front is safe because a bare `"model"` key must
            // appear before any nested occurrence; we additionally require the
            // value to be a plain string so we never touch a nested object.
            var needle = "\"model\":";
            var idx = payload.IndexOf(needle, StringComparison.Ordinal);
            if (idx < 0) return null;
            var q = payload.IndexOf('"', idx + needle.Length);
            if (q < 0) return null;
            // Walk the string to its closing quote, honouring backslash escapes.
            int end = q + 1;
            while (end < payload.Length)
            {
                if (payload[end] == '\\') { end += 2; continue; }
                if (payload[end] == '"') break;
                end++;
            }
            if (end >= payload.Length) return null;
            // Serialize to get correct escaping (servedModel contains ':' but may
            // legitimately contain quotes/backslashes), then splice the ESCAPED
            // INNER text between the existing quotes — payload[..q+1] already
            // carries the opening quote and payload[end..] the closing one, so
            // adding our own quotes here would double them.
            var escaped = JsonSerializer.Serialize(model);
            var inner = escaped.Length >= 2 ? escaped[1..^1] : escaped;
            return payload[..(q + 1)] + inner + payload[end..];
        }
    }

    /// <summary>Extract usage from the final SSE chunk if the upstream includes it.</summary>
    static (Stats.Usage U, int Cached) SseUsage(byte[] raw)
    {
        foreach (var line in Encoding.UTF8.GetString(raw).Split('\n'))
        {
            var t = line.Trim();
            if (!t.StartsWith("data: ")) continue;
            var payload = t[6..].Trim();
            if (payload is "[DONE]" or "") continue;
            JsonObject? obj;
            try { obj = JsonNode.Parse(payload) as JsonObject; } catch { continue; }
            if (obj is null) continue;
            if (obj["usage"] is JsonObject u)
            {
                var reasoning = u["completion_tokens_details"]?["reasoning_tokens"]?.GetValue<int>() ?? 0;
                var cached = u["prompt_tokens_details"]?["cached_tokens"]?.GetValue<int>() ?? 0;
                return (new Stats.Usage(
                    u["prompt_tokens"]?.GetValue<int>() ?? 0,
                    u["completion_tokens"]?.GetValue<int>() ?? 0,
                    reasoning), cached);
            }
        }
        return (new Stats.Usage(0, 0, 0), 0);
    }

    static void RecordStat(HttpContext ctx, string endpoint, bool stream, int status, long latencyMs,
        (bool Injected, int Chunks, int Bytes) rag, Stats.Usage usage)
    {
        var (consumer, _, _) = Rag.RequestHeaders(ctx);
        Stats.Record(new Stats.Rec(
            DateTimeOffset.UtcNow.ToUnixTimeSeconds(), consumer, endpoint, stream,
            status, latencyMs, new Stats.RagMeta(rag.Injected, rag.Chunks, rag.Bytes), usage));
    }

    static async Task<string> ErrorSnippetAsync(HttpResponseMessage resp)
    {
        try
        {
            var s = await resp.Content.ReadAsStringAsync();
            return s.Length <= 400 ? s : s[..400];
        }
        catch { return ""; }
    }

    /// <summary>Short human reason for a failed provider attempt (from its error body).</summary>
    static string ShortReason(string? raw)
    {
        if (string.IsNullOrEmpty(raw)) return "";
        try
        {
            var par = JsonNode.Parse(raw) as JsonObject;
            var em = par?["error"]?["message"]?.GetValue<string>();
            if (!string.IsNullOrEmpty(em))
                return em.Length <= 90 ? em : em[..90];
        }
        catch { /* fall back to raw snippet */ }
        var s = raw!.Replace('\n', ' ').Trim();
        return s.Length <= 90 ? s : s[..90];
    }

    /// <summary>Clear aggregated "everything failed" message: one line per provider.</summary>
    static string SummarizeFailures(string model, IReadOnlyList<ReqLog.Attempt> attempts)
    {
        if (attempts.Count == 0) return $"all providers failed for model '{model}'";
        var parts = new List<string>();
        for (var i = 0; i < attempts.Count; i++)
        {
            var a = attempts[i];
            var reason = a.Outcome == "unreachable"
                ? $"unreachable ({ShortReason(a.Detail)})"
                : $"HTTP {a.Status} {ShortReason(a.Detail)}".Trim();
            parts.Add($"[{i + 1}] {a.Provider}: {reason}");
        }
        var msg = $"all providers failed for model '{model}': {string.Join("; ", parts)}.";

        var allRateLimited = attempts.Count > 0 && attempts.All(a =>
            a.Status == 429 || (a.Detail?.Contains("rate limit", StringComparison.OrdinalIgnoreCase) ?? false));
        var balanceShort = attempts.Any(a => (a.Detail?.Contains("insufficient balance", StringComparison.OrdinalIgnoreCase) ?? false)
            || (a.Detail?.Contains("insufficient_quota", StringComparison.OrdinalIgnoreCase) ?? false));

        if (allRateLimited)
            msg += " Every provider is quota-limited right now — no free tier is available at this moment. " +
                   "Stop and retry after a reset window (usually the next hour/day); it is not a transient glitch.";
        else if (balanceShort)
            msg += " At least one provider needs a balance top-up — consider adding a small credit balance " +
                   "to unlock its free tier again (e.g. OpenRouter: even $10 unlocks 1000 free-model requests/day).";
        else
            msg += " No fallback served this request — rate limits are time-based, so retry in a bit.";
        return msg;
    }

    // ---------------- degraded responses ----------------

    /// <summary>Human-readable assistant text for a request that outlived every
    /// backend retry.
    ///
    /// Written to be coherent in a transcript rather than to be an error string:
    /// the caller asked a question, so it gets an answer-shaped reply that says
    /// what happened and when to try again. The usage window's reset time is
    /// included when we know it, because "try later" is much more useful when
    /// "later" has a timestamp.</summary>
    /// <summary>Only real upstream attempts. The recovery block appends sentinel
    /// entries ("recovery-enter", "recovery-probe") that carry status 0 by
    /// design; leaving them in made the rate-limit test below see a non-429 and
    /// report "no backend could be reached" when every attempt had in fact been
    /// rate-limited.</summary>
    static IEnumerable<ReqLog.Attempt> UpstreamAttempts(IReadOnlyList<ReqLog.Attempt> attempts) =>
        attempts.Where(a => a.Outcome is not ("recovery-enter" or "recovery-probe"));

    enum Exhausted { RateLimited, Credentials, Mixed }

    /// <summary>Why the chain ran out, from the statuses actually observed.
    ///
    /// The test this replaces was binary: the attempts were either all
    /// rate-limited or, by elimination, "no model backend could be reached".
    /// An expired API key, a retired model id and an unreachable host all fell
    /// into that second branch, so the reply sent the reader off to check
    /// connectivity while the actual fault was a credential nobody had
    /// rotated. Classify on what the upstreams said, and keep a mixed bucket so
    /// a genuinely heterogeneous failure reports the statuses instead of
    /// picking a side.</summary>
    static Exhausted ClassifyExhausted(IReadOnlyList<ReqLog.Attempt> real)
    {
        if (real.Count == 0) return Exhausted.Mixed;

        static bool RateLimited(ReqLog.Attempt a) =>
            a.Status == 429
            || (a.Detail?.Contains("rate limit", StringComparison.OrdinalIgnoreCase) ?? false);
        static bool Creds(ReqLog.Attempt a) =>
            a.Status is 401 or 403
            || (a.Detail?.Contains("api_key", StringComparison.OrdinalIgnoreCase) ?? false)
            || (a.Detail?.Contains("unauthorized", StringComparison.OrdinalIgnoreCase) ?? false);

        if (real.All(Creds)) return Exhausted.Credentials;
        if (real.All(RateLimited)) return Exhausted.RateLimited;
        return Exhausted.Mixed;
    }

    static string DegradedText(string model, IReadOnlyList<ReqLog.Attempt> attempts)
    {
        var real = UpstreamAttempts(attempts).ToList();
        var kind = ClassifyExhausted(real);
        var codes = real.Select(a => a.Status).Distinct().OrderBy(s => s).ToList();
        var why = kind switch
        {
            Exhausted.RateLimited => "every model backend is currently rate-limited",
            Exhausted.Credentials => "every model backend rejected this gateway's credentials",
            _ when real.Count == 0 => "no model backend was attempted",
            _ => "the model backends failed (upstream status "
                 + string.Join("/", codes) + ")",
        };

        var sb = new System.Text.StringBuilder();
        sb.Append("I wasn't able to answer this one. ");
        sb.Append(char.ToUpperInvariant(why[0])).Append(why[1..]);
        sb.Append(", and this gateway has no capacity left to fall back on.");
        sb.Append(" Your request itself was fine — nothing is wrong with it.");

        if (kind == Exhausted.RateLimited)
        {
            if (ModelScorerService.NextQuotaReset() is { } reset)
            {
                var mins = (int)Math.Round((reset - DateTimeOffset.UtcNow).TotalMinutes);
                sb.Append(mins > 0
                    ? $" The current usage window resets in about {mins} minute(s), at {reset:HH:mm} UTC."
                    : $" The current usage window resets at {reset:HH:mm} UTC.");
            }
            else
            {
                sb.Append(" These limits are time-based, so trying again shortly should work.");
            }
        }
        else if (kind == Exhausted.Credentials)
        {
            // Retrying cannot help a dead key, and the old closing line told the
            // reader to try again shortly -- the advice was worse than useless.
            sb.Append(" This is a gateway-side fault, not yours: the provider API key needs replacing.");
        }
        else
        {
            sb.Append(" Retrying shortly may work once the upstream fault clears.");
        }

        // Triage trail. Without this the reply above is guesswork: an expired
        // credential and an unreachable host both collapse into "no model
        // backend could be reached", which sends the reader after the wrong
        // thing. Carry the chain that was actually walked plus the upstream
        // error text verbatim, so the cause is readable from the transcript.
        if (real.Count > 0)
        {
            sb.Append(" Attempts:");
            for (var i = 0; i < real.Count; i++)
            {
                var a = real[i];
                sb.Append(i == 0 ? " " : "; ");
                sb.Append(a.Provider.ToLowerInvariant()).Append('/').Append(a.Upstream)
                  .Append(" -> ").Append(a.Status);
                // Raw, untruncated. An operator reading this transcript is
                // debugging a provider outage and needs the provider's own
                // words -- codes, quota numbers, reset hints -- not a summary.
                // The text lands in the chat transcript, so anything the upstream
                // chooses to echo (a key fragment, a tenant id) is carried into
                // it verbatim; that is a deliberate trade for debuggability.
                var detail = a.Detail?.Trim();
                if (!string.IsNullOrEmpty(detail))
                    sb.Append(" (").Append(detail).Append(')');
            }
            sb.Append('.');
        }
        return sb.ToString();
    }

    /// <summary>A well-formed chat.completion carrying <see cref="DegradedText"/>.
    /// Usage is zeroed because no model produced any of these tokens.</summary>
    static JsonObject DegradedChat(string modelAlias, string text) => new()
    {
        ["id"] = $"chatcmpl-degraded-{Guid.NewGuid():N}",
        ["object"] = "chat.completion",
        ["created"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
        ["model"] = modelAlias,
        ["choices"] = new JsonArray(new JsonObject
        {
            ["index"] = 0,
            ["message"] = new JsonObject { ["role"] = "assistant", ["content"] = text },
            ["finish_reason"] = "stop",
        }),
        ["usage"] = new JsonObject
        {
            ["prompt_tokens"] = 0, ["completion_tokens"] = 0, ["total_tokens"] = 0,
        },
    };

    /// <summary>Flag a degraded 200 so callers and monitoring can still tell this
    /// apart from a real completion. Without this the degradation is invisible
    /// except to whoever reads the request log.</summary>
    static void MarkDegraded(HttpContext ctx, int upstreamStatus)
    {
        // On the streaming path the keep-alives have already committed the
        // response, and headers are read-only once it has started. Setting them
        // would throw and truncate the stream before the assistant message lands,
        // so the flag is only attached when there is still a chance to.
        if (ctx.Response.HasStarted) return;
        ctx.Response.Headers["X-LLM-Gateway-Status"] = "degraded";
        ctx.Response.Headers["X-LLM-Gateway-Upstream-Status"] = upstreamStatus.ToString();
    }

    // ---------------- /v1/chat/completions ----------------

    public static async Task ChatCompletions(HttpContext ctx)
    {
        var body = await ParseBody(ctx);
        if (body is null) throw new ApiError(400, "request body must be JSON");
        if (HasImageContent(body!["messages"] as JsonArray))
            throw new ApiError(400, "image input is disabled in llm-gateway. Send text-only requests.");
        var model = body!["model"]?.GetValue<string>();
        // Resolve the conversation once per request: with no cookie this does
        // an exact-prefix lookup, and every use below must see the same id.
        var sid = SessionId(ctx, body["messages"] as JsonArray);
        var routes = await Providers.ResolveAsync(model);
        var wantStream = body["stream"]?.GetValue<bool>() ?? false;

        // RAG: header-driven injection (fail-open; absent headers = legacy defaults)
        var rag = (Injected: false, Chunks: 0, Bytes: 0);
        if (Env.RagEnabled && (body["rag"]?.GetValue<bool>() ?? true))
        {
            var (consumer, docsEnabled, optin) = Rag.RequestHeaders(ctx);
            if (optin && body["messages"] is JsonArray msgs)
            {
                var injected = await Rag.InjectAsync(msgs, consumer, docsEnabled);
                body["messages"] = injected;
                rag = Rag.RagMeta(injected);
            }
        }

        if (!wantStream)
        {
            var nsT00 = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            var nsAttempts = new List<ReqLog.Attempt>();
            var nsWaitStart = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            string? lastErr = null;
        nsRetry:
            foreach (var (key, upstream) in routes)
            {
                var t0 = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                var payload = (JsonObject)body.DeepClone();
                payload["model"] = upstream;
                Providers.EnsureReasoningEcho(payload["messages"] as JsonArray, Providers.IsThinkingModel(upstream));
                Providers.NormalizeReasoningEffort(payload, key);
                HttpResponseMessage resp;
                try { (resp, _) = await Providers.UpstreamAsync(key, "/chat/completions", payload, false, sid); }
                catch (Exception e)
                {
                    lastErr = $"{Providers.ByKey(key)?.Label} unreachable: {e.GetType().Name}";
                    ModelMetrics.Failure($"{key}:{upstream}", 0);
                    nsAttempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                        "unreachable", 0, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0,
                        $"{e.GetType().Name}: {e.Message}"));
                    continue;
                }
                using (resp)
                {
                    var status = (int)resp.StatusCode;
                    // A single leg has nothing to fall through to, but a rate
                    // limit or a transient 5xx is still worth the retry budget
                    // and then a degraded reply rather than a raw upstream error.
                    var retryable = Providers.FallbackStatus.Contains(status)
                        && (routes.Count > 1
                            || (Env.DegradeOnFailure && Providers.DegradableStatus.Contains(status)));
                    if (retryable)
                    {
                        var detail = await ErrorSnippetAsync(resp);
                        lastErr = $"{Providers.ByKey(key)?.Label} HTTP {status}";
                        ModelMetrics.Failure($"{key}:{upstream}", status);
                        nsAttempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                            "fallback-status", status, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0, detail));
                        continue;
                    }
                    var raw = await resp.Content.ReadAsByteArrayAsync();
                    JsonNode? content = null;
                    try { content = JsonNode.Parse(raw); } catch { /* fall through to raw */ }

                    // X-LLM-Model: the model that ACTUALLY served this request.
                    // OCG legs: we dictate the model, so body echo == leg model.
                    // Hoisted above the session write so the attribution is
                    // recorded even when the body is not JSON.
                    var servedName = (content as JsonObject)?["model"]?.GetValue<string>();
                    var servedModel = string.IsNullOrEmpty(servedName) ? $"{key}:{upstream}" : $"{key}:{servedName}";
                    var (u, cached) = UsageFrom(content);

                    if (status < 400)
                    {
                        var reply = FirstChoice(content as JsonObject)?["message"]?["content"]?.GetValue<string>() ?? "";
                        TeeTurn(ctx, body["messages"] as JsonArray, reply);
                        if (Providers.IsThinkingModel(upstream)) Providers.CacheReasoningEcho(content);
                        SaveSessionFrom(ctx, new Sessions.Turn(
                            model ?? "", servedModel, key,
                            body["messages"] as JsonArray, reply, status,
                            DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0,
                            new Sessions.Usage(u.Prompt, u.Completion, u.Reasoning, cached),
                            ModelScorerService.Cost(servedModel, u.Prompt,
                                u.Completion + u.Reasoning, cached)), sid);
                        ModelMetrics.Success(servedModel, u.Prompt, u.Completion + u.Reasoning, cached,
                            ModelScorerService.Cost(servedModel, u.Prompt, u.Completion + u.Reasoning, cached));
                    }
                    else
                    {
                        ModelMetrics.Failure($"{key}:{upstream}", status);
                        nsAttempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                            "upstream-error", status, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0,
                            Encoding.UTF8.GetString(raw)[..Math.Min(400, raw.Length)]));
                    }
                    var latency = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0;
                    RecordStat(ctx, "chat", false, status, latency, rag, u);
                    nsAttempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                        "ok", status, latency, null));
                    ReqLog.Write(ctx, "chat", model, false,
                        status, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - nsT00, nsAttempts,
                        new Stats.RagMeta(rag.Injected, rag.Chunks, rag.Bytes), u, cached, null);
                    if (content is not null)
                    {
                        ctx.Response.StatusCode = status;
                        ctx.Response.ContentType = "application/json";
                        ctx.Response.Headers["X-LLM-Model"] = servedModel;
                        // Mirror the header into the body's model field. OpenAI-
                        // compatible clients (Hermes) read response.model, not
                        // headers, so without this they only ever see the bare
                        // upstream name with no provider prefix. Informational
                        // only — nothing downstream routes on this field.
                        if (content is JsonObject bodyObj) bodyObj["model"] = servedModel;
                        await ctx.Response.WriteAsync(content.ToJsonString());
                        return;
                    }
                    ctx.Response.StatusCode = status;
                    ctx.Response.ContentType = resp.Content.Headers.ContentType?.ToString() ?? "application/json";
                    await ctx.Response.Body.WriteAsync(raw);
                    return;
                }
            }
            // Every backend failed. Give them a bounded chance to come back before
            // answering, so a momentary blip degrades to a retry rather than a
            // refusal. Non-streaming callers are usually blocking on their own
            // client timeout, hence the short DegradeMaxWaitSeconds window.
            while (Env.DegradeOnFailure && !ctx.RequestAborted.IsCancellationRequested
                   && DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - nsWaitStart
                        < Env.DegradeMaxWaitSeconds * 1000L)
            {
                try { await Task.Delay(TimeSpan.FromSeconds(Env.RecoveryKeepAliveSeconds), ctx.RequestAborted); }
                catch (OperationCanceledException) { break; }

                var back = new List<(string Key, string Upstream)>();
                foreach (var (k, up) in routes)
                    if (await Providers.ProbeAsync(k) is { Reachable: true })
                        back.Add((k, up));
                if (back.Count > 0)
                {
                    routes = back;
                    nsAttempts = new List<ReqLog.Attempt>();
                    goto nsRetry;
                }
            }

            var nsSummary = SummarizeFailures(model ?? "auto", nsAttempts);
            // The log records the truth regardless of how we answer the client.
            ReqLog.Write(ctx, "chat", model, false,
                502, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - nsT00, nsAttempts,
                new Stats.RagMeta(rag.Injected, rag.Chunks, rag.Bytes), new Stats.Usage(0, 0, 0), 0,
                nsSummary);

            if (Env.DegradeOnFailure)
            {
                MarkDegraded(ctx, 502);
                ctx.Response.StatusCode = 200;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync(
                    DegradedChat(model ?? "auto", DegradedText(model ?? "auto", nsAttempts)).ToJsonString());
                return;
            }
            throw new ApiError(502, nsSummary);
        }

        // streaming: pick the first provider that responds 2xx, stream its SSE
        var t00 = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        var attempts = new List<ReqLog.Attempt>();
        // Recovery deadline, established once on the first entry into the
        // recovery block and deliberately NOT reset by a `goto streamRetry`.
        // Declaring the start time inside that block meant every retry started
        // the clock over, so the cap was never reached whenever the provider was
        // reachable-but-rejecting -- i.e. exactly a 429 quota wall.
        long recoveryDeadline = 0;
        string? streamErr = null;
    streamRetry:
        foreach (var (key, upstream) in routes)
        {
            var payload = (JsonObject)body.DeepClone();
            payload["model"] = upstream;
            Providers.EnsureReasoningEcho(payload["messages"] as JsonArray, Providers.IsThinkingModel(upstream));
                Providers.NormalizeReasoningEffort(payload, key);
            HttpResponseMessage resp;
            var t0 = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            try { (resp, _) = await Providers.UpstreamAsync(key, "/chat/completions", payload, true, sid); }
            catch (Exception e)
            {
                streamErr = $"{Providers.ByKey(key)?.Label} unreachable: {e.GetType().Name}";
                ModelMetrics.Failure($"{key}:{upstream}", 0);
                attempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                    "unreachable", 0, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0,
                    $"{e.GetType().Name}: {e.Message}"));
                continue;
            }
            var status = (int)resp.StatusCode;
            if (status >= 400)
            {
                streamErr = $"{Providers.ByKey(key)?.Label} HTTP {status}";
                ModelMetrics.Failure($"{key}:{upstream}", status);
                // Same reasoning as the non-streaming path: with one leg, only
                // genuine config/auth errors pass through verbatim.
                var passThrough = routes.Count == 1
                    && !(Env.DegradeOnFailure && Providers.DegradableStatus.Contains(status));
                if (passThrough)
                {
                    var raw = await resp.Content.ReadAsByteArrayAsync();
                    resp.Dispose();
                    attempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                        "upstream-error", status, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0,
                        Encoding.UTF8.GetString(raw)[..Math.Min(400, raw.Length)]));
                    ReqLog.Write(ctx, "chat", model, true,
                        status, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t00, attempts,
                        new Stats.RagMeta(rag.Injected, rag.Chunks, rag.Bytes), new Stats.Usage(0, 0, 0), 0, null);
                    if (!ctx.Response.HasStarted)
                    {
                        ctx.Response.StatusCode = status;
                        ctx.Response.ContentType = resp.Content.Headers.ContentType?.ToString() ?? "application/json";
                    }
                    await ctx.Response.Body.WriteAsync(raw);
                    return;
                }
                attempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                    "fallback-status", status, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0,
                    await ErrorSnippetAsync(resp)));
                resp.Dispose();
                continue;
            }

            // relay SSE incrementally; accumulate for post-processing
            if (!ctx.Response.HasStarted)
            {
                ctx.Response.StatusCode = 200;
                ctx.Response.ContentType = "text/event-stream";
                ctx.Response.Headers["Cache-Control"] = "no-cache";
                // X-LLM-Model on streaming too — must be set before the body
                // starts. Streaming never re-parses the SSE for the resolved
                // model, so this is the LEG spec (key:upstream). For OCG that
                // is the model we dictated; OR free legs resolve in-band.
                ctx.Response.Headers["X-LLM-Model"] = $"{key}:{upstream}";
            }
            var chunks = new MemoryStream();
            try
            {
                await using var stream = await resp.Content.ReadAsStreamAsync();
                var buf = new byte[81920];
                int read;
                // Rewrite "model" in each complete SSE line to servedModel so
                // OpenAI-compatible clients (Hermes) see the provider-qualified
                // name in chunk objects, not just the X-LLM-Model header. Reads
                // split at arbitrary byte offsets, so carry the trailing partial
                // line and never re-emit it until it terminates.
                var servedModel = $"{key}:{upstream}";
                var pending = new MemoryStream();
                while ((read = await stream.ReadAsync(buf)) > 0)
                {
                    chunks.Write(buf, 0, read);
                    pending.Write(buf, 0, read);
                    var pendingArr = pending.ToArray();
                    var lastNl = Array.LastIndexOf(pendingArr, (byte)'\n');
                    if (lastNl < 0) continue;
                    var complete = System.Text.Encoding.UTF8.GetString(pendingArr, 0, lastNl + 1);
                    pending.SetLength(0);
                    pending.Write(pendingArr, lastNl + 1, pendingArr.Length - lastNl - 1);
                    var rewritten = RewriteSseModel(complete, servedModel);
                    await ctx.Response.Body.WriteAsync(rewritten);
                    await ctx.Response.Body.FlushAsync();
                }
                // Flush any trailing partial line (stream ended without newline).
                if (pending.Length > 0)
                {
                    var tail = RewriteSseModel(System.Text.Encoding.UTF8.GetString(pending.ToArray()), servedModel);
                    await ctx.Response.Body.WriteAsync(tail);
                    await ctx.Response.Body.FlushAsync();
                }
            }
            finally
            {
                var raw = chunks.ToArray();
                var reply = SseReplyText(raw);
                var (u, cached) = SseUsage(raw);
                var servedModel = $"{key}:{upstream}";
                if (status < 400)
                {
                    TeeTurn(ctx, body["messages"] as JsonArray, reply);
                    if (Providers.IsThinkingModel(upstream)) Providers.CacheReasoningEchoFromSse(raw);
                    if (!string.IsNullOrEmpty(sid) && !string.IsNullOrWhiteSpace(reply))
                    {
                        Sessions.RecordTurn(sid, new Sessions.Turn(
                            model ?? "", servedModel, key,
                            body["messages"] as JsonArray, reply, status,
                            DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0,
                            new Sessions.Usage(u.Prompt, u.Completion, u.Reasoning, cached),
                            ModelScorerService.Cost(servedModel, u.Prompt,
                                u.Completion + u.Reasoning, cached)));
                    }
                    ModelMetrics.Success(servedModel, u.Prompt, u.Completion + u.Reasoning, cached,
                        ModelScorerService.Cost(servedModel, u.Prompt, u.Completion + u.Reasoning, cached));
                }
                var latency = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0;
                RecordStat(ctx, "chat", true, status, latency, rag, u);
                attempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                    "ok", status, latency, null));
                ReqLog.Write(ctx, "chat", model, true,
                    status, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t00, attempts,
                    new Stats.RagMeta(rag.Injected, rag.Chunks, rag.Bytes), u, cached, null);
                resp.Dispose();
            }
            return;
        }
        // All providers failed. Do NOT end the turn — keep the SSE stream open,
        // emitting keep-alive comments until a provider comes back (or the client
        // leaves / the cap is hit), then retry the chain from the top.
        if (!ctx.RequestAborted.IsCancellationRequested)
        {
            if (recoveryDeadline == 0)
                recoveryDeadline = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
                                    + Env.RecoveryMaxWaitSeconds * 1000L;
            attempts.Add(new ReqLog.Attempt("-", "-", "recovery-enter", 0,
                DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t00,
                $"chain exhausted after {attempts.Count} attempt(s); keep-alive up to {Env.RecoveryMaxWaitSeconds}s"));
            // Only set status/headers when the response has not already started.
            // A recovery retry (goto streamRetry) means a previous pass already
            // began streaming; re-setting StatusCode then throws
            // InvalidOperationException -> unhandled -> client sees a truncated
            // 200 and retries. Same guard as the single-route passthrough above.
            if (!ctx.Response.HasStarted)
            {
                ctx.Response.StatusCode = 200;
                ctx.Response.ContentType = "text/event-stream";
                ctx.Response.Headers["Cache-Control"] = "no-cache";
            }
            while (true)
            {
                try
                {
                    var now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                    await ctx.Response.WriteAsync(
                        $"data: {{\"id\":\"gen-keepalive\",\"object\":\"chat.completion.chunk\",\"created\":{now},\"model\":{JsonSerializer.Serialize(model ?? "auto")},\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"\"}},\"finish_reason\":null}}]}}\n\n");
                    await ctx.Response.WriteAsync($": keep-alive {now}\n\n");
                    await ctx.Response.Body.FlushAsync();
                }
                catch (Exception e) when (e is OperationCanceledException or IOException or ObjectDisposedException)
                {
                    return; // client went away mid-keepalive
                }
                try { await Task.Delay(TimeSpan.FromSeconds(Env.RecoveryKeepAliveSeconds), ctx.RequestAborted); }
                catch (OperationCanceledException) { return; } // client went away

                var recovered = new List<(string Key, string Upstream)>();
                foreach (var (k, up) in routes)
                    if (await Providers.ProbeAsync(k) is { Reachable: true } probe)
                        recovered.Add((k, up));
                attempts.Add(new ReqLog.Attempt("-", "-", "recovery-probe", 0,
                    DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t00,
                    $"{routes.Count - recovered.Count}/{routes.Count} leg(s) down; up=[{string.Join(",", recovered.Select(r => r.Key + ":" + r.Upstream))}]"));

                // Check the cap BEFORE acting on recovery. A provider that is
                // reachable but rejecting (a 429 quota wall answers /models
                // perfectly happily) used to send us straight back to
                // streamRetry, so the cap was never reached and the stream spun
                // keep-alives forever -- strictly worse than an error, because
                // the client just waits. The cap is a hard bound on total
                // recovery time, retries included.
                if (DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() > recoveryDeadline)
                    break;

                if (recovered.Count > 0)
                {
                    routes = recovered;
                    attempts = new List<ReqLog.Attempt>();
                    goto streamRetry;
                }
            }
            var gaveup = SummarizeFailures(model ?? "auto", attempts);
            ReqLog.Write(ctx, "chat", model, true,
                502, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t00, attempts,
                new Stats.RagMeta(rag.Injected, rag.Chunks, rag.Bytes), new Stats.Usage(0, 0, 0), 0, gaveup);
            try
            {
                if (Env.DegradeOnFailure)
                {
                    // The stream has been open on 200 with keep-alives for the
                    // whole recovery window, so an error frame here would be the
                    // only non-conversational thing the client ever sees. Emit a
                    // real assistant message instead and close it cleanly.
                    var text = DegradedText(model ?? "auto", attempts);
                    var esc = JsonSerializer.Serialize(text);
                    var created = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                    MarkDegraded(ctx, 502);
                    await ctx.Response.WriteAsync(
                        $"data: {{\"id\":\"chatcmpl-degraded\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{JsonSerializer.Serialize(model ?? "auto")},\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":{esc}}},\"finish_reason\":null}}]}}\n\n");
                    await ctx.Response.WriteAsync(
                        $"data: {{\"id\":\"chatcmpl-degraded\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{JsonSerializer.Serialize(model ?? "auto")},\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n");
                }
                else
                {
                    await ctx.Response.WriteAsync($"data: {{\"error\":{{\"message\":{JsonSerializer.Serialize(gaveup)}}}}}\n\n");
                }
                await ctx.Response.WriteAsync("data: [DONE]\n\n");
                await ctx.Response.Body.FlushAsync();
            }
            catch (Exception e) when (e is OperationCanceledException or IOException or ObjectDisposedException)
            {
                // client already gone; nothing to say
            }
            return;
        }
        ReqLog.Write(ctx, "chat", model, true,
            502, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t00, attempts,
            new Stats.RagMeta(rag.Injected, rag.Chunks, rag.Bytes), new Stats.Usage(0, 0, 0), 0,
            SummarizeFailures(model ?? "auto", attempts));
    }

    // ---------------- /v1/responses (OpenAI Responses API) ----------------

    static JsonArray ResponsesInputToMessages(JsonNode? input, string? instructions)
    {
        var messages = new JsonArray();
        if (!string.IsNullOrEmpty(instructions))
            messages.Add(new JsonObject { ["role"] = "system", ["content"] = instructions });

        if (input is JsonValue sv && sv.TryGetValue<string>(out var str))
        {
            if (!string.IsNullOrWhiteSpace(str))
                messages.Add(new JsonObject { ["role"] = "user", ["content"] = str });
            return messages;
        }
        if (input is not JsonArray arr)
            throw new ApiError(400, "input must be a string or array of items");

        foreach (var node in arr)
        {
            var item = node as JsonObject;
            if (item is null) continue;
            var type = item["type"]?.GetValue<string>();
            switch (type)
            {
                case "message":
                {
                    var role = item["role"]?.GetValue<string>() ?? "user";
                    var content = item["content"];
                    string text;
                    if (content is JsonValue cv && cv.TryGetValue<string>(out var s)) text = s;
                    else if (content is JsonArray parts)
                    {
                        var sb = new StringBuilder();
                        foreach (var p in parts)
                        {
                            if (p is not JsonObject po) continue;
                            var pt = po["type"]?.GetValue<string>();
                            if (pt is "input_text" or "output_text" or "text")
                                sb.Append(po["text"]?.GetValue<string>() ?? "");
                        }
                        text = sb.ToString();
                    }
                    else text = "";
                    if (!string.IsNullOrWhiteSpace(text))
                        messages.Add(new JsonObject { ["role"] = role, ["content"] = text });
                    break;
                }
                case "function_call":
                {
                    var msg = new JsonObject
                    {
                        ["role"] = "assistant",
                        ["content"] = null,
                        ["tool_calls"] = new JsonArray(new JsonObject
                        {
                            ["id"] = item["call_id"]?.GetValue<string>() ?? $"call_{Guid.NewGuid():N}"[..13],
                            ["type"] = "function",
                            ["function"] = new JsonObject
                            {
                                ["name"] = item["name"]?.GetValue<string>() ?? "",
                                ["arguments"] = item["arguments"]?.GetValue<string>() ?? "{}",
                            },
                        }),
                    };
                    // DeepSeek thinking-mode requires reasoning_content echoed back
                    if (item["reasoning_content"]?.GetValue<string>() is { Length: > 0 } rc)
                        msg["reasoning_content"] = rc;
                    messages.Add(msg);
                    break;
                }
                case "function_call_output":
                    messages.Add(new JsonObject
                    {
                        ["role"] = "tool",
                        ["tool_call_id"] = item["call_id"]?.GetValue<string>() ?? "",
                        ["content"] = item["output"]?.ToString() ?? "",
                    });
                    break;
                case "reasoning":
                    continue; // reasoning summaries don't map to chat messages
                case "input_image":
                case "image":
                case "image_url":
                    throw new ApiError(400, "image input is disabled in llm-gateway. Send text-only requests.");
                default:
                    throw new ApiError(400, $"unsupported input item type '{type}'");
            }
        }
        return messages;
    }

    static JsonObject ChatToResponse(JsonNode? chat, string modelAlias)
    {
        var choice = FirstChoice(chat as JsonObject) ?? new JsonObject();
        var msg = choice["message"] as JsonObject ?? new JsonObject();
        var output = new JsonArray();
        var content = msg["content"]?.GetValue<string>() ?? "";
        if (content.Length > 0)
        {
            output.Add(new JsonObject
            {
                ["id"] = $"msg_{Guid.NewGuid():N}",
                ["type"] = "message",
                ["status"] = "completed",
                ["role"] = "assistant",
                ["content"] = new JsonArray(new JsonObject { ["type"] = "output_text", ["text"] = content }),
            });
        }
        if (msg["tool_calls"] is JsonArray tcs)
            foreach (var tc in tcs)
            {
                var fn = tc?["function"] as JsonObject ?? new JsonObject();
                var fc = new JsonObject
                {
                    ["id"] = $"fc_{Guid.NewGuid():N}",
                    ["type"] = "function_call",
                    ["status"] = "completed",
                    ["call_id"] = tc?["id"]?.GetValue<string>() ?? "",
                    ["name"] = fn["name"]?.GetValue<string>() ?? "",
                    ["arguments"] = fn["arguments"]?.GetValue<string>() ?? "{}",
                };
                if (msg["reasoning_content"]?.GetValue<string>() is { Length: > 0 } rc) fc["reasoning_content"] = rc;
                output.Add(fc);
            }
        var usage = chat?["usage"] as JsonObject ?? new JsonObject();
        var finish = choice["finish_reason"]?.GetValue<string>();
        var status = finish is null or "stop" or "tool_calls" or "length" ? "completed" : "incomplete";
        return new JsonObject
        {
            ["id"] = $"resp_{Guid.NewGuid():N}",
            ["object"] = "response",
            ["created_at"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            ["status"] = status,
            ["model"] = modelAlias,
            ["output"] = output,
            ["output_text"] = content,
            ["usage"] = new JsonObject
            {
                ["input_tokens"] = usage["prompt_tokens"]?.GetValue<int>() ?? 0,
                ["output_tokens"] = usage["completion_tokens"]?.GetValue<int>() ?? 0,
                ["total_tokens"] = usage["total_tokens"]?.GetValue<int>() ?? 0,
            },
        };
    }

    public static async Task Responses(HttpContext ctx)
    {
        var body = await ParseBody(ctx);
        if (body is null) throw new ApiError(400, "request body must be JSON");
        if (body!["previous_response_id"]?.GetValue<string>() is { Length: > 0 })
            throw new ApiError(400, "previous_response_id is not supported (pass full input history)");

        var model = body["model"]?.GetValue<string>() ?? "auto";

        // Block image payloads supplied as top-level input items up front, so
        // we never spend credits upstream. ResponsesInputToMessages would
        // otherwise silently drop `input_image` items.
        if (body["input"] is JsonArray topIn)
        {
            foreach (var n in topIn)
            {
                if (n is JsonObject o && (o["type"]?.GetValue<string>()?.ToLowerInvariant() is "input_image" or "image" or "image_url"))
                    throw new ApiError(400, "image input is disabled in llm-gateway. Send text-only requests.");
            }
        }

        // translate Responses API request -> chat.completions request
        var chatPayload = new JsonObject
        {
            ["messages"] = ResponsesInputToMessages(body["input"], body["instructions"]?.GetValue<string>()),
        };
        if (chatPayload["messages"] is not JsonArray mArr || mArr.Count == 0)
            throw new ApiError(400, "input is empty");

        var rag = (Injected: false, Chunks: 0, Bytes: 0);
        if (Env.RagEnabled && (body["rag"]?.GetValue<bool>() ?? true))
        {
            var (consumer, docsEnabled, optin) = Rag.RequestHeaders(ctx);
            if (optin)
            {
                var injected = await Rag.InjectAsync(mArr, consumer, docsEnabled);
                chatPayload["messages"] = injected;
                rag = Rag.RagMeta(injected);
            }
        }

        foreach (var (src, dst) in new[]
        {
            ("max_output_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p"),
            ("stop", "stop"), ("presence_penalty", "presence_penalty"), ("frequency_penalty", "frequency_penalty"),
            ("seed", "seed"), ("user", "user"),
        })
        {
            if (body[src] is not null) chatPayload[dst] = body[src]!.DeepClone();
        }
        if (body["tools"] is not null)
        {
            chatPayload["tools"] = body["tools"]!.DeepClone();
            if (body["tool_choice"] is not null) chatPayload["tool_choice"] = body["tool_choice"]!.DeepClone();
        }

        if (body["stream"]?.GetValue<bool>() ?? false)
            throw new ApiError(501, "streaming /v1/responses is not supported yet; use stream:false or /v1/chat/completions");

        var routes = await Providers.ResolveAsync(model);
        var t00 = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        var attempts = new List<ReqLog.Attempt>();
        var respWaitStart = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        string? lastErr = null;
    respRetry:
        foreach (var (key, upstream) in routes)
        {
            var t0 = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            var payload = (JsonObject)chatPayload.DeepClone();
            payload["model"] = upstream;
            Providers.EnsureReasoningEcho(payload["messages"] as JsonArray, Providers.IsThinkingModel(upstream));
                Providers.NormalizeReasoningEffort(payload, key);
            HttpResponseMessage resp;
            try { (resp, _) = await Providers.UpstreamAsync(key, "/chat/completions", payload, false,
                SessionId(ctx, body["messages"] as JsonArray)); }
            catch (Exception e)
            {
                lastErr = $"{Providers.ByKey(key)?.Label} unreachable: {e.GetType().Name}";
                attempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                    "unreachable", 0, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0,
                    $"{e.GetType().Name}: {e.Message}"));
                continue;
            }
            using (resp)
            {
                var status = (int)resp.StatusCode;
                var retryable = Providers.FallbackStatus.Contains(status)
                    && (routes.Count > 1
                        || (Env.DegradeOnFailure && Providers.DegradableStatus.Contains(status)));
                if (retryable)
                {
                    var detail = await ErrorSnippetAsync(resp);
                    lastErr = $"{Providers.ByKey(key)?.Label} HTTP {status}";
                    attempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                        "fallback-status", status, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0, detail));
                    continue;
                }
                var raw = await resp.Content.ReadAsByteArrayAsync();
                JsonNode? chat;
                try { chat = JsonNode.Parse(raw); }
                catch { throw new ApiError(502, $"upstream returned non-JSON (HTTP {status})"); }

                if (status < 400)
                {
                    var reply = FirstChoice(chat as JsonObject)?["message"]?["content"]?.GetValue<string>() ?? "";
                    TeeTurn(ctx, chatPayload["messages"] as JsonArray, reply);
                    if (Providers.IsThinkingModel(upstream)) Providers.CacheReasoningEcho(chat);
                }
                else
                {
                    attempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                        "upstream-error", status, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0,
                        Encoding.UTF8.GetString(raw)[..Math.Min(400, raw.Length)]));
                }
                var latency = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0;
                var (u, cached) = UsageFrom(chat);
                RecordStat(ctx, "responses", false, status, latency, rag, u);
                attempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                    "ok", status, latency, null));

                if (status >= 400)
                {
                    ReqLog.Write(ctx, "responses", model, false,
                        status, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t00, attempts,
                        new Stats.RagMeta(rag.Injected, rag.Chunks, rag.Bytes), u, cached, null);
                    ctx.Response.StatusCode = status;
                    ctx.Response.ContentType = "application/json";
                    await ctx.Response.WriteAsync(chat!.ToJsonString());
                    return;
                }
                ReqLog.Write(ctx, "responses", model, false,
                    status, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t00, attempts,
                    new Stats.RagMeta(rag.Injected, rag.Chunks, rag.Bytes), u, cached, null);
                ctx.Response.StatusCode = 200;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync(ChatToResponse(chat, model).ToJsonString());
                return;
            }
        }
        while (Env.DegradeOnFailure && !ctx.RequestAborted.IsCancellationRequested
               && DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - respWaitStart
                    < Env.DegradeMaxWaitSeconds * 1000L)
        {
            try { await Task.Delay(TimeSpan.FromSeconds(Env.RecoveryKeepAliveSeconds), ctx.RequestAborted); }
            catch (OperationCanceledException) { break; }
            var back = new List<(string Key, string Upstream)>();
            foreach (var (k, up) in routes)
                if (await Providers.ProbeAsync(k) is { Reachable: true })
                    back.Add((k, up));
            if (back.Count > 0)
            {
                routes = back;
                attempts = new List<ReqLog.Attempt>();
                goto respRetry;
            }
        }
        var respSummary = SummarizeFailures(model, attempts);
        ReqLog.Write(ctx, "responses", model, false,
            502, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t00, attempts,
            new Stats.RagMeta(rag.Injected, rag.Chunks, rag.Bytes), new Stats.Usage(0, 0, 0), 0,
            respSummary);
        if (Env.DegradeOnFailure)
        {
            // Reuse the chat->response shaper so the degraded payload has exactly
            // the shape a real /v1/responses reply would have.
            var degraded = DegradedChat(model, DegradedText(model, attempts));
            MarkDegraded(ctx, 502);
            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(ChatToResponse(degraded, model).ToJsonString());
            return;
        }
        throw new ApiError(502, respSummary);
    }
}
