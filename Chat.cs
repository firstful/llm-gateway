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

    static string? SessionId(HttpContext ctx)
        => ctx.Request.Cookies[Env.SessionCookie];

    static void SaveSessionFrom(HttpContext ctx, JsonArray? messages, string model, string reply)
    {
        var sid = SessionId(ctx);
        if (string.IsNullOrEmpty(sid) || string.IsNullOrWhiteSpace(reply)) return;
        var msgs = new JsonArray();
        if (messages is not null)
            foreach (var m in messages) msgs.Add(m?.DeepClone());
        msgs.Add(new JsonObject { ["role"] = "assistant", ["content"] = reply });
        Sessions.Save(sid, msgs, model);
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

    // ---------------- /v1/chat/completions ----------------

    public static async Task ChatCompletions(HttpContext ctx)
    {
        var body = await ParseBody(ctx);
        if (body is null) throw new ApiError(400, "request body must be JSON");
        var model = body!["model"]?.GetValue<string>();
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
            string? lastErr = null;
            foreach (var (key, upstream) in routes)
            {
                var t0 = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                var payload = (JsonObject)body.DeepClone();
                payload["model"] = upstream;
                Providers.EnsureReasoningEcho(payload["messages"] as JsonArray, Providers.IsThinkingModel(upstream));
                HttpResponseMessage resp;
                try { (resp, _) = await Providers.UpstreamAsync(key, "/chat/completions", payload, false); }
                catch (Exception e)
                {
                    lastErr = $"{Providers.ByKey(key)?.Label} unreachable: {e.GetType().Name}";
                    nsAttempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                        "unreachable", 0, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0,
                        $"{e.GetType().Name}: {e.Message}"));
                    continue;
                }
                using (resp)
                {
                    var status = (int)resp.StatusCode;
                    if (Providers.FallbackStatus.Contains(status) && routes.Count > 1)
                    {
                        var detail = await ErrorSnippetAsync(resp);
                        lastErr = $"{Providers.ByKey(key)?.Label} HTTP {status}";
                        nsAttempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                            "fallback-status", status, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0, detail));
                        continue;
                    }
                    var raw = await resp.Content.ReadAsByteArrayAsync();
                    JsonNode? content = null;
                    try { content = JsonNode.Parse(raw); } catch { /* fall through to raw */ }

                    if (status < 400)
                    {
                        var reply = FirstChoice(content as JsonObject)?["message"]?["content"]?.GetValue<string>() ?? "";
                        TeeTurn(ctx, body["messages"] as JsonArray, reply);
                        if (Providers.IsThinkingModel(upstream)) Providers.CacheReasoningEcho(content);
                        SaveSessionFrom(ctx, body["messages"] as JsonArray, model ?? "", reply);
                    }
                    else
                    {
                        nsAttempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                            "upstream-error", status, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0,
                            Encoding.UTF8.GetString(raw)[..Math.Min(400, raw.Length)]));
                    }
                    var latency = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0;
                    var (u, cached) = UsageFrom(content);
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
                        await ctx.Response.WriteAsync(content.ToJsonString());
                        return;
                    }
                    ctx.Response.StatusCode = status;
                    ctx.Response.ContentType = resp.Content.Headers.ContentType?.ToString() ?? "application/json";
                    await ctx.Response.Body.WriteAsync(raw);
                    return;
                }
            }
            ReqLog.Write(ctx, "chat", model, false,
                502, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - nsT00, nsAttempts,
                new Stats.RagMeta(rag.Injected, rag.Chunks, rag.Bytes), new Stats.Usage(0, 0, 0), 0,
                SummarizeFailures(model ?? "auto", nsAttempts));
            throw new ApiError(502, SummarizeFailures(model ?? "auto", nsAttempts));
        }

        // streaming: pick the first provider that responds 2xx, stream its SSE
        var t00 = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        var attempts = new List<ReqLog.Attempt>();
        string? streamErr = null;
    streamRetry:
        foreach (var (key, upstream) in routes)
        {
            var payload = (JsonObject)body.DeepClone();
            payload["model"] = upstream;
            Providers.EnsureReasoningEcho(payload["messages"] as JsonArray, Providers.IsThinkingModel(upstream));
            HttpResponseMessage resp;
            var t0 = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            try { (resp, _) = await Providers.UpstreamAsync(key, "/chat/completions", payload, true); }
            catch (Exception e)
            {
                streamErr = $"{Providers.ByKey(key)?.Label} unreachable: {e.GetType().Name}";
                attempts.Add(new ReqLog.Attempt(Providers.ByKey(key)?.Label ?? key, upstream,
                    "unreachable", 0, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0,
                    $"{e.GetType().Name}: {e.Message}"));
                continue;
            }
            var status = (int)resp.StatusCode;
            if (status >= 400)
            {
                streamErr = $"{Providers.ByKey(key)?.Label} HTTP {status}";
                if (routes.Count == 1)
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
            }
            var sid = SessionId(ctx);
            var chunks = new MemoryStream();
            try
            {
                await using var stream = await resp.Content.ReadAsStreamAsync();
                var buf = new byte[81920];
                int read;
                while ((read = await stream.ReadAsync(buf)) > 0)
                {
                    chunks.Write(buf, 0, read);
                    await ctx.Response.Body.WriteAsync(buf.AsMemory(0, read));
                    await ctx.Response.Body.FlushAsync();
                }
            }
            finally
            {
                var raw = chunks.ToArray();
                var reply = SseReplyText(raw);
                if (status < 400)
                {
                    TeeTurn(ctx, body["messages"] as JsonArray, reply);
                    if (Providers.IsThinkingModel(upstream)) Providers.CacheReasoningEchoFromSse(raw);
                    if (!string.IsNullOrEmpty(sid) && !string.IsNullOrWhiteSpace(reply))
                    {
                        var msgs = new JsonArray();
                        if (body["messages"] is JsonArray orig)
                            foreach (var m in orig) msgs.Add(m?.DeepClone());
                        msgs.Add(new JsonObject { ["role"] = "assistant", ["content"] = reply });
                        Sessions.Save(sid, msgs, model ?? "");
                    }
                }
                var latency = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t0;
                var (u, cached) = SseUsage(raw);
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
            var waitStart = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            var capMs = Env.RecoveryMaxWaitSeconds * 1000L;
            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "text/event-stream";
            ctx.Response.Headers["Cache-Control"] = "no-cache";
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
                if (recovered.Count > 0)
                {
                    routes = recovered;
                    attempts = new List<ReqLog.Attempt>();
                    goto streamRetry;
                }
                if (DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - waitStart > capMs)
                    break;
            }
            var gaveup = SummarizeFailures(model ?? "auto", attempts);
            ReqLog.Write(ctx, "chat", model, true,
                502, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t00, attempts,
                new Stats.RagMeta(rag.Injected, rag.Chunks, rag.Bytes), new Stats.Usage(0, 0, 0), 0, gaveup);
            try
            {
                await ctx.Response.WriteAsync($"data: {{\"error\":{{\"message\":{JsonSerializer.Serialize(gaveup)}}}}}\n\n");
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
        string? lastErr = null;
        foreach (var (key, upstream) in routes)
        {
            var t0 = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            var payload = (JsonObject)chatPayload.DeepClone();
            payload["model"] = upstream;
            Providers.EnsureReasoningEcho(payload["messages"] as JsonArray, Providers.IsThinkingModel(upstream));
            HttpResponseMessage resp;
            try { (resp, _) = await Providers.UpstreamAsync(key, "/chat/completions", payload, false); }
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
                if (Providers.FallbackStatus.Contains(status) && routes.Count > 1)
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
        ReqLog.Write(ctx, "responses", model, false,
            502, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - t00, attempts,
            new Stats.RagMeta(rag.Injected, rag.Chunks, rag.Bytes), new Stats.Usage(0, 0, 0), 0,
            SummarizeFailures(model, attempts));
        throw new ApiError(502, SummarizeFailures(model, attempts));
    }
}
