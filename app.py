"""llm-gateway — OpenAI-compatible LLM gateway for the Sutliff homelab.

Serves /v1/chat/completions + /v1/responses, aggregating multiple upstream LLM
providers. Model names are prefixed by provider (case-insensitive):

  OCG-<model>    -> opencode-go   (https://opencode.ai/zen/go/v1)
  DPSK-<model>   -> deepseek      (https://api.deepseek.com/v1)
  AGNO-<model>   -> any provider that is available AND serving <model>
  auto           -> OCG-<AUTO_MODEL> with fallback to DPSK-<AUTO_MODEL>

LAN-only service (llm.homelab.internal); never exposed through the tunnel.
"""

import json
import os
import time
import uuid

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

APP_NAME = "llm-gateway"
PORT = int(os.environ.get("PORT", "8080"))
AUTO_MODEL = os.environ.get("AUTO_MODEL", "deepseek-v4-flash")
MODELS_CACHE_TTL = int(os.environ.get("MODELS_CACHE_TTL", "300"))

# RAG injection (v1.1.0) — query the embeddings service with the last user
# message and prepend retrieved context as a system message. Fail-open:
# any error leaves the request untouched.
RAG_ENABLED = os.environ.get("RAG_ENABLED", "1") not in ("0", "false", "False")
RAG_BASE_URL = os.environ.get("RAG_BASE_URL", "http://embeddings.homelab.internal").rstrip("/")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.15"))
RAG_MAX_CHARS = int(os.environ.get("RAG_MAX_CHARS", "1800"))

# provider key -> (label, base_url, api_key)
PROVIDERS = {
    "ocg": {
        "label": "OCG",
        "base_url": os.environ["OCG_BASE_URL"].rstrip("/"),
        "api_key": os.environ["OCG_API_KEY"],
    },
    "dpsk": {
        "label": "DPSK",
        "base_url": os.environ["DPSK_BASE_URL"].rstrip("/"),
        "api_key": os.environ["DPSK_API_KEY"],
    },
}
# AGNO preference order (first available+serving wins)
AGNO_ORDER = [k for k in os.environ.get("AGNO_ORDER", "ocg,dpsk").split(",") if k in PROVIDERS]

# fallback on these upstream outcomes for auto/AGNO chains
_FALLBACK_STATUS = {401, 403, 408, 429, 500, 502, 503, 504}

app = FastAPI(title=APP_NAME, version="1.1.0")

_models_cache: dict[str, tuple[float, set[str] | None]] = {}


def _headers(provider: dict) -> dict:
    return {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }


async def _provider_models(key: str) -> set[str] | None:
    """Cached set of model ids a provider serves (None = couldn't verify)."""
    now = time.monotonic()
    hit = _models_cache.get(key)
    if hit and now - hit[0] < MODELS_CACHE_TTL:
        return hit[1]
    prov = PROVIDERS[key]
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{prov['base_url']}/models", headers=_headers(prov))
            r.raise_for_status()
            ids = {m.get("id") for m in r.json().get("data", []) if m.get("id")}
    except Exception:
        ids = None
    _models_cache[key] = (now, ids)
    return ids


class _ApiError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail


@app.exception_handler(_ApiError)
async def _api_error_handler(_req, exc: _ApiError):
    return JSONResponse(
        status_code=exc.status,
        content={"error": {"message": exc.detail, "type": "invalid_request_error", "code": None}},
    )


def _resolve(model: str | None) -> list[tuple[str, str]]:
    """Return ordered (provider_key, upstream_model) chain for a model name."""
    if not model or not str(model).strip():
        raise _ApiError(400, "model is required")
    m = str(model).strip()
    low = m.lower()

    if low == "auto":
        chain = []
        for k in AGNO_ORDER:
            chain.append((k, AUTO_MODEL))
        if not chain:
            chain = [("ocg", AUTO_MODEL), ("dpsk", AUTO_MODEL)]
        return chain

    for key in ("ocg", "dpsk", "agno"):
        prefix = key + "-"
        if low.startswith(prefix):
            upstream = m[len(prefix):].strip()
            if not upstream:
                raise _ApiError(400, f"model '{m}': missing upstream model after '{key}-'")
            if key in ("ocg", "dpsk"):
                return [(key, upstream)]
            # AGNO: providers that are up AND serving this model, in order
            chain = []
            for pk in AGNO_ORDER:
                ids = await _provider_models(pk)
                if ids is None:
                    continue  # can't verify availability -> skip for AGNO
                if upstream in ids or upstream == AUTO_MODEL:
                    chain.append((pk, upstream))
            if not chain:
                raise _ApiError(404, f"AGNO: no provider available and serving '{upstream}'")
            return chain

    raise _ApiError(
        404,
        f"unknown model '{m}'. Use a provider prefix (OCG-/DPSK-/AGNO-) or 'auto'.",
    )


async def _upstream(method: str, path: str, body: dict, key: str):
    """Fire one request at one provider. Returns (client, response)."""
    prov = PROVIDERS[key]
    url = f"{prov['base_url']}{path}"
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=600, write=60, pool=10)
    )
    try:
        r = await client.request(method, url, headers=_headers(prov), json=body)
    except Exception:
        await client.aclose()
        raise
    return client, r


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": APP_NAME,
        "providers": list(PROVIDERS),
        "rag": {"enabled": RAG_ENABLED, "base_url": RAG_BASE_URL},
    }


@app.get("/v1/models")
async def list_models():
    ids = ["auto"]
    for key, prov in PROVIDERS.items():
        served = await _provider_models(key)
        if served is None:
            served = {AUTO_MODEL}  # can't verify -> advertise the default only
        for mid in sorted(served):
            ids.append(f"{prov['label']}-{mid}")
    # AGNO-<model> for the union of everything any provider serves
    union: set[str] = set()
    for key in PROVIDERS:
        served = await _provider_models(key)
        if served:
            union |= served
    if not union:
        union = {AUTO_MODEL}
    for mid in sorted(union):
        ids.append(f"AGNO-{mid}")
    # de-dup preserving order
    seen, out = set(), []
    for mid in ids:
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": int(time.time()), "owned_by": "llm-gateway"}
            for mid in out
        ],
    }


async def _rag_retrieve(query: str) -> list[dict]:
    """Ask the embeddings service for top-k semantic hits. Fail-open: [] on any error."""
    if not RAG_ENABLED:
        return []
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.post(
                f"{RAG_BASE_URL}/search",
                json={"query": query, "top_k": RAG_TOP_K, "min_score": RAG_MIN_SCORE},
            )
            r.raise_for_status()
            return r.json().get("results", [])
    except Exception:
        return []


def _rag_system_message(query: str, hits: list[dict]) -> str:
    """Format retrieved chunks into a compact system-prompt context block."""
    parts = []
    for h in hits[:RAG_TOP_K]:
        src = h.get("source", "?")
        title = h.get("title", "")
        text = (h.get("text") or "")[:RAG_MAX_CHARS]
        score = h.get("score", 0)
        parts.append(f"--- [{src}] {title} (score {score})\n{text}")
    return (
        "Retrieved context (homelab docs + session history, use only if relevant):\n\n"
        + "\n\n".join(parts)
    )


def _last_user_message(messages: list) -> str:
    """Find the most recent user turn to query RAG against."""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
            return m["content"].strip()
    return ""


async def _inject_rag(messages: list) -> list:
    """Return messages with a RAG context system message injected (fail-open: unchanged)."""
    query = _last_user_message(messages)
    if not query:
        return messages
    hits = await _rag_retrieve(query)
    if not hits:
        return messages
    ctx = _rag_system_message(query, hits)
    # prepend as the FIRST system message (before any client system prompt is fine —
    # providers treat the first system message as the primary prompt; appending context
    # to it would mutate the client's prompt. A separate leading system message is the
    # standard RAG pattern and keeps the client's system prompt intact.)
    return [{"role": "system", "content": ctx}] + messages


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise _ApiError(400, "request body must be JSON")
    model = body.get("model")
    routes = _resolve(model)
    want_stream = bool(body.get("stream"))

    # RAG: inject retrieved context into the message list (fail-open)
    if RAG_ENABLED and body.get("rag") is not False:
        messages = body.get("messages") or []
        if isinstance(messages, list):
            messages = await _inject_rag(messages)
            body = {**body, "messages": messages}

    if not want_stream:
        last_err = None
        for key, upstream in routes:
            payload = dict(body)
            payload["model"] = upstream
            try:
                client, r = await _upstream("POST", "/chat/completions", payload, key)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                await client.aclose()
                last_err = f"{PROVIDERS[key]['label']} unreachable: {type(e).__name__}"
                continue
            if r.status_code in _FALLBACK_STATUS and len(routes) > 1:
                last_err = f"{PROVIDERS[key]['label']} HTTP {r.status_code}"
                await client.aclose()
                continue
            try:
                content = r.json()
            except Exception:
                content = None
            await client.aclose()
            if content is not None:
                return JSONResponse(status_code=r.status_code, content=content)
            return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
        raise _ApiError(502, f"all providers failed for model '{model}': {last_err or 'unknown'}")

    # streaming: pick the first provider that responds 2xx, stream its SSE
    last_err = None
    for key, upstream in routes:
        payload = dict(body)
        payload["model"] = upstream
        try:
            client, r = await _upstream("POST", "/chat/completions", payload, key)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            await client.aclose()
            last_err = f"{PROVIDERS[key]['label']} unreachable: {type(e).__name__}"
            continue
        if r.status_code >= 400:
            last_err = f"{PROVIDERS[key]['label']} HTTP {r.status_code}"
            # still report the body of the first failure if it's the only route
            if len(routes) == 1:
                await client.aclose()
                return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
            await client.aclose()
            continue

        async def stream(client=client, r=r):
            try:
                async for chunk in r.aiter_bytes():
                    yield chunk
            finally:
                await client.aclose()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    raise _ApiError(502, f"all providers failed for model '{model}': {last_err or 'unknown'}")


# ---------------- /v1/responses (OpenAI Responses API) ----------------

def _responses_input_to_messages(inp, instructions=None):
    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    if isinstance(inp, str):
        if inp.strip():
            messages.append({"role": "user", "content": inp})
        return messages
    if not isinstance(inp, list):
        raise _ApiError(400, "input must be a string or array of items")
    for item in inp:
        t = item.get("type")
        if t == "message":
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text")
                )
            else:
                text = ""
            if text.strip():
                messages.append({"role": role, "content": text})
        elif t == "function_call":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id") or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments") or "{}",
                    },
                }],
            })
        elif t == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id") or "",
                "content": str(item.get("output", "")),
            })
        elif t == "reasoning":
            continue  # reasoning summaries don't map to chat messages
        else:
            raise _ApiError(400, f"unsupported input item type '{t}'")
    return messages


def _chat_to_response(chat: dict, model_alias: str) -> dict:
    choice = (chat.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    output = []
    if msg.get("content"):
        output.append({
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": msg["content"]}],
        })
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) or {}
        output.append({
            "id": f"fc_{uuid.uuid4().hex}",
            "type": "function_call",
            "status": "completed",
            "call_id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
        })
    usage = chat.get("usage", {}) or {}
    finish = choice.get("finish_reason")
    status = "completed" if finish in (None, "stop", "tool_calls", "length") else "incomplete"
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model_alias,
        "output": output,
        "output_text": msg.get("content") or "",
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


@app.post("/v1/responses")
async def responses(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise _ApiError(400, "request body must be JSON")

    if body.get("previous_response_id"):
        raise _ApiError(400, "previous_response_id is not supported (pass full input history)")

    model = body.get("model") or "auto"

    # translate Responses API request -> chat.completions request
    chat_payload: dict = {
        "messages": _responses_input_to_messages(body.get("input"), body.get("instructions")),
    }
    if not chat_payload["messages"]:
        raise _ApiError(400, "input is empty")
    # RAG: inject retrieved context into the message list (fail-open)
    if RAG_ENABLED and body.get("rag") is not False:
        chat_payload["messages"] = await _inject_rag(chat_payload["messages"])
    for src, dst in (
        ("max_output_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop", "stop"),
        ("presence_penalty", "presence_penalty"),
        ("frequency_penalty", "frequency_penalty"),
        ("seed", "seed"),
        ("user", "user"),
    ):
        if src in body and body[src] is not None:
            chat_payload[dst] = body[src]
    if body.get("tools"):
        chat_payload["tools"] = body["tools"]
        if body.get("tool_choice"):
            chat_payload["tool_choice"] = body["tool_choice"]

    if body.get("stream"):
        raise _ApiError(501, "streaming /v1/responses is not supported yet; use stream:false or /v1/chat/completions")

    routes = _resolve(model)
    last_err = None
    for key, upstream in routes:
        payload = dict(chat_payload)
        payload["model"] = upstream
        try:
            client, r = await _upstream("POST", "/chat/completions", payload, key)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            await client.aclose()
            last_err = f"{PROVIDERS[key]['label']} unreachable: {type(e).__name__}"
            continue
        if r.status_code in _FALLBACK_STATUS and len(routes) > 1:
            last_err = f"{PROVIDERS[key]['label']} HTTP {r.status_code}"
            await client.aclose()
            continue
        try:
            chat = r.json()
        except Exception:
            await client.aclose()
            raise _ApiError(502, f"upstream returned non-JSON (HTTP {r.status_code})")
        await client.aclose()
        if r.status_code >= 400:
            return JSONResponse(status_code=r.status_code, content=chat)
        return JSONResponse(content=_chat_to_response(chat, str(model)))
    raise _ApiError(502, f"all providers failed for model '{model}': {last_err or 'unknown'}")
