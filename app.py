"""llm-gateway — OpenAI-compatible LLM gateway for the Sutliff homelab.

Serves /v1/chat/completions + /v1/responses, aggregating multiple upstream LLM
providers. Model names are prefixed by provider (case-insensitive):

  OCG-<model>    -> opencode-go   (https://opencode.ai/zen/go/v1)
  DPSK-<model>   -> deepseek      (https://api.deepseek.com/v1)
  LOCAL-<model>  -> local GPU     (llama-ctrl -> llama-server on the RX 470)
  AGNO-<model>   -> any provider that is available AND serving <model>
  auto           -> OCG-<AUTO_MODEL> with fallback to DPSK-<AUTO_MODEL>, then
                    LOCAL-<LOCAL_MODEL> (final fallback)

v1.7.0: chat sessions are persisted server-side (SQLite + cookie), the UI
shows live provider + local-LLM status, and AGNO-<model> is only advertised
when >=2 providers actually serve it.

LAN-only service (llm.homelab.internal); never exposed through the tunnel.
Serves a simple chat UI at /ui (model selection + streaming chat).
"""

import asyncio
import json
import os
import sqlite3
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

APP_NAME = "llm-gateway"
APP_VERSION = "1.7.4"
PORT = int(os.environ.get("PORT", "8080"))
AUTO_MODEL = os.environ.get("AUTO_MODEL", "deepseek-v4-flash")
# local GPU fallback (llama-ctrl) — the LAST rung of the auto fallback chain
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "lfm2.5-2.6b")
MODELS_CACHE_TTL = int(os.environ.get("MODELS_CACHE_TTL", "300"))

# RAG injection (v1.2.0) — header-driven control plane:
#   X-Consumer: <id>       scope session embeddings to this consumer (docs stay shared)
#   X-RAG-Docs: 1|0        include docs source (default on)
#   X-RAG-Optin: 1|0       master switch — 0 disables RAG for this request (default on)
# Behavior is backward-compatible: a request with NO RAG headers behaves exactly like
# v1.1 (docs + all sessions, opt-in on). Fail-open: any search error leaves the
# request untouched.
RAG_ENABLED = os.environ.get("RAG_ENABLED", "1") not in ("0", "false", "False")
RAG_BASE_URL = os.environ.get("RAG_BASE_URL", "http://embeddings.homelab.internal").rstrip("/")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.15"))
RAG_MAX_CHARS = int(os.environ.get("RAG_MAX_CHARS", "1800"))

# Live session ingestion (v1.3.0) — tee every turn through a deferred queue into
# the embeddings service. The gateway sees each request/response as it happens,
# so it enqueues the user msg + assistant reply as a paired chunk
# (id 'live:<consumer>:<ts>'). A background flusher drains the queue to
# /ingest/sessions. This is the PRIMARY session-memory path (near-real-time);
# the NFS snapshot (/refresh sources) is backup/catchup only (initial backfill +
# recovery after queue loss).
LIVE_INGEST_ENABLED = os.environ.get("LIVE_INGEST_ENABLED", "1") not in ("0", "false", "False")
LIVE_FLUSH_SECONDS = float(os.environ.get("LIVE_FLUSH_SECONDS", "3"))
LIVE_FLUSH_BATCH = int(os.environ.get("LIVE_FLUSH_BATCH", "20"))

_live_queue: asyncio.Queue = asyncio.Queue()
_live_flusher_task = None

# ---------------- chat session persistence (v1.7.0) ----------------
# Server-side chat history keyed by a cookie. The UI sends the cookie on every
# request; a session survives browser close / window loss — the full transcript
# (plus any in-flight partial reply) is stored and restored on return.
SESSION_DB = os.environ.get("SESSION_DB", "/data/sessions.db")
SESSION_COOKIE = "llm_session"
SESSION_TTL_DAYS = int(os.environ.get("SESSION_TTL_DAYS", "30"))

_sqlite_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    """Thread-safe SQLite connection (per-call, WAL)."""
    conn = sqlite3.connect(SESSION_DB, timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db():
    Path(os.path.dirname(SESSION_DB) or ".").mkdir(parents=True, exist_ok=True)
    with _sqlite_lock, _db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                messages TEXT NOT NULL,      -- JSON array of {role, content}
                model TEXT NOT NULL DEFAULT 'auto',
                updated_at REAL NOT NULL
            )"""
        )


def _load_session(sid: str) -> tuple[list[dict], str] | None:
    if not sid:
        return None
    try:
        with _sqlite_lock, _db() as conn:
            row = conn.execute(
                "SELECT messages, model FROM sessions WHERE id = ?", (sid,)
            ).fetchone()
        if not row:
            return None
        return json.loads(row[0]), row[1]
    except Exception:
        return None


def _save_session(sid: str, messages: list[dict], model: str):
    try:
        with _sqlite_lock, _db() as conn:
            conn.execute(
                """INSERT INTO sessions (id, messages, model, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     messages = excluded.messages,
                     model = excluded.model,
                     updated_at = excluded.updated_at""",
                (sid, json.dumps(messages), model, time.time()),
            )
    except Exception:
        pass


def _session_id_from(request: Request) -> str:
    return request.cookies.get(SESSION_COOKIE, "")


# ---- live local-LLM status cache (v1.7.0) ----
# The gateway proxies llama-ctrl's /status for the UI. It's cached briefly so
# the UI doesn't hammer ctrl (which would wake the server on status polls).
LOCAL_CTRL_URL = os.environ.get(
    "LOCAL_CTRL_URL", "http://llama-ctrl.llama.svc.cluster.local"
).rstrip("/")
_local_status: dict | None = None
_local_status_at = 0.0
_LOCAL_STATUS_TTL = 5.0


def _local_status_sync() -> dict | None:
    """Best-effort snapshot of llama-ctrl /status (None = ctrl unreachable)."""
    global _local_status, _local_status_at
    now = time.monotonic()
    if _local_status is not None and now - _local_status_at < _LOCAL_STATUS_TTL:
        return _local_status
    try:
        r = httpx.get(f"{LOCAL_CTRL_URL}/status", timeout=3)
        r.raise_for_status()
        _local_status = r.json()
    except Exception:
        _local_status = None
    _local_status_at = now
    return _local_status

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
    "local": {
        "label": "LOCAL",
        "base_url": os.environ.get(
            "LOCAL_BASE_URL", "http://llama-ctrl.llama.svc.cluster.local"
        ).rstrip("/"),
        # no auth — llama-ctrl is LAN-only in the same trust domain
        "api_key": os.environ.get("LOCAL_API_KEY", ""),
    },
}
# AGNO preference order (first available+serving wins)
AGNO_ORDER = [k for k in os.environ.get("AGNO_ORDER", "ocg,dpsk").split(",") if k in PROVIDERS]

# fallback on these upstream outcomes for auto/AGNO chains
_FALLBACK_STATUS = {401, 403, 408, 429, 500, 502, 503, 504}

# DeepSeek thinking-mode echo: upstream (OCG/DPSK deepseek-v4-flash) REQUIRES
# `reasoning_content` to be echoed back on assistant tool_call messages in
# follow-up turns, or it 400s ("The reasoning_content in the thinking mode must
# be passed back to the API"). Some clients (Hermes' non-streaming path, bare
# OpenAI clients) omit it. We cache what the upstream returns (keyed by
# tool_call id) and inject it — or an empty echo, which upstream accepts — so
# multi-turn tool loops never break. Applies to both /v1/chat/completions and
# /v1/responses.
_reasoning_echo: dict[str, str] = {}
_REASONING_ECHO_MAX = 4096


def _is_thinking_model(name: str) -> bool:
    low = (name or "").lower()
    return any(tok in low for tok in ("deepseek", "glm", "reasoner", "think"))


def _cache_reasoning_echo(content: dict | None) -> None:
    """Cache reasoning_content from an upstream chat.completions response,
    keyed by the tool_call ids it accompanied."""
    if not isinstance(content, dict):
        return
    for choice in content.get("choices") or []:
        msg = choice.get("message") or {}
        rc = msg.get("reasoning_content")
        if not rc:
            continue
        for tc in msg.get("tool_calls") or []:
            cid = tc.get("id") if isinstance(tc, dict) else None
            if cid:
                _reasoning_echo[cid] = rc
    if len(_reasoning_echo) > _REASONING_ECHO_MAX:
        for k in list(_reasoning_echo)[: len(_reasoning_echo) - _REASONING_ECHO_MAX]:
            _reasoning_echo.pop(k, None)


def _cache_reasoning_echo_from_sse(raw: bytes) -> None:
    """Harvest reasoning_content + tool_call ids from a streaming SSE body."""
    pending = ""
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue
        delta = ((obj.get("choices") or [{}])[0]).get("delta") or {}
        rc = delta.get("reasoning_content")
        if rc:
            pending += rc
        for tc in delta.get("tool_calls") or []:
            cid = tc.get("id") if isinstance(tc, dict) else None
            if cid and pending:
                _reasoning_echo[cid] = pending
        if ((obj.get("choices") or [{}])[0]).get("finish_reason"):
            pending = ""
    if len(_reasoning_echo) > _REASONING_ECHO_MAX:
        for k in list(_reasoning_echo)[: len(_reasoning_echo) - _REASONING_ECHO_MAX]:
            _reasoning_echo.pop(k, None)


def _ensure_reasoning_echo(messages, thinking: bool) -> None:
    """Guarantee assistant tool_call messages carry reasoning_content when the
    upstream model is a thinking model. Injects cached reasoning (best) or an
    empty echo (accepted by upstream) so tool loops never 400."""
    if not thinking or not isinstance(messages, list):
        return
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        if not msg.get("tool_calls"):
            continue
        if msg.get("reasoning_content") is not None:
            continue  # client already echoed it (explicitly or not)
        injected = ""
        for tc in msg.get("tool_calls") or []:
            cid = tc.get("id") if isinstance(tc, dict) else None
            if cid and cid in _reasoning_echo:
                injected = _reasoning_echo[cid]
                break
        msg["reasoning_content"] = injected


# ---------------- request statistics (rolling + telemetry) ----------------
# Reed 2026-08-07: track per-request stats over time (RAG bytes injected, token
# usage, latency) so real-world RAG overhead is measurable instead of guessed.
# In-memory rolling window for GET /stats; optionally flushed to telemetry
# (raw-text POST, <=4096B) for durable 90-day history. Fail-open everywhere:
# stats must NEVER break the request path.

_RAG_PREFIX = (
    "The following is authoritative context from the homelab operator's docs and "
    "session history (retrieved via RAG)."
)
STATS_WINDOW = int(os.environ.get("STATS_WINDOW", "5000"))
STATS_FLUSH_SECONDS = int(os.environ.get("STATS_FLUSH_SECONDS", "30"))
TELEMETRY_URL = os.environ.get("TELEMETRY_URL", "").rstrip("/")
TELEMETRY_MAX_BYTES = int(os.environ.get("TELEMETRY_MAX_BYTES", "3800"))

_stats_window: deque = deque(maxlen=STATS_WINDOW)
_stats_pending: deque = deque()          # not yet flushed to telemetry
_stats_since_start: dict = {
    "requests": 0, "with_rag": 0, "rag_bytes": 0, "rag_chunks": 0,
    "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
    "latency_ms": 0, "errors": 0, "dropped_telemetry": 0,
}


def _record_stat(rec: dict) -> None:
    """Append one request record to the rolling window + running totals."""
    if not isinstance(rec, dict):
        return
    _stats_window.append(rec)
    _stats_pending.append(rec)
    t = _stats_since_start
    t["requests"] += 1
    rag = rec.get("rag") or {}
    if rag.get("injected"):
        t["with_rag"] += 1
        t["rag_bytes"] += rag.get("bytes") or 0
        t["rag_chunks"] += rag.get("chunks") or 0
    u = rec.get("usage") or {}
    t["prompt_tokens"] += u.get("prompt") or 0
    t["completion_tokens"] += u.get("completion") or 0
    t["reasoning_tokens"] += u.get("reasoning") or 0
    t["latency_ms"] += rec.get("latency_ms") or 0
    if rec.get("status", 0) >= 400:
        t["errors"] += 1


def _rag_meta(messages: list) -> dict:
    """Extract injected-RAG metadata from the (possibly RAG-prepended) message list."""
    if not messages or not isinstance(messages[0], dict):
        return {"injected": False, "chunks": 0, "bytes": 0}
    first = messages[0]
    content = first.get("content") or ""
    if first.get("role") != "system" or not str(content).startswith(_RAG_PREFIX):
        return {"injected": False, "chunks": 0, "bytes": 0}
    n_chunks = str(content).count("--- [")
    return {"injected": True, "chunks": n_chunks, "bytes": len(str(content))}


def _sse_usage(raw: bytes) -> dict:
    """Extract usage from the final SSE chunk if the upstream includes it."""
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload in ("[DONE]", ""):
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if obj.get("usage"):
            u = obj["usage"]
            return {
                "prompt": u.get("prompt_tokens", 0),
                "completion": u.get("completion_tokens", 0),
                "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            }
    return {}


def _stats_line(rec: dict) -> str:
    """One compact JSON line for telemetry (~150B). Keys are short to stay far
    under the 4096-byte telemetry body cap when batched."""
    rag = rec.get("rag") or {}
    u = rec.get("usage") or {}
    return json.dumps({
        "ts": rec.get("ts", 0),
        "c": (rec.get("consumer") or "-")[:24],
        "ep": rec.get("endpoint", "chat"),
        "s": 1 if rec.get("stream") else 0,
        "st": rec.get("status", 0),
        "ms": rec.get("latency_ms", 0),
        "ri": 1 if rag.get("injected") else 0,
        "rc": rag.get("chunks", 0),
        "rb": rag.get("bytes", 0),
        "pt": u.get("prompt", 0),
        "ct": u.get("completion", 0),
        "rt": u.get("reasoning", 0),
    }, separators=(",", ":"))


async def _stats_flusher():
    """Background task: drain pending stats to telemetry in <=3800B batches.
    Fail-open: any error drops the batch (counted) — never retried forever."""
    while True:
        try:
            await asyncio.sleep(STATS_FLUSH_SECONDS)
            if not TELEMETRY_URL or not _stats_pending:
                continue
            lines: list[str] = []
            batch_size = 0
            while _stats_pending and batch_size < TELEMETRY_MAX_BYTES:
                line = _stats_line(_stats_pending.popleft())
                if batch_size + len(line) + 1 > TELEMETRY_MAX_BYTES:
                    _stats_pending.appendleft(json.loads(line))
                    break
                lines.append(line)
                batch_size += len(line) + 1
            if not lines:
                continue
            try:
                async with httpx.AsyncClient(timeout=5) as c:
                    r = await c.post(TELEMETRY_URL, content="\n".join(lines))
                    if r.status_code >= 400:
                        raise RuntimeError(f"telemetry HTTP {r.status_code}")
            except Exception:
                _stats_since_start["dropped_telemetry"] += len(lines)
        except Exception:
            await asyncio.sleep(STATS_FLUSH_SECONDS)

app = FastAPI(title=APP_NAME, version=APP_VERSION)
_init_db()

# Simple chat UI (model selection + streaming chat) — served at / and /ui
_UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
async def chat_ui():
    return FileResponse(_UI_PATH, media_type="text/html")


@app.get("/local/status", include_in_schema=False)
async def local_status():
    """Live llama-ctrl status (cached 5s) — drives the UI's LOCAL indicator."""
    st = _local_status_sync()
    if st is None:
        return JSONResponse({"reachable": False, "detail": "ctrl unreachable"})
    return JSONResponse({"reachable": True, **st})


@app.get("/local/providers", include_in_schema=False)
async def local_providers():
    """Provider availability for the UI status bar (cached model lists)."""
    out = {}
    for key, prov in PROVIDERS.items():
        served = await _provider_models(key)
        out[key] = {
            "label": prov["label"],
            "reachable": served is not None,
            "models": sorted(served) if served is not None else [],
        }
    return JSONResponse(out)


@app.get("/v1/session", include_in_schema=False)
async def get_session(request: Request):
    """Get-or-create the chat session. Returns the cookie (Set-Cookie on new),
    the stored transcript, and the last model. The UI calls this on load."""
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        sid = uuid.uuid4().hex
        data = {"id": sid, "messages": [], "model": "auto", "new": True}
        resp = JSONResponse(data)
        resp.set_cookie(
            SESSION_COOKIE, sid, max_age=SESSION_TTL_DAYS * 86400,
            httponly=True, samesite="lax", path="/",
        )
        return resp
    loaded = _load_session(sid)
    messages, model = (loaded if loaded else ([], "auto"))
    return JSONResponse({"id": sid, "messages": messages, "model": model, "new": False})


@app.post("/v1/session/clear", include_in_schema=False)
async def clear_session(request: Request):
    """Wipe the current session's transcript (cookie stays valid)."""
    sid = _session_id_from(request)
    if sid:
        try:
            with _sqlite_lock, _db() as conn:
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
        except Exception:
            pass
    return JSONResponse({"ok": True})

_models_cache: dict[str, tuple[float, set[str] | None]] = {}


def _headers(provider: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"
    return headers


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


async def _resolve(model: str | None) -> list[tuple[str, str]]:
    """Return ordered (provider_key, upstream_model) chain for a model name."""
    if not model or not str(model).strip():
        raise _ApiError(400, "model is required")
    m = str(model).strip()
    low = m.lower()

    if low == "auto":
        # cloud providers first (AGNO order), then the local GPU as the
        # FINAL fallback. local serves LOCAL_MODEL, not AUTO_MODEL.
        chain = [(k, AUTO_MODEL) for k in AGNO_ORDER if k != "local"]
        if not chain:
            chain = [("ocg", AUTO_MODEL), ("dpsk", AUTO_MODEL)]
        if "local" in PROVIDERS:
            chain.append(("local", LOCAL_MODEL))
        return chain

    for key in ("ocg", "dpsk", "agno", "local"):
        prefix = key + "-"
        if low.startswith(prefix):
            upstream = m[len(prefix):].strip()
            if not upstream:
                raise _ApiError(400, f"model '{m}': missing upstream model after '{key}-'")
            if key in ("ocg", "dpsk", "local"):
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
        f"unknown model '{m}'. Use a provider prefix (OCG-/DPSK-/LOCAL-/AGNO-) or 'auto'.",
    )


async def _upstream(method: str, path: str, body: dict, key: str, stream: bool = False):
    """Fire one request at one provider. Returns (client, response).

    stream=True: SSE relayed INCREMENTALLY (the chat/completions streaming
    path) — caller consumes via aiter_bytes(); body is NOT pre-read.
    stream=False (default): httpx reads eagerly, so r.json()/r.content work
    directly. v1.7.2 set stream=True unconditionally, which 500'd every
    non-streaming request (httpx.ResponseNotRead on r.json()/r.content) —
    fixed in v1.7.3 by making it explicit per call site.
    """
    prov = PROVIDERS[key]
    url = f"{prov['base_url']}{path}"
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=600, write=60, pool=10)
    )
    try:
        req = client.build_request(method, url, headers=_headers(prov), json=body)
        r = await client.send(req, stream=stream)
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
        "live_ingest": {"enabled": LIVE_INGEST_ENABLED, "queued": _live_queue.qsize()},
    }


@app.get("/stats")
async def stats(limit: int = 20):
    """Per-request stats over the rolling window + totals since process start.
    RAG overhead is visible as rag bytes injected (rag_bytes) against prompt
    tokens (usage.prompt). limit clamps the recent-request sample."""
    win = list(_stats_window)
    n = len(win)
    rag_recs = [r for r in win if (r.get("rag") or {}).get("injected")]
    with_rag = len(rag_recs)

    def _pct(arr, p):
        if not arr:
            return 0
        s = sorted(arr)
        return s[min(len(s) - 1, int(len(s) * p))]

    agg = {
        "window_requests": n,
        "window_with_rag": with_rag,
        "rag_inject_rate_pct": round(100 * with_rag / n, 1) if n else 0,
        "rag_bytes": {
            "total": sum((r.get("rag") or {}).get("bytes", 0) for r in rag_recs),
            "avg": round(sum((r.get("rag") or {}).get("bytes", 0) for r in rag_recs) / with_rag, 1) if with_rag else 0,
            "p50": _pct([(r.get("rag") or {}).get("bytes", 0) for r in rag_recs], 0.5),
            "p95": _pct([(r.get("rag") or {}).get("bytes", 0) for r in rag_recs], 0.95),
        },
        "rag_tokens_est": {
            "avg": round(sum((r.get("rag") or {}).get("bytes", 0) for r in rag_recs) / 4 / with_rag, 1) if with_rag else 0,
        },
        "prompt_tokens": {
            "avg": round(sum((r.get("usage") or {}).get("prompt", 0) for r in win) / n, 1) if n else 0,
            "p95": _pct([(r.get("usage") or {}).get("prompt", 0) for r in win], 0.95),
        },
        "completion_tokens": {"avg": round(sum((r.get("usage") or {}).get("completion", 0) for r in win) / n, 1) if n else 0},
        "reasoning_tokens": {"avg": round(sum((r.get("usage") or {}).get("reasoning", 0) for r in win) / n, 1) if n else 0},
        "latency_ms": {
            "avg": round(sum(r.get("latency_ms", 0) for r in win) / n, 1) if n else 0,
            "p95": _pct([r.get("latency_ms", 0) for r in win], 0.95),
        },
        "errors": sum(1 for r in win if r.get("status", 0) >= 400),
        "by_consumer": {},
    }
    for r in win:
        c = r.get("consumer") or "-"
        b = agg["by_consumer"].setdefault(c, {"requests": 0, "with_rag": 0, "rag_bytes": 0, "prompt_tokens": 0})
        b["requests"] += 1
        if (r.get("rag") or {}).get("injected"):
            b["with_rag"] += 1
            b["rag_bytes"] += (r.get("rag") or {}).get("bytes", 0)
        b["prompt_tokens"] += (r.get("usage") or {}).get("prompt", 0)

    t = dict(_stats_since_start)
    if t["requests"]:
        t["avg_latency_ms"] = round(t["latency_ms"] / t["requests"], 1)
        t["rag_bytes_per_request"] = round(t["rag_bytes"] / t["requests"], 1)
    else:
        t["avg_latency_ms"] = 0
        t["rag_bytes_per_request"] = 0
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "window": STATS_WINDOW,
        "aggregates_window": agg,
        "totals_since_start": t,
        "telemetry": {"url": TELEMETRY_URL, "flush_seconds": STATS_FLUSH_SECONDS},
        "recent": list(win)[-max(1, min(limit, 100)):],
    }


# ---------------- live session ingestion (deferred queue) ----------------

def _tee_live_turn(consumer: str | None, user_msg: str, reply: str):
    """Enqueue one turn pair for background embedding. Fail-safe: never raises,
    never blocks the response path. consumer None = nothing to attribute → skip."""
    if not LIVE_INGEST_ENABLED or not consumer or not user_msg.strip():
        return
    try:
        _live_queue.put_nowait({
            "consumer": consumer,
            "user_msg": user_msg.strip(),
            "reply": reply.strip() if reply else "",
            "ts": int(time.time()),
        })
    except Exception:
        pass


async def _live_flusher():
    """Background task: drain the queue in batches to embeddings /ingest/sessions."""
    while True:
        try:
            await asyncio.sleep(LIVE_FLUSH_SECONDS)
            batch: list[dict] = []
            while not _live_queue.empty() and len(batch) < LIVE_FLUSH_BATCH:
                try:
                    batch.append(_live_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if not batch:
                continue
            # group by consumer, one POST per consumer
            by_consumer: dict[str, list[dict]] = {}
            for item in batch:
                by_consumer.setdefault(item["consumer"], []).append(item)
            for consumer, items in by_consumer.items():
                chunks = []
                for item in items:
                    text = f"user: {item['user_msg']}"
                    if item["reply"]:
                        text += f"\nassistant: {item['reply']}"
                    chunks.append({
                        "id": f"live:{consumer}:{item['ts']}",
                        "title": f"live {consumer} {time.strftime('%Y-%m-%d %H:%M', time.localtime(item['ts']))}",
                        "text": text,
                        "meta": {"consumer": consumer, "live": True, "ts": item["ts"]},
                    })
                try:
                    async with httpx.AsyncClient(timeout=10) as c:
                        r = await c.post(
                            f"{RAG_BASE_URL}/ingest/sessions",
                            json={"consumer": consumer, "chunks": chunks},
                        )
                        r.raise_for_status()
                except Exception:
                    # fail-open: drop the batch — NFS snapshot catchup covers it
                    pass
        except Exception:
            await asyncio.sleep(LIVE_FLUSH_SECONDS)


@app.on_event("startup")
async def _start_live_flusher():
    global _live_flusher_task
    if LIVE_INGEST_ENABLED and _live_flusher_task is None:
        _live_flusher_task = asyncio.create_task(_live_flusher())
    if TELEMETRY_URL:
        asyncio.create_task(_stats_flusher())


@app.on_event("shutdown")
async def _stop_live_flusher():
    global _live_flusher_task
    if _live_flusher_task:
        _live_flusher_task.cancel()
        _live_flusher_task = None


@app.get("/v1/models")
async def list_models():
    ids = ["auto"]
    for key, prov in PROVIDERS.items():
        served = await _provider_models(key)
        if served is None:
            served = {AUTO_MODEL}  # can't verify -> advertise the default only
        for mid in sorted(served):
            ids.append(f"{prov['label']}-{mid}")
    # AGNO-<model> only when >=2 providers actually serve it — a single
    # provider serving a model already has its own prefix (OCG-/DPSK-/LOCAL-);
    # advertising AGNO- for it would be redundant AND misleading (it implies a
    # choice of backends where none exists).
    union: set[str] = set()
    for key in PROVIDERS:
        served = await _provider_models(key)
        if served:
            union |= served
    if len(PROVIDERS) >= 2:
        for mid in sorted(union):
            count = 0
            for key in PROVIDERS:
                served = await _provider_models(key)
                if served and mid in served:
                    count += 1
            if count >= 2:
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


def _rag_request_headers(request: Request) -> tuple[str | None, bool, bool]:
    """Extract the RAG control plane from request headers.

    Returns (consumer, docs_enabled, optin). Absent headers = legacy defaults:
    consumer None (all sessions), docs on, opt-in on. Header value "0"/"false"
    disables; anything else enables.
    """
    hdr = request.headers
    consumer = (hdr.get("x-consumer") or "").strip() or None
    docs_raw = (hdr.get("x-rag-docs") or "").strip().lower()
    optin_raw = (hdr.get("x-rag-optin") or "").strip().lower()
    docs_enabled = docs_raw not in ("0", "false", "no")
    optin = optin_raw not in ("0", "false", "no")
    return consumer, docs_enabled, optin


async def _rag_retrieve(query: str, consumer: str | None, docs_enabled: bool) -> list[dict]:
    """Ask the embeddings service for top-k semantic hits. Fail-open: [] on any error.

    Balanced retrieval: when docs are enabled, query docs and sessions SEPARATELY
    (top_k//2 each) and interleave. Session narratives routinely outrank docs on
    operational queries (observed: 0.63 session vs 0.31 docs for infra-doctor) —
    a single pooled top-k starves the docs answer. Docs and sessions both get a seat.
    """
    if not RAG_ENABLED:
        return []

    async def _search(payload: dict) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.post(f"{RAG_BASE_URL}/search", json=payload)
                r.raise_for_status()
                return r.json().get("results", [])
        except Exception:
            return []

    per_source_k = max(1, (RAG_TOP_K + 1) // 2)  # ceil: TOP_K=3 → 2+2, capped to 3 at the end
    merged: list[dict] = []

    # sessions bucket (scoped to consumer when present)
    sess_payload: dict = {"query": query, "top_k": per_source_k, "min_score": RAG_MIN_SCORE, "sources": ["session"]}
    if consumer:
        sess_payload["consumers"] = [consumer]
    merged.extend(await _search(sess_payload))

    # docs bucket (shared, always included when enabled)
    if docs_enabled:
        merged.extend(await _search({"query": query, "top_k": per_source_k, "min_score": RAG_MIN_SCORE, "sources": ["docs"]}))

    # interleave: session, docs, session, docs... (stable by source bucket order)
    sessions = [h for h in merged if h.get("source") == "session"]
    docs = [h for h in merged if h.get("source") == "docs"]
    out: list[dict] = []
    for i in range(max(len(sessions), len(docs))):
        if i < len(sessions):
            out.append(sessions[i])
        if i < len(docs):
            out.append(docs[i])
    return out[:RAG_TOP_K]


def _rag_system_message(query: str, hits: list[dict]) -> str:
    """Format retrieved chunks into a compact system-prompt context block."""
    parts = []
    for h in hits[:RAG_TOP_K]:
        src = h.get("source", "?")
        consumer = h.get("consumer") or ""
        title = h.get("title", "")
        text = (h.get("text") or "")[:RAG_MAX_CHARS]
        score = h.get("score", 0)
        who = f" [{consumer}]" if consumer else ""
        parts.append(f"--- [{src}{who}] {title} (score {score})\n{text}")
    return (
        "The following is authoritative context from the homelab operator's docs and "
        "session history (retrieved via RAG). It is FACT, not a hint — prefer it over "
        "guessing or tool checks when it answers the user's question:\n\n"
        + "\n\n".join(parts)
    )


def _last_user_message(messages: list) -> str:
    """Find the most recent user turn to query RAG against."""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
            return m["content"].strip()
    return ""


def _sse_reply_text(raw: bytes) -> str:
    """Reconstruct assistant content from OpenAI SSE stream bytes."""
    parts = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload in ("[DONE]", ""):
            continue
        try:
            obj = json.loads(payload)
            delta = ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content")
            if isinstance(delta, str):
                parts.append(delta)
        except Exception:
            continue
    return "".join(parts)


async def _inject_rag(messages: list, consumer: str | None, docs_enabled: bool) -> list:
    """Return messages with a RAG context system message injected (fail-open: unchanged)."""
    query = _last_user_message(messages)
    if not query:
        return messages
    hits = await _rag_retrieve(query, consumer, docs_enabled)
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
    routes = await _resolve(model)
    want_stream = bool(body.get("stream"))

    # RAG: header-driven injection (fail-open; absent headers = legacy defaults)
    rag_meta = {"injected": False, "chunks": 0, "bytes": 0}
    if RAG_ENABLED and body.get("rag") is not False:
        consumer, docs_enabled, optin = _rag_request_headers(request)
        if optin:
            messages = body.get("messages") or []
            if isinstance(messages, list):
                messages = await _inject_rag(messages, consumer, docs_enabled)
                rag_meta = _rag_meta(messages)
                body = {**body, "messages": messages}

    if not want_stream:
        last_err = None
        for key, upstream in routes:
            t0 = time.time()
            payload = dict(body)
            payload["model"] = upstream
            _ensure_reasoning_echo(payload.get("messages"), _is_thinking_model(upstream))
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
            # live session ingest: tee the turn (deferred queue) on a 2xx
            if r.status_code < 400:
                consumer, _, _ = _rag_request_headers(request)
                _tee_live_turn(
                    consumer,
                    _last_user_message(body.get("messages") or []),
                    ((content or {}).get("choices") or [{}])[0].get("message", {}).get("content") or "",
                )
                # remember reasoning_content so follow-up tool loops echo it
                if _is_thinking_model(upstream):
                    _cache_reasoning_echo(content)
                # persist the conversation server-side (v1.7.0)
                sid = _session_id_from(request)
                if sid:
                    try:
                        msgs = body.get("messages") or []
                        reply = ((content or {}).get("choices") or [{}])[0].get("message", {}).get("content") or ""
                        if reply.strip():
                            msgs = [*msgs, {"role": "assistant", "content": reply}]
                        _save_session(sid, msgs, str(body.get("model", "")))
                    except Exception:
                        pass
            await client.aclose()
            # record stats (non-streaming; usage comes from the JSON body)
            u = (content or {}).get("usage") or {}
            _record_stat({
                "ts": int(time.time()),
                "consumer": _rag_request_headers(request)[0],
                "endpoint": "chat",
                "stream": False,
                "status": r.status_code,
                "latency_ms": int((time.time() - t0) * 1000),
                "rag": rag_meta,
                "usage": {
                    "prompt": u.get("prompt_tokens", 0),
                    "completion": u.get("completion_tokens", 0),
                    "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                },
            })
            if content is not None:
                return JSONResponse(status_code=r.status_code, content=content)
            return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
        raise _ApiError(502, f"all providers failed for model '{model}': {last_err or 'unknown'}")

    # streaming: pick the first provider that responds 2xx, stream its SSE
    last_err = None
    for key, upstream in routes:
        payload = dict(body)
        payload["model"] = upstream
        _ensure_reasoning_echo(payload.get("messages"), _is_thinking_model(upstream))
        try:
            client, r = await _upstream("POST", "/chat/completions", payload, key, stream=True)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            await client.aclose()
            last_err = f"{PROVIDERS[key]['label']} unreachable: {type(e).__name__}"
            continue
        if r.status_code >= 400:
            last_err = f"{PROVIDERS[key]['label']} HTTP {r.status_code}"
            # still report the body of the first failure if it's the only route
            if len(routes) == 1:
                try:
                    await r.aread()
                except Exception:
                    pass
                await client.aclose()
                return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
            await client.aclose()
            continue

        async def stream(client=client, r=r):
            chunks = []
            t0 = time.time()
            sid = _session_id_from(request)
            try:
                async for chunk in r.aiter_bytes():
                    chunks.append(chunk)
                    yield chunk
            finally:
                # live session ingest: reconstruct the assistant reply from SSE deltas
                if r.status_code < 400:
                    try:
                        consumer, _, _ = _rag_request_headers(request)
                        reply = _sse_reply_text(b"".join(chunks))
                        _tee_live_turn(
                            consumer,
                            _last_user_message(body.get("messages") or []),
                            reply,
                        )
                    except Exception:
                        pass
                    # remember reasoning_content from the stream for follow-up loops
                    if _is_thinking_model(upstream):
                        _cache_reasoning_echo_from_sse(b"".join(chunks))
                # persist the conversation server-side (v1.7.0): whatever was
                # produced — including a partial reply if the client disconnected
                # mid-stream — is stored so the UI can restore it on return.
                if sid:
                    try:
                        msgs = body.get("messages") or []
                        reply = _sse_reply_text(b"".join(chunks))
                        if reply.strip():
                            msgs = [*msgs, {"role": "assistant", "content": reply}]
                        _save_session(sid, msgs, str(body.get("model", "")))
                    except Exception:
                        pass
                # record stats (streaming; usage from the final SSE chunk when present)
                try:
                    _record_stat({
                        "ts": int(time.time()),
                        "consumer": _rag_request_headers(request)[0],
                        "endpoint": "chat",
                        "stream": True,
                        "status": r.status_code,
                        "latency_ms": int((time.time() - t0) * 1000),
                        "rag": rag_meta,
                        "usage": _sse_usage(b"".join(chunks)),
                    })
                except Exception:
                    pass
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
            msg: dict = {
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
            }
            # DeepSeek thinking-mode requires reasoning_content echoed back on
            # assistant tool_call messages; preserve it through the round-trip.
            if item.get("reasoning_content"):
                msg["reasoning_content"] = item["reasoning_content"]
            messages.append(msg)
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
        fc_item = {
            "id": f"fc_{uuid.uuid4().hex}",
            "type": "function_call",
            "status": "completed",
            "call_id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
        }
        if msg.get("reasoning_content"):
            fc_item["reasoning_content"] = msg["reasoning_content"]
        output.append(fc_item)
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
    # RAG: header-driven injection (fail-open; absent headers = legacy defaults)
    rag_meta = {"injected": False, "chunks": 0, "bytes": 0}
    if RAG_ENABLED and body.get("rag") is not False:
        consumer, docs_enabled, optin = _rag_request_headers(request)
        if optin:
            chat_payload["messages"] = await _inject_rag(chat_payload["messages"], consumer, docs_enabled)
            rag_meta = _rag_meta(chat_payload["messages"])
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

    routes = await _resolve(model)
    last_err = None
    for key, upstream in routes:
        t0 = time.time()
        payload = dict(chat_payload)
        payload["model"] = upstream
        _ensure_reasoning_echo(payload.get("messages"), _is_thinking_model(upstream))
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
        # live session ingest: tee the turn on a 2xx
        if r.status_code < 400:
            consumer, _, _ = _rag_request_headers(request)
            _tee_live_turn(
                consumer,
                _last_user_message(chat_payload["messages"]),
                ((chat.get("choices") or [{}])[0].get("message", {}).get("content") or ""),
            )
            # remember reasoning_content so follow-up tool loops echo it
            if _is_thinking_model(upstream):
                _cache_reasoning_echo(chat)
        await client.aclose()
        # record stats (non-streaming; usage comes from the chat JSON body)
        u = chat.get("usage") or {}
        _record_stat({
            "ts": int(time.time()),
            "consumer": _rag_request_headers(request)[0],
            "endpoint": "responses",
            "stream": False,
            "status": r.status_code,
            "latency_ms": int((time.time() - t0) * 1000),
            "rag": rag_meta,
            "usage": {
                "prompt": u.get("prompt_tokens", 0),
                "completion": u.get("completion_tokens", 0),
                "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            },
        })
        if r.status_code >= 400:
            return JSONResponse(status_code=r.status_code, content=chat)
        return JSONResponse(content=_chat_to_response(chat, str(model)))
    raise _ApiError(502, f"all providers failed for model '{model}': {last_err or 'unknown'}")
