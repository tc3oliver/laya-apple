"""`laya-apple serve`: a local decision server with a Jev-compatible API.

    laya-apple serve                       # http://127.0.0.1:8642, model "auto"
    TYPESAFE_BASE_URL=http://127.0.0.1:8642 <your Jev client>

Which checkpoint answers a request, when the request names none (Jev clients send a Jev id
such as "jev-latest"): with `--model auto`, the default, the one upstream's Router picks by
the state's language: `laya` (English) for English text, `laya-multilingual` otherwise
(laya_apple/lang.py, vendored from upstream). `--model NAME` pins one checkpoint instead,
for example `--model typed-decisions` for the workflows it was fine-tuned on. A request
that names a checkpoint always gets that checkpoint.

The wire format follows upstream `laya.serve` (NandhaKishorM/laya), which exposes Laya
over TypeSafe Jev's `POST /v1/systemone` protocol: a request `{model?, state, questions}`
answers `{model, answers, usage, routing}`, so a client written against Jev keeps working
when its base URL points here. What differs from upstream, on purpose:

- It binds 127.0.0.1 and answers only requests whose `Host` is a loopback name, so a web
  page in a local browser cannot reach it through DNS rebinding. Any other bind address
  needs `--allow-remote` and `LAYA_API_KEY`. `POST /v1/systemone` needs
  `Content-Type: application/json`, which a browser cannot send cross-site without a
  preflight this server never grants.
- There is no global inference lock. Each model is one `Laya(execution="workers")`
  instance and requests go through `apredict`, so the GPU and the ANE serve concurrently.
- `ane_startup="background"`: the server answers on MLX as soon as MLX is ready while the
  ANE artifacts load; `/health` reports the ANE as `warming`, `ready` or `unavailable`.
- Every response carries a `laya_apple` block (checkpoint, device, routing reason,
  request id, `truncated`). A state cut to fit the model's max length is reported there;
  upstream cuts it too, without saying so.

Configuration is by flag, with environment fallbacks for the service manager:
`LAYA_APPLE_SERVE_MODEL`, `LAYA_APPLE_SERVE_HOST`, `LAYA_APPLE_SERVE_PORT`, and
`LAYA_API_KEY` (upstream's name).
With `LAYA_API_KEY` set, requests need `Authorization: Bearer <key>`, compared in constant
time. Without it any key is accepted, because Jev clients always send one.

fastapi and uvicorn come from the `[serve]` extra and are imported only here.
"""

# No `from __future__ import annotations`: FastAPI must see the real `Request` class on the
# route handlers below, which import it locally.
import asyncio
import hmac
import ipaddress
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Callable

from .errors import InvalidRequestError, LayaAppleError, UnsupportedModelError, UnsupportedShapeError
from .registry import models, resolve

_log = logging.getLogger("laya_apple.serve")

DEFAULT_HOST = "127.0.0.1"
# Not 8000: upstream laya.serve and common local LLM servers (oMLX, vLLM) default to it.
DEFAULT_PORT = 8642
AUTO = "auto"
DEFAULT_MODEL = AUTO
# What upstream's Router falls back to when the language is undecided or there is no text.
AUTO_FALLBACK = "laya"

# Upstream's guardrails (laya/serve.py), so a client sees the same limits and status codes.
MAX_QUESTIONS = 64
MAX_STATE_CHARS = 50000
MAX_BODY_BYTES = 2 * 1024 * 1024

# Upstream's checkpoint names and aliases (laya.router, v0.3.20). Any other `model` value,
# such as a Jev model id ("jev-latest"), means "the server's default", as upstream.
UPSTREAM_NAMES = {"english": "laya", "multilingual": "laya-multilingual", "typed-decisions": "laya-typed-decisions"}
_ALIASES = {
    "en": "english",
    "laya": "english",
    "default": "english",
    "multi": "multilingual",
    "ml": "multilingual",
    "laya-multilingual": "multilingual",
    "typed": "typed-decisions",
    "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions",
    "decisions": "typed-decisions",
}
_UPSTREAM_KEY = {v: k for k, v in UPSTREAM_NAMES.items()}
# Upstream reads the root repo id as "let the router choose", not "pin English".
_ROUTER_CHOICE = {"convaiinnovations/laya"}
# Upstream's typed-decisions workflows (laya/router.py v0.3.20), matched on the exact set of
# question ids. Reported in `routing.workflow`; like upstream with LAYA_AUTO_TASK off, a
# match does not change the checkpoint.
TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
    "security_incidents": {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

Loader = Callable[[str], Any]


def _upstream_name(name: str) -> str | None:
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    return UPSTREAM_NAMES.get(key)


def checkpoint(name: str) -> str:
    """A laya-apple checkpoint name from ours, an HF id or an upstream name; else raise."""
    return resolve(_upstream_name(name) or name.strip()).name


def match_workflow(questions: Any) -> str | None:
    ids = set(questions or {})
    return next((wf for wf, sig in TYPED_DECISION_WORKFLOWS.items() if ids == sig), None)


def resolve_model(requested: Any, default: str) -> str:
    """Map a client's `model` field onto a checkpoint name, or `default` (which may be AUTO).
    `model: "auto"` asks for language routing whatever the server default is."""
    if not isinstance(requested, str) or not requested.strip():
        return default
    name = requested.strip()
    if name.lower() == AUTO:
        return AUTO
    if name.lower() in _ROUTER_CHOICE:
        return default
    if (upstream := _upstream_name(name)) is not None:
        return upstream
    try:
        return resolve(name).name
    except UnsupportedModelError:
        return default


def route_by_language(state: Any) -> tuple[str, str, dict]:
    """(checkpoint, reason, detection): upstream Router's language branch, verbatim in effect
    (laya/router.py v0.3.20, `default="english"`, no lang hints)."""
    from .lang import analyse

    det = analyse(state)
    if det["script"] == "unknown":
        name, reason = AUTO_FALLBACK, "no letters detected in state; using default (english)"
    elif det["script"] != "latin":
        name = "laya-multilingual"
        reason = "non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it" % (
            det["script"],
            100 * float(det["non_latin_fraction"]),
        )
    elif not det["is_english"]:
        name = "laya-multilingual"
        if det["language"]:
            reason = "Latin script but language looks like %r, not English" % det["language"]
        else:
            reason = (
                "Latin script, language not identified but %.0f%% non-English letters; "
                "not safe for the English checkpoint" % (100 * float(det["diacritic_rate"]))
            )
    elif det["language_undecided"]:
        name = AUTO_FALLBACK
        reason = "Latin script, language not identified and no non-English letters; using default (english)"
    else:
        name, reason = "laya", "English Latin text"
    return name, reason, det


def host_name(header: str) -> str:
    """The host part of a Host header: "[::1]:8642" -> "::1", "localhost:8642" -> "localhost"."""
    header = header.strip().lower()
    if header.startswith("["):
        return header[1:].split("]", 1)[0]
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_limits(state: Any, questions: Any) -> None:
    """Upstream's pre-tokenization checks: 400 for a non-object, 413 for oversized input."""
    from fastapi import HTTPException

    if not isinstance(questions, dict):
        raise HTTPException(status_code=400, detail="'questions' must be an object")
    if len(questions) > MAX_QUESTIONS:
        raise HTTPException(status_code=413, detail="too many questions (%d > %d)" % (len(questions), MAX_QUESTIONS))
    try:
        state_len = len(state) if isinstance(state, str) else len(str(state))
    except Exception:
        state_len = MAX_STATE_CHARS + 1
    if state_len > MAX_STATE_CHARS:
        raise HTTPException(status_code=413, detail="state too large (%d > %d chars)" % (state_len, MAX_STATE_CHARS))


async def read_body_capped(request: Any) -> bytes:
    """Stream the body and stop at MAX_BODY_BYTES, whatever Content-Length claims."""
    from fastapi import HTTPException

    total, chunks = 0, []
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")
        chunks.append(chunk)
    return b"".join(chunks)


def ane_status(laya: Any) -> dict:
    """The ANE path's state for /health: warming, ready or unavailable (with the reason)."""
    from . import routing

    state = getattr(laya, "ane_state", None)
    reason = getattr(state, "unavailable", None)
    if reason == routing.ANE_STARTING:
        return {"status": "warming"}
    if reason is None and getattr(laya, "ane", None) is not None and state is not None and state.buckets:
        return {"status": "ready", "buckets": list(state.buckets)}
    return {"status": "unavailable", "reason": reason or "no_ane_path"}


def default_loader(*, device: str = "auto", offline: bool = False) -> Loader:
    def load(name: str):
        from . import Laya

        return Laya.from_pretrained(
            name,
            device=device,
            execution="workers",
            ane_startup="background" if device == "auto" else "wait",
            local_files_only=offline,
        )

    return load


def create_app(
    *,
    default_model: str = DEFAULT_MODEL,
    preload: tuple[str, ...] = (),
    loader: Loader | None = None,
    api_key: str | None = None,
    device: str = "auto",
    loopback_only: bool = True,
):
    """Build the FastAPI app. `loader(name)` returns a Laya-like object (tests inject one).

    `default_model` is a checkpoint or AUTO (language routing between laya and
    laya-multilingual). The default's checkpoints and every `preload` model load at startup;
    another supported model loads the first time a request names it.

    loopback_only: refuse requests whose Host header is not a loopback name (the bind is
    loopback; this stops DNS rebinding from a browser)."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    auto = default_model.strip().lower() == AUTO
    default_model = AUTO if auto else checkpoint(default_model)
    startup = ["laya", "laya-multilingual"] if auto else [default_model]
    names = list(dict.fromkeys([*startup, *(checkpoint(m) for m in preload)]))
    loader = loader or default_loader()
    expected_auth = ("Bearer " + api_key).encode("utf-8", "surrogateescape") if api_key else b""
    instances: dict[str, Any] = {}
    loads: dict[str, asyncio.Task] = {}

    async def load(name: str) -> Any:
        """One load per model, shared by concurrent requests. The load runs as its own task,
        so a request cancelled mid-load neither aborts it nor leaks the instance: it is
        stored when it finishes, and closed at shutdown like every other."""
        if name in instances:
            return instances[name]
        task = loads.get(name)
        if task is None or (task.done() and task.exception() is not None):

            async def run_load():
                _log.info("loading %s", name)
                instances[name] = await asyncio.to_thread(loader, name)
                return instances[name]

            task = loads[name] = asyncio.create_task(run_load())
        return await asyncio.shield(task)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            for name in names:
                await load(name)
            yield
        finally:
            pending = [t for t in loads.values() if not t.done()]
            if pending:  # a thread cannot be cancelled: wait for it, then close what it built
                await asyncio.gather(*pending, return_exceptions=True)
            for laya in instances.values():
                try:
                    laya.close()
                except Exception:
                    _log.exception("closing a model failed")

    from . import __version__

    app = FastAPI(
        title="laya-apple serve",
        summary="Local Laya decisions over a Jev-compatible /v1/systemone API",
        version=__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def loopback_host_only(request: Request, call_next):
        if loopback_only and host_name(request.headers.get("host", "")) not in LOOPBACK_HOSTS:
            return JSONResponse(status_code=421, content={"detail": "this server answers loopback requests only"})
        return await call_next(request)

    @app.exception_handler(Exception)
    async def unexpected(_request: Request, exc: Exception):
        _log.error("request failed", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "inference failed"})

    def check_auth(request: Request) -> None:
        if not api_key:
            return
        supplied = (request.headers.get("authorization") or "").encode("utf-8", "surrogateescape")
        if not hmac.compare_digest(supplied, expected_auth):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    def health_body() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "default_model": default_model,  # a checkpoint, or "auto" (language routing)
            "loaded": sorted(instances),
            "device": device,
            "ane": {name: ane_status(laya) for name, laya in sorted(instances.items())},
        }

    @app.get("/health")
    def health() -> dict:
        return health_body()

    @app.get("/healthz")
    def healthz() -> dict:
        return health_body()

    @app.get("/v1/models")
    def list_models(request: Request) -> dict:
        """Jev's shape, `{"models": [{name, description, release_date}]}`; the SDKs read it
        strictly, so the extra fields are only additions."""
        check_auth(request)
        entries = [
            {
                "name": AUTO,
                "description": "Routes by the state's language: laya for English, laya-multilingual otherwise",
                "release_date": "",
                "loaded": auto,
                "default": auto,
            }
        ]
        for spec in models().values():
            entries.append(
                {
                    "name": spec.name,
                    "description": f"{spec.repo}@{spec.revision[:12]} ({spec.encoder}, {spec.max_len} tokens)",
                    "release_date": "",
                    "loaded": spec.name in instances,
                    "default": spec.name == default_model,
                }
            )
        return {"models": entries}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        check_auth(request)
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            raise HTTPException(status_code=415, detail="Content-Type must be application/json")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")
        raw = await read_body_capped(request)
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):
            raise HTTPException(status_code=400, detail="request body must be valid JSON")
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(status_code=400, detail="request body must be an object with a 'questions' field")
        state, questions = body.get("state"), body["questions"]
        check_limits(state, questions)
        requested = body.get("model")
        name = resolve_model(requested, default_model)
        workflow = match_workflow(questions)
        if checkpoint_named(requested):  # upstream names the normalised checkpoint key
            routing = routing_block(name, reason="explicit model=%r" % _UPSTREAM_KEY[name])
        elif name == AUTO:
            name, reason, detection = route_by_language(state)
            routing = routing_block(name, reason=reason, detection=detection, workflow=workflow)
        else:
            routing = routing_block(name, reason="server default model (--model %s)" % name, workflow=workflow)
        if not questions:  # upstream answers an empty question set with nothing, not an error
            return JSONResponse(
                content={
                    "model": "laya-rl-agent",
                    "answers": {},
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "routing": routing,
                }
            )
        # Upstream serializes a missing state as JSON null ("null"); Laya reads None as "".
        state = "null" if state is None else state
        try:
            laya = await load(name)
        except Exception:
            _log.exception("loading %s failed", name)
            raise HTTPException(status_code=503, detail=f"model {name} is unavailable on this server")
        try:
            t0 = time.perf_counter()
            # submit() tokenizes before it queues the device job; keep that off the event loop.
            future = await asyncio.to_thread(laya.submit, state, questions)
            result = await asyncio.wrap_future(future)
            infer_ms = (time.perf_counter() - t0) * 1000.0
        except (InvalidRequestError, UnsupportedShapeError) as e:
            raise HTTPException(status_code=422, detail=str(e))
        except ValueError as e:  # upstream: question validation errors are safe to return
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:  # never leak paths, weights or memory state to the client
            _log.exception("inference failed for model=%s", name)
            raise HTTPException(status_code=500, detail="inference failed")
        return JSONResponse(
            content=response_body(result, routing),
            headers={"Server-Timing": f"inference;dur={infer_ms:.2f}", "X-Inference-Time-Ms": f"{infer_ms:.2f}"},
        )

    return app


def checkpoint_named(requested: Any) -> bool:
    """Whether the client's `model` names a checkpoint (as opposed to a Jev id or nothing)."""
    if not isinstance(requested, str) or not requested.strip() or requested.strip().lower() in _ROUTER_CHOICE:
        return False
    try:
        checkpoint(requested)
    except UnsupportedModelError:
        return False
    return True


def routing_block(name: str, *, reason: str, detection: dict | None = None, workflow: str | None = None) -> dict:
    """Upstream's `routing` keys: {model, repo, reason, detection, workflow}. `repo` names the
    standalone repository the pinned weights come from; stock upstream names its bundle
    repository (convaiinnovations/laya/<subfolder>) instead."""
    spec = resolve(name)
    return {
        "model": _UPSTREAM_KEY[spec.name],
        "repo": spec.repo,
        "reason": reason,
        "detection": detection,
        "workflow": workflow,
    }


def response_body(result: Any, routing: dict) -> dict:
    """Upstream's `{model, answers, usage, routing}` plus the `laya_apple` block."""
    rt = result.runtime
    return {
        "model": result.model,
        "answers": result.answers,
        "usage": result.usage,
        "routing": routing,
        "laya_apple": {
            "model": rt.model,
            "model_revision": rt.model_revision,
            "device": rt.device,
            "backend": rt.backend,
            "routing_reason": rt.routing_reason,
            "request_id": rt.request_id,
            "truncated": rt.truncated,
            "sequence_length": rt.sequence_length,
            "question_count": rt.question_count,
            "latency_ms": round(rt.latency_ms, 3),
            "queue_wait_ms": rt.queue_wait_ms,
            "device_ms": rt.device_ms,
        },
    }


def run(
    *,
    host: str | None = None,
    port: int | None = None,
    model: str | None = None,
    preload: tuple[str, ...] = (),
    device: str = "auto",
    allow_remote: bool = False,
    offline: bool = False,
    log_level: str = "info",
) -> int:
    try:
        import uvicorn
    except ImportError:
        raise LayaAppleError("laya-apple serve needs the [serve] extra: pip install 'laya-apple[serve]'") from None

    host = host or os.environ.get("LAYA_APPLE_SERVE_HOST") or DEFAULT_HOST
    raw_port = port if port is not None else os.environ.get("LAYA_APPLE_SERVE_PORT", DEFAULT_PORT)
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        raise LayaAppleError(f"invalid port {raw_port!r}: must be an integer 1-65535") from None
    if not 1 <= port <= 65535:
        raise LayaAppleError(f"invalid port {port}: must be an integer 1-65535")
    api_key = os.environ.get("LAYA_API_KEY") or None
    loopback = is_loopback(host)
    if not loopback:
        if not allow_remote:
            raise LayaAppleError(
                f"refusing to bind {host!r}: laya-apple serve is loopback-only by default. "
                "Pass --allow-remote to expose it, and set LAYA_API_KEY."
            )
        if not api_key:
            raise LayaAppleError(f"refusing to bind {host!r} without LAYA_API_KEY: set a key before exposing it")
        _log.warning("laya-apple serve is listening on %s, which is not a loopback address", host)
        print(f"warning: listening on {host}, reachable from other machines", flush=True)
    model = model or os.environ.get("LAYA_APPLE_SERVE_MODEL") or DEFAULT_MODEL
    app = create_app(
        default_model=model,
        preload=preload,
        loader=default_loader(device=device, offline=offline),
        api_key=api_key,
        device=device,
        loopback_only=loopback,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level)
    return 0
