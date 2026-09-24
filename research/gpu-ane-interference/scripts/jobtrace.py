"""Research-only backend phase hooks, keyed by the runtime's request id.

The request lifecycle (submit, prepare, routing, queue, dispatch, service, response) and the
queue snapshots the router used come from the runtime itself: `Laya(..., trace=callback)`
emits one `laya_apple.RequestTrace` per request (laya_apple/trace.py). This module adds only
what a product runtime should not carry: a split of each backend forward into device spans
(`mx.eval` for MLX, Core ML `predict` for the ANE) and host work (everything else), and the
executing thread's CPU time. ledger.py joins the two on `request_id`.

Where the hooks run:
- A thread-placed backend runs in this process on the device's dispatcher thread.
  `install_phase_hooks()` patches the backend classes here, and `tag_dispatcher(worker)`
  wraps that DeviceWorker's `_run(job_id, rows)` so the forward knows its job id.
- A process-placed backend runs in a worker started as `python -m laya_apple.executor`.
  `hooks/sitecustomize.py` (on PYTHONPATH with LAYA_TRACE_DIR set) installs the same class
  hooks there and wraps the worker's connection so each received job's id is known while
  its forward runs. Records are written to LAYA_TRACE_DIR/phases-<pid>.json at exit.

The job id on the worker protocol is the request id (laya_apple/executor.py). Neither hook
changes an argument, a result, a queue or a routing decision; each only reads the clock and
the id around the call it wraps.

Every timestamp is `time.monotonic_ns()`, the clock the runtime trace uses. On macOS it is
one system-wide clock (mach time), so worker and parent timestamps share one axis
(`clock_info()` records the implementation).
"""

from __future__ import annotations

import atexit
import json
import multiprocessing.connection
import os
import threading
import time
from pathlib import Path

now = time.monotonic_ns

# ----------------------------------------------------------------------------- clock


def clock_info() -> dict:
    info = time.get_clock_info("monotonic")
    return {"implementation": info.implementation, "monotonic": info.monotonic, "resolution_s": info.resolution}


# ----------------------------------------------------------------------------- job id

_CURRENT = threading.local()  # the request id whose forward runs on this thread, or None


def current_request_id():
    return getattr(_CURRENT, "request_id", None)


def tag_dispatcher(worker) -> None:
    """A thread-placed DeviceWorker: expose each job's id to the forward it runs (instance-level)."""
    orig = worker._run

    def _run(job_id, rows):
        _CURRENT.request_id = job_id
        try:
            return orig(job_id, rows)
        finally:
            _CURRENT.request_id = None

    worker._run = _run


class _TaggingConnection:
    """A worker's connection: a received job (job_id, rows) sets the current request id."""

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def recv(self):
        msg = self._conn.recv()
        if isinstance(msg, tuple) and len(msg) == 2 and isinstance(msg[0], int):
            _CURRENT.request_id = msg[0]
        return msg


def _tag_worker_connection() -> None:
    """Wrap multiprocessing.connection.Client before `python -m laya_apple.executor` imports it."""
    client = multiprocessing.connection.Client

    def tagging_client(*args, **kwargs):
        return _TaggingConnection(client(*args, **kwargs))

    multiprocessing.connection.Client = tagging_client


# ----------------------------------------------------------------------------- backend side

_PHASES: list = []
_LOCK = threading.Lock()


class _TimedMx:
    """Stands in for the `mlx.core` module inside one MLXBackend.forward; times mx.eval."""

    def __init__(self, mx, spans):
        self._mx, self._spans = mx, spans

    def __getattr__(self, name):
        return getattr(self._mx, name)

    def eval(self, *args):
        t = now()
        self._mx.eval(*args)
        self._spans.append((t, now()))


class _TimedModel:
    """Stands in for one Core ML model inside ANEBackend.forward; times predict."""

    def __init__(self, model, spans):
        self._model, self._spans = model, spans

    def __getattr__(self, name):
        return getattr(self._model, name)

    def predict(self, feats):
        t = now()
        out = self._model.predict(feats)
        self._spans.append((t, now()))
        return out


def _record(kind, items, t0, t1, c0, c1, spans):
    rec = {
        "request_id": current_request_id(),  # None: a forward outside a job (warm-up, references)
        "device": kind,
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "rows": len(items),
        "max_len": max(len(it["ids"]) for it in items),
        "t0": t0,
        "t1": t1,
        "cpu_ns": c1 - c0,
        "device_spans": spans,
    }
    with _LOCK:
        _PHASES.append(rec)


def install_phase_hooks() -> None:
    """Wrap MLXBackend.forward and ANEBackend.forward in this process (idempotent)."""
    from laya_apple.backends import coreml_ane, mlx

    if getattr(mlx.MLXBackend.forward, "_laya_trace", False):
        return
    mlx_forward, ane_forward = mlx.MLXBackend.forward, coreml_ane.ANEBackend.forward

    def gpu_forward(self, items):
        spans: list = []
        real = self.mx
        self.mx = _TimedMx(real, spans)
        c0, t0 = time.thread_time_ns(), now()
        try:
            return mlx_forward(self, items)
        finally:
            t1, c1 = now(), time.thread_time_ns()
            self.mx = real
            _record("gpu", items, t0, t1, c0, c1, spans)

    def ane_fwd(self, items):
        spans: list = []
        real = self.models
        self.models = {b: _TimedModel(m, spans) for b, m in real.items()}
        c0, t0 = time.thread_time_ns(), now()
        try:
            return ane_forward(self, items)
        finally:
            t1, c1 = now(), time.thread_time_ns()
            self.models = real
            _record("ane", items, t0, t1, c0, c1, spans)

    gpu_forward._laya_trace = ane_fwd._laya_trace = True
    mlx.MLXBackend.forward, coreml_ane.ANEBackend.forward = gpu_forward, ane_fwd


def phases() -> list:
    with _LOCK:
        return list(_PHASES)


def clear_phases() -> None:
    with _LOCK:
        _PHASES.clear()


def dump_phases(path: Path) -> None:
    Path(path).write_text(json.dumps(phases()))


def install_in_worker() -> None:
    """Called from hooks/sitecustomize.py: hooks, job-id tagging, and a dump at exit."""
    out = os.environ.get("LAYA_TRACE_DIR")
    if not out:
        return
    _tag_worker_connection()
    install_phase_hooks()
    atexit.register(lambda: dump_phases(Path(out) / f"phases-{os.getpid()}.json"))
