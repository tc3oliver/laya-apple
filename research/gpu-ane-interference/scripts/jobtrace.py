"""Per-request timing for laya-apple's heterogeneous executor, without changing its behaviour.

Two layers. Neither changes a routing decision, a queue, or a backend's output: each wrapper
only reads the clock around the call it wraps and returns that call's result unchanged.

1. Parent side (`instrument_worker`): instance-level wrappers on one `DeviceWorker`'s
   `submit` and `_run`, recording per job

       queue_enter   the job entered the device's FIFO queue (DeviceWorker.submit)
       device_start  the dispatcher took it and started the device call (_run entry)
       device_end    the device call returned to the dispatcher (_run exit)

   `_run` is the whole device occupancy as the FIFO sees it: for a process-placed device
   it includes the IPC round trip, for a thread-placed device it is the backend call.

2. Backend side (`install_phase_hooks`): wrappers on `MLXBackend.forward` and
   `ANEBackend.forward` that split each forward into device spans (`mx.eval` for MLX, Core
   ML `predict` for the ANE) and host work (everything else), and record the executing
   thread's CPU time. In a thread-placed backend they are installed in this process; in a
   worker process they are installed by `hooks/sitecustomize.py`, which Python imports at
   start-up when `hooks/` is on PYTHONPATH and LAYA_TRACE_DIR is set. Records are written
   to LAYA_TRACE_DIR/phases-<pid>.json at exit.

Every timestamp is `time.perf_counter()`. On macOS it is `mach_absolute_time()`, one
system-wide monotonic clock, so parent and worker timestamps are directly comparable
(checked by `clock_info()`).
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path

perf = time.perf_counter

# ----------------------------------------------------------------------------- clock


def clock_info() -> dict:
    info = time.get_clock_info("perf_counter")
    return {"implementation": info.implementation, "monotonic": info.monotonic, "resolution_s": info.resolution}


# ----------------------------------------------------------------------------- parent side


class JobTable:
    """Per-request records keyed by the identity of the request's rows list.

    The rows list is created once per request (Laya.prepare) and passed unchanged to
    DeviceWorker.submit and on to _run, so id(rows) names one job for its whole life."""

    def __init__(self):
        self._lock = threading.Lock()
        self._live: dict[int, dict] = {}
        self._tl = threading.local()

    def open(self, rows, **fields) -> dict:
        rec = dict(fields)
        with self._lock:
            self._live[id(rows)] = rec
        rec["_key"] = id(rows)
        self._tl.rec = rec
        return rec

    def expect_new(self) -> None:
        """The next submit on this thread is a new request (Laya.submit path): open a fresh
        record even if a finished request's rows, not yet collected, had the same id."""
        self._tl.new = True

    def take_new(self) -> bool:
        new = getattr(self._tl, "new", False)
        self._tl.new = False
        return new

    def last_opened(self) -> dict | None:
        """The record most recently opened on this thread. Laya.submit creates the rows and
        calls DeviceWorker.submit on the caller's thread, so after Laya.submit returns this is
        that request's record."""
        return getattr(self._tl, "rec", None)

    def get(self, rows) -> dict | None:
        return self._live.get(id(rows))

    def close(self, rec: dict) -> dict:
        """Forget a finished request (its rows may be garbage-collected and the id reused)."""
        with self._lock:
            self._live.pop(rec.pop("_key"), None)
        return rec


def instrument_worker(worker, jobs: JobTable) -> None:
    """Record queue_enter / device_start / device_end on one DeviceWorker (instance-level)."""
    orig_submit, orig_run, backlog = worker.submit, worker._run, worker.backlog_ms

    def submit(rows, estimate_ms):
        rec = None if jobs.take_new() else jobs.get(rows)
        if rec is None:  # a request submitted through Laya.submit: its rows are new here
            rec = jobs.open(rows)
        rec["device"] = worker.kind
        rec["backlog_at_enter_ms"] = backlog()
        rec["estimate_ms"] = estimate_ms
        rec["queue_enter"] = perf()
        return orig_submit(rows, estimate_ms)

    def _run(rows):
        rec = jobs.get(rows)
        t0 = perf()
        try:
            return orig_run(rows)
        finally:
            if rec is not None:
                rec["device_start"], rec["device_end"] = t0, perf()

    worker.submit, worker._run = submit, _run


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
        t = perf()
        self._mx.eval(*args)
        self._spans.append((t, perf()))


class _TimedModel:
    """Stands in for one Core ML model inside ANEBackend.forward; times predict."""

    def __init__(self, model, spans):
        self._model, self._spans = model, spans

    def __getattr__(self, name):
        return getattr(self._model, name)

    def predict(self, feats):
        t = perf()
        out = self._model.predict(feats)
        self._spans.append((t, perf()))
        return out


def _record(kind, items, t0, t1, c0, c1, spans):
    rec = {
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "kind": kind,
        "rows": len(items),
        "max_len": max(len(it["ids"]) for it in items),
        "t0": t0,
        "t1": t1,
        "cpu_ms": (c1 - c0) * 1e3,
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
        c0, t0 = time.thread_time(), perf()
        try:
            return mlx_forward(self, items)
        finally:
            t1, c1 = perf(), time.thread_time()
            self.mx = real
            _record("gpu", items, t0, t1, c0, c1, spans)

    def ane_fwd(self, items):
        spans: list = []
        real = self.models
        self.models = {b: _TimedModel(m, spans) for b, m in real.items()}
        c0, t0 = time.thread_time(), perf()
        try:
            return ane_forward(self, items)
        finally:
            t1, c1 = perf(), time.thread_time()
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
    """Called from hooks/sitecustomize.py: hooks plus a dump at interpreter exit."""
    out = os.environ.get("LAYA_TRACE_DIR")
    if not out:
        return
    install_phase_hooks()
    atexit.register(lambda: dump_phases(Path(out) / f"phases-{os.getpid()}.json"))
