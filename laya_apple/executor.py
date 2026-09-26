"""Heterogeneous execution: one FIFO queue per device.

Placement per device, chosen from measurements (research/v0.2-concurrency/):

- GPU (MLX): a worker process. Two backends in one interpreter measurably interfere
  (the GPU stream lost ~11% with the ANE in a sibling thread).
- ANE (Core ML): an in-process thread or a worker process, chosen per model from
  measurement (laya_apple/data/placement.json).
  - Thread: with the GPU busy, a device fed over cross-process IPC runs its host-side
    work 4-6x slower (a cache-resident probe slows from ~0.3 to 1.2-1.7 ms). An in-process
    ANE is not slowed this way.
  - Process: Core ML's Python predict holds the GIL for much of an ANE call, which costs
    the caller's interpreter more at high short-request rates (mmBERT-base, ~250 req/s).

Process workers are started as `python -m laya_apple.executor` subprocesses, not
multiprocessing children, so the caller's main module is never re-imported (works from a
REPL or notebook, no `if __name__ == "__main__"` requirement). They connect back over an
authenticated AF_UNIX multiprocessing connection. The parent sends (job_id, rows), where
job_id is the request's id; the worker runs backend.forward(rows) and replies
(job_id, "ok", (logits, act, start_ns, end_ns)) or (job_id, "error", exception). start_ns and
end_ns bracket the forward in the worker (time.monotonic_ns, a system-wide clock on macOS).

Jobs on one device run in FIFO order, one at a time, which is how both devices behave
anyway: the ANE serialises work, and concurrent MLX streams share one GPU.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import Future
from multiprocessing.connection import Client, Listener

from .errors import BackendUnavailableError, LayaAppleError
from .trace import QueueSnapshot

PLACEMENTS = ("process", "thread")


def load_backend(kind: str, args: dict):
    from pathlib import Path

    from .registry import resolve

    spec = resolve(args["model"])
    ckpt = Path(args["checkpoint"])
    if kind == "gpu":
        from .backends.mlx import MLXBackend

        return MLXBackend(spec, ckpt, args["pad_id"], dtype=args["dtype"], batch_size=args["batch_size"])
    from .backends.coreml_ane import ANEBackend

    return ANEBackend(
        spec,
        ckpt,
        args["pad_id"],
        args["local_attention"],
        args["buckets"],
        strict=args["strict"],
        async_models=bool(args.get("async_models", False)),  # ane_handoff=True only
    )


def warm(kind: str, backend, pad_id: int) -> None:
    """Run each compiled shape once so the first real request does not pay for it.

    ANE: one prediction per loaded bucket. MLX: one small forward (kernel compilation;
    MLX specialises some kernels per shape, so later new lengths may still pay a little).
    The outputs are discarded.
    """
    if kind == "ane":
        for b in backend.buckets:  # a row of exactly b tokens selects bucket b
            backend.forward([{"ids": [pad_id] * b, "markers": [1, 2], "qtype": 0}])
    else:
        backend.forward([{"ids": [pad_id] * 8, "markers": [1, 2], "qtype": 0}])


def backend_info(kind: str, backend) -> dict:
    info = {"name": backend.name, "device": backend.device}
    if kind == "ane":
        info["offered"] = list(backend.offered)
        info["artifact_sha256"] = dict(backend.artifact_sha256)
        info["load_errors"] = {b: _portable(e) for b, e in backend.load_errors.items()}
        info["probes"] = {b: dict(v) for b, v in backend.probes.items()}
    return info


def _forward_timed(backend, rows):
    start_ns = time.monotonic_ns()
    logits, act = backend.forward(rows)
    return logits, act, start_ns, time.monotonic_ns()


def _worker_main(kind: str, args: dict, conn) -> None:
    import warnings

    # laya-apple's own load-time warnings are re-issued by the parent from the worker's info;
    # anything else (Core ML, coremltools, NumPy) stays visible on the worker's stderr.
    warnings.filterwarnings("ignore", message="laya-apple:", category=RuntimeWarning)
    try:
        backend = load_backend(kind, args)
        warm(kind, backend, args["pad_id"])
    except BaseException as e:  # report the load failure with its own type
        conn.send(("load_error", _portable(e)))
        return
    conn.send(("ready", backend_info(kind, backend)))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        job_id, rows = msg
        try:
            conn.send((job_id, "ok", _forward_timed(backend, rows)))
        except BaseException as e:
            conn.send((job_id, "error", _portable(e)))


def _portable(e: BaseException) -> BaseException:
    """An exception that survives pickling: LayaAppleError subclasses as-is, others wrapped."""
    if isinstance(e, LayaAppleError):
        return e
    return LayaAppleError(f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}")


class DeviceWorker:
    """One device: a FIFO job queue, a dispatcher thread, and the backend it feeds.

    placement="process": the backend lives in a worker process (IPC per job).
    placement="thread": the backend lives in this process and runs on the dispatcher thread.

    `snapshot()` reads the queue state in one step: the backlog (the sum of service
    estimates of queued jobs plus the remaining estimate of the running one, the queue-state
    input of scheduling.decide_queued), the number of queued jobs, and whether one is running.

    A job's Future resolves to (logits, act, queue_enter_ns, dispatch_ns, service_start_ns,
    service_end_ns), all time.monotonic_ns().

    `activity` (optional, set by the owner before the first submit; ane_handoff=True only): an
    object with started() and ended(), such as laya_apple.handoff.GpuActivity. submit() calls
    started() for every job and ended() exactly once when that job's Future is done, whatever
    the outcome: a result, an error, a cancellation, a dead worker, or a job failed by close().
    """

    activity = None

    def __init__(
        self, kind: str, args: dict, *, placement: str = "process", start_timeout: float = 600.0, wait: bool = True
    ):
        """Start loading. With wait=False, call `wait_ready()` before use, so several devices
        load in parallel."""
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}")
        self.kind, self.placement = kind, placement
        self._start_timeout = start_timeout
        self._args = args
        self.info = None
        self._backend = None
        self._closed = False
        self._proc = self._listener = self._conn = None
        if placement == "process":
            key = os.urandom(32)
            self._listener = Listener(family="AF_UNIX", authkey=key)
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "laya_apple.executor"], stdin=subprocess.PIPE, text=True
            )
            self._proc.stdin.write(
                json.dumps({"kind": kind, "args": args, "address": self._listener.address, "key": key.hex()})
            )
            self._proc.stdin.close()
        else:
            self._loaded: dict = {}
            self._loader = threading.Thread(target=self._load_here, name=f"laya-{kind}-load", daemon=True)
            self._loader.start()
        if wait:
            self.wait_ready()

    def _load_here(self):
        try:
            backend = load_backend(self.kind, self._args)
            warm(self.kind, backend, self._args["pad_id"])
            self._loaded["backend"] = backend
        except BaseException as e:
            self._loaded["error"] = e

    def wait_ready(self):
        """Block until the backend has loaded and warmed, or raise its load error."""
        if self.info is not None:
            return
        if self.placement == "thread":
            self._loader.join()
            if "error" in self._loaded:
                raise self._loaded["error"]
            if self._closed:  # closed while loading (background start-up): release, do not serve
                self._loaded.clear()
                raise BackendUnavailableError(f"{self.kind} worker closed during start-up")
            self._backend = self._loaded["backend"]
            info = backend_info(self.kind, self._backend)
        else:
            info = self._connect()
        # Everything the router and submit() touch exists before `info` marks the worker ready.
        self._jobs: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._queued_ms = 0.0
        self._queued_jobs = 0
        self._running = False
        self._running_est = 0.0
        self._running_since = 0
        self._dead: BaseException | None = None
        self._thread = threading.Thread(target=self._dispatch, name=f"laya-{self.kind}-dispatch", daemon=True)
        self._thread.start()
        self.info = info

    def _connect(self):
        kind, start_timeout = self.kind, self._start_timeout
        accepted: dict = {}

        def accept():
            try:
                accepted["conn"] = self._listener.accept()
            except BaseException as e:  # listener closed on timeout
                accepted["error"] = e

        try:
            t = threading.Thread(target=accept, daemon=True)
            t.start()
            deadline = time.monotonic() + start_timeout
            while t.is_alive() and time.monotonic() < deadline and self._proc.poll() is None:
                t.join(0.05)
            if "conn" not in accepted:
                self._proc.kill()
                self._proc.wait()
                raise BackendUnavailableError(
                    f"{kind} worker did not connect (exit code {self._proc.returncode}, timeout {start_timeout:.0f} s)"
                )
            self._conn = accepted["conn"]
        finally:
            self._listener.close()
        if not self._conn.poll(start_timeout):
            self._proc.kill()
            raise BackendUnavailableError(f"{kind} worker did not finish loading within {start_timeout:.0f} s")
        status, payload = self._conn.recv()
        if status == "load_error":
            self._proc.wait(timeout=10)
            raise payload
        return payload

    @property
    def ready(self) -> bool:
        return self.info is not None

    @property
    def alive(self) -> bool:
        if getattr(self, "_dead", None) is not None:
            return False
        if self.placement == "process" and self.info is not None and self._proc.poll() is not None:
            # exited before the dispatcher noticed: report it now, not after a request fails on it
            self._dead = BackendUnavailableError(f"{self.kind} worker exited with code {self._proc.returncode}")
            return False
        return True

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def read_queue(self) -> tuple | None:
        """(backlog_ms, queued_jobs, running) read in one step under the lock, or None before
        the worker is ready. A plain tuple: the router's hot path builds no object."""
        if self.info is None:
            return None
        with self._lock:
            backlog = self._queued_ms
            if self._running_est:
                backlog += max(0.0, self._running_est - (time.monotonic_ns() - self._running_since) / 1e6)
            return backlog, self._queued_jobs, self._running

    def snapshot(self) -> QueueSnapshot | None:
        """The queue state as one consistent reading (read_queue), or None before ready."""
        state = self.read_queue()
        return QueueSnapshot._make(state) if state is not None else None

    def backlog_ms(self) -> float:
        snap = self.snapshot()
        return snap.backlog_ms if snap is not None else 0.0

    def submit(self, rows, estimate_ms: float, job_id: int) -> Future:
        """Queue one job; `job_id` (the request's id) names it on the worker protocol."""
        fut: Future = Future()
        activity = self.activity
        # With the handoff (activity attached), a dead worker's job never counts as GPU activity, so
        # it cannot start a handoff episode; alive also notices a process that exited unreported.
        # Without it (the default), submit is unchanged.
        if activity is not None and self.alive:
            activity.started()
            fut.add_done_callback(lambda _f: activity.ended())
        if self._dead is not None:
            fut.set_exception(BackendUnavailableError(f"{self.kind} worker is not running: {self._dead}"))
            return fut
        with self._lock:
            self._queued_ms += estimate_ms
            self._queued_jobs += 1
        self._jobs.put((fut, rows, estimate_ms, time.monotonic_ns(), job_id))
        return fut

    def _run(self, job_id, rows):
        if self._backend is not None:
            return _forward_timed(self._backend, rows)
        self._conn.send((job_id, rows))
        rid, status, payload = self._conn.recv()
        if rid != job_id:
            raise LayaAppleError(f"{self.kind} worker protocol error: reply {rid} for job {job_id}")
        if status == "error":
            raise payload
        return payload

    def _dispatch(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            fut, rows, est, enq, job_id = job
            dispatch_ns = time.monotonic_ns()
            with self._lock:
                self._queued_ms -= est
                self._queued_jobs -= 1
                self._running, self._running_est, self._running_since = True, est, dispatch_ns
            if not fut.set_running_or_notify_cancel():
                with self._lock:
                    self._running, self._running_est = False, 0.0
                continue
            try:
                if self._dead is not None:
                    raise BackendUnavailableError(f"{self.kind} worker is not running: {self._dead}")
                logits, act, start_ns, end_ns = self._run(job_id, rows)
                fut.set_result((logits, act, enq, dispatch_ns, start_ns, end_ns))
            except (EOFError, OSError, BrokenPipeError) as e:
                if self.placement != "process":  # an in-process backend error is just an error
                    fut.set_exception(e)
                    continue
                self._dead = e
                fut.set_exception(BackendUnavailableError(f"{self.kind} worker exited: {e!r}"))
            except BaseException as e:
                fut.set_exception(e)
            finally:
                with self._lock:
                    self._running, self._running_est = False, 0.0

    def close(self, timeout: float = 10.0):
        """Finish queued jobs (up to `timeout`), then stop. Jobs still queued after that fail
        with BackendUnavailableError; none is left pending."""
        self._closed = True
        if self.info is None:  # never became ready: nothing queued
            if self.placement == "process":
                self._listener.close()
                if self._proc.poll() is None:
                    self._proc.kill()
                self._proc.wait(timeout=timeout)
            return
        self._jobs.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            self._dead = self._dead or BackendUnavailableError("closed")
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                break
            if job is not None and job[0].set_running_or_notify_cancel():
                job[0].set_exception(BackendUnavailableError(f"{self.kind} worker closed before this request ran"))
        if self.placement == "thread":
            self._backend = None
            return
        try:
            self._conn.send(None)
        except (OSError, BrokenPipeError):
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=timeout)
        self._conn.close()


def _main() -> None:
    cfg = json.loads(sys.stdin.read())
    conn = Client(cfg["address"], family="AF_UNIX", authkey=bytes.fromhex(cfg["key"]))
    try:
        _worker_main(cfg["kind"], cfg["args"], conn)
    finally:
        conn.close()


if __name__ == "__main__":
    _main()
