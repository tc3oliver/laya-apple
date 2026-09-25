"""PB-ASYNC: #83's prebound Core ML predict, sent through the asynchronous prediction API (research only).

Everything #83's PrebindModel builds at load is reused unchanged, by subclassing it: the input
MLMultiArrays and their NumPy views, the Foundation-only feature provider, the output
MLMultiArrays set as MLPredictionOptions.outputBackings and their NumPy views, the strong
references, and the per-bucket non-blocking lock. Only the predict call differs:

  PB-SYNC (#83)                                    PB-ASYNC (this file)
  write inputs into the views (NumPy)              the same
  push an autorelease pool                         the same
  predictionFromFeatures:options:error: through    predictionFromFeatures:options:completionHandler:
    predict_options_stamped.m, ctypes.CDLL           as a plain PyObjC send (PyObjC releases the GIL
    (GIL released for the whole predict)             around the send as for any send; it returns once
                                                     Core ML has queued the work)
  -                                                wait on the bucket's threading.Event (GIL released
                                                     while waiting), timeout TIMEOUT_S -> raise
  pop the pool                                     the same
  copy the outputs out of the backed views         the same

The completion handler is one Python callable per bucket, created at load and kept referenced
(Completion.handler). Core ML calls it on one of its own threads; PyObjC takes the GIL for it.
Per call it stamps time.monotonic_ns() on entry, records the native thread id, the dispatch
queue label and whether an error came back, counts the callback, and sets the event. It does not
read the output provider: the outputs are in the backings. Exactly one callback per submit is
enforced: a callback with no submit pending (a duplicate, or one after a timeout) is recorded as
an anomaly, and a bucket that timed out refuses further submits.

Backings are verified at load through the async path, with #83's sentinel method
(verify_backings). An output that is not backed ("read") is read from the provider the handler
then keeps; its mode is in `self.modes`.

STAMPS gets one (submit_before, submit_after, callback_entry, wake) per predict, all
time.monotonic_ns(): before the send, after the send returned, the handler's first line (on the
callback thread), and the waiter resuming. CALLBACKS gets (callback_entry, native thread id,
error) per callback, LABELS counts the queue labels, ANOMALIES lists late or duplicate
callbacks. Appended from the ANE thread and the callback threads.

Research only: needs pyobjc-framework-CoreML (`uv run --with pyobjc-framework-CoreML==12.2.2`);
the async API needs macOS 14 or later.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PB83 = HERE.parents[1] / "coreml-prebind-predict" / "scripts"
sys.path.insert(0, str(PB83))

import prebind  # noqa: E402  #83's, unchanged
from prebind import SENTINEL, _address, _cached_view  # noqa: E402

TIMEOUT_S = 5.0
CALLBACK_THREAD_NAME = "coreml-callback"

STAMPS: list[tuple[int, int, int, int]] = []
CALLBACKS: list[tuple[int, int, int]] = []
LABELS: Counter = Counter()
ANOMALIES: list[dict] = []

_libsystem = None


def queue_label() -> str:
    """dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL) of the calling thread."""
    global _libsystem
    if _libsystem is None:
        lib = ctypes.PyDLL("/usr/lib/libSystem.B.dylib")  # PyDLL: a tiny call, the GIL is kept
        lib.dispatch_queue_get_label.restype = ctypes.c_char_p
        lib.dispatch_queue_get_label.argtypes = [ctypes.c_void_p]
        _libsystem = lib
    label = _libsystem.dispatch_queue_get_label(None)
    return label.decode(errors="replace") if label else ""


class Completion:
    """One bucket's completion: the handler Core ML calls, and the event its submitter waits on.

    States: idle -> pending (arm) -> idle (handler) ; pending -> timed_out (wait timed out, final).
    Pure Python: unit-tested with a fake model."""

    def __init__(self, stamps: list | None = None, callbacks: list | None = None, labels=None, anomalies=None):
        self.event = threading.Event()
        self._lock = threading.Lock()
        self.state = "idle"
        self.submits = 0
        self.callbacks = 0
        self.keep_output = False  # load-time backing check, or a "read" output
        self.output = None
        self.error = None
        self.entry_ns = 0
        self._stamps = STAMPS if stamps is None else stamps
        self._cb = CALLBACKS if callbacks is None else callbacks
        self._labels = LABELS if labels is None else labels
        self._anomalies = ANOMALIES if anomalies is None else anomalies
        self.handler = self._handler  # one bound method, created once, kept referenced

    def arm(self) -> None:
        with self._lock:
            if self.state == "timed_out":
                raise RuntimeError("async predict timed out earlier on this bucket: it takes no further submits")
            if self.state != "idle":
                raise RuntimeError("async predict submitted before the previous one completed")
            self.state = "pending"
            self.submits += 1
            self.output = self.error = None
            self.event.clear()

    def _handler(self, output, error) -> None:
        t = time.monotonic_ns()
        th = threading.current_thread()  # registers Core ML's thread (a _DummyThread) for the snapshots
        if th.name != CALLBACK_THREAD_NAME:
            th.name = CALLBACK_THREAD_NAME
        tid = threading.get_native_id()
        self._labels[queue_label()] += 1
        failed = error is not None
        with self._lock:
            self.callbacks += 1
            self._cb.append((t, tid, int(failed)))
            if self.state != "pending":
                self._anomalies.append({"kind": "late" if self.state == "timed_out" else "duplicate", "t_ns": t})
                return
            self.state = "idle"
            self.entry_ns = t
            self.error = error
            if self.keep_output:
                self.output = output
        self.event.set()

    def wait(self, timeout: float = TIMEOUT_S) -> int:
        """Block (GIL released) until the handler ran; return the wake stamp."""
        if not self.event.wait(timeout):
            with self._lock:
                if self.state == "pending":
                    self.state = "timed_out"
                    raise TimeoutError(f"async predict did not complete within {timeout} s")
        return time.monotonic_ns()

    def submit(self, send, timeout: float = TIMEOUT_S) -> tuple[int, int, int, int]:
        """arm, send(handler), wait; returns and records (submit_before, submit_after,
        callback_entry, wake). Raises on a Core ML error."""
        self.arm()
        sb = time.monotonic_ns()
        send(self.handler)
        sa = time.monotonic_ns()
        wake = self.wait(timeout)
        s = (sb, sa, self.entry_ns, wake)
        self._stamps.append(s)
        if self.error is not None:
            raise RuntimeError(f"Core ML async predict failed: {self.error}")
        return s


class AsyncPrebindModel(prebind.PrebindModel):
    """#83's PrebindModel with only the predict call replaced (and the load-time backing check
    made through the same call). #83's __init__ still compiles and holds its shim (self._call);
    this class never calls it."""

    def _completion(self) -> Completion:
        c = self.__dict__.get("_async")
        if c is None:
            c = self._async = Completion()
        return c

    def _send(self, handler) -> None:
        self._model.predictionFromFeatures_options_completionHandler_(self._provider, self._options, handler)

    def verify_backings(self) -> dict:
        """#83's sentinel check, through the async call: 'backed' if the backing was filled with
        exactly the output the handler received, else 'read'."""
        import objc

        c = self._completion()
        for view in self._in_views.values():
            view[...] = 0
        for view in self._out_views.values():
            view[...] = np.frombuffer(bytes([SENTINEL]) * view.dtype.itemsize, view.dtype)[0]
        evidence = {}
        c.keep_output = True
        try:
            with objc.autorelease_pool():
                c.submit(self._send)
                out = c.output
                for name, backing in self._out_views.items():
                    arr = out.featureValueForName_(name).multiArrayValue()
                    returned = prebind.nogil_predict._view(arr, prebind.nogil_predict._types()[arr.dataType()])
                    written = not np.all(np.ascontiguousarray(backing).view(np.uint8) == SENTINEL)
                    equal = returned.dtype == backing.dtype and np.array_equal(
                        np.ascontiguousarray(returned).view(np.uint8), np.ascontiguousarray(backing).view(np.uint8)
                    )
                    evidence[name] = {
                        "same_address": _address(returned) == _address(backing),
                        "backing_written": bool(written),
                        "equal_to_returned": bool(equal),
                        "dtype": str(backing.dtype),
                        "shape": list(backing.shape),
                    }
                    self.modes[name] = "backed" if written and equal else "read"
                c.output = None
        finally:
            c.keep_output = any(m != "backed" for m in self.modes.values())
        self.backing_evidence = evidence
        return evidence

    def predict(self, feats: dict) -> dict:
        import objc

        if not self._busy.acquire(blocking=False):
            raise RuntimeError("prebound bucket already in flight: its buffers would be overwritten")
        try:
            for name, view in self._in_views.items():  # #83's conversion
                view[...] = np.asarray(feats[name], dtype=view.dtype).reshape(view.shape)
            c = self._completion()
            result = {}
            with objc.autorelease_pool():
                c.submit(self._send)  # the one send, then the wait
                if c.keep_output:
                    for name, mode in self.modes.items():
                        if mode != "backed":
                            dtype, shape, strides = self._out_meta[name]
                            arr = c.output.featureValueForName_(name).multiArrayValue()
                            v = _cached_view(arr, dtype, shape, strides)
                            result[name] = v.astype(np.float32) if v.dtype == np.float16 else np.array(v)
                    c.output = None
            for name, mode in self.modes.items():
                if mode == "backed":
                    v = self._out_views[name]
                    result[name] = v.astype(np.float32) if v.dtype == np.float16 else np.array(v)
            return result
        finally:
            self._busy.release()


_stamper = None
_stamper_lock = threading.Lock()


def _stamper_lib():
    """Compile async_stamper.m once per process; return (the LayaAsyncStamper class, the stamp
    reader). Phase 0 only."""
    global _stamper
    with _stamper_lock:
        if _stamper is None:
            import objc

            build = Path(tempfile.mkdtemp(prefix="laya-async-stamper-"))
            try:
                out = build / "libasync_stamper.dylib"
                subprocess.run(
                    ["xcrun", "clang", "-dynamiclib", "-O2", "-fobjc-arc", "-fblocks", "-framework", "Foundation"]
                    + ["-framework", "CoreML", str(HERE / "async_stamper.m"), "-o", str(out)],
                    check=True,
                    capture_output=True,
                )
                lib = ctypes.CDLL(str(out))  # registers the class
            finally:
                shutil.rmtree(build, ignore_errors=True)  # the loaded image stays mapped
            lib.laya_async_native_stamp.restype = ctypes.c_uint64
            lib.laya_async_native_stamp.argtypes = []
            objc.registerMetaDataForSelector(
                b"LayaAsyncStamper",
                b"predict:features:options:handler:",
                {
                    "arguments": {
                        5: {
                            "callable": {
                                "retval": {"type": b"v"},
                                "arguments": {0: {"type": b"^v"}, 1: {"type": b"@"}, 2: {"type": b"@"}},
                            }
                        }
                    }
                },
            )
            _stamper = (objc.lookUpClass("LayaAsyncStamper"), lib.laya_async_native_stamp)
    return _stamper


class StampedAsyncPrebindModel(AsyncPrebindModel):
    """PB-ASYNC-STAMPED (Phase 0 only): the same send made from a C block that stamps the native
    completion first. NATIVE gets (native_completion, callback_entry) per predict. A semantics
    check of the completion path, not performance evidence; never used in the screen."""

    NATIVE: list[tuple[int, int]] = []

    def _send(self, handler) -> None:
        cls, _ = _stamper_lib()
        cls.predict_features_options_handler_(self._model, self._provider, self._options, handler)

    def predict(self, feats: dict) -> dict:
        out = super().predict(feats)
        self.NATIVE.append((int(_stamper_lib()[1]()), self._completion().entry_ns))
        return out


def install(backend) -> None:
    """Swap every loaded bucket of one ANEBackend instance to the async prebound path."""
    from laya_apple.artifacts import COMPILED, artifact_dir

    backend.models = {b: AsyncPrebindModel(artifact_dir(backend.spec, b) / COMPILED) for b in backend.models}
