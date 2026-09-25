"""Public pthread QoS calls for the dependency QoS screen (research only; ../criteria.md).

ctypes bindings to libSystem's pthread/qos.h (macOS 10.10+):
  pthread_self() -> pthread_t
  pthread_threadid_np(pthread_t, uint64_t *) -> int          (the kernel thread id, for alignment)
  pthread_override_qos_class_start_np(pthread_t, qos_class_t, int) -> pthread_override_t (NULL on failure)
  pthread_override_qos_class_end_np(pthread_override_t) -> int
  pthread_set_qos_class_self_np(qos_class_t, int) -> int
  pthread_get_qos_class_np(pthread_t, qos_class_t *, int *) -> int

The calls go through ctypes.PyDLL, as recount.py's do: the GIL is kept across each call, so an
override start or end on the client thread hands the GIL to no one.

OverrideRegistry keeps every override object of one run: each start returns a token (or None when
the start returned NULL, counted as a failure), each end is matched to its token once (a second end
of the same token is recorded as an end error and never reaches libSystem, so no object is freed
twice), and end_all() ends whatever is still outstanding at the end of the run.
"""

from __future__ import annotations

import ctypes
import threading
import time

QOS_CLASS_USER_INTERACTIVE = 0x21
QOS_CLASS_USER_INITIATED = 0x19
QOS_CLASS_DEFAULT = 0x15
QOS_CLASS_UTILITY = 0x11
QOS_CLASS_BACKGROUND = 0x09
QOS_CLASS_UNSPECIFIED = 0x00

_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.PyDLL("/usr/lib/libSystem.B.dylib")
        L.pthread_self.argtypes = []
        L.pthread_self.restype = ctypes.c_void_p
        L.pthread_threadid_np.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        L.pthread_threadid_np.restype = ctypes.c_int
        L.pthread_override_qos_class_start_np.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int]
        L.pthread_override_qos_class_start_np.restype = ctypes.c_void_p
        L.pthread_override_qos_class_end_np.argtypes = [ctypes.c_void_p]
        L.pthread_override_qos_class_end_np.restype = ctypes.c_int
        L.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
        L.pthread_set_qos_class_self_np.restype = ctypes.c_int
        L.pthread_get_qos_class_np.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_int),
        ]
        L.pthread_get_qos_class_np.restype = ctypes.c_int
        _lib = L
    return _lib


def pthread_self() -> int:
    return lib().pthread_self()


def thread_id(pthread: int) -> int | None:
    """The kernel thread id of a pthread_t (threading.get_native_id() on macOS), or None."""
    out = ctypes.c_uint64(0)
    return int(out.value) if lib().pthread_threadid_np(pthread, ctypes.byref(out)) == 0 else None


def get_qos(pthread: int) -> dict:
    """The thread's requested QoS: {"rc", "qos", "relpri"} (overrides are not visible here)."""
    q, r = ctypes.c_uint(0), ctypes.c_int(0)
    rc = lib().pthread_get_qos_class_np(pthread, ctypes.byref(q), ctypes.byref(r))
    return {"rc": int(rc), "qos": int(q.value), "relpri": int(r.value)}


def set_qos_self(qos: int, relpri: int = 0) -> int:
    return int(lib().pthread_set_qos_class_self_np(qos, relpri))


def override_start(pthread: int, qos: int, relpri: int) -> int | None:
    return lib().pthread_override_qos_class_start_np(pthread, qos, relpri)


def override_end(handle: int) -> int:
    return int(lib().pthread_override_qos_class_end_np(handle))


class OverrideRegistry:
    """Every dependency override of one run, with its bookkeeping (thread-safe).

    Per override (token = index): start_call_ns (before the start call), start_ns (after it
    returned), end_ns (before the end call), end_ret_ns (after it returned), end_rc, and how it was
    ended ("client" or "end_all"). The start / end functions are injectable for tests."""

    def __init__(self, qos: int = QOS_CLASS_USER_INITIATED, relpri: int = 0, start_fn=None, end_fn=None):
        self.qos, self.relpri = qos, relpri
        self._start = start_fn or override_start
        self._end = end_fn or override_end
        self._lock = threading.Lock()
        self._open: dict[int, int] = {}  # token -> override object
        self.rows: list[list] = []  # [start_call_ns, start_ns, end_ns, end_ret_ns, end_rc, by]
        self.starts = self.ends = self.null_starts = self.end_errors = 0
        self.double_ends = 0
        self.ended_by_end_all = 0
        self.outstanding_at_end: int | None = None

    def start(self, target: int | None) -> int | None:
        c = time.monotonic_ns()
        h = self._start(target, self.qos, self.relpri) if target else None
        t = time.monotonic_ns()
        with self._lock:
            if not h:
                self.null_starts += 1
                return None
            self.starts += 1
            token = len(self.rows)
            self.rows.append([c, t, 0, 0, None, None])
            self._open[token] = h
            return token

    def end(self, token: int | None, by: str = "client") -> int | None:
        if token is None:  # the start failed: nothing to end
            return None
        with self._lock:
            h = self._open.pop(token, None)
            if h is None:  # unknown or already ended: recorded, never passed to libSystem
                self.end_errors += 1
                self.double_ends += 1
                return None
        e = time.monotonic_ns()
        rc = self._end(h)
        r = time.monotonic_ns()
        with self._lock:
            row = self.rows[token]
            row[2:] = [e, r, rc, by]
            if by == "client":
                self.ends += 1
            else:
                self.ended_by_end_all += 1
            if rc != 0:
                self.end_errors += 1
        return rc

    @property
    def outstanding(self) -> int:
        with self._lock:
            return len(self._open)

    def end_all(self) -> int:
        """End every override still outstanding; returns how many there were. The first call's count
        is kept as outstanding_at_end."""
        with self._lock:
            left = sorted(self._open)
        if self.outstanding_at_end is None:
            self.outstanding_at_end = len(left)
        for token in left:
            self.end(token, by="end_all")
        return len(left)

    def summary(self) -> dict:
        with self._lock:
            return {
                "qos": self.qos,
                "relpri": self.relpri,
                "starts": self.starts,
                "ends": self.ends,
                "null_starts": self.null_starts,
                "end_errors": self.end_errors,
                "double_ends": self.double_ends,
                "ended_by_end_all": self.ended_by_end_all,
                "outstanding_at_end": self.outstanding_at_end,
                "outstanding": len(self._open),
            }

    def record(self) -> dict:
        """summary() plus the per-override times as column arrays."""
        with self._lock:
            rows = [list(r) for r in self.rows]
        cols = ("start_call_ns", "start_ns", "end_ns", "end_ret_ns", "end_rc", "ended_by")
        return {**self.summary(), "overrides": {c: [r[i] for r in rows] for i, c in enumerate(cols)}}
