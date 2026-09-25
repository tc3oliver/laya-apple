"""GIL-contention records that do not depend on GPU completions (criteria.md, not gating).

GilProbe   a daemon thread that sleeps to a 1 ms grid and records, for every wake, how late it
           ran Python code again (scheduled tick -> running). Beyond the OS wake-up latency,
           that lateness is time spent waiting for the GIL, whichever thread held it.
thread_cpu one snapshot of every thread's CPU time (user + system, ns) in this process, from
           Mach thread_info(THREAD_EXTENDED_INFO), keyed by the thread id that
           threading.get_native_id() returns (THREAD_IDENTIFIER_INFO). The Mach calls are made
           through ctypes.PyDLL, which keeps the GIL: a snapshot hands the GIL to no one.
"""

from __future__ import annotations

import ctypes
import threading
import time
from array import array

# ------------------------------------------------------------------ Mach per-thread CPU

_THREAD_EXTENDED_INFO = 5
_THREAD_IDENTIFIER_INFO = 4


class _ExtendedInfo(ctypes.Structure):
    _fields_ = [
        ("pth_user_time", ctypes.c_uint64),
        ("pth_system_time", ctypes.c_uint64),
        ("pth_cpu_usage", ctypes.c_int32),
        ("pth_policy", ctypes.c_int32),
        ("pth_run_state", ctypes.c_int32),
        ("pth_flags", ctypes.c_int32),
        ("pth_sleep_time", ctypes.c_int32),
        ("pth_curpri", ctypes.c_int32),
        ("pth_priority", ctypes.c_int32),
        ("pth_maxpriority", ctypes.c_int32),
        ("pth_name", ctypes.c_char * 64),
    ]


class _IdentifierInfo(ctypes.Structure):
    _fields_ = [("thread_id", ctypes.c_uint64), ("thread_handle", ctypes.c_uint64), ("dispatch_qaddr", ctypes.c_uint64)]


_mach = None


def _lib():
    global _mach
    if _mach is None:
        lib = ctypes.PyDLL("/usr/lib/libSystem.B.dylib")  # PyDLL: the GIL is kept across calls
        lib.task_threads.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        lib.thread_info.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        lib.mach_port_deallocate.argtypes = [ctypes.c_uint, ctypes.c_uint]
        lib.vm_deallocate.argtypes = [ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t]
        _mach = (lib, ctypes.c_uint.in_dll(lib, "mach_task_self_").value)
    return _mach


def thread_cpu() -> dict[int, int]:
    """{native thread id: user + system CPU ns} for every thread of this process."""
    lib, task = _lib()
    threads, n = ctypes.POINTER(ctypes.c_uint)(), ctypes.c_uint()
    if lib.task_threads(task, ctypes.byref(threads), ctypes.byref(n)) != 0:
        raise OSError("task_threads failed")
    out = {}
    try:
        for i in range(n.value):
            port = threads[i]
            ext, ident = _ExtendedInfo(), _IdentifierInfo()
            ce = ctypes.c_uint(ctypes.sizeof(ext) // 4)
            ci = ctypes.c_uint(ctypes.sizeof(ident) // 4)
            ok = lib.thread_info(port, _THREAD_EXTENDED_INFO, ctypes.byref(ext), ctypes.byref(ce)) == 0
            ok = ok and lib.thread_info(port, _THREAD_IDENTIFIER_INFO, ctypes.byref(ident), ctypes.byref(ci)) == 0
            if ok:
                out[int(ident.thread_id)] = int(ext.pth_user_time + ext.pth_system_time)
            lib.mach_port_deallocate(task, port)
    finally:
        addr = ctypes.cast(threads, ctypes.c_void_p).value
        if addr:
            lib.vm_deallocate(task, addr, n.value * ctypes.sizeof(ctypes.c_uint))
    return out


def snapshot() -> dict:
    """A named snapshot: {"t_ns", "threads": {name: cpu_ns}}. Python threads by their name; any
    other thread (Core ML, MLX, libdispatch) summed under "other"."""
    t = time.monotonic_ns()
    cpu = thread_cpu()
    names = {th.native_id: th.name for th in threading.enumerate() if th.native_id is not None}
    threads: dict[str, int] = {}
    for tid, ns in cpu.items():
        name = names.get(tid, "other")
        threads[name] = threads.get(name, 0) + ns
    return {"t_ns": t, "threads": threads}


# ------------------------------------------------------------------ GIL probe


def next_tick(tick: int, now: int, period: int) -> int:
    """The first grid tick after `now`: late wakes skip the ticks they missed, not bunch them."""
    tick += period
    if tick <= now:
        tick += ((now - tick) // period + 1) * period
    return tick


class GilProbe(threading.Thread):
    def __init__(self, period_ns: int = 1_000_000):
        super().__init__(name="gil-probe", daemon=True)
        self.period = period_ns
        self.ticks = array("q")  # scheduled tick, time.monotonic_ns()
        self.late = array("q")  # ns from the tick to running Python code again
        self.cpu_ns = 0
        self._stop_flag = False

    def run(self):
        c0 = time.thread_time_ns()
        tick = time.monotonic_ns() + self.period
        while not self._stop_flag:
            d = tick - time.monotonic_ns()
            if d > 0:
                time.sleep(d / 1e9)
            now = time.monotonic_ns()
            self.ticks.append(tick)
            self.late.append(now - tick)
            tick = next_tick(tick, now, self.period)
        self.cpu_ns = time.thread_time_ns() - c0

    def stop(self):
        self._stop_flag = True
        self.join()
