"""Per-thread, per-perf-level CPU counters (research only): the primary scheduler instrumentation.

proc_pidinfo(pid, PROC_PIDTHREADCOUNTS = 34, thread_id, buf, size) fills, for one thread, a
header (uint16 len, uint16, uint32) and, per CPU perf level (sysctl hw.nperflevels; perflevel0 is
"Performance", perflevel1 "Efficiency" on M4 Max), the thread's cumulative instructions, cycles,
user and system time (mach time units) and energy (nJ) on cores of that level. The thread ids of
another process come from proc_pidinfo(pid, PROC_PIDLISTTHREADIDS = 28, ...). Both are a private,
undocumented interface (XNU bsd/sys/proc_info_private.h); a failed read is recorded, never
raised. The calls go through ctypes.PyDLL, which keeps the GIL: a sample hands the GIL to no one.

Sampler: a daemon thread "laya-recount" that wakes on a SAMPLE_PERIOD_NS grid of
time.monotonic_ns() and reads the target threads: in this process the Python threads named
MainThread, client-short, client-long, laya-ane-dispatch, laya-gpu-dispatch and
coreml-callback, and any extra thread id handed in (the completion callbacks' ids); in every
extra process (the GPU worker), all of its threads, listed again at each sample. Per sample it
keeps the stamp and, per (pid, thread id), the counts per level. Its own thread CPU is measured.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time

PROC_PIDLISTTHREADIDS = 28
PROC_PIDTHREADCOUNTS = 34
SAMPLE_PERIOD_NS = 100_000_000
TARGET_NAMES = (
    "MainThread",
    "client-short",
    "client-long",
    "laya-ane-dispatch",
    "laya-gpu-dispatch",
    "coreml-callback",
)
FIELDS = ("instructions", "cycles", "cpu_ns", "energy_nj")  # per level, in this order
INTERFACE = "proc_pidinfo PROC_PIDTHREADCOUNTS (34) / PROC_PIDLISTTHREADIDS (28): private, XNU proc_info_private.h"

_lib = None


class _Level(ctypes.Structure):
    _fields_ = [
        ("instructions", ctypes.c_uint64),
        ("cycles", ctypes.c_uint64),
        ("user_mach", ctypes.c_uint64),
        ("system_mach", ctypes.c_uint64),
        ("energy_nj", ctypes.c_uint64),
    ]


def nperflevels() -> int:
    out = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(out))
    lib = _libsystem()
    if lib.sysctlbyname(b"hw.nperflevels", ctypes.byref(out), ctypes.byref(size), None, 0) != 0:
        return 1
    return int(out.value)


def perflevel_names(n: int) -> list[str]:
    names = []
    lib = _libsystem()
    for i in range(n):
        buf = ctypes.create_string_buffer(64)
        size = ctypes.c_size_t(64)
        ok = lib.sysctlbyname(f"hw.perflevel{i}.name".encode(), buf, ctypes.byref(size), None, 0) == 0
        names.append(buf.value.decode() if ok else f"perflevel{i}")
    return names


def _libsystem():
    global _lib
    if _lib is None:
        lib = ctypes.PyDLL("/usr/lib/libSystem.B.dylib")  # PyDLL: the GIL is kept across calls
        lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        lib.proc_pidinfo.restype = ctypes.c_int
        lib.sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.mach_timebase_info.argtypes = [ctypes.c_void_p]
        _lib = lib
    return _lib


def _timebase() -> tuple[int, int]:
    class TB(ctypes.Structure):
        _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]

    tb = TB()
    _libsystem().mach_timebase_info(ctypes.byref(tb))
    return tb.numer, tb.denom


class Reader:
    def __init__(self):
        self.levels = nperflevels()

        class Counts(ctypes.Structure):
            _fields_ = [("len", ctypes.c_uint16), ("r0", ctypes.c_uint16), ("r1", ctypes.c_uint32)]
            _fields_ += [("counts", _Level * self.levels)]

        self._buf = Counts()
        self._size = ctypes.sizeof(Counts)
        self._numer, self._denom = _timebase()
        self._ids = (ctypes.c_uint64 * 512)()
        self.lib = _libsystem()

    def thread(self, pid: int, tid: int) -> list[int] | None:
        """[instructions, cycles, cpu_ns, energy_nj] per level, flattened; None if the read failed."""
        n = self.lib.proc_pidinfo(pid, PROC_PIDTHREADCOUNTS, tid, ctypes.byref(self._buf), self._size)
        if n <= 0:
            return None
        out = []
        for c in self._buf.counts:
            mach = c.user_mach + c.system_mach
            out += [c.instructions, c.cycles, mach * self._numer // self._denom, c.energy_nj]
        return out

    def thread_ids(self, pid: int) -> list[int] | None:
        n = self.lib.proc_pidinfo(pid, PROC_PIDLISTTHREADIDS, 0, self._ids, ctypes.sizeof(self._ids))
        if n <= 0:
            return None
        return [int(self._ids[i]) for i in range(n // 8)]


class Sampler(threading.Thread):
    """Samples the target threads every SAMPLE_PERIOD_NS until stop()."""

    def __init__(self, extra_tids=None, extra_pids=None, period_ns: int = SAMPLE_PERIOD_NS):
        super().__init__(name="laya-recount", daemon=True)
        self.period = period_ns
        self.extra_tids = extra_tids or (lambda: ())  # callables: evaluated at every sample
        self.extra_pids = extra_pids or (lambda: ())
        self.reader = Reader()
        self.pid = os.getpid()
        self.t: list[int] = []  # sample stamps
        self.series: dict[str, dict] = {}  # "pid:tid" -> {"name", "rows": [[sample index, *counts]]}
        self.errors: dict[str, int] = {}
        self.cpu_ns = 0
        self._stop_flag = False

    def _add(self, key: str, name: str, pid: int, tid: int, k: int) -> None:
        v = self.reader.thread(pid, tid)
        if v is None:
            self.errors[key] = self.errors.get(key, 0) + 1
            return
        s = self.series.get(key)
        if s is None:
            s = self.series[key] = {"name": name, "pid": pid, "tid": tid, "rows": []}
        s["rows"].append([k, *v])

    def sample(self) -> None:
        k = len(self.t)
        self.t.append(time.monotonic_ns())
        seen = set()
        for th in threading.enumerate():
            if th.native_id is not None and th.name in TARGET_NAMES:
                seen.add(th.native_id)
                self._add(f"{self.pid}:{th.native_id}", th.name, self.pid, th.native_id, k)
        for tid in self.extra_tids():
            if tid not in seen:
                self._add(f"{self.pid}:{tid}", "coreml-callback", self.pid, tid, k)
        for pid in self.extra_pids():
            ids = self.reader.thread_ids(pid)
            if ids is None:
                self.errors[f"{pid}:list"] = self.errors.get(f"{pid}:list", 0) + 1
                continue
            for tid in ids:
                self._add(f"{pid}:{tid}", "gpu-worker", pid, tid, k)

    def run(self):
        c0 = time.thread_time_ns()
        tick = time.monotonic_ns()
        while not self._stop_flag:
            try:
                self.sample()
            except Exception as e:  # recorded, the run goes on
                self.errors[repr(e)] = self.errors.get(repr(e), 0) + 1
            tick += self.period
            d = tick - time.monotonic_ns()
            if d > 0:
                time.sleep(d / 1e9)
            else:
                tick = time.monotonic_ns()
        self.cpu_ns = time.thread_time_ns() - c0

    def stop(self) -> None:
        self._stop_flag = True
        self.join()

    def record(self) -> dict:
        t = self.t
        span = (t[-1] - t[0]) if len(t) > 1 else 0
        gaps = [b - a for a, b in zip(t, t[1:])]
        return {
            "interface": INTERFACE,
            "levels": self.reader.levels,
            "level_names": perflevel_names(self.reader.levels),
            "fields_per_level": list(FIELDS),
            "period_ns": self.period,
            "t_ns": t,
            "series": self.series,
            "errors": self.errors,
            "sampler_cpu_ns": self.cpu_ns,
            "sampler_cpu_fraction_of_core": self.cpu_ns / span if span else None,
            "interval_ns": {
                "n": len(gaps),
                "p50": sorted(gaps)[len(gaps) // 2] if gaps else None,
                "max": max(gaps) if gaps else None,
                "min": min(gaps) if gaps else None,
            },
        }


_libc = None


def set_os_thread_name(name: str) -> None:
    """pthread_setname_np for the calling thread (macOS: only the calling thread), so the name
    also shows in tools that read OS thread names."""
    global _libc
    try:
        if _libc is None:
            import ctypes

            _libc = ctypes.PyDLL("/usr/lib/libSystem.B.dylib")
        _libc.pthread_setname_np(name.encode()[:63])
    except Exception:
        pass
