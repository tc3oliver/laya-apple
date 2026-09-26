"""Adaptive Core ML execution for an in-process ANE: a staged handoff with a slow-state breaker.

Per ANE forward it answers which Core ML path runs that forward: "sync" (coremltools'
predict, the production path of 1.4) or "async" (the prebound asynchronous predict,
laya_apple/backends/coreml_async.py).

    ARMED ──(a forward while the GPU is active)──> SAFE_SYNC  (a new episode)
    SAFE_SYNC: the first `guard` (64) forwards of the episode run sync; after the guard-th one
      the state is ASYNC_HEALTHY
    ASYNC_HEALTHY: forwards run async; every completed request whose forward ran async is
      observed (C3): prepare_ms > 0.3 ms -> count + 1, otherwise count = 0; at 3 the breaker
      opens
    BREAKER_OPEN: forwards run sync for the rest of the episode (no retry, no half-open state)
    any state ──(GPU not active, or no forward for more than `gap_s`)──> ARMED  (re-armed)
    any state ──(disable(reason))──> DISABLED: sync, permanently

Roles:
- **The 64-forward guard** only steps over the known transient at the start of a hetero
  episode. It is not the safety mechanism.
- **The safety mechanism is the breaker.** C3 watches PB-ASYNC from the handoff onward.
  Once C3 trips, every forward whose path is chosen afterwards runs the production path until
  the episode ends. A request already running is never cancelled or re-sent.
- **The evidence** is in research/coreml-adaptive-breaker/:
  - Phase 0: C3 caught 16 of 16 recorded slow states from `prepare_ms` alone.
  - Phase 1: falling back recovered A-like latency within 164-414 ms in 12 of 12 slow
    episodes.
- **The research lineage:** the guard is research/coreml-staged-handoff/'s H64 cell. C3 and
  the breaker are research/coreml-adaptive-breaker/scripts/breaker.py.

Signals:
- **prepare_ms:** `Laya.prepare`'s duration (tokenising and layout) on the submitting thread.
  It is measured for every request while the handoff is in use, whether or not trace= is set.
- **Observed** on the ANE dispatcher thread as the request completes, before its Future
  resolves. In a closed loop the next forward's decision therefore already sees it.
- **"GPU active" at a forward:** the GPU worker has a job queued or running, or its last job
  ended at most `gap_s` ago (GpuActivity, fed by the GPU worker's submit and the job's Future
  completing, whatever the outcome).
- **An episode ends** when the GPU has been idle for more than `gap_s`, or when no forward came
  for more than `gap_s`: hetero -> single device -> hetero re-arms.
- **ARMED runs sync:** outside a hetero episode the ANE path is 1.4's production path.
- **The guard counts ANE forwards** (one per ANE job), not requests seen by clients. The
  load-time warm-up runs before the handoff is attached and is not counted.

Only real requests move the state: no background thread, no sleep, no timer, no CPU
performance-level counter, no private API and no QoS call. Thread-safe: one lock per object.
An invariant check disables the handoff, with one warning, if its state is ever found
inconsistent. snapshot() exposes all of it (Laya.info()["ane_handoff"]).
"""

from __future__ import annotations

import threading
import time
import warnings

ARMED, SAFE_SYNC, ASYNC_HEALTHY, BREAKER_OPEN, DISABLED = (
    "armed",
    "safe_sync",
    "async_healthy",
    "breaker_open",
    "disabled",
)
SYNC, ASYNC = "sync", "async"
STATES = (ARMED, SAFE_SYNC, ASYNC_HEALTHY, BREAKER_OPEN, DISABLED)
GUARD = 64  # research/coreml-staged-handoff/'s H64
GAP_S = 1.0
C3_THRESHOLD_MS = 0.3  # research/coreml-adaptive-breaker/ (frozen)
C3_CONSECUTIVE = 3


class GpuActivity:
    """GPU jobs in flight and the end of the last one, from the GPU worker's submit and
    completion hooks (executor.DeviceWorker.activity)."""

    def __init__(self, clock=time.monotonic_ns):
        self._lock = threading.Lock()
        self._clock = clock
        self._inflight = 0
        self._last_end_ns: int | None = None

    def started(self) -> None:
        with self._lock:
            self._inflight += 1

    def ended(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._last_end_ns = self._clock()

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def active(self, now_ns: int, gap_ns: int) -> bool:
        with self._lock:
            if self._inflight > 0:
                return True
            return self._last_end_ns is not None and now_ns - self._last_end_ns <= gap_ns


class StagedHandoff:
    """The per-forward Core ML path decision: 64 safe sync forwards, then PB-ASYNC under the
    C3 breaker, per hetero episode."""

    def __init__(self, gpu: GpuActivity, guard: int = GUARD, gap_s: float = GAP_S, clock=time.monotonic_ns):
        if guard < 1:
            raise ValueError("guard must be >= 1")
        self.guard = guard
        self.gap_s = gap_s
        self._gpu = gpu
        self._gap_ns = int(gap_s * 1e9)
        self._clock = clock
        self._lock = threading.Lock()
        self.state = ARMED
        self.count = 0  # sync forwards of the current episode's guard
        self.c3 = 0  # consecutive host-slow async requests in the current episode
        self.episodes = 0
        self.trips = 0
        self.disabled_reason: str | None = None
        self.forwards = {SYNC: 0, ASYNC: 0}
        self._last_ns: int | None = None
        self._last_path = SYNC  # the path of the latest decided forward

    def _inconsistency(self) -> str | None:
        """Why the current state is impossible for this machine, or None. Caller holds the lock."""
        s, c, k = self.state, self.count, self.c3
        if s not in STATES:
            return f"unknown state {s!r}"
        if not isinstance(c, int) or not isinstance(k, int):
            return f"count {c!r}, c3 {k!r}"
        if s in (ARMED, DISABLED) and (c != 0 or k != 0):
            return f"count {c}, c3 {k} in state {s}"
        if s == SAFE_SYNC and (not 1 <= c < self.guard or k != 0):
            return f"count {c}, c3 {k} in state {s} (guard {self.guard})"
        if s == ASYNC_HEALTHY and (c != self.guard or not 0 <= k < C3_CONSECUTIVE):
            return f"count {c}, c3 {k} in state {s} (guard {self.guard})"
        if s == BREAKER_OPEN and (c != self.guard or k != C3_CONSECUTIVE):
            return f"count {c}, c3 {k} in state {s} (guard {self.guard})"
        if set(self.forwards) != {SYNC, ASYNC} or min(self.forwards.values()) < 0 or self.episodes < 0:
            return f"counters {self.forwards!r}, episodes {self.episodes!r}"
        if not 0 <= self.trips <= self.episodes:
            return f"trips {self.trips!r}, episodes {self.episodes!r}"
        return None

    def _corrupt(self) -> str | None:
        """Disable on an impossible state; the reason, once. Caller holds the lock."""
        if self.state == DISABLED or (why := self._inconsistency()) is None:
            return None
        reason = f"handoff state corrupted: {why}"
        self.state, self.count, self.c3, self.disabled_reason = DISABLED, 0, 0, reason
        if set(self.forwards) != {SYNC, ASYNC}:
            self.forwards = {SYNC: 0, ASYNC: 0}
        return reason

    @staticmethod
    def _warn_corrupted(reason: str | None) -> None:
        if reason is not None:
            warnings.warn(
                f"laya-apple: {reason}; adaptive execution is disabled: every later ANE forward runs the "
                "coremltools path",
                RuntimeWarning,
                stacklevel=3,
            )

    def decide(self) -> tuple[str, str, int]:
        """(path, state after the decision, guard count) for one forward."""
        now = self._clock()
        gpu_active = self._gpu.active(now, self._gap_ns)
        with self._lock:
            corrupted = self._corrupt()
            if self.state == DISABLED:
                path = SYNC
            else:
                paused = self._last_ns is None or now - self._last_ns > self._gap_ns
                self._last_ns = now
                if not gpu_active:
                    self.state, self.count, self.c3 = ARMED, 0, 0
                    path = SYNC
                else:
                    if self.state == ARMED or paused:
                        self.state, self.count, self.c3 = SAFE_SYNC, 0, 0
                        self.episodes += 1
                    if self.state == SAFE_SYNC:
                        self.count += 1
                        if self.count >= self.guard:
                            self.state = ASYNC_HEALTHY
                        path = SYNC
                    elif self.state == ASYNC_HEALTHY:
                        path = ASYNC
                    else:  # BREAKER_OPEN
                        path = SYNC
            self.forwards[path] += 1
            self._last_path = path
            out = (path, self.state, self.count)
        self._warn_corrupted(corrupted)
        return out

    def observe(self, prepare_ms: float) -> bool:
        """Feed one completed ANE request's prepare time, on the thread that ran its forward, before
        the next forward is decided. Only a request whose forward ran async, in ASYNC_HEALTHY,
        counts. True only for the call that opened the breaker. Never raises."""
        with self._lock:
            if self.state != ASYNC_HEALTHY or self._last_path != ASYNC:
                return False
            try:
                slow = float(prepare_ms) > C3_THRESHOLD_MS
            except (TypeError, ValueError):
                return False
            self.c3 = self.c3 + 1 if slow else 0
            if self.c3 < C3_CONSECUTIVE:
                return False
            self.state = BREAKER_OPEN
            self.trips += 1
            return True

    def disable(self, reason: str) -> bool:
        """Pin every later forward to sync, permanently. True only for the call that disabled
        it, so the caller warns once."""
        with self._lock:
            if self.state == DISABLED:
                return False
            self.state, self.count, self.c3, self.disabled_reason = DISABLED, 0, 0, reason
            return True

    @property
    def enabled(self) -> bool:
        return self.state != DISABLED

    def snapshot(self) -> dict:
        """A consistent, JSON-serialisable copy of the state, for info() and debugging."""
        with self._lock:
            disabled = self.state == DISABLED
            return {
                "enabled": not disabled,
                "disabled": disabled,
                "disabled_reason": self.disabled_reason,
                "state": self.state,
                "guard": self.guard,
                "gap_s": self.gap_s,
                "count": self.count,
                "breaker": {
                    "rule": f"{C3_CONSECUTIVE} consecutive async requests with prepare > {C3_THRESHOLD_MS} ms",
                    "count": self.c3,
                    "open": self.state == BREAKER_OPEN,
                    "trips": self.trips,
                },
                "episodes": self.episodes,
                "forwards": dict(self.forwards),
                "consistent": self._inconsistency() is None,
            }
