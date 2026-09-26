"""Staged Core ML handoff for an in-process ANE (opt-in; research/coreml-staged-handoff/).

Per ANE forward it answers which Core ML path runs that forward: "sync" (coremltools'
predict, today's production path) or "async" (the prebound asynchronous predict,
laya_apple/backends/coreml_async.py).

    ARMED ──(a forward while the GPU is active)──> SYNC_GUARD
    SYNC_GUARD: the first `guard` forwards of the episode run sync; after the guard-th one
      the state is ASYNC_STEADY
    ASYNC_STEADY: forwards run async
    any state ──(GPU not active, or no forward for more than `gap_s`)──> ARMED
    any state ──(disable(reason))──> DISABLED: sync, permanently

- "GPU active" at a forward: the GPU worker has a job queued or running, or its last job
  ended at most `gap_s` ago (GpuActivity, fed by the GPU worker's submit and the job's
  Future completing, whatever the outcome: result, error, cancellation, dead worker, close).
- An episode ends when the GPU has been idle for more than `gap_s`, or when no forward came
  for more than `gap_s` (hetero -> single device -> hetero re-arms).
- ARMED runs sync: outside a hetero episode the ANE path is today's production path.
- The guard counts ANE forwards (one per ANE job), not requests seen by clients. Every ANE
  forward counts, including tie-band forwards (buckets outside auto's own, sent to the ANE by
  queue-aware routing) and multi-row jobs (one decision for the whole job). The load-time
  warm-up runs before the handoff is attached and is not counted.

Only real requests move the state: no background thread, no sleep, no timer, no CPU
performance-level counter, no private API and no QoS call. Everything is evaluated lazily
when a forward arrives. Thread-safe: one lock per object.

The state machine is the one the research screen and the H64 confirmation ran
(research/coreml-staged-handoff/scripts/handoff.py, guard 64), with three additions that do
not change a decision of an enabled handoff: DISABLED (disable()), per-path forward counts,
and an invariant check that disables the handoff, with one warning, if its state was ever
found inconsistent. snapshot() exposes all of it.
"""

from __future__ import annotations

import threading
import time
import warnings

ARMED, SYNC_GUARD, ASYNC_STEADY, DISABLED = "armed", "sync_guard", "async_steady", "disabled"
SYNC, ASYNC = "sync", "async"
STATES = (ARMED, SYNC_GUARD, ASYNC_STEADY, DISABLED)
GUARD = 64  # the H64 cell of research/coreml-staged-handoff/
GAP_S = 1.0


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
    """The per-forward Core ML path decision (guard = 64: the confirmed H64 cell)."""

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
        self.count = 0  # sync forwards in the current episode
        self.episodes = 0
        self.disabled_reason: str | None = None
        self.forwards = {SYNC: 0, ASYNC: 0}
        self._last_ns: int | None = None

    def _inconsistency(self) -> str | None:
        """Why the current state is impossible for this machine, or None. Caller holds the lock."""
        s, c = self.state, self.count
        if s not in STATES:
            return f"unknown state {s!r}"
        if not isinstance(c, int) or (s in (ARMED, DISABLED) and c != 0):
            return f"count {c!r} in state {s}"
        if s == SYNC_GUARD and not 1 <= c < self.guard:
            return f"count {c} in state {s} (guard {self.guard})"
        if s == ASYNC_STEADY and c != self.guard:
            return f"count {c} in state {s} (guard {self.guard})"
        if set(self.forwards) != {SYNC, ASYNC} or min(self.forwards.values()) < 0 or self.episodes < 0:
            return f"counters {self.forwards!r}, episodes {self.episodes!r}"
        return None

    def decide(self) -> tuple[str, str, int]:
        """(path, state after the decision, guard count) for one forward."""
        now = self._clock()
        gpu_active = self._gpu.active(now, self._gap_ns)
        corrupted = None
        with self._lock:
            if self.state != DISABLED and (why := self._inconsistency()) is not None:
                corrupted = f"handoff state corrupted: {why}"
                self.state, self.count, self.disabled_reason = DISABLED, 0, corrupted
                if set(self.forwards) != {SYNC, ASYNC}:  # the reason keeps the corrupted counters
                    self.forwards = {SYNC: 0, ASYNC: 0}
            if self.state == DISABLED:
                self.forwards[SYNC] += 1
                out = (SYNC, self.state, 0)
            else:
                paused = self._last_ns is None or now - self._last_ns > self._gap_ns
                self._last_ns = now
                if not gpu_active:
                    self.state, self.count = ARMED, 0
                    path = SYNC
                else:
                    if self.state == ARMED or paused:
                        self.state, self.count = SYNC_GUARD, 0
                        self.episodes += 1
                    if self.state == SYNC_GUARD:
                        self.count += 1
                        if self.count >= self.guard:
                            self.state = ASYNC_STEADY
                        path = SYNC
                    else:
                        path = ASYNC
                self.forwards[path] += 1
                out = (path, self.state, self.count)
        if corrupted is not None:
            warnings.warn(
                f"laya-apple: {corrupted}; the staged handoff is disabled: every later ANE forward runs the "
                "coremltools path",
                RuntimeWarning,
                stacklevel=2,
            )
        return out

    def disable(self, reason: str) -> bool:
        """Pin every later forward to sync, permanently. True only for the call that disabled
        it, so the caller warns once."""
        with self._lock:
            if self.state == DISABLED:
                return False
            self.state, self.count, self.disabled_reason = DISABLED, 0, reason
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
                "episodes": self.episodes,
                "forwards": dict(self.forwards),
                "consistent": self._inconsistency() is None,
            }
