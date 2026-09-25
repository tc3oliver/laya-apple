"""The staged-handoff state machine (research copy; ../criteria.md).

Per ANE forward of the auto instance it answers which Core ML path runs that forward:
"sync" (today's production path, coremltools with the GIL held) or "async" (PB-ASYNC).

    ARMED ──(short forward while the GPU is active)──> SYNC_GUARD
    SYNC_GUARD: the first `guard` short forwards of the episode run sync; after the guard-th
      one the state is ASYNC_STEADY
    ASYNC_STEADY: short forwards run async
    any state ──(GPU not active, or no short forward for more than `gap_s`)──> ARMED

- "GPU active" at a short forward: the GPU worker has a job queued or running, or its last job
  ended at most `gap_s` ago (GpuActivity). Only real requests move it: no sleep, no fake
  inference, no background thread; everything is evaluated lazily when a short forward arrives.
- An episode ends when the GPU has been idle for more than `gap_s`, or when no short forward
  came for more than `gap_s` (hetero -> single device -> hetero re-arms).
- ARMED runs sync: outside a hetero episode the ANE path is today's production path.
- The guard counts forwards, not requests seen by clients: one forward per ANE job.

Thread-safe (one lock); the research harness calls decide() from the ANE dispatcher thread and
the GpuActivity hooks from the client and GPU dispatcher threads.
"""

from __future__ import annotations

import threading
import time

ARMED, SYNC_GUARD, ASYNC_STEADY = "armed", "sync_guard", "async_steady"
SYNC, ASYNC = "sync", "async"
STATES = (ARMED, SYNC_GUARD, ASYNC_STEADY)
GAP_S = 1.0


class GpuActivity:
    """GPU jobs in flight and the end of the last one, from the submit and completion hooks."""

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

    def active(self, now_ns: int, gap_ns: int) -> bool:
        with self._lock:
            if self._inflight > 0:
                return True
            return self._last_end_ns is not None and now_ns - self._last_end_ns <= gap_ns


class StagedHandoff:
    """The per-forward path decision of cells H32 / H64 (guard = 32 / 64)."""

    def __init__(self, guard: int, gpu: GpuActivity, gap_s: float = GAP_S, clock=time.monotonic_ns):
        if guard < 1:
            raise ValueError("guard must be >= 1")
        self.guard = guard
        self._gpu = gpu
        self._gap_ns = int(gap_s * 1e9)
        self._clock = clock
        self._lock = threading.Lock()
        self.state = ARMED
        self.count = 0  # sync forwards in the current episode
        self._last_short_ns: int | None = None
        self.episodes = 0

    def decide(self) -> tuple[str, str, int]:
        """(path, state after the decision, guard count) for one short forward."""
        now = self._clock()
        gpu_active = self._gpu.active(now, self._gap_ns)
        with self._lock:
            paused = self._last_short_ns is None or now - self._last_short_ns > self._gap_ns
            self._last_short_ns = now
            if not gpu_active:
                self.state, self.count = ARMED, 0
                return SYNC, self.state, self.count
            if self.state == ARMED or paused:
                self.state, self.count = SYNC_GUARD, 0
                self.episodes += 1
            if self.state == SYNC_GUARD:
                self.count += 1
                if self.count >= self.guard:
                    self.state = ASYNC_STEADY
                return SYNC, self.state, self.count
            return ASYNC, self.state, self.count


class Fixed:
    """Cells A (always sync) and B (always async): the same interface, no state."""

    def __init__(self, path: str):
        if path not in (SYNC, ASYNC):
            raise ValueError(path)
        self.path = path
        self.state = "fixed"
        self.count = 0
        self.episodes = 0

    def decide(self) -> tuple[str, str, int]:
        return self.path, self.state, 0
