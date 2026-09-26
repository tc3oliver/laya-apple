"""The adaptive breaker of cell R (research copy; ../phase1.md). Frozen before Phase 1's data.

Per ANE forward of the auto instance it answers which Core ML path runs that forward: "async"
(PB-ASYNC) or "sync" (the production coremltools path, 1.4's A). It uses one detector, C3, and no
other signal.

    IDLE ──(an ANE forward while the GPU is active)──> HEALTHY  (a new episode; C3 count = 0)
    HEALTHY: forwards run async; every completed ANE request is observed, and from 1.0 s after
      the episode start (ARM_S) it counts:
      prepare_ms > 0.3 ms  -> count += 1, otherwise count = 0
      count reaches 3      -> OPEN (one trip, recorded with the request that caused it)
    OPEN: forwards run sync for the rest of the episode. No async retry, no half-open state,
      no cooldown.
    any state ──(GPU not active, or no ANE forward for more than 1.0 s)──> IDLE  (episode end, re-arm)
    IDLE: forwards run async (outside a hetero episode R is B).

- **C3 (frozen):** 3 consecutive ANE requests with `RequestTrace.prepare_ms > 0.3 ms`, counted in
  completion order.
  - Only requests observed while the episode is HEALTHY and at least ARM_S = 1.0 s after its
    start (the first HEALTHY decision) count. Every hetero start has a short transient with
    prepare > 0.3 ms in every cell, production A included (#102 / #103: C3 from t0 trips in 40
    of 40 A windows, from t0 + 1 s in none); Phase 0 validated C3 only after that transient.
  - The count resets to 0 at every episode start, and on every request with prepare ≤ 0.3 ms.
- **The observation** is the trace callback of the public `trace=` option: one RequestTrace per
  completed request, on the ANE dispatcher thread, before that request's Future resolves.
  - In a closed loop, therefore, the forward after the tripping request is already sync.
- **"GPU active" and the 1.0 s gap** are the staged handoff's `handoff.GpuActivity` and gap rule,
  unchanged. Only real requests move them: no sleep, no probe, no background thread.

Thread-safe: one lock covers decide() and observe().
"""

from __future__ import annotations

import threading
import time

IDLE, HEALTHY, OPEN = "idle", "healthy", "open"
SYNC, ASYNC = "sync", "async"
STATES = (IDLE, HEALTHY, OPEN)
THRESHOLD_MS = 0.3
CONSECUTIVE = 3
GAP_S = 1.0
ARM_S = 1.0
ARMING = "arming"  # observation label: HEALTHY, but before ARM_S; never counted


class Breaker:
    """Cell R's per-forward path decision and its C3 detector."""

    def __init__(self, gpu, gap_s: float = GAP_S, clock=time.monotonic_ns):
        self._gpu = gpu
        self._gap_ns = int(gap_s * 1e9)
        self._clock = clock
        self._lock = threading.Lock()
        self.state = IDLE
        self.count = 0
        self.episodes = 0
        self.trips: list[tuple] = []  # (trip_ns, episode, request_id, prepare_ms)
        self.observations: list[tuple] = []  # (t_ns, request_id, prepare_ms, state before, count after)
        self._last_short_ns: int | None = None
        self._episode_start_ns: int | None = None
        self._arm_ns = int(ARM_S * 1e9)

    def decide(self) -> tuple[str, str, int]:
        """(path, state after the decision, C3 count) for one ANE forward."""
        now = self._clock()
        gpu_active = self._gpu.active(now, self._gap_ns)
        with self._lock:
            paused = self._last_short_ns is None or now - self._last_short_ns > self._gap_ns
            self._last_short_ns = now
            if not gpu_active or paused:
                self.state, self.count = IDLE, 0
            if not gpu_active:
                return ASYNC, self.state, self.count
            if self.state == IDLE:
                self.state, self.count = HEALTHY, 0
                self.episodes += 1
                self._episode_start_ns = now
            if self.state == OPEN:
                return SYNC, self.state, self.count
            return ASYNC, self.state, self.count

    def observe(self, trace) -> None:
        """Feed one completed request (a RequestTrace). Only ANE requests in an armed HEALTHY episode count."""
        if trace.target != "ane":
            return
        now = self._clock()
        prep = trace.prepare_ms
        with self._lock:
            before = self.state
            if before == HEALTHY and now - self._episode_start_ns < self._arm_ns:
                before = ARMING
            if before == HEALTHY:
                self.count = self.count + 1 if prep > THRESHOLD_MS else 0
                if self.count >= CONSECUTIVE:
                    self.state = OPEN
                    self.trips.append((now, self.episodes, trace.request_id, prep))
            self.observations.append((now, trace.request_id, prep, before, self.count))
