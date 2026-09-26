# Phase 1: recovery experiment — preregistration

Issue #104, draft PR #105. This file, the breaker and the analysis are committed and pushed before
the first Phase 1 run. Nothing below changes once that run has started. After the data exists, a
change can only be an addendum, and an addendum never changes this phase's outcome.

## The question

When PB-ASYNC is already in a sustained slow state, and the frozen C3 detector trips and routes
every new ANE request to production A, does user-visible behaviour return to A-like within 1 s?

**What Phase 1 does not re-test:**
- C3's recall, which Phase 0 answered in [`replay.md`](replay.md);
- why the slow state happens: no scheduler, P-core, QoS or placement question.

**The feedback loop:**

| part | what | status |
|---|---|---|
| sensor | `RequestTrace.prepare_ms` | Phase 0, frozen |
| detector | C3 | Phase 0, frozen |
| controller | the breaker | tested here |
| actuator | PB-ASYNC → production A | tested here |
| recovery | A-like within 1.0 s of the trip | tested here |

**Framing.** Related systems work motivates the design principle: static heterogeneous execution
can fail under runtime interference, so lightweight online feedback with a safe fallback is a
reasonable direction. That work is motivation and precedent only. No README, doc or paper text
from this track claims that it proves C3 or this breaker correct. The evidence for 1.5 comes only
from this track's own data: the Phase 0 replay, this experiment, and the typed, product-mix and
soak validations that would follow.

## Frozen detector and breaker

[`scripts/breaker.py`](scripts/breaker.py) is the only policy code of cell R.

**C3:**
- Trips on 3 consecutive ANE requests with `prepare_ms > 0.3 ms`. The threshold is strict:
  0.3 ms itself is not host-slow.
- `prepare_ms = prepared_ns − submit_ns`, from the RequestTrace. It is fed through the public
  `trace=` callback: one call per completed request, on the ANE dispatcher thread, before the
  request's Future resolves.
- Requests are counted in completion order, which is also submit order for the serial ANE
  dispatcher. GPU requests are ignored.
- **Arming:** C3 counts only requests that complete at least **1.0 s after the episode start**
  (the first HEALTHY decision). Earlier observations are logged as `arming` and change nothing.
  - **Why:** every hetero start has a short transient with prepare > 0.3 ms in every cell,
    production A included. The first requests of #102's A windows had prepare 1.19, 0.43, 0.36,
    0.53 ms, and so on.
  - In #102 and #103's raw data, C3 counting from t0 trips within 1 s in 40 of 40 A windows.
  - Counting from t0 + 1 s, it trips in A 0/40, B 4/4, H64 0/26 and P 1/14. The P trip is #103's
    failing P r5 w5.
  - Phase 0 validated C3 only after this transient: the H and P replays start at the handoff, and
    the A reference starts at t0 + 1 s.
  - Without arming, R would trip on the transient in every episode, and Phase 1 would not test
    recovery from an established slow state.
  - The price is known and reported: the breaker does not protect the first 1.0 s of an episode.
  - Decided with the user on 2026-09-26, before any Phase 1 run. The rule, 3 consecutive
    > 0.3 ms, is unchanged.
- **Reset:**
  - to 0 on every ANE request with `prepare_ms ≤ 0.3 ms`;
  - to 0 at every episode start;
  - no count outside an armed HEALTHY episode. Observations in IDLE, arming or OPEN change
    nothing.
- One counter, one threshold. Not compared or tuned again: no C2, EWMA, 3-of-5 or other
  threshold.

**The breaker states.**

| state | when | ANE forward path |
|---|---|---|
| IDLE | outside a hetero episode | async, as cell B |
| HEALTHY | from the first ANE forward while the GPU is active | async; C3 counting from 1.0 s |
| OPEN | from a C3 trip to the end of the episode | production A |

- **The trip itself:**
  - it sets OPEN under the same lock, so detector trip = breaker open (one timestamp);
  - it records `(trip_ns, episode, request_id, prepare_ms)`.
- **In OPEN:**
  - no async retry, no half-open state, no cooldown retry within the episode;
  - the count stops.
- **Episode end and re-arm:**
  - the GPU is not active, or no ANE forward came for more than 1.0 s;
  - these are `handoff.GpuActivity` and the 1.0 s gap rule from #102, unchanged;
  - the state becomes IDLE and the count 0. The next ANE forward while the GPU is active starts a
    new HEALTHY episode.

**Actuation semantics:**
- **A request's path is chosen once,** by `decide()` at the start of its ANE forward, on the ANE
  dispatcher thread.
- **Request K** is the 3rd consecutive bad prepare. It finishes on the path its forward already
  took, PB-ASYNC. C3 sees K only when K completes.
- **K+1 onward:** every ANE request whose path is chosen after OPEN runs production A.
- **In flight:**
  - an ANE request whose path was chosen before OPEN is never cancelled, re-sent or replayed;
  - with the serial ANE dispatcher and the closed-loop short client this is at most request K
    itself;
  - the analysis counts async forwards decided after the trip, which must be 0.
- **Recorded:**
  - the detector trip timestamp, equal to the breaker-open timestamp;
  - the last PB-ASYNC request id and the first production-A request id, mapped from each
    decision to the one ANE forward whose service interval contains it;
  - trip → first A in ms.

## Cells, harness, runtime

**All cells** run #102's staged-handoff harness, `research/coreml-staged-handoff/scripts/run_config.py`,
loaded unchanged by [`scripts/run_config.py`](scripts/run_config.py).
- That is #94's run: laya at L128 / L512, #92's four conditions, 2 cycles of 20 s windows (2
  hetero windows per run), one fresh process per run.
- Both Core ML paths are loaded and warm.

**The cells:**
- **A:** always the production coremltools path.
- **B:** always PB-ASYNC, with no breaker. It confirms the slow-state phenotype still reproduces.
  C3 is applied offline, armed the same way, and reported only.
- **R:** B plus the frozen breaker. It is B until C3 trips.

**Runtime:** `laya_apple` tree: `1993c972654520e0ce5c046febbc5ab3eab6a20c`, the same as `main`.
- Every run records HEAD, the `laya_apple` tree, and whether any code the harness executes is
  dirty: `laya_apple/`, this track's `scripts/`, the harness scripts of #102, #94, #92, #77 and
  #83, `benchmarks/ane-process-isolation/` and `scripts/`.
- A different tree, or dirty paths, makes the run INVALID.

**Run order:** fixed, 12 runs, from `phase1_analyze.py runs`.

```
R1 A1 R2 B1 | R3 B2 R4 A2 | R5 A3 R6 B3
```

That gives R 12 episodes, A 6 and B 6.

**Runner:** [`scripts/run_phase1.sh`](scripts/run_phase1.sh).
- The `machine_snapshot.py` from #103 runs before and after every run.
- **Crash rule:**
  - A / B: re-run once in place; a second failure makes the phase INCONCLUSIVE;
  - R: any crash is FAIL and stops the phase;
  - this applies to `-b` re-runs too;
  - once the runner stops, the slots not run are not pending: the outcome is FAIL or
    INCONCLUSIVE from what stopped it.
- **Machine rule:** a run failing the qualification's machine item (qualification.md item 5) is
  re-run once as `<run>-b` at the end. A second machine failure makes the phase INCONCLUSIVE.

**Environment:** the same discipline as #103's qualification and evaluation.
- oMLX stopped, on AC power.
- OrbStack and the Aerial wallpaper left as they are.
- No other agent, build, test, benchmark or LLM workload during a run.
- **No stimulus.** No manual OrbStack load, CPU burner, forced E-core, QoS change or scheduler
  trick.
- If the preregistered runs do not produce enough natural slow state, the phase is INCONCLUSIVE.
  The workload is never changed to get a PASS or a FAIL.

## Signals: two separate families

- **Detector:** `prepare_ms` only.
- **Ground truth, onset and recovery:** user-visible latency (short e2e = response − submit) and
  completion cadence.
- **`prepare_ms` in recovery:**
  - it enters only as one extra *necessary* condition: at most 2 host-slow requests per window;
  - it can only delay a recovery, never establish one;
  - "prepare recovered, therefore the product recovered" is never an argument here;
  - it is also reported as a diagnostic.

## Definitions (every window measured from its t0, the hetero window start)

**m_A:** the pooled median e2e of the short requests in Phase 1's A hetero windows, over
[t0 + 1 s, end). A slow latency is > 1.2 × m_A.

**Sustained slow state:** Phase 0's ground truth, unchanged.
- Take 1 s spans from the episode start: t0 for B and R, t0 + 1 s for A.
- A span is slow when its short median e2e is > 1.2 × m_A.
- The episode is sustained when 2 or more consecutive spans are slow.
- **Onset** is the start of the first span of the first such run, moved back in 0.1 s steps while
  the 1 s window starting 0.1 s earlier is still slow, and never before the episode start.
- This counts slow-state episodes in A, B and R, and gives B's onset.

**The A-like envelope:**
- rolling windows of **W = 25 consecutive short requests**, sliding by one request, over every
  Phase 1 A hetero window from t0 + 1 s;
- per window: median e2e, P95 e2e (numpy linear), span (submit of the 25th − submit of the 1st:
  the short completion cadence), and the host-slow count;
- envelope = the **P99.9** of each measure, pooled over all Phase 1 A windows.

**A-like window:** all four hold.
1. median ≤ the envelope median;
2. P95 ≤ the envelope P95;
3. span ≤ the envelope span;
4. at most 2 host-slow requests.

**Onset in R:**
- Phase 0's 1 s spans cannot see a slow state that the breaker cuts short within about a second.
- So R's onset uses the recovery's latency resolution. Take the first rolling window that starts
  at or after the arm time (the episode start + 1.0 s) and at or before the trip, and whose median
  or P95 is outside the envelope.
- The onset is the submit time of that window's first request whose e2e is above the envelope
  P95. Prepare is not read.
- None means no user-visible slowness between the arm time and the trip.
- Searching from the arm time keeps the start transient out. It also means a slow state that
  begins at t0 is measured from the arm time. Its blind first second is the arming cost above, and
  is reported as such.

**R timing, per tripped episode:**
- slow onset → detector trip;
- detector trip → first A: the first sync decision after the trip;
- first A → A-like recovery;
- detector trip → sustained A-like recovery. This is the primary metric.
- Also recorded: the first bad prepare, the first of the 3 C3 requests.

**Sustained A-like recovery point:**
- Windows start at the first A request.
- The recovery point is the first window start t such that every window starting in
  [t, t + 1.0 s) is A-like, and the episode's data reaches t + 1.0 s. That is about 67 windows at
  the A rate.
- **None** means no such point before the episode ends.
- Beyond that second, the rest of the episode is covered by the persistence check below.

**Confirmed slow at the trip:** the median e2e of the short requests submitted in
[max(t0, trip − 1 s), the first A request's submit) is > 1.2 × m_A. That is the user-visible
second before the fallback.

**Suspected false trip:** a trip with no R onset and not confirmed slow. Its product cost is
reported: GPU-return P50 / P99, aggregate req/s and short P99 after the fallback, against A and B.
- It is recorded, not failed. A false positive only gives up PB-ASYNC for the rest of the episode
  and runs the known-safe A.
- Its cost is judged later, in production validation.

**After the fallback:** [first A request's submit + 1 s, end) of a tripped R episode.
- **Aggregate req/s:** all auto-instance completions over the span.
- **Short P99.**
- **GPU-return P50 / P99.** GPU return may revert to A's roughly 4.3 ms; that is allowed.
- **Persistence:**
  - 2 or more consecutive slow 1 s spans (#103's H6), or
  - 2 or more consecutive full 0.5 s bins with a host-slow share ≥ 0.10 (#103's H5),
  - on a grid from first A + 1 s.

**Correctness, every run:**
- 0 mismatches in every window;
- routing per condition: solo_short → ANE, solo_long → GPU, hetero short → ANE and long → GPU,
  gpu_only → GPU;
- **no request loss:** in every hetero window, the client count per stream equals the trace count
  per target.

## Outcome

**FAIL (hard: any R run; it stops the adaptive route).**
- R crash, mismatch, routing failure or request loss.
- **Breaker corruption.** Trips, observations and decisions belong to a window by their request:
  an ANE request submitted in [t0, end). They are not assigned by their own timestamps. A decision
  is mapped to the one ANE forward whose service interval contains it.
  - more than one trip in an episode;
  - a trip on a request outside every hetero window;
  - the breaker's episode count ≠ the number of HEALTHY runs in its own decision log;
  - an ANE request of a window without exactly one observation;
  - handoff errors;
  - an async decision after the trip in its episode (a retry);
  - a sync decision before the trip, or without one;
  - an online C3 count that breaks its own rule: armed HEALTHY observations move it, others
    leave it unchanged;
  - an observation whose prepare differs from the trace;
  - an online trip on a different request than C3 applied offline to the episode's trace. This is
    the race or duplicate-trip check.

**FAIL (recovery; only when the validity items below hold, otherwise INCONCLUSIVE).**
- Any tripped R episode with trip → sustained A-like recovery > 1.0 s, or none.
- Any tripped R episode where the slow state persists after the fallback: H5 or H6 from
  first A + 1 s.
- Any R episode with a sustained slow state (Phase 0 truth over [t0, end)) and no trip: a miss.
- Any tripped R episode with an R onset and onset → trip > 250 ms.
- The breaker re-arming inside a hetero window. An ANE forward gap > 1 s would mean the episode
  ran async again.
- **A trip on a window's last request** has no request after it. It is recorded, with no
  recovery measured, and is not a failure.
- **After the fallback,** pooled over tripped R episodes:
  - aggregate req/s < 0.95 × A's pooled req/s over [t0 + 1 s, end);
  - or the median tripped-episode P99 > allowed(the median A episode P99), where
    allowed(x) = max(1.05x, x + 1 ms).

**INCONCLUSIVE:**
- **Validity (A side, harness, machine):**
  - any run with the wrong runtime tree or dirty paths, or a protocol deviation;
  - an A or B crash twice, or an A / B mismatch, routing failure or request loss;
  - a second machine failure;
  - more than 1 of the 6 A episodes that is sustained slow, or has no A-like point within 1.0 s of
    t0 + 1 s. That would be A's own check against its own envelope.
- **B phenotype not reproduced:** fewer than 3 of the 6 B episodes sustained.
- **Not enough natural slow state:** fewer than 6 tripped R episodes, or fewer than 4 confirmed
  slow at the trip.

**PASS:** no FAIL, valid, and enough slow state. Then, and only then, the production breaker is
implemented (#104's plan, items 8 onward).

**On FAIL:**
- the PB-ASYNC production route closes;
- 1.4 stays;
- no other detector, cooldown, half-open, C2, threshold, guard, QoS or scheduler variant is tried;
- a native runtime is a future version's question.

## Report

Analysis: [`scripts/phase1_analyze.py`](scripts/phase1_analyze.py), which writes `phase1_tables.md` and
`phase1_results.json`. It was written and unit-tested (`tests/unit/test_breaker_phase1.py`) before any
Phase 1 run. An independent review checked it against this file before the freeze.

**The report lists:**
- the outcome;
- **C3:** the sustained episodes detected; onset → trip median / P95 / worst, from the arm time;
  false trips; B's offline armed trips;
- **recovery:**
  - trip → first A;
  - first A → A-like and trip → A-like: median / P95 / worst;
  - the worst episode;
  - for suspected false trips: GPU-return P50 / P99, req/s and P99 after the fallback, against
    the medians of A and of healthy B;
  - throughput, latency and correctness after the fallback;
- **A, B, R:** slow-state counts; R's trips; R's recoveries within 1 s.

The harness smoke run that checked the wiring (5 s windows, one cycle) went to a scratch directory.
It is not Phase 1 data. It ran the breaker before arming was added, with oMLX running.
