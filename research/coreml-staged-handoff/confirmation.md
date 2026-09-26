# H64 confirmation: preregistration

Issue #101, draft PR #102. This is a separate confirmation study, committed and pushed before any
of its runs.

**It changes nothing in [`criteria.md`](criteria.md).**
- The staged-handoff screen's outcome stays RESEARCH-CLOSED / NO PRODUCT CHANGE.
- H64 r3 cycle 0 (12.15150 > 12.15123 ms) stays a FAIL.

This study does not re-score that result. It asks a new question, with a new design fixed here
before its data exists. Nothing below changes once the first confirmation run has started.

## Question

Does H64 reproducibly avoid the #96 / #99 post-handoff slow state, while providing user-visible
latency that is materially non-inferior to production A?

## Cells and harness

- **Cells:** only A and H64, exactly as in the screen: `scripts/run_config.py --cell A|H64`, with
  the same load, the same instrumentation and the same state machine (`scripts/handoff.py`, guard
  64, gap 1.0 s).
- **Not tested:** H32, any other N, QoS, a warm-up, a probe, a wall-clock guard, and any other
  scheduler change.
- **Protocol:** laya, L128 / L512, #92's full mix, 2 cycles of 20 s windows. That gives two
  transitions per run: cycle 0 after solo_long, cycle 1 after gpu_only.
- **Environment:** one fresh process per run. The machine is idle and on AC power, with oMLX / the
  local LLM stopped. No other agent, build, test, benchmark or background job runs while the
  campaign runs.
- Raw output goes to `raw-conf/`.

**The order is fixed: 16 runs in 4 blocks.**

| block | runs |
|---|---|
| 1 | A r1, H64 r1, H64 r2, A r2 |
| 2 | H64 r3, A r3, A r4, H64 r4 |
| 3 | A r5, H64 r5, H64 r6, A r6 |
| 4 | H64 r7, A r7, A r8, H64 r8 |

**Pairing.** Each H64 run is paired with the A run next to it in its block:

| block | pairs |
|---|---|
| 1 | H64 r1 ↔ A r1, H64 r2 ↔ A r2 |
| 2 | H64 r3 ↔ A r3, H64 r4 ↔ A r4 |
| 3 | H64 r5 ↔ A r5, H64 r6 ↔ A r6 |
| 4 | H64 r7 ↔ A r7, H64 r8 ↔ A r8 |

Transitions are paired by predecessor: cycle 0 ↔ cycle 0 (after solo_long), and cycle 1 ↔ cycle 1
(after gpu_only). That gives 8 run pairs and 16 transition pairs.

## Crash and validity rules

- **An H64 run that exits non-zero** keeps its log and is re-run once in place. The H64 crash still
  counts as a correctness failure, because a crash is never excused.
- **An A run that exits non-zero** is re-run once in place. A second failure makes the study
  INVALID.
- **A validity** is checked on every hetero window, with #94's guard, as in the screen:
  - short P99 < 13 ms;
  - transient ≤ 1 s;
  - steady host-slow < 0.10;
  - aggregate in [109.9, 135.0] req/s;
  - 0 mismatches and 0 routing failures.
- **An A run that fails validity** is re-run once, as `A rN-b`, right after the campaign's last
  run. The re-run replaces it in its pair. If the re-run also fails, the study is INVALID.
- **Protocol.** Any run with a wrong protocol (cycles, window length, guard, handoff not installed)
  makes the study INVALID.
- INVALID is used only for these machine, harness and run-validity cases.

## Measures (per transition; t0 = hetero window start, end = window end)

The screen's `analyze.py` definitions are reused unchanged:
- **t_h:** the first async decision of the episode;
- **structure:** exactly one episode in the window, and exactly 64 sync forwards before t_h;
- **window P99:** part_a's short `p99_ms`;
- **onset P99:** over [t0, t0 + 4 s);
- **native mean:** async forwards in [t_h, end); A's predict mean is over the whole window;
- **transient from t_h:** #94's transient, with origin t_h;
- **steady host-slow:** over [t0 + 10, t0 + 20);
- **GPU return:** GPU requests received in [t_h + 1 s, end);
- **aggregate req/s.**

**Post-handoff E residency (new, from the research counters).**
- **Bins:** the 0.5 s bins [t_h + 0.5k, t_h + 0.5(k+1)) up to the window end.
- **Groups:** as in the screen's mechanism table, the parent's active threads and the auto GPU
  worker, each CPU-weighted. Active means ≥ 20 ms of CPU in [t0, t0 + 4 s).
- **E-resident bin:** either group has ≥ 2 ms of CPU in the bin and an E share ≥ 0.5.
- **Long E residency:** 2 or more consecutive E-resident bins (≥ 1.0 s).
- The counters are research-only, and production never reads them.

## Hard safety gates (every one of the 16 H64 transitions must pass all)

1. **Correctness.**
   - 0 mismatches and 0 routing failures, in every window of the run;
   - no crash;
   - the structure is valid, with exactly 64 sync forwards before t_h.
2. **No #96 / #99 state.** No long E residency after t_h.
3. **Host slow.**
   - transient from t_h ≤ 1.0 s;
   - steady host-slow share < 0.10.
4. **GPU isolation.**
   - GPU-return P50 ≤ 1.0 ms and P99 ≤ 1.0 ms. P95 is reported.
   - The P99 gate is supported by the screen: the 19 H transitions without post-handoff residency
     had P99 0.11–0.28 ms, and H32 r3 cycle 1's residency had 1.76 ms.
5. **Throughput.** The hetero window's aggregate req/s must be ≥ 0.95 × its paired A transition's.
6. **Core ML.**
   - native mean ≤ 1.05 × the paired A transition's predict mean;
   - and ≤ that mean + 0.3 ms.

## User-visible latency

For each of the 16 transition pairs, delta_p99 = H64 window P99 − the paired A window P99 (ms).

- **Population gate, both must hold:**
  - the median of the 16 deltas is ≤ +0.5 ms;
  - the one-sided 95% bootstrap upper bound of that median is ≤ +1.0 ms.
- **Outlier guard:** any delta_p99 > +2.0 ms fails.

**Bootstrap, fixed here.**
- **What is resampled:** the 8 run pairs, with replacement. Each drawn pair contributes both of its
  transition deltas, so the statistic is always the median of 16 values.
- **Why run pairs, not transitions:** the two transitions of a run share a process and a load, so
  they are not independent.
- **Settings:** B = 20,000 resamples, `numpy.random.default_rng(20260926)`, percentile method.
- **Upper bound:** the 95th percentile of the resampled medians.
- **Reported alongside, not gates:** the transition-level bootstrap bound, and the same deltas for
  onset P99, P95, median latency and req/s.

## Outcome

- **CONFIRMED** only if all of the following hold:
  - all 16 H64 transitions pass every hard safety gate;
  - there is no long E residency;
  - the population gate passes;
  - the outlier guard passes;
  - there are 0 correctness issues.

  The result then reads: "H64 confirmed as the leading production candidate."
- **FAILED** if any of these occurs:
  - a #96 / #99-type E residency;
  - a correctness failure;
  - a persistent host-slow state;
  - GPU isolation fails;
  - throughput fails;
  - the latency confirmation fails.

  The H64 route closes, production stays A, and no other N is tried.
- **INVALID:** only as defined under the crash and validity rules.

**After CONFIRMED.** The production phases follow on the same PR, each only if the previous one
passed: Phase 4, then 5, 5b, 6 and 7 (addenda 2 and 3).
- The guard becomes 64.
- The soak runs at least 60 hetero episodes.
- The prototype (local `proto/staged-handoff`, d023cc0) is re-reviewed and implemented cleanly,
  never cherry-picked blind.

**After FAILED.** The result is added to the closure, and PR #102 merges as a research closure.

Analysis: [`scripts/confirm_analyze.py`](scripts/confirm_analyze.py), written and unit-tested
before any confirmation run.
