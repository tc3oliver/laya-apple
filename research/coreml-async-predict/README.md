# Core ML asynchronous prediction as the GPU-isolation execution path

**Status: run. Outcome: async does not solve the slow state; #89 resumes.**
- **How the outcome was reached.** The preregistration ([`criteria.md`](criteria.md), #91) was
  merged before Phase 0 of record and before any screen run.
- **Phase 0.** All 8 feasibility gates passed.
- **Round 1.** PB-SYNC was slow in 3 of 3 hetero windows and PB-ASYNC in 2 of 3, while A had 0 of
  3.
- **What follows.** By the preregistered rule, round 2 did not run, async is not claimed to fix
  anything, and the slow-state trigger screen (#89) is next.
- **The numbers:** [`tables.md`](tables.md), [`results.json`](results.json); raw data in
  [`raw/`](raw/).

## Results

### Phase 0 (`raw/phase0.json`, run after the merge)

- **Every gate passed:**
  - correctness: PB-SYNC and PB-ASYNC are bit-identical to coremltools on L64, L96 and L128, at
    model and `forward` level;
  - output backings are used;
  - 3,473 submits produced 3,473 callbacks, with 0 errors and 0 timeouts;
  - 1000 repeats per bucket grew RSS by at most 0.36 MB;
  - the callback runs on Core ML's GCD queue (`com.apple.root.default-qos.overcommit`), not on the
    calling thread;
  - the submit P50 is 0.009 ms;
  - a main-thread probe's P50 lateness during async predictions is 0.34 ms.
- **The calling thread:** 4 crossings per forward (1 send, 3 for the pool), 0 re-entries, and 1
  Python callback per forward.
- **One slow submit.** The maximum submit was 566 ms, against a P50 of 0.009 ms. It happened once
  and was not explained.
- **PB-ASYNC-STAMPED** measured native completion to Python callback entry at a P50 of
  0.009 ms. That is a semantics check, not performance evidence.

### Round 1 (hetero windows; slow if short P99 ≥ 13.0 ms)

| configuration | slow windows | short P99 per window, ms | aggregate req/s | GPU return P50 | mismatches |
|---|---|---|---|---|---|
| A | 0 of 3 | 12.91, 11.13, 11.01 | 122.1–122.7 | 4.30 ms | 0 |
| PB-SYNC | 3 of 3 | 16.87, 16.67, 16.77 | 102.1–113.9 | 0.24 ms | 0 |
| PB-ASYNC | 2 of 3 | 16.96, 12.77, 16.26 | 123.1–127.6 | 0.035 ms | 0 |

### Not gating: the two slow runs differ in host CPU

These records are observations. They do not change the outcome above.

- **PB-SYNC's slow windows** show R1's process-wide host-CPU inflation. Against A, PB-SYNC's
  pooled windows have:
  - an ANE features stage of 0.78 ms, against 0.16;
  - GPU thread CPU per forward of 7.7 ms, against 1.8;
  - client-short CPU of 889–1338 ms per window, against 284–296.
- **PB-ASYNC's slow windows show little of it.** Its pooled windows have:
  - a features stage of 0.16 ms, the same as A;
  - GPU thread CPU per forward of 2.4 ms, and 2.9 ms in its worst window;
  - client-short CPU of 327–480 ms per window.

  Its aggregate throughput is at or above A's, and its GPU return P50 is 0.035 ms.
- **What is and is not shown.** PB-ASYNC's short P99 crosses R1's 13 ms state split in 2 of 3
  windows. Whether that tail comes from the same mechanism as PB-SYNC's slow state is not tested
  here.

## Post-hoc: where the PB-ASYNC short tail occurs (existing data only, not a gate)

`scripts/tail_decomposition.py` splits every short request of #92's hetero windows into stages. It
uses the recorded request trace, the binding's stamps and the client's own latency. The outputs
are [`tail_decomposition.md`](tail_decomposition.md) and `tail_decomposition.json`.
- **This analysis does not change #92's outcome.** Async did not pass the 13 ms classifier, round 2
  did not run, and async is not claimed to solve anything.
- **The tail is a transient episode at the start of each hetero window.** It is neither a shift of
  the whole distribution nor scattered requests.
  - A request counts as "host-slow" when its client-thread `prepare` is above 0.3 ms (normally
    about 0.12 ms). By that measure, each of PB-ASYNC's windows opens with a host-slow episode of
    0.8–3.4 s and then stays normal for the rest of the window.
  - The two windows above 13 ms are the two with the longest episodes.
  - Outside the episodes, PB-ASYNC's client P99 is 10.4 ms, against A's 11.9 ms over all
    requests.
  - PB-ASYNC's slowest requests cluster in time: P(next of the slowest 10% | slowest 10%) is 0.94,
    against 0.12 for A and 0.02 for PB-SYNC.
- **The three cells differ in how long the host-slow state lasts:**

  | cell | host-slow state |
  |---|---|
  | A | at most about 0.3 s at each window's start |
  | PB-ASYNC | 0.8–3.4 s at the start |
  | PB-SYNC | almost the whole window: 100% of two windows, and 11 s of the third |

- **The episode has the same signature as PB-SYNC's slow state.** CPU time rises with wall time,
  so the work itself runs slower; it is not only waiting. In PB-ASYNC's slowest 10% of requests:
  - ANE dispatcher CPU per forward is 1.35 ms, against 0.27 ms;
  - client-thread `prepare` is 0.66 ms, against 0.12 ms.

  That is about 5× in every Python stage, close to PB-SYNC's whole-window levels (1.46 ms and
  0.59 ms).
- **Where the extra time goes.** Against the fastest 90%, the slowest 1% of PB-ASYNC requests add
  about 9 ms:

  | stage | added |
  |---|---|
  | features | +3.8 ms |
  | Core ML plus the callback taking the GIL | +2.9 ms |
  | client `prepare` | +0.8 ms |
  | action head | +0.3 ms |
  | queue, async send, callback → waiter and the response path | about 0.1 ms each |

  The slowest 1% also overlap more GPU completions during service: 0.86 on average, against 0.28.
- **What the recorded data cannot separate:**
  - native completion from the callback's GIL acquisition inside the "Core ML + callback" span
    (the stamping C block ran only in Phase 0);
  - which CPU cores or frequencies the threads ran on;
  - what starts the episode at hetero onset, and what ends it.

## Question

Can Core ML's official asynchronous prediction path provide GPU completion isolation without
triggering the host-side slow state seen with our manually GIL-released synchronous path?

## Why

- **R1** (`research/coreml-prebind-full-protocol/`) released the GIL ourselves around a synchronous
  Core ML prediction. That removed the GPU completion wait, but on laya it put hetero windows into
  a host-side slow state.
- **Core ML has an official asynchronous prediction API** that had not been tested:
  `-[MLModel predictionFromFeatures:options:completionHandler:]`, available from macOS 14.
- **#89**, the slow-state trigger screen, is paused before any run in favour of this question.

## Design

laya only, L128 / L512, with the GPU in a worker process, on R1's full protocol.

| configuration | ANE prediction call | role |
|---|---|---|
| A | coremltools, synchronous, GIL held | production, negative control |
| PB-SYNC | R1's PB: synchronous call, GIL released by our shim | positive control |
| PB-ASYNC | the same prebound objects, run through the async API with one completion handoff | candidate |

- **Phase 0** checks feasibility: correctness, output backings, callbacks, repeat stability,
  threading, and GIL behaviour.
- **The screen** is one run per configuration (A, PB-SYNC, PB-ASYNC). Round 1 is a screen only.
  A reversed replication round runs automatically only when round 1 shows the full strong pattern
  (A normal, PB-SYNC slow, PB-ASYNC normal, 0 mismatches, PB-ASYNC isolation). Strong causal
  evidence requires round 2 to show that pattern again. Otherwise the result is INCONCLUSIVE.
- **The classifier** is R1's: a window is slow if its short P99 is ≥ 13 ms. It classifies state
  only.
- **Maximum:** 6 runs, about 28 min, plus Phase 0.

## Run

```sh
# Phase 0 (feasibility gates), then round 1 (3 runs, ~14 min) and, only as criteria.md requires,
# round 2. Idle machine on AC power, local LLM server stopped. Resumable.
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-async-predict/scripts/run_all.sh
uv run python research/coreml-async-predict/scripts/analyze.py      # --check to verify
```

Unit tests: `tests/unit/test_async_predict.py`.
