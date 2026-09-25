# Core ML asynchronous prediction as the GPU-isolation execution path

**Status: preregistered, not run.** The criteria are in [`criteria.md`](criteria.md), committed
before any screen run. Issue #90.

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
