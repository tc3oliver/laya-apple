# Criteria: Core ML asynchronous prediction as the GPU-isolation execution path

This file is written and committed before any screen run. Nothing in it changes after screen data
is seen. Phase 0 is a feasibility check with its own gates below. Harness smoke checks are not
data and are not committed. Issue #90.

## Purpose and question

**The 1.5 objective** is an execution model that can go into production and that combines:
- GPU completion isolation;
- no host-side slow state;
- no ANE latency or throughput regression.

**What is known so far:**
- **R1** (#86, #87, #88): releasing the GIL ourselves around a *synchronous* Core ML prediction
  removes the GPU completion wait. On laya, though, it puts hetero windows into a host-side slow
  state.
- **Core ML's official concurrent-prediction path had not been tested.**
- **The slow-state trigger screen (#89)** is paused before any run, in favour of this question:

> Can Core ML's official asynchronous prediction path provide GPU completion isolation without
> triggering the host-side slow state seen with our manually GIL-released synchronous path?

This is not a question about whether async is faster than sync.

## The API

**Declared** in `CoreML.framework/Headers/MLModel.h` (macOS SDK 27.0; this machine runs macOS
26.6.2):

```objc
- (void)predictionFromFeatures:(id<MLFeatureProvider>)input
                       options:(MLPredictionOptions *)options
             completionHandler:(void (^)(_Nullable id<MLFeatureProvider> output,
                                         NSError * _Nullable error))completionHandler
                 API_AVAILABLE(macos(14.0), ios(17.0), watchos(10.0), tvos(17.0))
                 NS_REFINED_FOR_SWIFT NS_SWIFT_DISABLE_ASYNC;
```

**Apple's own description.** The header documents it as "Run a prediction on a model
asynchronously". It is the Objective-C form of Swift's `prediction(from:options:) async`.
- **PyObjC exposure.** PyObjC 12.2.2 exposes it as
  `predictionFromFeatures_options_completionHandler_`, with the handler typed as a block (`@?`)
  returning void.
- **Scope.** `laya_apple` does not depend on PyObjC. The binding here is research only, added
  through `uv run --with pyobjc-framework-CoreML==12.2.2`.

## Configurations

The model is laya only, with L128 short and L512 long requests. The GPU runs in a worker process in
every configuration.

| | ANE prediction call | GIL | role |
|---|---|---|---|
| **A** | coremltools, synchronous | held for the call | production, negative control |
| **PB-SYNC** | R1's PB: `predictionFromFeatures:options:error:` through #83's shim, called with `ctypes.CDLL` | released by us for the call | positive control, which should show the slow state |
| **PB-ASYNC** | `predictionFromFeatures:options:completionHandler:`, sent through PyObjC | as for any PyObjC send; the calling thread then waits with the GIL released | candidate |

**PB-SYNC and PB-ASYNC share everything but the call.** They are the same `PrebindModel` objects,
built at load:
- the input MLMultiArrays and their NumPy views;
- the Foundation-only feature provider;
- the output MLMultiArrays, set as `MLPredictionOptions.outputBackings`, and their NumPy views;
- strong ownership of all of these, and the per-bucket non-blocking lock.

**The code-path difference, per forward:**

| step | PB-SYNC | PB-ASYNC |
|---|---|---|
| 1 | write the inputs (NumPy) | write the inputs (NumPy) |
| 2 | push an autorelease pool | push an autorelease pool |
| 3 | call the `ctypes.CDLL` shim. It releases the GIL and sends the synchronous `predictionFromFeatures:options:error:` on this thread | send `predictionFromFeatures:options:completionHandler:` through PyObjC. It returns once the prediction has been submitted |
| 4 | – | wait on a per-bucket event, with the GIL released and a 5 s timeout |
| 5 | – | Core ML calls the handler on its own queue. The handler takes the GIL once, stamps, records the thread and queue, and sets the event |
| 6 | pop the pool | pop the pool |
| 7 | read the outputs from the backed views (NumPy) | read the outputs from the backed views (NumPy) |

**The handler:**
- It is created once per bucket, at load, and kept strongly referenced.
- It does not read the output provider, because the backings carry the outputs.
- Exactly one callback per submit is required.
- A bucket's next submit happens only after the previous completion.

## Phase 0: feasibility (before the screen)

`scripts/phase0.py` writes `raw/phase0.json`. The screen runs only if every gate passes. If a gate
fails, the reason is recorded under "PB-ASYNC cannot be implemented safely" below.

| # | gate | passes if |
|---|---|---|
| 1 | callable | the API responds, and every submit completes within 5 s |
| 2 | correctness | PB-ASYNC's outputs are bit-identical to coremltools on every laya ANE bucket, at model and `ANEBackend.forward` level; PB-SYNC also |
| 3 | output backings | every output is `backed` under the async path, checked by #83's sentinel method |
| 4 | callbacks | exactly one callback per submit, 0 errors, and 0 duplicate or late callbacks across every Phase 0 call |
| 5 | repeat stability | 1000 forwards per bucket (L64, L96, L128) with varying inputs, and no hang, timeout or exception. Resident memory grows by at most 64 MB per bucket, and every 100th forward is bit-identical to coremltools |
| 6 | threading | the callback runs on a thread other than the calling thread |
| 7 | submit returns early | on L128, the P50 of submit_after − submit_before is below 1 ms, and below 0.2 × the P50 of callback_entry − submit_before |
| 8 | the GIL is free | while a background thread runs 200 async L128 predictions, a main-thread 1 ms sleep-grid probe has a P50 lateness below 1 ms |

**Also recorded in Phase 0, not gating:**
- callback_entry − submit_before, which is the native time plus the handoff into Python;
- wake − callback_entry, the waiter handoff;
- the callback threads and dispatch-queue labels;
- the crossings per forward on the calling thread, counted with #83's `crossings.py`;
- the Python entries on the callback thread (1 per forward expected), and any re-entries.

**PB-ASYNC-STAMPED, research only, Phase 0 only.** It wraps the handler in a C block that stamps
with #83's clock and then calls the Python block.
- **What it is for:** checking semantics only. That covers native completion, when the callback
  enters Python relative to it, and whether the callback thread must take the GIL.
- **What it is not:** its latencies are **not performance evidence** for PB-ASYNC. They are
  labelled that way in `phase0.json` and the tables.
- **The screen never uses it.**
- If it proves impractical, that is recorded.

**A harness-development run before this file was merged.** While the harness was being written,
`phase0.py` and three 2 s smokes were run once into a scratch directory outside the repository.
- This happened before this file's rules were final.
- Their outputs were not inspected by whoever interprets the results, and they are not used.
- Phase 0 is run again after this file is merged, and that run is the Phase 0 of record.

**Anything unexpected is recorded as observed, not explained away.** Examples: the callback
entering Python earlier than expected, hidden Python-backed proxies, or re-entries.

## The screen

**Protocol: R1's, unchanged.**
- It is #77's full #57 mix: `run_mix.py` running `bench_concurrency.py --part a`.
- The windows are solo_short, solo_long, hetero and gpu_only, over 3 cycles of 20 s, giving **3
  hetero windows per run**.
- There is one closed-loop client per stream. Each stream repeats `make_request(seed=0)` with one
  question, so every configuration sends identical requests.
- One fresh process per run.
- R1's records are kept identically in every configuration. PB-ASYNC adds its stamps, the
  callback threads and the CPU of the callback threads.
- **Not added:** PB-H, the hetero warm-up, the 1 ms probe, QoS, a Swift worker, or GPU pacing.

**Order.**
- Round 1 runs A, PB-SYNC, PB-ASYNC.
- Round 2 runs only under the trigger below, in the reverse order: PB-ASYNC, PB-SYNC, A.
- Over both rounds that is ABC CBA, so every configuration has the same mean position.

**The state classifier is R1's.**
- A hetero window is **slow** if its short-stream P99 is ≥ 13.0 ms, and **normal** otherwise.
- The 13 ms split is R1's observed gap between the two clusters. It classifies state only. It is
  not a production threshold.
- A run is **slow** if at least 2 of its 3 windows are slow, and **normal** if none is.

**GPU isolation is preserved** when both hold:
- PB-ASYNC's GPU-return P50, pooled over its hetero windows, is ≤ 1 ms;
- it is at least 5× better than A's.

**Round 1 is a screen only.** Round 1 alone never supports the conclusion that async solved the slow
state.

**The strong pattern.** Round 1 shows the strong pattern when all five of these hold:
1. A is normal (0 of 3 windows slow);
2. PB-SYNC is slow (at least 2 of 3 windows slow);
3. PB-ASYNC is normal (0 of 3 windows slow);
4. correctness holds: 0 mismatches in every window of all three runs;
5. PB-ASYNC's GPU completion isolation holds for its run: hetero GPU-return P50 ≤ 1 ms, and A's P50
   / PB-ASYNC's P50 ≥ 5.

**The replication trigger is fixed now, so no human decides it after seeing round 1.** Round 2
runs **automatically** only if round 1 shows the strong pattern. It uses the reverse order:
PB-ASYNC, PB-SYNC, A.

**After round 2.** Only round 2's own three runs are judged:
- **They show the strong pattern again** (all five conditions): this is **strong causal
  evidence**, and PB-ASYNC becomes the leading 1.5 candidate. It means:
  - manually releasing the GIL around a synchronous prediction is not the right production
    concurrency architecture;
  - official async Core ML execution is the path to productionize.

  Next comes productionizing the async prebound path, then the full 1.5 acceptance gate, each in
  its own PR.
- **Otherwise: INCONCLUSIVE.**
  - Async is not claimed to solve the slow state.
  - Nothing is productionized.
  - The next step is a human decision.

**If round 1 does not show the strong pattern, round 2 does not run.** The screen stops on the
first branch that applies:

| round 1 | outcome |
|---|---|
| any A window slow | INCONCLUSIVE: the negative control was slow, so the round is invalid for causal interpretation |
| PB-SYNC has at most 1 slow window | INCONCLUSIVE: the positive control was not reproduced. Async is not concluded to have fixed anything; the positive control / conditioning is repeated first, as a human decision |
| PB-SYNC slow and PB-ASYNC slow (at least 2 of 3 windows) | async does not solve the slow state; resume #89 (PB-H, PB-W, PB-P) |
| anything else | mixed: not interpreted, human decision. For example: any mismatch, PB-ASYNC isolation lost, or PB-ASYNC with 1 of 3 windows slow |

**No third round runs automatically.**

**PB-ASYNC cannot be implemented safely through PyObjC** if Phase 0 fails. The reason is recorded
as one of these, and #89 resumes:
- the API is unavailable;
- the callback or GIL semantics are unsuitable;
- a lifetime or buffer problem;
- unacceptable overhead.

**Time.**
- Phase 0: a few minutes.
- One screen run: about 4.7 min (R1).
- Round 1: 3 runs, about 14 min.
- Round 2: 3 runs, about 14 min.
- Maximum: 6 runs, about 28 min, plus Phase 0.

## Outputs (all non-gating), per configuration, run and hetero window

- **State:** the slow flag, and slow windows out of 3.
- **Throughput and GPU:**
  - short P99;
  - aggregate req/s;
  - GPU return P50;
  - mismatches.
- **ANE stages:**
  - features;
  - pre (entry to submit, or entry to the native call);
  - submit (PB-ASYNC);
  - completion into Python (PB-ASYNC: callback_entry − submit_before; PB-SYNC: the native call and
    the re-acquire);
  - the waiter handoff (PB-ASYNC);
  - post;
  - tail.
- **Host CPU:**
  - ANE thread CPU per forward;
  - GPU thread CPU per forward;
  - client-short and client-long CPU;
  - callback thread CPU and callback latency (PB-ASYNC).

**Machine:**
- idle, on AC power;
- the local LLM server and other GPU/ANE services stopped, and restored afterwards;
- the Core ML E5 cache not cleared.

**Versions:** as locked in `uv.lock`, plus pyobjc-framework-CoreML 12.2.2 through `uv run --with`.

**What this does not change:** production code, `laya_apple/data/placement.json`, or any release.
