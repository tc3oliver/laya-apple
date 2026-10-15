# Preregistration: W8 ANE L64 production gate on macOS 27

Written and committed before any macOS 27 W8 artifact is built or measured. Nothing here
changes after data is seen. A criterion that fails is reported as failed with its metric and
magnitude. A change of method after results is a new preregistration, not an edit to this one.

Prior work: [`research/ane-w8/`](../ane-w8/README.md) (issue #11). It measured `w8-pt` on
macOS 26.6.2, coremltools 9.0: L64 `laya-typed-decisions` W8/FP16 predict-P50 ratio 0.649,
parity 0.0170 max probability error, 0 hard mismatches, 2 near-tie flips, in 3 independent
processes. It did not test any serving path, concurrency, GPU co-load or the adaptive
breaker. That research is not edited.

## Question

Does the existing `w8-pt` configuration, on `laya-typed-decisions` L64 only, reduce
end-to-end ANE serving latency on macOS 27 while keeping the unchanged correctness gate and
the existing reliability behaviour? If the answer is not a clear yes, nothing is integrated.

## Scope

- Model/bucket: `laya-typed-decisions`, L64, B=1, `device="ane"`. Nothing else.
- Arms: **A = production FP16 L64**, built by the production pipeline
  (`laya-apple artifacts build`). **B = `w8-pt` L64**, built by
  `research/ane-w8/scripts/build_w8.py --config w8-pt` (palettizer config unchanged:
  kmeans, 8-bit, per tensor, `weight_threshold=65536`, FP16 LUT, FP16 compute and I/O).
- No other quantization setting is tried. No setting is tuned after seeing data.
- `laya_apple/` is not modified. `research/ane-w8/` is not modified. New code lives in
  `research/ane-w8-l64-gate/`. The serving harness injects the W8 compiled model into the
  existing ANE backend from the research side. That injection is not the production path and
  is stated as a limitation of every serving result.

## Environment and isolation

- Same machine, OS build, coremltools, MLX and Python for both arms; recorded per run with
  thermal state and power mode. Environment changes between runs invalidate the pairing.
- Artifacts and Core ML caches: `LAYA_APPLE_CACHE` and the research root point to a new
  directory under `~/Developer/scratch/w8-l64-gate/`. The active cache
  (`/Volumes/Data/cache/laya-apple`) and existing artifacts are never written. `HF_HUB_OFFLINE=1`.
- Nothing else uses the machine during a measurement: no pytest, no `uv` lock, no second
  benchmark. oMLX production is stopped with `bootout` only for measurement windows that need
  a quiet system or a GPU the harness owns, and is restored and verified afterwards.
- At the end the scratch directory and any Core ML e5rt cache growth it caused are removed
  after path verification.

## Metrics and their sources

Unavailable means unavailable: energy and any counter that cannot be read reliably are
recorded as `unavailable` with the reason. No estimate is substituted.

| Metric | Source |
|---|---|
| Parity: probability error, hard mismatches, near-tie flips, finite, repeat-identical | production `ane_parity` and goldens, unchanged tolerance (`laya_apple/parity/__init__.py`) |
| Placement | production `compute_plan_summary` / `check_ane_placement`; runtime `probe_placement` (`PROBE_MAX_RATIO` unchanged) |
| Forward latency | `research/ane-w8/scripts/latency.py` protocol (20 warm-up, 10 cycles x 100-forward windows, AB/BA alternating, per-pair B/A ratio, geometric mean, 95% t-interval) |
| Cold / warm | load time and first-call latency per fresh process; warm = after the 20 warm-up forwards |
| Percentiles | P50, P95, P99 from raw per-request samples |
| Serving | end-to-end latency per request at the client, throughput, error rate |
| GPU LLM | tokens/s and TTFT of a local MLX LLM generating on the GPU while ANE requests run |
| Resources | process RSS and, where readable without privileged tools, CPU time. GPU/ANE utilisation counters needing `sudo powermetrics` are `unavailable` |

## Design

- Interleaved A/B. Within a run, arm order alternates per cycle (AB, BA). Fixed seeds
  recorded in each raw file. Same request sequence for both arms in every pair.
- Independent replication: at least 3 fresh processes per measurement, a different process
  for each replicate, never one long process counted as three.
- Raw per-request or per-window samples are committed (kept under 1 MiB per file;
  larger raw data would go external with a manifest, per `AGENTS.md`).
- Intervals: 95% t-interval over replicate ratios; bootstrap as sensitivity only.

### Phase 2: macOS 27 re-validation

Build both arms; run the unchanged parity gate, compute-plan and runtime placement probe;
repeat-inference consistency; cold/warm latency; forward-latency protocol above with 3
fresh-process replicates.

### Phase 3: serving, screen first

Screen (about 5 to 10 minutes): single-process, alternating windows of A and B for
(S1) L64 short request alone. Stop rule below. If the screen shows an effect, run the full set,
each with 3 replicates, interleaved:

- S1 single L64 short request, concurrency 1.
- S2 mixed load: 80% L64-eligible short requests, 20% longer or multi-question requests that
  route elsewhere, at the production routing defaults.
- S3 heterogeneous: S2 while a local MLX LLM generates continuously on the GPU. The LLM is
  loaded by the harness from the local model cache. oMLX production is not used as the load.
- S4 concurrency sweep: 1, 2, 4, 8 clients on S2.
- S5 interference: S3 with LLM tokens/s and TTFT compared between the arm A and arm B runs,
  and against an LLM-only baseline.
- S6 adaptive/fallback: with B injected, run the existing breaker/handoff scenarios
  (quiet, GPU busy, ANE unavailable, `device="ane"` strict failure) and compare behaviour with A.
  The C3 threshold and `PROBE_MAX_RATIO` are not changed; if B trips or fails to trip
  differently from A, that is a recorded finding.

## Gate: every condition is required

All thresholds are fixed here. B is the candidate, A the baseline.

1. **Correctness.** The production gate passes unchanged: probability error <= 0.02, 0 hard
   mismatches, all outputs finite, repeat-identical. Near-tie flips are listed, not hidden.
2. **Placement.** Compute plan 100% ANE, 0 device transitions, runtime probe PASS.
3. **Forward reproduction.** In each of 3 fresh processes the B/A predict-P50 ratio is
   <= 0.80, and the pooled geometric-mean 95% interval has an upper bound < 0.85.
4. **End-to-end benefit.** In S1 or S2 (the L64 workloads), B has an end-to-end P50 or
   throughput improvement of at least 5% over A, with a 95% interval that excludes zero, and
   the direction holds in at least 2 of 3 replicates. A forward-only improvement does not count.
5. **No regression.** Across S1 to S4, B vs A: end-to-end P99 no worse than +5%; throughput no
   worse than -3%; error rate not higher (any new error is a failure); hard mismatches 0.
   For S3/S5: GPU LLM tokens/s no worse than -3% and TTFT P95 no worse than +5% against the
   arm A run. Each bound is checked at every tested concurrency and a miss anywhere fails.
6. **Reliability.** S6 shows B does not break adaptive execution, fallback or strict-ANE
   behaviour compared with A; the fast suite and the `ane`-marked tests pass unchanged on
   this branch; FP16 artifacts and defaults are untouched (no `laya_apple/` diff).
7. **Real use.** A recorded corpus of real client requests, at least 200 requests, shows that
   at least 25% of `laya-typed-decisions` ANE-eligible requests fall in the L64 bucket
   (token length <= 64 after tokenization, within the shipped routing limits). Synthetic
   benchmark lengths do not count. If no such corpus exists on this machine, condition 7
   is `unproven` and counts as FAIL; the length distribution is not estimated.

Any failed condition gives **Research Only / No Ship**. No tolerance, threshold or condition
is relaxed to obtain a pass.

## Stop rules

- Phase 2 parity, placement or probe fails: stop. Record the failure. Run no serving benchmark.
- Phase 2 forward ratio fails condition 3 in the first replicate: run the remaining replicates
  once to record it, then stop. No serving benchmark.
- Phase 3 screen shows end-to-end B vs A within +/-2% of A at P50 (no effect), or a P99 worse
  than +5%: stop after the screen. Run no full set.
- A run with a thermal-state change, a competing process, an environment mismatch or an
  A/B asymmetry is discarded and repeated once with the reason recorded. A second failure
  stops that step.
- A single run is never described as a stable improvement. A claim needs the 3 replicates.

## Deliverables

- `research/ane-w8-l64-gate/README.md`: methods, results tables, per-condition PASS/FAIL, limits.
- `research/ane-w8-l64-gate/raw/`: raw samples and build/parity/placement records.
- A research PR (not merged by the agent). A production PR is opened only if all 7 conditions
  pass, as a separate PR using the existing artifact lifecycle, W8 opt-in, manifest records of
  the quantization format, fail-closed on unvalidated artifacts, and no change to routing
  defaults, public API, Goldens, tolerances or release gates.
