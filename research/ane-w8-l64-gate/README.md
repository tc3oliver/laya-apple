# W8 ANE L64 production gate on macOS 27: result

**Verdict: Research Only / No Ship.** Conditions 5 and 7 of the preregistered gate fail.
`research/ane-w8/` and `laya_apple/` are unchanged; nothing was integrated.

Protocol, thresholds and stop rules: [`PREREGISTRATION.md`](PREREGISTRATION.md), committed
before any measurement. No threshold, tolerance or condition was changed afterwards. Prior
work (macOS 26.6.2): [`../ane-w8/README.md`](../ane-w8/README.md).

## Setup

macOS 27.0 (26A428), Apple M4 Max, coremltools 9.0, mlx 0.32.2, torch 2.7.0, Python 3.12.14,
laya_apple 1.6.3, AC power, low-power mode off, no thermal warning before or after any run
(`raw/phase2/thermal_*.txt`, `raw/phase3/thermal_*.txt`). Model `laya-typed-decisions`
(revision f9ab0b228f0f), bucket L64, B=1.

- **A** = production FP16, built with `laya-apple artifacts build`.
- **B** = `w8-pt` from `research/ane-w8/scripts/build_w8.py` (kmeans, 8-bit, per tensor,
  FP16 LUT, FP16 compute and I/O).
- Isolated artifact cache under `~/Developer/scratch/w8-l64-gate/`. The active cache was not written.
- Every measurement is 3 fresh-process replicates, A/B interleaved (AB/BA by cycle), same
  request sequence for both arms, fixed seeds.
- **Limits that apply to every number below.**
  - The 95% intervals are over 3 replicates (t, df=2, critical value 4.3027), so they are wide.
  - System load was not zero: load average about 2 to 3 from OrbStack, other sessions and
    Spotlight. Pair ratios within a replicate were tight, but the preregistered "nothing else
    runs" was met only for our own processes.
  - B is injected from the research side (`scripts/`) into an unmodified
    `Laya`; this is not the production artifact path, which rejects non-FP16 artifacts today.
  - Client concurrency is thread-level over one inline `Laya` (lock-serialized), not parallel ANE use.
  - Energy and GPU/ANE utilisation counters are `unavailable` (need privileged tools). Nothing is estimated.

## Phase 2: parity, placement, forward latency

Raw: `raw/phase2/`. Scripts: `scripts/run_w8.py`, `screen_arm.py`, `phase2_ratios.py`.

| | A (FP16) | B (W8) |
|---|---|---|
| Max probability error (gate 0.02) | 0.0046 | 0.01696 |
| Hard mismatches (gate 0) | 0 | 0 |
| Near-tie flips (listed) | none | 2: `L64-s13` row 5, margin 0.0091; `lang-es` row 0, margin 0.0223 |
| Finite, repeat-identical | yes | yes |
| Compute plan | 10594 ops, all ANE, 0 transitions | same |
| Runtime probe, ratio to CPU_ONLY (limit 0.8) | 0.511 | 0.322 |
| Weights | 738.9 MB | 369.7 MB |
| Cold first call / warm P50 / P95 / P99, one process each | 17.9 ms / 7.96 / 8.07 / 8.12 ms | 15.0 ms / 5.18 / 5.27 / 5.32 ms |
| RSS, same process | 1994 MB | 1662 MB |

"Cold" is a fresh process whose on-device ANE compile was already warmed by the build; it is
not a first-ever compile.

Forward predict-P50, B/A, `latency.py` protocol, 3 fresh processes: 0.6498 [0.6443, 0.6555],
0.6500 [0.6491, 0.6509], 0.6541 [0.6518, 0.6565]; pooled 0.6513 [0.6453, 0.6573]. Features +
predict + tail: pooled 0.6560 [0.6506, 0.6614]. Baseline artifact hash verified identical in
all three. Max abs logit difference to A on the workload row: 0.0520.
The macOS 26.6.2 reference was 0.649, so the improvement reproduces on macOS 27.

## Phase 3: serving

Raw: `raw/phase3/{s1,mix,s3,s6,llm_only}/`. Scripts: `scripts/screen_s1*.py`, `serve_mix*.py`,
`llm_load.py`, `llm_analyze.py`, `s6_adaptive.py`.

**S1, single L64 short request** (`device="ane"`, concurrency 1, end-to-end `Laya.predict`):
P50 B/A 0.659 [0.655, 0.663] (A 8.2 ms, B 5.4 ms), P99 0.666 [0.663, 0.670], throughput 1.519x,
errors 0/0, same answer in every window.

**S2/S4, mixed load** (79% L64 single-question requests to ANE; the rest route to GPU;
`device="auto"`; latency includes lock queueing), B/A pooled, no LLM:

| clients | P50 | P99 | throughput |
|---|---|---|---|
| 1 | 0.658 | 0.900 [0.860, 0.941] | 1.294x |
| 2 | 0.661 | 0.996 [0.992, 1.000] | 1.288x |
| 4 | 0.789 | 0.939 [0.886, 0.994] | 1.287x |
| 8 | 0.770 | 0.901 [0.859, 0.946] | 1.289x |

Errors 0 in both arms. 0 answer-choice mismatches between A and B over 8000 requests per cell
(answer choice only; probabilities were not compared in the serving runs).

**S3/S5, mixed load with a continuous local MLX LLM on the GPU** (Qwen3.5-0.8B-MLX-4bit, greedy,
128 new tokens). LLM alone: 485 tokens/s, TTFT P50/P95 64.4/64.7 ms. During the mix:
A about 302 tokens/s, B about 288 to 291 tokens/s.

| clients | LLM tokens/s, B/A per replicate (gate >= 0.97) | LLM TTFT P95, B/A per replicate (gate <= 1.05) |
|---|---|---|
| 1 | 0.962, 0.953, 0.962 | 1.023, 1.012, 1.016 |
| 2 | 0.962, 0.961, 0.959 | **1.052**, 1.002, 1.017 |
| 4 | 0.960, 0.962, 0.961 | 0.988, 1.004, 1.017 |
| 8 | 0.955, 0.960, 0.962 | 1.009, 0.992, 1.002 |

Mix under LLM load, B/A: P50 0.667 / 0.674 / 0.808 / 0.806; P99 1.032 [0.866, 1.231] /
0.982 / 0.919 / 0.927; throughput 1.238 to 1.244x. The c=1 P99 point estimate is inside the
+5% bound but its interval is wide and includes values above it.

**S6, adaptive and fallback** (production `Laya`, `execution="workers"`, adaptive handoff).
A and B behaved identically in all 3 replicates: quiet = armed; GPU busy = safe_sync at
request 0, async_healthy at request 63; breaker trips 0 in both; injected async-predict failure
= request served on ANE and handoff disabled with the recorded reason; re-arm after GPU idle;
strict `device="ane"` serves L64. B's async model output equals B's sync output on the
production load check. Not covered: an "ANE unavailable" case (arm-independent), and any
real GPU-busy episode that would actually trip the C3 threshold (none occurred), so breaker
trip behaviour with W8 remains untested.

## Gate

| # | Condition | Result | Evidence |
|---|---|---|---|
| 1 | Unchanged parity gate | PASS | prob error 0.01696 <= 0.02, 0 hard mismatches, finite, repeat-identical; 2 near-tie flips listed |
| 2 | 100% ANE, 0 transitions, probe PASS | PASS | 10594/10594 ops, 0 transitions, probe 0.322 <= 0.8 |
| 3 | Forward reproduction | PASS | each replicate <= 0.80 (0.650 to 0.654), pooled upper 0.657 < 0.85 |
| 4 | End-to-end benefit | PASS | S1 P50 0.659 and S2 c=1 P50 0.658, intervals exclude 1, 3 of 3 replicates |
| 5 | No regression (P99, throughput, errors, GPU LLM) | **FAIL** | LLM tokens/s B/A < 0.97 in 12 of 12 replicate cells (about -4%, 0.953 to 0.962); TTFT P95 1.052 in one replicate. P99, throughput, errors and mismatches stay within bounds in S1 to S4 |
| 6 | Reliability | PASS, limited coverage | S6 A = B; fast suite 1496 passed; `-m ane` 67 passed, 1 skipped (standard cache); no `laya_apple/` diff. Coverage gaps as listed under S6 |
| 7 | Real short-request distribution | **FAIL (unproven)** | no recorded real request corpus exists on this machine; every candidate is synthetic or unrelated. No distribution is estimated |

Per the preregistration, one failed condition is enough: **No Ship**. The production PR is not opened.

## Reading the S5 failure

The ANE-side gain is real and reproduces: about 35% lower forward and end-to-end latency at
L64, half the weight size, no tail regression, same answers. The gate fails on the GPU LLM
side. The S3 load is closed-loop, so the faster arm B completes about 24% more requests per
second, and more of the non-ANE requests (3-question and L256) reach the GPU while the LLM is
generating. The offered load therefore differs between arms, and the data cannot tell "W8
disturbs the GPU" from "B does more GPU work". This is an interpretation that was not tested.
The preregistration did not specify an open-loop S5 and the criterion was not changed. An
open-loop, fixed-arrival-rate S5 would be a new small preregistered experiment. It cannot
change this verdict.

## Other notes

- The first full `pytest -m ane` run used the isolated cache, which holds only the L64 FP16
  artifact: 12 tests failed on `ArtifactMissingError` for the missing L96 and L128 artifacts.
  The same suite against the standard artifact cache passed (67 passed, 1 skipped). The branch has no change to
  `laya_apple/`, `tests/` or `data/`.
- `research/ane-w8/scripts/` write build records into `research/ane-w8/raw`. `scripts/run_w8.py`
  redirects that output to `raw/phase2/build` so the historical data is not touched.
- oMLX production was not running during this work (no LaunchAgent loaded, nothing on :8000);
  nothing was stopped or restored. The GPU LLM load used model files only, not the service.
- If a real request capture appears, condition 7 can be evaluated directly. The Phase 2 and
  Phase 3 results remain valid for the re-run, and the S5 question above is the remaining blocker.
