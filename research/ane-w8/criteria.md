# Criteria: W8-palettized ANE artifacts under the unchanged artifact gate

Issue: [#11](https://github.com/tc3oliver/laya-apple/issues/11). This file was written and
committed before any W8 artifact was built or measured. Nothing in it changes after data is
seen. A failing cell is reported as a failure, with its metric and magnitude, and is a
documented no-ship. Smoke checks made while the scripts were written are not data and are not
committed.

## Question

Can the shipped BC1S ANE artifacts be stored with 8-bit palettized weights and still pass the
unchanged artifact gate (`CONTRIBUTING.md`, "Artifact release policy"), per model and per
shipped bucket? What does that do to artifact size and to short-L ANE latency against the
shipped FP16 artifact?

## Background

- `CONTRIBUTING.md` already fixes the rule: a quantized export ships only if it passes the same
  gate as the shipped artifacts; a failing one stays research and is never registered, offered
  or routed to.
- `research/phase-0-feasibility/ane-long-context.md` ("Implications") lists W8 weight
  compression as a candidate for short-L ANE value, because projections and MLP dominate at
  L ≤ 128. It was not measured in this repository.
- The shipped artifacts keep every conv weight in FP16. The weight file is most of the artifact:
  about 707 MB per ModernBERT-large bucket and 240 MB per mmBERT-base bucket (the embedding
  table stays on the host and is not in the graph).

## Cells

The shipped explicit ANE buckets at 1.5.0 (`docs/support-matrix.md`), 10 cells:

| model | buckets | golden rows in the gate (tokens ≤ L) |
|---|---|---|
| `laya` | 64, 96, 128 | 41, 93, 122 |
| `laya-typed-decisions` | 64, 96, 128 | 41, 93, 122 |
| `laya-multilingual` | 64, 96, 128, 256 | 47, 97, 121, 146 |

The list is fixed here. It is not read from the registry, so a bucket or routing change made
elsewhere (for example 256/512 buckets) does not change this experiment.

## Compression

One primary configuration, one conditional secondary. Everything else in the build is the
production pipeline.

| id | coremltools 9.0 `OpPalettizerConfig` | runs on |
|---|---|---|
| **`w8-pt`** (primary) | `mode="kmeans", nbits=8, granularity="per_tensor"` | all 10 cells |
| `w8-gc32` (secondary) | `mode="kmeans", nbits=8, granularity="per_grouped_channel", group_size=32` | only cells where `w8-pt` **fails the parity gate** |

For both:
- applied with `palettize_weights(mlmodel, OptimizationConfig(op_type_configs={"conv": ...}))`
  to the converted FP16 ML program, before it is saved;
- `weight_threshold=65536`: every projection and MLP matrix (all ≥ 768 × 768) is palettized;
  no bias, LayerNorm weight, RoPE table or scorer output vector is (all ≤ 4,096 elements).
  The build record lists every conv weight that stayed dense;
- the LUT is FP16. Compute and I/O stay FP16. No per-channel scale, no joint compression,
  no activation quantization;
- k-means runs through coremltools' bundled exact 1-D k-means (`kmeans1d`), which is
  deterministic. The build refuses to run if it is unavailable, so it cannot silently fall
  back to scikit-learn's randomized KMeans.

Why the secondary runs only after a parity failure: a grouped LUT can only improve numerics
over a per-tensor LUT at the same bit width. It is a second chance on correctness, not a second
latency candidate. It does not run for a placement, probe or latency failure. Each
configuration keeps its own verdict; the secondary never replaces or hides the primary's.

## Build pipeline

`scripts/build_w8.py`, per cell, all production code unchanged except step 5:

1. pinned checkpoint, `verify_weights` against the pinned SHA-256;
2. PyTorch FP32 reference (`conversion.torch_reference.load_model`) and the production
   `conversion.bc1s.ConvBody`;
3. the production FP32 layout check (`conversion.build._layout_check`), < 1e-3;
4. trace and `ct.convert` with the production arguments (`FLOAT16` compute, FP16 I/O, B=1,
   fixed L, `macOS15` target);
5. **palettize** (above);
6. save, compile to `model.mlmodelc`, remove the `.mlpackage`.

The artifact stays under the research root `/Volumes/Data/cache/laya-apple-research/w8/`,
which the scripts refuse to place inside `LAYA_APPLE_CACHE`. It is never registered and the
runtime never loads it.

## The gate: four separate results per cell and configuration

None of these is loosened, re-implemented or re-counted.

1. **Placement.** `laya_apple.artifacts.compute_plan_summary` on `CPU_AND_NE`, then
   `check_ane_placement`: ANE ops > 0, 0 CPU ops, 0 GPU ops, 0 device transitions. The
   operators that dequantize the palettized weights are counted exactly as the production
   function counts them. If the check fails because of them, that is a FAIL of this
   experiment. Changing the check would need its own preregistered PR. The per-op-type plan
   is recorded as a diagnostic.
2. **Parity.** `laya_apple.parity.ane.ane_parity` at the cell's bucket, on this machine's
   Neural Engine, against the upstream PyTorch FP32 goldens. It applies the unchanged FP16
   gate of `laya_apple/parity/__init__.py`:
   - calibrated probability max |Δ| ≤ 0.02, action probability max |Δ| ≤ 0.02;
   - 0 hard mismatches, where a hard mismatch is a top-1 flip with reference margin ≥ 0.04;
   - near-tie flips (margin < 0.04) listed, not failed;
   - exact prompt tokens, all outputs finite, and a repeated call bitwise identical.

   A second pass with the same forward records every row's raw outputs. `analyze.py`
   recomputes the per-row table from them and must reproduce the summary of record.
3. **Placement probe.** `laya_apple.backends.coreml_ane.probe_placement` unchanged: the
   loaded `CPU_AND_NE` model's fastest-of-5 time on the runtime's pad probe input must be ≤ 0.8×
   the same artifact on `CPU_ONLY`.
4. **Latency** (gated, as non-regression). This is the paired gate below: W8 predict P50 must
   not exceed 1.05× the shipped FP16 artifact's.

**Size** is recorded, not gated: the `weight.bin` bytes and the whole `model.mlmodelc` bytes,
W8 against the shipped FP16 artifact of the same cell.

**Per cell and configuration:**
- **ship candidate**: all four results are PASS;
- **documented no-ship**: any of the four is FAIL, or the build fails at any step. The step and
  reason are recorded;
- **inconclusive**: none is FAIL but latency is INCONCLUSIVE. No ship claim follows. Deciding
  it would need a separately preregistered replication.

All four results are always reported side by side. They are never merged into one pass/fail
that hides a failing dimension.

A ship candidate is not shipped by this experiment. Promotion would be its own production PR:
- manifest and registry support for a W8 artifact, with tests;
- a rebuild through `laya-apple artifacts build`;
- the full gate on the tested profile;
- a CHANGELOG entry.

## Latency protocol and paired gate

`scripts/latency.py`, one process per run, in an exclusive hardware slot: oMLX stopped, no
other benchmark, build, test or package install running.

- **Arms.**
  - A: the shipped FP16 artifact, loaded through `load_verified` from `LAYA_APPLE_CACHE`.
  - B: the cell's W8 artifact, `CPU_AND_NE`.
- **Input.** One exact-length request, `make_request(tokenizer, config, L, n_questions=1,
  seed=0)`, as `laya-apple benchmark` builds it. The row must be exactly L tokens.
- **Warm-up.** 20 `predict` calls per arm, not recorded.
- **Windows.** 10 cycles. Each cycle has one 100-forward window per arm, ordered A B in even
  cycles and B A in odd cycles.
- **Per forward.** `predict_ms` (`model.predict` only, features prebuilt per forward outside the
  timer) and `forward_ms` (features + `predict` + host tail, the backend's boundary). Every
  sample is kept.
- **Pair.** Arm B's window and arm A's window of the same cycle, 10 pairs.
- **Statistic.** Per pair, B predict P50 ÷ A predict P50. The geometric mean of the 10 ratios,
  with a 95% Student t interval on the log ratios (df = 9).
- **Verdict (limit 1.05).**
  - PASS if the upper bound ≤ 1.05;
  - FAIL if the lower bound > 1.05;
  - INCONCLUSIVE otherwise.
- **Sensitivity only.** A seeded percentile bootstrap over pairs (10,000 resamples). It never
  changes the verdict.
- **Descriptive, not gated.** The `forward_ms` ratio with its interval; each arm's median window
  P50; load time of each arm; the max |Δlogit| between the arms on the workload row.
- **Wording.** "W8 is faster" is written only if the predict-ratio upper bound is < 1.00.
  Otherwise the result is "no measurable speed-up", even when the cell passes.

Numbers from this protocol are not compared with `benchmarks/v1.0.md` or with Phase −1
latency tables without saying that the methodology differs: those are pooled runs, not paired
windows.

## Run order and stop rules

1. Build every cell for `w8-pt`. This is CPU-heavy; the parity step uses the ANE.
2. Build `w8-gc32`, only for cells whose `w8-pt` parity result is FAIL.
3. Latency and probe for every cell and configuration whose build reached `done`, in the
   exclusive slot.
4. `analyze.py`.

Other rules:
- Each cell is run once.
- A run interrupted by a documented external event (crash, power, another process found running)
  is re-run once in full. The interrupted data is kept under `raw/aborted/`.
- No re-run because a result is unwelcome.
- No automatic extension: no more cycles, no other bit widths (4, 6), no other granularity, and no
  per-layer exclusions within this experiment. Any of those is a new preregistration.

## Predictions (written down to be checked, not criteria)

- Placement: PASS. The dequantizing operators are compile-time constants and should not appear
  as CPU work. This is the largest structural risk, and it is checked, not assumed.
- Parity with `w8-pt`: PASS on all cells, with probability error well under 0.02. 256 FP16
  centroids per matrix of this size usually cost less accuracy than FP16 compute already does.
- Size: `weight.bin` about 0.50–0.52× of FP16.
- Latency: within ±5% of FP16 at L64–L128. ANE weight traffic is not expected to bind at B=1.
  The prior report of about 2% on multilingual came from other code and is not evidence here.

## What does not change

- The parity tolerances, the placement check and the probe threshold.
- Routing, the registry and the shipped artifacts.
- `laya_apple/` is not modified. The scripts import it; it never imports them.

## Known limitations, stated in advance

- One machine and profile: M4 Max, macOS 26.6.2, coremltools 9.0. Nothing is claimed for others.
- The gate covers the golden rows that fit each bucket (table above). Any row longer than the
  bucket is outside that cell's gate, as for the shipped artifacts.
- Latency is one exact-length request per bucket at B=1. No concurrency, no GPU load, no
  energy measurement.
- The W8 artifact's on-device ANE compile happens in the build process. The latency process
  reuses Core ML's cache for the same path. Its load time is recorded but not gated.

## Outputs

- `raw/<model>/L<L>-<config>/build.json`: every step, timing, placement, per-op plan,
  inventory, parity summary and error.
- `raw/<model>/L<L>-<config>/parity_rows.jsonl`: raw outputs per golden row.
- `raw/<model>/L<L>-<config>/latency.json`: every sample, both probes, and the environment
  before and after.
- `raw/<model>/L<L>-fp16/baseline_size.json`: size and SHA-256 of the shipped artifact
  compared against.
- `results.json`, `tables.md` (`analyze.py`), and the report in `README.md`.
