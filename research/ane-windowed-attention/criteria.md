# Criteria: windowed (local) attention BC1S graph, at the shipped and the long buckets

Issues: [#15](https://github.com/tc3oliver/laya-apple/issues/15) (shipped buckets) and
[#14](https://github.com/tc3oliver/laya-apple/issues/14) (long context). One experiment covers
both. This file was written and committed before any windowed artifact of this experiment was
built or measured. Nothing in it changes after data is seen. A failing cell is reported as a
failure, with its metric and magnitude. Smoke checks made while the scripts were written are
not data and are not committed.

## Question

The shipped BC1S graph computes dense L×L attention scores in all 28 encoder layers. 18 of
those layers are sliding-window (±64 tokens), and the graph only masks the scores outside the
window. Phase −1 built an exact block-local rewrite (`ConvAttention._windowed`) and measured it
on `laya-typed-decisions` only, in pooled runs:
- full-body forward P50 −13% at L512 and −25% at L1024;
- +2% at L128 and −2% at L256;
- parity PASS with rows bucketed to 128/256/512/1024;
- a much longer ANE compile: 55 s at L128, 227 s at L512
  (`research/phase-0-feasibility/ane-long-context.md`, H2; `raw/bench/latency.jsonl`, tag
  `windowed`).

This experiment carries that rewrite onto the production graph and runs it through the
unchanged artifact gate, with a paired latency comparison against the masked graph:

- **#15:** does it help at the buckets shipped today (64/96/128)?
- **#14:** does it help at 256/512? How does it compare with MLX there?

## The rewrite

`scripts/windowed_body.py`. It is applied to the production `laya_apple.conversion.bc1s.ConvBody`:
- the attention module of every local layer (`layer.kind != "full_attention"`, the production
  body's own test) is wrapped. The wrapper reuses that module's own qkv/out convolutions and RoPE
  tables;
- queries are processed in blocks of 64 tokens, the Phase −1 value, not tuned;
- each block visits only keys in [start − 64, end + 64), clipped to [0, L), with the host's
  additive local mask sliced to the same span;
- the global layers, norms, MLPs, the decision head and the scorer are the production graph,
  untouched;
- the computation of `_windowed` is copied unchanged from
  `research/phase-0-feasibility/scripts/ane_model.py`.

**What the rewrite can save, by construction.** Per local layer and head, it still computes
this share of the dense L×L score entries:

| L | 64 | 96 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| share computed | 1.000 | 1.000 | 1.000 | 0.625 | 0.344 |

At L ≤ 128 every 64-query block's key span already covers the whole sequence, so the rewrite
cannot remove score work there. It can only add slicing and concatenation. This is stated
before the data. It is the reason the #15 cells are expected to fail the latency criterion.

## Cells

`laya` and `laya-typed-decisions`, the two ModernBERT-large checkpoints with 18 local layers.
`laya-multilingual` (mmBERT-base) is out of scope: the rewrite was validated only on
ModernBERT-large.

| cells | issue | baseline arm (A) | golden rows in the gate, laya / typed |
|---|---|---|---|
| L64, L96, L128 | #15 | the **shipped** artifact in `LAYA_APPLE_CACHE`, loaded through `load_verified` | 41 / 41, 93 / 93, 122 / 122 |
| L256, L512 | #14 | a **masked** artifact built by this experiment's script, identical except for the rewrite | 146 / 146, 176 / 170 |

At 256/512 the masked baseline must pass the same placement, parity and probe checks. If it
fails any of them, that cell's latency result is INCONCLUSIVE (no valid baseline). The masked
results are reported as well.

## Build pipeline

`scripts/build_windowed.py`, per cell and variant. Production code is used unchanged except for
the rewrite:

1. pinned checkpoint, `verify_weights`;
2. PyTorch FP32 reference and the production `ConvBody`, then the rewrite (windowed variant only);
3. the production FP32 layout check (`conversion.build._layout_check`, windowed body against the
   PyTorch reference) must be < 1e-3, or the cell stops here;
4. an FP32 exactness diagnostic, windowed body vs masked body, max |Δlogit| and |ΔCLS| on:
   - an exact-length row;
   - the shortest golden row, padded to L (the padded-query path).

   The Phase −1 value was ≤ 4.6e-6. This is reported, not gated;
5. trace and `ct.convert` with the production arguments (`FLOAT16` compute, FP16 I/O, B=1,
   fixed L, `macOS15` target), then compile to `model.mlmodelc`;
6. the first `CPU_AND_NE` load time (the on-device ANE compile), recorded.

Artifacts stay under `/Volumes/Data/cache/laya-apple-research/windowed/`. The scripts refuse to
place it inside `LAYA_APPLE_CACHE`. Nothing is registered, offered or routed to.

## The gate: four separate results per windowed cell

None of these is loosened, re-implemented or re-counted.

1. **Placement.** `compute_plan_summary` on `CPU_AND_NE` and `check_ane_placement`: 100% of
   ops on the ANE, 0 CPU/GPU ops, 0 device transitions. The MIL op count and per-op-type plan
   are recorded as diagnostics.
2. **Parity.** `laya_apple.parity.ane.ane_parity` at the cell's bucket, on this machine's
   Neural Engine, against the upstream PyTorch FP32 goldens. The unchanged FP16 gate applies:
   - calibrated probability max |Δ| ≤ 0.02, action probability max |Δ| ≤ 0.02;
   - 0 hard mismatches, where a hard mismatch is a top-1 flip with reference margin ≥ 0.04;
   - near-tie flips listed, not failed;
   - exact prompt tokens, all outputs finite, and a repeated call bitwise identical.

   Every golden row that fits the bucket is evaluated at that bucket, so short rows run heavily
   padded at 256/512. This is stricter than Phase −1, which bucketed each row to the smallest
   length that held it. A recording pass keeps every row's raw outputs, and `analyze.py`
   recomputes the per-row table from them.
3. **Placement probe.** `probe_placement` unchanged: the loaded model's fastest-of-5 time on
   the pad probe input must be ≤ 0.8× the same artifact on `CPU_ONLY`.
4. **Latency** (gated, as improvement). The paired gate below: windowed predict P50 must be
   ≤ 0.95× the masked graph's.

**Per cell:**
- **graph-change candidate**: all four results are PASS;
- **documented no-change**: any of the four is FAIL, or the build fails at any step;
- **inconclusive**: none is FAIL but latency is INCONCLUSIVE, which includes having no valid
  baseline.

The four results are always reported side by side, and never merged into one pass/fail.

**What a result does and does not lead to.**
- A graph-change candidate at a shipped bucket (#15) would be proposed as its own production PR:
  - a new graph id in the manifest and registry, with tests;
  - rebuilt artifacts;
  - the full gate on the tested profile;
  - the benchmark comparison `CONTRIBUTING.md` asks for.
- A candidate at 256/512 (#14) makes those lengths eligible for validation only. Whether they are
  offered or routed is decided by `scripts/derive_routing.py` in its own PR with measurements,
  not here.
- This experiment changes no routing, bucket, registry entry or shipped artifact.

## Latency protocol and paired gate

`scripts/latency.py`, one process per run, in an exclusive hardware slot: oMLX stopped, no
other benchmark, build, test or package install running.

- **Arms.**
  - A: the baseline (above).
  - B: the windowed artifact, `CPU_AND_NE`.
  - M, at 256/512 only, descriptive: the MLX FP16 backend (`MLXBackend.forward`).
- **Input.** One exact-length request, `make_request(tokenizer, config, L, n_questions=1,
  seed=0)`, as `laya-apple benchmark` builds it. The row must be exactly L tokens.
- **Warm-up.** 20 calls per arm, not recorded.
- **Windows.** 10 cycles. Each cycle has one 100-forward window of A and one of B, ordered A B in
  even cycles and B A in odd cycles. At 256/512 a 100-forward window of M follows in each cycle.
- **Per ANE forward.** `predict_ms` (`model.predict` only) and `forward_ms` (features +
  `predict` + host tail, the backend boundary).
- **Per MLX forward.** `forward_ms` (`MLXBackend.forward` on the prepared row). Every sample
  is kept.
- **Pair.** B's window and A's window of the same cycle, 10 pairs.
- **Statistic.** Per pair, B predict P50 ÷ A predict P50. The geometric mean of the 10 ratios,
  with a 95% Student t interval on the log ratios (df = 9).
- **Verdict (limit 0.95).**
  - PASS if the upper bound ≤ 0.95, meaning at least 5% faster at 95% confidence;
  - FAIL if the lower bound > 0.95;
  - INCONCLUSIVE otherwise.
- **Why 0.95.** The rewrite costs a larger graph and a longer ANE compile (Phase −1: 55 s vs
  about 43 s at L128, 227 s at L512). A gain below 5% does not pay for that. The Phase −1 effect
  at L512 (−13%) is well beyond the limit.
- **Sensitivity only.** A seeded percentile bootstrap over pairs (10,000 resamples). It never
  changes the verdict.
- **Descriptive, not gated.**
  - the `forward_ms` ratio with its interval;
  - each arm's median window P50;
  - both arms' load times and the first-load (compile) time from the build;
  - the max |Δlogit| between A and B on the workload row;
  - at 256/512, windowed/MLX and masked/MLX forward ratios per cycle, with intervals.

  The MLX window always runs last in a cycle. Its position is fixed, not balanced, so it is
  descriptive only.

The paired baseline is measured in the same session and replaces the historical numbers as the
comparison. `benchmarks/v1.0.md` and the Phase −1 tables are cited for context only, and never
compared without saying that their methodology differs (pooled runs, not paired windows).

## Run order and stop rules

1. Build the windowed variant at every cell. This is CPU-heavy; the parity step uses the ANE.
2. Build the masked variant at 256/512.
3. Run latency and the probe for every cell whose builds reached `done`, in the exclusive slot.
4. Run `analyze.py`.

Other rules:
- A cell whose FP32 layout check fails stops before conversion and is a documented no-change.
- Each cell is run once.
- A run interrupted by a documented external event is re-run once in full. The interrupted data
  is kept under `raw/aborted/`.
- No re-run because a result is unwelcome.
- No automatic extension: no other block size, no L1024, no multilingual, no more cycles. Any of
  those is a new preregistration.

## Predictions (written down to be checked, not criteria)

- Layout check < 1e-3, and FP32 exactness ≤ 1e-5, at every cell.
- Placement: PASS at every cell (Phase −1: 100% ANE, 0 transitions for typed at 128–1024).
- Parity: PASS at every cell. The padded-query path at 256/512 is the new risk, because Phase −1
  never ran short rows padded that far.
- Latency:
  - L64/96/128: FAIL (ratio ≈ 1.00–1.03);
  - L256: FAIL or INCONCLUSIVE (Phase −1: −2%);
  - L512: PASS (Phase −1: −13%).
- Against MLX at 256/512: windowed ANE is still slower at L512 (Phase −1: 47.5 vs 35.2 ms).
- First load: well above the masked graph's, growing with L.

## What does not change

- The parity tolerances, the placement check and the probe threshold.
- Routing, the registry and the shipped artifacts.
- `laya_apple/` is not modified. The scripts import it; it never imports them.

## Known limitations, stated in advance

- One machine and profile: M4 Max, macOS 26.6.2, coremltools 9.0.
- One block size (64) and one formulation. Other exact formulations (larger head groups, chunked
  keys with online softmax) are not tested.
- Latency is one exact-length request per bucket at B=1. No concurrency, no GPU load, no energy.
- The MLX comparison is descriptive and unbalanced in position.

## Outputs

- `raw/<model>/L<L>-{windowed,masked}/build.json`: every step and timing, layout and exactness,
  placement and per-op plan, first load, parity summary, error.
- `raw/<model>/L<L>-{windowed,masked}/parity_rows.jsonl`: raw outputs per golden row.
- `raw/<model>/L<L>-windowed/latency.json`: every sample of every arm, both probes, and the
  environment before and after.
- `results.json`, `tables.md` (`analyze.py`), and the report in `README.md`.
