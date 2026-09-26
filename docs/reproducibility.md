# Reproducibility

Every headline number in the [README](../README.md) traces to a raw file in the
repository, a command that regenerates that file, and a command that renders the table.
This page lists them. Setup (checkpoints, ANE artifacts, `LAYA_APPLE_CACHE`,
`HF_HUB_OFFLINE`) is in [`benchmarks.md`](benchmarks.md#setup).

## Environment

Read from the raw files, not typed from memory:

| Field | Value | Recorded in |
|---|---|---|
| SoC | Apple M4 Max, 16 CPU cores (12 performance, 4 efficiency) | `platform.soc` in `raw.jsonl`, `placement-*.json`, `concurrency-*.json`, `probe.json`; `environment.cpu_cores` in `parity/*/coreml_*.json` |
| Memory | 68719476736 bytes | `environment.memory_bytes` in `parity/*/coreml_*.json` |
| macOS | 26.6.2, build 25G83 | `platform.macos`, `platform.macos_build` |
| Python | 3.12.14 | `platform.python` |
| coremltools | 9.0 | `platform.coremltools` |
| MLX | 0.32.2 | `environment.packages.mlx` in `parity/*/coreml_*.json` and `comparators-latency.jsonl`; `uv.lock` for the laya-apple environment |
| NumPy | 2.1.3 | `environment.packages.numpy`; `uv.lock` |
| torch (comparators, goldens) | 2.7.0 | `environment.packages.torch` |
| transformers / upstream `laya` / `laya-coreml` (comparators) | 5.17.0 / 0.3.5 / 0.1.1 | `environment.packages` |
| laya-apple | 1.0.0 | `laya_apple` in `placement-*.json`, `concurrency-*.json` |

All paths are under `benchmarks/v1.0/`. `raw.jsonl` records `laya_apple: 0.3.0`: it was
recorded before `__version__` was set to 1.0.0 in the release commit. The comparator
files also record the pinned checkpoint revisions and weight hashes
(`environment.models`).

## Claim → source

Report sections are in [`benchmarks/v1.0.md`](../benchmarks/v1.0.md). Raw files are under
`benchmarks/v1.0/` unless a path says otherwise. "Regenerate" commands are listed in full
under [Re-running everything](#re-running-everything).

| README claim | Report section | Raw file(s) | Regenerate | Render | Environment |
|---|---|---|---|---|---|
| GPU + ANE gives 2.9–4.6× GPU-only throughput, 0 answer changes; closed-loop table (41.9 → 122.5 req/s, 2.92×, …) | [Closed loop](../benchmarks/v1.0.md#heterogeneous-gpu--ane-serving-closed-loop) | `placement-laya-thread.json`, `placement-laya-multilingual-process.json`, `placement-laya-typed-decisions-thread.json` | `scripts/bench_concurrency.py --part a` (step 2) | `scripts/v1_report.py hetero` | `platform`, `args` in each file |
| Open-loop P99 table (e.g. laya Poisson 43.1 req/s: 170.7 → 30.3 ms) | [Open loop](../benchmarks/v1.0.md#open-loop) | `concurrency-laya.json`, `concurrency-laya-multilingual.json`, `concurrency-laya-typed-decisions.json` | `scripts/bench_v1.sh`, part 3 (step 3) | `scripts/v1_report.py openloop` | `platform`, `args` |
| Parity table: ordinary Core ML on `CPU_AND_NE` fails (12 / 85 / 19 hard), `CPU_AND_GPU` and PyTorch MPS results | [Correctness](../benchmarks/v1.0.md#correctness-vs-upstream-pytorch-fp32) | `parity/<model>/coreml_units-*.json`, `parity/<model>/torch_device-mps.json` | `scripts/bench_v1.sh`, part 2 (step 3) | `scripts/v1_report.py parity` | `environment` in each file |
| Parity table: laya-apple MLX FP16, ANE FP16; MLX FP32 ≤ 1.1e-5 | [Correctness](../benchmarks/v1.0.md#correctness-vs-upstream-pytorch-fp32) | `parity/<model>/laya-apple-{gpu-float16,gpu-float32,ane-float16}.json` | `laya-apple parity` (step 4) | `scripts/v1_report.py parity` | not recorded in these files |
| laya ANE near-tie flip: one row, upstream margin 0.0035 | [Correctness](../benchmarks/v1.0.md#correctness-vs-upstream-pytorch-fp32) | `parity/laya/laya-apple-ane-float16.json`, field `near_tie_flips[].ref_margin` | `laya-apple parity laya --device ane` (step 4) | read the field; `v1_report.py parity` shows the count | not recorded in the file |
| Ordinary Core ML on `CPU_AND_NE` runs on the CPU at laya-typed-decisions L1024 (2201.8 ms) | [Correctness](../benchmarks/v1.0.md#correctness-vs-upstream-pytorch-fp32) | `comparators-latency.jsonl` (`units: cpu_ne`, `length: 1024`) | `scripts/bench_v1.sh`, part 1 (step 3) | `scripts/v1_report.py latency` | `environment` per record |
| Single-request forward P50 tables | [Forward latency](../benchmarks/v1.0.md#single-request-forward-latency) | `raw.jsonl` (laya-apple), `comparators-latency.jsonl` (PyTorch, Core ML ordinary) | `scripts/release_bench.py` (step 1); `scripts/bench_v1.sh`, part 1 (step 3) | `scripts/v1_report.py latency` | `platform` / `environment` per record |
| Enumerated-shape export: 217 ms at L128 on laya | [Forward latency](../benchmarks/v1.0.md#single-request-forward-latency) | `comparators-latency.jsonl` | `scripts/bench_v1.sh`, part 1 (step 3) | `scripts/v1_report.py latency` (216.8) | `environment` |
| End to end adds 0.08–0.61 ms; typed-decisions L128 `auto` 9.91 ms forward, 10.06 ms `predict` | [End-to-end](../benchmarks/v1.0.md#end-to-end-latency) | `raw.jsonl` (`device_requested: auto`) | `scripts/release_bench.py` (step 1) | `scripts/v1_report.py e2e` | `platform` |
| Auto routing: MLX 33.2 ms at L128 with 4 questions; multilingual L256 ANE 8.3 vs MLX 8.6 ms, MLX L128 6.4 ms | [Auto routing](../benchmarks/v1.0.md#auto-routing) | `raw.jsonl` | `scripts/release_bench.py` (step 1) | `scripts/v1_report.py e2e`, `latency` | `platform` |
| Auto routing: typed-decisions L512 ANE 54 ms vs MLX 35 ms; laya ANE 39.4 ms at L128 with 4 questions (Phase -1, not v1.0) | — | `research/phase-0-feasibility/raw/bench/latency.jsonl` | `research/phase-0-feasibility/` harness | `research/phase-0-feasibility/reports/latency.md` | `research/phase-0-feasibility/environment.md` |
| v1.0 re-run reproduced v0.1 within ±1.8% over 50 configurations | [Reproducibility](../benchmarks/v1.0.md#reproducibility-single-request-latency-against-v01) | `raw.jsonl`, `benchmarks/v0.1/raw.jsonl` | `scripts/release_bench.py` (step 1) | `scripts/compare_bench.py latency benchmarks/v0.1/raw.jsonl benchmarks/v1.0/raw.jsonl` | `platform` |
| Option order: upstream changes its decision in 35% (typed-decisions) and 22.5% (laya) of cases; MLX FP32 reproduces upstream exactly | [Option order](../benchmarks/v1.0.md#option-order-robustness) | `option-order-upstream-*.json`, `option-order-{gpu-float32,gpu-float16,ane-float16}.json` | `scripts/option_order.py run`, `research/option-order/upstream.py` (step 5) | `scripts/option_order.py compare` | not recorded in these files |
| Cold start 3–5 minutes per model (v0.3, not v1.0) | — | `benchmarks/v0.3/coldstart.json` | `scripts/bench_coldstart.py` | [`benchmarks/v0.3.md`](../benchmarks/v0.3.md) | in the file |
| Per-stream P99 under concurrency is above its solo value (v0.2, not v1.0) | — | `benchmarks/v0.2/` | `scripts/bench_concurrency.py` | [`benchmarks/v0.2.md`](../benchmarks/v0.2.md) | in the files |
| `laya-apple serve`: 792 of 792 requests match unmodified upstream `laya.serve` 0.3.20 within the FP16 parity gate (1.3.0, not v1.0) | [`benchmarks/serve-compat/README.md`](../benchmarks/serve-compat/README.md), "Results" | `benchmarks/serve-compat/*.json`: `passed` per file and per case, 9 files, 792 cases | `scripts/upstream_serve_pinned.py`, `laya-apple serve --preload typed-decisions`, then `scripts/compare_serve_upstream.py` per file ([Reproduce](../benchmarks/serve-compat/README.md#reproduce)) | count `cases[].passed` over the 9 files | "Setup" in the report; not recorded in the files |
| `laya-apple serve`: 365 of those answered on the Neural Engine | same | `benchmarks/serve-compat/*-split.json`, `cases[].ours_device == "ane"` (122 + 121 + 122) | the `--split` runs above | count `ours_device` | same |
| `laya-apple serve`: tested unmodified with 7 Jev clients | [`integrations/jev-plugins/README.md`](../integrations/jev-plugins/README.md) | [`integrations/jev-plugins/runs/2026-09-25.md`](../integrations/jev-plugins/runs/2026-09-25.md): versions, install and run commands, decoded output per client | by hand, per client, as in the run record | — | `integrations/jev-plugins/README.md`, "Tested" (Apple M4 Max); macOS not recorded |
| `laya-apple serve` beside a local LLM (run 2, 1.4.0, not v1.0): short-decision P99 with the LLM busy 47.2 ms (`auto`) against 122.3 ms (`--device gpu`); LLM idle 43.2 and 55.9 ms | [`benchmarks/serve/README.md`](../benchmarks/serve/README.md), run 2; [`m4-max-r2/tables.md`](../benchmarks/serve/m4-max-r2/tables.md), "Decisions" | `benchmarks/serve/m4-max-r2/results.json`: `cells.<config>/<kind>.classes.short_1q.p99_ms_median`; per-window raw data in `m4-max-r2/raw/` | `benchmarks/serve/run.sh benchmarks/serve/<machine>` ([Run](../benchmarks/serve/README.md#run)) | `benchmarks/serve/analyze.py benchmarks/serve/m4-max-r2` | `raw/campaign.json`, `platform` (Apple M4 Max, macOS 26.6.2); LLM server version in `llm_status_start` |
| Same run: LLM 42.2 tok/s alone, −4.0% beside serve `auto`, −5.3% beside `--device gpu`; the 1.31-point gap against the 1.28-point LLM-alone window spread (G1, G2) | same, "LLM throughput" and "Criteria" | `results.json`: `cells.llm_alone.llm_tok_s_median`, `llm_tok_s_drop`, `llm_alone_noise`, `results.gpu_free` | same | same | same |
| Same run: 0 hard mismatches and 0 errors over 7,728 decisions, max probability error 0.0039 (C1, E1) | same, "Decisions" and "Criteria" | `results.json`: `results.correctness`; `n`, `hard_mismatches`, `errors` per cell and class (4 cells × (1,572 + 360)) | same | same | same |
| Adaptive ANE execution (1.5.0, not v1.0): 154 production episodes of laya and laya-typed-decisions all stayed on the fast path; GPU return 0.035–0.043 ms against 4.28–8.60 ms on the 1.4 path; throughput 1.038–1.042×; median episode P99 0.18–0.53 ms lower; 0 mismatches, routing failures, lost requests or crashes | [`research/coreml-adaptive-breaker/val_tables.md`](../research/coreml-adaptive-breaker/val_tables.md) | `research/coreml-adaptive-breaker/val_results.json`: `phases.<2,3,5>.stats`; raw runs in `raw-val/` | `research/coreml-adaptive-breaker/scripts/run_val.sh <phase>` ([`validation.md`](../research/coreml-adaptive-breaker/validation.md)) | `research/coreml-adaptive-breaker/scripts/val_analyze.py` | `raw-val/*.before.json` / `.after.json` (machine), `runtime` in each run |
| Adaptive ANE execution recovery: in 12 of 12 slow episodes the breaker tripped within 40 ms and latency was A-like within 164–414 ms; the slow state's median short latency 12.3 ms against 10.0 ms | [`research/coreml-adaptive-breaker/phase1_tables.md`](../research/coreml-adaptive-breaker/phase1_tables.md) | `phase1_results.json`: `stats.R`, `stats.B`; raw runs in `raw/` | `research/coreml-adaptive-breaker/scripts/run_phase1.sh` ([`phase1.md`](../research/coreml-adaptive-breaker/phase1.md)) | `research/coreml-adaptive-breaker/scripts/phase1_analyze.py` | `raw/*.before.json` / `.after.json`, `runtime` in each run |
| Same run: about 4% of `auto` short decisions on the GPU by the backlog spill (130 of 3,144) | same, "Validity" (V4) | `results.json`: `validity.V4_auto_short_policy_path.spill_share_windows`; `cells.auto/*.classes.short_1q.devices.gpu` (67 + 63) | same | same | same |
| Same run: multi-question P99 with the LLM busy against idle, 117.1 against 84.1 ms (`auto`) and 153.1 against 79.4 ms (`--device gpu`) | same, "Decisions" | `results.json`: `cells.<config>/<kind>.classes.mixed_3q.p99_ms_median` | same | same | same |
| Serve run 1 is invalid (V2 and V4 failed) | [`benchmarks/serve/README.md`](../benchmarks/serve/README.md#run-1-m4-max-invalid) | `benchmarks/serve/m4-max/results.json`: `valid`, `validity` | — | `benchmarks/serve/analyze.py benchmarks/serve/m4-max` | `m4-max/raw/campaign.json` |

The README figure `docs/readme/serve-llm-load.svg` is drawn from the run-2 `results.json`
and `raw/campaign.json` above by `scripts/generate_readme_svgs.py`, which refuses a run
that is not valid under its own checks; `--check` fails if the committed figure differs
from the data.

Three README figures are not benchmark data:
- the Quickstart output (`high`, its probabilities and 11.2 ms);
- "probabilities within 0.002" on `device="gpu"`;
- the `laya-apple serve` terminal figure (`docs/readme/serve-demo.svg`, answer `fix_code`
  on the ANE). It is one recorded session, `docs/readme/serve-demo.json`, captured by
  `scripts/capture_serve_demo.py` and rendered by `scripts/generate_readme_svgs.py`.

The first two come from running the Quickstart; re-run it to see the values on your
machine. Re-run `scripts/capture_serve_demo.py` for the third.

## Re-running everything

Run on a quiet machine on AC power, with nothing else using the GPU or ANE, after the
[setup](benchmarks.md#setup) steps. Steps 1 and 3 stop early if free disk is below the
minimum, and warn if the Core ML E5 caches are large; see
[Disk space and the Core ML E5 cache](benchmarks.md#disk-space-and-the-core-ml-e5-cache).
From the repository root:

```bash
export LAYA_APPLE_CACHE=/path/to/cache
export HF_HUB_OFFLINE=1

# 1. laya-apple single-request latency: 50 configurations, 2 passes
uv run python scripts/release_bench.py benchmarks/v1.0/raw.jsonl

# 2. closed-loop GPU + ANE mix, per model, with its measured ANE placement
uv run python scripts/bench_concurrency.py --model laya --short 128 --long 512 \
  --ane-placement thread --part a --output benchmarks/v1.0/placement-laya-thread.json
uv run python scripts/bench_concurrency.py --model laya-multilingual --short 96 --long 1024 \
  --ane-placement process --part a --output benchmarks/v1.0/placement-laya-multilingual-process.json
uv run python scripts/bench_concurrency.py --model laya-typed-decisions --short 128 --long 1024 \
  --ane-placement thread --part a --output benchmarks/v1.0/placement-laya-typed-decisions-thread.json

# 3. comparators (latency + parity) and the open-loop mix
LAYA_APPLE_ARTIFACTS=/path/to/research-artifacts scripts/bench_v1.sh

# 4. laya-apple parity against the shipped goldens (repeat per model)
uv run laya-apple --offline parity laya --device gpu --dtype float32 \
  > benchmarks/v1.0/parity/laya/laya-apple-gpu-float32.json
uv run laya-apple --offline parity laya --device gpu --dtype float16 \
  > benchmarks/v1.0/parity/laya/laya-apple-gpu-float16.json
uv run laya-apple --offline parity laya --device ane \
  > benchmarks/v1.0/parity/laya/laya-apple-ane-float16.json

# 5. option order (upstream.py runs in the research venv, see research/option-order/)
uv run python scripts/option_order.py write-cases
uv run python scripts/option_order.py run --device gpu --dtype float32 \
  --output benchmarks/v1.0/option-order-gpu-float32.json
uv run python scripts/option_order.py run --device gpu --dtype float16 \
  --output benchmarks/v1.0/option-order-gpu-float16.json
uv run python scripts/option_order.py run --device ane --dtype float16 \
  --output benchmarks/v1.0/option-order-ane-float16.json
(cd research/phase-0-feasibility && \
  uv run python ../option-order/upstream.py --model laya \
    --output ../../benchmarks/v1.0/option-order-upstream-laya.json && \
  uv run python ../option-order/upstream.py --model laya-typed-decisions \
    --output ../../benchmarks/v1.0/option-order-upstream-laya-typed-decisions.json)

# 6. render
uv run python scripts/v1_report.py all
uv run python scripts/compare_bench.py latency benchmarks/v0.1/raw.jsonl benchmarks/v1.0/raw.jsonl
uv run python scripts/compare_bench.py concurrency benchmarks/v0.2 benchmarks/v1.0
uv run python scripts/option_order.py compare \
  benchmarks/v1.0/option-order-gpu-float32.json benchmarks/v1.0/option-order-upstream-laya.json
uv run python scripts/option_order.py compare \
  benchmarks/v1.0/option-order-gpu-float32.json \
  benchmarks/v1.0/option-order-upstream-laya-typed-decisions.json
```

Notes:
- `bench_v1.sh` runs the comparators with the Phase -1 harness in
  `research/phase-0-feasibility/`, which has its own venv (upstream `laya`, torch,
  `laya-coreml`). `LAYA_APPLE_ARTIFACTS` points at that harness's Core ML packages.
  `MODELS` and `BUDGET` override the model list and the per-configuration time budget.
- `laya-apple parity` prints JSON on stdout and exits non-zero if the gate fails.
- `benchmarks/v1.0/probe.json` (the placement probe evidence) comes from
  `.venv/bin/python scripts/bench_probe.py --runs 5 --out benchmarks/v1.0/probe.json`.

## Expected runtime

From the timestamps and durations in the v1.0 raw files, on the tested machine:

| Step | Recorded |
|---|---|
| 2. closed loop | runs started at 11:32:53, 11:37:32 and 11:42:10 UTC: about 5 minutes per model |
| 3. comparator latency | first to last record, 11:56:19 to 12:09:38 UTC |
| 3. comparator parity | 10.1–187.1 s per file (`seconds`) |
| 3. open loop | runs started at 12:23:26, 12:26:18 and 12:29:10 UTC: about 3 minutes per model |
| 5. upstream option order | 39.1 s (laya), 39.3 s (laya-typed-decisions) |

Steps 1, 4 and the laya-apple option-order runs do not record their duration. Building
the ANE artifacts beforehand includes the on-device Core ML compile, 3–5 minutes per
model on a fresh artifact location ([`benchmarks/v0.3.md`](../benchmarks/v0.3.md)).

## Your own machine

To measure a different Mac, use the community benchmark bundle
(`uv run python scripts/hardware_report.py`); see
[`community-benchmarks.md`](community-benchmarks.md). Results from another machine are
a different profile and are not merged into the v1.0 tables.
