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

Two README figures are not benchmark data: the Quickstart output (`high`, its
probabilities and 11.2 ms) and "probabilities within 0.002" on `device="gpu"`. Both come
from running the Quickstart; re-run it to see the values on your machine.

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
