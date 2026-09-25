# Same-machine comparison of Apple-silicon Laya runtimes

**Question.** On one Mac, with the same checkpoint where possible, how do the available
Apple-silicon Laya runtimes compare on correctness against upstream Laya, latency,
throughput and (optionally) energy, per request shape?

**Status.** Method, pins and tooling only. No campaign has been run, so this directory has no
results yet. `raw/`, `results.json` and `tables.md` appear after the first run. This README
was written before any measurement. A change to the method after a run is recorded here, and
numbers measured under different methods are not compared.

## Runtimes under test

Each runtime is installed into its own throwaway uv environment (Python 3.12) and never into
the laya-apple environment. Pins and install lines are in [`runtimes.json`](runtimes.json).

| Runtime | Pinned version | Install | Weights | License |
|---|---|---|---|---|
| laya-apple | this checkout (`git rev-parse HEAD` is recorded per run) | `uv sync --inexact --extra ane` | upstream snapshots pinned in `laya_apple.registry` | Apache-2.0 |
| [laya-mlx](https://github.com/mizorewww/laya-mlx) | `laya-mlx==0.2.0` (PyPI) | `uv pip install laya-mlx==0.2.0` | the same upstream snapshots (`laya_mlx.load` takes a local checkpoint directory) | Apache-2.0 |
| [laya-coreml](https://github.com/mizorewww/laya-coreml) | `laya-coreml==0.1.0` (PyPI) | `uv pip install laya-coreml==0.1.0` | the project's own Core ML bundles on Hugging Face, pinned by revision (below) | Apache-2.0 |
| [laya-fast](https://github.com/DJLougen/laya-fast) | commit `3a9d534642cadb54c4cdb531f9abfa694acbefc8` (no PyPI release) | source tarball at that commit; `uv pip install -r requirements-lock.txt -r requirements-export.txt` | `convert.py` (MLX fp16) of the upstream `laya` snapshot, plus `ane/export.py` bodies | Apache-2.0 |
| upstream Laya (reference) | `laya==0.3.20`, torch 2.7.0, transformers 5.17.0, tokenizers 0.23.2 | `uv pip install ...` | the same upstream snapshots, PyTorch CPU FP32 | Apache-2.0 |

The laya-mlx and laya-coreml source trees, at commits `0a859518634112655cb97c745dbf04f5191aaf13`
and `4619e0483f07adf39068532e85b42ec2347edb83`, are downloaded only for their published
fixtures. The runtimes under test are the PyPI wheels.

### Configurations

Each runtime is measured in its default configuration, plus the configuration its own README
recommends for speed. They are separate rows and are never averaged together.

| Runtime | Variant | What it is | Models |
|---|---|---|---|
| laya-apple | `auto` | default: MLX GPU, plus validated ANE artifacts for short single questions | all three |
| laya-apple | `gpu` | MLX GPU only | all three |
| laya-mlx | `fp16` | default: `dtype="float16"`, eager | all three |
| laya-mlx | `fp16-opt` | documented opt-in: `compile=True, pad_to_multiple=16, cache_prompts=True` | all three |
| laya-coreml | `default` | enumerated-shape Core ML export, `cpu_gpu` (the package default) | all three |
| laya-coreml | `ane` | `aac6fef/laya-multilingual-coreml-ane`, fixed B1/L96, `cpu_ne` | laya-multilingual only |
| laya-fast | `mlx` | `LayaMLX(converted-fp16, float16, compile=True)`, the CLI default | laya only |
| laya-fast | `fast` | `LayaFast` with ANE bodies for buckets 64, 80, 96, 128, 256, 512 | laya only |

laya-coreml bundle revisions: `aac6fef/laya-coreml@fff78b2d`,
`aac6fef/laya-multilingual-coreml@8139e908`, `aac6fef/laya-typed-decisions-coreml@28d24fa8`,
`aac6fef/laya-multilingual-coreml-ane@39d6a9b3`. They total about 3.1 GB and are downloaded
into the Hugging Face cache (`HF_HOME`).

### Weights

Upstream checkpoints: `convaiinnovations/laya@c5d78730`,
`convaiinnovations/laya-multilingual@052592a1` and
`convaiinnovations/laya-typed-decisions@f9ab0b22`, as pinned by laya-apple. The comparison
reuses the snapshots already in the Hugging Face cache and downloads none of them again.

- **laya-apple, laya-mlx, upstream:** load these exact snapshot directories.
- **laya-coreml:** each bundle's manifest records the weight hash it was exported from.
  `analyze.py` compares it with the pinned hash, and the weights table shows the result per
  configuration. A bundle exported from other weights is marked "different weights" and is
  not a same-weights comparison.
- **laya-fast:** its README pins `convaiinnovations/laya@1c5edc17`. `drive.py setup` compares
  the two revisions through the Hugging Face tree API. Only `README.md` differs; the weights,
  configuration, encoder configuration and tokenizer files are identical. `convert.py` therefore
  runs on laya-apple's snapshot. The converted weights (~0.8 GB) and ANE bodies (~0.7 GB each)
  go to `--model-root` (default: `compare-v1.4/` next to `HF_HOME`, so on the same volume as
  the Hugging Face cache).
- **Upstream basis.** laya-mlx and laya-coreml adapt upstream prompt and output code at
  `573e5b62`, with the v0.3.5 temperature clamp. laya-fast ports the Hugging Face reference
  code at `1c5edc17`. The goldens come from upstream 0.3.20. Cases that exercise prompt
  changes made after 0.3.5 are therefore reported as a separate parity group; they are not
  merged into the 0.3.5-identical rows.

## Fixtures

| Set | Source | Cases | Reference |
|---|---|---|---|
| `goldens` | `laya_apple/parity/goldens` (upstream 0.3.20, PyTorch CPU FP32) | 34 / 37 / 37 per model | committed golden answers |
| `published-laya-coreml` | laya-coreml `benchmarks/results/reference.json` at the pinned commit (the 63-question suite laya-mlx also validates against) | 16 per model | upstream 0.3.20 CPU FP32 answers, recorded in the same run |
| `published-laya-fast` | laya-fast `benchmarks/benchmark.py::make_fixtures` at the pinned commit | 4 (English) | same |
| `shapes` | [`shapes.json`](shapes.json), written for this project (`make_shapes.py`) | 5 per model | same |

The third-party fixtures are read from the pinned source trees at run time. They are not
copied into this repository; `raw/<run>/fixtures.json` records the sha256 of every fixture
file used.

The golden set is split into two groups: cases whose prompt behaviour is identical in upstream
0.3.5 and 0.3.20, and the `drift-*` cases, which exercise upstream changes after 0.3.5
([`docs/correctness.md`](../../docs/correctness.md)). A runtime that follows 0.3.5 is expected
to differ from the goldens on the second group. That difference is reported, not hidden, and
it is not the same finding as a numerical error.

### Request shapes (latency and throughput)

| Shape | Request | Prompt tokens (laya / multilingual / typed) |
|---|---|---|
| `short` | 1 choice question | 62 / 65 / 62 |
| `long` | 1 noul question, long state | 401 / 441 / 401 |
| `mixed3` | choice + score + noul in one call | ≤ 73 / 76 / 73 per question |
| `uniform3` | three noul questions in one call (control for `mixed3`) | ≤ 73 / 76 / 73 |
| `batch16` | 16 short questions, all three types | ≤ 77 / 82 / 77 |

`mixed3` is the request shape described in laya-coreml issue #5: one call mixing question
types. `uniform3` has the same state and question count with a single type, so the effect of
mixing types can be separated from the effect of the question count. Every shape fits a
96-token fixed-shape artifact except `long`. Token counts come from laya_apple's tokenizer,
which reproduces upstream's prompt exactly.

## Method

Fixed before the first run:

- **Machine state.** oMLX and other GPU/ANE workloads stopped (the driver refuses to start
  while oMLX is running), power connected, no other benchmark running. `manifest.json` records
  the load average, power source and `pmset -g therm` at the start and the end.
- **Isolation.** One runtime process at a time. Each configuration runs in a fresh process
  started with its own environment's python (`adapter.py`). Load time is recorded and is not
  part of any latency.
- **Timing boundary.** End to end in the calling process: `predict`/`system_one` from the
  request dict to the answers dict. This includes prompt building, tokenization, inference,
  calibration and formatting. It is measured with `time.perf_counter`, one call at a time,
  with no think time.
- **Warm-up.** For every shape, one call (this also detects an unsupported shape), then 20
  warm-up calls, all discarded.
- **Windows.** One 20 s window per shape per round: calls repeat until 20 s have passed, so
  the last call may run past the window. Every call's latency is kept in `raw/`.
- **Rounds and ordering.** 3 rounds. In round r, the configuration order is rotated by r and
  each configuration's shape order is rotated by r + its position. No configuration and no
  shape always runs first. A 15 s pause follows every process.
- **Latency statistics.** P50 and P99 over all windows' samples pooled, plus the range of
  per-window P50s. A P99 from fewer than 1,000 samples is marked, because it rests on
  ≤ 10 samples above it.
- **Throughput.** Closed loop, one client: calls and questions completed per second of window
  wall time. This is not a concurrent-serving measurement. laya-apple's heterogeneous
  `execution="workers"` serving is out of scope here and is measured in
  `benchmarks/switchyard/`.
- **Parity.** Every fixture set, before the latency rounds, in its own process per
  configuration. The first answered case is repeated twice to check that repeated calls give
  identical answers.
- **Energy (optional).** Set `--energy-cmd` or `COMPARE_ENERGY_CMD` to a sampler command that
  contains `{out}`. The contract matches `research/energy-sampler`: it runs until SIGINT and
  writes JSON with `mean_power_w` and, optionally, CLOCK_UPTIME_RAW `samples`.
  - The adapter starts the sampler 1 s before each window, stops it after the window, and
    stamps the window with CLOCK_UPTIME_RAW.
  - `analyze.py` computes the window's SoC energy as soc(end) − soc(start) from the sampler's
    cumulative rails (CPU + GPU + ANE + DRAM), linearly interpolated at both window edges, and
    divides by the window length. Without a timeline that brackets the window, it falls back
    to the sampler's whole-life mean. That value includes the 1 s lead-in and the tail up to
    SIGINT, so it is labelled as a fallback in `results.json`.
  - Each round starts with a 20 s idle window, whose power is subtracted, giving net J per call
    and per question.
  - Without a sampler, the energy table says "not measured" and nothing else changes.

### Parity gate

The FP16 gate of [`docs/correctness.md`](../../docs/correctness.md), with tolerance 0.02,
applied to the public answers every runtime returns:

- probability error ≤ 0.02 over each question's options (noul: [1 − p, p]);
- act-probability error ≤ 0.02;
- 0 hard mismatches (the selected option differs and the reference top-1/top-2 margin
  ≥ 0.04);
- 0 label-set mismatches;
- repeated calls identical.

Near-tie flips (a different selection inside the 0.04 band) are listed row by row. Cases a
runtime refuses, for example a capacity error on a fixed-shape bundle, are counted and listed
separately; they are not answered rows.

This is an answer-level comparison. Public answers are rounded to 4 decimals, so errors below
1e-4 are not resolved. The v1.0 tables in `docs/correctness.md` compare unrounded logits, and
are a different measurement.

### Separate tables

`tables.md` has five tables: weights, parity, latency, throughput and energy. They are never
merged. There is no composite score and no overall winner.

## Run

```bash
export LAYA_APPLE_CACHE=<the laya-apple artifact cache holding this machine's ANE artifacts>
uv run python benchmarks/compare-v1.4/drive.py plan     # matrix and estimated duration
bash benchmarks/compare-v1.4/campaign.sh <run-id>       # setup, run, analyze
uv run python benchmarks/compare-v1.4/analyze.py --check
uv run python benchmarks/compare-v1.4/drive.py teardown --models   # remove envs and model files
```

Estimated duration on the reference machine (Apple M4 Max), from `drive.py plan`:
- first setup: about 35 minutes;
- campaign: about 3.6 h without energy sampling and 3.9 h with it (18 configurations).

`drive.py smoke` runs one golden case and one short latency window per configuration. Its
output stays under the scratch directory and is not committed.

Machine-specific paths (environments, snapshot directories, converted models) go to
`config.json` under the scratch directory (default `~/Developer/scratch/compare-v1.4`).
`raw/` holds measurements, versions and hashes only.

## Files

| File | Role |
|---|---|
| `runtimes.json` | pins, install lines, configurations |
| `make_shapes.py`, `shapes.json` | request shapes and their token counts (`--check`) |
| `adapter.py` | runs inside each runtime's environment; parity and latency tasks; JSON out |
| `drive.py` | `plan`, `setup`, `smoke`, `run`, `teardown` |
| `analyze.py` | `raw/` → `results.json`, `tables.md` (`--check`) |
| `campaign.sh` | the full campaign |

## Known limitations of this method

- laya-fast supports only the English `laya` checkpoint, so laya-fast rows exist for `laya`
  only.
- laya-coreml's `ane` bundle exists only for laya-multilingual and accepts at most 96 prompt
  tokens. Longer golden rows are refused and listed.
- laya-fast's lock file resolves torch 2.14.0 with coremltools 9.0. coremltools warns that it
  has not been tested with that torch version. The pinned lock is used as published.
- The published fixture sets have no committed goldens. Their reference is the upstream
  runtime run inside the same campaign, with the same version, device and dtype as the
  goldens.
- One machine. Other chips, memory sizes and macOS versions will differ.
