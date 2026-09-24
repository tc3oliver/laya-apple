# Switchyard-v1 benchmark

`laya-apple switchyard`: a headless, open-loop benchmark rendered as a rail-junction game.
Trains arrive on a seeded timetable while background long-context requests load the GPU; each
train is one Laya routing decision. This directory is the official campaign for the frozen
`switchyard-v1` workload — not a demo run.

## Method

The workload, measurement boundaries, round order, ANE-state classification and the
`result.json` schema are documented for readers in
[`docs/switchyard.md`](../../docs/switchyard.md). This README does not restate them; it only
reports numbers measured against that contract.

## Criteria and definitions

- **Late.** Decision latency (`response_ns - arrival_ns`) > 100 ms.
- **Misrouted.** On time, but the answer differs from the oracle platform.
- **Delivered.** On time and correct.
- **Miss rate at N ms.** Fraction of trains whose decision latency exceeds N, at 25, 50, 100,
  250 and 500 ms.
- **Metric order** (`docs/switchyard.md`): late, P99 decision latency, P99 queue wait,
  delivered, P50 / P95. There is no composite score — throughput, latency and correctness are
  never merged into one PASS/FAIL.
- **Configs.** `gpu_only` and `hybrid` (GPU + ANE) run the identical seeded timetable. Round
  order is fixed by seed parity (`design.ordering` in `result.json`), not alternated between
  runs: with the standard seed (11, odd), `hybrid` always runs first. `design.counterbalance`
  is `"none"` — order effects are not controlled for in v1, and a systematic first-round vs.
  second-round difference cannot be distinguished from a `gpu_only` vs. `hybrid` difference
  from this campaign alone.
- A run reports a single config only when the ANE state is not `ready`
  (`ane.state` / `ane.reason` in `result.json`); this is expected on a machine without a
  calibrated ANE profile, not a failure.

## Reproduce

Each machine's campaign is its own new directory under `benchmarks/switchyard/`, named for the
machine it ran on (e.g. `v1-m4-max`):

```sh
# idle machine on AC power, other GPU/ANE consumers stopped (local model servers, browsers)
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh benchmarks/switchyard/run.sh benchmarks/switchyard/v1-m4-max
uv run python scripts/switchyard_report.py benchmarks/switchyard/v1-m4-max
# or, after adding this campaign's <!-- switchyard-report:begin/end <label> --> markers below:
uv run python scripts/switchyard_report.py --update-readme benchmarks/switchyard/v1-m4-max
uv run python scripts/switchyard_report.py --check benchmarks/switchyard/v1-m4-max
```

`run.sh` refuses to start if the campaign directory already exists and has data (raw data is
never overwritten — start a new campaign directory instead), and it refuses to start if another
laya-apple process is already running. It writes three runs to
`<campaign>/raw/run-001/`, `<campaign>/raw/run-002/`, `<campaign>/raw/run-003/`, and does not
stop or start any service itself.

Each campaign gets its own `### <campaign-dir-name>` subsection here, with its own
`<!-- switchyard-report:begin <label> -->` / `<!-- switchyard-report:end <label> -->` markers
(`label` = the campaign directory's name) for `switchyard_report.py --check` /
`--update-readme` to fill. Add the subsection and markers for a new campaign by hand before
running `--update-readme` for it the first time.

## Results

### v1-m4-max

<!-- switchyard-report:begin v1-m4-max -->

## Switchyard-v1 report

### raw/run-001/result.json

- standard: `True` · ANE state: `ready` · round order: `seed_parity` ['hybrid', 'gpu_only'] · counterbalance: `none` · decision disagreements: 0

| config | late (count/rate) | P99 decision ms | P99 queue ms | delivered | P50 / P95 ms | misrouted | miss@25/50/100/250/500 ms |
|---|---|---:|---:|---:|---:|---:|---|
| hybrid | 0 / 0.000 | 54.60 | 39.71 | 1422 | 14.70 / 41.16 | 0 | 0.213/0.021/0.000/0.000/0.000 |
| gpu_only | 1408 / 0.990 | 3170.55 | 3158.82 | 14 | 1498.85 / 2840.39 | 0 | 0.997/0.994/0.990/0.969/0.892 |

### raw/run-002/result.json

- standard: `True` · ANE state: `ready` · round order: `seed_parity` ['hybrid', 'gpu_only'] · counterbalance: `none` · decision disagreements: 0

| config | late (count/rate) | P99 decision ms | P99 queue ms | delivered | P50 / P95 ms | misrouted | miss@25/50/100/250/500 ms |
|---|---|---:|---:|---:|---:|---:|---|
| hybrid | 0 / 0.000 | 54.48 | 39.64 | 1422 | 14.74 / 41.01 | 0 | 0.210/0.023/0.000/0.000/0.000 |
| gpu_only | 1408 / 0.990 | 3107.72 | 3095.94 | 14 | 1472.90 / 2816.46 | 0 | 0.998/0.994/0.990/0.969/0.895 |

### raw/run-003/result.json

- standard: `True` · ANE state: `ready` · round order: `seed_parity` ['hybrid', 'gpu_only'] · counterbalance: `none` · decision disagreements: 0

| config | late (count/rate) | P99 decision ms | P99 queue ms | delivered | P50 / P95 ms | misrouted | miss@25/50/100/250/500 ms |
|---|---|---:|---:|---:|---:|---:|---|
| hybrid | 0 / 0.000 | 54.68 | 42.82 | 1422 | 14.89 / 41.35 | 0 | 0.212/0.021/0.000/0.000/0.000 |
| gpu_only | 1407 / 0.989 | 3137.24 | 3125.58 | 15 | 1488.20 / 2825.74 | 0 | 0.997/0.994/0.989/0.969/0.895 |

### Cross-run spread (min–max)

| config | runs | late count | P99 decision ms | P99 queue ms | delivered | P50 ms | P95 ms |
|---|---:|---|---|---|---|---|---|
| gpu_only | 3 | 1407–1408 | 3107.72–3170.55 | 3095.94–3158.82 | 14–15 | 1472.90–1498.85 | 2816.46–2840.39 |
| hybrid | 3 | 0 | 54.48–54.68 | 39.64–42.82 | 1422 | 14.70–14.89 | 41.01–41.35 |
<!-- switchyard-report:end v1-m4-max -->
