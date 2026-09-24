# Switchyard images

`switchyard-gpu-only.png`, `switchyard-gpu-ane.png` and `switchyard-result.png` (each
3200×1800, a 1600×900 page at 2×) come from one recorded Switchyard run on an Apple M4 Max
with macOS 26.6.2:
[`benchmarks/switchyard/v1-m4-max/raw/run-001/`](../../benchmarks/switchyard/v1-m4-max/raw/run-001/).

| Input | SHA-256 |
|---|---|
| `result.json` | `61a9e9d86f204cf6d8620bcdecaa7064f5d1e6b6bbd7344a0943200491daace2` |
| `trace.jsonl` | `c4029d9d482ee4654dd0baa4933da22a81ddda7698b636383e87accbf24e2570` |

The run is standard: workload switchyard-v1, seed 11, 60 s per round, laya-typed-decisions
at revision `f9ab0b2`, laya-apple 1.1.0, rounds in the order GPU + ANE, then GPU only.

## How they were made

The replay page was rebuilt from a copy of the run directory, so the raw data was not
touched:

```bash
laya-apple switchyard --replay <copy of run-001> --no-open
```

laya-apple was the uncommitted working tree of the `feat/switchyard` branch (base
`cd59fb6`), which includes the replay page's signal fix: a signal is green only while the
train that just got its answer passes it. Headless Chrome, driven over the DevTools
Protocol with a 1600×900 viewport at device scale factor 2, then opened that page.

| Image | What it shows | Page state |
|---|---|---|
| `switchyard-gpu-only.png` | The GPU-only round in the middle of a rush hour. The 11th burst starts at 30.0 s | round 2 (GPU only), data time 30.8 s, paused |
| `switchyard-gpu-ane.png` | The GPU + ANE round at the same data time, on the same timetable | round 1 (GPU + ANE), data time 30.8 s, paused |
| `switchyard-result.png` | The result card | produced by the page's own **Save PNG** button, which draws the card at 2× |

## Where the numbers come from

- **Result card:** every number comes from `run-001/result.json`:
  - Late trains: `configs.<label>.summary.late`, with the train count from `configs.<label>.game.trains`.
  - P99 decision latency: `summary.decision_latency.p99_ms`.
  - P99 queue wait: `summary.queue_wait.p99_ms`.
  - The "Slower than" row: `summary.miss_rate_at_ms`.
  - The footer: `machine.platform`, `model`, `workload.seed`, `design.sequence` and `laya_apple`.
  - Percentages are rounded, never to 0% or 100% unless the value is exactly 0 or 1. Below 10% and above 99% they keep one decimal, so the GPU-only 25 ms rate of 0.997 reads 99.7%.
- **Replay screenshots:** the status row shows running values at 30.8 s: trains late so far, and the P99 of the answers received so far. The page computes them from the trains in the replay data (built from `trace.jsonl`), so they are not the final round totals.
- **Train positions:** every position in the rail yard comes from that train's recorded times. A train reaches its queue at `arrival_ms` and leaves the signal at `response_ms`.
