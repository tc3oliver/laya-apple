# Heterogeneous serving video

`heterogeneous-serving.mp4` (1920×1080, 20 s), `heterogeneous-serving.gif` (960×540) and
the poster frame `heterogeneous-serving.png` show one burst of the v1.0 open-loop bursty
workload for laya-typed-decisions on the tested Apple M4 Max (macOS 26.6.2). The same 102
requests, with the same arrival times, are served GPU-only and then GPU + ANE.

| On screen | Source |
|---|---|
| Short-request P99 1592.9 → 79.5 ms, 0 answer mismatches, 35.8 req/s offered | The published result: [`benchmarks/v1.0/concurrency-laya-typed-decisions.json`](../../benchmarks/v1.0/concurrency-laya-typed-decisions.json), `part_b["bursty@35"]`, also in [`benchmarks/v1.0.md`](../../benchmarks/v1.0.md) |
| The per-request timeline: arrival, start, finish and device of every request in the burst | [`heterogeneous-serving-trace.json`](heterogeneous-serving-trace.json), a per-request re-run of the same workload |

The published benchmark stores per-class percentiles, not individual requests, so the
timeline comes from a re-run. The re-run uses the benchmark's own arrival generator
(`arrivals()` in `scripts/bench_concurrency.py`, seed 11), class picks (seed 7) and
request shapes, with laya-apple 1.0.2. Its short-request P99 (1591.4 ms GPU-only,
52.9 ms GPU + ANE) is recorded in the trace metadata. The headline P99 on screen is always
the published number, not the re-run.

- Each request runs on exactly one device.
- The playback is 3.4× slower than real time. The mapping keeps every ordering and every
  ratio between waiting and execution time.
- A single short request takes about the same time on either device (9.9 ms on the ANE
  against 12.2 ms on MLX, laya-typed-decisions at 128 tokens). The difference in the video
  is queueing.
