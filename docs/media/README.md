# Heterogeneous serving video

`heterogeneous-serving.mp4` (1920×1080, 30 s), `heterogeneous-serving.gif` (800×450,
5 fps, 30 s) and the poster frame `heterogeneous-serving.png` are cut from two recordings
on the tested Apple M4 Max (macOS 26.6.2):

| Time | Shows |
|---|---|
| 0–13 s | One recorded Lane Runner game played by the same model on the MLX GPU and on the Apple Neural Engine, side by side, then the results of all ten recorded runs per device |
| 13–30 s | One burst of the v1.0 open-loop bursty workload for laya-typed-decisions. The same 102 requests, with the same arrival times, are served GPU-only and then GPU + ANE |

The poster frame is from the second part.

## 0–13 s: Lane Runner, MLX GPU vs Apple Neural Engine

The game runs in [laya-playground-apple](https://github.com/tc3oliver/laya-playground-apple),
a fork of [laya-playground](https://github.com/wdobry/laya-playground) with laya-apple as its
backend. The model is upstream's checkpoint for that game, `convaiinnovations/laya`, not
laya-typed-decisions. Each device was measured alone; the video replays recorded run 1 of
10 per device, and part of the game is fast-forwarded 8×, labelled on screen.

| On screen | Source |
|---|---|
| The game, each decision and its latency | [`results/english/gpu-run-001.json` and `ane-run-001.json`](https://github.com/tc3oliver/laya-playground-apple/tree/6e0aa6718937e62ea58bc04f16b529f6089709ed/results/english) |
| Latency P50 9.2 / 8.2 ms, P95 9.7 / 8.3 ms, 185 rows cleared, 0 crashes | [`results/english/SUMMARY.md`](https://github.com/tc3oliver/laya-playground-apple/blob/6e0aa6718937e62ea58bc04f16b529f6089709ed/results/english/SUMMARY.md): 10 runs per device, 36,000 decisions each |

This part compares one request at a time. On all 10 seeds both devices ended the game
identically (rows cleared, crashes, score); 9 of 10 also played the same action sequence,
and the run shown is the one that did not. It says nothing about serving under load.

## 13–30 s: GPU-only vs GPU + ANE serving

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
