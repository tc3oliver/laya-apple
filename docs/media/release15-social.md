# laya-apple 1.5 release media

`release15-social.mp4` (1920×1080, 30 fps, 28 s, silent, for social posts) and
`release15-social.gif` (960×540, 10 fps, 28 s, the README figure) tell the 1.5 story: the GPU
finished but its result waited on the GIL; asynchronous Core ML execution removes that wait;
the asynchronous path can then fall into a host-side slow state; 1.5 detects it and falls back
to the 1.4 path. They are made with Remotion outside this repository, so the runtime has no
Node dependency. Every number on screen is computed from the files below at tag `v1.5.0`; none
is typed in by hand.

| Input (at `v1.5.0`) | SHA-256 |
|---|---|
| [`research/coreml-adaptive-breaker/val_results.json`](../../research/coreml-adaptive-breaker/val_results.json) | `d0c016212a1a46379a6ef931210f7610b7d1fb9bba4a4c873dea8df8dcfc6e97` |
| [`research/coreml-adaptive-breaker/phase1_results.json`](../../research/coreml-adaptive-breaker/phase1_results.json) | `12f715327884177e2758346e646c3e75e53a7c5d369ba9622800b01c26682530` |
| [`research/coreml-staged-handoff/raw-eval/mix-laya-P-r5.json.gz`](../../research/coreml-staged-handoff/raw-eval/mix-laya-P-r5.json.gz) | `a91a471a6f81b10f38a64252ee7b620ebe22290f31db9804f0252c0031600c2a` |
| [`research/coreml-adaptive-breaker/raw/laya-R-r3.json.gz`](../../research/coreml-adaptive-breaker/raw/laya-R-r3.json.gz) | `a239a030c77ec9ab6ecfb29945a6a4064108e7da419ae4162c73039abb5a423f` |

## Timing

| Time | Shows |
|---|---|
| 0–3 s | The GPU finishes; its result hits the GIL. "The GPU finished. The result didn't." |
| 3–6 s | The completion path: the reply is read, the dispatcher waits in `take_gil` while the ANE thread's synchronous Core ML predict holds the GIL; the timer lands on 8.60 ms |
| 6–10 s | Asynchronous Core ML: the result goes straight through; 8.60 → 0.043 ms |
| 10–11 s | "Problem solved?" "Not quite." |
| 11–17 s | A recorded run: asynchronous execution after a fixed handoff, healthy, then a host-side slow state that holds to the end of the window |
| 17–18.5 s | "Static fixes weren't enough": process isolation, dispatcher QoS, fixed handoff |
| 18.5–23 s | The route switch (schematic) over a recorded episode of the controlled recovery test: detect, fall back, recover |
| 23–24.5 s | 12 / 12 already-slow episodes recovered, within 164–414 ms |
| 24.5–28 s | laya-apple 1.5: 154 validation episodes, 0 correctness failures, 12/12 controlled recoveries |

## Where the numbers come from

| Shown | Value | Source |
|---|---|---|
| GPU result return 8.60 → 0.043 ms (P50), ~200× | 8.600, 0.0426 ms | `val_results.json`, phase 3 (laya-typed-decisions), `stats.gpu_return_p50_ms.{A_median,P_healthy_median}` |
| The GPU reply is read, then the dispatcher waits in `take_gil` for the synchronous predict | — | [`research/coreml-gil-completion-path/README.md`](../../research/coreml-gil-completion-path/README.md) |
| Healthy 9.8 ms, slow 12.3 ms, onset 5.5 s, slow for 14.5 s, prepare > 0.3 ms 0% → 100% | medians and host-slow shares before and after the onset | `mix-laya-P-r5.json.gz`, hetero window 5: the episode [#103](https://github.com/tc3oliver/laya-apple/pull/103) reports falling into the slow state. The onset is the start of the first 0.5 s bin from which every later bin is at least 90% host-slow |
| Process isolation, dispatcher QoS, fixed handoff ✕ | no numbers on screen | the verdicts of [#57](https://github.com/tc3oliver/laya-apple/pull/57), [#99](https://github.com/tc3oliver/laya-apple/pull/99), [#102](https://github.com/tc3oliver/laya-apple/pull/102) and [#103](https://github.com/tc3oliver/laya-apple/pull/103) |
| The recovery chart: trip, back in range after 414 ms | `trip_s` 1.104, `recovered_at_s` 1.518 | `laya-R-r3.json.gz` window 0 and `phase1_results.json`: the episode named in `stats.worst_recovery` |
| 12 / 12, 164–414 ms | 12 tripped and recovered within 1 s; min and max `trip_to_recovery_ms` | `phase1_results.json`, `stats.R` and the R episodes |
| 154 validation episodes, 0 correctness failures | 12 + 78 + 64 | `val_results.json`, phases 2, 3 and 5: `stats.p_episodes`; every phase PASS with an empty `fail` list |

## Rules

- The first 10 s are a schematic, labelled on screen. The timer is one linear clock that lands
  on the measured P50.
- The two recorded traces come from different experiments and are framed apart. The 11–17 s
  run is a production-evaluation run with no breaker. The recovery chart sits in its own panel
  labelled "controlled recovery test".
- Latency axes are labelled in milliseconds. The slow state is shown at its measured size.
- GPU result return is not inference time, and the video says so where the number appears.
- Asynchronous Core ML is described as avoiding the synchronous path's long GIL hold, not as
  releasing the GIL. E-core residency is not on screen: its link to the slow state is a
  correlation.
