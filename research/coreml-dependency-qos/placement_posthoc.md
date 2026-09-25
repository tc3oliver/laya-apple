# Post-hoc: where the hetero-onset E-core residency sits (#99 raw data only)

Not a gate; #99's outcome is unchanged. Definitions and the classification rule are in `scripts/placement_posthoc.py`, fixed before its output was read.

Overall class: **P3**. Per transition: B-r1 c0 P3, B-r1 c1 P3, O-r1 c0 P3, O-r1 c1 P3, Q-r1 c0 P3, Q-r1 c1 P3

## Group aggregates over W = [t0 + 0.5, t0 + 4) (active threads, CPU-weighted)

| run | cycle | after | group | active threads | CPU ms | E share | IPC | rate P / E | dominance |
|---|---|---|---|---|---|---|---|---|---|
| laya-B-r1 | 0 | solo_long | chain | 3 | 764.3 | 0.99 | 1.73 | 1.68 / 1.44 | E |
| laya-B-r1 | 0 | solo_long | parent-other | 2 | 250.1 | 0.99 | 2.67 | 1.79 / 1.24 | E |
| laya-B-r1 | 0 | solo_long | worker | 3 | 960.7 | 0.99 | 1.80 | 1.62 / 1.51 | E |
| laya-B-r1 | 1 | gpu_only | chain | 3 | 734.6 | 0.99 | 1.82 | 2.58 / 1.36 | E |
| laya-B-r1 | 1 | gpu_only | parent-other | 2 | 243.1 | 0.99 | 2.78 | 2.58 / 1.17 | E |
| laya-B-r1 | 1 | gpu_only | worker | 3 | 937.0 | 0.99 | 1.86 | 2.56 / 1.41 | E |
| laya-O-r1 | 0 | solo_long | chain | 3 | 780.1 | 1.00 | 1.78 | – / 1.39 | E |
| laya-O-r1 | 0 | solo_long | parent-other | 2 | 257.0 | 1.00 | 2.77 | – / 1.18 | E |
| laya-O-r1 | 0 | solo_long | worker | 4 | 971.1 | 1.00 | 1.84 | – / 1.45 | E |
| laya-O-r1 | 1 | gpu_only | chain | 3 | 791.3 | 1.00 | 1.77 | – / 1.36 | E |
| laya-O-r1 | 1 | gpu_only | parent-other | 2 | 252.2 | 1.00 | 2.77 | – / 1.18 | E |
| laya-O-r1 | 1 | gpu_only | worker | 3 | 973.0 | 1.00 | 1.87 | – / 1.42 | E |
| laya-Q-r1 | 0 | solo_long | chain | 3 | 413.1 | 0.70 | 2.62 | 3.48 / 1.47 | E |
| laya-Q-r1 | 0 | solo_long | parent-other | 2 | 126.5 | 0.76 | 3.73 | 4.01 / 1.25 | E |
| laya-Q-r1 | 0 | solo_long | worker | 3 | 552.7 | 0.67 | 2.54 | 3.03 / 1.53 | E |
| laya-Q-r1 | 1 | gpu_only | chain | 3 | 776.3 | 1.00 | 1.81 | – / 1.35 | E |
| laya-Q-r1 | 1 | gpu_only | parent-other | 2 | 261.3 | 1.00 | 2.77 | – / 1.14 | E |
| laya-Q-r1 | 1 | gpu_only | worker | 3 | 990.2 | 1.00 | 1.85 | – / 1.40 | E |

## Group E share by bucket (active threads, CPU-weighted)

| run | cycle | group | baseline -2-0 | 0-0.5 | 0.5-1 | 1-2 | 2-4 | 4-8 | late 10-20 |
|---|---|---|---|---|---|---|---|---|---|
| laya-B-r1 | 0 | chain | 1.00 | 0.93 | 1.00 | 0.95 | 1.00 | 0.99 | 0.98 |
| laya-B-r1 | 0 | parent-other | 1.00 | 0.98 | 1.00 | 0.98 | 1.00 | 0.99 | 0.98 |
| laya-B-r1 | 0 | worker | 0.95 | 0.86 | 1.00 | 0.96 | 1.00 | 0.99 | 0.98 |
| laya-B-r1 | 1 | chain | 0.66 | 0.98 | 1.00 | 1.00 | 0.98 | 1.00 | 0.90 |
| laya-B-r1 | 1 | parent-other | 0.66 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.92 |
| laya-B-r1 | 1 | worker | – | 0.93 | 1.00 | 1.00 | 0.98 | 1.00 | 0.90 |
| laya-O-r1 | 0 | chain | 0.90 | 0.92 | 1.00 | 1.00 | 1.00 | 0.09 | 0.00 |
| laya-O-r1 | 0 | parent-other | 0.92 | 0.96 | 1.00 | 1.00 | 1.00 | 0.12 | 0.00 |
| laya-O-r1 | 0 | worker | 1.00 | 0.86 | 1.00 | 1.00 | 1.00 | 0.08 | 0.00 |
| laya-O-r1 | 1 | chain | 0.93 | 0.92 | 1.00 | 1.00 | 1.00 | 0.99 | 0.45 |
| laya-O-r1 | 1 | parent-other | 0.92 | 0.97 | 1.00 | 1.00 | 1.00 | 0.99 | 0.50 |
| laya-O-r1 | 1 | worker | 0.86 | 0.86 | 1.00 | 1.00 | 1.00 | 0.99 | 0.43 |
| laya-Q-r1 | 0 | chain | 0.99 | 0.92 | 1.00 | 0.91 | 0.00 | 0.00 | 0.00 |
| laya-Q-r1 | 0 | parent-other | 0.99 | 0.97 | 1.00 | 0.94 | 0.00 | 0.00 | 0.00 |
| laya-Q-r1 | 0 | worker | 0.95 | 0.86 | 1.00 | 0.93 | 0.00 | 0.00 | 0.00 |
| laya-Q-r1 | 1 | chain | 0.79 | 0.92 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 |
| laya-Q-r1 | 1 | parent-other | 0.79 | 0.98 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 |
| laya-Q-r1 | 1 | worker | – | 0.85 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 |

## Every window: E share of all sampled CPU, parent and GPU worker

gpu_only runs on the separate GPU-only instance, whose threads are not sampled.

| run | cycle | condition | parent E share (CPU s) | worker E share (CPU s) |
|---|---|---|---|---|
| laya-B-r1 | 0 | solo_short | 0.00 (1.0) | 0.00 (0.0) |
| laya-B-r1 | 0 | solo_long | 0.01 (0.3) | 0.01 (1.6) |
| laya-B-r1 | 0 | hetero | 0.98 (5.7) | 0.98 (5.5) |
| laya-B-r1 | 0 | gpu_only | 0.02 (0.1) | 0.00 (0.0) |
| laya-B-r1 | 1 | gpu_only | 0.03 (0.1) | 0.00 (0.0) |
| laya-B-r1 | 1 | hetero | 0.96 (5.2) | 0.95 (5.0) |
| laya-B-r1 | 1 | solo_long | 0.04 (0.3) | 0.02 (1.6) |
| laya-B-r1 | 1 | solo_short | 0.03 (1.0) | 0.00 (0.0) |
| laya-O-r1 | 0 | solo_short | 0.00 (1.0) | 0.00 (0.0) |
| laya-O-r1 | 0 | solo_long | 0.01 (0.3) | 0.01 (1.6) |
| laya-O-r1 | 0 | hetero | 0.52 (2.3) | 0.46 (2.5) |
| laya-O-r1 | 0 | gpu_only | 0.02 (0.1) | 0.00 (0.0) |
| laya-O-r1 | 1 | gpu_only | 0.02 (0.1) | 0.00 (0.0) |
| laya-O-r1 | 1 | hetero | 0.86 (4.0) | 0.82 (3.9) |
| laya-O-r1 | 1 | solo_long | 0.04 (0.3) | 0.03 (1.6) |
| laya-O-r1 | 1 | solo_short | 0.03 (1.0) | 0.22 (0.0) |
| laya-Q-r1 | 0 | solo_short | 0.00 (1.0) | 0.00 (0.0) |
| laya-Q-r1 | 0 | solo_long | 0.01 (0.3) | 0.00 (1.6) |
| laya-Q-r1 | 0 | hetero | 0.29 (1.7) | 0.25 (2.0) |
| laya-Q-r1 | 0 | gpu_only | 0.02 (0.1) | 0.00 (0.0) |
| laya-Q-r1 | 1 | gpu_only | 0.02 (0.1) | 0.00 (0.0) |
| laya-Q-r1 | 1 | hetero | 0.99 (5.8) | 0.99 (5.5) |
| laya-Q-r1 | 1 | solo_long | 0.01 (0.3) | 0.00 (1.6) |
| laya-Q-r1 | 1 | solo_short | 0.02 (1.0) | 0.10 (0.0) |

## Per active thread: E share over W, and the E->P switch time

Switch: start (s from t0) of the first 0.5 s bin from which every later bin with CPU is P-dominant; `never` = still E-dominant somewhere up to t0 + 20 s.

| run | cycle | group | thread | onset CPU ms | E share W | IPC W | rate P / E W | switch s |
|---|---|---|---|---|---|---|---|---|
| laya-B-r1 | 0 | chain | client-short:26549318 | 241.7 | 0.99 | 2.26 | 1.70 / 1.44 | 19.5 |
| laya-B-r1 | 0 | chain | coreml-callback:26548819 | 95.0 | 0.99 | 0.75 | 1.62 / 1.41 | 19.5 |
| laya-B-r1 | 0 | chain | laya-ane-dispatch:26548830 | 514.8 | 0.99 | 1.67 | 1.69 / 1.45 | 19.5 |
| laya-B-r1 | 0 | parent-other | client-long:26549319 | 235.0 | 0.99 | 2.98 | 1.79 / 1.24 | never |
| laya-B-r1 | 0 | parent-other | laya-gpu-dispatch:26548851 | 47.8 | 0.99 | 1.19 | 1.76 / 1.25 | never |
| laya-B-r1 | 0 | worker | gpu-worker:26548522 | 796.2 | 0.99 | 2.00 | 1.68 / 1.51 | 19.5 |
| laya-B-r1 | 0 | worker | gpu-worker:26548833 | 165.5 | 0.98 | 1.24 | 1.55 / 1.48 | never |
| laya-B-r1 | 0 | worker | gpu-worker:26549324 | 142.4 | 0.99 | 1.23 | 1.50 / 1.47 | 19.5 |
| laya-B-r1 | 1 | chain | client-short:26549896 | 239.4 | 0.99 | 2.35 | 2.63 / 1.36 | 17.0 |
| laya-B-r1 | 1 | chain | coreml-callback:26548819 | 88.8 | 0.99 | 0.85 | 2.60 / 1.34 | 17.0 |
| laya-B-r1 | 1 | chain | laya-ane-dispatch:26548830 | 505.3 | 0.99 | 1.74 | 2.56 / 1.37 | 17.0 |
| laya-B-r1 | 1 | parent-other | client-long:26549897 | 233.4 | 0.99 | 3.06 | 2.59 / 1.17 | 17.5 |
| laya-B-r1 | 1 | parent-other | laya-gpu-dispatch:26548851 | 45.3 | 0.99 | 1.31 | 2.53 / 1.16 | 17.0 |
| laya-B-r1 | 1 | worker | gpu-worker:26548522 | 807.8 | 0.99 | 2.03 | 2.57 / 1.41 | 17.0 |
| laya-B-r1 | 1 | worker | gpu-worker:26548844 | 133.4 | 0.99 | 1.34 | 2.57 / 1.38 | 17.0 |
| laya-B-r1 | 1 | worker | gpu-worker:26549324 | 153.2 | 0.99 | 1.34 | 2.54 / 1.42 | never |
| laya-O-r1 | 0 | chain | client-short:26551802 | 257.2 | 1.00 | 2.25 | – / 1.39 | 4.0 |
| laya-O-r1 | 0 | chain | coreml-callback:26551161 | 96.5 | 1.00 | 0.86 | – / 1.35 | 4.0 |
| laya-O-r1 | 0 | chain | laya-ane-dispatch:26551217 | 525.1 | 1.00 | 1.72 | – / 1.40 | 4.0 |
| laya-O-r1 | 0 | parent-other | client-long:26551803 | 238.3 | 1.00 | 3.07 | – / 1.18 | 4.5 |
| laya-O-r1 | 0 | parent-other | laya-gpu-dispatch:26551232 | 47.6 | 1.00 | 1.26 | – / 1.17 | never |
| laya-O-r1 | 0 | worker | gpu-worker:26551136 | 831.4 | 1.00 | 2.01 | – / 1.45 | 4.5 |
| laya-O-r1 | 0 | worker | gpu-worker:26551219 | 101.1 | 1.00 | 1.33 | – / 1.44 | 4.0 |
| laya-O-r1 | 0 | worker | gpu-worker:26551468 | 163.1 | 1.00 | 1.31 | – / 1.43 | 4.0 |
| laya-O-r1 | 0 | worker | gpu-worker:26551756 | 39.0 | 1.00 | 1.35 | – / 1.41 | 6.0 |
| laya-O-r1 | 1 | chain | client-short:26552436 | 261.6 | 1.00 | 2.23 | – / 1.36 | 11.5 |
| laya-O-r1 | 1 | chain | coreml-callback:26551161 | 97.4 | 1.00 | 0.85 | – / 1.34 | 11.5 |
| laya-O-r1 | 1 | chain | laya-ane-dispatch:26551217 | 532.4 | 1.00 | 1.71 | – / 1.37 | 11.5 |
| laya-O-r1 | 1 | parent-other | client-long:26552437 | 235.0 | 1.00 | 3.07 | – / 1.19 | 12.0 |
| laya-O-r1 | 1 | parent-other | laya-gpu-dispatch:26551232 | 47.4 | 1.00 | 1.29 | – / 1.16 | never |
| laya-O-r1 | 1 | worker | gpu-worker:26551136 | 823.5 | 1.00 | 2.04 | – / 1.42 | 11.5 |
| laya-O-r1 | 1 | worker | gpu-worker:26551219 | 157.3 | 1.00 | 1.34 | – / 1.39 | 11.5 |
| laya-O-r1 | 1 | worker | gpu-worker:26551998 | 128.8 | 1.00 | 1.34 | – / 1.40 | never |
| laya-Q-r1 | 0 | chain | client-short:26553829 | 142.3 | 0.72 | 3.40 | 3.84 / 1.47 | 2.0 |
| laya-Q-r1 | 0 | chain | coreml-callback:26553290 | 57.7 | 0.66 | 1.14 | 1.81 / 1.43 | 2.0 |
| laya-Q-r1 | 0 | chain | laya-ane-dispatch:26553302 | 305.1 | 0.70 | 2.47 | 3.68 / 1.47 | 2.0 |
| laya-Q-r1 | 0 | parent-other | client-long:26553830 | 132.1 | 0.76 | 4.12 | 4.10 / 1.25 | 2.0 |
| laya-Q-r1 | 0 | parent-other | laya-gpu-dispatch:26553317 | 26.4 | 0.74 | 1.70 | 3.60 / 1.24 | never |
| laya-Q-r1 | 0 | worker | gpu-worker:26553261 | 477.5 | 0.72 | 2.85 | 3.64 / 1.54 | 2.0 |
| laya-Q-r1 | 0 | worker | gpu-worker:26553304 | 130.6 | 0.56 | 1.68 | 2.15 / 1.51 | 2.0 |
| laya-Q-r1 | 0 | worker | gpu-worker:26553835 | 95.9 | 0.55 | 1.68 | 2.13 / 1.48 | 2.0 |
| laya-Q-r1 | 1 | chain | client-short:26554593 | 249.8 | 1.00 | 2.31 | – / 1.34 | never |
| laya-Q-r1 | 1 | chain | coreml-callback:26553290 | 91.1 | 1.00 | 0.85 | – / 1.33 | never |
| laya-Q-r1 | 1 | chain | laya-ane-dispatch:26553302 | 522.5 | 1.00 | 1.73 | – / 1.35 | never |
| laya-Q-r1 | 1 | parent-other | client-long:26554594 | 246.9 | 1.00 | 3.07 | – / 1.14 | never |
| laya-Q-r1 | 1 | parent-other | laya-gpu-dispatch:26553317 | 48.4 | 1.00 | 1.29 | – / 1.13 | never |
| laya-Q-r1 | 1 | worker | gpu-worker:26553261 | 834.6 | 1.00 | 2.02 | – / 1.41 | never |
| laya-Q-r1 | 1 | worker | gpu-worker:26553985 | 167.9 | 1.00 | 1.33 | – / 1.36 | never |
| laya-Q-r1 | 1 | worker | gpu-worker:26554598 | 137.6 | 1.00 | 1.32 | – / 1.41 | never |
