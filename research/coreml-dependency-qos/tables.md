# Dependency QoS screen: PB-ASYNC's hetero onset under B, O and Q (laya, L128 / L512)

Outcome: STOP this USER_INITIATED dispatcher-QoS route: no further QoS variants and no USER_INTERACTIVE trial-and-error; a USER_INTERACTIVE ceiling test may be opened later only as a separately preregistered diagnostic, by human decision, never as a mitigation candidate from this screen; the next step is re-evaluating the architecture or scheduler-state priming

Crashed runs and re-runs (`raw/failed/`): none


## Round 1: runs

| run | mismatches | routing failures | native mean ms | GPU return P50 ms | agg req/s per window | agg req/s pooled | dispatcher tid ok | QoS load / after set / end | overrides | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| laya-B-r1 | 0 | – | 9.848 | 0.238 | 100.9 / 103.6 | 102.2 | True | 0x15 / – / 0x15 | – | 189 |
| laya-O-r1 | 0 | – | 9.574 | 0.042 | 122.9 / 111.9 | 117.4 | True | 0x15 / – / 0x15 | 7664 starts / 7664 ends / 0 NULL / 0 end errors / 0 outstanding; start 1.6 / 15.1 µs, end 1.6 / 13.8 µs (P50 / P99); off-ANE 0 | 189 |
| laya-Q-r1 | 0 | – | 9.645 | 0.160 | 126.5 / 100.1 | 113.2 | True | 0x15 / 0x19 / 0x19 | – | 189 |

## Round 1: transitions

Onset [t0, t0 + 4 s); W = [t0 + 0.5 s, t0 + 4 s). E-resident: dispatcher E share over W >= 0.50; avoided: <= 0.10; between: partial (not avoided).

| run | cycle | after | onset E share dispatcher / client-short / callback | onset ANE CPU/fwd ms | W E share dispatcher / client-short / callback | class | CHAIN-E | transient s | peak short P99 ms |
|---|---|---|---|---|---|---|---|---|---|
| laya-B-r1 | 0 | solo_long | 0.98 / 0.98 / 0.98 | 1.402 | 0.99 / 0.99 / 0.98 | E-resident | True | 20.0 | 27.43 |
| laya-B-r1 | 1 | gpu_only | 0.99 / 0.99 / 0.99 | 1.431 | 0.99 / 0.99 / 0.99 | E-resident | True | 17.5 | 72.82 |
| laya-O-r1 | 0 | solo_long | 0.99 / 0.99 / 0.99 | 1.444 | 1.00 / 1.00 / 1.00 | E-resident | True | 4.5 | 28.02 |
| laya-O-r1 | 1 | gpu_only | 0.99 / 0.99 / 0.99 | 1.460 | 1.00 / 1.00 / 1.00 | E-resident | True | 12.0 | 28.88 |
| laya-Q-r1 | 0 | solo_long | 0.74 / 0.76 / 0.70 | 0.711 | 0.70 / 0.71 / 0.65 | E-resident | True | 2.0 | 28.07 |
| laya-Q-r1 | 1 | gpu_only | 0.99 / 0.99 / 0.99 | 1.464 | 1.00 / 1.00 / 1.00 | E-resident | True | 20.0 | 27.92 |

## Round 1: onset buckets (seconds from t0, #94's buckets)

| run | cycle | bucket | n | short P99 ms | host-slow | E share dispatcher / client-short / callback | ANE CPU/fwd ms | native ms |
|---|---|---|---|---|---|---|---|---|
| laya-B-r1 | 0 | -2-0 | 0 | – | – | 0.45 / 0.97 / 0.14 | – | – |
| laya-B-r1 | 0 | 0-0.5 | 37 | 42.24 | 0.97 | 0.94 / 0.95 / 0.93 | 1.239 | 10.978 |
| laya-B-r1 | 0 | 0.5-1 | 38 | 17.15 | 1.00 | 1.00 / 1.00 / 1.00 | 1.520 | 9.924 |
| laya-B-r1 | 0 | 1-2 | 79 | 17.30 | 0.97 | 0.95 / 0.95 / 0.94 | 1.295 | 9.854 |
| laya-B-r1 | 0 | 2-4 | 150 | 17.49 | 1.00 | 1.00 / 1.00 / 1.00 | 1.468 | 9.846 |
| laya-B-r1 | 0 | 4-5 | 73 | 17.50 | 1.00 | 1.00 / 1.00 / 1.00 | 1.602 | 9.932 |
| laya-B-r1 | 0 | steady 10-20 | 781 | 17.21 | 0.94 | 0.98 / 0.98 / 0.97 | 1.373 | 9.836 |
| laya-B-r1 | 1 | -2-0 | 0 | – | – | 0.83 / 0.68 / 0.76 | – | – |
| laya-B-r1 | 1 | 0-0.5 | 37 | 38.00 | 0.95 | 0.98 / 0.98 / 0.97 | 1.347 | 10.585 |
| laya-B-r1 | 1 | 0.5-1 | 39 | 16.91 | 1.00 | 1.00 / 1.00 / 1.00 | 1.378 | 9.833 |
| laya-B-r1 | 1 | 1-2 | 79 | 17.18 | 1.00 | 1.00 / 1.00 / 1.00 | 1.396 | 9.860 |
| laya-B-r1 | 1 | 2-4 | 140 | 50.42 | 0.96 | 0.98 / 0.99 / 0.99 | 1.486 | 10.055 |
| laya-B-r1 | 1 | 4-5 | 75 | 17.34 | 1.00 | 1.00 / 1.00 / 1.00 | 1.550 | 9.921 |
| laya-B-r1 | 1 | steady 10-20 | 833 | 17.18 | 0.65 | 0.90 / 0.91 / 0.88 | 1.057 | 9.707 |
| laya-O-r1 | 0 | -2-0 | 0 | – | – | 1.00 / 0.87 / – | – | – |
| laya-O-r1 | 0 | 0-0.5 | 35 | 42.27 | 0.97 | 0.92 / 0.93 / 0.91 | 1.443 | 11.155 |
| laya-O-r1 | 0 | 0.5-1 | 38 | 17.19 | 1.00 | 1.00 / 1.00 / 1.00 | 1.458 | 9.898 |
| laya-O-r1 | 0 | 1-2 | 77 | 17.02 | 1.00 | 1.00 / 1.00 / 1.00 | 1.454 | 9.820 |
| laya-O-r1 | 0 | 2-4 | 153 | 17.42 | 1.00 | 1.00 / 1.00 / 1.00 | 1.435 | 9.860 |
| laya-O-r1 | 0 | 4-5 | 98 | 16.68 | 0.09 | 0.32 / 0.33 / 0.27 | 0.398 | 9.473 |
| laya-O-r1 | 0 | steady 10-20 | 1011 | 10.38 | 0.00 | 0.00 / 0.00 / 0.00 | 0.264 | 9.406 |
| laya-O-r1 | 1 | -2-0 | 0 | – | – | 1.00 / 0.90 / 1.00 | – | – |
| laya-O-r1 | 1 | 0-0.5 | 37 | 42.08 | 1.00 | 0.92 / 0.92 / 0.89 | 1.301 | 10.965 |
| laya-O-r1 | 1 | 0.5-1 | 36 | 18.03 | 1.00 | 1.00 / 1.00 / 1.00 | 1.613 | 9.945 |
| laya-O-r1 | 1 | 1-2 | 77 | 17.09 | 1.00 | 1.00 / 1.00 / 1.00 | 1.444 | 9.872 |
| laya-O-r1 | 1 | 2-4 | 152 | 17.30 | 1.00 | 1.00 / 1.00 / 1.00 | 1.469 | 9.847 |
| laya-O-r1 | 1 | 4-5 | 75 | 17.14 | 1.00 | 1.00 / 1.00 / 1.00 | 1.500 | 9.899 |
| laya-O-r1 | 1 | steady 10-20 | 970 | 16.68 | 0.13 | 0.45 / 0.47 / 0.40 | 0.425 | 9.456 |
| laya-Q-r1 | 0 | -2-0 | 0 | – | – | 0.68 / 0.97 / 0.32 | – | – |
| laya-Q-r1 | 0 | 0-0.5 | 36 | 46.02 | 0.97 | 0.93 / 0.93 / 0.91 | 1.331 | 11.061 |
| laya-Q-r1 | 0 | 0.5-1 | 40 | 16.80 | 1.00 | 1.00 / 1.00 / 1.00 | 1.380 | 9.817 |
| laya-Q-r1 | 0 | 1-2 | 80 | 17.17 | 0.90 | 0.90 / 0.90 / 0.89 | 1.233 | 9.782 |
| laya-Q-r1 | 0 | 2-4 | 203 | 10.16 | 0.00 | 0.00 / 0.00 / 0.00 | 0.264 | 9.383 |
| laya-Q-r1 | 0 | 4-5 | 101 | 10.26 | 0.00 | 0.00 / 0.00 / 0.00 | 0.277 | 9.395 |
| laya-Q-r1 | 0 | steady 10-20 | 1013 | 10.29 | 0.00 | 0.00 / 0.00 / 0.00 | 0.263 | 9.387 |
| laya-Q-r1 | 1 | -2-0 | 0 | – | – | 0.78 / 0.79 / 0.00 | – | – |
| laya-Q-r1 | 1 | 0-0.5 | 35 | 42.32 | 0.97 | 0.93 / 0.94 / 0.91 | 1.379 | 11.168 |
| laya-Q-r1 | 1 | 0.5-1 | 39 | 17.53 | 1.00 | 1.00 / 1.00 / 1.00 | 1.367 | 9.862 |
| laya-Q-r1 | 1 | 1-2 | 76 | 17.13 | 1.00 | 1.00 / 1.00 / 1.00 | 1.475 | 9.923 |
| laya-Q-r1 | 1 | 2-4 | 150 | 17.10 | 1.00 | 1.00 / 1.00 / 1.00 | 1.503 | 9.912 |
| laya-Q-r1 | 1 | 4-5 | 77 | 16.90 | 1.00 | 1.00 / 1.00 / 1.00 | 1.460 | 9.867 |
| laya-Q-r1 | 1 | steady 10-20 | 768 | 17.36 | 0.99 | 0.99 / 0.99 / 0.99 | 1.440 | 9.872 |

## Round 1: B validity

Valid: **True**. Aggregate req/s range per hetero window: [96.5, 141.7].


## Round 1: candidates

| candidate | classes | MECH | LAT | CHAIN-E | placement change | failed guards | outcome |
|---|---|---|---|---|---|---|---|
| O | E-resident, E-resident | False | False | True | changed nothing | – | the dispatcher's placement did not change enough; no replication |
| Q | E-resident, E-resident | False | False | True | changed nothing | – | the dispatcher's placement did not change enough; no replication |

After round 1: STOP this USER_INITIATED dispatcher-QoS route: no further QoS variants and no USER_INTERACTIVE trial-and-error; a USER_INTERACTIVE ceiling test may be opened later only as a separately preregistered diagnostic, by human decision, never as a mitigation candidate from this screen; the next step is re-evaluating the architecture or scheduler-state priming

