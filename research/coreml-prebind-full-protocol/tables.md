# R1: prebound predict under the full #57 product-mix protocol (fast fail, slow pass)

Pre-campaign check (`raw/check.json`): PB bit-identical everywhere True, C bit-identical everywhere True

| model | status | C at n=6 (reference) | runs present | not used for the verdict |
|---|---|---|---|---|
| laya | FUTILITY STOP at n=12 | FAIL | 10 | – |
| laya-typed-decisions | stopped (R1 outcome reached) | FAIL | 6 | – |

Outcome: FUTILITY STOP at n=12 on laya: short_p99; R1 FAIL; protocol-history experiment next

Crashed runs and re-runs (`raw/failed/`): none

- laya: C FAIL (n=6), PB FAIL: no statement on handoff reduction follows from this pair

## Interim looks: PB against A, futility only (99% t-intervals)

FUTILITY STOP if any mismatch, isolation fails, or a 99% interval lies entirely beyond a budget (short or long P99 lower bound > 1.05×, aggregate upper bound < 0.95×). Never a PASS.

| model | n | aggregate | short P99 | long P99 | GPU return P50 A / PB (gain) | mismatches A / PB | decision |
|---|---|---|---|---|---|---|---|
| laya | 6 | 0.917 [0.771, 1.091] INCONCLUSIVE | 1.362 [0.967, 1.918] INCONCLUSIVE | 1.011 [0.873, 1.171] INCONCLUSIVE | 4.367 / 0.201 (×21.7) | 0 / 0 | **CONTINUE** |
| laya | 12 | 0.910 [0.833, 0.994] INCONCLUSIVE | 1.409 [1.223, 1.623] FAIL | 1.034 [0.975, 1.097] INCONCLUSIVE | 4.349 / 0.214 (×20.3) | 0 / 0 | **FUTILITY STOP** (short_p99) |
| laya-typed-decisions | 6 | 1.044 [1.039, 1.049] PASS | 0.980 [0.879, 1.092] INCONCLUSIVE | 0.920 [0.867, 0.975] PASS | 8.500 / 0.040 (×212.5) | 0 / 0 | **CONTINUE** |

## Paired gate (95% t): PB at n=18 (the verdict), C at n=6 (reference only)

| model | candidate | pairs | aggregate ≥ 0.95× | short P99 ≤ 1.05× | long P99 ≤ 1.05× | GPU return P50 ≤ 1 ms and ≥ 5× vs A | correctness | verdict | bootstrap (aggregate / short / long) |
|---|---|---|---|---|---|---|---|---|---|
| laya | C (n=6) | 6 | 0.915 [0.814, 1.028] INCONCLUSIVE | 1.312 [1.119, 1.538] FAIL | 0.998 [0.919, 1.084] INCONCLUSIVE | ×79.4 PASS | PASS | **FAIL** | [0.844, 0.990] INCONCLUSIVE / [1.157, 1.423] FAIL / [0.934, 1.043] PASS |
| laya-typed-decisions | C (n=6) | 6 | 0.978 [0.938, 1.021] INCONCLUSIVE | 1.407 [1.109, 1.785] FAIL | 0.981 [0.930, 1.035] PASS | ×188.9 PASS | PASS | **FAIL** | [0.947, 1.005] INCONCLUSIVE / [1.184, 1.642] FAIL / [0.940, 1.004] PASS |

## Not gating: window history (the condition run just before each hetero window)

bench_concurrency alternates the order per cycle, so a hetero window follows solo_long in even cycles and gpu_only in odd cycles. Ratios: geometric mean of the matched-pair ratios to A (C on rounds 1–2).

| model | preceded by | A short P99 median | C short P99 median | PB short P99 median | C/A short P99 (pairs) | PB/A short P99 (pairs) | C/A aggregate | PB/A aggregate |
|---|---|---|---|---|---|---|---|---|
| laya | gpu_only | 11.16 | 13.46 | 16.72 | 1.143 (2) | 1.404 (4) | 0.962 | 0.925 |
| laya | solo_long | 11.07 | 16.09 | 16.84 | 1.405 (4) | 1.412 (8) | 0.893 | 0.902 |
| laya-typed-decisions | gpu_only | 11.36 | 13.69 | 11.58 | 1.184 (2) | 1.018 (2) | 1.002 | 1.042 |
| laya-typed-decisions | solo_long | 11.40 | 17.84 | 11.07 | 1.533 (4) | 0.961 (4) | 0.967 | 1.045 |

## Not gating: crossings, GIL re-acquire wait, ANE stages (hetero windows)

| model | config | crossings per forward | re-acquire P50 / P99 / mean ms | features | pre | native | re-acquire | post | tail | predict (whole) |
|---|---|---|---|---|---|---|---|---|---|---|
| laya | P | 0 (predict 0, sends 0, pool 0); re-entries 0 | – | 0.158 | – | – | – | – | 0.061 | 9.623 |
| laya | PB | 4 (predict 1, sends 0, pool 3); re-entries 0 | 0.004 / 1.239 / 0.036 | 0.638 | 0.085 | 9.670 | 0.004 | 0.032 | 0.309 | 9.809 |
| laya | C | 108 (predict 1, sends 104, pool 3); re-entries 62 | 0.003 / 1.924 / 0.100 | 0.165 | 0.134 | 9.499 | 0.003 | 0.100 | 0.061 | 9.738 |
| laya-typed-decisions | P | 0 (predict 0, sends 0, pool 0); re-entries 0 | – | 0.158 | – | – | – | – | 0.061 | 9.627 |
| laya-typed-decisions | PB | 4 (predict 1, sends 0, pool 3); re-entries 0 | 0.002 / 0.733 / 0.017 | 0.155 | 0.014 | 9.370 | 0.002 | 0.016 | 0.058 | 9.404 |
| laya-typed-decisions | C | 108 (predict 1, sends 104, pool 3); re-entries 62 | 0.002 / 0.372 / 0.015 | 0.156 | 0.125 | 9.443 | 0.002 | 0.099 | 0.055 | 9.671 |

## Not gating: GPU completion collisions and the ANE tail

Fixed exposure window: a GPU `received_ns` in [ANE service_start − 1 ms, service_start + 0.3 ms]. Tail: ANE request latency above A's pooled hetero P99. Odds ratio with 0.5 added to each cell.

| model | config | tail threshold ms | in tail | colliding | tail requests colliding | odds ratio | GPU completions overlapping a forward (mean / any) |
|---|---|---|---|---|---|---|---|
| laya | P | 11.27 | 0.010 | 0.249 | 0.277 | 1.2 | 0.194 / 0.194 |
| laya | PB | 11.27 | 0.572 | 0.040 | 0.047 | 1.6 | 0.284 / 0.284 |
| laya | C | 11.27 | 0.361 | 0.028 | 0.002 | 0.1 | 0.299 / 0.299 |
| laya-typed-decisions | P | 11.56 | 0.010 | 0.061 | 0.331 | 8.0 | 0.110 / 0.110 |
| laya-typed-decisions | PB | 11.56 | 0.007 | 0.018 | 0.037 | 2.4 | 0.134 / 0.134 |
| laya-typed-decisions | C | 11.56 | 0.111 | 0.010 | 0.029 | 3.7 | 0.142 / 0.142 |

## Not gating: thread CPU and the slow-CPU flag

Thread CPU: ms per hetero window (median). Slow-CPU flag: candidate windows whose client-short CPU per request exceeds 1.25× that of A's matched window.

| model | config | client-short | client-long | ane-dispatch | gpu-dispatch | ANE thread CPU/forward | GPU thread CPU/forward | slow-CPU windows |
|---|---|---|---|---|---|---|---|---|
| laya | P | 307 | 220 | 1256 | 59 | 0.598 | 1.798 | – (reference) |
| laya | PB | 1308 | 764 | 2932 | 223 | 1.192 | 6.084 | 1 of 1 computable (12 pairs) |
| laya | C | – | 651 | 2919 | 139 | 1.605 | 4.573 | 0 of 1 computable (6 pairs) |
| laya-typed-decisions | P | – | 225 | 1317 | 37 | 0.620 | 1.997 | – (reference) |
| laya-typed-decisions | PB | – | 236 | 839 | 28 | 0.369 | 1.924 | not computable (0 of 6) |
| laya-typed-decisions | C | – | 342 | 1752 | 42 | 0.948 | 2.878 | not computable (0 of 6) |
