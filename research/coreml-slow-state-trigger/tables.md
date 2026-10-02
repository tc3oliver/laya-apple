# Slow-state trigger test (laya, L128 / L512, R1's full protocol)

Pre-campaign check (`raw/check.json`): PB-R bit-identical True, PB-H bit-identical True, PB-R releases the GIL True (probe lateness P50 0.338 ms), PB-H holds the GIL True (lateness P50 8.947 ms, native predict P50 9.487 ms); all ok True

Outcome: round 2: PB-H confirmed; PB-W confirmed; PB-P confirmed (PB-R 5 of 6 slow, A 0 of 6 slow)

- PB-H: GIL release is one necessary condition for the slow-state transition; next: who takes CPU/GIL during the GIL-released period
- PB-W: hetero transition / scheduler residency state is key; next: minimal conditioning/state-retention mechanism
- PB-P: the 1 ms probe changes scheduler/CPU residency; next: minimal wake mechanism and its overhead; the probe is not a production fix

Next: none

Crashed runs and re-runs (`raw/failed/`): none

Runs not used by the rule: none


## Hetero windows

Slow: short P99 ≥ 13.0 ms. CPU: ms per window (client threads), ms per forward (ANE and GPU executing threads). Stages: P50 ms.

| run | cycle | short P99 | slow | aggregate req/s | client-short CPU | client-long CPU | ANE CPU/fwd | GPU CPU/fwd | features | native | re-acquire | tail | predict | probe P50 / P99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laya-A-r1 | 0 | 11.74 | normal | 120.9 | 313 | 228 | 0.644 | 1.800 | 0.162 | – | – | 0.062 | 9.685 | – |
| laya-A-r1 | 1 | 11.94 | normal | 120.7 | 301 | 223 | 0.629 | 1.764 | 0.160 | – | – | 0.062 | 9.682 | – |
| laya-A-r1 | 2 | 11.35 | normal | 121.2 | 303 | 217 | 0.635 | 1.775 | 0.163 | – | – | 0.063 | 9.686 | – |
| laya-A-r2 | 0 | 10.98 | normal | 121.7 | 284 | 217 | 0.597 | 1.689 | 0.157 | – | – | 0.060 | 9.690 | – |
| laya-A-r2 | 1 | 11.41 | normal | 120.9 | 307 | 227 | 0.645 | 1.772 | 0.164 | – | – | 0.062 | 9.694 | – |
| laya-A-r2 | 2 | 11.08 | normal | 121.8 | 286 | 218 | 0.600 | 1.693 | 0.157 | – | – | 0.061 | 9.678 | – |
| laya-PB-H-r1 | 0 | 10.53 | normal | 124.7 | 287 | 223 | 0.383 | 1.686 | 0.157 | 9.432 | 0.002 | 0.062 | 9.458 | – |
| laya-PB-H-r1 | 1 | 10.89 | normal | 124.4 | 299 | 231 | 0.396 | 1.736 | 0.160 | 9.436 | 0.002 | 0.062 | 9.463 | – |
| laya-PB-H-r1 | 2 | 10.54 | normal | 124.7 | 285 | 221 | 0.382 | 1.670 | 0.157 | 9.433 | 0.002 | 0.061 | 9.458 | – |
| laya-PB-H-r2 | 0 | 10.58 | normal | 124.7 | 288 | 222 | 0.383 | 1.685 | 0.158 | 9.434 | 0.002 | 0.062 | 9.460 | – |
| laya-PB-H-r2 | 1 | 10.71 | normal | 124.6 | 293 | 228 | 0.384 | 1.716 | 0.160 | 9.429 | 0.001 | 0.062 | 9.455 | – |
| laya-PB-H-r2 | 2 | 10.60 | normal | 124.9 | 313 | 247 | 0.376 | 1.696 | 0.177 | 9.381 | 0.001 | 0.056 | 9.403 | – |
| laya-PB-P-r1 | 0 | 10.40 | normal | 128.9 | 289 | 241 | 0.364 | 1.647 | 0.157 | 9.426 | 0.001 | 0.061 | 9.449 | 0.342 / 0.582 |
| laya-PB-P-r1 | 1 | 10.39 | normal | 128.8 | 286 | 242 | 0.367 | 1.655 | 0.158 | 9.430 | 0.001 | 0.061 | 9.452 | 0.342 / 0.578 |
| laya-PB-P-r1 | 2 | 10.39 | normal | 128.9 | 285 | 244 | 0.367 | 1.649 | 0.158 | 9.428 | 0.001 | 0.061 | 9.451 | 0.342 / 0.573 |
| laya-PB-P-r2 | 0 | 10.41 | normal | 128.7 | 298 | 247 | 0.371 | 1.698 | 0.159 | 9.441 | 0.001 | 0.061 | 9.464 | 0.341 / 0.583 |
| laya-PB-P-r2 | 1 | 10.47 | normal | 128.5 | 306 | 257 | 0.380 | 1.699 | 0.171 | 9.440 | 0.001 | 0.060 | 9.464 | 0.340 / 0.585 |
| laya-PB-P-r2 | 2 | 10.47 | normal | 128.5 | 302 | 256 | 0.378 | 1.698 | 0.166 | 9.446 | 0.001 | 0.061 | 9.469 | 0.340 / 0.576 |
| laya-PB-R-r1 | 0 | 12.13 | normal | 127.7 | 332 | 273 | 0.426 | 1.867 | 0.159 | 9.458 | 0.002 | 0.063 | 9.488 | – |
| laya-PB-R-r1 | 1 | 16.58 | **slow** | 120.1 | 602 | 538 | 0.760 | 3.548 | 0.158 | 9.474 | 0.002 | 0.062 | 9.503 | – |
| laya-PB-R-r1 | 2 | 16.96 | **slow** | 103.5 | 1199 | 1130 | 1.715 | 7.849 | 0.789 | 9.835 | 0.006 | 0.367 | 9.993 | – |
| laya-PB-R-r2 | 0 | 16.52 | **slow** | 122.0 | 537 | 483 | 0.680 | 3.152 | 0.158 | 9.454 | 0.002 | 0.063 | 9.484 | – |
| laya-PB-R-r2 | 1 | 16.88 | **slow** | 103.5 | 1230 | 1161 | 1.748 | 8.054 | 0.791 | 9.813 | 0.006 | 0.366 | 9.973 | – |
| laya-PB-R-r2 | 2 | 16.91 | **slow** | 101.3 | 1309 | 1244 | 1.896 | 8.736 | 0.798 | 9.827 | 0.006 | 0.377 | 9.986 | – |
| laya-PB-W-r1 | 0 | 10.38 | normal | 128.9 | 289 | 240 | 0.359 | 1.625 | 0.155 | 9.457 | 0.002 | 0.060 | 9.486 | – |
| laya-PB-W-r1 | 1 | 10.43 | normal | 128.6 | 294 | 246 | 0.371 | 1.679 | 0.155 | 9.461 | 0.002 | 0.061 | 9.491 | – |
| laya-PB-W-r1 | 2 | 10.38 | normal | 129.0 | 287 | 241 | 0.357 | 1.619 | 0.155 | 9.461 | 0.002 | 0.060 | 9.489 | – |
| laya-PB-W-r2 | 0 | 10.42 | normal | 129.0 | 297 | 249 | 0.367 | 1.659 | 0.157 | 9.439 | 0.001 | 0.059 | 9.466 | – |
| laya-PB-W-r2 | 1 | 10.40 | normal | 129.1 | 296 | 248 | 0.364 | 1.654 | 0.159 | 9.432 | 0.001 | 0.059 | 9.459 | – |
| laya-PB-W-r2 | 2 | 10.38 | normal | 129.3 | 308 | 265 | 0.357 | 1.634 | 0.174 | 9.377 | 0.001 | 0.054 | 9.400 | – |

## Runs

| run | slow windows | GPU return P50 ms | mismatches | routing failures | predicts with native stamps | warm-ups (short / long requests, mismatches) | wall s |
|---|---|---|---|---|---|---|---|
| laya-A-r1 | 0 of 3 | 4.678 | 0 | 0 | 0 of 5805 | – | 276 |
| laya-A-r2 | 0 of 3 | 4.694 | 0 | 0 | 0 of 5832 | – | 275 |
| laya-PB-H-r1 | 0 of 3 | 3.693 | 0 | 0 | 5982 of 5982 | – | 276 |
| laya-PB-H-r2 | 0 of 3 | 3.674 | 0 | 0 | 5987 of 5987 | – | 276 |
| laya-PB-P-r1 | 0 of 3 | 0.034 | 0 | 0 | 6058 of 6058 | – | 276 |
| laya-PB-P-r2 | 0 of 3 | 0.037 | 0 | 0 | 6038 of 6038 | – | 276 |
| laya-PB-R-r1 | 2 of 3 | 0.051 | 0 | 0 | 5456 of 5456 | – | 276 |
| laya-PB-R-r2 | 3 of 3 | 0.239 | 0 | 0 | 5044 of 5044 | – | 276 |
| laya-PB-W-r1 | 0 of 3 | 0.033 | 0 | 0 | 6057 of 6057 | c0: 198 / 56, 0; c1: 199 / 56, 0; c2: 198 / 56, 0 | 282 |
| laya-PB-W-r2 | 0 of 3 | 0.037 | 0 | 0 | 6072 of 6072 | c0: 199 / 56, 0; c1: 199 / 56, 0; c2: 199 / 56, 0 | 282 |

## Cells, pooled over their runs

| cell | rounds | slow windows | GPU return P50 ms | features | pre | native | re-acquire | post | tail | ANE CPU/fwd | GPU CPU/fwd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1, 2 | 0 of 6 | 4.685 | 0.160 | – | – | – | – | 0.062 | 0.625 | 1.749 |
| PB-R | 1, 2 | 5 of 6 | 0.174 | 0.475 | 0.050 | 9.634 | 0.004 | 0.026 | 0.164 | 1.144 | 5.363 |
| PB-H | 1, 2 | 0 of 6 | 3.685 | 0.161 | 0.015 | 9.427 | 0.001 | 0.007 | 0.061 | 0.384 | 1.698 |
| PB-W | 1, 2 | 0 of 6 | 0.035 | 0.157 | 0.014 | 9.442 | 0.001 | 0.007 | 0.059 | 0.362 | 1.645 |
| PB-P | 1, 2 | 0 of 6 | 0.035 | 0.160 | 0.014 | 9.436 | 0.001 | 0.006 | 0.061 | 0.371 | 1.674 |
