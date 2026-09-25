# One prediction crossing on the ANE request path (hetero-only v1.0 closed-loop mix)

**Not the verdict.** contaminated-laya: runs from raw/contaminated/, reported alongside and outside the verdict (criteria.md, addendum).

| model | config | aggregate req/s | short req/s | short P99 | long req/s | long P99 | GPU return P50 / P99 | mismatches |
|---|---|---|---|---|---|---|---|---|
| laya | A (production) | 121.4 | 97.1 | 10.97 | 24.3 | 43.02 | 4.545 / 6.816 | 0 |
| laya | C | 125.9 | 98.1 | 10.90 | 27.9 | 36.32 | 0.041 / 0.255 | 0 |
| laya | PB | 128.5 | 100.7 | 10.50 | 27.8 | 36.50 | 0.045 / 0.257 | 0 |

## Criteria: paired gate (candidate vs the model's production configuration, per model)

Geometric mean of the matched-pair ratios (same round, same cycle) with its 95% t-interval (df = n − 1); PASS / FAIL / INCONCLUSIVE as in criteria.md.

| model | candidate | pairs | aggregate ≥ 0.95× | short P99 ≤ 1.05× | long P99 ≤ 1.05× | GPU return P50 ≤ 1 ms (and ≥ 5× vs A) | correctness | model |
|---|---|---|---|---|---|---|---|---|
| laya | C vs A | 6 | 1.037 [1.034, 1.039] PASS | 0.991 [0.967, 1.016] PASS | 0.844 [0.829, 0.860] PASS | ×110.9 PASS | PASS | **PASS** |
| laya | PB vs A | 6 | 1.060 [1.056, 1.064] PASS | 0.954 [0.933, 0.977] PASS | 0.848 [0.831, 0.865] PASS | ×101.0 PASS | PASS | **PASS** |

Sensitivity only (never the verdict): seeded percentile bootstrap over pairs, 10,000 resamples.

| model | candidate | aggregate | short P99 | long P99 |
|---|---|---|---|---|
| laya | C vs A | [1.035, 1.039] PASS | [0.974, 1.008] PASS | [0.835, 0.854] PASS |
| laya | PB vs A | [1.057, 1.063] PASS | [0.938, 0.969] PASS | [0.836, 0.860] PASS |

- PB PASS on laya: supports 'the many PyObjC/GIL handoffs are a significant execution-layer cost' for this model (not attributed to GPU-completion collisions alone)

Outcome: partial model set (laya): no preregistered outcome

C's verdicts are a same-campaign reference for PB. The paired gate applies from this preregistration on; it is not applied to #77, whose FAIL stands.

## Not gating: crossings, GIL re-acquire wait, ANE stages

Crossings: one forward on the short stream's bucket, counted at load (`crossings.py`; a lower bound for C). Stages: medians over in-process ANE forwards in hetero windows (A has no native stamps).

| model | config | crossings per forward | re-acquire P50 / P99 / mean ms | features | pre | native | re-acquire | post | tail | predict (whole) |
|---|---|---|---|---|---|---|---|---|---|---|
| laya | A | 0 (predict 0, sends 0, pool 0); re-entries 0 | – | 0.169 | – | – | – | – | 0.062 | 9.719 |
| laya | C | 108 (predict 1, sends 104, pool 3); re-entries 62 | 0.001 / 0.117 / 0.005 | 0.170 | 0.138 | 9.470 | 0.001 | 0.087 | 0.058 | 9.699 |
| laya | PB | 4 (predict 1, sends 0, pool 3); re-entries 0 | 0.001 / 0.061 / 0.005 | 0.174 | 0.016 | 9.383 | 0.001 | 0.007 | 0.064 | 9.409 |

## Not gating: GPU completion collisions and the ANE tail

Fixed exposure window: a GPU `received_ns` in [ANE service_start − 1 ms, service_start + 0.3 ms]. Tail: ANE request `response_ns − submit_ns` above the production configuration's P99 (hetero windows). Odds ratio with 0.5 added to each cell.

| model | config | tail threshold ms | in tail | colliding | tail requests colliding | odds ratio | GPU completions overlapping a forward (mean / any) | … overlapping its pre stage (mean / any) |
|---|---|---|---|---|---|---|---|---|
| laya | A | 10.98 | 0.010 | 0.243 | 0.291 | 1.3 | 0.233 / 0.233 | – |
| laya | C | 10.98 | 0.006 | 0.048 | 0.250 | 6.9 | 0.280 / 0.280 | 0.029 / 0.029 |
| laya | PB | 10.98 | 0.002 | 0.030 | 0.310 | 15.2 | 0.273 / 0.273 | 0.013 / 0.013 |

## Not gating: GIL probe, thread CPU, slow-CPU flag

GIL probe: lateness of a 1 ms sleep grid in the benchmark process (median over hetero windows of the per-window P99). Thread CPU: ms per hetero window (median). Slow-CPU flag: candidate windows whose client-short CPU per request exceeds 1.25x that of P's matched window.

| model | config | probe P99 ms | probe > 1 ms | probe CPU (core) | client-short | client-long | ane-dispatch | gpu-dispatch | ANE thread CPU/forward | GPU thread CPU/forward | slow-CPU windows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| laya | A | 10.375 | 0.7095 | 0.0031 | 290 | 231 | 1294 | 59 | 0.617 | 1.815 | – (reference) |
| laya | C | 0.584 | 0.0002 | 0.0094 | – | 256 | 1388 | 51 | 0.656 | 1.763 | 0 of 2 computable (6 pairs) |
| laya | PB | 0.624 | 0.0005 | 0.0086 | – | 271 | 887 | 55 | 0.383 | 1.855 | 0 of 2 computable (6 pairs) |
