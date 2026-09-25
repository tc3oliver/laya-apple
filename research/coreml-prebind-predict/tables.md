# One prediction crossing on the ANE request path (hetero-only v1.0 closed-loop mix)

| model | config | aggregate req/s | short req/s | short P99 | long req/s | long P99 | GPU return P50 / P99 | mismatches |
|---|---|---|---|---|---|---|---|---|
| laya | A (production) | 122.3 | 97.8 | 10.76 | 24.5 | 41.82 | 4.390 / 5.615 | 0 |
| laya | C | 126.3 | 98.5 | 10.80 | 27.8 | 36.20 | 0.038 / 0.253 | 0 |
| laya | PB | 129.5 | 101.6 | 10.29 | 27.9 | 36.13 | 0.034 / 0.176 | 0 |
| laya-typed-decisions | A (production) | 110.4 | 98.1 | 11.13 | 12.3 | 82.03 | 8.779 / 9.550 | 0 |
| laya-typed-decisions | C | 113.0 | 99.1 | 11.11 | 13.9 | 72.44 | 0.044 / 0.257 | 0 |
| laya-typed-decisions | PB | 115.8 | 101.9 | 10.76 | 13.9 | 72.41 | 0.039 / 0.270 | 0 |
| laya-multilingual | B (production) | 242.6 | 211.2 | 5.85 | 31.5 | 32.85 | 0.085 / 0.365 | 0 |
| laya-multilingual | C | 188.0 | 158.2 | 9.02 | 29.8 | 35.75 | 0.165 / 0.899 | 0 |
| laya-multilingual | PB | 293.4 | 259.9 | 4.41 | 33.5 | 30.13 | 0.032 / 0.727 | 0 |

## Criteria: paired gate (candidate vs the model's production configuration, per model)

Geometric mean of the matched-pair ratios (same round, same cycle) with its 95% t-interval (df = n − 1); PASS / FAIL / INCONCLUSIVE as in criteria.md.

| model | candidate | pairs | aggregate ≥ 0.95× | short P99 ≤ 1.05× | long P99 ≤ 1.05× | GPU return P50 ≤ 1 ms (and ≥ 5× vs A) | correctness | model |
|---|---|---|---|---|---|---|---|---|
| laya | C vs A | 6 | 1.032 [1.029, 1.036] PASS | 1.004 [1.000, 1.008] PASS | 0.864 [0.858, 0.870] PASS | ×115.5 PASS | PASS | **PASS** |
| laya | PB vs A | 6 | 1.060 [1.057, 1.062] PASS | 0.957 [0.953, 0.962] PASS | 0.862 [0.857, 0.867] PASS | ×129.1 PASS | PASS | **PASS** |
| laya-typed-decisions | C vs A | 6 | 1.022 [1.020, 1.024] PASS | 0.996 [0.990, 1.002] PASS | 0.882 [0.880, 0.885] PASS | ×199.5 PASS | PASS | **PASS** |
| laya-typed-decisions | PB vs A | 6 | 1.049 [1.047, 1.051] PASS | 0.963 [0.952, 0.974] PASS | 0.882 [0.881, 0.883] PASS | ×225.1 PASS | PASS | **PASS** |
| laya-multilingual | C vs B | 6 | 0.833 [0.698, 0.995] INCONCLUSIVE | 1.423 [1.063, 1.904] FAIL | 1.066 [0.976, 1.164] INCONCLUSIVE | ≤ 1 ms only (vs B) PASS | PASS | **FAIL** |
| laya-multilingual | PB vs B | 6 | 1.171 [0.905, 1.515] INCONCLUSIVE | 0.851 [0.524, 1.384] INCONCLUSIVE | 0.952 [0.812, 1.116] INCONCLUSIVE | ≤ 1 ms only (vs B) PASS | PASS | **INCONCLUSIVE** |

Sensitivity only (never the verdict): seeded percentile bootstrap over pairs, 10,000 resamples.

| model | candidate | aggregate | short P99 | long P99 |
|---|---|---|---|---|
| laya | C vs A | [1.030, 1.035] PASS | [1.002, 1.007] PASS | [0.860, 0.867] PASS |
| laya | PB vs A | [1.058, 1.061] PASS | [0.955, 0.960] PASS | [0.859, 0.866] PASS |
| laya-typed-decisions | C vs A | [1.021, 1.024] PASS | [0.991, 0.999] PASS | [0.881, 0.884] PASS |
| laya-typed-decisions | PB vs A | [1.048, 1.050] PASS | [0.954, 0.970] PASS | [0.881, 0.883] PASS |
| laya-multilingual | C vs B | [0.743, 0.934] FAIL | [1.173, 1.726] FAIL | [1.007, 1.128] INCONCLUSIVE |
| laya-multilingual | PB vs B | [0.961, 1.364] PASS | [0.636, 1.247] INCONCLUSIVE | [0.868, 1.079] INCONCLUSIVE |

- PB PASS on laya: supports 'the many PyObjC/GIL handoffs are a significant execution-layer cost' for this model (not attributed to GPU-completion collisions alone)
- PB PASS on laya-typed-decisions: supports 'the many PyObjC/GIL handoffs are a significant execution-layer cost' for this model (not attributed to GPU-completion collisions alone)
- PB INCONCLUSIVE on laya-multilingual: reported as inconclusive; no production change follows; a larger preregistered replication (for example 9 cycles, n = 18 pairs) decides

Outcome: PB INCONCLUSIVE on laya-multilingual: no production change; a larger preregistered replication decides

C's verdicts are a same-campaign reference for PB. The paired gate applies from this preregistration on; it is not applied to #77, whose FAIL stands.

## Not gating: crossings, GIL re-acquire wait, ANE stages

Crossings: one forward on the short stream's bucket, counted at load (`crossings.py`; a lower bound for C). Stages: medians over in-process ANE forwards in hetero windows (A has no native stamps).

| model | config | crossings per forward | re-acquire P50 / P99 / mean ms | features | pre | native | re-acquire | post | tail | predict (whole) |
|---|---|---|---|---|---|---|---|---|---|---|
| laya | A | 0 (predict 0, sends 0, pool 0); re-entries 0 | – | 0.161 | – | – | – | – | 0.060 | 9.668 |
| laya | C | 108 (predict 1, sends 104, pool 3); re-entries 62 | 0.001 / 0.098 / 0.004 | 0.172 | 0.146 | 9.461 | 0.001 | 0.089 | 0.055 | 9.698 |
| laya | PB | 4 (predict 1, sends 0, pool 3); re-entries 0 | 0.001 / 0.017 / 0.003 | 0.157 | 0.014 | 9.373 | 0.001 | 0.006 | 0.060 | 9.395 |
| laya-typed-decisions | A | 0 (predict 0, sends 0, pool 0); re-entries 0 | – | 0.159 | – | – | – | – | 0.059 | 9.651 |
| laya-typed-decisions | C | 108 (predict 1, sends 104, pool 3); re-entries 62 | 0.001 / 0.237 / 0.007 | 0.158 | 0.134 | 9.416 | 0.001 | 0.083 | 0.055 | 9.638 |
| laya-typed-decisions | PB | 4 (predict 1, sends 0, pool 3); re-entries 0 | 0.001 / 0.363 / 0.008 | 0.158 | 0.014 | 9.326 | 0.001 | 0.005 | 0.058 | 9.348 |
| laya-multilingual | B | – | – | – | – | – | – | – | – | – |
| laya-multilingual | C | 108 (predict 1, sends 104, pool 3); re-entries 62 | 0.002 / 1.661 / 0.042 | 0.264 | 0.446 | 4.103 | 0.002 | 0.276 | 0.215 | 4.868 |
| laya-multilingual | PB | 4 (predict 1, sends 0, pool 3); re-entries 0 | 0.001 / 0.363 / 0.009 | 0.083 | 0.011 | 3.537 | 0.001 | 0.005 | 0.051 | 3.555 |

## Not gating: GPU completion collisions and the ANE tail

Fixed exposure window: a GPU `received_ns` in [ANE service_start − 1 ms, service_start + 0.3 ms]. Tail: ANE request `response_ns − submit_ns` above the production configuration's P99 (hetero windows). Odds ratio with 0.5 added to each cell.

| model | config | tail threshold ms | in tail | colliding | tail requests colliding | odds ratio | GPU completions overlapping a forward (mean / any) | … overlapping its pre stage (mean / any) |
|---|---|---|---|---|---|---|---|---|
| laya | A | 10.73 | 0.010 | 0.249 | 0.483 | 2.9 | 0.239 / 0.239 | – |
| laya | C | 10.73 | 0.022 | 0.042 | 0.587 | 45.0 | 0.282 / 0.282 | 0.027 / 0.027 |
| laya | PB | 10.73 | 0.001 | 0.031 | 0.056 | 2.7 | 0.274 / 0.274 | 0.011 / 0.011 |
| laya-typed-decisions | A | 11.13 | 0.010 | 0.087 | 0.576 | 15.2 | 0.117 / 0.117 | – |
| laya-typed-decisions | C | 11.13 | 0.007 | 0.012 | 0.525 | 131.2 | 0.139 / 0.139 | 0.012 / 0.012 |
| laya-typed-decisions | PB | 11.13 | 0.002 | 0.018 | 0.320 | 28.0 | 0.135 / 0.135 | 0.014 / 0.014 |
| laya-multilingual | B | 6.56 | 0.010 | 0.033 | 0.016 | 0.5 | 0.146 / 0.146 | – |
| laya-multilingual | C | 6.56 | 0.161 | 0.041 | 0.155 | 9.4 | 0.169 / 0.169 | 0.051 / 0.051 |
| laya-multilingual | PB | 6.56 | 0.007 | 0.036 | 0.570 | 40.1 | 0.130 / 0.130 | 0.013 / 0.013 |

## Not gating: GIL probe, thread CPU, slow-CPU flag

GIL probe: lateness of a 1 ms sleep grid in the benchmark process (median over hetero windows of the per-window P99). Thread CPU: ms per hetero window (median). Slow-CPU flag: candidate windows whose client-short CPU per request exceeds 1.25x that of P's matched window.

| model | config | probe P99 ms | probe > 1 ms | probe CPU (core) | client-short | client-long | ane-dispatch | gpu-dispatch | ANE thread CPU/forward | GPU thread CPU/forward | slow-CPU windows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| laya | A | 10.321 | 0.7098 | 0.0032 | – | 221 | 1240 | 57 | 0.587 | 1.737 | – (reference) |
| laya | C | 0.589 | 0.0002 | 0.0080 | 291 | 252 | 1377 | 48 | 0.643 | 1.699 | not computable (0 of 6) |
| laya | PB | 0.577 | 0.0001 | 0.0092 | 290 | 241 | 785 | 45 | 0.336 | 1.699 | not computable (0 of 6) |
| laya-typed-decisions | A | 9.991 | 0.7082 | 0.0030 | – | 214 | 1264 | 33 | 0.596 | 1.875 | – (reference) |
| laya-typed-decisions | C | 0.585 | 0.0029 | 0.0098 | – | 245 | 1320 | 28 | 0.616 | 1.843 | not computable (0 of 6) |
| laya-typed-decisions | PB | 0.583 | 0.0027 | 0.0095 | – | 235 | 779 | 26 | 0.334 | 1.817 | not computable (0 of 6) |
| laya-multilingual | B | 1.574 | 0.0167 | 0.0094 | – | 768 | 668 | 123 | 0.753 | 3.355 | – (reference) |
| laya-multilingual | C | 2.204 | 0.0420 | 0.0193 | – | 1071 | 6505 | 215 | 1.565 | 4.506 | not computable (0 of 6) |
| laya-multilingual | PB | 0.646 | 0.0004 | 0.0080 | – | 359 | 1431 | 57 | 0.307 | 1.974 | not computable (0 of 6) |

## Pre-campaign check (`raw/check.json`)

- PB bit-identical to coremltools on every bucket of every model: True
- C bit-identical to coremltools on every bucket of every model: True
- PB output modes seen: backed
