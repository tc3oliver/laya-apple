# Core ML predict without the GIL on the product mix (v1.0 closed-loop mix, hetero windows)

| model | config | aggregate req/s | short req/s | short P99 | long req/s | long P99 | GPU return P50 / P99 | mismatches |
|---|---|---|---|---|---|---|---|---|
| laya | A (production) | 122.0 | 97.5 | 11.43 | 24.5 | 43.26 | 4.354 / 6.865 | 0 |
| laya | C | 124.5 | 96.8 | 12.91 | 27.7 | 40.29 | 0.050 / 0.292 | 0 |
| laya | D | 108.0 | 82.2 | 15.89 | 25.8 | 42.74 | 0.010 / 0.023 | 0 |
| laya-typed-decisions | A (production) | 110.2 | 97.8 | 11.60 | 12.4 | 82.31 | 8.530 / 9.584 | 0 |
| laya-typed-decisions | C | 102.2 | 88.8 | 21.24 | 13.4 | 83.32 | 0.064 / 1.182 | 0 |
| laya-typed-decisions | D | 110.9 | 97.1 | 12.98 | 13.8 | 80.71 | 0.002 / 0.014 | 0 |
| laya-multilingual | B (production) | 200.5 | 171.4 | 7.92 | 29.1 | 36.34 | 0.147 / 0.754 | 0 |
| laya-multilingual | C | 182.1 | 152.9 | 9.58 | 29.2 | 36.46 | 0.192 / 0.942 | 0 |
| laya-multilingual | D | 169.2 | 140.2 | 12.51 | 29.0 | 36.90 | 0.011 / 0.019 | 0 |

## Criteria (candidate vs the model's production configuration)

| model | candidate | aggregate ≥ 0.95× | short P99 ≤ 1.05× | long P99 ≤ 1.05× | GPU return P50 ≤ 1 ms and ≥ 5× better | correctness | model |
|---|---|---|---|---|---|---|---|
| laya | C vs A | +2.0% pass | +12.9% **FAIL** | -6.9% pass | ×87.1 pass | pass | **FAIL** |
| laya | D vs A | -11.5% **FAIL** | +39.1% **FAIL** | -1.2% pass | ×435.4 pass | pass | **FAIL** |
| laya-typed-decisions | C vs A | -7.2% **FAIL** | +83.0% **FAIL** | +1.2% pass | ×133.3 pass | pass | **FAIL** |
| laya-typed-decisions | D vs A | +0.7% pass | +11.8% **FAIL** | -1.9% pass | ×4265.0 pass | pass | **FAIL** |
| laya-multilingual | C vs B | -9.1% **FAIL** | +20.9% **FAIL** | +0.4% pass | ≤ 1 ms only (vs B) pass | pass | **FAIL** |
| laya-multilingual | D vs B | -15.6% **FAIL** | +57.9% **FAIL** | +1.5% pass | ≤ 1 ms only (vs B) pass | pass | **FAIL** |

- C: **FAIL** (laya, laya-typed-decisions, laya-multilingual)
- D: **FAIL** (laya, laya-typed-decisions, laya-multilingual)

Stop rule: C and D both fail: FAIL recorded; the Swift-worker fallback research is preregistered next

## Not gating: GIL re-acquire wait and executing-thread CPU

| model | config | ANE GIL re-acquire P50 / P95 / P99 / mean ms | ANE predict (native) P50 | ANE thread CPU per forward | ANE placement | GPU thread CPU per forward | GPU service P50 | GPU placement |
|---|---|---|---|---|---|---|---|---|
| laya | A | – | – | 0.610 | thread | 1.840 | 35.76 | process |
| laya | C | 0.002 / 0.007 / 0.352 / 0.013 | 9.49 | 0.773 | thread | 2.081 | 35.32 | process |
| laya | D | 0.004 / 0.518 / 1.537 / 0.071 | 9.72 | 1.918 | thread | 5.285 | 37.22 | thread |
| laya-typed-decisions | A | – | – | 0.611 | thread | 1.991 | 71.59 | process |
| laya-typed-decisions | C | 0.003 / 0.010 / 0.448 / 0.021 | 9.45 | 1.370 | thread | 4.124 | 71.18 | process |
| laya-typed-decisions | D | 0.002 / 0.007 / 1.091 / 0.033 | 9.44 | 0.700 | thread | 2.104 | 71.15 | thread |
| laya-multilingual | B | – | – | 1.327 | process | 5.938 | 31.59 | process |
| laya-multilingual | C | 0.003 / 0.024 / 2.040 / 0.069 | 4.23 | 1.967 | thread | 5.690 | 31.41 | process |
| laya-multilingual | D | 0.003 / 2.713 / 3.835 / 0.358 | 4.23 | 2.006 | thread | 5.920 | 32.03 | thread |
