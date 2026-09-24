# ANE process isolation: production regression gate (v1.0 closed-loop mix, hetero windows)

| model | placement | aggregate req/s | short req/s | short P99 | long req/s | long P99 | GPU return P50 / P99 | mismatches |
|---|---|---|---|---|---|---|---|---|
| laya | thread | 122.5 | 97.9 | 11.36 | 24.6 | 42.41 | 4.283 / 6.686 | 0 |
| laya | process | 125.7 | 98.0 | 13.34 | 27.7 | 41.55 | 0.031 / 0.226 | 0 |
| laya-typed-decisions | thread | 109.8 | 97.1 | 11.57 | 12.7 | 82.49 | 8.333 / 9.499 | 0 |
| laya-typed-decisions | process | 92.2 | 79.6 | 20.05 | 12.6 | 83.57 | 0.204 / 1.228 | 0 |

## Criteria (process vs thread)

| model | aggregate ≥ 0.95× | short P99 ≤ 1.05× | long P99 ≤ 1.05× | GPU return P50 ≤ 1 ms and ≥ 5× better | correctness | model |
|---|---|---|---|---|---|---|
| laya | +2.6% pass | +17.5% **FAIL** | -2.0% pass | ×138.2 pass | pass | **FAIL** |
| laya-typed-decisions | -16.1% **FAIL** | +73.3% **FAIL** | +1.3% pass | ×40.8 pass | pass | **FAIL** |

Overall: **FAIL**
