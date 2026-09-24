# Equal GPU offered load (30 req/s target): thread vs process ANE placement

## Workload equality and stability

| model | placement | GPU offered req/s | GPU completed req/s | arrival SHA-256 (c0) | stable cycles | errors | mismatches |
|---|---|---|---|---|---|---|---|
| laya | thread | 31.55 | 31.55 | `e537412cf4e7026a…` | 3/3 | 0 | 0 |
| laya | process | 31.55 | 31.55 | `e537412cf4e7026a…` | 3/3 | 0 | 0 |
| laya-typed-decisions | thread | 31.55 | 31.55 | `e537412cf4e7026a…` | 3/3 | 0 | 0 |
| laya-typed-decisions | process | 31.55 | 31.55 | `e537412cf4e7026a…` | 3/3 | 0 | 0 |

## CPU, both devices busy (3 cycles × 20 s)

| model | placement | tree CPU ms / request | GPU worker ms / GPU req | parent + ANE worker ms / ANE req | GPU forward thread CPU mean | ANE req/s | GPU return P50 / P99 |
|---|---|---|---|---|---|---|---|
| laya | thread | 1.82 | 2.49 | 1.58 | 1.64 | 92.2 | 7.462 / 11.500 |
| laya | process | 1.80 | 2.28 | 1.65 | 1.59 | 96.2 | 0.033 / 3.189 |
| laya-typed-decisions | thread | 1.80 | 2.51 | 1.55 | 1.63 | 92.3 | 7.501 / 11.378 |
| laya-typed-decisions | process | 1.83 | 2.29 | 1.67 | 1.60 | 95.6 | 0.034 / 3.066 |

## CPU seconds per role, both devices busy (sum of 3 windows)

| model | placement | parent | GPU worker | ANE worker | other | tree | GPU done | ANE done |
|---|---|---|---|---|---|---|---|---|
| laya | thread | 8.75 | 4.72 | – | 0.00 | 13.47 | 1893 | 5529 |
| laya | process | 5.98 | 4.32 | 3.52 | 0.00 | 13.82 | 1893 | 5775 |
| laya-typed-decisions | thread | 8.59 | 4.75 | – | 0.00 | 13.34 | 1893 | 5537 |
| laya-typed-decisions | process | 6.08 | 4.34 | 3.52 | 0.00 | 13.94 | 1893 | 5734 |

## GPU alone, same arrivals

| model | placement | tree CPU ms / request | GPU worker ms / GPU req |
|---|---|---|---|
| laya | thread | 13.77 | 7.65 |
| laya | process | 13.88 | 7.73 |
| laya-typed-decisions | thread | 13.89 | 7.79 |
| laya-typed-decisions | process | 13.70 | 7.56 |

## Verdict (criterion: process / thread tree CPU per request ≤ 1.5)

| model | workload valid | ratio | CPU |
|---|---|---|---|
| laya | True | ×0.99 | pass |
| laya-typed-decisions | True | ×1.02 | pass |
