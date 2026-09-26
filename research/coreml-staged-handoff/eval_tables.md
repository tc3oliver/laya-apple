# H64 production evaluation (evaluation.md)

Outcome: CLOSE: phase 1 failed: H5 mix-laya-P-r5 w5: 30 consecutive host-slow bins. H64 is closed and 1.4 stays.

Runtime under test: laya_apple tree d541fdc820f59fcb7b27a5f20110f828892be16f, pyproject 2d2d3d684b086c0c651cd1ea96e09550ce2e5e85, uv.lock 10f6ba84dd28791684cc102d651c1d5c38cc0178, clean. Failed logs: none

| phase | status | reasons |
|---|---|---|
| 1 | FAIL | H5 mix-laya-P-r5 w5: 30 consecutive host-slow bins |
| 2 | not reached (would be pending) | mix-laya-typed-decisions-P-r1; mix-laya-typed-decisions-A-r1; mix-laya-typed-decisions-A-r2; mix-laya-typed-decisions-P-r2; mix-laya-typed-decisions-A-r3; mix-laya-typed-decisions-P-r3; mix-laya-typed-decisions-P-r4; mix-laya-typed-decisions-A-r4 |
| 3 | not reached (would be pending) | mix-laya-multilingual-A-r1 |
| 4 | not reached (would be pending) | product-laya-P-r1; product-laya-A-r1; product-laya-A-r2; product-laya-P-r2; product-laya-A-r3; product-laya-P-r3; product-laya-P-r4; product-laya-A-r4 |
| 5 | not reached (would be pending) | soak55-laya-P-r1; soak55-laya-A-r1; soak55-laya-A-r2; soak55-laya-P-r2 |

## Phase 1: laya, L128 / L512, mix

Status: **FAIL**
- fail: H5 mix-laya-P-r5 w5: 30 consecutive host-slow bins; H6 mix-laya-P-r5 w5: 15 consecutive slow spans

### Runs and validity

| run | used | runtime ok | protocol ok | machine | crash logs | mismatches | routing | workers alive | H1 | H2 |
|---|---|---|---|---|---|---|---|---|---|---|
| mix-laya-P-r1 | yes | True | True | ok | 0 | 0 | 0 | True | – | – |
| mix-laya-A-r1 | yes | True | True | ok | 0 | 0 | 0 | True | – | – |
| mix-laya-A-r2 | yes | True | True | ok | 0 | 0 | 0 | True | – | – |
| mix-laya-P-r2 | yes | True | True | ok | 0 | 0 | 0 | True | – | – |
| mix-laya-A-r3 | yes | True | True | ok | 0 | 0 | 0 | True | – | – |
| mix-laya-P-r3 | yes | True | True | ok | 0 | 0 | 0 | True | – | – |
| mix-laya-P-r4 | yes | True | True | ok | 0 | 0 | 0 | True | – | – |
| mix-laya-A-r4 | yes | True | True | ok | 0 | 0 | 0 | True | – | – |
| mix-laya-P-r5 | superseded | True | True | before: OrbStack Helper at 92% CPU | 0 | 0 | 0 | True | – | – |
| mix-laya-P-r5-b | yes | True | True | ok | 0 | 0 | 0 | True | – | – |
| mix-laya-A-r5 | yes | True | True | ok | 0 | 0 | 0 | True | – | – |
| mix-laya-A-r6 | yes | True | True | ok | 0 | 0 | 0 | True | – | – |
| mix-laya-P-r6 | yes | True | True | ok | 0 | 0 | 0 | True | – | – |

### Hard gates (P)

| gate | value | limit | pass |
|---|---|---|---|
| H1 correctness | 0 findings | 0 | True |
| H2 handoff state | 0 findings | 0 | True |
| H3 episode GPU-return P50 (max) | 0.231 | <= 1.0 | True |
| H3 pooled GPU-return P99 | 0.208 | <= 1.0 | True |
| H4 min pooled aggregate ratio | 1.047 | >= 0.95 | True |
| H5 longest host-slow bins | 30 | < 2 | False |
| H6 longest slow spans | 15 | < 2 | False |
| H7 P tail events | 0 (A 0) | <= 2 | True |
| pooled P GPU-return P50 (outcome) | 0.035 | <= 1.0 | True |

P episodes: 12 of 12 expected. A signatures (H5 / H6): 0 of 12 A episodes.

### Latency non-inferiority (delta P99 = P episode - paired A episode)

- median delta: -0.365 ms (<= +0.5): True
- cluster bootstrap one-sided 95% upper bound (6 clusters of 2, B = 20000, seed 20260926): -0.256 ms (<= +1.0): True
- transition-level upper bound (report only): -0.068 ms
- median deltas (report only): episode median -0.187, P95 -0.556, P99.9 +0.082 ms, req/s +5.92
- pass: **True**

### Pooled per cell

| cell | median | P95 | P99 | P99.9 | GPU return P50 / P95 / P99 (n) |
|---|---|---|---|---|---|
| P | 9.86 | 10.07 | 10.80 | 15.86 | 0.035 / 0.069 / 0.208 (6104) |
| A | 10.05 | 10.64 | 11.14 | 16.13 | 4.325 / 5.044 / 5.726 (5606) |

A pooled hetero short median (slow-span reference): 10.045 ms

### Throughput per run pair

| P | A | P agg req/s | A agg req/s | ratio | episode ratios |
|---|---|---|---|---|---|
| mix-laya-P-r1 | mix-laya-A-r1 | 128.4 | 122.6 | 1.047 | 1.048, 1.047 |
| mix-laya-P-r2 | mix-laya-A-r2 | 128.4 | 122.1 | 1.051 | 1.053, 1.050 |
| mix-laya-P-r3 | mix-laya-A-r3 | 128.4 | 122.4 | 1.049 | 1.050, 1.049 |
| mix-laya-P-r4 | mix-laya-A-r4 | 128.3 | 122.2 | 1.050 | 1.050, 1.050 |
| mix-laya-P-r5-b | mix-laya-A-r5 | 128.4 | 122.5 | 1.048 | 1.047, 1.048 |
| mix-laya-P-r6 | mix-laya-A-r6 | 128.5 | 122.6 | 1.048 | 1.048, 1.048 |

### Tail events (episode P99 > A median episode P99 11.04 + 2.0 ms)

none

### Episodes

| run | window | after | t_h s | median | P95 | P99 | P99.9 | agg req/s | GPU return P50 / P95 / P99 | host-slow bins (longest) | slow spans (longest) | latency source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mix-laya-P-r1 | 2 | solo_long | 0.766 | 9.85 | 10.12 | 11.09 | 15.81 | 128.4 | 0.034 / 0.076 / 0.286 | 0 (0) | 0 (0) | client |
| mix-laya-P-r1 | 5 | gpu_only | 0.751 | 9.86 | 10.06 | 10.72 | 15.35 | 128.4 | 0.033 / 0.070 / 0.148 | 0 (0) | 0 (0) | client |
| mix-laya-A-r1 | 2 | solo_long | – | 10.05 | 10.57 | 10.91 | 15.43 | 122.5 | 4.348 / 4.724 / 5.401 | 0 (0) | 0 (0) | client |
| mix-laya-A-r1 | 5 | gpu_only | – | 10.03 | 10.57 | 11.01 | 15.21 | 122.7 | 4.271 / 4.577 / 5.308 | 0 (0) | 0 (0) | client |
| mix-laya-A-r2 | 2 | solo_long | – | 10.04 | 10.67 | 12.72 | 17.36 | 122.0 | 4.294 / 5.409 / 6.529 | 0 (0) | 0 (0) | client |
| mix-laya-A-r2 | 5 | gpu_only | – | 10.06 | 10.66 | 11.06 | 15.17 | 122.3 | 4.377 / 4.715 / 5.286 | 0 (0) | 0 (0) | client |
| mix-laya-P-r2 | 2 | solo_long | 0.746 | 9.86 | 10.05 | 10.61 | 14.23 | 128.5 | 0.036 / 0.068 / 0.140 | 0 (0) | 0 (0) | client |
| mix-laya-P-r2 | 5 | gpu_only | 0.739 | 9.87 | 10.06 | 10.63 | 14.48 | 128.4 | 0.035 / 0.064 / 0.181 | 0 (0) | 0 (0) | client |
| mix-laya-A-r3 | 2 | solo_long | – | 10.05 | 10.63 | 11.02 | 15.98 | 122.4 | 4.386 / 4.844 / 5.297 | 0 (0) | 0 (0) | client |
| mix-laya-A-r3 | 5 | gpu_only | – | 10.05 | 10.64 | 10.83 | 15.21 | 122.4 | 4.365 / 4.643 / 5.425 | 0 (0) | 0 (0) | client |
| mix-laya-P-r3 | 2 | solo_long | 0.751 | 9.85 | 10.06 | 10.71 | 15.45 | 128.5 | 0.034 / 0.085 / 0.245 | 0 (0) | 0 (0) | client |
| mix-laya-P-r3 | 5 | gpu_only | 0.752 | 9.87 | 10.08 | 10.61 | 15.24 | 128.3 | 0.033 / 0.062 / 0.149 | 0 (0) | 0 (0) | client |
| mix-laya-P-r4 | 2 | solo_long | 0.762 | 9.85 | 10.06 | 10.75 | 16.76 | 128.4 | 0.035 / 0.064 / 0.240 | 0 (0) | 0 (0) | client |
| mix-laya-P-r4 | 5 | gpu_only | 0.771 | 9.88 | 10.12 | 11.82 | 15.83 | 128.1 | 0.037 / 0.074 / 0.170 | 0 (0) | 0 (0) | client |
| mix-laya-A-r4 | 2 | solo_long | – | 10.05 | 10.70 | 11.26 | 16.31 | 122.3 | 4.325 / 5.198 / 5.504 | 0 (0) | 0 (0) | client |
| mix-laya-A-r4 | 5 | gpu_only | – | 10.07 | 10.73 | 10.92 | 16.21 | 122.0 | 4.417 / 5.275 / 5.780 | 0 (0) | 0 (0) | client |
| mix-laya-P-r5 | 2 | solo_long | 0.742 | 9.86 | 10.10 | 10.80 | 14.49 | 128.5 | 0.041 / 0.070 / 0.129 | 0 (0) | 0 (0) | client |
| mix-laya-P-r5 | 5 | gpu_only | 0.734 | 12.09 | 16.52 | 17.03 | 18.05 | 108.0 | 0.231 / 1.420 / 2.001 | 30 (30) | 15 (15) | client |
| mix-laya-P-r5-b | 2 | solo_long | 0.759 | 9.85 | 10.06 | 10.69 | 14.85 | 128.5 | 0.037 / 0.071 / 0.140 | 0 (0) | 0 (0) | client |
| mix-laya-P-r5-b | 5 | gpu_only | 0.787 | 9.86 | 10.08 | 12.14 | 17.19 | 128.3 | 0.037 / 0.066 / 0.221 | 0 (0) | 0 (0) | client |
| mix-laya-A-r5 | 2 | solo_long | – | 10.03 | 10.55 | 11.43 | 15.36 | 122.6 | 4.258 / 4.523 / 5.338 | 0 (0) | 0 (0) | client |
| mix-laya-A-r5 | 5 | gpu_only | – | 10.05 | 10.61 | 10.93 | 14.90 | 122.4 | 4.352 / 4.820 / 5.734 | 0 (0) | 0 (0) | client |
| mix-laya-A-r6 | 2 | solo_long | – | 10.04 | 10.56 | 11.07 | 16.28 | 122.6 | 4.271 / 4.548 / 5.471 | 0 (0) | 0 (0) | client |
| mix-laya-A-r6 | 5 | gpu_only | – | 10.04 | 10.60 | 11.38 | 15.28 | 122.5 | 4.280 / 4.778 / 5.543 | 0 (0) | 0 (0) | client |
| mix-laya-P-r6 | 2 | solo_long | 0.752 | 9.85 | 10.08 | 10.65 | 16.48 | 128.5 | 0.034 / 0.064 / 0.203 | 0 (0) | 0 (0) | client |
| mix-laya-P-r6 | 5 | gpu_only | 0.760 | 9.85 | 10.05 | 10.84 | 15.62 | 128.4 | 0.034 / 0.064 / 0.186 | 0 (0) | 0 (0) | client |

## Phase 2: laya-typed-decisions, L128 / L1024, mix

Status: **not reached**
- pending: mix-laya-typed-decisions-P-r1; mix-laya-typed-decisions-A-r1; mix-laya-typed-decisions-A-r2; mix-laya-typed-decisions-P-r2; mix-laya-typed-decisions-A-r3; mix-laya-typed-decisions-P-r3; mix-laya-typed-decisions-P-r4; mix-laya-typed-decisions-A-r4

### Runs and validity

| run | used | runtime ok | protocol ok | machine | crash logs | mismatches | routing | workers alive | H1 | H2 |
|---|---|---|---|---|---|---|---|---|---|---|
| mix-laya-typed-decisions-P-r1 | missing | – | – | – | 0 | – | – | – | – | – |
| mix-laya-typed-decisions-A-r1 | missing | – | – | – | 0 | – | – | – | – | – |
| mix-laya-typed-decisions-A-r2 | missing | – | – | – | 0 | – | – | – | – | – |
| mix-laya-typed-decisions-P-r2 | missing | – | – | – | 0 | – | – | – | – | – |
| mix-laya-typed-decisions-A-r3 | missing | – | – | – | 0 | – | – | – | – | – |
| mix-laya-typed-decisions-P-r3 | missing | – | – | – | 0 | – | – | – | – | – |
| mix-laya-typed-decisions-P-r4 | missing | – | – | – | 0 | – | – | – | – | – |
| mix-laya-typed-decisions-A-r4 | missing | – | – | – | 0 | – | – | – | – | – |

## Phase 3: laya-multilingual, L128 / L512, mix

Status: **not reached**
- pending: mix-laya-multilingual-A-r1

### Runs and validity

| run | used | runtime ok | protocol ok | machine | crash logs | mismatches | routing | workers alive | H1 | H2 |
|---|---|---|---|---|---|---|---|---|---|---|
| mix-laya-multilingual-A-r1 | missing | – | – | – | 0 | – | – | – | – | – |

### Multilingual smoke

pending

## Phase 4: laya, L128 / L512, product

Status: **not reached**
- pending: product-laya-P-r1; product-laya-A-r1; product-laya-A-r2; product-laya-P-r2; product-laya-A-r3; product-laya-P-r3; product-laya-P-r4; product-laya-A-r4

### Runs and validity

| run | used | runtime ok | protocol ok | machine | crash logs | mismatches | routing | workers alive | H1 | H2 |
|---|---|---|---|---|---|---|---|---|---|---|
| product-laya-P-r1 | missing | – | – | – | 0 | – | – | – | – | – |
| product-laya-A-r1 | missing | – | – | – | 0 | – | – | – | – | – |
| product-laya-A-r2 | missing | – | – | – | 0 | – | – | – | – | – |
| product-laya-P-r2 | missing | – | – | – | 0 | – | – | – | – | – |
| product-laya-A-r3 | missing | – | – | – | 0 | – | – | – | – | – |
| product-laya-P-r3 | missing | – | – | – | 0 | – | – | – | – | – |
| product-laya-P-r4 | missing | – | – | – | 0 | – | – | – | – | – |
| product-laya-A-r4 | missing | – | – | – | 0 | – | – | – | – | – |

## Phase 5: laya, L128 / L512, soak55

Status: **not reached**
- pending: soak55-laya-P-r1; soak55-laya-A-r1; soak55-laya-A-r2; soak55-laya-P-r2

### Runs and validity

| run | used | runtime ok | protocol ok | machine | crash logs | mismatches | routing | workers alive | H1 | H2 |
|---|---|---|---|---|---|---|---|---|---|---|
| soak55-laya-P-r1 | missing | – | – | – | 0 | – | – | – | – | – |
| soak55-laya-A-r1 | missing | – | – | – | 0 | – | – | – | – | – |
| soak55-laya-A-r2 | missing | – | – | – | 0 | – | – | – | – | – |
| soak55-laya-P-r2 | missing | – | – | – | 0 | – | – | – | – | – |
