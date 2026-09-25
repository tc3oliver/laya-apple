# Async transient: what happens at a hetero transition (laya, L128 / L512)

Outcome: C: supports a perf-level placement / performance-state hypothesis

Crashed runs and re-runs (`raw/failed/`): none


## Runs

| run | mismatches | recount samples / series / sampler CPU | wall s |
|---|---|---|---|
| laya-A-r1 | 0 | 1861 / 29 / 0.110% | 186 |
| laya-PB-ASYNC-r1 | 0 | 1869 / 32 / 0.128% | 187 |
| laya-PB-ASYNC-r2 | 0 | 1869 / 34 / 0.137% | 187 |
| laya-A-r2 | 0 | 1862 / 29 / 0.106% | 187 |

## Transitions

| cell | run | cycle | after | transient s | recovered | present | peak short P99 ms | client CPU/req short |
|---|---|---|---|---|---|---|---|---|
| A | laya-A-r1 | 0 | solo_long | 0.5 | True | True | 21.43 | 0.149 |
| A | laya-A-r1 | 1 | gpu_only | 0.5 | True | True | 19.05 | 0.146 |
| A | laya-A-r2 | 0 | solo_long | 0.5 | True | True | 15.91 | 0.146 |
| A | laya-A-r2 | 1 | gpu_only | 0.5 | True | True | 22.55 | 0.149 |
| PB-ASYNC | laya-PB-ASYNC-r1 | 0 | solo_long | 3.0 | True | True | 26.51 | 0.214 |
| PB-ASYNC | laya-PB-ASYNC-r1 | 1 | gpu_only | 4.0 | True | True | 31.52 | 0.242 |
| PB-ASYNC | laya-PB-ASYNC-r2 | 0 | solo_long | 0.5 | True | True | 15.40 | 0.146 |
| PB-ASYNC | laya-PB-ASYNC-r2 | 1 | gpu_only | 15.0 | True | True | 27.62 | 0.619 |

## Buckets (seconds from t0)

| run | cycle | bucket | n | short P50 / P99 | prepare | host-slow | features | native | cb delay P50 | handoff | post | action head | ANE CPU/fwd | GPU CPU/fwd | GPU return P50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laya-A-r1 | 0 | -2-0 | 0 | – / – | – | – | – | – | – | – | – | – | – | – | – |
| laya-A-r1 | 0 | 0-0.5 | 40 | 10.62 / 42.66 | 0.308 | 0.33 | 0.686 | – | – | – | – | 0.135 | 1.517 | 4.145 | 4.328 |
| laya-A-r1 | 0 | 0.5-1 | 49 | 10.04 / 10.69 | 0.124 | 0.00 | 0.255 | – | – | – | – | 0.063 | 0.567 | 1.695 | 4.438 |
| laya-A-r1 | 0 | 1-2 | 98 | 10.06 / 10.79 | 0.125 | 0.00 | 0.264 | – | – | – | – | 0.068 | 0.586 | 1.771 | 4.418 |
| laya-A-r1 | 0 | 2-4 | 197 | 10.04 / 10.68 | 0.121 | 0.00 | 0.264 | – | – | – | – | 0.062 | 0.565 | 1.705 | 4.349 |
| laya-A-r1 | 0 | 4-5 | 99 | 10.03 / 10.73 | 0.120 | 0.00 | 0.276 | – | – | – | – | 0.062 | 0.558 | 1.682 | 4.296 |
| laya-A-r1 | 0 | steady 10-20 | 984 | 10.04 / 10.73 | 0.122 | 0.00 | 0.268 | – | – | – | – | 0.065 | 0.571 | 1.727 | 4.329 |
| laya-A-r1 | 1 | -2-0 | 0 | – / – | – | – | – | – | – | – | – | – | – | – | – |
| laya-A-r1 | 1 | 0-0.5 | 42 | 10.41 / 39.61 | 0.239 | 0.21 | 0.403 | – | – | – | – | 0.103 | 1.256 | 3.214 | 4.590 |
| laya-A-r1 | 1 | 0.5-1 | 49 | 10.06 / 10.88 | 0.125 | 0.00 | 0.237 | – | – | – | – | 0.062 | 0.570 | 1.713 | 4.374 |
| laya-A-r1 | 1 | 1-2 | 99 | 10.04 / 10.75 | 0.122 | 0.00 | 0.267 | – | – | – | – | 0.061 | 0.564 | 1.679 | 4.306 |
| laya-A-r1 | 1 | 2-4 | 197 | 10.04 / 10.66 | 0.120 | 0.00 | 0.273 | – | – | – | – | 0.062 | 0.560 | 1.666 | 4.323 |
| laya-A-r1 | 1 | 4-5 | 98 | 10.04 / 10.68 | 0.122 | 0.00 | 0.259 | – | – | – | – | 0.064 | 0.571 | 1.728 | 4.322 |
| laya-A-r1 | 1 | steady 10-20 | 985 | 10.04 / 10.77 | 0.121 | 0.00 | 0.261 | – | – | – | – | 0.063 | 0.564 | 1.701 | 4.316 |
| laya-A-r2 | 0 | -2-0 | 0 | – / – | – | – | – | – | – | – | – | – | – | – | – |
| laya-A-r2 | 0 | 0-0.5 | 44 | 10.41 / 28.84 | 0.178 | 0.14 | 0.351 | – | – | – | – | 0.086 | 1.073 | 2.592 | 4.822 |
| laya-A-r2 | 0 | 0.5-1 | 49 | 10.06 / 10.85 | 0.127 | 0.00 | 0.248 | – | – | – | – | 0.067 | 0.588 | 1.738 | 4.352 |
| laya-A-r2 | 0 | 1-2 | 98 | 10.06 / 10.78 | 0.123 | 0.00 | 0.255 | – | – | – | – | 0.063 | 0.574 | 1.743 | 4.351 |
| laya-A-r2 | 0 | 2-4 | 197 | 10.04 / 10.72 | 0.120 | 0.00 | 0.260 | – | – | – | – | 0.062 | 0.569 | 1.709 | 4.326 |
| laya-A-r2 | 0 | 4-5 | 99 | 10.05 / 10.72 | 0.119 | 0.00 | 0.247 | – | – | – | – | 0.061 | 0.567 | 1.697 | 4.352 |
| laya-A-r2 | 0 | steady 10-20 | 981 | 10.06 / 10.81 | 0.123 | 0.00 | 0.259 | – | – | – | – | 0.064 | 0.579 | 1.747 | 4.385 |
| laya-A-r2 | 1 | -2-0 | 0 | – / – | – | – | – | – | – | – | – | – | – | – | – |
| laya-A-r2 | 1 | 0-0.5 | 38 | 11.65 / 45.07 | 0.372 | 0.45 | 0.722 | – | – | – | – | 0.178 | 1.838 | 5.373 | 4.328 |
| laya-A-r2 | 1 | 0.5-1 | 49 | 10.05 / 10.70 | 0.123 | 0.00 | 0.275 | – | – | – | – | 0.061 | 0.569 | 1.723 | 4.367 |
| laya-A-r2 | 1 | 1-2 | 99 | 10.02 / 10.76 | 0.121 | 0.00 | 0.279 | – | – | – | – | 0.062 | 0.567 | 1.727 | 4.507 |
| laya-A-r2 | 1 | 2-4 | 197 | 10.03 / 10.73 | 0.120 | 0.00 | 0.257 | – | – | – | – | 0.062 | 0.569 | 1.708 | 4.314 |
| laya-A-r2 | 1 | 4-5 | 98 | 10.05 / 10.66 | 0.119 | 0.00 | 0.268 | – | – | – | – | 0.061 | 0.566 | 1.716 | 4.358 |
| laya-A-r2 | 1 | steady 10-20 | 981 | 10.05 / 10.76 | 0.121 | 0.00 | 0.269 | – | – | – | – | 0.063 | 0.579 | 1.733 | 4.398 |
| laya-PB-ASYNC-r1 | 0 | -2-0 | 0 | – / – | – | – | – | – | – | – | – | – | – | – | – |
| laya-PB-ASYNC-r1 | 0 | 0-0.5 | 36 | 12.17 / 40.73 | 0.645 | 0.97 | 1.062 | 11.025 | 0.030 | 0.056 | 0.041 | 0.342 | 1.325 | 8.152 | 0.202 |
| laya-PB-ASYNC-r1 | 0 | 0.5-1 | 39 | 12.39 / 17.00 | 0.719 | 1.00 | 1.107 | 9.563 | 0.033 | 0.059 | 0.043 | 0.369 | 1.454 | 8.542 | 0.240 |
| laya-PB-ASYNC-r1 | 0 | 1-2 | 77 | 12.48 / 17.11 | 0.736 | 1.00 | 1.230 | 9.840 | 0.034 | 0.104 | 0.045 | 0.391 | 1.502 | 8.700 | 0.266 |
| laya-PB-ASYNC-r1 | 0 | 2-4 | 184 | 9.90 / 17.17 | 0.289 | 0.26 | 0.472 | 9.528 | 0.012 | 0.031 | 0.018 | 0.170 | 0.619 | 3.948 | 0.048 |
| laya-PB-ASYNC-r1 | 0 | 4-5 | 102 | 9.86 / 9.95 | 0.118 | 0.00 | 0.157 | 9.391 | 0.012 | 0.021 | 0.008 | 0.061 | 0.259 | 1.686 | 0.037 |
| laya-PB-ASYNC-r1 | 0 | steady 10-20 | 1013 | 9.86 / 10.34 | 0.120 | 0.00 | 0.166 | 9.390 | 0.011 | 0.020 | 0.008 | 0.062 | 0.263 | 1.684 | 0.035 |
| laya-PB-ASYNC-r1 | 1 | -2-0 | 0 | – / – | – | – | – | – | – | – | – | – | – | – | – |
| laya-PB-ASYNC-r1 | 1 | 0-0.5 | 37 | 12.06 / 47.57 | 0.631 | 0.97 | 1.027 | 11.029 | 0.028 | 0.098 | 0.037 | 0.317 | 1.252 | 7.381 | 0.205 |
| laya-PB-ASYNC-r1 | 1 | 0.5-1 | 37 | 12.45 / 18.38 | 0.716 | 1.00 | 1.417 | 9.910 | 0.034 | 0.071 | 0.044 | 0.378 | 1.426 | 8.283 | 0.251 |
| laya-PB-ASYNC-r1 | 1 | 1-2 | 75 | 12.70 / 17.39 | 0.765 | 1.00 | 1.379 | 9.929 | 0.036 | 0.074 | 0.050 | 0.415 | 1.563 | 9.119 | 0.289 |
| laya-PB-ASYNC-r1 | 1 | 2-4 | 158 | 12.31 / 17.20 | 0.604 | 0.87 | 1.054 | 9.841 | 0.033 | 0.057 | 0.041 | 0.332 | 1.270 | 7.338 | 0.235 |
| laya-PB-ASYNC-r1 | 1 | 4-5 | 101 | 9.87 / 10.31 | 0.119 | 0.00 | 0.164 | 9.402 | 0.011 | 0.020 | 0.008 | 0.062 | 0.263 | 1.686 | 0.037 |
| laya-PB-ASYNC-r1 | 1 | steady 10-20 | 1011 | 9.88 / 10.34 | 0.119 | 0.00 | 0.164 | 9.406 | 0.011 | 0.021 | 0.008 | 0.061 | 0.263 | 1.696 | 0.034 |
| laya-PB-ASYNC-r2 | 0 | -2-0 | 0 | – / – | – | – | – | – | – | – | – | – | – | – | – |
| laya-PB-ASYNC-r2 | 0 | 0-0.5 | 44 | 10.36 / 29.00 | 0.264 | 0.27 | 0.375 | 10.242 | 0.012 | 0.031 | 0.020 | 0.143 | 0.577 | 3.296 | 0.098 |
| laya-PB-ASYNC-r2 | 0 | 0.5-1 | 51 | 9.87 / 10.19 | 0.126 | 0.00 | 0.171 | 9.384 | 0.010 | 0.022 | 0.008 | 0.063 | 0.267 | 1.692 | 0.038 |
| laya-PB-ASYNC-r2 | 0 | 1-2 | 101 | 9.87 / 10.11 | 0.122 | 0.00 | 0.163 | 9.393 | 0.011 | 0.019 | 0.008 | 0.063 | 0.263 | 1.699 | 0.038 |
| laya-PB-ASYNC-r2 | 0 | 2-4 | 203 | 9.86 / 10.31 | 0.121 | 0.00 | 0.165 | 9.390 | 0.009 | 0.021 | 0.008 | 0.063 | 0.270 | 1.716 | 0.037 |
| laya-PB-ASYNC-r2 | 0 | 4-5 | 101 | 9.86 / 10.12 | 0.119 | 0.00 | 0.163 | 9.390 | 0.010 | 0.020 | 0.007 | 0.060 | 0.259 | 1.673 | 0.035 |
| laya-PB-ASYNC-r2 | 0 | steady 10-20 | 1012 | 9.86 / 10.36 | 0.122 | 0.00 | 0.167 | 9.388 | 0.010 | 0.020 | 0.008 | 0.062 | 0.266 | 1.715 | 0.037 |
| laya-PB-ASYNC-r2 | 1 | -2-0 | 0 | – / – | – | – | – | – | – | – | – | – | – | – | – |
| laya-PB-ASYNC-r2 | 1 | 0-0.5 | 36 | 12.24 / 43.51 | 0.632 | 0.97 | 1.010 | 11.119 | 0.032 | 0.057 | 0.040 | 0.341 | 1.326 | 7.706 | 0.223 |
| laya-PB-ASYNC-r2 | 1 | 0.5-1 | 39 | 12.33 / 17.01 | 0.694 | 1.00 | 1.197 | 9.861 | 0.032 | 0.059 | 0.043 | 0.374 | 1.420 | 8.361 | 0.260 |
| laya-PB-ASYNC-r2 | 1 | 1-2 | 76 | 12.62 / 17.12 | 0.741 | 1.00 | 1.275 | 9.870 | 0.034 | 0.063 | 0.045 | 0.407 | 1.517 | 8.929 | 0.255 |
| laya-PB-ASYNC-r2 | 1 | 2-4 | 154 | 12.44 / 17.24 | 0.690 | 0.99 | 1.171 | 9.865 | 0.035 | 0.084 | 0.044 | 0.382 | 1.432 | 8.485 | 0.235 |
| laya-PB-ASYNC-r2 | 1 | 4-5 | 75 | 12.71 / 17.83 | 0.780 | 1.00 | 1.188 | 9.893 | 0.040 | 0.069 | 0.050 | 0.436 | 1.578 | 9.027 | 0.288 |
| laya-PB-ASYNC-r2 | 1 | steady 10-20 | 893 | 9.93 / 17.15 | 0.363 | 0.41 | 0.600 | 9.584 | 0.013 | 0.046 | 0.023 | 0.196 | 0.760 | 4.715 | 0.048 |

## Per-thread counters by bucket

E share: CPU time on E cores / all. Relative effective cycle rate: cycles / CPU ns per level (relative only, not a clock frequency). CPU/unit: per short request (client-short), long request (client-long), ANE forward (ANE dispatcher, callback), GPU forward (GPU worker).

| run | cycle | bucket | thread | CPU ms | E share | relative effective cycle rate P / E | IPC | CPU/unit ms |
|---|---|---|---|---|---|---|---|---|
| laya-A-r1 | 0 | -2-0 | ane-dispatch | 8.2 | 0.20 | 1.84 / 1.32 | 3.75 | – |
| laya-A-r1 | 0 | -2-0 | client-short | 6.3 | 0.95 | 1.29 / 1.08 | 1.21 | – |
| laya-A-r1 | 0 | -2-0 | client-long | 6.8 | 0.95 | 1.28 / 1.07 | 1.50 | – |
| laya-A-r1 | 0 | -2-0 | gpu-dispatch | 0.2 | 1.00 | – / 1.35 | 0.63 | – |
| laya-A-r1 | 0 | -2-0 | gpu-worker | 18.1 | 0.58 | 1.76 / 1.83 | 3.83 | – |
| laya-A-r1 | 0 | -2-0 | main | 20.8 | 0.94 | 2.59 / 1.12 | 1.93 | – |
| laya-A-r1 | 0 | 0-0.5 | ane-dispatch | 56.9 | 0.37 | 2.74 / 1.85 | 3.04 | 1.422 |
| laya-A-r1 | 0 | 0-0.5 | client-short | 13.6 | 0.52 | 3.20 / 1.71 | 3.29 | 0.341 |
| laya-A-r1 | 0 | 0-0.5 | client-long | 12.1 | 0.63 | 3.35 / 1.57 | 3.93 | 1.010 |
| laya-A-r1 | 0 | 0-0.5 | gpu-dispatch | 2.8 | 0.51 | 2.95 / 1.70 | 1.42 | – |
| laya-A-r1 | 0 | 0-0.5 | gpu-worker | 80.1 | 0.52 | 2.55 / 1.84 | 2.84 | 6.676 |
| laya-A-r1 | 0 | 0-0.5 | main | 0.0 | – | – / – | – | – |
| laya-A-r1 | 0 | 0.5-1 | ane-dispatch | 30.5 | 0.00 | 4.06 / – | 3.51 | 0.622 |
| laya-A-r1 | 0 | 0.5-1 | client-short | 7.0 | 0.00 | 4.29 / – | 4.34 | 0.143 |
| laya-A-r1 | 0 | 0.5-1 | client-long | 5.2 | 0.00 | 4.43 / – | 5.02 | 0.432 |
| laya-A-r1 | 0 | 0.5-1 | gpu-dispatch | 1.4 | 0.00 | 3.64 / – | 2.01 | – |
| laya-A-r1 | 0 | 0.5-1 | gpu-worker | 36.7 | 0.00 | 3.11 / – | 3.32 | 3.059 |
| laya-A-r1 | 0 | 0.5-1 | main | 0.0 | – | – / – | – | – |
| laya-A-r1 | 0 | 1-2 | ane-dispatch | 62.5 | 0.00 | 4.05 / – | 3.42 | 0.638 |
| laya-A-r1 | 0 | 1-2 | client-short | 14.3 | 0.00 | 4.28 / – | 4.19 | 0.146 |
| laya-A-r1 | 0 | 1-2 | client-long | 10.4 | 0.00 | 4.40 / – | 4.87 | 0.416 |
| laya-A-r1 | 0 | 1-2 | gpu-dispatch | 2.9 | 0.00 | 3.69 / – | 1.97 | – |
| laya-A-r1 | 0 | 1-2 | gpu-worker | 74.5 | 0.00 | 3.16 / – | 3.18 | 3.106 |
| laya-A-r1 | 0 | 1-2 | main | 0.0 | – | – / – | – | – |
| laya-A-r1 | 0 | 2-4 | ane-dispatch | 120.8 | 0.00 | 4.06 / – | 3.54 | 0.613 |
| laya-A-r1 | 0 | 2-4 | client-short | 27.5 | 0.00 | 4.27 / – | 4.34 | 0.139 |
| laya-A-r1 | 0 | 2-4 | client-long | 20.7 | 0.00 | 4.43 / – | 4.95 | 0.422 |
| laya-A-r1 | 0 | 2-4 | gpu-dispatch | 5.6 | 0.00 | 3.65 / – | 2.06 | – |
| laya-A-r1 | 0 | 2-4 | gpu-worker | 148.5 | 0.00 | 3.12 / – | 3.32 | 2.969 |
| laya-A-r1 | 0 | 2-4 | main | 0.0 | – | – / – | – | – |
| laya-A-r1 | 0 | 4-5 | ane-dispatch | 59.7 | 0.00 | 4.07 / – | 3.59 | 0.603 |
| laya-A-r1 | 0 | 4-5 | client-short | 13.6 | 0.00 | 4.27 / – | 4.40 | 0.137 |
| laya-A-r1 | 0 | 4-5 | client-long | 10.1 | 0.00 | 4.41 / – | 4.99 | 0.419 |
| laya-A-r1 | 0 | 4-5 | gpu-dispatch | 2.7 | 0.00 | 3.67 / – | 2.13 | – |
| laya-A-r1 | 0 | 4-5 | gpu-worker | 71.9 | 0.00 | 3.12 / – | 3.36 | 2.997 |
| laya-A-r1 | 0 | 4-5 | main | 0.0 | – | – / – | – | – |
| laya-A-r1 | 0 | steady 10-20 | ane-dispatch | 609.4 | 0.00 | 4.06 / 2.53 | 3.51 | 0.619 |
| laya-A-r1 | 0 | steady 10-20 | client-short | 138.2 | 0.00 | 4.27 / 2.23 | 4.30 | 0.140 |
| laya-A-r1 | 0 | steady 10-20 | client-long | 103.3 | 0.00 | 4.41 / 2.43 | 4.92 | 0.420 |
| laya-A-r1 | 0 | steady 10-20 | gpu-dispatch | 28.3 | 0.00 | 3.67 / 2.46 | 2.03 | – |
| laya-A-r1 | 0 | steady 10-20 | gpu-worker | 738.6 | 0.00 | 3.14 / 2.08 | 3.28 | 3.003 |
| laya-A-r1 | 0 | steady 10-20 | main | 0.1 | 0.00 | 4.44 / – | 2.13 | – |
| laya-A-r1 | 1 | -2-0 | ane-dispatch | 3.8 | 0.14 | 2.05 / 1.74 | 3.40 | – |
| laya-A-r1 | 1 | -2-0 | client-short | 5.8 | 0.94 | 1.95 / 1.12 | 1.32 | – |
| laya-A-r1 | 1 | -2-0 | client-long | 5.7 | 0.96 | 1.68 / 1.13 | 1.49 | – |
| laya-A-r1 | 1 | -2-0 | gpu-dispatch | 0.1 | 0.49 | 1.71 / 1.75 | 0.97 | – |
| laya-A-r1 | 1 | -2-0 | gpu-worker | 7.7 | 0.40 | 1.98 / 2.21 | 3.63 | – |
| laya-A-r1 | 1 | -2-0 | main | 20.3 | 0.96 | 2.59 / 1.07 | 1.99 | – |
| laya-A-r1 | 1 | 0-0.5 | ane-dispatch | 52.7 | 0.12 | 2.78 / 2.12 | 3.29 | 1.254 |
| laya-A-r1 | 1 | 0-0.5 | client-short | 11.5 | 0.31 | 3.21 / 1.69 | 3.58 | 0.274 |
| laya-A-r1 | 1 | 0-0.5 | client-long | 9.8 | 0.48 | 3.42 / 1.78 | 4.01 | 0.889 |
| laya-A-r1 | 1 | 0-0.5 | gpu-dispatch | 2.3 | 0.35 | 2.88 / 2.05 | 1.51 | – |
| laya-A-r1 | 1 | 0-0.5 | gpu-worker | 75.1 | 0.24 | 2.39 / 2.21 | 3.32 | 6.826 |
| laya-A-r1 | 1 | 0-0.5 | main | 0.0 | – | – / – | – | – |
| laya-A-r1 | 1 | 0.5-1 | ane-dispatch | 30.5 | 0.00 | 4.06 / – | 3.50 | 0.622 |
| laya-A-r1 | 1 | 0.5-1 | client-short | 7.1 | 0.00 | 4.27 / – | 4.32 | 0.144 |
| laya-A-r1 | 1 | 0.5-1 | client-long | 5.2 | 0.00 | 4.43 / – | 4.92 | 0.433 |
| laya-A-r1 | 1 | 0.5-1 | gpu-dispatch | 1.4 | 0.00 | 3.66 / – | 2.03 | – |
| laya-A-r1 | 1 | 0.5-1 | gpu-worker | 36.8 | 0.00 | 3.09 / – | 3.30 | 3.068 |
| laya-A-r1 | 1 | 0.5-1 | main | 0.0 | – | – / – | – | – |
| laya-A-r1 | 1 | 1-2 | ane-dispatch | 60.2 | 0.00 | 4.04 / – | 3.57 | 0.608 |
| laya-A-r1 | 1 | 1-2 | client-short | 13.7 | 0.00 | 4.29 / – | 4.35 | 0.139 |
| laya-A-r1 | 1 | 1-2 | client-long | 10.5 | 0.00 | 4.40 / – | 4.97 | 0.421 |
| laya-A-r1 | 1 | 1-2 | gpu-dispatch | 2.8 | 0.00 | 3.63 / – | 2.10 | – |
| laya-A-r1 | 1 | 1-2 | gpu-worker | 73.5 | 0.00 | 3.13 / – | 3.36 | 2.940 |
| laya-A-r1 | 1 | 1-2 | main | 0.0 | – | – / – | – | – |
| laya-A-r1 | 1 | 2-4 | ane-dispatch | 119.9 | 0.00 | 4.06 / – | 3.57 | 0.609 |
| laya-A-r1 | 1 | 2-4 | client-short | 27.2 | 0.00 | 4.28 / – | 4.38 | 0.138 |
| laya-A-r1 | 1 | 2-4 | client-long | 20.3 | 0.00 | 4.42 / – | 4.99 | 0.414 |
| laya-A-r1 | 1 | 2-4 | gpu-dispatch | 5.4 | 0.00 | 3.66 / – | 2.13 | – |
| laya-A-r1 | 1 | 2-4 | gpu-worker | 144.0 | 0.00 | 3.11 / – | 3.38 | 2.938 |
| laya-A-r1 | 1 | 2-4 | main | 0.0 | – | – / – | – | – |
| laya-A-r1 | 1 | 4-5 | ane-dispatch | 61.0 | 0.00 | 4.05 / – | 3.51 | 0.622 |
| laya-A-r1 | 1 | 4-5 | client-short | 13.9 | 0.00 | 4.28 / – | 4.28 | 0.142 |
| laya-A-r1 | 1 | 4-5 | client-long | 10.6 | 0.00 | 4.41 / – | 4.89 | 0.424 |
| laya-A-r1 | 1 | 4-5 | gpu-dispatch | 2.9 | 0.00 | 3.63 / – | 2.01 | – |
| laya-A-r1 | 1 | 4-5 | gpu-worker | 75.3 | 0.00 | 3.15 / – | 3.26 | 3.011 |
| laya-A-r1 | 1 | 4-5 | main | 0.0 | – | – / – | – | – |
| laya-A-r1 | 1 | steady 10-20 | ane-dispatch | 603.1 | 0.00 | 4.05 / – | 3.56 | 0.612 |
| laya-A-r1 | 1 | steady 10-20 | client-short | 137.0 | 0.00 | 4.27 / – | 4.35 | 0.139 |
| laya-A-r1 | 1 | steady 10-20 | client-long | 102.5 | 0.00 | 4.41 / – | 4.97 | 0.417 |
| laya-A-r1 | 1 | steady 10-20 | gpu-dispatch | 27.7 | 0.00 | 3.65 / 1.02 | 2.09 | – |
| laya-A-r1 | 1 | steady 10-20 | gpu-worker | 731.2 | 0.00 | 3.12 / – | 3.33 | 2.972 |
| laya-A-r1 | 1 | steady 10-20 | main | 0.0 | 0.00 | 4.43 / – | 2.12 | – |
| laya-A-r2 | 0 | -2-0 | ane-dispatch | 8.8 | 0.00 | 2.61 / – | 3.76 | – |
| laya-A-r2 | 0 | -2-0 | client-short | 6.1 | 0.29 | 1.71 / 1.19 | 1.11 | – |
| laya-A-r2 | 0 | -2-0 | client-long | 6.9 | 0.24 | 1.76 / 1.21 | 1.44 | – |
| laya-A-r2 | 0 | -2-0 | gpu-dispatch | 0.2 | 0.00 | 2.04 / – | 1.33 | – |
| laya-A-r2 | 0 | -2-0 | gpu-worker | 18.0 | 0.00 | 2.30 / – | 5.38 | – |
| laya-A-r2 | 0 | -2-0 | main | 18.4 | 0.92 | 2.60 / 1.30 | 1.82 | – |
| laya-A-r2 | 0 | 0-0.5 | ane-dispatch | 41.7 | 0.00 | 2.95 / – | 3.39 | 0.949 |
| laya-A-r2 | 0 | 0-0.5 | client-short | 9.4 | 0.00 | 3.02 / – | 4.09 | 0.213 |
| laya-A-r2 | 0 | 0-0.5 | client-long | 7.6 | 0.00 | 2.89 / – | 4.77 | 0.633 |
| laya-A-r2 | 0 | 0-0.5 | gpu-dispatch | 1.9 | 0.00 | 2.64 / – | 1.83 | – |
| laya-A-r2 | 0 | 0-0.5 | gpu-worker | 48.3 | 0.00 | 2.55 / – | 3.46 | 4.024 |
| laya-A-r2 | 0 | 0-0.5 | main | 0.0 | – | – / – | – | – |
| laya-A-r2 | 0 | 0.5-1 | ane-dispatch | 31.3 | 0.00 | 4.04 / 2.52 | 3.42 | 0.639 |
| laya-A-r2 | 0 | 0.5-1 | client-short | 7.2 | 0.00 | 4.27 / 2.14 | 4.21 | 0.147 |
| laya-A-r2 | 0 | 0.5-1 | client-long | 5.2 | 0.00 | 4.38 / 2.16 | 4.96 | 0.431 |
| laya-A-r2 | 0 | 0.5-1 | gpu-dispatch | 1.5 | 0.01 | 3.66 / 2.44 | 1.93 | – |
| laya-A-r2 | 0 | 0.5-1 | gpu-worker | 36.6 | 0.00 | 3.14 / – | 3.23 | 3.050 |
| laya-A-r2 | 0 | 0.5-1 | main | 0.0 | – | – / – | – | – |
| laya-A-r2 | 0 | 1-2 | ane-dispatch | 61.4 | 0.00 | 4.06 / 2.52 | 3.48 | 0.626 |
| laya-A-r2 | 0 | 1-2 | client-short | 14.2 | 0.00 | 4.26 / 2.14 | 4.28 | 0.144 |
| laya-A-r2 | 0 | 1-2 | client-long | 10.5 | 0.00 | 4.43 / 2.16 | 4.94 | 0.421 |
| laya-A-r2 | 0 | 1-2 | gpu-dispatch | 3.0 | 0.00 | 3.67 / 2.44 | 1.96 | – |
| laya-A-r2 | 0 | 1-2 | gpu-worker | 75.4 | 0.00 | 3.16 / – | 3.23 | 3.014 |
| laya-A-r2 | 0 | 1-2 | main | 0.0 | – | – / – | – | – |
| laya-A-r2 | 0 | 2-4 | ane-dispatch | 121.8 | 0.00 | 4.05 / – | 3.53 | 0.618 |
| laya-A-r2 | 0 | 2-4 | client-short | 27.3 | 0.00 | 4.27 / – | 4.36 | 0.138 |
| laya-A-r2 | 0 | 2-4 | client-long | 20.1 | 0.00 | 4.43 / – | 5.02 | 0.411 |
| laya-A-r2 | 0 | 2-4 | gpu-dispatch | 5.6 | 0.00 | 3.67 / – | 2.04 | – |
| laya-A-r2 | 0 | 2-4 | gpu-worker | 146.8 | 0.00 | 3.12 / – | 3.30 | 2.995 |
| laya-A-r2 | 0 | 2-4 | main | 0.0 | – | – / – | – | – |
| laya-A-r2 | 0 | 4-5 | ane-dispatch | 60.7 | 0.00 | 4.04 / – | 3.54 | 0.613 |
| laya-A-r2 | 0 | 4-5 | client-short | 13.7 | 0.00 | 4.28 / – | 4.37 | 0.139 |
| laya-A-r2 | 0 | 4-5 | client-long | 10.4 | 0.00 | 4.40 / – | 5.01 | 0.414 |
| laya-A-r2 | 0 | 4-5 | gpu-dispatch | 2.9 | 0.00 | 3.63 / – | 2.07 | – |
| laya-A-r2 | 0 | 4-5 | gpu-worker | 73.6 | 0.00 | 3.12 / – | 3.35 | 2.942 |
| laya-A-r2 | 0 | 4-5 | main | 0.0 | – | – / – | – | – |
| laya-A-r2 | 0 | steady 10-20 | ane-dispatch | 618.8 | 0.00 | 4.02 / 2.50 | 3.49 | 0.631 |
| laya-A-r2 | 0 | steady 10-20 | client-short | 139.1 | 0.00 | 4.24 / 2.58 | 4.29 | 0.142 |
| laya-A-r2 | 0 | steady 10-20 | client-long | 102.9 | 0.00 | 4.38 / – | 4.96 | 0.420 |
| laya-A-r2 | 0 | steady 10-20 | gpu-dispatch | 29.0 | 0.00 | 3.64 / 2.59 | 2.00 | – |
| laya-A-r2 | 0 | steady 10-20 | gpu-worker | 739.3 | 0.00 | 3.15 / 2.55 | 3.26 | 3.018 |
| laya-A-r2 | 0 | steady 10-20 | main | 0.0 | 0.00 | 3.56 / – | 0.68 | – |
| laya-A-r2 | 1 | -2-0 | ane-dispatch | 11.5 | 0.23 | 1.79 / 1.15 | 3.82 | – |
| laya-A-r2 | 1 | -2-0 | client-short | 7.5 | 0.84 | 1.41 / 1.07 | 1.17 | – |
| laya-A-r2 | 1 | -2-0 | client-long | 8.4 | 0.84 | 1.64 / 1.06 | 1.40 | – |
| laya-A-r2 | 1 | -2-0 | gpu-dispatch | 0.3 | 1.00 | – / 1.02 | 1.06 | – |
| laya-A-r2 | 1 | -2-0 | gpu-worker | 27.9 | 0.52 | 1.80 / 1.73 | 3.92 | – |
| laya-A-r2 | 1 | -2-0 | main | 19.2 | 0.95 | 2.59 / 1.26 | 1.81 | – |
| laya-A-r2 | 1 | 0-0.5 | ane-dispatch | 64.0 | 0.54 | 2.59 / 1.76 | 2.76 | 1.685 |
| laya-A-r2 | 1 | 0-0.5 | client-short | 15.8 | 0.66 | 3.07 / 1.69 | 3.01 | 0.415 |
| laya-A-r2 | 1 | 0-0.5 | client-long | 14.1 | 0.67 | 2.76 / 1.57 | 3.68 | 1.282 |
| laya-A-r2 | 1 | 0-0.5 | gpu-dispatch | 3.0 | 0.59 | 2.89 / 1.60 | 1.38 | – |
| laya-A-r2 | 1 | 0-0.5 | gpu-worker | 89.0 | 0.60 | 2.32 / 1.72 | 2.61 | 8.092 |
| laya-A-r2 | 1 | 0-0.5 | main | 0.0 | – | – / – | – | – |
| laya-A-r2 | 1 | 0.5-1 | ane-dispatch | 30.3 | 0.00 | 4.05 / – | 3.54 | 0.619 |
| laya-A-r2 | 1 | 0.5-1 | client-short | 6.9 | 0.00 | 4.29 / – | 4.40 | 0.141 |
| laya-A-r2 | 1 | 0.5-1 | client-long | 5.1 | 0.00 | 4.44 / – | 5.04 | 0.421 |
| laya-A-r2 | 1 | 0.5-1 | gpu-dispatch | 1.4 | 0.00 | 3.72 / – | 1.95 | – |
| laya-A-r2 | 1 | 0.5-1 | gpu-worker | 36.4 | 0.00 | 3.10 / – | 3.30 | 3.033 |
| laya-A-r2 | 1 | 0.5-1 | main | 0.0 | – | – / – | – | – |
| laya-A-r2 | 1 | 1-2 | ane-dispatch | 60.8 | 0.00 | 4.03 / – | 3.55 | 0.614 |
| laya-A-r2 | 1 | 1-2 | client-short | 13.7 | 0.00 | 4.24 / – | 4.42 | 0.138 |
| laya-A-r2 | 1 | 1-2 | client-long | 10.6 | 0.00 | 4.41 / – | 4.90 | 0.423 |
| laya-A-r2 | 1 | 1-2 | gpu-dispatch | 2.9 | 0.00 | 3.67 / – | 2.01 | – |
| laya-A-r2 | 1 | 1-2 | gpu-worker | 73.4 | 0.00 | 3.10 / – | 3.36 | 2.935 |
| laya-A-r2 | 1 | 1-2 | main | 0.0 | – | – / – | – | – |
| laya-A-r2 | 1 | 2-4 | ane-dispatch | 121.5 | 0.00 | 4.03 / – | 3.55 | 0.617 |
| laya-A-r2 | 1 | 2-4 | client-short | 27.3 | 0.00 | 4.26 / – | 4.38 | 0.139 |
| laya-A-r2 | 1 | 2-4 | client-long | 20.1 | 0.00 | 4.43 / – | 5.03 | 0.411 |
| laya-A-r2 | 1 | 2-4 | gpu-dispatch | 5.5 | 0.00 | 3.68 / – | 2.08 | – |
| laya-A-r2 | 1 | 2-4 | gpu-worker | 145.7 | 0.00 | 3.09 / – | 3.37 | 2.973 |
| laya-A-r2 | 1 | 2-4 | main | 0.0 | – | – / – | – | – |
| laya-A-r2 | 1 | 4-5 | ane-dispatch | 60.5 | 0.00 | 4.04 / – | 3.55 | 0.617 |
| laya-A-r2 | 1 | 4-5 | client-short | 13.5 | 0.00 | 4.28 / – | 4.39 | 0.137 |
| laya-A-r2 | 1 | 4-5 | client-long | 10.0 | 0.00 | 4.42 / – | 5.00 | 0.399 |
| laya-A-r2 | 1 | 4-5 | gpu-dispatch | 2.8 | 0.00 | 3.67 / – | 2.04 | – |
| laya-A-r2 | 1 | 4-5 | gpu-worker | 72.3 | 0.00 | 3.10 / – | 3.33 | 2.894 |
| laya-A-r2 | 1 | 4-5 | main | 0.0 | – | – / – | – | – |
| laya-A-r2 | 1 | steady 10-20 | ane-dispatch | 615.8 | 0.00 | 3.99 / – | 3.52 | 0.628 |
| laya-A-r2 | 1 | steady 10-20 | client-short | 135.9 | 0.00 | 4.26 / – | 4.35 | 0.139 |
| laya-A-r2 | 1 | steady 10-20 | client-long | 102.2 | 0.00 | 4.38 / – | 4.97 | 0.417 |
| laya-A-r2 | 1 | steady 10-20 | gpu-dispatch | 28.2 | 0.00 | 3.64 / – | 2.05 | – |
| laya-A-r2 | 1 | steady 10-20 | gpu-worker | 739.0 | 0.00 | 3.09 / – | 3.31 | 3.016 |
| laya-A-r2 | 1 | steady 10-20 | main | 0.3 | 0.00 | 4.43 / – | 2.12 | – |
| laya-PB-ASYNC-r1 | 0 | -2-0 | ane-dispatch | 0.5 | 1.00 | – / 2.11 | 0.54 | – |
| laya-PB-ASYNC-r1 | 0 | -2-0 | client-short | 7.1 | 0.82 | 1.31 / 1.16 | 1.28 | – |
| laya-PB-ASYNC-r1 | 0 | -2-0 | client-long | 11.2 | 0.90 | 1.42 / 1.05 | 1.50 | – |
| laya-PB-ASYNC-r1 | 0 | -2-0 | gpu-dispatch | 0.4 | 1.00 | – / 1.10 | 1.09 | – |
| laya-PB-ASYNC-r1 | 0 | -2-0 | callback | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r1 | 0 | -2-0 | gpu-worker | 1.8 | 0.72 | 2.22 / 1.70 | 0.75 | – |
| laya-PB-ASYNC-r1 | 0 | -2-0 | main | 21.0 | 0.95 | 2.58 / 1.11 | 1.88 | – |
| laya-PB-ASYNC-r1 | 0 | 0-0.5 | ane-dispatch | 56.6 | 0.94 | 1.86 / 1.52 | 1.73 | 1.572 |
| laya-PB-ASYNC-r1 | 0 | 0-0.5 | client-short | 26.6 | 0.95 | 1.90 / 1.50 | 2.37 | 0.740 |
| laya-PB-ASYNC-r1 | 0 | 0-0.5 | client-long | 22.4 | 0.97 | 2.62 / 1.42 | 3.11 | 1.869 |
| laya-PB-ASYNC-r1 | 0 | 0-0.5 | gpu-dispatch | 4.5 | 0.95 | 2.62 / 1.43 | 1.23 | – |
| laya-PB-ASYNC-r1 | 0 | 0-0.5 | callback | 10.6 | 0.93 | 1.96 / 1.46 | 0.84 | 0.294 |
| laya-PB-ASYNC-r1 | 0 | 0-0.5 | gpu-worker | 159.4 | 0.87 | 1.95 / 1.54 | 2.50 | 13.282 |
| laya-PB-ASYNC-r1 | 0 | 0-0.5 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r1 | 0 | 0.5-1 | ane-dispatch | 68.0 | 1.00 | – / 1.34 | 1.79 | 1.743 |
| laya-PB-ASYNC-r1 | 0 | 0.5-1 | client-short | 32.0 | 1.00 | – / 1.34 | 2.40 | 0.821 |
| laya-PB-ASYNC-r1 | 0 | 0.5-1 | client-long | 34.4 | 1.00 | – / 1.08 | 3.09 | 2.865 |
| laya-PB-ASYNC-r1 | 0 | 0.5-1 | gpu-dispatch | 6.4 | 1.00 | – / 1.07 | 1.35 | – |
| laya-PB-ASYNC-r1 | 0 | 0.5-1 | callback | 11.6 | 1.00 | – / 1.33 | 0.89 | 0.297 |
| laya-PB-ASYNC-r1 | 0 | 0.5-1 | gpu-worker | 141.8 | 1.00 | – / 1.40 | 1.87 | 11.819 |
| laya-PB-ASYNC-r1 | 0 | 0.5-1 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r1 | 0 | 1-2 | ane-dispatch | 137.7 | 1.00 | – / 1.31 | 1.76 | 1.812 |
| laya-PB-ASYNC-r1 | 0 | 1-2 | client-short | 65.2 | 1.00 | – / 1.30 | 2.37 | 0.846 |
| laya-PB-ASYNC-r1 | 0 | 1-2 | client-long | 65.8 | 1.00 | – / 1.10 | 3.06 | 2.861 |
| laya-PB-ASYNC-r1 | 0 | 1-2 | gpu-dispatch | 13.0 | 1.00 | – / 1.08 | 1.30 | – |
| laya-PB-ASYNC-r1 | 0 | 1-2 | callback | 24.0 | 1.00 | – / 1.31 | 0.86 | 0.315 |
| laya-PB-ASYNC-r1 | 0 | 1-2 | gpu-worker | 274.7 | 1.00 | – / 1.39 | 1.86 | 11.941 |
| laya-PB-ASYNC-r1 | 0 | 1-2 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r1 | 0 | 2-4 | ane-dispatch | 136.8 | 0.66 | 3.82 / 1.23 | 2.62 | 0.740 |
| laya-PB-ASYNC-r1 | 0 | 2-4 | client-short | 62.5 | 0.67 | 4.01 / 1.23 | 3.57 | 0.340 |
| laya-PB-ASYNC-r1 | 0 | 2-4 | client-long | 57.0 | 0.71 | 4.25 / 1.05 | 4.26 | 1.076 |
| laya-PB-ASYNC-r1 | 0 | 2-4 | gpu-dispatch | 11.6 | 0.69 | 3.76 / 1.05 | 1.76 | – |
| laya-PB-ASYNC-r1 | 0 | 2-4 | callback | 25.8 | 0.60 | 1.82 / 1.20 | 1.24 | 0.140 |
| laya-PB-ASYNC-r1 | 0 | 2-4 | gpu-worker | 307.6 | 0.62 | 3.12 / 1.29 | 2.68 | 5.803 |
| laya-PB-ASYNC-r1 | 0 | 2-4 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r1 | 0 | 4-5 | ane-dispatch | 31.0 | 0.00 | 4.06 / – | 3.27 | 0.304 |
| laya-PB-ASYNC-r1 | 0 | 4-5 | client-short | 13.8 | 0.00 | 4.24 / – | 4.45 | 0.135 |
| laya-PB-ASYNC-r1 | 0 | 4-5 | client-long | 11.1 | 0.00 | 4.42 / – | 5.06 | 0.396 |
| laya-PB-ASYNC-r1 | 0 | 4-5 | gpu-dispatch | 2.3 | 0.00 | 3.91 / – | 2.21 | – |
| laya-PB-ASYNC-r1 | 0 | 4-5 | callback | 7.6 | 0.00 | 1.72 / – | 1.65 | 0.074 |
| laya-PB-ASYNC-r1 | 0 | 4-5 | gpu-worker | 82.3 | 0.00 | 3.19 / – | 3.26 | 2.938 |
| laya-PB-ASYNC-r1 | 0 | 4-5 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r1 | 0 | steady 10-20 | ane-dispatch | 314.9 | 0.00 | 4.03 / – | 3.26 | 0.311 |
| laya-PB-ASYNC-r1 | 0 | steady 10-20 | client-short | 138.8 | 0.00 | 4.25 / – | 4.41 | 0.137 |
| laya-PB-ASYNC-r1 | 0 | steady 10-20 | client-long | 113.0 | 0.00 | 4.42 / – | 5.05 | 0.405 |
| laya-PB-ASYNC-r1 | 0 | steady 10-20 | gpu-dispatch | 22.8 | 0.00 | 3.91 / – | 2.26 | – |
| laya-PB-ASYNC-r1 | 0 | steady 10-20 | callback | 70.6 | 0.00 | 1.83 / – | 1.66 | 0.070 |
| laya-PB-ASYNC-r1 | 0 | steady 10-20 | gpu-worker | 817.6 | 0.00 | 3.17 / – | 3.30 | 2.931 |
| laya-PB-ASYNC-r1 | 0 | steady 10-20 | main | 0.3 | 0.00 | 4.47 / – | 2.06 | – |
| laya-PB-ASYNC-r1 | 1 | -2-0 | ane-dispatch | 1.2 | 1.00 | – / 1.02 | 1.45 | – |
| laya-PB-ASYNC-r1 | 1 | -2-0 | client-short | 6.4 | 0.93 | 1.62 / 1.05 | 1.20 | – |
| laya-PB-ASYNC-r1 | 1 | -2-0 | client-long | 6.8 | 0.94 | 1.67 / 1.06 | 1.49 | – |
| laya-PB-ASYNC-r1 | 1 | -2-0 | gpu-dispatch | 0.2 | 1.00 | – / 1.02 | 1.09 | – |
| laya-PB-ASYNC-r1 | 1 | -2-0 | callback | 0.1 | 0.00 | 1.51 / – | 0.59 | – |
| laya-PB-ASYNC-r1 | 1 | -2-0 | gpu-worker | 21.0 | 0.51 | 1.85 / 1.78 | 3.84 | – |
| laya-PB-ASYNC-r1 | 1 | -2-0 | main | 20.9 | 0.95 | 2.57 / 1.09 | 1.94 | – |
| laya-PB-ASYNC-r1 | 1 | 0-0.5 | ane-dispatch | 52.8 | 0.91 | 1.88 / 1.61 | 1.73 | 1.428 |
| laya-PB-ASYNC-r1 | 1 | 0-0.5 | client-short | 26.0 | 0.92 | 1.97 / 1.60 | 2.31 | 0.703 |
| laya-PB-ASYNC-r1 | 1 | 0-0.5 | client-long | 25.5 | 0.91 | 1.75 / 1.33 | 3.20 | 2.126 |
| laya-PB-ASYNC-r1 | 1 | 0-0.5 | gpu-dispatch | 4.8 | 0.90 | 1.80 / 1.33 | 1.31 | – |
| laya-PB-ASYNC-r1 | 1 | 0-0.5 | callback | 9.7 | 0.88 | 1.86 / 1.58 | 0.82 | 0.263 |
| laya-PB-ASYNC-r1 | 1 | 0-0.5 | gpu-worker | 132.8 | 0.82 | 1.88 / 1.64 | 2.33 | 11.063 |
| laya-PB-ASYNC-r1 | 1 | 0-0.5 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r1 | 1 | 0.5-1 | ane-dispatch | 65.2 | 1.00 | – / 1.44 | 1.68 | 1.761 |
| laya-PB-ASYNC-r1 | 1 | 0.5-1 | client-short | 32.0 | 1.00 | – / 1.42 | 2.28 | 0.864 |
| laya-PB-ASYNC-r1 | 1 | 0.5-1 | client-long | 31.7 | 1.00 | – / 1.15 | 3.02 | 2.644 |
| laya-PB-ASYNC-r1 | 1 | 0.5-1 | gpu-dispatch | 6.0 | 1.00 | – / 1.17 | 1.25 | – |
| laya-PB-ASYNC-r1 | 1 | 0.5-1 | callback | 11.9 | 1.00 | – / 1.41 | 0.81 | 0.322 |
| laya-PB-ASYNC-r1 | 1 | 0.5-1 | gpu-worker | 135.2 | 1.00 | – / 1.51 | 1.81 | 11.270 |
| laya-PB-ASYNC-r1 | 1 | 0.5-1 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r1 | 1 | 1-2 | ane-dispatch | 140.7 | 1.00 | – / 1.28 | 1.72 | 1.876 |
| laya-PB-ASYNC-r1 | 1 | 1-2 | client-short | 66.3 | 1.00 | – / 1.30 | 2.30 | 0.884 |
| laya-PB-ASYNC-r1 | 1 | 1-2 | client-long | 63.0 | 1.00 | – / 1.11 | 3.08 | 2.741 |
| laya-PB-ASYNC-r1 | 1 | 1-2 | gpu-dispatch | 12.9 | 1.00 | – / 1.08 | 1.28 | – |
| laya-PB-ASYNC-r1 | 1 | 1-2 | callback | 24.8 | 1.00 | – / 1.24 | 0.84 | 0.331 |
| laya-PB-ASYNC-r1 | 1 | 1-2 | gpu-worker | 289.8 | 1.00 | – / 1.32 | 1.86 | 12.599 |
| laya-PB-ASYNC-r1 | 1 | 1-2 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r1 | 1 | 2-4 | ane-dispatch | 241.7 | 0.94 | 2.59 / 1.42 | 1.83 | 1.529 |
| laya-PB-ASYNC-r1 | 1 | 2-4 | client-short | 110.9 | 0.94 | 2.68 / 1.42 | 2.49 | 0.702 |
| laya-PB-ASYNC-r1 | 1 | 2-4 | client-long | 109.5 | 0.96 | 3.08 / 1.20 | 3.22 | 2.235 |
| laya-PB-ASYNC-r1 | 1 | 2-4 | gpu-dispatch | 21.7 | 0.95 | 2.73 / 1.20 | 1.34 | – |
| laya-PB-ASYNC-r1 | 1 | 2-4 | callback | 42.9 | 0.93 | 1.78 / 1.38 | 0.86 | 0.271 |
| laya-PB-ASYNC-r1 | 1 | 2-4 | gpu-worker | 497.7 | 0.94 | 2.51 / 1.52 | 1.93 | 10.157 |
| laya-PB-ASYNC-r1 | 1 | 2-4 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r1 | 1 | 4-5 | ane-dispatch | 31.5 | 0.00 | 4.05 / – | 3.24 | 0.312 |
| laya-PB-ASYNC-r1 | 1 | 4-5 | client-short | 13.9 | 0.00 | 4.26 / – | 4.42 | 0.138 |
| laya-PB-ASYNC-r1 | 1 | 4-5 | client-long | 11.2 | 0.00 | 4.43 / – | 5.13 | 0.400 |
| laya-PB-ASYNC-r1 | 1 | 4-5 | gpu-dispatch | 2.6 | 0.00 | 3.96 / – | 2.19 | – |
| laya-PB-ASYNC-r1 | 1 | 4-5 | callback | 7.1 | 0.00 | 1.81 / – | 1.66 | 0.071 |
| laya-PB-ASYNC-r1 | 1 | 4-5 | gpu-worker | 82.4 | 0.00 | 3.13 / – | 3.32 | 2.944 |
| laya-PB-ASYNC-r1 | 1 | 4-5 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r1 | 1 | steady 10-20 | ane-dispatch | 313.7 | 0.00 | 4.03 / 0.99 | 3.26 | 0.310 |
| laya-PB-ASYNC-r1 | 1 | steady 10-20 | client-short | 138.7 | 0.00 | 4.24 / – | 4.44 | 0.137 |
| laya-PB-ASYNC-r1 | 1 | steady 10-20 | client-long | 113.1 | 0.00 | 4.42 / – | 5.08 | 0.405 |
| laya-PB-ASYNC-r1 | 1 | steady 10-20 | gpu-dispatch | 22.4 | 0.00 | 3.92 / – | 2.28 | – |
| laya-PB-ASYNC-r1 | 1 | steady 10-20 | callback | 69.5 | 0.00 | 1.82 / – | 1.69 | 0.069 |
| laya-PB-ASYNC-r1 | 1 | steady 10-20 | gpu-worker | 822.4 | 0.00 | 3.15 / – | 3.31 | 2.948 |
| laya-PB-ASYNC-r1 | 1 | steady 10-20 | main | 0.1 | 0.00 | 4.42 / – | 2.18 | – |
| laya-PB-ASYNC-r2 | 0 | -2-0 | ane-dispatch | 0.7 | 0.00 | 1.87 / – | 2.17 | – |
| laya-PB-ASYNC-r2 | 0 | -2-0 | client-short | 5.9 | 0.76 | 1.70 / 1.12 | 1.28 | – |
| laya-PB-ASYNC-r2 | 0 | -2-0 | client-long | 6.7 | 0.66 | 1.82 / 1.12 | 1.68 | – |
| laya-PB-ASYNC-r2 | 0 | -2-0 | gpu-dispatch | 0.2 | 0.00 | 1.82 / – | 1.47 | – |
| laya-PB-ASYNC-r2 | 0 | -2-0 | callback | 0.1 | 0.00 | 2.57 / – | 0.74 | – |
| laya-PB-ASYNC-r2 | 0 | -2-0 | gpu-worker | 18.3 | 0.05 | 2.31 / 1.02 | 5.66 | – |
| laya-PB-ASYNC-r2 | 0 | -2-0 | main | 19.4 | 0.95 | 2.57 / 1.22 | 1.87 | – |
| laya-PB-ASYNC-r2 | 0 | 0-0.5 | ane-dispatch | 29.7 | 0.37 | 2.71 / 2.07 | 2.43 | 0.676 |
| laya-PB-ASYNC-r2 | 0 | 0-0.5 | client-short | 13.6 | 0.38 | 2.81 / 2.06 | 3.38 | 0.308 |
| laya-PB-ASYNC-r2 | 0 | 0-0.5 | client-long | 12.0 | 0.46 | 2.93 / 1.88 | 4.08 | 0.857 |
| laya-PB-ASYNC-r2 | 0 | 0-0.5 | gpu-dispatch | 2.6 | 0.47 | 2.81 / 1.88 | 1.48 | – |
| laya-PB-ASYNC-r2 | 0 | 0-0.5 | callback | 6.0 | 0.37 | 1.90 / 1.99 | 1.02 | 0.137 |
| laya-PB-ASYNC-r2 | 0 | 0-0.5 | gpu-worker | 69.4 | 0.33 | 2.50 / 2.08 | 2.91 | 4.958 |
| laya-PB-ASYNC-r2 | 0 | 0-0.5 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r2 | 0 | 0.5-1 | ane-dispatch | 16.1 | 0.00 | 4.02 / – | 3.19 | 0.316 |
| laya-PB-ASYNC-r2 | 0 | 0.5-1 | client-short | 7.3 | 0.00 | 4.26 / – | 4.32 | 0.143 |
| laya-PB-ASYNC-r2 | 0 | 0.5-1 | client-long | 5.9 | 0.00 | 4.44 / – | 5.00 | 0.421 |
| laya-PB-ASYNC-r2 | 0 | 0.5-1 | gpu-dispatch | 1.2 | 0.00 | 3.89 / – | 2.19 | – |
| laya-PB-ASYNC-r2 | 0 | 0.5-1 | callback | 3.2 | 0.00 | 1.86 / – | 1.68 | 0.064 |
| laya-PB-ASYNC-r2 | 0 | 0.5-1 | gpu-worker | 40.9 | 0.00 | 3.19 / – | 3.30 | 2.921 |
| laya-PB-ASYNC-r2 | 0 | 0.5-1 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r2 | 0 | 1-2 | ane-dispatch | 31.5 | 0.00 | 4.04 / – | 3.24 | 0.312 |
| laya-PB-ASYNC-r2 | 0 | 1-2 | client-short | 14.2 | 0.00 | 4.23 / – | 4.37 | 0.141 |
| laya-PB-ASYNC-r2 | 0 | 1-2 | client-long | 11.6 | 0.00 | 4.43 / – | 4.98 | 0.431 |
| laya-PB-ASYNC-r2 | 0 | 1-2 | gpu-dispatch | 2.3 | 0.00 | 3.91 / – | 2.20 | – |
| laya-PB-ASYNC-r2 | 0 | 1-2 | callback | 7.4 | 0.00 | 1.90 / – | 1.56 | 0.073 |
| laya-PB-ASYNC-r2 | 0 | 1-2 | gpu-worker | 82.1 | 0.00 | 3.19 / – | 3.30 | 3.040 |
| laya-PB-ASYNC-r2 | 0 | 1-2 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r2 | 0 | 2-4 | ane-dispatch | 64.7 | 0.00 | 3.99 / – | 3.20 | 0.319 |
| laya-PB-ASYNC-r2 | 0 | 2-4 | client-short | 28.4 | 0.00 | 4.25 / – | 4.32 | 0.140 |
| laya-PB-ASYNC-r2 | 0 | 2-4 | client-long | 23.2 | 0.00 | 4.40 / – | 4.98 | 0.414 |
| laya-PB-ASYNC-r2 | 0 | 2-4 | gpu-dispatch | 4.6 | 0.00 | 3.93 / – | 2.21 | – |
| laya-PB-ASYNC-r2 | 0 | 2-4 | callback | 13.3 | 0.00 | 1.75 / – | 1.78 | 0.066 |
| laya-PB-ASYNC-r2 | 0 | 2-4 | gpu-worker | 164.8 | 0.00 | 3.19 / – | 3.27 | 2.943 |
| laya-PB-ASYNC-r2 | 0 | 2-4 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r2 | 0 | 4-5 | ane-dispatch | 31.1 | 0.00 | 4.04 / – | 3.29 | 0.308 |
| laya-PB-ASYNC-r2 | 0 | 4-5 | client-short | 13.9 | 0.00 | 4.25 / – | 4.40 | 0.138 |
| laya-PB-ASYNC-r2 | 0 | 4-5 | client-long | 11.5 | 0.00 | 4.43 / – | 5.03 | 0.410 |
| laya-PB-ASYNC-r2 | 0 | 4-5 | gpu-dispatch | 2.2 | 0.00 | 3.93 / – | 2.33 | – |
| laya-PB-ASYNC-r2 | 0 | 4-5 | callback | 6.9 | 0.00 | 1.84 / – | 1.69 | 0.068 |
| laya-PB-ASYNC-r2 | 0 | 4-5 | gpu-worker | 80.5 | 0.00 | 3.16 / – | 3.38 | 2.875 |
| laya-PB-ASYNC-r2 | 0 | 4-5 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r2 | 0 | steady 10-20 | ane-dispatch | 317.6 | 0.00 | 4.01 / – | 3.23 | 0.314 |
| laya-PB-ASYNC-r2 | 0 | steady 10-20 | client-short | 140.8 | 0.00 | 4.22 / – | 4.35 | 0.139 |
| laya-PB-ASYNC-r2 | 0 | steady 10-20 | client-long | 115.1 | 0.00 | 4.39 / – | 5.00 | 0.413 |
| laya-PB-ASYNC-r2 | 0 | steady 10-20 | gpu-dispatch | 23.1 | 0.00 | 3.90 / – | 2.22 | – |
| laya-PB-ASYNC-r2 | 0 | steady 10-20 | callback | 67.5 | 0.00 | 1.89 / – | 1.66 | 0.067 |
| laya-PB-ASYNC-r2 | 0 | steady 10-20 | gpu-worker | 817.5 | 0.00 | 3.19 / – | 3.29 | 2.930 |
| laya-PB-ASYNC-r2 | 0 | steady 10-20 | main | 0.3 | 0.00 | 4.45 / – | 1.96 | – |
| laya-PB-ASYNC-r2 | 1 | -2-0 | ane-dispatch | 1.3 | 0.69 | 2.15 / 1.02 | 1.71 | – |
| laya-PB-ASYNC-r2 | 1 | -2-0 | client-short | 6.0 | 0.84 | 1.48 / 1.09 | 1.28 | – |
| laya-PB-ASYNC-r2 | 1 | -2-0 | client-long | 7.2 | 0.89 | 1.36 / 1.07 | 1.37 | – |
| laya-PB-ASYNC-r2 | 1 | -2-0 | gpu-dispatch | 0.2 | 1.00 | – / 1.02 | 1.04 | – |
| laya-PB-ASYNC-r2 | 1 | -2-0 | callback | 0.1 | 0.31 | 2.32 / 2.35 | 0.68 | – |
| laya-PB-ASYNC-r2 | 1 | -2-0 | gpu-worker | 18.6 | 0.62 | 1.80 / 1.55 | 3.80 | – |
| laya-PB-ASYNC-r2 | 1 | -2-0 | main | 19.8 | 0.96 | 2.57 / 1.23 | 1.81 | – |
| laya-PB-ASYNC-r2 | 1 | 0-0.5 | ane-dispatch | 55.3 | 0.94 | 1.96 / 1.54 | 1.69 | 1.536 |
| laya-PB-ASYNC-r2 | 1 | 0-0.5 | client-short | 26.1 | 0.94 | 2.00 / 1.53 | 2.32 | 0.726 |
| laya-PB-ASYNC-r2 | 1 | 0-0.5 | client-long | 27.8 | 0.97 | 2.62 / 1.28 | 2.96 | 2.317 |
| laya-PB-ASYNC-r2 | 1 | 0-0.5 | gpu-dispatch | 5.3 | 0.96 | 2.62 / 1.28 | 1.19 | – |
| laya-PB-ASYNC-r2 | 1 | 0-0.5 | callback | 10.1 | 0.93 | 2.03 / 1.55 | 0.80 | 0.280 |
| laya-PB-ASYNC-r2 | 1 | 0-0.5 | gpu-worker | 139.8 | 0.85 | 1.96 / 1.60 | 2.30 | 11.647 |
| laya-PB-ASYNC-r2 | 1 | 0-0.5 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r2 | 1 | 0.5-1 | ane-dispatch | 65.9 | 1.00 | – / 1.45 | 1.70 | 1.690 |
| laya-PB-ASYNC-r2 | 1 | 0.5-1 | client-short | 31.5 | 1.00 | – / 1.44 | 2.30 | 0.808 |
| laya-PB-ASYNC-r2 | 1 | 0.5-1 | client-long | 31.1 | 1.00 | – / 1.18 | 3.06 | 2.595 |
| laya-PB-ASYNC-r2 | 1 | 0.5-1 | gpu-dispatch | 6.1 | 1.00 | – / 1.16 | 1.26 | – |
| laya-PB-ASYNC-r2 | 1 | 0.5-1 | callback | 12.2 | 1.00 | – / 1.40 | 0.87 | 0.313 |
| laya-PB-ASYNC-r2 | 1 | 0.5-1 | gpu-worker | 134.6 | 1.00 | – / 1.49 | 1.85 | 11.215 |
| laya-PB-ASYNC-r2 | 1 | 0.5-1 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r2 | 1 | 1-2 | ane-dispatch | 138.5 | 1.00 | – / 1.32 | 1.72 | 1.823 |
| laya-PB-ASYNC-r2 | 1 | 1-2 | client-short | 65.7 | 1.00 | – / 1.33 | 2.31 | 0.864 |
| laya-PB-ASYNC-r2 | 1 | 1-2 | client-long | 62.6 | 1.00 | – / 1.15 | 3.08 | 2.720 |
| laya-PB-ASYNC-r2 | 1 | 1-2 | gpu-dispatch | 12.2 | 1.00 | – / 1.15 | 1.29 | – |
| laya-PB-ASYNC-r2 | 1 | 1-2 | callback | 24.3 | 1.00 | – / 1.30 | 0.89 | 0.320 |
| laya-PB-ASYNC-r2 | 1 | 1-2 | gpu-worker | 287.6 | 1.00 | – / 1.36 | 1.87 | 12.504 |
| laya-PB-ASYNC-r2 | 1 | 1-2 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r2 | 1 | 2-4 | ane-dispatch | 266.0 | 0.97 | 1.69 / 1.40 | 1.72 | 1.727 |
| laya-PB-ASYNC-r2 | 1 | 2-4 | client-short | 124.2 | 0.97 | 1.70 / 1.39 | 2.31 | 0.807 |
| laya-PB-ASYNC-r2 | 1 | 2-4 | client-long | 118.6 | 0.98 | 1.64 / 1.18 | 3.10 | 2.470 |
| laya-PB-ASYNC-r2 | 1 | 2-4 | gpu-dispatch | 23.5 | 0.98 | 1.63 / 1.17 | 1.27 | – |
| laya-PB-ASYNC-r2 | 1 | 2-4 | callback | 47.0 | 0.97 | 1.63 / 1.37 | 0.84 | 0.305 |
| laya-PB-ASYNC-r2 | 1 | 2-4 | gpu-worker | 548.8 | 0.98 | 1.64 / 1.44 | 1.86 | 11.433 |
| laya-PB-ASYNC-r2 | 1 | 2-4 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r2 | 1 | 4-5 | ane-dispatch | 141.8 | 1.00 | – / 1.29 | 1.69 | 1.916 |
| laya-PB-ASYNC-r2 | 1 | 4-5 | client-short | 68.0 | 1.00 | – / 1.27 | 2.27 | 0.907 |
| laya-PB-ASYNC-r2 | 1 | 4-5 | client-long | 65.1 | 1.00 | – / 1.10 | 3.01 | 2.712 |
| laya-PB-ASYNC-r2 | 1 | 4-5 | gpu-dispatch | 13.2 | 1.00 | – / 1.11 | 1.24 | – |
| laya-PB-ASYNC-r2 | 1 | 4-5 | callback | 25.1 | 1.00 | – / 1.26 | 0.81 | 0.340 |
| laya-PB-ASYNC-r2 | 1 | 4-5 | gpu-worker | 289.7 | 1.00 | – / 1.36 | 1.86 | 12.073 |
| laya-PB-ASYNC-r2 | 1 | 4-5 | main | 0.0 | – | – / – | – | – |
| laya-PB-ASYNC-r2 | 1 | steady 10-20 | ane-dispatch | 811.9 | 0.78 | 3.81 / 1.36 | 2.38 | 0.909 |
| laya-PB-ASYNC-r2 | 1 | steady 10-20 | client-short | 375.3 | 0.79 | 4.06 / 1.36 | 3.20 | 0.420 |
| laya-PB-ASYNC-r2 | 1 | steady 10-20 | client-long | 354.9 | 0.82 | 4.16 / 1.16 | 3.92 | 1.370 |
| laya-PB-ASYNC-r2 | 1 | steady 10-20 | gpu-dispatch | 70.7 | 0.82 | 3.76 / 1.15 | 1.67 | – |
| laya-PB-ASYNC-r2 | 1 | steady 10-20 | callback | 148.2 | 0.74 | 1.79 / 1.33 | 1.12 | 0.166 |
| laya-PB-ASYNC-r2 | 1 | steady 10-20 | gpu-worker | 1758.0 | 0.75 | 3.08 / 1.43 | 2.47 | 6.788 |
| laya-PB-ASYNC-r2 | 1 | steady 10-20 | main | 0.1 | 0.00 | 4.41 / – | 2.08 | – |

## Validity guard (A vs its historical phenotype, #92 / #93)

Concern if any A hetero window has: short P99 >= 13.0 ms, transient > 1.0 s, steady host-slow share >= 10%, aggregate req/s outside [109.9, 135.0], or a mismatch / failed routing.

| run | cycle | short P99 ms | transient s | steady host-slow | aggregate req/s | mismatches | routing failures | flags |
|---|---|---|---|---|---|---|---|---|
| laya-A-r1 | 0 | 11.00 | 0.5 | 0.000 | 122.5 | 0 | – | – |
| laya-A-r1 | 1 | 10.80 | 0.5 | 0.000 | 122.6 | 0 | – | – |
| laya-A-r2 | 0 | 10.89 | 0.5 | 0.001 | 122.5 | 0 | – | – |
| laya-A-r2 | 1 | 11.51 | 0.5 | 0.000 | 122.1 | 0 | – | – |

Validity concern: **False**


## Readings

- C: supports a perf-level placement / performance-state hypothesis
