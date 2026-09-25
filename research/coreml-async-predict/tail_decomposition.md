# PB-ASYNC short-tail decomposition (post-hoc, #92 raw data, not a gate)

Mean ms per short request in hetero windows (P50 in parentheses). Stages are consecutive; `client` is the latency bench_concurrency measured around `laya.predict`.

| stage | A (all windows) | PB-ASYNC normal window | PB-ASYNC tail windows | PB-ASYNC slowest 10% | PB-ASYNC slowest 5% | PB-ASYNC slowest 1% | PB-SYNC (all windows, reference) |
|---|---:|---:|---:|---:|---:|---:|---:|
| requests | 5874 | 2000 | 3887 | 589 | 294 | 59 | 4915 |
| client | 10.213 (10.040) | 10.003 (9.868) | 10.290 (9.875) | 13.049 (12.407) | 14.341 (13.027) | 18.893 (16.968) | 12.203 (12.136) |
| traced | 10.166 (10.017) | 9.977 (9.849) | 10.250 (9.856) | 12.883 (12.284) | 14.103 (12.841) | 18.454 (16.852) | 12.090 (12.014) |
| outside traced span | 0.046 (0.020) | 0.026 (0.020) | 0.040 (0.020) | 0.166 (0.110) | 0.238 (0.123) | 0.438 (0.120) | 0.113 (0.109) |
| pre-service (submit → service_start) | 0.147 (0.135) | 0.163 (0.135) | 0.223 (0.136) | 0.777 (0.787) | 0.942 (0.890) | 1.160 (1.072) | 0.730 (0.785) |
| prepare | 0.124 (0.117) | 0.139 (0.117) | 0.192 (0.117) | 0.661 (0.679) | 0.801 (0.769) | 0.950 (0.920) | 0.620 (0.676) |
| route | 0.006 (0.006) | 0.008 (0.006) | 0.012 (0.006) | 0.046 (0.047) | 0.054 (0.052) | 0.064 (0.060) | 0.043 (0.047) |
| check + enqueue | 0.003 (0.003) | 0.004 (0.003) | 0.005 (0.003) | 0.018 (0.018) | 0.022 (0.020) | 0.027 (0.024) | 0.017 (0.018) |
| queue / dispatcher wake | 0.012 (0.008) | 0.011 (0.008) | 0.013 (0.008) | 0.045 (0.037) | 0.056 (0.043) | 0.107 (0.048) | 0.044 (0.037) |
| dispatch → service_start | 0.001 (0.001) | 0.002 (0.001) | 0.002 (0.001) | 0.007 (0.006) | 0.008 (0.007) | 0.012 (0.008) | 0.006 (0.006) |
| service | 9.989 (9.853) | 9.781 (9.685) | 9.981 (9.691) | 11.944 (11.293) | 12.974 (11.699) | 17.120 (15.595) | 11.209 (11.032) |
| features | 0.262 (0.157) | 0.198 (0.161) | 0.297 (0.157) | 1.148 (0.786) | 1.658 (0.979) | 3.920 (4.394) | 0.935 (0.777) |
| predict (whole) | 9.661 (9.617) | – | – | – | – | – | – |
| pre (inputs, pool) | – | 0.020 (0.016) | 0.028 (0.015) | 0.105 (0.092) | 0.128 (0.110) | 0.156 (0.147) | 0.100 (0.089) |
| to native | – | – | – | – | – | – | 0.008 (0.008) |
| native predict | – | – | – | – | – | – | 9.758 (9.715) |
| GIL re-acquire | – | – | – | – | – | – | 0.053 (0.006) |
| post (pool, outputs) | – | 0.008 (0.005) | 0.011 (0.006) | 0.035 (0.035) | 0.040 (0.039) | 0.042 (0.040) | 0.035 (0.037) |
| async send | – | 0.011 (0.009) | 0.020 (0.009) | 0.088 (0.065) | 0.115 (0.073) | 0.115 (0.113) | – |
| Core ML + callback GIL | – | 9.445 (9.409) | 9.498 (9.417) | 10.135 (9.846) | 10.516 (9.932) | 12.352 (10.358) | – |
| callback → waiter wake | – | 0.025 (0.022) | 0.026 (0.022) | 0.074 (0.061) | 0.097 (0.068) | 0.132 (0.068) | – |
| action head (exit → service_end) | 0.065 (0.060) | 0.073 (0.060) | 0.102 (0.061) | 0.360 (0.362) | 0.420 (0.407) | 0.402 (0.390) | 0.321 (0.346) |
| post-service (service_end → response) | 0.031 (0.029) | 0.033 (0.028) | 0.046 (0.028) | 0.162 (0.163) | 0.188 (0.181) | 0.174 (0.169) | 0.151 (0.163) |
| service_end → finish (future resolved) | 0.003 (0.003) | 0.003 (0.003) | 0.004 (0.003) | 0.014 (0.014) | 0.016 (0.016) | 0.016 (0.015) | 0.013 (0.014) |
| format answers | 0.028 (0.026) | 0.030 (0.025) | 0.042 (0.026) | 0.148 (0.149) | 0.172 (0.164) | 0.158 (0.154) | 0.137 (0.148) |
| GPU completions during service | 0.193 (0.000) | 0.275 (0.000) | 0.276 (0.000) | 0.328 (0.000) | 0.388 (0.000) | 0.864 (1.000) | 0.289 (0.000) |

## Client latency percentiles (ms)

| group | P50 | P90 | P95 | P99 | P99.9 |
|---|---:|---:|---:|---:|---:|
| A (all windows) | 10.04 | 10.51 | 10.57 | 11.93 | 16.47 |
| PB-ASYNC normal window | 9.87 | 9.96 | 10.34 | 12.77 | 17.83 |
| PB-ASYNC tail windows | 9.88 | 11.98 | 12.58 | 16.69 | 17.35 |

## Excess of PB-ASYNC's slowest 1% over its fastest 90% (mean ms per stage)

| stage | excess ms |
|---|---:|
| client | +9.018 |
| traced | +8.600 |
| outside traced span | +0.418 |
| pre-service (submit → service_start) | +1.022 |
| prepare | +0.830 |
| route | +0.057 |
| check + enqueue | +0.024 |
| queue / dispatcher wake | +0.099 |
| dispatch → service_start | +0.011 |
| service | +7.432 |
| features | +3.754 |
| pre (inputs, pool) | +0.140 |
| post (pool, outputs) | +0.036 |
| async send | +0.106 |
| Core ML + callback GIL | +2.944 |
| callback → waiter wake | +0.112 |
| action head (exit → service_end) | +0.340 |
| post-service (service_end → response) | +0.146 |
| service_end → finish (future resolved) | +0.013 |
| format answers | +0.133 |
| GPU completions during service | +0.595 |

## Host-slow share over each hetero window (client-thread prepare > 0.3 ms)

| cell | window | window short P99 | host-slow share | per 2 s of the window | client P99 host-slow / otherwise |
|---|---|---:|---:|---|---|
| A | 0 | 12.91 | 1.5% | 16% 0% 0% 0% 0% 0% 0% 0% 0% 0% | 49.39 / 10.75 |
| A | 1 | 11.13 | 0.8% | 8% 0% 0% 0% 0% 0% 0% 0% 0% 0% | 49.08 / 10.65 |
| A | 2 | 11.01 | 0.7% | 7% 0% 0% 0% 0% 0% 0% 0% 0% 0% | 50.01 / 10.70 |
| PB-SYNC | 0 | 16.87 | 99.9% | 99% 100% 100% 100% 100% 100% 100% 100% 100% 100% | 16.87 / 11.05 |
| PB-SYNC | 1 | 16.67 | 52.1% | 100% 100% 99% 100% 100% 80% 0% 0% 0% 0% | 16.79 / 10.34 |
| PB-SYNC | 2 | 16.77 | 99.5% | 99% 99% 100% 100% 100% 98% 100% 100% 100% 99% | 16.77 / 12.10 |
| PB-ASYNC | 0 | 16.96 | 15.6% | 99% 99% 0% 0% 0% 0% 0% 0% 0% 0% | 17.24 / 10.42 |
| PB-ASYNC | 1 | 12.77 | 3.4% | 38% 0% 0% 0% 0% 0% 0% 0% 0% 0% | 31.93 / 10.44 |
| PB-ASYNC | 2 | 16.26 | 9.0% | 99% 15% 0% 0% 0% 0% 0% 0% 0% 0% | 17.29 / 10.38 |

## CPU time against wall time (ANE dispatcher thread), by client-latency group

| cell | group | ANE thread CPU / forward | wall of CPU-active stages | client-thread prepare |
|---|---|---:|---:|---:|
| A | fastest 90% | 0.561 | 0.289 | 0.118 |
| A | slowest 10% | 0.870 | 0.669 | 0.173 |
| A | slowest 1% | 3.097 | 1.472 | 0.600 |
| A | P(next of slowest 10% \| slowest 10%) | 0.12 | | |
| PB-SYNC | fastest 90% | 1.458 | 1.123 | 0.589 |
| PB-SYNC | slowest 10% | 2.238 | 3.876 | 0.898 |
| PB-SYNC | slowest 1% | 3.394 | 4.740 | 0.982 |
| PB-SYNC | P(next of slowest 10% \| slowest 10%) | 0.02 | | |
| PB-ASYNC | fastest 90% | 0.265 | 0.260 | 0.120 |
| PB-ASYNC | slowest 10% | 1.348 | 1.736 | 0.662 |
| PB-ASYNC | slowest 1% | 1.877 | 4.651 | 0.950 |
| PB-ASYNC | P(next of slowest 10% \| slowest 10%) | 0.94 | | |

Callback threads seen (native ids): 1
