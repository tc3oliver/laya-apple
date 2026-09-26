# H64 confirmation: A vs H64 at the hetero transition (laya, L128 / L512)

Outcome: FAILED: latency confirmation (outlier guard: laya-H64-r8 c1); the H64 route closes, production stays A, and no other N is tried

Crash logs (`raw-conf/failed/`): none

## Runs

| run | protocol ok | mismatches | routing failures | crashed | episodes | forwards sync / async | wall s |
|---|---|---|---|---|---|---|---|
| laya-A-r1 | True | 0 | – | False | 0 | 7844 / 0 | 189 |
| laya-A-r2 | True | 0 | – | False | 0 | 7846 / 0 | 189 |
| laya-A-r3 | True | 0 | – | False | 0 | 7846 / 0 | 189 |
| laya-A-r4 | True | 0 | – | False | 0 | 7849 / 0 | 189 |
| laya-A-r5 | True | 0 | – | False | 0 | 7857 / 0 | 189 |
| laya-A-r6 | True | 0 | – | False | 0 | 7845 / 0 | 189 |
| laya-A-r7 | True | 0 | – | False | 0 | 7850 / 0 | 189 |
| laya-A-r8 | True | 0 | – | False | 0 | 7851 / 0 | 189 |
| laya-H64-r1 | True | 0 | – | False | 2 | 4063 / 3900 | 189 |
| laya-H64-r2 | True | 0 | – | False | 2 | 4066 / 3894 | 189 |
| laya-H64-r3 | True | 0 | – | False | 2 | 4064 / 3901 | 189 |
| laya-H64-r4 | True | 0 | – | False | 2 | 4057 / 3894 | 189 |
| laya-H64-r5 | True | 0 | – | False | 2 | 4059 / 3889 | 189 |
| laya-H64-r6 | True | 0 | – | False | 2 | 4070 / 3895 | 189 |
| laya-H64-r7 | True | 0 | – | False | 2 | 4059 / 3889 | 189 |
| laya-H64-r8 | True | 0 | – | False | 2 | 4059 / 3879 | 189 |

## A validity

| run | valid | failed checks |
|---|---|---|
| laya-A-r1 | True | – |
| laya-A-r2 | True | – |
| laya-A-r3 | True | – |
| laya-A-r4 | True | – |
| laya-A-r5 | True | – |
| laya-A-r6 | True | – |
| laya-A-r7 | True | – |
| laya-A-r8 | True | – |

## Hard safety gates (every H64 transition)

| H64 transition | A | after | mismatches | routing | crash | structure | sync_before_t_h | e_residency | transient_from_th | steady_host_slow | gpu_return_p50 | gpu_return_p99 | throughput | native_ratio | native_plus | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laya-H64-r1 c0 | laya-A-r1 c0 | solo_long | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.036 <= 1.000 pass | 0.103 <= 1.000 pass | 128.34 >= 116.13 pass | 9.39 <= 10.16 pass | 9.39 <= 9.98 pass | – |
| laya-H64-r1 c1 | laya-A-r1 c1 | gpu_only | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.037 <= 1.000 pass | 0.196 <= 1.000 pass | 128.46 >= 116.11 pass | 9.41 <= 10.16 pass | 9.41 <= 9.98 pass | – |
| laya-H64-r2 c0 | laya-A-r2 c0 | solo_long | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.036 <= 1.000 pass | 0.169 <= 1.000 pass | 128.19 >= 116.36 pass | 9.40 <= 10.14 pass | 9.40 <= 9.96 pass | – |
| laya-H64-r2 c1 | laya-A-r2 c1 | gpu_only | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.036 <= 1.000 pass | 0.212 <= 1.000 pass | 128.27 >= 115.92 pass | 9.39 <= 10.18 pass | 9.39 <= 9.99 pass | – |
| laya-H64-r3 c0 | laya-A-r3 c0 | solo_long | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.035 <= 1.000 pass | 0.113 <= 1.000 pass | 128.24 >= 116.22 pass | 9.40 <= 10.15 pass | 9.40 <= 9.96 pass | – |
| laya-H64-r3 c1 | laya-A-r3 c1 | gpu_only | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.034 <= 1.000 pass | 0.145 <= 1.000 pass | 128.63 >= 116.05 pass | 9.39 <= 10.17 pass | 9.39 <= 9.98 pass | – |
| laya-H64-r4 c0 | laya-A-r4 c0 | solo_long | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.036 <= 1.000 pass | 0.161 <= 1.000 pass | 128.30 >= 116.30 pass | 9.39 <= 10.14 pass | 9.39 <= 9.96 pass | – |
| laya-H64-r4 c1 | laya-A-r4 c1 | gpu_only | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.036 <= 1.000 pass | 0.233 <= 1.000 pass | 128.13 >= 116.26 pass | 9.40 <= 10.15 pass | 9.40 <= 9.96 pass | – |
| laya-H64-r5 c0 | laya-A-r5 c0 | solo_long | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.042 <= 1.000 pass | 0.171 <= 1.000 pass | 128.24 >= 115.82 pass | 9.39 <= 10.17 pass | 9.39 <= 9.98 pass | – |
| laya-H64-r5 c1 | laya-A-r5 c1 | gpu_only | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.042 <= 1.000 pass | 0.282 <= 1.000 pass | 128.06 >= 116.11 pass | 9.37 <= 10.15 pass | 9.37 <= 9.97 pass | – |
| laya-H64-r6 c0 | laya-A-r6 c0 | solo_long | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.038 <= 1.000 pass | 0.196 <= 1.000 pass | 128.25 >= 116.31 pass | 9.40 <= 10.14 pass | 9.40 <= 9.96 pass | – |
| laya-H64-r6 c1 | laya-A-r6 c1 | gpu_only | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.038 <= 1.000 pass | 0.227 <= 1.000 pass | 128.30 >= 116.08 pass | 9.39 <= 10.16 pass | 9.39 <= 9.98 pass | – |
| laya-H64-r7 c0 | laya-A-r7 c0 | solo_long | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.001 < 0.100 pass | 0.039 <= 1.000 pass | 0.159 <= 1.000 pass | 127.87 >= 116.11 pass | 9.41 <= 10.16 pass | 9.41 <= 9.98 pass | – |
| laya-H64-r7 c1 | laya-A-r7 c1 | gpu_only | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.037 <= 1.000 pass | 0.176 <= 1.000 pass | 128.35 >= 116.19 pass | 9.39 <= 10.15 pass | 9.39 <= 9.96 pass | – |
| laya-H64-r8 c0 | laya-A-r8 c0 | solo_long | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.037 <= 1.000 pass | 0.255 <= 1.000 pass | 128.33 >= 116.01 pass | 9.38 <= 10.16 pass | 9.38 <= 9.98 pass | – |
| laya-H64-r8 c1 | laya-A-r8 c1 | gpu_only | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | 64 == 64 pass | 0 <= 1 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.039 <= 1.000 pass | 0.213 <= 1.000 pass | 127.35 >= 116.33 pass | 9.40 <= 10.15 pass | 9.40 <= 9.96 pass | – |

## Post-handoff E residency

0.5 s bins from t_h; a bin is E-resident if parent-active or the worker has >= 2 ms CPU and E share >= 0.5.

| H64 transition | bins | E-resident bins (s from t_h) | longest consecutive | long |
|---|---|---|---|---|
| laya-H64-r1 c0 | 39 | none | 0 | False |
| laya-H64-r1 c1 | 39 | none | 0 | False |
| laya-H64-r2 c0 | 39 | none | 0 | False |
| laya-H64-r2 c1 | 39 | none | 0 | False |
| laya-H64-r3 c0 | 39 | none | 0 | False |
| laya-H64-r3 c1 | 39 | none | 0 | False |
| laya-H64-r4 c0 | 39 | none | 0 | False |
| laya-H64-r4 c1 | 39 | none | 0 | False |
| laya-H64-r5 c0 | 39 | none | 0 | False |
| laya-H64-r5 c1 | 39 | none | 0 | False |
| laya-H64-r6 c0 | 39 | none | 0 | False |
| laya-H64-r6 c1 | 39 | none | 0 | False |
| laya-H64-r7 c0 | 39 | none | 0 | False |
| laya-H64-r7 c1 | 39 | none | 0 | False |
| laya-H64-r8 c0 | 39 | none | 0 | False |
| laya-H64-r8 c1 | 39 | none | 0 | False |

## User-visible latency (H64 − paired A)

delta_p99 is the gate; the other deltas are reported only.

| pair | cycle | H64 window P99 | A window P99 | delta P99 | delta onset P99 | delta P95 | delta median | delta req/s |
|---|---|---|---|---|---|---|---|---|
| laya-H64-r1 ↔ laya-A-r1 | 0 | 10.65 | 11.06 | -0.408 | 0.163 | -0.569 | -0.191 | 6.1 |
| laya-H64-r1 ↔ laya-A-r1 | 1 | 10.57 | 11.40 | -0.828 | -4.171 | -0.517 | -0.169 | 6.2 |
| laya-H64-r2 ↔ laya-A-r2 | 0 | 10.81 | 11.00 | -0.188 | 0.507 | -0.526 | -0.162 | 5.7 |
| laya-H64-r2 ↔ laya-A-r2 | 1 | 10.79 | 12.06 | -1.273 | -0.817 | -0.612 | -0.193 | 6.3 |
| laya-H64-r3 ↔ laya-A-r3 | 0 | 10.77 | 12.08 | -1.310 | -0.292 | -0.508 | -0.168 | 5.9 |
| laya-H64-r3 ↔ laya-A-r3 | 1 | 10.60 | 11.64 | -1.047 | -2.631 | -0.587 | -0.199 | 6.5 |
| laya-H64-r4 ↔ laya-A-r4 | 0 | 10.75 | 10.95 | -0.200 | 0.679 | -0.548 | -0.179 | 5.9 |
| laya-H64-r4 ↔ laya-A-r4 | 1 | 10.81 | 10.90 | -0.090 | 0.117 | -0.527 | -0.162 | 5.7 |
| laya-H64-r5 ↔ laya-A-r5 | 0 | 10.75 | 11.08 | -0.330 | -0.026 | -0.593 | -0.196 | 6.3 |
| laya-H64-r5 ↔ laya-A-r5 | 1 | 12.71 | 10.95 | 1.751 | 1.819 | -0.546 | -0.202 | 5.8 |
| laya-H64-r6 ↔ laya-A-r6 | 0 | 10.74 | 11.05 | -0.304 | -0.180 | -0.542 | -0.167 | 5.8 |
| laya-H64-r6 ↔ laya-A-r6 | 1 | 10.77 | 11.01 | -0.244 | -0.573 | -0.577 | -0.194 | 6.1 |
| laya-H64-r7 ↔ laya-A-r7 | 0 | 12.12 | 11.18 | 0.938 | -0.303 | -0.236 | -0.181 | 5.6 |
| laya-H64-r7 ↔ laya-A-r7 | 1 | 10.79 | 12.18 | -1.397 | 0.114 | -0.545 | -0.179 | 6.0 |
| laya-H64-r8 ↔ laya-A-r8 | 0 | 12.18 | 11.21 | 0.970 | 0.417 | -0.596 | -0.206 | 6.2 |
| laya-H64-r8 ↔ laya-A-r8 | 1 | 13.78 | 11.35 | 2.433 | 3.235 | -0.342 | -0.165 | 4.9 |

- Median of the 16 deltas: -0.274 <= 0.500 pass
- Cluster (run-pair) bootstrap one-sided 95% upper bound (B = 20000, seed 20260926): 0.304 <= 1.000 pass
- Transition-level bootstrap upper bound (report only): -0.139
- Worst delta (outlier guard): 2.43 <= 2.00 FAIL
- Population gate: **True**; outlier guard: **False**

## GPU return, throughput, correctness

| cell | GPU return P50 | P95 | P99 | aggregate req/s |
|---|---|---|---|---|
| H64 (from t_h + 1 s) | 0.034–0.042 | 0.058–0.097 | 0.103–0.282 | 127.3–128.6 |
| A (whole window, paired) | 4.306–4.476 | 4.627–5.407 | 5.414–6.293 | 121.9–122.5 |

Correctness (H64): mismatches 0, routing failures 0, crashed runs 0, structurally valid transitions 16 of 16
