# Staged handoff screen: A, B, H32, H64 at the hetero transition (laya, L128 / L512)

Outcome: round 2 (H32) pending: H32 r3, A r3, H32 r4, A r4, H32 r5

Crashed runs and re-runs (`raw/failed/`): none


## Runs

| run | seconds | protocol ok | mismatches | routing failures | crashed | episodes | forwards sync / async | wall s |
|---|---|---|---|---|---|---|---|---|
| laya-A-r1 | 20 | True | 0 | – | False | 0 | 7839 / 0 | 189 |
| laya-A-r2 | 20 | True | 0 | – | False | 0 | 7846 / 0 | 189 |
| laya-B-r1 | 20 | True | 0 | – | False | 0 | 0 / 7085 | 189 |
| laya-B-r2 | 20 | True | 0 | – | False | 0 | 0 / 7088 | 189 |
| laya-H32-r1 | 20 | True | 0 | – | False | 2 | 3988 / 3956 | 188 |
| laya-H32-r2 | 20 | True | 0 | – | False | 2 | 4002 / 3961 | 189 |
| laya-H64-r1 | 20 | True | 0 | – | False | 2 | 4074 / 3878 | 189 |
| laya-H64-r2 | 20 | True | 0 | – | False | 2 | 4067 / 3898 | 189 |

## Transitions

t_h in s from t0. Transient: #94's, from t0 and from t_h. GPU return over [t_h + 1 s, end) (A: whole window).

| run | cycle | after | t_h s | structure (episodes / sync before t_h / valid) | onset P99 | window P99 | native mean (n) | predict mean | transient t0 / t_h s | steady host-slow | GPU return P50 / P95 / P99 | agg req/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laya-A-r1 | 0 | solo_long | – | – / – / – | 16.05 | 12.72 | – (0) | 9.668 | 0.5 / – | 0.000 | 4.386 / 5.085 / 5.799 | 122.1 |
| laya-A-r1 | 1 | gpu_only | – | – / – / – | 14.77 | 11.49 | – (0) | 9.668 | 0.5 / – | 0.000 | 4.374 / 5.148 / 6.246 | 122.1 |
| laya-A-r2 | 0 | solo_long | – | – / – / – | 14.50 | 11.12 | – (0) | 9.658 | 0.5 / – | 0.000 | 4.330 / 4.836 / 6.245 | 122.4 |
| laya-A-r2 | 1 | gpu_only | – | – / – / – | 15.14 | 11.00 | – (0) | 9.659 | 0.5 / – | 0.000 | 4.326 / 4.852 / 5.593 | 122.4 |
| laya-B-r1 | 0 | solo_long | 0.000 | – / – / – | 17.46 | 17.44 | 9.871 (1532) | 10.321 | 20.0 / 20.0 | 0.999 | 0.248 / 1.465 / 2.017 | 100.3 |
| laya-B-r1 | 1 | gpu_only | 0.000 | – / – / – | 17.38 | 17.35 | 9.878 (1535) | 10.300 | 20.0 / 20.0 | 1.000 | 0.246 / 1.541 / 2.158 | 100.4 |
| laya-B-r2 | 0 | solo_long | 0.000 | – / – / – | 17.42 | 17.37 | 9.862 (1534) | 10.304 | 20.0 / 20.0 | 0.995 | 0.246 / 1.403 / 2.034 | 100.4 |
| laya-B-r2 | 1 | gpu_only | 0.000 | – / – / – | 17.16 | 17.37 | 9.894 (1529) | 10.331 | 20.0 / 20.0 | 1.000 | 0.247 / 1.263 / 2.083 | 100.1 |
| laya-H32-r1 | 0 | solo_long | 0.426 | 1 / 32 / True | 14.40 | 10.57 | 9.404 (1978) | 9.508 | 0.5 / 0.0 | 0.000 | 0.040 / 0.071 / 0.171 | 128.3 |
| laya-H32-r1 | 1 | gpu_only | 0.440 | 1 / 32 / True | 14.45 | 10.68 | 9.406 (1977) | 9.509 | 0.5 / 0.0 | 0.000 | 0.040 / 0.076 / 0.222 | 128.2 |
| laya-H32-r2 | 0 | solo_long | 0.441 | 1 / 32 / True | 14.83 | 10.56 | 9.390 (1982) | 9.494 | 0.5 / 0.0 | 0.000 | 0.038 / 0.063 / 0.190 | 128.5 |
| laya-H32-r2 | 1 | gpu_only | 0.441 | 1 / 32 / True | 14.38 | 10.74 | 9.405 (1979) | 9.506 | 0.5 / 0.0 | 0.000 | 0.038 / 0.064 / 0.147 | 128.3 |
| laya-H64-r1 | 0 | solo_long | 0.762 | 1 / 64 / True | 14.94 | 11.00 | 9.416 (1935) | 9.531 | 0.5 / 0.0 | 0.000 | 0.045 / 0.097 / 0.276 | 127.6 |
| laya-H64-r1 | 1 | gpu_only | 0.758 | 1 / 64 / True | 14.49 | 10.74 | 9.396 (1943) | 9.507 | 0.5 / 0.0 | 0.001 | 0.038 / 0.083 / 0.179 | 128.0 |
| laya-H64-r2 | 0 | solo_long | 0.763 | 1 / 64 / True | 14.58 | 10.69 | 9.388 (1949) | 9.492 | 0.5 / 0.0 | 0.000 | 0.039 / 0.067 / 0.163 | 128.4 |
| laya-H64-r2 | 1 | gpu_only | 0.767 | 1 / 64 / True | 14.89 | 11.10 | 9.389 (1949) | 9.493 | 0.5 / 0.0 | 0.000 | 0.038 / 0.071 / 0.231 | 128.3 |

## Mechanism (report only, never a gate)

E share (relative cycle rate P / E) of parent-active and the auto GPU worker; spans from t_h (A: t0). Switch: E->P switch time from t0 (placement_posthoc's rule).

| run | cycle | group | 0-0.5 | 0.5-4 | 4-end | switch s | P-dominant and stays |
|---|---|---|---|---|---|---|---|
| laya-A-r1 | 0 | parent | 0.68 (2.15 / 1.53) | 0.00 (4.14 / –) | 0.00 (4.13 / –) | 0.5 | False |
| laya-A-r1 | 0 | worker | 0.73 (2.15 / 1.73) | 0.00 (3.14 / –) | 0.00 (3.17 / –) | 0.5 | False |
| laya-A-r1 | 1 | parent | 0.54 (2.42 / 1.61) | 0.00 (4.08 / –) | 0.00 (4.11 / –) | 0.5 | False |
| laya-A-r1 | 1 | worker | 0.60 (2.29 / 1.74) | 0.00 (3.26 / –) | 0.00 (3.37 / 2.60) | 0.5 | False |
| laya-A-r2 | 0 | parent | 0.49 (2.58 / 1.68) | 0.00 (4.13 / –) | 0.00 (4.13 / –) | 0.0 | False |
| laya-A-r2 | 0 | worker | 0.60 (2.41 / 1.88) | 0.00 (3.16 / –) | 0.00 (3.38 / –) | 0.5 | False |
| laya-A-r2 | 1 | parent | 0.68 (3.30 / 1.74) | 0.00 (4.13 / –) | 0.00 (4.12 / –) | 0.5 | False |
| laya-A-r2 | 1 | worker | 0.70 (2.75 / 1.82) | 0.00 (3.13 / –) | 0.00 (3.37 / –) | 0.5 | False |
| laya-B-r1 | 0 | parent | 0.93 (1.99 / 1.44) | 1.00 (– / 1.30) | 1.00 (1.54 / 1.33) | never | False |
| laya-B-r1 | 0 | worker | 0.84 (1.90 / 1.59) | 1.00 (– / 1.42) | 1.00 (1.50 / 1.45) | never | False |
| laya-B-r1 | 1 | parent | 0.92 (1.98 / 1.55) | 0.99 (1.57 / 1.34) | 1.00 (1.74 / 1.32) | never | False |
| laya-B-r1 | 1 | worker | 0.82 (1.91 / 1.73) | 0.99 (1.55 / 1.46) | 1.00 (1.68 / 1.43) | never | False |
| laya-B-r2 | 0 | parent | 0.91 (1.91 / 1.66) | 1.00 (– / 1.26) | 0.99 (1.65 / 1.33) | never | False |
| laya-B-r2 | 0 | worker | 0.82 (2.03 / 1.69) | 1.00 (– / 1.40) | 0.99 (1.68 / 1.45) | never | False |
| laya-B-r2 | 1 | parent | 0.81 (1.76 / 1.66) | 1.00 (– / 1.35) | 1.00 (1.52 / 1.31) | never | False |
| laya-B-r2 | 1 | worker | 0.74 (1.88 / 1.76) | 1.00 (– / 1.48) | 1.00 (1.48 / 1.43) | never | False |
| laya-H32-r1 | 0 | parent | 0.00 (3.95 / –) | 0.00 (3.89 / –) | 0.00 (3.91 / 1.00) | 0.0 | True |
| laya-H32-r1 | 0 | worker | 0.00 (3.18 / –) | 0.00 (3.19 / –) | 0.00 (3.44 / –) | 0.0 | True |
| laya-H32-r1 | 1 | parent | 0.00 (3.90 / –) | 0.00 (3.90 / –) | 0.00 (3.91 / –) | 0.0 | True |
| laya-H32-r1 | 1 | worker | 0.00 (3.14 / –) | 0.00 (3.18 / –) | 0.00 (3.44 / –) | 0.5 | True |
| laya-H32-r2 | 0 | parent | 0.00 (3.97 / –) | 0.00 (3.91 / –) | 0.00 (3.90 / –) | 0.0 | True |
| laya-H32-r2 | 0 | worker | 0.00 (3.20 / –) | 0.00 (3.21 / –) | 0.00 (3.18 / –) | 0.5 | True |
| laya-H32-r2 | 1 | parent | 0.00 (3.94 / –) | 0.00 (3.90 / –) | 0.00 (3.90 / –) | 0.5 | True |
| laya-H32-r2 | 1 | worker | 0.00 (3.20 / –) | 0.00 (3.17 / –) | 0.00 (3.41 / –) | 0.5 | True |
| laya-H64-r1 | 0 | parent | 0.00 (3.92 / –) | 0.00 (3.91 / –) | 0.00 (3.82 / 2.15) | 0.5 | True |
| laya-H64-r1 | 0 | worker | 0.00 (3.22 / –) | 0.00 (3.20 / –) | 0.00 (3.26 / 2.53) | 0.5 | True |
| laya-H64-r1 | 1 | parent | 0.00 (3.90 / –) | 0.00 (3.90 / –) | 0.00 (3.80 / 2.53) | 0.0 | True |
| laya-H64-r1 | 1 | worker | 0.00 (3.13 / –) | 0.00 (3.16 / –) | 0.00 (3.16 / 2.49) | 0.5 | True |
| laya-H64-r2 | 0 | parent | 0.00 (3.92 / –) | 0.00 (3.89 / –) | 0.00 (3.89 / –) | 0.0 | True |
| laya-H64-r2 | 0 | worker | 0.00 (3.20 / –) | 0.00 (3.19 / –) | 0.00 (3.45 / –) | 0.5 | True |
| laya-H64-r2 | 1 | parent | 0.00 (3.93 / –) | 0.00 (3.89 / –) | 0.00 (3.90 / –) | 0.5 | True |
| laya-H64-r2 | 1 | worker | 0.00 (3.22 / –) | 0.00 (3.18 / –) | 0.00 (3.43 / –) | 0.5 | True |

## Round 1

A references (median over 4 A transitions): onset P99 14.95 ms, window P99 11.30 ms, predict mean 9.663 ms, aggregate 122.3 req/s

A validity: **True**

| run | cycle | short P99 ms | transient s | steady host-slow | aggregate req/s | failed |
|---|---|---|---|---|---|---|
| laya-A-r1 | 0 | 12.72 | 0.5 | 0.000 | 122.1 | – |
| laya-A-r1 | 1 | 11.49 | 0.5 | 0.000 | 122.1 | – |
| laya-A-r2 | 0 | 11.12 | 0.5 | 0.000 | 122.4 | – |
| laya-A-r2 | 1 | 11.00 | 0.5 | 0.000 | 122.4 | – |

B phenotype: **True** (4 of 4 transitions with a transient >= 2.0 s: 20.0, 20.0, 20.0, 20.0)


### H32

| transition | after | onset_p99 | window_p99 | native_ratio | native_plus | transient_from_th | steady_host_slow | gpu_return_p50 | throughput | mismatches | routing | crash | structure | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laya-H32-r1 c0 | solo_long | 14.40 <= 15.95 pass | 10.57 <= 12.30 pass | 9.40 <= 10.15 pass | 9.40 <= 9.96 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.040 <= 1.000 pass | 128.28 >= 116.16 pass | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | – |
| laya-H32-r1 c1 | gpu_only | 14.45 <= 15.95 pass | 10.68 <= 12.30 pass | 9.41 <= 10.15 pass | 9.41 <= 9.96 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.040 <= 1.000 pass | 128.24 >= 116.16 pass | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | – |
| laya-H32-r2 c0 | solo_long | 14.83 <= 15.95 pass | 10.56 <= 12.30 pass | 9.39 <= 10.15 pass | 9.39 <= 9.96 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.038 <= 1.000 pass | 128.47 >= 116.16 pass | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | – |
| laya-H32-r2 c1 | gpu_only | 14.38 <= 15.95 pass | 10.74 <= 12.30 pass | 9.40 <= 10.15 pass | 9.40 <= 9.96 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.038 <= 1.000 pass | 128.32 >= 116.16 pass | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | – |

Candidate passes: **True** (4 transitions)
- 'P-dominant at the handoff and stays' holds in every transition: True


### H64

| transition | after | onset_p99 | window_p99 | native_ratio | native_plus | transient_from_th | steady_host_slow | gpu_return_p50 | throughput | mismatches | routing | crash | structure | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laya-H64-r1 c0 | solo_long | 14.94 <= 15.95 pass | 11.00 <= 12.30 pass | 9.42 <= 10.15 pass | 9.42 <= 9.96 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.045 <= 1.000 pass | 127.59 >= 116.16 pass | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | – |
| laya-H64-r1 c1 | gpu_only | 14.49 <= 15.95 pass | 10.74 <= 12.30 pass | 9.40 <= 10.15 pass | 9.40 <= 9.96 pass | 0.000 <= 1.000 pass | 0.001 < 0.100 pass | 0.038 <= 1.000 pass | 128.00 >= 116.16 pass | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | – |
| laya-H64-r2 c0 | solo_long | 14.58 <= 15.95 pass | 10.69 <= 12.30 pass | 9.39 <= 10.15 pass | 9.39 <= 9.96 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.039 <= 1.000 pass | 128.39 >= 116.16 pass | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | – |
| laya-H64-r2 c1 | gpu_only | 14.89 <= 15.95 pass | 11.10 <= 12.30 pass | 9.39 <= 10.15 pass | 9.39 <= 9.96 pass | 0.000 <= 1.000 pass | 0.000 < 0.100 pass | 0.038 <= 1.000 pass | 128.27 >= 116.16 pass | 0 == 0 pass | 0 == 0 pass | False == False pass | True == True pass | – |

Candidate passes: **True** (4 transitions)
- 'P-dominant at the handoff and stays' holds in every transition: True

After round 1: leader H32; round 2 runs: H32 r3, A r3, H32 r4, A r4, H32 r5


## Round 2: H32

A references: pending

Round 2 passed: –

