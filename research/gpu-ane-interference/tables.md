### Solo baselines (closed loop, one device, one stream)

| Model | ANE placement | Stream | L | n | req/s | service mean | P50 | P95 | P99 | P99 95% CI | e2e P99 | forward | device exec | host | CPU ms | mismatches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| typed-decisions | thread | gpu_M | 128 | 6025 | 80.4 | 12.28 | 12.30 | 12.40 | 12.47 | 12.46–12.48 | 12.67 | 12.23 | 11.84 | 0.39 | 1.51 | 0 |
| typed-decisions | thread | gpu_L | 1024 | 1041 | 13.9 | 71.10 | 71.12 | 71.24 | 71.36 | 71.32–71.48 | 72.36 | 71.02 | 70.57 | 0.44 | 1.73 | 0 |
| typed-decisions | thread | ane_S | 64 | 9114 | 121.4 | 8.12 | 8.13 | 8.17 | 8.24 | 8.23–8.25 | 8.38 | 8.12 | 8.00 | 0.11 | 0.36 | 0 |
| typed-decisions | thread | ane_B | 128 | 7428 | 99.0 | 9.94 | 9.96 | 10.06 | 10.10 | 10.09–10.11 | 10.27 | 9.94 | 9.73 | 0.21 | 0.56 | 0 |
| typed-decisions | process | gpu_M | 128 | 6023 | 80.2 | 12.28 | 12.31 | 12.40 | 12.48 | 12.46–12.49 | 12.69 | 12.23 | 11.84 | 0.39 | 1.51 | 0 |
| typed-decisions | process | gpu_L | 1024 | 1041 | 13.9 | 71.08 | 71.11 | 71.26 | 71.41 | 71.35–71.43 | 72.41 | 71.00 | 70.54 | 0.45 | 1.77 | 0 |
| typed-decisions | process | ane_S | 64 | 9048 | 120.6 | 8.18 | 8.18 | 8.22 | 8.29 | 8.28–8.31 | 8.43 | 8.13 | 8.02 | 0.11 | 0.36 | 0 |
| typed-decisions | process | ane_B | 128 | 7371 | 98.2 | 10.02 | 10.02 | 10.11 | 10.20 | 10.17–10.22 | 10.42 | 9.97 | 9.75 | 0.21 | 0.56 | 0 |
| multilingual | thread | gpu_M | 128 | 11368 | 151.7 | 6.47 | 6.50 | 6.57 | 6.65 | 6.64–6.66 | 6.79 | 6.43 | 6.13 | 0.30 | 1.20 | 0 |
| multilingual | thread | gpu_L | 1024 | 2517 | 33.6 | 29.29 | 29.29 | 29.39 | 29.47 | 29.46–29.50 | 30.11 | 29.22 | 28.86 | 0.35 | 1.24 | 0 |
| multilingual | thread | ane_S | 64 | 21506 | 286.8 | 3.39 | 3.40 | 3.43 | 3.48 | 3.48–3.49 | 3.59 | 3.39 | 3.29 | 0.10 | 0.30 | 0 |
| multilingual | thread | ane_B | 256 | 8818 | 117.5 | 8.34 | 8.36 | 8.41 | 8.47 | 8.46–8.48 | 8.66 | 8.34 | 7.98 | 0.36 | 0.89 | 0 |
| multilingual | process | gpu_M | 128 | 11358 | 151.5 | 6.48 | 6.51 | 6.58 | 6.66 | 6.65–6.67 | 6.80 | 6.44 | 6.14 | 0.30 | 1.21 | 0 |
| multilingual | process | gpu_L | 1024 | 2517 | 33.6 | 29.30 | 29.31 | 29.41 | 29.52 | 29.49–29.54 | 30.13 | 29.23 | 28.87 | 0.36 | 1.26 | 0 |
| multilingual | process | ane_S | 64 | 21134 | 282.9 | 3.45 | 3.45 | 3.52 | 3.59 | 3.58–3.59 | 3.72 | 3.41 | 3.30 | 0.10 | 0.31 | 0 |
| multilingual | process | ane_B | 256 | 8772 | 116.9 | 8.38 | 8.40 | 8.45 | 8.55 | 8.53–8.57 | 8.74 | 8.33 | 7.97 | 0.35 | 0.88 | 0 |

### Concurrent matrix (both devices closed loop): inflation against solo

| Model | ANE placement | Cell | Stream | n | service mean ms (solo) | service × (95% CI) | device exec × | Δ host ms | Δ dispatch ms | Δ CPU ms | P50 | P95 | P99 | e2e P99 × | req/s × | mismatches |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| typed-decisions | thread | gpu_M+ane_S | ane_S | 9027 | 8.18 (8.12) | ×1.008 (1.007–1.008) | ×0.994 | +0.11 | -0.00 | +0.02 | 8.14 | 8.34 | 8.42 | ×1.03 | ×0.991 | 0 |
| typed-decisions | thread | gpu_M+ane_S | gpu_M | 4514 | 16.43 (12.28) | ×1.339 (1.338–1.339) | ×1.036 | +0.02 | +3.70 | +0.07 | 16.41 | 16.58 | 16.86 | ×1.35 | ×0.749 | 0 |
| typed-decisions | thread | gpu_M+ane_B | ane_B | 7373 | 9.99 (9.94) | ×1.005 (1.004–1.006) | ×0.994 | +0.11 | +0.00 | +0.05 | 9.94 | 10.14 | 10.25 | ×1.03 | ×0.993 | 0 |
| typed-decisions | thread | gpu_M+ane_B | gpu_M | 3687 | 20.14 (12.28) | ×1.641 (1.640–1.641) | ×1.037 | +0.02 | +7.40 | +0.07 | 20.13 | 20.36 | 20.61 | ×1.66 | ×0.611 | 0 |
| typed-decisions | thread | gpu_L+ane_S | ane_S | 9011 | 8.20 (8.12) | ×1.010 (1.009–1.011) | ×0.997 | +0.10 | -0.00 | +0.01 | 8.10 | 9.09 | 9.16 | ×1.11 | ×0.990 | 0 |
| typed-decisions | thread | gpu_L+ane_S | gpu_L | 1002 | 74.04 (71.10) | ×1.041 (1.041–1.042) | ×1.007 | +0.06 | +2.41 | +0.10 | 73.96 | 74.45 | 75.61 | ×1.06 | ×0.963 | 0 |
| typed-decisions | thread | gpu_L+ane_B | ane_B | 7390 | 9.97 (9.94) | ×1.003 (1.002–1.004) | ×0.996 | +0.07 | -0.00 | +0.03 | 9.85 | 10.80 | 10.89 | ×1.08 | ×0.996 | 0 |
| typed-decisions | thread | gpu_L+ane_B | gpu_L | 929 | 79.64 (71.10) | ×1.120 (1.118–1.122) | ×1.007 | +0.07 | +8.01 | +0.14 | 80.03 | 80.46 | 80.78 | ×1.13 | ×0.890 | 0 |
| typed-decisions | process | gpu_M+ane_S | ane_S | 8000 | 8.98 (8.18) | ×1.097 (1.096–1.099) | ×1.043 | +0.30 | +0.14 | +0.65 | 9.12 | 9.57 | 9.90 | ×1.25 | ×0.858 | 0 |
| typed-decisions | process | gpu_M+ane_S | gpu_M | 5314 | 13.53 (12.28) | ×1.102 (1.100–1.104) | ×1.019 | +0.90 | +0.12 | +2.77 | 13.77 | 14.24 | 14.63 | ×1.23 | ×0.856 | 0 |
| typed-decisions | process | gpu_M+ane_B | ane_B | 6326 | 11.23 (10.02) | ×1.121 (1.119–1.123) | ×1.056 | +0.50 | +0.15 | +1.10 | 11.23 | 11.57 | 11.97 | ×1.24 | ×0.858 | 0 |
| typed-decisions | process | gpu_M+ane_B | gpu_M | 5203 | 13.79 (12.28) | ×1.123 (1.121–1.125) | ×1.025 | +1.06 | +0.14 | +3.30 | 13.77 | 14.26 | 14.70 | ×1.24 | ×0.865 | 0 |
| typed-decisions | process | gpu_L+ane_S | ane_S | 9062 | 8.16 (8.18) | ×0.997 (0.997–0.998) | ×0.994 | +0.01 | +0.01 | +0.01 | 8.14 | 8.27 | 8.57 | ×1.05 | ×1.002 | 0 |
| typed-decisions | process | gpu_L+ane_S | gpu_L | 1041 | 71.19 (71.08) | ×1.001 (1.001–1.002) | ×1.001 | +0.05 | +0.00 | +0.10 | 71.23 | 71.40 | 71.45 | ×1.00 | ×1.000 | 0 |
| typed-decisions | process | gpu_L+ane_B | ane_B | 6382 | 11.11 (10.02) | ×1.109 (1.106–1.113) | ×1.045 | +0.49 | +0.16 | +1.01 | 9.99 | 13.07 | 13.96 | ×1.88 | ×0.806 | 0 |
| typed-decisions | process | gpu_L+ane_B | gpu_L | 982 | 73.03 (71.08) | ×1.027 (1.026–1.029) | ×1.005 | +1.38 | +0.22 | +3.73 | 71.39 | 75.97 | 76.42 | ×1.15 | ×0.919 | 0 |
| multilingual | thread | gpu_M+ane_S | ane_S | 13075 | 5.27 (3.39) | ×1.555 (1.550–1.559) | ×1.444 | +0.41 | +0.00 | +0.67 | 5.38 | 6.47 | 6.82 | ×2.05 | ×0.586 | 0 |
| multilingual | thread | gpu_M+ane_S | gpu_M | 6532 | 10.98 (6.47) | ×1.696 (1.689–1.702) | ×1.112 | +1.19 | +2.62 | +2.95 | 11.33 | 12.71 | 13.14 | ×2.04 | ×0.553 | 0 |
| multilingual | thread | gpu_M+ane_B | ane_B | 8583 | 8.53 (8.34) | ×1.024 (1.023–1.024) | ×1.003 | +0.18 | -0.00 | +0.12 | 8.54 | 8.60 | 8.72 | ×1.05 | ×0.974 | 0 |
| multilingual | thread | gpu_M+ane_B | gpu_M | 8583 | 8.59 (6.47) | ×1.327 (1.326–1.327) | ×1.006 | +0.03 | +2.05 | +0.03 | 8.59 | 8.72 | 8.89 | ×1.34 | ×0.754 | 0 |
| multilingual | thread | gpu_L+ane_S | ane_S | 13729 | 4.89 (3.39) | ×1.442 (1.436–1.448) | ×1.209 | +0.80 | +0.00 | +1.01 | 4.45 | 8.22 | 9.27 | ×2.76 | ×0.638 | 0 |
| multilingual | thread | gpu_L+ane_S | gpu_L | 1974 | 34.92 (29.29) | ×1.192 (1.190–1.194) | ×1.017 | +2.27 | +2.86 | +5.45 | 34.98 | 36.88 | 37.86 | ×1.37 | ×0.784 | 0 |
| multilingual | thread | gpu_L+ane_B | ane_B | 8714 | 8.42 (8.34) | ×1.010 (1.009–1.010) | ×0.994 | +0.13 | -0.00 | +0.04 | 8.26 | 8.95 | 9.02 | ×1.06 | ×0.989 | 0 |
| multilingual | thread | gpu_L+ane_B | gpu_L | 2179 | 33.90 (29.29) | ×1.157 (1.157–1.158) | ×1.009 | +0.03 | +4.30 | +0.07 | 33.88 | 34.11 | 34.79 | ×1.18 | ×0.865 | 0 |
| multilingual | process | gpu_M+ane_S | ane_S | 14399 | 4.91 (3.45) | ×1.421 (1.418–1.424) | ×1.341 | +0.22 | +0.10 | +0.54 | 5.07 | 5.53 | 5.84 | ×1.70 | ×0.667 | 0 |
| multilingual | process | gpu_M+ane_S | gpu_M | 9555 | 7.45 (6.48) | ×1.150 (1.149–1.151) | ×1.028 | +0.69 | +0.10 | +2.14 | 7.41 | 7.97 | 8.18 | ×1.30 | ×0.837 | 0 |
| multilingual | process | gpu_M+ane_B | ane_B | 7160 | 10.01 (8.38) | ×1.195 (1.194–1.196) | ×1.119 | +0.60 | +0.08 | +1.35 | 9.98 | 10.58 | 10.85 | ×1.32 | ×0.817 | 0 |
| multilingual | process | gpu_M+ane_B | gpu_M | 9969 | 7.20 (6.48) | ×1.112 (1.111–1.113) | ×1.026 | +0.49 | +0.08 | +1.62 | 7.18 | 7.60 | 7.99 | ×1.25 | ×0.878 | 0 |
| multilingual | process | gpu_L+ane_S | ane_S | 14782 | 4.60 (3.45) | ×1.333 (1.329–1.338) | ×1.170 | +0.37 | +0.20 | +0.84 | 4.47 | 5.21 | 6.85 | ×2.03 | ×0.696 | 0 |
| multilingual | process | gpu_L+ane_S | gpu_L | 2175 | 32.08 (29.30) | ×1.095 (1.093–1.098) | ×1.022 | +1.81 | +0.32 | +4.79 | 32.06 | 32.92 | 33.49 | ×1.22 | ×0.864 | 0 |
| multilingual | process | gpu_L+ane_B | ane_B | 6284 | 11.03 (8.38) | ×1.317 (1.314–1.321) | ×1.143 | +1.24 | +0.26 | +2.31 | 10.89 | 12.22 | 13.10 | ×1.66 | ×0.717 | 0 |
| multilingual | process | gpu_L+ane_B | gpu_L | 2207 | 31.68 (29.30) | ×1.081 (1.079–1.085) | ×1.018 | +1.56 | +0.29 | +4.26 | 31.48 | 32.83 | 33.35 | ×1.22 | ×0.877 | 0 |

### Aggregate throughput of the matrix cells

| Model | ANE placement | Cell | GPU req/s (solo) | ANE req/s (solo) | GPU busy | ANE busy |
|---|---|---|---:|---:|---:|---:|
| typed-decisions | thread | gpu_M+ane_S | 120.4 (121.4) | 60.2 (80.4) | 0.99 | 0.98 |
| typed-decisions | thread | gpu_M+ane_B | 98.3 (99.0) | 49.2 (80.4) | 0.99 | 0.98 |
| typed-decisions | thread | gpu_L+ane_S | 120.2 (121.4) | 13.4 (13.9) | 0.99 | 0.99 |
| typed-decisions | thread | gpu_L+ane_B | 98.6 (99.0) | 12.4 (13.9) | 0.99 | 0.98 |
| typed-decisions | process | gpu_M+ane_S | 103.5 (120.6) | 68.7 (80.2) | 0.96 | 0.96 |
| typed-decisions | process | gpu_M+ane_B | 84.3 (98.2) | 69.4 (80.2) | 0.96 | 0.95 |
| typed-decisions | process | gpu_L+ane_S | 120.9 (120.6) | 13.9 (13.9) | 0.99 | 0.99 |
| typed-decisions | process | gpu_L+ane_B | 79.2 (98.2) | 12.8 (13.9) | 0.96 | 0.95 |
| multilingual | thread | gpu_M+ane_S | 168.0 (286.8) | 83.9 (151.7) | 0.96 | 0.92 |
| multilingual | thread | gpu_M+ane_B | 114.4 (117.5) | 114.4 (151.7) | 0.98 | 0.98 |
| multilingual | thread | gpu_L+ane_S | 183.0 (286.8) | 26.3 (33.6) | 0.92 | 0.90 |
| multilingual | thread | gpu_L+ane_B | 116.2 (117.5) | 29.0 (33.6) | 0.98 | 0.98 |
| multilingual | process | gpu_M+ane_S | 188.8 (282.9) | 126.8 (151.5) | 0.95 | 0.94 |
| multilingual | process | gpu_M+ane_B | 95.5 (116.9) | 133.0 (151.5) | 0.96 | 0.96 |
| multilingual | process | gpu_L+ane_S | 196.8 (282.9) | 29.0 (33.6) | 0.93 | 0.91 |
| multilingual | process | gpu_L+ane_B | 83.8 (116.9) | 29.4 (33.6) | 0.93 | 0.92 |

### Host-side controls (one closed-loop stream, no second device)

| Model | ANE placement | Stream | Aggressor | service × (95% CI) | device exec × | Δ dispatch ms | Δ host ms | e2e P99 × | n |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| typed-decisions | thread | ane_B | 4 CPU-burning processes | ×1.000 (1.000–1.001) | ×0.998 | -0.00 | +0.02 | ×1.01 | 7409 |
| typed-decisions | thread | ane_B | pure-Python thread in the caller | ×9.544 (9.514–9.576) | ×5.085 | -0.00 | +45.18 | ×11.67 | 750 |
| typed-decisions | thread | ane_B | memory copy process (139 GB/s) | ×1.001 (1.000–1.001) | ×0.992 | +0.00 | +0.09 | ×1.01 | 7349 |
| typed-decisions | thread | gpu_L | 4 CPU-burning processes | ×0.992 (0.992–0.992) | ×0.991 | +0.01 | +0.05 | ×1.00 | 1050 |
| typed-decisions | thread | gpu_L | pure-Python thread in the caller | ×1.120 (1.118–1.123) | ×0.997 | +8.71 | +0.02 | ×1.41 | 851 |
| typed-decisions | thread | gpu_L | memory copy process (140 GB/s) | ×1.004 (1.004–1.004) | ×0.997 | +0.11 | +0.36 | ×1.01 | 1032 |
| typed-decisions | process | ane_B | 4 CPU-burning processes | ×0.997 (0.996–0.998) | ×0.994 | +0.00 | +0.03 | ×1.01 | 7373 |
| typed-decisions | process | ane_B | pure-Python thread in the caller | ×1.981 (1.968–1.992) | ×1.174 | +8.10 | +0.02 | ×3.93 | 3229 |
| typed-decisions | process | ane_B | memory copy process (139 GB/s) | ×1.005 (1.004–1.006) | ×0.992 | +0.04 | +0.09 | ×1.04 | 7250 |
| typed-decisions | process | gpu_L | 4 CPU-burning processes | ×0.992 (0.992–0.992) | ×0.991 | +0.01 | +0.04 | ×0.99 | 1050 |
| typed-decisions | process | gpu_L | pure-Python thread in the caller | ×1.119 (1.116–1.121) | ×0.998 | +8.57 | +0.02 | ×1.41 | 855 |
| typed-decisions | process | gpu_L | memory copy process (140 GB/s) | ×1.004 (1.004–1.004) | ×0.998 | +0.10 | +0.34 | ×1.01 | 1032 |
| multilingual | thread | ane_B | 4 CPU-burning processes | ×1.020 (1.020–1.020) | ×1.015 | -0.00 | +0.05 | ×1.04 | 8627 |
| multilingual | thread | ane_B | pure-Python thread in the caller | ×12.380 (12.322–12.436) | ×6.101 | -0.00 | +54.18 | ×14.86 | 689 |
| multilingual | thread | ane_B | memory copy process (143 GB/s) | ×1.021 (1.020–1.024) | ×1.009 | -0.00 | +0.10 | ×1.05 | 8554 |
| multilingual | thread | gpu_L | 4 CPU-burning processes | ×0.986 (0.986–0.986) | ×0.984 | +0.01 | +0.03 | ×1.00 | 2545 |
| multilingual | thread | gpu_L | pure-Python thread in the caller | ×1.302 (1.298–1.307) | ×0.996 | +8.93 | +0.03 | ×1.99 | 1651 |
| multilingual | thread | gpu_L | memory copy process (141 GB/s) | ×1.005 (1.005–1.005) | ×0.990 | +0.11 | +0.31 | ×1.02 | 2471 |
| multilingual | process | ane_B | 4 CPU-burning processes | ×1.023 (1.022–1.024) | ×1.016 | +0.00 | +0.06 | ×1.04 | 8560 |
| multilingual | process | ane_B | pure-Python thread in the caller | ×2.276 (2.261–2.292) | ×1.313 | +8.15 | +0.05 | ×4.49 | 3320 |
| multilingual | process | ane_B | memory copy process (143 GB/s) | ×1.026 (1.025–1.026) | ×1.012 | +0.03 | +0.09 | ×1.05 | 8467 |
| multilingual | process | gpu_L | 4 CPU-burning processes | ×0.986 (0.986–0.986) | ×0.985 | +0.01 | +0.02 | ×1.00 | 2545 |
| multilingual | process | gpu_L | pure-Python thread in the caller | ×1.294 (1.290–1.299) | ×0.996 | +8.72 | +0.01 | ×1.98 | 1681 |
| multilingual | process | gpu_L | memory copy process (141 GB/s) | ×1.004 (1.004–1.004) | ×0.990 | +0.10 | +0.29 | ×1.01 | 2478 |

### Load sweep: victim mean service × against the other device's realised busy fraction

| Model | ANE placement | Victim | Other device | points (busy fraction → service ×) |
|---|---|---|---|---|
| typed-decisions | thread | ane_B | gpu_L | 0.00 → ×1.000, 0.29 → ×1.004, 0.52 → ×1.005, 0.78 → ×1.005, 0.99 → ×1.003 |
| typed-decisions | thread | ane_B | gpu_M | 0.00 → ×1.000, 0.40 → ×1.008, 0.81 → ×1.008, 0.99 → ×1.005, 1.00 → ×1.008 |
| typed-decisions | thread | gpu_L | ane_B | 0.00 → ×1.000, 0.24 → ×1.014, 0.50 → ×1.034, 0.76 → ×1.064, 0.98 → ×1.120 |
| typed-decisions | thread | gpu_M | ane_B | 0.00 → ×1.000, 0.25 → ×1.108, 0.49 → ×1.202, 0.77 → ×1.393, 0.98 → ×1.641 |
| typed-decisions | process | ane_B | gpu_L | 0.00 → ×1.000, 0.27 → ×1.003, 0.48 → ×1.001, 0.72 → ×0.995, 0.96 → ×1.109 |
| typed-decisions | process | ane_B | gpu_M | 0.00 → ×1.000, 0.27 → ×1.009, 0.51 → ×1.004, 0.75 → ×0.998, 0.96 → ×1.121 |
| typed-decisions | process | gpu_L | ane_B | 0.00 → ×1.000, 0.24 → ×0.999, 0.49 → ×0.999, 0.74 → ×0.999, 0.95 → ×1.027 |
| typed-decisions | process | gpu_M | ane_B | 0.00 → ×1.000, 0.24 → ×0.997, 0.49 → ×0.997, 0.75 → ×0.998, 0.95 → ×1.123 |
| multilingual | thread | ane_B | gpu_L | 0.00 → ×1.000, 0.29 → ×1.007, 0.57 → ×1.009, 0.85 → ×1.011, 0.98 → ×1.010 |
| multilingual | thread | ane_B | gpu_M | 0.00 → ×1.000, 0.37 → ×1.012, 0.68 → ×1.018, 0.98 → ×1.024, 0.99 → ×1.023 |
| multilingual | thread | gpu_L | ane_B | 0.00 → ×1.000, 0.24 → ×1.028, 0.50 → ×1.066, 0.78 → ×1.112, 0.98 → ×1.157 |
| multilingual | thread | gpu_M | ane_B | 0.00 → ×1.000, 0.25 → ×1.142, 0.51 → ×1.240, 0.79 → ×1.313, 0.98 → ×1.327 |
| multilingual | process | ane_B | gpu_L | 0.00 → ×1.000, 0.26 → ×1.002, 0.50 → ×0.998, 0.73 → ×0.995, 0.93 → ×1.317 |
| multilingual | process | ane_B | gpu_M | 0.00 → ×1.000, 0.28 → ×1.003, 0.52 → ×1.002, 0.76 → ×1.002, 0.96 → ×1.195 |
| multilingual | process | gpu_L | ane_B | 0.00 → ×1.000, 0.24 → ×0.997, 0.49 → ×0.998, 0.76 → ×0.998, 0.92 → ×1.081 |
| multilingual | process | gpu_M | ane_B | 0.00 → ×1.000, 0.24 → ×1.002, 0.50 → ×1.004, 0.85 → ×1.058, 0.96 → ×1.112 |

### Sparse load: one device alone, open loop

| Model (placement) | Stream | offered load | n | service × (95% CI) | CPU ms × | device exec × |
|---|---|---:|---:|---|---:|---:|
| typed-decisions (thread) | ane_B | 0.05 | 351 | ×2.02 (1.99–2.05) | ×6.54 | ×1.89 |
| typed-decisions (thread) | ane_B | 0.1 | 732 | ×1.95 (1.93–1.97) | ×6.03 | ×1.83 |
| typed-decisions (thread) | ane_B | 0.25 | 1995 | ×1.66 (1.64–1.67) | ×4.51 | ×1.57 |
| typed-decisions (thread) | ane_B | 0.5 | 3801 | ×1.26 (1.25–1.27) | ×2.14 | ×1.23 |
| typed-decisions (thread) | ane_B | 0.1 + 1 CPU-busy process | 732 | ×1.12 (1.11–1.14) | ×1.15 | ×1.12 |
| typed-decisions (thread) | gpu_L | 0.05 | 48 | ×1.25 (1.21–1.28) | ×5.91 | ×1.19 |
| typed-decisions (thread) | gpu_L | 0.1 | 78 | ×1.19 (1.16–1.21) | ×5.38 | ×1.14 |
| typed-decisions (thread) | gpu_L | 0.25 | 225 | ×1.10 (1.09–1.11) | ×4.16 | ×1.07 |
| typed-decisions (thread) | gpu_L | 0.5 | 474 | ×1.05 (1.04–1.05) | ×2.43 | ×1.03 |
| typed-decisions (thread) | gpu_M | 0.05 | 267 | ×1.78 (1.74–1.83) | ×6.95 | ×1.50 |
| typed-decisions (thread) | gpu_M | 0.1 | 579 | ×1.64 (1.62–1.66) | ×6.78 | ×1.36 |
| typed-decisions (thread) | gpu_M | 0.25 | 1608 | ×1.40 (1.39–1.42) | ×5.72 | ×1.18 |
| typed-decisions (thread) | gpu_M | 0.5 | 3093 | ×1.22 (1.21–1.23) | ×4.13 | ×1.07 |
| typed-decisions (thread) | gpu_M | 0.1 + 1 CPU-busy process | 579 | ×1.24 (1.23–1.26) | ×1.09 | ×1.24 |
| multilingual (process) | ane_B | 0.05 | 411 | ×2.49 (2.47–2.51) | ×6.20 | ×2.22 |
| multilingual (process) | ane_B | 0.1 | 882 | ×2.38 (2.36–2.39) | ×5.90 | ×2.12 |
| multilingual (process) | ane_B | 0.25 | 2358 | ×1.87 (1.85–1.89) | ×4.13 | ×1.71 |
| multilingual (process) | ane_B | 0.5 | 4548 | ×1.29 (1.28–1.30) | ×1.79 | ×1.25 |
| multilingual (process) | ane_B | 0.1 + 1 CPU-busy process | 882 | ×1.15 (1.14–1.17) | ×1.13 | ×1.15 |
| multilingual (process) | gpu_L | 0.05 | 93 | ×1.51 (1.43–1.58) | ×7.25 | ×1.39 |
| multilingual (process) | gpu_L | 0.1 | 216 | ×1.34 (1.31–1.37) | ×6.85 | ×1.22 |
| multilingual (process) | gpu_L | 0.25 | 621 | ×1.22 (1.21–1.23) | ×5.99 | ×1.12 |
| multilingual (process) | gpu_L | 0.5 | 1314 | ×1.09 (1.09–1.10) | ×3.31 | ×1.05 |
| multilingual (process) | gpu_M | 0.05 | 525 | ×2.19 (2.15–2.22) | ×7.39 | ×1.73 |
| multilingual (process) | gpu_M | 0.1 | 1185 | ×1.90 (1.87–1.92) | ×6.78 | ×1.48 |
| multilingual (process) | gpu_M | 0.25 | 2976 | ×1.59 (1.57–1.60) | ×5.54 | ×1.26 |
| multilingual (process) | gpu_M | 0.5 | 5793 | ×1.34 (1.33–1.34) | ×4.26 | ×1.10 |
| multilingual (process) | gpu_M | 0.1 + 1 CPU-busy process | 1185 | ×1.32 (1.30–1.34) | ×1.07 | ×1.32 |

### Product path, v0.2 Part A mix (closed loop, Laya.submit, v0.2 router)

| Model | ANE placement | Stream | req/s (solo) | P50 | P95 | P99 (solo) | Δ P99 | v0.2 Δ P99 | queue mean | service mean (solo) | tail: pre / queue / service / post ms | n | mismatches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| typed-decisions | thread | short L128 | 98.4 (98.6) | 10.03 | 10.98 | 11.07 (10.29) | +8% | +8% | 0.02 | 9.95 (9.95) | 0.15 / 0.18 / 10.67 / 0.17 | 7385 | 0 |
| typed-decisions | thread | long L1024 | 12.3 (13.9) | 81.19 | 81.60 | 81.79 (72.34) | +13% | +13% | 0.10 | 79.97 (71.12) | 0.87 / 2.00 / 79.78 / 0.28 | 925 | 0 |
| typed-decisions | process | short L128 | 71.0 (98.3) | 14.01 | 15.37 | 20.77 (10.38) | +100% | +91% | 0.18 | 12.63 (9.99) | 1.10 / 5.14 / 15.43 / 0.61 | 5335 | 0 |
| typed-decisions | process | long L1024 | 12.3 (13.9) | 81.07 | 83.03 | 83.62 (72.32) | +16% | +15% | 0.31 | 74.99 (71.11) | 6.59 / 1.63 / 75.54 / 0.38 | 927 | 0 |
| multilingual | thread | short L96 | 158.4 (246.8) | 5.75 | 9.45 | 10.91 (4.15) | +163% | +153% | 0.10 | 5.56 (3.93) | 0.57 / 1.12 / 9.14 / 0.54 | 11918 | 0 |
| multilingual | thread | long L1024 | 26.4 (33.6) | 37.67 | 40.70 | 41.89 (30.11) | +39% | +38% | 0.18 | 34.78 (29.29) | 3.52 / 0.77 / 38.50 / 0.72 | 1988 | 0 |
| multilingual | process | short L96 | 171.8 (246.4) | 5.73 | 6.54 | 8.10 (4.20) | +93% | +99% | 0.05 | 5.22 (3.94) | 0.51 / 1.20 / 6.61 / 0.68 | 12842 | 0 |
| multilingual | process | long L1024 | 29.0 (33.5) | 34.32 | 35.79 | 36.55 (30.15) | +21% | +21% | 0.22 | 31.96 (29.30) | 2.84 / 0.88 / 33.21 / 0.20 | 2177 | 0 |

### Product path, open loop (v0.2 Part B class mix): per class

| Model | ANE placement | offered req/s | class | n | GPU share | P50 | P95 | P99 | queue mean | service mean | tail: queue / service ms | mismatches |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| typed-decisions | thread | 15 | all | 1137 | 0.45 | 25.8 | 110.7 | 173.1 | 7.8 | 31.7 | 139.5 / 52.1 | 0 |
| typed-decisions | thread | 25 | all | 1827 | 0.43 | 25.2 | 131.3 | 189.1 | 12.6 | 28.9 | 184.4 / 54.1 | 0 |
| typed-decisions | thread | 35 | all | 2613 | 0.43 | 22.4 | 183.1 | 255.9 | 21.2 | 26.8 | 247.0 / 51.3 | 0 |
| typed-decisions | process | 15 | all | 1137 | 0.45 | 24.6 | 110.4 | 172.6 | 8.0 | 31.5 | 136.5 / 52.2 | 0 |
| typed-decisions | process | 25 | all | 1827 | 0.44 | 23.9 | 132.2 | 185.6 | 12.8 | 28.5 | 177.8 / 53.5 | 0 |
| typed-decisions | process | 35 | all | 2613 | 0.45 | 22.2 | 175.4 | 236.1 | 20.6 | 26.6 | 239.5 / 50.2 | 0 |
| multilingual | thread | 120 | all | 8925 | 0.40 | 7.0 | 142.3 | 203.0 | 22.0 | 9.7 | 198.5 / 18.1 | 0 |
| multilingual | thread | 40 | all | 2946 | 0.43 | 15.1 | 57.5 | 79.8 | 4.3 | 15.4 | 60.1 / 23.2 | 0 |
| multilingual | thread | 80 | all | 5829 | 0.42 | 12.6 | 69.6 | 104.9 | 8.0 | 11.8 | 94.0 / 23.3 | 0 |
| multilingual | process | 120 | all | 8925 | 0.40 | 7.8 | 131.3 | 182.8 | 20.2 | 9.8 | 182.1 / 17.2 | 0 |
| multilingual | process | 40 | all | 2946 | 0.43 | 13.9 | 52.2 | 73.5 | 4.2 | 15.2 | 59.2 / 23.7 | 0 |
| multilingual | process | 80 | all | 5829 | 0.43 | 13.1 | 67.5 | 95.8 | 8.3 | 12.2 | 87.0 / 22.0 | 0 |

### Routing hindsight (product open loop, short 1-question class)

| Model | ANE placement | offered req/s | decision | n | misses | miss rate | mean miss cost ms |
|---|---|---:|---|---:|---:|---:|---:|
| typed-decisions | thread | 15 | ane_backlog_shorter_on_gpu|busy | 26 | 0 | 0.0% | 0.0 |
| typed-decisions | thread | 15 | validated_short_single_question_path|busy | 182 | 0 | 0.0% | 0.0 |
| typed-decisions | thread | 15 | validated_short_single_question_path|idle | 446 | 61 | 13.7% | 3.0 |
| typed-decisions | thread | 25 | ane_backlog_shorter_on_gpu|busy | 24 | 4 | 16.7% | 5.2 |
| typed-decisions | thread | 25 | validated_short_single_question_path|busy | 483 | 0 | 0.0% | 0.0 |
| typed-decisions | thread | 25 | validated_short_single_question_path|idle | 555 | 45 | 8.1% | 2.7 |
| typed-decisions | thread | 35 | ane_backlog_shorter_on_gpu|busy | 37 | 3 | 8.1% | 8.0 |
| typed-decisions | thread | 35 | validated_short_single_question_path|busy | 999 | 3 | 0.3% | 3.4 |
| typed-decisions | thread | 35 | validated_short_single_question_path|idle | 485 | 112 | 23.1% | 3.9 |
| typed-decisions | process | 15 | ane_backlog_shorter_on_gpu|busy | 33 | 0 | 0.0% | 0.0 |
| typed-decisions | process | 15 | validated_short_single_question_path|busy | 190 | 2 | 1.1% | 7.5 |
| typed-decisions | process | 15 | validated_short_single_question_path|idle | 431 | 130 | 30.2% | 1.8 |
| typed-decisions | process | 25 | ane_backlog_shorter_on_gpu|busy | 48 | 0 | 0.0% | 0.0 |
| typed-decisions | process | 25 | validated_short_single_question_path|busy | 521 | 13 | 2.5% | 8.5 |
| typed-decisions | process | 25 | validated_short_single_question_path|idle | 493 | 85 | 17.2% | 2.2 |
| typed-decisions | process | 35 | ane_backlog_shorter_on_gpu|busy | 71 | 5 | 7.0% | 2.9 |
| typed-decisions | process | 35 | validated_short_single_question_path|busy | 1027 | 9 | 0.9% | 6.7 |
| typed-decisions | process | 35 | validated_short_single_question_path|idle | 423 | 151 | 35.7% | 4.5 |
| multilingual | thread | 120 | ane_backlog_shorter_on_gpu|busy | 22 | 4 | 18.2% | 2.0 |
| multilingual | thread | 120 | validated_short_single_question_path|busy | 4684 | 0 | 0.0% | 0.0 |
| multilingual | thread | 120 | validated_short_single_question_path|idle | 679 | 2 | 0.3% | 1.5 |
| multilingual | thread | 40 | ane_backlog_shorter_on_gpu|busy | 53 | 5 | 9.4% | 5.2 |
| multilingual | thread | 40 | validated_short_single_question_path|busy | 566 | 6 | 1.1% | 6.8 |
| multilingual | thread | 40 | validated_short_single_question_path|idle | 1112 | 53 | 4.8% | 2.6 |
| multilingual | thread | 80 | ane_backlog_shorter_on_gpu|busy | 90 | 18 | 20.0% | 5.1 |
| multilingual | thread | 80 | validated_short_single_question_path|busy | 2126 | 13 | 0.6% | 5.3 |
| multilingual | thread | 80 | validated_short_single_question_path|idle | 1237 | 175 | 14.1% | 3.2 |
| multilingual | process | 120 | ane_backlog_shorter_on_gpu|busy | 73 | 2 | 2.7% | 4.0 |
| multilingual | process | 120 | validated_short_single_question_path|busy | 4695 | 32 | 0.7% | 4.2 |
| multilingual | process | 120 | validated_short_single_question_path|idle | 617 | 49 | 7.9% | 3.4 |
| multilingual | process | 40 | ane_backlog_shorter_on_gpu|busy | 59 | 0 | 0.0% | 0.0 |
| multilingual | process | 40 | validated_short_single_question_path|busy | 599 | 44 | 7.3% | 5.5 |
| multilingual | process | 40 | validated_short_single_question_path|idle | 1073 | 86 | 8.0% | 3.0 |
| multilingual | process | 80 | ane_backlog_shorter_on_gpu|busy | 128 | 0 | 0.0% | 0.0 |
| multilingual | process | 80 | validated_short_single_question_path|busy | 2234 | 95 | 4.3% | 7.5 |
| multilingual | process | 80 | validated_short_single_question_path|idle | 1091 | 434 | 39.8% | 2.9 |

### Scheduler prototype A/B: typed-decisions (ANE thread)

Calibrated: GPU service × (1 + 0.012) + 7.72 ms while the ANE is busy; ANE service × (1 + 0.000) + 0.06 ms while the GPU is busy.

| Workload | offered req/s | class | n | P50 base → proto | P95 | P99 | P99 × (95% CI) | mean × | GPU share | queue mean | service mean | req/s | mismatches |
|---|---:|---|---:|---|---|---|---|---:|---|---|---|---|---|
| heavy | 100 | A | 6600 | 24.9 → 27.5 | 74.2 → 72.0 | 101.4 → 97.6 | ×0.962 (0.91–1.01) | ×1.044 | 0.088 → 0.062 | 16.0 → 17.4 | 11.0 → 10.8 | | |
| heavy | 100 | L | 750 | 156.4 → 143.9 | 396.6 → 387.1 | 594.4 → 615.1 | ×1.035 (0.92–1.19) | ×0.953 | 1.000 → 1.000 | 106.5 → 97.5 | 75.7 → 75.9 | | |
| heavy | 100 | all | 7350 | 28.1 → 30.4 | 158.4 → 145.5 | 339.7 → 321.0 | ×0.945 (0.87–1.05) | – | – | – | – | 98.0 → 98.0 | 0 / 0 |
| heavy | 120 | A | 8019 | 452.9 → 501.2 | 996.3 → 1034.5 | 1069.6 → 1103.5 | ×1.032 (1.02–1.05) | ×1.048 | 0.051 → 0.051 | 458.8 → 481.4 | 10.7 → 10.7 | | |
| heavy | 120 | L | 906 | 782.1 → 809.6 | 1280.4 → 1303.6 | 1404.0 → 1407.8 | ×1.003 (0.96–1.06) | ×1.000 | 1.000 → 1.000 | 691.6 → 691.4 | 76.1 → 76.2 | | |
| heavy | 120 | all | 8925 | 472.4 → 516.2 | 1026.3 → 1063.2 | 1207.2 → 1220.1 | ×1.011 (0.98–1.05) | – | – | – | – | 119.0 → 119.0 | 0 / 0 |
| heavy | 80 | A | 5223 | 16.8 → 18.2 | 42.7 → 44.3 | 70.4 → 68.2 | ×0.969 (0.89–1.05) | ×1.053 | 0.089 → 0.049 | 5.5 → 6.8 | 11.2 → 10.8 | | |
| heavy | 80 | L | 606 | 107.2 → 99.3 | 328.2 → 308.8 | 439.3 → 405.0 | ×0.922 (0.83–1.02) | ×0.967 | 1.000 → 1.000 | 57.4 → 53.1 | 75.6 → 75.6 | | |
| heavy | 80 | all | 5829 | 18.2 → 19.9 | 110.2 → 104.0 | 248.1 → 247.8 | ×0.999 (0.85–1.11) | – | – | – | – | 77.7 → 77.7 | 0 / 0 |
| open | 15 | A | 654 | 21.9 → 21.8 | 28.6 → 29.9 | 38.4 → 40.4 | ×1.051 (0.92–1.20) | ×1.011 | 0.043 → 0.000 | 0.6 → 1.2 | 18.3 → 17.8 | | |
| open | 15 | L | 111 | 90.3 → 88.8 | 172.0 → 174.3 | 199.2 → 200.0 | ×1.004 (0.87–1.17) | ×1.011 | 1.000 → 1.000 | 20.7 → 21.6 | 78.0 → 78.0 | | |
| open | 15 | Md | 264 | 50.2 → 50.5 | 117.8 → 117.8 | 137.5 → 137.8 | ×1.002 (0.94–1.09) | ×1.000 | 1.000 → 1.000 | 12.9 → 13.1 | 41.7 → 41.9 | | |
| open | 15 | S4 | 108 | 54.2 → 52.7 | 155.2 → 158.8 | 215.6 → 216.4 | ×1.004 (0.80–1.29) | ×0.998 | 1.000 → 1.000 | 24.4 → 24.2 | 40.7 → 40.9 | | |
| open | 15 | all | 1137 | 25.6 → 26.0 | 111.3 → 112.4 | 171.3 → 174.4 | ×1.018 (0.92–1.13) | – | – | – | – | 15.2 → 15.2 | 0 / 0 |
| open | 25 | A | 1062 | 17.9 → 18.1 | 30.7 → 33.2 | 41.8 → 41.3 | ×0.987 (0.94–1.14) | ×1.016 | 0.024 → 0.000 | 0.8 → 1.3 | 15.4 → 15.2 | | |
| open | 25 | L | 168 | 92.3 → 91.7 | 192.5 → 186.6 | 272.1 → 262.3 | ×0.964 (0.71–1.35) | ×0.993 | 1.000 → 1.000 | 31.7 → 31.2 | 76.4 → 76.3 | | |
| open | 25 | Md | 417 | 54.7 → 55.2 | 146.8 → 146.2 | 181.4 → 179.7 | ×0.991 (0.85–1.11) | ×0.990 | 1.000 → 1.000 | 27.4 → 26.7 | 39.5 → 39.5 | | |
| open | 25 | S4 | 180 | 51.3 → 49.9 | 162.5 → 161.7 | 295.9 → 295.9 | ×1.000 (0.68–1.41) | ×0.997 | 1.000 → 1.000 | 30.7 → 30.6 | 38.6 → 38.6 | | |
| open | 25 | all | 1827 | 25.1 → 25.7 | 134.4 → 134.1 | 189.3 → 186.6 | ×0.986 (0.89–1.07) | – | – | – | – | 24.4 → 24.4 | 0 / 0 |
| open | 35 | A | 1521 | 11.6 → 11.5 | 28.2 → 29.5 | 40.0 → 39.0 | ×0.973 (0.92–1.11) | ×1.000 | 0.025 → 0.000 | 0.8 → 1.1 | 12.5 → 12.2 | | |
| open | 35 | L | 288 | 114.3 → 114.1 | 242.4 → 238.4 | 319.4 → 319.3 | ×1.000 (0.77–1.30) | ×0.994 | 1.000 → 1.000 | 54.5 → 54.3 | 73.5 → 73.3 | | |
| open | 35 | Md | 552 | 66.4 → 64.5 | 223.7 → 223.7 | 248.9 → 248.4 | ×0.998 (0.95–1.04) | ×0.995 | 1.000 → 1.000 | 52.7 → 52.3 | 37.4 → 37.5 | | |
| open | 35 | S4 | 252 | 52.4 → 51.3 | 216.3 → 216.4 | 349.0 → 345.8 | ×0.991 (0.92–1.02) | ×0.996 | 1.000 → 1.000 | 39.5 → 39.4 | 37.1 → 36.9 | | |
| open | 35 | all | 2613 | 22.1 → 22.4 | 186.3 → 185.3 | 256.2 → 255.9 | ×0.999 (0.91–1.10) | – | – | – | – | 34.8 → 34.8 | 0 / 0 |

Verdict against the pre-registered criteria:

| Workload | levels | short P99 improved (levels) | throughput within 2% | other-class regressions | mismatches | passes |
|---|---:|---:|---|---:|---:|---|
| heavy | 3 | 0 | True | 0 | 0 | no |
| open | 3 | 0 | True | 0 | 0 | no |

### Scheduler prototype A/B: multilingual (ANE process)

Calibrated: GPU service × (1 + 0.073) + 0.26 ms while the ANE is busy; ANE service × (1 + 0.306) + 0.09 ms while the GPU is busy.

| Workload | offered req/s | class | n | P50 base → proto | P95 | P99 | P99 × (95% CI) | mean × | GPU share | queue mean | service mean | req/s | mismatches |
|---|---:|---|---:|---|---|---|---|---:|---|---|---|---|---|
| heavy | 180 | A | 11994 | 10.0 → 11.2 | 34.0 → 36.9 | 55.0 → 70.5 | ×1.282 (1.17–1.37) | ×1.081 | 0.143 → 0.150 | 7.3 → 8.2 | 5.7 → 5.9 | | |
| heavy | 180 | L | 1380 | 46.0 → 48.3 | 119.4 → 122.6 | 150.7 → 159.1 | ×1.056 (0.97–1.11) | ×1.054 | 1.000 → 1.000 | 23.3 → 25.8 | 31.0 → 31.3 | | |
| heavy | 180 | all | 13374 | 11.2 → 12.3 | 54.4 → 59.5 | 106.4 → 106.3 | ×0.999 (0.94–1.08) | – | – | – | – | 178.3 → 178.3 | 0 / 0 |
| heavy | 240 | A | 16062 | 19.9 → 12.3 | 202.5 → 165.0 | 260.9 → 210.8 | ×0.808 (0.80–0.82) | ×0.657 | 0.107 → 0.104 | 51.0 → 31.9 | 4.8 → 4.6 | | |
| heavy | 240 | L | 1857 | 88.4 → 78.2 | 240.8 → 239.6 | 302.7 → 297.8 | ×0.984 (0.92–1.09) | ×0.916 | 1.000 → 1.000 | 76.1 → 67.5 | 30.3 → 30.0 | | |
| heavy | 240 | all | 17919 | 26.5 → 14.8 | 207.7 → 176.0 | 265.9 → 225.4 | ×0.848 (0.83–0.87) | – | – | – | – | 238.9 → 238.9 | 0 / 0 |
| heavy | 300 | A | 20241 | 1118.1 → 691.1 | 1686.4 → 1903.0 | 2523.4 → 2157.8 | ×0.855 (0.84–0.86) | ×0.782 | 0.082 → 0.080 | 1207.7 → 943.6 | 4.4 → 4.3 | | |
| heavy | 300 | L | 2232 | 1193.5 → 836.6 | 1692.6 → 2327.4 | 2043.3 → 2419.9 | ×1.184 (1.14–1.26) | ×0.945 | 1.000 → 1.000 | 1200.5 → 1132.4 | 29.7 → 29.5 | | |
| heavy | 300 | all | 22473 | 1123.5 → 738.1 | 1687.0 → 1936.0 | 2506.0 → 2283.3 | ×0.911 (0.90–0.94) | – | – | – | – | 299.6 → 299.6 | 0 / 0 |
| open | 120 | A | 5385 | 4.6 → 4.5 | 13.7 → 12.9 | 25.4 → 23.2 | ×0.913 (0.80–1.00) | ×0.959 | 0.014 → 0.012 | 1.5 → 1.3 | 4.6 → 4.5 | | |
| open | 120 | L | 867 | 69.0 → 67.6 | 168.5 → 168.7 | 189.2 → 188.6 | ×0.997 (0.93–1.06) | ×0.993 | 1.000 → 1.000 | 47.7 → 47.2 | 29.7 → 29.6 | | |
| open | 120 | Md | 1767 | 54.2 → 54.0 | 160.1 → 161.0 | 198.4 → 198.9 | ×1.002 (0.92–1.07) | ×0.993 | 1.000 → 1.000 | 50.1 → 49.8 | 15.5 → 15.4 | | |
| open | 120 | S4 | 906 | 49.7 → 48.2 | 159.3 → 159.6 | 195.7 → 195.3 | ×0.998 (0.93–1.08) | ×0.990 | 1.000 → 1.000 | 50.2 → 49.7 | 11.5 → 11.5 | | |
| open | 120 | all | 8925 | 8.0 → 7.8 | 130.7 → 130.2 | 178.6 → 178.5 | ×1.000 (0.98–1.02) | – | – | – | – | 119.0 → 119.0 | 0 / 0 |
| open | 40 | A | 1731 | 11.8 → 11.8 | 17.6 → 18.3 | 22.0 → 22.3 | ×1.014 (0.96–1.09) | ×1.014 | 0.034 → 0.024 | 1.0 → 1.2 | 9.8 → 9.9 | | |
| open | 40 | L | 306 | 43.3 → 42.7 | 75.0 → 73.5 | 85.8 → 88.3 | ×1.029 (0.82–1.23) | ×0.997 | 1.000 → 1.000 | 9.5 → 9.4 | 34.3 → 34.3 | | |
| open | 40 | Md | 624 | 26.7 → 27.2 | 58.2 → 57.5 | 74.4 → 73.2 | ×0.984 (0.90–1.11) | ×0.998 | 1.000 → 1.000 | 8.0 → 7.9 | 19.8 → 19.9 | | |
| open | 40 | S4 | 285 | 23.4 → 23.4 | 58.1 → 59.8 | 101.9 → 102.4 | ×1.005 (0.65–1.53) | ×1.001 | 1.000 → 1.000 | 9.0 → 9.1 | 16.8 → 16.8 | | |
| open | 40 | all | 2946 | 13.7 → 13.8 | 51.8 → 52.1 | 74.4 → 73.1 | ×0.983 (0.91–1.08) | – | – | – | – | 39.3 → 39.3 | 0 / 0 |
| open | 80 | A | 3453 | 9.2 → 9.0 | 19.2 → 19.5 | 26.1 → 26.6 | ×1.018 (0.93–1.12) | ×0.999 | 0.037 → 0.035 | 1.7 → 1.8 | 7.1 → 7.0 | | |
| open | 80 | L | 588 | 43.5 → 43.5 | 92.3 → 95.2 | 116.9 → 117.2 | ×1.003 (0.85–1.23) | ×1.004 | 1.000 → 1.000 | 18.1 → 18.4 | 31.2 → 31.1 | | |
| open | 80 | Md | 1182 | 28.7 → 28.9 | 81.9 → 82.8 | 98.9 → 101.4 | ×1.025 (0.95–1.12) | ×1.015 | 1.000 → 1.000 | 17.8 → 18.4 | 16.9 → 16.9 | | |
| open | 80 | S4 | 606 | 20.9 → 22.3 | 64.9 → 66.9 | 121.8 → 133.9 | ×1.100 (0.71–1.42) | ×1.024 | 1.000 → 1.000 | 14.4 → 15.1 | 13.2 → 13.2 | | |
| open | 80 | all | 5829 | 12.9 → 12.8 | 65.3 → 67.0 | 93.9 → 96.9 | ×1.032 (0.98–1.09) | – | – | – | – | 77.7 → 77.7 | 0 / 0 |

Verdict against the pre-registered criteria:

| Workload | levels | short P99 improved (levels) | throughput within 2% | other-class regressions | mismatches | passes |
|---|---:|---:|---|---:|---:|---|
| heavy | 3 | 2 | True | 1 | 0 | no |
| open | 3 | 0 | True | 0 | 0 | no |

