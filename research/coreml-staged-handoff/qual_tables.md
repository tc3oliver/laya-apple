# Environment qualification: production A, product schedule (laya, L128 / L512)

Outcome: environment PASS

## Runs

| run | mismatches | routing | crashed | workers alive | failed | machine findings |
|---|---|---|---|---|---|---|
| product-laya-A-r1 | 0 | 0 | False | {'ane': True, 'gpu': True} | – | – |
| product-laya-A-r2 | 0 | 0 | False | {'ane': True, 'gpu': True} | – | – |
| product-laya-A-r3 | 0 | 0 | False | {'ane': True, 'gpu': True} | – | – |

## Hetero windows

| run | window | n short | median | P95 | P99 | P99.9 | agg req/s | GPU return P50 / P95 / P99 | host-slow bins (longest) | slow spans (longest) |
|---|---|---|---|---|---|---|---|---|---|---|
| product-laya-A-r1 | 1 | 1955 | 10.04 | 10.69 | 11.41 | 15.19 | 122.2 | 4.368 / 6.305 / 7.377 | 0 | 0 |
| product-laya-A-r1 | 3 | 1965 | 10.03 | 10.56 | 11.69 | 15.93 | 122.9 | 4.285 / 4.599 / 5.340 | 0 | 0 |
| product-laya-A-r1 | 5 | 1957 | 10.04 | 10.62 | 11.21 | 15.27 | 122.4 | 4.316 / 5.627 / 6.355 | 0 | 0 |
| product-laya-A-r1 | 7 | 1962 | 10.03 | 10.56 | 11.06 | 14.74 | 122.6 | 4.286 / 4.732 / 5.962 | 0 | 0 |
| product-laya-A-r1 | 10 | 1961 | 10.05 | 10.61 | 11.25 | 15.76 | 122.6 | 4.317 / 4.870 / 6.004 | 0 | 0 |
| product-laya-A-r1 | 11 | 1949 | 10.08 | 10.82 | 12.15 | 15.97 | 121.9 | 4.373 / 5.580 / 6.390 | 1 | 0 |
| product-laya-A-r2 | 1 | 1958 | 10.04 | 10.62 | 11.13 | 15.71 | 122.4 | 4.308 / 4.662 / 5.740 | 0 | 0 |
| product-laya-A-r2 | 3 | 1960 | 10.04 | 10.62 | 11.79 | 16.82 | 122.5 | 4.305 / 4.730 / 5.649 | 0 | 0 |
| product-laya-A-r2 | 5 | 1956 | 10.05 | 10.66 | 11.13 | 14.81 | 122.3 | 4.331 / 4.730 / 6.092 | 0 | 0 |
| product-laya-A-r2 | 7 | 1957 | 10.05 | 10.67 | 10.85 | 14.24 | 122.3 | 4.365 / 4.768 / 5.892 | 0 | 0 |
| product-laya-A-r2 | 10 | 1958 | 10.05 | 10.67 | 11.31 | 15.06 | 122.3 | 4.354 / 4.883 / 6.127 | 0 | 0 |
| product-laya-A-r2 | 11 | 1954 | 10.06 | 10.70 | 11.49 | 15.18 | 122.2 | 4.384 / 4.818 / 5.756 | 0 | 0 |
| product-laya-A-r3 | 1 | 1960 | 10.04 | 10.58 | 11.07 | 15.43 | 122.5 | 4.292 / 4.802 / 5.892 | 0 | 0 |
| product-laya-A-r3 | 3 | 1964 | 10.04 | 10.58 | 11.22 | 14.51 | 122.7 | 4.286 / 4.857 / 6.091 | 0 | 0 |
| product-laya-A-r3 | 5 | 1959 | 10.04 | 10.60 | 11.34 | 15.70 | 122.4 | 4.282 / 4.989 / 6.081 | 0 | 0 |
| product-laya-A-r3 | 7 | 1960 | 10.04 | 10.60 | 10.89 | 15.41 | 122.5 | 4.285 / 4.722 / 5.457 | 0 | 0 |
| product-laya-A-r3 | 10 | 1960 | 10.06 | 10.59 | 11.25 | 15.14 | 122.5 | 4.356 / 4.769 / 5.847 | 0 | 0 |
| product-laya-A-r3 | 11 | 1960 | 10.06 | 10.61 | 11.03 | 15.30 | 122.5 | 4.369 / 4.798 / 5.930 | 0 | 0 |

## Pooled per run

| run | median | P95 | P99 | P99.9 | GPU return P50 / P95 / P99 |
|---|---|---|---|---|---|
| product-laya-A-r1 | 10.04 | 10.64 | 11.36 | 15.56 | 4.321 / 5.433 / 6.736 |
| product-laya-A-r2 | 10.05 | 10.66 | 11.26 | 15.69 | 4.343 / 4.770 / 5.861 |
| product-laya-A-r3 | 10.04 | 10.60 | 11.20 | 15.47 | 4.315 / 4.808 / 5.921 |

## Machine snapshots

| run | when | load | CPU idle % | memory free % | swap | thermal | power | top CPU |
|---|---|---|---|---|---|---|---|---|
| product-laya-A-r1 | before | 1.37 1.47 1.39 | 96.7 | 93 | total = 2048.00M  used = 758.94M  free = 1289.06M  (encrypted) | none | AC | OrbStack Helper 15%, WallpaperAerialsExtension 8%, claude 4% |
| product-laya-A-r1 | after | 2.34 3.17 2.29 | 97.2 | 93 | total = 2048.00M  used = 758.94M  free = 1289.06M  (encrypted) | none | AC | WallpaperAerialsExtension 8%, OrbStack Helper 5%, WindowServer 2% |
| product-laya-A-r2 | before | 2.34 3.17 2.29 | 97.5 | 93 | total = 2048.00M  used = 758.94M  free = 1289.06M  (encrypted) | none | AC | WallpaperAerialsExtension 6%, iTerm2 3%, OrbStack Helper 3% |
| product-laya-A-r2 | after | 1.46 2.63 2.27 | 97.5 | 93 | total = 2048.00M  used = 758.94M  free = 1289.06M  (encrypted) | none | AC | OrbStack Helper 36%, WallpaperAerialsExtension 4%, WindowServer 3% |
| product-laya-A-r3 | before | 1.46 2.63 2.27 | 98.0 | 93 | total = 2048.00M  used = 758.94M  free = 1289.06M  (encrypted) | none | AC | WallpaperAerialsExtension 6%, WindowServer 2%, VTDecoderXPCService 2% |
| product-laya-A-r3 | after | 1.22 1.88 2.02 | 97.6 | 93 | total = 2048.00M  used = 758.94M  free = 1289.06M  (encrypted) | none | AC | WallpaperAerialsExtension 9%, claude 3%, WindowServer 2% |
