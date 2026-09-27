# intra-request-split: look `f1`

| gate | ratio, geometric mean [CI] | verdict |
|---|---|---|
| G1 8x512 split / laya-fast (window P50) | 0.770 [0.769, 0.771] (n=10) | PASS |
| G2 no_regression 8x128 split / gpu (window P50) | 0.639 [0.637, 0.641] (n=10) | PASS |
| G2 no_regression 8x512 split / gpu (window P50) | 0.672 [0.672, 0.672] (n=10) | PASS |
| G2 no_regression 1x512 split / gpu (window P50) | 1.002 [0.999, 1.004] (n=10) | PASS |
| G2 no_regression 32x64 split / gpu (window P50) | 0.661 [0.660, 0.662] (n=10) | PASS |
| G2 gain 8x512 split / gpu (window P50) | 0.672 [0.672, 0.672] (n=10) | PASS |
| G3 correctness | 1232 split answers; hard 0, flips 0, exceedances 0 (shared 0) | PASS |

Confidence 0.99. Decision: **CONTINUE**
