# Async prebound predict screen (laya, L128 / L512, R1's full protocol)

Phase 0 (`raw/phase0.json`): 1_callable PASS, 2_correctness PASS, 3_backings PASS, 4_callbacks PASS, 5_repeat PASS, 6_threading PASS, 7_submit_returns_early PASS, 8_gil PASS

PB-ASYNC-STAMPED native completion → Python callback entry (semantics check, not performance evidence): P50 0.009 ms, P99 0.023 ms, n = 200

Outcome: async does not solve the slow state: resume #89

Next: none

Crashed runs and re-runs (`raw/failed/`): none

Runs not used by the rule: none


## Hetero windows

Slow: short P99 ≥ 13.0 ms. Thread CPU: ms per window; ANE and GPU: ms per forward. Stages: P50 ms; completion is callback entry − submit (PB-ASYNC) or the native predict (PB-SYNC); handoff is wake − callback entry (PB-ASYNC) or the GIL re-acquire (PB-SYNC).

| run | cycle | short P99 | slow | aggregate req/s | GPU return P50 | mismatches | client-short CPU | client-long CPU | callback CPU | ANE CPU/fwd | GPU CPU/fwd | features | pre | submit | completion | handoff | post | tail | callbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laya-A-r1 | 0 | 12.91 | normal | 122.1 | 4.360 | 0 | 296 | 225 | – | 0.603 | 1.827 | 0.157 | – | – | – | – | – | 0.060 | – |
| laya-A-r1 | 1 | 11.13 | normal | 122.7 | 4.285 | 0 | 284 | 216 | – | 0.585 | 1.753 | 0.157 | – | – | – | – | – | 0.060 | – |
| laya-A-r1 | 2 | 11.01 | normal | 122.6 | 4.272 | 0 | 286 | 218 | – | 0.587 | 1.758 | 0.157 | – | – | – | – | – | 0.060 | – |
| laya-PB-ASYNC-r1 | 0 | 16.96 | **slow** | 123.1 | 0.035 | 0 | 480 | 439 | 204 | 0.449 | 2.895 | 0.158 | 0.015 | 0.009 | 9.439 | 0.023 | 0.006 | 0.061 | 1921 (0 errors) |
| laya-PB-ASYNC-r1 | 1 | 12.77 | normal | 127.6 | 0.035 | 0 | 327 | 283 | 154 | 0.304 | 1.948 | 0.161 | 0.016 | 0.009 | 9.417 | 0.022 | 0.005 | 0.060 | 1999 (0 errors) |
| laya-PB-ASYNC-r1 | 2 | 16.26 | **slow** | 125.6 | 0.035 | 0 | 400 | 358 | 178 | 0.369 | 2.384 | 0.157 | 0.015 | 0.009 | 9.416 | 0.022 | 0.006 | 0.060 | 1964 (0 errors) |
| laya-PB-SYNC-r1 | 0 | 16.87 | **slow** | 102.1 | 0.247 | 0 | 1338 | 1294 | – | 1.809 | 8.916 | 0.795 | 0.094 | – | 9.737 | 0.006 | 0.039 | 0.370 | – |
| laya-PB-SYNC-r1 | 1 | 16.67 | **slow** | 113.9 | 0.184 | 0 | 889 | 832 | – | 1.080 | 5.474 | 0.535 | 0.068 | – | 9.635 | 0.004 | 0.027 | 0.264 | – |
| laya-PB-SYNC-r1 | 2 | 16.77 | **slow** | 102.4 | 0.246 | 0 | 1335 | 1285 | – | 1.777 | 8.866 | 0.792 | 0.093 | – | 9.729 | 0.006 | 0.038 | 0.363 | – |

## Runs

| run | slow windows | GPU return P50 ms | mismatches | routing failures | predicts with binding stamps | submits / callbacks / errors / anomalies | callback threads | backings | wall s |
|---|---|---|---|---|---|---|---|---|---|
| laya-A-r1 | 0 of 3 | 4.302 | 0 | 0 | 0 of 5874 | – | – | – | 277 |
| laya-PB-ASYNC-r1 | 2 of 3 | 0.035 | 0 | 0 | 5887 of 5887 | 11929 / 11929 / 0 / 0 | 3 | backed | 278 |
| laya-PB-SYNC-r1 | 3 of 3 | 0.242 | 0 | 0 | 4915 of 4915 | – | – | backed | 278 |

## Cells, pooled over their runs

| cell | rounds | slow windows | mismatches | GPU return P50 ms | features | pre | submit | completion | handoff | post | tail | ANE CPU/fwd | GPU CPU/fwd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 0 of 3 | 0 | 4.302 | 0.157 | – | – | – | – | – | 0.060 | 0.592 | 1.779 |
| PB-SYNC | 1 | 3 of 3 | 0 | 0.242 | 0.777 | 0.089 | – | 9.715 | 0.006 | 0.037 | 0.346 | 1.536 | 7.695 |
| PB-ASYNC | 1 | 2 of 3 | 0 | 0.035 | 0.159 | 0.015 | 0.009 | 9.422 | 0.022 | 0.006 | 0.060 | 0.373 | 2.405 |
