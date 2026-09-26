# Serve decisions beside a local LLM: m4-max-r3

Criteria revision `r3` (see README.md).

Offered decision load 8 req/s (open loop: per-client seeded Poisson timetables, identical in every window); LLM `Qwen3.8-27B-oQ4e-mtp`. Medians over windows; P99 is the median of per-window P99s.

## LLM throughput

| cell | windows | LLM tok/s (window) | per-request decode tok/s | server TTFT s | vs LLM alone |
|---|---|---|---|---|---|
| LLM alone | 9 | 43.3 | 42.9 | 5.84 | window spread +2.0% |
| serve gpu + LLM | 4 | 41.2 | 41.9 | 6.09 | -4.7% |
| serve auto + LLM | 4 | 42.3 | 42.4 | 5.95 | -2.2% |

## Decisions

| cell | class | n | P50 ms | P99 ms | errors | hard mismatches | max prob err | near-tie flips | devices |
|---|---|---|---|---|---|---|---|---|---|
| gpu unloaded | short_1q | 1572 | 30.41 | 54.62 | 0 | 0 | 0.0000 | 0 | {'gpu': 1572} |
| gpu unloaded | mixed_3q | 360 | 46.37 | 72.17 | 0 | 0 | 0.0000 | 0 | {'gpu': 360} |
| gpu + LLM | short_1q | 1572 | 33.62 | 79.51 | 0 | 0 | 0.0000 | 0 | {'gpu': 1572} |
| gpu + LLM | mixed_3q | 360 | 54.06 | 97.01 | 0 | 0 | 0.0000 | 0 | {'gpu': 360} |
| auto unloaded | short_1q | 1572 | 29.08 | 43.42 | 0 | 0 | 0.0039 | 0 | {'ane': 1496, 'gpu': 76} |
| auto unloaded | mixed_3q | 360 | 48.53 | 86.62 | 0 | 0 | 0.0000 | 0 | {'gpu': 360} |
| auto + LLM | short_1q | 1572 | 25.15 | 41.72 | 0 | 0 | 0.0039 | 0 | {'ane': 1510, 'gpu': 62} |
| auto + LLM | mixed_3q | 360 | 54.69 | 104.85 | 0 | 0 | 0.0000 | 0 | {'gpu': 360} |

## Validity

| check | ok | detail |
|---|---|---|
| V1_open_loop_client_lag_p99_ms | yes | worst=3.3569596600000007 |
| V2_llm_saturated | yes | worst_busy_coverage=1.0, worst_generating_coverage=0.9874606118, llm_errors=0 |
| V3_llm_exclusive | yes | windows_failed=[], windows_unchecked=[] |
| V4_auto_short_policy_path | yes | worst_policy_share=1.0, spill_share_windows={'4': 0.043256997455470736, '5': 0.05089058524173028, '7': 0.04834605597964377, '8': 0.04071246819338423, '13': 0.05089058524173028, '14': 0.030534351145038167, '22': 0.043256997455470736, '23': 0.043256997455470736}, other_routes={} |
| V5_references_on_expected_devices | yes |  |

## Criteria (separate results, no overall verdict)

| result | criterion | value | limit | verdict |
|---|---|---|---|---|
| gpu_free | G1_llm_cost_auto | +2.2% | ≤ +5.0% | pass |
| gpu_free | G2_auto_below_gpu_only | +2.6% | > +2.0% (LLM-alone spread) | pass |
| decision_latency | L1_short_p99_loaded_abs_ms | 41.72 | ≤ 50 | pass |
| decision_latency | L2_short_p99_loaded_vs_unloaded | 0.96 | ≤ 1.5 | pass |
| correctness | C1_fp16_gate | prob 0.0039, act 0.0000, hard 0, flips 0 | ≤ 0.02, ≤ 0.02, 0, listed | pass |
| correctness | E1_errors | 0.00 | ≤ 0 | pass |

## Adaptive ANE execution (recorded, not gated)

A1, adaptive execution in use in every `auto` window: yes. It decides only how the run is described.

| cell | windows | episodes | breaker trips | ANE forwards sync | async | async share |
|---|---|---|---|---|---|---|
| auto/decisions | 4 | 64 | 0 | 1604 | 0 | 0.0% |
| auto/decisions_llm | 4 | 69 | 0 | 1618 | 0 | 0.0% |

| window | cell | state before → after | episodes | trips | sync | async |
|---|---|---|---|---|---|---|
| 4 | auto/decisions_llm | safe_sync → safe_sync | 16 | 0 | 403 | 0 |
| 5 | auto/decisions | safe_sync → safe_sync | 16 | 0 | 400 | 0 |
| 7 | auto/decisions | safe_sync → safe_sync | 16 | 0 | 401 | 0 |
| 8 | auto/decisions_llm | safe_sync → safe_sync | 17 | 0 | 404 | 0 |
| 13 | auto/decisions | safe_sync → safe_sync | 15 | 0 | 400 | 0 |
| 14 | auto/decisions_llm | safe_sync → safe_sync | 19 | 0 | 408 | 0 |
| 22 | auto/decisions_llm | safe_sync → safe_sync | 17 | 0 | 403 | 0 |
| 23 | auto/decisions | safe_sync → safe_sync | 17 | 0 | 403 | 0 |
