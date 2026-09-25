# Serve decisions beside a local LLM: m4-max-r2

Criteria revision `r2` (see README.md).

Offered decision load 8 req/s (open loop: per-client seeded Poisson timetables, identical in every window); LLM `Qwen3.8-27B-oQ4e-mtp`. Medians over windows; P99 is the median of per-window P99s.

## LLM throughput

| cell | windows | LLM tok/s (window) | per-request decode tok/s | server TTFT s | vs LLM alone |
|---|---|---|---|---|---|
| LLM alone | 9 | 42.2 | 45.5 | 6.48 | window spread +1.3% |
| serve gpu + LLM | 4 | 40.0 | 43.0 | 6.73 | -5.3% |
| serve auto + LLM | 4 | 40.5 | 44.3 | 6.70 | -4.0% |

## Decisions

| cell | class | n | P50 ms | P99 ms | errors | hard mismatches | max prob err | near-tie flips | devices |
|---|---|---|---|---|---|---|---|---|---|
| gpu unloaded | short_1q | 1572 | 30.19 | 55.85 | 0 | 0 | 0.0000 | 0 | {'gpu': 1572} |
| gpu unloaded | mixed_3q | 360 | 46.40 | 79.39 | 0 | 0 | 0.0000 | 0 | {'gpu': 360} |
| gpu + LLM | short_1q | 1572 | 33.66 | 122.31 | 0 | 0 | 0.0000 | 0 | {'gpu': 1572} |
| gpu + LLM | mixed_3q | 360 | 53.90 | 153.11 | 0 | 0 | 0.0000 | 0 | {'gpu': 360} |
| auto unloaded | short_1q | 1572 | 28.82 | 43.19 | 0 | 0 | 0.0039 | 0 | {'ane': 1505, 'gpu': 67} |
| auto unloaded | mixed_3q | 360 | 48.37 | 84.07 | 0 | 0 | 0.0000 | 0 | {'gpu': 360} |
| auto + LLM | short_1q | 1572 | 24.23 | 47.18 | 0 | 0 | 0.0039 | 0 | {'ane': 1509, 'gpu': 63} |
| auto + LLM | mixed_3q | 360 | 54.80 | 117.08 | 0 | 0 | 0.0000 | 0 | {'gpu': 360} |

## Validity

| check | ok | detail |
|---|---|---|
| V1_open_loop_client_lag_p99_ms | yes | worst=3.1390281 |
| V2_llm_saturated | yes | worst_busy_coverage=1.0, worst_generating_coverage=0.91289696945, llm_errors=0 |
| V3_llm_exclusive | yes | windows_failed=[], windows_unchecked=[] |
| V4_auto_short_policy_path | yes | worst_policy_share=1.0, spill_share_windows={'4': 0.04071246819338423, '5': 0.043256997455470736, '7': 0.03816793893129771, '8': 0.035623409669211195, '13': 0.043256997455470736, '14': 0.043256997455470736, '22': 0.04071246819338423, '23': 0.04580152671755725}, other_routes={} |
| V5_references_on_expected_devices | yes |  |

## Criteria (separate results, no overall verdict)

| result | criterion | value | limit | verdict |
|---|---|---|---|---|
| gpu_free | G1_llm_cost_auto | +4.0% | ≤ +5.0% | pass |
| gpu_free | G2_auto_below_gpu_only | +1.3% | > +1.3% (LLM-alone spread) | pass |
| decision_latency | L1_short_p99_loaded_abs_ms | 47.18 | ≤ 50 | pass |
| decision_latency | L2_short_p99_loaded_vs_unloaded | 1.09 | ≤ 1.5 | pass |
| correctness | C1_fp16_gate | prob 0.0039, act 0.0000, hard 0, flips 0 | ≤ 0.02, ≤ 0.02, 0, listed | pass |
| correctness | E1_errors | 0.00 | ≤ 0 | pass |
