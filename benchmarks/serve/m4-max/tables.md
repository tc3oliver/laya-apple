# Serve decisions beside a local LLM: m4-max

Offered decision load 8 req/s (open loop: per-client seeded Poisson timetables, identical in every window); LLM `Qwen3.8-27B-oQ4e-mtp`. Medians over windows; P99 is the median of per-window P99s.

## LLM throughput

| cell | windows | LLM tok/s (window) | per-request decode tok/s | server TTFT s | vs LLM alone |
|---|---|---|---|---|---|
| LLM alone | 9 | 42.4 | 42.5 | 6.00 | window spread +0.9% |
| serve gpu + LLM | 4 | 39.8 | 43.1 | 6.72 | -6.1% |
| serve auto + LLM | 4 | 40.5 | 44.0 | 6.67 | -4.5% |

## Decisions

| cell | class | n | P50 ms | P99 ms | errors | hard mismatches | max prob err | near-tie flips | devices |
|---|---|---|---|---|---|---|---|---|---|
| gpu unloaded | short_1q | 1572 | 30.53 | 56.39 | 0 | 0 | 0.0000 | 0 | {'gpu': 1572} |
| gpu unloaded | mixed_3q | 360 | 46.70 | 75.98 | 0 | 0 | 0.0000 | 0 | {'gpu': 360} |
| gpu + LLM | short_1q | 1572 | 34.00 | 144.49 | 0 | 0 | 0.0000 | 0 | {'gpu': 1572} |
| gpu + LLM | mixed_3q | 360 | 54.55 | 160.01 | 0 | 0 | 0.0000 | 0 | {'gpu': 360} |
| auto unloaded | short_1q | 1572 | 29.02 | 44.23 | 0 | 0 | 0.0039 | 0 | {'ane': 1505, 'gpu': 67} |
| auto unloaded | mixed_3q | 360 | 48.70 | 84.08 | 0 | 0 | 0.0000 | 0 | {'gpu': 360} |
| auto + LLM | short_1q | 1572 | 26.56 | 44.21 | 0 | 0 | 0.0039 | 0 | {'ane': 1503, 'gpu': 69} |
| auto + LLM | mixed_3q | 360 | 55.44 | 118.67 | 0 | 0 | 0.0000 | 0 | {'gpu': 360} |

## Validity

| check | ok | detail |
|---|---|---|
| V1_open_loop_client_lag_p99_ms | yes | worst=3.126446380000001 |
| V2_llm_saturated | **no** | worst_coverage=0.8954767764, llm_errors=0 |
| V3_llm_exclusive | yes | windows_failed=[], windows_unchecked=[] |
| V4_auto_short_on_ane | **no** | worst_share=0.9516539440203562 |
| V5_references_on_expected_devices | yes |  |

## Criteria (separate results, no overall verdict)

| result | criterion | value | limit | verdict |
|---|---|---|---|---|
| gpu_free | G1_llm_cost_auto | +4.5% | ≤ +5.0% | **invalid** |
| gpu_free | G2_auto_below_gpu_only | +1.6% | > +0.9% (LLM-alone spread) | **invalid** |
| decision_latency | L1_short_p99_loaded_abs_ms | 44.21 | ≤ 50 | **invalid** |
| decision_latency | L2_short_p99_loaded_vs_unloaded | 1.00 | ≤ 1.5 | **invalid** |
| correctness | C1_fp16_gate | prob 0.0039, act 0.0000, hard 0, flips 0 | ≤ 0.02, ≤ 0.02, 0, listed | **invalid** |
| correctness | E1_errors | 0.00 | ≤ 0 | **invalid** |
