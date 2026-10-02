# Release gate — laya-apple 1.6.0

- generated: 2026-10-02T23:42:55+0800
- git revision: `486997f3e174f9301e202bd8bd95300c0e768d3a`
- python: 3.12.14 (main, Sep  1 2026, 14:09:38) [Clang 22.1.3 ]
- platform: {"soc": "Apple M4 Max", "macos": "27.0", "macos_build": "26A428", "coremltools": "9.0"}
- quick mode: False
- soak seconds: 0
- **result: FAIL**

| step | status | duration (s) | required |
|---|---|---|---|
| ruff check | pass | 0.26 | True |
| ruff format --check | pass | 0.03 | True |
| derive_routing --check | pass | 0.07 | True |
| make_goldens --check | pass | 0.06 | True |
| derive_placement --check | pass | 0.04 | True |
| pytest | fail | 1073.93 | True |
| clean install matrix | pass | 129.23 | True |
| artifacts verify | pass | 13.53 | True |
| manifest schema | pass | 0.0 | True |
| no silent fallback tests | pass | 0.01 | True |
| docs versions | pass | 0.0 | True |
| soak | skipped | 0.0 | False |

## pytest — fail

```
tests/integration/test_mlx_compile_checkpoints.py:27: RuntimeWarning: laya-apple: checkpoint temperatures that are invalid or outside [0.5, 5.0] are replaced (choice:11+=0.10058280825614929 -> 0.5), as upstream does; treat confidence from those entries as uncalibrated
    summary = evaluate(

tests/integration/test_mlx_length_buckets_checkpoints.py::test_bucketed_batches_pass_parity_and_keep_decisions[laya-float16]
tests/integration/test_mlx_length_buckets_checkpoints.py::test_bucketed_batches_pass_parity_and_keep_decisions[laya-float32]
tests/integration/test_mlx_length_buckets_checkpoints.py::test_bucketed_batches_pass_parity_and_keep_decisions[laya-typed-decisions-float16]
tests/integration/test_mlx_length_buckets_checkpoints.py::test_bucketed_batches_pass_parity_and_keep_decisions[laya-typed-decisions-float32]
  ./tests/integration/test_mlx_length_buckets_checkpoints.py:40: RuntimeWarning: laya-apple: checkpoint temperatures that are invalid or outside [0.5, 5.0] are replaced (choice:11+=0.10058280825614929 -> 0.5), as upstream does; treat confidence from those entries as uncalibrated
    summary = evaluate(

tests/parity/test_ane_async_parity.py::test_async_path_passes_parity_and_equals_coremltools[laya]
tests/parity/test_ane_async_parity.py::test_async_path_passes_parity_and_equals_coremltools[laya-typed-decisions]
  ./tests/parity/test_ane_async_parity.py:67: RuntimeWarning: laya-apple: checkpoint temperatures that are invalid or outside [0.5, 5.0] are replaced (choice:11+=0.10058280825614929 -> 0.5), as upstream does; treat confidence from those entries as uncalibrated
    summary = evaluate(

tests/parity/test_ane_parity.py::test_ane_passes_parity_gate_for_each_present_bucket[laya]
tests/parity/test_ane_parity.py::test_ane_passes_parity_gate_for_each_present_bucket[laya-typed-decisions]
  ./tests/parity/test_ane_parity.py:24: RuntimeWarning: laya-apple: checkpoint temperatures that are invalid or outside [0.5, 5.0] are replaced (choice:11+=0.10058280825614929 -> 0.5), as upstream does; treat confidence from those entries as uncalibrated
    summary = evaluate(

tests/parity/test_mlx_parity.py::test_mlx_passes_parity_gate[laya-float16]
tests/parity/test_mlx_parity.py::test_mlx_passes_parity_gate[laya-float32]
tests/parity/test_mlx_parity.py::test_mlx_passes_parity_gate[laya-typed-decisions-float16]
tests/parity/test_mlx_parity.py::test_mlx_passes_parity_gate[laya-typed-decisions-float32]
  ./tests/parity/test_mlx_parity.py:15: RuntimeWarning: laya-apple: checkpoint temperatures that are invalid or outside [0.5, 5.0] are replaced (choice:11+=0.10058280825614929 -> 0.5), as upstream does; treat confidence from those entries as uncalibrated
    summary = evaluate(

tests/unit/test_prompt.py::test_answers_match_upstream_on_every_golden[laya]
tests/unit/test_prompt.py::test_answers_match_upstream_on_every_golden[laya-typed-decisions]
  ./.venv/lib/python3.12/site-packages/_pytest/fixtures.py:1005: RuntimeWarning: laya-apple: checkpoint temperatures that are invalid or outside [0.5, 5.0] are replaced (choice:11+=0.10058280825614929 -> 0.5), as upstream does; treat confidence from those entries as uncalibrated
    fixture_result = fixturefunc(**kwargs)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/integration/test_no_silent_fallback.py::test_placement_probe_refuses_a_model_that_runs_like_the_cpu[10-10-False]
1 failed, 1612 passed, 9 skipped, 76 warnings in 1072.73s (0:17:52)

```
