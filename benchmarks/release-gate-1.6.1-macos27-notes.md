# Release gate for 1.6.1 on macOS 27: notes

Tracker: [#139](https://github.com/tc3oliver/laya-apple/issues/139). The generated
reports are [`release-gate-1.6.1-macos27.json`](release-gate-1.6.1-macos27.json) and
[`release-gate-1.6.1-macos27.md`](release-gate-1.6.1-macos27.md). This file records how the
run was made and what its one failure is.

## Result

**FAIL.** 1 required step failed. The failure is a test that depends on `time.sleep`
precision (details below). It is not a product regression. The gate is unchanged and the run
is recorded as it came out. 1.6.1 is not qualified by this run.

| step | status | duration (s) |
|---|---|---|
| ruff check | pass | 0.26 |
| ruff format --check | pass | 0.03 |
| derive_routing --check | pass | 0.07 |
| make_goldens --check | pass | 0.06 |
| derive_placement --check | pass | 0.04 |
| pytest (full, incl. integration/parity/ane) | **fail**: 1 failed, 1612 passed, 9 skipped | 1073.93 |
| clean install matrix (3.11/3.12/3.13 x base/ane) | pass | 129.23 |
| artifacts verify | pass | 13.53 |
| manifest schema | pass | 0.0 |
| no silent fallback tests | pass | 0.01 |
| docs versions | pass | 0.0 |
| soak | skipped (no `--soak`, as for 1.6.0) | 0.0 |

The gate's wall time was 1219 s (23:22:37 to 23:42:56 +0800, 2026-10-02).

## What was run

- Candidate: `main` at `486997f` (after #152 and #153). The version was **not** bumped, so
  the reports say `laya-apple 1.6.0`, and the `docs versions` step checked 1.6.0. The
  release PR has to show version sync again after the bump.
- Command: `LAYA_APPLE_CACHE=<cache> HF_HUB_OFFLINE=1 uv run python scripts/release_gate.py --out benchmarks/release-gate-1.6.1-macos27`.
  The run was full, not `--quick`. The environment was
  `uv sync --extra dev --extra ane --extra convert --extra serve`.
- The ANE artifacts were the ones built and parity-validated on this OS in
  `research/macos27-validation/`.
- The Hugging Face Hub was offline. The clean-install matrix resolved its dependencies
  from PyPI through `uv`, as in earlier gates.
- Machine: Apple M4 Max, macOS 27.0 (26A428), coremltools 9.0, Python 3.12.14.
  - oMLX was stopped for the run. No other heavy work ran during it.
  - `pmset -g therm` reported no thermal or performance warning before or after the run.
  - The load average was 2.37 / 1.81 / 1.57 before the run and 2.44 / 2.09 / 1.79 after.

### Methodology differences from the 1.6.0 gate

- **OS:** 1.6.0 was qualified on macOS 26.6.2 (25G83), and this run used macOS 27.0 (26A428).
  The machine and coremltools 9.0 are the same. The maintainer has no macOS 26 machine
  available for 1.6.1.
- **Routing profile:** this OS uses the shipped profile from #152
  (`Apple_M4_Max-macos27-coremltools9.0.json`) instead of `routing.json`.
- **Artifact cache:** the run used the macOS 27 artifact cache, not the 26.6.2 one.
- **Install matrix timing:** the install matrix took 129 s, against 36 s for 1.6.0. Part of
  the `uv` cache was cold. Its result does not depend on timing.

## The failure

`tests/integration/test_no_silent_fallback.py::test_placement_probe_refuses_a_model_that_runs_like_the_cpu[10-10-False]`

### What the test does

- It replaces both models in `coreml_ane.probe_placement` with fakes that each
  `time.sleep(10 ms)`.
- It expects the probe to refuse the model. The probe compares the fastest of 5 runs of each
  model and refuses when `ane_ms / cpu_ms > PROBE_MAX_RATIO` (0.8).

### Why it fails here

- On this machine, `time.sleep(0.01)` often oversleeps by up to about 5 ms. The likely cause
  is macOS timer coalescing.
- A histogram of 1500 sleeps, measured right after the gate, had these durations:

  | duration (ms) | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
  |---|---|---|---|---|---|---|---|
  | sleeps | 83 | 171 | 200 | 156 | 171 | 718 | 1 |

- So the fastest of 5 sleeps can be about 15 ms for one fake model and about 10 ms for the
  other. The ratio then falls to about 0.67, and the probe accepts the model.

### Measurements

- **Direct probe calls:** `probe_placement` was called 200 times with the test's two 10 ms
  fakes. It accepted the model 16 times out of 200 (8%). Examples:
  - `ane_ms` 10.13 against `cpu_ms` 14.95, ratio 0.678;
  - `ane_ms` 11.13 against `cpu_ms` 15.01, ratio 0.742.
- **Isolated pytest runs:** run alone, the test's three cases passed in 10 out of 10 runs.

### What this means for the release

- **Product code:** the shipped probe compares a real ANE model with the same artifact on
  `CPU_ONLY`. It does not rely on `sleep` precision.
- **Real artifacts:** `test_real_artifacts_pass_the_probe_and_report_it` passed in the same
  run, and all `parity` and `ane` tests passed.
- **Gate:** no gate, tolerance or threshold was changed.

### Unknowns

- It is not known whether macOS 26.6.2 has the same sleep jitter. The 1.6.0 gate passed this
  test.
- No macOS 26 machine is available to compare.

### Before 1.6.1 can qualify

1. Make the test deterministic in its own `fix` PR. For example, fake the timings instead of
   sleeping.
2. Run the gate again.
