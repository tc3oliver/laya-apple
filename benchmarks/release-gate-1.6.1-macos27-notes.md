# Release gate for 1.6.1 on macOS 27: notes

Tracker: [#139](https://github.com/tc3oliver/laya-apple/issues/139). This file covers the two
full runs of the 1.6.1 release gate, how they were made, and the cause of run 1's failure.

| run | candidate | result | generated reports |
|---|---|---|---|
| 1 | `main` at `486997f` | **FAIL**: 1 required step | [`.json`](release-gate-1.6.1-macos27.json), [`.md`](release-gate-1.6.1-macos27.md) |
| 2 | `main` at `5d8da47`, which includes the test fix [#156](https://github.com/tc3oliver/laya-apple/pull/156) | **PASS** | [`.json`](release-gate-1.6.1-macos27-run2.json), [`.md`](release-gate-1.6.1-macos27-run2.md) |

- Run 1 stays recorded unchanged as the failure record.
- Both runs used the same command, environment, machine, OS and artifact cache. They differ
  in three ways:
  - the candidate commit;
  - other load on the machine, which was present during run 2 (see "Revision and load");
  - how warm the `uv` cache was for the install matrix: 129 s in run 1 against 25 s in run 2.
- No gate, tolerance or threshold changed between them.

## Run 2 (PASS)

| step | status | duration (s) |
|---|---|---|
| ruff check | pass | 0.06 |
| ruff format --check | pass | 0.02 |
| derive_routing --check | pass | 0.07 |
| make_goldens --check | pass | 0.05 |
| derive_placement --check | pass | 0.04 |
| pytest (full, incl. integration/parity/ane) | pass: 1643 passed, 9 skipped | 1097.1 |
| clean install matrix (3.11/3.12/3.13 x base/ane) | pass | 25.08 |
| artifacts verify | pass | 15.21 |
| manifest schema | pass | 0.0 |
| no silent fallback tests | pass | 0.01 |
| docs versions | pass | 0.0 |
| soak | skipped (no `--soak`, as for 1.6.0) | 0.0 |

The gate's wall time was 1139 s (00:06:26 to 00:25:25 +0800, 2026-10-03).

### Revision and load

- **Recorded revision:** the report records `f41fb41`. That is `origin/main` at `5d8da47`
  plus only run 1's result files. `laya_apple/`, `tests/` and `scripts/` are identical to
  `5d8da47`.
  - `f41fb41` is a commit on the PR branch of
    [#155](https://github.com/tc3oliver/laya-apple/pull/155). It is reachable as
    `refs/pull/155/head`, but it will not be on `main` after the squash merge.
  - Run 2's files and these notes were added in later commits on that branch.
- **Machine state:** oMLX was stopped. `pmset -g therm` reported no thermal or performance
  warning before or after the run.
- **Load average:** 2.43 / 2.09 / 1.91 before the run, and 3.43 / 6.93 / 6.34 after it.
- **Other load:** other work ran on the machine for much of the run.
  - The 15-minute average went from 1.91 to 6.34. That rise cannot come from load that
    started only at the midpoint of a 19-minute run.
  - What that work was is not known.
- **Why it matters:** no benchmark timing is part of this gate's result. The gate does
  contain timing-dependent tests, though, as run 1's failure showed. Run 2 passed every
  step under this load.

### pytest count

Run 2 collected 1652 tests (1643 passed + 9 skipped), against 1622 in run 1 (1 failed +
1612 passed + 9 skipped).

- All 30 extra tests come from #154.
- #156 only rewrote an existing test. It adds no test function or parametrization.

## Run 1 (FAIL)

**FAIL.** 1 required step failed. The failure was a test that depended on `time.sleep`
precision (details below). It was not a product regression. It is fixed by
[#156](https://github.com/tc3oliver/laya-apple/pull/156) and run 2 passed.

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

## Setup (both runs)

- Candidates: run 1 used `main` at `486997f` (after #152 and #153), and run 2 used `main` at
  `5d8da47` (after #154 and #156).
- Version: the version was **not** bumped, so the reports say `laya-apple 1.6.0`, and the
  `docs versions` step checked 1.6.0. The release PR has to show version sync again after the
  bump.
- Command: `LAYA_APPLE_CACHE=<cache> HF_HUB_OFFLINE=1 uv run python scripts/release_gate.py --out benchmarks/release-gate-1.6.1-macos27`.
  Run 2 wrote to `-run2` instead. Both runs were full, not `--quick`. The environment was
  `uv sync --extra dev --extra ane --extra convert --extra serve`.
- The ANE artifacts were the ones built and parity-validated on this OS in
  `research/macos27-validation/`.
- The Hugging Face Hub was offline. The clean-install matrix resolved its dependencies
  from PyPI through `uv`, as in earlier gates.
- Machine: Apple M4 Max, macOS 27.0 (26A428), coremltools 9.0, Python 3.12.14.
  - oMLX was stopped for both runs.
  - `pmset -g therm` reported no thermal or performance warning before or after either run.
  - Run 1's load average was 2.37 / 1.81 / 1.57 before and 2.44 / 2.09 / 1.79 after, and no
    other heavy work ran during it.
  - Run 2's load is under "Revision and load" above.

### Methodology differences from the 1.6.0 gate

- **OS:** 1.6.0 was qualified on macOS 26.6.2 (25G83), and this run used macOS 27.0 (26A428).
  The machine and coremltools 9.0 are the same. The maintainer has no macOS 26 machine
  available for 1.6.1.
- **Routing profile:** this OS uses the shipped profile from #152
  (`Apple_M4_Max-macos27-coremltools9.0.json`) instead of `routing.json`.
- **Artifact cache:** the runs used the macOS 27 artifact cache, not the 26.6.2 one.
- **Manifest coverage:** `manifest schema` validated 10 manifests here, against 15 for 1.6.0.
  - The macOS 27 cache holds fewer artifacts. All of them are `bc1s-masked` B1 graphs, as in
    the 26.6.2 cache.
  - These buckets are in the 26.6.2 cache but not in the macOS 27 one:
    - `laya`: L256 and L512;
    - `laya-multilingual`: L512;
    - `laya-typed-decisions`: L256 and L512.
  - So `artifacts verify` and the ANE parity tests covered only these buckets:
    - L64, L96 and L128 for every model;
    - L256 for `laya-multilingual`.
  - The comparison was made by listing each cache's `*/*/*/manifest.json`, read-only.
- **Routing check:** `derive_routing --check` covers only `laya_apple/data/routing.json`. It
  does not check the shipped macOS 27 profile
  (`laya_apple/data/profiles/Apple_M4_Max-macos27-coremltools9.0.json`).
  - That file is checked by the tests in `tests/unit/test_profiles_unit.py`, which ran in the
    pytest step. For example, `test_shipped_profile_is_the_committed_calibration` and
    `test_shipped_profile_rederives_and_matches_the_registry`.
- **Install matrix timing:** the install matrix took 129 s in run 1 and 25 s in run 2,
  against 36 s for 1.6.0. Part of the `uv` cache was cold in run 1. Its result does not
  depend on timing.

## Run 1's failure

`tests/integration/test_no_silent_fallback.py::test_placement_probe_refuses_a_model_that_runs_like_the_cpu[10-10-False]`

### What the test does

- It replaces both models in `coreml_ane.probe_placement` with fakes that each
  `time.sleep(10 ms)`.
- It expects the probe to refuse the model. The probe compares the fastest of 5 runs of each
  model and refuses when `ane_ms / cpu_ms > PROBE_MAX_RATIO` (0.8).

### Why it fails here

- On this machine, `time.sleep(0.01)` often oversleeps by up to about 5 ms. The likely cause
  is macOS timer coalescing.
- A histogram of 1500 sleeps was measured about 1 minute after run 1 ended. No gate step or
  other known heavy work was running then. The load average was 1.68 / 1.93 / 1.75. The
  sleeps had these durations:

  | duration (ms) | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
  |---|---|---|---|---|---|---|---|
  | sleeps | 83 | 171 | 200 | 156 | 171 | 718 | 1 |

- So the fastest of 5 sleeps can be about 15 ms for one fake model and about 10 ms for the
  other. The ratio then falls to about 0.67, and the probe accepts the model.

### Measurements

- **Direct probe calls:** after run 1 ended, `probe_placement` was called 200 times with the test's two 10 ms
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

### Fix

[#156](https://github.com/tc3oliver/laya-apple/pull/156) makes the test deterministic.

- **How:** the fake models no longer sleep. Each `predict` advances a shared fake clock by
  exactly its duration, and the test installs that clock as `time.perf_counter`, the clock
  `_fastest_ms` uses.
- **Result:** the ratio is now exactly `loaded_ms / cpu_ms`.
- **What stayed the same:** every case keeps its expected outcome. `laya_apple/` code,
  `PROBE_MAX_RATIO` and every other threshold are unchanged.
- **Verification:** run 2 includes the fix and passed.
