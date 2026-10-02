# Build and parity validation on macOS 27.0 (Apple M4 Max)

Issue: [#17](https://github.com/tc3oliver/laya-apple/issues/17). Every number below is in
`raw/`.

## Question

Does laya-apple's own build-and-parity pipeline still produce correct, fully Neural Engine
placed BC1S artifacts when only macOS changes?

Every validated artifact so far was built on macOS 26.6.2 with coremltools 9.0. The only
macOS 27 evidence was one community `--quick` run on an M4 Pro
([`hardware-results/apple-m4-pro-macos27/`](../../hardware-results/apple-m4-pro-macos27/summary.md),
#32). There, the SoC changed together with macOS. This run uses the release machine itself
(the same Apple M4 Max) after it was upgraded to macOS 27.0, with coremltools still 9.0.
Only macOS changed.

## Answer

**On this machine, macOS 27.0 changed nothing that the build, placement or parity gates
measure.** All 10 shipped model/bucket pairs passed:

- every one built;
- every one is placed 100% on the Neural Engine with 0 device transitions;
- every one passed the unchanged FP16 parity gate with 0 hard mismatches;
- every one passed the runtime placement probe and `artifacts verify`.

The compute plans have the same op counts as on 26.6.2, and the parity error is the same to
four decimals. `laya`'s one listed near-tie flip (`lang-fr` row 0) is the same as on 26.6.2.
Prior third-party work (laya-coreml, on an M3 Max with macOS 27.2) reported an
enumerated-shape package running on the GPU
([`../phase-0-feasibility/typed-decisions-ane.md`](../phase-0-feasibility/typed-decisions-ane.md)).
Nothing like that appeared here, but this run is a different macOS build (27.0) and tests only
the fixed-shape BC1S graphs on `CPU_AND_NE`, not enumerated shapes.

With these artifacts, the ANE test tiers run again on this machine. The ANE parity, GPU/ANE
agreement, export/import and prebuilt-fetch tests pass. 11 tests fail, all for one reason
that is not a runtime defect: they assume `auto` runs on a shipped routing profile
([Test tiers](#test-tiers)).

`platform_validated_for_auto_ane` is `false`, as designed. No shipped routing profile matches
macOS 27, so `device="auto"` stays on MLX (`platform_not_validated`). This run does not
change that, and it measured no routing, latency or throughput.

## Machine and software

From `raw/environment.txt` and `raw/info-*.json`:

| | |
|---|---|
| SoC | Apple M4 Max (`Mac16,9`), 64 GB |
| macOS | 27.0 (26A428) |
| Python | 3.12.14 |
| coremltools | 9.0 |
| MLX | 0.32.2 |
| PyTorch | 2.7.0 |
| NumPy | 2.1.3 |
| pyobjc-framework-CoreML | 12.2.2 |
| laya-apple | 1.6.0 at `adf28da92d6f` (the `main` commit of #146) |

The previous profile on this machine was macOS 26.6.2 (25G83), with the same coremltools.

The code that built the artifacts is the code at `adf28da`. Every manifest records
`git_revision: adf28da…` and a `code_sha256` for the four conversion files (`bc1s.py`,
`torch_reference.py`, `build.py`, `backends/coreml_ane.py`). All four hashes equal the
SHA-256 of those files at `adf28da` (`git show adf28da:<path> | shasum -a 256`). Against the
26.6.2 builds, only `coreml_ane.py` differs: #146 moved the placement probe's input into a
shared `probe_features()` function, with the same row, and changed nothing in conversion or
in the probe itself. The working tree was not recorded separately (no `git status` capture).

## Method

- Fresh cache: `LAYA_APPLE_CACHE` pointed at a new, empty directory (`<cache>` in `raw/`).
  The existing 26.6.2 artifacts were neither used nor modified.
- `HF_HUB_OFFLINE=1`. The pinned checkpoints were already in the Hugging Face cache, and
  the runtime verified their SHA-256 before use.
- Steps, in order:
  1. `laya-apple info laya-typed-decisions`, before any build (`raw/info-before-build.json`:
     all buckets `missing`).
  2. `laya-apple artifacts build <model>` for `laya-typed-decisions`, `laya` and
     `laya-multilingual`, all of their explicit ANE buckets (`raw/build-*.log`). The build
     records the layout check, the Core ML compute plan on `CPU_AND_NE` and the parity gate in
     each manifest (`raw/manifests/`).
  3. `laya-apple parity <model> --device ane` (`raw/parity-ane-*.json`).
  4. `laya-apple info <model>` (`raw/info-*.json`) and `laya-apple artifacts verify <model>`
     (`raw/verify-*.txt`; this reruns the file hash and the compute-plan check).
  5. Placement probe: `Laya.from_pretrained(<model>, device="ane")`, which runs the runtime
     placement probe (`backends.coreml_ane.probe_placement`) on every bucket. The
     per-bucket records are in `raw/probe-*.json`.
- The steps ran one at a time; no test or other laya-apple process overlapped them. A local
  LLM server on this machine (oMLX) stayed running and was not stopped; its load was not
  controlled. None of the gates above depends on load except the probe, which passed with a
  wide margin.
- Absolute paths in `raw/` are rewritten: `<checkout>` (the laya-apple checkout), `<cache>`
  (the fresh `LAYA_APPLE_CACHE`), `<tmp>` (pytest's temporary directory).

The probe ratio is a correctness gate: the loaded model must take at most 0.8× the time of
the same artifact on `CPU_ONLY`. Its times are five-run minimums from one load. They are
not a latency measurement and are not compared with anything here.

## Results

Compute plan, from `raw/manifests/` (`placement`); parity, from `raw/parity-ane-*.json`
and the build's own gate in the manifests; probe, from `raw/probe-*.json`.

| model | L | ops on ANE (cost share) | transitions | parity rows | prob max | hard | near-tie flips | probe ratio (ANE / CPU ms) |
|---|---:|---|---:|---:|---:|---:|---|---|
| laya-typed-decisions | 64 | 10594 / 10594 (100%) | 0 | 41 | 0.0046 | 0 | none | 0.509 (7.919 / 15.569) |
| laya-typed-decisions | 96 | 10594 / 10594 (100%) | 0 | 93 | 0.0046 | 0 | none | 0.366 (8.777 / 24.0) |
| laya-typed-decisions | 128 | 10594 / 10594 (100%) | 0 | 122 | 0.0077 | 0 | none | 0.364 (9.456 / 25.966) |
| laya | 64 | 10594 / 10594 (100%) | 0 | 41 | 0.0093 | 0 | none | 0.474 (7.395 / 15.586) |
| laya | 96 | 10594 / 10594 (100%) | 0 | 93 | 0.0126 | 0 | lang-fr row 0 (margin 0.0035) | 0.352 (8.449 / 24.017) |
| laya | 128 | 10594 / 10594 (100%) | 0 | 122 | 0.0126 | 0 | lang-fr row 0 (margin 0.0035) | 0.362 (9.359 / 25.857) |
| laya-multilingual | 64 | 6390 / 6390 (100%) | 0 | 47 | 0.0079 | 0 | none | 0.435 (3.297 / 7.587) |
| laya-multilingual | 96 | 6390 / 6390 (100%) | 0 | 97 | 0.0128 | 0 | none | 0.334 (3.752 / 11.218) |
| laya-multilingual | 128 | 6390 / 6390 (100%) | 0 | 121 | 0.0128 | 0 | none | 0.337 (4.109 / 12.206) |
| laya-multilingual | 256 | 6390 / 6390 (100%) | 0 | 146 | 0.0128 | 0 | none | 0.334 (7.833 / 23.476) |

- **Gate:** probability error ≤ 0.02, 0 hard mismatches, near-tie flips listed
  ([`docs/correctness.md`](../../docs/correctness.md)). Unchanged.
- **`laya-apple parity --device ane`** over each model's largest bucket, every golden row that
  fits: `laya-typed-decisions` 122 rows, prob max 0.0077; `laya` 122 rows, 0.0126, the
  `lang-fr` near-tie flip; `laya-multilingual` 146 rows, 0.0128. All passed, 0 hard
  mismatches, all finite, repeat-identical.
- **Layout check** (FP32 BC1S against the FP32 PyTorch reference, before conversion):
  max |Δlogit| 1.43e-06 to 5.96e-06 (`raw/build-*.log`), against a limit of 1e-3.
- **`artifacts verify`:** OK for all 10.
- **`laya-apple info`:** platform `Apple M4 Max / 27.0 / 26A428 / coremltools 9.0`,
  `platform_validated_for_auto_ane: false`, every bucket `validated`.

### Against macOS 26.6.2 on the same machine

[`benchmarks/prebuilt-artifacts-1.6.0.md`](../../benchmarks/prebuilt-artifacts-1.6.0.md)
records the same 10 pairs on this machine under macOS 26.6.2 (coremltools 9.0).

- **Op counts:** the same (10594 for the ModernBERT-large models, 6390 for
  `laya-multilingual`). 100% on the ANE and 0 transitions on both.
- **Parity prob max:** the same, to the four decimals that file reports, for all 10 pairs.
- **Near-tie flips:** the same (only `laya` `lang-fr` row 0, at L96 and L128).
- **Artifact tree hashes differ between the 26.6.2 and 27.0 builds.** This run built each pair
  once, so it did not separate the macOS version from build non-determinism. The 26.6.2 hash is `imported.artifact_sha256_check` in
  [`benchmarks/prebuilt-artifacts-1.6.0.json`](../../benchmarks/prebuilt-artifacts-1.6.0.json).
  That `.md` lists archive hashes, which are a different thing. The 27.0 hash is
  `integrity.artifact_sha256` in `raw/manifests/`.

| model | L | 26.6.2 artifact SHA-256 | 27.0 artifact SHA-256 |
|---|---:|---|---|
| laya-typed-decisions | 64 | `1273fcd304953d4e…` | `b078691b4df9b1be…` |
| laya-typed-decisions | 96 | `4145579a0a09dbad…` | `154967b904035858…` |
| laya-typed-decisions | 128 | `5db688bf78039fd5…` | `4792ff211eff86b6…` |
| laya | 64 | `f6263eb65898fd68…` | `ce3ee6df35fcf8be…` |
| laya | 96 | `1d0eb7da10157e75…` | `7ff8b2a5b67aa8a0…` |
| laya | 128 | `85ee23bf8edc58ca…` | `f47a0fe400365f1e…` |
| laya-multilingual | 64 | `5501b83508f5125c…` | `318726186f424249…` |
| laya-multilingual | 96 | `8fcc0f79e982248f…` | `afec4677072df621…` |
| laya-multilingual | 128 | `62c1f978aec16056…` | `7d06f1db9ea86262…` |
| laya-multilingual | 256 | `044e8e748edca85a…` | `77440a63e1228ed2…` |

## Test tiers

The tiers that need built ANE artifacts could not run on macOS 27 before: the cached
artifacts carry the 26.6.2 platform profile, and the runtime refuses them on another
profile by design. With the artifacts above, using the same cache and `HF_HUB_OFFLINE=1`:

- **`test_prebuilt_fetch.py` and `test_export_import.py`** (`-m "integration or parity or
  ane"`): **4 passed** (`raw/pytest-prebuilt-export.txt`). Export, import, the parity gate
  and the placement probe work end to end on macOS 27 for one artifact, `laya-multilingual`
  L64. The fetch test does not touch the network: it monkeypatches `prebuilt._download` to
  serve a local stand-in repository built from that artifact. The published repository has no
  macOS 27 archive set, so a real `laya-apple artifacts fetch` on macOS 27 still raises
  `ArtifactMissingError`, as `docs/support-matrix.md` says.
- **Every `integration`, `parity` and `ane` test, without `stress`:** **229 passed, 11 failed,
  19 skipped** (`raw/pytest-integration-parity-ane.txt`). Five test cases were left out on
  purpose, because they run benchmark code paths: the three model parametrizations of
  `tests/integration/test_benchmark.py`, and `test_switchyard.py`'s two `test_smoke_*`
  tests. Switchyard's oracle gates (GPU and ANE) ran and passed.

| file | passed | failed | skipped |
|---|---:|---:|---:|
| `parity/test_ane_parity.py` | 3 | 0 | 0 |
| `parity/test_ane_async_parity.py` | 2 | 0 | 1 |
| `parity/test_mlx_parity.py` | 6 | 0 | 0 |
| `integration/test_gpu_ane_agreement.py` | 3 | 0 | 0 |
| `integration/test_export_import.py` | 3 | 0 | 0 |
| `integration/test_prebuilt_fetch.py` | 1 | 0 | 0 |
| `integration/test_cli.py` | 9 | 0 | 0 |
| `integration/test_predict.py` | 25 | 0 | 0 |
| `integration/test_switchyard.py` (smoke tests excluded) | 3 | 0 | 0 |
| `integration/test_routing_e2e.py` | 8 | 4 | 3 |
| `integration/test_no_silent_fallback.py` | 15 | 4 | 0 |
| `integration/test_workers.py` | 12 | 2 | 3 |
| `integration/test_profiles.py` | 8 | 1 | 0 |
| `integration/test_request_trace_workers.py` | 0 | 0 | 8 |
| other files (MLX, serve, shortlist, option order, token cache, `unit/` cases the marker expression selects) | 131 | 0 | 4 |

**All 11 failures have one cause, and it is not a defect in the runtime.** The tests assume
the machine runs a shipped routing profile, so that `device="auto"` loads the ANE. On macOS
27 no shipped profile matches. `auto` therefore stays on MLX with `platform_not_validated`
before it looks at artifacts, workers or probes. This is the documented behaviour
([`docs/no-silent-fallback.md`](../../docs/no-silent-fallback.md), row 1: `auto` goes to MLX
on an unvalidated platform, decided before the request runs and recorded). The failures:

- `test_routing_e2e.py::test_auto_with_empty_artifact_cache_routes_to_gpu_with_artifact_unavailable`
  (3 models): expected `ane_artifact_unavailable`, got `platform_not_validated`.
- `test_routing_e2e.py::test_artifact_revision_mismatch_under_explicit_ane_raises[laya-multilingual]`:
  the explicit `device="ane"` half passed (`ArtifactRevisionError` raised). The `auto` half
  expected a `RuntimeWarning` for the rejected bucket, but `auto` never loads the ANE here,
  so there is no bucket to reject.
- `test_no_silent_fallback.py`:
  - `test_auto_with_a_rejected_bucket_warns_and_records_the_reason` and
    `test_placement_probe_failure_drops_the_bucket_under_auto_and_raises_under_ane`: the
    expected "L64 rejected" / "L96 rejected" warning is not emitted, for the same reason.
    In the second test, the explicit `device="ane"` half passed
    (`ComputeUnitMismatchError` raised).
  - `test_dead_ane_worker_under_auto_is_recorded_and_warned_once`: the first request was
    expected on `ane`, and it ran on `gpu`.
  - `test_background_startup_failure_is_warned_and_recorded`: expected
    `ane_runtime_unavailable`, got `platform_not_validated`.
- `test_workers.py`:
  - `test_dead_gpu_worker_fails_loudly_and_never_falls_back`: `KeyError: 'ane'`. `auto` with
    workers starts no ANE worker on an unvalidated profile.
  - `test_background_ane_startup_serves_mlx_first_then_ane`: expected `ane_starting`, got
    `platform_not_validated`.
- `test_profiles.py::test_local_profile_ignored_on_a_shipped_profile`: the test writes a local
  profile and expects the shipped one to win. On macOS 27 there is no shipped profile, so the
  local one is used (`routing_profile == "local:…/Apple_M4_Max-macos27-coremltools9.0.json"`),
  as designed.

The 14 "no validated ANE artifacts" skips in `test_routing_e2e.py`, `test_workers.py` and
`test_request_trace_workers.py` have the same cause. The artifacts are present and
validated, but those fixtures check the buckets `auto` loaded, and `auto` loaded none. The
remaining skips are fixture scope (`test_option_order.py`; `laya-multilingual` in
`test_ane_async_parity.py`).

- **A shipped macOS 27 routing profile would make all 11 pass.**
- **A local calibrated profile** from `laya-apple calibrate` would probably make 10 of them
  pass. This is untested: calibration is a timing measurement and was out of scope here.
- **It can never make `test_profiles.py::test_local_profile_ignored_on_a_shipped_profile`
  pass**, because that test asserts `routing_profile == "shipped"`.
- **A possible follow-up:** these tests could skip, rather than fail, when
  `platform_validated_for_auto_ane` is false. This PR leaves that for its own change.

No test, tolerance or gate was changed.

## Limits

- **One machine, one macOS build** (27.0, 26A428). Other SoCs, and other macOS 27 builds,
  are unmeasured.
- **coremltools did not change** (9.0). The coremltools half of #17 is still open.
- **Correctness and placement only.** No latency, throughput, routing or calibration was
  measured; the shipped routing table is unchanged, and `auto` stays on MLX on this profile.
- **Explicit buckets only.** The longer validated ANE lengths (up to 1024; `laya` up to 512)
  and the research-only windowed-attention graph were not built.
- **Fixed-shape BC1S only.** The placement change in the prior third-party work (M3 Max,
  macOS 27.2) was for enumerated shapes, which laya-apple does not ship. This run does not
  test them, or macOS 27.2.
- **No macOS 27 prebuilt archives.** These artifacts were not published, and `artifacts
  fetch` on macOS 27 still finds none.
