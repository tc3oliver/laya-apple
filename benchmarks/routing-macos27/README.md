# Routing calibration on macOS 27.0 (Apple M4 Max)

Issues: [#17](https://github.com/tc3oliver/laya-apple/issues/17),
[#151](https://github.com/tc3oliver/laya-apple/issues/151). Every number below is in `raw/`.

## Question

The release machine (Apple M4 Max, 64 GB) was upgraded from macOS 26.6.2 to 27.0, with
coremltools still 9.0. [`research/macos27-validation/`](../../research/macos27-validation/README.md)
built and parity-validated the ANE artifacts of all three models there, but measured no
routing, so `device="auto"` stayed on MLX (`platform_not_validated`).

Does the unchanged routing rule, applied to measurements on this profile, give `auto` ANE
buckets, and do they hold across two passes?

## Answer

**Yes. Both passes give the same auto ANE buckets as the shipped 26.6.2 table for every
model:** 64, 96 and 128 for `laya`, `laya-multilingual` and `laya-typed-decisions`.
`laya-multilingual` L256 stays explicit-only, as on 26.6.2. The criterion written down before
the full run ([`PLAN.md`](PLAN.md)) is met, so pass 1's profile ships, unedited, as
`laya_apple/data/profiles/Apple_M4_Max-macos27-coremltools9.0.json`. The plan was written
before pass 1 but committed only with the results, so the history does not attest that order.

The narrowest margin is L96 of the two ModernBERT-large models: the ANE at 96 tokens beats MLX
at 64 tokens by 0.21 to 0.27 ms (about 2–3%). On 26.6.2 the same comparison had a 0.37–0.41 ms
margin. That bucket is the one to watch if this profile is ever re-measured.

## Machine and software

| | |
|---|---|
| SoC | Apple M4 Max (`Mac16,9`), 64 GB |
| macOS | 27.0 (26A428) |
| coremltools | 9.0 |
| MLX | 0.32.2 |
| laya-apple | 1.6.0 at `9be538239c7c` (`main`, after #150) |
| ANE artifacts | the ones built and parity-validated in `research/macos27-validation/` (hashes in each pass's profile, `artifacts`) |
| Power | AC, no thermal or performance warning recorded before or after any run (`raw/*-conditions-*.txt`) |

The local LLM server on this machine (oMLX) was stopped for the whole window. No test, build
or other laya-apple process ran during a measurement. Load averages (1 min) before/after:
screen 1.62/0.84, pass 1 1.30/1.42, pass 2 1.51/1.30.

## Method

All runs: `LAYA_APPLE_CACHE` set to the macOS 27 cache, `HF_HUB_OFFLINE=1`,
`laya-apple --offline calibrate ...`. `calibrate` loads each model on MLX and on the ANE,
measures the backend forward P50 (synchronised) for every cell, and applies the unchanged rule
in `laya_apple/derivation.py`, the one behind the shipped table
([`docs/support-matrix.md`](../../docs/support-matrix.md#how-the-auto-ane-buckets-were-derived)).

| Run | Command | Models, in order | Time |
|---|---|---|---:|
| Screen | `calibrate laya-typed-decisions` (defaults: warmup 5, iters 30) | typed | 75 s |
| Pass 1 | `calibrate laya-typed-decisions laya laya-multilingual --warmup 10 --iters 300` | typed, laya, multilingual | 1104 s |
| Pass 2 | `calibrate laya-multilingual laya laya-typed-decisions --warmup 10 --iters 300` | multilingual, laya, typed | 1101 s |

Measurement time in total: 2280 s (38 min).

Pass 1 and the screen ran on a clean checkout of `9be5382`. Pass 2 ran with this PR's change
to routing-profile selection (`profiles.py`, `model.py`) already in the working tree. That
change decides which profile `auto` uses and is not on the measured path: `calibrate` loads
explicit `device="gpu"` and `device="ane"` models and times their backends directly.

### Differences from the shipped 26.6.2 table

The shipped table (`laya_apple/data/routing.json`) comes from the Phase -1 benchmark harness
([`research/phase-0-feasibility/methodology.md`](../../research/phase-0-feasibility/methodology.md)),
not from `calibrate`. The rule is the same; the measurement is not:

| | Phase -1 (26.6.2, shipped) | This run (27.0) |
|---|---|---|
| Harness | `research/phase-0-feasibility` benchmark | `laya-apple calibrate` (the documented procedure for a profile) |
| Process | a fresh process per model × backend | one process per pass; each model's MLX and ANE instances loaded together, MLX cells measured first |
| Warmup | 10 calls | 10 calls |
| Samples per cell | time-budgeted, 50–300 (about 4 s); 300 for every routing-relevant short cell | fixed 300 |
| Order control | two passes, reversed backend order | two passes, reversed model order (backend order within a model is fixed by `calibrate`) |
| ANE service times | 1, 4 and 8 questions, up to 1024 tokens | 1 question, offered buckets only (what `calibrate` records) |

Numbers from the two methodologies are shown side by side below for orientation, not as a
measured macOS 26 → 27 change. Under these methods the ANE forwards read 0.03–0.28 ms
(1–4%) slower on 27.0 (pass 1) and the MLX forwards within 1%; this run does not separate macOS from
the methodology change.

## Results

ANE and MLX are forward P50 in ms. "MLX at" is the length the rule compares against (the
previous bucket; for the first bucket, the bucket itself). Parity is the build gate recorded in
each artifact's manifest (all passed, 0 hard mismatches).

| model | bucket | parity prob max | ANE pass 1 | ANE pass 2 | MLX at | MLX pass 1 | MLX pass 2 | auto pass 1 | auto pass 2 | 26.6.2 shipped ANE / MLX (auto) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| laya-typed-decisions | 64 | 0.0046 | 8.29 | 8.26 | 64 | 9.35 | 9.42 | yes | yes | 8.07 / 9.39 (yes) |
| laya-typed-decisions | 96 | 0.0046 | 9.15 | 9.16 | 64 | 9.35 | 9.42 | yes | yes | 9.02 / 9.39 (yes) |
| laya-typed-decisions | 128 | 0.0077 | 10.05 | 10.06 | 96 | 11.55 | 11.58 | yes | yes | 9.93 / 11.57 (yes) |
| laya | 64 | 0.0093 | 8.28 | 8.26 | 64 | 9.43 | 9.39 | yes | yes | 8.10 / 9.40 (yes) |
| laya | 96 | 0.0126 | 9.16 | 9.16 | 64 | 9.43 | 9.39 | yes | yes | 9.00 / 9.40 (yes) |
| laya | 128 | 0.0126 | 10.04 | 10.04 | 96 | 11.60 | 11.57 | yes | yes | 9.86 / 11.59 (yes) |
| laya-multilingual | 64 | 0.0079 | 3.46 | 3.48 | 64 | 5.39 | 5.36 | yes | yes | 3.43 / 5.42 (yes) |
| laya-multilingual | 96 | 0.0128 | 3.97 | 4.04 | 64 | 5.39 | 5.36 | yes | yes | 3.85 / 5.42 (yes) |
| laya-multilingual | 128 | 0.0128 | 4.43 | 4.48 | 96 | 6.33 | 6.32 | yes | yes | 4.31 / 6.36 (yes) |
| laya-multilingual | 256 | 0.0128 | 8.63 | 8.66 | 128 | 6.42 | 6.44 | no | no | 8.35 / 6.47 (no) |

- **Pass to pass:** every MLX cell (1, 4 and 8 questions; 65 cells over the three models)
  within 0.7%; every ANE bucket within 0.07 ms.
- **Screen** (30 samples, `laya-typed-decisions`): ANE 8.21 / 9.07 / 9.97 ms against MLX
  9.39 / 9.39 / 11.59 ms, auto 64/96/128 (`raw/screen-*`). Consistent with both passes.

## What ships

- `laya_apple/data/profiles/Apple_M4_Max-macos27-coremltools9.0.json` is byte-identical to
  `raw/pass1-profile.json`; a unit test checks this, re-derives its buckets with the shipped
  rule, and checks the pinned revisions.
- It applies to Apple M4 Max, macOS 27.x, coremltools 9.0 (the existing profile match: SoC,
  macOS major, coremltools). There, `platform_validated_for_auto_ane` is `true` and
  `Laya.info()["routing_profile"]` is `"shipped"`. Its auto buckets and service times replace
  routing.json's for that profile, the same way a local calibrated profile does on an
  unvalidated one.
- Unlike a local profile, it is not tied to this machine's artifact hashes: as with
  routing.json, any artifact must still pass parity and placement on the machine that uses it.
- `routing.json`, the 26.6.2 profile, the rule, the parity tolerances and every gate are
  unchanged.

## Test tiers

**With a local profile only** (before the shipped file existed; the cache's local profile from
pass 2), the 11 tests that failed in `research/macos27-validation/`: **6 passed, 5 failed**
(`raw/pytest-11-local-profile.txt`).

- Passed: the four `test_no_silent_fallback.py` tests and the two `test_workers.py` tests.
- Failed: `test_routing_e2e.py::test_auto_with_empty_artifact_cache_routes_to_gpu_with_artifact_unavailable`
  (3 models) and `test_artifact_revision_mismatch_under_explicit_ane_raises[laya-multilingual]`
  point `LAYA_APPLE_CACHE` at an empty directory, which holds no local profile, so `auto`
  reports `platform_not_validated`; `test_profiles.py::test_local_profile_ignored_on_a_shipped_profile`
  needs a shipped profile.

**With the shipped profile** (same cache, `HF_HUB_OFFLINE=1`):

- The same 11 tests: **all 11 passed** (`raw/pytest-11-shipped-profile.txt`, 13 cases: the
  revision-mismatch test also ran for `laya` and `laya-typed-decisions`).
- Every `integration`, `parity` and `ane` test without `stress`, leaving out the same five
  benchmark-path cases as `research/macos27-validation/` (`test_benchmark.py`, and
  `test_switchyard.py`'s two `test_smoke_*` tests): **254 passed, 0 failed, 5 skipped**
  (`raw/pytest-integration-parity-ane.txt`), against 229 passed, 11 failed and 19 skipped
  before. The 14 "no validated ANE artifacts" skips now run, because `auto` loads the ANE.
  The 5 remaining skips are fixture scope (`test_option_order.py`; `laya-multilingual` in
  `test_ane_async_parity.py`).

## Limits

- **One machine, one macOS build** (27.0, 26A428), coremltools 9.0. The profile covers every
  M4 Max on macOS 27.x with coremltools 9.0 because that is how profiles match; other macOS 27
  builds are not measured.
- **Single-question routing only**, as on 26.6.2. Multi-question requests never auto-route to
  the ANE. The ANE service times hold 1 question at the offered buckets only.
- **No release benchmark, serve run or adaptive-execution (`ane_handoff`) run on macOS 27.**
  Those results stay 26.6.2-only.
- **No macOS 27 prebuilt artifacts.** On this profile `auto` uses the ANE once the artifacts are
  built locally (`laya-apple artifacts build`).
