# Prebuilt ANE artifacts — clean-cache fetch check, laya-apple 1.6.0

This check covers the published repository
[`tc3oliver/laya-apple-artifacts`](https://huggingface.co/tc3oliver/laya-apple-artifacts) at
Hugging Face commit `93181067cfee9c6117a7919321eb303ec36fcbd4` (its `main`). That commit holds
10 archives, one for each model/bucket pair offered on the one published profile.

- **Profile:** Apple M4 Max, macOS 26.6.2 (25G83), coremltools 9.0. It is both the build machine
  and the machine this check ran on.
- **Package:** laya-apple 1.6.0, built from a clean `git archive` of `4f65423` (the release
  branch). It was installed from the wheel alone into a fresh Python 3.12 virtual environment
  and run outside the source tree; `laya_apple.__file__` resolved to site-packages.
- **Cache:** a new, empty `LAYA_APPLE_CACHE`. Checkpoints came from the existing Hugging Face
  cache, and no archive was cached locally beforehand.
- **Date:** 2026-09-27, 07:54–08:20 UTC.

## Method

For each of laya-typed-decisions, laya and laya-multilingual:

```bash
laya-apple artifacts fetch MODEL      # no --repo: uses the default repository
laya-apple artifacts verify MODEL
```

`fetch` downloads each archive and checks its SHA-256 against the repository's `index.json`.
It then imports the archive through `artifacts import`, which checks the manifest, the
platform profile, the file hash and the compute plan, and runs the full FP16 parity gate
against the shipped goldens. Last, it loads each bucket once and runs the runtime placement
probe. The raw per-bucket import records (parity, placement, provenance) are in
[`prebuilt-artifacts-1.6.0.json`](prebuilt-artifacts-1.6.0.json).

## Result: all 10 pairs passed

| model | L | archive SHA-256 | parity prob max | hard | near-tie flips | placement: ANE, transitions | probe ratio (ANE / CPU) | parity / register+load (s) |
|---|---|---|---|---|---|---|---|---|
| laya | 64 | `bb4b3a35f88ae2af…` | 0.0093 | 0 | none | 100% (10594 ops), 0 | 0.483 (7.823 / 16.209 ms) | 34.1 / 83.1 |
| laya | 96 | `779395ff629c24f6…` | 0.0126 | 0 | lang-fr row 0 (margin 0.0035) | 100% (10594 ops), 0 | 0.352 (8.621 / 24.492 ms) | 35.6 / 85.4 |
| laya | 128 | `7e7e1ef3459b02ef…` | 0.0126 | 0 | lang-fr row 0 (margin 0.0035) | 100% (10594 ops), 0 | 0.35 (9.357 / 26.766 ms) | 36.0 / 86.4 |
| laya-multilingual | 64 | `488fa6b03b440a36…` | 0.0079 | 0 | none | 100% (6390 ops), 0 | 0.427 (3.25 / 7.615 ms) | 17.3 / 40.2 |
| laya-multilingual | 96 | `201f868b8769c8e5…` | 0.0128 | 0 | none | 100% (6390 ops), 0 | 0.306 (3.498 / 11.433 ms) | 18.1 / 41.4 |
| laya-multilingual | 128 | `cf508dc8126f789f…` | 0.0128 | 0 | none | 100% (6390 ops), 0 | 0.318 (3.982 / 12.526 ms) | 18.2 / 41.3 |
| laya-multilingual | 256 | `0c954dbd6ad6a4fa…` | 0.0128 | 0 | none | 100% (6390 ops), 0 | 0.302 (7.437 / 24.61 ms) | 21.4 / 45.9 |
| laya-typed-decisions | 64 | `959e18c65f92a514…` | 0.0046 | 0 | none | 100% (10594 ops), 0 | 0.495 (7.858 / 15.863 ms) | 34.4 / 84.7 |
| laya-typed-decisions | 96 | `9500017f7b329c95…` | 0.0046 | 0 | none | 100% (10594 ops), 0 | 0.346 (8.55 / 24.73 ms) | 35.7 / 86.7 |
| laya-typed-decisions | 128 | `8abd996f5019eda8…` | 0.0077 | 0 | none | 100% (10594 ops), 0 | 0.354 (9.37 / 26.498 ms) | 36.0 / 86.4 |

- **Parity:** every pair passed the unchanged gate (probability error ≤ 0.02, 0 hard
  mismatches). The near-tie flips are listed above, not hidden.
- **Placement:** 100% of ops ran on the Neural Engine with 0 transitions. The probe ratio was
  0.302–0.495 against a limit of ≤ 0.8.
- **`artifacts verify`:** OK for all 10.
- **Fetch wall time:** laya-typed-decisions 570.8 s, laya 564.6 s, laya-multilingual 375.8 s.
  Each includes the download, the parity gate and the first registered load, and that first
  load includes Core ML's on-device compile.

## Limits of this check

- **It ran on the build machine, with an empty cache.** It shows that the archives, the index
  and the fetch → import → parity → probe path are intact. It does not show that the archives
  work on a second machine of the same profile, which `docs/publishing.md` asks for before
  the repository is announced.
- **It covers one profile only.** Every other SoC, macOS major or coremltools version gets
  `ArtifactMissingError` and builds locally.
- **Fetching does not remove the on-device compile.** The cold-start numbers are in
  [`research/coreml-compile-cache/screen.md`](../research/coreml-compile-cache/screen.md).
