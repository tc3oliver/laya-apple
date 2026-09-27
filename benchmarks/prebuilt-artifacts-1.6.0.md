# Prebuilt ANE artifacts — clean-cache fetch check, laya-apple 1.6.0

This check covers the published repository
[`tc3oliver/laya-apple-artifacts`](https://huggingface.co/tc3oliver/laya-apple-artifacts) at
Hugging Face commit `93181067cfee9c6117a7919321eb303ec36fcbd4`, the revision laya-apple 1.6.0
pins as `DEFAULT_PREBUILT_REVISION`. That commit holds
10 archives, one for each model/bucket pair offered on the one published profile.

- **Profile:** Apple M4 Max, macOS 26.6.2 (25G83), coremltools 9.0. It is both the build machine
  and the machine this check ran on.
- **Package:** laya-apple 1.6.0, built from a clean `git archive` of `0ce1156` (the release
  branch, with the revision pin). It was installed from the wheel alone into a fresh Python 3.12 virtual environment
  and run outside the source tree; `laya_apple.__file__` resolved to site-packages.
- **Cache:** a new, empty `LAYA_APPLE_CACHE`. Checkpoints came from the existing Hugging Face
  cache, and no archive was cached locally beforehand.
- **Date:** 2026-09-27, 11:41–12:06 UTC.

## Method

For each of laya-typed-decisions, laya and laya-multilingual:

```bash
laya-apple artifacts fetch MODEL      # no --repo, no --revision: the default repository at the pinned commit
laya-apple artifacts verify MODEL
```

`fetch` downloads each archive and checks its SHA-256 against the repository's `index.json`.
Every `imported.from` in the raw file reads `hf://tc3oliver/laya-apple-artifacts@93181067…`,
so the pinned commit, not `main`, was read. It then imports the archive through `artifacts import`, which checks the manifest, the
platform profile, the file hash and the compute plan, and runs the full FP16 parity gate
against the shipped goldens. Last, it loads each bucket once and runs the runtime placement
probe. The raw per-bucket import records (parity, placement, provenance) are in
[`prebuilt-artifacts-1.6.0.json`](prebuilt-artifacts-1.6.0.json).

## Result: all 10 pairs passed

| model | L | archive SHA-256 | parity prob max | hard | near-tie flips | placement: ANE, transitions | probe ratio (ANE / CPU) | parity / register+load (s) |
|---|---|---|---|---|---|---|---|---|
| laya | 64 | `bb4b3a35f88ae2af…` | 0.0093 | 0 | none | 100% (10594 ops), 0 | 0.483 (7.83 / 16.203 ms) | 34.0 / 83.4 |
| laya | 96 | `779395ff629c24f6…` | 0.0126 | 0 | lang-fr row 0 (margin 0.0035) | 100% (10594 ops), 0 | 0.345 (8.55 / 24.757 ms) | 35.5 / 85.4 |
| laya | 128 | `7e7e1ef3459b02ef…` | 0.0126 | 0 | lang-fr row 0 (margin 0.0035) | 100% (10594 ops), 0 | 0.351 (9.362 / 26.644 ms) | 35.9 / 87.2 |
| laya-multilingual | 64 | `488fa6b03b440a36…` | 0.0079 | 0 | none | 100% (6390 ops), 0 | 0.423 (3.202 / 7.565 ms) | 17.4 / 40.1 |
| laya-multilingual | 96 | `201f868b8769c8e5…` | 0.0128 | 0 | none | 100% (6390 ops), 0 | 0.319 (3.606 / 11.301 ms) | 18.1 / 41.3 |
| laya-multilingual | 128 | `cf508dc8126f789f…` | 0.0128 | 0 | none | 100% (6390 ops), 0 | 0.313 (3.935 / 12.579 ms) | 18.2 / 41.3 |
| laya-multilingual | 256 | `0c954dbd6ad6a4fa…` | 0.0128 | 0 | none | 100% (6390 ops), 0 | 0.309 (7.427 / 24.043 ms) | 21.3 / 46.1 |
| laya-typed-decisions | 64 | `959e18c65f92a514…` | 0.0046 | 0 | none | 100% (10594 ops), 0 | 0.494 (7.878 / 15.953 ms) | 34.6 / 84.6 |
| laya-typed-decisions | 96 | `9500017f7b329c95…` | 0.0046 | 0 | none | 100% (10594 ops), 0 | 0.349 (8.594 / 24.617 ms) | 35.7 / 86.8 |
| laya-typed-decisions | 128 | `8abd996f5019eda8…` | 0.0077 | 0 | none | 100% (10594 ops), 0 | 0.348 (9.321 / 26.757 ms) | 36.2 / 87.4 |

- **Parity:** every pair passed the unchanged gate (probability error ≤ 0.02, 0 hard
  mismatches). The near-tie flips are listed above, not hidden.
- **Placement:** 100% of ops ran on the Neural Engine with 0 transitions. The probe ratio was
  0.309–0.494 against a limit of ≤ 0.8.
- **`artifacts verify`:** OK for all 10.
- **Fetch wall time:** laya-typed-decisions 575.0 s, laya 565.3 s, laya-multilingual 376.8 s.
  Each includes the download, the parity gate and the first registered load, and that first
  load includes Core ML's on-device compile.

## Limits of this check

- **It ran on the build machine, with an empty cache.** It shows that the archives, the index
  and the fetch → import → parity → probe path are intact. It is not an independent check on a
  second machine of the same profile; none has been done yet. Under
  [`docs/publishing.md`](../docs/publishing.md) that check is recommended, not required, for
  publishing the first profile, because every receiving machine re-validates each artifact
  (integrity, platform, parity, placement) before it registers it. A second-machine record
  will be added here when one is available.
- **It covers one profile only.** Every other SoC, macOS major or coremltools version gets
  `ArtifactMissingError` and builds locally.
- **Fetching does not remove the on-device compile.** The cold-start numbers are in
  [`research/coreml-compile-cache/screen.md`](../research/coreml-compile-cache/screen.md).
