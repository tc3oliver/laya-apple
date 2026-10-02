# Prebuilt ANE artifacts for macOS 27 — clean-cache fetch check, laya-apple 1.6.2

This check covers the macOS 27 archives in
[`tc3oliver/laya-apple-artifacts`](https://huggingface.co/tc3oliver/laya-apple-artifacts) at
Hugging Face commit `44a54765f89d434e9d85f401126cfee7d2059e7d`, the revision laya-apple 1.6.2
pins as `DEFAULT_PREBUILT_REVISION`. That commit adds 10 archives under
`apple-m4-max-macos27-coremltools9.0/`, one per model/bucket pair, to the 10 macOS 26 archives
that 1.6.0 and 1.6.1 read at `93181067cfee9c6117a7919321eb303ec36fcbd4`. Its `index.json` has
20 entries; the 10 macOS 26 entries are identical to the old pin's, field for field.

- **Profile:** Apple M4 Max, macOS 27.0 (26A428), coremltools 9.0. It is both the build machine
  and the machine this check ran on.
- **Two runs,** each with the package installed into a fresh Python 3.12 virtual environment
  and run outside the source tree (`laya_apple.__file__` resolved to site-packages), and each
  with a new, empty `LAYA_APPLE_CACHE`:
  - **Run A:** laya-apple 1.6.1 from PyPI (`laya-apple[ane]==1.6.1`), with
    `--revision 44a54765f89d434e9d85f401126cfee7d2059e7d`. It checks the upload with the
    released code, before the pin changed.
  - **Run B:** laya-apple 1.6.2, a wheel built from a clean `git archive` of `37f726c` (the
    release branch, with the new pin), with no `--revision`. It checks that the pin resolves to
    the upload.
- **Checkpoints** came from the existing Hugging Face cache. No archive was cached locally
  beforehand.
- **Date:** 2026-10-02, run A 19:00–19:22 UTC, run B 19:24–19:47 UTC. The 1-minute load average
  was 1.9 at the start of run A and 1.7 at the start of run B. At the end of run B the 5-minute
  load average was 8.2, so something else ran during its last minutes; run B's timings may
  include that load, but no check depends on timing except the probe ratio.

## Method

For each of laya-typed-decisions, laya and laya-multilingual:

```bash
laya-apple artifacts fetch MODEL [--revision 44a54765f89d434e9d85f401126cfee7d2059e7d]   # run A only
laya-apple artifacts verify MODEL
```

`fetch` downloads each archive and checks its SHA-256 against the repository's `index.json`.
It then imports the archive through `artifacts import`, which checks the manifest, the
platform profile, the file hash and the compute plan, and runs the full FP16 parity gate
against the shipped goldens. It probes placement on the staged copy, moves it into place,
loads it there and probes again; the second probe is the one reported. Every `imported.from`
in the raw file reads `hf://tc3oliver/laya-apple-artifacts@44a54765…`, in both runs. The raw
per-bucket records (parity, placement, provenance, probe) are in
[`prebuilt-artifacts-1.6.2-macos27.json`](prebuilt-artifacts-1.6.2-macos27.json); `run` is
`released-1.6.1` or `wheel-1.6.2`.

## Result: all 10 pairs passed, in both runs

Acceptance per pair: the archive's SHA-256 matches the index, parity passes (probability error
≤ 0.02, 0 hard mismatches, near-tie flips listed), placement is 100% Neural Engine with 0
transitions, the probe ratio is ≤ 0.8, and `artifacts verify` says OK.

### Run A: laya-apple 1.6.1, `--revision 44a54765…`

| model | L | archive SHA-256 | parity prob max | hard | near-tie flips | placement: ANE, transitions | probe ratio (ANE / CPU) | parity / register+load (s) |
|---|---|---|---|---|---|---|---|---|
| laya | 64 | `63c6953a8d0ef9a5…` | 0.0093 | 0 | none | 100% (10594 ops), 0 | 0.522 (8.195 / 15.704 ms) | 25.1 / 67.4 |
| laya | 96 | `f2c2e67a13a762a5…` | 0.0126 | 0 | lang-fr row 0 (margin 0.0035) | 100% (10594 ops), 0 | 0.364 (8.85 / 24.308 ms) | 26.7 / 69.7 |
| laya | 128 | `18b04aac082e07c2…` | 0.0126 | 0 | lang-fr row 0 (margin 0.0035) | 100% (10594 ops), 0 | 0.362 (9.408 / 25.955 ms) | 27.1 / 70.4 |
| laya-multilingual | 64 | `b595810c2bdeac83…` | 0.0079 | 0 | none | 100% (6390 ops), 0 | 0.404 (3.11 / 7.691 ms) | 14.3 / 36.0 |
| laya-multilingual | 96 | `81391a845102ccfd…` | 0.0128 | 0 | none | 100% (6390 ops), 0 | 0.258 (3.889 / 15.095 ms) | 18.7 / 41.9 |
| laya-multilingual | 128 | `77efadaea4038eb6…` | 0.0128 | 0 | none | 100% (6390 ops), 0 | 0.257 (4.213 / 16.41 ms) | 17.8 / 42.6 |
| laya-multilingual | 256 | `b714b02598b96d59…` | 0.0128 | 0 | none | 100% (6390 ops), 0 | 0.29 (7.865 / 27.086 ms) | 28.6 / 47.0 |
| laya-typed-decisions | 64 | `89c38790b6d16ec4…` | 0.0046 | 0 | none | 100% (10594 ops), 0 | 0.506 (7.987 / 15.797 ms) | 25.4 / 68.5 |
| laya-typed-decisions | 96 | `a2a8a4987a468a75…` | 0.0046 | 0 | none | 100% (10594 ops), 0 | 0.354 (8.632 / 24.372 ms) | 26.7 / 70.7 |
| laya-typed-decisions | 128 | `98cb2470ff963c99…` | 0.0077 | 0 | none | 100% (10594 ops), 0 | 0.366 (9.487 / 25.903 ms) | 27.0 / 70.5 |

### Run B: laya-apple 1.6.2, pinned revision

| model | L | archive SHA-256 | parity prob max | hard | near-tie flips | placement: ANE, transitions | probe ratio (ANE / CPU) | parity / register+load (s) |
|---|---|---|---|---|---|---|---|---|
| laya | 64 | `63c6953a8d0ef9a5…` | 0.0093 | 0 | none | 100% (10594 ops), 0 | 0.511 (8.0 / 15.648 ms) | 27.7 / 72.6 |
| laya | 96 | `f2c2e67a13a762a5…` | 0.0126 | 0 | lang-fr row 0 (margin 0.0035) | 100% (10594 ops), 0 | 0.359 (8.681 / 24.193 ms) | 27.4 / 71.2 |
| laya | 128 | `18b04aac082e07c2…` | 0.0126 | 0 | lang-fr row 0 (margin 0.0035) | 100% (10594 ops), 0 | 0.361 (9.378 / 25.985 ms) | 27.9 / 73.1 |
| laya-multilingual | 64 | `b595810c2bdeac83…` | 0.0079 | 0 | none | 100% (6390 ops), 0 | 0.405 (3.038 / 7.506 ms) | 15.7 / 43.2 |
| laya-multilingual | 96 | `81391a845102ccfd…` | 0.0128 | 0 | none | 100% (6390 ops), 0 | 0.336 (3.724 / 11.08 ms) | 20.0 / 39.3 |
| laya-multilingual | 128 | `77efadaea4038eb6…` | 0.0128 | 0 | none | 100% (6390 ops), 0 | 0.335 (4.064 / 12.129 ms) | 15.7 / 38.2 |
| laya-multilingual | 256 | `b714b02598b96d59…` | 0.0128 | 0 | none | 100% (6390 ops), 0 | 0.284 (7.888 / 27.745 ms) | 18.6 / 44.3 |
| laya-typed-decisions | 64 | `89c38790b6d16ec4…` | 0.0046 | 0 | none | 100% (10594 ops), 0 | 0.491 (7.607 / 15.506 ms) | 27.7 / 75.5 |
| laya-typed-decisions | 96 | `a2a8a4987a468a75…` | 0.0046 | 0 | none | 100% (10594 ops), 0 | 0.362 (8.97 / 24.782 ms) | 32.2 / 82.8 |
| laya-typed-decisions | 128 | `98cb2470ff963c99…` | 0.0077 | 0 | none | 100% (10594 ops), 0 | 0.356 (9.32 / 26.166 ms) | 31.9 / 79.1 |

- **Parity:** every pair passed the unchanged gate in both runs, with identical results in the
  two runs (the same `imported.parity` and `imported.placement` records, and the same artifact
  hash). The near-tie flips are listed above, not hidden.
- **Placement:** 100% of ops ran on the Neural Engine with 0 transitions. The probe ratio was 0.257–0.522 in run A and 0.284–0.511 in run B, against a limit of ≤ 0.8.
- **`artifacts verify`:** OK for all 10, in both runs.
- **Fetch wall time** (download, parity gate and the first registered load, which includes Core
  ML's on-device compile):

  | model | run A (s) | run B (s) |
  |---|---:|---:|
  | laya-typed-decisions | 470 | 526 |
  | laya | 467 | 482 |
  | laya-multilingual | 386 | 363 |

- **Same op counts and parity errors as the macOS 26.6.2 archives**
  ([`prebuilt-artifacts-1.6.0.md`](prebuilt-artifacts-1.6.0.md)).

## Core ML compile cache

Core ML writes its on-device compile to `~/Library/Caches/<process name>/com.apple.e5rt.e5bundlecache/`
and never evicts it. Each run grew the cache of the Python process (`python`) as follows
(`du -sk` before and after, GB = 10^9 bytes):

| run | before (GB) | after (GB) | growth (GB) |
|---|---:|---:|---:|
| A | 5.37 | 26.79 | 21.42 |
| B | 26.79 | 48.21 | 21.42 |

Each import compiles twice, once for the staged parity gate and once at the registered path,
so 10 buckets mean 20 compiles. The check caches under `LAYA_APPLE_CACHE` were deleted after
the record was taken; the Core ML cache was left as it was.

## Limits of this check

- **It ran on the build machine, with an empty cache.** It shows that the archives, the index,
  the pin and the fetch → import → parity → probe path are intact. It is not an independent
  check on a second machine of the same profile; none has been done, for either profile.
  Under [`docs/publishing.md`](../docs/publishing.md) that check is recommended, not required,
  because every receiving machine re-validates each artifact before it registers it.
- **Only macOS 27.0 was measured.** The profile matches every macOS 27.x on an M4 Max with
  coremltools 9.0, so later 27.x releases will select these archives without having been
  measured. The import's compute-plan check, parity gate and placement probe still run there.
- **Fetching does not remove the on-device compile.** A cold start of laya-typed-decisions
  (64/96/128) took 210.152 s (median of 3) on macOS 27.0
  ([`research/coreml-compile-cache/results.md`](../research/coreml-compile-cache/results.md)).
