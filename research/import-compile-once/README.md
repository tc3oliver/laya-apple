# Validate imports at the registered path so the compile is paid once

Issue: [#8](https://github.com/tc3oliver/laya-apple/issues/8) (cold start), following
[#121](https://github.com/tc3oliver/laya-apple/issues/121) decision 1 (`research/coreml-compile-cache/`: Phase B
"pays twice"). Preregistration: [#162](https://github.com/tc3oliver/laya-apple/issues/162), copied verbatim below.
It was opened before any data and is not edited after the first row is written. Status: **harness ready, nothing
measured yet.**

## Preregistration (verbatim from #162)

Preregistration for `research/import-compile-once/` (part of #8; follows #121 decision 1). No data has been collected. The criteria below are not edited after the first row is written; a change of scope goes into a new preregistration.

**Question.** Can `import_artifact` (as used by `artifacts fetch`) pay Core ML's on-device ANE compile once instead of twice, by running the compute-plan check, the full parity gate and the placement probe on the model loaded at the registered path while `manifest.json` is withheld, then publishing the manifest atomically — without weakening any guarantee of `docs/no-silent-fallback.md` (rows 4, 5, 9, 10, 23)?

**Arms.**
- `baseline`: `laya_apple.lifecycle.import_artifact` on main `484a645`, unmodified, with fetch's hooks (`probe` = staged `_probe`, `registered_probe` = registered `_probe`).
- `final-path`: research prototype `research/import-compile-once/proto_import.py` (mechanism A, first install only; `--force`/replace delegates to baseline). Extract to staging and hash-check there; under `build_lock` rename into the final path with `manifest.json` withheld (kept as `manifest.pending.json` plus a `PENDING.json` with the pid); at the final path run the compute plan, the full parity gate and the placement probe on that one load; publish with `os.replace(manifest.pending.json → manifest.json)` and unlink the marker. Any failure renames to `.failed-<pid>` and removes it. Same hooks, same checks; only their location and order change.

**Model.** laya-typed-decisions. Timing at L128 (comparable to #121 Phase B); injected failures at L64. Archives from `artifacts export` of the validated macOS 27 cache. Every import goes into a new empty `LAYA_APPLE_CACHE` on the internal disk. Each import and each reader runs in its own child process of the same interpreter (same e5rt process name).

**Metrics, per import.** `T` = total wall time of the import call. Sub-phases: `S` static checks (hash, compute plan), `P` parity gate incl. its load, `Q` probes, `R` registered-path load and probe (baseline only). `E` = count and bytes of new e5rt bundle directories during the import (listing diff of `~/Library/Caches/python*/com.apple.e5rt.e5bundlecache/<build>/`). Reader: a new process runs `load_verified` and the probe at the registered path; `F` = its first-load time, `W'` = a second load in the same process, plus its own e5rt growth.

**Validity guard (any failure → that arm is INVALID: a correctness finding with its own issue, never counted as a saving).**
1. Parity summary (excluding timing fields) identical to baseline's, `passed=True`, 0 hard mismatches; every probe ratio ≤ 0.8 (`PROBE_MAX_RATIO` unchanged).
2. The reader after import loads on CPU_AND_NE, its probe is ≤ 0.8, and its predictions on the probe features are bit-identical to baseline's reader.
3. A concurrent poller (`load_verified(spec, 128)` every 0.5 s during each `final-path` import) never gets a model before `manifest.json` appears; before that it only gets `ArtifactMissingError`, and nothing is moved into `quarantine/`.
4. Injected failures with `final-path`: (i) parity reported failed at the final path; (ii) the probe at the final path raises `ComputeUnitMismatchError`; (iii, full run only) `--force` replace of a registered L64 combined with (ii): the old artifact is restored with byte-identical `manifest.json` and identical `tree_sha256`; (iv, full run only) SIGKILL during the gate, then a re-import. For (i)–(ii): the call raises, nothing is registered, the final path is absent, the poller never sees a manifest, `quarantine/` is empty. For (iv): the final path has no manifest, readers get `ArtifactMissingError`, and the re-import succeeds with no leftover.

**Classification (valid arms only; `B` = median `T` of baseline, `Eb` = median `E` bytes of baseline).**
- **SAVES:** every `final-path` repeat has `T ≤ 0.80 × B` and `B − T ≥ 15 s`, `E ≤ 0.6 × Eb`, and its reader `F ≤` baseline-median `F` + 5 s.
- **NO EFFECT:** every repeat has `T ≥ 0.90 × B` or `E ≥ 0.9 × Eb`.
- **INCONCLUSIVE:** anything else, including repeats that disagree.
- Reported, no criterion: whether a reader reuses the import's compile (`F ≤ W' + 10 s`) in each arm.

**Screen first (about 20 min, n = 1).** Baseline L128, then final-path L128 with the poller, then injection (i) at L64. Screen labels use the thresholds above with n = 1. screen-NO-EFFECT → stop and write up. screen-INVALID → stop and open a correctness issue. screen-SAVES → full run in a separate slot: 3 + 3 L128 interleaved (B, A, B, A, B, A) plus injections (i)–(iv). screen-INCONCLUSIVE → the maintainer decides. A screen never declares SAVES.

**Decisions, fixed before data.**
1. Full-run SAVES with every guard passing → one PR `perf(artifacts): validate imports at the registered path` rebuilding mechanism A in `lifecycle.import_artifact` (first install only, fetch and CLI import), with the prune rule for an abandoned pending directory, recovery from a leftover one, unit tests for each guard item, no-silent-fallback row 23 and the `prebuilt.py` docstring updated; #121 Phase B re-run before and after, comparison in the PR.
2. `--force` replace stays on the current path whatever the result.
3. NO EFFECT or INCONCLUSIVE → record, close this mechanism, `import_artifact` unchanged.
4. INVALID → no ship whatever the timings, plus a correctness issue.
5. Nothing here changes `PROBE_MAX_RATIO`, the parity tolerances or #121's verdicts.

**Machine conditions.** Mac Studio M4 Max, macOS 27.0 (26A428), coremltools 9.0. Exclusive slot, local LLM service stopped and restored after, no tests or other benchmarks running, mains power, ≥ 200 GB free on both volumes. Recorded before/after each step: `sw_vers`, `uptime`, `pmset -g therm`, `pmset -g batt`, `df`, e5rt listing and size, git SHA, coremltools version.

**Limitations.** One machine, one model, n ≤ 3. e5rt bundles are a proxy for compiles, not an API. The cache is keyed by process name, so a server with a different process name may compile again regardless. The prototype is a research copy of `import_artifact`; its diff against main is recorded. Concurrent prune is not exercised here (covered by the production PR's tests). The unexplained P < R from #121 is reported via sub-phases, with no criterion.

**Cleanup.** Remove the temp caches and export directory, and the e5rt bundle directories this run created (before/after name diff). Restore the local LLM service.



## How to run

Exclusive slot: the local LLM service stopped (and restored afterwards), no tests or other benchmarks, mains
power, ≥ 200 GB free on both volumes. `<cache>` is the laya-apple cache holding the artifacts validated on macOS
27; it is only read, by `artifacts export`. `<data>` is the second volume whose free space the preflight checks.

```bash
uv sync --extra dev --extra ane            # once, before the slot
export LAYA_APPLE_SOURCE_CACHE=<cache> LAYA_IC_DATA_VOLUME=<data>
zsh research/import-compile-once/run-screen.sh   # the screen, about 20 min
zsh research/import-compile-once/run-full.sh     # only after screen-SAVES, in a separate slot
```

Each script (shared steps in `common.sh`):
1. `harness.py preflight`: records `sw_vers`, `uptime`, `pmset -g therm`, `pmset -g batt`, `df`, the e5rt
   listing and size, the git SHA and the coremltools version; stops on < 200 GB free, a thermal warning or no
   mains power.
2. `harness.py e5rt-snapshot`: the names in every `~/Library/Caches/python*/com.apple.e5rt.e5bundlecache/<build>/`
   at the start of the run, for the cleanup.
3. Exports L128 and L64 from `<cache>` into a new `mktemp` directory.
4. One `harness.py step` per import: the screen runs baseline L128, final-path L128 with the poller, then
   injection (i) at L64; the full run B, A, B, A, B, A at L128, then injections (i)–(iv) at L64.
5. `analyze.py` prints the guard and the screen or full label, and writes `analysis.json`.
6. On exit, always: `harness.py cleanup` removes the e5rt entries that are new since step 2 (a name diff; each one
   must be a hex-named directory directly inside an e5rt build directory, and anything else is reported, not
   removed), the temp caches and the export directory (each must be a `laya-ic-*` directory directly inside the
   temp directory). Then a closing snapshot, and `harness.py redact` replaces local absolute paths in the output
   with `<cache>`, `<data>`, `<tmp>`, `<repo>` and `~`.

Output: `raw-<date>-screen/` (or `-full/`): one `<label>.json` per step and its child records under `steps/`.

### What a step records

- A new empty `LAYA_APPLE_CACHE` under the internal-disk temp directory, removed at the end of the step.
- Every Core ML call runs in a child process of the same interpreter: the import, the reader, the poller and the
  post-step inspection.
- **Import**: `T` (the import call) and the sub-phases, timed around the laya_apple helpers both arms call:
  - `S`: `verify_files`, `tree_sha256`, `compute_plan_summary` outside `load_verified`;
  - `P`: `ane_parity`, which includes its own load;
  - `Q`: the probe hook (baseline: the staged probe; final-path: its one probe, at the registered path);
  - `R`: `load_verified` and the registered probe (baseline only).
- **`E`**: the e5rt listing diff around the import child; only new entries are sized.
- **Post**: whether the final path and its manifest exist, pending files, `.old-`/`.failed-` leftovers, staging and
  `quarantine/` contents, the `manifest.json` SHA-256, `tree_sha256`, and the registered parity summary and placement.
- **Reader** (after a successful import): a new process runs `load_verified` (`F`), then
  `probe_placement` on that model, predictions on the probe features (dtype, shape and SHA-256 per output), and a
  second `load_verified` (`W'`), plus its own e5rt growth.
- **Poller** (`--poller`): a child calling `load_verified(spec, bucket)` every 0.5 s from before the import starts
  until it ends. It writes one line per attempt (outcome, `quarantine/` listing). It stops before loading once
  `manifest.json` exists, and after one successful load, so it adds as little as possible to `E`.
- **Injections** (final-path):
  - (i) the parity summary at the registered path is reported `passed=False` after the real gate ran;
  - (ii) the probe at the registered path raises `ComputeUnitMismatchError` after the real probe ran;
  - (iii) a first install of L64, then `--force` with (ii) (the replace goes through main's path, so the failing
    probe is its registered probe); the old manifest's SHA-256 and `tree_sha256` are recorded before and after;
  - (iv) SIGKILL 3 s after the gate starts; then a reader, a re-import and the post state.

## Differences from `import_artifact`

`proto_import.py` is a research copy of `laya_apple.lifecycle.import_artifact` at `484a645`. On a first install
(`force=False`) it differs only as follows; `force=True` calls main's `import_artifact` unchanged.

| Step | main `import_artifact` | `proto_import_artifact` |
|---|---|---|
| archive, manifest, `expect`, profile, file hash | staging | staging (same code) |
| `artifact_sha256_check` (`tree_sha256`) | staging, after the probe | staging, before the lock |
| compute plan (`compute_plan_summary`) | staging, under the lock | registered path, under the lock |
| weight check, parity gate | staging | registered path |
| probe hooks | `probe` on staging; `registered_probe` at the registered path | one probe at the registered path: `registered_probe` if given, else `probe` |
| move into place | `os.rename(stage, final)` with `manifest.json` | `os.rename(stage, final)` with `manifest.pending.json` and `PENDING.json` (pid) |
| registered-path load | `load_verified(full=True)` (hash, compute plan, load) after the move | none; the parity gate's load is the one load at that path |
| registration | the rename | `os.replace(manifest.pending.json → manifest.json)`, then the marker is unlinked |
| verification stamp | written by that `load_verified` | written after publishing, with `load_verified`'s own key and the placement above |
| failure after the move | rename to `.failed-<pid>`, remove it, restore the old copy | rename to `.failed-<pid>`, remove it (no old copy: first install only) |
| a registered path holding `PENDING.json` and no manifest | — | removed under the lock (`.failed-<pid>`, then deleted) before installing; the lock means its import is gone |
| a registered path with a manifest, `force=False` | refused | refused |

`build_lock`, the staging `BUILDING.json` marker, `MAX_IMPORT_BYTES` and the tar checks are main's. `prune` is
unchanged and does not know `PENDING.json`; concurrent prune is out of scope (preregistration, Limitations).

## Files

- `proto_import.py`: mechanism A (first install only).
- `harness.py`: the steps, the children, the e5rt listing diff, the cleanup, the redaction and the machine
  snapshots.
- `analyze.py`: the guard and the classification, with the preregistered thresholds, standard library only.
- `run-screen.sh`, `run-full.sh`, `common.sh`: the runners.
- `tests/unit/test_import_compile_once.py`: analyze's thresholds and guard, the e5rt diff and cleanup path
  safety, and the prototype with Core ML stubbed. No hardware is needed.
