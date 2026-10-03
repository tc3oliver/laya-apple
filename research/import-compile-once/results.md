# Results: validate imports at the registered path

Issue: [#8](https://github.com/tc3oliver/laya-apple/issues/8). Preregistration:
[#162](https://github.com/tc3oliver/laya-apple/issues/162), with [addendum
1](https://github.com/tc3oliver/laya-apple/issues/162#issuecomment-5965336692) (implementation choices, posted
before any data). Criteria unedited. Raw data: [`raw-2026-10-03-screen/`](raw-2026-10-03-screen/),
[`raw-2026-10-03-full/`](raw-2026-10-03-full/), [`raw-2026-10-03-phaseb/`](raw-2026-10-03-phaseb/). Runners:
[`run-screen.sh`](run-screen.sh), [`run-full.sh`](run-full.sh), [`run-phaseb.sh`](run-phaseb.sh) (redacted copy).

Local absolute paths in the raw files are replaced by `<cache>`, `<data>`, `<tmp>`, `<repo>` and `~`
(`harness.py redact`; the worktree path in the Phase B import logs by hand). No measured value was changed.

**Result: SAVES, every guard passes. Decision 1 is taken.**

## Setup

Mac Studio M4 Max, macOS 27.0 (26A428), coremltools 9.0, Python 3.12.14, laya_apple 1.6.2. Harness and prototype
at 0d49789, whose `laya_apple/` is identical to main 484a645 (`git diff --stat 484a645 0d49789 -- laya_apple`
is empty), so the baseline arm is main's `import_artifact` (`setup.txt`). Model laya-typed-decisions, timing at
L128, injections at L64. Every import into a new empty `LAYA_APPLE_CACHE` on the internal disk. Exclusive slot
with the local LLM service stopped (not separately recorded in the raw files, apart from the two minor overlaps
under Limitations). AC power and no thermal or performance warning in any `machine_before`/`machine_after`
record. Run on 2026-10-03 (UTC): screen 04:04–04:12, full run 04:12–04:34,
Phase B re-run 04:36–04:49.

`git_dirty` is true in the machine records because the raw output directories were untracked during the run.

## Preregistered results

### Screen (n = 1)

Source: `raw-2026-10-03-screen/analysis.txt`, `analysis.json`.

| Row | `T` s | `S` | `P` | `Q` | `R` | `E` count | `E` GB | Reader `F` s | `W'` s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-L128-1 | 147.2 | 41.1 | 28.2 | 3.7 | 69.8 | 6 | 2.979 | 0.32 | 0.15 |
| final-path-L128-1 | 74.7 | 40.9 | 26.7 | 3.7 | — | 3 | 1.489 | 0.33 | 0.15 |

Guard: baseline-L128-1, final-path-L128-1 and inject-i-L64 pass. Label: **screen-SAVES**, so the full run went
ahead in a separate slot.

### Full run (3 + 3, interleaved B, A, B, A, B, A)

Source: `raw-2026-10-03-full/analysis.txt`, `analysis.json`.

| Row | `T` s | `S` | `P` | `Q` | `R` | `E` count | `E` GB | Reader `F` s | `W'` s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-L128-1 | 145.7 | 41.2 | 27.0 | 3.7 | 70.3 | 6 | 2.979 | 0.35 | 0.16 |
| baseline-L128-2 | 146.1 | 41.0 | 26.8 | 3.7 | 71.2 | 6 | 2.979 | 0.34 | 0.15 |
| baseline-L128-3 | 145.9 | 41.2 | 27.2 | 3.7 | 70.3 | 6 | 2.979 | 0.33 | 0.15 |
| final-path-L128-1 | 75.4 | 41.4 | 26.8 | 3.7 | — | 3 | 1.489 | 0.33 | 0.15 |
| final-path-L128-2 | 75.6 | 41.4 | 27.0 | 3.7 | — | 3 | 1.489 | 0.33 | 0.16 |
| final-path-L128-3 | 75.3 | 41.2 | 26.9 | 3.7 | — | 3 | 1.489 | 0.33 | 0.15 |

`B` = 145.90 s, `Eb` = 2.979 GB (2 978 528 906 bytes), baseline median `F` = 0.34 s.

SAVES rule, per final-path repeat:
- `T ≤ 0.80 × B` = 116.72 s: 75.37, 75.63, 75.28 s. Yes.
- `B − T ≥ 15 s`: 70.53, 70.27, 70.62 s. Yes.
- `E ≤ 0.6 × Eb` = 1.787 GB: 1.489 GB each. Yes.
- `F ≤` 0.34 + 5 s: 0.33 s each. Yes.

NO EFFECT rule: false for every repeat.

**Label: SAVES.**

### Validity guard

All ten rows pass (`analysis.json`, `guard`, all empty). From the step records:

1. Parity summary identical to baseline's, `passed=True`, 0 hard mismatches, 0 near-tie flips. Probe ratios
   0.354–0.372 (≤ 0.8).
2. Readers load on CPU_AND_NE with probe ratios 0.362–0.372, predictions bit-identical to the baseline reader.
   No reader created an e5rt bundle (`e5rt_reader.count` = 0 in every row).
3. Poller: 147 (screen), 149, 149 and 148 attempts during the final-path imports, every one
   `ArtifactMissingError`; `quarantine/` empty on every attempt. In final-path-L128-3 the poller saw
   `manifest.json` and stopped before loading (`manifest_seen`).
4. Injections:
   - (i) parity reported failed (screen and full): the call raised, nothing registered, final path absent, no
     pending files, `quarantine/` empty, 135 poller attempts all `ArtifactMissingError`.
   - (ii) probe raises `ComputeUnitMismatchError`: same outcome, 142 poller attempts all `ArtifactMissingError`.
   - (iii) `--force` replace with (ii): the call raised; `manifest.json` SHA-256 and `tree_sha256` identical
     before and after, no leftovers, `quarantine/` empty.
   - (iv) SIGKILL 3.0 s after the gate started: final path holds `PENDING.json` and `manifest.pending.json` and
     no `manifest.json`; a reader got `ArtifactMissingError`; the re-import succeeded in 71.5 s with no pending
     files, leftovers or quarantine entries.

### Reported, no criterion

- Reader reuses the import's compile (`F ≤ W' + 10 s`): true in every row of both arms.
- Sub-phases (full run, baseline): `S` 41.0–41.2 s, `P` 26.8–27.2 s, `Q` 3.7 s, `R` 70.3–71.2 s. The
  final-path arm has the same `S`, `P` and `Q` and no `R`. #121's `P < R` stays unexplained.
- `E` breakdown: baseline adds 4 bundles of ~745 MB and 2 of 64 bytes per import; final-path adds 2 of ~745 MB
  and 1 of 64 bytes (`e5rt_import.new` in each step record).

## Phase B re-run (decision 1)

The production change (branch `perf/validate-imports-at-registered-path`, 2be8b43) measured with #121's Phase B
method: `laya-apple artifacts import` of the L128 export archive, 3 times, each into a new empty cache. Script:
`run-phaseb.sh`. Source: `raw-2026-10-03-phaseb/`.

| | Commit | import-1 | import-2 | import-3 | New e5rt bundles per import |
|---|---|---:|---:|---:|---:|
| before | `laya_apple/` of 484a645 (run from 0d49789) | 151.10 s | 154.03 s | 149.18 s | 4, 4, 4 |
| after | 2be8b43 | 76.35 s | 91.42 s | 78.88 s | 2, 2, 2 |

Wall time from `*-summary.txt`; bundle counts from the `*-e5-*.lst` listing diffs. Before, each import log
reports the parity gate on the staged copy (29.5, 29.5, 28.6 s) and then "registered and loaded" (72.9, 73.9,
69.3 s). After, the gate runs at the registered path (29.6, 29.6, 30.5 s) and there is no second load.

import-2 after (91.42 s) is about 13–15 s slower than the other two, with the same gate time and the same 2
bundles. It is unexplained. The load average rose during the after runs (3.24 at the start, 4.35 at the end,
`after-before.txt`, `after-after.txt`); this was not an exclusive preregistered slot.

## Decisions (as preregistered)

1. **Taken.** Full-run SAVES with every guard passing. The production PR rebuilds mechanism A in
   `lifecycle.import_artifact`; its Phase B comparison is the table above.
2. `--force` replace stays on the current path. Held: the production change is first install only.
3. Not applicable (not NO EFFECT or INCONCLUSIVE).
4. Not applicable (no INVALID).
5. Held: `PROBE_MAX_RATIO`, the parity tolerances and #121's verdicts are unchanged.

## Interpretation (not a criterion)

The saving is the registered-path load: final-path `T` is about baseline `T` minus `R`, and `E` drops by
exactly 2 large bundles. Running the gate and probe at the registered path compiles once there, and a new
process (the reader) reuses that compile in 0.3 s. `S` (about 41 s, the hash and compute plan) is now the
largest remaining phase of an import and is untouched by this change.

## Limitations

- One machine, one model, n = 3 per arm (n = 1 screen).
- e5rt bundle directories are a proxy for compiles, not an API.
- Core ML keys that cache by process name. All children here are the same interpreter; a server with a
  different process name may compile again regardless.
- The prototype is a research copy of `import_artifact`; its differences are listed in [`README.md`](README.md).
- Concurrent prune is not exercised (left to the production PR's tests).
- Phase B is n = 3 per side, run outside the preregistered slot, and import-2 after is unexplained.
- Minor: during the screen's export phase (before any timed import), a review agent ran one brief `uv run`.
  While the full run was going, another agent edited code and ran one instant, standard-library-only `python3`
  string replace. Neither overlapped a timed import in a way the records show; both are noted as minor.

## Cleanup

The harness removed the e5rt entries new since each run started: 15 after the screen, 50 after the full run
(`cleanup.json`, none skipped), plus the temp caches and export directory. Phase B's temp caches and export
directory were removed by the script, and its e5rt bundles were cleared after the run. The local LLM service was
restored after the run.
