# Results: where Core ML's on-device ANE compile is reused

Issue: [#8](https://github.com/tc3oliver/laya-apple/issues/8). Preregistration and addenda:
[#121](https://github.com/tc3oliver/laya-apple/issues/121) (criteria unedited; addendum 1 platform/cache,
addendum 2 re-run after an invalid first attempt, both posted before the data they cover). Screen:
[`screen.md`](screen.md) (macOS 26.6.2, not pooled here). Raw data:
[`raw-2026-10-02-macos27/`](raw-2026-10-02-macos27/). Runners: [`run-slot1b.sh`](run-slot1b.sh)
(Phase A) and [`run-slot2.sh`](run-slot2.sh) (Phases B and C), as redacted copies.

The raw copies have local absolute paths replaced by `<wt-121>`, `<r121-raw>`, `<scratchpad>`, `<tmp>`,
`<cache>` and `<data>`. No measured value was changed: the bench JSON files are byte-identical to what the
bench wrote.

## Setup

Mac Studio M4 Max, macOS 27.0 (26A428), laya-apple 57caed2 (1.6.0, ships the macOS 27 M4 Max routing
profile, #152), coremltools 9.0, Python 3.12.14. Model laya-typed-decisions, buckets 64/96/128,
`ane_startup="wait"`. Exclusive slot, local LLM service stopped, no tests running, mains power. `sw_vers`,
`uptime`, `pmset -g therm`, `df` and the Core ML e5rt cache size were recorded before and after every arm and
phase (`machine-*.txt`). No thermal or performance warning at any point. Run on 2026-10-02: Phase A
13:16–14:15 UTC, Phase B 14:16–14:24, Phase C 14:24–14:27.

Invalid first attempt (`attempt1-invalid-9be5382/`): stopped after `fresh-copy`; all 3 rows served on the GPU
(`platform_not_validated`), no ANE probes; on 9be5382 the only validated routing profile was macOS 26.6.2 and
the bench's fresh caches do not carry the local calibration, so `auto` stayed on MLX. INVALID, counts toward
nothing; its `ready_s` (2.339, 1.244 and 1.243 s) is not a `C`.

## Phase A

`C` = median `ready_s` of `a-fresh-copy.json` = 210.152 s (0.5×`C` = 105.076 s). `W` = median `warm_ready_s`
over all 12 rows = 2.0675 s (`W` + 10 = 12.068 s). Validity guard: all 12 rows pass (ANE after ready; probes
64/96/128 with ratios 0.353–0.525; answers identical to fresh-copy row 0).

| Arm | `ready_s` r0 / r1 / r2 | Median | ÷`C` | Prep load median | Classification |
|---|---|---:|---:|---:|---|
| fresh-copy | 209.467 / 210.152 / 210.433 | 210.152 | 1.000 | — | baseline (`C`) |
| move | 87.967 / 88.473 / 87.750 | 87.967 | 0.419 | 211.425 | INCONCLUSIVE |
| same-path-recopy | 211.257 / 210.940 / 210.700 | 210.940 | 1.004 | 211.403 | RECOMPILED |
| touch | 5.693 / 5.717 / 5.692 | 5.693 | 0.027 | 211.385 | REUSED |

`move`: all rows between the bounds, SD 0.37 s: a stable partial reuse.

## Phase B

L128 exported, imported 3× into new empty caches.

| Import | `P` parity gate (s) | `R` registered + loaded (s) | `R` ÷ `P` |
|---:|---:|---:|---:|
| 1 | 26.8 | 66.3 | 2.47 |
| 2 | 26.8 | 66.4 | 2.48 |
| 3 | 26.9 | 66.6 | 2.48 |

All `R` ≥ 0.5×`P`: the import pays twice.

## Phase C (observational)

The compile is not in `DARWIN_USER_CACHE_DIR` (only Metal/Safari entries, `c-new-paths.txt`, `c-sizes.txt`).
It is in `~/Library/Caches/<process name>/com.apple.e5rt.e5bundlecache/<macOS build>/<hash>/`
(`c-e5rt-cache.txt`), as the screen found on 26.6.2. One 3-bucket cold load writes 6 bundles of 0.727 GB
+ 3 × 4 KB = 4.36 GB, equal to Phase A's per-load growth; each Phase B import wrote 2 bundles (1.45 GB).
Nothing evicts them: the run added 80.7 GB; the 26A428 directory ends at 102.8 GB / 379 entries. Other
processes keep their own e5rt caches. Sizes in GB are `du -sk` kilobyte counts divided by 10^6.

## Decisions (as preregistered)

1. Undecided; no code change (Phase B pays twice but `move` is INCONCLUSIVE, not RECOMPILED; stays-as-is
   needs `move` REUSED or pays once, and neither holds). Any change to where the parity gate runs needs a new
   separately preregistered experiment.
2. No code change; `same-path-recopy` RECOMPILED and `touch` REUSED reported; no INVALID arm.
3. Not shipped: a cache keyed by macOS build and process name.
4. Not met: `C` = 210 s ≫ 30 s. Prebuilt artifacts remove the build, not the compile;
   `ane_startup="background"` stays documented; #8's limitation stays published.

## Interpretation (not a criterion)

Reuse appears keyed on file identity, not timestamps: `touch` keeps it (5.7 s, ≈ 3.6 s above warm), new
inodes at the same path lose all of it (`same-path-recopy`), and a rename keeps the inodes but changes the
path and re-pays ≈ 42% (`move`), matching the 26.6.2 screen's ≈ 42% (114.9 of 273.5 s). Phase B's `R` (≈ 66 s,
one bucket) is ≈ ⅓ of the 3-bucket cold start, so the registered load pays roughly a full compile of that
bucket; the staged parity load is shorter (≈ 27 s). Not explained: why the staged load is cheaper.

## Limitations

One machine, model and macOS build; 3 repeats; answers compared by JSON value equality on rounded answers; the
e5rt cache was never cleared (22 GB of earlier entries were already present, and every arm used new paths);
macOS 27 timings are not compared with 26.6.2 or v0.3 except the qualitative 42% observation; Phase C
additionally recorded e5rt sizes and paths beyond the preregistered listing, following the screen.
