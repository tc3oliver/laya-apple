# Screen result (scope addendum of #121)

Machine: Mac Studio M4 Max, macOS 26.6.2 (25G83), coremltools 9.0, laya-typed-decisions,
buckets 64/96/128, `ane_startup="wait"`, one repeat, exclusive slot with the local LLM
service stopped. Raw data: `raw/c-fresh-copy.json`, `raw/screen-move.json`,
`raw/c-new-paths.txt`, `raw/c-sizes.txt`, `raw/c-e5rt-cache.txt`,
`raw/screen-machine-{before,after}.txt`. This is the screen, not the preregistered
verdict (n = 1).

| Arm | `ready_s` | `warm_ready_s` | Preparation load |
|---|---:|---:|---:|
| fresh-copy (`C`) | 273.5 | 2.68 (`W`) | — |
| move | 114.9 | 2.65 | 271.1 |

**Validity guard: passed.** Both rows served the request after ready on the ANE. Every
bucket's placement probe ratio was ≤ 0.8 (0.35–0.50). The move row's answers are
bit-identical to the fresh-copy row's.

**Screen label: screen-INCONCLUSIVE.** The move took 114.9 s:
- that is above the screen-REUSED bound, `W + 10 s` = 12.7 s;
- it is below the screen-RECOMPILED bound, `0.5 × C` = 136.7 s.

A path change kept about 58% of the cold cost (114.9 of 273.5 s were paid again), not all of
it and not none.

**Where the compile is stored (Phase C).** Core ML writes its on-device ANE compile to
`~/Library/Caches/<process name>/com.apple.e5rt.e5bundlecache/<macOS build>/<hash>/`, not to
the per-user cache directory the plan listed. That layout makes the cache per executable
name and per macOS build.
- The screen wrote about 15.6 GB there: 14 bundles of 0.7–1.4 GB each.
- In total the cache now holds 15 GB (`python`) and 24 GB (`python3`), including entries
  from earlier runs.
- New locations add bundles that nothing in laya-apple evicts.

**Decisions from the addendum:**
- Following the addendum, the result is reported here, and the maintainer decides whether a
  repeat or Phase B is worth a slot. Nothing further ran.
- Decision 3 is unchanged: no cross-machine cache shipping. The observed layout, keyed by macOS
  build and process name, makes a shipped cache build-specific.
- Decision 4: `C` = 273.5 s, so the < 30 s cold-start target on a new machine is **not met**
  by prebuilt artifacts. The compile remains, and `ane_startup="background"` stays the way to
  serve during it.
