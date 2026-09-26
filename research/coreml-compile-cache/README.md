# Where Core ML's on-device ANE compile is reused

Issue: [#8](https://github.com/tc3oliver/laya-apple/issues/8) (cold start). Preregistration:
[#121](https://github.com/tc3oliver/laya-apple/issues/121), which holds the "Preregistered
criteria" below verbatim, opened before any data. They are not edited after a run. Status:
**planned, no data; the ~17 min screen in the scope addendum runs first**.

## Question

`benchmarks/v0.3.md` ("Cold start") measured 177–299 s from `Laya.from_pretrained(...,
ane_startup="wait")` to ready at a fresh artifact location, against 1.9–2.7 s warm. Almost
all of that time is Core ML's on-device ANE compile, which Core ML caches **per location**.
`laya-apple artifacts fetch` (prebuilt artifacts, V16-7) removes the conversion build. It
cannot remove that compile, because the compile happens on the receiving machine when the
artifact is first loaded.

1. **What keys the reuse?** Does Core ML reuse its compile after the artifact is moved
   (same inodes, new path), re-copied onto the same path (same path, new inodes), or
   touched (same path and inodes, new mtimes)?
2. **Does `artifacts import` (and so `fetch`) pay the compile twice?** The parity gate loads
   the model at the staging path. The import then renames the directory into place and
   loads it again there to pre-warm.
3. **Can a compiled `.mlmodelc`, or Core ML's compile cache, be shipped safely to another
   machine?** The export archive already ships `model.mlmodelc`, Core ML's compiled model.
   The question here is the device-specific ANE program Core ML builds from it at load time.

## What this plan can and cannot decide

- Q1 and Q2 are measured on one machine and can change laya-apple's own code. For example,
  running the parity gate at the final location if a rename loses the compile.
- Q3 cannot be promoted by any result here. Core ML documents no API to export or import its
  on-device compile. A transplanted system cache would depend on the macOS build and on
  undocumented cache keys, and would sit outside the checks laya-apple runs on an artifact.
  **A cross-machine shipped compile cache is a no-ship in advance**, whatever the timings. Phase C
  only observes where the cache lives and how large it is, so the write-up can say why.
- The row-9 and row-10 guarantees of `docs/no-silent-fallback.md` stay in force in every arm:
  - the compute plan is checked at first load and after any file change (the verification
    stamp is keyed on size, mtime and inode);
  - the runtime placement probe runs on every load.

  `PROBE_MAX_RATIO` and the compute-plan requirement are not changed.

## Method

Machine state for every phase: an exclusive hardware slot, no other benchmark or test run,
the local LLM service stopped as for other laya-apple benchmarks, and a normal power
source. Record `sw_vers`, the macOS build and `pmset -g therm` before and after each phase.
The bench script records the platform profile in every output file.

Model: `laya-typed-decisions` (buckets 64, 96, 128). All three buckets load in every run, as
in v0.3. Mode: `ane_startup="wait"`, so `ready_s` is the time until every offered bucket is
loaded, verified, compiled and probed. The v0.3 numbers were measured under an older macOS
and laya-apple version, so this plan re-measures its own baseline and compares only within
this run.

`<cache>` is the laya-apple cache holding validated, registered artifacts for the pinned
revision. The bench copies artifacts out of it and never modifies it.

### Phase A: location arms (Q1)

Order: baseline first, then the arms. Each command runs 3 repeats; each repeat starts from a
new copy of the registered artifacts.

```bash
export LAYA_APPLE_CACHE=<cache> HF_HUB_OFFLINE=1
OUT=research/coreml-compile-cache/raw
uv run python scripts/bench_coldstart.py laya-typed-decisions --modes wait --repeats 3 --location fresh-copy       --out $OUT/a-fresh-copy.json
uv run python scripts/bench_coldstart.py laya-typed-decisions --modes wait --repeats 3 --location move             --out $OUT/a-move.json
uv run python scripts/bench_coldstart.py laya-typed-decisions --modes wait --repeats 3 --location same-path-recopy --out $OUT/a-same-path-recopy.json
uv run python scripts/bench_coldstart.py laya-typed-decisions --modes wait --repeats 3 --location touch            --out $OUT/a-touch.json
```

Per row: `ready_s` (the outcome), `warm_ready_s` (the same process re-opening the location it
just loaded), `prep_load_s` (the preparation child's load, the arms only), `ane_probes`,
`after_ready_device` and `after_ready_answers`.

- **Cost:** a cold `wait` start was about 300 s for this model in v0.3.
  - `fresh-copy`: ≤ 16 min.
  - Each other arm pays one preparation compile per repeat, plus a measured start that is
    either about as long or a few seconds: ≤ 32 min each.
  - Worst case for Phase A: about 1 h 50 min.
- **Timing-sensitive:** yes.

### Phase B: import double compile (Q2)

1. Export one bucket from `<cache>`.
2. Import it into an empty cache three times, reading the phase times that `import_artifact`
   logs:
   - `parity gate ran in … s (includes the staged load)`;
   - `registered and loaded in … s`.

```bash
export HF_HUB_OFFLINE=1
EXP=$(mktemp -d)
LAYA_APPLE_CACHE=<cache> uv run laya-apple artifacts export laya-typed-decisions --length 128 --out "$EXP"
for i in 1 2 3; do
  EMPTY=$(mktemp -d)
  LAYA_APPLE_CACHE="$EMPTY" uv run laya-apple artifacts import "$EXP"/laya-typed-decisions-L128.tar.gz \
    2>&1 | tee research/coreml-compile-cache/raw/b-import-$i.log
  rm -rf "$EMPTY"
done
rm -rf "$EXP"
```

- **Cost:** up to 2 compiles per import, ≤ 3 × 10 min.
- **Timing-sensitive:** yes.

### Phase C: where the compile is stored (Q3, observational, read-only)

1. Start from a fresh copy.
2. List what the first load writes under the per-user cache directory, with sizes only.
   Nothing is copied, moved or deleted there, and file contents are not read.

```bash
export LAYA_APPLE_CACHE=<cache> HF_HUB_OFFLINE=1
MARK=$(mktemp); sleep 1
uv run python scripts/bench_coldstart.py laya-typed-decisions --modes wait --repeats 1 --location fresh-copy \
  --out research/coreml-compile-cache/raw/c-fresh-copy.json
UCACHE=$(getconf DARWIN_USER_CACHE_DIR)
find "$UCACHE" -newer "$MARK" -maxdepth 4 -print 2>/dev/null | sed "s|$UCACHE|<user-cache>|" \
  > research/coreml-compile-cache/raw/c-new-paths.txt
du -sk "$UCACHE"/* 2>/dev/null | sort -n | tail -20 | sed "s|$UCACHE|<user-cache>|" \
  > research/coreml-compile-cache/raw/c-sizes.txt
rm -f "$MARK"
```

- **Cost:** ≤ 6 min.
- **Timing-sensitive:** no, but it runs in the same slot.

## Preregistered criteria

Definitions, per arm of Phase A. Every row is one (repeat, `ready_s`) pair with `wait`, from
the bench's JSON.
- `C` = the median `ready_s` of `a-fresh-copy.json`;
- `W` = the median `warm_ready_s` over all Phase A rows.

**Validity guard (every row of every arm):**
- `after_ready_device == "ane"`;
- `ane_probes` holds all three buckets, each with `ratio ≤ 0.8`;
- `after_ready_answers` is bit-identical to the first `fresh-copy` row's.

Any failing row makes its arm **INVALID**. An invalid arm is reported, never counted as reuse.

**Classification (per arm, valid arms only):**
- **REUSED:** all 3 rows have `ready_s ≤ W + 10 s`.
- **RECOMPILED:** all 3 rows have `ready_s ≥ 0.5 × C`.
- **INCONCLUSIVE:** anything else, including repeats that disagree.

**Phase B, per import:**
- `P` = the parity-gate time;
- `R` = the registered-load time.

The import **pays twice** if all 3 imports have `R ≥ 0.5 × P`, and **pays once** if all 3
have `R ≤ 10 s`. Anything else is inconclusive.

**Decisions, fixed before the data:**
1. If `move` is RECOMPILED and Phase B pays twice, the proposal is to run the import's parity
   gate at the final location, still under the build lock, with the artifact registered only
   when it passes. It goes in its own PR, with Phase B re-run before and after. If `move` is
   REUSED, or Phase B pays once, `import_artifact` stays as it is.
2. `same-path-recopy` and `touch` are reported as they come out. They describe when an
   evicted or restored cache costs a recompile (backups, `rsync`, `artifacts import
   --force`). They change no code unless an arm is INVALID. An INVALID arm is a correctness
   finding and gets its own issue.
3. Shipping or transplanting Core ML's compile cache to another machine is **not shipped**,
   whatever the timings (see "What this plan can and cannot decide"). Phase C only
   documents why.
4. The V16-7 target of a cold start under 30 s on a new machine is **met** only if
   `fresh-copy` has `C < 30 s`. Otherwise the write-up records that prebuilt artifacts
   remove the build but not the compile. It also keeps `ane_startup="background"` (MLX within
   about 1.4 s in v0.3) as the documented way to serve during the compile, and #8's
   limitation stays published.

No criterion here is edited after the first Phase A row is written. A change of scope after
the results goes into a new, separately preregistered experiment. It never changes this
verdict.

## Scope addendum: the screen runs first

Posted on [#121](https://github.com/tc3oliver/laya-apple/issues/121#issuecomment-5848321081) before any data.

Per a maintainer policy of no multi-hour runs up front, this experiment is cut to a **screen of about 20 minutes**. Nothing has been measured yet. The criteria above are not edited. This addendum only reduces what runs first and adds screen-level labels for n = 1.

### The screen (one model, one repeat, `wait`)

```bash
export LAYA_APPLE_CACHE=<cache> HF_HUB_OFFLINE=1
OUT=research/coreml-compile-cache/raw
# Phase C (read-only). Its fresh-copy run is also the screen's baseline C.
MARK=$(mktemp); sleep 1
uv run python scripts/bench_coldstart.py laya-typed-decisions --modes wait --repeats 1 --location fresh-copy --out $OUT/c-fresh-copy.json
UCACHE=$(getconf DARWIN_USER_CACHE_DIR)
find "$UCACHE" -newer "$MARK" -maxdepth 4 -print 2>/dev/null | sed "s|$UCACHE|<user-cache>|" > $OUT/c-new-paths.txt
du -sk "$UCACHE"/* 2>/dev/null | sort -n | tail -20 | sed "s|$UCACHE|<user-cache>|" > $OUT/c-sizes.txt
rm -f "$MARK"
# Phase A, one arm: move (same files and inodes, new path)
uv run python scripts/bench_coldstart.py laya-typed-decisions --modes wait --repeats 1 --location move --out $OUT/screen-move.json
```

- **Estimated time:**
  - Phase C: about 6 min (one cold compile of ~300 s, plus the warm re-open).
  - `move`: up to about 11 min (a preparation compile of ~300 s, then a measured start that is either a few seconds or ~300 s).
  - **Total ≤ about 17 min.** It is timing-sensitive, so it needs an exclusive slot.
- **Why `move` only:** it is the arm that decides whether a location change keeps the compile. That covers both relocating a cache and `artifacts import`'s rename from staging to the final path (Q2). `same-path-recopy` and `touch` describe cache eviction on restore. They are deferred.

### Screen labels (n = 1)

`C` = `ready_s` of `c-fresh-copy.json`; `W` = its `warm_ready_s`. The validity guard above applies unchanged.
- **screen-REUSED:** `move` has `ready_s ≤ W + 10 s`.
- **screen-RECOMPILED:** `move` has `ready_s ≥ 0.5 × C`.
- **screen-INCONCLUSIVE:** anything else.

### What runs next

- **screen-REUSED:** the preregistered Phase A (3 repeats, all arms) and Phase B run as written, each in a slot approved separately.
- **screen-RECOMPILED:** no reusable compile across a path change. The rest of Phase A and Phase B do not run up front. The screen result is written up as a screen, not as the preregistered verdict, and decision 1 is not taken on n = 1. Any follow-up (for example, Phase B alone, to check whether `import` compiles twice) needs its own approval.
- **screen-INCONCLUSIVE:** the result is reported, and the maintainer decides whether a repeat is worth a slot.

Decisions 3 and 4 above (no shipped compile cache; the < 30 s target is met only if `C < 30 s`) are unaffected and can be read from the screen's `C`.

## Files

- `raw/`: the bench JSON and import logs above, unedited. A re-run goes in a new directory
  with its own date.
- `results.md`: the classification table and the decisions, written from `raw/` after the
  run.
