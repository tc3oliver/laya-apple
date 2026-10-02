# Slow-state trigger screen: results

**Status: run (2026-10-02, macOS 27.0). Outcome: PB-H, PB-W and PB-P are each confirmed as removing
the slow state. No single cause is attributed.**
- **The positive control reproduced.** PB-R, the #88 path, was slow in 5 of 6 hetero windows. The
  negative control, A, was slow in 0 of 6.
- **Three changes, each one difference from PB-R, each removed it.** PB-H (the same native call
  with the GIL held), PB-W (a fixed 2 s hetero warm-up before each hetero window) and PB-P (a 1 ms
  GIL probe thread) each had 0 of 6 slow windows. PB-H's contrast is wider than "the GIL"; see the
  limitations.
- **What follows.** The criteria ([`criteria.md`](criteria.md), unchanged since it was committed)
  report each confirmed reading and rank none. There is no automatic third round: the rule's next
  step is "none", so anything further is a new human decision.
- **The numbers:** [`tables.md`](tables.md), [`results.json`](results.json),
  [`background.md`](background.md) and [`raw/check.json`](raw/check.json). The per-run raw files are external evidence; see
  [Evidence](#evidence).

## Outcome, as the preregistered rule printed it

`scripts/analyze.py` produced this text from the raw runs ([`tables.md`](tables.md), copied
verbatim):

> Outcome: round 2: PB-H confirmed; PB-W confirmed; PB-P confirmed (PB-R 5 of 6 slow, A 0 of 6 slow)
>
> - PB-H: GIL release is one necessary condition for the slow-state transition; next: who takes CPU/GIL during the GIL-released period
> - PB-W: hetero transition / scheduler residency state is key; next: minimal conditioning/state-retention mechanism
> - PB-P: the 1 ms probe changes scheduler/CPU residency; next: minimal wake mechanism and its overhead; the probe is not a production fix
>
> Next: none

Crashed runs and re-runs (`raw/failed/`): none. Runs not used by the rule: none.

## Setup

- **Question and cells.** As in [`README.md`](README.md) and [`criteria.md`](criteria.md): laya only,
  L128 short and L512 long requests, R1's full protocol, 3 hetero windows per run, one fresh process
  per run. Cells A (production coremltools, GIL held), PB-R (R1's PB), PB-H (PB-R's prebound path
  with the GIL held), PB-W (#83's 2 s hetero warm-up) and PB-P (#83's 1 ms GIL probe). A hetero
  window is slow if its short P99 is at least 13.0 ms. That classifies state only.
- **Criteria addendum.** [Addendum 1 on #89](https://github.com/tc3oliver/laya-apple/issues/89#issuecomment-5954732480)
  was posted at 14:33:54 UTC, before the first run at 14:34:43 UTC. It left the criteria unchanged
  and recorded the platform and code below. The criteria were committed at `0420cd9` on the branch
  `research/slow-state-trigger`, which is kept: the repository squash-merges, so that branch is the
  durable record of the commit. They are rebased here as `6ee5f7b`, with an identical patch-id.
- **Platform.** One Mac Studio, Apple M4 Max, macOS 27.0 (26A428). Python 3.12.14, coremltools 9.0,
  MLX 0.32.2, pyobjc-framework-CoreML 12.2.2 (every run records these versions).
- **Code.** `laya_apple` 1.4.0 at `0420cd9b234498ea026e2d50e176d0faad94e3b5`, the preregistration
  and harness commit on `research/slow-state-trigger`. No harness file was edited. This is not
  `main` (1.6.0), where 1.5's adaptive execution is the default for this configuration.
- **Dates.** 2026-10-02, 22:34 to 23:21 local time (14:34 to 15:21 UTC).
- **Configuration check** (before the pre-campaign check; not data). The addendum required the 1.4
  auto instance to report the local macOS 27 routing profile and ANE buckets 64, 96 and 128, and the
  campaign would not start otherwise. The check passed on the local profile. Each run's own
  `auto_info` records the same: routing profile `local:<cache>/profiles/Apple_M4_Max-macos27-coremltools9.0.json`,
  ANE buckets [64, 96, 128], `ane_ready` true, no ANE load errors, execution `workers`, ANE
  placement `thread`.
- **Pre-campaign check** (`raw/check.json`, `all_ok` true). All four preregistered conditions held:
  - PB-R is bit-identical to coremltools on every laya ANE bucket (L64, L96, L128), at model and
    `ANEBackend.forward` level;
  - PB-H is bit-identical on the same terms;
  - PB-R releases the GIL: the main-thread 1 ms probe's P50 lateness was 0.338 ms, against a limit
    of 1 ms;
  - PB-H holds the GIL: the probe's P50 lateness was 8.947 ms, with native `predict` P50 at
    9.487 ms. The limit is at least half the native time.
- **Run order.** Round 1: A, PB-R, PB-H, PB-W, PB-P. Round 2 applied because PB-R's round-1 run was
  slow and A's had no slow window. It ran A, PB-R and the three candidates (each was 3 of 3 normal
  in round 1), in the same fixed order. Every run took about 276 s (PB-W 282 s).
- **Machine.** Preregistered: idle, on AC power, the local LLM server stopped (stated in addendum 1;
  not recorded in the raw files). The recorded `conditions_before` samples show background load; see
  the limitations and [`background.md`](background.md).

## Results

Hetero windows, short P99 per window in cycle order. A window at or above 13.0 ms is slow
(bold). GPU return P50 is per run: the time from the end of GPU service to the reply reaching the
parent, from the request trace.

### Round 1

| cell | slow windows | short P99 per window, ms | GPU return P50, ms |
|---|---|---|---|
| A | 0 of 3 | 11.74, 11.94, 11.35 | 4.678 |
| PB-R | 2 of 3 | 12.13, **16.58**, **16.96** | 0.051 |
| PB-H | 0 of 3 | 10.53, 10.89, 10.54 | 3.693 |
| PB-W | 0 of 3 | 10.38, 10.43, 10.38 | 0.033 |
| PB-P | 0 of 3 | 10.40, 10.39, 10.39 | 0.034 |

### Round 2

| cell | slow windows | short P99 per window, ms | GPU return P50, ms |
|---|---|---|---|
| A | 0 of 3 | 10.98, 11.41, 11.08 | 4.694 |
| PB-R | 3 of 3 | **16.52**, **16.88**, **16.91** | 0.239 |
| PB-H | 0 of 3 | 10.58, 10.71, 10.60 | 3.674 |
| PB-W | 0 of 3 | 10.42, 10.40, 10.38 | 0.037 |
| PB-P | 0 of 3 | 10.41, 10.47, 10.47 | 0.037 |

### Both rounds

| cell | slow windows | GPU return P50, ms (pooled) |
|---|---|---|
| A | 0 of 6 | 4.685 |
| PB-R | 5 of 6 | 0.174 |
| PB-H | 0 of 6 | 3.685 |
| PB-W | 0 of 6 | 0.035 |
| PB-P | 0 of 6 | 0.035 |

All ten runs have 0 mismatches against the inline references and 0 routing failures. Every
hetero-window `predict` of the PB cells carries the binding's native stamps (for example PB-R round
2: 5044 of 5044). The full per-window tables, with the ANE stages and CPU columns, are in
[`tables.md`](tables.md).

## Findings

### Preregistered

These come from the rules in [`criteria.md`](criteria.md), applied as written.
- **The pre-campaign check passed**, so the campaign ran.
- **Round 1 required a round 2.** PB-R was slow (2 of 3 windows) and A had no slow window. PB-H,
  PB-W and PB-P were each 3 of 3 normal, so all three were candidates.
- **All three candidates are confirmed.** A candidate needs 0 of 6 slow windows, PB-R at least 4 of
  6 and A 0 of 6. PB-R had 5 of 6, A 0 of 6, and each candidate 0 of 6.
- **The readings** are the three in [the outcome above](#outcome-as-the-preregistered-rule-printed-it).
  Each is a statement about a trigger, not a fix. The criteria say that with more than one confirmed
  candidate each reading is reported and none is ranked, and PB-P's reading adds that the probe is
  not taken as a production fix.
- **The next step is "none".** No third round runs automatically.

### Descriptive (not gating, not preregistered readings)

Every number below is in [`tables.md`](tables.md) or [`results.json`](results.json), or is a mean
or range of values listed there.
- **PB-R's slow windows are far above the others.** Their short P99 is 16.52 to 16.96 ms. PB-R's one
  normal window is 12.13 ms. Every window of the other four cells is at or below 11.94 ms (A 10.98
  to 11.94, PB-H 10.53 to 10.89, PB-W 10.38 to 10.43, PB-P 10.39 to 10.47).
- **Host CPU work is inflated in PB-R's slow windows.**
  - GPU-thread CPU per forward averages 6.27 ms over the five slow windows (3.15 to 8.74 ms). The
    other cells, pooled over their six windows, are 1.698 ms (PB-H), 1.645 (PB-W), 1.674 (PB-P) and
    1.749 (A). PB-R's one normal window is 1.867 ms.
  - ANE-thread CPU per forward averages 1.36 ms over the slow windows, against 0.384 (PB-H), 0.362
    (PB-W) and 0.371 (PB-P).
  - Client-short thread CPU is 537 to 1309 ms per slow window, against 284 to 313 ms in every
    window of the other four cells (332 ms in PB-R's normal window).
- **Native `predict` moves much less than the host stages.** PB-R's pooled native P50 is 9.634 ms,
  against 9.427 ms for PB-H (PB-W 9.442, PB-P 9.436). Per window, native P50 is 9.458 ms in PB-R's
  normal window, 9.454 and 9.474 ms in two slow windows, and 9.81 to 9.84 ms in the other three
  (about +0.36 to +0.38 ms, 4%). In those three, the features stage P50 is 0.79 to 0.80 ms against
  0.16 ms and the tail stage 0.37 to 0.38 ms against 0.06 ms.
- **The slow windows are not all alike.** Two of PB-R's five (round 1 cycle 1 and round 2 cycle 0)
  show the 16.5 ms short P99 with the features stage still at 0.158 ms, CPU per forward about
  twice normal (GPU thread 3.55 and 3.15 ms) and aggregate throughput of 120.1 and 122.0 req/s. The
  other three show 5 times the features stage, GPU-thread CPU of 7.85 to 8.74 ms and 101.3 to
  103.5 req/s. This screen does not test why.
- **GPU completion isolation, per run.** GPU return P50 is 0.033 to 0.037 ms in PB-W and PB-P,
  0.051 and 0.239 ms in PB-R, 3.67 to 3.69 ms in PB-H and 4.68 to 4.69 ms in A. PB-W and PB-P
  therefore kept the GIL-released isolation with 0 of 6 slow windows each. PB-H holds the GIL, so it
  still has the GPU reply wait (3.685 ms pooled, against A's 4.685). It is a causal control, not a
  candidate execution path.
- **Throughput.** Aggregate hetero req/s is 120.7 to 121.8 in A, 124.4 to 124.9 in PB-H, 128.6 to
  129.3 in PB-W and 128.5 to 128.9 in PB-P. PB-R's windows run from 101.3 to 127.7.
- **PB-P's probe.** Its lateness P50 is 0.340 to 0.342 ms and P99 0.573 to 0.585 ms in every
  hetero window, over 272,649 and 272,609 ticks in the two runs.
- **PB-W's warm-ups** sent 198 or 199 short and 56 long requests before each hetero window, with 0
  mismatches.

## Limitations

- **n = 6 windows per cell.** Each cell has two runs of three hetero windows. The outcome rule is a
  count rule; no statistical test was preregistered, and none is claimed. PB-R's own count is 5 of
  6, so the positive control is itself not 6 of 6.
- **One machine and one OS version.** Every run is on one M4 Max on macOS 27.0. R1's slow state was
  observed on macOS 26.6.2, and this screen says nothing about 26.6.2. It shows the state on
  macOS 27.0 for PB-R.
- **Three confirmed candidates, so no single-cause attribution.** Each of the three changes, one
  difference from PB-R each, coincided with 0 of 6 slow windows. The screen cannot say whether they act through one
  shared mechanism, such as scheduler or CPU residency, or separately. The criteria rank none, and
  neither does this page.
- **PB-H's contrast is wider than "the GIL".** PB-H differs from PB-R only in how the one native
  call is bound ([`criteria.md`](criteria.md)). Holding the GIL also puts back the GPU reply wait
  (GPU return P50 3.685 ms), so every downstream effect of that wait belongs to the same contrast.
  "One necessary condition" is the preregistered wording for this contrast. PB-W and PB-P release
  the GIL and stay normal, so GIL release alone is not shown to be sufficient.
- **The PB path under 1.4 only.** The cells test the synchronous no-GIL thread path under 1.4. They
  do not test 1.5's adaptive execution (PB-ASYNC with the breaker), and the result neither validates
  nor invalidates it. laya-typed-decisions and laya-multilingual were not run; R1's
  typed-decisions PB was 0 of 6 slow.
- **Fixed run order.** A, PB-R, PB-H, PB-W, PB-P in both rounds, as preregistered. Cell is therefore
  aligned with position in time, and machine history (time since boot, thermal state) is not
  separated from the cell.
- **Background load was present, and it is a confound for all three readings.** The raw runs record
  the machine's load averages and its five busiest processes (`ps -Ao %cpu=`) once before each
  window, after the 2.0 s idle and before the window, and never during one. [`background.md`](background.md)
  reduces all 120 windows of the ten runs; `scripts/background.py` generates it from the raw files.
  `ps` `%cpu` is a decaying average over up to a minute, with 100 as one core, so a value says a
  process was recently busy, not that it ran during the window. Only the top five are kept. What the
  samples show, descriptively:
  - **A constant background.** WallpaperAerialsExtension is in the top five before all 120 windows
    (6.0 to 14.0%), VTDecoderXPCService before 115 (3.2 to 5.3%), OrbStack Helper before 115 (2.4 to
    93.4%) and WindowServer before 109 (2.6 to 9.1%). Processes named `claude` (51 windows, 0.9 to
    42.1%) and `python` (29 windows, 2.1 to 6.8%) are recorded by name only; the data do not say which
    instance. The first three appear in every window group, including the cells with no slow window,
    so they do not separate the cells. Together they mean the machine was not idle.
  - **Load.** The 1 minute load before a hetero window is 0.96 to 1.72 in the cells with no slow
    window, 1.04 before PB-R's normal window and 1.12 to 2.07 before its slow windows. The two
    highest of all hetero windows are PB-R's (2.07 before round 2 cycle 0, 1.77 before cycle 1). The
    load average includes the benchmark's own threads, so it does not isolate a background cause.
    The largest rise between consecutive samples of any run, 1.29 to 2.63, is in PB-R round 2, during
    cycle 0's solo_short window. The next window of that run is its first hetero window, slow at
    16.52 ms with 2.07 before it. No process in either of those samples is above 8.8%.
  - **Spikes.** A process at or above 50% appears in the sample before 4 of the 24 hetero windows of
    the cells with no slow window (A round 2 cycle 0: mobileassetd 98.5%; PB-P round 1 cycle 2:
    Google Chrome Helper (Renderer) 95.0%; PB-P round 2 cycles 0 and 2: OrbStack Helper 92.7% and
    88.4%), all normal, and before none of PB-R's six. A process at or above 20% appears before 7 of
    those 24, 0 of PB-R's one normal window and 1 of its 5 slow windows (round 2 cycle 1: OrbStack
    Helper 36.5%). The other four slow windows have a highest sample of 8.1 to 11.0%. Earlier in the
    same cycles, before PB-R's solo_short windows, launchd is at 30.9% (round 1 cycle 2) and
    PerfPowerServices at 70.1% with runningboardd at 59.3% (round 2 cycle 2); those are not in the
    hetero windows' own samples.
  - **What this does and does not say.** No background spike coincides with PB-R's slow windows more
    often than with the windows that stayed normal, so the samples do not point to one. They also
    cannot see activity during a window, and background activity was neither controlled nor
    excluded. It remains a possible confound for each of the three readings, which compare cells run
    in a fixed order over 47 minutes on a machine that was never idle. No causal claim is made.
- **The 13.0 ms classifier states a window's state; it is not a threshold.** The one PB-R window at
  12.13 ms is above every window of the other cells and below the split.
- **Absolute paths were redacted from the committed data.** Each raw run recorded the local routing
  profile as an absolute path to the local cache directory, which sits on the external data volume.
  In all ten raw files that cache directory prefix was replaced with `<cache>`, and only that string
  changed. Decompressed, each file is byte-identical to the original except for those characters,
  and `analyze.py --check` reproduces `results.json` and `tables.md` from the redacted files. The
  hashes below are of the redacted files.

## Evidence

The ten raw run files total 10,178,485 bytes (9.71 MiB), and four of them are over the 1 MiB cap
(PB-P round 1 and 2, PB-W round 1 and 2). By [`docs/evidence.md`](../../docs/evidence.md) a pull
request that adds more than 5 MiB of raw data publishes the bulk externally, so all ten are
external evidence. They are on the GitHub release `evidence-coreml-slow-state-trigger-20261002` and
listed in [`evidence.json`](evidence.json). A downloaded file counts only if its SHA-256 matches:

| path | size, bytes | SHA-256 |
|---|---|---|
| `raw/laya-A-r1.json.gz` | 856,044 | `7574cec1533a3b1f0d93039170ceccc8d064cb4d12c04720215d48231352c04a` |
| `raw/laya-A-r2.json.gz` | 857,771 | `5fd3c839209272d91bb3bd011630b9581deccf86c885c9c54a0f4857c61e4445` |
| `raw/laya-PB-H-r1.json.gz` | 1,037,815 | `ce5bea6db30bc3b31ebe92aebd3acff0008faceec1ec35dd16615ad7165cfb12` |
| `raw/laya-PB-H-r2.json.gz` | 1,040,369 | `30cc7fca1e481c658d935bea8211d0a56202845272d84657c2f23ca3b2ac3419` |
| `raw/laya-PB-P-r1.json.gz` | 1,120,135 | `ddc58d163dfd9c228d0fcd9d6723c534dec4871d7404a683fa26f0449de8b562` |
| `raw/laya-PB-P-r2.json.gz` | 1,119,301 | `4983ce3929b0a28d8f5e547b36b8bb684d88474eeece6e39a1f09e1c087f5625` |
| `raw/laya-PB-R-r1.json.gz` | 1,004,749 | `aa628d0146de5f07935be8d8dd6dc2cd287670c180d6d2879f0d4ca9881d650f` |
| `raw/laya-PB-R-r2.json.gz` | 970,241 | `3c5f70021196253ccc7dc9cce4cb0b90b911981977fd2682c3d1c752ad8e99e7` |
| `raw/laya-PB-W-r1.json.gz` | 1,084,175 | `8727e1428d90bec9fde8cb967102b53f3d1a48f26e302ff87a27f3719f5557ed` |
| `raw/laya-PB-W-r2.json.gz` | 1,087,885 | `d6273128a06907ffc99e35c6c568c10f289dd11b16f614e29aea7887f927c729` |

Everything in this page can be read without downloading them. They are needed only to re-derive
`results.json`, `tables.md` and `background.md`: fetch each file to its `path` under this directory
as [`docs/evidence.md`](../../docs/evidence.md) describes, then run
`uv run python research/coreml-slow-state-trigger/scripts/analyze.py --check` and
`python research/coreml-slow-state-trigger/scripts/background.py --check`.
