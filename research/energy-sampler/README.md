# Energy per decision: a no-sudo sampler and the J/decision harness

Issue: [#13](https://github.com/tc3oliver/laya-apple/issues/13). Machine: Mac Studio M4 Max
(Mac16,9), macOS 26.6.2. Model: laya-typed-decisions.

**Status: run 1 is invalid under its own criteria (see [Run 1: invalid](#run-1-invalid)). No
result from this track is claimable yet, and no cross-check data has been recorded. Run 2 is
preregistered with unchanged thresholds (see [Run 2](#run-2-preregistered)).**
`results.json` and `tables.md` are generated from `raw/`. The criteria below were written
before any measurement and are applied unchanged by `scripts/analyze.py`.

## Question

At equal offered load, how much energy does one decision cost with `device="gpu"`,
`device="ane"` and `device="auto"`, once idle power is subtracted? The existing benchmarks
measure latency and throughput only. They say nothing about whether the ANE path costs more
or less energy per decision.

## Power sources evaluated (no sudo)

Both candidates were read as the logged-in user, without sudo, from Python through `ctypes`
alone (`scripts/sources.py`). No third-party package is needed, and nothing is added to
`pyproject.toml`. Run `python3 research/energy-sampler/scripts/sources.py` to print what a
machine exposes. The observations below were made interactively on the tested machine; they
are not in `raw/`.

| | SMC (`AppleSMC` user client) | IOReport (`libIOReport`, group "Energy Model") |
|---|---|---|
| What | `PSTR`: total system power, W, type `flt ` (float32) | Cumulative energy counters per rail |
| Channels used | `PSTR` (also readable: `PDTR`, `PHPC`, `PZC0`; `PPBR` and `PMVC` do not exist on M4 Max, SMC result 132) | `CPU Energy` (mJ), `GPU Energy` (nJ), `ANE` (mJ), `DRAM` (mJ), out of 331 channels |
| Update rate | Once per second: 3 value changes in 3 s of 200 µs polling, both idle and under GPU load | Every ≤10 ms: every 10 ms delta was non-zero at idle |
| Resolution | float W | 1 mJ (CPU, ANE, DRAM), 1 nJ (GPU) |
| Cost per read | ~0.02 ms | Almost all kernel time in `IOReportCreateSamples`: ~0.1 ms user + ~3.6 ms system back to back for all 331 channels, ~2.9 ms for a 4-channel subscription; 6–11 ms per sample when taken periodically (see [Sampler](#sampler)) |
| Energy of an interval | Integral of 1 Hz readings (zero-order hold); ±1 update of edge uncertainty | Difference of two counter samples; covers the whole interval, with no aliasing |
| Per rail | No: whole system (SoC, fans, SSD, PSU losses, other processes' peripherals) | Yes: CPU, GPU, ANE, DRAM separately |

Under an MLX matmul loop, `GPU Energy` gave 36.8 W and the coarser `GPU` channel 36.9 W, while
`PSTR` read 104 W.

**Choice.** IOReport is the primary source. It measures per-rail energy directly, covers
every interval exactly, and gives ANE and GPU energy separately. The SoC energy used below is
`CPU Energy + GPU Energy + ANE + DRAM`. `PSTR` is recorded alongside as a system-level
secondary result. It is reported separately and never merged with the IOReport numbers.

**SoC dependence.** IOReport channel names and SMC keys differ by SoC and macOS version. The
names above were verified on M4 Max / macOS 26.6.2 only. `sampler.py` fails if a rail is
missing, and `sources.py` lists what another machine exposes.

**What IOReport is.** It is the SoC's own energy model, the same source `powermetrics` reads.
It is not an external meter. The cross-check below confirms that the sampler reads it
correctly. It cannot show that the model is physically accurate. `PSTR` is the independent,
coarser system-level view.

## Sampler

`scripts/sampler.py` runs as a separate process, so its CPU cost never holds the harness's
GIL:

```
python3 research/energy-sampler/scripts/sampler.py --out PATH.json [--interval 1.0] [--pstr-interval 0.5] [--seconds N]
```

- It samples IOReport every `--interval` s (default 1.0 s) and polls `PSTR` every
  `--pstr-interval` s (default 0.5 s).
- Samples are stamped with `CLOCK_UPTIME_RAW`, which is system-wide, and with the wall clock.
- It stops on SIGINT or SIGTERM and writes the JSON atomically. The keys are listed in the
  module docstring: `energy_j`, `mean_power_w`, `rails_j`, the cumulative `samples`, the `pstr`
  changes and `meta`.
- `meta.sampler_loop_cpu_frac` records the sampling loop's own CPU time as a fraction of one
  core, and `meta.sampler_loop_cpu_ms_per_sample` the same per sample.
- The sampler runs through idle windows too, so its own cost cancels in the idle subtraction.
- The energy of a window is the cumulative counter linearly interpolated at the window's
  start and end (`energy.window_energy`).

**Sampler versions.** `meta.sampler_version` records which one produced a run.

- **Version 1 (run 1).** Subscribed to the whole group (331 channels), sampled every 0.5 s,
  and built an `IOReportCreateSamplesDelta` object per sample, whose four rails were found by
  converting channel names to Python strings. The README expected about 0.7% of a core from
  a 3.3 ms back-to-back cost. Run 1 measured 11.9 ms per sample and 2.38%. A sample taken
  once per interval costs about 2.5–3 times as much as one taken back to back (below). This
  is consistent with each sample running on a core that has just woken and is not at full
  clock; the cause was not measured further.
- **Version 2 (run 2).** Subscribes to the four rails only, reads their cumulative counters
  directly (`IOReportEnergy.read`: no delta object, channel order checked with `CFEqual`,
  no string conversion per sample), samples every 1.0 s, and wakes the main thread every
  5 s instead of every 0.2 s.

Breakdown of one version-1 sample, measured back to back with no model loaded, under `nice`,
on the M4 Max while other work was running:

| Step | CPU per sample |
|---|---|
| `IOReportCreateSamples`, 331 channels | 3.6 ms (0.1 ms user, 3.6 ms system) |
| `IOReportCreateSamples`, 4 channels | 2.9 ms |
| `IOReportCreateSamplesDelta` + reading 4 cached rails | 0.24 ms |
| Delta + full walk of 331 channels with string conversion (first sample, or after a reorder) | 1.2 ms |
| SMC `PSTR` read | 0.02 ms |

The kernel call dominates, so the cost is set mostly by how often a sample is taken.
Sampler-only runs with no model, back to back under the same conditions (6 s each, twice;
then 15 s):

| Sampler | Interval | Loop CPU, % of one core | CPU per sample |
|---|---|---|---|
| version 1 | 0.5 s | 2.34%, 2.38% | 10.1, 11.0 ms |
| version 2 | 0.5 s | 1.37%, 1.32% | 5.9, 6.1 ms |
| version 2 | 1.0 s | 0.81%, 0.74% (15 s run: 0.83%) | 6.1, 6.4 ms (7.3 ms) |

Version 1 reproduces run 1's 2.38% in this setting, so the setting is comparable to the
campaign. The short runs overstate the steady-state fraction slightly, because the first
sample is counted against only a few seconds. At 1.0 s, version 2 is below the 2% limit
by a factor of about 2.5. These measurements were made interactively; they are not in
`raw/`. Criterion 3 is judged on each run's own recorded `sampler_loop_cpu_frac`.

**Why 1.0 s is enough.** A window's energy is the cumulative counter interpolated at the
window's edges, so the interval affects only the two edge intervals of a 30 s (or 60 s)
window, not the energy between them. Decimating run 1's 0.5 s series to 1.0 s and
re-running the unchanged analysis changes:

- no loaded window's net J/decision by more than 0.10%;
- no window's mean SoC power by more than 1.95%. The largest changes are in idle windows
  near 0.1 W, and the idle spread changes by at most 0.0015 W.

At 2.0 s the changes are 0.15% and 2.9%. The `powermetrics` cross-check pairs 1 s
reference intervals with the sampler's interpolated energy over the same interval, inside
phases trimmed by 3 s at each edge, so a 1 s sampler interval does not change what is
compared.

## Harness

`scripts/harness.py` uses the public API: one `Laya.from_pretrained(model, device=...)` per
config, with inline execution, and `predict(context=..., questions=...)`.

- **Configs:** `gpu`, `ane`, `auto`. All three are loaded, and every request shape is run
  once untimed, before the first window.
- **Shapes:** fixed request cycles, built with `laya_apple.workload.make_request`, which
  generates text and uses no third-party prompts:
  - `short`: single-question requests alternating L64/L128 across question types. `auto`
    routes all of them to the ANE.
  - `mixed`: L64/L128/L96 single-question requests plus one 4-question L128 request per
    cycle, 7 decisions per 4 requests. `auto` sends the 4-question request to MLX.
- **Offered load:** fixed rate, open loop, deterministic spacing 1/rate.
  - `short`: 30 req/s (30 decisions/s).
  - `mixed`: 20 req/s (35 decisions/s).
  - The slowest config's service time is about 10–12 ms per short request and about 39 ms
    for the 4-question request on the ANE, so both rates keep every config below about 40%
    busy. A request whose scheduled start has passed starts at once, and its lateness is
    recorded.
  - A decision is one answered question.
- **Windows:** 10 s warm-up (not measured), then 30 s measured. An idle window runs nothing
  for the same 40 s. From method version 2 (run 2), an idle window measures 60 s after its
  10 s settle, and a window hit by a CPU burst is repeated (see [Run 2](#run-2-preregistered)).
- **Order:** ABBA per shape, `idle, gpu, ane, auto, idle, auto, ane, gpu` × 2, then a closing
  `idle`. That gives 17 windows per shape, 4 measured windows per config, and an idle window
  on each side of every loaded block.
- **Baseline:** idle power at a loaded window's midpoint, linearly interpolated between the
  idle windows around it. This removes slow drift. Net J/decision =
  (E_window − P_idle × T) / decisions completed in the window. It is computed for the SoC,
  per rail and for `PSTR`.
- **Recorded:** per request, the scheduled, start and end times, the question count, the
  device, the routing reason, the answer and any error. The raw run also records the whole
  sampler series and the machine metadata: SoC, macOS, versions, git commit, power source,
  and whether `omlx-server` was running.

## Acceptance criteria (preregistered)

The thresholds below are also recorded in [`criteria-r1.json`](criteria-r1.json), which is
what `analyze.py` reads. [`criteria-r2.json`](criteria-r2.json) keeps every one of them.

**Sampler vs `powermetrics`.** The sampler is accepted only if all of these hold on the
cross-check run:

1. In every phase, with 3 s trimmed at each edge, the sampler's mean power agrees with
   `powermetrics`:
   - the CPU, GPU and ANE rails, and their sum, within **±5%** of `powermetrics` where the
     reference is ≥1 W;
   - within **±0.1 W** absolute where it is below 1 W.
2. Each loaded phase has **≥20** paired 1 s samples.
3. The sampler loop's CPU cost is **≤2% of one core**.

**Campaign validity.** These are reported per window and per shape, not merged:

4. A loaded window is valid only if all of these hold:
   - achieved decisions/s is within **±2%** of the offered rate;
   - P99 start lateness is ≤ one arrival interval;
   - no request raised.

   Invalid windows are excluded and counted in the tables.
5. For a shape's results to be used, its idle windows must span at most **10%** of the
   smallest net loaded SoC power in that shape. Otherwise, baseline drift is too large to
   subtract.

**Comparison.** For each shape, `ane` and `auto` are compared with `gpu`:

- **lower** or **higher**: every valid window of one config lies below (or above) every
  valid window of `gpu` in net J/decision;
- **not separated**: anything else.

The median ratio is reported with the verdict. The same rule is applied separately to the
`PSTR` result. Throughput, latency and correctness are not part of this result.

## Cross-check against `powermetrics`

Terminal 1. The operator runs this; nothing in this repository runs sudo. Run 2 uses
`-n 300` instead of the original `-n 200` (see [Run 2](#procedure)):

```
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 300 \
  -o research/energy-sampler/raw/crosscheck-powermetrics.txt
```

Terminal 2, within about 20 s:

```
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 \
  uv run python research/energy-sampler/scripts/crosscheck.py run
```

The fixed workload is:

| Phase | Duration | Load |
|---|---|---|
| idle | 20 s | none |
| GPU | 40 s | L128 single-question requests, closed loop |
| idle | 20 s | none |
| ANE | 40 s | L128 single-question requests, closed loop |
| idle | 20 s | none |

`analyze.py` (or `crosscheck.py compare`) handles the alignment:

1. It parses the `powermetrics` text samples.
2. It finds the whole-second timestamp shift that best matches the GPU rail. The header
   stamps have 1 s resolution.
3. For each `powermetrics` interval, it computes the sampler's mean power over that same
   interval, using the wall-clock stamps.
4. It averages both inside each trimmed phase and applies criteria 1–3.

## Run 1: invalid

Raw data: `raw/campaign-laya-typed-decisions.json.gz` (the campaign above, method version 1,
sampler interval 0.5 s, no cross-check run). Apple M4 Max (Mac16,9), macOS 26.6.2,
laya-apple 1.3.0, AC power, `omlx-server` not running. `results.json` and `tables.md`
reproduce it with `analyze.py --check`.

The run fails two of its own preregistered criteria:

| Criterion | Limit | Run 1 | Met |
|---|---|---|---|
| 3. Sampler loop CPU | ≤ 2% of one core | 2.38% (32.4 CPU-s over 1363 s, 2728 samples: 11.9 ms per sample) | no |
| 5. Idle spread, `short` | ≤ 10% of the smallest net loaded SoC power | spread 0.194 W vs 1.653 W (limit 0.165 W) | no |
| 5. Idle spread, `mixed` | ≤ 10% of the smallest net loaded SoC power | spread 0.447 W vs 2.397 W (limit 0.240 W) | no |

Criterion 5 makes a shape's results unusable, and it fails in both shapes. The J/decision
numbers and comparison verdicts that `tables.md` prints for this run are therefore **not
claimable**, and no verdict is drawn from them. They stay in the generated tables only
because the tables are regenerated from `raw/` unchanged. Criterion 4 (loaded window
validity) was met in every window, and criteria 1–2 need a cross-check run, which was not
recorded. The file is kept unchanged as the record of the first attempt.

## Run 2 (preregistered)

This section and [`criteria-r2.json`](criteria-r2.json) were written after run 1 was found
invalid and before any run-2 or cross-check data existed.

**Criteria.** Every acceptance threshold is unchanged: the `thresholds` in `criteria-r2.json`
equal those in `criteria-r1.json`, and a unit test checks this. Only the sampler
implementation and the idle/disturbance protocol change. `analyze.py` analyses each run under
the revision stored with it (`meta.criteria`). Run 1 records none and has method version 1, so
it is analysed under r1 and its `results.json` and `tables.md` are unchanged byte for byte. If
run 2 also fails a criterion, it is recorded as invalid in the same way. Neither r2 nor a
threshold is changed after run-2 data exists.

### What drove run 1's idle spread

From run 1's raw sampler series (0.5 s resolution) and its window stamps:

- **Two windows cause the spread.** Eight of the ten idle windows have mean SoC power
  0.119–0.163 W. Among those eight, the spread is 0.037 W in `short` and 0.025 W in `mixed`,
  against limits of 0.165 W and 0.240 W. The other two are `short` window 16 (0.320 W) and
  `mixed` window 12 (0.565 W).
- **Both are bursts of CPU work from other processes.** In `short` window 16, a single burst
  17.3 s into the measured span reached 8.6 W SoC (7.8 W on the CPU rail) for 0.5 s, and
  stayed at 1–2 W for another 2 s. In `mixed` window 12, repeated bursts of 2–8 W ran through
  the settle and the measured span. The GPU and ANE rails stayed at or below 0.02 W, and the
  harness sleeps during idle windows.
- **Not cooling after load.** After a loaded window, the first 2 s of the 10 s settle average
  0.2–0.9 W. From then on, the settle reads like the measured span that follows it, apart
  from the bursts in `mixed` window 12. The settle is long enough and stays at 10 s.
- **Loaded windows are hit too.** The same kind of burst appears in 3 of the 24 loaded
  windows (`short` windows 3, 7 and 9). This does not affect criterion 4, but it adds noise
  to net J/decision.
- Run 1 did not record which processes ran, so the bursts cannot be attributed.

The statistic used below is the **CPU-rail excess**: over a window's measured span, the mean
minus the median of the CPU rail's power in 5 s bins. A short burst raises the mean but not
the median. In run 1, the five burst windows have an excess of 0.160–0.313 W. The other 29
windows are at or below 0.044 W. Idle windows without a burst are at or below 0.014 W.

### Protocol changes (method version 2)

1. **Sampler version 2** at a 1.0 s interval (see [Sampler](#sampler)).
2. **Idle windows measure 60 s** after the unchanged 10 s settle, instead of 30 s. This halves
   how far any burst below the repeat limit can move the baseline. Loaded windows are
   unchanged: 10 s warm-up, then 30 s measured.
3. **Disturbance record, every window.**
   - The CPU rail's energy in 5 s bins over the measured span. A harness thread reads it
     through a one-channel IOReport subscription, in idle and loaded windows alike.
   - CPU seconds per process from `ps` at the window's start and end: the 10 largest,
     recorded by process name only (last path component). The harness and the sampler are
     labelled as such.
4. **Disturbance repeat rule, every window.**
   - A window whose CPU-rail excess is above **0.10 W** is repeated at once, in the same
     position. Idle and loaded windows follow the same rule, so the baseline and the loaded
     energy exclude the same kind of disturbance.
   - A window is repeated at most 2 times, and a run at most 8 times.
   - The analysis uses the last attempt of each window: the first undisturbed one, or the
     last one when a limit is reached.
   - Every attempt stays in `raw/` and in `results.json`. `tables.md` counts the repeated
     attempts, the disturbed attempts that were kept, and, for information only, the idle
     spread over all attempts.
   - The 0.10 W limit was chosen from run 1: it separates the five burst windows (≥ 0.160 W)
     from all the others (≤ 0.044 W).
5. **Quiet machine, checked by the operator before starting:**
   - no other interactive or agent sessions;
   - the local LLM server and container runtimes stopped;
   - a static wallpaper. An animated Aerial wallpaper decodes video: its extension and a
     video decoder service appeared in the process list of a short no-model smoke test of
     the disturbance record;
   - Spotlight indexing and Time Machine idle;
   - AC power, and 5 minutes without use before starting.

Everything else is unchanged: the shapes, rates, ABBA order, warm-up, loaded window length,
idle interpolation, net J/decision, the comparison rule, and `PSTR` as a separate secondary
result.

### Procedure

On the quiet machine, run the cross-check first, because it validates the sampler.

1. **Cross-check.** The operator runs this in terminal 1; nothing in this repository runs
   sudo:

   ```
   sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 300 \
     -o research/energy-sampler/raw/crosscheck-powermetrics.txt
   ```

   Then, within about 20 s, in terminal 2:

   ```
   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 \
     uv run python research/energy-sampler/scripts/crosscheck.py run
   ```

   This takes about 3–4 minutes: 140 s of phases plus loading the model. `-n 300` (5 min)
   covers the phases even with a slow model load; run 1's procedure had `-n 200`. The
   cross-check run records method version 2 and is judged under r2. The sampler's CPU
   (criterion 3) is reported for it separately.
2. **Campaign**, with the same `LAYA_APPLE_CACHE` and `HF_HUB_OFFLINE=1`:

   ```
   sh research/energy-sampler/scripts/run.sh
   ```

   It writes `raw/campaign-r2-laya-typed-decisions.json.gz`. Per shape, 5 idle windows take
   70 s each and 12 loaded windows take 40 s each, so the windows take 27.7 min for both
   shapes. With loading and the untimed pass, expect about 30 minutes. Each repeat adds 40
   or 70 s, so with all 8 repeats a run takes at most about 40 minutes.
3. **Analysis:** `uv run python research/energy-sampler/scripts/analyze.py`. Before committing
   `raw/`, read the `cpu_by_process` names: they are process names from the machine.
4. **Report** every criterion on its own, for the cross-check and for each shape, as for
   run 1.

## Reproduce

```
uv sync --extra ane --extra dev
export LAYA_APPLE_CACHE=/path/to/cache HF_HUB_OFFLINE=1
# quiet machine (Run 2, protocol change 5); AC power
sh research/energy-sampler/scripts/run.sh          # campaign, about 30 min
uv run python research/energy-sampler/scripts/analyze.py --check
```

Unit tests for the pure logic (`scripts/energy.py`, `scripts/analyze.py`) are in
`tests/unit/test_energy_sampler.py`. They are part of the fast suite:

```
uv run pytest -q tests/unit/test_energy_sampler.py
```

## Limitations

- Only one SoC and macOS version were verified. Channel names and SMC keys are SoC-specific.
- IOReport is a model estimate, not an external meter. `powermetrics` itself says its power
  values "are estimated and may be inaccurate". The cross-check validates the reading, not
  the model.
- `PSTR` updates at 1 Hz. Whether each value is an average or a point reading is not
  documented, so it is used only as a secondary, whole-window result.
- Only one process is measured, but the rails are system-wide. Any other GPU/ANE user adds
  to them. Criterion 5 rejects a run whose idle windows show that kind of interference.
- The run-2 repeat rule catches bursts of CPU work only. It is calibrated on run 1's bursts,
  so a burst below 0.10 W of CPU-rail excess, or interference on another rail, is not
  repeated. It remains in the window, and criterion 5 still applies.
- The repeat rule makes the run's length depend on the machine's background activity. The
  per-process record shows what ran, but it cannot tell which process caused a burst when
  several were active.
- Inline execution serves one request at a time. Energy under `execution="workers"`, with
  both devices busy at once, is not covered by this harness.
