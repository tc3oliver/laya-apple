# Energy per decision: a no-sudo sampler and the J/decision harness

Issue: [#13](https://github.com/tc3oliver/laya-apple/issues/13). Machine: Mac Studio M4 Max
(Mac16,9), macOS 26.6.2. Model: laya-typed-decisions.

**Status: method and criteria only. No campaign or cross-check data has been recorded yet.**
`results.json` and `tables.md` are generated from `raw/` and stay empty until those runs
exist. The criteria below were written before any measurement and are applied unchanged by
`scripts/analyze.py`.

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
| Cost per read | ~0.2 ms | ~3.3 ms of CPU per sample, whether the subscription holds all 331 channels or only 4 |
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
python3 research/energy-sampler/scripts/sampler.py --out PATH.json [--interval 0.5] [--seconds N]
```

- It samples IOReport every `--interval` s (default 0.5 s) and polls `PSTR` on the same tick.
- Samples are stamped with `CLOCK_UPTIME_RAW`, which is system-wide, and with the wall clock.
- It stops on SIGINT or SIGTERM and writes the JSON atomically. The keys are listed in the
  module docstring: `energy_j`, `mean_power_w`, `rails_j`, the cumulative `samples`, the `pstr`
  changes and `meta`.
- `meta.sampler_loop_cpu_frac` records the sampling loop's own CPU time as a fraction of one
  core. At 3.3 ms per sample, the expected cost at 0.5 s is about 0.7%.
- The sampler runs through idle windows too, so its own cost cancels in the idle subtraction.
- The energy of a window is the cumulative counter linearly interpolated at the window's
  start and end (`energy.window_energy`).

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
  for the same 40 s.
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

Terminal 1. The operator runs this; nothing in this repository runs sudo:

```
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 200 \
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

## Reproduce

```
uv sync --extra ane --extra dev
export LAYA_APPLE_CACHE=/path/to/cache HF_HUB_OFFLINE=1
# stop other GPU/ANE users first (a local LLM server included); AC power
sh research/energy-sampler/scripts/run.sh          # campaign, about 25 min
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
- Inline execution serves one request at a time. Energy under `execution="workers"`, with
  both devices busy at once, is not covered by this harness.
