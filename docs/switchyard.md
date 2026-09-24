# Switchyard

`laya-apple switchyard` is an open-loop benchmark shown as a rail junction. Trains arrive
on a seeded timetable. Each train needs one Laya decision: which platform it goes to.
Long requests arrive in the background at the same time and load the GPU. The benchmark
runs the same timetable twice, once on the GPU only and once on the GPU plus the Neural
Engine (ANE). It then reports how many trains got their decision late.

The benchmark runs headless. The browser only shows a replay of the recorded run, never
a live measurement.

```
laya-apple switchyard [--seed N] [--duration S] [--out DIR] [--no-open] [--setup-ane] [--replay DIR]
```

This document states the rules and the measurement method before any number is
published. It contains no performance results.

## Rules of the game

- **Train.** One `choice` question: `ROUTE A / B / C`. Exactly one platform is clear.
  Each of the other two is either occupied by another train or closed for maintenance.
  The clear platform is the correct answer (the *oracle*).
- **Late.** The decision latency is over the deadline. The deadline is **100 ms**,
  preregistered here before any Switchyard measurement.
- **Misrouted.** The decision is on time, but the answer is not the oracle.
- **Delivered.** The decision is on time and correct.

"Late" wins over "misrouted": a late train counts as late whatever its answer. Every run
also reports the miss rate at **25, 50, 100, 250 and 500 ms**, so the result never
depends on the 100 ms choice alone. The deadline and these thresholds are fixed for
workload `switchyard-v1`. Changing them needs a new workload version.

## Workload `switchyard-v1`

| Parameter | Value |
|---|---|
| Model | `laya-typed-decisions`, pinned revision (`laya_apple/registry.py`) |
| Arrivals | bursty: 1 s at 3x the rate, then 2 s off. Nominal mean rate 40 req/s. `laya_apple.workload.arrivals`, bit-identical to `scripts/bench_concurrency.arrivals` |
| Timed window | 60 s per round |
| Warmup | 5 s, with its own schedule (seed + 100), not measured |
| Mix | 60% train (1 question, ≤ 128 tokens) · 20% `medium_1q` (L512) · 10% `long_1q` (L1024) · 10% `short_4q` (L128, 4 questions) |
| Background requests | `laya_apple.workload.make_request` with seeds 2 / 3 / 4, as in v1.0 part B. No deadline |
| Seeds | default 11. Arrivals use `Random(seed)`, class picks `Random(seed + 1)` and train contents `Random(seed + 2)` |
| Deadline | 100 ms |
| Drain | each round waits for every request to finish, with a 600 s timeout |

A run with any seed or duration other than the default is recorded as
`"standard": false`. `mode` is a separate field: `"standard"` is the only value in v1,
and `"live"` is reserved for a future showcase mode. `standard` is the comparability
flag. `mode` says how the run was made.

`submission_eligible` is true for every standard run, whether or not the GPU + ANE round
ran, so GPU-only Macs can submit too.

With the default seed, the 60 s schedule has 2410 requests, 1422 of them trains. Its
`schedule_sha256` is `bf84ed5699072c8d561ae8c65b61bd6fd71b89c2d467f2b2c5686caad0ccf99d`
(`world.schedule_sha256(world.schedule())`).

### Train prompt (frozen)

```
context:  Line 3 inbound. Train 4127 is approaching the junction. Platform A is occupied by
          train 2290. Platform B is closed for maintenance. Platform C is clear and open.
question: {"platform": {"type": "choice",
           "instructions": "Train 4127 must be routed to the clear platform. Which platform is clear?",
           "criteria": ["A", "B", "C"]}}
```

The line (1–6), the train id and the ids of the occupying trains vary with the seed. The
platform sentences always come in the order A, B, C. With one clear platform and two
platforms that are each occupied or closed, there are 3 × 2 × 2 = 12 core patterns.
`prompt_template_sha256` in `result.json` fingerprints this wording
(`a7569c1f9b8e5592b7fc1e7a14f78d099ba26cee64037fa596998db45084e7fe`).

### Oracle gate

The gate is 12 patterns × 20 surface variants (`world.oracle_cases()`, seed 0). The
argmax must equal the oracle for 100% of cases, on the GPU and on the ANE. Every train
prompt must also be ≤ 128 tokens, so that `auto` can route it to the ANE. The check is
`tests/integration/test_switchyard.py`. With the frozen wording, train prompts are 61–67
tokens.

The wording was the only thing allowed to change to pass the gate. The checkpoint was
not changed. Every wording tried is listed below. Each row changes only the listed
sentence(s) of the original contract wording and keeps the rest.

The original contract wording:
- Header: "Line {line} inbound. Train {train} is approaching the junction."
- Occupied platform: "Platform {p} is occupied by train {other}."
- Closed platform: "Platform {p} is closed for maintenance."
- Clear platform: "Platform {p} is clear."
- Question: "Which platform should train {train} be routed to?"

Accuracy is the share of cases where the argmax matched the oracle, on the 240-case gate
set. The minimum margin is the smallest gap between the oracle's probability and the
runner-up's, over the correctly answered cases. The held-out set is 12 patterns × 50
variants from seed 1 (600 cases). It was run only for wordings that passed the gate.

| # | Wording change | GPU gate | ANE gate | Min margin (gate) | GPU held-out (min margin) |
|---|---|---:|---:|---:|---|
| 1 | none (original contract wording) | 160/240 | 160/240 | 0.018 | — |
| 2 | question: "Which platform is clear for train {train}?" | 238/240 | — | 0.025 | — |
| 3 | question: "Which platform is clear?" | 231/240 | — | 0.002 | — |
| 4 | question: "Train {train} must be routed to the clear platform. Which platform is clear?" | 237/240 | — | 0.032 | — |
| 5 | question: "Which platform is free for train {train} to enter?" | 230/240 | — | 0.013 | — |
| 6 | clear: "Platform {p} is clear and open." | 190/240 | — | 0.017 | — |
| 7 | clear: "Platform {p} is clear and available for train arrivals." | 189/240 | — | 0.003 | — |
| 8 | closed: "Platform {p} is closed." | 163/240 | — | 0.001 | — |
| 9 | occupied: "Platform {p} is blocked by train {other}.", closed: "Platform {p} is closed." | 215/240 | — | 0.001 | — |
| 10 | criteria with descriptions `{"A": "platform A", "B": "platform B", "C": "platform C"}` | 184/240 | — | 0.023 | — |
| 11 | #2 + #6 | 240/240 | — | 0.102 | 600/600 (0.048) |
| 12 | #2 + #6 + occupied "blocked by train {other}" | 240/240 | — | 0.112 | 600/600 (0.029) |
| 13 | question "Which platform is clear and open for train {train}?" + #6 | 240/240 | — | 0.076 | 600/600 (0.071) |
| 14 | **#4 + #6 (frozen)** | **240/240** | **240/240** | 0.129 (ANE 0.128) | 600/600 (0.116); ANE 600/600 (0.123) |

Rule used to choose: among the wordings at 100% on the gate set, take the one with the
largest minimum margin on the held-out set. This makes a near-tie flip between GPU FP16
and ANE FP16 as unlikely as possible. Row 14 was then checked on the ANE.

Measured on an Apple M4 Max, macOS 26.6.2, laya-typed-decisions at `f9ab0b2`. The GPU
column uses MLX FP16 and the ANE column uses the validated ANE artifacts, both run
inline. The wording is now frozen. Any change to it is a new workload version.

Source: `benchmarks/switchyard/oracle/gpu.json` (every wording) and
`benchmarks/switchyard/oracle/ane.json` (wordings 1 and 14), written by
`scripts/switchyard_oracle.py`. Each file keeps every case: pattern, line, train id,
oracle, answer, margin and token count, plus the platform profile and model revision.
The margins in the table are rounded to three decimals.

## Measurement boundaries

All timestamps are `time.monotonic_ns()`, the same clock axis as
`laya_apple.trace.RequestTrace`.

| Quantity | Definition |
|---|---|
| arrival | `arrival_ns = round_start_ns + scheduled offset` |
| decision latency | `response_ns − arrival_ns`. Includes submit lag, queueing, inference and formatting |
| queue wait | `RequestTrace.queue_ms` (`dispatch_ns − queue_enter_ns`) |
| inference | `RequestTrace.service_ms` (`service_end_ns − service_start_ns`) |
| submit lag | `submit_ns − arrival_ns` |
| overhead | decision latency − queue wait − inference |

The replay timeline and the queue-wait metrics use different starting points. On the
timeline, a request counts as waiting from its arrival until its inference starts
(`service_start_ns − arrival_ns`), so the drawn wait includes the submit lag. The "queue
wait" metrics in `result.json` and on the result card are `queue_ms = dispatch_ns −
queue_enter_ns`, the time spent in the device's FIFO only.

- **Driver.** Open-loop, modelled on part B of `scripts/bench_concurrency.py`. One
  submitter thread waits for each scheduled arrival with the same loop as that script
  (`time.sleep(0.0002)` until the arrival time). A spin loop would hold the GIL against
  the thread-placed ANE dispatcher in the same process and slow it down. It then calls
  `Laya.submit` and never waits for a response before the next arrival. `Laya.submit`
  tokenises the request on the submitter thread, as the product does; that time is part
  of the submit lag. `secondary.submit_lag_all_requests` (every request, background
  included) shows how closely the open loop kept to the timetable.
- **Configurations.** Both run through `Laya(execution="workers")`. `gpu_only` uses
  `device="gpu"`. `hybrid` uses `device="auto"` with the measured ANE placement.
- **Tracing.** The trace callback only puts each `RequestTrace` on a `SimpleQueue`.
  After the drain, every trace is joined to its result on `request_id`. A request that
  fails, times out or has no trace fails the round. Nothing is dropped silently.
- **Percentiles.** Computed with `laya_apple.benchmark.stats`, per round.
- **Which requests count.** Train metrics use trains only; every scheduled arrival falls
  inside the timed window. Background classes are reported separately. Pooled summaries
  (`configs.<label>.summary`) concatenate the samples of every round with that
  configuration. They never average percentiles.
- **Separate results.** Late trains, decision latency, queue wait, correctness
  (misrouted, `route_accuracy`, `decision_disagreements`) and throughput
  (`decisions_per_s`, `completed_req_s`) are reported as separate results. They are never
  merged into one pass/fail.
- **Conditions.** Each round records its start and end conditions: the load average, the
  five busiest processes (CPU %, executable name only) and the `pmset -g therm` notes. It
  also records other running laya-apple processes (the `laya-apple` console script,
  `python -m laya_apple…` or a `scripts/bench_*.py` in the first two arguments), by pid
  and executable name only. Command lines are never recorded.
- **Duration.** `--duration` must be a whole number of burst cycles
  (`workload.burst_period_s`, 3 s): 3, 6, 9, … s. Any other value is refused before
  anything loads. `workload.offered.req_s` is the rate actually offered over the window.
  It equals the nominal mean (`nominal_rate_req_s`, 40 req/s) only over whole burst
  cycles: a 1 s window, for example, is all burst and would offer about 115 req/s.

## Round order

With an even seed, `gpu_only` runs first. With an odd seed, `hybrid` runs first. The
default seed, 11, therefore runs `hybrid` first. There is no counterbalancing within a
run. The order is recorded in `design`
(`{"ordering": "seed_parity", "counterbalance": "none", "sequence": [...]}`). One `Laya`
is closed before the next is loaded, so the two configurations never share the machine.
Both rounds use the same timetable. `comparison.same_schedule_sha256` confirms that from
what each round actually submitted. It is `null` when fewer than two rounds ran.

## Why the GPU + ANE number is not the v1.0 number

The v1.0 open-loop benchmark (`benchmarks/v1.0/concurrency-laya-typed-decisions.json`,
part B) reports a GPU + ANE short-request P50 of 11.1 ms. Switchyard's GPU + ANE decision
latency is not comparable with it, and must not be compared without saying so:

| | v1.0 part B | Switchyard v1 |
|---|---|---|
| Rate | bursty at 35 req/s (its rates were 20 and 35) | bursty at 40 req/s |
| Latency starts at | `Laya.submit` (`runtime.latency_ms`) | the scheduled arrival (includes submit lag) |
| Window | 20 s | 60 s |
| Class sequence | `Random(7)` picks | `Random(seed + 1)` picks, seed 11 |
| Short request | exactly 128 tokens (ANE bucket 128) | 61–67 tokens (ANE bucket 64) |

At 40 req/s bursty, about 70 trains arrive in each burst second. At about 11 ms of ANE
time each, the ANE is busy about three quarters of every burst, so trains queue behind
each other on the ANE. Late trains in the GPU + ANE round are an honest result of the
preregistered workload. The rate was not changed after seeing them.

## ANE states

The ANE state is decided from the runtime's own checks only: `_coremltools_available()`,
`platform_validated()`, a local calibrated profile and `artifacts.load_verified` for the
auto buckets.

| State | Condition | Behaviour |
|---|---|---|
| `ready` | coremltools, verified auto-bucket artifacts and a validated or calibrated profile. Also, the warmup routed at least 1 train to the ANE | runs `gpu_only` and `hybrid` |
| `setup_available` | Apple silicon, but coremltools, the artifacts or the calibration are missing. Also used when the warmup sent no train to the ANE | runs `gpu_only` only. Reports the reason and the setup command |
| `unavailable` | an artifact failed parity on this Mac, or Core ML cannot run the ANE path | runs `gpu_only` only. Reports the reason, with no command |
| `unavailable` (after the round) | the GPU + ANE round failed, or its trains stopped reaching the ANE | see below |

After the GPU + ANE round, the trains are checked. The round is demoted when no train ran
on the ANE, or when any train was routed with `ane_runtime_unavailable` (the ANE worker
died). A demoted round keeps its data and has `ane_verified: false`. The ANE state
becomes `unavailable` with reason `ane_lost_during_round`, and `comparison.available` is
false. Neither the terminal nor the result card presents it as a comparison. If loading,
warming up or running the GPU + ANE round raises, that round is dropped. The state
becomes `unavailable` with reason `ane_round_failed` and the error in `ane.detail`, and
the GPU-only round still runs. GPU-only rounds have `ane_verified: null`.

`result.json` records the raw reason code as `ane.reason` and a plain-language sentence
as `ane.reason_text` (`laya_apple/demos/switchyard/messages.py`). When there is an
underlying error, `ane.detail` holds it.

| `ane.reason` | `ane.reason_text` |
|---|---|
| `ane_runtime_unavailable` | The Neural Engine runtime (Core ML Tools) is not installed |
| `ane_artifact_unavailable` | The Neural Engine model is not built on this Mac yet |
| `ane_artifact_stale` | The Neural Engine model needs to be rebuilt on this Mac |
| `platform_not_validated` | This Mac has not been calibrated for the Neural Engine yet |
| `ane_unused_in_warmup` | The router sent no trains to the Neural Engine during warmup |
| `ane_parity_failed` | A Neural Engine model failed the correctness check on this Mac |
| `ane_placement_failed` | Core ML could not run the model on the Neural Engine |
| `not_apple_silicon` | This machine is not an Apple silicon Mac |
| `ane_lost_during_round` | The Neural Engine stopped being used during the round |
| `ane_round_failed` | The GPU + ANE round failed on this Mac |
| `ane_convert_unavailable` | The Neural Engine build tools (laya-apple[convert]) are not installed |
| `ane_calibration_failed` | Measuring this Mac for the Neural Engine failed |

`ane_runtime_unavailable`, `ane_artifact_unavailable`, `ane_artifact_stale`,
`platform_not_validated`, `ane_unused_in_warmup` and `ane_convert_unavailable` are
`setup_available`. All the others are `unavailable`.

The setup command is exactly:

```
uvx --from "laya-apple[convert]" laya-apple switchyard --setup-ane
```

`--setup-ane` first builds every offered bucket that does not verify on this Mac, with
the parity gate (`laya-apple artifacts build`). `calibrate` needs all of them. It then runs
`laya-apple calibrate` if no shipped profile matches this Mac. Last, it runs the
benchmark. If a build or the calibration fails, the failure becomes the ANE state (a
parity failure is `unavailable`) and the GPU-only round still runs.

Exit codes:
- **0:** the run finished, whatever the ANE state. A missing or failed ANE never makes
  the command exit with a non-zero code.
- **1:** the GPU-only round failed. Any rounds that finished are still written.
- **2:** a usage problem. The machine cannot run MLX (not Apple silicon macOS),
  `--duration` is not a whole number of 3 s burst cycles, or `--replay DIR` does not point
  to a folder a Switchyard run wrote (a missing or unreadable `result.json` or
  `trace.jsonl`). In both cases the command prints one line saying why. `--replay` works on any
  machine.

## Outputs

Each run writes three files to `--out`. The default is
`./switchyard-results/<UTC YYYYmmddTHHMMSSZ>-<soc>/`.

- `result.json`: `schema: "laya-apple/switchyard-result"`, `schema_version: 1`,
  validated against `laya_apple/data/switchyard-result.schema.json`. Machine paths are
  written as `<LAYA_APPLE_CACHE>`, `<HF_HOME>` or `~`. `workload.burst_period_s` and
  `workload.burst_on_s` give the burst shape (3 s and 1 s), from the same constants
  `laya_apple.workload.arrivals` uses. `configs.<label>.summary` has the shape of a round's
  `systems`, and `configs.<label>.game` the shape of its `game`, both pooled over that
  configuration's rounds. Each round also records its `round_start_ns` and a
  `timetable_sha256` of what it submitted.
- `trace.jsonl`: one line per request. Each line holds `RequestTrace.to_dict()` plus
  `round`, `index` (the position in the schedule), `class` and `arrival_ns`. Train lines
  also have `line`, `train_id`, `pattern`, `answer`, `probabilities`, `oracle` and
  `outcome`. Every number in `result.json` can be recomputed from this file with
  `laya_apple.demos.switchyard.metrics`.
- `replay.html`: a self-contained page. It is `static/index.html` with the stylesheet,
  the script and the replay data inlined. It needs no network and opens from `file://`.
  `--replay DIR` rebuilds it from `DIR`'s `result.json` and `trace.jsonl`.

## Replay data

`replay.html` embeds one JSON object, built by
`laya_apple.demos.switchyard.replay.replay_data`. It is the only interface between the
benchmark and the page. `replay.bundle` fills three markers in `static/index.html` in a
single pass:
- `/*__SWITCHYARD_CSS__*/` in a `<style>`, with `static/style.css`;
- `/*__SWITCHYARD_JS__*/` in a `<script>`, with `static/app.js`;
- `__SWITCHYARD_DATA__` in `<script type="application/json" id="switchyard-data">`, with
  the JSON. Every `</` is escaped as `<\/`.

```
{
  "format": "switchyard-replay", "format_version": 1,
  "result": { the full result.json object },
  "rounds": [{
    "index", "config", "duration_s", "deadline_ms",
    "trains": [{
      "id", "line", "platforms": {"A": ..., "B": ..., "C": ...}, "oracle", "answer",
      "device", "outcome", "arrival_ms", "queue_enter_ms", "dispatch_ms",
      "service_start_ms", "service_end_ms", "response_ms", "latency_ms", "queue_ms",
      "service_ms"
    }],
    "background": [{
      "class", "device", "arrival_ms", "queue_enter_ms", "dispatch_ms",
      "service_start_ms", "service_end_ms", "response_ms", "latency_ms"
    }]
  }]
}
```

- **Times.** Every `*_ms` time is relative to that round's `round_start_ns`, rounded to
  3 decimals. Trains and background requests are sorted by arrival.
- **Values.** `outcome` is `delivered`, `late` or `misrouted`. `device` is `gpu` or
  `ane`. `line` is 1–6. Platform states are `clear`, `occupied` or `closed`.
- **Recomputing.** The counts and percentiles on the result card can be recomputed from
  this data, within the 3-decimal rounding.
- **Synthetic data.** Synthetic test data under `tests/fixtures/switchyard/` is labelled
  `"synthetic": true`. It is never used for documentation.
