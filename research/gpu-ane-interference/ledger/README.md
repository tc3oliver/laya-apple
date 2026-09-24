# Runtime-trace pipeline: representative re-run (issue #6)

This directory re-measures the study's headline result with the lifecycle taken from the
runtime itself, not from the harness.

- **Runtime lifecycle:** the runtime emits `laya_apple.RequestTrace`
  (`Laya(..., trace=callback)`). It is the only source of submit, prepare, routing, queue,
  dispatch, service, response and the router's queue snapshots.
- **Research-only detail:** the harness adds the device spans, host work and CPU time. It
  joins them to the trace on `request_id`.
- **What this re-run checks:** that the new pipeline joins completely and reproduces the
  known decomposition, and what it adds. It covers a representative subset, not the 1.49M
  requests of the full campaign.

**Status [Measured].** Apple M4 Max, macOS 26.6.2 (25G83), Python 3.12.14, MLX 0.32.2,
coremltools 9.0, laya-apple 1.0.2 with request tracing (PR #43, `f817218`; the harness in
this directory ran on top of it). `laya-typed-decisions`. AC power, oMLX stopped. 0 answer
mismatches in every window of every run.

## What changed in the harness

| Before | Now |
|---|---|
| `jobtrace.JobTable` + `instrument_worker` wrapped `DeviceWorker.submit` / `_run` and stamped `queue_enter`, `device_start` and `device_end` itself, keyed on `id(rows)` | removed; the runtime trace carries these and more (`dispatch`, the worker's `service_start/end`, `received`) |
| `ProductStream` found its request's record through thread-local tricks around `Laya.submit` | the stream keeps `RuntimeInfo.request_id`; the trace is looked up by it |
| backlog, estimate, reason and snapshot copied from `RuntimeInfo` or read with an extra `backlog_ms()` call | from the trace: the very reading the router used, plus queue depth and running flags |
| the device plan fed `DeviceWorker.submit` directly | the device plan submits through `Laya.submit` of an explicit-device instance (`device="gpu"`, `device="ane"`), both in the same process (see Methodology change) |
| backend phases joined to requests by time containment | each backend phase record carries `request_id`, and the join is by id; time containment is now a check |

What stays research-only (`scripts/jobtrace.py`):
- `mx.eval` spans (MLX) and Core ML `predict` spans (ANE);
- host work, which is the forward minus the device spans;
- thread CPU time;
- the GIL probe (`gil_probe.py`, unchanged);
- the aggressors and the contention prototype.

The request id reaches the backend hooks in two ways, depending on placement:
- **Thread-placed backend:** the hook wraps `DeviceWorker._run(job_id, rows)` on that
  instance.
- **Worker process:** `hooks/sitecustomize.py` wraps the worker's connection, so a received
  `(job_id, rows)` names the forward that follows.

The job id on the worker protocol is the request id (PR #43).

## Canonical ledger

`scripts/ledger.py` builds one record per request from `RequestTrace` + backend phase +
generator row. The schema and every definition are in its docstring. The durations are
classified:

| Kind | Durations |
|---|---|
| measured (runtime) | prepare, route, enqueue, queue, dispatch, service, return, occupancy (dispatch→received), postprocess, e2e, completion (routed→received) |
| measured (hooks) | forward, device_exec, cpu_ms |
| derived | host = forward − device_exec |
| unattributed | service − forward (and all of service if a request had no backend record: none did) |
| generator | generator_lag = submit − scheduled arrival (open loop) |

Identities hold by construction:
- occupancy = dispatch + service + return;
- e2e = prepare + route + enqueue + queue + occupancy + postprocess.

**Mapping to the historical harness:**
- its `service` is `occupancy` here;
- its `dispatch` (service − forward) is `dispatch + return` here;
- its `forward` is `service` here.

## Methodology change

In the device plan, the historical harness called `DeviceWorker.submit` itself. It did its
own prompt build, ANE check, estimate and answer formatting.

It now calls `Laya.submit` on two explicit-device instances in one process:
- `device="gpu"`: a GPU worker process;
- `device="ane"`: the ANE on a thread or in a process.

This is the same topology as `auto` (a thread-placed ANE shares the interpreter with the
GPU's dispatcher). The per-request host work is the same kinds of work, now done by the
runtime. One difference: each instance's router sees only its own device's queue, so the
trace's snapshot of the other device is empty in the device plan. Cross-device concurrency is
measured instead as each request's overlap with the other device's occupancy.

Numbers are compared with the historical run side by side below, with this difference in
mind.

## Representative workload

Run with `sh scripts/run_ledger.sh`. The raw files are in `raw/` and `overhead/`, compact
JSON, integer µs.

| Run | Cells | Windows | In-window requests |
|---|---|---|---:|
| `raw/device-laya-typed-decisions-thread` | solo:gpu_M (GPU L128), solo:ane_B (ANE L128), matrix:gpu_M+ane_B | 3 cycles × 25 s, 2 s warm-up, 0.4 s tail guard, 8 seeds per shape | 24,628 |
| `raw/device-laya-typed-decisions-process` | same, ANE in a worker process | same | 24,890 |
| `raw/product-laya-typed-decisions-thread` | `auto` router, open loop: product:open@25 (v0.2 Part B mix) and product:heavy@100 (90% short / 10% long) | same | 9,177 |
| `overhead/product-...-off` | the product cells, runtime tracing and hooks off | same | 9,177 |
| `overhead/solo-...-{hooks,trace,off}` | solo:gpu_M, solo:ane_B, with hooks + trace, trace only, neither | 2 cycles × 15 s | 5,370 each |
| `equivalence-*.json` | 7 shapes × 8 seeds, sequential, trace off and on | — | 56 per placement |

The first full run of this campaign was discarded, and everything above is from the second.
The reason: a harness bug stamped open-loop client completion when the collector reached the
request, not when it resolved. It changed only the client-side e2e of the open-loop runs; the
runtime trace was unaffected.

The discarded run is not committed. Its headline numbers agreed with the committed run
(thread-placed GPU occupancy ×1.645, dispatch + return +7.45 ms), except `device_exec`
×1.036 against ×1.066 here. Its solo GPU `mx.eval` was 11.82 ms against 11.52 ms here.

## Validation

**Join (`tables.md`, "Join").** 100% of traces join exactly one backend record by
`request_id`, in all three runs: 26,945 / 26,945, 27,238 / 27,238 and 10,071 / 10,071.
- **Checks, all zero:** backend spans outside the trace's service window, device mismatches,
  traces with several backend records, backend records without a trace, generator rows
  without a trace.
- **Records without a request id:** only 1–4 per run, all before the first request. These are
  the worker processes' warm-up forwards: one MLX forward, plus one per ANE bucket in a
  process-placed ANE worker. The thread-placed ANE warms up in this process, and those
  forwards are cleared with the inline reference forwards before measurement starts. The
  rest are counted and categorised, never dropped.

**RuntimeInfo vs RequestTrace.**
- `RuntimeInfo.queue_wait_ms`, `device_ms` and `latency_ms` are computed from the trace's own
  timestamps, so they are `queue_ms`, `service_ms` and `e2e_ms` exactly. The PR #43 tests pin
  this to within 1e-6 ms, with real worker processes.
- The boundary difference that matters is `device_ms` against occupancy. `device_ms` is the
  forward in the process that runs it. Occupancy adds the dispatch and return legs, and it is
  what the historical harness called service.
- The hook's own forward span sits inside the runtime's service window for 100% of requests.
  The gap (unattributed) averages 0.004–0.010 ms, with P99 ≤ 0.035 ms and max 0.32 ms. That
  is ≤0.09% of e2e.

**Tracing changes neither decisions nor answers.**
- **Idle queues** (`equivalence-*.json`). The 56 requests route identically (device, reason,
  buckets) with bit-identical answers, trace off against trace + hooks on, for both ANE
  placements. Every request is traced once and joined 1:1.
- **Under open-loop load** (`overhead.json`), routing depends on timing, so the shares are
  compared.

  | Cell | `ane_backlog_shorter_on_gpu`, on | off |
  |---|---:|---:|
  | heavy@100 | 7.96% | 7.92% |
  | open@25 | 1.26% | 1.31% |

  Every other reason is identical to four digits, and both runs have 0 mismatches.

**Overhead** (`overhead.json`, client clock, solo cells). The two costs are kept separate:
- runtime tracing (trace − off): ANE L128 mean e2e +0.17%, GPU L128 −0.09%;
- runtime tracing plus research hooks (hooks − off): ANE +0.18%, GPU +0.06%.

Both are within run-to-run noise. The runtime tracing cost on its own is in
`benchmarks/tracing.md`: +0.96 µs per request in the host path, no measurable serving change.

## Reproduction: GPU+ANE concurrent service decomposition

Solo against concurrent (matrix), mean ms per in-window request, with the Δ 95% bootstrap
CI in `tables.md`. The figure is `figures/service-decomposition.svg`.

**Thread-placed ANE, GPU L128 stream:**

| | solo | with ANE busy | Δ | share of Δ occupancy |
|---|---:|---:|---:|---:|
| dispatch (send leg) | 0.03 | 0.03 | −0.00 | 0% |
| **return (reply leg)** | 0.05 | 7.41 | **+7.36** | **91.1%** |
| device_exec (`mx.eval`) | 11.52 | 12.28 | +0.76 (×1.066) | 9.3% |
| host (in the worker) | 0.45 | 0.42 | −0.03 | −0.4% |
| unattributed | 0.01 | 0.00 | −0.00 | −0.0% |
| **occupancy** | 12.05 | 20.14 | +8.08 (**×1.670**) | |

| Historical (`../results.json`) against this run | historical | this run |
|---|---:|---:|
| Δ dispatch (historical) against Δ dispatch + return | +7.40 ms | +7.36 ms |
| service × against occupancy × | ×1.641 | ×1.670 |
| `mx.eval` × | ×1.037 | ×1.066 |
| host Δ | +0.02 | −0.03 |

**The known result reproduces.** GPU inflation with a thread-placed ANE is host-side waiting
between the worker and the parent, not device execution. `mx.eval` accounts for 9% of the
increase in this run (5% in the discarded replicate).

**The new decomposition sharpens it.**
- **The wait is entirely in the reply leg.** The send leg does not change.
- **Its size is set by the ANE's `predict`** (`tables.md`, "GPU return leg"; data in
  `results.json` `return_alignment`).
  - All 3,680 GPU forwards ended while an ANE `predict` was running.
  - The parent's dispatcher got the result a median 0.125 ms after that `predict` ended
    (P5 −0.106, P95 +0.302 ms).
  - The mean return leg, 7.41 ms, matches the `predict` time left when the GPU forward
    ended, 7.25 ms, with correlation 0.83.

  This is what a reply waiting for the GIL looks like, if Core ML `predict` holds it; the GIL
  probe in `../README.md` measured that directly. The timestamps show the alignment. They do
  not observe the GIL itself.
- **The process placement is the control.**
  - In `matrix:gpu_M+ane_B` with the ANE in its own process, 4,517 of 5,195 GPU forwards
    also ended during an ANE `predict`. Their reply leg is 0.12 ms, uncorrelated with the
    predict time left (−0.11).
  - There, GPU occupancy grows ×1.121 (historical ×1.123). The increase is host work inside
    the worker (+1.06 ms, ×3.7, 71% of Δ; historical +1.06) and `mx.eval` (×1.022; historical
    ×1.025).
  - The GPU worker's CPU time per request rises 1.51 → 4.75 ms (historical 1.51 → 4.8).
- **The ANE stream is barely affected with a thread placement:** occupancy ×1.013
  (historical ×1.005).
- **With a process placement, both devices pay about ×1.12.** For the ANE, that is
  `predict` ×1.055 plus host ×3.4.

The timeline in `figures/request-timeline.svg` (300 ms of the thread-placed matrix cell)
shows the pattern. Every GPU request ends in a ~7 ms return segment that closes when the
ANE's current request finishes.

## Scheduler estimation error

These are the router's own numbers from the trace; nothing is re-modelled:
- **Predicted completion** = the target's backlog in the snapshot the router used plus the
  service estimate charged to the queue (`scheduling.decide_queued` compares
  `backlog + service estimate`).
- **Actual completion** = routed → received.
- **The split:** signed error = queue error (routed → dispatch against the backlog) + service
  error (occupancy against the estimate). The residual is 0 over 108,213 records.

Results are in `tables.md`, "Completion prediction error", `prediction_error.csv` and
`figures/prediction-error.svg`.

**Product run (router under open load, thread placement, 9,177 requests).**

| Group | n | mean signed | median | P90 \|err\| | mean rel. | queue err | service err |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 9177 | +4.07 | +0.91 | 12.65 | +15.6% | +2.06 | +2.01 |
| target ANE, GPU busy | 6053 | +1.03 | +0.51 | 1.98 | +6.4% | +0.53 | +0.51 |
| target ANE, GPU idle | 1001 | +4.33 | +1.92 | 11.94 | +43.6% | +0.22 | +4.11 |
| target GPU, ANE busy | 1392 | +15.17 | +12.17 | 28.29 | +37.1% | +9.44 | +5.73 |
| target GPU, ANE idle | 731 | +7.68 | +6.59 | 16.28 | +12.7% | +3.22 | +4.46 |
| reason `ane_backlog_shorter_on_gpu` | 608 | +17.68 | +16.95 | 33.74 | +69.3% | +10.55 | +7.13 |

**Findings:**
- **The router's completion estimate is biased late on every group**, never early on average.
- **It is worst for the GPU, and worst of all when the ANE is busy:** +15.2 ms, +37%.
- **The requests diverted from a busy ANE to the GPU (`ane_backlog_shorter_on_gpu`) are the
  worst predicted of all: +17.7 ms, +69%.**
  - That decision is taken exactly when the ANE is busy, which with a thread placement is
    when a GPU request pays the ~7 ms return-leg wait.
  - Both the GPU backlog (queue error +10.6 ms) and the service (+7.1 ms) are
    under-estimated.
  - Whether the diversions were wrong decisions needs a counterfactual this ledger does not
    compute. The historical hindsight analysis found few misses.
- **The ANE with an idle GPU also has a large service error (+4.1 ms, +44%).** That fits the
  low-load power-state slowdown in `../README.md` ("Sparse load"). This run does not isolate
  it.

**Device runs (explicit devices, no routing choice, 49,518 requests pooled):**
- The service estimate is accurate on an uncontended device:
  - ANE L128, thread: mean signed −0.04 ms;
  - requests with no overlap with the other device: −0.02 ms mean.
- It misses by up to 8 ms when the other device overlaps:
  - GPU L128 with a thread-placed ANE busy: mean +2.9 ms, P90 \|err\| 8.0 ms;
  - overlap > 0.5 in the process placement: +1.47 ms (+13.4%).

**What this means for #6's two possible conclusions.** The ledger supports both, and it
separates them:
- **(A) The routing model is missing a term.** Completion under concurrency is
  under-predicted, systematically and by device.
- **(B) The error is dominated by an execution-placement effect.** The thread placement's
  reply-leg GIL wait, and the process placement's host slowdown, sit outside the router's
  control.

The data does not favour a routing change over an execution change on its own. The
historical A/B of a contention-aware router (`../README.md`, "Scheduler prototype") failed its
criteria. The mechanism shown here, a GIL-bound reply leg, is removable by execution changes,
for example taking the reply off the GIL-holding path, which would remove most of the bias at
its source.

## Files and commands

| Path | What |
|---|---|
| `raw/*.json.gz` | the three runs: traces, backend records, generator rows, windows, environment |
| `overhead/*.json.gz` | trace-off and hooks-off runs for overhead and routing equivalence |
| `equivalence-*.json` | idle-queue decision and answer equivalence (`scripts/equivalence.py`) |
| `results.json`, `tables.md`, `prediction_error.csv` | `scripts/analyze_ledger.py` |
| `overhead.json` | `scripts/overhead.py` |
| `figures/*.svg` | `scripts/figures_ledger.py` (from `results.json` only) |

```bash
export LAYA_APPLE_CACHE=/path/to/cache HF_HUB_OFFLINE=1
sh research/gpu-ane-interference/scripts/run_ledger.sh          # ~35 min: every run above, then analysis and figures
uv run python research/gpu-ane-interference/scripts/ledger.py RUN.json.gz --jsonl ledger.jsonl   # one run's ledger, one request per line
uv run python research/gpu-ane-interference/scripts/analyze_ledger.py --check
uv run python research/gpu-ane-interference/scripts/figures_ledger.py --check
uv run python research/gpu-ane-interference/scripts/overhead.py --check
```
