# Criteria: GIL-released Core ML predict on the product mix

Written and committed before any campaign run. Nothing in this file is changed after the
data is seen. A failure is reported as a failure, with the metric and its magnitude. The
harness smoke checks made while writing the scripts are not data and are not committed.

## Configurations

| | ANE | Core ML `predict` | GPU (MLX) |
|---|---|---|---|
| **A** | thread in the caller | coremltools, GIL held | worker process |
| **B** | worker process | coremltools, in the worker | worker process |
| **C** | thread in the caller | PyObjC-built features, GIL released (`scripts/nogil.py`) | worker process |
| **D** | thread in the caller | as C, GIL released | thread in the caller, no IPC |

Each model's production configuration today (`laya_apple/data/placement.json`):

| model | production | shapes (short / long) |
|---|---|---|
| laya | A | L128 / L512 |
| laya-typed-decisions | A | L128 / L1024 |
| laya-multilingual | B | L96 / L1024 |

Each candidate, C and D, is compared with the model's production configuration. The other
of A / B is not run: #57 already measured process against thread placement for laya and
laya-typed-decisions on this workload.

## Gate: the #57 criteria, copied verbatim

From `benchmarks/ane-process-isolation/README.md` (#57), "Criteria (written and committed
before the campaign ran)":

> | # | criterion | definition |
> |---|---|---|
> | 1 | correctness | 0 answer mismatches in every window of every run, both placements |
> | 2 | aggregate throughput | process ≥ 0.95 × thread |
> | 3 | short-stream P99 | process ≤ 1.05 × thread |
> | 4 | long (GPU) stream P99 | process ≤ 1.05 × thread |
> | 5 | GPU completion isolation | process GPU return P50 ≤ 1 ms, **and** thread P50 / process P50 ≥ 5 |
>
> **Definitions:**
> - **Windows.** Only the `hetero` windows count: both streams run through the one
>   `device="auto"`, `execution="workers"` instance. Each placement has 2 runs × 3 cycles = 6
>   windows.
> - **Aggregate throughput.** For each stream, take the median of its req/s over the 6
>   windows; add the two streams' medians. This is `bench_concurrency.py`'s own aggregate,
>   over the pooled windows.
> - **Stream P99.** The median over the 6 windows of that stream's per-window P99 latency.
>   This is `bench_concurrency.py`'s own summary, over the pooled windows.
> - **GPU return.** `received_ns − service_end_ns` of every GPU request whose reply arrives
>   inside a `hetero` window, pooled over the 6 windows. P50 of that pool.

**How it is read here.** This is the only substitution; the thresholds and definitions are
unchanged:
- "process" is the candidate (C or D). "thread" is the model's production configuration (A
  for laya and laya-typed-decisions, B for laya-multilingual). "Both placements" is the
  candidate and the production configuration. "Placement" is configuration.
- **laya-multilingual:** the second half of criterion 5 ("thread P50 / process P50 ≥ 5")
  does not apply. Its production configuration B already has no thread-placed `predict` in
  the parent, so it has no completion wait to improve on. Criterion 5 there is "candidate GPU
  return P50 ≤ 1 ms".

**Verdict.**
- A candidate PASSes on a model only if every criterion holds.
- A candidate PASSes overall only if it PASSes on all three models.
- C and D are judged independently.

`scripts/analyze.py` implements this. It computes each configuration's summary with #57's
own `placement_summary` (imported from `benchmarks/ane-process-isolation/analyze.py`), and
the thresholds are #57's `LIMITS`.

## Stop rules

- **C and D both fail:** FAIL is recorded. The Swift-worker fallback research is
  preregistered next.
- **D passes:** D is the candidate production design.
- **Only C passes:** C is the candidate production design, and the GPU worker stays a
  process.

A candidate production design is not a production change. Any runtime change is its own PR,
with its own tests and gates.

## Valid runs

- A run that crashes or fails to start is re-run once, in the same position of the order.
  Both the failure and the re-run are reported. No completed run is discarded.
- In every `hetero` window, the short stream must be served only by the ANE and the long
  stream only by the GPU. `analyze.py` reports any run where this does not hold. Such a run
  is reported, and it is not silently dropped.

## Not gating: recorded and reported, never part of the verdict

- **The ANE thread's GIL re-acquire wait after `predict` returns** (C and D): the time from
  the native return of `predictionFromFeatures:error:` to the moment the ANE thread holds the
  GIL again. P50 / P95 / P99 / mean over every `predict` in a `hetero` window. A has no such
  wait, because it never releases the GIL. In B the call is in another interpreter.
- **CPU time per request on the executing threads**, GPU and ANE: `time.thread_time_ns()`
  of the thread that runs the backend's forward, across the forward, for forwards started
  in a `hetero` window. In-process, that thread is the device's dispatcher thread. In a
  worker process, it is the worker's thread (`scripts/worker_entry.py`).
- The GPU forward's service span (`service_end − service_start`), per configuration.

## Known property of the fixed gate for D

In D the GPU has no reply to carry, so its return leg (`received_ns − service_end_ns`) is
bookkeeping inside the dispatcher thread, and criterion 5 is met almost by construction.
- Any GIL wait the GPU dispatcher has after `mx.eval` falls inside its service span.
- For D, that cost can only show in criteria 2 and 4 (aggregate throughput and long-stream
  P99). The GPU service span above reports it, but does not gate it.
- Criterion 5 is applied to D as written.

## Protocol

- **Workload:** #57's `benchmarks/ane-process-isolation/run_mix.py`, which runs
  `scripts/bench_concurrency.py --part a` unchanged. 20 s windows, 3 cycles per run, one
  closed-loop client per stream, answers checked against inline references computed per
  device. The heterogeneous instance's configuration is the only difference between runs.
- **Order per model:** P C D D C P, where P is the production configuration, one run after
  another. This is ABBA extended to three configurations: every configuration's mean
  position is equal. The models run in the order laya, laya-typed-decisions,
  laya-multilingual.
- **Machine:** idle, on AC power, with oMLX and other GPU/ANE services stopped. The Core ML
  E5 cache is not cleared.
- **Versions:** as locked in `uv.lock` (coremltools 9.0, mlx 0.32.x), plus
  pyobjc-framework-CoreML 12.2.2 through `uv run --with`, for research only.
