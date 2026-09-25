# Criteria: one prediction crossing on the ANE request path

This file was written and committed before any campaign run or bit-identity check. Nothing
in it changes after data is seen. A failure is reported as a failure, with its metric and
magnitude. Harness smoke checks made while the scripts were written are not data and are not
committed.

## Purpose and question

**Purpose.** Test whether reducing the ANE request path's many PyObjC/GIL handoffs to the
minimum restores ANE short-tail latency while keeping GPU completion isolation.

**Question.** If the ANE request path's many PyObjC/GIL handoffs are collapsed into a single
prediction crossing, can production-level ANE short-tail latency be restored while GPU
completion isolation is kept?

The experiment does not test which mechanism behind those handoffs matters, for example GPU
completions taking the GIL. It changes one thing, the number of handoffs, and measures the
outcome.

## Background

- **#77's result.** In #77 (issue #66, `research/coreml-nogil-product-mix/`), a thread-placed
  ANE with a GIL-released PyObjC `predict` (configuration C) kept GPU completion isolation. It
  failed #57's gate on all three models because the short (ANE) stream's P99 regressed
  against production. **That FAIL stands.** This experiment does not revise it.
- **C's handoffs.** C's request path makes dozens of Objective-C calls per forward, and
  PyObjC releases and re-acquires the GIL around every one of them:
  - before `predict`: allocate the input MLMultiArrays, read their shape, strides and data
    pointer, build the MLFeatureValues and the feature provider, push an autorelease pool;
  - after `predict`: list the output names, get each feature value and its MLMultiArray, read
    the shape, strides, data type and data pointer, pop the pool.
- **Why that may cost time.** Each re-acquire is a point where another Python thread can take
  the GIL, and the ANE thread can then wait before it runs again. The handoffs also cost
  bridge CPU time of their own.
- **This experiment** reduces the handoffs to the minimum, keeps everything else the same,
  and measures the outcome.

## Configurations

These are fixed now for all three models. They were not chosen from #77's results, and no
model or configuration is dropped based on them.

| | ANE | Core ML `predict` | GPU (MLX) |
|---|---|---|---|
| **P** | production: **A** (thread, coremltools, GIL held) for laya and laya-typed-decisions; **B** (worker process, coremltools) for laya-multilingual | as in production | worker process |
| **C** | thread | #77's binding, unchanged: PyObjC-built inputs and outputs, GIL released for `predict` | worker process |
| **PB** | thread | prebound (below): one prediction crossing per forward | worker process |

P and C are exactly #77's: same code, loaded from `research/coreml-nogil-product-mix/scripts/`
by path. PB has C's topology. Only the ANE binding differs.

### PB, the prebound binding (research only)

Nothing in `laya_apple/` or its packaging changes. `laya_apple` imports nothing from
`research/`.

**At model load, for each bucket, once:**
- The same verified `model.mlmodelc` is loaded with `CPU_AND_NE`, as in #46 and #77.
- An MLMultiArray is allocated for each input with the model's declared shape and data type.
  A NumPy view over each array's data is taken once.
- One MLFeatureValue per input, and one MLDictionaryFeatureProvider over them, are built.
  They are reused for every forward, so the provider references the same arrays each time.
- An MLMultiArray is allocated for each output with the model's declared shape and data type.
  These are set as `MLPredictionOptions.outputBackings`, and a NumPy view over each is taken
  once.
- Strong references to the arrays, feature values, provider and options are held for the
  bucket's lifetime.
- Every object Core ML reads during a prediction is a Foundation object, never a Python
  container that PyObjC proxies into Objective-C:
  - the array shapes are NSArrays of NSNumbers;
  - the feature dictionary and the output-backings dictionary are NSDictionaries with NSString
    keys.

  A Python-backed proxy (OC_PythonArray, OC_PythonDictionary and the like) makes Core ML call
  back into Python during `predict`, and each callback takes the GIL again inside the one
  crossing. This is checked at load: PB refuses to load if any such proxy reaches Core ML.
- The raw pointers of the model, provider and options, and the ctypes error slot and stamp
  buffer, are taken once at load. The request path touches no PyObjC object other than the
  autorelease pool.

**On the request path, per forward:**
1. The features are written into the preallocated input views. This is NumPy only.
2. An autorelease pool is pushed.
3. One `predictionFromFeatures:options:error:` message is sent.
4. The pool is popped.
5. The outputs are copied out of the backed output views into new NumPy arrays, with
   coremltools' names, dtypes and shapes (FP16 widened to float32). This is NumPy only.

**How the one `predict` send is made.** It goes through
`scripts/predict_options_stamped.m`, called with `ctypes.CDLL`, which releases the GIL once
for the call:
- This is #77's `predict_stamped.m` with one extra argument, the options object. It is
  compiled with the system clang into a temporary directory when a run starts, and no binary
  is committed.
- The shim only timestamps the one `predict` send. Every input, output, feature value,
  provider and options object is built by PyObjC at load time. The shim builds nothing and
  copies nothing, so it is not an input-building native path.
- C sends its `predict` through #77's shim in the same way. C and PB therefore share the same
  `predict`-call instrumentation, including the native before/after stamps.

**Risks, and how each is handled:**
- **Output backings may not be honoured.** This includes FP16 outputs on the ANE.
  - At load, after warm-up, one forward is run for each bucket. The returned provider's
    MLMultiArray for each output is compared with its backing, by data address and by value.
  - An output whose backing is honoured is read from its backed view: mode `backed`.
  - For any other output, the request path adds the fewest extra crossings needed to read it
    from the returned provider: feature value, MLMultiArray and data pointer, with the
    shape and strides cached at load. That is mode `read`.
  - The mode of each output is recorded in every run and in the bit-identity check.
- **Buffer lifetime.** Each bucket's objects are held by strong references on the model
  object for as long as the backend exists.
- **Buffer reuse.** Each bucket has a non-blocking lock. A forward that finds its bucket
  already in flight raises, so an overlapping forward cannot silently corrupt a shared
  buffer. The executor runs at most one ANE job at a time, so this is never expected to fire.
  It would fail the run, which is then reported.

## Before the campaign: bit-identity check

`scripts/check_prebind.py` runs after the machine is free and before the campaign, on every
ANE bucket of all three models.
- It compares PB's outputs with coremltools' outputs for the same `model.mlmodelc`.
- It uses the warm-up row plus 10 generated rows per bucket, with different token ids,
  lengths, markers and question types each time. Reused buffers therefore have to change
  between calls.
- It also compares the full `ANEBackend.forward` logits and actions.
- **Condition to start the campaign:** PB is bit-identical to coremltools on every compared
  output of every bucket of every model.
- C is checked the same way, and its result is recorded. This costs seconds. It is not a
  re-validation of C, which #77 already validated.
- The same script records each output's backing mode and the crossings per forward for C and
  PB (defined below). Its output goes into `raw/check.json`.
- `run_all.sh` runs this check before the campaign.

If PB is not bit-identical, the campaign does not run, and that result is reported.

## Gate: #57's engineering budgets, judged by a paired statistical gate

### The budgets: #57's criteria, copied verbatim

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

**How the budgets are read here.** This is the same substitution #77 made:
- "process" is the candidate (C or PB). "thread" is the model's production configuration P.
  "Both placements" means the candidate and P. "Placement" means configuration.
- **laya-multilingual:** the second half of criterion 5 ("thread P50 / process P50 ≥ 5")
  does not apply. Its P is B, which has no thread-placed `predict` in the parent and so no
  completion wait to improve on. Criterion 5 there is "candidate GPU return P50 ≤ 1 ms".

The thresholds (1.05×, 0.95×, 1 ms, 5×) are unchanged.

### The verdict rule: a paired statistical gate

This rule applies from this preregistration on. It is **not applied retroactively**: #77 and
older results keep the verdicts they were scored with, and #77's FAIL stands.

Criteria 2, 3 and 4 are judged on matched window pairs, not on one pooled point estimate.
Individual requests are never treated as independent samples.
- **Matching unit.**
  - Round 1 is each configuration's first run in the order below, and round 2 its second.
  - In a run, cycle k is the k-th `hetero` window that the run records (k = 0, 1, 2; see
    "Protocol").
  - A pair is the candidate's `hetero` window of round r, cycle k and P's `hetero` window of
    the same round r and the same cycle k.
  - The order P C PB PB C P with 3 cycles per run gives 2 × 3 = 6 pairs per candidate per
    model.
  - A cycle present in only one run of a pair is an error, reported as such. It is never
    silently dropped.
- **Per pair,** the ratio candidate / P of:
  - the short-stream P99 (the window's P99, as `bench_concurrency.py` computes it);
  - the long-stream P99;
  - the aggregate throughput (short req/s + long req/s in that window).
- **Statistic:** the geometric mean of the n pair ratios.
- **Primary interval:** a 95% Student-t interval on the log ratios (mean ± t(0.975, n − 1) ·
  SD / √n), back-transformed.
- **Sensitivity only:** a percentile bootstrap over pairs, with 10,000 resamples and seed
  20260925. It is reported beside the primary interval and never changes the verdict.
- **Short P99 and long P99:**
  - PASS if the interval's upper bound is ≤ 1.05;
  - FAIL if its lower bound is > 1.05;
  - otherwise INCONCLUSIVE.
- **Aggregate throughput:**
  - PASS if the interval's lower bound is ≥ 0.95;
  - FAIL if its upper bound is < 0.95;
  - otherwise INCONCLUSIVE.
- **Correctness (criterion 1):** unchanged. 0 mismatches in every window of every run of the
  candidate and P.
- **GPU completion isolation (criterion 5):** the absolute engineering gate, unchanged and
  computed as in #77. GPU return P50 over the pooled `hetero` windows must be ≤ 1 ms, and
  ≥ 5× better than A. Against B, only ≤ 1 ms applies. PASS or FAIL.
- **Per model and per candidate:**
  - PASS only if every criterion passes;
  - FAIL if any criterion fails;
  - otherwise INCONCLUSIVE.
- **Independent judgments.** Each model's verdict is reported on its own. C and PB are judged
  independently. C's verdict here is a same-campaign reference point for PB and does not
  revise #77.

`scripts/gate.py` implements the rule and `scripts/analyze.py` applies it. #57's
`placement_summary` and `LIMITS` are imported from
`benchmarks/ane-process-isolation/analyze.py`. Correctness and criterion 5 use #77's
`verdict`. The pooled point estimates are reported as context, never as the verdict.

### Power, estimated before any data

This estimate uses #77's per-window data. That protocol had solo windows between hetero windows,
so the estimate is approximate for the hetero-only protocol here. Each `hetero` window was paired with production's
window of the same round and cycle, with n = 6. t(0.975, 5) = 2.571.
- **Log-scale SD** is the SD of the pair log ratios.
- **Half-width** is t · SD / √6, on the log scale. For comparison, log 1.05 = 0.049.
- **A r2 vs A r1 and B r2 vs B r1** pair production with itself, with n = 3. That gives a
  null comparison.

| model | metric | log SD (C vs P) | log SD (D vs P) | log SD (P vs P, null) | half-width at n = 6 |
|---|---|---|---|---|---|
| laya | short P99 | 0.164 | 0.085 | 0.100 | 0.09–0.17 |
| laya | long P99 | 0.079 | 0.012 | 0.008 | 0.01–0.08 |
| laya | aggregate | 0.015 | 0.120 | 0.006 | 0.01–0.13 |
| laya-typed-decisions | short P99 | 0.246 | 0.078 | 0.035 | 0.04–0.26 |
| laya-typed-decisions | long P99 | 0.054 | 0.043 | 0.035 | 0.04–0.06 |
| laya-typed-decisions | aggregate | 0.079 | 0.006 | 0.005 | 0.01–0.08 |
| laya-multilingual | short P99 | 0.021 | 0.059 | 0.018 | 0.02–0.06 |
| laya-multilingual | long P99 | 0.011 | 0.022 | 0.017 | 0.01–0.02 |
| laya-multilingual | aggregate | 0.010 | 0.105 | 0.001 | 0.00–0.11 |

**What this means for the verdicts.**
- **laya short P99:** with n = 6, this is expected to be INCONCLUSIVE unless the effect is
  large. Even with a true ratio of 1.00, the upper bound would be about 1.10–1.18.
- **laya-typed-decisions:** short and long P99 are resolvable only if PB's pair-to-pair spread
  is close to production's own.
- **laya-multilingual,** and aggregate throughput in most cells, can resolve a 5% difference.
- **A larger replication.** At an SD of about 0.10, a 5% budget needs about 18 pairs: 9 cycles
  per run in the same order. That would be about 1 h 15 min for 18 runs under the hetero-only
  protocol.

**Decision.** The design stays at 3 cycles per run, the same number of `hetero` windows as #57
and #77 (n = 6 pairs). The windows between them change, as described under "Protocol". An
INCONCLUSIVE result is reported as such, and no production change follows from it. A larger
preregistered replication (for example, 9 cycles and n = 18) decides it.

## Preregistered interpretation and follow-ups

Applied per model. A pass on one model is not carried over to another.
- **PB PASS on a model:** this supports "the many PyObjC/GIL handoffs are a significant
  execution-layer cost" for that model. It is not attributed to GPU-completion collisions
  alone.
- **PB PASS on all three models:** a production binding is considered next, in its own PR
  with its own tests and gates. A research pass is not a production change.
- **PB INCONCLUSIVE on a model:** the result is reported as inconclusive, with each
  criterion's interval. No production change follows. A larger preregistered replication
  (for example 9 cycles, n = 18 pairs) decides it. INCONCLUSIVE is neither a pass nor a
  fail, and it does not start the FAIL branch below.
- **PB FAIL on any model:** that model's verdict is recorded as a FAIL, with the failing
  metric and its interval. The paused equal-load experiment (branch
  `research/coreml-nogil-equal-load`) runs next.
  - **Equal-load passes:** workload and SoC resource contention are considered next.
  - **Equal-load fails:** the CPU slow state, host scheduling and Python runtime contention
    are the next priorities.

**Not in this experiment, recorded as out of scope:**
- thread QoS classes and CPU scheduling policy, which are not varied or set in any
  configuration;
- a Swift worker;
- a native full-forward path.

## Prior evidence: observational, association only (non-gating)

This is an observational collision analysis of #77's raw data. It shows association only, not
cause.
- **Exposure window.** An ANE request is counted as colliding if a GPU completion falls inside
  a fixed window, `[ANE service_start − 1 ms, ANE service_start + 0.3 ms]`. A GPU completion
  here is the parent-side `received_ns` of a GPU request. The window was chosen after the data
  was seen.
- **Findings for C:**
  - **laya:** collisions explain very little of the tail: odds ratio 1.6 [1.0, 2.4], about 2%
    attributable.
  - **laya-typed-decisions:** collisions explain very little.
    - The regression is mainly an upward shift of the whole latency distribution: 24% of
      requests are above production P99, and 1.3% collide (about 1% attributable).
    - Production A shows a similar association (odds ratio 5.4).
  - **laya-multilingual:** a strong association (odds ratio 209 [144, 305]) that production B
    does not show (odds ratio 0.2). Collisions account for about 25% of its tail.
    - This is confounded with the CPU slow state, which covers every C window.

This evidence does not affect the gate, the verdict or the interpretation above. This
experiment computes the same fixed-window collision measure for P, C and PB as a diagnostic
(below), so the comparison after the runs uses the same definition. The GIL-contention
measures below do not depend on GPU completions. They are there because a broad shift such
as laya-typed-decisions' could come from GIL contention with other Python threads: the
long-stream client preparing and tokenizing requests, the GPU dispatcher reading the worker's
reply, the dispatchers themselves.

## Not gating: recorded and reported, never part of the verdict

Every record below except the crossings count is taken during the campaign.
- **Every configuration carries the same instrumentation.** The only exception is P = B, whose
  ANE forward runs in a worker process. There, the parent-side stage stamps do not apply.
- **The instrumentation differs from #77's.** It adds the GIL probe, per-thread CPU and the
  full request trace. Numbers are compared within this campaign only. Where one is set beside
  #77's, the difference in instrumentation is stated.

**Recorded measures:**
- **Objective-C crossings per forward, for C and PB.**
  - Counted with `sys.monitoring` over one forward per bucket, at load time after warm-up, and
    again in the bit-identity check. It is never counted inside a measured window.
  - **Top-level crossings** are calls made from the forward's own Python code whose callable
    is one of these:
    - a PyObjC message send (`objc.native_selector`);
    - a ctypes foreign-function call (the `predict` shim);
    - the exit of `objc.autorelease_pool`, which releases the pool.

    The pool push counts as the message sends it makes (`alloc`, `init`).
  - **Re-entries** count separately. A re-entry is Python code starting on the calling thread
    while a crossing's native call is still in flight (CALL, C_RETURN, C_RAISE and PY_START
    events). Any sends made inside a re-entry count as `nested_send`.
    - A re-entry is a hidden GIL acquisition inside a crossing, for example Core ML calling
      back into a Python-backed proxy.
  - Implicit sends that PyObjC makes with no Python call and no Python callback are not
    counted. These include a proxy's release when it is freed and conversions done in C. The
    count is therefore a lower bound for C.
  - **PB's expected count per forward, fixed before any campaign data:**
    - 1 prediction crossing (`ctypes`);
    - 2 sends for the pool push and 1 for the pool pop;
    - 0 other sends;
    - 0 re-entries.

    That is 4 in total. Everything else Core ML reads was prebuilt from Foundation objects at
    load.
  - **The autorelease pool cannot be removed.** The prediction's result and Core ML's
    temporaries are autoreleased, and a Python thread has no pool of its own. Without a pool
    per forward they leak until IOSurface allocation fails (#46). C has the same pool.
  - **Pre-campaign measurement** (`raw/check.json`, every bucket of all three models; PB was
    bit-identical everywhere):
    - PB: total 4 (predict 1, pool 3), 0 sends, 0 re-entries.
    - C: total 108 top-level crossings (104 sends, predict 1, pool 3), plus 62 re-entries with
      112 nested sends inside its `predict`, from Python-backed shape arrays and dictionaries.
    - An earlier PB draft passed shapes as Python lists. It made 20 re-entries (40 nested
      sends) inside the prediction. It was fixed before any campaign run, and the criteria did
      not change.
- **GIL re-acquire wait, for C and PB.** For each `predict`, the time from the shim's native
  "after" stamp to the Python stamp taken once ctypes has the GIL back. Reported as P50, P95,
  P99 and mean over the forwards in `hetero` windows.
- **ANE stage timings, in-process ANE only (A, C, PB), per forward in `hetero` windows:**

  | stage | from | to | what it covers |
  |---|---|---|---|
  | features | service_start | predict entry | `ane_features` |
  | pre | predict entry | native before | building or writing the inputs, the pool push, the call into the shim |
  | native | native before | native after | the `predict` send itself |
  | re-acquire | native after | Python after | waiting for the GIL back |
  | post | Python after | predict exit | reading the outputs, the pool pop |
  | tail | predict exit | service_end | the action head |

  A holds the GIL for the whole `predict` and has no native stamps. For A, the three middle
  stages are reported as one `predict` stage.
- **GPU completion collisions:**
  - **Fixed exposure window.** This uses the prior-evidence definition: an ANE request collides
    if any GPU `received_ns` falls in `[service_start − 1 ms, service_start + 0.3 ms]`.
  - **Tail.** An ANE short-stream request in a `hetero` window is in the tail if its
    `response_ns − submit_ns` is above the P99 of the same quantity for the model's P, pooled
    over P's `hetero` windows.
  - **Reported:** the fraction of requests in the tail, the fraction of tail requests that
    collide, and the odds ratio of tail given collision. The odds ratio adds 0.5 to every
    cell of the 2×2 table.
- **Overlap counts, per ANE forward:** how many parent-side GPU completion intervals
  `[received_ns, response_ns]` overlap the forward (`[service_start, service_end]`), and how
  many overlap its pre stage (`[service_start, native before]`, in-process C and PB only). The
  GPU trace keeps both `received_ns` and `response_ns` for this. #77 discarded
  `response_ns`.
- **Full request trace.** Every request of the heterogeneous instance, both targets, keeps all
  `RequestTrace` timestamps.
- **Executing-thread CPU per forward, ANE and GPU.** `time.thread_time_ns()` across the
  backend's forward, on the thread that runs it, for every forward. This is #77's measure.
- **Per-thread CPU per window.**
  - The CPU time (user + system) of every thread in the benchmark process is read with Mach
    `thread_info`. Threads are named: the two client threads (`client-short`,
    `client-long`), the dispatcher threads (`laya-ane-dispatch` and `laya-gpu-dispatch`, which
    also reads the GPU worker's reply), the probe, and others.
  - Snapshots are taken just before each window's start and just after its end, outside the
    window, so the reading does not load the measured interval.
- **GIL probe.**
  - A daemon thread in the benchmark process sleeps to a 1 ms grid. On each wake it records
    how late it woke: the time from the scheduled tick to the moment it runs Python code again.
    Any lateness beyond the OS wake-up latency is time spent waiting for the GIL.
  - The raw lateness and its P50, P99, mean and fraction above 1 ms are kept for every
    `hetero` window.
  - Its overhead is its own thread CPU, reported as a fraction of one core per window.
  - The probe is the same in P, C and PB. It measures GIL contention whichever thread causes
    it, not only GPU completions.
- **Slow-CPU-state flag, per candidate `hetero` window.** The protocol has no solo windows, so
  the reference is P's matched window, the same pair as the gate uses.
  - The flag uses the `client-short` thread's CPU per request in the candidate window,
    divided by the same quantity in P's matched window. That thread runs the same code in every
    configuration: preparing and routing the request and formatting its answers.
  - The window is flagged when the ratio exceeds 1.25.
  - The same ratio for the ANE executing thread's median CPU per forward is reported
    alongside, not flagged. That thread's code differs by binding.
  - P's own windows are the reference and carry no flag.
  - A direct CPU frequency or residency reading (IOReport) is not taken. It costs about 3.3 ms
    of CPU per sample (`research/energy-sampler/`), which would perturb the measurement.
- **Mismatches,** per window, as `bench_concurrency.py` counts them. These also gate, under
  criterion 1.

## Valid runs

- A run that crashes or fails to start is re-run once, in the same position in the order.
  Both the failure and the re-run are reported. No completed run is discarded.
- Every run must record 3 `hetero` windows, one per cycle, so every pair exists.
- In every `hetero` window, the short stream must be served only by the ANE, and the long
  stream only by the GPU. `analyze.py` reports any run where this does not hold. Such a run is
  reported, not silently dropped.
- A PB run that raises on its buffer-reuse assertion counts as a crashed run.

## Protocol

- **Workload: hetero-only.** This is #57's Part A reduced to the windows the gate uses, and a
  protocol change from #57 and #77, made before any data.
  - The solo_short, solo_long and gpu_only windows are removed. So are the gpu_only
    instance and Part B, which feed nothing in this gate.
  - The hetero sample is not reduced: there are 3 `hetero` windows per run, one per cycle,
    so 6 matched pairs per candidate per model.
  - The comparison is within this experiment only. Numbers here are not set beside #57's or
    #77's without saying that those had solo windows between hetero windows and this does not.
  - The pieces are `scripts/bench_concurrency.py`'s own: `closed_loop`, `stats`,
    `conditions`, its request construction and its inline references. They are driven by
    `scripts/run_config.py`, and the output keeps `run_mix.py`'s layout, so #57's
    `placement_summary` reads it unchanged.
- **Per run:**
  - load the heterogeneous instance and the inline references;
  - run bench_concurrency's warm-up of 5 requests per stream;
  - then run 3 cycles. Each cycle is:
    - 2.0 s idle;
    - **a fixed warm-up**: both streams, closed-loop, through the same instance, for 2.0 s.
      It is not measured. Its request counts and mismatches are recorded. The duration and
      request mix are the same in every configuration and before every hetero window;
    - 0.5 s lead;
    - one measured `hetero` window of 20 s.
- **Identical request streams.** Each stream repeats one fixed request, built by
  bench_concurrency's `make_request(seed=0)` with one question:
  - L128, L96, L512 or L1024 per the shapes below;
  - the same request content in window k of every configuration and every run;
  - it is the same request #57 and #77 used.

  The clients are closed-loop: each sends its next request when the previous one returns, so
  there is no arrival schedule to seed. The request sequence of every window is fixed by
  that seed. Removing the solo windows therefore changes no window's request history, apart
  from what precedes it, which is the same fixed warm-up in every configuration.
- **Other settings:**
  - one closed-loop client per stream;
  - answers checked against the inline references computed per device;
  - the v1.0 shapes: laya L128 / L512, laya-typed-decisions L128 / L1024, laya-multilingual
    L96 / L1024;
  - the heterogeneous instance's ANE binding is the only difference between runs.
- **GPU topology:** a worker process in every configuration, as in #77.
- **Order per model:** P C PB PB C P, one run after another. This is ABBA extended to three
  configurations, so every configuration has the same mean position. That gives
  2 runs × 3 cycles = 6 `hetero` windows per configuration and model.
  - The models run in this order: laya, laya-typed-decisions, laya-multilingual.
  - That is 18 runs in total, about 85 s each: 3 × 24.5 s of cycles, plus load and
    references. That is about 30 min in all, after the bit-identity check (about 15 s).
- **Machine:** idle, on AC power, with the local LLM server and other GPU/ANE services
  stopped. The Core ML E5 cache is not cleared.
- **Versions:** as locked in `uv.lock` (coremltools 9.0, mlx 0.32.x). pyobjc-framework-CoreML
  12.2.2 is added through `uv run --with`, for research only.
