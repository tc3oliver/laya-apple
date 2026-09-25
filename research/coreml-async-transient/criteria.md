# Criteria: the transient host slowdown at hetero onset (A and PB-ASYNC)

This file is written and committed before any formal run. Nothing in it changes after data is
seen. Harness smoke runs check structure only: coverage, alignment, cleanup and the presence of
fields. They are not data and are not committed. Issue #94.

## Question

> What causes the transient host slowdown immediately after entering heterogeneous GPU+ANE
> serving?

**This is a root-cause experiment, not a production gate.** Its purpose is to decide which layer
1.5 must fix. It is not meant to make a benchmark pass.

**#92's and #93's verdicts are unchanged.** No production code changes. The #89 five-cell screen is
not resumed.

## Background (#93, existing data only)

- **PB-ASYNC's tail in #92 is a transient.** Each hetero window of PB-ASYNC starts with a host-slow
  episode of 0.8–3.4 s, and then stays normal:
  - GPU return P50 is 0.035 ms;
  - aggregate throughput is at or above A's;
  - client P99 outside the episode is 10.4 ms.
- **The episode has PB-SYNC's host-slow signature:** CPU time per Python stage is about 5×, and it
  rises together with wall time.
- **Episode length by configuration:**
  - A: at most about 0.3 s;
  - PB-ASYNC: 0.8–3.4 s;
  - PB-SYNC: most of each window.
- **Not separable in #92's data:**
  - native completion, from the callback taking the GIL;
  - which cores the threads ran on, and how they were scheduled.

## Configurations

The model is laya, with L128 short and L512 long requests. The GPU runs in a worker process.

| cell | ANE prediction | per-forward stamps |
|---|---|---|
| **A** | production: coremltools, synchronous, GIL held | entry, exit |
| **PB-ASYNC** | #90's async prebound path (`predictionFromFeatures:options:completionHandler:`) | entry, async submit (before and after the send), **native Core ML completion**, Python callback entry, waiter wake, exit |

**How native completion is stamped.** For PB-ASYNC, the handler is wrapped in a small C block
(#90's `async_stamper.m`). The block stamps the native completion with #83's clock, which is
`time.monotonic_ns()`, and then calls the Python block.
- **Same callback path.** The Python handler, its single GIL handoff and the waiter are unchanged.
- **In every PB-ASYNC run.** The wrapper is part of the PB-ASYNC cell in every run, so every
  PB-ASYNC transition carries the same instrumentation.

**Not in this experiment:**
- PB-SYNC, PB-H, PB-W or PB-P;
- QoS changes, a Swift worker or GPU pacing;
- any warm-up used as a mitigation, or a probe;
- a System Trace / Thread Activity recording (dropped before any formal run; see "Instrumentation
  dropped after the smoke" below).

## Protocol

- **Workload.** It is #92's full protocol (`run_mix.py` running `bench_concurrency.py --part a`)
  with **2 cycles** of 20 s windows. Each run has exactly two single-device → hetero transitions:
  - **cycle 0:** solo_short, solo_long, **hetero**, gpu_only, so the hetero window follows
    solo_long;
  - **cycle 1:** gpu_only, **hetero**, solo_long, solo_short, so the hetero window follows
    gpu_only.
- **Around every window:** 2.0 s idle and a 0.5 s lead.
- **Clients and requests:** one closed-loop client per stream, with identical requests
  (`make_request(seed=0)`, one question).
- **One fresh process per run.**
- **Order (fixed):** A, PB-ASYNC, PB-ASYNC, A. Every run has cycles 0 and 1, so each cell has 4
  transitions, 2 after solo_long and 2 after gpu_only.
- **The same instrumentation in every run:** the per-thread counters in both cells, and the native
  stamps in every PB-ASYNC run. They cover all 8 transitions.
- **t0** is the hetero window's start (`start_at`). Every measure is aligned to it.

## Evidence hierarchy

| tier | evidence | covers |
|---|---|---|
| **Primary** | native Core ML completion stamps, the request trace and request-stage timings, and per-thread perf-level counters (`PROC_PIDTHREADCOUNTS`) | all 8 transitions, both cells |
| **Optional** | IOReport cluster telemetry | only if available unchanged; never blocks the experiment |

There is no secondary tier. No System Trace or Thread Activity recording is taken, so there is no
runnable, waker or migration evidence, and no reading rests on it.

## Primary evidence (every transition)

Every transition records the following:
- **The per-forward stamps above.** For PB-ASYNC these give:
  - **native Core ML duration:** native completion − submit_after;
  - native completion → Python callback entry;
  - callback entry → waiter wake;
  - post;
  - action head.
- **The request trace, per short request:**
  - prepare, route, enqueue, queue / dispatcher wake;
  - dispatch → service start, features;
  - service end → response;
  - client latency.
- **Executing-thread CPU per forward:** the ANE dispatcher thread, and the GPU worker thread.
- **Per-thread CPU snapshots per window:** client-short, client-long, laya-ane-dispatch,
  laya-gpu-dispatch, and the Core ML callback thread(s).

**Per-thread perf-level counters.** These are research instrumentation only, and never production
code.
- **The interface.** `proc_pidinfo(pid, PROC_PIDTHREADCOUNTS, thread_id, ...)` returns, for each
  thread and each CPU performance level, the instructions, cycles, CPU time and energy.
  - This machine has 2 levels (`hw.nperflevels`): `Performance`, with 12 logical CPUs, and
    `Efficiency`, with 4.
  - It is an undocumented XNU interface (`proc_info_private.h`, flavor 34), called through ctypes.
  - A failed read is recorded, and the run continues.
- **The sampler.** A thread named "laya-recount" samples every 100 ms, for the whole run, in both
  cells. It reads these threads:
  - client-short and client-long;
  - laya-ane-dispatch and laya-gpu-dispatch;
  - every Core ML callback thread seen;
  - the main thread;
  - all threads of the GPU worker process, listed each sample with `PROC_PIDLISTTHREADIDS`.

  Its own CPU is recorded.
- **Feasibility, measured before this file was merged.** One read of 8 threads takes a median of
  141 µs. At 100 ms that is 0.17% of one core. A background-QoS thread's cycles are attributed to
  `Efficiency`, and another process's threads can be read.
- **Derived per thread and bucket:**
  - CPU time on P and on E, and the E share;
  - **relative effective cycle rate** per level (cycles / CPU ns). This is not a precise physical
    CPU frequency. It is read as a relative change within the same perf level: A against
    PB-ASYNC, and before against after the transition;
  - IPC (instructions / cycles);
  - CPU per request or forward.

These records decide every mechanistic reading.

**What the counters cannot show.** They record where CPU time was spent and at what cycle rate. They
do not record runnable waits, wakeups, wakers or migrations. No reading below names a scheduler
wakeup, runnable latency or a specific scheduler heuristic as the cause.

## Instrumentation dropped after the smoke (before any formal run)

The first draft of this file (never merged) had a secondary tier: a Thread Activity trace (`xctrace`) of one
transition in each PB-ASYNC run, with a rule for whether the trace perturbed the transient. The
structure-only smoke showed the recording is not usable on this machine:
- it saw 8 of 35 marker wakes (23%), against the 90% completeness the harness required;
- the ANE dispatcher had 364 running intervals over a span with 489 forwards;
- each recording wrote about 12.8 GB of temporary ktrace data and blocked the run for about 173 s
  of processing.

The trace, its perturbation rule and every runnable, waker or migration reading are therefore
removed, before any formal run and before any formal data exists. No other recording setup is
tried. The primary evidence and every threshold below are unchanged from the first draft.

**Transient definitions** (#93's, used below):
- **Host-slow request:** a short request whose client-thread `prepare` is above 0.3 ms.
- **Bins:** 0.5 s bins from t0 to t0 + 20 s. The host-slow share of a bin is taken over the short
  requests submitted in it.
- **Recovery point:** the start of the first bin from which every later bin, up to t0 + 20 s, has a
  host-slow share below 10%.
- **Transient duration:** recovery point − t0, or 0 if the first bin already qualifies. A
  transient is **present** if its duration is ≥ 0.5 s.
- **Peak short P99:** the maximum, over 1 s bins in [t0, t0 + 5 s), of the P99 of the short
  client latency.

## Analysis

**Buckets**, aligned at t0:

| buckets |
|---|
| −2–0 s |
| 0–0.5 s |
| 0.5–1 s |
| 1–2 s |
| 2–4 s |
| 4–5 s |
| steady: 10–20 s |

The onset, peak and recovery of the transient are reported per transition.

**Per bucket and transition, reported:**
- **Latency:** short client P50 / P99 and the host-slow share.
- **ANE stages:**
  - features wall;
  - native Core ML duration;
  - native completion → callback entry;
  - callback → waiter wake;
  - post;
  - action head.
- **Host CPU:**
  - ANE dispatcher CPU per forward;
  - GPU host CPU per forward;
  - client-thread prepare, plus client CPU per request per window.
- **Perf-level counters:** E share, relative effective cycle rate per level, and IPC.
- **IOReport cluster frequency and residency,** if it is recorded (below).

**Period definitions.** The transient period is [t0, recovery point). Steady is [t0 + 10 s,
t0 + 20 s).

**How the readings are evaluated.** Every reading that applies is reported. There is no single
verdict. Readings A–D are hypotheses for the next step, not proven causes.

| # | condition | reading |
|---|---|---|
| A | PB-ASYNC's mean native Core ML duration over its transient periods is more than 1.10× its steady mean **and** more than 0.3 ms above it | the Core ML / ANE runtime transition becomes the leading hypothesis |
| B | Not (A). The P50 of native completion → callback entry over PB-ASYNC's transient periods is more than 3× its steady P50 **and** more than 0.2 ms above it | the callback / GIL handoff becomes the leading hypothesis |
| C | PB-ASYNC's mean ANE-dispatcher CPU per forward over its transient periods is at least 2× its steady mean, **and** the per-thread counters show a placement or cycle-rate change (below) | supports a perf-level placement / performance-state hypothesis. No specific scheduler heuristic is named |
| D | PB-ASYNC's CPU per forward rises as in (C), but the counters show no placement change and no drop in the relative effective cycle rate | the slowdown is not explained by the measured placement / cycle-rate factors. If IPC also drops (below): supports further investigation of shared-resource / memory-hierarchy effects. Cache contention is not concluded |
| E | PB-ASYNC has a transient in at least one transition, every PB-ASYNC transition recovers before t0 + 5 s, and its steady state keeps: GPU return P50 ≤ 1 ms; aggregate req/s ≥ 0.95 × A's steady; short client P99 ≤ 1.05 × A's steady | a production-safe priming / residency strategy is studied next. The existing 2 s benchmark warm-up is not a fix |

**A placement or cycle-rate change**, for (C) and (D), comes from the per-thread counters, in every
transition. It is any of these for the laya-ane-dispatch, client-short or Core ML callback
threads, transient against a reference:
- the E-level share of CPU time rises by at least 25 percentage points;
- the P-level relative effective cycle rate is at most 0.8× the reference.

**The IPC drop for (D):** IPC at most 0.8× the reference on one of those threads, with no E/P
placement change and no drop in the relative effective cycle rate. It is reported, and it only
qualifies reading D as above.

The reference is [t0 + 4 s, t0 + 5 s) when the transition recovered by t0 + 4 s. Otherwise it is
[t0 − 2 s, t0).

**If PB-ASYNC shows no transient** in any of its 4 transitions: "phenotype not reproduced: no causal
reading".

**A is the baseline.** Its short transient (#93) is reported in the same buckets. A has no native
completion stamp, so readings A and B are evaluated on PB-ASYNC only.

## Validity guard (A as the internal control)

The counters' overhead is low, but the smoke cannot show they perturb nothing: one pair of 5 s
windows with the counters on and off differed, in a direction and size that one window cannot
separate from noise. Both cells therefore run with the same counters, and A is the internal
validity control against its historical phenotype (#92's A, and #93's analysis of it).

**A validity concern is flagged** if any A hetero window shows any of:
- a short client P99 over the whole window ≥ 13.0 ms (R1's state classifier; #92's A windows were
  11.0–12.9 ms);
- a transient duration above 1.0 s (#93: A's host-slow state lasted at most about 0.3 s);
- a host-slow share ≥ 10% over steady [t0 + 10 s, t0 + 20 s), a sustained host-slow state;
- an aggregate req/s outside [109.9, 135.0], which is 0.9× to 1.1× #92's A range of 122.1–122.7;
- any mismatch or wrong routing.

**If it is flagged:** every number is still reported, the concern is stated with the window and the
trigger, and readings A–E are not drawn. Other anomalies seen in A are also recorded as a validity
concern. No conclusion is forced.

## IOReport (supplemental)

Cluster frequency and residency are recorded only if an existing sampler in this repository can be
used unchanged, at up to 10 Hz, with small overhead. Otherwise the run records "IOReport skipped"
and the reason. Cluster-level telemetry is never used as evidence of which core a particular
thread ran on.

## Valid runs and time

- **Crashes.** A run that crashes is re-run once, in the same position. A second failure stops the
  experiment, and it is reported.
- **Required in every run:** 2 hetero windows with 0 mismatches and correct routing.
- **Machine:**
  - idle, on AC power;
  - the local LLM server and other GPU/ANE services stopped, and restored afterwards;
  - the Core ML E5 cache not cleared.
- **Versions:** as locked in `uv.lock`, plus pyobjc-framework-CoreML 12.2.2 through
  `uv run --with`. The shims and the C block are compiled with `xcrun clang` at run start (research
  only).
- **Time.**
  - Each run has 8 windows of 20 s with 2.5 s gaps, plus load and references: about 4 min.
  - 4 runs: about 16 min.
