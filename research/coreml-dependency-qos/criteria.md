# Criteria: a dependency QoS override for PB-ASYNC's hetero-onset E-core residency

This file is written and committed before any screen run. Nothing in it changes after screen data
is seen. Harness smoke runs check structure only: coverage, the override bookkeeping, alignment and
the presence of fields. They are not data and are not committed. Issue #97.

## Question

> Can a public, production-viable QoS mechanism prevent the PB-ASYNC request dependency chain
> from remaining on E-cores during hetero onset, while preserving GPU completion isolation and
> normal steady-state behaviour?

**This is a mitigation screen, not a root-cause screen.** It asks whether one public mechanism
changes the placement #96 measured, and whether the latency transient goes with it. It does not
ask why the scheduler places the threads on E.

**Unchanged:** the verdicts of #92, #93 and #96, production code, `laya_apple/data/placement.json`,
and any release. The #89 screen is not resumed.

## What #96 measured

In PB-ASYNC's hetero transients (`research/coreml-async-transient/`):
- the laya-ane-dispatch, client-short and Core ML callback threads had an E-core CPU share of
  0.91–1.00, against 0.00 in the steady state;
- the ANE dispatcher's CPU per forward rose from 0.38 to 1.40 ms (means over the transient and
  steady periods);
- each transient ended when those threads moved back to the P cores;
- native Core ML duration (1.05×) and the native completion → callback handoff (+0.02 ms) did not
  explain the regression;
- the counters cannot say why the scheduler made that placement.

In 3 of #96's 4 PB-ASYNC transitions the ANE dispatcher stayed on E well past t0 + 0.5 s. In one
(run 2, cycle 0) it was back on P by t0 + 0.5 s, with a 0.5 s transient.

## History: QoS was tested before

**This is not the first QoS test in this repository.** v0.2 set USER_INTERACTIVE QoS on the
worker and dispatcher threads, verified that it took effect (0x15 → 0x21), and it did not improve
the request-driven process slowdown of that time (`research/v0.2-concurrency/README.md`,
"Mitigations that did not help").

**Why it is tested again:**
- the execution topology differs: the ANE runs in-process on the async prebound path (PB-ASYNC),
  with the GPU in a worker process;
- #96 is the first direct measurement that the transient coincides with E-core residency of the
  request's threads;
- the lead candidate is a different mechanism: a temporary dependency override, not a raised
  requested QoS.

## The mechanisms (public API only)

Both are declared in the macOS SDK's `pthread/qos.h`, available from macOS 10.10, and called
through ctypes from research code.

**`pthread_override_qos_class_start_np(pthread_t, qos_class_t, int)` /
`pthread_override_qos_class_end_np(pthread_override_t)`.** In the header's words, starting an
override "expresses that an item of pending work classified with the specified QOS class and
relative priority depends on the completion of the work currently being executed by the thread".
- While overrides are in effect, the target runs at the maximum of all overrides and its own
  requested QoS.
- The override does not change the requested QoS, and is not visible through
  `pthread_get_qos_class_np()`.
- The start returns an override object, or NULL on failure. Every object must be ended exactly once,
  or the target stays elevated. The starting and ending threads need not be the same.

**`pthread_set_qos_class_self_np(qos_class_t, int)`** sets the calling thread's requested QoS,
readable back with `pthread_get_qos_class_np()`.

**Not used:** private scheduler or affinity interfaces, core pinning, USER_INTERACTIVE, or any QoS
on threads other than the ANE dispatcher.

## Cells

laya only, L128 short and L512 long, GPU in a worker process. Every cell is #94's PB-ASYNC with the
same instrumentation (below).

| cell | change from #94's PB-ASYNC |
|---|---|
| **B** | none: the baseline |
| **O** | a dependency override: for each short request, from just before it is submitted until its response returns, an override of `QOS_CLASS_USER_INITIATED` (relative priority 0) on the ANE dispatcher thread |
| **Q** | a requested QoS: the ANE dispatcher thread calls `pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0)` once, on itself, when it loads |

**O in detail:**
- The ANE dispatcher's `pthread_t` is captured on the dispatcher thread itself (`pthread_self()`)
  when that thread starts, before its first job. Model load and warm-up run on a separate loader
  thread, so this is the dispatcher's earliest point. The capture runs in every cell.
- The override covers every short request of the serving instance, in solo_short windows as well
  as hetero. The warm-up predicts before part_a are not covered.
- The client thread of each short request starts the override and ends it in a `finally`, so an
  exception cannot leak it. Each request has its own override object, so concurrent overrides are
  independent. A registry of outstanding overrides is checked, and any left are ended, at the end
  of the run.
- Recorded: start and end counts, NULL starts, non-zero end results, the outstanding count at the
  end, and each override's start and end time.
- Nothing else changes: not the long stream, the GPU path, the callback queue or the Core ML
  call.

**Q in detail:** the call is made at the same point as the capture: on the dispatcher thread, when
it starts, before its first job. The dispatcher's requested QoS is read back with `pthread_get_qos_class_np()` right
after the call, and at the end of the run.

**Every cell records** the dispatcher's requested QoS at load and at the end. In B and O it is
expected to be DEFAULT (0x15). For O this is not a check of the override: by the header's
semantics an override does not change the requested QoS and is invisible to
`pthread_get_qos_class_np()`, so a DEFAULT read-back in O is expected, not a failure.

**Supplemental, never gating:** the dispatcher's scheduling priority, read with `proc_pidinfo`
(`PROC_PIDTHREADID64INFO`, research only), with and without an override active. Whether O
changed execution is judged only from the primary evidence: the ANE dispatcher's P/E residency,
its relative effective cycle rate, its CPU per forward, and the onset transient duration.

**USER_INTERACTIVE is not tested.** If USER_INITIATED changes nothing, a USER_INTERACTIVE ceiling
test is not run from this screen (see the outcomes).

## Protocol and instrumentation (#94's, unchanged)

- #92's full protocol (`run_mix.py` → `bench_concurrency.py --part a`) with 2 cycles of 20 s
  windows. Each run has two transitions: cycle 0 after solo_long, cycle 1 after gpu_only.
- 2.0 s idle and a 0.5 s lead; one closed-loop client per stream; identical requests
  (`make_request(seed=0)`).
- One fresh process per run.
- **The same in every run:**
  - native Core ML completion stamps;
  - the request trace;
  - the per-thread `PROC_PIDTHREADCOUNTS` counters every 100 ms (research only);
  - per-window thread CPU snapshots.

  #94's `recount.py` and its derived measures are reused unchanged.
- No System Trace, no IOReport, no warm-up change, no probe, no Swift worker, no GPU pacing.

## Measures, per transition

t0 is the hetero window's start. #94's definitions are used:
- **host-slow request:** client-thread `prepare` > 0.3 ms;
- 0.5 s bins;
- **recovery point**, and **transient duration** (present if ≥ 0.5 s);
- **peak short P99:** the maximum, over 1 s bins in [t0, t0 + 5 s), of the P99 of the short client
  latency.

**Onset measures, over [t0, t0 + 4 s), reported in #94's buckets:**
1. the ANE dispatcher's E-core CPU share;
2. client-short's E-core share;
3. the callback threads' E-core share;
4. the ANE dispatcher's CPU per forward;
5. the transient duration.

**The deciding placement measure** is the ANE dispatcher's E-core share of CPU time over
**W = [t0 + 0.5 s, t0 + 4 s)**. W starts at t0 + 0.5 s because A itself briefly used E cores in its
first 0.5 s bin (#96: E share 0.00–0.54), and was on P after that.
- **E-resident (#96-type):** E share over W ≥ 0.50.
- **Avoided:** E share over W ≤ 0.10.
- **Unassessable:** no counter data for the dispatcher over W. It counts as neither E-resident nor
  avoided.
- **Between these:** partial. It is reported, and it counts as not avoided.

## Checks per run

**Guards: a candidate (O or Q) against B in the same round:**
- **native Core ML duration:** the mean over both hetero windows is not more than 1.05× B's mean
  **and** 0.3 ms above it;
- **GPU completion isolation:** the GPU-return P50, pooled over both hetero windows, is ≤ 1 ms;
- **correctness:** 0 mismatches and correct routing in every window;
- **throughput:** aggregate req/s, pooled over both hetero windows (all requests of both streams
  over the total window time), is ≥ 0.95× B's;
- **O only, structural validity of the override:**
  - every `pthread_override_qos_class_start_np()` returned non-NULL;
  - every override has exactly one matching end;
  - every `pthread_override_qos_class_end_np()` returned 0;
  - 0 overrides outstanding at the end of the run;
  - the target's lifetime is correct: every override starts after the dispatcher's `pthread_t`
    was captured and ends before the dispatcher thread exits;
  - at least one override was active in each hetero window.

  The requested-QoS read-back is not part of O's validity (above);
- **Q only:** the read-back requested QoS is USER_INITIATED (0x19) after the call and at the end.

**B validity: the phenotype must reproduce.** B is valid in a round when all of these hold:
- at least 1 of its 2 transitions is E-resident;
- 0 mismatches, and a GPU-return P50 ≤ 1 ms;
- aggregate req/s per hetero window within [96.5, 141.7]: 0.9× to 1.1× the range of PB-ASYNC's
  hetero windows in #92 and #96, 107.3–128.8. The low end comes from #96's 15 s transient, since
  the phenotype itself costs throughput.

If B is not valid, the round is **INCONCLUSIVE**: no candidate is judged in it, nothing replicates
automatically, and the next step is a human decision.

## Outcomes

A candidate is judged in a round only when B is valid in that round.
- **MECH:** both of its transitions avoided E residency (the ANE dispatcher's E share over W is
  ≤ 0.10).
- **LAT:** both of its transitions have a transient duration ≤ 1.0 s.
- **CHAIN-E:** in at least one of its transitions, client-short or the Core ML callback threads
  have an E-core share over W of at least 0.50. #96 measured that the affected chain is more than
  the dispatcher: client-short and the callback also ran on E during the transients.

| MECH | LAT | guards | round outcome for the candidate |
|---|---|---|---|
| yes | yes | pass | **PASS**: it replicates (below) |
| no | yes | any | latency improved, but #96's mechanism did not change: QoS is not claimed to fix it; no replication |
| yes | no, with CHAIN-E | any | **dispatcher-only QoS changes placement but is insufficient.** The rest of the dependency chain stays on E. This is not a disproof of the QoS route; no replication |
| yes | no, without CHAIN-E | any | QoS affects placement but is insufficient; not productionized; no replication |
| no | no | any | the dispatcher's placement did not change enough; no replication |
| yes | yes | fail | the failed guard is reported; no replication |

**Placement change, per candidate.** A candidate *changed nothing* if none of its transitions
avoided E residency. It changed placement *in some transitions* if one did but not both (no
MECH).

**Stop rule.** If B is valid and both O and Q changed nothing in round 1 (every O and Q transition
stayed E-resident or partial on the dispatcher): **STOP this USER_INITIATED dispatcher-QoS
route.**
- No further QoS variants, and no USER_INTERACTIVE trial-and-error.
- A USER_INTERACTIVE ceiling test may be opened later only as a separately preregistered
  *diagnostic*, by human decision. It is never a mitigation candidate from this screen.
- The next step is re-evaluating the architecture or scheduler-state priming.

**If a candidate was "dispatcher-only QoS changes placement but is insufficient"** (and nothing
passed): this allows exactly one follow-up, an architecture decision. The decision is whether a
clean, public, production-safe way exists to propagate QoS along the whole dependency chain,
covering client → dispatcher → Core ML callback. The follow-up does not try USER_INTERACTIVE, a
warm-up, a probe or any other parameter.

**Anything else without a PASS** (placement changed in some transitions only, or a latency change without MECH)
is reported as observed, nothing replicates, and the next step is a human decision. It does not
start parameter trials.

**Replication (round 2, automatic).** Round 2 runs only if round 1 has at least one PASS and a
valid B.
- **Round 2 runs:** the passing candidates, in the reverse of their round-1 order, then B. Round 1
  runs B, O, Q, so round 2 is Q, O, B; or O, B; or Q, B.
- **Judging:** round 2 is judged alone, by the same rules, including B's validity.
- **Leading candidate:** a candidate that passes both rounds.
- **Otherwise INCONCLUSIVE** for that candidate: not productionized, and the next step is a human
  decision.
- **No third round.**

**Preference, fixed now.** If O and Q both lead, O is the leading candidate: a temporary dependency
override is preferred over a permanently raised QoS, which may cost power and system scheduling
for no benefit. Q is reported as the fallback.

**A leading candidate is not a production change.** Before any productionization, a separate
change has to cover:
- the override lifetime with several concurrent requests, cancellation, exception paths and
  shutdown;
- no override leak;
- energy;
- typed-decisions;
- the full product-mix gate.

## Valid runs and time

- **Crashes.** A run that crashes is re-run once, in the same position. A second failure stops the
  screen, and it is reported.
- **Machine:**
  - idle, on AC power;
  - the local LLM server and other GPU/ANE services stopped, and restored afterwards;
  - the Core ML E5 cache not cleared.
- **Versions:** as locked in `uv.lock`, plus pyobjc-framework-CoreML 12.2.2 through `uv run --with`.
- **Time:** about 3.1 min per run (#96). Round 1 is 3 runs, about 10 min. Round 2 is at most 3 runs,
  about 10 min. The maximum is 6 runs, about 20 min.
