# A dependency QoS override for the hetero-onset E-core residency

**Status: run. Outcome: STOP this USER_INITIATED dispatcher-QoS route.**
- **How the outcome was reached.** The preregistration ([`criteria.md`](criteria.md), #98) was
  merged before any screen run. Issue #97.
- **Round 1.** Neither O nor Q moved the ANE dispatcher off the E cores in any transition. B was
  valid, so by the preregistered stop rule round 2 did not run.
- **What follows.** No further QoS variants and no USER_INTERACTIVE trial-and-error. The next step
  is re-evaluating the architecture or scheduler-state priming.
- **The numbers:** [`tables.md`](tables.md), [`results.json`](results.json); raw data in
  [`raw/`](raw/). No run crashed.

## Results (round 1)

| run | transition | ANE dispatcher E share over [t0 + 0.5, t0 + 4) | class | transient s |
|---|---|---|---|---|
| B | after solo_long | 0.99 | E-resident | 20.0 |
| B | after gpu_only | 0.99 | E-resident | 17.5 |
| O | after solo_long | 1.00 | E-resident | 4.5 |
| O | after gpu_only | 1.00 | E-resident | 12.0 |
| Q | after solo_long | 0.70 | E-resident | 2.0 |
| Q | after gpu_only | 1.00 | E-resident | 20.0 |

- **B is valid.** Both of its transitions are E-resident, with 0 mismatches, a GPU-return P50 of
  0.24 ms, and 100.9 and 103.6 req/s per hetero window.
- **The manipulations took effect structurally:**
  - **O:** 7,664 overrides started and 7,664 ended, with 0 NULL starts, 0 end errors and 0
    outstanding. Start and end each cost 1.6 µs P50.
  - **Q:** the requested QoS read back as 0x19 after the call and at the end.
  - A harness probe before the merge showed that an override raises a blocked thread's scheduling
    priority from 31 to 37, and releases it on end.
- **O and Q each "changed nothing"** by the preregistered definition: no transition had the
  dispatcher's E share at or below 0.10. CHAIN-E held in every transition.
- **Guards passed for both candidates.**
  - native Core ML mean: 9.57 ms (O) and 9.65 ms (Q), against 9.85 ms (B);
  - GPU-return P50: 0.042 and 0.160 ms;
  - 0 mismatches.

**Observations, not gating.**
- **The three threads move together.** In every transition the dispatcher, client-short and
  callback threads have nearly the same E share, for example 0.98 / 0.98 / 0.98 in B, and
  0.70 / 0.71 / 0.65 in Q's first transition. That includes the callback threads, which run on a
  Core ML dispatch queue that neither O nor Q touches. What makes them move together is not tested
  here.
- **B's transients this time (20.0 and 17.5 s) are longer than #96's PB-ASYNC transients**
  (0.5–15 s). The transient durations vary a lot from run to run, so the differences in transient
  length between B, O and Q in one run each are not read as effects.

## Post-hoc: where the E-core residency sits (#99 raw data only, not a gate)

`scripts/placement_posthoc.py` splits #99's per-thread counters by process and thread group. The
output is [`placement_posthoc.md`](placement_posthoc.md) and `placement_posthoc.json`. The group
definitions, the active-thread rule and the P1–P4 classification are fixed in the script's
docstring, written before its output was read. **This does not change #99's outcome.**

**The groups:**
- **chain:** laya-ane-dispatch, client-short and the Core ML callback;
- **parent-other:** MainThread, client-long and laya-gpu-dispatch;
- **worker:** every thread of the GPU worker process.

Only active threads (at least 20 ms of CPU in [t0, t0 + 4 s)) enter a group's CPU-weighted
aggregate.

**Classification: P3, cross-process, in all 6 transitions.**
- **Every group is E-dominant together.** Over [t0 + 0.5, t0 + 4 s), chain, parent-other and
  worker are E-dominant in every transition: 0.99–1.00, and 0.67–0.76 in Q's first transition.
- **The two processes switch back together.** The E→P switch times of the active threads in both
  processes fall in the same 0.5 s bin, or within one bin of it:

  | run, transition | switch |
  |---|---|
  | O cycle 0 | 4.0–4.5 s |
  | Q cycle 0 | 2.0 s |
  | O cycle 1 | 11.5–12.0 s |
  | B cycle 1 | 17.0–17.5 s |
  | B cycle 0 | 19.5 s |
  | Q cycle 1 | never within 20 s |

- **Group aggregates are not averaging different threads.** IPC and the relative effective cycle
  rate on E are similar across the groups.
- **The residency is specific to the hetero windows.** In the single-device windows of the same
  runs, the sampled CPU of both processes is on P (E share ≤ 0.04). In the hetero windows it is
  0.25–0.99.
- The [t0 − 2 s, t0) baseline has only 0–31 ms of CPU per group. Its E shares are not read.

**The limits of this data:**
- **Sampled threads only.** The recount sampler reads only the listed parent threads and the auto
  instance's GPU worker. Other parent threads, the separate GPU-only instance and other processes
  on the machine are not in the data. "Cross-process" here means the two laya processes. It is not
  shown to be system-wide.
- **No cause.** What places both processes on E during hetero serving, and what moves them back,
  is not observable in these counters.

## Question

Can a public, production-viable QoS mechanism prevent the PB-ASYNC request dependency chain from
remaining on E-cores during hetero onset, while preserving GPU completion isolation and normal
steady-state behaviour?

## Why

- **#96 measured the placement.** During PB-ASYNC's hetero transients, the ANE dispatcher,
  client-short and Core ML callback threads ran almost entirely on E cores. The transient ended
  when they moved back to P.
- **Other layers were ruled out as the explanation.** Native Core ML duration and the callback
  handoff did not explain the regression.
- **This is a mitigation screen.** It tests one public mechanism against that placement. It does
  not ask why the scheduler chose it.
- **QoS was tested before.** v0.2's USER_INTERACTIVE QoS did not help the request-driven slowdown of
  that time (`research/v0.2-concurrency/README.md`). The topology, the execution path and the
  measured mechanism differ now.

## Design

| | setting |
|---|---|
| model | laya, L128 / L512, GPU in a worker process |
| B | #94's PB-ASYNC, unchanged |
| O | B plus a `pthread_override_qos_class_start_np` USER_INITIATED override on the ANE dispatcher while each short request is outstanding |
| Q | B plus `pthread_set_qos_class_self_np(USER_INITIATED)` on the ANE dispatcher |
| instrumentation | #94's: native Core ML stamps, request trace, per-thread perf-level counters |
| deciding measure | ANE dispatcher E-core share over [t0 + 0.5 s, t0 + 4 s), together with the transient duration |
| rounds | round 1: B, O, Q. Round 2 runs automatically for passing candidates, in reverse order, then B |
| time | about 10 min per round, 20 min at most |

## Run

```sh
# Idle machine on AC power, local LLM server stopped. Resumable.
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-dependency-qos/scripts/run_all.sh
uv run python research/coreml-dependency-qos/scripts/analyze.py      # --check to verify
```

Unit tests: `tests/unit/test_dependency_qos.py`.
