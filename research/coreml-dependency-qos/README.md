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
