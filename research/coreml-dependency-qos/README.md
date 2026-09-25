# A dependency QoS override for the hetero-onset E-core residency

**Status: preregistered, not run.** The criteria are in [`criteria.md`](criteria.md), committed
before any screen run. Issue #97.

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
