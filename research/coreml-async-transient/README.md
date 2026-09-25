# The transient host slowdown at hetero onset

**Status: preregistered, not run.** The criteria are in [`criteria.md`](criteria.md), committed
before any formal run. Issue #94.

## Question

What causes the transient host slowdown immediately after entering heterogeneous GPU+ANE serving?

## Why

- **The transient.** #93 found that PB-ASYNC's short tail in #92 is a host-slow episode of
  0.8–3.4 s at the start of each hetero window.
- **After it, the async path meets most of the 1.5 goal:**
  - GPU completion isolation;
  - throughput at or above production;
  - client P99 of 10.4 ms outside the episode.
- **Purpose.** This experiment decides which layer 1.5 must fix. It is not a production gate.

## Design

| | setting |
|---|---|
| model | laya, L128 / L512, GPU in a worker process |
| cells | A (production) and PB-ASYNC (#90's async path, with native Core ML completion stamped) |
| runs | 4 fresh processes: A, PB-ASYNC, PB-ASYNC, A |
| transitions | 2 per run, one after solo_long and one after gpu_only; 4 per cell |
| evidence | every transition: native Core ML completion, the request stages, and per-thread perf-level counters (CPU time, cycles and instructions on P and E, sampled every 100 ms). No System Trace: the smoke showed it misses most scheduling events on this machine |
| validity guard | A is the internal control: if it departs from its #92 phenotype (a slow window, a transient above 1 s, a sustained host-slow state, throughput out of range, a mismatch), no causal reading is drawn |
| time | about 16 min |

## Run

```sh
# Idle machine on AC power, local LLM server stopped. Resumable.
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-async-transient/scripts/run_all.sh
uv run python research/coreml-async-transient/scripts/analyze.py      # --check to verify
```

Unit tests: `tests/unit/test_async_transient.py`.
