# The transient host slowdown at hetero onset

**Status: run. Reading C: the evidence supports a perf-level placement / performance-state
hypothesis.**
- **How the reading was reached.** The preregistration ([`criteria.md`](criteria.md), #95) was
  merged before any formal run. Issue #94.
- **Validity.** A matched its #92 phenotype in all 4 windows, so the validity guard did not fire.
- **The numbers:** [`tables.md`](tables.md), [`results.json`](results.json); raw data in
  [`raw/`](raw/). No run crashed.

## Results

### Validity guard (A, every hetero window)

| | A here | #92's A |
|---|---|---|
| short P99 per window | 10.80–11.51 ms | 11.0–12.9 ms |
| aggregate req/s | 122.1–122.6 | 122.1–122.7 |
| transient | 0.5 s in each window | at most about 0.3 s (#93) |
| steady host-slow share | ≤ 0.1% | – |
| mismatches | 0 | 0 |

### The transient reproduced

PB-ASYNC had a transient in all 4 transitions: 3.0, 4.0, 0.5 and 15.0 s. The 15 s one (run 2,
after gpu_only) covers most of the steady period, so PB-ASYNC's steady short P99 is 16.36 ms
against A's 10.78 ms. Reading E does not apply.

### Readings (criteria.md)

| # | measure, PB-ASYNC transient vs steady | applies |
|---|---|---|
| A | native Core ML duration: 9.93 ms mean against 9.44 ms (1.05×, +0.49 ms) | no |
| B | native completion → callback entry: P50 0.034 ms against 0.011 ms (+0.023 ms) | no |
| C | ANE dispatcher CPU per forward: 1.40 ms against 0.38 ms (3.7×), **and** a placement or cycle-rate change in every transition | **yes** |
| D | – (C applies) | no |
| E | not every transition recovered by t0 + 5 s | no |

**What the counters show.** In PB-ASYNC's transient buckets, the laya-ane-dispatch, client-short
and Core ML callback threads run almost entirely on the Efficiency cores:
- E share 0.91–1.00, against 0.00 in the steady state.
- Relative effective cycle rate on E: about 1.3–1.5, against about 4.0 on P.
- IPC: about 1.7, against about 3.3.
- ANE dispatcher CPU per forward: about 1.8 ms on E, against 0.31 ms on P.
- The transient ends when these threads move back to P. For example, run 1 cycle 0 is on E from
  0 to 2 s, mixed from 2 to 4 s (E share 0.66), and on P from 4 s on.
- The 15 s transient is the same pattern held longer: the ANE dispatcher's E share is 1.00 up to
  t0 + 5 s and 0.78 over the steady period.

A shows the same move only in its first 0.5 s bucket, at a smaller share (ANE dispatcher E share
0.00–0.54), and is on P from t0 + 0.5 s on.

**What this does not show.** The counters record where CPU time ran and at what cycle rate. They do
not show why the scheduler placed the threads on E, or which heuristic did it; no System Trace was
taken. Why PB-ASYNC stays on E so much longer than A is not tested here. One candidate for a next
experiment is that PB-ASYNC's ANE dispatcher uses about half of A's CPU per forward in the steady
state (0.31 against 0.62 ms), which may change how busy the scheduler judges these threads to be.
That is a hypothesis, not a finding.

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
