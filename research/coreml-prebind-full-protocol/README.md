# Prebound predict under the full product-mix protocol (R1)

**Status: run, FAIL (futility stop at n = 12 on laya).**
- **How the verdict was reached.** The criteria are in [`criteria.md`](criteria.md), committed
  before any campaign data. Addendum 1 ("fast fail, slow pass") was merged before any result was
  read.
- **The stop.** At the n = 12 interim look, laya's PB short P99 against production A was
  1.409 × [1.223, 1.623] (99% t-interval). That is entirely above the 1.05 budget, so R1 stopped
  for futility.
- **What follows.** Per the preregistered outcome, R1 is a FAIL, and a protocol-history causal
  experiment is preregistered next. There is no production change, and no move to a Swift worker
  or to GPU pacing.
- **#77's and #83's verdicts are unchanged.**
- **The numbers:** [`tables.md`](tables.md), [`results.json`](results.json); raw data in
  [`raw/`](raw/).

## Results

16 runs: block 1 (P C PB PB C P) on both models, then laya block 2 (PB P P PB).
- There were 0 mismatches and no crashes.
- PB and C were bit-identical to coremltools on every ANE bucket of both models
  (`raw/check.json`).
- Each run took 277–281 s.

| look | model | aggregate | short P99 | long P99 | GPU return P50, A → PB | decision |
|---|---|---|---|---|---|---|
| n = 6 (99%) | laya | 0.917 [0.771, 1.091] | 1.362 [0.967, 1.918] | 1.011 [0.873, 1.171] | 4.37 → 0.20 ms (×21.7) | continue |
| n = 6 (99%) | laya-typed-decisions | 1.044 [1.039, 1.049] | 0.980 [0.879, 1.092] | 0.920 [0.867, 0.975] | 8.50 → 0.04 ms (×212.5) | continue |
| n = 12 (99%) | laya | 0.910 [0.833, 0.994] | **1.409 [1.223, 1.623]** | 1.034 [0.975, 1.097] | 4.35 → 0.21 ms (×20.3) | **futility stop** |

laya-typed-decisions was not run past block 1: a futility stop on either model ends R1.

**C, the mechanistic control, at n = 6 (95%, reference only).**
- It FAILed on both models: short P99 1.312 [1.119, 1.538] on laya and 1.407 [1.109, 1.785] on
  typed.
- **#77's C regression reproduces under the full protocol.** Under #83's hetero-only protocol it
  did not.

### Not gating: a two-state short tail

The measurements below are observations, not a verdict or a cause. Per-run numbers are in
`raw/`; the window-history split is in `tables.md`.
- **Two states.** Every C and PB hetero window's short P99 sits in one of two clusters:
  - normal, 10.4–12.3 ms, which is A's level (10.9–12.2 ms);
  - slow, 13.6–21.2 ms.

  The split is at 13 ms.
- **Only the hetero windows.** A's hetero windows and every configuration's solo_short windows
  stay normal. PB's solo_short P99 is 10.0–10.2 ms, against A's 10.3–10.7 ms.
- **How often each configuration is slow:**

  | model | configuration | hetero windows slow |
  |---|---|---|
  | laya | PB | 11 of 12 |
  | laya | C | 5 of 6 |
  | laya-typed-decisions | PB | 0 of 6 |
  | laya-typed-decisions | C | 5 of 6 |

- **The slow state inflates host CPU work across the process.** Against A:
  - laya PB's ANE features stage is 0.64 ms, against 0.16;
  - its post-forward tail is 0.31 ms, against 0.06;
  - GPU-thread CPU per forward is 6.1 ms, against 1.8;
  - the client-short thread's CPU per window is 1308 ms, against 307.

  This is the host-side slowdown #58 described for process isolation.
- **The native `predict` does not change**: 9.67 ms for laya PB, against 9.62 ms for A's whole
  `predict`.
- **Window history does not separate the states.** They occur after both solo_long and gpu_only.

## Question

Under #57's and #77's original, complete product-mix protocol, does PB (#83's prebound binding)
pass the production gate on both laya and laya-typed-decisions, and keep GPU completion
isolation?

## Why

- **#83** recorded PB PASS on laya and laya-typed-decisions under a hetero-only protocol. Under
  that protocol, C also passed, although #77 had recorded C as a FAIL on both models under the
  full protocol. So #83's pass depends on the protocol or on what ran before each hetero window.
- **This experiment** repeats P, C and PB under the full protocol, with enough matched pairs to
  resolve the 5% budgets.
- **#77's and #83's verdicts stand as recorded.**

## Design

| | ANE `predict` | role |
|---|---|---|
| **P** | production A: thread, coremltools, GIL held | reference |
| **C** | #77's GIL-released binding, unchanged | mechanistic control |
| **PB** | #83's prebound binding, unchanged | candidate |

- **Models:** laya (L128 / L512) and laya-typed-decisions (L128 / L1024). laya-multilingual keeps
  its process-isolated production path and is out of this gate.
- **The GPU:** a worker process in every configuration.
- **Workload:** #77's, unchanged: `run_mix.py` → `bench_concurrency.py --part a`. Each run has
  solo_short, solo_long, hetero and gpu_only windows over 3 cycles of 20 s.
- **Order:** rotating ABBA blocks. The paired gate is #83's (`gate.py`), applied to matched hetero
  windows.
- **Looks (Addendum 1):** futility-only looks at n = 6 and n = 12 (99% intervals). The formal gate
  runs at n = 18. There is no automatic extension.
- **The GIL probe:** #83's 1 ms probe thread is not run.

## Run

```sh
# Pre-campaign check (~15 s), then at most 28 runs x ~4.65 min (~2 h 10 min), stopping early on
# futility (Addendum 1). Idle machine on
# AC power, the local LLM server and other GPU/ANE services stopped. Resumable.
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-prebind-full-protocol/scripts/run_all.sh
uv run python research/coreml-prebind-full-protocol/scripts/analyze.py      # --check to verify
```

| file | what |
|---|---|
| `criteria.md` | question, scope, protocol, gate, power, sequential looks, outcomes, non-gating records |
| `scripts/design.py` | run order, rounds, stages; `next` prints what the sequential rule requires |
| `scripts/run_config.py` | one run: #77's full mix, the configuration's injection, #83's passive records |
| `scripts/run_all.sh` | the check, then the stages the rule requires |
| `scripts/analyze.py` | the looks, the verdicts, `results.json` and `tables.md` (`--check`) |

Unit tests: `tests/unit/test_prebind_full_protocol.py`.
