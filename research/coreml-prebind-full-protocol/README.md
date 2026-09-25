# Prebound predict under the full product-mix protocol (R1)

**Status: preregistered, not run.** The criteria are in [`criteria.md`](criteria.md). They were
committed before any campaign data or pre-campaign check.

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
- **Stages:** 18 matched pairs per candidate at the first look. If PB is INCONCLUSIVE, the look
  is extended to 36 and then 54 pairs.
- **The GIL probe:** #83's 1 ms probe thread is not run.

## Run

```sh
# Pre-campaign check (~15 s), then stage 1 (36 runs x ~4.8 min, ~2 h 53 min); further stages
# only as criteria.md's sequential rule requires (up to 108 runs, ~8 h 38 min). Idle machine on
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
