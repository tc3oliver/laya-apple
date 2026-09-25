# What triggers the host-side slow state (trigger screen)

**Status: paused before any campaign run (2026-09-26).**
- **Why.** A higher-priority architectural alternative was identified before any campaign run and
  before any result was inspected: Core ML's official asynchronous prediction API (#90,
  `research/coreml-async-predict/`).
- **What exists.** The harness is written and statically validated. The pre-campaign check passed
  in a smoke location. No campaign data exists.
- **When it resumes.** This screen resumes if async prediction does not remove the host-side slow
  state, or cannot be implemented safely.
- **The criteria** are in [`criteria.md`](criteria.md), and they are unchanged. Issue #89.

## Question

What condition causes the no-GIL ANE thread to enter the host-side slow state under
heterogeneous GPU+ANE serving?

## Why

R1 (`research/coreml-prebind-full-protocol/`) stopped for futility.
- **Two states.** No-GIL hetero windows fall into one of two states: normal, with short P99 at
  production's level, or slow, at 13.6–21 ms with host CPU work inflated across the process.
- **Production** never entered the slow state.
- **This screen** looks for the trigger. It is not a production gate.

## Cells

laya only, L128 / L512, on R1's full protocol. One run is 3 hetero windows.

| cell | what differs from R1's PB | role |
|---|---|---|
| A | production coremltools, GIL held | negative control |
| PB-R | nothing | positive control |
| PB-H | the same native `predict` call, bound so the GIL is held | causal control for GIL release |
| PB-W | #83's fixed 2 s hetero warm-up before each hetero window | conditioning |
| PB-P | #83's 1 ms GIL probe thread | wake / residency |

- **Classifier.** A window is slow if its short P99 is ≥ 13 ms. That classifies state only; it is
  not a threshold.
- **Round 1** runs every cell once.
- **Round 2** runs only A, PB-R and any cell that stayed fully normal. It runs only if PB-R was
  slow and A was not.
- **At most** 10 runs, about 50 min.

## Run

```sh
# Pre-campaign check (bit identity, GIL release / hold), then round 1 (5 runs, ~24 min) and, only
# as criteria.md requires, round 2. Idle machine on AC power, local LLM server stopped. Resumable.
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-slow-state-trigger/scripts/run_all.sh
uv run python research/coreml-slow-state-trigger/scripts/analyze.py      # --check to verify
```

| file | what |
|---|---|
| `criteria.md` | question, cells, protocol, pre-campaign check, classifier, stop rules, interpretation |
| `scripts/run_config.py` | one run of one cell: R1's harness plus the cell's single difference |
| `scripts/check_cells.py` | bit identity of PB-R and PB-H; GIL release (PB-R) and hold (PB-H) |
| `scripts/design.py` | the rounds and the next step |
| `scripts/run_all.sh` | the check, then the rounds the rules require |
| `scripts/analyze.py` | states, outcomes, `results.json` and `tables.md` (`--check`) |

Unit tests: `tests/unit/test_slow_state_trigger.py`.
