# intra-request-split: outcome of the r1 campaign

**Recorded outcome: stopped. G1–G3 PASS. G4a INVALID. The serve mix (G4b) was not run.
No promotion to the product: V16-4 is not done.**

The campaign ran under [`criteria.json`](../criteria.json) r1 and
[`criteria-addendum-1.json`](../criteria-addendum-1.json) (preregistered in #123). One machine:
Apple M4 Max, macOS 26.6.2, laya-apple 1.5.0, local LLM server stopped, exclusive slot.

## Sequence

| Step | Result | Data |
|---|---|---|
| Addendum 1 screen (stop-only) | SCREEN CONTINUE: 8×512 split / MLX-only 0.673 (6 of 6 pairs < 1.00); split 163.4 ms, laya-fast 211.7 ms (median window P50); 0 hard mismatches | `raw/screen/`, `screen.json` |
| Block 1, look F1 (99%, 10 pairs) | CONTINUE | `raw/latency/*-b1-*`, `raw/laya-fast/*-b1-*`, `phase-a-f1.*` |
| Block 2, look F2 (95%, 20 pairs) | G1, G2, G3 PASS: continue to the mixes | `raw/*/*-b2-*`, `phase-a-f2.*` |
| Switchyard mix | stopped itself after block 1 (F3 futility rule); every round invalid on client lag | `raw/mix/switchyard.json` |
| Serve mix | not run (F3: skipped after the Switchyard stop) | – |
| `analyze.py --look final` | prints FAIL (see "Deviation") | `phase-a-final.*` |

## Gates

| Gate | Result | Recorded as |
|---|---|---|
| G1 8×512 split / laya-fast, window P50 | 0.770 [0.769, 0.770], n = 20 | PASS |
| G2 split / today's MLX path, window P50: 8×128, 8×512, 1×512, 32×64 | 0.639, 0.672, 1.001, 0.661 (every 95% upper bound ≤ 1.002), n = 20 each | PASS (no regression, and a gain on 8×512) |
| G3 correctness against the FP32 reference | 2,960 split answers (2,464 in the latency runs, 496 multi-question answers of the Switchyard split rounds): 0 hard mismatches, 0 near-tie flips, 0 errors above 0.02 | PASS |
| G4a Switchyard mix | all four rounds have an open-loop client-lag P99 of 11.8–13.5 ms, above the preregistered 5 ms validity limit | **INVALID** |
| G4b serve mix | not run | – |

8×512 medians of the window P50s: split 163.0 ms, laya-fast 211.8 ms, today's MLX path 242.5 ms.

## Deviation: the order of the futility stop and the lag validity rule

criteria.json r1 makes a mix round with a client-lag P99 above 5 ms invalid, and an invalid
round cannot be part of a verdict (`gates.validity.mix_lag`). The harness does not follow that
order:

- `mix.py` applies the F3 futility stop after block 1 without checking the lag rule first. It
  stopped the Switchyard run on two invalid pairs (ratios 5.663 and 3.196).
- `analyze.py`'s `g4` checks the recorded stop before the lag rule. So `--look final` prints
  **G4a FAIL** and an overall verdict of **FAIL** (`phase-a-final.json`).

The preregistered reading takes the validity rule first. All four rounds are invalid, so G4a is
**INVALID**, not FAIL. The campaign is recorded as stopped, not as a FAIL verdict. The code is
left as it ran. Both readings are recorded here, and the raw data supports either computation.

## Exploratory, not a gate result

The Switchyard rounds are invalid for gating (client lag above 5 ms in both configs). They are
reported descriptively only. Pairs are adjacent rounds of one block.

| Round | Config | Train P50 / P95 / P99 (ms) | Late trains (> 100 ms) | 4-question P99 (ms) | Medium / long 1-question P99 (ms) | Adaptive-execution state at round end |
|---|---|---|---:|---:|---|---|
| 1 | base | 16.9 / 55.5 / 89.3 | 9 | 1,497.5 | 1,484 / 1,541 | `safe_sync` |
| 2 | split | 152.5 / 431.8 / 505.8 | 887 | 527.7 | 1,178 / 1,178 | `async_healthy` |
| 3 | split | 144.8 / 378.8 / 448.8 | 920 | 488.0 | 1,167 / 1,187 | `async_healthy` |
| 4 | base | 18.6 / 90.5 / 140.4 | 61 | 1,506.0 | 1,527 / 1,569 | `safe_sync` |

- **Trains got much slower.** Their P99 under split was 3.2× and 5.7× base's in the two pairs,
  and late trains rose from 9 and 61 to 887 and 920 per round of about 1,420 trains. No train
  answer was wrong against the oracle, and there were 0 hard mismatches in the multi-question
  answers.
- **Multi-question requests got faster.** The 4-question P99 was about 3× better under split
  (488–528 ms against about 1,500 ms).
- **The ANE path differed by config.** Adaptive execution ended split rounds in `async_healthy`
  and base rounds in `safe_sync`. The two configs did not run the same ANE path.
- **Unverified hypothesis: head-of-line blocking on the ANE.** The split queues 1–4 ANE rows per
  multi-question request (k was 1–4) ahead of the single-question trains that `auto` also routes
  to the ANE. This was not tested here: no queue decomposition was analysed, and the path
  difference above is a second candidate cause.
- **Base itself drifted:** train P99 went from 89.3 to 140.4 ms between rounds 1 and 4.

## What this means

- The split clears its latency and correctness gates on this machine for isolated
  multi-question requests (G1–G3).
- Whether it harms single-question latency in a mixed workload was not measured validly: G4a is
  invalid and G4b did not run. The exploratory Switchyard rounds point to a large harm, but they
  are not a verdict.
- The split is not promoted. V16-4 is not done. Any follow-up needs a new preregistered
  experiment. Its harness must meet the 5 ms client-lag rule and must apply validity before any
  stop rule.
