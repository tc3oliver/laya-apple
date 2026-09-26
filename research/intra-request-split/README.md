# Split one multi-question request across the ANE and the GPU

This is research only. The `laya_apple` package never imports it, and nothing here changes the
product.

**Status: preregistered, no data.** The criteria are in [`criteria.json`](criteria.json), copied
verbatim into the preregistration issue
[#123](https://github.com/tc3oliver/laya-apple/issues/123) before any run. They do not change after data is seen.
Harness smoke checks made while the scripts were written are not data and are not committed.

## Question

A multi-question request (N questions sharing one context) runs today as one MLX batch on the GPU:
`device="auto"` never sends a multi-question request to the Neural Engine
(`laya_apple/routing.py`, reason `multiple_questions`). The ANE artifacts are fixed-shape, one row
at a time (B=1). Both devices can run at once, so one request could be split: k rows run on the
ANE one after another while the GPU batches the other N − k.

Does that split answer the request faster than today's GPU-only path, and faster than
[laya-fast](https://github.com/DJLougen/laya-fast) measured on the same machine, without changing
any decision, and without making the single-question P99 of the product mixes worse?

## What the recorded data predicts

The policy's own cost model, the shipped forward P50s in `laya_apple/data/routing.json`
(`service_ms`, from `research/phase-0-feasibility/raw/bench/latency.jsonl`), for `laya` on the
release profile:

| Workload (questions × tokens) | GPU only, predicted | ANE, per row | Policy's k | Predicted makespan |
|---|---:|---:|---:|---:|
| 8×128 | 62.1 ms | 9.86 ms | 4 | 39.4 ms |
| 8×512 | 241.6 ms | 54.5 ms | 3 | 163.5 ms |
| 1×512 | 35.2 ms | 54.5 ms | 0 (never split) | 35.2 ms |
| 32×64 | 116.3 ms (extrapolated past 8 questions) | 8.10 ms | 10 | 81.5 ms |

These are predictions from single-device measurements, not results. Two effects they leave out:
- **Contention.** `research/gpu-ane-interference/` measured the GPU's service time rising by up to
  ×1.70 while a thread-placed ANE ran Core ML's GIL-holding `predict`. laya-apple 1.5's adaptive execution
  runs the ANE on the asynchronous, GIL-free path while it is healthy; the split runs through it.
- **The mixes.** In Switchyard (`short_4q`) and in the serve benchmark (`mixed_3q`), multi-question
  requests share the ANE with single-question requests that `auto` sends there. ANE rows of a
  split request queue ahead of them.

## Candidate policy

[`scripts/split.py`](scripts/split.py), `plan_split` and `SplitSubmitter`:
- Only requests with 2 or more questions, on `Laya(device="auto", execution="workers")`.
  A 1-question request is never touched.
- A row is ANE-eligible if a loaded bucket fits it (length ≤ bucket, ≤ 32 options).
- The ANE takes the k longest eligible rows, as one job on the product's ANE worker, one row
  after another. The GPU takes the rest as one MLX batch on the product's GPU worker. Both jobs
  are queued at the same moment on the product's own FIFOs.
- k minimises the predicted makespan,
  `max(ANE backlog + Σ ANE P50(bucket), GPU backlog + GPU P50(longest GPU row, GPU rows))`, with
  the instance's own `ServiceModel` and the device backlogs read once from the queues. Ties go to
  the smaller k (today's path).
- No fallback: a device error fails the request.
- Answers come from the product's `format_answers`. The adaptive handoff's breaker is fed exactly
  as `Laya.submit` feeds it for an ANE request.

The harness opens the product instance as `Laya.from_pretrained(model, device="auto",
execution="workers")`. The only change is in this process's memory: the model's explicit ANE
buckets gain the 512 bucket the workloads need, so the ANE worker loads it. The auto tie band is
pinned to laya-apple 1.5.0's (`common.TIE_BUCKETS_1_5`), so no single-question request routes
differently.

## Arms and workloads

| Arm | What runs |
|---|---|
| `split` | the policy above |
| `gpu` | `Laya.submit`, unchanged: laya-apple 1.5's path for these requests (the GPU) |
| `ane` | every row as one ANE job on the product's ANE worker (report only) |
| `laya_fast` | laya-fast at a pinned commit, unmodified, through its public `LayaFast(...).system_one` (8×512 on `laya` only) |

Workloads: `laya_apple.workload.make_request(tokenizer, config, tokens, n_questions, seed)`, whose
longest prompt row is exactly `tokens` long, with questions cycling through its question pool:
**8×128, 8×512, 1×512, 32×64** (questions × tokens), seeds 0–3.
[`scripts/fixtures.py`](scripts/fixtures.py) writes them once, with every row's token length,
before any run; every later step reads that file. laya-fast gets the same state and questions.

## Protocol

**Latency runs** ([`scripts/latency.py`](scripts/latency.py), one process per run):
1. Open the product instance; refuse to run if the ANE is not ready or a bucket did not load.
2. Correctness pass: every fixture request once per arm, not timed.
3. Warm-up: 3 requests per workload and arm.
4. 5 cycles. In each, the workloads in a fixed order (8×128, 8×512, 1×512, 32×64), the three arms
   in an order rotated by (cycle + workload index). A window is 1 s idle, 2 discarded requests,
   then 30 measured requests, closed loop (the next is sent when the previous answer is back),
   cycling the 4 request variants.
5. Latency is caller-observed: from the call that submits the request (before tokenisation) to
   its answers in the calling thread. The window statistic is the P50 of its 30 requests.

**laya-fast runs** ([`scripts/laya_fast_driver.py`](scripts/laya_fast_driver.py)) use the same
window protocol on 8×512, with laya-fast's own interpreter in its own checkout, calling
`system_one(state, questions)`. The driver checks the pinned commit, that its Python sources are
unmodified, and that its 512 body loaded. Its outputs are not evaluated here. laya-fast picks its
split from cost constants in its own source; it runs as shipped.

**Order.** Runs alternate in ABBA blocks, one process at a time: block 1 is laya-apple, laya-fast,
laya-fast, laya-apple; block 2 is laya-fast, laya-apple, laya-apple, laya-fast. Each laya-apple
run is paired with the adjacent laya-fast run of its block, window by window on the same cycle:
10 pairs per block.

**Mix proxies** ([`scripts/mix.py`](scripts/mix.py), one process each, one product instance):
- `switchyard`: the frozen switchyard-v1 timetable and payloads (`laya-typed-decisions`, 40 req/s
  bursty, 60 s, its warm-up schedule), without the game, the GPU-only round or the result schema.
  Measured: trains (1 question, ≤ 128 tokens); late means over 100 ms; answers checked against
  the oracle platform.
- `serve`: benchmarks/serve's decision timetable and requests (`scripts/bench_serve.py`
  `client_schedule` and `build_requests`: 8 clients at 1 req/s Poisson, 80% `short_1q` at 96
  tokens, 20% `mixed_3q` at 128 tokens, seed 11), in process: no HTTP and no LLM. Measured:
  `short_1q`.
- Rounds `base` (every request through `Laya.submit`) and `split` (multi-question requests through
  the policy) replay the identical open-loop timetable, in 3 ABBA blocks (6 pairs). Latency is
  completion minus scheduled arrival. Each distinct multi-question payload is answered once, idle,
  before the first round; that answer is the correctness reference for both configs.

**Machine.** The release profile (Apple M4 Max), exclusive, on AC power. The local LLM server and
every other GPU/ANE consumer are stopped. The Core ML E5 cache is not cleared.

## Criteria

The paired gate of `research/coreml-prebind-predict/scripts/gate.py`, unchanged: per pair the
ratio candidate / reference, the geometric mean of the ratios, and a Student-t interval on the log
ratios (df = n − 1), back-transformed. A seeded bootstrap is reported as sensitivity only.

| Gate | Pairs | PASS | FAIL | Otherwise |
|---|---|---|---|---|
| **G1** 8×512 on `laya`: split / laya-fast, window P50 | 20 | 95% upper < 1.00 | 95% lower ≥ 1.00 | INCONCLUSIVE |
| **G2** split / today's GPU path, window P50, `laya`: no regression on 8×128, 8×512, 1×512 and 32×64 | 20 each | 95% upper ≤ 1.05 on every workload | 95% lower > 1.05 on any | INCONCLUSIVE |
| **G2** gain on 8×512 | 20 | 95% upper < 1.00 | 95% lower ≥ 1.00 | INCONCLUSIVE |
| **G3** correctness of every split answer (the correctness passes, every measured request, every multi-question answer of the split mix rounds) | – | 0 hard mismatches and probability and action errors ≤ 0.02, against the FP32 reference | any hard mismatch, or an error above 0.02 that today's GPU arm does not also show on the same request | an error above 0.02 that today's GPU arm shows too (reported as a product finding) |
| **G4a** Switchyard mix: split / base, train P99 per round | 6 | 95% upper ≤ 1.05, and no more late or oracle-wrong trains than base | 95% lower > 1.05, or more late or oracle-wrong trains | INCONCLUSIVE |
| **G4b** serve mix: split / base, `short_1q` P99 per round | 6 | 95% upper ≤ 1.05 | 95% lower > 1.05 | INCONCLUSIVE |

- **The FP32 reference** ([`scripts/reference.py`](scripts/reference.py)) is the PyTorch CPU FP32
  model that `laya-apple artifacts build` checks every ANE artifact against
  (`laya_apple/conversion/torch_reference.py`), run on every fixture row, one row at a time, with
  no padding. Answers go through the product's `format_answers`; the comparison is
  `scripts/bench_serve.py`'s FP16 gate (hard mismatch: the decision differs and the reference
  top-1/top-2 margin is ≥ 0.04; near-tie flips are listed, never failed).
- **Verdict:** PASS if G1, G2, G3, G4a and G4b all pass; FAIL if any fails; otherwise
  INCONCLUSIVE. Each gate is reported on its own.
- **Validity:** every `gpu`-arm answer came from the GPU with the expected reason; each run's ANE
  was ready with the needed buckets (otherwise the run aborts); laya-fast loaded its 512 body; mix
  rounds with an open-loop client lag P99 above 5 ms are invalid. A crashed run is re-run once in
  the same position; a second crash stops the campaign.

**Sequence and fast fail** (criteria.json, `sequence_and_fast_fail`):
- **F0.** Fixtures and references first; then the first latency run's correctness pass. A split
  hard mismatch, or a non-shared error above 0.02, stops the campaign with a FAIL.
- **F1, after block 1 (10 pairs).** 99% intervals, and only a stop: FAIL if G1's lower bound
  ≥ 1.00, if any G2 no-regression lower bound > 1.05, if G2's gain lower bound ≥ 1.00, or on any
  G3 failure. Never a PASS.
- **F2, after block 2 (20 pairs).** The gates at 95%. G1, G2 and G3 must all pass before the mixes
  run. INCONCLUSIVE stops the campaign: there is no automatic extension.
- **F3, the mixes.** Switchyard first, then serve. Each stops itself after its first block if both
  pairs' ratios exceed 1.20 (a FAIL). A Switchyard FAIL skips serve.

**Addendum 1: a stop-only screen before block 1**
([`criteria-addendum-1.json`](criteria-addendum-1.json), posted on #123 before any data). About
5 minutes: one laya-apple run on 8×512 with only the split and GPU arms, 6 cycles in ABBA order,
then one 3-cycle laya-fast run. The campaign stops, reported as a screen stop and not as a
verdict, if any of the 6 split / GPU window ratios is ≥ 1.00 or their geometric mean is ≥ 0.95
(S1), on any split hard mismatch (S2), or if the split's median window P50 is ≥ 1.10 × laya-fast's
(S3). Otherwise block 1 starts as above. The screen can never pass anything, and no screen data
enters a gate. The placement-probe step also waits for the screen to survive.

**Phase B** runs only after a PASS: `laya-typed-decisions` and `laya-multilingual`, 4 laya-apple
runs each (no laya-fast), G2 and G3 per model. A model without a Phase B PASS keeps today's
GPU-only path for multi-question requests in any production change.

**Promotion.** A PASS is not a product change. The split would be rebuilt with tests in
`laya_apple/scheduling.py` in its own pull request, gated again by the real Switchyard campaign
(`benchmarks/switchyard/run.sh`) and by the serve campaign beside a local LLM
(`benchmarks/serve/run.sh`, `criteria-r2.json` unchanged), each compared with the laya-apple 1.5
campaign re-run on the same machine. The mix proxies here have no HTTP layer and no LLM load.

## laya-fast, the same-machine reference

- Repository: <https://github.com/DJLougen/laya-fast> (Apache-2.0), commit
  `3a9d534642cadb54c4cdb531f9abfa694acbefc8`.
- A throwaway checkout and venv under `~/Developer/scratch/v16/laya-fast`, removed after the
  campaign. Nothing of it is committed here except the driver's timing records.
- Install as its README says, with uv in place of pip:
  `uv venv --python 3.12 .venv`, then `uv pip install --python .venv/bin/python -r
  requirements-lock.txt` and `-r requirements-export.txt`. The installed versions are recorded in
  every run file.
- Checkpoint: `convaiinnovations/laya` at its README's revision
  `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, downloaded into the checkout and converted with its
  `convert.py --dtype float16`. Its `model.safetensors` must hash to laya-apple's pinned
  `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c` (the registry records the two
  revisions' weights as byte-identical).
- ANE body: `ane/export.py --model converted-fp16 --length 512 --output ane/body512` (its
  README). Only the 512 body: every 8×512 row needs it.

## Hardware queue

Every step runs from the repository root with `LAYA_APPLE_CACHE` set to the artifact cache and
`HF_HUB_OFFLINE=1` (except the laya-fast download). Steps marked *timing* need the exclusive slot.

| # | Step | Command | Minutes | Timing |
|---|---|---|---:|---|
| 0 | the 512 ANE artifact for `laya` (V16-2) | built with laya-apple's own build and parity gate; see V16-2 | ~4 | no (heavy) |
| 1 | fixtures and FP32 references | `sh research/intra-request-split/scripts/run.sh prep` | ~3 | no (CPU) |
| 2 | laya-fast checkout, venv, checkpoint, conversion, 512 body | README commands above, in `~/Developer/scratch/v16/laya-fast` | ~20 | no (network, heavy) |
| 2a | addendum 1 screen, then its look (stop-only) | `LAYA_FAST_DIR=... sh .../run.sh screen` | ~5 | **yes** |
| 3 | block 1: 2 laya-apple runs (~7 min each), 2 laya-fast runs (~2 min each) | `LAYA_FAST_DIR=... sh .../run.sh block1` | ~18 | **yes** |
| 4 | F1 look | `sh .../run.sh look1` | <1 | no |
| 5 | block 2 | `sh .../run.sh block2` | ~18 | **yes** |
| 6 | F2 look | `sh .../run.sh look2` | <1 | no |
| 7 | Switchyard mix, 12 rounds | `sh .../run.sh switchyard` | ~16 | **yes** |
| 8 | serve mix, 12 rounds | `sh .../run.sh serve` | ~15 | **yes** |
| 9 | verdict | `sh .../run.sh final` | <1 | no |
| B | phase B, per model: 512 artifact, `prep-b MODEL`, `phase-b MODEL` (4 runs) | `sh .../run.sh prep-b laya-typed-decisions` … | ~32 per model | **yes** |

The laya-fast checkout is removed after step 5 (or after F1 if it stops the campaign).

## Threats to validity

- **One machine.** Every number applies to the release profile only.
- **Separate processes for the reference.** laya-fast runs in its own process between laya-apple
  runs; adjacent pairing and the ABBA order balance drift, but machine state is not identical
  across processes. Each run records load average and `pmset -g therm` at start and end.
- **The cost model is the shipped one.** The policy's k comes from the Phase -1 forward P50s, not
  from a re-measurement; if they are off on this machine today, the split is judged with them.
- **The mix proxies are not the product benchmarks.** No HTTP, no LLM, no Switchyard game layer.
  The promotion gate uses the real campaigns.
- **30 requests per window** make the window P50 stable but its P99 close to the maximum; P99 per
  window is reported, not gated, in the latency runs.

## Files

| File | What |
|---|---|
| `criteria.json` | the preregistered criteria (r1) |
| `scripts/split.py` | the policy and its runner on the product's workers |
| `scripts/common.py` | workloads, fixtures, the product instance, the FP16 comparison |
| `scripts/fixtures.py`, `scripts/reference.py` | the fixture requests and their FP32 reference answers |
| `scripts/latency.py`, `scripts/laya_fast_driver.py`, `scripts/mix.py` | the runs |
| `scripts/design.py`, `scripts/analyze.py` | pairing, looks and verdicts; `results/` |
| `scripts/run.sh` | the campaign, one step at a time |
| `tests/unit/test_intra_request_split.py` | the policy, pairing and gate logic on synthetic data |
