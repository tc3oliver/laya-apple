# Criteria: prebound predict under the full product-mix protocol (R1)

This file is written and committed before any campaign run or pre-campaign check. Nothing in it
changes after data is seen. A failure is reported as a failure, with its metric and magnitude.
Harness smoke checks made while the scripts were written are not data and are not committed.

## Question

Under #57's and #77's original, complete product-mix protocol, does PB (#83's prebound binding)
pass the production gate on both laya and laya-typed-decisions, and keep GPU completion
isolation?

## Why this replication

- **#83** recorded PB PASS on laya and laya-typed-decisions, and INCONCLUSIVE on
  laya-multilingual, under a **hetero-only** protocol (`research/coreml-prebind-predict/`).
- **Under that protocol, C also passed** on laya and laya-typed-decisions. #77 recorded C as a
  FAIL on both models under the full protocol, with solo and gpu_only windows between its hetero
  windows (`research/coreml-nogil-product-mix/`). #83's result therefore depends on the protocol
  or on what ran before each hetero window, and #83 cannot be productionized as it stands.
- **#77's and #83's verdicts stand as recorded.** This experiment does not re-score or
  reinterpret either of them.
- **The rest of the evidence this design rests on:**
  - #51: at a fixed GPU offered load, releasing the GIL cost the ANE nothing measurable
    (`research/coreml-placement-deconfounding/`).
  - #57: ANE process isolation removes the GPU completion wait but fails the gate on laya and
    laya-typed-decisions (`benchmarks/ane-process-isolation/`).
  - #58: the process regression came from host CPU, queue and dispatch slowdown, not from GPU
    pacing. GPU pacing is not revisited here.

## Scope

- **Models:** laya and laya-typed-decisions. Their production configuration today is A (a
  thread-placed ANE, coremltools, the GIL held; `laya_apple/data/placement.json`).
- **laya-multilingual is out of this gate.** Its production path (B, a process-placed ANE) is
  already isolated and is not a migration candidate. It gets regression validation later, not a
  verdict here.
- **Configurations:**

  | | ANE | Core ML `predict` | GPU (MLX) | role |
  |---|---|---|---|---|
  | **P** | thread | coremltools, GIL held (A) | worker process | production |
  | **C** | thread | #77's binding, unchanged (`coreml-nogil-product-mix/scripts/nogil.py`) | worker process | mechanistic control |
  | **PB** | thread | #83's prebound binding, unchanged (`coreml-prebind-predict/scripts/prebind.py`) | worker process | candidate |

- **The binding and the GPU topology are not changed.** Nothing in `laya_apple/` changes, and
  the package imports nothing from `research/`.

## Protocol

**Workload: #57's and #77's, unchanged.** `scripts/run_config.py` drives
`benchmarks/ane-process-isolation/run_mix.py`, which runs `scripts/bench_concurrency.py --part a`
unchanged. That is exactly how #77's `run_config.py` ran it:
- solo_short, solo_long, hetero and gpu_only windows, with the gpu_only instance. The order is
  that one in even cycles and reversed in odd cycles (bench_concurrency's own);
- 3 cycles per run, 20 s windows, 2.0 s idle and 0.5 s lead before each window;
- one closed-loop client per stream, each repeating the request built by
  `make_request(seed=0)` with one question;
- answers checked against inline coremltools and MLX references;
- the v1.0 shapes: laya L128 / L512, laya-typed-decisions L128 / L1024.

P, C and PB use the identical workload, request construction and seed. The #83 hetero-only
shortcut, with its fixed pre-window warm-up, is not used.

**Instrumentation.** It is #77's, plus #83's passive records. None of these adds work inside a
measured window beyond what #77 or #83 already did there:
- per-forward executing-thread CPU;
- two clock reads per `predict` in every configuration, as in #83;
- the request trace. The trace callback existed in #77 too, for GPU returns;
- per-thread CPU snapshots of each auto-instance window, taken by each client thread during the
  lead and after its own window end, exactly as #83 took them;
- crossings counted at load.

**#83's 1 ms GIL probe thread is not run.** It wakes every millisecond inside the measured
window, which may change the CPU's scheduling and frequency state. It is one of the differences
between #77's protocol and #83's. It stays a candidate for the protocol-history experiment, not
a variable here.

**Run order.** The order is ABBA-style blocks of six runs per model, in three rotating orders
(`scripts/design.py`):

| block | order |
|---|---|
| 1, 4, 7 | P C PB PB C P |
| 2, 5, 8 | C PB P P PB C |
| 3, 6, 9 | PB P C C P PB |

- **Balance.** Every configuration has mean position 3.5 in every block. In each stage, each
  configuration runs first once and last once.
- **Rounds.** A configuration's round r is its r-th run. Block b holds rounds 2b − 1 and 2b of
  all three configurations, so each matched pair is always within one block, at most two runs
  apart.
- **Stages.** A stage is three blocks per model: 18 runs per model, 6 per configuration, which
  gives 18 matched hetero windows per candidate. Within a stage the two models alternate block
  by block, so neither model's runs sit in one end of the session.

**Machine:**
- idle, on AC power;
- the local LLM server and other GPU/ANE services stopped, and restored after the campaign;
- the Core ML E5 cache not cleared.

**Versions:** as locked in `uv.lock` (coremltools 9.0, mlx 0.32.x). pyobjc-framework-CoreML
12.2.2 is added through `uv run --with`, for research only. The shims are compiled with
`xcrun clang` into a temporary directory at run start, as in #77 and #83.

## Before the campaign: bit-identity check

#83's `check_prebind.py` is run, unchanged, with `--models laya laya-typed-decisions`, into
`raw/check.json`.
- **Condition to start:** PB is bit-identical to coremltools on every output of every ANE bucket
  of both models.
- The check also records each output's backing mode and the crossings per forward (PB expected:
  4, with 0 re-entries).
- `run_all.sh` does not start the campaign otherwise, and that result is reported.

## Gate

The budgets are #57's engineering budgets, unchanged. The verdict rule is #83's paired
statistical gate, unchanged (`coreml-prebind-predict/scripts/gate.py`, loaded by path).

- **Sampling unit: the matched hetero window.**
  - A pair is the candidate's hetero window of round r and cycle k, together with P's hetero
    window of the same round r and cycle k.
  - Individual requests are never treated as independent samples.
  - A cycle present in only one run of a pair is an error, reported as such.
- **Per pair,** the ratio candidate / P of:
  - the short-stream P99;
  - the long-stream P99;
  - the aggregate throughput (short req/s + long req/s).
- **Statistic and interval.** The statistic is the geometric mean of the pair ratios. Its 95%
  Student-t interval is taken on the log ratios (df = n − 1) and back-transformed.
- **Rules:**

  | criterion | PASS | FAIL | otherwise |
  |---|---|---|---|
  | correctness | 0 mismatches in every window (all four conditions) of every run of P and the candidate | any mismatch | – |
  | aggregate throughput | CI lower bound ≥ 0.95 | CI upper bound < 0.95 | INCONCLUSIVE |
  | short P99 | CI upper bound ≤ 1.05 | CI lower bound > 1.05 | INCONCLUSIVE |
  | long P99 | CI upper bound ≤ 1.05 | CI lower bound > 1.05 | INCONCLUSIVE |
  | GPU completion isolation | candidate GPU return P50 ≤ 1 ms **and** P's P50 / candidate's P50 ≥ 5 | otherwise | – |

  GPU return is `received_ns − service_end_ns` of every GPU request whose reply arrives inside a
  hetero window, pooled over the look's windows. This is #57's definition, computed by #77's
  `verdict`.
- **Per model and candidate:** PASS if every criterion passes, FAIL if any fails, INCONCLUSIVE
  otherwise.
- **The deciding verdict is PB's, per model.** C is judged at the same looks, as a same-campaign
  mechanistic reference. C's verdict decides nothing.
- **Reported beside the verdict, never changing it:**
  - #83's seeded percentile bootstrap over pairs (10,000 resamples, seed 20260925);
  - the looks-adjusted interval below;
  - the pooled point estimates, labelled as such.

## Sample size, power and sequential looks

**Power, estimated before any data.**
- **The spreads.** The log SDs are of matched pair ratios in the recorded windows. #77's use the
  full protocol, the one this experiment uses. #83's use the hetero-only protocol:

  | model | metric | #77 C vs A | #77 A vs A (null) | #83 PB vs A |
  |---|---|---|---|---|
  | laya | short P99 | 0.164 | 0.100 | 0.004 |
  | laya | long P99 | 0.079 | 0.008 | 0.005 |
  | laya | aggregate | 0.015 | 0.006 | 0.002 |
  | laya-typed-decisions | short P99 | 0.246 | 0.035 | 0.011 |
  | laya-typed-decisions | long P99 | 0.054 | 0.035 | 0.001 |
  | laya-typed-decisions | aggregate | 0.079 | 0.005 | 0.002 |

- **Half-width** of the 95% interval on the log scale, t(0.975, n − 1) / √n × SD:
  - 0.497 SD at n = 18;
  - 0.338 SD at n = 36;
  - 0.273 SD at n = 54.

  For comparison, log 1.05 = 0.049.
- **Reading, short P99 at n = 18.** At the full-protocol null spread (SD 0.10), the upper bound
  passes only if the true ratio is ≤ about 1.00. At C's spread (0.164), it must be ≤ about 0.97.
  At typed's C spread (0.246), it must be ≤ about 0.93.
- **Why not fewer pairs.** The full protocol's spreads are 3–54× #83's (short P99, C vs A: 41×
  on laya, 22× on typed). No smaller n can be justified from the existing variance, so n = 18 is
  the first look.

**Sequential rule (fixed now).**
- **Looks.** A look is taken per model at the end of stage 1, 2 and 3: 18, 36 and 54 pairs, on
  all rounds up to that stage.
- **The final look for a model** is the first at which PB's verdict is PASS or FAIL. Runs made
  after a model's final look are reported, labelled as not used, and never enter its verdict.
- **INCONCLUSIVE at a look:** the model gets the next stage, 18 more runs in the same design. A
  pooled point estimate never overrides INCONCLUSIVE.
- **Cap.** After the look at 54 pairs, INCONCLUSIVE is final: "INCONCLUSIVE at the cap".
- **A started stage is completed.** A stage, once started, is completed for every model it
  includes, and stage 1 always completes for both models. This holds even after an interruption
  such as a crash.
- **No new stage after a FAIL.** If PB FAILs on either model at any look, R1's outcome is FAIL,
  and no new stage starts for the other model.
- **Multiplicity.** Up to three looks at a nominal 95% raise the chance of a false PASS or FAIL
  above 5%. The verdict uses 95% intervals, as the gate specifies. At every look the report also
  gives the Bonferroni interval over three looks (98.33%) and the verdict it would give, as
  sensitivity only.
- **Unattended.** `scripts/run_all.sh` applies this rule mechanically (`design.py next` from
  `analyze.py`).

**Machine time.**
- **One run:** about 4.8 min. #77's 18 full-protocol runs took 277–281 s each, plus uv start-up
  and the shim build.
- **The check:** about 15 s.
- **Stage 1:** 36 runs, about 2 h 53 min.
- **Each extension stage:** about 2 h 53 min for both models, or 1 h 26 min for one.
- **Maximum:** three stages for both models, 108 runs, about 8 h 38 min.

## Outcomes and what follows

Applied to PB's final verdicts:
- **PB PASS on both models:** proceed to R2, production binding feasibility (PB-direct-PyObjC
  first). A research pass is not a production change.
- **PB FAIL on either model:** R1 is a FAIL.
  - No move to a Swift or native worker, and GPU pacing is not revived.
  - A protocol-history causal experiment is preregistered next. It runs P, C and PB matched,
    with hetero preceded by each of: a fixed warm-up, solo_short, solo_long, and gpu_only. It
    identifies which preceding workload or state triggers the regression.
- **PB INCONCLUSIVE at the cap:** 1.5.0 stays blocked, and the next step is preregistered
  separately.

**C, the mechanistic control. This is a reading of the result, not a verdict.**
- **C FAIL, PB PASS:** #77's regression reproduces under the full protocol, and PB avoids it.
  That supports handoff reduction as the mechanism.
- **C PASS, PB PASS:** #77's C regression does not reproduce in this campaign. PB's pass holds
  under the full protocol. It is not attributed to handoff reduction, and the #77/#83 divergence
  stays an open question.
- **Otherwise:** reported as observed.

## Valid runs

- **Crashes.** A run that crashes or fails to start is re-run once, in the same position of the
  order.
  - `run_all.sh` keeps the failed attempt's console output in `raw/failed/`, and `analyze.py`
    lists it.
  - A second failure of the same run stops the campaign, and the stop is reported.
  - No completed run is discarded.
- **Completeness.** Every run must have 20 s windows and 3 cycles, and must record 3 hetero
  windows, one per cycle. `analyze.py` refuses any run that does not.
- **Routing.** In every hetero window, the short stream must be served only by the ANE and the
  long stream only by the GPU. `analyze.py` lists every configuration, round, cycle and stream
  where this does not hold. Such a run
  is not silently dropped.
- A PB run that raises on its buffer-reuse assertion counts as a crashed run.
- **Contamination.** If anything unrelated overlaps a run, as happened in #83's first laya block,
  the rule is fixed in an addendum before any result is read. The affected block is re-run in the
  same order, and the original runs are kept, labelled and outside the verdict.

## Not gating: recorded and reported, never part of the verdict

- **The mechanism records, per configuration, over hetero windows** (#83's definitions):
  - the GIL re-acquire wait (C and PB);
  - the ANE stage timings;
  - executing-thread CPU per forward;
  - per-thread CPU per window and the slow-CPU flag;
  - fixed-exposure-window GPU completion collisions and overlap counts;
  - crossings and backing modes at load.
- **Window history.** For each hetero window, the condition that ran immediately before it is
  recorded:
  - solo_long in even cycles;
  - gpu_only in odd cycles.

  The candidate / P short-P99 ratios are reported split by that preceding condition, as input to
  the protocol-history question. The split never gates.

## Addendum 1 (2026-09-25): fast fail, slow pass

Written during the campaign. No run file, result, table or summary had been read when it was
written, and it is merged before the first interim look is computed. The budgets (5% on the P99s,
5% on aggregate throughput, GPU return ≤ 1 ms and ≥ 5×) are unchanged, and #77's and #83's
verdicts are unchanged.

**Why.** The original design ran a fixed 36-run stage and then extended automatically, up to 108
runs and about 8 h 38 min, with no human decision. That is not justified when PB is already
clearly failing, or when the result stays unresolved. This addendum lets R1 stop early for
futility. It never lets it pass early.

**What happened before this addendum.**
- The campaign started at 22:54 with the original controller.
- At 23:17, during `laya-C-r2`, only the controller shells were stopped. The run in progress
  finished normally.
- A fixed-list runner then completed block 1 of both models. Block 1 is identical under the
  original design and this one. The runner reads no results.
- Runs completed at that point: `laya-A-r1`, `laya-C-r1`, `laya-PB-r1`, `laya-PB-r2`, with
  `laya-C-r2` in progress.

### 1. C runs only in block 1

The production go/no-go is P against PB. C is the mechanistic control, and its verdict decides
nothing.
- C runs only in each model's first block (P C PB PB C P). That gives 6 matched hetero pairs per
  model.
- Those pairs are kept as the #77/#83 protocol-dependence diagnostic. C is judged there with the
  paired gate, for reference only, and the window-history split is reported.
- No C run is scheduled after block 1. If PB fails, C is used again in the protocol-history
  experiment.
- PB's formal gate does not change because C is shortened.

### 2. The design after block 1

Per model, only P and PB run, in ABBA blocks:

| block | order | rounds |
|---|---|---|
| 1 (unchanged) | P C PB PB C P | 1–2 of P, C, PB |
| 2 | PB P P PB | 3–4 of P, PB |
| 3 | P PB PB P | 5–6 of P, PB |

**Campaign order:**
1. laya block 1, then typed block 1.
2. **Look at n = 6** for both models.
3. laya block 2, then the **look at n = 12** for laya.
4. typed block 2, then the **look at n = 12** for typed.
5. laya block 3, then typed block 3.
6. **Final look at n = 18** for both models.

A model's look is taken as soon as its block is complete. The n = 6 look waits until both models'
first blocks exist.

### 3. Early futility checks at n = 6 and n = 12 (never a PASS)

These checks compare PB with P on the rounds so far. A look can only give **FUTILITY STOP**, which
is a FAIL, or **CONTINUE**. PB stops, and R1 ends with a FAIL, if **any** of these holds:

| criterion | futility stop if |
|---|---|
| correctness | any mismatch in any window of a P or PB run used |
| GPU completion isolation | PB's pooled hetero GPU-return P50 > 1 ms, or P's P50 / PB's P50 < 5 |
| short P99 | 99% t-interval lower bound > 1.05 |
| long P99 | 99% t-interval lower bound > 1.05 |
| aggregate throughput | 99% t-interval upper bound < 0.95 |
| crashes | a second crash of the same PB run |

- The intervals use the paired gate's pairs and the t-interval on the log ratios, at 99%, which is
  more conservative than the final gate.
- Anything else means continue. A 99% interval that sits inside the budget is **not** a pass.
- **A futility stop on either model ends the campaign**, because PASS needs both models.
- **A second crash of a P run** also stops the campaign. It is reported as invalid, not as a
  futility verdict.

### 4. Formal PASS only at n = 18, and no automatic extension

At n = 18 (rounds 1–6), the original gate applies unchanged:
- 0 mismatches;
- the 95% t-interval: short P99 upper bound ≤ 1.05, long P99 upper bound ≤ 1.05, aggregate lower
  bound ≥ 0.95;
- GPU return P50 ≤ 1 ms, and ≥ 5× better than A.

**Outcomes:**
- **PASS on both laya and laya-typed-decisions:** R2.
- **Any FAIL:** R1 is a FAIL, and the protocol-history experiment is preregistered next.
- **Otherwise INCONCLUSIVE:** **the campaign stops.** Stages 2 and 3 and the looks at 36 and 54
  pairs are removed, and nothing runs without a human decision.

**What an INCONCLUSIVE result reports**, per criterion:
- the point estimate (geometric mean);
- the 95% interval;
- the log-SD of the pair ratios;
- which criterion is inconclusive;
- the pairs needed, from the stage-1 spread. That is the smallest n for which
  t(0.975, n − 1) · SD / √n is below |log(limit) − log(geometric mean)|, or "not resolvable at
  this effect size".

Any continuation is a new, targeted, preregistered replication.

The Bonferroni-over-looks interval of the original design is dropped: there is one look that can
pass. The bootstrap stays as sensitivity only.

### 5. What this addendum replaces

In the sections above, these parts are replaced:
- the run order after block 1, and the stages;
- "Sample size, power and sequential looks", except the power table;
- the extension outcomes, and the machine-time figures.

**Everything else stands:**
- the protocol and the configurations;
- the gate's definitions and budgets;
- the valid-run rules, including one re-run of a crashed run;
- the non-gating records.

**Run count and machine time,** at about 4.65 min per run as measured in this campaign:

| | runs |
|---|---|
| per model: block 1, block 2, block 3 | 6 + 4 + 4 = 14 |
| both models | 28 |
| the original stage 1 | 36 |
| the original design at most | 108 |
