# H64 production evaluation: preregistration

Issue #101, draft PR #103. This file is committed and pushed before any evaluation run. Nothing
below changes once the first evaluation run has started. From then on the runtime semantics do
not change either.

**Recorded verdicts are unchanged.**
- The screen ([`criteria.md`](criteria.md)) stays RESEARCH-CLOSED.
- The H64 confirmation ([`confirmation.md`](confirmation.md)) stays FAILED on its preregistered
  outlier guard.

This is a product evaluation of the real runtime, not scheduler research. It adds no N, QoS,
priming, warm-up, probe or private API.

## Question

In a clean, real production runtime, does H64 steadily give users PB-ASYNC's GPU isolation and
throughput advantage without introducing a reproducible latency regression?

## Runtime under test

**Code:** `laya_apple/`, `pyproject.toml` and `uv.lock` at commit
**8d9e798e51abaf59a28c3ce6d2d9b27ddccfcbba** (tree `laya_apple` = d541fdc820f59fcb7b27a5f20110f828892be16f).
- `prod_run.py` records `git rev-parse HEAD:laya_apple`, the blob ids of `pyproject.toml`
  (2d2d3d684b086c0c651cd1ea96e09550ce2e5e85) and `uv.lock`
  (10f6ba84dd28791684cc102d651c1d5c38cc0178), and whether those paths are dirty.
- A run with a different tree or blob, or with those paths dirty, is INVALID.

**Cells:**
- **A:** `Laya.from_pretrained(model, device="auto", execution="workers")` with the default
  `ane_handoff=False`. This is 1.4's production path, unchanged.
- **P:** the same, with `ane_handoff=True`: the H64 staged handoff.
  - The first 64 ANE forwards of a hetero episode run the synchronous coremltools path, and
    forward 65 onward runs the prebound async Core ML path.
  - The episode ends, and the handoff re-arms, when the GPU or the short stream is idle for more
    than 1.0 s.
  - Production counts every ANE forward. In these workloads every ANE forward is a short
    request: L128 goes to the ANE and L512 / L1024 to the GPU, and routing is checked per window.

**Harness:** [`scripts/prod_run.py`](scripts/prod_run.py).
- One fresh process per run: #92's closed-loop clients, answers checked against inline GPU / ANE
  references, and 5 warm-up predicts per stream and instance.
- It records every request of the auto instance through its RequestTrace, logs P's
  `StagedHandoff.decide` calls, and takes the handoff snapshot at the end.
- No P/E counter is recorded in this evaluation. Nothing decides on one.

## Environment

- **As the qualification left it** ([`qualification.md`](qualification.md), PASS on
  `qual_tables.md`):
  - oMLX / the local LLM server stopped, AC power;
  - OrbStack and the Aerial wallpaper left as they are, so the qualification baseline is kept;
  - no other agent, build, test, benchmark or LLM workload while a run is in progress.
- **Snapshots:** `scripts/machine_snapshot.py` runs right before and right after every run.
- **A run whose snapshots fail qualification item 5** is re-run once at the end of its phase. A
  second machine failure makes the phase INVALID.

## Phases and runs (the order is fixed; a phase runs only if the previous one passed)

| phase | model, short / long | schedule (`prod_run.py`) | runs, in order | P episodes |
|---|---|---|---|---|
| 1 laya | laya, 128 / 512 | `mix`: #92's four conditions, reversed on the odd cycle, 2 cycles, 20 s windows | P A A P, A P P A, P A A P | 12 |
| 2 typed | laya-typed-decisions, 128 / 1024 | `mix` | P A A P, A P P A | 8 |
| 3 multilingual smoke | laya-multilingual, 128 / 512 | `mix` with 5 s windows, `--expect-rejected` | A (default) × 1 | – |
| 4 product mix | laya, 128 / 512 | `product` | P A A P, A P P A | 24 |
| 5 soak | laya, 128 / 512 | `soak55` | P A A P | 132 |

**The product schedule.** Idle 10 s, then: hetero, solo_short, hetero, solo_long, hetero, gpu_only,
hetero, solo_long, solo_short, hetero, hetero. Hetero windows are 20 s, single-device windows 10 s.
- That is 6 hetero windows per run.
- **Correction to `qualification.md`:** it says 7 per run and 21 in all. Its data has 6 per run, 18
  in all. No rule depended on the count.
- It covers idle → hetero, every single-device state → hetero, hetero → single → hetero, and
  hetero → hetero.

**The soak55 schedule.** 55 units per run, each "predecessor window, 1.0 s gap, hetero 8 s".
- The predecessor cycles through idle 3 s, solo_short 4 s, solo_long 4 s, gpu_only 4 s and a hetero
  8 s window: 11 of each.
- Every hetero window is an episode, including the hetero predecessors, which make the rapid
  repeated transitions. That gives 66 episodes per run, 132 per cell.
- Each window starts 0.5 s after its gap. Every pause between windows is therefore 1.5 s, longer
  than the 1.0 s re-arm gap, so every hetero window starts a new episode.

**Pairing.**
- Each P run is paired with the A run adjacent to it in its block of four: P1 ↔ A1 and P2 ↔ A2 in
  "P A A P", and the same in "A P P A".
- Episodes are paired by their index in the schedule, so both episodes of a pair have the same
  predecessor.

## Per-episode measures (t0 = hetero window start, end = window end)

- **t_h:** P's first async decision in the window.
- **Short latency:** the client latency of each short request. Per episode: median, P95, P99 and
  P99.9. Per phase and cell, the same four pooled over all hetero requests.
- **GPU return:** received − service_end of the GPU requests received in [t_h + 1 s, end). For A,
  from t0 + 1 s.
- **Aggregate req/s:** the sum of the window's streams' req/s.
- **Host-slow:** a request whose prepare (prepared − submit) is > 0.3 ms, in 0.5 s bins over
  [t_h, end). For A, over [t0 + 1 s, end).
- **Slow spans:** 1 s spans over [t_h, end) whose short median latency is > 1.2 × the phase's pooled
  A hetero short median. For A, over [t0 + 1 s, end).

## Hard failures (any one in any P run: the phase fails, and the evaluation stops at CLOSE)

- **H1. Correctness:**
  - any mismatch, in any window;
  - any routing failure, in any window: hetero short → ANE and long → GPU; solo_short → ANE;
    solo_long → GPU; gpu_only → GPU;
  - a crash (a non-zero exit);
  - a worker not alive at the end;
  - the handoff disabled at the end, or its snapshot not `consistent`.
- **H2. Handoff state corruption:**
  - in a hetero window, the decisions must follow exactly this order:
    - zero or more leading `armed` / sync decisions (count 0), from a short request that
      arrives before the long request's GPU job starts;
    - then exactly 64 guard decisions, counts 1..64 in order, all sync;
    - then t_h, and async only after it;
  - anything else fails H2, including an `armed` decision once the guard has started, 63 or 65
    guard decisions, or no t_h;
  - any decision in a solo_short window is not `armed` / sync;
  - the snapshot's episode count differs from the number of hetero windows.
- **H3. GPU isolation lost:**
  - any P episode with GPU-return P50 > 1.0 ms;
  - or a phase's pooled P GPU-return P99 > 1.0 ms.
- **H4. Sustained throughput regression:** in any run pair, P's pooled hetero aggregate is
  < 0.95 × A's. Per-episode ratios are reported.
- **H5. Persistent host-slow:** any P episode with 2 or more consecutive 0.5 s bins at a host-slow
  share ≥ 0.10.
- **H6. Multi-second catastrophic latency state:** any P episode with 2 or more consecutive slow
  spans.
- **H7. Repeated tail regression.**
  - **Tail event:** an episode whose P99 exceeds the phase's median A episode P99 by more than
    2.0 ms. This applies to episodes of both cells.
  - **Fails when:** P's tail count in a phase is > max(2, 2 × A's tail count in that phase).

## Latency non-inferiority (every phase except the multilingual smoke must pass)

- **delta_p99** = a P episode's P99 − its paired A episode's P99.
  - Pass needs both: the median delta ≤ +0.5 ms, and the one-sided 95% cluster-bootstrap upper
    bound of that median ≤ +1.0 ms.
  - Bootstrap: B = 20,000, `numpy.random.default_rng(20260926)`, percentile method. The upper
    bound is the 95th percentile of the resampled medians.
- **Clusters.**
  - Phases 1, 2 and 4: the run pairs, two per block: 6, 4 and 4 of them. Each drawn pair
    contributes all its episode deltas.
  - Phase 5: the soak has only 2 run pairs, so its clusters are consecutive 6-episode groups of a
    run pair. One group is one cycle of the five predecessor kinds plus its rapid second hetero
    window. That gives 11 groups per pair, 22 clusters.
- **Reported, not gated:** the same paired deltas for episode median, P95 and P99.9, and for req/s;
  the transition-level bootstrap bound; and pooled median / P95 / P99 / P99.9 per cell.
- **Tail events are listed individually, with a label.**
  - **"Isolated tail":** the episode's P95 and median are within +0.3 ms of its pair, its req/s is
    ≥ 0.95 × its pair's, its GPU-return P50 is ≤ 1 ms, and it has no host-slow bin and no slow
    span. These are the conditions of #102's H64 r8 cycle 1.
  - **Otherwise "tail with other symptoms".**
  - An isolated tail is a recorded tail event, not an execution failure. Only H7 counts tail
    events against P.

## Validity (INVALID only for the machine, the harness or the A side)

A phase is INVALID when any of these holds:
- a run's runtime tree differs from d541fdc8…, or those paths are dirty;
- a protocol deviation: the wrong schedule, model, lengths or cell;
- an A run crashes twice;
- A has a mismatch or routing failure;
- A has H5 or H6 signatures in more than 10% of its episodes in the phase;
- a machine snapshot fails twice, as above.

An INVALID phase stops the evaluation, and the environment is reported. Runs are not repeated until
the numbers look good.

## Multilingual smoke (phase 3)

- **Pass needs all of the following:**
  - the default configuration (A) runs with 0 mismatches, 0 routing failures and no crash;
  - `info()` shows `ane_placement == "process"` and no `ane_handoff` key;
  - `Laya.from_pretrained("laya-multilingual", device="auto", execution="workers",
    ane_handoff=True)` raises `ValueError` before loading.
- **Not done:** no placement change and no thread-mode comparison.

## Outcome

- **PRODUCT-CANDIDATE** only if all of the following hold:
  - phases 1, 2, 4 and 5 pass every hard gate, with 0 correctness failures;
  - each of those phases passes the latency non-inferiority;
  - the multilingual smoke passes;
  - P's pooled GPU-return P50 is ≤ 1.0 ms in every phase. A's is about 4.3 ms, so the GPU
    completion improvement is preserved;
  - the soak has ≥ 100 P episodes.

  The result then reads: "H64 is the leading 1.5 production candidate." The PR is not merged
  without the user.
- **CLOSE** if any P hard failure occurs (a multi-second slowdown, correctness, throughput, and so
  on), or any phase's latency non-inferiority fails.
  - H64 is closed, 1.4 stays, and no new scheduler or N variant is tried.
  - The runtime change is then withdrawn from the PR, and the evaluation is kept as a record.
- **INVALID:** only as defined above.

Analysis: [`scripts/eval_analyze.py`](scripts/eval_analyze.py), written and unit-tested before any
evaluation run.
