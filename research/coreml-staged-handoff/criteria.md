# Staged handoff: preregistered criteria

Issue #101. This file is committed and pushed before any formal run of this directory. The
commit's SHA and time are recorded on the issue and the draft PR. Nothing below changes after
formal data exists; a later phase's protocol may be added in an addendum committed before that
phase's first run, and an addendum never changes a threshold, a gate or an outcome rule already
written here.

## Question

Can PB-ASYNC become a production-safe GPU+ANE execution path that is never worse than today's
production behavior during a hetero transition, while keeping GPU completion isolation in steady
state?

## What is already known (not retested)

- The synchronous Core ML path holds the GIL and delays GPU completions (#77, #83).
- A GPU worker process isolates GPU completion (production today, cell A).
- PB-ASYNC is correct, its native Core ML time is normal, and it isolates GPU completion. Once its
  threads run on P cores, its latency and throughput are normal (#92, #96).
- At hetero onset PB-ASYNC's threads stay on E cores for 0.5–20 s, and short requests are slow
  while they do (#96). Dispatcher QoS does not prevent it (#99). The parent's sampled threads and the
  auto instance's GPU worker move to E and back together (#100).
- Phase 0 ([`phase0.md`](phase0.md), existing data only): a brief E-core start of under 0.5 s
  appears in every window type on both paths. A's hetero windows are on P from 0.5 s. Only
  PB-ASYNC's hetero windows stay on E for seconds. By its preregistered rule the classification
  is G3.

## Hypothesis: staged handoff

At the start of a hetero episode, the first N real short requests take today's synchronous path.
After that, short requests take PB-ASYNC. When the hetero overlap ends, the state re-arms. The
guard counts real requests: no sleep, no fake inference, no background thread, no P/E counter in
the decision.

The state machine is [`scripts/handoff.py`](scripts/handoff.py), fixed at this commit:

- **ARMED.** Sync. A short forward that finds the GPU active starts an episode: SYNC_GUARD, count 0.
- **SYNC_GUARD.** Sync. Each short forward adds 1 to the count. After the N-th, the state is
  ASYNC_STEADY, so forward N+1 is the first async one.
- **ASYNC_STEADY.** Async.
- **Episode end.** In any state, when the GPU is not active, or no short forward came for more than
  1.0 s, the state goes back to ARMED. That covers hetero → single device → hetero.
- **GPU active** means the auto instance's GPU worker has a job queued or running, or its last job
  ended at most 1.0 s earlier. It is observed from that worker's own submit and completion.

## Cells

| cell | ANE path per forward of the auto instance |
|---|---|
| A | always sync (coremltools), today's production path |
| B | always PB-ASYNC, the known-problematic baseline (#96 / #99) |
| H32 | staged handoff, N = 32 |
| H64 | staged handoff, N = 64 |

- **The same load in every cell.** It is #94's PB-ASYNC run (`run_config.py`): both Core ML paths
  are loaded and warmed for every bucket, and the cells differ only in the per-forward decision.
- **A therefore differs from production in one respect:** the unused PB-ASYNC models are loaded
  too. The A validity guard below checks A against #96's A.
- **Instrumentation is #94's, in every cell:** RequestTrace, native Core ML completion stamps (async
  forwards only), per-thread `PROC_PIDTHREADCOUNTS` every 100 ms (parent's listed threads and the
  auto instance's GPU worker), per-window thread CPU, GPU return. Added here: one decision record
  per ANE forward (time, path, state, count).

## Protocol

- **Setup.** laya, L128 / L512, #92's full product mix with 2 cycles of 20 s windows: solo_short,
  solo_long, hetero, gpu_only, reversed on the odd cycle. That gives two hetero transitions per run:
  cycle 0 after solo_long, cycle 1 after gpu_only.
- **Environment.** One fresh process per run, the GPU in a worker process, identical request
  generation. The machine is idle and on AC power, with the local LLM server stopped.
- **Round 1 (screen):** H32 A H64 B | B H64 A H32, one run each. That is 8 runs, and every cell gets
  4 transitions.
- **Round 2 (replication):** only for a round-1 leader L. Runs L A L A L, which gives L 6 more
  transitions and A 4 more.
  - A fallback round 2 of H64 is allowed only if H32 led round 1, failed round 2, and H64 passed
    round 1.
  - There is no other round, and no other N or variant.
- **Crash rule** (as #99): a run that exits non-zero keeps its log and is re-run once in place. A
  second failure stops the campaign.
  - A crash of an H run after the auto instance served its first request fails that candidate's
    correctness.

## Measures (per transition; t0 = the hetero window's start)

- **Short latency:** part_a's per-request client latency of the short stream.
- **t_h (handoff time).** For H cells, the decision time of the first async forward of the episode
  in this window. For B, t_h = t0. The transition is structurally valid only if all three hold:
  - exactly one episode started in the window;
  - t_h exists;
  - exactly N forwards of the episode before t_h were sync.
- **Host-slow and transient:** #94's definitions:
  - host-slow request: prepare > 0.3 ms;
  - 0.5 s bins, recovered below 10%;
  - the transient duration is the start of the first bin from which every later bin is below 10%.
  - For the steady gate they are computed on the requests submitted in [t_h, end), with t_h as the
    origin.
- **Native Core ML (async forwards):** native completion − submit_after.
- **A's reference Core ML duration:** the coremltools predict call, entry → exit. It is the only
  Core ML duration A has; its predicts carry no native stamp.
- **GPU return:** run_mix's per-GPU-request return time (the worker's end to the parent's receipt).
- **Aggregate throughput:** #92's aggregate req/s of the hetero window, the sum of the two streams'
  req/s.

**A references** (the A transitions of the rounds in use: round 1's 4 for the screen, rounds 1 + 2's
8 for the replication), each the median over those transitions:
- A_onset_p99, A_window_p99: short P99 over [t0, t0 + 4 s) and over the whole window;
- A_predict_mean: A's Core ML predict mean over the whole hetero window;
- A_agg: aggregate req/s per hetero window.

## User-facing gates (per H transition; every one must pass)

1. **Transition safety.** Short P99 over [t0, t0 + 4 s) ≤ allowed(A_onset_p99), and short P99 over
   the whole window ≤ allowed(A_window_p99).
   - allowed(x) = max(1.05 × x, x + 1.0 ms).
   - This fixes, in advance, the rule for when the ratio and the absolute rule disagree: the more
     permissive one applies.
2. **ANE steady behavior after the handoff.**
   - Native Core ML mean of the async forwards in [t_h, end) ≤ 1.05 × A_predict_ref, and ≤
     A_predict_ref + 0.3 ms. Both must hold.
   - No persistent host-slow state after the handoff: the transient from t_h lasts ≤ 1.0 s, and
     host-slow share over [t0 + 10 s, t0 + 20 s) < 0.10.
3. **GPU completion isolation.** GPU-return P50 of the GPU requests received in [t_h + 1.0 s, end)
   ≤ 1.0 ms. P95 and P99 are reported.
4. **Throughput.** The hetero window's aggregate req/s ≥ 0.95 × A_agg.
5. **Correctness.** Checked in every window of the run:
   - 0 mismatches;
   - 0 routing failures (short → ANE, long → GPU in hetero, as #99);
   - no crash;
   - the transition is structurally valid.

**A candidate passes a round** only if every one of its transitions in that round passes every gate.
One failing transition fails the candidate. Nothing is averaged across transitions.

## Validity

- **A validity (every round).** Each A window must pass #94's guard, as in #96:
  - whole-window short P99 < 13 ms;
  - transient ≤ 1 s;
  - steady host-slow < 10%;
  - aggregate req/s in [109.9, 135.0];
  - 0 mismatches and 0 routing failures.

  If any A window fails, that round is INCONCLUSIVE and no candidate is judged in it.
- **B phenotype (round 1).** At least 2 of B's 4 transitions must have a transient ≥ 2.0 s (#94's
  measure from t0). Otherwise the problem did not reproduce: round 1 is INCONCLUSIVE, because an H
  pass could then not be read as avoiding it.
- **Not reasons for INCONCLUSIVE:** a candidate failing a gate, and a candidate crashing, are
  candidate failures.

## Mechanism evidence (research only; never a gate)

Per H transition, from the counters:
- the E share of the parent's active threads and of the auto GPU worker over [t_h, t_h + 0.5 s),
  [t_h + 0.5, t_h + 4 s) and [t_h + 4 s, end);
- the E→P switch time, per #100;
- the relative effective cycle rate on P and E.

It is read as "P-dominant at the handoff and stays" when the E share is ≤ 0.25 in all three spans.
If the gates pass and this reading does not hold, the result says so. No scheduler root cause is
claimed either way.

## Selection and outcomes

- **After round 1.**
  - If A is invalid, or B does not reproduce the phenotype: INCONCLUSIVE.
  - Leader: H32 if it passes; otherwise H64 if it passes.
  - If neither passes: **STOP the staged-handoff route → RESEARCH-CLOSED.** No H96, H128,
    wall-clock or other variant is run.
- **After round 2.** The leader passes only if all 6 new transitions pass, with A valid.
  - Any failing transition fails it; a long #99-type regression in one transition is enough.
  - If H32 fails, H64 is replicated once, but only if it passed round 1.
  - If no candidate replicates: RESEARCH-CLOSED.
- **After replication:** Phase 4, a production prototype in `laya_apple/`. The later phases follow
  if each earlier one passes, with their protocols in an addendum committed before their first run:
  - Phase 5: laya-typed-decisions, A vs the leader, the same gates;
  - Phase 6: the full product-mix transitions;
  - Phase 7: a soak of dozens of hetero transitions.
- **The three outcomes:**
  - **PRODUCT-CANDIDATE:** laya, typed, product mix, soak and correctness all pass.
  - **RESEARCH-CLOSED / NO PRODUCT CHANGE:** a candidate fails, and production is unchanged.
  - **INCONCLUSIVE:** only for a validity, machine or harness failure. A candidate's failure is
    never written as INCONCLUSIVE.
- `laya-multilingual` keeps its process placement. It gets only a regression smoke if shared code
  changes.

## Addendum 1 (committed after round 1, before any round-2 run; protocol only)

The review of `analyze.py` noted that the H64 fallback round 2 would reuse the A runs of H32's
round 2 as its own. If the fallback runs, it gets its own interleaved A runs: H64 r3, A r5, H64 r4,
A r6, H64 r5. Its A references and its A validity use A r1, r2, r5 and r6. No threshold, gate or
outcome rule changes. H32's round 2 is as written above: H32 r3, A r3, H32 r4, A r4, H32 r5.
