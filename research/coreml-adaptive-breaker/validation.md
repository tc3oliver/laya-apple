# 1.5 release validation — preregistration (evidence reuse + targeted production runs)

Issue #104, PR #105. This file, the runner and the analysis are committed and pushed before the
first validation run. Nothing below changes once that run has started, and the runtime semantics
do not change either.

This is release validation of the production runtime, not new research. It adds no detector,
threshold, guard, QoS, placement, priming or scheduler variant.

The components have independent evidence already, so it is reused rather than re-run. The new
runs answer one question:

> Once the validated components are wired into the real production runtime, is there an
> integration regression, and do typed-decisions and a real product mix hold?

## Evidence reused (not re-run)

| component | evidence | where |
|---|---|---|
| H64 steps over the hetero-onset transient | 10 screen + 16 confirmation episodes, clean handoffs; #103: 13 of 14 production P episodes healthy | #102 `tables.md`, `confirm_tables.md`; #103 `eval_tables.md` |
| C3 detects the slow state from `prepare_ms` | 67 replayed async episodes: 16/16 sustained caught, worst 126 ms; 1/48 guarded false trip | Phase 0, [`replay.md`](replay.md) |
| Falling back to A recovers | 12/12 tripped R episodes A-like within 164–414 ms; 0.999 × A throughput after the fallback | Phase 1, [`phase1_tables.md`](phase1_tables.md) |
| The production A baseline | #103's qualification (3 product runs) and evaluation A runs; #102's A runs | #103, #102 |
| The multilingual process placement | unchanged: laya-multilingual is never eligible, and its execution path has no change | `laya_apple/model.py` (`_handoff_eligible`) |
| The runtime itself | code review; fast suite; the Apple-silicon integration, parity and ANE suites, including the async path against the goldens | PR #105 |

## Runtime under test

**Code:** `laya_apple/`, `pyproject.toml` and `uv.lock` at commit **902813721010caf02b81535929ff858e26314a3e**.
- laya_apple tree: `5dca50e0ce5f53712f659902bb3b05b9f62baf5b`
- pyproject.toml blob: `b19061939d1618484b4f089b4ad24bbc6cbe104b`
- uv.lock blob: `10f6ba84dd28791684cc102d651c1d5c38cc0178`

A run with another tree or blob, or with the runtime, lock or harness scripts dirty, is INVALID.

**Adaptive execution** is `laya_apple/handoff.py`: per hetero episode, 64 safe sync ANE forwards,
then PB-ASYNC under the C3 breaker.
- **C3:** 3 consecutive completed async ANE requests with `prepare_ms > 0.3 ms`, armed from the
  handoff.
- **Breaker open:** production A for the rest of the episode.
- **Re-arm:** at the episode's end.
- It is the default (`ane_handoff=None`) for laya and laya-typed-decisions with
  `execution="workers"` and `device="auto"`. That is also serve's path (`serve.default_loader`).

**Cells:**
- **P:** `Laya.from_pretrained(model, device="auto", execution="workers")`, with no ane_handoff
  argument. This is the production default.
- **A:** the same with `ane_handoff=False`, 1.4's path. It is only a small same-day reference,
  not an A/B matrix.

**Runner:** [`scripts/val_run.py`](scripts/val_run.py).
- One fresh process per run.
- #92's closed-loop clients; answers checked against inline GPU and ANE references.
- Every auto RequestTrace is recorded, and every `decide()` and breaker trip of P.

**Environment:** as in #103's qualification.
- oMLX stopped, AC power.
- OrbStack and the Aerial wallpaper as they are; ordinary desktop background is allowed.
- No other agent, build, test, benchmark or LLM workload during a run.
- No stimulus: no CPU burner, no forced E-core, no QoS or scheduler trick. A slow state is never
  induced.
- A machine snapshot before and after every run. A run failing the qualification's machine item
  is re-run once as `<run>-b`; a second failure makes the phase INVALID.
- **Crash rule:** an A run is re-run once in place, and a second crash makes the phase INVALID.
  A P crash is FAIL.

## New runs (in order; a phase runs only when every earlier one passed)

| phase | model, short / long | schedule, runs | new P hetero episodes | A reference episodes |
|---|---|---|---|---|
| 2 laya combined runtime | laya, 128 / 512 | `product`: P A P | 12 | 6 |
| 3 typed-decisions | laya-typed-decisions, 128 / 1024 | `product`: P A P; then `soak55`: P | 12 + 66 = 78 | 6 |
| 4 multilingual smoke | laya-multilingual, 128 / 512 | `mix`, 5 s windows, `--expect-rejected`: P (the default call) × 1 | – | – |
| 5 targeted product mix | laya, 128 / 512 | `bursty`: P A P | 16 | 8 |
| 6 production-adaptive soak | laya, 128 / 512 | `soak55`: P (+ P r2 only on an extension trigger) | 66 (132) | – (phases 2 and 5's A) |

**Coverage:**
- **Phase 2, `product`:** healthy handoff and sustained async in 20 s windows. It also covers
  idle → hetero, solo_short → hetero, solo_long → hetero, gpu_only → hetero, hetero → two
  single-device windows → hetero, and hetero → hetero. Every hetero window is a new episode, so
  the re-arm is exercised 6 times per run.
- **Phase 3, typed-decisions.** It gets the largest budget, because it is the only checkpoint
  with no async detector history:
  - the same `product` coverage;
  - a 66-episode `soak55` run for the typed false-trip rate, with rapid repeated transitions (hetero → hetero at 1.5 s).
- **Phase 5, targeted:** the burst pattern the other phases lack. The short stream runs 0.4 s on
  and 0.2 s off against a continuous long stream; pauses below the 1.0 s gap keep each window one
  episode. Phase 2's `product` run already covers the product transitions.
- **Phase 6:**
  - 66 new laya episodes of the combined runtime over about 15 min.
  - Predecessors are idle, solo_short, solo_long, gpu_only and hetero, so every transition
    repeats 11 to 22 times.
- **Serve:** `laya-apple serve` loads models through `serve.default_loader` with the same default
  call (`execution="workers"`, `device="auto"`, no ane_handoff), plus `ane_startup="background"`.
  Adaptive execution attaches once the ANE finishes loading; in auto mode it never raises there.
  The integration suite's serve HTTP test loads a model through `default_loader` and checks the
  answers.

**Phase 6 extension (the only conditional run):** a second `soak55` P run (r2) is required when
r1 passes but shows any of these:
- a suspected false trip (an unexplained trip);
- a borderline result: throughput ratio < 1.02, or median episode P99 more than +0.3 ms over A;
- the phase is then judged on both runs.

Any anomaly in the state machine is already a FAIL.

**Workload compared with the full plan:**
- **The full plan:** 48 runs, about 183 min.
- **This plan:** 10 runs, about 65 min (+15 min if phase 6 extends).
- That is about 64% less, and 55% less if phase 6 extends.

## Definitions

Phase 1's ([`phase1.md`](phase1.md)), with each phase's A reference:
- **The A reference:** its own A runs; phase 6 uses phases 2 and 5's.
- **m_A:** the pooled median short e2e over [t0 + 1 s, end).
- **The A-like envelope:** P99.9 of rolling 25-request windows (median, P95, span), per condition:
  hetero and hetero_bursty separately.
- Plus at most 2 host-slow requests per window.

**Per P hetero episode:**
- the handoff t_h: its first async decision;
- the trip, if any: inside one request's completion;
- the first production-A request: the first `breaker_open` decision.

A trip is either:
- **confirmed slow:** the median short e2e from max(t_h, trip − 1 s) to the first A request is
  > 1.2 × m_A;
- **suspected false:** not confirmed, and no rolling window between t_h and the trip is outside
  the envelope.

**Recovery:** the trip to the first sustained A-like point (every window starting within the
next 1.0 s A-like).

**Persistence after the fallback:** 2 consecutive slow 1 s spans, or 2 consecutive full 0.5 s
host-slow bins, from the first A request + 1 s.

**Exposed slow state:** an untripped P episode with 2 or more consecutive slow 1 s spans from t_h.

**Async residency:** async ANE forwards / all ANE forwards of the P hetero windows.

**Steady span:** [t0 + 1 s, end), the same for both cells. It is past P's 64-forward guard (about
0.75 s).
- Throughput is all auto completions per second over it.
- Latency is the short e2e (RequestTrace) of requests submitted in it.

## Gates (phases 2, 3, 5, 6; any failure fails the phase and blocks the release)

**Correctness:**
- 0 mismatches and 0 routing failures, in every window;
- 0 request loss: per auto window and stream, client count = trace count;
- 0 crashes; both workers alive at the end.

**State machine (P):**
- In every hetero episode, in this order:
  - zero or more leading armed (sync) decisions;
  - exactly 64 guard decisions, counts 1..64, sync;
  - async_healthy (async);
  - optionally breaker_open (sync) to the end.
- No async decision after breaker_open (no retry).
- At most one trip, and a trip if and only if breaker_open appears.
- Every trip inside exactly one request's completion.
- The end snapshot: enabled and consistent, with episodes = hetero windows and trips = logged
  trips.
- `info()` shows `ane_handoff`: the default really used adaptive execution.

**Safety:**
- every trip recovers to sustained A-like within 1.0 s;
- no persistence after any fallback;
- no exposed slow state.

A slow state is not required to occur. If one occurs naturally, it must be handled.

**Performance:**
- **GPU isolation:** every untripped P episode has GPU-return P50 ≤ 1.0 ms over
  [t_h + 1 s, end). A is about 4.3 ms.
- **Throughput:** P ≥ 1.00 × the A reference, pooled over the phase's episodes (steady span).
- **Latency:** the median P episode P99 ≤ the median A-reference episode P99 + 0.5 ms (steady
  span).

**Product value:**
- ≥ 80% of P hetero episodes end without a trip;
- async residency ≥ 70%;
- suspected false trips ≤ 10% of P hetero episodes;
- in typed-decisions, a failure of this last rule is the "many false trips" release blocker. The
  detector is not tuned for it.

**Soak:** at least 60 new P hetero episodes analysed (66 planned; 132 with the extension).

**Reported, not gated:**
- trip rate;
- onset → trip, trip → first A and trip → recovery (median / P95 / worst);
- short e2e median / P95 / P99 / P99.9;
- GPU-return medians.

## Multilingual smoke (phase 4)

The pass needs all of the following:
- 0 mismatches, routing failures and request loss; workers alive;
- `info()` shows `ane_placement == "process"` and no `ane_handoff` key, at start and end;
- no handoff decision or trip logged;
- `ane_handoff=True` raises `ValueError`, which rejects it explicitly.

## Outcomes

- **PASS** for a phase: no gate fails, and it is valid.
- **FAIL:** any gate fails. This is a release blocker: 1.4 stays, the adaptive route closes, and
  there is no further mitigation.
- **INVALID:** only for the runtime pins, the protocol, A-side correctness or crashes, the
  machine, or the A reference itself (sustained slow in more than 10% of its episodes). The phase
  stops and the environment is reported; runs are not repeated until the numbers look good.
- **All five phases PASS**, with runtime review, integration and parity already passed: release
  v1.5.0 with adaptive execution as the default.

Analysis: [`scripts/val_analyze.py`](scripts/val_analyze.py), which writes `val_tables.md` and
`val_results.json`. It was unit-tested (`tests/unit/test_breaker_validation.py`) before the first
run. Runner: [`scripts/run_val.sh`](scripts/run_val.sh).

## Addendum 1 (2026-09-26, after phase 2 PASS, during phase 3; no gate changed)

The user asked for the remaining validation to be compressed so that no evidence is produced
twice. The pins, the definitions and every gate above are unchanged. Phases 2 and 3 run as
registered: phase 2 passed, and phase 3 completes as planned with no extra typed runs. Phase 4 runs
as registered. Only the structure of the last two phases changes.

**Phases 5 and 6 become one production product-mix soak (new phase 5).** Phase 6 and its
conditional second soak are removed.
- **P, 1 run, schedule `productsoak`:** 48 units of "predecessor, 1.0 s gap, hetero 8 s". The
  predecessors cycle through idle 3 s, solo_short 4 s, solo_long 4 s, gpu_only 4 s, hetero 8 s and
  hetero_bursty 8 s. Every hetero window is an episode: 64 new P episodes in one run (56 hetero,
  8 bursty). They cover:
  - idle, solo_short, solo_long and gpu_only → hetero;
  - hetero → single device → hetero;
  - hetero → hetero at 1.5 s;
  - bursty hetero;
  - the auto instance's real routing: short → ANE and long → GPU in hetero, the single-device
    windows, and the GPU-only instance.
- **The A reference.** Hetero episodes use phase 2's A runs, as phase 6 would have. Bursty
  episodes need a same-day bursty reference, because their pauses change the rolling-window span
  and the throughput. So there is one short `bursty` A run (8 episodes, about 2 min).
- **Gates.** All the phase-5/6 gates above apply to this one data set, with these changes:
  - the minimum is 60 new P hetero episodes;
  - throughput and latency are compared per condition (hetero with hetero A, bursty with bursty
    A), and every condition must pass;
  - there is no extension run.
- **The workload:** phases 5 + 6 were about 32 min (47 with the extension). Now they are about
  17 min: the productsoak run about 15 min, the bursty A run about 2 min.

**The release follows** when phases 2, 3, 4 and this phase 5 all pass: no Phase 7, confirmation
run, second soak or new gate.
