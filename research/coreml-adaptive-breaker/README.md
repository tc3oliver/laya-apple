# Adaptive PB-ASYNC execution with a slow-state breaker

This is research only. The `laya_apple` package never imports it.

The track takes up where the static handoff stopped. #102 and #103 closed that route: H64 kept the
onset clean, but #103's P r5 fell into the #96 / #99 slow state about 4 s after the handoff and
stayed there until the window ended. So a request-count guard cannot make PB-ASYNC safe.

The question here is a product question: can the runtime see from its own public signals that
PB-ASYNC has gone slow, and fall back to the production sync path (1.4's A) before users notice a
sustained regression?

- **In scope:** detecting the slow state and falling back to A.
- **Not in scope:** controlling scheduling. No QoS, affinity, private API, warm-up, priming or new
  guard length.

## Phase 0: detector replay on recorded data

[`scripts/replay.py`](scripts/replay.py) produces [`replay.md`](replay.md) and `replay.json`. It
takes no new measurement.

**Episodes replayed:**
- every hetero window of the recorded PB-ASYNC, B, O, Q, H32, H64 and P runs, 67 in all: #92, #96,
  #99, #102 (the screen and the confirmation) and #103;
- 65 A windows, replayed as a reference only.

**Signal.** `prepare_ms` is `RequestTrace.prepared_ns − submit_ns`. It covers the ANE-routed
requests from the async start. The runtime already measures it, and it is known before the request
is routed. A request is host-slow when `prepare_ms > 0.3 ms`, #94's rule.

**Ground truth: user-facing latency.** It never reads prepare.
- A sustained episode has 2 or more consecutive 1 s spans whose median e2e is > 1.2 × the study's
  A median, #103's H6 rule.
- Onset is the earliest 1 s window of that run that is already slow.

| detector | recall | delay from onset, ms (median / P95 / worst) | requests | false trips, healthy | A trips |
|---|---|---|---|---|---|
| C2 (2 consecutive) | 16/16 | 61 / 71 / 84 | 1 / 4 / 6 | 4/51 | 1/65 |
| **C3 (3 consecutive)** | 16/16 | 109 / 122 / 126 | 5 / 6 / 7 | 4/51 | 1/65 |
| R3of5 | 16/16 | 85 / 98 / 101 | 3 / 5 / 7 | 4/51 | 1/65 |
| R4of8 | 16/16 | 97 / 110 / 113 | 4 / 6 / 8 | 4/51 | 1/65 |
| EWMA (α 0.25) | 16/16 | 7 / 29 / 78 | 0 / 2 / 7 | 4/51 | 2/65 |
| MED5 | 16/16 | 97 / 101 / 102 | 4 / 5 / 7 | 4/51 | 1/65 |

**Readings.**
- **Prepare separates the states.** Healthy guarded async has prepare P99.9 of 0.238 ms, with 7
  isolated host-slow requests in 93,663. The sustained state has a median of 0.554 ms and a
  host-slow share of 0.58.
- **Every candidate catches all 16 sustained episodes**, including #103's P r5 w5. For that episode
  C3 trips 78 ms after onset.
- **Every candidate makes the same 4 false trips.** Three are onset transients in the unguarded
  cells (#92, #96, #99), which recovered by themselves.
- **The fourth is a 5-request burst in #102's H32 r3 w0** at 15.56 s. It lasted about 55 ms and
  barely moved latency.
- **Guarded episodes, the production shape:** 1 false trip in 48.
- **The same kind of burst occurs in A:** a 3-request run in #103's qualification, A r1 w5.
- **A false trip only costs the rest of that episode's async benefit.** It falls back to A.
- **#102's isolated P99 tail (confirmation H64 r8, cycle 1) trips nothing.**

**Recommendation: C3, 3 consecutive host-slow ANE requests.**
- It has one counter and one threshold.
- Its worst delay, 126 ms, is inside the 250 ms target.
- It avoids the 2-request run that C2 would trip on in A.

The data does not separate C2 from C3 on false trips.

**Limits.**
- Every replayed episode is laya at L128. No typed-decisions async data exists, so the 0.3 ms
  threshold is unverified there. The typed validation checks false trips.
- Phase 0 says nothing about whether falling back to A actually recovers. That is Phase 1's
  question.

## Phase 1: recovery experiment: PASS

The preregistration is [`phase1.md`](phase1.md), frozen at ad82202; the results are in
[`phase1_tables.md`](phase1_tables.md). The runs used #102's harness, 12 of them:
- **A** (production sync): 0 of 6 episodes slow.
- **B** (PB-ASYNC with no breaker): 6 of 6 slow. The phenotype still reproduces: median
  12.3 ms, P95 16.7 ms, about 100 req/s.
- **R** (PB-ASYNC plus the frozen C3 breaker, armed 1 s into the episode past the hetero-start
  transient):
  - 12 of 12 episodes tripped, all confirmed slow at the trip;
  - onset → trip 26 / 35 / 40 ms (median / P95 / worst);
  - trip → first A 0.78 / 1.19 / 1.30 ms;
  - trip → sustained A-like 215 / 364 / 414 ms;
  - after the fallback: 0.999 × A throughput, and a median episode P99 of 10.73 ms against A's
    10.71 ms;
  - 0 correctness, routing, request-loss or breaker failures.

## Production runtime and release validation: PASS

**Production:** the runtime is `laya_apple/handoff.py`. It runs 64 synchronous forwards per
hetero episode, then PB-ASYNC under C3, which is armed right at the handoff: the 64-forward
guard steps over the start transient. When C3 trips, the rest of the episode runs production
A; the breaker re-arms when the episode ends. It is the default for laya and
laya-typed-decisions.

**Validation** ([`validation.md`](validation.md), frozen at 5ea03ad, addendum 1;
[`val_tables.md`](val_tables.md)) reused the evidence above and ran targeted production runs
only.

| phase | P episodes | stayed async | trips | async residency | GPU return P50, P / A | throughput P / A | median episode P99, P − A |
|---|---|---|---|---|---|---|---|
| 2 laya product mix | 12 | 12 | 0 | 96.8% | 0.037 / 4.29 ms | 1.042 | −0.19 ms |
| 3 typed-decisions (product + soak) | 78 | 78 | 0 | 93.4% | 0.043 / 8.60 ms | 1.038 | −0.53 ms |
| 4 multilingual smoke | – | – | – | – | – | – | – |
| 5 product-mix soak (incl. bursts) | 64 | 64 | 0 | 91.5% | 0.035 / 4.28 ms | 1.042 | −0.18 ms |

**All phases passed**, with 0 mismatches, routing failures, request loss or crashes.
- **Phase 4:** laya-multilingual kept process placement, did not use adaptive execution, and
  `ane_handoff=True` was rejected.
- **Not exercised naturally:** no slow state occurred in any validation run, so the breaker's
  production trip path was not triggered there. Its recovery evidence is Phase 1, and its logic
  is covered by the unit tests, which match it against the research breaker.
