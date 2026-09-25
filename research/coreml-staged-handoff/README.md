# Staged handoff: PB-ASYNC behind a request-count guard (1.5 execution closure)

**Status: run. Outcome: RESEARCH-CLOSED / NO PRODUCT CHANGE.** Neither candidate replicated.
- **Records.** The criteria, gates and outcome rules are in [`criteria.md`](criteria.md), committed
  before any formal run (2978d1a), with addenda 1–4 each committed before the runs it concerns.
  Issue #101.
- **Data.** The numbers are in [`tables.md`](tables.md) and [`results.json`](results.json); the raw
  data is in [`raw/`](raw/). No run crashed.
- **Production code is unchanged.** The production prototype is not part of this change, and
  Phases 4–7 were not reached.

## Question

Can PB-ASYNC become a production-safe GPU+ANE path that is never worse than today's production path
(A) during a hetero transition, and keeps GPU completion isolation in steady state?

**The hypothesis tested:**
- **Staged handoff.** The first N short ANE forwards of each hetero episode take A's synchronous
  path; PB-ASYNC takes over after that.
- **N:** 32 (H32) or 64 (H64).
- **What the guard counts:** real requests only. There is no sleep, no fake inference and no
  counter in the decision.
- **State machine:** [`scripts/handoff.py`](scripts/handoff.py).

## Results

**Round 1 (8 runs; A, B, H32 and H64 with 4 transitions each).**

| cell | hetero window short P99 ms | GPU return P50 ms | aggregate req/s | E-core residency after the switch |
|---|---|---|---|---|
| A (valid) | 11.00–12.72 | 4.33–4.39 | 122.1–122.4 | – (on P from 0.5 s) |
| B | 17.16–17.44 | 0.25 | 100.1–100.4 | 4 of 4, all 20 s |
| H32 | 10.56–10.74 | 0.038–0.040 | 128.2–128.5 | 0 of 4 |
| H64 | 10.69–11.10 | 0.038–0.045 | 127.6–128.4 | 0 of 4 |

- **H32 and H64 passed every gate in every transition.** After the handoff, their Core ML time
  (9.39–9.42 ms native) was below A's predict call (9.66 ms), and their GPU completion was isolated.
  H32 led.

**Round 2 (replication).**
- **H32 (H32 r3–r5, A r3–r4): INCONCLUSIVE as preregistered.** A r3's cycle-1 window had a P99 of
  13.33 ms, over the A guard's 13 ms.
  - Independently of A, H32 failed gates that do not use A:
    - **r3 cycle 1:** parent and GPU worker were E-resident from the handoff for 11.5 s (E share
      1.00). That is #96 / #99's state, with a window P99 of 17.12 ms and 112.8 req/s.
    - **r3 cycle 0:** the transient from t_h lasted 16.0 s. The counters show P throughout.
  - H32 is therefore not replicated under any reading (addendum 4).
- **H64 fallback (H64 r3–r5, A r5–r6, A valid): not replicated.**
  - Five of its six transitions passed every gate.
  - **H64 r3 cycle 0 failed the whole-window P99 gate by 0.27 µs:** 12.15150 ms against a limit of
    12.15123 ms. Every other gate of that transition passed.
  - The gates are preregistered and are applied as written.

**Across both rounds (10 transitions per candidate):**

| | H32 | H64 |
|---|---|---|
| E-core residency starting at the handoff (#96 / #99 state) | 1 of 10 | 0 of 10 |
| transitions failing any gate | 3 of 10 | 1 of 10 (by 0.27 µs, P99 only) |
| handoff time after t0 | 0.43–0.44 s | 0.76–0.77 s |

## What this does and does not show

- **PB-ASYNC's steady-state benefits are real.** When its threads run on P cores:
  - GPU return P50 is about 0.04 ms, against A's 4.3 ms;
  - the short P99 is lower than A's;
  - aggregate throughput is about 5% higher than A's;
  - answers are correct.
- **A request-count guard is not tied to what it is meant to wait for.** A's own E→P promotion at
  hetero onset took 0–0.5 s in most transitions, but 1.0 s in A r3 cycle 1 (Phase 0 and the
  mechanism table).
  - A guard that ends at a fixed count can hand over before the promotion. H32's r3 cycle-1 failure
    started exactly at the handoff.
  - H64 hands over later and had no residency in 10 transitions. That does not show it cannot
    happen.
- **No scheduler root cause is claimed.** The P/E counters are research-only.
- **Routes already ruled out:**
  - moving the ANE into a worker process: it isolates GPU completion but costs ANE latency and throughput;
  - no-GIL synchronous prediction (PB-SYNC): its host-side slowdown stays;
  - dispatcher QoS (#99);
  - process placement of the chain (#100: both processes move to E together).

## Phase 0 (existing #96 / #99 data)

[`phase0.md`](phase0.md), [`scripts/onset_phase0.py`](scripts/onset_phase0.py). Classification:
**G3** by its preregistered rule.
- A brief E-core start of under 0.5 s appears in every window type on both paths.
- A's hetero windows are on P from 0.5 s. Only PB-ASYNC's hetero windows stay on E for seconds.

## Not reached

- **Phases 4–7:** the production path, typed-decisions, the full product mix and the soak.
  - Their protocols are addendum 2 and 3, with [`scripts/prod_run.py`](scripts/prod_run.py) and
    [`scripts/prod_analyze.py`](scripts/prod_analyze.py). They are kept for reference and have no
    data.
- **The production prototype** was written and unit-tested during the campaign, but never run on
  hardware. It is not part of this change, and `laya_apple/` is unchanged.

## Run

```sh
# Idle machine on AC power, local LLM server stopped. Resumable.
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-staged-handoff/scripts/run_all.sh 1
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-staged-handoff/scripts/run_all.sh 2 H32
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-staged-handoff/scripts/run_all.sh 2 H64 fallback
uv run python research/coreml-staged-handoff/scripts/analyze.py      # --check to verify
```

Unit tests: `tests/unit/test_staged_handoff.py`.
