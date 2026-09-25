# Criteria: what triggers the host-side slow state (trigger screen)

This file is written and committed before any campaign run or pre-campaign check. Nothing in it
changes after data is seen. Harness smoke checks made while the scripts were written are not
data and are not committed. Issue #89.

## Purpose and question

**The 1.5 objective** is an execution model that can go into production and that combines:
- GPU completion isolation;
- no host-side slow state;
- no ANE latency or throughput regression.

**Where R1 left it.** R1 (#86, #87, #88) showed that PB alone is not that model. It stopped for
futility on laya, and C, the #77 binding, failed on both models.

**This experiment is a trigger screen, not a production gate.** Its one question:

> What condition causes the no-GIL ANE thread to enter the host-side slow state under
> heterogeneous GPU+ANE serving?

## What R1 observed (the phenomenon, not gating there)

- **Two states.** The short P99 of each C or PB hetero window fell into one of two clusters:
  normal, at 10.4–12.3 ms (A's level), or slow, at 13.6–21.2 ms.
- **Host CPU in the slow state.** Host CPU work was inflated across the process:
  - the ANE features stage;
  - the GPU thread's CPU per forward;
  - the client threads' CPU.

  The native Core ML `predict` did not change.
- **Where it never appeared:** in A's hetero windows and in every configuration's solo_short
  windows.
- **How often each configuration was slow:**

  | model | configuration | hetero windows slow |
  |---|---|---|
  | laya | PB | 11 of 12 |
  | laya | C | 5 of 6 |
  | laya-typed-decisions | PB | 0 of 6 |
  | laya-typed-decisions | C | 5 of 6 |

- **#83 is not a single-variable contrast with R1.** Under #83's protocol, C and PB stayed
  normal. That protocol differed from R1's in three ways at once:
  1. hetero-only windows, with no solo_short, solo_long or gpu_only windows and no gpu_only
     instance;
  2. a fixed 2 s hetero warm-up before each hetero window;
  3. a 1 ms GIL probe thread.

  This screen separates two of them, (2) and (3), and adds the most direct causal control: the
  same prebound path with the GIL held.

## Cells

The model is laya only, with L128 short and L512 long requests. laya-typed-decisions and
laya-multilingual are not run.

| cell | ANE `predict` | GIL during `predict` | added to R1's protocol | role |
|---|---|---|---|---|
| **A** | coremltools (production) | held | – | negative control |
| **PB-R** | R1's PB (#83's `prebind.py`, unchanged) | released | – | positive control |
| **PB-H** | PB-R's prebound path, identical | **held** | – | causal control for GIL release |
| **PB-W** | PB-R | released | #83's fixed 2 s hetero warm-up before each hetero window | conditioning |
| **PB-P** | PB-R | released | #83's 1 ms GIL probe thread, for the whole run | wake / residency |

**How PB-H is built.** PB-H is PB-R with one difference, how the one native call is bound:
- **The same as PB-R:** `prebind.install`, the prebound input buffers, the feature provider, the
  `outputBackings`, the options, the compiled shim and its clock stamps.
- **The one difference:** PB-R calls the shim's function through `ctypes.CDLL`, which releases
  the GIL for the call. PB-H calls the same native function, at the same address, through
  `ctypes.PYFUNCTYPE`, which keeps the GIL.

**PB-W and PB-P are run separately.** Neither has the other's addition.

**Not revisited here,** because the earlier evidence was negative:
- thread QoS;
- the Python switch interval;
- spin polling;
- GPU pacing (#58).

## Protocol

- **Workload: R1's, unchanged.** It is #77's full #57 mix: `run_mix.py` running
  `bench_concurrency.py --part a`.
  - The windows are solo_short, solo_long, hetero and gpu_only, in bench_concurrency's
    alternating order, over 3 cycles of 20 s. That gives **3 hetero windows per run**.
  - Before every window there is 2.0 s idle and a 0.5 s lead.
  - There is one closed-loop client per stream. Each stream repeats `make_request(seed=0)` with
    one question, so every cell sends identical requests.
  - Answers are checked against inline references.
- **Fixed, identical conditioning.** Every cell runs the same window sequence before each hetero
  window. PB-W alone adds its warm-up: 2.0 s idle, then 2.0 s of both streams, closed-loop and
  not measured, then the 0.5 s lead.
- **One fresh process per run**, as in R1.
- **Records: R1's, identical in every cell:**
  - executing-thread CPU per forward;
  - the `predict` stamps;
  - crossings at load;
  - backings;
  - the request trace;
  - per-thread CPU snapshots.

  PB-P also records the probe's lateness.
- **Machine:**
  - idle, on AC power;
  - the local LLM server and other GPU/ANE services stopped, and restored afterwards;
  - the Core ML E5 cache not cleared.
- **Versions:** as locked in `uv.lock`. pyobjc-framework-CoreML 12.2.2 is added through
  `uv run --with`, for research only.

## Before the campaign (`scripts/check_cells.py`, `raw/check.json`)

The campaign starts only if all four of these hold:
1. **PB-R bit-identical:** its outputs match coremltools bit for bit on every laya ANE bucket,
   and so does `ANEBackend.forward`.
2. **PB-H bit-identical,** on the same terms.
3. **PB-R releases the GIL.** While a background thread runs 50 predicts on the L128 bucket, a
   main-thread 1 ms probe shows a P50 lateness below 1 ms.
4. **PB-H holds the GIL.** In the same test, the probe's P50 lateness is at least half the median
   native `predict` time.

## Classification (state, not acceptance)

- **A window's state.** A hetero window is **slow** if its short-stream P99 is ≥ 13.0 ms, and
  **normal** otherwise. The 13 ms split is R1's observed gap between the two clusters. It
  classifies state only. It is not a production threshold.
- **A run's state:**
  - **slow:** at least 2 of its 3 hetero windows are slow;
  - **normal:** 0 of 3 are slow.

## Run order and stop rules

**Round 1:** one run per cell, in the fixed order A, PB-R, PB-H, PB-W, PB-P. Results are read for
the first time after all five runs.

**Round 2 applies only if PB-R's round-1 run is slow and A's has no slow window.**
- Each of PB-H, PB-W and PB-P whose round-1 run is normal (3 of 3 windows normal) is a
  candidate.
- Round 2 runs A, PB-R and the candidates, in the fixed order.
- No other cell runs a second time.

**A candidate is confirmed** when all three hold over rounds 1 and 2:
- the candidate has 0 of 6 windows slow;
- PB-R has at least 4 of 6 slow;
- A has 0 of 6 slow.

**No third round runs automatically.** Anything beyond round 2 needs a new, human decision.

**Positive control not reproduced** (PB-R's round-1 run has at most 1 slow window):
- One more PB-R run is made, as its round 2.
- The screen then stops. It draws no causal conclusion in either case.
- The PB-R re-run's state is reported.

**Negative control slow** (any A window slow): the screen stops, with no causal conclusion.

**Time.**
- One run takes about 4.7 min, as measured in R1. PB-W adds 6 s.
- The pre-campaign check takes under a minute.
- Round 1: 5 runs, about 24 min.
- Round 2: at most 5 runs, about 24 min.
- Maximum: 10 runs, about 50 min.

## Preregistered interpretation

These readings apply to confirmed candidates. Each is a statement about a trigger, not a fix.
1. **PB-R slow, PB-H normal:** GIL release is one necessary condition for the transition into the
   slow state.
   - **Next:** find out who takes CPU or the GIL during the GIL-released period and causes the
     process-wide slowdown.
2. **PB-R slow, PB-W normal:** a hetero-transition or scheduler-residency state is key.
   - **Next:** find the minimal conditioning or state-retention mechanism.
3. **PB-R slow, PB-P normal:** #83's probe changes scheduler or CPU residency.
   - **Next:** find the minimal wake mechanism and measure its overhead.
   - The probe is not taken as a production fix.
4. **PB-H, PB-W and PB-P all slow** (each run slow): stop tuning the Python thread path.
   - **Next:** native / Swift worker architecture research.
5. **PB-R does not reproduce the slow state:** no causal conclusion from this round. The positive
   control is re-run (above).

**Other outcomes:**
- **Any other mix** is reported as partial, for example a candidate with 1 of 3 windows slow, or
  one that is normal in round 1 but not confirmed. It is not interpreted, and the next step is a
  human decision.
- **More than one confirmed candidate:** each reading is reported, and none is ranked.

## Outputs (all non-gating)

Per cell, per run and per hetero window:
- the slow flag, and slow windows out of 3;
- short P99;
- aggregate req/s;
- GPU return P50;
- the ANE stages: features, pre, native, re-acquire, post, tail;
- ANE and GPU thread CPU per forward;
- client-short and client-long CPU;
- for PB-P, the probe's lateness;
- for PB-W, the warm-up's request counts.

This screen does not change production code or `laya_apple/data/placement.json`, and it prepares
no release.
