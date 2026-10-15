# Open-loop S5: is the GPU LLM slowdown caused by W8?

**Outcome: No W8-specific LLM slowdown detected at matched offered load.** (The preregistered
rule's label for this branch is "No inherent W8 interference detected"; this README uses the
narrower wording because the experiment shows matched-load behaviour on one machine, not every
load.)
At equal offered load, a GPU LLM runs 1.2% *faster* next to the W8 arm than next to FP16
(tokens/s B/A 1.012, 1.013, 1.012; TTFT P95 B/A 1.004, 0.995, 1.001).

This is an independent record. It does **not** change the verdict of
[`../ane-w8-l64-gate/`](../ane-w8-l64-gate/README.md) (PR #174: **Research Only / No Ship**;
conditions 5 and 7 failed there and still stand as recorded). Condition 7 (a real L64 request
distribution) remains unproven, so this result does not pass the gate. No `laya_apple/` code,
runtime default, golden, tolerance or release gate is touched by this record.

Protocol: [`PREREGISTRATION.md`](PREREGISTRATION.md) (committed before any data; one amendment,
also before any data, widens the client-lag validity bound, see below). This was the last
experiment of the S5 question: **the S5 study ends here and no follow-up experiment is opened.**

## Question and design

In PR #174's closed-loop S3, the LLM generated about 4% fewer tokens/s next to arm B (W8) than next
to arm A (FP16), while B completed about 24% more requests/s (A 87.0, B 107.8 req/s under LLM
load). Is the slowdown inherent to running W8 on the ANE next to a GPU LLM, or the effect of B
serving more GPU-routed requests?

Open-loop load: a dispatcher thread issues requests at seeded Poisson arrival times, so arms A and
B receive the byte-identical schedule and request sequence (checked per cycle by hash) regardless
of how fast they are served. Everything else is unchanged from the gate: model
`laya-typed-decisions` L64, unmodified inline `Laya(device="auto")` with B injected from the
research side, mix 80% L64 1q / 10% L64 3q / 10% L256 1q, Qwen3.5-0.8B-MLX-4bit (128 tokens,
greedy, `ane-w8-l64-gate/scripts/llm_load.py` unchanged, run as a subprocess).

- `eq` = 52 req/s for both arms (primary; equal offered load, so completed requests are identical).
- `hi` = 65 req/s for both arms (1.25x, dose-response, descriptive).
- `null` = arm A twice at the `eq` rate (noise floor of the A/B alternation).
- 30 s windows, 8 cycles per cell, AB/BA alternating by cycle, 3 fresh-process replicates run
  sequentially, LLM-only 60 s blocks before and after each replicate. 144 windows, 0 drops,
  0 errors, all 3 replicates valid (8/8 cycles in every cell).

Environment: macOS 27.0 (26A428), Apple M4 Max, coremltools 9.0, mlx 0.32.2, laya_apple 1.6.3,
AC power, no thermal warning before or after any run (`raw/env_*.txt`, `raw/rep*/meta*.json`).
The artifacts were rebuilt in a new isolated cache (the active cache was not written): the rebuilt
B passes the unchanged parity gate (prob error 0.01696, 0 hard mismatches, the same 2 near-tie
flips as the gate), 10594/10594 ops on the ANE with 0 transitions (`raw/build/`). The rebuilt A
artifact hash differs from the gate's (`e4a2a1e3c26a...` vs `7e8446c81b56...`; the rebuild is not
byte-reproducible); its parity, placement and probe match.

## Primary result: `eq`, B/A per replicate

| replicate | LLM tokens/s A | B | **B/A** (bound >= 0.97) | LLM TTFT P95 A / B (ms) | **B/A** (bound <= 1.05) |
|---|---|---|---|---|---|
| 1 | 337.1 | 341.3 | **1.012** | 77.0 / 77.3 | **1.004** |
| 2 | 336.8 | 341.2 | **1.013** | 76.8 / 76.4 | **0.995** |
| 3 | 337.4 | 341.5 | **1.012** | 76.8 / 76.8 | **1.001** |

Pooled mean B/A (mean of the per-replicate ratios) 1.013. All three replicates are inside both
bounds, so the rule gives its "no interference" branch ("No inherent W8 interference detected"). The "confirmed" arm of the rule
(tokens/s < 0.97 in >= 2 of 3 and pooled < 0.97) is not met in any replicate.

Noise floor, `null` A'/A (tokens/s, TTFT P95): (0.998, 1.006), (1.000, 0.998), (1.000, 1.008). All
within 0.015, so the design is not noise-limited; the +1.2% is several times the null deviation (at most 0.002 in tokens/s).
LLM alone (not part of any decision rule), first 3 generations dropped: 386.9 / 386.2, 387.4 /
385.5, 387.2 / 385.9 tokens/s (pre / post block of replicates 1, 2, 3), TTFT P95 about 64.9 ms.

Descriptive sensitivity (not preregistered, not used for the outcome): excluding cycle 0 (A's
first window has about 1 s of warm-up) gives tokens/s B/A 1.013, 1.013, 1.012; splitting by window
order gives 1.012 to 1.013 in both AB and BA cycles of every replicate. The ratio does not depend
on which arm ran first.

## Fixed request volume vs more total load (`hi`, dose-response)

`hi` raises the offered rate by 25% for both arms; A and B again replay one identical schedule.

| replicate | `hi` B/A tokens/s | `hi` B/A TTFT P95 | A(hi)/A(eq) tokens/s | B(hi)/B(eq) tokens/s |
|---|---|---|---|---|
| 1 | 1.017 | 1.000 | 0.968 | 0.972 |
| 2 | 1.015 | 0.998 | 0.969 | 0.971 |
| 3 | 1.016 | 1.002 | 0.968 | 0.972 |

Two separate effects follow from the data:

- **Interference at fixed request volume** (`eq` and `hi` B/A): none against W8. B is 1.2% to 1.7%
  faster for the LLM at the same offered load.
- **Total resource competition**: raising the offered rate by 25% costs the LLM about 3.0% to 3.2%
  of its tokens/s in *both* arms (A 0.968 to 0.969, B 0.971 to 0.972), with TTFT P95 +0.7% to +1.3%.
  This is the same size as the closed-loop S3 gap (-3.8% to -4.7%, with B completing +24%). The
  data are consistent with the S3 slowdown being the cost of B serving more GPU-routed requests,
  not a property of W8. The closed-loop gap was not directly decomposed (different methodology,
  different LLM-alone baseline, see Limits), so this is the supported reading, not a proof. The
  `eq` to `hi` comparison uses different schedules and is descriptive.

Why this does not rescue W8: in a closed loop a faster ANE arm will always attract more GPU work
per second, so the gate's condition 5 as written (S3 against arm A) can still fail for a faster
arm. This experiment explains the failure; it does not edit it.

## Load, queue, lag (pooled over valid cycles, per replicate 1 / 2 / 3 identical to within rounding)

| cell | arm | offered = completed (req/s) | drops | queue mean / P99 / max | client lag P50 / P99 / max (ms) |
|---|---|---|---|---|---|
| eq | A | 52.8 | 0 | 1.1 / 2 / 4 | 0.0 / 7.6 / 11.5 to 12.3 |
| eq | B | 52.8 | 0 | 1.0 / 2 / 4 | 0.0 / 4.8 / 5.4 to 7.8 |
| hi | A | 64.6 | 0 | 1.1 / 3 / 4 to 6 | 0.0 / 7.7 / 16.9 to 17.4 |
| hi | B | 64.6 | 0 | 1.1 / 2 / 6 | 0.0 / 4.9 / 8.9 to 9.6 |
| null | A, A' | 52.8 | 0 | 1.1 / 2 / 4 to 5 | 0.0 / 7.6 / 8.3 to 55.4 |

The client-lag P99 is 7.4 to 7.9 ms in every A window and 4.7 to 5.0 ms in every B window. The
strict original 5 ms bound is therefore met only by B and is reported next to the amended 14 ms
bound that decided validity (see Amendment 1). The lag tracks one ANE predict (A 8 ms, B 5 ms)
because Core ML predict holds the GIL; the lock serialises requests anyway, so start times are not
affected, and every latency below is measured from the scheduled arrival.

## ANE and GPU request latency, B/A (reference, no verdict)

Response time = scheduled arrival to end; service time = worker pickup to end (includes lock wait,
so service partly measures queueing). Ranges are over the 3 replicates.

| cell | device | service P50 | service P99 | response P50 | response P99 |
|---|---|---|---|---|---|
| eq | ANE | 0.670 to 0.673 | 0.736 to 0.799 | 0.446 to 0.461 | 0.738 to 0.790 |
| eq | GPU-routed | 0.977 to 0.988 | 0.789 to 0.889 | 0.835 to 0.849 | 0.792 to 0.887 |
| hi | ANE | 0.334 to 0.354 | 0.534 to 0.578 | 0.445 to 0.466 | 0.529 to 0.580 |
| hi | GPU-routed | 0.702 to 0.724 | 0.609 to 0.685 | 0.733 to 0.739 | 0.605 to 0.696 |
| null | ANE | 0.999 to 1.008 | 0.973 to 1.047 | 0.990 to 0.995 | 0.977 to 1.048 |

At equal offered load the ANE P99 is 20% to 26% lower with W8 (the gate's +5% bound is not
approached). The much larger gaps at `hi` are queueing: A runs at utilisation about 0.75 there.
Answer choices: 0 mismatches between A and B on the identical requests in all 144 windows
(12672 requests per replicate at `eq`, 15496 at `hi`; answer choice only).

## What this establishes, and what it does not

Established (for this load model, this machine and this LLM):

1. Holding the arrival schedule, request mix and LLM workload identical, replacing the FP16 L64 model
   with W8 does not slow the GPU LLM. It speeds it by about 1.2% (3 of 3 replicates).
2. A 25% higher offered rate slows the LLM by about 3% in both arms, matching the size of the
   closed-loop S3 gap. The most likely cause of that gap is the extra GPU-routed work B
   completes, not W8.

Not established, and recorded as limits:

- Condition 7 (a real L64 request distribution) is still unproven, and PR #174's verdict is
  unchanged: **Research Only / No Ship**. Passing here does not pass the gate.
- The reason B is slightly faster for the LLM was not investigated (no counters are available to
  attribute it; for example less ANE work or less memory traffic per request). It is an observation.
- 3 replicates, intervals not computed (the three ratios agree to 0.001, so a t-interval over df=2
  would be uninformative about anything but replicate scatter). The closed-loop S3 numbers and
  these numbers use different methodologies and are compared only in direction and size.
- The LLM-alone baseline here (about 387 tokens/s) differs from the gate's (485 tokens/s) although
  the interpreter, model and script are the same. The gate's LLM-alone block was run in a different
  session state; the cause was not investigated. The B/A ratios are within-experiment and do not
  depend on it, but absolute tokens/s should not be compared across the two records.
- The dispatcher lives in the same process as the workers, so client lag follows one predict
  (Amendment 1). Thread-level concurrency over one lock-serialised `Laya`, research-side injection
  of B and the single mix are unchanged limits from the gate. Energy and GPU/ANE counters are
  `unavailable`. System load average was about 2 to 3 from other sessions and OrbStack in all runs
  (`raw/env_*.txt`), equal for both arms by interleaving.
- Notes on the metrics: "service time" includes lock wait; the `null` cell's second window follows
  the same model and so understates swap/alternation noise; "pooled mean" is the mean of the
  per-replicate ratios; the first 3 generations of each LLM-only block are dropped.
- In `raw/*/meta*.json` and the per-window B files the home directory prefix was replaced by `~`
  after the runs (path metadata only; no measured value was touched, and the analysis output is
  byte-identical before and after).
- The Core ML e5rt cache grew from these builds and runs; its growth was not measured.
- oMLX production was not running and was not touched; the LLM load used model files only.

## Files

- `PREREGISTRATION.md` and Amendment 1.
- `scripts/serve_openloop.py` (harness), `scripts/analyze_openloop.py` (analysis, decision rules),
  `scripts/run_w8.py` (builds the rebuilt B artifact in the isolated root).
- `raw/screen/` (valid screen, 3 cycles, informational: eq B/A tokens/s 1.013, TTFT P95 0.999),
  `raw/rep{1,2,3}/` (per-window per-request records, `llm.jsonl`, `index.jsonl`, `meta*.json`),
  `raw/build/` (build, parity, placement and probe records), `raw/env_*.txt`.

Reproduce the tables: `python scripts/analyze_openloop.py raw/rep1 raw/rep2 raw/rep3` (needs numpy).
