# Laya as a first-pass router: shadow benchmark report

Status: offline shadow measurement, 2026-10-14. Nothing deployed; Laya controls nothing.
Branch `research/ctk-routing-shadow`. Raw data and method: [`README.md`](README.md), `results/`.

## Verdict: No Change (research only)

On this workload Laya-first routing did not beat rules-first routing, and Laya adds an inference
layer without a safe reduction in Claude calls. The rules-first hybrid needed fewer Claude calls
(37.5% against 67.4%) with no measurable accuracy loss against Claude-only (0.892 against 0.887, a
gap of under two requests in 344). The evidence does not support integrating Laya into CTK routing,
and this study recommends against it. Nothing here is integrated at runtime.

**Read the accuracy numbers by name.** *Rules-only* (0.735) and *Laya-only* (0.584) are single
classifiers that never call Claude. *Rules-first hybrid* (0.892) and *Laya-first hybrid* (0.855) are
cascades where Claude answers every request the first stage is unsure about. A hybrid beats its first
stage because Claude does the hard part; the two kinds of number must not be compared with each other.
The terminology table in [`README.md`](README.md#terminology-three-different-accuracy-numbers) gives
all five with their Claude-call share.

## Setup

- Task: classify a coding request into 6 labels (4 low-risk classes, `needs-human`, `needs-more-info`).
- Eval: 344 synthetic requests (id 120, confusable 80, ood 60, abstain 34, long 50), kept apart from
  the 186-request dev set that tuned rules, Laya criteria wording and the cascade threshold.
- Arms: rules; Laya `laya-typed-decisions` via `Laya.predict`, one `choice` question, MLX GPU;
  Claude `haiku` via `claude -p` with JSON schema. No Claude call failed (344/344 eval, 186/186 dev).

## Results (eval)

| arm | accuracy | abstain rate | ECE | needs-human sent to a low-risk class | p50 / p95 / p99 ms | cost |
|---|---|---|---|---|---|---|
| rules | 0.735 | 0.326 | 0.221 (rule margin, not a probability) | 0.000 | ~0 | none |
| Laya | 0.584 | 0.064 | 0.205 | **0.594** | 12.3 / 26.4 / 28.1 | local; memory/energy unavailable |
| Claude haiku | **0.887** | 0.192 | 0.077 (verbalized) | 0.062 | 3464 / 4772 / 5660 | US$0.183 (0.00053 per call) |

Accuracy by slice (rules / Laya / Claude): id 1.00 / 0.80 / 0.98; confusable 0.59 / 0.35 / 0.74;
ood 0.15 / 0.53 / 0.95; abstain 0.94 / 0.38 / 0.88; long 0.90 / 0.64 / 0.84. Per-class precision and
recall are in `results/metrics.md`. On dev: rules 0.952, Laya 0.651, Claude 0.978.

Laya's two weak spots matter for routing. It rarely abstains (6.4%; recall on `needs-more-info`
0.36), and it sent 59% of true `needs-human` requests to a low-risk class (recall 0.34), the unsafe
direction. Its confidence is poorly calibrated (ECE 0.205) and the criteria wording moved dev
accuracy from 0.54 (bare label names) to 0.65 (described labels), so it is prompt-fragile. This
checkpoint is a generic typed-decision model, not tuned for this label set.

## Cascade: does Laya reduce Claude calls?

Laya decides first; Claude is called when Laya's `answer_confidence` is below tau or Laya picks
`needs-more-info`. Tau was chosen on dev as the largest call reduction within 1 point of
Claude-only accuracy (tau = 0.4).

| strategy | Claude calls | accuracy | mean latency ms | p95 ms | Claude cost |
|---|---|---|---|---|---|
| Claude only | 100% | 0.887 | 7193* | 9543* | US$0.183 |
| rules first, Claude on unsure | **37.5%** | **0.892** | 1331 | 4150 | US$0.068 |
| Laya first, Claude on low confidence | 67.4% | 0.855 | 2446 | 4643 | US$0.124 |

\* Includes queueing from running 3 Claude calls at once; single-call wall latency is about 3.5 s.

- Laya at the dev-chosen tau cuts 33% of Claude calls but loses 3.2 points of accuracy on eval, so
  the dev threshold does not hold up out of sample (the dev tolerance was 1 point).
- An ex-post sweep on eval (not used for any decision) finds no tau that saves calls without
  accuracy loss: at tau >= 0.7 every request escalates (100% of calls, accuracy 0.887), so Laya is
  then pure overhead of about 12 ms per request. Raising tau to 0.6 gives 94.8% of calls and
  0.872 accuracy.
- The rules-first hybrid beats the Laya-first hybrid on both axes here: fewer calls (37.5% vs 67.4%)
  and higher accuracy (0.892 vs 0.855). Caveat: the rules were written by the same author as the
  templates; on the OOD slice they score 0.15 as a standalone classifier, so their advantage will
  shrink on real traffic.
- Answer to the question: Laya reduces Claude calls only by accepting errors, and the errors include
  unsafe `needs-human` misroutes. Otherwise it is an extra layer.
- Money is not the issue either: Claude haiku costs about US$0.0005 per classification here, so even
  a free perfect first pass would save about US$0.18 per 344 requests. The potential gain is latency
  (3.5 s per call), and Laya does not capture it safely.

## Not measured or unverified

- Laya resource use (memory, energy) and the ANE path: not measured (GPU run only; no artifact built).
- Real CTK traffic: all data is synthetic, from one author's templates; results may not transfer.
- Claude run-to-run variance: one run per request, no repeats; Laya is deterministic.
- Larger Claude models, and fine-tuning or calibrating Laya for this label set, were not tried.
- The Claude model id behind the `haiku` alias was not stored per call.

## Repository test results

This branch adds files only under `research/ctk-routing-shadow/`. `git diff main --name-only`
outside that directory is empty, so the production source is identical to `main` at `b23eb98`
(no production code difference). `pyproject.toml` already excludes `research/` from ruff.

Full suite on the branch, `uv run pytest -q` (2026-10-14, Apple silicon Mac, default extras):

| passed | failed | errors | skipped | duration |
|---|---|---|---|---|
| 1686 | 22 | 9 | 47 | 12 min 53 s |

All 31 failures and errors are in `tests/integration/` (`test_routing_e2e` 9, `test_no_silent_fallback` 8,
`test_workers` 6, `test_profiles` 4, `test_export_import` 2, `test_prebuilt_fetch` 1,
`test_switchyard` 1). CI does not run these tests.

What was and was not checked:

- The full-suite run above was made on the branch only. It was **not** repeated on `main`.
- Separately, after that run, the same 31 test ids were run on a detached worktree of `main`
  (`b23eb98`, imported from that worktree, same Python environment): the same 31 fail or error,
  none extra, none missing (22 failed, 9 errors). This covers only those 31 ids, in isolation, not the
  other 1686 tests, and it is a single run.
- The visible cause is environmental: in 18 of the 31 the message is
  `BackendUnavailableError: device='ane' needs coremltools: install the [ane] extra`, i.e. the
  environment did not have the `ane` extra, and the other messages were not triaged one by one. So
  this does not prove all 31 are pre-existing defects of `main`; it shows the branch does not change
  them.
- The CI-equivalent fast suite is reported on the pull request, not here.

## Follow-up

None proposed. The evidence does not justify any integration, so there is nothing to propose (and
nothing deployed). Revisit only if a real workload shows high-volume, low-risk requests where a
tuned or calibrated Laya clears both an accuracy and a needs-human-safety gate.
