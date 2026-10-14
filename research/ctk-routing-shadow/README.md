# CTK routing shadow benchmark

**Question.** Can `laya-typed-decisions`, through the existing Typed Decision API, route low-risk
coding requests well enough to cut calls to Claude? Offline and read-only: the benchmark reads
decisions and compares them with labels. It never routes, gates or approves real work, and nothing
here is wired into a runtime. Verdict and numbers: [`REPORT.md`](REPORT.md).

Research code: `laya_apple` never imports it.

**Status: research only, no runtime integration, no recommendation to integrate.** Laya-first routing
did not beat rules-first routing (see the terminology below). All data is synthetic, Claude ran once
per request, and memory and energy were not measured; the full list is under
[Limits](#limits-and-unverified).

## Layout

| Path | What |
|---|---|
| `gen_data.py` | Seeded generator for `data/dev.jsonl` and `data/eval.jsonl`; writes `data/SPLIT.md` (counts, hashes) |
| `rules.py` | Deterministic keyword arm, tuned on dev only |
| `run_laya.py` | Laya arm: one `choice` question over the six labels (`device="gpu"`, offline checkpoint); writes `results/laya_BLOCKED.txt` instead if it cannot load |
| `run_claude.py` | Claude arm: `claude -p --model haiku` (CLI from `$CLAUDE_BIN`, default `claude` on PATH), `--json-schema`, no tools, no session persistence |
| `analyze.py` | Accuracy, per-class P/R, abstain rate, ECE, latency, cost, cascades |
| `run.sh` | Whole pipeline |
| `results/` | Raw per-request outputs, `metrics.json`, `metrics.md` |

## Labels

`lint-fix`, `small-bugfix`, `refactor`, `docs` (the four low-risk classes a cheap path could handle),
`needs-human` (security, production data, permissions, credentials, design decisions) and
`needs-more-info` (too vague; the correct answer is to abstain). Label policy: the primary intent of
the request decides.

## Data and split

Synthetic, generated from templates with fixed seeds (dev 1001, eval 2002); no third-party prompt or
benchmark text. **dev** (186) is the annotation set: it designed the rules, chose the Laya criteria
wording and the cascade threshold. **eval** (344) was only measured. Eval texts never equal a dev text.
Eval slices: `id` (same templates, new slot values), `confusable` (cross-class cues, mostly new
templates), `ood` (terse, non-English, odd framing; templates dev never saw), `abstain` (underspecified),
`long` (request buried in 2-4 noise blocks of logs, diffs and tracebacks). See `data/SPLIT.md`.

## Terminology: three different "accuracy" numbers

| Name | What answers each request | Eval accuracy | Claude calls |
|---|---|---|---|
| **Rules-only** | the keyword rules, never Claude | 0.735 | 0% |
| **Laya-only** | Laya, never Claude | 0.584 | 0% |
| **Claude-only** | Claude haiku on every request | 0.887 | 100% |
| **Rules-first hybrid** | rules when confident, otherwise Claude | 0.892 | 37.5% |
| **Laya-first hybrid** | Laya when confident (tau 0.4), otherwise Claude | 0.855 | 67.4% |

Rules-only and Laya-only are single classifiers. The two hybrids are cascades and score better than
their first stage because Claude answers the hard requests. Compare hybrids with hybrids and
classifiers with classifiers. The 0.5-point gap between rules-first (0.892) and Claude-only (0.887)
is under two requests out of 344 and is not a finding.

## Run

```bash
research/ctk-routing-shadow/run.sh          # SKIP_CLAUDE=1 reuses results/claude_*.jsonl
python3 research/ctk-routing-shadow/analyze.py described   # recompute metrics from results/ only
```

`analyze.py` needs neither Laya nor `claude`; it regenerates `results/metrics.json` and `metrics.md`
from the stored raw outputs, which is the cheapest way to check the numbers. A full `run.sh` needs
the `convaiinnovations/laya-typed-decisions` checkpoint in the local Hugging Face cache and a
logged-in `claude` (about US$0.4 of haiku calls for dev + eval). The Claude arm sets no `ANTHROPIC_*`
variable and does not skip permissions. `gen_data.py` is deterministic: it rewrites `data/` and
`data/SPLIT.md` byte for byte (dev sha256 prefix `5260b91c28ba4a86`, eval `054cafbf0825e2fb`).

Recorded run: 2026-10-14, Claude Code 2.1.299, model alias `haiku` (the resolved model id was not
stored per call), Laya on the MLX GPU, Apple silicon Mac, laya-apple `main` at `b23eb98`.

## Limits and unverified

- **All data is synthetic**, generated from templates by one author. The templates share that
  author's phrasing, so the rules arm reads better on in-distribution slices than it would on real
  traffic (rules score 0.15 on the OOD slice). Nothing here is real CTK traffic.
- **Claude ran once per request** (no repeats), 3 calls at a time. Run-to-run variance is unmeasured
  and its wall latency includes some client contention. Laya is deterministic.
- **Memory and energy were not measured** for any arm. Laya latency is wall time on the MLX GPU only;
  the ANE path was not built or tested.
- The cascade threshold (tau 0.4) was chosen on dev and did not hold on eval (accuracy fell by 3.2
  points against the 1-point dev tolerance). The tau sweep on eval is post hoc and decided nothing.
- Larger Claude models, and calibrating or fine-tuning Laya for this label set, were not tried.
- Test results for this branch, and what they do and do not show, are in
  [`REPORT.md`](REPORT.md#repository-test-results).
