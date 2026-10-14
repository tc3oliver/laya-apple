# CTK routing shadow benchmark

**Question.** Can `laya-typed-decisions`, through the existing Typed Decision API, route low-risk
coding requests well enough to cut calls to Claude? Offline and read-only: the benchmark reads
decisions and compares them with labels. It never routes, gates or approves real work, and nothing
here is wired into a runtime. Verdict and numbers: [`REPORT.md`](REPORT.md).

Research code: `laya_apple` never imports it.

## Layout

| Path | What |
|---|---|
| `gen_data.py` | Seeded generator for `data/dev.jsonl` and `data/eval.jsonl`; writes `data/SPLIT.md` (counts, hashes) |
| `rules.py` | Deterministic keyword arm, tuned on dev only |
| `run_laya.py` | Laya arm: one `choice` question over the six labels (`device="gpu"`, offline checkpoint); writes `results/laya_BLOCKED.txt` instead if it cannot load |
| `run_claude.py` | Claude arm: `~/.local/bin/claude -p --model haiku`, `--json-schema`, no tools, no session persistence |
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

## Run

```bash
research/ctk-routing-shadow/run.sh          # SKIP_CLAUDE=1 reuses results/claude_*.jsonl
```

Needs the `convaiinnovations/laya-typed-decisions` checkpoint in the local Hugging Face cache and a
logged-in `claude`. The Claude arm calls the CLI by absolute path with the ambient environment; it
sets no `ANTHROPIC_*` variable and does not skip permissions.

## Limits

Synthetic data from one author, so the templates share the author's phrasing; the rules arm in
particular reads better on in-distribution slices than it would on real traffic. Claude calls ran 3 at
a time, so its wall latency includes some client contention. Laya ran on the MLX GPU only (the ANE
artifact path was not built or tested).
