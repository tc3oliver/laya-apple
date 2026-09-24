# EXP-001 quick probe: semantic tool-output admission

**Status:** pre-registered. Not run yet.

This is a small feasibility probe, not the full EXP-001. It answers two questions with
the least work:
1. **Upside.** How much ordinary tool output could an ideal semantic gate keep out of
   the Main LLM's context in real coding work? (Stage A)
2. **Signal.** Does an existing Laya checkpoint, zero-shot, show a usable KEEP/DROP
   signal on that workload? (Stage B)

Out of scope: a dataset, a public-data importer, fine-tuning, any runtime integration
(Pi, Hermes Agent, OpenClaw), an installer, and any change to `laya_apple`. Scope follows
[EXP-000B](EXP-000B-ordinary-tool-results.md): ordinary tool results only.

## Data

**Source.** Local Claude Code sessions for this repository only
(`~/.claude/projects/-Users-oliver-Developer-src-personal-laya-apple/`). Sessions of
other repositories are not read. There are no Pi sessions for this repository, so none
are used.

**Selection.** `Bash` tool calls whose command is one of:
- tests: `pytest`;
- search: `rg`, `grep`;
- build or lint: `ruff`, `uv build`, `twine check`, `mypy`, a compiler;
- git state: `git log`, `git status`.

Excluded: file reads (`cat`, `sed -n`, `head`, `tail`, the `Read` tool), `git diff`,
`git show`, patches, user-provided content, and any output touching credentials or
`.env`.

A call qualifies when its output is at least 400 tokens. The most recent qualifying
calls are taken, newest first, up to 20 in total and at most 6 per tool type. If fewer
than 10 qualify, the probe runs on what exists and says so.

**Privacy.** Raw sessions and raw tool output stay in a local scratch directory and are
never committed. Before any processing, API keys, tokens, credentials, e-mail addresses,
private hostnames and absolute home paths are redacted. The repository gets only
sanitized per-chunk records (id, tool type, chunk kind, token count, labels, model
outputs) and aggregate results. No chunk text is committed.

**Tokens.** Counted with `tiktoken` `o200k_base`, as a stand-in for the Main LLM's
tokenizer.

## Chunking

Minimal logical chunks, split by simple rules:
- **pytest:** session header; each failure block; each traceback; warnings summary;
  runs of passing/progress lines; the final summary line.
- **rg / grep:** one chunk per file group (consecutive lines from the same file).
- **ruff / build / compiler:** one chunk per diagnostic block; the summary.
- **git log / status:** one chunk per commit, or per status section.

Very small pieces are merged into their neighbour. The rules are not tuned after
labels are seen.

## Stage A: value ceiling (no Laya)

Each chunk gets an oracle label:
- **REQUIRED:** the agent's later actions or answers depend on it (a failing test, a
  match it then edited or read, an error it then fixed);
- **USEFUL:** it informs the agent, but the work could continue without it;
- **IRRELEVANT:** nothing later depends on it.

The evidence is the session after the call: the next assistant text, the next tool calls,
the files read or edited, and the task outcome. The labeler is the strong model running
this probe, and it labels **conservatively**: any doubt between two labels takes the
more-kept one. The labels are frozen, and their digest is recorded, before Stage B runs.

Mapping: REQUIRED and USEFUL → KEEP; IRRELEVANT → DROP.

**Oracle reduction** = IRRELEVANT tokens / all chunk tokens. It is reported overall and
per tool type.

**Stage A verdict (fixed now):**

| Oracle reduction | Verdict |
|---|---|
| < 15% | **NO-GO.** The quick probe stops, and Laya is not tested |
| 15%–30% | **MARGINAL.** Not a success. Stage B does not run as part of this probe |
| ≥ 30% | **Clear upside.** Stage B runs |

## Stage B: Laya capability probe (only if Stage A ≥ 30%)

**Sample.** Every chunk if there are at most 100; otherwise a stratified sample of 100
by oracle label (seed 0), keeping all REQUIRED chunks.

**Model.** `convaiinnovations/laya-typed-decisions` at the revision pinned in
`laya_apple/registry.py`, through the public `Laya.predict` API, zero-shot, no
fine-tuning. The runtime is not modified.

**Query.** One `choice` question per chunk:
- context: the chunk text;
- instructions: the task (a one-line, redacted summary of the user request), the tool
  name and its arguments, and the question "Should this part of the tool output be kept
  in the coding agent's context?";
- criteria: `["keep", "drop"]`, in that order.

**Decision.** DROP when `p(drop) > p(keep)` (argmax). No threshold is tuned. The
probabilities are recorded. Chunks longer than the model's window are truncated by the
runtime, and the count of such chunks is reported.

**Latency.** `result.runtime.latency_ms` per chunk, after 3 warm-up calls. P50 and P95.

**Metrics.**
- **REQUIRED recall** (primary): the share of REQUIRED chunks Laya keeps;
- DROP precision: the share of Laya's DROPs that the oracle marks IRRELEVANT;
- overall KEEP/DROP agreement with the oracle;
- simulated token reduction: Laya-dropped tokens / sample tokens;
- latency P50 / P95;
- descriptive only: AUROC of `p(keep)` against oracle KEEP.

**Stage B verdict (fixed now).** This is a signal probe, not a production gate.
- **PROMISING:** REQUIRED recall ≥ 95%, **and** simulated reduction ≥ 10 percentage
  points above the keep-all baseline (0%).
- Otherwise **NO CLEAR SIGNAL.** That means only that the current checkpoint does not
  give enough zero-shot signal. It does not mean the decision cannot be learned.

## What this probe cannot claim

- One developer, one repository and at most 20 tool calls. The numbers carry wide
  uncertainty and are not a workload estimate.
- The oracle labels come from one model with hindsight, not from human annotation or a
  counterfactual rerun. "IRRELEVANT" means no visible later dependence.
- Claude Code sessions stand in for Pi or Hermes Agent workloads.
- Whether to start a Pi shadow-mode proof of concept is a recommendation based on this
  probe, not a pre-registered GO.
