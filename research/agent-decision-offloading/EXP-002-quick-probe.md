# EXP-002 quick probe: Main-LLM turn avoidance

**Status:** complete. **Stage A: NO-GO** (0 of 30 turns bypassable). Stage B was not
run. **Overall: NO-GO** (see [Result](#result)).

[EXP-001's quick probe](EXP-001-quick-probe.md) is NO-GO, and tool-output admission is
closed for the current workload. This probe does not extend it. It asks a different
question, in two stages:
1. **Value (Stage A).** In real coding-agent work, what share of the Main LLM turns that
   follow a tool result only make a small control decision that a fixed action could
   replace?
2. **Seam (Stage B).** Can Hermes Agent short-circuit that Main LLM invocation through an
   official plugin or middleware API, without a runtime fork?

Out of scope: running Laya, fine-tuning, a dataset, Pi / OpenClaw / Claude Code
adapters, an installer, and any change to `laya_apple`.

## Stage A: value ceiling

### Data

**Source.** Claude Code sessions of this repository only
(`~/.claude/projects/-Users-oliver-Developer-src-personal-laya-apple/`), top-level
sessions and subagent transcripts. There are no Pi sessions for this repository.

Excluded:
- the session that runs this probe, because it contains the probe's own design turns;
- a case whose tool call ran in another repository or a third-party checkout, or read
  Claude Code session files, as in the
  [EXP-001 probe](EXP-001-quick-probe.md#selection-as-implemented).

**Case.** One case is a tool result, or one batch of parallel tool results, followed by
the next Main LLM turn. A turn is one model response: all assistant entries that share a
message id.

**Selection.** The most recent cases, newest first, up to **30**, at most 10 per session
file. Nothing more is added if fewer exist.

**Privacy.** Raw session content stays in local scratch and is never committed. The
repository gets only anonymised per-case metadata and aggregate results.

### Per-case metadata

| Field | Values |
|---|---|
| `previous_tool_type` | tool name. For `Bash`, one of test, search, build_lint, git, gh, file_read, script, other |
| `next_action_type` | the tool name(s) the turn calls, or `final_text` when it only writes text |
| `has_substantive_generation` | true if the turn writes code, prose beyond a one-line status, or tool arguments not copied from earlier context |
| `fixed_action` | one of `RETRY`, `CONTINUE`, `STOP`, `INSPECT_FAILURE`, `RUN_TARGETED_TEST`, `ESCALATE`, or `NONE` |
| `label` | `BYPASSABLE` or `NOT_BYPASSABLE` |
| tokens | from the turn's recorded `usage`: `input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens` |

### BYPASSABLE (fixed now)

A turn is `BYPASSABLE` only if **all** of these hold:
1. Its main result is only control flow or a single tool action.
2. It generates no new code, no natural-language content and no complex arguments. A
   tool call counts only if its arguments are fixed or copied verbatim from earlier
   context, for example rerunning the same command.
3. It maps to one action in the fixed set above.
4. It needs no change to the earlier context.

Any ambiguous case is `NOT_BYPASSABLE`. "Looks easy" does not make a turn bypassable.
Hidden reasoning (thinking) does not disqualify a turn by itself; only the turn's output
is judged.

The labeler is the model running this probe, with an independent re-label by a separate
verifier. The labels are frozen, and their digest recorded, before the verdict is
computed.

### Metrics

- total post-tool model turns;
- `BYPASSABLE` turns, and their share (the verdict metric);
- the distribution of `fixed_action` and of `next_action_type`;
- the input tokens entering the bypassable turns, and their share of all input tokens
  across the cases. Input tokens = `input_tokens` + `cache_read_input_tokens` +
  `cache_creation_input_tokens`. Reported only if the `usage` data is present.

### Stage A verdict (fixed now)

| Bypassable turns | Verdict |
|---|---|
| < 10% | **NO-GO.** Stop. Stage B does not run |
| 10%–20% | **MARGINAL.** Stage B runs |
| ≥ 20% | **PROMISING.** Stage B runs |

## Stage B: Hermes short-circuit feasibility (only if Stage A ≥ 10%)

No Laya. Hermes Agent v0.21.4 (`7de8728cba33`), the version EXP-000 used. EXP-000
recorded the Hermes `llm_execution` middleware as UNKNOWN for this capability.

**Sentinel spike.** Use the loopback mock provider
([`spikes/common/mock_llm.py`](spikes/common/mock_llm.py) or a variant of it), and count
its requests.

| Case | Flow | Expected Main-LLM requests after the tool result |
|---|---|---|
| control | tool → Main LLM | +1 |
| short-circuit | tool → plugin/middleware returns a synthetic, valid response | +0 |
| fallback | tool → the plugin's decision fails (raises or times out) | +1: the request reaches the Main LLM |

**Stage B PASS requires all of:**
1. no runtime fork (the checkout stays clean);
2. an official plugin or middleware API;
3. the provider request is actually avoided: the mock's request count is +0 in the
   short-circuit case and +1 in the control;
4. the agent loop accepts the synthetic result: the turn completes normally, and the
   synthetic response is in the session history;
5. the existing history is unchanged: the message digests before the short-circuit point
   are the same as in the control;
6. a failure in the plugin falls back safely to the Main LLM (the fallback case).

Otherwise **FAIL**. Only spike behaviour counts, backed by a source reference. A source
reading alone does not.

## Overall verdict (fixed now)

**GO** only if Stage A is at least 10% **and** Stage B passes. Only then may a later
experiment study whether Laya can make these decisions correctly.

Otherwise **NO-GO**. The Agent Decision Offloading direction is then paused, and no
further experiments are added to it.

## What this probe cannot claim

- One developer, one repository, at most 30 turns, all from Claude Code. The numbers
  carry wide uncertainty.
- The labels come from models with hindsight, not from a counterfactual rerun of the
  agent.
- Claude Code turns stand in for Hermes Agent or Pi turns.
- A Stage B PASS shows that the seam exists. It says nothing about the quality of the
  decisions.

## Result

The pre-registration was committed in `c1fde64` before any session was read.

```text
EXP-002 Quick Probe

Stage A: How much theoretical value exists?
  Post-tool model turns          30 (2 sessions, 4 session files)
  BYPASSABLE turns               0
  Bypassable turn %              0%
  Input tokens, all 30 turns     4,608,057
  Input tokens, bypassable       0 (0%)
  Verdict                        NO-GO (< 10%)

Stage B: Can Hermes actually avoid the model invocation?
  Not run (Stage A < 10%). The Hermes llm_execution seam stays UNKNOWN.

Overall                          NO-GO
```

**Action distribution.** The next turn called `Bash` in 23 cases, `Bash` twice in 1,
`Edit` twice in 1 and `Read` in 1. It wrote only text in 4 cases: 3 final reports and 1
verification report.

**Closest candidates.** In 5 turns the intent matches a fixed action, but the turn still
generated its own arguments:

| Intent | Turns | Why not BYPASSABLE |
|---|---|---|
| `RUN_TARGETED_TEST` | 2 | One turn ran the two checks named in its task prompt, but it combined them and added `2>&1 \| tail`. The other ran `ruff` on files it chose |
| `INSPECT_FAILURE` | 2 | After a failed patch, the turn chose a new grep pattern, or a file offset and length, to find the cause |
| `RETRY` | 1 | An empty grep was retried with a different command, not the same one |

Every other turn wrote new code, a new command or a report.

The labels were frozen before the verdict was computed (SHA-256 `3e42df77…ca3554`). The
per-case metadata is in [`results/exp-002-quick-probe/`](results/exp-002-quick-probe/).

### Selection as implemented

- The first extraction treated each tool result as a case. Claude Code writes the
  several tool calls of one response as separate entries, with results in between. So
  some "next turns" were the same model response.
- Before any case was labelled, the extraction was fixed to the pre-registered unit: all
  results of one response, followed by the next response with a different message id and
  no new user message in between.
- The pool had 790 cases, after excluding this probe's own session and the EXP-001
  exclusions. The 30 newest were taken, at most 10 per session file.

### Answers

- **Stage A.** In this workload, almost no post-tool Main LLM turn only makes a small
  control decision. Each one writes a new command, code or a report. An ideal fixed-action
  offloader would avoid 0 of 30 turns.
- **Stage B.** Not tested. Whether Hermes can short-circuit a model invocation stays
  UNKNOWN, as EXP-000 recorded it.
- **Overall: NO-GO.** Under the rule fixed in advance, the Agent Decision Offloading
  direction is paused. No further experiments are added to it.

### Limitations

- Everything under [What this probe cannot claim](#what-this-probe-cannot-claim) applies.
- The 30 cases come from 2 sessions. Most are from one editing burst and two verifier
  subagents, so they are not independent.
- The agent here is Claude Code on a frontier model, which tends to combine steps into
  one rich command. An agent that takes smaller steps, such as polling loops, CI waits
  or retry-heavy workflows, could have more control-only turns. This probe did not
  measure one.
- `previous_tool_type` is a coarse classification of the command text. It is descriptive
  only and does not enter the verdict.
