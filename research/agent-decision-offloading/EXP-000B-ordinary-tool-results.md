# EXP-000B: Ordinary tool-result integration feasibility

**Status:** pre-registered, not yet run.

## Why this experiment exists

[EXP-000](EXP-000-integration-feasibility.md) is **NO-GO** under its pre-registered
interpretation. Its criteria did not say which runtime outputs count as a "tool result".
After the spikes ran, that question was settled in a way that narrowed the evaluation
population. Under the narrower scope, Pi and Hermes Agent pass C1–C7. That is a post-hoc
finding, and EXP-000 does not claim it as its verdict.

EXP-000B fixes the narrower scope **before** it runs, then tests it with a small
confirmatory run. It does not change C1–C7.

**What EXP-000B cannot claim.** The scope below was chosen after the EXP-000 results
were known, and the confirmatory cases reuse EXP-000's spike code. EXP-000B confirms
that the result reproduces under a scope fixed in advance. It is not evidence independent
of EXP-000, and it says nothing about paths outside that scope.

## Scope (fixed before the run)

A **tool result** is the output of an ordinary tool execution, returned to the agent
loop through the runtime's normal tool-result path.

Not in scope:
- **Subagent-generated summaries.** For example, Hermes Agent's asynchronous
  `delegate_task` completion summary, `[ASYNC DELEGATION COMPLETE …]`.
- **Runtime-generated text.** This covers blocked, unknown or invalid calls, status text,
  and error text produced outside the tool.

Paths outside this scope are not tested here. Their limitations, as recorded in the
[EXP-000 matrix](results/integration-matrix.md), stay in force.

| Runtime | In EXP-000B |
|---|---|
| Pi 0.87.1 (`f07218c4d4bb`) | yes |
| Hermes Agent v0.21.4 (`7de8728cba33`) | yes |
| OpenClaw | no. Its EXP-000 result (C1, C3, C4 PARTIAL) is unchanged |

The upstream versions are the ones EXP-000 used.

## Criteria (unchanged)

C1–C7 exactly as defined in
[EXP-000](EXP-000-integration-feasibility.md#go--no-go-criteria), with EXP-000's
[verdict rules](EXP-000-integration-feasibility.md#verdict-rules).

**EXP-000B PASS requires both Pi and Hermes Agent to PASS all seven of C1–C7.**
Otherwise EXP-000B is FAIL.

## Confirmatory run

One negative control and one filtered case per runtime, run with the existing spike
scripts and the common mock. Nothing else in the spikes changes.

| Runtime | Negative control | Filtered case (two user turns) |
|---|---|---|
| Pi | `control_tool_nofilter` | `ext_tool` |
| Hermes Agent | `negative_control` | `plugin_tool` |

```bash
# Pi
LAYA_SPIKE_RESULTS="$PWD/research/agent-decision-offloading/results/exp-000b/pi" \
LAYA_SPIKE_CASES="control_tool_nofilter ext_tool" \
  research/agent-decision-offloading/spikes/pi/run.sh

# Hermes Agent
LAYA_SPIKE_RESULTS="$PWD/research/agent-decision-offloading/results/exp-000b/hermes" \
  research/agent-decision-offloading/spikes/hermes/run.sh negative_control plugin_tool
```

The committed evidence goes to `results/exp-000b/`. EXP-000's own result files are not
touched.

**Validity.** The run counts only if each negative control delivers the raw sentinel to
the mock (`raw_sentinel_anywhere_in_request: true`). If the harness fails (a Pi timeout,
a mock that never starts, a case with no mock request), the run is repeated and the
failure is recorded. A valid run that fails a criterion is **not** repeated.

**Pass evidence for the filtered case:**

| # | Evidence |
|---|---|
| C1 | The side channel records the tool name, the arguments and the raw output `AAA LAYA_SENTINEL BBB` |
| C2 | The side channel records an awaited delay of at least 50 ms |
| C3 | The mock's tool message is exactly `AAA [FILTERED_BY_LAYA_SPIKE] BBB`, and `raw_sentinel_anywhere_in_request` is `false` on every request |
| C4 | The EXP-000 source reference, plus behaviour that agrees with it. Pi: the extension records `tool_result_in_session_when_hook_ran: false`. Hermes: the `post_tool_call` record shows no raw sentinel, and the session database has 0 raw rows |
| C5 | The side channel holds the raw output |
| C6 | The turn-1 message digests are an unchanged prefix of turn 2 |
| C7 | Loaded only as a Pi extension or a Hermes plugin. The Pi package is the pinned npm release. The Hermes checkout is at the pinned commit with a clean `git status` |

## Forward decision

- **EXP-000B PASS:** the research track's forward decision is **GO**. EXP-001 (Semantic
  Tool-Output Admission) may then be pre-registered, with its scope limited to ordinary
  tool results.
- **EXP-000B FAIL:** the forward decision is **NO-GO**, and EXP-001 does not start.

## Result

Not run yet.
