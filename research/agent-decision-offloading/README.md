# Agent decision offloading

**Question.** laya-apple's decision runtime is local and low-latency. Can it make
decisions inside real coding-agent workloads, for example deciding which parts of a
tool's output reach the Main LLM?

Agent offloading is a **paused** research track in laya-apple (see `AGENTS.md`, "Scope").
Everything in this directory is **research code**:
- the spikes are disposable and minimal;
- `laya_apple` never imports them;
- they are not the production integrations.

An experiment that passes its recorded criteria can later be promoted into production
code, which is rebuilt to production standards in its own pull request.

## Closure summary

```text
EXP-000   Integration feasibility       NO-GO under original scope
EXP-000B  Ordinary tool-result seam     PASS
EXP-001   Semantic context admission    NO-GO
EXP-002   Main-LLM turn avoidance       NO-GO
Status                                  PAUSED
```

- **What was shown.** Pi and Hermes Agent can replace an ordinary tool result before it
  enters the Main LLM's context, without a fork (EXP-000B).
- **What was not found.** Any measurable value in this repository's Claude Code sessions.
  - An ideal chunk-level admission gate would drop 0% of qualifying tool output
    (EXP-001).
  - 0 of 30 post-tool Main LLM turns were only a fixed control action. Those turns
    carried 4,608,057 input tokens, none of it in bypassable turns (EXP-002).
- **What was not tested.**
  - No Laya checkpoint was run.
  - The results do not show that Laya is unsuited to these decisions, only that this
    workload offers too little to decide.
  - Whether Hermes Agent can short-circuit a Main LLM invocation stays UNKNOWN.
- **Resume only if** new evidence shows a workload with measurable offloading value,
  such as substantial redundant tool-output context or a meaningful fraction of
  fixed-action post-tool LLM turns.

Every NO-GO result, its evidence and its limitations stay recorded below.

## Experiments

| Experiment | Question | Status |
|---|---|---|
| [EXP-000](EXP-000-integration-feasibility.md) | Can OpenClaw, Hermes Agent and Pi intercept and replace a tool result before it first enters Main LLM context, without forking the runtime? | **NO-GO** under the pre-registered interpretation. Only Pi passes C1–C7. Hermes Agent and OpenClaw are PARTIAL ([matrix](results/integration-matrix.md)) |
| [EXP-000B](EXP-000B-ordinary-tool-results.md) | The same question, with the scope fixed in advance to ordinary tool results | **PASS**: Pi and Hermes Agent pass C1–C7 |
| [EXP-001 quick probe](EXP-001-quick-probe.md) | Is there enough upside in semantic tool-output admission, and does a current Laya checkpoint show a zero-shot KEEP/DROP signal? | **NO-GO** at Stage A: the oracle reduction is 0% on 10 tool results. Laya was not tested |
| [EXP-002 quick probe](EXP-002-quick-probe.md) | Are enough post-tool Main LLM turns control-only to be worth offloading, and can Hermes Agent short-circuit such a turn? | **NO-GO**: 0 of 30 turns are bypassable. The Hermes spike was not run |

**Direction status: paused.** EXP-002 is NO-GO, so under its pre-registered rule the
agent decision offloading direction is paused and no further experiments are added.

EXP-000B's forward decision was **GO** toward EXP-001, for ordinary tool results only. The
EXP-001 and EXP-002 quick probes that followed are both NO-GO. The limitations recorded
for EXP-000 remain in force.

## Full EXP-001 (not started)

The EXP-001 [quick probe](EXP-001-quick-probe.md) found no upside for chunk-level
admission in this repository's sessions, so the full EXP-001 below has not started.

The full experiment would be:

```text
EXP-001 Semantic Tool-Output Admission

Goal:
Can a local Laya decision model identify task-relevant
tool-output chunks before they enter Main LLM context,
with near-zero critical-information loss?
```

- **Scope:** ordinary tool results, as defined in EXP-000B.
- **Candidate runtimes:** Pi first, then Hermes Agent.
- **Status:** not started. The quick probe is NO-GO. Any full EXP-001 needs its own
  pre-registration.

## Layout

```text
EXP-000-integration-feasibility.md   question, fixed Go/No-Go criteria, results
EXP-000B-ordinary-tool-results.md    pre-registered ordinary-tool-result scope, result
EXP-001-quick-probe.md               pre-registered value / signal probe, result
EXP-002-quick-probe.md               pre-registered turn-avoidance probe, result
spikes/common/mock_llm.py           scripted stand-in for the Main LLM (no model)
spikes/common/mock_llm_parallel.py  two-call variant for the parallel-batch cases
spikes/{openclaw,hermes,pi}/        one minimal sentinel spike per runtime
results/integration-matrix.md       the capability matrix with evidence
results/exp-000b/{pi,hermes}/       EXP-000B confirmatory evidence
results/exp-001-quick-probe/         sanitized per-chunk labels and summary (no text)
results/exp-002-quick-probe/         sanitized per-case metadata and summary (no text)
```
