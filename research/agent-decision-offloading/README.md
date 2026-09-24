# Agent decision offloading

**Question.** laya-apple's decision runtime is local and low-latency. Can it make
decisions inside real coding-agent workloads, for example deciding which parts of a
tool's output reach the Main LLM?

Agent offloading is one of laya-apple's product directions (see `AGENTS.md`, "Scope").
Everything in this directory is **research code**:
- the spikes are disposable and minimal;
- `laya_apple` never imports them;
- they are not the production integrations.

An experiment that passes its recorded criteria can later be promoted into production
code, which is rebuilt to production standards in its own pull request.

## Experiments

| Experiment | Question | Status |
|---|---|---|
| [EXP-000](EXP-000-integration-feasibility.md) | Can OpenClaw, Hermes Agent and Pi intercept and replace a tool result before it first enters Main LLM context, without forking the runtime? | **GO**: Pi and Hermes Agent pass C1–C7; OpenClaw is partial ([matrix](results/integration-matrix.md)) |

## Next stage (not started)

EXP-000 is GO, so the next experiment is:

```text
EXP-001 Semantic Tool-Output Admission

Goal:
Can a local Laya decision model identify task-relevant
tool-output chunks before they enter Main LLM context,
with near-zero critical-information loss?
```

- **Candidate runtimes:** Pi first, then Hermes Agent.
- **Status:** EXP-001 has no criteria, code or data yet. It will be pre-registered in
  its own document before any work starts.

## Layout

```text
EXP-000-integration-feasibility.md   question, fixed Go/No-Go criteria, results
spikes/common/mock_llm.py           scripted stand-in for the Main LLM (no model)
spikes/common/mock_llm_parallel.py  two-call variant for the parallel-batch cases
spikes/{openclaw,hermes,pi}/        one minimal sentinel spike per runtime
results/integration-matrix.md       the capability matrix with evidence
```
