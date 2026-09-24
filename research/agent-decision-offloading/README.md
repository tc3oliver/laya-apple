# Agent decision offloading

**Question.** laya-apple's decision runtime is local and low-latency. Can it make
decisions inside real coding-agent workloads, for example deciding which parts of a
tool's output reach the Main LLM?

Everything here is **research code**. The spikes are disposable. `laya_apple` never
imports them, and they are not official plugins, adapters or installers for any agent
(see `AGENTS.md`, "Scope").

## Experiments

| Experiment | Question | Status |
|---|---|---|
| [EXP-000](EXP-000-integration-feasibility.md) | Can OpenClaw, Hermes Agent and Pi intercept and replace a tool result before it first enters Main LLM context, without forking the runtime? | Criteria fixed; spikes pending |

Later experiments depend on EXP-000 being GO.

## Layout

```text
EXP-000-integration-feasibility.md   question, fixed Go/No-Go criteria, results
spikes/common/mock_llm.py           scripted stand-in for the Main LLM (no model)
spikes/{openclaw,hermes,pi}/        one minimal sentinel spike per runtime
results/integration-matrix.md       the capability matrix with evidence
```
