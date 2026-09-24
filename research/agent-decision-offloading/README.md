# Agent decision offloading

**Question.** laya-apple's decision runtime is local and low-latency. Can it make
decisions inside real coding-agent workloads, for example deciding which parts of a
tool's output reach the Main LLM?

Agent offloading is a research track and a candidate product direction for laya-apple
(see `AGENTS.md`, "Scope"). It becomes a product direction only after EXP-001 and the
later value experiments pass. Everything in this directory is **research code**:
- the spikes are disposable and minimal;
- `laya_apple` never imports them;
- they are not the production integrations.

An experiment that passes its recorded criteria can later be promoted into production
code, which is rebuilt to production standards in its own pull request.

## Experiments

| Experiment | Question | Status |
|---|---|---|
| [EXP-000](EXP-000-integration-feasibility.md) | Can OpenClaw, Hermes Agent and Pi intercept and replace a tool result before it first enters Main LLM context, without forking the runtime? | **NO-GO** under the pre-registered interpretation. Only Pi passes C1–C7. Hermes Agent and OpenClaw are PARTIAL ([matrix](results/integration-matrix.md)) |
| [EXP-000B](EXP-000B-ordinary-tool-results.md) | The same question, with the scope fixed in advance to ordinary tool results | **PASS**: Pi and Hermes Agent pass C1–C7 |
| [EXP-001 quick probe](EXP-001-quick-probe.md) | Is there enough upside in semantic tool-output admission, and does a current Laya checkpoint show a zero-shot KEEP/DROP signal? | **NO-GO** at Stage A: the oracle reduction is 0% on 10 tool results. Laya was not tested |

**Forward decision: GO** (from EXP-000B), for ordinary tool results only. The limitations
recorded for EXP-000 remain in force.

## Next stage

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
spikes/common/mock_llm.py           scripted stand-in for the Main LLM (no model)
spikes/common/mock_llm_parallel.py  two-call variant for the parallel-batch cases
spikes/{openclaw,hermes,pi}/        one minimal sentinel spike per runtime
results/integration-matrix.md       the capability matrix with evidence
results/exp-000b/{pi,hermes}/       EXP-000B confirmatory evidence
results/exp-001-quick-probe/         sanitized per-chunk labels and summary (no text)
```
