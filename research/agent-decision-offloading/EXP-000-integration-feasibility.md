# EXP-000: Agent integration feasibility

**Status:** criteria fixed, spikes not yet run.

## Question

Do OpenClaw, Hermes Agent and Pi each offer an extension point that can do the
following without forking the agent runtime?
1. Intercept a tool result.
2. Process it locally.
3. Replace it before it first enters the Main LLM's context.

If none of them can, the later Laya experiments are not worth running.

| In scope | Out of scope |
|---|---|
| OpenClaw, Hermes Agent, Pi | Claude Code, and any other agent |

**This experiment uses no Laya model.** It runs no inference, no KEEP/DROP decision and
no semantic relevance scoring. It collects no dataset and does no fine-tuning. It tests
only whether a transform can be inserted at the right point.

## Go / No-Go criteria

These criteria were fixed before any spike ran. The commit that adds this file comes
before every spike commit. They are not relaxed after results are known.

**GO** requires **at least two** runtimes to PASS all seven of these capabilities:

| # | Capability |
|---|---|
| C1 | Raw tool-result interception: tool name, arguments and raw output available to the extension |
| C2 | Async processing: the interception point can `await` an async function before the result is admitted |
| C3 | Model-bound replacement: the Main LLM receives only `AAA [FILTERED_BY_LAYA_SPIKE] BBB`. `LAYA_SENTINEL` appears nowhere in the model request |
| C4 | Before first admission: the transform runs before the result is appended to the conversation/context, not by rewriting it afterwards |
| C5 | Raw result retention: the original is stored on a side channel (a temp file is enough) while the transformed text goes to the model |
| C6 | No historical rewrite: the earlier messages A→B→C reach the model unchanged. Only the new D is transformed |
| C7 | No runtime fork: done entirely through a plugin, extension, hook or middleware. Upstream core source is not modified |

**It also requires that OpenClaw or Hermes Agent is one of the passing runtimes.**

Otherwise **EXP-000 = NO-GO**. A NO-GO stops EXP-001 (semantic tool-output admission),
the Laya relevance experiments, dataset collection, fine-tuning and any installer work.
Limitations are not worked around to keep the research going.

### Verdict rules

- **PASS:** shown by the spike against the mock LLM and backed by an upstream source
  reference. For C4 and C7, a source reference plus spike behaviour that agrees with it
  is enough.
- **PARTIAL:** it works only for some tool classes, code paths or configurations. The
  conditions are listed. For GO, a PARTIAL counts as not passed.
- **FAIL:** does not work, or needs a runtime fork (`FAIL — runtime fork required`).
- **UNKNOWN:** could not be determined. For GO, this counts as not passed.
- A result is never marked PASS on the strength of an API name or documentation
  alone.

## Common sentinel test

The same test runs on each runtime.

1. The agent is pointed at the **scripted mock Main LLM**
   ([`spikes/common/mock_llm.py`](spikes/common/mock_llm.py)). It is a loopback
   OpenAI-compatible Chat Completions server.
   - On the first turn, it asks the agent to call the sentinel tool.
   - On the turn that carries the tool result, it records the exact tool-role message
     content the model received. It also checks whether `LAYA_SENTINEL` appears
     anywhere in the request body.
2. The sentinel tool returns `AAA LAYA_SENTINEL BBB`.
3. The spike's extension intercepts that raw result.
   - It `await`s an artificial async delay of about 50 ms to stand in for
     `local_decision(...)`.
   - It writes the raw result, tool name and arguments to a temp file.
   - It returns `AAA [FILTERED_BY_LAYA_SPIKE] BBB`.
4. **Primary proof:** the mock's evidence shows that the tool message is
   `AAA [FILTERED_BY_LAYA_SPIKE] BBB` and that `raw_sentinel_anywhere_in_request` is
   `false`.

   This proof also rules out one false positive: the original result entering the
   context with a FILTERED note appended afterwards.
5. **C6 check:** a second user turn follows. The mock hashes every message it receives.
   The prefix of earlier messages must be identical across turns, except for messages
   the runtime itself appends.

Using a mock instead of a real provider makes the proof exact and free of credentials.
The request body the mock records is what an actual provider would have received.

**Third-party text.** The mock's full request log contains the agent's system prompt
and tool definitions. It stays outside the repository and is never committed. Only the
evidence summary (tool-message contents, booleans, digests) is committed.

## Per-runtime capability investigation (no implementation)

Recorded as SUPPORTED / PARTIAL / NOT SUPPORTED / UNKNOWN, each with API and source
evidence:
- **Pre-tool decision:** allow, deny, modify or redirect a tool call before it runs.
- **Pre-LLM decision:** see the context before a model invocation and continue, modify
  or bypass it.
- **LLM-turn avoidance:** after a tool result and a local decision, retry, continue or
  stop without calling the Main LLM.

Runtime-specific checks:
- **OpenClaw.** Tool-result middleware (Tokenjuice-style).
  - Is the raw result visible, and is async supported?
  - Is the result model-bound?
  - Do dynamic, plugin, native and provider tools behave differently?
  - Are there tool classes whose result cannot be replaced?
- **Hermes Agent.** `transform_tool_result`, `pre_tool_call`, `pre_llm_call`,
  `post_tool_call`. Does `transform_tool_result` run before the conversation append?
- **Pi.** `tool_call`, `tool_result`, `before_agent_start`, the agent lifecycle, and
  terminate/follow-up control.

## Upstream versions

Filled in by each spike: repository, commit, version and date read.

| Runtime | Repository | Commit / version | Date read |
|---|---|---|---|
| OpenClaw | | | |
| Hermes Agent | | | |
| Pi | | | |

## Results

Not yet run. See [`results/integration-matrix.md`](results/integration-matrix.md).
