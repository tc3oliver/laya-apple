# EXP-000 integration matrix

The verdict rules are in [EXP-000](../EXP-000-integration-feasibility.md#verdict-rules),
and the scope of C1–C7 is in
[Scope clarification](../EXP-000-integration-feasibility.md#scope-clarification-recorded-after-results).

Every PASS and PARTIAL cites an API, an upstream source reference and a spike result. The
spike READMEs hold the full evidence:
[OpenClaw](../spikes/openclaw/README.md), [Hermes Agent](../spikes/hermes/README.md) and
[Pi](../spikes/pi/README.md).

## Upstream versions

| Runtime | Repository | Version / commit | Date read |
|---|---|---|---|
| OpenClaw | github.com/openclaw/openclaw | release `v2026.9.6`, tag commit `eb377ac59e6c`; npm `openclaw@2026.9.6` | 2026-09-24 |
| Hermes Agent | github.com/NousResearch/hermes-agent | `main` @ `7de8728cba33` (v0.21.4) | 2026-09-24 |
| Pi | github.com/earendil-works/pi (formerly badlogic/pi-mono) | tag `v0.87.1` @ `f07218c4d4bb`; npm `@earendil-works/pi-coding-agent@0.87.1` | 2026-09-24 |

## Matrix

| Capability | OpenClaw | Hermes | Pi |
|---|---|---|---|
| Intercept raw tool result (C1) | PARTIAL | PASS | PASS |
| Access tool name/args (C1) | PARTIAL | PASS | PASS |
| Async processing (C2) | PASS | PASS | PASS |
| Replace model-bound result (C3) | PARTIAL | PASS | PASS |
| Before first context admission (C4) | PARTIAL | PASS | PASS |
| Preserve raw result separately (C5) | PASS | PASS | PASS |
| No historical rewrite (C6) | PASS | PASS | PASS |
| No runtime fork (C7) | PASS | PASS | PASS |
| Pre-tool hook | PARTIAL | PARTIAL | PARTIAL |
| Pre-LLM hook | PARTIAL | PARTIAL | PARTIAL |
| Potential LLM-turn bypass | PARTIAL | UNKNOWN | PARTIAL |

**All seven of C1–C7 pass for:** Hermes Agent and Pi. OpenClaw passes all seven only in
its embedded runtime (see below).

## Evidence

### What the Main LLM received

Every case ran against the scripted mock
([`spikes/common/mock_llm.py`](../spikes/common/mock_llm.py)). The mock quotes the tool
message it received and checks whether `LAYA_SENTINEL` appears anywhere in the request
body. Where a case ran a second user turn (the C6 check), both turns gave the same result.
Shell commands print the sentinel with `printf 'AAA LAYA_%s BBB\n' SENTINEL`, so the
literal never appears in the model's own tool-call arguments.

| Runtime | Case | Tool message received | Raw sentinel anywhere in request |
|---|---|---|---|
| OpenClaw | control, plugin tool, no filter | `AAA LAYA_SENTINEL BBB` | true |
| OpenClaw | plugin tool | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false** |
| OpenClaw | built-in `exec` | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false** |
| OpenClaw | tool that throws | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false** |
| OpenClaw | default Tool Search `tool_call` bridge | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false** |
| Hermes | control, no filter | `AAA LAYA_SENTINEL BBB` | true |
| Hermes | plugin tool; plugin tool via `tool_call` bridge | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false** |
| Hermes | built-in `terminal` | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false** |
| Hermes | two concurrent `read_file` calls | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` ×2 | **false** |
| Hermes | tool that raises | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false** |
| Hermes | background terminal, control (`transform_tool_result` only) | notification contains `AAA LAYA_SENTINEL BBB` | true |
| Hermes | background terminal + `transform_terminal_output` | notification contains `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false** |
| Hermes | plugin-side 0.2 s deadline, 2 s decision | `[LAYA_SPIKE_DEADLINE_FALLBACK]` | **false** |
| Hermes | Hermes hook timeout 0.01 s, no plugin deadline (fail-open probe) | `AAA LAYA_SENTINEL BBB` | true |
| Hermes | concurrent batch, inline tool raises, control (no guard) | `Error executing tool 'session_search': local index read failed: AAA LAYA_SENTINEL BBB` | true |
| Hermes | same, with the plugin's `tool_execution` exception guard | `[LAYA_SPIKE_EXCEPTION_FALLBACK]` | **false** |
| Pi | control, no extension | `AAA LAYA_SENTINEL BBB` | true |
| Pi | extension tool | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false** |
| Pi | built-in `bash` | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false** |
| Pi | built-in `bash`, exit code 3 | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` (`isError`) | **false** |
| Pi | parallel batch, control, no filter | `AAA LAYA_SENTINEL BBB` ×2 | true |
| Pi | parallel batch of two extension-tool calls; hooks overlapped; admitted in source order | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` ×2 | **false** |

### C1–C7

| Capability | OpenClaw | Hermes | Pi |
|---|---|---|---|
| C1 intercept, name/args | `api.registerAgentToolResultMiddleware`; the event carries `toolName`, `args` and the raw `result` (`src/agents/harness/tool-result-middleware.ts:404-479`, `src/agents/embedded-agent-runner/extensions.ts:55-117`). Side channel holds the raw result, name and args for 4 tool classes. **PARTIAL:** no hook sees Codex-native tools in the Codex app-server harness, external CLI/ACP backends, or provider-hosted tools | `transform_tool_result(tool_name, args, result, …)` (`model_tools.py:848-865`, `:955-959`). Raw result captured for plugin, built-in, parallel, bridged and error cases | `pi.on("tool_result")` via `afterToolCall` (`packages/agent/src/agent-loop.ts:816-861`, `coding-agent/src/core/agent-session.ts:551-584`). Raw result captured for extension tool, `bash`, `bash` error and a parallel batch |
| C2 async | The middleware is awaited inside the awaited `afterToolCall` (`tool-result-middleware.ts:444`, `agent-loop.ts:1267`). The awaited delay is in each side-channel record | An `async def` hook runs to completion under `asyncio.run` on a worker thread, and the tool loop blocks on it (`hermes_cli/plugins_dispatch.py:199-207`, `:269-355`). The awaited delay is in each side-channel record; two concurrent calls overlapped | Handlers awaited (`extensions/runner.ts:1090`, `agent-loop.ts:829`). The awaited delay is in each side-channel record (`awaited_ms`); two parallel hooks overlapped (`summary.json` → `parallel.hooks_overlap`) |
| C3 replace | The returned result becomes the `toolResult` content (`agent-loop.ts:1278-1285`, `:1520-1541`). Mock: filtered text only. **PARTIAL:** the same paths as C1 cannot be replaced | First `str` return replaces the result (`model_tools.py:862`). Mock: filtered text only | Returned `content` replaces the result (`agent-loop.ts:843`). Only `content` goes to the provider (`ai/src/api/openai-completions.ts:1398-1424`). Mock: filtered text only |
| C4 before admission | Middleware runs in `finalizeExecutedToolCall`, before `emitToolResultMessage` pushes the message to `context.messages` (`agent-loop.ts:1256-1291` then `:1520-1541`). The model-bound `content` persisted in the session is the filtered text. Raw text can persist outside it: in `details` when the filter replaces only `content`, and in Tool Search nested records (`results/*.state-scan.jsonl`). Neither is sent to the model. **PARTIAL:** same paths as C1 | Transform at `model_tools.py:959` runs before `messages.append` (`agent/tool_executor.py:1101`). `state.db`: 0 raw rows with the filter, 8 without | Hook runs before `createToolResultMessage` (`agent-loop.ts:880`), the state push (`agent.ts:573`) and `appendMessage` (`agent-session.ts:940`). The extension recorded `tool_result_in_session_when_hook_ran: false` |
| C5 raw retained | The plugin writes the raw result to its own side channel | Same | Same |
| C6 no history rewrite | In all 9 cases, the turn-1 message digests are a prefix of turn 2 once the one runtime-appended tail message is excluded. That message's digest is identical on every request | In turn 2 (`--resume`), the prefix digests are unchanged in all cases | In turn 2 (`-c`), the 4/4 prefix digests are unchanged |
| C7 no fork | Public plugin API plus a manifest contract; unmodified npm release | Directory plugin plus `hermes plugins enable`; the checkout stayed clean | `--extension <file>`; unmodified npm release |

### Pre-tool, pre-LLM and LLM-turn bypass (investigation only, not implemented)

| Capability | OpenClaw | Hermes | Pi |
|---|---|---|---|
| Pre-tool hook | **PARTIAL.** `before_tool_call` can allow, deny (`block`) or modify (`params`) (`src/plugins/hook-before-tool-call-result.ts:14-35`). It has no redirect. Source only; not exercised by the spike | **PARTIAL.** `pre_tool_call` can allow, `block`, `approve` (escalate) or `modify` (`hermes_cli/plugins.py:1854-1906`). It has no native redirect; `tool_execution` middleware can substitute a result. The spike saw it fire before every tool; directives not exercised | **PARTIAL.** `tool_call` can allow, `block` (optionally with `terminate`) or modify `input` in place (`agent-loop.ts:722-750`). It has no redirect; a same-name tool override is the closest. The spike saw it fire before execution |
| Pre-LLM hook | **PARTIAL.** `llm_input` is observe-only (`src/plugins/hook-types.ts:1078`). `before_prompt_build` runs at prompt build, not per model call. A `contextEngine.assemble` runs per call but cannot touch the current turn's pending messages (`tool-result-context-guard.ts:413-432`) | **PARTIAL.** `pre_llm_call` fires once per user turn, not before the model call that follows a tool result (`agent/turn_context.py:745-798`). `llm_request` middleware can replace provider kwargs per request (`agent/turn_api_request.py:141-143`); not exercised | **PARTIAL.** Continue and modify are supported: the `context` / `context_with_system` event fires before every model request and can return request-local messages (`extensions/runner.ts:1190-1251`), and `before_provider_request` can replace the payload (`:1253-1282`). The spike saw a `context` event before each request; modify was not exercised. Bypass is not supported: no hook can return a response instead of a request |
| Potential LLM-turn bypass | **PARTIAL (STOP only).** The middleware can set `terminate: true`. The loop then ends without another provider call only if every result in the batch terminates (`agent-loop.ts:916-921`). No CONTINUE or RETRY without the model | **UNKNOWN.** `llm_execution` middleware may return a response without calling the provider (`hermes_cli/middleware.py:170-201`). Whether the loop accepts a synthesized response is untested | **PARTIAL (STOP only).** Pre-tool `{block, terminate}` or a tool's own `terminate` ends the loop when the whole batch terminates (`agent-loop.ts:271`, `:685-687`). `tool_result` cannot set `terminate` (`agent-session.ts:578-583`). No CONTINUE without the model |

## Limitations by runtime

### OpenClaw
- Replacement works only for tools that OpenClaw's own embedded agent loop executes. It
  does not work for:
  - Codex-native tools under the Codex app-server harness, which are observe-only;
  - external CLI and ACP backends;
  - provider-hosted tools.

  An official OpenAI endpoint may route to the Codex harness.
- `details` is persisted but not model-bound. For `exec` it carries the raw output, so a
  filter must replace it too.
- The Tool Search bridge persists the inner tool's raw result as a nested record that is
  excluded from context. The middleware sees only the outer `tool_call`.
- The middleware fails **closed**: a throw becomes a generic error text, never the raw
  result.

### Hermes Agent
- **Ordinary tool results:** C1–C7 PASS.
- **Async `delegate_task` completion summaries** (`[ASYNC DELEGATION COMPLETE …]`) bypass
  `transform_tool_result`. The runtime adds them to the context directly.
  - The subagent's own tool results still go through the filter.
  - Supporting admission of subagent summaries would need a separate study of another
    lifecycle seam.
  - This limitation is not a success criterion.
- **Background terminal output** re-enters as a notification and bypasses
  `transform_tool_result`. `transform_terminal_output` covers it before admission (spike
  `background_terminal`). Registering that hook also rewrites foreground terminal output.
  Hermes writes the unfiltered output to `logs/process-results/`.
- **The hook fails open.** On a Hermes hook timeout (default 30 s) or an exception, the
  raw result is admitted.
  - A plugin-side deadline that returns a safe fallback prevents it (spike
    `plugin_deadline_fallback`).
  - A Hermes filter therefore has to be written defensively, with both a deadline and an
    exception guard.
  - A plugin without one leaks (spike `hook_timeout_failopen`).
- **Async runs on a fresh event loop per call,** on a worker thread. It is not a shared
  agent loop.
- **Exceptions on the concurrent path skip the transform.** When an inline tool raises
  in a concurrent batch, Hermes wraps the exception text into an error message without
  running `transform_tool_result` (`agent/tool_executor.py:1303-1306`). The control case
  shows the raw text reaching the model.
  - A `tool_execution` middleware in the filter plugin can catch the exception and return
    a safe fallback; spike `concurrent_exception_guard` shows this.
  - The guard cannot wrap these:
    - NeMo Relay managed execution, which was not tested;
    - `BaseException` subclasses, for which Hermes substitutes its own text with no raw
      data;
    - error strings that `model_tools.py:961-966` returns for exceptions raised
      outside the middleware chain.
  - The guard returns a fixed fallback. That shows the exception is contained. It does
    not show that an awaited local decision works on this path.
- **Blocked, unknown and invalid calls** produce runtime-generated text. It is outside the
  C1–C7 scope, as for the other two runtimes.

### Pi
- **Handler errors fail open.** An exception in a `tool_result` handler admits the raw
  result (`extensions/runner.ts:1109-1118`). The filter must catch its own errors.
- **`details` is kept unless replaced.** It is persisted, not sent to the model. A
  truncated bash output leaves the full raw output in a temp file.
- **Streaming partial output is not filtered.** `tool_execution_update` events reach the
  UI and extensions before the hook. They are not model-bound.
- **Paths without `tool_result`.** User `!` shell commands and extension `sendMessage`
  enter the context without it. These are not tool results.
