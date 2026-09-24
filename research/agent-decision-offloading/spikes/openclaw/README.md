# EXP-000 spike: OpenClaw

Can an OpenClaw plugin replace a tool result before it first enters the Main LLM's
context, without forking OpenClaw? Criteria and verdict rules:
[`EXP-000-integration-feasibility.md`](../../EXP-000-integration-feasibility.md).

**Short answer.** Yes for OpenClaw's own embedded agent runtime. The public
`api.registerAgentToolResultMiddleware(...)` plugin API sees the raw result, is awaited,
and its output is the only version the model receives. Every tool class the spike ran
passed: plugin tool, built-in `exec`, a thrown tool error, and the default Tool Search
`tool_call` bridge. The API does **not** cover every OpenClaw execution path:
- Codex-native tools in the Codex app-server harness are observe-only;
- CLI backends and ACP agents run tools in another process;
- provider-hosted (server-side) tools never produce a result OpenClaw can see.

## Upstream version

| Field | Value |
|---|---|
| Repository | https://github.com/openclaw/openclaw (not a fork; the MIT-licensed OpenClaw Foundation project, npm package `openclaw`) |
| Release read and run | `v2026.9.6`, tag commit `eb377ac59e6c9fd6c7705028034812becf00271b`, npm `latest` = `2026.9.6` (published 2026-09-23) |
| Default branch at clone time | `main` @ `fef6b1290e412761888865da5b61ee1c0ce29586` (its `package.json` still says `2026.9.5`). The source was read at the release tag so it matches the installed package |
| Date read | 2026-09-24 |
| Runtime under test | OpenClaw embedded runner (`executionTrace.runner = "embedded"`), provider `api: "openai-completions"` against the mock |

All `file:line` references below are at the release tag `v2026.9.6`.

## Files

| File | Purpose |
|---|---|
| `plugins/laya-sentinel-tool/` | Plugin (dynamic) tools: `laya_sentinel` returns `AAA LAYA_SENTINEL BBB`; `laya_sentinel_error` throws that string |
| `plugins/laya-spike-filter/` | The extension under test. It is one `registerAgentToolResultMiddleware` handler that awaits about 50 ms, appends `{toolName, args, raw result}` to the side channel, and returns `AAA [FILTERED_BY_LAYA_SPIKE] BBB` |
| `run.sh` | Reproduces every case. It installs `openclaw@2026.9.6` into scratch, gives each case an isolated `HOME`/`OPENCLAW_HOME`/state/config and a dummy key, runs the mock on `127.0.0.1:18081` and two `openclaw agent --local` turns in one session. Then it kills the mock |
| `results/<case>.evidence.jsonl` | The mock's evidence summary: tool-message text, sentinel booleans and message digests |
| `results/<case>.side-channel.jsonl` | The raw result the filter kept (C5), with paths stripped |
| `results/<case>.state-scan.jsonl` | Which files in OpenClaw's persisted state still contain the raw sentinel |
| `results/summary.jsonl` | One line per case, including the C6 prefix check |

The mock's raw request log (system prompt, tool definitions) and OpenClaw's state stay in
scratch, in a fresh per-run directory
(`~/Developer/scratch/agent-offloading/openclaw-spike/runs/<timestamp>/<case>/`), and are not
committed. The script deletes nothing: it overwrites each `results/` file by name.

## Real lifecycle (embedded runtime)

1. **Tool execution.** The agent loop in `packages/agent-core/src/agent-loop.ts` runs
   the tool.
   - While it runs, partial output goes out as `tool_execution_update` events to UI and
     channel progress (`agent-loop.ts:1096-1113`). These events are not model context.
2. **Raw result → `afterToolCall`.** `finalizeExecutedToolCall` calls
   `batch.config.afterToolCall({toolCall, args, result, isError})` and awaits it
   (`agent-loop.ts:1256-1291`).
   - It runs only when `executed.executionStarted` is true (`:1265`).
   - It replaces `content`, `details`, `isError` and `terminate` field by field
     (`:1278-1285`).
   - The contract says it runs "before `tool_execution_end` and tool-result message
     events are emitted" (`packages/agent-core/src/types.ts:379-393`).
3. **Session hook.** The session installs `afterToolCall`, which awaits
   `runner.emitToolResult({type: "tool_result", toolName, input: args, content, details,
   isError})` (`src/agents/sessions/agent-session-base.ts:255-285`).
   - `emitToolResult` chains the `tool_result` handlers. Later handlers see the
     replaced content (`src/agents/sessions/extensions/runner.ts:721-756`).
4. **Plugin middleware.** Every embedded session gets a `tool_result` handler.
   - The handler is added in `src/agents/embedded-agent-runner/extensions.ts:163-170`.
   - That handler awaits `runner.applyToolResultMiddleware({toolName, args, isError,
     result})` and returns its `content`, `details` and `terminate` (`:55-117`).
   - The runner in `src/agents/harness/tool-result-middleware.ts:404-479` works as
     follows:
     - it loads the plugin middlewares for runtime `"openclaw"`;
     - if none are registered, it returns the result unchanged (`:432-434`);
     - otherwise it awaits each handler in turn (`:442-444`) and validates each output;
     - on an invalid output or a throw it **fails closed**. The text becomes "Tool output
       unavailable due to post-processing error." and the raw text is not used
       (`:347-360`, `:449-475`).
5. **Context append.** Only after finalization does `emitToolResultMessage` build the
   `role: "toolResult"` message from `finalized.result.content`
   (`agent-loop.ts:1520-1541`).
   - The `message_end` event pushes it into `context.messages`
     (`agent-stream-response.ts:182-186`). The session also persists it on `message_end`.
6. **Model request.** The next provider call converts `context.messages`
   (`agent-stream-response.ts:132-146`: `transformContext` → `convertToLlm`) and
   streams them to the provider.
   - The tool message holds only `content`. `details` is not model-bound: the
     `c_exec_content_only` case keeps raw `details` and the mock still saw no sentinel.

**Plugin API and gates.** Registration is `api.registerAgentToolResultMiddleware(handler,
{runtimes, matcher?})` (`src/plugins/registry-registrars-tools-hooks.ts:131-205`). Three
gates apply:
- the manifest must declare `contracts.agentToolResultMiddleware: ["openclaw"]` (`:150-160`);
- a non-bundled plugin must be explicitly enabled, which `plugins.entries.<id>.enabled:
  true` provides (`:68-70`, `:161-167`);
- the handler timeout comes only from an operator hook policy (`:175`). With no policy
  there is no timeout, because `withTimeout(..., 0)` is a no-op.

## What the Main LLM saw

The quotes below are the mock's `tool_messages` on the tool-result request of turn 1 and
turn 2 (`results/*.evidence.jsonl`, `results/summary.jsonl`).

| Case | Tool (class) | Filter | Tool message received (turn 1 = turn 2) | `raw_sentinel_anywhere_in_request` t1 / t2 | C6 prefix unchanged | Raw sentinel in persisted state |
|---|---|---|---|---|---|---|
| `a_control_plugin` | `laya_sentinel` (plugin) | no | `AAA LAYA_SENTINEL BBB` | true / true | yes | yes (session DB) |
| `b_plugin` | `laya_sentinel` (plugin) | yes | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false / false** | yes | **no** |
| `c_control_exec` | `exec` (built-in) | no | `AAA LAYA_SENTINEL BBB` | true / true | yes | yes |
| `c_exec` | `exec` (built-in) | yes | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false / false** | yes | **no** |
| `c_exec_content_only` | `exec`, filter keeps raw `details` | yes | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false / false** | yes | yes: `details.aggregated` persisted |
| `d_control_error` | `laya_sentinel_error` (thrown error) | no | `{"status": "error", "tool": "laya_sentinel_error", "error": "AAA LAYA_SENTINEL BBB"}` | true / true | yes | yes |
| `d_error` | `laya_sentinel_error` (thrown error) | yes | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false / false** | yes | **no** |
| `e_control_toolsearch` | `tool_call` bridge → `laya_sentinel` (default config) | no | JSON envelope `{"tool": {...}, "result": {"content": [{"text": "AAA LAYA_SENTINEL BBB"}]}}` | true / true | yes | yes |
| `e_toolsearch` | `tool_call` bridge → `laya_sentinel` (default config) | yes | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | **false / false** | yes | yes: nested-tool transcript record (see limitations) |

The exec command is `printf "AAA LAYA_%s BBB\n" SENTINEL`. The sentinel therefore appears
only in the tool's output and never in the tool-call arguments. A `true` flag can come
only from the result.

**Side channel (C5).** Each filtered case has one record (`results/*.side-channel.jsonl`)
holding the raw content, the raw `details`, the tool name and the arguments. The measured
awaited delay (`awaitedDelayMs`) was 52 ms in every filtered case. For `exec` the raw `details` also carry `aggregated`, the raw
output. The record shows what the middleware receives:
- for the error case, `isError: true` and the runtime-built error envelope;
- for the Tool Search case, `toolName: "tool_call"`, with the inner tool in `args.id` and
  in the raw envelope.

**C6 is a prefix match once the runtime-appended tail message is excluded.** Every model
request ends with one user message that OpenClaw appends per request (an internal
runtime-context block). Its digest is identical on every request, and it always comes
last. The check drops that tail from turn 1's `message_digests`. What remains must equal
the first digests of turn 2, position by position (`c6_turn1_prefix_unchanged_in_turn2`
in `results/summary.jsonl`). A separate field, `c6_runtime_tail_digest_same`, checks that
the tail itself is the same on both turns.
- The prefix covers the system prompt, the user turn 1, the assistant tool call and the
  already-filtered tool message.
- For example, `results/b_plugin.evidence.jsonl` has `message_digests[0:4]` =
  `ad0848e8…`, `54508e83…`, `c6c2959e…`, `86b60153…` on both turns, and the tail is
  `10df915c…`.
- Digests differ between runs because the user message carries a timestamp. Read them
  from `results/`, not from this README.

The transformed result was stored once. It was not rewritten later.

## Verdicts

These verdicts are for the runtime the spike ran: the OpenClaw embedded runtime, which
is what an OpenClaw install runs for a non-Codex provider. The last column covers
OpenClaw's other execution paths, from source only. Under the EXP-000 rule, if those
paths count as "tool classes or code paths" of OpenClaw, the replacement rows (C1, C3,
C4) are **PARTIAL** runtime-wide.

| Capability | Embedded runtime verdict | API / source | Spike result | Other paths (source) |
|---|---|---|---|---|
| Intercept raw tool result (C1) | **PASS** | `api.registerAgentToolResultMiddleware`; `event.result` is the unmodified tool result (`tool-result-middleware.ts:441-444`, `extensions.ts:66-93`) | The side channel holds the raw `AAA LAYA_SENTINEL BBB` for plugin, exec, error and bridge | Codex harness: dynamic tools yes, native tools observe-only. CLI/ACP: no. Provider-hosted: nothing to intercept |
| Access tool name/args (C1) | **PASS** | `event.toolName`, `event.args` (the params after `before_tool_call` adjustment, `extensions.ts:81-89`), `toolCallId`, `isError`, `cwd` | Name and arguments recorded for every case | Tool Search: the outer name is `tool_call`, the inner tool is in `args.id` |
| Async processing (C2) | **PASS** | The handler is awaited (`tool-result-middleware.ts:444`), inside the awaited `afterToolCall` (`agent-loop.ts:1267`) | The awaited 50 ms `setTimeout` completed (`awaitedDelayMs` = 52) before the model request | No default timeout; an operator hook policy can bound it (`registry-registrars-tools-hooks.ts:175`) |
| Replace model-bound result (C3) | **PASS** | The returned `{result}` becomes `finalized.result` → `toolResult.content` (`agent-loop.ts:1278-1285`, `:1520-1541`) | The mock received only `AAA [FILTERED_BY_LAYA_SPIKE] BBB`; `raw_sentinel_anywhere_in_request=false` on both turns in 4/4 filtered classes | Codex native: **cannot** replace. CLI/ACP/provider-hosted: no hook |
| Before first context admission (C4) | **PASS** | `afterToolCall` → middleware runs before `emitToolResultMessage`, whose `message_end` pushes to `context.messages` (`agent-loop.ts:1256-1291` then `:1520-1541`; `agent-stream-response.ts:182-186`) | The raw result never reached any request; the persisted session holds the filtered result (b, c, d), which rules out rewrite-after-append | UI/channel partial updates and Tool Search nested records happen before it (limitations) |
| Preserve raw result separately (C5) | **PASS** | The middleware owns the side channel (a plain file here); the raw result is in `event.result` | 1 side-channel record per filtered case with the raw content and details | The runtime itself does not keep the raw result once `content` and `details` are both replaced |
| No historical rewrite (C6) | **PASS** | The middleware runs once per new tool result; the transcript stores its output | Turn-1 digests, without the runtime-appended tail, are an unchanged prefix of turn 2 in all 9 cases | — |
| No runtime fork (C7) | **PASS** | A public plugin API plus a manifest contract, loaded with `plugins.load.paths` and `plugins.entries.<id>.enabled` | The unmodified npm `openclaw@2026.9.6` was used; no upstream file changed | — |
| Pre-tool hook | **PARTIAL** (allow/deny/modify; no native redirect) | Typed hook `before_tool_call` → `{block, blockReason, params, requireApproval}` (`src/plugins/hook-before-tool-call-result.ts:14-35`, `src/plugins/hook-types.ts:1138`). It can rewrite `params`; nothing redirects the call to a different tool | Not implemented (investigation only) | The Codex harness relays `PreToolUse` for native tools (`native-hook-relay-events.ts`) |
| Pre-LLM hook | **PARTIAL** | `llm_input` is observe-only (returns `void`, `hook-types.ts:1078`). `before_prompt_build` edits the system prompt, prepended/appended context and tool allowlist (`hook-before-agent-start.types.ts:32-50`), at prompt build, not per model call. A `contextEngine` slot plugin's `assemble()` runs before each model call, but the current turn's pending user/tool messages are appended unmodified after the assembled history (`src/agents/embedded-agent-runner/tool-result-context-guard.ts:413-432`) | Not implemented | — |
| Potential LLM-turn bypass | **PARTIAL (STOP only)** | Middleware may return `result.terminate: true` (`types.ts:549-553`; forwarded at `extensions.ts:113`). The loop then ends without another provider call only if **every** result in the batch sets it (`agent-loop.ts:916-921`, `:322-327`). `before_agent_reply` → `{handled: true, reply}` skips the model only at the start of a run (`run-orchestrator.ts:504`), not between a tool result and the next model call | Not implemented, not spike-verified | — |

### Capability investigation (no implementation)

| Decision | Status | Evidence |
|---|---|---|
| Pre-tool: allow | SUPPORTED | `before_tool_call` returning nothing proceeds (`hook-before-tool-call-result.ts:14-35`) |
| Pre-tool: deny | SUPPORTED | `{block: true, blockReason}`. The call becomes a blocked tool result instead (`src/agents/agent-tools.before-tool-call.policy.ts:103`, `:363`) |
| Pre-tool: modify | SUPPORTED | `{params}` replaces the arguments. The middleware then sees the adjusted arguments (`extensions.ts:81-89`) |
| Pre-tool: redirect | NOT SUPPORTED | There is no field to substitute another tool; only a block with a reason or modified params |
| Pre-LLM: continue | SUPPORTED | Every hook is optional; with no return the request proceeds |
| Pre-LLM: modify | PARTIAL | `before_prompt_build` (prompt, context, tool allowlist), or a `contextEngine.assemble` that rewrites history. The current turn's pending tool exchange is host-owned (`tool-result-context-guard.ts:413-432`) |
| Pre-LLM: bypass | PARTIAL | `before_agent_reply` `{handled: true}` only before a run starts (`run-orchestrator.ts:504`). There is no per-model-call bypass; `llm_input` cannot return a decision (`hook-types.ts:1078`) |
| LLM-turn avoidance: STOP | PARTIAL | Middleware `terminate: true` on every result in the batch (`agent-loop.ts:916-921`) |
| LLM-turn avoidance: CONTINUE | NOT SUPPORTED | A non-terminating batch always leads to the next provider request (`agent-loop.ts:322-327`) |
| LLM-turn avoidance: RETRY | NOT SUPPORTED | No plugin API re-runs a tool through the runtime. A middleware could only do its own work in-process |

## Tool classes and limitations

| Class | Replaceable before admission? | Evidence |
|---|---|---|
| Plugin/dynamic tools (embedded) | **Yes** | spike `b_plugin` |
| Built-in tools such as `exec` (embedded) | **Yes** | spike `c_exec` |
| Error results from an executed tool that threw | **Yes**; the middleware sees `isError: true` and the runtime's error envelope | spike `d_error` |
| Tool Search bridge `tool_call` (the default when `tools.toolSearch` is unset) | **Yes, as the outer envelope only**. The middleware runs once, with `toolName: "tool_call"` and the inner tool in `args.id`. The inner result is not passed through the middleware separately | spike `e_toolsearch`; side channel has 1 record |
| Code Mode (`tool_search_code`, JS calling tools) | Expected: outer result only (same outer-tool shape) | Not run |
| MCP tools in the embedded runtime | Expected yes: they run through the same agent loop | Not run |
| Pre-execution outcomes (blocked, argument validation, unknown tool, steering skip, tool-loop intervention) | **No middleware call.** These are runtime-generated texts, not tool output | `agent-loop.ts:1265` (`executionStarted` guard), `:1406-1426`, `:1441-1487` |
| Images | Allowed in both directions (validated `type: "image"` blocks up to 5 M chars) | `tool-result-middleware.ts:44-51`; not run |
| Codex app-server harness, OpenClaw dynamic tools | Yes, by source: runtime `"codex"` middleware before the result goes back to Codex | `extensions/codex/src/app-server/dynamic-tools.ts:596-604`; not run (needs Codex and OpenAI auth) |
| Codex app-server harness, Codex-native tools (shell, patch, native web search) | **No, observe-only.** Upstream comment: "codex-rs PostToolUse hooks cannot replace tool_response … a transformed result reaches only after_tool_call observers, never the model" | `src/agents/harness/native-hook-relay-events.ts:241-283` (comment at `:251-253`) |
| CLI backends (external agent CLIs) and ACP agents | **No.** Tools execute inside the external process; OpenClaw only fires the observe-only `after_tool_call` | `src/agents/cli-runner/execute-events.ts:159`; no middleware call in `src/agents/cli-runner` |
| Provider-hosted tools, such as OpenAI Responses `web_search` injected into the payload | **No.** The provider runs them server-side; there is no tool result in OpenClaw to intercept | `src/agents/codex-native-web-search-core.ts:169-217` |

Further limitations:
- **`details` must be replaced as well.** `details` is not sent to the model: in
  `c_exec_content_only` the mock saw no sentinel. It is persisted, though. For `exec` it
  carries `aggregated`, the raw output. A filter that replaces only `content` leaves the
  raw output in the session database.
- **Tool Search nested records.** The inner call is written to the transcript as an
  `openclaw.nested-tool.v1` record with `excludeFromContext: true`, and to trajectory
  events. This happens before the outer middleware runs
  (`src/agents/embedded-agent-runner/run/attempt-tool-search-executor.ts:66-98`). It did
  not reach the model on either turn, but the raw text is persisted.
- **Live progress.** `tool_execution_update` partial results stream to UI and channel
  progress before the middleware runs (`agent-loop.ts:1096-1113`). They are not model
  context.
- **Model provider only.** The mock was an OpenAI-compatible Chat Completions provider.
  An official OpenAI endpoint may route to the Codex harness instead
  (`docs/providers/openai/runtimes.md`), and there Codex-native tool results are not
  replaceable.
- **Activation.** A non-bundled plugin must be explicitly enabled to register middleware.
  OpenClaw also warns that it cannot verify the provenance of a `load.paths` plugin; the
  plugin still loads.
- **Observe-only hook.** The `after_tool_call` plugin hook runs after finalization. It
  sees the filtered result and cannot change it
  (`embedded-agent-subscribe.handlers.tools.completion.ts:615-639`).

## Runtime fork needed?

**No.** The replacement uses only the public plugin API and an unmodified npm release.
Tool results that the API cannot replace (Codex-native tools, CLI/ACP backends and
provider-hosted tools) would need upstream changes in codex-rs or in those external
agents. A fork of OpenClaw would not be enough.

## Reproduce

```bash
./run.sh              # all 9 cases, about 2 minutes after the first install
./run.sh b_plugin     # one case
```

Requirements: Node 24 (mise), npm, `python3`, `curl`, and a free port 18081.
`LAYA_SPIKE_SCRATCH` overrides the scratch directory. The script only writes there and to
`results/`.
