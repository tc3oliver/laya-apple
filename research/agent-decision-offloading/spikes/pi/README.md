# EXP-000 spike: Pi

**Result: all seven criteria C1–C7 PASS for Pi 0.87.1, without a runtime fork.**

A `tool_result` extension handler is awaited inside the agent loop. It runs after the tool
executes and before Pi builds the `toolResult` message. That is before the message
enters agent state, before it is written to the session JSONL, and before any model
request. The spike used an extension tool, built-in `bash`, `bash` on its error path, and
a parallel batch of two extension-tool calls. In each case the mock Main LLM received only `AAA [FILTERED_BY_LAYA_SPIKE] BBB`. The raw
output went to a side-channel file, and the second turn replayed the earlier messages
byte-identically.

The criteria and verdict rules are in
[`../../EXP-000-integration-feasibility.md`](../../EXP-000-integration-feasibility.md).

## Upstream version

| Item | Value |
|---|---|
| Repository | `github.com/earendil-works/pi`. `github.com/badlogic/pi-mono` redirects here. The old npm package `@mariozechner/pi-coding-agent` (last version 0.73.1) is deprecated with the message "please use @earendil-works/pi-coding-agent" |
| Source read at | tag `v0.87.1`, commit `f07218c4d4bbc12bef056a7058c3dd49dfe41abe` (2026-09-22) |
| `main` when cloned | `b45597504eeaba1f11a9920a1d1048c361ed4b8e`, 16 commits after the tag. None of them touches `agent-loop.ts`, `agent-session.ts` or `runner.ts`, and the `types.ts` changes only add provider-stream and model-config types |
| Packages run | `@earendil-works/pi-coding-agent` 0.87.1, `@earendil-works/pi-agent-core` 0.87.1, `@earendil-works/pi-ai` 0.87.1. `npm` `latest` was 0.87.1 |
| Date read | 2026-09-24 |
| Node | 24.21.0 (mise) |

All `file:line` references below are at `v0.87.1`. The prefixes are `agent/` =
`packages/agent/src/`, `session/` = `packages/coding-agent/src/core/`, and `ai/` =
`packages/ai/src/`.

## Real lifecycle of a tool result

The CLI's `AgentSession` drives `Agent` (`sdk.ts:366`). `Agent` runs `runAgentLoop`
(`agent/agent.ts:434`). The coding agent does not use the `harness/` or the experimental
`pico3` runtime for this path.

1. **Pre-tool.** `prepareToolCall` validates the arguments and then awaits
   `config.beforeToolCall` (`agent/agent-loop.ts:719-750`). `AgentSession` installs this
   as the `tool_call` extension event (`session/agent-session.ts:530-549`). The event
   carries `toolName`, `toolCallId` and a mutable `input`. A handler can return
   `{block, reason, terminate}` (`session/extensions/types.ts:1217-1226`). If a handler
   throws, the tool is blocked (fail-closed).
2. **Execution.** `executePreparedToolCall` awaits `tool.execute(...)`
   (`agent/agent-loop.ts:773-814`). Streaming partial output is emitted as
   `tool_execution_update` events while the tool runs (`:786-798`). These events reach
   extensions, the UI and JSON/RPC stdout **before** the hook, but they are not
   model-bound and are not persisted. A thrown error becomes an `isError` result
   (`:804-810`) and still continues to step 3.
3. **Hook: `finalizeExecutedToolCall`** (`agent/agent-loop.ts:816-861`) awaits
   `config.afterToolCall` with `{toolCall, args, result, isError}`. It merges the
   returned `content` / `details` / `isError` / `usage` / `terminate`:

   ```ts
   content: afterResult.content ?? result.content,
   details: afterResult.details ?? result.details,
   ```

   `AgentSession` installs `afterToolCall` as the `tool_result` extension event
   (`session/agent-session.ts:551-584`). The event carries `toolName`, `toolCallId`,
   `input`, `content`, `details`, `isError` and `usage`. The runner awaits each handler in
   load order, and each handler sees the earlier handlers' changes
   (`session/extensions/runner.ts:1082-1132`). `AgentSession` forwards `content`,
   `details`, `isError` and `usage`, but **not `terminate`** (`agent-session.ts:578-583`,
   `types.ts:1241-1246`). If a `tool_result` handler throws, the runner logs the error
   and continues with the unmodified result (`runner.ts:1109-1118`). This is
   **fail-open**.
4. **Admission.** Only after step 3 does `createToolResultMessage` build the
   `toolResult` message from the finalized result (`agent/agent-loop.ts:880-893`). This
   is the only place that creates a `toolResult` message. `emitToolResultMessage` then
   emits `message_start` and `message_end` (`:895-898`). On `message_end`, `Agent` pushes
   the message into `state.messages` (`agent/agent.ts:571-574`) and then awaits its
   listeners (`:605-607`). The `AgentSession` listener runs the extension `message_end`
   handlers first (`agent-session.ts:918`, `:1093-1111`). Those handlers may still
   replace the message in place. The listener then persists the message with
   `sessionManager.appendMessage(event.message)` (`:922-941`). Finally the loop pushes
   the message into its working context (`agent/agent-loop.ts:273-276`).
5. **Next model request.** `prepareNextTurn` and `prepareRequest` rebuild the context
   from the session projection (`agent-session.ts:610-616`, `:693-697`). The request
   therefore contains what was persisted. `streamAssistantResponse` then runs
   `transformContext`, which is the per-request `context` / `context_with_system`
   extension event (`sdk.ts:390-394`, `runner.ts:1190-1251`). It also runs
   `convertToLlm`, and the provider serialises the tool message from `content` text
   only. `details` is not sent (`ai/api/openai-completions.ts:1398-1424`).
   `before_provider_request` can replace the raw payload (`sdk.ts:351-355`,
   `runner.ts:1253-1282`).
6. **Turn end.** `finishTurn` dispatches `turn_end`. A handler can return
   `continue: true`, which asks for one more **model** request
   (`agent-session.ts:675-685`, `agent/agent-loop.ts:285-313`). If every result in a
   batch has `terminate === true`, the inner loop stops without another model request
   (`agent/agent-loop.ts:271`, `:685-687`).

**Coverage.** Built-in and extension tools are wrapped into the same `AgentTool` list
(`agent-session.ts:3195-3196`) and share one execution path. An extension tool that
overrides a built-in by name also uses this path. Sequential batches call the hook per
call (`agent/agent-loop.ts:554-564`). Parallel is the default mode
(`agent/agent.ts:250`; the coding agent does not override it), and a batch goes sequential
only if a tool in it declares `executionMode: "sequential"` (`agent-loop.ts:512-519`).
Neither `laya_sentinel` nor the built-ins declare one. In a parallel batch, the
`tool_call` hooks run one after another in source order (`:593-601`). Each call's
execute and `tool_result` hook then run concurrently in its own closure (`:616-637`,
`:643-645`). The messages are created and emitted only after all of them finish, in
source order (`:646-651`). The spike confirms this in case (e). The following outcomes skip `afterToolCall` because no tool ran and their
text is generated by the runtime:
- tool not found (`:710-717`);
- argument-validation failure (`:764-770`);
- a `tool_call` block (`:739-749`);
- abort (`:732-757`, `:617-625`);
- a length-truncated assistant message (`:475-500`).

`before_agent_start` runs once per user prompt, before the loop starts
(`agent-session.ts:1700-1707`). It can change the system prompt and inject a custom
message. It is not a per-tool-result hook.

## What the Main LLM actually saw

The mock returned one tool call per case, except in (e). There,
`../common/mock_llm_parallel.py` (a wrapper around the common mock that only widens the
reply to two calls) returned two `laya_sentinel` calls in one assistant message. Evidence files: `results/<case>.evidence.jsonl`
(the mock's own summary), `results/<case>.side_channel.jsonl` (the extension's records)
and `results/summary.json` (the reduction by `summarize.py`).

| Case | Tool | Tool message the mock received | `raw_sentinel_anywhere_in_request` | Session `toolResult` content |
|---|---|---|---|---|
| (a) `control_bash_noext`: no extension | built-in `bash` | `"AAA LAYA_SENTINEL BBB\n"` | `true` | raw |
| (a) `control_tool_nofilter`: extension loaded, replacement off | `laya_sentinel` | `"AAA LAYA_SENTINEL BBB"` | `true` | raw |
| (b) `ext_tool` | `laya_sentinel` (extension) | `"AAA [FILTERED_BY_LAYA_SPIKE] BBB"` | **`false`** (both turns) | filtered |
| (c) `builtin_bash` | built-in `bash` | `"AAA [FILTERED_BY_LAYA_SPIKE] BBB"` | **`false`** (both turns) | filtered |
| (c′) `builtin_bash_literal`: `echo AAA LAYA_SENTINEL BBB` | built-in `bash` | `"AAA [FILTERED_BY_LAYA_SPIKE] BBB"` | `true`, see note | filtered |
| (d) `builtin_bash_error`: exit code 3 | built-in `bash` | `"AAA [FILTERED_BY_LAYA_SPIKE] BBB"` | **`false`** | filtered, `isError: true` |
| (e) control `control_parallel_nofilter`: 2 calls, replacement off | `laya_sentinel` ×2 | `["AAA LAYA_SENTINEL BBB", "AAA LAYA_SENTINEL BBB"]` | `true` | raw ×2 |
| (e) `parallel_ext_tool`: 2 calls in one batch | `laya_sentinel` ×2 | `["AAA [FILTERED_BY_LAYA_SPIKE] BBB", "AAA [FILTERED_BY_LAYA_SPIKE] BBB"]` | **`false`** (both turns) | filtered ×2 |

- **Note on (c′).** The literal command puts `LAYA_SENTINEL` into the assistant's own
  tool-call arguments, and the mock's whole-request check sees those arguments too. The
  tool message is still filtered.
- **Command used in (c) and (d).** These cases use
  `printf 'AAA LAYA_%s BBB\n' SENTINEL`. It prints the same output as (c′) but keeps the
  literal out of the arguments, so the whole-request check sees only the tool result.
- **Raw output the hook received** (from the side channel):
  - `laya_sentinel`: `"AAA LAYA_SENTINEL BBB"`;
  - `bash`: `"AAA LAYA_SENTINEL BBB\n"`;
  - `bash` error: `"AAA LAYA_SENTINEL BBB\n\n\nCommand exited with code 3"`, with
    `is_error: true`.

  Each record also has the tool name and arguments, `awaited_ms` of about 50, the 50 ms `setTimeout` measured with `Date.now()` (the exact values are in the side channel), and
  `tool_result_in_session_when_hook_ran: false`.
- **Ordering.** The records appear in this order:
  1. `tool_call_hook`;
  2. `tool_result_hook`;
  3. `message_end_tool_result`, which is already filtered and has
     `in_session_at_message_end: false`;
  4. `turn_end_tool_result`, which is filtered and has `in_session_at_turn_end: true`.
     This confirms that the session probe works;
  5. a `context_event` whose `tool_result_texts` is `["AAA [FILTERED_BY_LAYA_SPIKE] BBB"]`
     and `raw_present: false`.

  The mock's tool-result request arrives after the hook ends. In every filtered case,
  the `t` of the `tool_result_hook` record in `results/<case>.side_channel.jsonl` is
  earlier than the `t` of the first record in `results/<case>.evidence.jsonl`.
- **Parallel batch (e).** The two calls are `{"tag":"first","delay_ms":20}` and
  `{"tag":"second","delay_ms":0}`. The first call's `execute()` sleeps 20 ms, which is
  less than the 50 ms hook delay, so it finishes last while its hook still overlaps the
  other call's hook. From `results/summary.json` (`parallel_ext_tool.parallel`):
  - both `tool_call_hook` records have the same timestamp, in source order
    `first, second`;
  - the `tool_result` hooks overlapped (`hooks_overlap: true`):
    - `first` started before `second` ended, as the `start`/`end` values in
      `hook_intervals` show;
  - the hooks completed in the order `second, first`;
  - `message_end` came in the order `first, second`. Both messages were already
    filtered and neither was in the session yet;
  - in the mock's tool-result request, the assistant's tool-call order and the
    tool-message order are both `first, second` (`admitted_in_source_order: true`).
    `summarize.py` reads only the call ids from the raw log to derive these orders;
  - the negative control shows the same ordering, with the raw text in both messages.
  - **C6:** turn 2 has 7 messages, and its first 5 message digests equal turn 1's
    (`system, user, assistant, tool, tool`; `prefix_identical: true`).
- **C6.** The second user turn resumes the session with `-c` in a new process, so the
  history is reloaded from the session JSONL. In `ext_tool` and `builtin_bash`, turn 2
  has 6 messages (`system, user, assistant, tool, assistant, user`). Its first 4 message
  digests equal the turn-1 request's 4 digests (`prefix_identical: true`,
  `differing_indices: []`). The only additions are the runtime-appended assistant reply
  and the new user message. The hook did not run again in turn 2.
- **Other surfaces.** The only place the raw sentinel appeared in the filtered cases was
  one JSON-mode `tool_execution_update` event: bash's streaming partial output from
  step 2. The persisted session holds no raw sentinel in any filtered case, including in
  `details`.

## Capability verdicts

| Capability | Verdict | API / source | Spike |
|---|---|---|---|
| Intercept raw tool result (C1) | **PASS** | `pi.on("tool_result")` fed from `afterToolCall` (`agent/agent-loop.ts:827-839`, `agent-session.ts:551-564`) | Raw output captured for an extension tool, built-in `bash`, and `bash` erroring |
| Access tool name / args (C1) | **PASS** | `ToolResultEvent.toolName` / `input` / `toolCallId` (`types.ts:1040-1093`) | Side channel records the name and args for every case |
| Async processing (C2) | **PASS** | Runner `await handler(...)` (`runner.ts:1090`), loop `await config.afterToolCall` (`agent-loop.ts:829`) | `awaited_ms` of about 50, and the mock request comes after the hook ends. In the parallel batch, the two awaited hooks overlap and admission still waits for both |
| Replace model-bound result (C3) | **PASS** | Returned `content` replaces `result.content` (`agent-loop.ts:843`). The provider sends `content` only (`openai-completions.ts:1398-1424`) | Mock tool message is `AAA [FILTERED_BY_LAYA_SPIKE] BBB`, and `raw_sentinel_anywhere_in_request: false` in (b), (c), (d) and the parallel batch (e). The negative controls show `true` |
| Before first context admission (C4) | **PASS** | Hook in `finalizeExecutedToolCall` runs before `createToolResultMessage`, before the state push (`agent.ts:573`) and before `appendMessage` (`agent-session.ts:940`) | `tool_result_in_session_when_hook_ran: false`. `message_end` and every `context` event saw only filtered text |
| Preserve raw result separately (C5) | **PASS** | The extension owns the side channel. Pi persists only the returned content (`agent-session.ts:940`) | `results/*.side_channel.jsonl` holds the raw output, and the session holds only the filtered text |
| No historical rewrite (C6) | **PASS** | The hook runs once per executed call. Later requests re-project the persisted session (`agent-session.ts:610-616`) | Turn-2 prefix digests are identical in (b) and (c) (4/4) and in (e) (5/5) |
| No runtime fork (C7) | **PASS** | `--extension <file>` loaded through jiti. The npm package is pinned, and no core file is changed | `run.sh` installs the stock 0.87.1 |
| Pre-tool hook | **PARTIAL**: allow, deny and modify are available; there is no native redirect. The spike shows the hook fires, is awaited, and gets the name and args before execution | `tool_call` → `beforeToolCall` (`agent-loop.ts:722-750`, `agent-session.ts:530-549`) | `tool_call_hook` records come before execution. The spike did not exercise block or mutate |
| Pre-LLM hook | **PARTIAL**: continue and modify are available; bypass is NOT SUPPORTED. The spike shows the hook fires before every model request | `context` / `context_with_system` (`runner.ts:1190-1251`) and `before_provider_request` (`runner.ts:1253-1282`) | A `context_event` precedes each of the 2–3 requests per case. The spike did not exercise modify |
| Potential LLM-turn bypass | **PARTIAL** (source only, not spiked) | STOP: `tool_call` `{block, terminate}` or a tool's own `terminate: true` ends the loop without a model call when every result in the batch terminates (`agent-loop.ts:271`, `:685-687`, `:739-748`). `tool_result` cannot set `terminate` (`agent-session.ts:578-583`). No API produces an assistant turn without the model | Not implemented (investigation only) |

### Capability investigation (not implemented)

| Decision | Status | Evidence |
|---|---|---|
| Pre-tool **allow** | SUPPORTED | Return `undefined` from `tool_call`. Exercised by the spike |
| Pre-tool **deny** | SUPPORTED | `{block: true, reason}` becomes an `isError` result with the reason text (`agent-loop.ts:739-749`). A handler that throws also blocks (`agent-session.ts:543-547`) |
| Pre-tool **modify** | SUPPORTED | Mutate `event.input` in place. Pi does not re-validate it afterwards (`types.ts:1023-1028`) |
| Pre-tool **redirect** | PARTIAL | No reroute-to-another-tool result exists. An extension can register a tool with a built-in's name to replace its implementation (`examples/extensions/tool-override.ts`), or block and explain in the reason |
| Pre-LLM **continue** | SUPPORTED | Return nothing from `context` |
| Pre-LLM **modify** | SUPPORTED | `context` returns `messages`, which are request-local and not persisted (`runner.ts:1203-1207`). `context_with_system` owns the whole transcript. `before_provider_request` replaces the payload |
| Pre-LLM **bypass** | NOT SUPPORTED by lifecycle hooks | `context` and `before_provider_request` must return a request, and none of them can return a synthetic response. A `pi.registerProvider()` provider with a custom `streamSimple` could answer without the Main LLM, but only while that provider/model is selected. This was not tested (`docs/custom-provider.md`) |
| LLM-turn avoidance **STOP** | PARTIAL | After a local decision, only a tool the extension owns (or a same-name override) can return `terminate: true`. The `tool_result` hook cannot. Pre-tool `{block, terminate: true}` stops, but the result is an error result. `ctx.abort()` aborts the run. Whether the provider call is avoided or merely aborted was not tested: UNKNOWN |
| LLM-turn avoidance **CONTINUE** | NOT SUPPORTED without the model | `turn_end` / `agent_before_settle` `continue: true`, `sendMessage({triggerTurn})` and `sendUserMessage` (steer / followUp) all cause another model request (`agent-loop.ts:293-313`, `types.ts:1482-1496`) |
| LLM-turn avoidance **RETRY** | PARTIAL | No runtime retry-tool API exists. A `tool_result` handler can re-run work itself before returning, because it is async and awaited, and then admit only the final result. This was not tested |

## Limitations

- **Fail-open on handler error.** An exception in a `tool_result` handler is logged, and
  the result continues unmodified, **raw**, to the model (`runner.ts:1109-1118`). A
  production filter must catch its own errors and return a safe replacement.
- **`details` is kept unless it is overwritten.** Returning `details: undefined` keeps
  the tool's original `details` (`agent-loop.ts:844`), which is persisted in the session
  but not sent to the model. When bash output is truncated,
  `details.fullOutputPath` points to a temp file with the full raw output, and the
  original model-facing text names that path (`tools/bash.ts:321-338`). The spike
  outputs were too small to truncate. A filter must also replace `details` if raw data
  must not be retained, and the temp file remains on disk.
- **Streaming partial output is not filtered.** `tool_execution_update` carries raw
  partial output to extensions, the TUI and JSON/RPC stdout before the hook. The spike
  saw it in the `bash` JSON event stream. It is not model-bound and is not persisted.
- **Paths that bypass `tool_result`.** User-initiated `!` shell commands are persisted as
  `bashExecution` messages (`agent-session.ts:3514`, `:3546`). They enter the context
  without `tool_result`. The runtime's own error texts for non-executed calls also skip
  the hook (see Coverage). Extension messages (`sendMessage`) enter the context directly.
- **Hook order between extensions.** Handlers run in load order. An extension loaded
  earlier sees the raw result, and a later `tool_result` or `message_end` handler can
  change the content again. `message_end` can replace a `toolResult` message after the
  hook and before persistence.
- **Not spiked:**
  - parallel batches of built-in tools, or mixed built-in and extension tools. They
    share the parallel code path exercised in (e);
  - large or truncated bash outputs;
  - image content;
  - RPC and TUI modes;
  - the experimental `pico3` / `micro` runtime.
- **Concurrent handlers.** In a parallel batch the `tool_result` handlers for different
  calls run concurrently, so a filter with shared state must be safe under concurrency.
  Admission still waits for the slowest handler in the batch.
- **C6 across processes only.** Turn 2 ran as a new process resuming the session, not as
  a second prompt inside one process. The mock compares the model-visible prefix only.
- **Only one provider.** Only the OpenAI Chat Completions serializer was exercised.

**Runtime fork needed: no.**

## Reproduce

```sh
./run.sh   # uses $LAYA_SPIKE_SCRATCH or ~/Developer/scratch/agent-offloading/pi
```

`run.sh` does the following:
1. Installs `@earendil-works/pi-coding-agent@0.87.1` into `<scratch>/install`, without a
   global install.
2. Runs Pi with `env -i` and an isolated `HOME`, `PI_CODING_AGENT_DIR`, session dir and
   working dir, all under `<scratch>/runs/<timestamp>-<pid>`. It sets `PI_OFFLINE=1` and passes
   `--no-extensions --no-skills --no-prompt-templates --no-context-files`.
3. Uses a `models.json` provider `laya-mock` (`openai-completions`,
   `http://127.0.0.1:18083/v1`) with a dummy key.
4. Starts `../common/mock_llm.py`, or `../common/mock_llm_parallel.py` for (e), on port
   18083 for each case, then stops it.
5. Runs Pi in `--mode json`, then runs a second turn with `-c`.

**Hang protection.** Each case records its turns in `results/<case>.status.jsonl`, and
`summary.json` gives each case a `status`. The status is `ok` only if every turn exited
0 and sent at least one request to the mock; otherwise it is `timeout`, `error`,
`no_request` or `missing`. A case that is not `ok` never counts as a pass.
- **Stdin.** Pi's stdin is `/dev/null`. When stdin is not a TTY, Pi reads it to EOF
  before doing anything else (`main.ts:874-876`, `readPipedStdin` at `:79-96`). An
  inherited stdin pipe that never closes therefore blocks Pi forever, with no request,
  no events and no stderr.
- **Run directory.** Each run writes to a new `<scratch>/runs/<timestamp>-<pid>/`, and
  nothing is deleted.
- **Watchdog.** Every Pi invocation runs as
  `perl -e 'alarm shift; exec @ARGV' $PI_TIMEOUT node cli.js …`. The default timeout is
  60 s; override it with `LAYA_SPIKE_PI_TIMEOUT`. The pending `SIGALRM` survives the
  `exec` and kills Pi (exit 142, recorded as `timeout`). macOS has no `timeout` or
  `gtimeout`.
- **Mock start.** The script waits up to 5 s for `GET /v1/models`, and fails the run if
  the mock does not answer.
- **Port check.** Before each case, the script fails if something is already listening
  on 18083, rather than talking to a leftover mock.

**Root cause of the hung run on 2026-09-24 at 13:46.** Pi was running with empty events
and stderr, and the mock had no request.
- **Mechanism: confirmed.** A single probe in scratch ran Pi with stdin from an open pipe
  (`sleep 60 | pi …`). It sent no request and wrote no events or stderr until the 15 s
  alarm killed it (exit 142). The same command with `< /dev/null` finished in under
  1 s and sent 2 requests.
- **Link to that particular run: inferred, not proven.** The stdin of the hung run was
  not recorded.
- **Editing `run.sh` during the run: ruled out as the cause.** BSD `sed -i` writes a new
  file, which gets a new inode (checked with `stat`), so a running bash keeps reading
  the old file. An edit could also only change lines bash had not yet read. It cannot
  stop a Pi process that has already started from sending its request.

Do not edit `run.sh` while it is running regardless: an in-place write that keeps the
inode, as editor tools and `open(..., "w")` do, can corrupt what a running bash reads
next.

The raw mock log, the Pi event streams and the session files stay in scratch. They
contain Pi's system prompt and tool definitions, and are never committed.

## Files

- `laya-sentinel-extension.ts`: the extension. It contains the sentinel tool, the
  `tool_result` replacement and the ordering observers.
- `run.sh`: reproduces every case.
- `summarize.py`: reduces the scratch run to the committable summaries.
- `../common/mock_llm_parallel.py`: shared two-call wrapper around the common mock, used unchanged for (e).
- `results/*.evidence.jsonl`: the mock's evidence summaries.
- `results/*.side_channel.jsonl`: the extension's records.
- `results/summary.json`: the reduction.
