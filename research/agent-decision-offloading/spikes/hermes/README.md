# EXP-000 spike: Hermes Agent

Can a Hermes Agent plugin intercept a tool result, process it asynchronously, and
replace it before the result first enters the Main LLM's context, without forking
Hermes? The criteria are fixed in
[`EXP-000-integration-feasibility.md`](../../EXP-000-integration-feasibility.md).

**Short answer:** yes, for every tool result that goes back to the model as a
tool-role message, through the `transform_tool_result` plugin hook. Hermes needs no
fork. The caveats that matter are listed under [Limitations](#limitations):
- **Hermes fails open** when a hook times out or raises. A plugin can close that gap
  itself with its own deadline and a safe fallback, which the `plugin_deadline_fallback`
  case shows.
- **Background terminal output** reaches the model as a notification, not a tool
  result. `transform_terminal_output` covers it before it is first added to the
  conversation, which the `background_terminal` case shows.
- **An exception on the concurrent path** is admitted as `Error executing tool …`
  with no `transform_tool_result`. A `tool_execution` middleware in the filter plugin
  contains it, which the `concurrent_exception_guard` case shows.
- **Background `delegate_task` summaries** remain uncovered.

**Numbers in this README.** Run-specific values (timings, row counts, log lines) are
named by their field in `results/summary.json` or `results/<case>.*.jsonl`, all from
one full `run.sh` run. Where a value is quoted, it is copied from those files.

## Upstream version

| Field | Value |
|---|---|
| Repository | <https://github.com/NousResearch/hermes-agent> (the official NousResearch repo, not a fork; default branch `main`) |
| Commit read and run | `7de8728cba339065329f141cf92686bf06d2c171` (`main`, committed 2026-09-24) |
| Version | `pyproject.toml` `0.21.4`; `hermes --version` reports `Hermes Agent v0.21.4 (2026.9.21) · upstream 7de8728c` |
| Latest release | `v2026.9.21` ("Hermes Agent v0.21.4"), tag commit `d337b736aa1e8ebecfab043842d13e4a2d2f48a3` |
| Date read | 2026-09-24 |
| Install | `uv sync --frozen --python 3.12` in the pinned checkout; the checkout stayed unmodified (`git status` clean after every run) |

All `file:line` references below are to that commit.

## How to reproduce

```bash
research/agent-decision-offloading/spikes/hermes/run.sh            # all cases
research/agent-decision-offloading/spikes/hermes/run.sh plugin_tool # one case
```

`run.sh` clones and pins Hermes under `$LAYA_SPIKE_SCRATCH` (default
`~/Developer/scratch/agent-offloading/hermes`) and installs it with uv. For each case it
does the following:
- creates a fresh `HERMES_HOME`;
- copies the plugins into `HERMES_HOME/plugins/` and turns them on with
  `hermes plugins enable`;
- starts the mock Main LLM on `127.0.0.1:18082`;
- runs `hermes chat -Q -q …`, then runs a second user turn with `--resume <session>`;
- stops the mock.

Each run writes to a new `runs/<timestamp>/` directory in scratch. The script never
deletes earlier runs, and `summarize.py` reads the newest run of each case.

Hermes runs under `env -i` with a throwaway `HOME`. It never sees `~/.hermes`, a real
`.env` or a provider key; the only key it sees is a dummy. The model config is
`provider: custom`, `base_url: http://127.0.0.1:18082/v1`. The mock's raw request log,
the per-case Hermes homes and the hook's side-channel file stay in scratch.
`summarize.py` writes only derived data to `results/`.

| File | Purpose |
|---|---|
| `plugins/laya_sentinel_tool/` | Plugin tools `laya_sentinel` (returns `AAA LAYA_SENTINEL BBB`) and `laya_sentinel_raise` (raises with that text) |
| `plugins/laya_spike_filter/` | The spike. An `async def` `transform_tool_result` callback, observe-only `pre_tool_call`, `post_tool_call` and `pre_llm_call`, and a `tool_execution` middleware that turns any exception from `next_call` into `[LAYA_SPIKE_EXCEPTION_FALLBACK]` (setting `exception_guard`, default on) |
| `plugins/laya_spike_fault/` | Test-only fault injector for the concurrent exception cases (see [below](#concurrent-exception-path)) |
| `plugins/laya_spike_terminal_filter/` | An `async def` `transform_terminal_output` callback, same 50 ms wait and side channel. Used only in `background_terminal` |
| `plugins/laya_spike_deadline/` | A `transform_tool_result` callback that races a deliberately slow decision (2.0 s) against its own 0.2 s deadline and returns `[LAYA_SPIKE_DEADLINE_FALLBACK]` on expiry or error. Used only in `plugin_deadline_fallback` |
| `../common/mock_llm_parallel.py` | Wraps `common/mock_llm.py` (unchanged) and returns **two** tool calls. The common mock always returns one, so it cannot trigger Hermes's concurrent batch |
| `run.sh`, `summarize.py` | Reproduction and evidence summary |
| `results/<case>.evidence.jsonl` | The mock's committable evidence |
| `results/<case>.sidechannel.jsonl` | The hook's raw-result side channel (the spike's own data; scratch paths replaced) |
| `results/summary.json` | Per-case summary: booleans, counts, digests, and a scan of each Hermes home for raw-sentinel hits |

## The real lifecycle (source)

The CLI agent loop is synchronous. One assistant turn with tool calls goes through
these steps:

1. **Dispatch choice.** `run_agent.py:1333-1346`:
   - one call runs sequentially;
   - several calls are split by `_plan_tool_batch_segments`
     (`agent/tool_dispatch_helpers.py:164`) into concurrent runs of parallel-safe tools
     and sequential barriers.

   Parallel-safe tools are the read-only built-ins at
   `agent/tool_dispatch_helpers.py:31-45`, plus MCP servers that opt in. Plugin tools and
   `terminal` are always sequential.
2. **Pre-tool.** `pre_tool_call` runs once per call.
   - Directives are parsed at `hermes_cli/plugins.py:1854-1906`:
     - `block` with a message, where the message becomes the tool result;
     - `approve`, which escalates to a human;
     - `modify`, whose returned args are shallow-merged.
   - A block is turned into an error result at `model_tools.py:776-788`, and the tool
     does not run. A timeout or exception fails closed as a block.
3. **Execution.** Registry tools cover built-ins, plugin tools and MCP tools. MCP tools
   are registered into the same registry (`tools/mcp_tool_registration.py:347`). They run
   through:
   - `model_tools.handle_function_call`;
   - `_execute_tool`;
   - `registry.dispatch`, which catches handler exceptions and returns
     `{"error": …}` (`tools/registry.py:887-915`).
4. **Raw result, then post and transform.** In `model_tools.py:955-959`:

   ```python
   result = _execute_tool(...)
   _emit(result, duration_ms=duration_ms)          # post_tool_call
   return _apply_transform_tool_result_hook(function_name, function_args, result, duration_ms, ids)
   ```

   `_apply_transform_tool_result_hook` (`model_tools.py:848-865`) calls
   `invoke_hook("transform_tool_result", tool_name=, args=, result=, tool_call_id=, …, status=, error_type=, error_message=)`
   and returns `next((r for r in hook_results if isinstance(r, str)), result)`. So:
   - the first `str` return replaces the result;
   - a non-`str` return, or an error, leaves the original in place (fail-open).
5. **Inline and agent-level tools.** These bypass the registry: todo, memory,
   session_search, memory-provider tools, context-engine tools and `delegate_task`.
   - They get the same helper through `agent/inline_tool_executors.py:61-83`.
   - The sequential path calls it at `agent/tool_executor.py:1744-1749`, unless the
     dispatch has already fired the hook (`transform_applied=True` for registry tools,
     `:1662`).
   - The concurrent path calls it at `agent/agent_runtime_helpers.py:2418`.
6. **Admission.** `_commit_tool_result` (`agent/tool_executor.py:1024-1114`) receives
   the already-transformed value. In order, it:
   - appends guardrail notices (`run_agent.py:1282-1312`);
   - spills large results to disk through `maybe_persist_tool_result` (`:1080`);
   - builds the tool message (`make_tool_result_message`,
     `agent/tool_dispatch_helpers.py:400-422`);
   - **`messages.append(tool_message)` (`:1101`)**;
   - flushes to the session DB (`:1102`).

   The concurrent batch commits in the original call order (`:1467-1495`) after all
   workers finish.
7. **Next model request.** It is built from `messages`:
   - `llm_request` middleware can rewrite the provider kwargs for that request only
     (`agent/turn_api_request.py:141-143`);
   - the `pre_api_request` observer fires (`:52-63`);
   - the call runs inside `llm_execution` middleware (`agent/turn_api_call.py:122-138`).

   `pre_llm_call` is **not** per request. It fires once per user turn, before the loop
   (`agent/turn_context.py:745-798`, called at `:1122`). Its return is context text that
   is added to the current user message at API time only (`:1230`).

**Hook ordering, measured by the spike.** The observers are in `results/summary.json`
under `hook_invocations`:
- **Sequential path** (plugin tool, terminal, error): `pre_llm_call` →
  `pre_tool_call` → `transform_tool_result` → `post_tool_call`, and `post_tool_call`
  already sees the **transformed** result. The inner post hook is suppressed
  (`agent/tool_executor.py:1636`), and the executor emits it after
  `handle_function_call` has transformed the result (`:1743`).
- **Concurrent path:** `pre_tool_call` ×2 → `post_tool_call` ×2 with the **raw**
  result → `transform_tool_result` ×2.

The hooks reference (`website/docs/user-guide/features/hooks.md:453`) says transform
runs "after `post_tool_call`". That is true only on the concurrent path. It does not
change admission, which is always after the transform.

**Is the hook sync or async?** `invoke_hook` is synchronous
(`hermes_cli/plugins_dispatch.py:209-243`). `transform_tool_result` is one of the
bounded hooks (`:42-46`), so each callback runs on a daemon worker thread
`hermes-hook-<name>`, with the caller blocking up to `plugins.hook_callback_timeout`
(default 30 s, `:153`, `:269-355`).
- **`async def` callbacks.** If a callback returns a coroutine, `_invoke_hook_callback`
  (`:199-207`) hands it to `resolve_plugin_command_result`
  (`hermes_cli/plugins.py:2075-2106`). That runs `asyncio.run(coro)` on the worker
  thread, which has no running loop. So Hermes really does await the callback to
  completion, on a fresh event loop per call, not on an agent loop.
- **The native async dispatcher.** `ainvoke_hook` (`plugins_dispatch.py:483-519`) is
  used only by the gateway's inbound `pre_gateway_dispatch`
  (`gateway/run_inbound.py:75`), never for tool hooks.
- **Timeouts and errors.** On timeout the worker is abandoned, the callback is skipped
  and logged, and it is suppressed for 60 s (`:340-352`). An exception is logged
  (`:239-240`). In both cases `transform_tool_result` **fails open**: the original
  result is admitted.

### Which tool paths reach `transform_tool_result`

| Path | Transform applied? | Evidence |
|---|---|---|
| Sequential registry tool: built-in, plugin, MCP | Yes, inside `handle_function_call` | Source `model_tools.py:959`. Spike: `plugin_tool`, `builtin_terminal` |
| Concurrent batch of parallel-safe tools | Yes, per call, on the worker thread before the ordered commit | Source `agent_runtime_helpers.py:2363-2440`, `tool_executor.py:1279`. Spike: `parallel_read_file`. Each hook call awaited ~52 ms and the two finished well under 1 ms apart, so they overlapped (`summary.json` → `cases.parallel_read_file.parallel_timing`) |
| Deferred tools via the `tool_call` bridge (Hermes's **default** for plugin and MCP tools) | Yes. The bridge re-dispatches the real tool through `handle_function_call`, and the hook sees the real name | Source `model_tools.py:706-749`, `:906-919`, `tools/tool_search.py:150-165`. Spike: `plugin_tool_bridge` |
| Connector batch through the bridge | Per entry: each entry runs `handle_function_call`. The assembled batch JSON is not transformed again | Source `tools/connectors/dispatch.py:21-60`. Not run |
| Handler raised; the registry returns an error result | Yes, with `status="error"` | Source `tools/registry.py:905-915`. Spike: `plugin_tool_error` |
| Inline and agent-level tools, memory and context-engine tools, `delegate_task` handle | Yes, through `apply_transform_tool_result` | Source `tool_executor.py:1744-1749`, `agent_runtime_helpers.py:2418`. Not run |
| Large results spilled to disk | The spill runs **after** the transform, on the transformed value | Source `tool_executor.py:1074-1086` |
| Exception escaping `handle_function_call` outside the registry, for example in middleware | **No.** Error text is returned without the transform | `model_tools.py:961-966` |
| Exception escaping a concurrent worker | **No `transform_tool_result`.** It is admitted as `Error executing tool '<name>': <exc>`. A plugin `tool_execution` middleware can contain it (see [Concurrent exception path](#concurrent-exception-path)) | `tool_executor.py:1303-1306`. Spike: `concurrent_exception_control`, `concurrent_exception_guard` |
| Blocked by `pre_tool_call`, timed out, cancelled, invalid args | No, but the tool never produced output. The result text is Hermes's own | `model_tools.py:787`, `tool_executor.py:1742`, `:1446-1463` |
| Background terminal processes (heartbeat, watch-match, completion notifications) | **No `transform_tool_result`.** The output re-enters as a user-role notification message. **`transform_terminal_output` does cover it**, and it runs before the event is queued (see below) | `tools/process_registry.py:1605-1623` (completion), `:680-697` (heartbeat), `:770-782` (watch match), `:2513-2533` (`_redact_process_result` → `transform_process_output` → the hook); formatting in `tools/process_registry_notifications.py:401-440`. Spike: `background_control`, `background_terminal` |
| Top-level `delegate_task` subagent results | **No.** Model delegations run in the background, and the child's summary re-enters as an `[ASYNC DELEGATION COMPLETE …]` message. The child's own tool results do go through the hook, because hooks are process-wide | `run_agent.py:1359-1372`, `tools/process_registry_notifications.py:263-288` |

### Background terminal output

`terminal` with `background: true, notify_on_complete: true` returns straight away.
The model gets `{"output": "Background process started", "session_id": …}` as a normal
tool result, which does go through `transform_tool_result`. The process's real output
reaches the model later, by a separate route:
1. **Completion event.** When the process exits, `ProcessRegistry` builds a
   `completion` event, calls `_redact_process_result(notification)`, and only then
   calls `completion_queue.put` (`tools/process_registry.py:1605-1623`). Heartbeat and
   watch-match events do the same (`:696-697`, `:781-782`).
2. **The hook.** `_redact_process_result` runs `transform_process_output`, which invokes
   `transform_terminal_output`, on the `output` and `output_preview` fields. Secret
   redaction runs after it (`:2513-2533`).
3. **Delivery.** A consumer drains the queue and formats each event with
   `format_process_notification` as `[IMPORTANT: Background process … Output: …]`.
   - In `hermes chat -Q -q`, `continue_quiet_notify_completions`
     (`hermes_cli/quiet_single_query.py:192`, called from `hermes_cli/cli_single_query.py:250`)
     lingers for the process and runs the text as a **follow-up user turn**.
   - The interactive CLI, TUI, gateway and bot DMs drain the same queue from their own
     loops.

The transform runs in the registry before the event is queued, so every surface sees
the transformed output. That is the first point where the output could enter any
conversation. It is also a different hook from `transform_tool_result`: a plugin
needs both. Registering `transform_terminal_output` also rewrites **foreground**
terminal output, before `transform_tool_result` sees it
(`tools/terminal_tool_result.py:136-146`).

## What the Main LLM actually saw

Quoted from `results/*.evidence.jsonl`. The mock records a tool-result turn on turn 1
and again on the resumed turn 2.

| Case | Filter | Tool message(s) the mock received | `raw_sentinel_anywhere_in_request` | Chat requests with the raw sentinel anywhere | C6 prefix identical |
|---|---|---|---|---|---|
| `negative_control` | off | `AAA LAYA_SENTINEL BBB` | `true`, `true` | 2/4 | yes (4/4 messages) |
| `plugin_tool` | on | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | `false`, `false` | 0/4 | yes (4/4) |
| `plugin_tool_bridge` | on | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | `false`, `false` | 0/4 | yes (4/4) |
| `builtin_terminal` | on | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | `false`, `false` | 0/4 | yes (4/4) |
| `parallel_read_file` | on | `AAA [FILTERED_BY_LAYA_SPIKE] BBB`, `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | `false`, `false` | 0/4 | yes (5/5) |
| `plugin_tool_error` | on | `AAA [FILTERED_BY_LAYA_SPIKE] BBB` | `false`, `false` | 0/4 | yes (4/4) |
| `hook_timeout_failopen` (limitation probe) | on, `hook_callback_timeout: 0.01` | `AAA LAYA_SENTINEL BBB` | `true`, `true` | 2/4 | yes (4/4) |
| `plugin_deadline_fallback` | deadline plugin, `hook_callback_timeout: 1.0` | `[LAYA_SPIKE_DEADLINE_FALLBACK]` | `false`, `false` | 0/4 | yes (4/4) |

The background cases have three tool-result requests: turn 1's tool result, the
notification follow-up turn, and turn 2. The notification is a **user** message, so
the table below quotes it directly. It is taken from the first request that carried
it (`results/summary.json` → `background_notifications`).

| Case | Plugins | Tool message (all three requests) | Notification text the mock received | `raw_sentinel_anywhere_in_request` (tool-result turn, notification turn, turn 2) | Chat requests with the raw sentinel |
|---|---|---|---|---|---|
| `background_control` | `transform_tool_result` filter only | `{"output": "Background process started", "session_id": "proc_…", "pid": …, "exit_code": 0, "error": null, "notify_on_complete": true}` | `[IMPORTANT: Background process proc_… completed normally (exit code 0).`<br>`Command: sleep 1; printf 'AAA LAYA_%s BBB\n' SENTINEL`<br>`Output:`<br>`AAA LAYA_SENTINEL BBB`<br>`]` | `false`, **`true`**, **`true`** | 2/5 |
| `background_terminal` | `transform_tool_result` + `transform_terminal_output` filters | same | `[IMPORTANT: Background process proc_… completed normally (exit code 0).`<br>`Command: sleep 1; printf 'AAA LAYA_%s BBB\n' SENTINEL`<br>`Output:`<br>`AAA [FILTERED_BY_LAYA_SPIKE] BBB]` | `false`, `false`, `false` | **0/5** |

What the background cases show:
- **The control confirms the gap.** `transform_tool_result` saw only
  `Background process started` (`matched: false`). The raw output reached the model in
  the notification, and Hermes stored it in `state.db` (`session_db.raw_sentinel_rows` > 0).
- **`transform_terminal_output` closes it.** With the terminal filter, the hook ran on
  the reader side (`results/background_terminal.sidechannel.jsonl`: `raw_output`
  `"AAA LAYA_SENTINEL BBB\n"`, a ~52 ms `await_ms`, and thread
  `hermes-hook-_on_transform_terminal_outpu`, which Hermes cuts to 40 characters). It
  finished **before the first request
  that carried the notification**:
  `terminal_output_hook_ran_before_first_admission: true`, measured by comparing the
  hook's timestamp with the mock's timestamp for that request.
- **No leak anywhere in the conversation.** No chat request held the sentinel, and
  `state.db` has 0 raw rows and 3 filtered rows.
- **Earlier messages unchanged.** C6 prefixes are identical across all three
  consecutive requests.

**Deadline fallback.** The plugin wraps its decision in
`asyncio.wait_for(decision, 0.2)`. The decision sleeps 2.0 s, and Hermes's own hook
timeout is set to 1.0 s, so without the plugin deadline this would fail open like
`hook_timeout_failopen`.
- The side channel shows `outcome: "deadline_fallback"` after about 200 ms
  (`results/plugin_deadline_fallback.sidechannel.jsonl`, `elapsed_ms`), with
  the raw `AAA LAYA_SENTINEL BBB` kept.
- Hermes logged no hook timeout: `summary.json` → `cases.plugin_deadline_fallback.agent_log.hook_timeouts.count`
  is 0, compared with 1 in `hook_timeout_failopen`.
- The mock received only `[LAYA_SPIKE_DEADLINE_FALLBACK]`, and 0/4 requests held the
  sentinel.

Notes on the cases:
- **Request counts.** "Chat requests" counts every `/chat/completions` request in the
  raw log. Each case has four:
  - turn 1's tool-call request;
  - turn 1's tool-result request;
  - one auxiliary request with no tools and only `[system, user]` (session titling);
  - turn 2.

  The background cases have a fifth request: the notification follow-up turn.

  The auxiliary request never carried the sentinel. The mock also saw Ollama-style
  `/api/show` probes, which carry no messages.
- **The terminal case uses `printf 'AAA LAYA_%s BBB\n' SENTINEL`.** The output is
  exactly `AAA LAYA_SENTINEL BBB`, but the literal never appears in the model-visible
  tool-call arguments. With `echo AAA LAYA_SENTINEL BBB`, the assistant message would
  contain the sentinel, and `raw_sentinel_anywhere_in_request` would be true whatever
  the hook did.
- **What the hook saw raw** (side channel):

  | Tool | Raw result |
  |---|---|
  | `laya_sentinel` | `AAA LAYA_SENTINEL BBB` |
  | `terminal` | `{"output": "AAA LAYA_SENTINEL BBB", "exit_code": 0, "error": null}` |
  | `read_file` | `{"content": "1\|AAA LAYA_SENTINEL BBB", …}` |
  | `laya_sentinel_raise` | `{"error": "[TOOL_ERROR] Tool execution failed: RuntimeError: AAA LAYA_SENTINEL BBB"}` |

  The whole result string is replaced.
- **Hermes's own session store** (`summary.json` → `cases.<case>.session_db`). In the
  negative control, `state.db` holds the raw sentinel (`raw_sentinel_rows` > 0). In
  every filtered case `raw_sentinel_rows` is 0:
  - the `transform_tool_result` and background cases store the filtered marker
    instead (`filtered_rows` > 0);
  - `plugin_deadline_fallback` and `concurrent_exception_guard` store their fallback
    text instead, so their `filtered_rows` is 0 too.

  So the raw result never entered Hermes's persisted transcript either, which supports
  C4.

## Concurrent exception path

On the concurrent path, `_ConcurrentBatch._dispatch_worker` catches any `Exception`
from the call and turns it into the tool result (`agent/tool_executor.py:1303-1306`):

```python
except Exception as tool_error:
    result = f"Error executing tool '{ref.name}': {tool_error}"
```

That text is committed as the tool message. `transform_tool_result` never runs for it,
because the transform lives inside the dispatch that raised.

**What can raise there.** Hermes's middleware chain (`hermes_cli/middleware.py:161-222`)
isolates exceptions raised by a middleware callback itself:
- a callback that raises before calling `next_call` is skipped, and the next layer runs
  (`:212`);
- one that raises after `next_call` succeeded has the downstream result kept (`:208`);
- only an exception from **below** the chain propagates (`:202-203`, `:210`).

`tool_request` middleware is isolated the same way (`hermes_cli/plugins_dispatch.py:591-602`).
Registry tools catch their own handler exceptions (`tools/registry.py:905-915`). So what
reaches line 1303 is an exception from:
- an inline executor. On the concurrent path the only parallel-safe inline tool is
  `session_search`; see `agent/inline_tool_executors.py:285-301` and
  `agent/tool_dispatch_helpers.py:31-45`;
- Hermes's own dispatch code between the chain and the tool;
- the optional NeMo Relay wrapper, outside the chain (see below).

**How the cases trigger it.** `session_search` handles bad arguments without raising in
every shape I tried. So the test-only plugin `laya_spike_fault` injects a fault
through the public middleware API:
- its `tool_execution` middleware calls `next_call` for the call with `query: "fault"`,
  with a `query` value whose `strip()` raises
  `RuntimeError("local index read failed: AAA LAYA_SENTINEL BBB")`;
- the sentinel is assembled at run time, so the model-visible arguments are just
  `{"query": "fault"}`;
- the other call, `{"query": "alpha"}`, is untouched.

The control's traceback confirms the source (`summary.json` →
`cases.concurrent_exception_control.agent_log.concurrent_invoke_tool_raised`). The
exception is raised in `tools/session_search_tool.py:604`, inside the inline executor
(`agent/inline_tool_executors.py:127` ← `agent/agent_runtime_helpers.py:2410`). It
travels up through both middleware frames to `_dispatch_worker` (`tool_executor.py:1276`),
where line 1306 logs `_invoke_tool raised for session_search: …`.

| Case | Filter plugin | Tool messages the mock received (both requests) | `raw_sentinel_anywhere_in_request` | Chat requests with the raw sentinel |
|---|---|---|---|---|
| `concurrent_exception_control` | `transform_tool_result` on, exception guard **off** | `{"success": true, "mode": "discover", "query": "alpha", …}` and **`Error executing tool 'session_search': local index read failed: AAA LAYA_SENTINEL BBB`** | **`true`**, **`true`** | 2/4 |
| `concurrent_exception_guard` | `transform_tool_result` on, exception guard **on** | `{"success": true, "mode": "discover", "query": "alpha", …}` and `[LAYA_SPIKE_EXCEPTION_FALLBACK]` | `false`, `false` | 0/4 |

In the guard case the side channel has a `tool_execution_exception` record with
`raw_exception: "local index read failed: AAA LAYA_SENTINEL BBB"` and the fallback as
its replacement. Hermes logged no `_invoke_tool raised` line, because the exception
never reached `_dispatch_worker`. `state.db` and every file in the Hermes home are free
of the sentinel (`hermes_home_files_with_raw_sentinel: []`). In the control, the
sentinel is in `state.db`, `agent.log` and `errors.log`.

**Does ordering let the guard wrap the raising layer?**
- **Order is load order.** Middleware runs in registration order, and the first
  registered is outermost (`middleware.py:166`, `:170-172`). Registration order is
  plugin load order: dependencies first, then alphabetical (`hermes_cli/plugins_manifest.py:221-260`).
- **No priority.** `register_middleware(kind, callback)` has no priority argument
  (`hermes_cli/plugins.py:916-921`), so a plugin cannot make itself outermost.
- **In these cases the guard was inner.** `laya_spike_fault` sorts before
  `laya_spike_filter`, so the fault injector was the **outer** layer.
- **Outermost is not required.** The raise comes from the tool below every middleware
  layer. The guard is the lowest layer, so it catches the exception before any outer
  layer sees it. An outer layer then gets the fallback string as a normal result. It
  cannot turn that back into a raw-text exception, because the chain drops a callback's
  own exceptions (the rules above).

**Does it wrap inline executors on the concurrent path?** Yes. The chain's terminal
call (`tool_executor.py:779-782`) runs `_dispatch_authorized_once`, which runs the
pre-tool hooks and guardrails and then `execute` (`:720`). `execute` is
`agent._invoke_tool(..., skip_tool_execution_middleware=True)` (`:1279`), which reaches
the inline executor (`agent_runtime_helpers.py:2405-2410`). The spike's exception
came from exactly that inline executor.

**Exception sources the guard cannot wrap:**
- **NeMo Relay.** `relay_tools.execute` (`tool_executor.py:787`; `agent/relay_tools.py:16-23`)
  wraps the whole pipeline outside the middleware chain. With NeMo Relay managed
  execution enabled, a Relay failure reaches line 1303 unguarded. It is off unless
  configured. Not tested.
- **`BaseException` subclasses** other than `KeyboardInterrupt`. Neither the chain nor
  line 1303 catches them. The worker dies, and Hermes admits its own text,
  `Error executing tool '<name>': thread did not return a result`
  (`tool_executor.py:1446-1463`). No raw data is in that text.
- **Exceptions raised outside the middleware chain.** Inside `handle_function_call`,
  `model_tools.py:961-966` catches exceptions and **returns** an error string without
  `transform_tool_result`.
  - Registry and connector dispatch both run as the terminal of the
    `tool_execution` middleware chain (`model_tools.py:834-844`). The guard therefore
    catches their exceptions before they reach `:961`.
  - What still reaches `:961` is an exception raised outside the chain, for example in
    `_pre_dispatch_guards`. That text comes from the runtime, not from the tool.
  - Not tested.

## Verdicts

| Capability | Verdict | API | Source | Spike result |
|---|---|---|---|---|
| Intercept raw tool result (C1) | **PASS** | `transform_tool_result` hook, `result` kwarg; for background terminal output, `transform_terminal_output`, `output` kwarg | `model_tools.py:848-865`, `:955-959`; `agent/inline_tool_executors.py:61-83`; `tools/process_registry.py:2513-2533` | The side channel holds the raw output for the plugin, built-in terminal, parallel `read_file`, bridged, error and background-process cases |
| Access tool name/args (C1) | **PASS** | Hook kwargs `tool_name`, `args`, plus `tool_call_id`, `session_id`, `turn_id`, `status` | `model_tools.py:859-861`; `hooks.md:453` | The side channel records the name and args, for example `terminal` with `{"command": "printf …"}`. The bridge case gives the real name `laya_sentinel`, not `tool_call` |
| Async processing (C2) | **PASS** (see limitations 1 and 2) | `async def` callback; Hermes resolves the coroutine with `asyncio.run` on the hook worker thread and blocks until it finishes or the hook timeout expires | `plugins_dispatch.py:199-207`, `:269-355`; `plugins.py:2075-2106` | `await_ms` between 52.0 and 52.2 in the committed side channels, on thread `hermes-hook-_on_transform_tool_result`, with a separate event loop per call. The replacement came from the awaited coroutine. `plugin_deadline_fallback`: `asyncio.wait_for` cancelled a 2.0 s decision at 0.2 s inside the hook |
| Replace model-bound result (C3) | **PASS** | First `str` return replaces the result | `model_tools.py:862` | Every filtered case: tool message `AAA [FILTERED_BY_LAYA_SPIKE] BBB`, `raw_sentinel_anywhere_in_request=false`, and 0/4 chat requests contain the sentinel anywhere. `background_terminal`: the notification reads `Output: AAA [FILTERED_BY_LAYA_SPIKE] BBB`, and 0/5 requests contain the sentinel |
| Before first context admission (C4) | **PASS** | The transform runs inside dispatch, before `_commit_tool_result` | Transform `model_tools.py:959` comes before `messages.append` (`tool_executor.py:1101`) and the DB flush (`:1102`) | Session `state.db`: 0 raw rows with the filter, 8 without. Not an append-then-rewrite: the raw text is in no request and in no persisted row. Background: `transform_terminal_output` runs before `completion_queue.put` (`process_registry.py:1622-1623`), and it finished before the first request that carried the notification |
| Preserve raw result separately (C5) | **PASS** | The plugin writes the raw result from the hook's `result` kwarg to its own side channel | Hook payload `model_tools.py:859-861` | `results/<case>.sidechannel.jsonl` holds `raw_output` while the model got the filtered text. Hermes keeps no raw copy itself |
| No historical rewrite (C6) | **PASS** | Nothing to call: the transform only touches the new result | Append-only `messages` (`tool_executor.py:1101`); `pre_llm_call` context is added to the current user message at API time only (`turn_context.py:1230`) | Turn 2 (`--resume`) prefix digests equal turn 1 for every message (system, user, assistant, tool) in all cases. The background cases are also identical across the notification turn |
| No runtime fork (C7) | **PASS** | Directory plugin `HERMES_HOME/plugins/<name>/` (`plugin.yaml` + `register(ctx)`), turned on with `hermes plugins enable` | `hermes_cli/plugins.py:456-486` (`register_tool`), `ctx.register_hook` | The pinned checkout is unmodified (`git status` clean); only the plugins and config were added in the isolated home |
| Pre-tool hook | **PARTIAL**. Allow, deny and modify exist, but the pre-registered decision set also includes **redirect**, which Hermes has no native directive for (see the capability table) | `pre_tool_call` → `block` / `approve` / `modify` | `plugins.py:1854-1906`; `model_tools.py:776-788` | The observer fired before every tool, before the transform, in all cases. No directive was exercised |
| Pre-LLM hook | **PARTIAL** | `pre_llm_call` (once per user turn, can only add context); `llm_request` middleware (per request, can replace kwargs) | `turn_context.py:745-798`; `turn_api_request.py:141-143` | `pre_llm_call` fired once per user turn, and **not** before the post-tool model call of the same turn. Per-request control needs `llm_request` middleware, which was not exercised |
| Potential LLM-turn bypass | **UNKNOWN** | `llm_execution` middleware may return "any provider response" without calling `next_call` | `hermes_cli/middleware.py:170-201`; `developer-guide/middleware.md:54` | Not exercised. Whether the loop accepts a synthesized response object has not been tested |

### Capability investigation (source only, not implemented)

| Capability | Status | Evidence |
|---|---|---|
| Pre-tool **allow** | SUPPORTED | `pre_tool_call` returns `None` (`plugins.py:1877-1906`) |
| Pre-tool **deny** | SUPPORTED | `{"action": "block", "message": …}`. The message becomes the tool result, the tool does not run, and the model is still called next (`model_tools.py:786-787`) |
| Pre-tool **modify** | SUPPORTED | `{"action": "modify", "args": {…}}`, shallow-merged (`plugins.py:1884-1889`). `tool_request` middleware can replace args wholesale (`middleware.py:102-132`) |
| Pre-tool **redirect** to another tool | PARTIAL | No directive changes the tool name. `tool_execution` middleware can skip `next_call` and return any result (`middleware.py:161-222`; docs "Wrap or replace the actual tool call"), and `ctx.dispatch_tool` can call another tool. That is a replacement, not a native redirect |
| Pre-LLM **continue** | SUPPORTED | `pre_llm_call` returns nothing |
| Pre-LLM **modify** | PARTIAL | `pre_llm_call` can only add context to the user message, once per turn. `llm_request` middleware can replace the provider kwargs per request, request-only (`turn_api_request.py:141-143`). A context engine's `select_context` can replace the request message list (`agent/context_engine.py:122-140`) |
| Pre-LLM **bypass** | PARTIAL | `llm_execution` middleware can return a response without calling `next_call` (`middleware.py:199-201`). Whether a synthesized response survives normalization is UNKNOWN |
| LLM-turn avoidance **RETRY** (re-run the tool without the LLM) | NOT SUPPORTED natively | `next_call` in execution middleware is single-use and raises on a second call (`middleware.py:179-191`). A transform hook could call `ctx.dispatch_tool` itself, but that is a workaround, not a lifecycle feature |
| LLM-turn avoidance **CONTINUE** | UNKNOWN | Would need an `llm_execution` middleware to synthesize a tool-call response instead of calling the provider. Not tested |
| LLM-turn avoidance **STOP** | UNKNOWN | Would need an `llm_execution` middleware to synthesize a final text response. The tool guardrail `halt` (`run_agent.py:1303-1310`) is internal, not a plugin directive. `pre_verify` only runs after a final response |

## Limitations

1. **Fail-open.** If the callback exceeds `plugins.hook_callback_timeout` (default
   30 s) or raises, Hermes logs it and admits the **original** result
   (`plugins_dispatch.py:239-240`, `:340-352`; `model_tools.py:862-864`). The
   `hook_timeout_failopen` probe shows this. The mock received `AAA LAYA_SENTINEL BBB`,
   and Hermes logged the timeout
   (`summary.json` → `cases.hook_timeout_failopen.agent_log.hook_timeouts.first_line`).
   After a timeout the callback is also suppressed for 60 s. Hermes has no
   fail-closed option for this hook; only `pre_tool_call` fails closed.

   **A plugin can prevent this itself.** `plugin_deadline_fallback` puts its own
   deadline well under `hook_callback_timeout`, catches its own errors, and always
   returns a string. The model got only `[LAYA_SPIKE_DEADLINE_FALLBACK]`, and Hermes
   logged no timeout. The code that stays fail-open is everything outside the
   callback's `try`:
   - Hermes failing to start the worker thread (`plugins_dispatch.py:332-339`);
   - the 60 s suppression window after an earlier real timeout (`:283-289`);
   - the cap of 3 abandoned workers (`:296-302`);
   - an exception inside Hermes's own dispatch, which `model_tools.py:863-864` swallows.

   None of these is reached while every callback finishes inside its own deadline.
   The same applies to `transform_terminal_output`, which has the same bounded, fail-open
   dispatch.
2. **Async is emulated, not native.** The coroutine runs under `asyncio.run` on a
   per-call worker thread while the tool loop blocks. It cannot share an event loop or
   loop-bound clients with the agent, because there is no agent loop in the CLI. Each
   call creates and tears down a loop. An abandoned (timed-out) coroutine's late
   result is discarded.
3. **Out-of-band channels skip `transform_tool_result`.**
   - **Background terminal processes.** Heartbeat, watch-match and completion output
     re-enter as user-role notification messages. `background_control` shows the raw
     output reaching the model when only `transform_tool_result` is registered.
     `background_terminal` shows that also registering `transform_terminal_output`
     keeps it out, because that hook runs before the event is queued. Only the
     completion event was run. Heartbeat and watch-match events go through the same
     `_redact_process_result` call; this is from source, not tested.
   - **Background `delegate_task` summaries.** They re-enter as
     `[ASYNC DELEGATION COMPLETE …]` messages with no transform hook. The child's own
     tool results are transformed. The summary is model-written text, not a raw tool
     result, but no hook sees it before admission. Not tested.

   Full coverage needs both hooks and a policy for the delegation summary.
4. **Two exception paths skip the transform.**
   - **Concurrent worker** (`agent/tool_executor.py:1303-1306`). `concurrent_exception_control`
     shows the exception text reaching the model. `concurrent_exception_guard` shows a
     plugin `tool_execution` middleware containing it, even as the inner layer. Still
     uncovered: NeMo Relay failures and `BaseException` subclasses. The latter carry
     no raw data. See [Concurrent exception path](#concurrent-exception-path).
   - **`model_tools.py:961-966`** returns an error string for an exception raised
     outside the middleware chain, for example in the pre-dispatch guards. That string
     is not transformed. Registry and connector exceptions are inside the chain, and the
     guard catches them. Not tested.
5. **First string wins.** With several `transform_tool_result` plugins, only the first
   non-`None` string is used. The rest are ignored, not chained (`model_tools.py:862`).
6. **The replacement is not always the final wire text.** For `web_extract`,
   `web_search`, `browser_*` and `mcp_*` tools, Hermes wraps the (transformed) content
   in untrusted-data delimiters when it is at least 32 characters
   (`agent/tool_dispatch_helpers.py:436-438`, `:515-524`). Guardrail notices can be
   appended (`run_agent.py:1282-1312`). No raw data is reintroduced, but the tool
   message is not byte-identical to the hook's return for those tools. Not tested.
7. **Raw results are still visible elsewhere on the machine:**
   - `post_tool_call` observers get the raw result on the concurrent path. Measured:
     `result_has_raw_sentinel: true` in `parallel_read_file`. That includes Hermes's
     bundled observability plugins such as Langfuse.
   - The registry logs handler exception text to `HERMES_HOME/logs/agent.log` and
     `errors.log` (`tools/registry.py:907`). Measured: both files hold the sentinel in
     `plugin_tool_error`.
   - Background processes write their untransformed output to
     `HERMES_HOME/logs/process-results/<proc>.json` (`tools/process_registry_results.py:30`).
     Measured: that file holds the sentinel in `background_terminal` even though the
     model and `state.db` saw only the filtered text.

   None of these are model-bound.
8. **Plugin tools are deferred by default.** Plugin and MCP tools sit behind the
   `tool_search`/`tool_describe`/`tool_call` bridge unless
   `tools.tool_search.enabled: off`. The spike covers both: `plugin_tool` with the
   bridge off, `plugin_tool_bridge` with the default on.
9. **Scope of the evidence.** Tested:
   - CLI `hermes chat -Q` with a custom OpenAI-compatible provider;
   - streaming Chat Completions;
   - sequential, concurrent and bridge paths;
   - one background completion notification in quiet one-shot mode.

   Not run:
   - the gateway, TUI and ACP surfaces;
   - MCP servers;
   - memory-provider and context-engine tools;
   - the Anthropic and Responses wire formats.

   The source shows those surfaces call the same executor, but no spike covers them.

## Runtime fork needed?

**No.** Everything runs through directory plugins and `hermes plugins enable` against
an unmodified checkout at `7de8728c`.

Under EXP-000's pre-registered interpretation, C1, C3 and C4 are **PARTIAL** because of
the `delegate_task` summaries, and the rest PASS. For ordinary tool results, all seven
criteria C1–C7 are **PASS**. That is a post-hoc finding of EXP-000, confirmed by the
pre-registered [EXP-000B](../../EXP-000B-ordinary-tool-results.md). The three follow-up
gaps are closed:
- **Background terminal output** is covered, with the same properties, by also
  registering `transform_terminal_output` (`background_terminal`).
- **Hermes's fail-open** is avoided by a plugin-side deadline and fallback
  (`plugin_deadline_fallback`).
- **Concurrent-path exceptions** are contained by a plugin `tool_execution`
  middleware (`concurrent_exception_guard`). It works whatever the plugin load order.

What remains:
- **Background `delegate_task` summaries** re-enter without any hook (limitation 3).
- **Uncontained exception and error-string sources** (limitation 4). These are NeMo
  Relay failures, which are optional and off by default, and error strings returned by
  `model_tools.py:961-966` for exceptions raised outside the middleware chain. Neither
  was tested.
- **The plugin must be written defensively.** A plugin without its own deadline still
  fails open (`hook_timeout_failopen`).
