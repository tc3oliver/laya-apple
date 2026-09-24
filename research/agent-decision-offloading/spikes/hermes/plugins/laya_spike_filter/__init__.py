"""EXP-000 spike: replace a tool result before Hermes appends it to the conversation.

The callback is ``async def``. Hermes does not await it on the agent's loop: its sync
hook dispatcher detects the coroutine and runs it with ``asyncio.run`` on a worker
thread while the tool loop blocks (``hermes_cli/plugins.py`` resolve_plugin_command_result).
The side-channel record says which thread and loop the coroutine actually ran on.

``pre_tool_call``, ``post_tool_call`` and ``pre_llm_call`` are registered as observers
only (they return ``None`` and decide nothing) so the side channel shows when each fires
relative to the transform.

A ``tool_execution`` middleware contains exceptions. An exception from the tool dispatch
skips ``transform_tool_result``, and on the concurrent path Hermes admits its text as the
tool result. The middleware catches any exception from ``next_call``, keeps the raw text
on the side channel, and returns a fixed fallback. Setting
``plugins.entries.laya_spike_filter.settings.exception_guard: false`` turns the catch off
(the negative control); the middleware then only passes the call through.
"""

import asyncio
import json
import os
import tempfile
import threading
import time

RAW_MARKER = "LAYA_SENTINEL"
FILTERED = "AAA [FILTERED_BY_LAYA_SPIKE] BBB"
EXCEPTION_FALLBACK = "[LAYA_SPIKE_EXCEPTION_FALLBACK]"


def _side_channel_path():
    return os.environ.get("LAYA_SPIKE_SIDECHANNEL") or os.path.join(
        tempfile.gettempdir(), "laya_spike_sidechannel.jsonl")


async def _local_decision(result):
    # Stand-in for local_decision(...): an artificial async wait, no model.
    await asyncio.sleep(0.05)
    return FILTERED


async def _on_transform_tool_result(tool_name="", args=None, result=None, **kwargs):
    started = time.monotonic()
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    matched = RAW_MARKER in text
    replacement = await _local_decision(result) if matched else None
    record = {
        "t": time.time(),
        "tool_name": tool_name,
        "args": args,
        "raw_output": result,
        "matched": matched,
        "replacement": replacement,
        "tool_call_id": kwargs.get("tool_call_id"),
        "status": kwargs.get("status"),
        "await_ms": round((time.monotonic() - started) * 1000, 1),
        "thread": threading.current_thread().name,
        "loop_id": id(asyncio.get_running_loop()),
    }
    _write({"event": "transform_tool_result", **record})
    return replacement  # None leaves non-sentinel results unchanged


def _write(record):
    with open(_side_channel_path(), "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _observer(event):
    def _observe(tool_name=None, args=None, result=None, **kwargs):
        rec = {"event": event, "t": time.time(), "tool_name": tool_name,
               "tool_call_id": kwargs.get("tool_call_id")}
        if event == "post_tool_call":
            rec["result_has_raw_sentinel"] = RAW_MARKER in str(result)
        if event == "pre_llm_call":
            history = kwargs.get("conversation_history") or []
            rec["history_roles"] = [m.get("role") for m in history if isinstance(m, dict)]
        _write(rec)
        return None
    return _observe


def _exception_guard(ctx):
    def _guard(tool_name=None, args=None, next_call=None, **kwargs):
        if ctx.get_config("exception_guard", True) is False:
            return next_call(args)
        try:
            return next_call(args)
        except Exception as exc:
            original = getattr(exc, "original", exc)  # the chain wraps downstream errors
            _write({"event": "tool_execution_exception", "t": time.time(), "tool_name": tool_name,
                    "args": args, "tool_call_id": kwargs.get("tool_call_id"),
                    "exception_type": type(original).__name__, "raw_exception": str(original),
                    "replacement": EXCEPTION_FALLBACK})
            return EXCEPTION_FALLBACK
    return _guard


def register(ctx):
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
    ctx.register_middleware("tool_execution", _exception_guard(ctx))
    for event in ("pre_tool_call", "post_tool_call", "pre_llm_call"):
        ctx.register_hook(event, _observer(event))
