"""EXP-000 spike: replace terminal output, including background-process output.

Background-process output never passes through ``transform_tool_result``: it re-enters
the conversation as a notification message. Hermes runs ``transform_terminal_output`` on
that output before it queues the notification (``tools/process_registry.py``
``_redact_process_result``). The callback is ``async def`` and is resolved the same way
as the tool-result hook: ``asyncio.run`` on a hook worker thread.
"""

import asyncio
import json
import os
import tempfile
import threading
import time

RAW_MARKER = "LAYA_SENTINEL"
FILTERED = "AAA [FILTERED_BY_LAYA_SPIKE] BBB"


def _side_channel_path():
    return os.environ.get("LAYA_SPIKE_SIDECHANNEL") or os.path.join(
        tempfile.gettempdir(), "laya_spike_sidechannel.jsonl")


async def _on_transform_terminal_output(command="", output="", returncode=None, **kwargs):
    started = time.monotonic()
    matched = RAW_MARKER in (output or "")
    if matched:
        await asyncio.sleep(0.05)  # stand-in for local_decision(...)
    record = {
        "event": "transform_terminal_output",
        "t": time.time(),
        "command": command,
        "raw_output": output,
        "returncode": returncode,
        "matched": matched,
        "replacement": FILTERED if matched else None,
        "tool_call_id": kwargs.get("tool_call_id"),
        "await_ms": round((time.monotonic() - started) * 1000, 1),
        "thread": threading.current_thread().name,
    }
    with open(_side_channel_path(), "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return FILTERED if matched else None


def register(ctx):
    ctx.register_hook("transform_terminal_output", _on_transform_terminal_output)
