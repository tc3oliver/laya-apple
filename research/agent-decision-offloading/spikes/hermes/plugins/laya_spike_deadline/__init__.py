"""EXP-000 spike: fail closed inside the plugin instead of relying on Hermes.

Hermes admits the ORIGINAL result when a ``transform_tool_result`` callback exceeds
``plugins.hook_callback_timeout`` or raises. This callback never lets that happen: it
races its local decision against its own deadline, set well under the Hermes timeout,
and returns a fixed fallback string when the decision is late or fails.
"""

import asyncio
import json
import os
import tempfile
import threading
import time

RAW_MARKER = "LAYA_SENTINEL"
FILTERED = "AAA [FILTERED_BY_LAYA_SPIKE] BBB"
FALLBACK = "[LAYA_SPIKE_DEADLINE_FALLBACK]"
# The run sets plugins.hook_callback_timeout to 1.0 s for this case.
DEADLINE_S = float(os.environ.get("LAYA_SPIKE_DEADLINE_S", "0.2"))
DECISION_S = float(os.environ.get("LAYA_SPIKE_DECISION_S", "2.0"))


def _side_channel_path():
    return os.environ.get("LAYA_SPIKE_SIDECHANNEL") or os.path.join(
        tempfile.gettempdir(), "laya_spike_sidechannel.jsonl")


async def _slow_local_decision(result):
    await asyncio.sleep(DECISION_S)  # deliberately slower than the deadline
    return FILTERED


async def _on_transform_tool_result(tool_name="", args=None, result=None, **kwargs):
    started = time.monotonic()
    try:
        replacement, outcome = await asyncio.wait_for(_slow_local_decision(result), DEADLINE_S), "decided"
    except asyncio.TimeoutError:
        replacement, outcome = FALLBACK, "deadline_fallback"
    except Exception as exc:  # any decision failure also fails closed
        replacement, outcome = FALLBACK, f"error_fallback:{type(exc).__name__}"
    record = {
        "event": "transform_tool_result",
        "t": time.time(),
        "tool_name": tool_name,
        "args": args,
        "raw_output": result,
        "replacement": replacement,
        "outcome": outcome,
        "deadline_s": DEADLINE_S,
        "decision_s": DECISION_S,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
        "thread": threading.current_thread().name,
    }
    with open(_side_channel_path(), "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return replacement


def register(ctx):
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
