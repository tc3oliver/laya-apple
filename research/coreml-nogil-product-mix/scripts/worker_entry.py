"""Process-worker entry used by run_config.py: `python -m laya_apple.executor`, unchanged, plus
the CPU time of the thread that runs each forward.

run_config.py starts every process worker of every configuration through this file (A, B, C
and D alike), so all of them pay the same two `time.thread_time_ns()` reads per forward. The
worker protocol, backend and warm-up are laya_apple.executor's own. When the worker exits
normally (its parent's close()), it writes
    $LAYA_NOGIL_CPU_DIR/<pid>.json = {"forwards": [[service_start_ns, service_end_ns, thread_cpu_ns], ...]}
"""

from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path

from laya_apple import executor

FORWARDS: list[tuple[int, int, int]] = []
_forward_timed = executor._forward_timed


def _timed(backend, rows):
    c0 = time.thread_time_ns()
    out = _forward_timed(backend, rows)
    FORWARDS.append((out[2], out[3], time.thread_time_ns() - c0))
    return out


def _dump():
    out = os.environ.get("LAYA_NOGIL_CPU_DIR")
    if out:
        Path(out, f"{os.getpid()}.json").write_text(json.dumps({"forwards": FORWARDS}))


if __name__ == "__main__":
    executor._forward_timed = _timed
    atexit.register(_dump)
    executor._main()
