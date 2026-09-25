"""Count the Objective-C bridge crossings one call makes from Python (criteria.md, not gating).

Uses sys.monitoring (Python 3.12+) on the calling thread. A top-level crossing is a call made
from the forward's own Python code whose callable is:
  objc_send   a PyObjC message send (objc.native_selector): PyObjC releases the GIL around it;
  pool_push   the same, made inside objc.autorelease_pool.__enter__ (NSAutoreleasePool alloc,
              init), counted apart so the pool is reported separately from the forward's sends;
  ctypes      a ctypes foreign function (the predict shims, loaded with CDLL): released once;
  pool_pop    the exit of objc.autorelease_pool, which releases the pool (one release send
              made from `del`, which no CALL event shows).

Re-entries. While a crossing is in flight the GIL is released; if the native side calls back
into Python (a Python-backed Objective-C proxy such as OC_PythonArray / OC_PythonDictionary /
OC_PythonNumber handed to Core ML, or PyObjC converting an NSNumber for such a proxy), that
callback must take the GIL again. Every such episode is counted as a `reentry` (PY_START on this
thread while a crossing's native call is in flight, CALL/C_RETURN/C_RAISE tracking which native
calls are in flight), and any sends made inside it as `nested_send`, not as top-level sends.
So `total` is the GIL crossings the forward's own code makes, and `reentries` the extra GIL
acquisitions hidden inside them.

Implicit sends PyObjC makes with no Python call and no Python callback (a proxy's release when
it is freed, conversions done in C) are not seen, so the count is a lower bound.
Used at load time and in check_prebind.py, never inside a measured window.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import types
from collections import Counter

KINDS = ("objc_send", "ctypes", "pool_push", "pool_pop")
_NATIVE_CROSSINGS = ("objc_send", "ctypes", "pool_push")  # kinds whose call runs native code


def classify(fn, caller: str = "") -> str | None:
    """The crossing kind of calling `fn` from code named `caller` (a co_qualname), or None."""
    if type(fn).__name__ == "native_selector":
        return "pool_push" if caller == "autorelease_pool.__enter__" else "objc_send"
    if isinstance(fn, ctypes._CFuncPtr):
        return "ctypes"
    func = getattr(fn, "__func__", None)
    if func is not None and func.__name__ == "__exit__" and func.__qualname__ == "autorelease_pool.__exit__":
        return "pool_pop"
    return None


def _is_python_function(fn) -> bool:
    """sys.monitoring reports C_RETURN / C_RAISE for every callable that is not one of these."""
    return isinstance(fn, types.FunctionType) or (
        isinstance(fn, types.MethodType) and isinstance(fn.__func__, types.FunctionType)
    )


def _name(fn) -> str:
    name = getattr(fn, "selector", None) or getattr(fn, "__name__", type(fn).__name__)
    return name.decode() if isinstance(name, bytes) else str(name)


def count(fn, *args, **kw) -> tuple[object, dict]:
    """Run fn(*args, **kw) on this thread; return (its result, crossings by kind, the total,
    re-entries into Python from inside a crossing, and every counted call site)."""
    mon = sys.monitoring
    tool = next((i for i in range(6) if mon.get_tool(i) is None), None)
    if tool is None:
        raise RuntimeError("no free sys.monitoring tool id")
    mon.use_tool_id(tool, "laya-prebind-crossings")
    me = threading.get_ident()
    seen, sites = Counter(), Counter()
    native: list[tuple[object, str | None]] = []  # native calls in flight on this thread
    state = {"reentry_depth": 0, "reentries": 0, "nested_sends": 0}

    def in_crossing() -> bool:
        return any(k in _NATIVE_CROSSINGS for _, k in native)

    def where(code) -> str:
        return f"{code.co_filename.rsplit('/', 1)[-1]}:{code.co_qualname}"

    def on_call(code, offset, callable_, arg0):
        if threading.get_ident() != me:
            return
        kind = classify(callable_, code.co_qualname)
        if kind is not None:
            if state["reentry_depth"]:
                state["nested_sends"] += 1
                sites[f"nested {kind} {_name(callable_)} <- {where(code)}"] += 1
            else:
                seen[kind] += 1
                sites[f"{kind} {_name(callable_)} <- {where(code)}"] += 1
        if not _is_python_function(callable_):
            native.append((callable_, kind))

    def on_c_exit(code, offset, callable_, arg0):
        if threading.get_ident() != me:
            return
        for i in range(len(native) - 1, -1, -1):
            if native[i][0] is callable_:
                del native[i:]
                return

    def on_start(code, offset):
        if threading.get_ident() != me:
            return
        if state["reentry_depth"]:
            state["reentry_depth"] += 1
        elif in_crossing():
            state["reentry_depth"] = 1
            state["reentries"] += 1
            sites[f"reentry {where(code)}"] += 1

    def on_end(code, offset, value):
        if threading.get_ident() == me and state["reentry_depth"]:
            state["reentry_depth"] -= 1

    ev = mon.events
    handlers = {
        ev.CALL: on_call,
        ev.C_RETURN: on_c_exit,
        ev.C_RAISE: on_c_exit,
        ev.PY_START: on_start,
        ev.PY_RETURN: on_end,
        ev.PY_UNWIND: on_end,
    }
    for e, h in handlers.items():
        mon.register_callback(tool, e, h)
    mon.set_events(tool, ev.CALL | ev.C_RETURN | ev.C_RAISE | ev.PY_START | ev.PY_RETURN | ev.PY_UNWIND)
    try:
        result = fn(*args, **kw)
    finally:
        mon.set_events(tool, 0)
        for e in handlers:
            mon.register_callback(tool, e, None)
        mon.free_tool_id(tool)
    counts = {k: seen.get(k, 0) for k in KINDS}
    counts["total"] = sum(counts.values())
    counts["reentries"] = state["reentries"]
    counts["nested_sends"] = state["nested_sends"]
    counts["sites"] = dict(sorted(sites.items()))  # "kind selector <- file:caller": calls
    return result, counts
