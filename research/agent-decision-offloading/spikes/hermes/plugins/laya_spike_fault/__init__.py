"""EXP-000 test-only fault injection for the concurrent exception path.

Hermes isolates exceptions raised by a middleware callback itself: the chain skips the
callback or keeps the downstream result (``hermes_cli/middleware.py`` ``call_at``). Only
an exception raised *below* the chain, by the tool dispatch, propagates. On the concurrent path
it reaches ``agent/tool_executor.py`` ``_dispatch_worker``, which turns it into
``"Error executing tool '<name>': <exc>"`` without ``transform_tool_result``.

So this plugin does not raise from its middleware. For the ``session_search`` call whose
query is ``fault``, it hands the dispatch a query value whose ``strip()`` raises. The
exception then comes from inside Hermes's inline ``session_search`` executor, the real
tool, and its message carries the sentinel. The sentinel is assembled at run time, so it
never appears in the model-visible tool-call arguments.
"""

FAULT_QUERY = "fault"


class _FaultingQuery(str):
    def strip(self, *args):
        raise RuntimeError("local index read failed: " + "AAA LAYA_" + "SENTINEL BBB")


def _inject_fault(tool_name=None, args=None, next_call=None, **kwargs):
    if tool_name == "session_search" and isinstance(args, dict) and args.get("query") == FAULT_QUERY:
        return next_call({**args, "query": _FaultingQuery(FAULT_QUERY)})
    return next_call(args)


def register(ctx):
    ctx.register_middleware("tool_execution", _inject_fault)
