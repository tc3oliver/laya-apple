"""EXP-000 sentinel tools: ``laya_sentinel`` returns ``AAA LAYA_SENTINEL BBB``;
``laya_sentinel_raise`` raises with that text, for the error-result path."""

SENTINEL_OUTPUT = "AAA LAYA_SENTINEL BBB"

SCHEMA = {
    "name": "laya_sentinel",
    "description": "Return the EXP-000 sentinel string.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


RAISE_SCHEMA = {
    "name": "laya_sentinel_raise",
    "description": "Fail with the EXP-000 sentinel string as the error message.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _handler(args, **kwargs):
    return SENTINEL_OUTPUT


def _raise_handler(args, **kwargs):
    raise RuntimeError(SENTINEL_OUTPUT)


def register(ctx):
    ctx.register_tool(name="laya_sentinel", toolset="laya_spike", schema=SCHEMA, handler=_handler)
    ctx.register_tool(name="laya_sentinel_raise", toolset="laya_spike", schema=RAISE_SCHEMA,
                      handler=_raise_handler)
