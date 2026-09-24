// EXP-000 sentinel tool: a plugin (dynamic) tool whose raw result is the sentinel.
export default {
  id: "laya-sentinel-tool",
  name: "EXP-000 sentinel tool",
  description: "Research spike: a plugin tool that returns the EXP-000 sentinel string.",
  register(api) {
    api.registerTool({
      name: "laya_sentinel",
      label: "Laya sentinel",
      description: "Return the EXP-000 sentinel string.",
      parameters: { type: "object", properties: {}, additionalProperties: false },
      async execute() {
        return { content: [{ type: "text", text: "AAA LAYA_SENTINEL BBB" }], details: {} };
      },
    });
    // Error class: the tool throws, so the runtime builds an error tool result from the message.
    api.registerTool({
      name: "laya_sentinel_error",
      label: "Laya sentinel (error)",
      description: "Fail with the EXP-000 sentinel string as the error message.",
      parameters: { type: "object", properties: {}, additionalProperties: false },
      async execute() {
        throw new Error("AAA LAYA_SENTINEL BBB");
      },
    });
  },
};
