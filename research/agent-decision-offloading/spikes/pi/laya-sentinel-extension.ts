/**
 * EXP-000 Pi sentinel spike. Loaded with `pi --extension <this file>`; no Pi source is patched.
 *
 * - Registers `laya_sentinel`, an extension tool whose raw output is `AAA LAYA_SENTINEL BBB`.
 * - `tool_result` handler: for any tool (built-in or extension) whose output carries the
 *   sentinel, await ~50 ms (stand-in for local_decision), write {tool, args, raw output} to
 *   the side-channel file, and return the filtered text as the model-bound content.
 * - Observers (never change anything) record `tool_call` firing, whether the tool-result
 *   message already existed in the session when the hook ran, and what `message_end`,
 *   `turn_end` and the per-request `context` event saw.
 *
 * Env: LAYA_SPIKE_SIDE_CHANNEL (jsonl path, required), LAYA_SPIKE_FILTER=0 disables the
 * replacement so the extension tool can serve as its own negative control.
 */
import { appendFileSync } from "node:fs";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const RAW = "LAYA_SENTINEL";
const FILTERED = "AAA [FILTERED_BY_LAYA_SPIKE] BBB";
const SIDE = process.env.LAYA_SPIKE_SIDE_CHANNEL ?? "";
const FILTER = process.env.LAYA_SPIKE_FILTER !== "0";

const record = (kind: string, data: Record<string, unknown>) => {
	if (SIDE) appendFileSync(SIDE, `${JSON.stringify({ kind, t: Date.now() / 1000, ...data })}\n`);
};
const textOf = (content: unknown) =>
	Array.isArray(content) ? content.map((p) => (p?.type === "text" ? p.text : "")).join("") : String(content ?? "");
const sessionHasToolResult = (ctx: any, toolCallId: string) =>
	ctx.sessionManager
		.getBranch()
		.some((e: any) => e.type === "message" && e.message?.role === "toolResult" && e.message.toolCallId === toolCallId);

const sentinelTool = defineTool({
	name: "laya_sentinel",
	label: "Laya sentinel",
	description: "EXP-000 sentinel tool. Returns a fixed marker string.",
	// Optional args exist only for the parallel case: `delay_ms` lets the first call of a
	// batch finish last, so admission order can be told apart from completion order.
	parameters: Type.Object({
		tag: Type.Optional(Type.String()),
		delay_ms: Type.Optional(Type.Number()),
	}),
	async execute(_id, params) {
		if (params.delay_ms) await new Promise((resolve) => setTimeout(resolve, params.delay_ms));
		return { content: [{ type: "text", text: "AAA LAYA_SENTINEL BBB" }], details: undefined };
	},
});

export default function (pi: ExtensionAPI) {
	pi.registerTool(sentinelTool);

	// Observer only: shows the pre-tool hook fires with name and args before execution.
	pi.on("tool_call", async (event) => {
		record("tool_call_hook", { tool_name: event.toolName, tool_call_id: event.toolCallId, args: event.input });
		return undefined;
	});

	pi.on("tool_result", async (event, ctx) => {
		const raw = textOf(event.content);
		if (!raw.includes(RAW)) return undefined;
		const inSessionBefore = sessionHasToolResult(ctx, event.toolCallId);
		const tStart = Date.now();
		await new Promise((resolve) => setTimeout(resolve, 50)); // stand-in for local_decision(...)
		record("tool_result_hook", {
			tool_name: event.toolName,
			tool_call_id: event.toolCallId,
			args: event.input,
			raw_output: raw,
			is_error: event.isError,
			hook_start: tStart / 1000,
			awaited_ms: Date.now() - tStart,
			tool_result_in_session_when_hook_ran: inSessionBefore,
			replaced: FILTER,
		});
		if (!FILTER) return undefined;
		return { content: [{ type: "text", text: FILTERED }] };
	});

	pi.on("message_end", async (event, ctx) => {
		const m = event.message as any;
		if (m.role !== "toolResult") return undefined;
		record("message_end_tool_result", {
			tool_call_id: m.toolCallId,
			text: textOf(m.content),
			is_error: m.isError,
			in_session_at_message_end: sessionHasToolResult(ctx, m.toolCallId),
		});
		return undefined;
	});

	// Validates the session probe above: by turn_end the same entries must be found.
	pi.on("turn_end", async (event, ctx) => {
		for (const m of event.toolResults) {
			record("turn_end_tool_result", {
				tool_call_id: m.toolCallId,
				text: textOf(m.content),
				in_session_at_turn_end: sessionHasToolResult(ctx, m.toolCallId),
			});
		}
		return undefined;
	});

	pi.on("context", async (event) => {
		const toolTexts = event.messages.filter((m: any) => m.role === "toolResult").map((m: any) => textOf(m.content));
		record("context_event", { tool_result_texts: toolTexts, raw_present: toolTexts.some((t) => t.includes(RAW)) });
		return undefined;
	});
}
