// EXP-000 filter: agent tool-result middleware that replaces the sentinel result.
//
// Every middleware call is recorded on the side channel (LAYA_SPIKE_SIDE_CHANNEL) with the
// tool name, arguments and raw result. A result carrying the sentinel is replaced after an
// awaited ~50 ms delay that stands in for local_decision(...).
// LAYA_SPIKE_FILTER_MODE=content-only keeps the original `details` (a leak probe).
import { appendFile } from "node:fs/promises";

const RAW = "LAYA_SENTINEL";
const FILTERED_TEXT = "AAA [FILTERED_BY_LAYA_SPIKE] BBB";

export default {
  id: "laya-spike-filter",
  name: "EXP-000 tool-result filter",
  description: "Research spike: replaces sentinel tool results through agent tool-result middleware.",
  register(api) {
    api.registerAgentToolResultMiddleware(
      async (event, ctx) => {
        const started = performance.now();
        await new Promise((resolve) => setTimeout(resolve, 50));
        const delayMs = performance.now() - started;
        const raw = JSON.stringify(event.result);
        const replace = raw.includes(RAW);
        const sideChannel = process.env.LAYA_SPIKE_SIDE_CHANNEL;
        if (sideChannel) {
          await appendFile(
            sideChannel,
            JSON.stringify({
              runtime: ctx.runtime,
              toolName: event.toolName,
              toolCallId: event.toolCallId,
              args: event.args,
              isError: event.isError ?? false,
              rawContent: event.result.content,
              rawDetails: event.result.details ?? null,
              awaitedDelayMs: Math.round(delayMs),
              replaced: replace,
            }) + "\n",
          );
        }
        if (!replace) {
          return undefined;
        }
        const keepDetails = process.env.LAYA_SPIKE_FILTER_MODE === "content-only";
        return {
          result: {
            content: [{ type: "text", text: FILTERED_TEXT }],
            details: keepDetails ? event.result.details : { layaSpike: "filtered" },
          },
        };
      },
      { runtimes: ["openclaw"] },
    );
  },
};
