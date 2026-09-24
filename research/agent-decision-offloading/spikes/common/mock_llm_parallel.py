"""Two-call variant of ``spikes/common/mock_llm.py`` for the parallel-batch cases (Hermes, Pi).

The common mock always answers with exactly one tool call, so it cannot make Hermes run
a concurrent batch. This wrapper reuses its handler unchanged (same logging, evidence
records and sentinel checks) and only widens the tool-call reply to two calls: one with
``--args`` and one with ``--second-args``. It adds no other behaviour.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mock_llm  # noqa: E402

SECOND_ARGS = "{}"


class ParallelHandler(mock_llm.Handler):
    def _message(self, reply):
        if "tool_call" not in reply:
            return super()._message(reply)
        name = reply["tool_call"]["name"]
        calls = [
            {"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
             "function": {"name": name, "arguments": arguments}}
            for arguments in (reply["tool_call"]["arguments"], SECOND_ARGS)
        ]
        return {"role": "assistant", "content": None, "tool_calls": calls}, "tool_calls"

    def _stream(self, req, reply):
        message, finish = self._message(reply)
        if "tool_calls" not in message:
            return super()._stream(req, reply)
        base = {"id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion.chunk",
                "created": int(time.time()), "model": req.get("model", "mock")}
        delta = {"role": "assistant", "content": None, "tool_calls": [
            {"index": i, "id": tc["id"], "type": "function", "function": tc["function"]}
            for i, tc in enumerate(message["tool_calls"])]}
        chunks = [
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        ]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()
        for c in chunks:
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    global SECOND_ARGS
    if "--second-args" in sys.argv:
        i = sys.argv.index("--second-args")
        SECOND_ARGS = sys.argv[i + 1]
        json.loads(SECOND_ARGS)
        del sys.argv[i:i + 2]
    mock_llm.Handler = ParallelHandler  # main() binds cfg to, and serves, mock_llm.Handler
    mock_llm.main()


if __name__ == "__main__":
    main()
