"""Scripted stand-in for the Main LLM, used by every EXP-000 sentinel spike.

Serves the OpenAI Chat Completions API (streaming and non-streaming) on loopback.
There is no model. The script is:

1. The request has no tool result yet: reply with one call to the sentinel tool.
2. The request carries a tool result: record exactly what the model would have seen,
   then reply with a final text message.

Every request body is appended to ``--raw-log``, which holds the agent's own system
prompt and tool definitions. That file is third-party text: keep it outside the repo
and never commit it. The ``--evidence`` file is the committable summary. It holds
only the tool-role message contents, the sentinel checks, and hashes of the earlier
messages, which the historical-context check compares across turns.

Usage:
    python mock_llm.py --port 18080 --tool laya_sentinel \
        --args '{}' --raw-log /tmp/x/raw.jsonl --evidence out/evidence.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RAW = "LAYA_SENTINEL"
FILTERED = "[FILTERED_BY_LAYA_SPIKE]"


def _text(content) -> str:
    """Flatten an OpenAI message ``content`` (a string or a list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return json.dumps(content)


def _digest(message: dict) -> str:
    return hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest()[:16]


class Handler(BaseHTTPRequestHandler):
    cfg: argparse.Namespace

    def log_message(self, *_):  # keep stdout for the summary lines
        pass

    def do_GET(self):  # /v1/models for runtimes that probe it
        body = json.dumps({"object": "list", "data": [{"id": "mock", "object": "model"}]})
        self._send(200, body.encode(), "application/json")

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        cfg = self.cfg
        with open(cfg.raw_log, "a") as f:
            f.write(json.dumps({"t": time.time(), "path": self.path, "body": req}) + "\n")

        messages = req.get("messages", [])
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        whole = json.dumps(req)
        tool_names = [t.get("function", {}).get("name") for t in req.get("tools", []) or []]

        if tool_msgs:
            record = {
                "t": time.time(),
                "path": self.path,
                "n_messages": len(messages),
                "message_roles": [m.get("role") for m in messages],
                "message_digests": [_digest(m) for m in messages],
                "tool_messages": [_text(m.get("content")) for m in tool_msgs],
                "raw_sentinel_anywhere_in_request": RAW in whole,
                "filtered_marker_anywhere_in_request": FILTERED in whole,
            }
            with open(cfg.evidence, "a") as f:
                f.write(json.dumps(record) + "\n")
            print("MOCK tool-result turn:", json.dumps(record["tool_messages"]), flush=True)
            reply = {"content": f"Observed tool result: {record['tool_messages'][-1]}"}
        elif cfg.tool in tool_names:
            reply = {"tool_call": {"name": cfg.tool, "arguments": cfg.args}}
        else:
            # An auxiliary request (title generation, summarisation) or a missing tool.
            print(f"MOCK no '{cfg.tool}' tool offered; tools={tool_names}", flush=True)
            reply = {"content": "ok"}

        if req.get("stream"):
            self._stream(req, reply)
        else:
            self._complete(req, reply)

    def _message(self, reply):
        if "tool_call" in reply:
            tc = reply["tool_call"]
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_" + uuid.uuid4().hex[:12],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                ],
            }, "tool_calls"
        return {"role": "assistant", "content": reply["content"]}, "stop"

    def _complete(self, req, reply):
        message, finish = self._message(reply)
        body = {
            "id": "chatcmpl-" + uuid.uuid4().hex[:12],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", "mock"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        self._send(200, json.dumps(body).encode(), "application/json")

    def _stream(self, req, reply):
        message, finish = self._message(reply)
        cid = "chatcmpl-" + uuid.uuid4().hex[:12]
        base = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": req.get("model", "mock")}
        if "tool_calls" in message:
            tc = message["tool_calls"][0]
            delta = {"role": "assistant", "content": None, "tool_calls": [
                {"index": 0, "id": tc["id"], "type": "function", "function": tc["function"]}]}
        else:
            delta = {"role": "assistant", "content": message["content"]}
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

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--tool", required=True, help="name of the tool the mock asks to call")
    p.add_argument("--args", default="{}", help="JSON string of the tool-call arguments")
    p.add_argument("--raw-log", required=True, help="full request log; never commit it")
    p.add_argument("--evidence", required=True, help="committable tool-result summary")
    cfg = p.parse_args()
    json.loads(cfg.args)
    for path in (cfg.raw_log, cfg.evidence):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Handler.cfg = cfg
    server = ThreadingHTTPServer(("127.0.0.1", cfg.port), Handler)
    print(f"MOCK listening on 127.0.0.1:{cfg.port}, tool={cfg.tool}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
