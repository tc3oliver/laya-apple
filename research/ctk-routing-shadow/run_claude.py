"""Claude arm: `claude -p --model haiku`, JSON-schema output, no tools, no session persistence.

Usage: run_claude.py SPLIT [LIMIT]. Resumable: ids already in the output file are skipped.
Calls ~/.local/bin/claude by absolute path with the ambient environment (no ANTHROPIC_* set here).
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
CLAUDE = os.path.expanduser("~/.local/bin/claude")
LABELS = ["lint-fix", "small-bugfix", "refactor", "docs", "needs-human", "needs-more-info"]
SYSTEM = (
    "You route software work requests. Reply only with the structured result. Labels: "
    "lint-fix (linter/formatter/style warning); small-bugfix (specific localized bug or failing test); "
    "refactor (restructure/rename, no behaviour change); docs (documentation or comments); "
    "needs-human (risky: security, production data, permissions, credentials, design decisions); "
    "needs-more-info (too vague to act on). Give confidence in [0,1]."
)
SCHEMA = json.dumps({"type": "object", "required": ["label", "confidence"], "additionalProperties": False,
                     "properties": {"label": {"type": "string", "enum": LABELS},
                                    "confidence": {"type": "number"}}})


def call(r):
    t0 = time.perf_counter()
    p = subprocess.run(
        [CLAUDE, "-p", "--model", "haiku", "--output-format", "json", "--json-schema", SCHEMA,
         "--system-prompt", SYSTEM, "--tools", "", "--no-session-persistence", "--disable-slash-commands",
         "--setting-sources", "", "--strict-mcp-config"],
        input=r["text"], capture_output=True, text=True, timeout=180)
    wall = (time.perf_counter() - t0) * 1000
    try:
        j = json.loads(p.stdout)
        s = j.get("structured_output") or json.loads(j["result"])
        return {"id": r["id"], "pred": s["label"], "conf": s["confidence"], "latency_ms": wall,
                "api_ms": j.get("duration_api_ms"), "cost_usd": j.get("total_cost_usd"),
                "usage": j.get("usage", {}).get("input_tokens", 0) + j.get("usage", {}).get("cache_creation_input_tokens", 0)
                + j.get("usage", {}).get("cache_read_input_tokens", 0)}
    except Exception as e:
        return {"id": r["id"], "error": f"{type(e).__name__}: {e}", "stderr": p.stderr[-300:], "latency_ms": wall}


def main():
    split = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    out = HERE / "results" / f"claude_{split}.jsonl"
    done = {json.loads(line)["id"] for line in out.open() if '"error"' not in line} if out.exists() else set()
    rows = [r for r in map(json.loads, (HERE / "data" / f"{split}.jsonl").open()) if r["id"] not in done][:limit]
    with ThreadPoolExecutor(3) as ex, out.open("a") as f:
        for res in ex.map(call, rows):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
