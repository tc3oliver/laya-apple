"""Ask a running `laya-apple serve` one decision over HTTP, the way a Jev client does.

    laya-apple serve                     # in another terminal
    python examples/serve_client.py      # standard library only
"""

import json
import os
import urllib.request

BASE_URL = os.environ.get("TYPESAFE_BASE_URL", "http://127.0.0.1:8642")

body = {
    "model": "jev-latest",  # a Jev id: the server's default (--model auto) picks the checkpoint
    "state": {"task": "fix the failing parser test", "last_tool": "edit parser.py", "tests": "not run yet"},
    "questions": {
        "next_step": {
            "type": "choice",
            "instructions": "What should the agent do next?",
            "criteria": {"run_tests": "run the test suite", "ask_user": "ask the user", "commit": "commit"},
        },
        "done": {"type": "noul", "instructions": "The task is complete."},
    },
}
request = urllib.request.Request(
    BASE_URL.rstrip("/") + "/v1/systemone",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer local-placeholder"},
)
with urllib.request.urlopen(request) as response:
    result = json.load(response)

print(json.dumps(result["answers"], indent=1))
print(f"routed to {result['routing']['model']}: {result['routing']['reason']}")
print(f"answered on {result['laya_apple']['device']} in {result['laya_apple']['latency_ms']:.1f} ms")
