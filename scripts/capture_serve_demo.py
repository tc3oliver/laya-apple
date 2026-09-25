"""Record the terminal session behind the README's `laya-apple serve` figure.

    LAYA_APPLE_CACHE=/path/to/cache uv run --extra serve --extra ane \\
        python scripts/capture_serve_demo.py --client-python /path/to/venv/bin/python

`--client-python` is a Python with the released Jev SDK installed, unmodified
(`uv venv && uv pip install typesafe-sdk==0.7.1`). The script needs `curl` and `jq` on PATH.

It runs an equivalent of every command the figure shows, and records each real output:

1. `laya-apple serve 2> serve.log &` from this checkout, on the default port 8642 (it
   refuses to start if the port is taken), then waits on `GET /health` until every loaded
   checkpoint reports its Neural Engine `ready`, so short decisions take the ANE path;
2. `cat next_step.py`: a small client of our own, written with the SDK's public API;
3. `python next_step.py` with `TYPESAFE_BASE_URL` pointed at the local server and a
   placeholder key;
4. the same request with `curl`, piped through `jq` to the server's `laya_apple` block,
   which names the checkpoint and the device that answered.

Then it stops the server and writes docs/readme/serve-demo.json. Everything recorded is our
own server's output or our own client code; the log file stays in a temporary directory.
`scripts/generate_readme_svgs.py` renders the figure, docs/readme/serve-demo.svg, from that
JSON, and its --check fails if the SVG drifts from the recording.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laya_apple import __version__  # noqa: E402
from laya_apple.artifacts import platform_profile  # noqa: E402

OUT = ROOT / "docs" / "readme" / "serve-demo.json"
PORT = 8642
BASE_URL = f"http://127.0.0.1:{PORT}"
PLACEHOLDER_KEY = "local-placeholder"

STATE = "pytest: 2 failed in tests/test_api.py right after the agent edited api/routes.py"
INSTRUCTIONS = "What should the coding agent do next?"
OPTIONS = ["fix_code", "run_tests", "commit"]

CLIENT = f"""from typesafe_sdk import Choice, TypeSafeClient

client = TypeSafeClient()  # reads TYPESAFE_BASE_URL and TYPESAFE_API_KEY
r = client.system_one(
    state={STATE!r},
    questions={{"next": Choice(instructions={INSTRUCTIONS!r},
                              criteria={{{", ".join(f"{o!r}: None" for o in OPTIONS)}}})}},
)
print(r.answers["next"].choice, r.answers["next"].probabilities)
"""

# The body the SDK sends for the request above: Choice(criteria={label: None}).
BODY = {
    "state": STATE,
    "questions": {"next": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {o: None for o in OPTIONS}}},
}
JQ = ".laya_apple | {model, device, routing_reason}"
# Split with a backslash-newline, as typed in a shell, so the recorded command fits the figure.
CURL = (
    f"curl -s {BASE_URL}/v1/systemone -H 'content-type: application/json' \\\n"
    f"    -d @next.json | jq -c {shlex.quote(JQ)}"
)


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _get(path: str) -> dict:
    with urllib.request.urlopen(BASE_URL + path, timeout=5) as r:
        return json.loads(r.read())


def _wait_ready(server: subprocess.Popen, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            sys.exit(f"laya-apple serve exited with status {server.returncode}")
        try:
            health = _get("/health")
        except OSError:
            time.sleep(1)
            continue
        states = [a.get("status") for a in health.get("ane", {}).values()]
        if states and all(s == "ready" for s in states):
            return health
        if any(s == "unavailable" for s in states):
            sys.exit(f"the Neural Engine is unavailable: {health['ane']}; build the artifacts first")
        time.sleep(1)
    sys.exit(f"the Neural Engine was not ready after {timeout_s:.0f} s")


def _run(cmd, *, cwd: Path, env: dict, shell: bool = False) -> str:
    r = subprocess.run(cmd, cwd=cwd, env=env, shell=shell, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        sys.exit(f"{cmd!r} failed with status {r.returncode}:\n{r.stdout}{r.stderr}")
    return r.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--client-python", required=True, help="a Python with typesafe-sdk installed")
    ap.add_argument("--ready-timeout", type=float, default=900, help="seconds to wait for the Neural Engine")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    if not _port_free(PORT):
        sys.exit(f"port {PORT} is in use; stop whatever listens there first")
    client_version = _run(
        [args.client_python, "-c", "import typesafe_sdk; print(typesafe_sdk.__version__)"], cwd=ROOT, env=os.environ
    ).strip()

    with tempfile.TemporaryDirectory(prefix="laya-serve-demo-") as d:
        work = Path(d)
        (work / "next_step.py").write_text(CLIENT)
        (work / "next.json").write_text(json.dumps(BODY))
        env = dict(os.environ)
        env.setdefault("HF_HUB_OFFLINE", "1")
        for k in ("LAYA_API_KEY", "LAYA_APPLE_SERVE_MODEL", "LAYA_APPLE_SERVE_HOST", "LAYA_APPLE_SERVE_PORT"):
            env.pop(k, None)  # the figure shows the defaults
        with open(work / "serve.log", "w") as log:
            server = subprocess.Popen(
                [sys.executable, "-m", "laya_apple.cli", "serve"], cwd=work, env=env, stdout=log, stderr=log
            )
        try:
            health = _wait_ready(server, args.ready_timeout)
            client_env = dict(env, TYPESAFE_BASE_URL=BASE_URL, TYPESAFE_API_KEY=PLACEHOLDER_KEY)
            client_out = _run([args.client_python, "next_step.py"], cwd=work, env=client_env)
            curl_out = _run(CURL, cwd=work, env=env, shell=True)
            response = json.loads(
                _run(
                    [
                        "curl",
                        "-s",
                        f"{BASE_URL}/v1/systemone",
                        "-H",
                        "content-type: application/json",
                        "-d",
                        "@next.json",
                    ],
                    cwd=work,
                    env=env,
                )
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    if not _port_free(PORT):
        print(f"warning: something still listens on {PORT}", file=sys.stderr)

    profile = platform_profile()
    record = {
        "about": "Recorded by scripts/capture_serve_demo.py; rendered to serve-demo.svg by "
        "scripts/generate_readme_svgs.py. Each output is the real output of an equivalent command.",
        "recorded": time.strftime("%Y-%m-%d"),
        "laya_apple": __version__,
        "platform": {"soc": profile.get("soc"), "macos": profile.get("macos")},
        "client": {"name": "typesafe-sdk", "version": client_version},
        "health": {k: health[k] for k in ("default_model", "ane") if k in health},
        "session": [
            {"cmd": "laya-apple serve 2> serve.log &", "out": ""},
            {"comment": "wait until GET /health reports the Neural Engine ready"},
            {"cmd": f"export TYPESAFE_BASE_URL={BASE_URL} TYPESAFE_API_KEY={PLACEHOLDER_KEY}", "out": ""},
            {"cmd": "cat next_step.py", "out": CLIENT},
            {"cmd": "python next_step.py", "out": client_out, "note": f"typesafe-sdk {client_version}, unmodified"},
            {"cmd": CURL, "out": curl_out},
        ],
        "request": BODY,
        "response": response,
    }
    args.out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    print(client_out, curl_out, sep="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
