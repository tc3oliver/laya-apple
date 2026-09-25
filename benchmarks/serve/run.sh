#!/bin/sh
# Serve decisions beside a local LLM (benchmarks/serve/): the whole campaign, then the analysis.
# scripts/bench_serve.py starts and stops `laya-apple serve` itself, once per block, on a free
# loopback port. It never starts, stops or reconfigures the LLM server: that server must already
# be running, serving its model, and used by nothing else for the whole campaign (the harness
# waits while the server reports other requests, and the analysis flags any window it shared).
#
# Refuses to start if another laya-apple process is running, or if CAMPAIGN_DIR already holds
# data: raw benchmark data is never overwritten (AGENTS.md). Every campaign is a new directory,
# named for the machine, e.g. benchmarks/serve/m4-max.
#
# Env (set by the caller, never hard-coded here):
#   LAYA_APPLE_CACHE  laya-apple cache directory, if it is not the default
#   HF_HUB_OFFLINE=1  no network model resolution
#   OMLX_SETTINGS     the LLM server's JSON settings file; its auth.api_key is read at run time
#                     and only ever sent in the Authorization header. Or LLM_API_KEY.
#   LLM_URL           default http://127.0.0.1:8000
#
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 OMLX_SETTINGS=... sh benchmarks/serve/run.sh benchmarks/serve/m4-max
set -e
cd "$(dirname "$0")/../.."

CAMPAIGN_DIR="$1"
if [ -z "$CAMPAIGN_DIR" ]; then
  echo "usage: run.sh CAMPAIGN_DIR   (e.g. benchmarks/serve/m4-max)" >&2
  exit 1
fi

other=$(pgrep -f "laya-apple (serve|switchyard)|laya_apple\.cli|scripts/bench_" | grep -v "^$$\$" || true)
if [ -n "$other" ]; then
  echo "refusing to start: another laya-apple process is already running (pid(s): $other)" >&2
  exit 1
fi

if [ -e "$CAMPAIGN_DIR" ] && [ -n "$(find "$CAMPAIGN_DIR" -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
  echo "refusing to start: $CAMPAIGN_DIR already has data; every campaign is a new directory" >&2
  exit 1
fi

uv run python scripts/bench_serve.py campaign --out "$CAMPAIGN_DIR" --llm-url "${LLM_URL:-http://127.0.0.1:8000}"
uv run python benchmarks/serve/analyze.py "$CAMPAIGN_DIR"
