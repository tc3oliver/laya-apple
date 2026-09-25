#!/bin/sh
# The energy-per-decision campaign behind research/energy-sampler/. Run on AC power, on an
# otherwise idle machine: stop every other GPU/ANE user first (the local LLM server included;
# harness.py records whether omlx-server was running). About 25 minutes.
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/energy-sampler/scripts/run.sh
set -e
cd "$(dirname "$0")/../../.."
D=research/energy-sampler/raw
mkdir -p $D
uv run python research/energy-sampler/scripts/harness.py --out $D/campaign-laya-typed-decisions.json.gz
uv run python research/energy-sampler/scripts/analyze.py
