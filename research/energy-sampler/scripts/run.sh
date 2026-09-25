#!/bin/sh
# Run 2 of the energy-per-decision campaign behind research/energy-sampler/ (method version 2,
# criteria-r2.json). Run on AC power, on a quiet machine (README.md, "Run 2", protocol change 5):
# stop every other GPU/ANE user first (the local LLM server included; harness.py records whether
# omlx-server was running). About 30 minutes, at most about 40 with every disturbance repeat.
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/energy-sampler/scripts/run.sh
# Run 1 (method version 1) is raw/campaign-laya-typed-decisions.json.gz; it is not re-run.
set -e
cd "$(dirname "$0")/../../.."
D=research/energy-sampler/raw
mkdir -p $D
uv run python research/energy-sampler/scripts/harness.py --out $D/campaign-r2-laya-typed-decisions.json.gz
uv run python research/energy-sampler/scripts/analyze.py
