#!/usr/bin/env bash
# The full same-machine runtime comparison (method: README.md in this directory).
#
#   bash benchmarks/compare-v1.4/campaign.sh RUN_ID
#
# Estimated duration on the reference machine (`drive.py plan` prints the breakdown):
#   setup, first time: ~35 min (environments, ~2.4 GB of Core ML bundles, laya-fast
#                      conversion and six ANE body exports); ~1 min when already set up
#   campaign:          ~3.6 h without energy sampling, ~3.9 h with it
#
# Before starting: stop oMLX and other GPU/ANE workloads, connect power, close other
# applications. The driver refuses to start while oMLX is running.
# Energy (optional): export COMPARE_ENERGY_CMD='python3 <path>/sampler.py --out {out}'.
set -euo pipefail
cd "$(dirname "$0")/../.."

RUN_ID="${1:?usage: campaign.sh RUN_ID}"
# LAYA_APPLE_CACHE must point at the cache holding this machine's validated ANE artifacts;
# without it laya-apple's `auto` rows cannot use the ANE.
: "${LAYA_APPLE_CACHE:?export LAYA_APPLE_CACHE (the laya-apple artifact cache) first}"
DRIVE=(uv run python benchmarks/compare-v1.4/drive.py)

uv run python benchmarks/compare-v1.4/make_shapes.py --check
"${DRIVE[@]}" plan
"${DRIVE[@]}" setup
"${DRIVE[@]}" run --run-id "$RUN_ID"
uv run python benchmarks/compare-v1.4/analyze.py
echo "done: benchmarks/compare-v1.4/raw/$RUN_ID, results.json, tables.md"
echo "remove the environments and model files with: ${DRIVE[*]} teardown --models"
