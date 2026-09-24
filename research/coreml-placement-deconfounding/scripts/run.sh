#!/bin/sh
# A / B / C at a fixed GPU offered load (research/coreml-placement-deconfounding/). One after
# another, idle machine on AC power, other GPU/ANE services stopped. The Core ML E5 cache is
# left as it is (not cleared) for all three.
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-placement-deconfounding/scripts/run.sh [RATE]
set -e
cd "$(dirname "$0")/../../.."
RATE=${1:-45}
S=research/coreml-placement-deconfounding/scripts
D=research/coreml-placement-deconfounding/raw
PY="uv run --with pyobjc-framework-CoreML==12.2.2 python"
ARGS="--gpu-rate $RATE --model laya-typed-decisions --plan device --cycles 3 --seconds 20"
mkdir -p $D
$PY $S/run_fixed.py --ane-predict coremltools --ane-placement thread $ARGS --out $D/A-thread-gpu$RATE.json.gz
$PY $S/run_fixed.py --ane-predict coremltools --ane-placement process $ARGS --out $D/B-process-gpu$RATE.json.gz
$PY $S/run_fixed.py --ane-predict pyobjc --ane-placement thread $ARGS --out $D/C-thread-nogil-gpu$RATE.json.gz
