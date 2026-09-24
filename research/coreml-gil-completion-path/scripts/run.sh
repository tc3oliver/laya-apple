#!/bin/sh
# The focused runs behind research/coreml-gil-completion-path/. One after another, on AC power,
# idle machine (other GPU/ANE services stopped).
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-gil-completion-path/scripts/run.sh
set -e
cd "$(dirname "$0")/../../.."
S=research/coreml-gil-completion-path/scripts
D=research/coreml-gil-completion-path/raw
PY="uv run --with pyobjc-framework-CoreML==12.2.2 python"
CELL="--model laya-typed-decisions --plan device --parts --cells solo:gpu_M solo:ane_B matrix:gpu_M+ane_B --cycles 3 --seconds 20"
mkdir -p $D
# 1. The 2x2 probe: {Core ML, not} x {GIL held, released} next to one runtime GPU worker.
$PY $S/completion_probe.py --out $D/completion-probe-laya-typed-decisions.json
# 2. GPU L128 + ANE L128 through the runtime: A thread (product), B process (control), C thread
#    with the GIL released around predict.
$PY $S/run_cell.py --ane-predict coremltools --ane-placement thread $CELL --out $D/A-thread.json.gz
$PY $S/run_cell.py --ane-predict coremltools --ane-placement process $CELL --out $D/B-process.json.gz
$PY $S/run_cell.py --ane-predict pyobjc --ane-placement thread $CELL --out $D/C-thread-nogil.json.gz
