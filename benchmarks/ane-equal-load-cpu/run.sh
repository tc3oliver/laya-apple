#!/bin/sh
# Thread vs process ANE placement at one equal GPU offered load (benchmarks/ane-equal-load-cpu/).
# One after another, idle machine on AC power, other GPU/ANE services stopped.
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh benchmarks/ane-equal-load-cpu/run.sh [RATE]
set -e
cd "$(dirname "$0")/../.."
RATE=${1:-30}
D=benchmarks/ane-equal-load-cpu/raw
PY="uv run --with pyobjc-framework-CoreML==12.2.2 python"
mkdir -p $D
for M in laya laya-typed-decisions; do
  for P in thread process; do
    $PY benchmarks/ane-equal-load-cpu/run_tree_cpu.py --gpu-rate $RATE --ane-predict coremltools \
      --ane-placement $P --model $M --plan device --cycles 3 --seconds 20 --out $D/$M-$P-gpu$RATE.json.gz
  done
done
