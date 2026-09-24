#!/bin/sh
# The ANE placement acceptance gate (#52). One process per measurement, one after another,
# idle machine on AC power, other GPU/ANE services stopped.
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh benchmarks/ane-placement-gate/run.sh
set -e
cd "$(dirname "$0")/../.."
D=benchmarks/ane-placement-gate/raw
PY="uv run python scripts/bench_ane_placement_gate.py"
mkdir -p $D
for m in laya laya-typed-decisions laya-multilingual; do
  for p in thread process; do
    $PY smoke --model $m --placement $p --cycles 2 --seconds 15 --out $D/smoke-$m-$p.json
    $PY lifecycle --model $m --placement $p --iterations 10 --requests 40 --out $D/lifecycle-$m-$p.json
  done
  $PY orphan --model $m --out $D/orphan-$m.json
done
uv run python scripts/ane_placement_gate_report.py
