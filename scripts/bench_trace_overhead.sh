#!/usr/bin/env bash
# Tracing overhead A/B: the base branch with trace off against this checkout with trace
# off, a no-op callback and an in-memory recorder. Conditions rotate each round so no
# condition always runs first or last.
#
#   BASE_DIR=/path/to/checkout-of-main LAYA_APPLE_CACHE=... scripts/bench_trace_overhead.sh
#
# BASE_DIR needs its own environment (uv sync --extra ane --extra dev in it). Close other
# GPU/ANE users first; the report records CPU but not other processes' GPU use.
set -euo pipefail
: "${BASE_DIR:?set BASE_DIR to a checkout of the base branch}"
ROUNDS=${ROUNDS:-5}
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$HERE/benchmarks/tracing/overhead.jsonl}
SCRIPT=$HERE/scripts/bench_trace_overhead.py
mkdir -p "$(dirname "$OUT")"
export HF_HUB_OFFLINE=1

conds=("main none" "branch none" "branch noop" "branch recorder")
for ((r = 0; r < ROUNDS; r++)); do
  for ((i = 0; i < ${#conds[@]}; i++)); do
    read -r label cond <<<"${conds[(i + r) % ${#conds[@]}]}"
    dir=$HERE
    [[ $label == main ]] && dir=$BASE_DIR
    (cd "$dir" && uv run --frozen python "$SCRIPT" --label "$label" --condition "$cond" --round "$r" --output "$OUT")
    sleep 2
  done
done
uv run --frozen python "$HERE/scripts/trace_overhead_report.py" "$OUT"
