#!/bin/sh
# The representative re-run behind research/gpu-ane-interference/ledger/: the runtime-trace
# pipeline on the subset that carries the study's headline result, not the full campaign.
# Runs strictly one after another, on AC power, idle machine.
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/gpu-ane-interference/scripts/run_ledger.sh
set -e
cd "$(dirname "$0")/../../.."
S=research/gpu-ane-interference/scripts
D=research/gpu-ane-interference/ledger
PY="uv run python $S/interference.py --model laya-typed-decisions"
CELLS="solo:gpu_M solo:ane_B matrix:gpu_M+ane_B"
mkdir -p $D/raw $D/overhead

# 1. GPU L128 alone, ANE L128 alone, both at once; ANE on a thread, then in a process.
for pl in thread process; do
  $PY --ane-placement $pl --plan device --parts --cells $CELLS --out $D/raw/device-laya-typed-decisions-$pl.json.gz
done
# 2. The router under load (for the prediction error): the Part B mix and a short-heavy mix.
$PY --ane-placement thread --plan product --only-open --rates 25 --heavy-rates 100 \
    --out $D/raw/product-laya-typed-decisions-thread.json.gz
# 3. The same product cells with tracing and hooks off: routing and answers under load.
$PY --ane-placement thread --plan product --only-open --rates 25 --heavy-rates 100 --no-trace --no-backend-hooks \
    --out $D/overhead/product-laya-typed-decisions-thread-off.json.gz
# 4. What each instrumentation layer costs: solo cells with both, runtime trace only, neither.
for v in "hooks:" "trace:--no-backend-hooks" "off:--no-trace --no-backend-hooks"; do
  $PY --ane-placement thread --plan device --parts --cells solo:gpu_M solo:ane_B --cycles 2 --seconds 15 ${v#*:} \
      --out $D/overhead/solo-laya-typed-decisions-thread-${v%%:*}.json.gz
done
# 5. Idle-queue equivalence: identical decisions and answers with tracing off and on.
for pl in thread process; do
  uv run python $S/equivalence.py --model laya-typed-decisions --ane-placement $pl \
      --out $D/equivalence-laya-typed-decisions-$pl.json
done
uv run python $S/analyze_ledger.py
uv run python $S/figures_ledger.py
