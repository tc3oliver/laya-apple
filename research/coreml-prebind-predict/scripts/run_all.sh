#!/bin/sh
# The whole campaign of research/coreml-prebind-predict/, unattended. Idle machine on AC
# power, the local LLM server and other GPU/ANE services stopped, Core ML E5 cache not cleared.
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-prebind-predict/scripts/run_all.sh
#
# 1. The pre-campaign check (check_prebind.py): PB bit-identical to coremltools on every ANE
#    bucket of all three models, or the campaign does not start (criteria.md).
# 2. Per model, the production configuration P and the candidates C and PB in the order
#    P C PB PB C P (ABBA extended to three configurations). 3 models x 6 runs = 18 runs of the
#    hetero-only mix (3 cycles of warm-up + 20 s hetero window), ~85 s each.
# A run whose output already exists is skipped, so an interrupted campaign resumes in order;
# run_config.py writes each file only once its run has finished.
set -e
cd "$(dirname "$0")/../../.."
[ -n "$LAYA_APPLE_CACHE" ] || { echo "set LAYA_APPLE_CACHE" >&2; exit 2; }
S=research/coreml-prebind-predict/scripts
D=research/coreml-prebind-predict/raw
PY="uv run --with pyobjc-framework-CoreML==12.2.2 python"
mkdir -p $D
echo "check (~15 s), then 18 runs x ~85 s (3 x [2 s idle, 2 s warm-up, 0.5 s lead, 20 s hetero], load, references): ~30 min"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
if [ -e $D/check.json ]; then echo "skip check (exists)"; else $PY $S/check_prebind.py --out $D/check.json; fi
uv run python -c "import json, sys; sys.exit(0 if json.load(open('$D/check.json'))['PB_bit_identical_everywhere'] else 1)" \
  || { echo "PB is not bit-identical to coremltools (raw/check.json): the campaign does not start" >&2; exit 1; }
run() {  # model short long config round
  out=$D/$1-$4-r$5.json.gz
  if [ -e "$out" ]; then echo "skip $out (exists)"; return; fi
  echo "$(date '+%H:%M:%S') $1 $4 r$5"
  $PY $S/run_config.py --config $4 --model $1 --short $2 --long $3 --output $out
}
for spec in "laya 128 512 A" "laya-typed-decisions 128 1024 A" "laya-multilingual 96 1024 B"; do
  set -- $spec
  run $1 $2 $3 $4 1
  run $1 $2 $3 C 1
  run $1 $2 $3 PB 1
  run $1 $2 $3 PB 2
  run $1 $2 $3 C 2
  run $1 $2 $3 $4 2
done
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
uv run python $S/analyze.py
