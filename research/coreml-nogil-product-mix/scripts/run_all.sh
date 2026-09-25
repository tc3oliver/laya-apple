#!/bin/sh
# The whole campaign of research/coreml-nogil-product-mix/, unattended. Idle machine on AC
# power, oMLX and other GPU/ANE services stopped, Core ML E5 cache not cleared.
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-nogil-product-mix/scripts/run_all.sh
#
# Per model, the production configuration P and the candidates C and D run in the order
# P C D D C P (ABBA extended to three configurations: every configuration's mean position is
# the same). 3 models x 6 runs = 18 runs of the #57 mix, ~4.8 min each.
# A run whose output already exists is skipped, so an interrupted campaign resumes in order;
# run_config.py writes each file only once its run has finished.
set -e
cd "$(dirname "$0")/../../.."
[ -n "$LAYA_APPLE_CACHE" ] || { echo "set LAYA_APPLE_CACHE" >&2; exit 2; }
S=research/coreml-nogil-product-mix/scripts
D=research/coreml-nogil-product-mix/raw
PY="uv run --with pyobjc-framework-CoreML==12.2.2 python"
mkdir -p $D
echo "18 runs x ~4.8 min (12 windows of 20 s, 2.5 s gaps, load and references): ~1 h 30 min in total"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
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
  run $1 $2 $3 D 1
  run $1 $2 $3 D 2
  run $1 $2 $3 C 2
  run $1 $2 $3 $4 2
done
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
uv run python $S/analyze.py
