#!/bin/sh
# The full measurement campaign behind research/gpu-ane-interference/README.md.
# Runs strictly one after another (never two benchmarks at once), on AC power, idle machine.
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/gpu-ane-interference/scripts/run_all.sh
set -e
cd "$(dirname "$0")/../../.."
D=research/gpu-ane-interference/raw
PY="uv run python research/gpu-ane-interference/scripts/interference.py"
mkdir -p $D
for pl in thread process; do
  $PY --model laya-typed-decisions --ane-placement $pl --plan device --out $D/device-laya-typed-decisions-$pl.json.gz
done
for pl in thread process; do
  $PY --model laya-multilingual --ane-placement $pl --plan device --part-a-short 96 --out $D/device-laya-multilingual-$pl.json.gz
done
for pl in thread process; do
  $PY --model laya-typed-decisions --ane-placement $pl --plan product --rates 15 25 35 --out $D/product-laya-typed-decisions-$pl.json.gz
  $PY --model laya-multilingual --ane-placement $pl --plan product --part-a-short 96 --rates 40 80 120 --out $D/product-laya-multilingual-$pl.json.gz
done
# Added after the first analysis: one device alone at low load (the idle-machine slowdown).
for m in laya-typed-decisions:thread laya-multilingual:process; do
  $PY --model ${m%%:*} --ane-placement ${m##*:} --plan device --parts sparse --only sparse \
      --out $D/sparse-${m%%:*}-${m##*:}.json.gz
done
