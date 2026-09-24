#!/bin/sh
# Thread vs process ANE placement on the v1.0 closed-loop heterogeneous mix
# (benchmarks/ane-process-isolation/). Order per model: thread, process, process, thread
# (ABBA), so temporal drift does not favour either placement. Idle machine on AC power, other
# GPU/ANE services stopped, Core ML E5 cache not cleared.
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh benchmarks/ane-process-isolation/run.sh
set -e
cd "$(dirname "$0")/../.."
D=benchmarks/ane-process-isolation/raw
mkdir -p $D
run() {  # model short long placement round
  uv run python benchmarks/ane-process-isolation/run_mix.py --model $1 --short $2 --long $3 \
    --ane-placement $4 --part a --output $D/$1-$4-r$5.json
}
for spec in "laya 128 512" "laya-typed-decisions 128 1024"; do
  set -- $spec
  run $1 $2 $3 thread 1
  run $1 $2 $3 process 1
  run $1 $2 $3 process 2
  run $1 $2 $3 thread 2
done
