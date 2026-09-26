#!/bin/sh
# The staged-handoff screen of research/coreml-staged-handoff/, unattended, one round per call:
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-staged-handoff/scripts/run_all.sh 1
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-staged-handoff/scripts/run_all.sh 2 H32
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-staged-handoff/scripts/run_all.sh 2 H64 fallback
# It stops no service: the machine is prepared beforehand (idle, on AC power, the local LLM server
# stopped). A run whose output already exists is skipped, so an interrupted round resumes in order;
# run_config.py writes each file only once its run has finished.
set -e
cd "$(dirname "$0")/../../.."
[ -n "$LAYA_APPLE_CACHE" ] || { echo "set LAYA_APPLE_CACHE" >&2; exit 2; }
S=research/coreml-staged-handoff/scripts
D=research/coreml-staged-handoff/raw
PY="uv run --with pyobjc-framework-CoreML==12.2.2 python"
mkdir -p $D
echo "started round $1 $(date '+%Y-%m-%d %H:%M:%S'); free: $(df -h / | tail -1 | awk '{print $4}')"

run() {  # cell rep
  # Crash rule: a run that exits non-zero keeps its console output in raw/failed/<run>.<attempt>.log
  # and is re-run once, in the same position; a second failure stops the campaign.
  name=laya-$1-r$2
  out=$D/$name.json.gz
  if [ -e "$out" ]; then echo "skip $out (exists)"; return; fi
  mkdir -p $D/failed
  while :; do
    failed=$(ls $D/failed/$name.*.log 2>/dev/null | wc -l | tr -d ' ')
    attempt=$((failed + 1))
    log=$D/failed/.$name.running
    echo "$(date '+%H:%M:%S') $1 r$2 (attempt $attempt)"
    { $PY $S/run_config.py --cell $1 --model laya --short 128 --long 512 --cycles 2 --output $out \
        </dev/null 2>&1 && echo 0 >"$log.status" || echo $? >"$log.status"; } | tee "$log"
    rc=$(cat "$log.status")
    rm -f "$log.status"
    if [ "$rc" = 0 ]; then rm -f "$log"; return; fi
    mv "$log" $D/failed/$name.$attempt.log
    echo "run $name failed (exit $rc), attempt $attempt; console output: $D/failed/$name.$attempt.log" >&2
    tail -n 40 $D/failed/$name.$attempt.log >&2
    if [ "$attempt" -ge 2 ]; then
      echo "second failure of $name: the campaign stops" >&2
      exit 1
    fi
  done
}

lines=$(mktemp)
trap 'rm -f "$lines"' EXIT
if [ "$1" = 2 ]; then
  uv run python $S/design.py runs 2 --leader "$2" ${3:+--fallback} >"$lines"
else
  uv run python $S/design.py runs 1 >"$lines"
fi
while read -r cell rep; do run "$cell" "$rep"; done <"$lines"
echo "finished round $1 $(date '+%Y-%m-%d %H:%M:%S'); free: $(df -h / | tail -1 | awk '{print $4}')"
