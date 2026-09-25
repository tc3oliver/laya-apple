#!/bin/sh
# The whole R1 campaign of research/coreml-prebind-full-protocol/, unattended. Idle machine on AC
# power, the local LLM server and other GPU/ANE services stopped, Core ML E5 cache not cleared.
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-prebind-full-protocol/scripts/run_all.sh
#
# 1. The pre-campaign check (#83's check_prebind.py on laya and laya-typed-decisions): PB
#    bit-identical to coremltools on every ANE bucket, or the campaign does not start.
# 2. The sequential design (design.py, criteria.md): stage 1 for both models, then whatever
#    `design.py next` requires (analyze.py applies the preregistered rule to the runs so far),
#    until it prints nothing. Each run is #77's full #57 mix (12 windows of 20 s), ~4.8 min.
# A run whose output already exists is skipped, so an interrupted campaign resumes in order;
# run_config.py writes each file only once its run has finished.
set -e
cd "$(dirname "$0")/../../.."
[ -n "$LAYA_APPLE_CACHE" ] || { echo "set LAYA_APPLE_CACHE" >&2; exit 2; }
S=research/coreml-prebind-full-protocol/scripts
D=research/coreml-prebind-full-protocol/raw
PB83=research/coreml-prebind-predict/scripts
PY="uv run --with pyobjc-framework-CoreML==12.2.2 python"
mkdir -p $D
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
if [ -e $D/check.json ]; then
  echo "skip check (exists)"
else
  $PY $PB83/check_prebind.py --models laya laya-typed-decisions --out $D/check.json || true  # the test below decides
fi
uv run python -c "
import json, sys
c = json.load(open('$D/check.json'))
sys.exit(0 if c['PB_bit_identical_everywhere'] and set(c['models']) == {'laya', 'laya-typed-decisions'} else 1)
" || { echo "PB is not bit-identical to coremltools on both models (raw/check.json): the campaign does not start" >&2; exit 1; }

run() {  # model short long config round
  # Crash rule (criteria.md): a run that exits non-zero keeps its console output in
  # raw/failed/<run>.<attempt>.log and is re-run once, in the same position; a second failure
  # (counting logs from before a resume) stops the campaign.
  name=$1-$4-r$5
  out=$D/$name.json.gz
  if [ -e "$out" ]; then echo "skip $out (exists)"; return; fi
  mkdir -p $D/failed
  while :; do
    failed=$(ls $D/failed/$name.*.log 2>/dev/null | wc -l | tr -d ' ')
    attempt=$((failed + 1))
    log=$D/failed/.$name.running
    echo "$(date '+%H:%M:%S') $1 $4 r$5 (attempt $attempt)"
    # an && / || list: set -e must not end the group before the exit status is written
    { $PY $S/run_config.py --config $4 --model $1 --short $2 --long $3 --output $out </dev/null 2>&1 \
        && echo 0 >"$log.status" || echo $? >"$log.status"; } | tee "$log"
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

prev=""
while :; do
  step=$(uv run python $S/design.py next)
  [ -n "$step" ] || break
  if [ "$step" = "$prev" ]; then
    echo "design.py next still requires '$step' after all its runs exist: stopping" >&2
    exit 1
  fi
  prev=$step
  set -- $step
  stage=$1
  shift
  runs=$(uv run python $S/design.py runs --stage "$stage" --models "$@")
  todo=0
  for f in $(echo "$runs" | awk -v d="$D" '{print d "/" $1 "-" $4 "-r" $5 ".json.gz"}'); do
    [ -e "$f" ] || todo=$((todo + 1))
  done
  echo "stage $stage ($*): $todo runs to do x ~4.8 min (~$((todo * 48 / 10)) min), $(date '+%Y-%m-%d %H:%M:%S')"
  echo "$runs" | while read -r m short long config rnd; do
    run "$m" "$short" "$long" "$config" "$rnd"
  done
done
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
uv run python $S/analyze.py
