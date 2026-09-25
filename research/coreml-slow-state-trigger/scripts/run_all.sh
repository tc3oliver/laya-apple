#!/bin/sh
# The whole slow-state trigger campaign of research/coreml-slow-state-trigger/, unattended. Idle
# machine on AC power, the local LLM server and other GPU/ANE services stopped, Core ML E5 cache
# not cleared.
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-slow-state-trigger/scripts/run_all.sh
#
# 1. The pre-campaign check (check_cells.py): PB-R and PB-H bit-identical to coremltools on every
#    laya ANE bucket, PB-R releasing and PB-H holding the GIL, or the campaign does not start.
#    Skipped if raw/check.json exists with all_ok.
# 2. The rounds (design.py, analyze.py's `decide`): `design.py next` prints the next round's runs
#    as "cell round" lines, or nothing once the design has reached an end. At most two rounds.
#    Each run is R1's full #57 mix (12 windows of 20 s), ~4.8 min.
# A run whose output already exists is skipped, so an interrupted campaign resumes in order;
# run_config.py writes each file only once its run has finished.
set -e
cd "$(dirname "$0")/../../.."
[ -n "$LAYA_APPLE_CACHE" ] || { echo "set LAYA_APPLE_CACHE" >&2; exit 2; }
S=research/coreml-slow-state-trigger/scripts
D=research/coreml-slow-state-trigger/raw
PY="uv run --with pyobjc-framework-CoreML==12.2.2 python"
mkdir -p $D
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
all_ok() {
  [ -e $D/check.json ] && uv run python -c "
import json, sys
sys.exit(0 if json.load(open('$D/check.json'))['all_ok'] is True else 1)
"
}
if all_ok; then
  echo "skip check (raw/check.json exists, all_ok)"
else
  $PY $S/check_cells.py --out $D/check.json || true  # the test below decides
  all_ok || { echo "the pre-campaign check failed (raw/check.json): the campaign does not start" >&2; exit 1; }
fi

run() {  # cell round
  # Crash rule (R1's): a run that exits non-zero keeps its console output in
  # raw/failed/<run>.<attempt>.log and is re-run once, in the same position; a second failure
  # (counting logs from before a resume) stops the campaign.
  name=laya-$1-r$2
  out=$D/$name.json.gz
  if [ -e "$out" ]; then echo "skip $out (exists)"; return; fi
  mkdir -p $D/failed
  while :; do
    failed=$(ls $D/failed/$name.*.log 2>/dev/null | wc -l | tr -d ' ')
    attempt=$((failed + 1))
    log=$D/failed/.$name.running
    echo "$(date '+%H:%M:%S') $1 r$2 (attempt $attempt)"
    # an && / || list: set -e must not end the group before the exit status is written
    { $PY $S/run_config.py --cell $1 --model laya --short 128 --long 512 --output $out </dev/null 2>&1 \
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
  runs=$(uv run python $S/design.py next)
  if [ -z "$runs" ]; then break; fi
  if [ "$runs" = "$prev" ]; then
    echo "design.py next still requires the same round after all its runs exist: stopping" >&2
    exit 1
  fi
  prev=$runs
  todo=0
  for f in $(echo "$runs" | awk -v d="$D" '{print d "/laya-" $1 "-r" $2 ".json.gz"}'); do
    [ -e "$f" ] || todo=$((todo + 1))
  done
  echo "round: $(echo "$runs" | awk '{printf "%s%s r%s", (NR > 1 ? ", " : ""), $1, $2}')"
  echo "$todo runs to do x ~4.8 min (~$((todo * 48 / 10)) min), $(date '+%Y-%m-%d %H:%M:%S')"
  echo "$runs" | while read -r cell rnd; do
    run "$cell" "$rnd"
  done
done
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
uv run python $S/analyze.py
