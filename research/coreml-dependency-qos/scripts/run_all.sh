#!/bin/sh
# The dependency QoS screen of research/coreml-dependency-qos/, unattended: round 1 (B, O, Q), then
# analyze.py; round 2 only if round 1 has a valid B and a PASS (design.py runs 2: the passing
# candidates in reverse round-1 order, then B), then analyze.py again. It stops no service: the
# machine is prepared beforehand (idle, on AC power, the local LLM server and other GPU/ANE services
# stopped; Core ML E5 cache not cleared).
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-dependency-qos/scripts/run_all.sh
# A run whose output already exists is skipped, so an interrupted campaign resumes in order;
# run_config.py writes each file only once its run has finished.
set -e
cd "$(dirname "$0")/../../.."
[ -n "$LAYA_APPLE_CACHE" ] || { echo "set LAYA_APPLE_CACHE" >&2; exit 2; }
S=research/coreml-dependency-qos/scripts
D=research/coreml-dependency-qos/raw
PY="uv run --with pyobjc-framework-CoreML==12.2.2 python"
mkdir -p $D
echo "started $(date '+%Y-%m-%d %H:%M:%S'); free: $(df -h / | tail -1 | awk '{print $4}')"

run() {  # cell round
  # Crash rule: a run that exits non-zero keeps its console output in
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

# run() exits the whole script on a second failure: the round loops read their lines from a file,
# not a pipe, so that exit is not confined to a subshell.
lines=$(mktemp)
trap 'rm -f "$lines"' EXIT

uv run python $S/design.py runs 1 >"$lines"
while read -r cell rnd; do run "$cell" "$rnd"; done <"$lines"
uv run python $S/analyze.py >/dev/null
echo "round 1 analysed: $(uv run python -c "import json; print(json.load(open('research/coreml-dependency-qos/results.json'))['outcome'])")"

uv run python $S/design.py runs 2 >"$lines"
if [ -s "$lines" ]; then
  while read -r cell rnd; do run "$cell" "$rnd"; done <"$lines"
else
  echo "round 2 does not run"
fi
echo "finished $(date '+%Y-%m-%d %H:%M:%S'); free: $(df -h / | tail -1 | awk '{print $4}')"
uv run python $S/analyze.py
