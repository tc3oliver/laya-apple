#!/bin/sh
# One phase of the 1.5 release validation (research/coreml-adaptive-breaker/validation.md):
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-adaptive-breaker/scripts/run_val.sh <phase>
# It runs only if every earlier phase passed, its own status is not already FAIL / INVALID and no P
# run of it crashed (val_analyze.py ready). The phase's runs go in the fixed order of
# val_analyze.py runs, each a fresh process between two machine snapshots (#103's
# machine_snapshot.py). Crash rule: an A run is re-run once in place (a second failure stops the
# phase: INVALID); a P crash stops the phase at once (FAIL). Then val_analyze.py reruns lists the
# machine re-runs (<run>-b) and, for phase 6, the extension run when r1 triggered it; they run in
# turn until the list is empty. It stops no service: the machine is prepared beforehand. A run
# whose output exists is skipped, so an interrupted phase resumes in order.
set -e
cd "$(dirname "$0")/../../.."
[ -n "$LAYA_APPLE_CACHE" ] || { echo "set LAYA_APPLE_CACHE" >&2; exit 2; }
[ -n "$1" ] || { echo "usage: run_val.sh <phase 2-6>" >&2; exit 2; }
phase=$1
S=research/coreml-adaptive-breaker/scripts
SNAP=research/coreml-staged-handoff/scripts/machine_snapshot.py
D=research/coreml-adaptive-breaker/raw-val
mkdir -p $D/failed
uv run python $S/val_analyze.py ready "$phase" </dev/null || exit 1
echo "phase $phase started $(date '+%Y-%m-%d %H:%M:%S')"

run() {  # schedule model cell rep short long seconds [extra flags]
  name=$1-$2-$3-r$4
  out=$D/$name.json.gz
  if [ -e "$out" ]; then echo "skip $out (exists)"; return 0; fi
  while :; do
    failed=$(ls $D/failed/$name.*.log 2>/dev/null | wc -l | tr -d ' ')
    attempt=$((failed + 1))
    log=$D/failed/.$name.running
    uv run python $SNAP $D/$name.before.json </dev/null
    echo "$(date '+%H:%M:%S') $name (attempt $attempt)"
    rc=0
    # shellcheck disable=SC2086  # $8 holds the optional extra flags
    uv run --extra ane python $S/val_run.py --cell "$3" --model "$2" --short "$5" --long "$6" \
      --schedule "$1" --seconds "$7" $8 --output "$out" </dev/null >"$log" 2>&1 || rc=$?
    uv run python $SNAP $D/$name.after.json </dev/null
    if [ "$rc" = 0 ]; then rm -f "$log"; return 0; fi
    mv "$log" $D/failed/$name.$attempt.log
    echo "run $name failed (exit $rc), attempt $attempt" >&2
    if [ "$3" = P ]; then echo "P crashed: FAIL; phase $phase stops" >&2; return 1; fi
    if [ "$attempt" -ge 2 ]; then echo "second failure of A $name: phase $phase stops (INVALID)" >&2; return 1; fi
  done
}

list=$(mktemp)
trap 'rm -f "$list"' EXIT
uv run python $S/val_analyze.py runs "$phase" </dev/null >"$list"
stop=0
while read -r sch model cell rep short long secs extra <&3; do
  run "$sch" "$model" "$cell" "$rep" "$short" "$long" "$secs" "$extra" || { stop=1; break; }
done 3<"$list"
while [ "$stop" = 0 ]; do  # machine re-runs, then (phase 6) the extension run, until none is left
  uv run python $S/val_analyze.py reruns "$phase" </dev/null >"$list"
  [ -s "$list" ] || break
  while read -r sch model cell rep short long secs extra <&3; do
    run "$sch" "$model" "$cell" "$rep" "$short" "$long" "$secs" "$extra" || { stop=1; break; }
  done 3<"$list"
done
echo "phase $phase finished $(date '+%Y-%m-%d %H:%M:%S')"
uv run python $S/val_analyze.py </dev/null >/dev/null
sed -n 3p research/coreml-adaptive-breaker/val_tables.md
exit $stop
