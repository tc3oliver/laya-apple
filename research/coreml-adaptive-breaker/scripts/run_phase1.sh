#!/bin/sh
# Phase 1, the recovery experiment (research/coreml-adaptive-breaker/phase1.md), unattended:
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-adaptive-breaker/scripts/run_phase1.sh
# The 12 runs go in the fixed order of phase1_analyze.py runs, each a fresh process between two
# machine snapshots (#103's machine_snapshot.py). Crash rule: an A or B run is re-run once in place
# (a second failure stops: INCONCLUSIVE); an R crash stops at once (FAIL). Then every run whose
# snapshots failed the qualification's machine item is re-run once as <run>-b (phase1_analyze.py
# reruns), and the analysis prints the outcome. It stops no service: the machine is prepared
# beforehand as #103's qualification left it. A run whose output exists is skipped, so an
# interrupted campaign resumes in order.
set -e
cd "$(dirname "$0")/../../.."
[ -n "$LAYA_APPLE_CACHE" ] || { echo "set LAYA_APPLE_CACHE" >&2; exit 2; }
S=research/coreml-adaptive-breaker/scripts
SNAP=research/coreml-staged-handoff/scripts/machine_snapshot.py
D=research/coreml-adaptive-breaker/raw
PY="uv run --with pyobjc-framework-CoreML==12.2.2 python"
mkdir -p $D/failed
echo "phase 1 started $(date '+%Y-%m-%d %H:%M:%S')"

run() {  # cell rep
  name=laya-$1-r$2
  out=$D/$name.json.gz
  if [ -e "$out" ]; then echo "skip $out (exists)"; return 0; fi
  while :; do
    failed=$(ls $D/failed/$name.*.log 2>/dev/null | wc -l | tr -d ' ')
    attempt=$((failed + 1))
    log=$D/failed/.$name.running
    uv run python $SNAP $D/$name.before.json </dev/null
    echo "$(date '+%H:%M:%S') $name (attempt $attempt)"
    rc=0
    $PY $S/run_config.py --cell "$1" --output "$out" </dev/null >"$log" 2>&1 || rc=$?
    uv run python $SNAP $D/$name.after.json </dev/null
    if [ "$rc" = 0 ]; then rm -f "$log"; return 0; fi
    mv "$log" $D/failed/$name.$attempt.log
    echo "run $name failed (exit $rc), attempt $attempt" >&2
    if [ "$1" = R ]; then echo "R crashed: a hard failure; phase 1 stops" >&2; return 1; fi
    if [ "$attempt" -ge 2 ]; then echo "second failure of $name: phase 1 stops (INCONCLUSIVE)" >&2; return 1; fi
  done
}

list=$(mktemp)
trap 'rm -f "$list"' EXIT
uv run python $S/phase1_analyze.py runs </dev/null >"$list"
stop=0
while read -r cell rep <&3; do run "$cell" "$rep" || { stop=1; break; }; done 3<"$list"
if [ "$stop" = 0 ]; then
  uv run python $S/phase1_analyze.py reruns </dev/null >"$list"
  while read -r cell rep <&3; do run "$cell" "$rep" || { stop=1; break; }; done 3<"$list"
fi
echo "phase 1 finished $(date '+%Y-%m-%d %H:%M:%S')"
uv run python $S/phase1_analyze.py </dev/null >/dev/null
sed -n 3p research/coreml-adaptive-breaker/phase1_tables.md
exit $stop
