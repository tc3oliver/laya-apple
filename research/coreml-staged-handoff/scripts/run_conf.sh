#!/bin/sh
# The H64 confirmation of research/coreml-staged-handoff/confirmation.md, unattended:
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-staged-handoff/scripts/run_conf.sh
# The 16 runs in the preregistered order, then any A validity re-runs (A rN-b) that
# confirm_analyze.py names, then the analysis. It stops no service: the machine is prepared
# beforehand (idle, AC power, oMLX stopped, nothing else running). A run whose output exists is
# skipped, so an interrupted campaign resumes in order.
set -e
cd "$(dirname "$0")/../../.."
[ -n "$LAYA_APPLE_CACHE" ] || { echo "set LAYA_APPLE_CACHE" >&2; exit 2; }
S=research/coreml-staged-handoff/scripts
D=research/coreml-staged-handoff/raw-conf
PY="uv run --with pyobjc-framework-CoreML==12.2.2 python"
mkdir -p $D/failed
echo "started $(date '+%Y-%m-%d %H:%M:%S'); free: $(df -h / | tail -1 | awk '{print $4}')"

run() {  # cell rep; crash rule: re-run once in place, a second failure stops the campaign
  name=laya-$1-r$2
  out=$D/$name.json.gz
  if [ -e "$out" ]; then echo "skip $out (exists)"; return; fi
  while :; do
    failed=$(ls $D/failed/$name.*.log 2>/dev/null | wc -l | tr -d ' ')
    attempt=$((failed + 1))
    log=$D/failed/.$name.running
    echo "$(date '+%H:%M:%S') $1 r$2 (attempt $attempt)"
    { $PY $S/run_config.py --cell $1 --model laya --short 128 --long 512 --cycles 2 --output $out \
        </dev/null 2>&1 && echo 0 >"$log.status" || echo $? >"$log.status"; } | tee "$log" | command grep -E "^A [01] |wrote research" || true
    rc=$(cat "$log.status")
    rm -f "$log.status"
    if [ "$rc" = 0 ]; then rm -f "$log"; return; fi
    mv "$log" $D/failed/$name.$attempt.log
    echo "run $name failed (exit $rc), attempt $attempt" >&2
    if [ "$attempt" -ge 2 ]; then echo "second failure of $name: the campaign stops" >&2; exit 1; fi
  done
}

for x in A:1 H64:1 H64:2 A:2  H64:3 A:3 A:4 H64:4  A:5 H64:5 H64:6 A:6  H64:7 A:7 A:8 H64:8; do
  run "${x%%:*}" "${x#*:}"
done
reruns=$(uv run python $S/confirm_analyze.py reruns) || { echo "confirm_analyze.py reruns failed" >&2; exit 1; }
for rep in $reruns; do run A "$rep"; done
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
uv run python $S/confirm_analyze.py >/dev/null
sed -n 3p research/coreml-staged-handoff/confirm_tables.md
