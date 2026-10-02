#!/bin/zsh
# Redacted copy of the runner used for #121: <...> marks a local absolute path that was removed, so it does not run as is.
set -u
S=<scratchpad>
OUT=$S/r121/raw
cd $S/wt-121 || exit 1
export LAYA_APPLE_CACHE=<cache> HF_HUB_OFFLINE=1
snap() {
  { date -u +%Y-%m-%dT%H:%M:%SZ; sw_vers; uptime; pmset -g therm; df -h / <data>;
    for d in $HOME/Library/Caches/python*(N); do [ -d "$d/com.apple.e5rt.e5bundlecache" ] && du -sk "$d"/com.apple.e5rt.e5bundlecache/* | sed "s|$HOME|~|"; done; } > "$1" 2>&1
}
freegb() { df -g "$1" | awk 'NR==2{print $4}'; }
{ git rev-parse HEAD; uv run python -c "import coremltools,laya_apple,platform;from laya_apple.model import platform_validated;print('coremltools',coremltools.__version__,'laya_apple',laya_apple.__version__,'python',platform.python_version(),'platform_validated',platform_validated())" 2>/dev/null; } > $OUT/setup.txt
for arm in fresh-copy move same-path-recopy touch; do
  r=$(freegb /); d=$(freegb <data>)
  if [ "$r" -lt 240 ] || [ "$d" -lt 240 ]; then echo "STOP disk / ${r}G data ${d}G before $arm" | tee $OUT/STOPPED.txt; exit 2; fi
  snap $OUT/machine-a-$arm-before.txt
  echo "$(date -u +%H:%M:%S) start $arm" >> $OUT/progress.log
  uv run python scripts/bench_coldstart.py laya-typed-decisions --modes wait --repeats 3 --location $arm \
     --out $OUT/a-$arm.json > $OUT/a-$arm.stdout.txt 2> $OUT/a-$arm.stderr.txt &
  bp=$!
  if [ $arm = fresh-copy ]; then
    while kill -0 $bp 2>/dev/null && [ ! -s $OUT/a-$arm.stdout.txt ]; do sleep 5; done
    first=$(head -1 $OUT/a-$arm.stdout.txt)
    if echo "$first" | command grep -q '"after_ready_device": "ane"' && echo "$first" | command grep -q '"64"' \
       && echo "$first" | command grep -q '"96"' && echo "$first" | command grep -q '"128"'; then
      echo "$(date -u +%H:%M:%S) sanity OK first row" >> $OUT/progress.log
    else
      echo "$(date -u +%H:%M:%S) sanity FAIL first row" >> $OUT/progress.log
      pkill -TERM -f "bench_coldstart.py" ; sleep 3
      echo "STOP sanity check failed on first fresh-copy row" | tee $OUT/STOPPED.txt; exit 5
    fi
  fi
  wait $bp; rc=$?
  snap $OUT/machine-a-$arm-after.txt
  echo "$(date -u +%H:%M:%S) end $arm rc=$rc" >> $OUT/progress.log
  if [ $rc -ne 0 ]; then echo "STOP $arm rc=$rc" | tee $OUT/STOPPED.txt; exit 3; fi
  if ! command grep -q "No thermal warning" $OUT/machine-a-$arm-after.txt; then echo "STOP thermal after $arm" | tee $OUT/STOPPED.txt; exit 4; fi
done
echo "$(date -u +%H:%M:%S) DONE" >> $OUT/progress.log
