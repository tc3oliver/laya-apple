#!/bin/zsh
# Redacted copy of the runner used for #121: <...> marks a local absolute path that was removed, so it does not run as is.
set -u
S=<scratchpad>
OUT=$S/r121/raw
CACHE=<cache>
cd $S/wt-121 || exit 1
export HF_HUB_OFFLINE=1
E5=$HOME/Library/Caches/python/com.apple.e5rt.e5bundlecache
snap() {
  { date -u +%Y-%m-%dT%H:%M:%SZ; sw_vers; uptime; pmset -g therm; df -h / <data>;
    for d in $HOME/Library/Caches/python*(N); do [ -d "$d/com.apple.e5rt.e5bundlecache" ] && du -sk "$d"/com.apple.e5rt.e5bundlecache/* | sed "s|$HOME|~|"; done; } > "$1" 2>&1
}
freegb() { df -g "$1" | awk 'NR==2{print $4}'; }
diskok() { r=$(freegb /); d=$(freegb <data>); if [ "$r" -lt 240 ] || [ "$d" -lt 240 ]; then echo "STOP disk / ${r}G data ${d}G before $1" | tee $OUT/STOPPED.txt; exit 2; fi; }
thermok() { command grep -q "No thermal warning" "$1" || { echo "STOP thermal $1" | tee $OUT/STOPPED.txt; exit 4; }; }
{ git rev-parse HEAD; uv run python -c "import coremltools,laya_apple,platform;from laya_apple.model import platform_validated;print('coremltools',coremltools.__version__,'laya_apple',laya_apple.__version__,'python',platform.python_version(),'platform_validated',platform_validated())" 2>/dev/null; } > $OUT/setup-slot2.txt

# ---- Phase B ----
diskok phase-b
snap $OUT/machine-b-before.txt
echo "$(date -u +%H:%M:%S) start phase-b export" >> $OUT/progress.log
EXP=$(mktemp -d)
LAYA_APPLE_CACHE=$CACHE uv run laya-apple artifacts export laya-typed-decisions --length 128 --out "$EXP" > $OUT/b-export.log 2>&1
rc=$?; ls -l "$EXP" | sed "s|$EXP|<exp>|" >> $OUT/b-export.log
ARC="$EXP"/laya-typed-decisions-L128.tar.gz
if [ $rc -ne 0 ] || [ ! -f "$ARC" ]; then echo "STOP export rc=$rc" | tee $OUT/STOPPED.txt; exit 3; fi
for i in 1 2 3; do
  diskok b-import-$i
  EMPTY=$(mktemp -d)
  echo "$(date -u +%H:%M:%S) start b-import-$i" >> $OUT/progress.log
  LAYA_APPLE_CACHE="$EMPTY" uv run laya-apple artifacts import "$ARC" > $OUT/b-import-$i.log 2>&1
  rc=$?
  echo "$(date -u +%H:%M:%S) end b-import-$i rc=$rc" >> $OUT/progress.log
  case "$EMPTY" in <tmp>/*|<tmp>/*) rm -rf "$EMPTY";; *) echo "not removing $EMPTY" >> $OUT/progress.log;; esac
  if [ $rc -ne 0 ]; then echo "STOP b-import-$i rc=$rc" | tee $OUT/STOPPED.txt; exit 3; fi
done
case "$EXP" in <tmp>/*|<tmp>/*) rm -rf "$EXP";; esac
snap $OUT/machine-b-after.txt
echo "$(date -u +%H:%M:%S) end phase-b" >> $OUT/progress.log
thermok $OUT/machine-b-after.txt

# ---- Phase C ----
diskok phase-c
snap $OUT/machine-c-before.txt
export LAYA_APPLE_CACHE=$CACHE
echo "$(date -u +%H:%M:%S) start phase-c" >> $OUT/progress.log
MARK=$(mktemp); sleep 1
uv run python scripts/bench_coldstart.py laya-typed-decisions --modes wait --repeats 1 --location fresh-copy --out $OUT/c-fresh-copy.json > $OUT/c-fresh-copy.stdout.txt 2> $OUT/c-fresh-copy.stderr.txt
rc=$?
UCACHE=$(getconf DARWIN_USER_CACHE_DIR)
find "$UCACHE" -newer "$MARK" -maxdepth 4 -print 2>/dev/null | sed "s|$UCACHE|<user-cache>|" > $OUT/c-new-paths.txt
du -sk "$UCACHE"/* 2>/dev/null | sort -n | tail -20 | sed "s|$UCACHE|<user-cache>|" > $OUT/c-sizes.txt
{ echo "# e5rt bundle cache: entries newer than the Phase C start (paths and sizes only)";
  find "$E5" -newer "$MARK" -maxdepth 2 -mindepth 2 -print 2>/dev/null | while read p; do du -sk "$p"; done | sed "s|$HOME|~|";
  echo "# per build totals"; du -sk "$E5"/* | sed "s|$HOME|~|";
  echo "# entry count per build"; for b in "$E5"/*; do echo "$(ls "$b" | wc -l | tr -d ' ') $(basename $b)"; done;
  echo "# other per-process e5rt caches"; ls -d $HOME/Library/Caches/*/com.apple.e5rt.e5bundlecache 2>/dev/null | sed "s|$HOME|~|"; } > $OUT/c-e5rt-cache.txt
rm -f "$MARK"
snap $OUT/machine-c-after.txt
echo "$(date -u +%H:%M:%S) end phase-c rc=$rc" >> $OUT/progress.log
if [ $rc -ne 0 ]; then echo "STOP phase-c rc=$rc" | tee $OUT/STOPPED.txt; exit 3; fi
thermok $OUT/machine-c-after.txt
echo "$(date -u +%H:%M:%S) DONE slot2" >> $OUT/progress.log
