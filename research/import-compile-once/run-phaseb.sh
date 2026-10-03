#!/bin/zsh
# usage: run.sh <worktree> <label> <outdir>
set -u
zmodload zsh/datetime
WT=$1; LABEL=$2; OUT=$3; mkdir -p $OUT
export HF_HUB_OFFLINE=1
CACHE=${LAYA_APPLE_CACHE:?}
E5S=($HOME/Library/Caches/python*/com.apple.e5rt.e5bundlecache(N))
cd $WT || exit 1
{ git rev-parse HEAD; date -u; sw_vers; uptime; pmset -g therm; pmset -g batt | head -1; } > $OUT/$LABEL-before.txt 2>&1
print -l ${^E5S}/*/*(N) | sort > $OUT/$LABEL-e5-before.lst
EXP=$(mktemp -d)
LAYA_APPLE_CACHE=$CACHE uv run laya-apple artifacts export laya-typed-decisions --length 128 --out "$EXP" > $OUT/$LABEL-export.log 2>&1 || exit 3
ARC=$EXP/laya-typed-decisions-L128.tar.gz
for i in 1 2 3; do
  EMPTY=$(mktemp -d)
  t0=$EPOCHREALTIME
  LAYA_APPLE_CACHE="$EMPTY" uv run laya-apple artifacts import "$ARC" > $OUT/$LABEL-import-$i.log 2>&1; rc=$?
  t1=$EPOCHREALTIME
  printf '%s import-%d rc=%d wall_s=%.2f\n' $LABEL $i $rc $(( t1 - t0 )) | tee -a $OUT/$LABEL-summary.txt
  print -l ${^E5S}/*/*(N) | sort > $OUT/$LABEL-e5-after-$i.lst
  rm -rf "$EMPTY"
  [ $rc -eq 0 ] || exit 3
done
rm -rf "$EXP"
{ date -u; uptime; pmset -g therm; } > $OUT/$LABEL-after.txt 2>&1
