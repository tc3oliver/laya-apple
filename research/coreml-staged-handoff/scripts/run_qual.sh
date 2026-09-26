#!/bin/sh
# The environment qualification of research/coreml-staged-handoff/qualification.md: 3 fresh-process
# production A runs of the product schedule, each between two machine snapshots. It stops no service.
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-staged-handoff/scripts/run_qual.sh
set -e
cd "$(dirname "$0")/../../.."
[ -n "$LAYA_APPLE_CACHE" ] || { echo "set LAYA_APPLE_CACHE" >&2; exit 2; }
S=research/coreml-staged-handoff/scripts
D=research/coreml-staged-handoff/raw-qual
mkdir -p $D/failed
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
for r in 1 2 3; do
  name=product-laya-A-r$r
  [ -e $D/$name.json.gz ] && { echo "skip $name"; continue; }
  uv run python $S/machine_snapshot.py $D/$name.before.json
  echo "$(date '+%H:%M:%S') $name"
  if ! uv run --extra ane python $S/prod_run.py --cell A --model laya --short 128 --long 512 \
      --schedule product --output $D/$name.json.gz </dev/null >$D/failed/.$name.log 2>&1; then
    mv $D/failed/.$name.log $D/failed/$name.1.log
    uv run python $S/machine_snapshot.py $D/$name.after.json
    echo "run $name failed: the qualification stops" >&2
    exit 1
  fi
  rm -f $D/failed/.$name.log
  uv run python $S/machine_snapshot.py $D/$name.after.json
done
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
uv run python $S/qual_analyze.py >/dev/null
sed -n 3p research/coreml-staged-handoff/qual_tables.md
