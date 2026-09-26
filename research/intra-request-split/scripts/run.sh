#!/bin/sh
# research/intra-request-split: the campaign, one step at a time (criteria.json,
# "sequence_and_fast_fail"). Each step refuses to redo a run whose raw file exists, and never
# starts the next step itself: the operator runs the steps in order and reads each look.
#
#   export LAYA_APPLE_CACHE=<cache> HF_HUB_OFFLINE=1
#   export LAYA_FAST_DIR=<scratch>/laya-fast      # the laya-fast checkout (README, "laya-fast")
#   sh research/intra-request-split/scripts/run.sh STEP
#
# STEP (timing-sensitive steps need the exclusive slot; see README "Hardware queue"):
#   prep            fixtures + FP32 references for laya                     CPU, not timed
#   screen          addendum 1's stop-only screen (~5 min), then its look    timing-sensitive
#   block1         laya-b1-p1, laya-fast-b1-p2, laya-fast-b1-p3, laya-b1-p4 timing-sensitive
#   look1           analyze --look f1 (CONTINUE or STOP FAIL)
#   block2          laya-fast-b2-p1, laya-b2-p2, laya-b2-p3, laya-fast-b2-p4 timing-sensitive
#   look2           analyze --look f2 (CONTINUE, STOP FAIL or STOP INCONCLUSIVE)
#   switchyard      the switchyard-v1 mix proxy                              timing-sensitive
#   serve           the serve decision mix proxy                             timing-sensitive
#   final           analyze --look final
#   prep-b MODEL    fixtures + references for a phase B model               CPU, not timed
#   phase-b MODEL   4 latency runs of MODEL (only after a PASS)              timing-sensitive
set -e
cd "$(dirname "$0")/../../.."
[ -n "$LAYA_APPLE_CACHE" ] || { echo "set LAYA_APPLE_CACHE" >&2; exit 2; }
T=research/intra-request-split
S=$T/scripts
# One environment for every step, synced once before the campaign
# (uv sync --extra ane --extra convert): nothing re-syncs mid-campaign.
PY="uv run --no-sync python"

other=$(pgrep -f "laya-apple (serve|switchyard)|laya_apple\.cli|scripts/bench_|intra-request-split/scripts/(latency|mix|laya_fast_driver)" | grep -v "^$$\$" || true)

ours() {  # run-id
  [ -e $T/raw/latency/$1.json ] && { echo "skip $1 (exists)"; return; }
  [ -z "$other" ] || { echo "refusing: another laya-apple process is running ($other)" >&2; exit 1; }
  $PY $S/latency.py --model laya --run-id "$1"
}

lf() {  # run-id
  [ -e $T/raw/laya-fast/$1.json ] && { echo "skip $1 (exists)"; return; }
  [ -n "$LAYA_FAST_DIR" ] || { echo "set LAYA_FAST_DIR" >&2; exit 2; }
  (cd "$LAYA_FAST_DIR" && .venv/bin/python "$OLDPWD/$S/laya_fast_driver.py" --laya-fast-dir . \
    --fixtures "$OLDPWD/$T/raw/fixtures-laya.json" --criteria "$OLDPWD/$T/criteria.json" \
    --out "$OLDPWD/$T/raw/laya-fast/$1.json" --run-id "$1")
}

case "$1" in
  prep)
    $PY $S/fixtures.py --model laya
    [ -e $T/raw/reference-laya.json ] || $PY $S/reference.py --model laya ;;
  screen)
    if [ -e $T/raw/screen/screen-laya.json ]; then echo "skip screen-laya (exists)"; else
      [ -z "$other" ] || { echo "refusing: another laya-apple process is running ($other)" >&2; exit 1; }
      $PY $S/latency.py --model laya --run-id screen-laya --screen
    fi
    if [ ! -e $T/raw/screen/screen-laya-fast.json ]; then
      [ -n "$LAYA_FAST_DIR" ] || { echo "set LAYA_FAST_DIR" >&2; exit 2; }
      (cd "$LAYA_FAST_DIR" && .venv/bin/python "$OLDPWD/$S/laya_fast_driver.py" --screen --laya-fast-dir . \
        --fixtures "$OLDPWD/$T/raw/fixtures-laya.json" --criteria "$OLDPWD/$T/criteria.json" \
        --out "$OLDPWD/$T/raw/screen/screen-laya-fast.json" --run-id screen-laya-fast)
    fi
    $PY $S/analyze.py --look screen ;;
  block1)
    ours laya-b1-p1; lf laya-fast-b1-p2; lf laya-fast-b1-p3; ours laya-b1-p4 ;;
  look1)
    $PY $S/analyze.py --look f1 ;;
  block2)
    lf laya-fast-b2-p1; ours laya-b2-p2; ours laya-b2-p3; lf laya-fast-b2-p4 ;;
  look2)
    $PY $S/analyze.py --look f2 ;;
  switchyard|serve)
    [ -e $T/raw/mix/$1.json ] && { echo "skip $1 (exists)"; exit 0; }
    $PY $S/mix.py --mix "$1" --run-id "$1" ;;
  final)
    $PY $S/analyze.py --look final ;;
  prep-b)
    $PY $S/fixtures.py --model "$2"
    [ -e $T/raw/reference-$2.json ] || $PY $S/reference.py --model "$2" ;;
  phase-b)
    for p in 1 2 3 4; do
      id="$2-b$(( (p + 1) / 2 ))-p$p"
      [ -e $T/raw/latency/$id.json ] && { echo "skip $id (exists)"; continue; }
      $PY $S/latency.py --model "$2" --run-id "$id"
      if [ "$p" = 2 ]; then
        d=$($PY $S/analyze.py --look b --model "$2" | tail -n 1)
        echo "phase B look after 2 runs: $d"
        [ "$d" = "STOP FAIL" ] && exit 1
      fi
    done
    $PY $S/analyze.py --look b --model "$2" ;;
  *)
    sed -n '2,24p' "$0"; exit 2 ;;
esac
