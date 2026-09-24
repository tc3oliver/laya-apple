#!/bin/sh
# Official switchyard-v1 campaign (benchmarks/switchyard/). Runs the standard-settings
# `laya-apple switchyard` demo three times, each into its own <campaign>/raw/run-00N/
# directory, for scripts/switchyard_report.py. Refuses to start if another laya-apple process
# (this repo's CLI or a scripts/bench_*.py) is already running, since the workload shares the
# machine's GPU/ANE with whatever else laya-apple is doing. Also refuses to start if <campaign>
# already exists and holds data: raw benchmark data is never overwritten (AGENTS.md). Every
# campaign is its own new directory, named for the machine it ran on, e.g. v1-m4-max.
#
# This script does not stop or start any service. Before running it, the operator stops
# every other GPU/ANE consumer on the machine (local model servers, browsers, other benchmarks).
#
# Env (set by the caller, never hard-coded here):
#   LAYA_APPLE_CACHE  laya-apple cache directory, if it is not the default
#   HF_HUB_OFFLINE=1  no network model resolution
#
#   LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh benchmarks/switchyard/run.sh benchmarks/switchyard/v1-m4-max
set -e
cd "$(dirname "$0")/../.."

CAMPAIGN_DIR="$1"
if [ -z "$CAMPAIGN_DIR" ]; then
  echo "usage: run.sh CAMPAIGN_DIR   (e.g. benchmarks/switchyard/v1-m4-max)" >&2
  exit 1
fi

other=$(pgrep -f "laya-apple switchyard|laya_apple\.cli|scripts/bench_" | grep -v "^$$\$" || true)
if [ -n "$other" ]; then
  echo "refusing to start: another laya-apple process is already running (pid(s): $other)" >&2
  exit 1
fi

if [ -e "$CAMPAIGN_DIR" ] && [ -n "$(find "$CAMPAIGN_DIR" -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
  echo "refusing to start: $CAMPAIGN_DIR already has data; every campaign is a new directory" \
    "(move or remove it, raw data is never overwritten)" >&2
  exit 1
fi

D="$CAMPAIGN_DIR/raw"
mkdir -p "$D"

for i in 1 2 3; do
  RUN_DIR="$D/run-00$i"
  if [ -e "$RUN_DIR" ] && [ -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]; then
    echo "refusing to overwrite existing raw data: $RUN_DIR" >&2
    exit 1
  fi
  mkdir -p "$RUN_DIR"
  pmset -g therm > "$RUN_DIR/therm-before.txt" 2>/dev/null || true
  uptime > "$RUN_DIR/loadavg-before.txt" 2>/dev/null || true

  uv run laya-apple switchyard --no-open --out "$RUN_DIR"

  pmset -g therm > "$RUN_DIR/therm-after.txt" 2>/dev/null || true
  uptime > "$RUN_DIR/loadavg-after.txt" 2>/dev/null || true
done
