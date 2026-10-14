#!/usr/bin/env bash
# Reproduce the shadow benchmark. Read-only: nothing here routes, gates or approves real work.
# Needs: the laya-typed-decisions checkpoint in the local HF cache, a logged-in `claude` CLI
# (on PATH, or set CLAUDE_BIN to its path).
# Set SKIP_CLAUDE=1 to reuse results/claude_*.jsonl (Claude calls are resumable and cost ~US$0.4 total).
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
python3 gen_data.py > /dev/null
for split in dev eval; do
  (cd "$ROOT" && HF_HUB_OFFLINE=1 uv run --extra serve python research/ctk-routing-shadow/run_laya.py "$split" --variant described)
  [ -n "${SKIP_CLAUDE:-}" ] || python3 run_claude.py "$split"
done
python3 analyze.py described
