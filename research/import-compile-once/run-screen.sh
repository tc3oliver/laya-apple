#!/bin/zsh
# #162 screen (n = 1): baseline L128, then final-path L128 with the poller, then injection (i) at L64.
# Exclusive slot: local LLM service stopped, no tests or other benchmarks, mains power.
#   LAYA_APPLE_SOURCE_CACHE=<validated macOS 27 cache> LAYA_IC_DATA_VOLUME=<data> zsh research/import-compile-once/run-screen.sh
RUN=screen
source "${0:A:h}/common.sh"
step baseline-L128-1 --arm baseline --bucket 128 --archive "$A128" || exit 3
step final-path-L128-1 --arm final-path --bucket 128 --archive "$A128" --poller || exit 3
step inject-i-L64 --arm final-path --bucket 64 --archive "$A64" --poller --inject parity-fail || exit 3
uv run python research/import-compile-once/analyze.py --mode screen "$OUT" --json "$OUT/analysis.json" \
  | tee "$OUT/analysis.txt"
