#!/bin/zsh
# #162 full run, only after screen-SAVES and in a separate slot: B, A, B, A, B, A at L128, then
# injections (i)-(iv) at L64. Same machine conditions as the screen.
#   LAYA_APPLE_SOURCE_CACHE=<validated macOS 27 cache> LAYA_IC_DATA_VOLUME=<data> zsh research/import-compile-once/run-full.sh
RUN=full
source "${0:A:h}/common.sh"
for i in 1 2 3; do
  step baseline-L128-$i --arm baseline --bucket 128 --archive "$A128" || exit 3
  step final-path-L128-$i --arm final-path --bucket 128 --archive "$A128" --poller || exit 3
done
step inject-i-L64 --arm final-path --bucket 64 --archive "$A64" --poller --inject parity-fail || exit 3
step inject-ii-L64 --arm final-path --bucket 64 --archive "$A64" --poller --inject probe-fail || exit 3
step inject-iii-L64 --arm final-path --bucket 64 --archive "$A64" --inject force-replace-probe-fail || exit 3
step inject-iv-L64 --arm final-path --bucket 64 --archive "$A64" --inject sigkill || exit 3
uv run python research/import-compile-once/analyze.py --mode full "$OUT" --json "$OUT/analysis.json" \
  | tee "$OUT/analysis.txt"
