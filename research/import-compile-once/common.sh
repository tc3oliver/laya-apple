# Sourced by run-screen.sh and run-full.sh (zsh): output directory, preflight, the run-start e5rt
# listing, the export into a temp directory and the cleanup trap. Not run on its own.
set -u
ROOT=${0:A:h:h:h}
cd "$ROOT" || exit 1
: "${LAYA_IC_DATA_VOLUME:?set LAYA_IC_DATA_VOLUME to the second volume whose free space is checked (>= 200 GB)}"
: "${LAYA_APPLE_SOURCE_CACHE:?set LAYA_APPLE_SOURCE_CACHE to the validated macOS 27 laya-apple cache (read by export only)}"
export HF_HUB_OFFLINE=1 LAYA_IC_DATA_VOLUME
OUT=${OUT:-research/import-compile-once/raw-$(date -u +%Y-%m-%d)-$RUN}
mkdir -p "$OUT"
H=(uv run python research/import-compile-once/harness.py)
TMPROOT=${TMPDIR:-$(getconf DARWIN_USER_TEMP_DIR)}
TMPROOT=${TMPROOT%/}
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$OUT/progress.log"; }
step() {
  log "start $1"
  "${H[@]}" step --label "$@" --out-dir "$OUT" 2> "$OUT/$1.stderr.txt"
  rc=$?
  log "end $1 rc=$rc"
  return $rc
}
finish() {
  log "cleanup: e5rt entries new since the run started, temp caches, the export directory"
  "${H[@]}" cleanup --before "$OUT/e5rt-run-before.json" --ledger "$OUT/created-paths.txt" --out "$OUT/cleanup.json"
  "${H[@]}" preflight --out-dir "$OUT/after" > /dev/null 2>&1
  "${H[@]}" redact --out-dir "$OUT" --source-cache "$LAYA_APPLE_SOURCE_CACHE"
  log "done; restore the local LLM service"
}

"${H[@]}" preflight --out-dir "$OUT" || { log "STOP preflight (disk, thermal or power)"; exit 2; }
"${H[@]}" e5rt-snapshot --out "$OUT/e5rt-run-before.json" || exit 2
touch "$OUT/created-paths.txt"
trap finish EXIT
EXP=$(mktemp -d "$TMPROOT/laya-ic-export.XXXXXX") || exit 2
echo "$EXP" >> "$OUT/created-paths.txt"
{ git rev-parse HEAD
  uv run python -c "import coremltools,laya_apple,platform;print('coremltools',coremltools.__version__,'laya_apple',laya_apple.__version__,'python',platform.python_version())"
} > "$OUT/setup.txt" 2>&1
log "export L128 and L64 from the source cache"
LAYA_APPLE_CACHE=$LAYA_APPLE_SOURCE_CACHE uv run laya-apple artifacts export laya-typed-decisions \
  --length 128 --length 64 --out "$EXP" > "$OUT/export.log" 2>&1 || { log "STOP export"; exit 3; }
A128=$EXP/laya-typed-decisions-L128.tar.gz
A64=$EXP/laya-typed-decisions-L64.tar.gz
