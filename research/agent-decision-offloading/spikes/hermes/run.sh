#!/usr/bin/env bash
# EXP-000 Hermes Agent sentinel spike. Reproduces every case end to end.
#
#   research/agent-decision-offloading/spikes/hermes/run.sh [case ...]
#
# Cases (default: all, in this order):
#   negative_control plugin_tool plugin_tool_bridge builtin_terminal parallel_read_file
#   plugin_tool_error hook_timeout_failopen plugin_deadline_fallback background_control
#   background_terminal concurrent_exception_control concurrent_exception_guard
#
# Everything third-party lives under $LAYA_SPIKE_SCRATCH (default
# ~/Developer/scratch/agent-offloading/hermes): the pinned Hermes checkout and its uv
# venv, one isolated HERMES_HOME per case, the mock's raw request log and the hook's
# side-channel file. Only the sanitized evidence lands in ./results/.
#
# Hermes runs under `env -i` with a throwaway HOME and HERMES_HOME, so it never reads
# ~/.hermes, a real .env or any provider credential. The only key it sees is a dummy.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOCK="$HERE/../common/mock_llm.py"
MOCK_PARALLEL="$HERE/../common/mock_llm_parallel.py"
RESULTS="${LAYA_SPIKE_RESULTS:-$HERE/results}"  # EXP-000B writes to its own directory
SCRATCH="${LAYA_SPIKE_SCRATCH:-$HOME/Developer/scratch/agent-offloading/hermes}"
HERMES_REPO="https://github.com/NousResearch/hermes-agent.git"
HERMES_SHA="7de8728cba339065329f141cf92686bf06d2c171"  # main on 2026-09-24, v0.21.4 line
PORT=18082
PY=3.12
# The sentinel tool lives in the plugin toolset `laya_spike`; hermes-cli is the default CLI bundle.
TOOLSETS="hermes-cli,laya_spike"

SRC="$SCRATCH/src"
HERMES_BIN="$SRC/.venv/bin/hermes"
MOCK_PID=""

cleanup() {
  if [[ -n "$MOCK_PID" ]] && kill -0 "$MOCK_PID" 2>/dev/null; then
    kill "$MOCK_PID" 2>/dev/null || true
    wait "$MOCK_PID" 2>/dev/null || true
  fi
  MOCK_PID=""
}
trap cleanup EXIT INT TERM

install_hermes() {
  mkdir -p "$SCRATCH"
  if [[ ! -d "$SRC/.git" ]]; then
    git clone --quiet "$HERMES_REPO" "$SRC"
  fi
  if [[ "$(git -C "$SRC" rev-parse HEAD)" != "$HERMES_SHA" ]]; then
    git -C "$SRC" fetch --quiet origin "$HERMES_SHA" || git -C "$SRC" fetch --quiet --unshallow origin || true
    git -C "$SRC" checkout --quiet "$HERMES_SHA"
  fi
  (cd "$SRC" && uv sync --frozen --python "$PY" --quiet)
  echo "hermes: $(git -C "$SRC" rev-parse HEAD) ($("$HERMES_BIN" --version 2>/dev/null | head -1 || echo '?'))"
}

# Run hermes in a scrubbed environment: no inherited credentials, throwaway HOME.
hermes_isolated() {
  local home="$1"; shift
  mkdir -p "$SCRATCH/workdir"
  cd "$SCRATCH/workdir"
  env -i \
    PATH="$SRC/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
    HOME="$SCRATCH/fakehome" \
    HERMES_HOME="$home" \
    TERM=dumb LANG=en_US.UTF-8 \
    LAYA_SPIKE_SIDECHANNEL="$SIDECHANNEL" \
    "$HERMES_BIN" "$@" </dev/null
}

start_mock() {  # start_mock <script> <tool> <args-json> <raw-log> <evidence> [extra args]
  local script="$1" tool="$2" args="$3" raw="$4" evidence="$5"; shift 5
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $PORT is busy; refusing to start" >&2; exit 1
  fi
  PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python "$PY" python "$script" --port "$PORT" --tool "$tool" --args "$args" \
    --raw-log "$raw" --evidence "$evidence" "$@" >"$CASE_DIR/mock.stdout" 2>&1 &
  MOCK_PID=$!
  for _ in $(seq 1 50); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
    sleep 0.1
  done
  echo "mock did not start" >&2; cat "$CASE_DIR/mock.stdout" >&2; exit 1
}

# run_case <name> <spike-plugins,comma-separated|-> <mock-script> <tool> <tool-args-json> <prompt>
#          [mock extra args]
# The sentinel-tool plugin is always enabled; the listed spike plugins are added to it.
run_case() {
  local name="$1" plugins="$2" script="$3" tool="$4" targs="$5" prompt="$6"; shift 6
  # A fresh directory per run: old runs are left in place, never deleted by this script.
  CASE_DIR="$RUN_DIR/$name"
  mkdir -p "$CASE_DIR/home/plugins" "$SCRATCH/fakehome" "$RESULTS"
  SIDECHANNEL="$CASE_DIR/sidechannel.jsonl"
  local home="$CASE_DIR/home" evidence="$RESULTS/$name.evidence.jsonl"
  rm -f "$evidence"

  cat >"$home/config.yaml" <<EOF
model:
  default: mock
  provider: custom
  base_url: http://127.0.0.1:$PORT/v1
  api_key: dummy-not-a-real-key
tools:
  tool_search:
    # off: every tool schema goes to the model directly. Plugin tools are otherwise
    # deferred behind the tool_search/tool_describe/tool_call bridge (tools/tool_search.py).
    enabled: ${TOOL_SEARCH:-off}
${EXTRA_CONFIG:-}
EOF
  cp -R "$HERE/plugins/laya_sentinel_tool" "$home/plugins/"
  hermes_isolated "$home" plugins enable laya_sentinel_tool >"$CASE_DIR/enable.log" 2>&1
  local p
  for p in ${plugins//,/ }; do
    [[ "$p" == - ]] && continue
    cp -R "$HERE/plugins/$p" "$home/plugins/"
    hermes_isolated "$home" plugins enable "$p" >>"$CASE_DIR/enable.log" 2>&1
  done
  hermes_isolated "$home" plugins list >"$CASE_DIR/plugins.txt" 2>&1 || true

  start_mock "$script" "$tool" "$targs" "$CASE_DIR/raw.jsonl" "$evidence" "$@"

  echo "== $name: turn 1"
  hermes_isolated "$home" chat -Q --yolo --source tool --max-turns 4 -t "$TOOLSETS" -q "$prompt" \
    >"$CASE_DIR/turn1.out" 2>"$CASE_DIR/turn1.err" || true
  local sid
  sid="$(command grep -Eo '[0-9]{8}_[0-9]{6}_[0-9a-f]+' "$CASE_DIR/turn1.out" "$CASE_DIR/turn1.err" | head -1 | cut -d: -f2 || true)"
  echo "   session: ${sid:-<none>}"
  if [[ -n "$sid" ]]; then
    echo "== $name: turn 2 (same session, C6 history check)"
    hermes_isolated "$home" chat -Q --yolo --source tool --max-turns 4 -t "$TOOLSETS" --resume "$sid" \
      -q "Second turn: summarise what the tool returned." \
      >"$CASE_DIR/turn2.out" 2>"$CASE_DIR/turn2.err" || true
  fi
  cleanup
  cat "$CASE_DIR/mock.stdout"
}

main() {
  local cases=("$@")
  [[ ${#cases[@]} -eq 0 ]] && cases=(negative_control plugin_tool plugin_tool_bridge builtin_terminal
    parallel_read_file plugin_tool_error hook_timeout_failopen plugin_deadline_fallback
    background_control background_terminal concurrent_exception_control concurrent_exception_guard)
  install_hermes
  RUN_DIR="$SCRATCH/runs/$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$RUN_DIR"
  echo "run dir: $RUN_DIR"
  for c in "${cases[@]}"; do
    case "$c" in
      negative_control)
        run_case negative_control - "$MOCK" laya_sentinel '{}' "Call the laya_sentinel tool." ;;
      plugin_tool)
        run_case plugin_tool laya_spike_filter "$MOCK" laya_sentinel '{}' "Call the laya_sentinel tool." ;;
      plugin_tool_bridge)
        # Hermes's default: the plugin tool is deferred and reached through the tool_call bridge.
        TOOL_SEARCH=on run_case plugin_tool_bridge laya_spike_filter "$MOCK" tool_call \
          '{"calls": [{"name": "laya_sentinel", "arguments": {}}]}' "Call the laya_sentinel tool." ;;
      builtin_terminal)
        # printf assembles the sentinel at run time, so the literal never appears in the
        # model-visible tool-call arguments (`echo AAA LAYA_SENTINEL BBB` would put it in
        # the assistant message and make raw_sentinel_anywhere_in_request true regardless).
        run_case builtin_terminal laya_spike_filter "$MOCK" terminal \
          '{"command": "printf '"'"'AAA LAYA_%s BBB\\n'"'"' SENTINEL"}' "Run the printf command." ;;
      parallel_read_file)
        # Relative paths: hermes runs with cwd=$SCRATCH/workdir (see hermes_isolated).
        mkdir -p "$SCRATCH/workdir"
        printf 'AAA LAYA_SENTINEL BBB\n' >"$SCRATCH/workdir/a.txt"
        printf 'AAA LAYA_SENTINEL BBB\n' >"$SCRATCH/workdir/b.txt"
        run_case parallel_read_file laya_spike_filter "$MOCK_PARALLEL" read_file '{"path": "a.txt"}' \
          "Read both files." --second-args '{"path": "b.txt"}' ;;
      plugin_tool_error)
        # The tool raises; the registry turns the exception into an error result.
        run_case plugin_tool_error laya_spike_filter "$MOCK" laya_sentinel_raise '{}' "Call the laya_sentinel_raise tool." ;;
      hook_timeout_failopen)
        # Limitation probe, not a pass case: a hook slower than plugins.hook_callback_timeout
        # is abandoned and Hermes admits the ORIGINAL result (fail-open).
        EXTRA_CONFIG=$'plugins:\n  hook_callback_timeout: 0.01' \
          run_case hook_timeout_failopen laya_spike_filter "$MOCK" laya_sentinel '{}' "Call the laya_sentinel tool." ;;
      plugin_deadline_fallback)
        # The plugin races a 2.0 s decision against its own 0.2 s deadline, under a 1.0 s
        # Hermes hook timeout that would otherwise fail open, and returns a fixed fallback.
        EXTRA_CONFIG=$'plugins:\n  hook_callback_timeout: 1.0' \
          run_case plugin_deadline_fallback laya_spike_deadline "$MOCK" laya_sentinel '{}' \
          "Call the laya_sentinel tool." ;;
      background_control|background_terminal)
        # A background process with notify_on_complete. Its output reaches the model only as
        # a completion notification, which quiet one-shot mode runs as a follow-up turn
        # (hermes_cli/quiet_single_query.py continue_quiet_notify_completions). The control
        # has the transform_tool_result filter only; the real case adds transform_terminal_output.
        local bg_plugins=laya_spike_filter
        [[ "$c" == background_terminal ]] && bg_plugins=laya_spike_filter,laya_spike_terminal_filter
        run_case "$c" "$bg_plugins" "$MOCK" terminal \
          '{"command": "sleep 1; printf '"'"'AAA LAYA_%s BBB\\n'"'"' SENTINEL", "background": true, "notify_on_complete": true}' \
          "Run the printf command in the background and tell me when it finishes." ;;
      concurrent_exception_control|concurrent_exception_guard)
        # Two parallel-safe session_search calls (session_search is an inline executor, so
        # it bypasses the registry's own exception handling). laya_spike_fault makes the
        # `fault` call raise inside the tool, with the sentinel in the message. The control
        # turns the filter plugin's exception guard off; the guard case leaves it on.
        local guard=true
        [[ "$c" == concurrent_exception_control ]] && guard=false
        EXTRA_CONFIG="plugins:
  entries:
    laya_spike_filter:
      settings:
        exception_guard: $guard" \
          run_case "$c" laya_spike_fault,laya_spike_filter "$MOCK_PARALLEL" session_search \
          '{"query": "alpha"}' "Search past sessions twice." --second-args '{"query": "fault"}' ;;
      *) echo "unknown case: $c" >&2; exit 2 ;;
    esac
  done
  cd "$HERE"
  uv run --no-project --python "$PY" python "$HERE/summarize.py" "$RESULTS" "$SCRATCH" "$RUN_DIR"
}

main "$@"
