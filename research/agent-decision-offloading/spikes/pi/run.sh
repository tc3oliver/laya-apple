#!/usr/bin/env bash
# EXP-000 Pi sentinel spike. Reproduces every case end to end:
#   installs the pinned Pi into a scratch dir, isolates Pi's config and HOME there,
#   points Pi at the scripted mock Main LLM (OpenAI Chat Completions on 127.0.0.1:18083),
#   runs each case non-interactively, then a second user turn in the same session (C6).
# Committable evidence goes to ./results; raw logs, sessions and event streams stay in scratch.
set -euo pipefail

PI_VERSION="0.87.1"
PORT=18083
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOCK="$HERE/../common/mock_llm.py"
# Two-call wrapper around the common mock (reuses its handler; only widens the reply).
MOCK_PARALLEL="$HERE/../common/mock_llm_parallel.py"
EXT="$HERE/laya-sentinel-extension.ts"
RESULTS="${LAYA_SPIKE_RESULTS:-$HERE/results}"   # EXP-000B writes to its own directory
SCRATCH="${LAYA_SPIKE_SCRATCH:-$HOME/Developer/scratch/agent-offloading/pi}"
PY=(uv run --no-project python)
PI_TIMEOUT="${LAYA_SPIKE_PI_TIMEOUT:-60}"   # watchdog per Pi invocation, seconds
CASES="${LAYA_SPIKE_CASES:-}"               # space-separated subset; empty runs every case

NODE="$(command -v node)"   # Node comes from mise; resolve it before HOME is replaced
INSTALL="$SCRATCH/install"
PI_CLI="$INSTALL/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"

# 1. Pinned install, local to scratch (no global install).
mkdir -p "$INSTALL"
if [ "$("$NODE" -p "require('$PI_CLI/../../../package.json').version" 2>/dev/null)" != "$PI_VERSION" ]; then
  [ -f "$INSTALL/package.json" ] || echo '{"private":true}' > "$INSTALL/package.json"
  (cd "$INSTALL" && npm install --no-fund --no-audit --save-exact "@earendil-works/pi-coding-agent@$PI_VERSION")
fi

# 2. Isolated run directory: fake HOME, agent dir, sessions, working dir. Never the real ~/.pi.
# A fresh directory per run: nothing is deleted, and earlier raw logs are kept.
RUN="$SCRATCH/runs/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$RUN/home" "$RUN/agent" "$RUN/work" "$RUN/logs" "$RESULTS"
cat > "$RUN/agent/models.json" <<JSON
{
  "providers": {
    "laya-mock": {
      "baseUrl": "http://127.0.0.1:$PORT/v1",
      "api": "openai-completions",
      "apiKey": "dummy-not-a-secret",
      "models": [{ "id": "mock", "name": "EXP-000 mock", "contextWindow": 128000, "maxTokens": 4096 }]
    }
  }
}
JSON

MOCK_PID=""
cleanup() { [ -n "$MOCK_PID" ] && kill "$MOCK_PID" 2>/dev/null && wait "$MOCK_PID" 2>/dev/null || true; }
trap cleanup EXIT

start_mock() {  # $1 case, $2 tool, $3 args, [$4 second-call args -> two-call mock]
  local mock=("$MOCK") second=()
  # Never talk to a leftover mock (or anything else) on the port: fail the run instead.
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $PORT is already in use; refusing to start case $1" >&2; exit 1
  fi
  [ -n "${4:-}" ] && mock=("$MOCK_PARALLEL") && second=(--second-args "$4")
  # The mock appends evidence, so clear this case's own named file. Other results files
  # (side channel copies, summary.json) are overwritten by summarize.py.
  rm -f "$RESULTS/$1.evidence.jsonl"
  rm -f "$RESULTS/$1.status.jsonl"
  "${PY[@]}" "${mock[@]}" --port "$PORT" --tool "$2" --args "$3" ${second[@]+"${second[@]}"} \
    --raw-log "$RUN/logs/$1.raw.jsonl" --evidence "$RESULTS/$1.evidence.jsonl" \
    > "$RUN/logs/$1.mock.out" 2>&1 &
  MOCK_PID=$!
  for _ in $(seq 50); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
    sleep 0.1
  done
  echo "mock did not start" >&2; exit 1
}

pi_run() {  # $1 case, $2 turn, $3 "ext"|"noext", $4 filter(1|0), $5 prompt, [$6 -c]
  local ext_args=()
  [ "$3" = "ext" ] && ext_args=(--extension "$EXT")
  # stdin is /dev/null: Pi reads piped stdin to EOF before its first request
  # (main.ts:874-876), so an inherited open pipe blocks it forever. `perl -e alarm` + exec
  # is the watchdog: SIGALRM survives the exec and kills Pi after $PI_TIMEOUT s (exit 142).
  (cd "$RUN/work" && env -i PATH="$(dirname "$NODE"):/usr/bin:/bin" HOME="$RUN/home" \
      PI_CODING_AGENT_DIR="$RUN/agent" PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0 \
      LAYA_SPIKE_SIDE_CHANNEL="$RUN/logs/$1.side.jsonl" LAYA_SPIKE_FILTER="$4" \
      perl -e 'alarm shift; exec @ARGV' "$PI_TIMEOUT" "$NODE" "$PI_CLI" --mode json --provider laya-mock --model mock \
      --session-dir "$RUN/sessions/$1" --no-extensions --no-skills --no-prompt-templates --no-context-files \
      ${ext_args[@]+"${ext_args[@]}"} ${6:-} "$5") < /dev/null > "$RUN/logs/$1.turn$2.events.jsonl" 2> "$RUN/logs/$1.turn$2.stderr"
}

requests() { [ -f "$RUN/logs/$1.raw.jsonl" ] && wc -l < "$RUN/logs/$1.raw.jsonl" | tr -d ' ' || echo 0; }

turn() {  # run one Pi turn and record its status; never aborts the script
  local before rc=0 status
  before=$(requests "$1")
  pi_run "$@" || rc=$?
  local sent=$(( $(requests "$1") - before ))
  case "$rc" in
    0) status=ok ;;
    142) status=timeout ;;
    *) status=error ;;
  esac
  [ "$status" = ok ] && [ "$sent" -eq 0 ] && status=no_request
  printf '{"turn":%s,"status":"%s","exit_code":%s,"mock_requests":%s,"timeout_s":%s}\n' \
    "$2" "$status" "$rc" "$sent" "$PI_TIMEOUT" >> "$RESULTS/$1.status.jsonl"
  [ "$status" = ok ] || echo "!! $1 turn $2: $status (exit $rc, $sent mock requests)" >&2
}

run_case() {  # $1 case, $2 ext|noext, $3 filter, $4 tool, $5 args, $6 second-turn(1|0), [$7 second-call args]
  [ -z "$CASES" ] || [[ " $CASES " == *" $1 "* ]] || return 0
  echo "== $1"
  start_mock "$1" "$4" "$5" "${7:-}"
  turn "$1" 1 "$2" "$3" "Call the $4 tool, then report its output."
  [ "$6" = "1" ] && turn "$1" 2 "$2" "$3" "Thanks. Anything else?" -c
  cleanup; MOCK_PID=""
  cat "$RUN/logs/$1.mock.out"
}

# The literal command puts LAYA_SENTINEL into the assistant's own tool-call arguments, which
# the mock's whole-request check also sees. The split variant prints the identical output
# without the literal in the arguments, so the whole-request check isolates the tool result.
BASH_LITERAL='{"command":"echo AAA LAYA_SENTINEL BBB"}'
BASH_ARGS='{"command":"printf '"'"'AAA LAYA_%s BBB\\n'"'"' SENTINEL"}'
BASH_ERR_ARGS='{"command":"printf '"'"'AAA LAYA_%s BBB\\n'"'"' SENTINEL; exit 3"}'

# (a) negative controls: raw sentinel must reach the mock.
run_case control_bash_noext   noext 1 bash          "$BASH_ARGS" 0
run_case control_tool_nofilter ext  0 laya_sentinel '{}'         0
# (b) extension-registered tool, (c) built-in bash, (d) built-in bash error path.
run_case ext_tool             ext   1 laya_sentinel '{}'         1
run_case builtin_bash         ext   1 bash          "$BASH_ARGS" 1
run_case builtin_bash_literal ext   1 bash          "$BASH_LITERAL" 0
run_case builtin_bash_error   ext   1 bash          "$BASH_ERR_ARGS" 0
# (e) parallel batch: two laya_sentinel calls in one assistant message. The tool sets no
# executionMode and Agent defaults to "parallel", so the batch takes executeToolCallsParallel.
# The first call sleeps 20 ms in execute() (less than the 50 ms hook), so the two hooks overlap
# and the first call still finishes last: admission order is then
# distinguishable from completion order.
PAR_FIRST='{"tag":"first","delay_ms":20}'
PAR_SECOND='{"tag":"second","delay_ms":0}'
run_case control_parallel_nofilter ext 0 laya_sentinel "$PAR_FIRST" 0 "$PAR_SECOND"
run_case parallel_ext_tool         ext 1 laya_sentinel "$PAR_FIRST" 1 "$PAR_SECOND"

# Sanitised summaries only: side channel, per-surface sentinel checks, C6 prefix comparison.
"${PY[@]}" "$HERE/summarize.py" "$RUN" "$RESULTS"
