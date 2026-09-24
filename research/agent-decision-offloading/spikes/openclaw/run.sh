#!/usr/bin/env bash
# EXP-000 OpenClaw sentinel spike. Reproduces every case in README.md.
#
# - Installs the pinned OpenClaw release into a scratch dir (no global install).
# - Every case gets its own isolated HOME / OPENCLAW_HOME / state / config under scratch,
#   a dummy API key, and the scripted mock Main LLM on 127.0.0.1:18081.
# - Two user turns per case run in the same session (turn 2 is the C6 history check).
# - Raw mock logs and OpenClaw state stay in scratch, in a fresh per-run directory. Only the
#   mock's evidence summary and the filter's side-channel record are copied into results/.
# - The script deletes nothing: every results/ file it writes is overwritten by name with `>`.
#
# Usage: ./run.sh            (all cases)
#        ./run.sh b_plugin   (one case)
set -euo pipefail

OPENCLAW_VERSION="2026.9.6"
PORT=18081
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOCK="$HERE/../common/mock_llm.py"
RESULTS="$HERE/results"
SCRATCH="${LAYA_SPIKE_SCRATCH:-$HOME/Developer/scratch/agent-offloading/openclaw-spike}"
INSTALL="$SCRATCH/install"
RUN_DIR="$SCRATCH/runs/$(date +%Y%m%dT%H%M%S)-$$"
PY="${PYTHON:-python3}"
NODE="$(command -v node)"  # resolved before HOME is swapped (mise-managed Node)

mkdir -p "$SCRATCH" "$RESULTS"

if [[ ! -x "$INSTALL/node_modules/.bin/openclaw" ]] ||
  [[ "$("$NODE" -p "require('$INSTALL/node_modules/openclaw/package.json').version")" != "$OPENCLAW_VERSION" ]]; then
  mkdir -p "$INSTALL"
  echo '{"name":"exp000-openclaw-install","private":true}' >"$INSTALL/package.json"
  (cd "$INSTALL" && npm install --no-fund --no-audit --loglevel=error "openclaw@$OPENCLAW_VERSION")
fi
OPENCLAW_MJS="$INSTALL/node_modules/openclaw/openclaw.mjs"

MOCK_PID=""
cleanup() {
  if [[ -n "$MOCK_PID" ]]; then kill "$MOCK_PID" 2>/dev/null || true; wait "$MOCK_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

# run_case <name> <mock tool> <mock args json> <load filter: yes|no> [filter mode] [tool search]
# Tool search defaults to off (direct tool schemas); "default" leaves tools.toolSearch unset.
run_case() {
  local name="$1" tool="$2" args="$3" filter="$4" mode="${5:-full}" search="${6:-off}"
  local dir="$RUN_DIR/$name"
  mkdir -p "$dir/home/.openclaw" "$dir/workspace"

  local paths="\"$HERE/plugins/laya-sentinel-tool\""
  local entries='"laya-sentinel-tool": {"enabled": true}'
  local allow='"laya-sentinel-tool"'
  if [[ "$filter" == yes ]]; then
    paths="$paths, \"$HERE/plugins/laya-spike-filter\""
    entries="$entries, \"laya-spike-filter\": {\"enabled\": true}"
    allow="$allow, \"laya-spike-filter\""
  fi
  local tools_cfg='"toolSearch": false'
  if [[ "$search" == default ]]; then tools_cfg=""; fi
  cat >"$dir/home/.openclaw/openclaw.json" <<EOF
{
  "models": {"providers": {"mock": {
    "baseUrl": "http://127.0.0.1:$PORT/v1",
    "apiKey": "dummy-not-a-secret",
    "api": "openai-completions",
    "models": [{"id": "mock", "name": "mock", "reasoning": false, "input": ["text"],
      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
      "contextWindow": 131072, "maxTokens": 4096}]}}},
  "agents": {"defaults": {"model": {"primary": "mock/mock"}, "workspace": "$dir/workspace"}},
  "tools": {$tools_cfg},
  "plugins": {"allow": [$allow], "load": {"paths": [$paths]}, "entries": {$entries}}
}
EOF

  # Refuse to run against someone else's server on the port: its evidence would not be ours.
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "run.sh: port $PORT is already listening before case $name; stop that process first" >&2
    exit 1
  fi
  "$PY" "$MOCK" --port "$PORT" --tool "$tool" --args "$args" \
    --raw-log "$dir/mock-raw.jsonl" --evidence "$dir/evidence.jsonl" >"$dir/mock.out" 2>&1 &
  MOCK_PID=$!
  local ready=no
  for _ in $(seq 50); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=yes; break; fi
    sleep 0.1
  done
  if [[ "$ready" != yes ]]; then
    echo "run.sh: mock did not answer /v1/models on port $PORT within 5 s (case $name); see $dir/mock.out" >&2
    exit 1
  fi

  oc() {
    (cd "$dir/workspace" && env HOME="$dir/home" OPENCLAW_HOME="$dir/home" \
      OPENCLAW_STATE_DIR="$dir/home/.openclaw" OPENCLAW_CONFIG_PATH="$dir/home/.openclaw/openclaw.json" \
      LAYA_SPIKE_SIDE_CHANNEL="$dir/side-channel.jsonl" LAYA_SPIKE_FILTER_MODE="$mode" \
      "$NODE" "$OPENCLAW_MJS" "$@" </dev/null)
  }
  # Exec needs a non-interactive approval policy; this writes only to the isolated home.
  oc exec-policy preset yolo >"$dir/exec-policy.out" 2>&1
  oc agent --local --session-id "exp000-$name" --timeout 120 --json \
    --message "Turn 1: call the requested tool." >"$dir/turn1.json" 2>"$dir/turn1.err"
  oc agent --local --session-id "exp000-$name" --timeout 120 --json \
    --message "Turn 2: summarise what you saw." >"$dir/turn2.json" 2>"$dir/turn2.err"

  cleanup
  MOCK_PID=""
  # The mock appends, so it writes into the fresh run dir; results/ gets an overwrite by name.
  cat "$dir/evidence.jsonl" >"$RESULTS/$name.evidence.jsonl"
  # The side channel holds the raw result (C5). Strip machine-specific paths before committing.
  if [[ "$filter" == yes ]]; then
    touch "$dir/side-channel.jsonl"
    sed -e "s|$dir|<case-dir>|g" -e "s|$HOME|~|g" "$dir/side-channel.jsonl" >"$RESULTS/$name.side-channel.jsonl"
  fi
  # Does the raw sentinel survive anywhere in OpenClaw's persisted state?
  local hits
  hits="$(command grep -rl --binary-files=text LAYA_SENTINEL "$dir/home/.openclaw" 2>/dev/null |
    sed "s|$dir/home/.openclaw/||" | sort | tr '\n' ' ')" || true
  echo "{\"case\": \"$name\", \"persisted_state_files_with_raw_sentinel\": \"${hits% }\"}" \
    >"$RESULTS/$name.state-scan.jsonl"
  echo "== $name: $(cat "$dir/mock.out" | command grep MOCK | tr '\n' ' ')"
}

want() { [[ $# -eq 0 || " $* " == *" $CASE "* ]]; }
SELECTED=("$@")

# Built-in exec: the command prints the sentinel without the tool-call arguments containing it,
# so a leak in the request can only come from the tool result.
EXEC_ARGS='{"command": "printf \"AAA LAYA_%s BBB\\n\" SENTINEL"}'
BRIDGE_ARGS='{"id": "laya_sentinel", "args": {}}'

for CASE in a_control_plugin b_plugin c_control_exec c_exec c_exec_content_only \
  d_control_error d_error e_control_toolsearch e_toolsearch; do
  want "${SELECTED[@]+"${SELECTED[@]}"}" || continue
  case "$CASE" in
    a_control_plugin) run_case "$CASE" laya_sentinel '{}' no ;;
    b_plugin) run_case "$CASE" laya_sentinel '{}' yes ;;
    c_control_exec) run_case "$CASE" exec "$EXEC_ARGS" no ;;
    c_exec) run_case "$CASE" exec "$EXEC_ARGS" yes ;;
    c_exec_content_only) run_case "$CASE" exec "$EXEC_ARGS" yes content-only ;;
    d_control_error) run_case "$CASE" laya_sentinel_error '{}' no ;;
    d_error) run_case "$CASE" laya_sentinel_error '{}' yes ;;
    # Default config: tools sit behind the Tool Search bridge and are called through tool_call.
    e_control_toolsearch) run_case "$CASE" tool_call "$BRIDGE_ARGS" no full default ;;
    e_toolsearch) run_case "$CASE" tool_call "$BRIDGE_ARGS" yes full default ;;
  esac
done

# One committable line per case: what the Main LLM saw on each turn and the C6 prefix check.
# Every request ends with a runtime-appended internal-context user message (same digest on
# every request), so turn 1's messages without that tail must be an unchanged prefix of turn 2.
"$PY" - "$RESULTS" <<'PY'
import json, pathlib, sys
results = pathlib.Path(sys.argv[1])
rows = []
for ev in sorted(results.glob("*.evidence.jsonl")):
    case = ev.name.removesuffix(".evidence.jsonl")
    turns = [json.loads(line) for line in ev.read_text().splitlines() if line.strip()]
    scan = results / f"{case}.state-scan.jsonl"
    side = results / f"{case}.side-channel.jsonl"
    t1, t2 = (turns + [None, None])[:2]
    prefix = t1["message_digests"][:-1] if t1 else []
    rows.append({
        "case": case,
        "n_tool_result_requests": len(turns),
        "turn1_tool_message": t1 and t1["tool_messages"][-1],
        "turn1_raw_sentinel_anywhere_in_request": t1 and t1["raw_sentinel_anywhere_in_request"],
        "turn2_tool_message": t2 and t2["tool_messages"][-1],
        "turn2_raw_sentinel_anywhere_in_request": t2 and t2["raw_sentinel_anywhere_in_request"],
        "c6_turn1_prefix_unchanged_in_turn2": bool(t1 and t2)
        and t2["message_digests"][: len(prefix)] == prefix,
        "c6_runtime_tail_digest_same": bool(t1 and t2)
        and t1["message_digests"][-1] == t2["message_digests"][-1],
        "side_channel_records": len(side.read_text().splitlines()) if side.exists() else 0,
        "persisted_state_files_with_raw_sentinel": (
            json.loads(scan.read_text())["persisted_state_files_with_raw_sentinel"]
            if scan.exists() else None
        ),
    })
(results / "summary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
for r in rows:
    print(json.dumps(r))
PY
