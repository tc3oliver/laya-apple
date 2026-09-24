"""Reduce one Pi spike run to committable evidence.

Reads the scratch run dir (raw mock logs, Pi JSON event streams, session JSONL files and
the extension's side channel) and writes only sanitised summaries into results/:

- ``<case>.side_channel.jsonl``: the extension's own records (tool name, args, raw output,
  hook ordering). It holds no third-party text.
- ``summary.json``: per case, what the mock saw, where ``LAYA_SENTINEL`` appears on every
  other surface (event stream, session file), and the C6 prefix comparison.

The raw mock log, the event streams and the session files stay in scratch: they contain
Pi's system prompt and tool definitions.

Usage: python summarize.py <scratch-run-dir> <results-dir>
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

RAW = "LAYA_SENTINEL"


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def event_surfaces(path: Path) -> dict[str, int]:
    """Count Pi JSON-mode events whose serialised form carries the raw sentinel."""
    hits: dict[str, int] = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        if RAW not in line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            key = "unparsed"
        else:
            key = ev.get("type", "?")
            msg = ev.get("message")
            if isinstance(msg, dict):
                key += f":{msg.get('role')}"
            if key.startswith("message_") and isinstance(msg, dict) and msg.get("role") == "assistant":
                # The model's own tool-call arguments, not the tool result.
                key += ":tool_call_arguments"
        hits[key] = hits.get(key, 0) + 1
    return hits


def session_surfaces(session_dir: Path) -> dict:
    """Where the raw sentinel appears in the persisted session (by entry and field)."""
    files = sorted(session_dir.rglob("*.jsonl")) if session_dir.exists() else []
    out = {"session_files": len(files), "tool_result_entries": [], "raw_sentinel_in": []}
    for f in files:
        for entry in jsonl(f):
            msg = entry.get("message") if entry.get("type") == "message" else None
            if isinstance(msg, dict) and msg.get("role") == "toolResult":
                text = "".join(p.get("text", "") for p in msg.get("content", []) if isinstance(p, dict))
                out["tool_result_entries"].append(
                    {"content_text": text, "is_error": msg.get("isError"),
                     "details_has_raw": RAW in json.dumps(msg.get("details"))}
                )
            if RAW in json.dumps(entry):
                where = entry.get("type")
                if isinstance(msg, dict):
                    where += f":{msg.get('role')}"
                out["raw_sentinel_in"].append(where)
    return out


def prefix_check(records: list[dict]) -> dict | None:
    """C6: the first tool-result request's messages must reappear unchanged in the next."""
    if len(records) < 2:
        return None
    a, b = records[0]["message_digests"], records[1]["message_digests"]
    same = [i for i in range(min(len(a), len(b))) if a[i] == b[i]]
    return {
        "turn1_n_messages": len(a),
        "turn2_n_messages": len(b),
        "turn1_roles": records[0]["message_roles"],
        "turn2_roles": records[1]["message_roles"],
        "prefix_identical": b[: len(a)] == a,
        "differing_indices": [i for i in range(len(a)) if i not in same],
        "turn2_tool_messages": records[1]["tool_messages"],
    }


def parallel_check(side: list[dict], raw_log: Path) -> dict | None:
    """For a two-call batch: hook overlap, and admission order against source order.

    Only tool-call ids and their order are read from the raw mock log; no text leaves it.
    """
    hooks = [r for r in side if r["kind"] == "tool_result_hook"]
    calls = [r for r in side if r["kind"] == "tool_call_hook"]
    if len(calls) < 2:
        return None
    tag = {c["tool_call_id"]: c["args"].get("tag") for c in calls}
    source_order, admitted_order = [], []
    for line in raw_log.read_text().splitlines():
        msgs = json.loads(line)["body"].get("messages", [])
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        if tool_msgs:
            asst = next(m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls"))
            source_order = [tag.get(tc["id"]) for tc in asst["tool_calls"]]
            admitted_order = [tag.get(m.get("tool_call_id")) for m in tool_msgs]
            break
    iv = sorted((h["hook_start"], h["t"], tag.get(h["tool_call_id"])) for h in hooks)
    return {
        "tool_call_hook_order": [tag[c["tool_call_id"]] for c in calls],
        "hook_intervals": [{"tag": t, "start": a, "end": b} for a, b, t in iv],
        "hooks_overlap": len(iv) >= 2 and iv[1][0] < iv[0][1],
        "hook_completion_order": [tag.get(h["tool_call_id"]) for h in sorted(hooks, key=lambda h: h["t"])],
        "message_end_order": [tag.get(r["tool_call_id"]) for r in side if r["kind"] == "message_end_tool_result"],
        "assistant_tool_call_order_in_request": source_order,
        "tool_message_order_in_request": admitted_order,
        "admitted_in_source_order": bool(source_order) and admitted_order == source_order,
    }


def main() -> None:
    run, results = Path(sys.argv[1]), Path(sys.argv[2])
    logs = run / "logs"
    cases = sorted({p.name.split(".")[0] for p in logs.glob("*.mock.out")})
    summary = {}
    for case in cases:
        evidence = jsonl(results / f"{case}.evidence.jsonl")
        side = logs / f"{case}.side.jsonl"
        if side.exists():
            shutil.copyfile(side, results / f"{case}.side_channel.jsonl")
        first = evidence[0] if evidence else {}
        turns = jsonl(results / f"{case}.status.jsonl")
        summary[case] = {
            # "ok" only when every Pi turn exited 0 and reached the mock; a timeout,
            # error or no_request turn never counts as a pass.
            "status": "ok" if turns and all(t["status"] == "ok" for t in turns)
            else next((t["status"] for t in turns if t["status"] != "ok"), "missing"),
            "turns": turns,
            "mock_tool_messages_turn1": first.get("tool_messages"),
            "raw_sentinel_anywhere_in_request_turn1": first.get("raw_sentinel_anywhere_in_request"),
            "filtered_marker_anywhere_in_request_turn1": first.get("filtered_marker_anywhere_in_request"),
            "raw_sentinel_anywhere_in_request_all_turns": [r["raw_sentinel_anywhere_in_request"] for r in evidence],
            "pi_event_stream_raw_hits": {
                f"turn{n}": event_surfaces(logs / f"{case}.turn{n}.events.jsonl") for n in (1, 2)
                if (logs / f"{case}.turn{n}.events.jsonl").exists()
            },
            "session": session_surfaces(run / "sessions" / case),
            "c6_prefix": prefix_check(evidence),
            "parallel": parallel_check(jsonl(side), logs / f"{case}.raw.jsonl"),
            "pi_stderr_nonempty": any(
                p.stat().st_size > 0 for p in logs.glob(f"{case}.turn*.stderr")
            ),
        }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for case, s in summary.items():
        print(f"{case}: status={s['status']} tool_messages={s['mock_tool_messages_turn1']} "
              f"raw_anywhere={s['raw_sentinel_anywhere_in_request_turn1']} "
              f"session_raw={s['session']['raw_sentinel_in']} "
              f"c6={None if not s['c6_prefix'] else s['c6_prefix']['prefix_identical']}")


if __name__ == "__main__":
    main()
