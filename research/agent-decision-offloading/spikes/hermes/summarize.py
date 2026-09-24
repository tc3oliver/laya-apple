"""Summarise the Hermes sentinel spike into ``results/summary.json``.

Reads the committed evidence (``results/<case>.evidence.jsonl``) plus, from scratch, the
mock's raw request log, the hook's side-channel file and the case's Hermes session
database. Only derived booleans, counts and digests leave scratch; the raw request log
(Hermes's system prompt and tool schemas) is never copied. The side-channel records are
the spike's own data and are copied with scratch paths replaced.

    python summarize.py <results-dir> <scratch-dir>
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

RAW = "LAYA_SENTINEL"
FILTERED = "AAA [FILTERED_BY_LAYA_SPIKE] BBB"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _db_hits(db: Path) -> dict:
    """Count rows in every text column of the Hermes session DB that hold each marker."""
    out = {"raw_sentinel_rows": 0, "filtered_rows": 0, "tables_scanned": 0}
    if not db.exists():
        out["error"] = "state.db not found"
        return out
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
        for table in tables:
            try:
                rows = con.execute(f'select * from "{table}"').fetchall()
            except sqlite3.Error:
                continue
            out["tables_scanned"] += 1
            for row in rows:
                text = " ".join(str(v) for v in row if isinstance(v, (str, bytes)))
                out["raw_sentinel_rows"] += RAW in text
                out["filtered_rows"] += "[FILTERED_BY_LAYA_SPIKE]" in text
    finally:
        con.close()
    return out


def _home_file_hits(home: Path) -> list[str]:
    """Files under the case's HERMES_HOME (plugins excluded) whose bytes hold the raw sentinel."""
    hits = []
    for f in sorted(home.rglob("*")):
        if not f.is_file() or "plugins" in f.relative_to(home).parts:
            continue
        try:
            if RAW.encode() in f.read_bytes():
                hits.append(str(f.relative_to(home)))
        except OSError:
            continue
    return hits


def _case_dir(name: str, scratch: Path) -> Path:
    """The newest ``runs/<stamp>/<name>`` that exists (``run.sh`` never deletes old runs)."""
    runs = sorted((scratch / "runs").glob(f"*/{name}")) if (scratch / "runs").exists() else []
    return runs[-1] if runs else scratch / "cases" / name


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content if isinstance(content, str) else ""


def _notifications(chat: list[dict], side: list[dict], scratch: Path) -> list[dict]:
    """Background-process notifications as the model received them: the first request that
    carried each one, its text (Hermes-formatted, holding only the spike's own command and
    output) and whether that request held the raw sentinel anywhere."""
    seen, out = set(), []
    for i, r in enumerate(chat):
        for m in r["body"].get("messages", []):
            text = _text(m.get("content"))
            if m.get("role") == "user" and "Background process" in text and text not in seen:
                seen.add(text)
                hooks = [h["t"] for h in side
                         if h.get("event") == "transform_terminal_output" and h.get("matched")]
                out.append({
                    "first_request_index": i,
                    "role": m.get("role"),
                    "text": text.replace(str(scratch), "$SCRATCH")[:800],
                    "raw_sentinel_in_text": RAW in text,
                    "raw_sentinel_anywhere_in_request": RAW in json.dumps(r["body"]),
                    # The first request carrying the notification is its first admission.
                    "terminal_output_hook_ran_before_first_admission":
                        bool(hooks) and min(hooks) < r["t"],
                })
    return out


def _agent_log(home: Path, scratch: Path) -> dict:
    """Counts of the Hermes log lines the README relies on, plus the first of each (with
    its timestamp prefix dropped and scratch paths replaced)."""
    patterns = {
        "hook_timeouts": "callback _on_transform_tool_result timed out",
        "concurrent_invoke_tool_raised": "_invoke_tool raised for",
    }
    log = home / "logs" / "agent.log"
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    out = {}
    for key, needle in patterns.items():
        hits = [line for line in lines if needle in line]
        first = hits[0].split(" ", 2)[-1].replace(str(scratch), "$SCRATCH") if hits else None
        out[key] = {"count": len(hits), "first_line": first}
    return out


def _parallel_timing(side: list[dict]) -> dict | None:
    """Two concurrent transform_tool_result calls: each await, and how far apart they ended."""
    hooks = [r for r in side if r.get("event") == "transform_tool_result" and r.get("matched")]
    if len(hooks) < 2:
        return None
    ends = sorted(r["t"] for r in hooks)
    return {"await_ms": [r.get("await_ms") for r in hooks],
            "completion_gap_us": round((ends[-1] - ends[0]) * 1e6, 1)}


def summarize_case(name: str, results: Path, scratch: Path) -> dict:
    case_dir = _case_dir(name, scratch)
    evidence = _jsonl(results / f"{name}.evidence.jsonl")
    raw = _jsonl(case_dir / "raw.jsonl")
    side = _jsonl(case_dir / "sidechannel.jsonl")

    chat = [r for r in raw if r["path"].endswith("/chat/completions")]
    raw_hits = [i for i, r in enumerate(chat) if RAW in json.dumps(r["body"])]

    c6 = None
    if len(evidence) >= 2:
        # Every later tool-result request must start with the earlier one's messages unchanged.
        pairs = []
        for a, b in zip(evidence, evidence[1:]):
            n = a["n_messages"]
            pairs.append({"from_roles": a["message_roles"], "to_roles": b["message_roles"],
                          "prefix_identical": a["message_digests"] == b["message_digests"][:n]})
        first, last = evidence[0], evidence[-1]
        n = first["n_messages"]
        c6 = {
            "turn1_roles": first["message_roles"],
            "turn2_roles": last["message_roles"],
            "prefix_len_compared": n,
            "prefix_identical": all(p["prefix_identical"] for p in pairs),
            "per_message_equal": [a == b for a, b in
                                  zip(first["message_digests"], last["message_digests"][:n])],
            "consecutive_requests": pairs,
            "turn2_tool_messages": last["tool_messages"],
        }

    sanitized_side = []
    for rec in side:
        text = json.dumps(rec).replace(str(scratch), "$SCRATCH")
        sanitized_side.append(json.loads(text))
    if sanitized_side:
        (results / f"{name}.sidechannel.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in sanitized_side))

    return {
        "tool_result_turns": [
            {"roles": e["message_roles"], "tool_messages": e["tool_messages"],
             "raw_sentinel_anywhere_in_request": e["raw_sentinel_anywhere_in_request"],
             "filtered_marker_anywhere_in_request": e["filtered_marker_anywhere_in_request"]}
            for e in evidence
        ],
        "chat_requests_total": len(chat),
        "chat_requests_with_raw_sentinel": len(raw_hits),
        "c6_history": c6,
        "background_notifications": _notifications(chat, side, scratch),
        "hook_invocations": [
            {k: r[k] for k in ("event", "tool_name", "matched", "await_ms", "thread", "status",
                               "result_has_raw_sentinel", "history_roles", "outcome",
                               "elapsed_ms", "deadline_s", "decision_s", "returncode",
                               "exception_type", "raw_exception", "replacement") if k in r}
            for r in sanitized_side
        ],
        "session_db": _db_hits(case_dir / "home" / "state.db"),
        "hermes_home_files_with_raw_sentinel": _home_file_hits(case_dir / "home"),
        "agent_log": _agent_log(case_dir / "home", scratch),
        "parallel_timing": _parallel_timing(sanitized_side),
    }


def main():
    results, scratch = Path(sys.argv[1]), Path(sys.argv[2])
    cases = sorted(p.name[: -len(".evidence.jsonl")] for p in results.glob("*.evidence.jsonl"))
    src = scratch / "src"
    head = (src / ".git" / "HEAD").read_text().strip() if (src / ".git" / "HEAD").exists() else "?"
    summary = {"hermes_checkout_head": head,
               "cases": {c: summarize_case(c, results, scratch) for c in cases}}
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for c, s in summary["cases"].items():
        turns = s["tool_result_turns"]
        print(f"{c}: tool_messages={[t['tool_messages'] for t in turns]} "
              f"raw_in_request={[t['raw_sentinel_anywhere_in_request'] for t in turns]} "
              f"raw_in_any_chat_request={s['chat_requests_with_raw_sentinel']}/{s['chat_requests_total']} "
              f"c6_prefix_identical={(s['c6_history'] or {}).get('prefix_identical')} "
              f"db_raw_rows={s['session_db'].get('raw_sentinel_rows')}")
        for n in s["background_notifications"]:
            print(f"    notification[{n['role']}] raw_in_text={n['raw_sentinel_in_text']} "
                  f"raw_in_request={n['raw_sentinel_anywhere_in_request']}: {n['text'][:200]!r}")


if __name__ == "__main__":
    main()
