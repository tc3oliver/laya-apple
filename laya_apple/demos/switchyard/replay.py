"""replay.html: one self-contained page that replays a recorded run.

`replay_data` turns result.json + trace.jsonl into the replay data JSON (the only interface
between the benchmark and the page; docs/switchyard.md). Times are milliseconds relative to
each round's `round_start_ns`, rounded to 3 decimals; trains and background requests are
sorted by arrival.

`bundle` fills static/index.html's three markers in one pass (replaced text is never
rescanned): `/*__SWITCHYARD_CSS__*/` with static/style.css, `/*__SWITCHYARD_JS__*/` with
static/app.js and `__SWITCHYARD_DATA__` with the data JSON (`</` escaped as `<\\/`). The page
needs no network and works from file://.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

from . import metrics, world

FORMAT, FORMAT_VERSION = "switchyard-replay", 1
CSS_MARKER, JS_MARKER, DATA_MARKER = "/*__SWITCHYARD_CSS__*/", "/*__SWITCHYARD_JS__*/", "__SWITCHYARD_DATA__"
_STAMPS = (
    ("arrival_ms", "arrival_ns"),
    ("queue_enter_ms", "queue_enter_ns"),
    ("dispatch_ms", "dispatch_ns"),
    ("service_start_ms", "service_start_ns"),
    ("service_end_ms", "service_end_ns"),
    ("response_ms", "response_ns"),
)


def _ms(ns: int) -> float:
    return round(ns / 1e6, 3)


def _times(rec: dict, start_ns: int) -> dict:
    return {key: _ms(rec[field] - start_ns) for key, field in _STAMPS}


def train_entry(rec: dict, start_ns: int) -> dict:
    return {
        "id": rec["train_id"],
        "line": rec["line"],
        "platforms": dict(zip(world.PLATFORMS, world.PATTERNS[rec["pattern"]])),
        "oracle": rec["oracle"],
        "answer": rec["answer"],
        "device": rec["target"],
        "outcome": rec["outcome"],
        **_times(rec, start_ns),
        "latency_ms": round(metrics.latency_ms(rec), 3),
        "queue_ms": round(metrics.queue_ms(rec), 3),
        "service_ms": round(metrics.service_ms(rec), 3),
    }


def background_entry(rec: dict, start_ns: int) -> dict:
    return {
        "class": rec["class"],
        "device": rec["target"],
        **_times(rec, start_ns),
        "latency_ms": round(metrics.latency_ms(rec), 3),
    }


def replay_data(result: dict, records: list[dict]) -> dict:
    by_round: dict = {}
    for rec in records:
        by_round.setdefault(rec["round"], []).append(rec)
    rounds = []
    for rnd in result["rounds"]:
        start = rnd["round_start_ns"]
        recs = sorted(by_round.get(rnd["index"], []), key=lambda r: (r["arrival_ns"], r["index"]))
        rounds.append(
            {
                "index": rnd["index"],
                "config": rnd["config"],
                "duration_s": result["workload"]["duration_s"],
                "deadline_ms": result["workload"]["deadline_ms"],
                "trains": [train_entry(r, start) for r in recs if r["class"] == world.TRAIN],
                "background": [background_entry(r, start) for r in recs if r["class"] != world.TRAIN],
            }
        )
    return {"format": FORMAT, "format_version": FORMAT_VERSION, "result": result, "rounds": rounds}


def static_files(static_dir: Path | None = None) -> tuple[str, str, str]:
    """(index.html, style.css, app.js) from `static_dir` or the installed package."""
    base = Path(static_dir) if static_dir is not None else resources.files("laya_apple.demos.switchyard") / "static"
    return tuple((base / name).read_text(encoding="utf-8") for name in ("index.html", "style.css", "app.js"))


def bundle(data: dict, static_dir: Path | None = None) -> str:
    template, css, js = static_files(static_dir)
    for marker in (CSS_MARKER, JS_MARKER, DATA_MARKER):
        if template.count(marker) != 1:
            raise ValueError(f"static/index.html must contain {marker} exactly once")
    if "</style" in css.lower() or "</script" in js.lower():
        raise ValueError("static/style.css or static/app.js closes its own <style>/<script> element")
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    parts = {CSS_MARKER: css, JS_MARKER: js, DATA_MARKER: payload}
    pattern = "|".join(re.escape(m) for m in parts)
    return re.sub(pattern, lambda m: parts[m.group(0)], template)


def write(out_dir: Path, result: dict, records: list[dict], static_dir: Path | None = None) -> Path:
    from .result import REPLAY_FILE

    path = Path(out_dir) / REPLAY_FILE
    path.write_text(bundle(replay_data(result, records), static_dir), encoding="utf-8")
    return path
