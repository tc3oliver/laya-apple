"""What each instrumentation layer costs, and what tracing changes under load (ledger/overhead/).

    uv run python research/gpu-ane-interference/scripts/overhead.py [--check]

Reads ledger/overhead/*.json.gz and the traced product run in ledger/raw/, writes
ledger/overhead.json and prints its tables. Only the client's own clock is used here
(arrival to completion as the workload generator saw it), because the trace-off runs have
no runtime trace:

  solo-*-{hooks,trace,off}   GPU L128 and ANE L128 alone, closed loop: with runtime tracing and
                             the research backend hooks, with runtime tracing only, and with
                             neither. trace - off is the runtime tracing cost; hooks - trace is
                             the research instrumentation cost.
  product-*-off vs raw/product-*  the router under open-loop load with everything off and on:
                             device and routing-reason shares, answer mismatches, client e2e.
                             Routing under load depends on timing, so the comparison is of
                             distributions, not request by request.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "ledger"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import load, rows  # noqa: E402


def client_records(run: dict):
    """(cell, stream) -> in-window generator rows, each with client e2e in ms."""
    out = defaultdict(list)
    for w in run["windows"]:
        for label, st in w["streams"].items():
            for r in rows(st["records"]):
                if r["in_window"]:
                    r["e2e_ms"] = (r["done_us"] - r["arrival_us"]) / 1e3
                    out[(w["cell"], label)].append(r)
    return out, {w["cell"]: w for w in run["windows"]}


def span_s(run: dict, cell: str) -> float:
    return sum(w["measure"][1] - w["measure"][0] for w in run["windows"] if w["cell"] == cell)


def stats(recs, seconds) -> dict:
    a = np.array([r["e2e_ms"] for r in recs])
    return {
        "n": int(a.size),
        "req_s": a.size / seconds,
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p99_ms": float(np.percentile(a, 99)),
        "mismatches": sum(not r["match"] for r in recs),
    }


def solo_section(paths: dict) -> dict:
    res = {}
    for variant, path in sorted(paths.items()):
        run = load(path)
        recs, _ = client_records(run)
        res[variant] = {f"{c}|{s}": stats(v, span_s(run, c)) for (c, s), v in sorted(recs.items())}
    base = res.get("off", {})
    for variant, cells in res.items():
        for k, v in cells.items():
            if k in base:
                v["mean_vs_off"] = v["mean_ms"] / base[k]["mean_ms"] - 1
                v["req_s_vs_off"] = v["req_s"] / base[k]["req_s"] - 1
    return res


def product_section(on_path: Path, off_path: Path) -> dict:
    res = {}
    for label, path in (("on", on_path), ("off", off_path)):
        run = load(path)
        recs, _ = client_records(run)
        by_cell = defaultdict(list)
        for (cell, _), v in recs.items():
            by_cell[cell] += v
        res[label] = {}
        for cell, v in sorted(by_cell.items()):
            n = len(v)
            res[label][cell] = {
                **stats(v, span_s(run, cell)),
                "devices": {k: c / n for k, c in sorted(Counter(r["device"] for r in v).items())},
                "reasons": {k: c / n for k, c in sorted(Counter(r["reason"] for r in v).items())},
            }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    solo = {p.name.split("-")[-1].split(".")[0]: p for p in sorted((ROOT / "overhead").glob("solo-*.json.gz"))}
    out = {"solo": solo_section(solo)}
    for off in sorted((ROOT / "overhead").glob("product-*-off.json.gz")):
        on = ROOT / "raw" / off.name.replace("-off", "")
        if on.exists():
            out[f"product:{on.stem.split('.')[0]}"] = product_section(on, off)
    text = json.dumps(out, indent=1, sort_keys=True) + "\n"
    target = ROOT / "overhead.json"
    if args.check:
        if not target.exists() or target.read_text() != text:
            raise SystemExit(f"{target} is out of date: re-run overhead.py")
        return
    target.write_text(text)
    for variant, cells in out["solo"].items():
        for k, v in cells.items():
            print(variant, k, {m: round(x, 4) if isinstance(x, float) else x for m, x in v.items()})
    for key, v in out.items():
        if key.startswith("product:"):
            for label, cells in v.items():
                for cell, s in cells.items():
                    print(key, label, cell, round(s["req_s"], 1), round(s["p99_ms"], 1), s["mismatches"], s["reasons"])


if __name__ == "__main__":
    main()
