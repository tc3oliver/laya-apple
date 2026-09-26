"""results.json and tables.md from raw/ (no hardware; exactly the rules in criteria.md).

    uv run python research/ane-w8/scripts/analyze.py

Per cell and config, four separate results, never merged into one pass/fail:
  placement  (build.json: check_ane_placement, unchanged)
  parity     (build.json: production ane_parity summary, unchanged gate)
  probe      (latency.json: probe_placement on the candidate, unchanged)
  latency    (latency.json: paired gate on predict P50, W8 / FP16 <= LATENCY_LIMIT)
plus size (reported, not gated). A cell is a ship candidate only if all four are PASS.
Per-row parity metrics are recomputed from the recorded raw outputs with the same formulas
as laya_apple.parity.evaluate and must reproduce the summary of record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CELLS, RAW, cell_name, item_key  # noqa: E402
from paired import FAIL, INCONCLUSIVE, PASS, bootstrap_interval, t_interval, upper_verdict  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
LATENCY_LIMIT = 1.05  # criteria.md: W8 predict P50 must not exceed 1.05x the FP16 artifact's
CONFIGS = ("w8-pt", "w8-gc32")


def _softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def per_row(model: str, length: int, rows_path: Path, tol: float) -> list[dict]:
    from laya_apple.hub import checkpoint_path
    from laya_apple.parity import load_goldens
    from laya_apple.prompt import Calibration
    from laya_apple.registry import resolve

    cfg = json.loads((checkpoint_path(resolve(model), local_files_only=True) / "rl_agent_config.json").read_text())
    calib = Calibration(cfg)
    got = {}
    for line in rows_path.read_text().splitlines():
        r = json.loads(line)
        got[r["key"]] = r
    out = []
    for case in load_goldens(model)["cases"]:
        for i, it in enumerate(case["items"]):
            if len(it["ids"]) > length:
                continue
            g = got[item_key(it)]
            k, qt = len(it["markers"]), it["qtype"]
            rl = np.asarray(case["logits"][i], np.float64)
            gl = np.asarray(g["logits"][:k], np.float64)
            rp, gp = calib.probabilities(rl, qt, k), calib.probabilities(gl, qt, k)
            srt = np.sort(rp)
            margin = float(srt[-1] - srt[-2]) if k > 1 else 1.0
            mismatch = int(gp.argmax()) != int(rp.argmax())
            out.append(
                {
                    "case": case["name"],
                    "row": i,
                    "tokens": len(it["ids"]),
                    "prob_max_abs": float(np.abs(gp - rp).max()),
                    "action_prob_max_abs": float(
                        np.abs(_softmax(g["action_logits"]) - _softmax(case["action_logits"][i])).max()
                    ),
                    "ref_margin": margin,
                    "mismatch": mismatch,
                    "hard": mismatch and margin >= 2 * tol,
                }
            )
    return out


def latency_stats(lat: dict) -> dict:
    by = {}
    for w in lat["windows"]:
        by.setdefault(w["cycle"], {})[w["arm"]] = w
    cycles = sorted(by)
    if any(set(by[c]) != {"baseline", "candidate"} for c in cycles):
        raise ValueError("a cycle is missing an arm")
    pred = [by[c]["candidate"]["predict_p50_ms"] / by[c]["baseline"]["predict_p50_ms"] for c in cycles]
    fwd = [by[c]["candidate"]["forward_p50_ms"] / by[c]["baseline"]["forward_p50_ms"] for c in cycles]
    ci = t_interval(pred)
    return {
        "pairs": len(cycles),
        "predict_ratios": pred,
        "predict_t": ci,
        "predict_bootstrap": bootstrap_interval(pred),
        "forward_ratios": fwd,
        "forward_t": t_interval(fwd),
        "baseline_predict_p50_ms": float(np.median([by[c]["baseline"]["predict_p50_ms"] for c in cycles])),
        "candidate_predict_p50_ms": float(np.median([by[c]["candidate"]["predict_p50_ms"] for c in cycles])),
        "verdict": upper_verdict(ci, LATENCY_LIMIT),
        "bootstrap_sensitivity": upper_verdict(bootstrap_interval(pred), LATENCY_LIMIT),
    }


def cell(model: str, length: int, config: str) -> dict | None:
    d = RAW / model / cell_name(length, config)
    if not (d / "build.json").exists():
        return None
    b = json.loads((d / "build.json").read_text())
    base = json.loads((RAW / model / f"L{length}-fp16" / "baseline_size.json").read_text())
    res = {"model": model, "length": length, "config": config, "build_step": b["step"], "error": b.get("error")}
    res["placement"] = b.get("placement_gate", FAIL if b["step"] in ("placement",) else None)
    res["placement_summary"] = b.get("placement")
    if "parity" in b:
        p = b["parity"]
        rows = per_row(model, length, d / "parity_rows.jsonl", p["tolerance"])
        recomputed = {
            "rows": len(rows),
            "prob_max_abs": max(r["prob_max_abs"] for r in rows),
            "hard_mismatches": sum(r["hard"] for r in rows),
        }
        res["parity"] = PASS if p["passed"] else FAIL
        res["parity_summary"] = {k: p[k] for k in p if k not in ("recording_pass",)}
        res["parity_rows"] = rows
        res["parity_recomputed_matches"] = bool(
            recomputed["rows"] == p["rows"]
            and recomputed["hard_mismatches"] == p["hard_mismatches"]
            and abs(recomputed["prob_max_abs"] - p["prob_max_abs"]) <= 1e-9
            and p.get("recording_pass_agrees", False)
        )
    else:
        res["parity"] = None
    res["size"] = {
        "fp16_weight_bin_bytes": base["weight_bin_bytes"],
        "w8_weight_bin_bytes": b.get("weight_bin_bytes"),
        "weight_ratio": (b["weight_bin_bytes"] / base["weight_bin_bytes"]) if b.get("weight_bin_bytes") else None,
        "fp16_compiled_bytes": base["compiled_bytes"],
        "w8_compiled_bytes": b.get("compiled_bytes"),
        "compiled_ratio": (b["compiled_bytes"] / base["compiled_bytes"]) if b.get("compiled_bytes") else None,
    }
    lat_path = d / "latency.json"
    if lat_path.exists():
        lat = json.loads(lat_path.read_text())
        res["probe"] = lat["candidate"].get("probe_gate")
        res["probe_detail"] = {"candidate": lat["candidate"].get("probe"), "baseline": lat["baseline"].get("probe")}
        res["latency"] = latency_stats(lat)
        res["latency_verdict"] = res["latency"]["verdict"]
    else:
        res["probe"] = res["latency_verdict"] = None
    dims = [res["placement"], res["parity"], res["probe"], res["latency_verdict"]]
    if all(x == PASS for x in dims):
        res["ship_candidate"] = "yes"
    elif any(x == FAIL for x in dims) or res["error"]:
        res["ship_candidate"] = "no (documented no-ship)"
    elif any(x == INCONCLUSIVE for x in dims):
        res["ship_candidate"] = "inconclusive"
    else:
        res["ship_candidate"] = "incomplete"
    return res


def fmt_ci(t):
    return f"{t['geomean']:.3f} [{t['lo']:.3f}, {t['hi']:.3f}]"


def main() -> None:
    results = [c for m in CELLS for L in CELLS[m] for cfg in CONFIGS if (c := cell(m, L, cfg)) is not None]
    (HERE / "results.json").write_text(json.dumps({"latency_limit": LATENCY_LIMIT, "cells": results}, indent=1) + "\n")
    lines = [
        "# W8 palettization: per-cell results",
        "",
        "Generated by `scripts/analyze.py` from `raw/`. Rules: `criteria.md`.",
        "",
        "| model | L | config | placement | parity | prob max | hard | near-tie flips | probe ratio | probe | "
        "predict P50 ratio W8/FP16 [95% CI] | latency | weights W8/FP16 | ship candidate |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---|---|---|---:|---|",
    ]
    for r in results:
        ps = r.get("parity_summary") or {}
        lat = r.get("latency")
        probe = (r.get("probe_detail") or {}).get("candidate") or {}
        lines.append(
            f"| {r['model']} | {r['length']} | {r['config']} | {r['placement']} | {r['parity']} | "
            f"{ps.get('prob_max_abs', float('nan')):.4f} | {ps.get('hard_mismatches', '–')} | "
            f"{len(ps.get('near_tie_flips', []))} | {probe.get('ratio', '–')} | {r['probe']} | "
            f"{fmt_ci(lat['predict_t']) if lat else '–'} | {r['latency_verdict']} | "
            f"{(r['size']['weight_ratio'] or float('nan')):.3f} | {r['ship_candidate']} |"
        )
    lines += ["", "## Near-tie flips (listed, not failed)", ""]
    for r in results:
        for f in (r.get("parity_summary") or {}).get("near_tie_flips", []):
            lines.append(f"- {r['model']} L{r['length']} {r['config']}: {f}")
    (HERE / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
