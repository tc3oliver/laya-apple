"""results.json and tables.md from raw/ (no hardware; exactly the rules in criteria.md).

    uv run python research/ane-windowed-attention/scripts/analyze.py

Per cell, four separate results for the windowed artifact, never merged into one pass/fail:
  placement  (build.json: check_ane_placement, unchanged)
  parity     (build.json: production ane_parity summary, unchanged gate)
  probe      (latency.json: probe_placement on the windowed artifact, unchanged)
  latency    (latency.json: paired gate on predict P50, windowed / masked <= IMPROVEMENT_LIMIT)
A cell is a graph-change candidate only if all four are PASS. At 256/512 the latency result
also needs a valid baseline: the research masked build must itself pass placement, parity and
the probe, otherwise the latency result is INCONCLUSIVE (no valid baseline).
Descriptive only: forward-boundary ratio, first-load time, MIL op count, the dense score
fraction, and at 256/512 the windowed (and masked) ANE forward vs MLX FP16 forward.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CELLS, LONG, RAW, cell_name, item_key  # noqa: E402
from paired import FAIL, INCONCLUSIVE, PASS, bootstrap_interval, t_interval, upper_verdict  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
IMPROVEMENT_LIMIT = 0.95  # criteria.md: windowed predict P50 must be <= 0.95x the masked graph's


def _softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def per_row(model: str, length: int, rows_path: Path, tol: float) -> list[dict]:
    """Per-row parity metrics recomputed from the raw outputs, same formulas as parity.evaluate."""
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


def build_result(model: str, length: int, variant: str) -> dict | None:
    d = RAW / model / cell_name(length, variant)
    if not (d / "build.json").exists():
        return None
    b = json.loads((d / "build.json").read_text())
    res = {
        "step": b["step"],
        "error": b.get("error"),
        "placement": b.get("placement_gate"),
        "placement_summary": b.get("placement"),
        "mil_ops": b.get("mil_ops"),
        "first_load_s": b["timings"].get("first_load_s"),
        "layout_check_max_abs_logit_fp32": b.get("layout_check_max_abs_logit_fp32"),
        "exactness_fp32_vs_masked": b.get("exactness_fp32_vs_masked"),
        "dense_score_fraction": b.get("dense_score_fraction"),
        "parity": None,
    }
    if "parity" in b:
        p = b["parity"]
        rows = per_row(model, length, d / "parity_rows.jsonl", p["tolerance"])
        res["parity"] = PASS if p["passed"] else FAIL
        res["parity_summary"] = {k: p[k] for k in p if k != "recording_pass"}
        res["parity_rows"] = rows
        res["parity_recomputed_matches"] = bool(
            len(rows) == p["rows"]
            and sum(r["hard"] for r in rows) == p["hard_mismatches"]
            and abs(max(r["prob_max_abs"] for r in rows) - p["prob_max_abs"]) <= 1e-9
            and p.get("recording_pass_agrees", False)
        )
    return res


def latency_stats(lat: dict) -> dict:
    by = {}
    for w in lat["windows"]:
        by.setdefault(w["cycle"], {})[w["arm"]] = w
    cycles = sorted(by)
    if any(not {"baseline", "candidate"} <= set(by[c]) for c in cycles):
        raise ValueError("a cycle is missing an ANE arm")
    pred = [by[c]["candidate"]["predict_p50_ms"] / by[c]["baseline"]["predict_p50_ms"] for c in cycles]
    fwd = [by[c]["candidate"]["forward_p50_ms"] / by[c]["baseline"]["forward_p50_ms"] for c in cycles]
    ci = t_interval(pred)
    out = {
        "pairs": len(cycles),
        "predict_ratios": pred,
        "predict_t": ci,
        "predict_bootstrap": bootstrap_interval(pred),
        "forward_ratios": fwd,
        "forward_t": t_interval(fwd),
        "verdict": upper_verdict(ci, IMPROVEMENT_LIMIT),
        "bootstrap_sensitivity": upper_verdict(bootstrap_interval(pred), IMPROVEMENT_LIMIT),
    }
    for arm in ("baseline", "candidate", "mlx"):
        if all(arm in by[c] for c in cycles):
            out[f"{arm}_forward_p50_ms"] = float(np.median([by[c][arm]["forward_p50_ms"] for c in cycles]))
            if arm != "mlx":
                out[f"{arm}_predict_p50_ms"] = float(np.median([by[c][arm]["predict_p50_ms"] for c in cycles]))
    if all("mlx" in by[c] for c in cycles):  # descriptive: ANE forward / MLX forward, per cycle
        for arm in ("candidate", "baseline"):
            r = [by[c][arm]["forward_p50_ms"] / by[c]["mlx"]["forward_p50_ms"] for c in cycles]
            out[f"{arm}_vs_mlx_forward"] = {"ratios": r, "t": t_interval(r)}
    return out


def cell(model: str, length: int) -> dict | None:
    w = build_result(model, length, "windowed")
    if w is None:
        return None
    res = {"model": model, "length": length, "issue": "#14" if length in LONG else "#15", "windowed": w}
    baseline_valid = True
    if length in LONG:
        m = build_result(model, length, "masked")
        res["masked_baseline"] = m
        baseline_valid = bool(m and m["placement"] == PASS and m["parity"] == PASS)
    lat_path = RAW / model / cell_name(length, "windowed") / "latency.json"
    res["probe"] = res["latency_verdict"] = None
    if lat_path.exists():
        lat = json.loads(lat_path.read_text())
        res["probe"] = lat["candidate"].get("probe_gate")
        res["probe_detail"] = {"candidate": lat["candidate"].get("probe"), "baseline": lat["baseline"].get("probe")}
        baseline_valid = baseline_valid and lat["baseline"].get("probe_gate") == PASS
        res["latency"] = latency_stats(lat)
        res["latency_verdict"] = res["latency"]["verdict"] if baseline_valid else INCONCLUSIVE
        res["baseline_valid"] = baseline_valid
    dims = [w["placement"], w["parity"], res["probe"], res["latency_verdict"]]
    if all(x == PASS for x in dims):
        res["graph_change_candidate"] = "yes"
    elif any(x == FAIL for x in dims) or w["error"]:
        res["graph_change_candidate"] = "no (documented)"
    elif any(x == INCONCLUSIVE for x in dims):
        res["graph_change_candidate"] = "inconclusive"
    else:
        res["graph_change_candidate"] = "incomplete"
    return res


def fmt_ci(t):
    return f"{t['geomean']:.3f} [{t['lo']:.3f}, {t['hi']:.3f}]"


def main() -> None:
    results = [c for m in CELLS for L in CELLS[m] if (c := cell(m, L)) is not None]
    (HERE / "results.json").write_text(
        json.dumps({"improvement_limit": IMPROVEMENT_LIMIT, "cells": results}, indent=1) + "\n"
    )
    head = (
        "| model | L | issue | dense score share | placement | parity | prob max | hard | flips | probe | "
        "predict P50 windowed/masked [95% CI] | latency | P50 ms masked → windowed | first load s | candidate |"
    )
    lines = ["# Windowed attention: per-cell results", "", "Generated by `scripts/analyze.py` from `raw/`.", "", head]
    lines.append("|---|---:|---|---:|---|---|---:|---:|---:|---|---|---|---|---:|---|")
    for r in results:
        w, lat = r["windowed"], r.get("latency")
        ps = w.get("parity_summary") or {}
        p50 = f"{lat['baseline_predict_p50_ms']:.2f} → {lat['candidate_predict_p50_ms']:.2f}" if lat else "–"
        lines.append(
            f"| {r['model']} | {r['length']} | {r['issue']} | {w['dense_score_fraction'] or float('nan'):.3f} | "
            f"{w['placement']} | {w['parity']} | {ps.get('prob_max_abs', float('nan')):.4f} | "
            f"{ps.get('hard_mismatches', '–')} | {len(ps.get('near_tie_flips', []))} | {r['probe']} | "
            f"{fmt_ci(lat['predict_t']) if lat else '–'} | {r['latency_verdict']} | {p50} | "
            f"{w['first_load_s'] or float('nan'):.0f} | {r['graph_change_candidate']} |"
        )
    lines += ["", "## 256/512 against MLX FP16 (descriptive, not gated)", ""]
    lines.append("| model | L | ANE masked fwd P50 | ANE windowed fwd P50 | MLX fwd P50 | windowed / MLX [95% CI] |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in results:
        lat = r.get("latency") or {}
        if "candidate_vs_mlx_forward" in lat:
            lines.append(
                f"| {r['model']} | {r['length']} | {lat['baseline_forward_p50_ms']:.2f} | "
                f"{lat['candidate_forward_p50_ms']:.2f} | {lat['mlx_forward_p50_ms']:.2f} | "
                f"{fmt_ci(lat['candidate_vs_mlx_forward']['t'])} |"
            )
    lines += ["", "## Near-tie flips (listed, not failed)", ""]
    for r in results:
        for f in (r["windowed"].get("parity_summary") or {}).get("near_tie_flips", []):
            lines.append(f"- {r['model']} L{r['length']} windowed: {f}")
    (HERE / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
