"""Serve decisions beside a local LLM: results.json and tables.md for one campaign.

    uv run python benchmarks/serve/analyze.py benchmarks/serve/<campaign> [--check] [--criteria FILE]

Inputs: <campaign>/raw/ from scripts/bench_serve.py. The criteria are the campaign's own copy,
<campaign>/criteria.json, which run.sh writes before the campaign starts; a campaign without
one (run 1, m4-max) uses criteria.json next to this file. Both were committed before the data
they apply to; this file implements them. Throughput, latency and correctness are reported as
separate results, with no overall PASS/FAIL.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("bench_serve", ROOT / "scripts" / "bench_serve.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

CLASSES = bench.CLASSES


def load_windows(raw: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(raw.glob("[0-9][0-9][0-9]-*.json"))]


def load_refs(raw: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(raw.glob("refs-step*.json"))]


def decision_metrics(w: dict) -> dict:
    recs = [r for r in w.get("decisions", []) if not r.get("warmup")]
    out = {"lag_p99_ms": bench.percentile([(r["sent_ns"] - r["sched_ns"]) / 1e6 for r in recs], 99), "classes": {}}
    for cls in CLASSES:
        rs = [r for r in recs if r["cls"] == cls]
        ok = [r for r in rs if r.get("status") == 200]
        s = bench.latency_stats([(r["done_ns"] - r["sched_ns"]) / 1e6 for r in ok])
        devices: dict = {}
        for r in ok:
            devices[r.get("device")] = devices.get(r.get("device"), 0) + 1
        s.update(
            offered=len(rs),
            errors=len(rs) - len(ok),
            hard_mismatches=sum(len(r.get("hard") or []) for r in ok),
            prob_err_max=max((r.get("prob_err", 0.0) for r in ok), default=0.0),
            act_err_max=max((r.get("act_err", 0.0) for r in ok), default=0.0),
            near_tie_flips=[
                {"window": w["index"], "variant": r["variant"], "questions": r["flips"]} for r in ok if r.get("flips")
            ],
            device_changed=sum(r.get("device") != r.get("ref_device") for r in ok),
            devices=devices,
        )
        out["classes"][cls] = s
    return out


def llm_metrics(w: dict) -> dict | None:
    if "llm" not in w:
        return None
    start, end = w["window_ns"]
    m = bench.llm_window_summary(w["llm"], start, end)
    rates = [
        (r.get("usage") or {}).get("generation_tokens_per_second")
        for r in w["llm"]["requests"]
        if (r.get("usage") or {}).get("generation_tokens_per_second")
    ]
    m["request_decode_tok_s_median"] = float(np.median(rates)) if rates else None
    # server-side time to first token: prefill (and the queue behind the other stream)
    ttft = [(r.get("usage") or {}).get("time_to_first_token") for r in w["llm"]["requests"]]
    ttft = [t for t in ttft if t is not None]
    m["server_ttft_s_median"] = float(np.median(ttft)) if ttft else None
    m["drain_aborted"] = w["llm"].get("drain_aborted", 0)
    return m


def exclusive(w: dict) -> dict:
    """V3: the LLM server idle at window start and no request but ours during the window."""
    b, a = w.get("llm_status_before") or {}, w.get("llm_status_after") or {}
    ours = len(w["llm"]["requests"]) if "llm" in w else 0
    if "total_requests" not in b or "total_requests" not in a:
        return {"checked": False, "ok": None}
    delta = a["total_requests"] - b["total_requests"]
    idle = b.get("active_requests") == 0 and b.get("waiting_requests") == 0
    return {"checked": True, "ok": bool(idle and delta == ours), "delta": delta, "ours": ours, "idle_at_start": idle}


def busy_coverage(w: dict) -> float:
    """V2 (run 2): fraction of the window with at least one harness LLM request in flight at the
    server, each request from its start to its last chunk, so prefill counts as busy."""
    start, end = w["window_ns"]
    spans = [(r["start_ns"], r["chunks"][-1][0]) for r in w["llm"]["requests"] if r.get("chunks") and "start_ns" in r]
    return bench.coverage(spans, start, end)


def short_routing(w: dict, spill_reason: str) -> dict:
    """V4 (run 2): how an auto window's short_1q answers were routed.

    policy_share counts the product policy's short path: the ANE, or the GPU with the
    scheduler's backlog spill reason. spill_share (not gated) is the spilled part alone."""
    ok = [r for r in w["decisions"] if not r.get("warmup") and r["cls"] == "short_1q" and r.get("status") == 200]
    ane = sum(r.get("device") == "ane" for r in ok)
    spill = sum(r.get("device") == "gpu" and r.get("reason") == spill_reason for r in ok)
    other: dict = {}
    for r in ok:
        if r.get("device") != "ane" and not (r.get("device") == "gpu" and r.get("reason") == spill_reason):
            k = f"{r.get('device')}:{r.get('reason')}"
            other[k] = other.get(k, 0) + 1
    n = len(ok)
    return {
        "index": w["index"],
        "n": n,
        "policy_share": (ane + spill) / n if n else None,
        "spill_share": spill / n if n else None,
        "other": other,
    }


def adaptive_execution(windows: list[dict], refs: list[dict], model: str) -> dict:
    """Run 3, recorded and not gated: the adaptive-execution state per auto decision window
    (bench_serve.handoff_delta, from serve /health before and after it), totals per auto cell,
    and A1: every auto serve process had it enabled and consistent at both ends of every window.
    A1 decides only how the run is described; no result or validity depends on it."""
    per_window, failing = [], []
    for w in windows:
        if w.get("config") != "auto" or "decisions" not in w:
            continue
        h = w.get("handoff") or {"present": False}
        per_window.append({"index": w["index"], "kind": w["kind"], **h})
        in_use = h.get("present") and all(
            h.get(k) is True for k in ("enabled_before", "enabled_after", "consistent_before", "consistent_after")
        )
        if not in_use:
            failing.append({"index": w["index"], "disabled_reason": h.get("disabled_reason")})

    def total(rows, key):
        vals = [r.get(key) for r in rows]
        return None if not vals or None in vals else sum(vals)

    cells = {}
    for kind in ("decisions", "decisions_llm"):
        rows = [r for r in per_window if r["kind"] == kind]
        c = {k: total(rows, k) for k in ("episodes", "trips", "forwards_sync", "forwards_async")}
        n = None if None in (c["forwards_sync"], c["forwards_async"]) else c["forwards_sync"] + c["forwards_async"]
        c["async_share"] = c["forwards_async"] / n if n else None
        c["windows"] = len(rows)
        cells[f"auto/{kind}"] = c
    startup = [
        {"step": b.get("step"), "handoff": bench.handoff_snapshot(b.get("health"), model)}
        for b in refs
        if b.get("config") == "auto"
    ]
    gpu_with_handoff = [
        w["index"]
        for w in windows
        if w.get("config") == "gpu" and bench.handoff_snapshot(w.get("serve_health_after"), model) is not None
    ]
    return {
        "windows": per_window,
        "cells": cells,
        "startup": startup,
        "gpu_windows_with_handoff": gpu_with_handoff,
        "A1_adaptive_execution_in_use": {"ok": bool(per_window) and not failing, "windows_failed": failing},
    }


def load_criteria(campaign: Path, override: Path | None = None) -> dict:
    """The campaign's own criteria copy if it has one, else the run-1 criteria.json."""
    for p in (override, campaign / "criteria.json", HERE / "criteria.json"):
        if p is not None and p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError("no criteria file")


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def v2(c2: dict, llm_ws: list[dict], raw_llm_ws: list[dict]) -> dict:
    """V2: run 1 gates generating coverage; run 2 (busy_coverage_min) gates busy coverage and
    reports generating coverage without a limit."""
    errors = sum(x["llm"]["errors"] for x in llm_ws)
    gen = [x["llm"]["generating_coverage"] for x in llm_ws]
    if "busy_coverage_min" not in c2:  # run 1
        return {
            "worst_coverage": min(gen, default=None),
            "llm_errors": errors,
            "ok": all(
                x["llm"]["generating_coverage"] >= c2["generating_coverage_min"]
                and x["llm"]["errors"] <= c2["llm_errors_max"]
                for x in llm_ws
            ),
        }
    busy = [busy_coverage(w) for w in raw_llm_ws]
    return {
        "worst_busy_coverage": min(busy, default=None),
        "worst_generating_coverage": min(gen, default=None),  # reported, not gated
        "llm_errors": errors,
        "ok": all(b >= c2["busy_coverage_min"] for b in busy)
        and all(x["llm"]["errors"] <= c2["llm_errors_max"] for x in llm_ws),
    }


def summarise(campaign: Path, criteria: Path | None = None) -> dict:
    crit = load_criteria(campaign, criteria)
    raw = campaign / "raw"
    meta = json.loads((raw / "campaign.json").read_text())
    windows = load_windows(raw)
    per_window = []
    for w in windows:
        per_window.append(
            {
                "index": w["index"],
                "kind": w["kind"],
                "config": w["config"],
                "decisions": decision_metrics(w) if "decisions" in w else None,
                "llm": llm_metrics(w),
                "exclusive": exclusive(w),
            }
        )

    def cell(config, kind):
        return [x for x in per_window if x["config"] == config and x["kind"] == kind]

    cells = {}
    for config, kind in [(None, "llm_alone")] + [(c, k) for c in bench.CONFIGS for k in ("decisions", "decisions_llm")]:
        ws = cell(config, kind)
        name = kind if config is None else f"{config}/{kind}"
        c = {"windows": len(ws)}
        if kind != "llm_alone":
            c["classes"] = {}
            for cls in CLASSES:
                vs = [x["decisions"]["classes"][cls] for x in ws]
                c["classes"][cls] = {
                    "p50_ms_median": med([v["p50_ms"] for v in vs]),
                    "p99_ms_median": med([v["p99_ms"] for v in vs]),
                    "p99_ms_windows": [v["p99_ms"] for v in vs],
                    "n": sum(v["n"] for v in vs),
                    "errors": sum(v["errors"] for v in vs),
                    "hard_mismatches": sum(v["hard_mismatches"] for v in vs),
                    "prob_err_max": max((v["prob_err_max"] for v in vs), default=0.0),
                    "act_err_max": max((v["act_err_max"] for v in vs), default=0.0),
                    "near_tie_flips": [f for v in vs for f in v["near_tie_flips"]],
                    "device_changed": sum(v["device_changed"] for v in vs),
                    "devices": {
                        d: sum(v["devices"].get(d, 0) for v in vs)
                        for d in sorted({d for v in vs for d in v["devices"]})
                    },
                }
        if kind != "decisions":
            toks = [x["llm"]["tok_s"] for x in ws]
            c["llm_tok_s_median"] = med(toks)
            c["llm_tok_s_windows"] = toks
            c["llm_request_decode_tok_s_median"] = med([x["llm"]["request_decode_tok_s_median"] for x in ws])
            c["llm_server_ttft_s_median"] = med([x["llm"]["server_ttft_s_median"] for x in ws])
        cells[name] = c

    alone = cells["llm_alone"]["llm_tok_s_windows"]
    alone_med = med(alone)
    noise = (max(alone) - min(alone)) / alone_med if alone and alone_med else None

    def drop(config):
        v = cells[f"{config}/decisions_llm"].get("llm_tok_s_median")
        return None if v is None or not alone_med else 1.0 - v / alone_med

    drops = {c: drop(c) for c in bench.CONFIGS}

    # validity
    dec_ws = [x for x in per_window if x["decisions"]]
    llm_ws = [x for x in per_window if x["llm"]]
    vc = crit["validity"]
    ref_ok = True
    for block in load_refs(raw):
        for cls, refs in block["references"].items():
            want = "ane" if (block["config"] == "auto" and cls == "short_1q") else "gpu"
            ref_ok &= all(r["device"] == want for r in refs)
    auto_short = [x["decisions"]["classes"]["short_1q"] for x in dec_ws if x["config"] == "auto"]
    validity = {
        "V1_open_loop_client_lag_p99_ms": {
            "worst": max((x["decisions"]["lag_p99_ms"] or 0.0 for x in dec_ws), default=None),
            "ok": all(
                (x["decisions"]["lag_p99_ms"] or 0.0) <= vc["V1_open_loop_client_lag_p99_ms"]["max"] for x in dec_ws
            ),
        },
        "V2_llm_saturated": v2(vc["V2_llm_saturated"], llm_ws, [w for w in windows if "llm" in w]),
        "V3_llm_exclusive": {
            "windows_failed": [x["index"] for x in per_window if x["exclusive"]["ok"] is False],
            "windows_unchecked": [x["index"] for x in per_window if not x["exclusive"]["checked"]],
            "ok": all(x["exclusive"]["ok"] for x in per_window),
        },
    }
    if "V4_auto_short_policy_path" in vc:  # run 2
        c4 = vc["V4_auto_short_policy_path"]
        routes = [
            short_routing(w, c4["spill_reason"]) for w in windows if w.get("config") == "auto" and "decisions" in w
        ]
        validity["V4_auto_short_policy_path"] = {
            "worst_policy_share": min((r["policy_share"] for r in routes if r["n"]), default=None),
            "spill_share_windows": {str(r["index"]): r["spill_share"] for r in routes},  # reported, not gated
            "other_routes": {
                k: sum(r["other"].get(k, 0) for r in routes) for k in sorted({k for r in routes for k in r["other"]})
            },
            "ok": bool(routes) and all(r["n"] and r["policy_share"] >= c4["min_share"] for r in routes),
        }
    else:  # run 1
        validity["V4_auto_short_on_ane"] = {
            "worst_share": min((s["devices"].get("ane", 0) / s["n"] for s in auto_short if s["n"]), default=None),
            "ok": all(
                s["n"] and s["devices"].get("ane", 0) / s["n"] >= vc["V4_auto_short_on_ane"]["min_share"]
                for s in auto_short
            ),
        }
    validity["V5_references_on_expected_devices"] = {"ok": ref_ok}
    valid = all(v["ok"] for v in validity.values())

    # criteria
    rc = crit["results"]
    auto_l, auto_u = cells["auto/decisions_llm"]["classes"]["short_1q"], cells["auto/decisions"]["classes"]["short_1q"]
    l1 = auto_l["p99_ms_median"]
    l2 = l1 / auto_u["p99_ms_median"] if l1 is not None and auto_u["p99_ms_median"] else None
    dec_cells = [
        cells[f"{c}/{k}"]["classes"][cls]
        for c in bench.CONFIGS
        for k in ("decisions", "decisions_llm")
        for cls in CLASSES
    ]
    c1 = rc["correctness"]["C1_fp16_gate"]
    results = {
        "gpu_free": {
            "G1_llm_cost_auto": {
                "value": drops["auto"],
                "limit": rc["gpu_free"]["G1_llm_cost_auto"]["max"],
                "pass": drops["auto"] is not None and drops["auto"] <= rc["gpu_free"]["G1_llm_cost_auto"]["max"],
            },
            "G2_auto_below_gpu_only": {
                "value": None if None in drops.values() else drops["gpu"] - drops["auto"],
                "limit": noise,
                "pass": None not in drops.values() and noise is not None and drops["gpu"] - drops["auto"] > noise,
            },
        },
        "decision_latency": {
            "L1_short_p99_loaded_abs_ms": {
                "value": l1,
                "limit": rc["decision_latency"]["L1_short_p99_loaded_abs_ms"]["max"],
                "pass": l1 is not None and l1 <= rc["decision_latency"]["L1_short_p99_loaded_abs_ms"]["max"],
            },
            "L2_short_p99_loaded_vs_unloaded": {
                "value": l2,
                "limit": rc["decision_latency"]["L2_short_p99_loaded_vs_unloaded"]["max"],
                "pass": l2 is not None and l2 <= rc["decision_latency"]["L2_short_p99_loaded_vs_unloaded"]["max"],
            },
        },
        "correctness": {
            "C1_fp16_gate": {
                "prob_err_max": max(c["prob_err_max"] for c in dec_cells),
                "act_err_max": max(c["act_err_max"] for c in dec_cells),
                "hard_mismatches": sum(c["hard_mismatches"] for c in dec_cells),
                "near_tie_flips": [f for c in dec_cells for f in c["near_tie_flips"]],
                "pass": max(c["prob_err_max"] for c in dec_cells) <= c1["prob_err_max"]
                and max(c["act_err_max"] for c in dec_cells) <= c1["act_err_max"]
                and sum(c["hard_mismatches"] for c in dec_cells) == c1["hard_mismatches"],
            },
            "E1_errors": {
                "value": sum(c["errors"] for c in dec_cells),
                "limit": rc["correctness"]["E1_errors"]["max"],
                "pass": sum(c["errors"] for c in dec_cells) <= rc["correctness"]["E1_errors"]["max"],
            },
        },
    }
    for group in results.values():
        for r in group.values():
            r["pass"] = bool(r["pass"])
            if not valid:
                r["verdict"] = "invalid"
            else:
                r["verdict"] = "pass" if r["pass"] else "fail"
    out = {
        "campaign": campaign.name,
        "smoke": meta.get("smoke", False),
        "laya_apple": meta.get("laya_apple"),
        "llm_model": meta["args"].get("llm_model"),
        "llm_status_start": {k: (meta.get("llm_status_start") or {}).get(k) for k in ("version",)},
        "arrivals": meta.get("arrivals"),
        "offered_req_s": meta["args"]["clients"] * meta["args"]["rate"],
        "llm_alone_noise": noise,
        "llm_tok_s_drop": drops,
        "cells": cells,
        "validity": validity,
        "valid": valid,
        "results": results,
        "windows": per_window,
    }
    if crit.get("revision"):  # absent in run 1's criteria, whose outputs stay as committed
        out["criteria_revision"] = crit["revision"]
    if "adaptive_execution" in crit.get("recorded_not_gated", {}):  # run 3; runs 1-2 stay as committed
        out["adaptive_execution"] = adaptive_execution(windows, load_refs(raw), meta["args"].get("model", "laya"))
        out["git"] = meta.get("git")
    return out


def f(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


def pct(x):
    return "–" if x is None else f"{x * 100:+.1f}%"


def tables(res: dict) -> str:
    L = [f"# Serve decisions beside a local LLM: {res['campaign']}\n"]
    if res["smoke"]:
        L.append("**Smoke run: not a result.**\n")
    if res.get("criteria_revision"):
        L.append(f"Criteria revision `{res['criteria_revision']}` (see README.md).\n")
    L.append(
        f"Offered decision load {res['offered_req_s']:g} req/s ({res['arrivals']}); LLM `{res['llm_model']}`. "
        f"Medians over windows; P99 is the median of per-window P99s.\n"
    )
    L += [
        "## LLM throughput\n",
        "| cell | windows | LLM tok/s (window) | per-request decode tok/s | server TTFT s | vs LLM alone |",
        "|---|---|---|---|---|---|",
    ]
    c = res["cells"]
    L.append(
        f"| LLM alone | {c['llm_alone']['windows']} | {f(c['llm_alone']['llm_tok_s_median'])} "
        f"| {f(c['llm_alone']['llm_request_decode_tok_s_median'])} | {f(c['llm_alone']['llm_server_ttft_s_median'], 2)} | window spread {pct(res['llm_alone_noise'])} |"
    )
    for cfg in ("gpu", "auto"):
        x = c[f"{cfg}/decisions_llm"]
        L.append(
            f"| serve {cfg} + LLM | {x['windows']} | {f(x['llm_tok_s_median'])} "
            f"| {f(x['llm_request_decode_tok_s_median'])} | {f(x['llm_server_ttft_s_median'], 2)} | {pct(-res['llm_tok_s_drop'][cfg] if res['llm_tok_s_drop'][cfg] is not None else None)} |"
        )
    L += [
        "\n## Decisions\n",
        "| cell | class | n | P50 ms | P99 ms | errors | hard mismatches | max prob err | near-tie flips | devices |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg in ("gpu", "auto"):
        for kind in ("decisions", "decisions_llm"):
            for cls, v in c[f"{cfg}/{kind}"]["classes"].items():
                L.append(
                    f"| {cfg} {'+ LLM' if kind == 'decisions_llm' else 'unloaded'} | {cls} | {v['n']} "
                    f"| {f(v['p50_ms_median'], 2)} | {f(v['p99_ms_median'], 2)} | {v['errors']} | {v['hard_mismatches']} "
                    f"| {f(v['prob_err_max'], 4)} | {len(v['near_tie_flips'])} | {v['devices']} |"
                )
    L += ["\n## Validity\n", "| check | ok | detail |", "|---|---|---|"]
    for k, v in res["validity"].items():
        detail = ", ".join(f"{kk}={vv}" for kk, vv in v.items() if kk != "ok")
        L.append(f"| {k} | {'yes' if v['ok'] else '**no**'} | {detail} |")
    L += [
        "\n## Criteria (separate results, no overall verdict)\n",
        "| result | criterion | value | limit | verdict |",
        "|---|---|---|---|---|",
    ]
    for group, rs in res["results"].items():
        for k, r in rs.items():
            if k == "C1_fp16_gate":
                val = f"prob {f(r['prob_err_max'], 4)}, act {f(r['act_err_max'], 4)}, hard {r['hard_mismatches']}, flips {len(r['near_tie_flips'])}"
                lim = "≤ 0.02, ≤ 0.02, 0, listed"
            elif k.startswith("G"):
                val, lim = (
                    pct(r["value"]),
                    (pct(r["limit"]) if k == "G2_auto_below_gpu_only" else f"≤ {pct(r['limit'])}"),
                )
                if k == "G2_auto_below_gpu_only":
                    lim = f"> {pct(r['limit'])} (LLM-alone spread)"
            else:
                val, lim = f(r["value"], 2), f"≤ {r['limit']:g}"
            v = r["verdict"]
            L.append(f"| {group} | {k} | {val} | {lim} | {v if v == 'pass' else '**' + v + '**'} |")
    if "adaptive_execution" in res:
        ae = res["adaptive_execution"]
        a1 = ae["A1_adaptive_execution_in_use"]
        L += [
            "\n## Adaptive ANE execution (recorded, not gated)\n",
            f"A1, adaptive execution in use in every `auto` window: {'yes' if a1['ok'] else '**no**'}"
            + (f" (failed: {a1['windows_failed']})" if a1["windows_failed"] else "")
            + ". It decides only how the run is described.\n",
            "| cell | windows | episodes | breaker trips | ANE forwards sync | async | async share |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, c in ae["cells"].items():
            share = "–" if c["async_share"] is None else f"{c['async_share'] * 100:.1f}%"
            L.append(
                f"| {name} | {c['windows']} | {c['episodes']} | {c['trips']} | {c['forwards_sync']} "
                f"| {c['forwards_async']} | {share} |"
            )
        L += [
            "\n| window | cell | state before → after | episodes | trips | sync | async |",
            "|---|---|---|---|---|---|---|",
        ]
        for w in ae["windows"]:
            L.append(
                f"| {w['index']} | auto/{w['kind']} | {w.get('state_before')} → {w.get('state_after')} "
                f"| {w.get('episodes')} | {w.get('trips')} | {w.get('forwards_sync')} | {w.get('forwards_async')} |"
            )
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--criteria", type=Path, help="criteria file (default: the campaign's criteria.json, else run 1's)")
    a = ap.parse_args(argv)
    res = summarise(a.campaign, a.criteria)
    js = json.dumps(res, indent=1, sort_keys=True) + "\n"
    md = tables(res)
    outs = ((a.campaign / "results.json", js), (a.campaign / "tables.md", md))
    if a.check:
        stale = [p.name for p, s in outs if not p.exists() or p.read_text() != s]
        if stale:
            sys.exit(f"stale: {stale}")
        print("outputs are up to date")
        return
    for p, s in outs:
        p.write_text(s)
    print(md)


if __name__ == "__main__":
    main()
