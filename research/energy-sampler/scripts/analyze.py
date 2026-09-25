"""Summarise the energy runs: results.json and tables.md from raw/.

    uv run python research/energy-sampler/scripts/analyze.py [--check] [--raw DIR --out DIR]

Inputs (raw/):
  campaign*.json.gz                 harness.py runs
  crosscheck-sampler.json.gz        crosscheck.py run
  crosscheck-powermetrics.txt       the operator's `sudo powermetrics` capture of the same run

--check regenerates both outputs in memory and fails if they differ from the committed files.

Each run is analysed under the criteria revision stored with it (criteria-<rev>.json next to
README.md): a run's meta.criteria names it, and a run without one (method_version 1, run 1) is
analysed under r1. Revisions keep the preregistered thresholds unchanged (README.md, "Run 2");
results are reported per criterion, with no combined PASS/FAIL.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from energy import (  # noqa: E402
    agreement,
    align_lag,
    baseline_at,
    integrate_power,
    net_energy,
    parse_powermetrics_text,
    per_decision,
    window_energy,
)

SOC = ("cpu", "gpu", "ane", "dram")


def load_criteria(rev: str) -> dict:
    """criteria-<rev>.json. The thresholds are preregistered (README.md); do not edit a
    revision after data exists for it, add a new one."""
    return json.loads((ROOT / f"criteria-{rev}.json").read_text())


def criteria_rev(meta: dict | None) -> str:
    """The criteria revision a run was recorded under: meta.criteria, else r1 (method
    version 1 predates the field)."""
    meta = meta or {}
    if meta.get("criteria"):
        return meta["criteria"]
    if meta.get("method_version", 1) == 1:
        return "r1"
    raise ValueError(f"run with method_version {meta.get('method_version')} names no criteria revision")


def load_gz(p: Path) -> dict:
    with gzip.open(p, "rt") as f:
        return json.load(f)


def series(sampler: dict, clock: int = 0) -> tuple[list[float], dict[str, list[float]]]:
    """(times s, {rail: cumulative J}) from sampler output. clock 0 = uptime, 1 = unix."""
    ts = [s[clock] / 1e9 for s in sampler["samples"]]
    return ts, {r: [s[2][r] for s in sampler["samples"]] for r in SOC}


def pstr_series(sampler: dict) -> tuple[list[float], list[float]]:
    return [p[0] / 1e9 for p in sampler["pstr"]], [p[1] for p in sampler["pstr"]]


def rng(xs) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)}


# ------------------------------------------------------------------------------ campaign


def window_row(w: dict, ts, cum, pts, pws, rate_tol: float | None = None) -> dict:
    m0, m1 = (x / 1e9 for x in w["measure_ns"])
    dur = m1 - m0
    rails = {r: window_energy(ts, cum[r], m0, m1) for r in SOC}
    soc = sum(rails.values())
    sys_e = integrate_power(pts, pws, m0, m1, hold=True) if pts and pts[0] <= m0 else None
    reqs = [r for r in w["requests"] if m0 <= r[5] / 1e9 <= m1]  # completed inside the window
    offered = [r for r in w["requests"] if m0 <= r[3] / 1e9 < m1]
    decisions = sum(r[2] for r in reqs)
    offered_decisions = sum(r[2] for r in offered)
    late = sorted((r[4] - r[3]) / 1e6 for r in offered)
    errors = sum(1 for r in w["requests"] if r[9])
    devices: dict = {}
    for r in reqs:
        devices[r[6]] = devices.get(r[6], 0) + r[2]
    row = {
        "shape": w["shape"],
        "config": w["config"],
        "index": w["index"],
        "mid_s": (m0 + m1) / 2,
        "duration_s": dur,
        "rails_w": {r: e / dur for r, e in rails.items()},
        "soc_w": soc / dur,
        "system_w": sys_e / dur if sys_e is not None else None,
        "decisions": decisions,
        "requests": len(reqs),
    }
    if w["config"] != "idle":
        rate_q = offered_decisions / dur
        row.update(
            offered_decisions_per_s=rate_q,
            achieved_decisions_per_s=decisions / dur,
            late_p99_ms=late[int(0.99 * (len(late) - 1))] if late else None,
            errors=errors,
            decisions_by_device=devices,
        )
        if rate_tol is None:
            rate_tol = load_criteria("r1")["thresholds"]["rate_tol"]
        ok_rate = rate_q > 0 and abs(decisions / dur - rate_q) / rate_q <= rate_tol
        ok_late = bool(late) and late[int(0.99 * (len(late) - 1))] <= 1e3 / w["rate"]
        row["valid"] = bool(ok_rate and ok_late and errors == 0)
    if "attempt" in w:  # method version 2: disturbance record and repeats
        row.update(
            attempt=w["attempt"],
            disturbed=w["disturbed"],
            superseded=w["superseded"],
            cpu_excess_w=w["cpu_excess_w"],
            cpu_by_process=w["cpu_by_process"],
        )
    return row


def campaign(run: dict) -> dict:
    th = load_criteria(criteria_rev(run.get("meta")))["thresholds"]
    ts, cum = series(run["sampler"])
    pts, pws = pstr_series(run["sampler"])
    all_rows = [window_row(w, ts, cum, pts, pws, th["rate_tol"]) for w in run["windows"]]
    # A superseded attempt was repeated because of a CPU burst (method version 2); the
    # attempt that replaced it is the one analysed. Every attempt stays in "windows".
    rows = [r for r in all_rows if not r.get("superseded")]
    out: dict = {"meta": run["meta"], "args": run["args"], "sampler_meta": run["sampler"]["meta"], "shapes": {}}
    loop_frac = run["sampler"]["meta"].get("sampler_loop_cpu_frac")
    out["sampler_overhead"] = {
        "loop_cpu_frac": loop_frac,
        "ok": loop_frac is not None and loop_frac <= th["sampler_max_cpu_frac"],
    }
    for shape in dict.fromkeys(r["shape"] for r in rows):
        rs = [r for r in rows if r["shape"] == shape]
        idle = [r for r in rs if r["config"] == "idle"]
        base_soc = [(r["mid_s"], r["soc_w"]) for r in idle]
        base_sys = [(r["mid_s"], r["system_w"]) for r in idle if r["system_w"] is not None]
        base_rail = {k: [(r["mid_s"], r["rails_w"][k]) for r in idle] for k in SOC}
        for r in rs:
            if r["config"] == "idle":
                continue
            b = baseline_at(base_soc, r["mid_s"])
            r["idle_soc_w"] = b
            r["net_soc_w"] = r["soc_w"] - b
            r["gross_j_per_decision"] = per_decision(r["soc_w"] * r["duration_s"], r["decisions"])
            r["net_j_per_decision"] = per_decision(
                net_energy(r["soc_w"] * r["duration_s"], r["duration_s"], b), r["decisions"]
            )
            r["net_rail_j_per_decision"] = {
                k: per_decision(
                    net_energy(
                        r["rails_w"][k] * r["duration_s"], r["duration_s"], baseline_at(base_rail[k], r["mid_s"])
                    ),
                    r["decisions"],
                )
                for k in SOC
            }
            if r["system_w"] is not None and base_sys:
                bs = baseline_at(base_sys, r["mid_s"])
                r["net_system_j_per_decision"] = per_decision((r["system_w"] - bs) * r["duration_s"], r["decisions"])
        loaded = [r for r in rs if r["config"] != "idle"]
        idle_w = [r["soc_w"] for r in idle]
        min_net = min((r["net_soc_w"] for r in loaded), default=None)
        spread = max(idle_w) - min(idle_w) if idle_w else None
        cells = {}
        for cfg in dict.fromkeys(r["config"] for r in loaded):
            cr = [r for r in loaded if r["config"] == cfg]
            valid = [r for r in cr if r["valid"]]
            cells[cfg] = {
                "windows": len(cr),
                "valid_windows": len(valid),
                "net_j_per_decision": rng([r["net_j_per_decision"] for r in valid]),
                "gross_j_per_decision": rng([r["gross_j_per_decision"] for r in valid]),
                "net_system_j_per_decision": rng([r.get("net_system_j_per_decision") for r in valid]),
                "net_rail_j_per_decision": {k: rng([r["net_rail_j_per_decision"][k] for r in valid]) for k in SOC},
                "net_soc_w": rng([r["net_soc_w"] for r in valid]),
                "achieved_decisions_per_s": rng([r["achieved_decisions_per_s"] for r in cr]),
                "decisions_by_device": {
                    d: sum(r["decisions_by_device"].get(d, 0) for r in valid)
                    for d in sorted({d for r in valid for d in r["decisions_by_device"]}, key=str)
                },
            }
        comparisons = {}
        ref = cells.get("gpu")
        for cfg, c in cells.items():
            if cfg == "gpu" or not ref:
                continue
            for metric in ("net_j_per_decision", "net_system_j_per_decision"):
                a, g = c[metric], ref[metric]
                if a.get("n", 0) == 0 or g.get("n", 0) == 0:
                    verdict, ratio = "no valid windows", None
                else:
                    ratio = a["median"] / g["median"] if g["median"] else None
                    verdict = "lower" if a["max"] < g["min"] else "higher" if a["min"] > g["max"] else "not separated"
                comparisons[f"{cfg}_vs_gpu:{metric}"] = {"median_ratio": ratio, "verdict": verdict}
        out["shapes"][shape] = {
            "idle": {
                "soc_w": rng(idle_w),
                "spread_w": spread,
                "min_net_loaded_w": min_net,
                "ok": spread is not None
                and min_net is not None
                and min_net > 0
                and spread <= th["idle_spread_max"] * min_net,
            },
            "cells": cells,
            "comparisons": comparisons,
            "windows": rs,
        }
        attempts = [r for r in all_rows if r["shape"] == shape and "attempt" in r]
        if attempts:
            sup = [r for r in attempts if r["superseded"]]
            idle_all = [r["soc_w"] for r in attempts if r["config"] == "idle"]
            out["shapes"][shape]["disturbance"] = {
                "attempts": len(attempts),
                "repeated": len(sup),
                "repeated_idle": sum(1 for r in sup if r["config"] == "idle"),
                "disturbed_but_kept": sum(1 for r in attempts if r["disturbed"] and not r["superseded"]),
                # informational: the idle spread had the repeated attempts been kept
                "idle_spread_all_attempts_w": max(idle_all) - min(idle_all) if idle_all else None,
                "superseded_windows": sup,
            }
    return out


# ------------------------------------------------------------------------------ cross-check


def crosscheck(sampler_path: Path, pm_path: Path) -> dict | None:
    if not (sampler_path.exists() and pm_path.exists()):
        return None
    run = load_gz(sampler_path)
    rev = criteria_rev(run.get("meta"))
    th = load_criteria(rev)["thresholds"]
    pm = parse_powermetrics_text(pm_path.read_text(errors="replace"))
    ts, cum = series(run["sampler"], clock=1)
    rails = {k: cum[k] for k in ("cpu", "gpu", "ane")}
    lag = align_lag(pm, ts, rails)
    phases = []
    for p in run["phases"]:
        u0, u1 = (x / 1e9 for x in p["unix_ns"])
        lo, hi = u0 + th["xc_trim_s"], u1 - th["xc_trim_s"]
        pairs = []
        for s in pm:
            t1 = s["t_end"] + lag
            t0 = t1 - s["elapsed_s"]
            if t0 >= lo and t1 <= hi and t0 >= ts[0] and t1 <= ts[-1]:
                ours = {k: window_energy(ts, rails[k], t0, t1) / (t1 - t0) for k in rails}
                pairs.append((s["elapsed_s"], ours, s))
        row = {"phase": p["phase"], "pairs": len(pairs)}
        if pairs:
            wsum = sum(e for e, _, _ in pairs)
            for k in ("cpu", "gpu", "ane"):
                o = sum(e * ours[k] for e, ours, _ in pairs) / wsum
                r = sum(e * s[f"{k}_w"] for e, _, s in pairs) / wsum
                row[k] = agreement(o, r, th["xc_rel_tol"], th["xc_abs_floor_w"], th["xc_abs_tol_w"])
            o = sum(row[k]["ours_w"] for k in ("cpu", "gpu", "ane"))
            r = sum(e * s["combined_w"] for e, _, s in pairs) / wsum
            row["combined"] = agreement(o, r, th["xc_rel_tol"], th["xc_abs_floor_w"], th["xc_abs_tol_w"])
        need = th["xc_min_pairs"] if p["phase"] != "idle" else 1
        row["ok"] = len(pairs) >= need and all(row[k]["ok"] for k in ("cpu", "gpu", "ane", "combined") if k in row)
        phases.append(row)
    loop_frac = run["sampler"]["meta"].get("sampler_loop_cpu_frac")
    out = {
        "powermetrics_samples": len(pm),
        "lag_s": lag,
        "phases": phases,
        "sampler_loop_cpu_frac": loop_frac,
        "ok": bool(phases) and all(p["ok"] for p in phases),
        "meta": run["meta"],
    }
    if rev != "r1":  # criterion 3 on the cross-check run, reported on its own
        out["criteria"] = rev
        out["sampler_overhead_ok"] = loop_frac is not None and loop_frac <= th["sampler_max_cpu_frac"]
    return out


# ------------------------------------------------------------------------------ output


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def tables(res: dict) -> str:
    L = ["# Energy per decision: tables", "", "Generated by `scripts/analyze.py` from `raw/`. Do not edit.", ""]
    xc = res["crosscheck"]
    L += ["## Sampler vs powermetrics cross-check", ""]
    if xc is None:
        L += ["No cross-check data in `raw/` yet.", ""]
    else:
        L += [
            f"Alignment lag {xc['lag_s']} s, {xc['powermetrics_samples']} powermetrics samples. "
            f"Criterion met: **{'yes' if xc['ok'] else 'no'}**.",
            "",
        ]
        if "sampler_overhead_ok" in xc:
            L += [
                f"Criteria revision {xc['criteria']}. Sampler loop CPU {fmt((xc['sampler_loop_cpu_frac'] or 0) * 100, 2)}% "
                f"of one core (criterion met: {'yes' if xc['sampler_overhead_ok'] else 'no'}).",
                "",
            ]
        L += ["| phase | pairs | rail | sampler W | powermetrics W | diff | ok |", "|---|---:|---|---:|---:|---:|---|"]
        for p in xc["phases"]:
            for k in ("cpu", "gpu", "ane", "combined"):
                if k in p:
                    a = p[k]
                    d = f"{a['rel'] * 100:+.1f}%" if a["rel"] is not None else f"{a['diff_w']:+.3f} W"
                    L.append(
                        f"| {p['phase']} | {p['pairs']} | {k} | {fmt(a['ours_w'])} | {fmt(a['ref_w'])} | {d} | {'yes' if a['ok'] else 'no'} |"
                    )
        L.append("")
    L += ["## J/decision at equal offered load", ""]
    if not res["campaigns"]:
        L += ["No campaign data in `raw/` yet.", ""]
    for name, c in res["campaigns"].items():
        m = c["meta"]
        L += [
            f"### {name}",
            "",
            f"{m['soc']} ({m['model']}), macOS {m['macos']}, laya-apple {m['laya_apple']}, "
            f"oMLX running: {m['omlx_running']}. Sampler loop CPU {fmt((c['sampler_overhead']['loop_cpu_frac'] or 0) * 100, 2)}% "
            f"of one core (criterion met: {'yes' if c['sampler_overhead']['ok'] else 'no'}).",
            "",
        ]
        if "criteria" in res:
            L += [
                f"Criteria revision: {res['criteria'][name]['revision']} (`criteria-{res['criteria'][name]['revision']}.json`).",
                "",
            ]
        for shape, s in c["shapes"].items():
            i = s["idle"]
            L += [
                f"**{shape}** — idle SoC {fmt(i['soc_w'].get('median'))} W, spread {fmt(i['spread_w'])} W "
                f"vs smallest net loaded {fmt(i['min_net_loaded_w'])} W (criterion met: {'yes' if i['ok'] else 'no'}).",
                "",
            ]
            if "disturbance" in s:
                d = s["disturbance"]
                L += [
                    f"Disturbance repeats: {d['repeated']} of {d['attempts']} attempts repeated ({d['repeated_idle']} idle); "
                    f"{d['disturbed_but_kept']} disturbed attempt(s) kept at the repeat limit. Idle spread over all "
                    f"attempts, repeated ones included: {fmt(d['idle_spread_all_attempts_w'])} W (informational).",
                    "",
                ]
            L += [
                "| config | valid/windows | decisions/s | net SoC W | net J/decision (median, min–max) | gross J/decision | net system J/decision (PSTR) | decisions by device |",
                "|---|---:|---:|---:|---|---:|---:|---|",
            ]
            for cfg, cell in s["cells"].items():
                n = cell["net_j_per_decision"]
                L.append(
                    f"| {cfg} | {cell['valid_windows']}/{cell['windows']} | {fmt(cell['achieved_decisions_per_s'].get('median'), 1)} "
                    f"| {fmt(cell['net_soc_w'].get('median'), 2)} | {fmt(n.get('median'), 4)} ({fmt(n.get('min'), 4)}–{fmt(n.get('max'), 4)}) "
                    f"| {fmt(cell['gross_j_per_decision'].get('median'), 4)} | {fmt(cell['net_system_j_per_decision'].get('median'), 4)} "
                    f"| {', '.join(f'{d}: {v}' for d, v in cell['decisions_by_device'].items())} |"
                )
            L += ["", "| comparison | median ratio | verdict |", "|---|---:|---|"]
            for k, v in s["comparisons"].items():
                L.append(f"| {k} | {fmt(v['median_ratio'])} | {v['verdict']} |")
            L.append("")
    return "\n".join(L) + "\n"


def build(raw: Path) -> dict:
    runs = {p.name.removesuffix(".json.gz"): load_gz(p) for p in sorted(raw.glob("campaign*.json.gz"))}
    campaigns = {name: campaign(run) for name, run in runs.items()}
    xc = crosscheck(raw / "crosscheck-sampler.json.gz", raw / "crosscheck-powermetrics.txt")
    res = {"crosscheck": xc, "campaigns": campaigns}
    revs = {name: criteria_rev(run.get("meta")) for name, run in runs.items()}
    if set(revs.values()) - {"r1"}:  # added with r2; run 1 alone keeps its original output
        res["criteria"] = {name: {"revision": rev, **load_criteria(rev)} for name, rev in revs.items()}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", type=Path, default=ROOT / "raw")
    ap.add_argument("--out", type=Path, default=ROOT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    res = build(a.raw)
    rj = json.dumps(res, indent=1, sort_keys=True) + "\n"
    tm = tables(res)
    if a.check:
        bad = [n for n, t in (("results.json", rj), ("tables.md", tm)) if (a.out / n).read_text() != t]
        if bad:
            print(f"out of date: {', '.join(bad)} (re-run analyze.py)", file=sys.stderr)
            return 1
        print("results.json and tables.md are up to date")
        return 0
    (a.out / "results.json").write_text(rj)
    (a.out / "tables.md").write_text(tm)
    print(tm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
