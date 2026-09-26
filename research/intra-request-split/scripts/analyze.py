"""Apply criteria.json to the recorded runs; write results.json and tables.md.

    uv run python research/intra-request-split/scripts/analyze.py --look f1     # after block 1
    uv run python research/intra-request-split/scripts/analyze.py --look f2     # after block 2
    uv run python research/intra-request-split/scripts/analyze.py --look final  # after the mixes
    uv run python research/intra-request-split/scripts/analyze.py --look b --model laya-typed-decisions

Prints one decision line last: CONTINUE, STOP FAIL, STOP INCONCLUSIVE, PASS, FAIL or
INCONCLUSIVE. Reads only raw/; never changes a raw file. The paired gate's arithmetic is
research/coreml-prebind-predict/scripts/gate.py, loaded by path and unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import design  # noqa: E402

GATE = common.load_by_path("split_gate", common.ROOT / "research/coreml-prebind-predict/scripts/gate.py")
PASS, FAIL, INCONCLUSIVE = GATE.PASS, GATE.FAIL, GATE.INCONCLUSIVE


def below(ci: dict, limit: float) -> str:
    """Must be strictly below `limit` (a gain)."""
    return PASS if ci["hi"] < limit else FAIL if ci["lo"] >= limit else INCONCLUSIVE


def not_above(ci: dict, limit: float) -> str:
    """Must not exceed `limit` (no regression)."""
    return PASS if ci["hi"] <= limit else FAIL if ci["lo"] > limit else INCONCLUSIVE


def stats(ratios, confidence: float) -> dict:
    if len(ratios) < 2:
        return {"n": len(ratios), "ratios": ratios}
    x = [math.log(r) for r in ratios]
    sd = (sum((v - sum(x) / len(x)) ** 2 for v in x) / (len(x) - 1)) ** 0.5
    return {
        "n": len(ratios),
        "ratios": ratios,
        "log_sd": sd,
        "t": GATE.t_interval(ratios, confidence),
        "bootstrap": GATE.bootstrap_interval(ratios, confidence=confidence),
    }


def load_dir(sub: str) -> dict:
    d = common.RAW / sub
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def windows_by(run: dict) -> dict:
    return {(w["cycle"], w["workload"], w["arm"]): w for w in run["windows"]}


def g1_ratios(ours: dict, lf: dict, crit: dict, blocks: int, model: str) -> list[dict]:
    order = crit["protocol"]["blocks"]["order"]
    out = []
    for b in range(1, blocks + 1):
        for po, pl in design.adjacent_pairs(order[b - 1], "ours", "lf"):
            o, f = ours.get(f"{model}-b{b}-p{po}"), lf.get(f"laya-fast-b{b}-p{pl}")
            if o is None or f is None:
                raise SystemExit(f"block {b}: run {model}-b{b}-p{po} or laya-fast-b{b}-p{pl} is missing")
            ow, fw = windows_by(o), windows_by(f)
            cycles = sorted({c for c, wl, _ in ow if wl == "8x512"})
            if cycles != sorted({c for c, wl, _ in fw if wl == "8x512"}):
                raise SystemExit(f"block {b}: the paired runs have different 8x512 cycles")
            for c in cycles:
                s, x = (
                    ow[(c, "8x512", "split")]["summary"]["p50_ms"],
                    fw[(c, "8x512", "laya_fast")]["summary"]["p50_ms"],
                )
                out.append({"block": b, "ours": po, "lf": pl, "cycle": c, "split_p50": s, "lf_p50": x, "ratio": s / x})
    return out


def g2_ratios(runs: list[dict], workload: str) -> list[dict]:
    out = []
    for run in runs:
        w = windows_by(run)
        for c in sorted({c for c, wl, _ in w if wl == workload}):
            s, g = w[(c, workload, "split")]["summary"]["p50_ms"], w[(c, workload, "gpu")]["summary"]["p50_ms"]
            out.append({"run": run["run_id"], "cycle": c, "split_p50": s, "gpu_p50": g, "ratio": s / g})
    return out


def g3(runs: list[dict], mixes: dict, tol: dict) -> dict:
    """Correctness of every split answer; the gpu arm's exceedances decide 'shared'."""
    hard, flips, exceed, gpu_exceed, rows = [], [], [], set(), 0
    arms_report: dict = {}

    def over(c):
        return c["prob_err"] > tol["prob_err_max"] or c["act_err"] > tol["act_err_max"]

    for run in runs:
        checks = [(c["workload"], c["seed"], c["arm"], c) for c in run["correctness"]]
        for w in run["windows"]:
            checks += [(w["workload"], r["seed"], w["arm"], r) for r in w["requests"]]
        for wl, seed, arm, c in checks:
            rep = arms_report.setdefault(
                arm, {"answers": 0, "hard": 0, "flips": 0, "prob_err_max": 0.0, "act_err_max": 0.0}
            )
            rep["answers"] += 1
            rep["hard"] += len(c["hard"])
            rep["flips"] += len(c["flips"])
            rep["prob_err_max"] = max(rep["prob_err_max"], c["prob_err"])
            rep["act_err_max"] = max(rep["act_err_max"], c["act_err"])
            if arm == "gpu" and over(c):
                gpu_exceed.add((run["model"], wl, seed))
            if arm != "split":
                continue
            rows += 1
            where = {"run": run["run_id"], "workload": wl, "seed": seed}
            hard += [{**where, "question": q} for q in c["hard"]]
            flips += [{**where, "question": q} for q in c["flips"]]
            if over(c):
                exceed.append({**where, "model": run["model"], "prob_err": c["prob_err"], "act_err": c["act_err"]})
    for name, m in mixes.items():
        for rnd in m["rounds"]:
            if rnd["config"] != "split":
                continue
            for c in rnd["multi_question_checks"]:
                rows += 1
                where = {"mix": name, "round": rnd["index"], "i": c["i"]}
                hard += [{**where, "question": q} for q in c["hard"]]
                flips += [{**where, "question": q} for q in c["flips"]]
                if over(c):
                    exceed.append({**where, "model": m["model"], "prob_err": c["prob_err"], "act_err": c["act_err"]})
    shared = [e for e in exceed if (e.get("model"), e.get("workload"), e.get("seed")) in gpu_exceed]
    other = [e for e in exceed if e not in shared]
    verdict = FAIL if hard or other else INCONCLUSIVE if shared else PASS
    return {
        "verdict": verdict,
        "split_answers_checked": rows,
        "hard_mismatches": hard,
        "near_tie_flips": flips,
        "exceedances": other,
        "shared_exceedances": shared,
        "by_arm": arms_report,
    }


def g4(mix: dict, gate: dict, confidence: float) -> dict:
    measured = mix["protocol"][mix["mix"]]["measured_class"]
    blocks = mix["protocol"]["blocks"]
    by = {(r["block"], r["position"]): r for r in mix["rounds"]}
    ratios, late, wrong, lag_bad = [], {"split": 0, "base": 0}, {"split": 0, "base": 0}, []
    for b, block in enumerate(blocks, 1):
        if not all((b, p) in by for p in range(1, len(block) + 1)):
            break
        for ps, pb in design.adjacent_pairs(block, "split", "base"):
            ratios.append(by[(b, ps)]["summary"][measured]["p99_ms"] / by[(b, pb)]["summary"][measured]["p99_ms"])
    for r in mix["rounds"]:
        late[r["config"]] += r["summary"].get("late") or 0
        wrong[r["config"]] += r.get("oracle_wrong") or 0
        if r["summary"]["lag_p99_ms"] > 5.0:
            lag_bad.append(r["index"])
    st = stats(ratios, confidence)
    if mix.get("stopped"):
        v = FAIL
    elif len(ratios) < gate["pairs_final"] or lag_bad:
        v = INCONCLUSIVE
    else:
        v = not_above(st["t"], 1.05)
        if mix["mix"] == "switchyard" and (late["split"] > late["base"] or wrong["split"] > wrong["base"]):
            v = FAIL
    return {
        "verdict": v,
        "stats": st,
        "late": late,
        "oracle_wrong": wrong,
        "invalid_rounds_lag": lag_bad,
        "stopped": mix.get("stopped"),
    }


def validity(runs: list[dict], lf_runs: list[dict], crit: dict) -> list[str]:
    """criteria.json gates.validity for the latency runs: every problem, or []."""
    problems = []
    for run in runs:
        for w in run["windows"]:
            if w["arm"] != "gpu":
                continue
            want = "sequence_exceeds_ane_auto_range" if w["workload"] == "1x512" else "multiple_questions"
            bad = [r["i"] for r in w["requests"] if r.get("device") != "gpu" or r.get("reason") != want]
            if bad:
                problems.append(
                    f"{run['run_id']} c{w['cycle']} {w['workload']}: gpu-arm requests {bad} not on the GPU/{want}"
                )
    want = sorted(crit["protocol"]["laya_fast"]["ane_bodies"])
    for run in lf_runs:
        if not set(want) <= set(run.get("ane_buckets_loaded") or []):
            problems.append(
                f"{run['run_id']}: laya-fast loaded ANE bodies {run.get('ane_buckets_loaded')}, needs {want}"
            )
        if run.get("laya_fast_commit") != crit["protocol"]["laya_fast"]["commit"]:
            problems.append(f"{run['run_id']}: laya-fast commit {run.get('laya_fast_commit')}")
    return problems


def phase_a(look: str, crit: dict) -> dict:
    model = crit["gates"]["G1_split_below_laya_fast"]["model"]
    ours = {k: v for k, v in load_dir("latency").items() if v["model"] == model}
    lf = load_dir("laya-fast")
    blocks = 1 if look == "f1" else 2
    conf = 0.99 if look == "f1" else 0.95
    runs = [ours[k] for k in sorted(ours) if int(k.split("-b")[1].split("-")[0]) <= blocks]
    res: dict = {"look": look, "confidence": conf, "blocks": blocks}

    r1 = g1_ratios(ours, lf, crit, blocks, model)
    s1 = stats([x["ratio"] for x in r1], conf)
    g1 = below(s1["t"], 1.00)
    res["G1"] = {"pairs": r1, "stats": s1, "verdict": g1}

    g2 = {"no_regression": {}, "gain": {}}
    for wl in crit["gates"]["G2_against_todays_gpu_path"]["no_regression"]["workloads"]:
        st = stats([x["ratio"] for x in g2_ratios(runs, wl)], conf)
        g2["no_regression"][wl] = {"stats": st, "verdict": not_above(st["t"], 1.05)}
    for wl in crit["gates"]["G2_against_todays_gpu_path"]["gain"]["workloads"]:
        st = stats([x["ratio"] for x in g2_ratios(runs, wl)], conf)
        g2["gain"][wl] = {"stats": st, "verdict": below(st["t"], 1.00)}
    parts = [v["verdict"] for part in g2.values() for v in part.values()]
    g2["verdict"] = GATE.combine(parts)
    res["G2"] = g2
    res["G3"] = g3(runs, {}, crit["gates"]["G3_correctness"])
    res["report_only"] = {
        "ane_over_gpu": {
            wl: stats([x["ratio"] for x in _arm_ratios(runs, wl, "ane", "gpu")], conf) for wl in common.WORKLOADS
        },
        "split_k": _k_counts(runs),
        "g1_pooled_p50": {
            "split": _median([x["split_p50"] for x in r1]),
            "laya_fast": _median([x["lf_p50"] for x in r1]),
        },
    }
    res["validity"] = validity(runs, [lf[k] for k in sorted(lf)], crit)
    if res["validity"]:
        res["decision"] = "STOP INVALID"
        return res
    if look == "f1":  # never a PASS here
        fail = g1 == FAIL or res["G3"]["verdict"] == FAIL or FAIL in parts
        res["decision"] = "STOP FAIL" if fail else "CONTINUE"
        return res
    core = GATE.combine([g1, g2["verdict"], res["G3"]["verdict"]])
    res["core_verdict"] = core
    if look == "f2":
        res["decision"] = {PASS: "CONTINUE", FAIL: "STOP FAIL", INCONCLUSIVE: "STOP INCONCLUSIVE"}[core]
        return res
    # final: the mixes
    mixes = load_dir("mix")
    g4s = {}
    for name, key in (("switchyard", "G4a_switchyard_mix"), ("serve", "G4b_serve_mix")):
        m = mixes.get(name)
        g4s[key] = g4(m, crit["gates"][key], 0.95) if m else {"verdict": INCONCLUSIVE, "missing": True}
    res.update(g4s)
    res["G3"] = g3(runs, mixes, crit["gates"]["G3_correctness"])
    verdict = GATE.combine([g1, g2["verdict"], res["G3"]["verdict"], *[v["verdict"] for v in g4s.values()]])
    res["verdict"] = verdict
    res["decision"] = verdict
    return res


def screen(runs: dict, add: dict) -> dict:
    """criteria-addendum-1.json's stop rule. Stop-only: CONTINUE or SCREEN STOP, never a verdict."""
    ours, lf = runs.get("screen-laya"), runs.get("screen-laya-fast")
    if ours is None or lf is None:
        raise SystemExit("raw/screen/ needs screen-laya.json and screen-laya-fast.json")
    w = windows_by(ours)
    cycles = sorted({c for c, wl, _ in w if wl == "8x512"})
    ratios = [
        w[(c, "8x512", "split")]["summary"]["p50_ms"] / w[(c, "8x512", "gpu")]["summary"]["p50_ms"] for c in cycles
    ]
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    split_p50 = _median([w[(c, "8x512", "split")]["summary"]["p50_ms"] for c in cycles])
    gpu_p50 = _median([w[(c, "8x512", "gpu")]["summary"]["p50_ms"] for c in cycles])
    lf_p50 = _median([x["summary"]["p50_ms"] for x in lf["windows"] if x["workload"] == "8x512"])
    hard = [
        (c.get("workload"), c.get("seed"), q) for c in ours["correctness"] if c["arm"] == "split" for q in c["hard"]
    ] + [
        (x["workload"], r["seed"], q)
        for x in ours["windows"]
        if x["arm"] == "split"
        for r in x["requests"]
        for q in r["hard"]
    ]
    s1 = len(ratios) == add["screen"]["cycles"] and all(r < 1.0 for r in ratios) and geo < 0.95
    s2 = not hard
    s3 = split_p50 < 1.10 * lf_p50
    stops = [n for n, ok in (("S1", s1), ("S2", s2), ("S3", s3)) if not ok]
    return {
        "look": "screen",
        "ratios_split_over_gpu": ratios,
        "geomean_split_over_gpu": geo,
        "median_window_p50_ms": {"split": split_p50, "gpu": gpu_p50, "laya_fast": lf_p50},
        "split_over_laya_fast_medians": split_p50 / lf_p50,
        "split_hard_mismatches": hard,
        "k": _k_counts([ours]),
        "stopped_by": stops,
        "decision": f"SCREEN STOP ({', '.join(stops)})" if stops else "SCREEN CONTINUE",
    }


def phase_b(model: str, look: str, crit: dict) -> dict:
    runs = [v for k, v in sorted(load_dir("latency").items()) if v["model"] == model]
    conf = 0.99 if len(runs) <= 2 else 0.95
    out: dict = {"model": model, "confidence": conf, "runs": [r["run_id"] for r in runs]}
    parts = []
    for wl in crit["gates"]["G2_against_todays_gpu_path"]["no_regression"]["workloads"]:
        st = stats([x["ratio"] for x in g2_ratios(runs, wl)], conf)
        if st["n"] >= 2:
            out[f"no_regression_{wl}"] = {"stats": st, "verdict": not_above(st["t"], 1.05)}
            parts.append(out[f"no_regression_{wl}"]["verdict"])
    out["G3"] = g3(runs, {}, crit["gates"]["G3_correctness"])
    parts.append(out["G3"]["verdict"])
    v = GATE.combine(parts)
    out["verdict"] = v
    out["decision"] = ("STOP FAIL" if v == FAIL else "CONTINUE") if conf == 0.99 else v
    return out


def _arm_ratios(runs, workload, a, b):
    out = []
    for run in runs:
        w = windows_by(run)
        for c in sorted({c for c, wl, _ in w if wl == workload}):
            out.append({"ratio": w[(c, workload, a)]["summary"]["p50_ms"] / w[(c, workload, b)]["summary"]["p50_ms"]})
    return out


def _k_counts(runs) -> dict:
    out: dict = {}
    for run in runs:
        for w in run["windows"]:
            if w["arm"] == "split":
                d = out.setdefault(w["workload"], {})
                for r in w["requests"]:
                    d[str(r["k"])] = d.get(str(r["k"]), 0) + 1
    return out


def _median(x):
    x = sorted(x)
    return None if not x else (x[len(x) // 2] if len(x) % 2 else (x[len(x) // 2 - 1] + x[len(x) // 2]) / 2)


def tables(res: dict) -> str:
    def ci(st):
        t = st.get("t")
        return "n < 2" if not t else f"{t['geomean']:.3f} [{t['lo']:.3f}, {t['hi']:.3f}] (n={t['n']})"

    lines = [f"# intra-request-split: look `{res.get('look', res.get('model'))}`", ""]
    if "G1" in res:
        lines += ["| gate | ratio, geometric mean [CI] | verdict |", "|---|---|---|"]
        lines.append(f"| G1 8x512 split / laya-fast (window P50) | {ci(res['G1']['stats'])} | {res['G1']['verdict']} |")
        for part, d in res["G2"].items():
            if part == "verdict":
                continue
            for wl, v in d.items():
                lines.append(f"| G2 {part} {wl} split / gpu (window P50) | {ci(v['stats'])} | {v['verdict']} |")
        g = res["G3"]
        lines.append(
            f"| G3 correctness | {g['split_answers_checked']} split answers; hard {len(g['hard_mismatches'])}, "
            f"flips {len(g['near_tie_flips'])}, exceedances {len(g['exceedances'])} (shared {len(g['shared_exceedances'])}) "
            f"| {g['verdict']} |"
        )
        for key in ("G4a_switchyard_mix", "G4b_serve_mix"):
            if key in res and "stats" in res[key]:
                lines.append(
                    f"| {key} split / base (measured-class P99) | {ci(res[key]['stats'])} | {res[key]['verdict']} |"
                )
    lines += ["", f"Confidence {res.get('confidence')}. Decision: **{res['decision']}**", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--look", required=True, choices=["screen", "f1", "f2", "final", "b"])
    p.add_argument("--model", help="phase B model")
    a = p.parse_args(argv)
    crit = common.criteria()
    if a.look == "screen":
        add = json.loads((common.TRACK / "criteria-addendum-1.json").read_text())
        res = screen(load_dir("screen"), add)
        common.write_json(common.TRACK / "results" / "screen.json", res)
        print(json.dumps({k: v for k, v in res.items() if k != "k"}, indent=1))
        print(res["decision"])
        return
    if a.look == "b":
        if not a.model:
            raise SystemExit("--look b needs --model")
        res = phase_b(a.model, a.look, crit)
        name = f"phase-b-{a.model}"
    else:
        res = phase_a(a.look, crit)
        name = f"phase-a-{a.look}"
    common.write_json(common.TRACK / "results" / f"{name}.json", res)
    (common.TRACK / "results" / f"{name}.md").write_text(tables(res))
    print(tables(res))
    print(res["decision"])


if __name__ == "__main__":
    main()
