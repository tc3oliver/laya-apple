"""Summarise the ANE placement acceptance gate (#52): results.json and tables.md.

    uv run python scripts/ane_placement_gate_report.py [--check]

Reads benchmarks/ane-placement-gate/raw/ (scripts/bench_ane_placement_gate.py). The gate
criteria below were written down before the measurement data was looked at, and are applied
unchanged. Process placement passes only if every criterion passes for every model:

  correctness   0 answer mismatches in every smoke window, both placements
  isolation     process: GPU return leg P50 <= 0.5 ms with the ANE busy (the #45 target)
  memory        process tree USS (steady) - thread tree USS <= 500 MB   [now phys_footprint; see below]
  startup       process ANE-instance start-up <= thread + 2.0 s
  cpu           process tree CPU per request (both devices busy) <= 1.5 x thread
  ipc           process ANE dispatch + return mean <= 0.5 ms per request
  shutdown      close() <= 2.0 s; after every close no worker process alive or zombie
  lifecycle     10 create/serve/close iterations [process only; see below]: parent RSS at the last iteration <= 1.05 x
                the second, file descriptors and threads not growing, no child left alive
  orphan        workers exit after an exception inside `with` and after the parent is
                SIGKILLed (within 30 s)

These thresholds are meant to catch a clearly unacceptable cost, not to tune a default.

Changes made after the first gate run, each for a measurement reason and applied before the
re-run's data was seen:
- memory: macOS denies USS for other processes (children included), so the first run
  recorded no worker memory at all. The criterion uses the same threshold on the tree's
  phys_footprint (what macOS charges each process), read with /usr/bin/footprint at the end
  of every window; RSS is reported beside it.
- lifecycle and shutdown are judged on the process placement only: the gate is about
  process placement's cost, and the thread placement is the incumbent it is compared with.
  Thread results are still reported.
The CPU criterion is unchanged. The per-device split (GPU worker CPU per GPU request, ANE side
per ANE request) is reported beside it, because the request mix differs between placements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "ane-placement-gate"
RAW = ROOT / "raw"
MODELS = ("laya", "laya-typed-decisions", "laya-multilingual")
PLACEMENTS = ("thread", "process")


def pct(a) -> dict:
    a = np.asarray(a, float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def smoke_summary(d: dict) -> dict:
    out = {
        "startup_s": d["startup_s"],
        "memory": d["memory"],
        "close_s": d["close_s"],
        "workers_alive_after_close": d["workers_alive_after_close"],
        "zombies_after_close": d["zombies_after_close"],
        "ane_length": d["ane_length"],
        "gpu_length": d["gpu_length"],
    }
    for cell in ("gpu_alone", "both"):
        ws = [w for w in d["windows"] if w["cell"] == cell]
        cat = lambda k: [x for w in ws for x in w[k]]  # noqa: E731
        secs = sum(w["seconds"] for w in ws)
        n = sum(w["gpu_requests"] + w["ane_requests"] for w in ws)
        cpu = {k: sum(w["cpu_s"][k] for w in ws) for k in ws[0]["cpu_s"]}
        out[cell] = {
            "gpu_req_s": sum(w["gpu_requests"] for w in ws) / secs,
            "ane_req_s": sum(w["ane_requests"] for w in ws) / secs,
            "mismatches": sum(w["gpu_mismatches"] + w["ane_mismatches"] for w in ws),
            "gpu_return_ms": pct(cat("gpu_return_ms")),
            "gpu_e2e_ms": pct(cat("gpu_e2e_ms")),
            "gpu_service_ms": pct(cat("gpu_service_ms")),
            "ane_e2e_ms": pct(cat("ane_e2e_ms")),
            "ane_service_ms": pct(cat("ane_service_ms")),
            "ane_ipc_ms": pct(cat("ane_ipc_ms")),
            "cpu_ms_per_request": {k: v / n * 1e3 for k, v in cpu.items()} if n else None,
            "tree_cpu_ms_per_request": sum(cpu.values()) / n * 1e3 if n else None,
            "gpu_worker_cpu_ms_per_gpu_request": cpu["gpu_worker"] / g * 1e3
            if (g := sum(w["gpu_requests"] for w in ws))
            else None,
            # the parent also runs the GPU's client side, in both placements
            "parent_plus_ane_worker_cpu_ms_per_ane_request": (cpu["parent"] + cpu.get("ane_worker", 0.0)) / a * 1e3
            if (a := sum(w["ane_requests"] for w in ws))
            else None,
        }
    fps = [w["footprint"] for w in d["windows"]]
    out["footprint_mb"] = {
        k: {
            "steady": float(np.median([x[k]["phys_footprint_mb"] for x in fps])),
            "peak": max(x[k]["phys_footprint_peak_mb"] for x in fps),
        }
        for k in fps[0]
    }
    out["footprint_mb"]["tree"] = {q: sum(v[q] for k, v in out["footprint_mb"].items()) for q in ("steady", "peak")}
    return out


def lifecycle_summary(d: dict) -> dict:
    it = d["iterations"]
    return {
        "iterations": len(it),
        "startup_s": pct([x["startup_s"] for x in it]),
        "close_s": pct([x["close_s"] for x in it]),
        "rss_mb": [x["rss_mb"] for x in it],
        "uss_mb": [x["uss_mb"] for x in it],
        "fds": [x["fds"] for x in it],
        "threads": [x["threads"] for x in it],
        "rss_growth_2nd_to_last": it[-1]["rss_mb"] / it[1]["rss_mb"] - 1,
        "workers_alive_after_close": sum(len(x["workers_alive_after_close"]) for x in it),
        "live_children_after_close": sum(len(x["live_children_after_close"]) for x in it),
        "zombie_children_after_close": sum(len(x["zombie_children_after_close"]) for x in it),
        "devices": {k: sum(x["devices"].get(k, 0) for x in it) for k in ("gpu", "ane")},
    }


def gate(res: dict) -> dict:
    out = {}
    for m in MODELS:
        s = res["models"][m]["smoke"]
        lc = res["models"][m]["lifecycle"]
        th, pr = s["thread"], s["process"]
        fp = lambda x: x["footprint_mb"]["tree"]["steady"]  # noqa: E731
        checks = {
            "correctness": all(x[c]["mismatches"] == 0 for x in (th, pr) for c in ("gpu_alone", "both")),
            "isolation": pr["both"]["gpu_return_ms"]["p50"] <= 0.5,
            "memory": fp(pr) - fp(th) <= 500.0,
            "startup": pr["startup_s"]["ane_instance"] <= th["startup_s"]["ane_instance"] + 2.0,
            "cpu": pr["both"]["tree_cpu_ms_per_request"] <= 1.5 * th["both"]["tree_cpu_ms_per_request"],
            "ipc": pr["both"]["ane_ipc_ms"]["mean"] <= 0.5,
            "shutdown": pr["close_s"] <= 2.0
            and not pr["workers_alive_after_close"]
            and not pr["zombies_after_close"]
            and lc["process"]["close_s"]["p99"] <= 2.0,
            "lifecycle": all(
                lc[p]["rss_growth_2nd_to_last"] <= 0.05
                and lc[p]["fds"][-1] <= lc[p]["fds"][1]
                and lc[p]["threads"][-1] <= lc[p]["threads"][1]
                and lc[p]["workers_alive_after_close"] == 0
                and lc[p]["live_children_after_close"] == 0
                and lc[p]["zombie_children_after_close"] == 0
                for p in ("process",)
            ),
            "orphan": not res["models"][m]["orphan"]["exception_in_with"]["alive_after"]
            and not res["models"][m]["orphan"]["parent_sigkill"]["alive_after_30s"],
        }
        out[m] = {"checks": checks, "pass": all(checks.values())}
    out["overall_pass"] = all(out[m]["pass"] for m in MODELS)
    return out


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def tables(res: dict) -> str:
    L = [
        "# ANE placement acceptance gate",
        "",
        "Generated by `scripts/ane_placement_gate_report.py` from `raw/`. ms unless noted.",
        "",
    ]
    L += [
        "## Gate",
        "",
        "| model | " + " | ".join(res["gate"][MODELS[0]]["checks"]) + " | pass |",
        "|---|" + "---|" * (len(res["gate"][MODELS[0]]["checks"]) + 1),
    ]
    for m in MODELS:
        g = res["gate"][m]
        L.append(
            f"| {m} | "
            + " | ".join("pass" if v else "**FAIL**" for v in g["checks"].values())
            + f" | {'PASS' if g['pass'] else '**FAIL**'} |"
        )
    L += ["", f"Overall: **{'PASS' if res['gate']['overall_pass'] else 'FAIL'}**", ""]
    L += [
        "## GPU completion with the ANE busy (smoke, GPU L128 + ANE smallest bucket)",
        "",
        "| model | ANE L | placement | GPU return alone P50 | GPU return both P50 / P95 / P99 | GPU e2e both P50 / P99 "
        "| GPU req/s both | ANE req/s both | ANE e2e both P50 / P99 | ANE IPC mean | mismatches |",
        "|" + "---|" * 11,
    ]
    for m in MODELS:
        for p in PLACEMENTS:
            s = res["models"][m]["smoke"][p]
            a, b = s["gpu_alone"], s["both"]
            r = b["gpu_return_ms"]
            L.append(
                f"| {m} | {s['ane_length']} | {p} | {f(a['gpu_return_ms']['p50'], 3)} | {f(r['p50'], 3)} / {f(r['p95'], 3)} / "
                f"{f(r['p99'], 3)} | {f(b['gpu_e2e_ms']['p50'])} / {f(b['gpu_e2e_ms']['p99'])} | {f(b['gpu_req_s'], 1)} | "
                f"{f(b['ane_req_s'], 1)} | {f(b['ane_e2e_ms']['p50'])} / {f(b['ane_e2e_ms']['p99'])} | "
                f"{f(b['ane_ipc_ms']['mean'], 3)} | {a['mismatches'] + b['mismatches']} |"
            )
    L += [
        "",
        "## Resources (smoke, both devices busy)",
        "",
        "Memory in MB: phys_footprint (steady = median at window ends; peak = lifetime peak) and tree RSS (median",
        "of 100 ms samples). CPU in ms: tree = all processes / all requests; GPU worker per GPU request; parent +",
        "ANE worker per ANE request (the parent also runs the GPU client side, in both placements).",
        "",
        "| model | placement | start-up s GPU / ANE instance | footprint parent / GPU w / ANE w | tree footprint steady / peak "
        "| tree RSS | tree CPU/request | GPU worker CPU/GPU req | parent+ANE CPU/ANE req | close s |",
        "|" + "---|" * 10,
    ]
    for m in MODELS:
        for p in PLACEMENTS:
            s = res["models"][m]["smoke"][p]
            fp = s["footprint_mb"]
            b = s["both"]
            ane_w = f(fp["ane_worker"]["steady"], 0) if "ane_worker" in fp else "–"
            L.append(
                f"| {m} | {p} | {f(s['startup_s']['gpu_instance'])} / {f(s['startup_s']['ane_instance'])} | "
                f"{f(fp['parent']['steady'], 0)} / {f(fp['gpu_worker']['steady'], 0)} / {ane_w} | "
                f"{f(fp['tree']['steady'], 0)} / {f(fp['tree']['peak'], 0)} | {f(s['memory']['tree']['rss_mb_steady'], 0)} | "
                f"{f(b['tree_cpu_ms_per_request'])} | {f(b['gpu_worker_cpu_ms_per_gpu_request'])} | "
                f"{f(b['parent_plus_ane_worker_cpu_ms_per_ane_request'])} | {f(s['close_s'])} |"
            )
    L += [
        "",
        '## Repeated create / serve / close (device="auto", 10 iterations, 40 requests each)',
        "",
        "| model | placement | start-up s P50 / max | close s P50 / max | parent RSS MB 2nd → last | growth | fds 2nd → last "
        "| threads 2nd → last | workers / children alive after close | zombies | requests gpu / ane |",
        "|" + "---|" * 11,
    ]
    for m in MODELS:
        for p in PLACEMENTS:
            lc = res["models"][m]["lifecycle"][p]
            L.append(
                f"| {m} | {p} | {f(lc['startup_s']['p50'])} / {f(max(lc['startup_s_all']))} | {f(lc['close_s']['p50'])} / "
                f"{f(max(lc['close_s_all']))} | {f(lc['rss_mb'][1], 0)} → {f(lc['rss_mb'][-1], 0)} | "
                f"{lc['rss_growth_2nd_to_last'] * 100:+.1f}% | {lc['fds'][1]} → {lc['fds'][-1]} | "
                f"{lc['threads'][1]} → {lc['threads'][-1]} | {lc['workers_alive_after_close']} / {lc['live_children_after_close']} "
                f"| {lc['zombie_children_after_close']} | {lc['devices']['gpu']} / {lc['devices']['ane']} |"
            )
    L += [
        "",
        "## Worker cleanup (process placement)",
        "",
        "| model | exception inside `with`: alive after | parent SIGKILL: alive after 30 s | seconds to exit |",
        "|---|---|---|---|",
    ]
    for m in MODELS:
        o = res["models"][m]["orphan"]
        L.append(
            f"| {m} | {len(o['exception_in_with']['alive_after'])} of {len(o['exception_in_with']['worker_pids'])} | "
            f"{len(o['parent_sigkill']['alive_after_30s'])} of {len(o['parent_sigkill']['worker_pids'])} | "
            f"{f(o['parent_sigkill']['seconds_to_exit'])} |"
        )
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    load = lambda name: json.loads((RAW / name).read_text())  # noqa: E731
    res = {"models": {}}
    env = None
    for m in MODELS:
        res["models"][m] = {"smoke": {}, "lifecycle": {}}
        for p in PLACEMENTS:
            d = load(f"smoke-{m}-{p}.json")
            env = env or d["environment"]
            res["models"][m]["smoke"][p] = smoke_summary(d)
            lc = load(f"lifecycle-{m}-{p}.json")
            s = lifecycle_summary(lc)
            s["startup_s_all"] = [x["startup_s"] for x in lc["iterations"]]
            s["close_s_all"] = [x["close_s"] for x in lc["iterations"]]
            res["models"][m]["lifecycle"][p] = s
        o = load(f"orphan-{m}.json")
        res["models"][m]["orphan"] = {k: o[k] for k in ("exception_in_with", "parent_sigkill")}
    res["environment"] = env
    res["gate"] = gate(res)
    outputs = {ROOT / "results.json": json.dumps(res, indent=1, sort_keys=True) + "\n", ROOT / "tables.md": tables(res)}
    if args.check:
        stale = [p.name for p, text in outputs.items() if not p.exists() or p.read_text() != text]
        if stale:
            raise SystemExit(f"out of date: {stale}; re-run scripts/ane_placement_gate_report.py")
        print("outputs are up to date")
        return
    for p, text in outputs.items():
        p.write_text(text)
    print(outputs[ROOT / "tables.md"])


if __name__ == "__main__":
    main()
