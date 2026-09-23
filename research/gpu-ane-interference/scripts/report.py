"""Print the README's tables (markdown) from results.json.

    uv run python research/gpu-ane-interference/scripts/report.py

Every number in research/gpu-ane-interference/README.md's tables comes from this output.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results.json").read_text())
MODELS = {"laya-typed-decisions": "typed-decisions", "laya-multilingual": "multilingual"}
PL = ("thread", "process")


def run(kind, model, pl):
    return RES["runs"][f"{kind}:{model}:{pl}"]


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def solo():
    print("### Solo baselines (closed loop, one device, one stream)\n")
    print("| Model | ANE placement | Stream | L | n | req/s | service mean | P50 | P95 | P99 | P99 95% CI | e2e P99 "
          "| forward | device exec | host | CPU ms | mismatches |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        for pl in PL:
            r = run("device", m, pl)
            for label in ("gpu_M", "gpu_L", "ane_S", "ane_B"):
                s = r["cells"][f"solo:{label}"]["streams"][label]
                L = r["shapes"][label.split("_")[1]]["length"]
                sv = s["service"]
                print(f"| {MODELS[m]} | {pl} | {label} | {L} | {s['n']} | {s['req_s']:.1f} | {sv['mean']:.2f} "
                      f"| {sv['p50']:.2f} | {sv['p95']:.2f} | {sv['p99']:.2f} | {f(sv['p99_ci95'][0])}–{f(sv['p99_ci95'][1])} "
                      f"| {s['e2e']['p99']:.2f} | {s['forward_ms']['mean']:.2f} | {s['device_exec']['mean']:.2f} "
                      f"| {s['host']['mean']:.2f} | {s['cpu_ms']['mean']:.2f} | {s['mismatches']} |")
    print()


def matrix():
    print("### Concurrent matrix (both devices closed loop): inflation against solo\n")
    print("| Model | ANE placement | Cell | Stream | n | service mean ms (solo) | service × (95% CI) | device exec × "
          "| Δ host ms | Δ dispatch ms | Δ CPU ms | P50 | P95 | P99 | e2e P99 × | req/s × | mismatches |")
    print("|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        for pl in PL:
            r = run("device", m, pl)
            for cell in ("matrix:gpu_M+ane_S", "matrix:gpu_M+ane_B", "matrix:gpu_L+ane_S", "matrix:gpu_L+ane_B"):
                for label, e in r["inflation"][cell].items():
                    s = r["cells"][cell]["streams"][label]
                    so = r["cells"][f"solo:{label}"]["streams"][label]
                    sv, ci = e["service"]["mean"], e["service"]["mean"]["ci95"]
                    print(f"| {MODELS[m]} | {pl} | {cell.split(':')[1]} | {label} | {s['n']} "
                          f"| {s['service']['mean']:.2f} ({so['service']['mean']:.2f}) | ×{sv['ratio']:.3f} ({ci[0]:.3f}–{ci[1]:.3f}) "
                          f"| ×{e['device_exec']['mean']['ratio']:.3f} | {e['host_delta_ms']:+.2f} | {e['dispatch_delta_ms']:+.2f} "
                          f"| {e['cpu_ms_delta_ms']:+.2f} | {s['service']['p50']:.2f} | {s['service']['p95']:.2f} "
                          f"| {s['service']['p99']:.2f} | ×{e['p99_e2e_ratio']:.2f} | ×{e['req_s_ratio']:.3f} | {s['mismatches']} |")
    print()


def aggregate():
    print("### Aggregate throughput of the matrix cells\n")
    print("| Model | ANE placement | Cell | GPU req/s (solo) | ANE req/s (solo) | GPU busy | ANE busy |")
    print("|---|---|---|---:|---:|---:|---:|")
    for m in MODELS:
        for pl in PL:
            r = run("device", m, pl)
            for cell in ("matrix:gpu_M+ane_S", "matrix:gpu_M+ane_B", "matrix:gpu_L+ane_S", "matrix:gpu_L+ane_B"):
                c = r["cells"][cell]
                parts = []
                for label, s in c["streams"].items():
                    so = r["cells"][f"solo:{label}"]["streams"][label]
                    parts.append(f"{s['req_s']:.1f} ({so['req_s']:.1f})")
                print(f"| {MODELS[m]} | {pl} | {cell.split(':')[1]} | {parts[0]} | {parts[1]} "
                      f"| {c['device_util']['gpu']['mean']:.2f} | {c['device_util']['ane']['mean']:.2f} |")
    print()


def controls():
    print("### Host-side controls (one closed-loop stream, no second device)\n")
    print("| Model | ANE placement | Stream | Aggressor | service × (95% CI) | device exec × | Δ dispatch ms "
          "| Δ host ms | e2e P99 × | n |")
    print("|---|---|---|---|---|---:|---:|---:|---:|---:|")
    for m in MODELS:
        for pl in PL:
            r = run("device", m, pl)
            for cell, v in r["inflation"].items():
                if not cell.startswith("control"):
                    continue
                ((label, e),) = v.items()
                agg = r["cells"][cell]["aggressors"][0]
                name = {"cpu": f"{agg['n']} CPU-burning processes", "membw": "memory copy process",
                        "gil": "pure-Python thread in the caller"}[agg["kind"]]
                if agg.get("gb_s_median"):
                    name += f" ({agg['gb_s_median']:.0f} GB/s)"
                sv = e["service"]["mean"]
                print(f"| {MODELS[m]} | {pl} | {label} | {name} | ×{sv['ratio']:.3f} ({sv['ci95'][0]:.3f}–{sv['ci95'][1]:.3f}) "
                      f"| ×{e['device_exec']['mean']['ratio']:.3f} | {e['dispatch_delta_ms']:+.2f} | {e['host_delta_ms']:+.2f} "
                      f"| ×{e['p99_e2e_ratio']:.2f} | {r['cells'][cell]['streams'][label]['n']} |")
    print()


def sweep():
    print("### Load sweep: victim mean service × against the other device's realised busy fraction\n")
    print("| Model | ANE placement | Victim | Other device | points (busy fraction → service ×) |")
    print("|---|---|---|---|---|")
    for m in MODELS:
        for pl in PL:
            for key, pts in run("device", m, pl)["sweep_curve"].items():
                if len(pts) < 3:
                    continue
                v, a = key.split("|")
                print(f"| {MODELS[m]} | {pl} | {v} | {a} | "
                      + ", ".join(f"{p['aggressor_util']:.2f} → ×{p['service_ratio']['ratio']:.3f}" for p in pts) + " |")
    print()


def sparse():
    print("### Sparse load: one device alone, open loop\n")
    print("| Model (placement) | Stream | offered load | n | service × (95% CI) | CPU ms × | device exec × |")
    print("|---|---|---:|---:|---|---:|---:|")
    for m, pl in (("laya-typed-decisions", "thread"), ("laya-multilingual", "process")):
        for label, pts in run("sparse", m, pl)["sparse_curve"].items():
            for p in pts:
                u = f"{p['offered_util']:g}" + (" + 1 CPU-busy process" if p["aggressors"] else "")
                sv = p["service_ratio"]
                print(f"| {MODELS[m]} ({pl}) | {label} | {u} | {p['n']} | ×{sv['ratio']:.2f} ({sv['ci95'][0]:.2f}–{sv['ci95'][1]:.2f}) "
                      f"| ×{p['cpu_ms_ratio']['ratio']:.2f} | ×{p['exec_ratio']['ratio']:.2f} |")
    print()


def part_a():
    print("### Product path, v0.2 Part A mix (closed loop, Laya.submit, v0.2 router)\n")
    print("| Model | ANE placement | Stream | req/s (solo) | P50 | P95 | P99 (solo) | Δ P99 | v0.2 Δ P99 | queue mean "
          "| service mean (solo) | tail: pre / queue / service / post ms | n | mismatches |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    v02 = json.loads((ROOT.parents[1] / "laya_apple/data/placement.json").read_text())["models"]
    for m in MODELS:
        for pl in PL:
            r = run("product", m, pl)
            for label, solo_cell, key in (("A", "product:solo_short", "short"), ("L", "product:solo_long", "long")):
                s = r["cells"]["product:hetero"]["streams"][label]
                so = r["cells"][solo_cell]["streams"][label]
                ta = r["tail_attribution"][f"product:hetero|{label}"]["tail_mean"]
                old = v02[m]["measured"][pl][f"{key}_p99_ms"]
                d = s["e2e"]["p99"] / so["e2e"]["p99"] - 1
                print(f"| {MODELS[m]} | {pl} | {key} L{r['shapes'][label]['length']} | {s['req_s']:.1f} ({so['req_s']:.1f}) "
                      f"| {s['e2e']['p50']:.2f} | {s['e2e']['p95']:.2f} | {s['e2e']['p99']:.2f} ({so['e2e']['p99']:.2f}) "
                      f"| {d:+.0%} | {old / _v02_solo(m, pl, key) - 1:+.0%} | {s['queue']['mean']:.2f} "
                      f"| {s['service']['mean']:.2f} ({so['service']['mean']:.2f}) "
                      f"| {ta['pre']:.2f} / {ta['queue']:.2f} / {ta['service']:.2f} / {ta['post']:.2f} | {s['n']} "
                      f"| {s['mismatches']} |")
    print()


def _v02_solo(m, pl, key):
    d = json.loads((ROOT.parents[1] / f"benchmarks/v0.2/placement-{m}-{pl}.json").read_text())
    return d["part_a"]["summary"][f"solo_{key}"][key]["p99_ms"]


def part_b():
    print("### Product path, open loop (v0.2 Part B class mix): per class\n")
    print("| Model | ANE placement | offered req/s | class | n | GPU share | P50 | P95 | P99 | queue mean | service mean "
          "| tail: queue / service ms | mismatches |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for m in MODELS:
        for pl in PL:
            r = run("product", m, pl)
            for cell, c in r["cells"].items():
                if c["kind"] != "product_open":
                    continue
                ((label, s),) = c["streams"].items()
                ta = r["tail_attribution"][f"{cell}|{label}"]["tail_mean"]
                print(f"| {MODELS[m]} | {pl} | {cell.split('@')[1]} | all | {s['n']} | "
                      f"{s['devices'].get('gpu', 0) / s['n']:.2f} | {s['e2e']['p50']:.1f} | {s['e2e']['p95']:.1f} "
                      f"| {s['e2e']['p99']:.1f} | {s['queue']['mean']:.1f} | {s['service']['mean']:.1f} "
                      f"| {ta['queue']:.1f} / {ta['service']:.1f} | {s['mismatches']} |")
    print()


def routing():
    print("### Routing hindsight (product open loop, short 1-question class)\n")
    print("| Model | ANE placement | offered req/s | decision | n | misses | miss rate | mean miss cost ms |")
    print("|---|---|---:|---|---:|---:|---:|---:|")
    for m in MODELS:
        for pl in PL:
            for key, v in run("product", m, pl)["routing_hindsight"].items():
                for reason, x in v.get("by_reason", {}).items():
                    print(f"| {MODELS[m]} | {pl} | {key.split('@')[1].split('|')[0]} | {reason} | {x['n']} | {x['miss']} "
                          f"| {x['miss_rate']:.1%} | {x['miss_cost_ms_mean']:.1f} |")
    print()


def ab():
    for m, pl in (("laya-typed-decisions", "thread"), ("laya-multilingual", "process")):
        key = f"sched:{m}:{pl}"
        if key not in RES["runs"]:
            continue
        r = RES["runs"][key]
        p = r["contention_params"]
        print(f"### Scheduler prototype A/B: {MODELS[m]} (ANE {pl})\n")
        print(f"Calibrated: GPU service × (1 + {p['gpu']['a']:.3f}) + {p['gpu']['d_ms']:.2f} ms while the ANE is busy; "
              f"ANE service × (1 + {p['ane']['a']:.3f}) + {p['ane']['d_ms']:.2f} ms while the GPU is busy.\n")
        print("| Workload | offered req/s | class | n | P50 base → proto | P95 | P99 | P99 × (95% CI) | mean × "
              "| GPU share | queue mean | service mean | req/s | mismatches |")
        print("|---|---:|---|---:|---|---|---|---|---:|---|---|---|---|---|")
        ab = r["scheduler_ab"]
        for name, e in ab["cells"].items():
            wl, rate = name.split(":")[1].split("@")
            for cls, c in list(e["classes"].items()) + [("all", e["all"])]:
                b, n = c["baseline"], c["contention"]
                pr = c["p99_ratio"]
                extra = ("| – | – | – " if cls == "all" else
                         f"| ×{c['mean_ratio']['ratio']:.3f} | {c['gpu_share'][0]:.3f} → {c['gpu_share'][1]:.3f} "
                         f"| {c['queue_mean'][0]:.1f} → {c['queue_mean'][1]:.1f} "
                         f"| {c['service_mean'][0]:.1f} → {c['service_mean'][1]:.1f} ")
                print(f"| {wl} | {rate} | {cls} | {b['n']} | {b['p50']:.1f} → {n['p50']:.1f} | {b['p95']:.1f} → {n['p95']:.1f} "
                      f"| {b['p99']:.1f} → {n['p99']:.1f} | ×{pr['ratio']:.3f} ({pr['ci95'][0]:.2f}–{pr['ci95'][1]:.2f}) "
                      + (extra if cls != "all" else "| – | – | – | – ")
                      + (f"| {e['req_s'][0]:.1f} → {e['req_s'][1]:.1f} | {e['mismatches'][0]} / {e['mismatches'][1]} |"
                         if cls == "all" else "| | |"))
        print("\nVerdict against the pre-registered criteria:\n")
        print("| Workload | levels | short P99 improved (levels) | throughput within 2% | other-class regressions "
              "| mismatches | passes |")
        print("|---|---:|---:|---|---:|---:|---|")
        for w, v in ab["verdict"].items():
            print(f"| {w.split(':')[1]} | {v['levels']} | {v['short_improved_levels']} | {v['throughput_ok_all']} "
                  f"| {v['other_regressions']} | {v['mismatches']} | {'**yes**' if v['passes'] else 'no'} |")
        print()


if __name__ == "__main__":
    for fn in (solo, matrix, aggregate, controls, sweep, sparse, part_a, part_b, routing, ab):
        fn()
