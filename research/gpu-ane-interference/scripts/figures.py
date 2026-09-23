"""Figures for research/gpu-ane-interference/README.md, from results.json only.

    uv run python research/gpu-ane-interference/scripts/figures.py           # writes figures/*.svg
    uv run python research/gpu-ane-interference/scripts/figures.py --check   # fails if stale

Colours follow scripts/generate_readme_svgs.py (a background card and a prefers-color-scheme
override); the series colours are one fixed categorical order, the same series always the
same colour. Every figure has a <title> naming what it shows; the README has the numbers as
tables as well.
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results.json"
OUT = ROOT / "figures"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
STYLE = """
.bg{fill:#ffffff;stroke:#d1d9e0}.t{fill:#1f2328}.m{fill:#59636e}.grid{stroke:#e6e9ec}.ax{stroke:#8c959f}
.s1{fill:#2a78d6;stroke:#2a78d6}.s2{fill:#eb6834;stroke:#eb6834}.s3{fill:#1baf7a;stroke:#1baf7a}
.s4{fill:#eda100;stroke:#eda100}.ln{fill:none;stroke-width:2}.ring{stroke:#ffffff;stroke-width:2}
.c1{fill:#8c959f}.c2{fill:#2a78d6}.c3{fill:#eb6834}.c4{fill:#1baf7a}
@media (prefers-color-scheme: dark){
.bg{fill:#0d1117;stroke:#3d444d}.t{fill:#e6edf3}.m{fill:#9198a1}.grid{stroke:#21262d}.ax{stroke:#656c76}
.s1{fill:#3987e5;stroke:#3987e5}.s2{fill:#d95926;stroke:#d95926}.s3{fill:#199e70;stroke:#199e70}
.s4{fill:#c98500;stroke:#c98500}.ring{stroke:#0d1117}
.c1{fill:#656c76}.c2{fill:#3987e5}.c3{fill:#d95926}.c4{fill:#199e70}
}
"""
MODELS = {"laya-typed-decisions": "typed-decisions", "laya-multilingual": "multilingual"}


def svg(w, h, title, body):
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-labelledby="title" font-family="{FONT}">',
            f'<title id="title">{escape(title)}</title>',
            f"<style>{STYLE}</style>",
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" class="bg"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def text(x, y, s, cls="t", size=12, anchor=None, weight=None):
    a = f'x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}"'
    if anchor:
        a += f' text-anchor="{anchor}"'
    if weight:
        a += f' font-weight="{weight}"'
    return f"<text {a}>{escape(s)}</text>"


class Panel:
    """A plot area with linear axes and light gridlines."""

    def __init__(self, x, y, w, h, xr, yr, xticks, yticks, xlabel, ylabel, title, xfmt="{:g}", yfmt="{:g}"):
        self.x, self.y, self.w, self.h, self.xr, self.yr = x, y, w, h, xr, yr
        self.body = [text(x, y - 12, title, size=13, weight=600)]
        for t in yticks:
            py = self.py(t)
            self.body.append(f'<line x1="{x}" x2="{x + w}" y1="{py:.1f}" y2="{py:.1f}" class="grid"/>')
            self.body.append(text(x - 6, py + 4, yfmt.format(t), cls="m", size=11, anchor="end"))
        for t in xticks:
            px = self.px(t)
            self.body.append(text(px, y + h + 16, xfmt.format(t), cls="m", size=11, anchor="middle"))
        self.body.append(f'<line x1="{x}" x2="{x + w}" y1="{y + h}" y2="{y + h}" class="ax"/>')
        self.body.append(text(x + w / 2, y + h + 34, xlabel, cls="m", size=11, anchor="middle"))
        self.body.append(
            f'<text x="{x - 44:.1f}" y="{y + h / 2:.1f}" class="m" font-size="11" text-anchor="middle" '
            f'transform="rotate(-90 {x - 44:.1f} {y + h / 2:.1f})">{escape(ylabel)}</text>'
        )

    def px(self, v):
        return self.x + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def py(self, v):
        return self.y + self.h - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def line(self, pts, cls, dashed=False, label=None):
        d = " ".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.body.append(f'<polyline points="{d}" class="{cls} ln" style="fill:none"{dash}/>')
        for x, y in pts:
            self.body.append(f'<circle cx="{self.px(x):.1f}" cy="{self.py(y):.1f}" r="4" class="{cls} ring"/>')
        if label:
            x, y = pts[-1]
            self.body.append(text(self.px(x) + 8, self.py(y) + 4, label, size=11))


def legend(x, y, items):
    out, cx = [], x
    for cls, label in items:
        out.append(f'<rect x="{cx}" y="{y - 9}" width="12" height="12" rx="2" class="{cls}"/>')
        out.append(text(cx + 17, y + 1, label, cls="m", size=12))
        cx += 17 + 7.2 * len(label) + 18
    return out


# ----------------------------------------------------------------------------- figures


def fig_inflation(res):
    """Mean service-time inflation per stream in every matrix cell, thread vs process."""
    rows = []
    for model, short in MODELS.items():
        for cell in ("matrix:gpu_M+ane_S", "matrix:gpu_M+ane_B", "matrix:gpu_L+ane_S", "matrix:gpu_L+ane_B"):
            for dev in ("gpu", "ane"):
                vals = {}
                for pl in ("thread", "process"):
                    run = res["runs"][f"device:{model}:{pl}"]
                    shapes = run["shapes"]
                    label = next(lab for lab in run["inflation"][cell] if lab.startswith(dev))
                    vals[pl] = run["inflation"][cell][label]["service"]["mean"]["ratio"]
                g, a = cell.split(":")[1].split("+")
                gl, al = shapes[g.split("_")[1]]["length"], shapes[a.split("_")[1]]["length"]
                who = f"GPU L{gl} (with ANE L{al})" if dev == "gpu" else f"ANE L{al} (with GPU L{gl})"
                rows.append((f"{short}: {who}", vals))
    W, top, rh = 900, 92, 22
    H = top + rh * len(rows) + 70
    x0, x1 = 360, W - 40
    lo, hi = 0.9, 1.8
    px = lambda v: x0 + (min(v, hi) - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
    b = [
        text(24, 32, "Service-time inflation with the other device busy (closed loop, mean vs solo)", size=16,
             weight=600),
        *legend(24, 60, [("s1", "ANE on a thread (GPU in a process)"), ("s2", "ANE in a worker process")]),
    ]
    for t in (1.0, 1.2, 1.4, 1.6, 1.8):
        b.append(f'<line x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{top - 8}" y2="{top + rh * len(rows)}" class="grid"/>')
        b.append(text(px(t), top + rh * len(rows) + 16, f"×{t:.1f}", cls="m", size=11, anchor="middle"))
    for i, (label, vals) in enumerate(rows):
        y = top + i * rh + 6
        b.append(text(x0 - 12, y + 4, label, size=11, anchor="end"))
        for pl, cls in (("thread", "s1"), ("process", "s2")):
            b.append(f'<circle cx="{px(vals[pl]):.1f}" cy="{y}" r="5" class="{cls} ring"/>')
    b.append(text(x0 + (x1 - x0) / 2, H - 24, "service time with the other device busy ÷ service time alone",
                  cls="m", size=11, anchor="middle"))
    title = "Service-time inflation per stream in each GPU+ANE matrix cell: " + "; ".join(
        f"{lab} thread ×{v['thread']:.2f} process ×{v['process']:.2f}" for lab, v in rows
    )
    return svg(W, H, title, b)


def fig_sweep(res):
    """Victim service ratio vs the aggressor device's realised utilisation."""
    W, H = 900, 560
    panels, b, title = [], [text(24, 32, "Victim service time vs the other device's utilisation (load sweep)",
                                 size=16, weight=600)], []
    series = [("gpu_M|ane_B", "s1", "GPU short, ANE busy"), ("gpu_L|ane_B", "s2", "GPU long, ANE busy"),
              ("ane_B|gpu_M", "s3", "ANE, GPU short busy"), ("ane_B|gpu_L", "s4", "ANE, GPU long busy")]
    b += legend(24, 60, [(c, lab) for _, c, lab in series])
    for i, model in enumerate(MODELS):
        for j, pl in enumerate(("thread", "process")):
            p = Panel(90 + j * 420, 110 + i * 230, 340, 150, (0, 1), (0.9, 1.7), (0, 0.25, 0.5, 0.75, 1),
                      (1.0, 1.2, 1.4, 1.6), "other device's busy fraction", "service ÷ solo",
                      f"{MODELS[model]}, ANE {'on a thread' if pl == 'thread' else 'in a process'}", yfmt="×{:.1f}")
            curve = res["runs"][f"device:{model}:{pl}"]["sweep_curve"]
            for key, cls, lab in series:
                pts = [(q["aggressor_util"], q["service_ratio"]["ratio"]) for q in curve.get(key, [])]
                if pts:
                    p.line(pts, cls)
                    title.append(f"{MODELS[model]} {pl} {lab}: " + ", ".join(f"{u:.2f}→×{r:.2f}" for u, r in pts))
            panels.append(p)
    for p in panels:
        b += p.body
    return svg(W, H, "Load sweep. " + "; ".join(title), b)


def fig_sparse(res):
    """One device alone at low offered load: service time vs its closed-loop solo value."""
    W, H = 900, 330
    b = [text(24, 32, "One device alone: service time rises as the machine idles (no second device running)",
              size=16, weight=600)]
    series = [("gpu_M", "s1", "GPU L128"), ("gpu_L", "s2", "GPU max length"), ("ane_B", "s3", "ANE longest bucket")]
    b += legend(24, 60, [(c, lab) for _, c, lab in series] + [("c1", "hollow: same, one CPU-busy process")])
    title = []
    for j, (model, pl) in enumerate((("laya-typed-decisions", "thread"), ("laya-multilingual", "process"))):
        p = Panel(90 + j * 420, 110, 340, 150, (0, 1), (0.8, 2.6), (0, 0.25, 0.5, 0.75, 1), (1.0, 1.5, 2.0, 2.5),
                  "offered load (fraction of closed-loop capacity)", "service ÷ closed-loop solo", MODELS[model],
                  yfmt="×{:.1f}")
        curve = res["runs"][f"sparse:{model}:{pl}"]["sparse_curve"]
        for key, cls, lab in series:
            pts = [(q["offered_util"], q["service_ratio"]["ratio"]) for q in curve.get(key, []) if not q["aggressors"]]
            pts.append((1.0, 1.0))  # closed loop = the reference
            p.line(pts, cls)
            for q in curve.get(key, []):
                if q["aggressors"]:
                    p.body.append(f'<circle cx="{p.px(q["offered_util"]):.1f}" cy="{p.py(q["service_ratio"]["ratio"]):.1f}" '
                                  f'r="5" class="{cls}" style="fill:none" stroke-width="2"/>')
            title.append(f"{MODELS[model]} {lab}: " + ", ".join(f"{u:g}→×{r:.2f}" for u, r in pts))
        b += p.body
    return svg(W, H, "Sparse load. " + "; ".join(title), b)


def fig_tail(res):
    """Part A short stream: the mean of each latency component over the requests at or above P99."""
    rows = []
    for model, short in MODELS.items():
        for pl in ("thread", "process"):
            ta = res["runs"][f"product:{model}:{pl}"]["tail_attribution"]
            for cond, key in (("alone", "product:solo_short|A"), ("with the long stream", "product:hetero|A")):
                t = ta[key]
                rows.append((f"{short}, ANE {pl}, {cond}", t["tail_mean"], t["p99_e2e_ms"]))
    W, top, rh = 900, 96, 30
    H = top + rh * len(rows) + 60
    x0, x1 = 330, W - 90
    mx = max(sum(r[1].values()) for r in rows) * 1.05
    comps = [("pre", "c1", "prompt + routing"), ("queue", "c2", "queue"), ("service", "c3", "service"),
             ("post", "c4", "wake + answer")]
    b = [text(24, 32, "Short-stream tail (requests at or above P99), closed loop: where the time goes", size=16,
              weight=600), *legend(24, 62, [(c, lab) for _, c, lab in comps])]
    for i, (label, parts, p99) in enumerate(rows):
        y = top + i * rh
        b.append(text(x0 - 12, y + 13, label, size=11, anchor="end"))
        cx = x0
        for k, cls, _ in comps:
            w = parts[k] / mx * (x1 - x0)
            if w > 0.5:
                b.append(f'<rect x="{cx:.1f}" y="{y}" width="{max(w - 2, 0.5):.1f}" height="18" rx="2" class="{cls}"/>')
            cx += w
        b.append(text(cx + 8, y + 13, f"P99 {p99:.1f} ms", cls="m", size=11))
    b.append(text(x0, H - 24, "milliseconds (mean over the tail requests)", cls="m", size=11))
    title = "Short-stream tail composition. " + "; ".join(
        f"{lab}: " + ", ".join(f"{k} {v:.2f} ms" for k, v in parts.items()) for lab, parts, _ in rows
    )
    return svg(W, H, title, b)


def fig_ab(res):
    """Scheduler prototype A/B: short-class P99 ratio (prototype / v0.2) per workload and load,
    pooled with its bootstrap CI, and per cycle."""
    cols = []
    for model, pl in (("laya-typed-decisions", "thread"), ("laya-multilingual", "process")):
        run = res["runs"].get(f"sched:{model}:{pl}")
        if not run:
            continue
        cells = run["scheduler_ab"]["cells"]
        for kind in ("open", "heavy"):
            for name in sorted((n for n in cells if f":{kind}@" in n), key=lambda n: float(n.split("@")[1])):
                e = cells[name]
                cyc = [n / b for b, n in e["short_p99_by_cycle"].values() if b and n]
                cols.append((MODELS[model], kind, name.split("@")[1], e["classes"]["A"]["p99_ratio"], cyc))
    W, H = 900, 400
    x0, x1, y0, y1 = 90, W - 30, 100, 290
    lo, hi = 0.0, 2.6
    py = lambda v: y1 - (min(max(v, lo), hi) - lo) / (hi - lo) * (y1 - y0)  # noqa: E731
    b = [text(24, 32, "Scheduler prototype vs v0.2 router: short-request P99 ratio (below 1 = prototype better)",
              size=16, weight=600),
         *legend(24, 60, [("s1", "pooled over 3 cycles, bootstrap 95% CI"), ("s2", "each cycle")])]
    for t in (0.5, 1.0, 1.5, 2.0, 2.5):
        b.append(f'<line x1="{x0}" x2="{x1}" y1="{py(t):.1f}" y2="{py(t):.1f}" class="grid"/>')
        b.append(text(x0 - 6, py(t) + 4, f"×{t:.1f}", cls="m", size=11, anchor="end"))
    b.append(f'<line x1="{x0}" x2="{x1}" y1="{py(0.9):.1f}" y2="{py(0.9):.1f}" class="ax" stroke-dasharray="4 4"/>')
    b.append(text(x0 + 4, py(0.9) + 14, "success threshold ×0.9", cls="m", size=10))
    step = (x1 - x0) / max(1, len(cols))
    title = []
    for i, (m, kind, rate, pr, cyc) in enumerate(cols):
        cx = x0 + step * (i + 0.5)
        lo_ci, hi_ci = pr["ci95"]
        b.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{py(hi_ci):.1f}" y2="{py(lo_ci):.1f}" class="s1" stroke-width="2"/>')
        b.append(f'<circle cx="{cx:.1f}" cy="{py(pr["ratio"]):.1f}" r="5" class="s1 ring"/>')
        for k, c in enumerate(cyc):
            b.append(f'<circle cx="{cx + 12 + 5 * k:.1f}" cy="{py(c):.1f}" r="3.5" class="s2"/>')
        b.append(text(cx, y1 + 16, f"{rate}", cls="m", size=10, anchor="middle"))
        title.append(f"{m} {kind} {rate} req/s ×{pr['ratio']:.2f} (CI {lo_ci:.2f}-{hi_ci:.2f}; cycles "
                     + ", ".join(f"×{c:.2f}" for c in cyc) + ")")
    groups = []
    for i, (m, kind, *_rest) in enumerate(cols):
        if not groups or groups[-1][0] != (m, kind):
            groups.append([(m, kind), i, i])
        groups[-1][2] = i
    for (m, kind), a, z in groups:
        cx = x0 + step * ((a + z) / 2 + 0.5)
        b.append(text(cx, y1 + 34, f"{m}, {'Part B mix' if kind == 'open' else '90% short'}", size=11, anchor="middle"))
        b.append(text(cx, y1 + 50, "offered req/s", cls="m", size=10, anchor="middle"))
    return svg(W, H, "Scheduler A/B, short P99 ratio prototype/v0.2: " + "; ".join(title), b)


FIGURES = {
    "inflation-matrix.svg": fig_inflation,
    "load-sweep.svg": fig_sweep,
    "sparse-load.svg": fig_sparse,
    "tail-composition.svg": fig_tail,
    "scheduler-ab.svg": fig_ab,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    res = json.loads(RESULTS.read_text())
    OUT.mkdir(exist_ok=True)
    stale = []
    for name, fn in FIGURES.items():
        content = fn(res)
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(name)
        else:
            path.write_text(content)
            print("wrote", path)
    if stale:
        sys.exit(f"stale figures: {stale}; re-run figures.py")


if __name__ == "__main__":
    main()
