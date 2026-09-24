"""Figures for the request ledger (research/gpu-ane-interference/ledger/), from ledger/results.json only.

    uv run python research/gpu-ane-interference/scripts/figures_ledger.py           # writes ledger/figures/*.svg
    uv run python research/gpu-ane-interference/scripts/figures_ledger.py --check   # fails if stale

Same hand-written SVG style as figures.py (its card, text, legend and palette); the eight
lifecycle phases get one fixed colour each in every figure. Every figure has a <title> with
the numbers it shows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures import legend, svg, text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ledger" / "results.json"
OUT = ROOT / "ledger" / "figures"
PHASES = (
    ("pre", "p1", "prepare + route + enqueue"),
    ("queue", "p2", "queue"),
    ("dispatch", "p3", "dispatch"),
    ("device_exec", "p4", "device exec"),
    ("host", "p5", "host"),
    ("unattributed", "p6", "unattributed"),
    ("return", "p7", "return"),
    ("postprocess", "p8", "postprocess"),
)
PHASE_STYLE = (
    "<style>.p1{fill:#8c959f}.p2{fill:#d1d9e0}.p3{fill:#eb6834}.p4{fill:#2a78d6}.p5{fill:#1baf7a}"
    ".p6{fill:#eda100}.p7{fill:#cf2e6e}.p8{fill:#8250df}"
    "@media (prefers-color-scheme: dark){.p1{fill:#656c76}.p2{fill:#3d444d}.p3{fill:#d95926}.p4{fill:#3987e5}"
    ".p5{fill:#199e70}.p6{fill:#c98500}.p7{fill:#db4f86}.p8{fill:#9a6ef0}}</style>"
)
LANES = (("gpu", "GPU"), ("ane", "ANE"))


def _legend_rows(x, y, items, per_row=4):
    out = []
    for i in range(0, len(items), per_row):
        out += legend(x, y + 20 * (i // per_row), items[i : i + per_row])
    return out


def fig_timeline(res):
    """Two lanes on one time axis: each request's phases, stacked in time order."""
    # the thread-placed device run carries the study's headline effect; else any device run
    keys = sorted(res["runs"])
    key = next((k for k in keys if k.startswith("device:") and k.endswith(":thread")), None)
    key = key or next((k for k in keys if k.startswith("device:")), keys[0])
    tl = res["runs"][key]["timeline"]
    rows = {dev: max([q["row"] for q in tl["lanes"][dev]], default=-1) + 1 for dev, _ in LANES}
    rh, gap = 16, 26
    W, x0, x1, top = 900, 80, 880, 118
    H = top + sum(max(r, 1) * (rh + 4) + gap for r in rows.values()) + 50
    span = tl["span_ms"]
    px = lambda v: x0 + min(max(v, 0.0), span) / span * (x1 - x0)  # noqa: E731
    b = [
        PHASE_STYLE,
        text(
            24,
            32,
            f"Request lifecycle, {key} — {tl['cell']} cycle {tl['cycle']} ({span:.0f} ms slice)",
            size=16,
            weight=600,
        ),
        *_legend_rows(24, 60, [(c, lab) for _, c, lab in PHASES]),
    ]
    for t in range(0, int(span) + 1, 50):
        b.append(f'<line x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{top - 6}" y2="{H - 44}" class="grid"/>')
        b.append(text(px(t), H - 30, f"{t}", cls="m", size=11, anchor="middle"))
    b.append(
        text((x0 + x1) / 2, H - 12, f"ms after {tl['t0_ms']:.1f} ms (run clock)", cls="m", size=11, anchor="middle")
    )
    y, summary = top, []
    for dev, name in LANES:
        reqs = tl["lanes"][dev]
        b.append(text(x0 - 10, y + 12, name, size=12, anchor="end", weight=600))
        for q in reqs:
            ry = y + q["row"] * (rh + 4)
            cx = q["start_ms"]
            for seg, dur in q["segments"]:
                a, z = px(cx), px(cx + max(dur, 0.0))
                if z - a > 0.2:
                    cls = next(c for s, c, _ in PHASES if s == seg)
                    b.append(f'<rect x="{a:.1f}" y="{ry}" width="{z - a:.1f}" height="{rh}" class="{cls}"/>')
                cx += max(dur, 0.0)
        if reqs:
            disp = sum(dict(q["segments"])["dispatch"] for q in reqs) / len(reqs)
            summary.append(f"{name}: {len(reqs)} requests, mean dispatch {disp:.2f} ms")
        y += max(rows[dev], 1) * (rh + 4) + gap
    title = f"Request timeline, {key}, {tl['cell']} cycle {tl['cycle']}: " + "; ".join(summary)
    return svg(W, H, title, b)


def _decomp_rows(res):
    out = []
    for key, run in res["runs"].items():
        for cell, labels in run.get("decomposition", {}).items():
            for label, d in labels.items():
                out.append((key, cell, label, d))
    return out


def fig_decomposition(res):
    """Per device stream: solo vs concurrent mean of each phase, stacked; historical dispatch delta noted."""
    rows = _decomp_rows(res)
    W, top, bh, gh = 900, 120, 16, 72
    H = top + gh * max(len(rows), 1) + 40
    x0, x1 = 330, 800
    mx = (
        max(
            [
                sum(max(d["components"][p][side] or 0.0, 0.0) for p, _, _ in PHASES)
                for *_, d in rows
                for side in ("solo_mean", "concurrent_mean")
            ]
            or [1.0]
        )
        * 1.05
    )
    b = [
        PHASE_STYLE,
        text(24, 32, "Where a request's time goes: solo vs with the other device busy (mean ms)", size=16, weight=600),
        *_legend_rows(24, 60, [(c, lab) for _, c, lab in PHASES]),
    ]
    title = []
    for i, (key, cell, label, d) in enumerate(rows):
        y = top + i * gh
        b.append(text(x0 - 72, y + 12, f"{key.split(':', 1)[1]} {label}", size=11, anchor="end"))
        b.append(text(x0 - 72, y + 12 + bh + 4, cell, cls="m", size=10, anchor="end"))
        for j, side in enumerate(("solo_mean", "concurrent_mean")):
            by = y + j * (bh + 4)
            b.append(text(x0 - 6, by + 12, "solo" if j == 0 else "concurrent", cls="m", size=10, anchor="end"))
            cx = x0
            for p, cls, _ in PHASES:
                w = max(d["components"][p][side] or 0.0, 0.0) / mx * (x1 - x0)
                if w > 0.3:
                    b.append(f'<rect x="{cx:.1f}" y="{by}" width="{w:.1f}" height="{bh}" class="{cls}"/>')
                cx += w
            e2e = d["components"]["e2e"][side]
            b.append(text(cx + 6, by + 12, f"{e2e:.1f} ms" if e2e is not None else "", cls="m", size=10))
        c = d["comparison"]["dispatch"]
        hist, new = c["historical_dispatch_delta_ms"], c["new_dispatch_plus_return_delta_ms"]
        note = f"Δ dispatch+return {new:+.2f} ms" if new is not None else ""
        if hist is not None:
            note += f" (historical Δ dispatch {hist:+.2f})"
        b.append(text(x0, y + 2 * (bh + 4) + 10, note, cls="m", size=10))
        title.append(
            f"{key} {cell} {label}: e2e solo {d['components']['e2e']['solo_mean']:.2f} ms, concurrent "
            f"{d['components']['e2e']['concurrent_mean']:.2f} ms; {note}"
        )
    if not rows:
        b.append(text(W / 2, top + 20, "no solo + matrix pair in these runs", cls="m", size=12, anchor="middle"))
    b.append(text(x0, H - 20, "milliseconds (mean over in-window requests)", cls="m", size=11))
    return svg(W, H, "Service decomposition. " + "; ".join(title), b)


def fig_prediction(res):
    """Mean signed completion error per (device, other-device state), whiskers P10-P90."""
    scopes = res["prediction"]["scopes"]
    order = sorted(scopes, key=lambda s: (scopes[s]["plan"] != "product", not s.startswith("pooled:"), s))
    scope = order[0] if order else None
    groups = scopes[scope]["groups"].get("target|other_device", {}) if scope else {}
    items = sorted(groups.items())
    W, H = 900, 380
    x0, x1, y0, y1 = 90, W - 30, 100, 290
    lo = min([g["p10_signed_ms"] for _, g in items] + [0.0])
    hi = max([g["p90_signed_ms"] for _, g in items] + [1.0])
    raw = (hi - lo) / 5
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    lo, hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
    py = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)  # noqa: E731
    b = [
        text(24, 32, f"Completion prediction error (actual − predicted), {scope}", size=16, weight=600),
        *legend(24, 60, [("s1", "mean signed error"), ("s2", "P10–P90 of signed error")]),
    ]
    for k in range(round((hi - lo) / step) + 1):
        v = round(lo + k * step, 9) or 0.0  # no "-0.0"
        b.append(f'<line x1="{x0}" x2="{x1}" y1="{py(v):.1f}" y2="{py(v):.1f}" class="grid"/>')
        b.append(text(x0 - 6, py(v) + 4, f"{v:+.1f}", cls="m", size=11, anchor="end"))
    b.append(f'<line x1="{x0}" x2="{x1}" y1="{py(0):.1f}" y2="{py(0):.1f}" class="ax"/>')
    b.append(
        f'<text x="{x0 - 56}" y="{(y0 + y1) / 2:.1f}" class="m" font-size="11" text-anchor="middle" '
        f'transform="rotate(-90 {x0 - 56} {(y0 + y1) / 2:.1f})">ms (positive: later than predicted)</text>'
    )
    cw = (x1 - x0) / max(len(items), 1)
    title = []
    for i, (name, g) in enumerate(items):
        cx = x0 + cw * (i + 0.5)
        m = g["mean_signed_ms"]
        b.append(
            f'<rect x="{cx - 14:.1f}" y="{min(py(m), py(0)):.1f}" width="28" '
            f'height="{abs(py(m) - py(0)):.1f}" rx="2" class="s1"/>'
        )
        b.append(
            f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{py(g["p90_signed_ms"]):.1f}" '
            f'y2="{py(g["p10_signed_ms"]):.1f}" class="s2" stroke-width="2"/>'
        )
        for v in (g["p10_signed_ms"], g["p90_signed_ms"]):
            b.append(
                f'<line x1="{cx - 6:.1f}" x2="{cx + 6:.1f}" y1="{py(v):.1f}" y2="{py(v):.1f}" class="s2" '
                f'stroke-width="2"/>'
            )
        dev, state = name.split("|")
        b.append(text(cx, y1 + 18, dev.upper(), size=11, anchor="middle"))
        b.append(text(cx, y1 + 34, f"other {state}", cls="m", size=10, anchor="middle"))
        b.append(text(cx, y1 + 50, f"n={g['n']}", cls="m", size=10, anchor="middle"))
        title.append(
            f"{name}: mean {m:+.2f} ms (P10 {g['p10_signed_ms']:+.2f}, P90 {g['p90_signed_ms']:+.2f}, n={g['n']})"
        )
    if not items:
        b.append(text(W / 2, (y0 + y1) / 2, "no traced requests", cls="m", size=12, anchor="middle"))
    return svg(W, H, f"Prediction error by device and other-device state, {scope}: " + "; ".join(title), b)


FIGURES = {
    "request-timeline.svg": fig_timeline,
    "service-decomposition.svg": fig_decomposition,
    "prediction-error.svg": fig_prediction,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    res = json.loads(args.results.read_text())
    stale = []
    for name, fn in FIGURES.items():
        content = fn(res)
        path = args.out / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(name)
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            print("wrote", path)
    if stale:
        sys.exit(f"stale figures: {stale}; re-run figures_ledger.py")


if __name__ == "__main__":
    main()
