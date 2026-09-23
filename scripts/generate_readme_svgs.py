"""Generate the README figures from the v1.0 benchmark data.

    uv run python scripts/generate_readme_svgs.py            # writes docs/readme/*.svg
    uv run python scripts/generate_readme_svgs.py --check    # fails if the committed files are stale

- hero-throughput.svg: mixed-workload throughput, GPU-only against GPU + ANE, read from
  benchmarks/v1.0/placement-*.json (the same numbers as `scripts/v1_report.py hetero`).
  The figure carries no release version, so it does not change with every release; the
  README text next to it names the benchmark report and its methodology.
- architecture.svg: request-level routing and concurrent GPU + ANE serving; the ANE limits
  are read from laya_apple/data/routing.json.

Colours are CSS classes with a prefers-color-scheme override, and each figure draws its own
background card with the same override. prefers-color-scheme inside an image follows the
OS, not the GitHub theme; with the card, text and background always match even when the
two disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

from v1_report import MODELS, PLACEMENT, V1

ROOT = Path(__file__).resolve().parents[1]
ROUTING = ROOT / "laya_apple" / "data" / "routing.json"
OUT = ROOT / "docs" / "readme"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

STYLE = """
.bg{fill:#ffffff;stroke:#d1d9e0}
.t{fill:#1f2328}.m{fill:#59636e}.acc{fill:#0969da}.gpu{fill:#8c959f}
.box{fill:#f6f8fa;stroke:#d1d9e0}.boxacc{fill:#ddf4ff;stroke:#0969da}
.ln{stroke:#59636e;fill:none}.lnacc{stroke:#0969da;fill:none}.grid{stroke:#d1d9e0}
@media (prefers-color-scheme: dark){
.bg{fill:#0d1117;stroke:#3d444d}
.t{fill:#e6edf3}.m{fill:#9198a1}.acc{fill:#4493f8}.gpu{fill:#656c76}
.box{fill:#151b23;stroke:#3d444d}.boxacc{fill:#0c2d6b;stroke:#4493f8}
.ln{stroke:#9198a1}.lnacc{stroke:#4493f8}.grid{stroke:#3d444d}
}
"""


def _svg(width: int, height: int, title: str, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title" font-family="{FONT}">',
            f'<title id="title">{escape(title)}</title>',
            f"<style>{STYLE}</style>",
            f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10" class="bg"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def _text(x, y, s, cls="t", size=13, weight=None, anchor=None) -> str:
    attrs = f'x="{x:g}" y="{y:g}" class="{cls}" font-size="{size}"'
    if weight:
        attrs += f' font-weight="{weight}"'
    if anchor:
        attrs += f' text-anchor="{anchor}"'
    return f"<text {attrs}>{escape(s)}</text>"


def throughput_data():
    """[(model, gpu_only req/s, gpu+ane req/s, ratio)] and the platform, from part A of the
    closed-loop mix. Aborts if the models disagree on platform or report any mismatch."""
    rows, platforms = [], set()
    for m in MODELS:
        d = json.loads((V1 / f"placement-{m}-{PLACEMENT[m]}.json").read_text())
        g = d["part_a"]["gate"]
        if g["mismatches"]:
            sys.exit(f"{m}: {g['mismatches']} mismatches in the v1.0 mix; the hero figure assumes none")
        rows.append((m, g["aggregate_gpu_only_req_s"], g["aggregate_hetero_req_s"], g["ratio"]))
        platforms.add((d["platform"]["soc"], d["platform"]["macos"]))
    if len(platforms) != 1:
        sys.exit(f"the v1.0 mix ran on more than one platform: {platforms}")
    return rows, platforms.pop()


def hero() -> str:
    rows, (soc, macos) = throughput_data()
    W, top, row_h = 880, 112, 66
    H = top + row_h * len(rows) + 52
    x0, bar_max, bar_h = 196, 440, 16
    scale = bar_max / max(r[2] for r in rows)
    b = [
        _text(24, 34, "laya-apple: mixed-workload throughput against GPU-only serving", size=18, weight=600),
        _text(
            24,
            58,
            "Heterogeneous MLX GPU + Apple Neural Engine serving · ANE paths parity-validated against upstream Laya",
            cls="m",
        ),
        '<rect x="24" y="76" width="12" height="12" rx="2" class="gpu"/>',
        _text(42, 87, "GPU-only (MLX)", cls="m", size=12),
        '<rect x="150" y="76" width="12" height="12" rx="2" class="acc"/>',
        _text(168, 87, "GPU + ANE (laya-apple)", cls="m", size=12),
        _text(W - 24, 87, "throughput gain", cls="m", size=12, anchor="end"),
    ]
    for i, (m, gpu, het, ratio) in enumerate(rows):
        y = top + i * row_h
        b.append(f'<line x1="24" x2="{W - 24}" y1="{y - 12}" y2="{y - 12}" class="grid"/>')
        b.append(_text(24, y + 25, m, size=14, weight=600))
        for j, (v, cls) in enumerate(((gpu, "gpu"), (het, "acc"))):
            by = y + j * (bar_h + 6)
            w = v * scale
            b.append(f'<rect x="{x0}" y="{by}" width="{w:.1f}" height="{bar_h}" rx="2" class="{cls}"/>')
            b.append(_text(x0 + w + 8, by + 12.5, f"{v:.1f} req/s", cls="t" if j else "m", size=12))
        b.append(_text(W - 24, y + 30, f"{ratio:.2f}×", cls="acc", size=28, weight=700, anchor="end"))
    b.append(f'<line x1="24" x2="{W - 24}" y1="{H - 64}" y2="{H - 64}" class="grid"/>')
    b.append(
        _text(
            24,
            H - 38,
            "One short and one long request stream per model through one "
            'Laya(execution="workers") instance; GPU-only runs both on the MLX worker.',
            cls="m",
            size=12,
        )
    )
    b.append(_text(24, H - 18, f"Measured on {soc} · macOS {macos}", cls="m", size=12))
    title = "Mixed-workload throughput, GPU-only vs GPU + ANE: " + ", ".join(
        f"{m} {gpu:.1f} to {het:.1f} req/s ({r:.2f}×)" for m, gpu, het, r in rows
    )
    return _svg(W, H, title, b)


def _box(x, y, w, h, lines, accent=False) -> list[str]:
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{"boxacc" if accent else "box"}"/>']
    ty = y + h / 2 - (len(lines) - 1) * 9 + 5
    for k, s in enumerate(lines):
        out.append(
            _text(
                x + w / 2,
                ty + k * 18,
                s,
                size=15 if k == 0 else 12,
                weight=600 if k == 0 else None,
                cls="t" if k == 0 else "m",
                anchor="middle",
            )
        )
    return out


def _elbow(x1, y1, xs, y2, x2, accent=False) -> str:
    """Right-angle connector: across to xs, then to y2, then across to x2."""
    cls, marker = ("lnacc", "ha") if accent else ("ln", "hm")
    return (
        f'<path d="M{x1},{y1} H{xs} V{y2} H{x2}" class="{cls}" stroke-width="1.6" '
        f'stroke-linejoin="round" marker-end="url(#{marker})"/>'
    )


def ane_limits() -> tuple[int, int]:
    """(max tokens, max questions) that `auto` sends to the ANE; the same for every model."""
    models = json.loads(ROUTING.read_text())["models"].values()
    limits = {(m["auto_ane_max_len"], m["auto_ane_max_questions"]) for m in models}
    if len(limits) != 1:
        sys.exit(f"auto ANE limits differ between models: {limits}; the figure assumes one")
    return limits.pop()


def architecture() -> str:
    max_len, max_q = ane_limits()
    questions = "one question" if max_q == 1 else f"≤ {max_q} questions"
    W, H = 880, 330
    b = [
        "<defs>"
        '<marker id="hm" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M0,0 L10,5 L0,10 z" class="m"/></marker>'
        '<marker id="ha" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M0,0 L10,5 L0,10 z" class="acc"/></marker>'
        "</defs>",
        *_box(24, 110, 140, 80, ["Request", "predict / submit"]),
        *_box(196, 100, 176, 100, ["Router", "decides before running,", "records routing_reason"]),
        *_box(
            620,
            24,
            236,
            96,
            ["Apple Neural Engine", "Core ML fixed-shape artifact,", "parity-validated on this Mac"],
            accent=True,
        ),
        *_box(620, 180, 236, 96, ["MLX GPU", "up to the model's max length,", "batches multiple questions"]),
        '<line x1="164" y1="150" x2="192" y2="150" class="ln" stroke-width="1.6" marker-end="url(#hm)"/>',
        _elbow(372, 140, 404, 72, 616, accent=True),
        _elbow(372, 160, 404, 228, 616),
        _text(512, 44, f"≤ {max_len} tokens, {questions},", cls="acc", size=12, anchor="middle"),
        _text(512, 62, "validated artifact present", cls="acc", size=12, anchor="middle"),
        _text(512, 248, "longer, several questions,", cls="m", size=12, anchor="middle"),
        _text(512, 266, "unvalidated Mac or no artifact", cls="m", size=12, anchor="middle"),
        '<line x1="680" y1="126" x2="680" y2="174" class="ln" stroke-width="1.2" stroke-dasharray="3 3"/>',
        _text(690, 138, 'with execution="workers":', cls="m", size=12),
        _text(690, 154, "serve independent", cls="m", size=12),
        _text(690, 170, "requests concurrently", cls="m", size=12),
        f'<line x1="24" x2="{W - 24}" y1="{H - 36}" y2="{H - 36}" class="grid"/>',
        _text(
            24,
            H - 14,
            'Routing on an idle machine. With execution="workers", the router also compares queue backlogs, '
            "so a request can go to the other engine.",
            cls="m",
            size=12,
        ),
    ]
    title = (
        f"laya-apple routing: requests of at most {max_len} tokens with {questions} and a validated artifact go "
        "to the Apple Neural Engine; longer, multi-question or unvalidated requests go to the MLX GPU; with "
        'execution="workers" both engines serve independent requests concurrently'
    )
    return _svg(W, H, title, b)


FIGURES = {"hero-throughput.svg": hero, "architecture.svg": architecture}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if a committed figure differs from the data")
    args = ap.parse_args(argv)
    stale = []
    for name, fn in FIGURES.items():
        path, svg = OUT / name, fn()
        if args.check:
            if not path.exists() or path.read_text() != svg:
                stale.append(name)
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(svg)
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        print(f"stale: {', '.join(stale)}; run scripts/generate_readme_svgs.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
