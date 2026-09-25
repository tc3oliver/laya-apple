"""Generate the README figures from the v1.0 benchmark data.

    uv run python scripts/generate_readme_svgs.py            # writes docs/readme/*.svg
    uv run python scripts/generate_readme_svgs.py --check    # fails if the committed files are stale
    uv run python scripts/generate_readme_svgs.py --social-png   # also renders docs/readme/social-preview.png

- hero-throughput.svg: mixed-workload throughput, GPU-only against GPU + ANE, read from
  benchmarks/v1.0/placement-*.json (the same numbers as `scripts/v1_report.py hetero`).
  The figure carries no release version, so it does not change with every release; the
  README text next to it names the benchmark report and its methodology.
- architecture.svg: request-level routing and concurrent GPU + ANE serving; the ANE limits
  are read from laya_apple/data/routing.json.
- serve-demo.svg: a terminal session in which the released Jev Python SDK, unmodified, gets
  its answer from `laya-apple serve`. It is drawn from docs/readme/serve-demo.json, which
  scripts/capture_serve_demo.py records from a real run; every output line is the recorded
  output of its command.
- serve-llm-load.svg: short-decision P99 and local-LLM tok/s, serve `--device gpu` against
  `auto`, with the LLM idle and busy, read from benchmarks/serve/m4-max-r2/results.json (the
  platform from its raw/campaign.json). Refuses a run that is not valid under its own checks.
- social-preview.png: the repository's 1280x640 link preview, with the best throughput gain
  from the same data and the hard-mismatch count from benchmarks/v1.0/parity. It is
  rendered by headless Google Chrome, so it is not part of --check; regenerate it with
  --social-png when the data changes, then upload it under Settings -> Social preview
  (GitHub has no API for that).

Colours are CSS classes with a prefers-color-scheme override, and each figure draws its own
background card with the same override. prefers-color-scheme inside an image follows the
OS, not the GitHub theme; with the card, text and background always match even when the
two disagree.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path

from v1_report import MODELS, PLACEMENT, V1, _mismatches, _parity

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


SERVE_DEMO = OUT / "serve-demo.json"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"
TERM = {"bg": "#0d1117", "bar": "#161b22", "edge": "#3d444d", "text": "#e6edf3", "out": "#c9d1d9"}
TERM.update(prompt="#3fb950", comment="#8b949e")


def _mono(x, y, spans: list[tuple[str, str]], size=13) -> str:
    """One terminal line: consecutive (text, colour) spans, whitespace preserved."""
    parts = "".join(f'<tspan fill="{c}">{escape(t)}</tspan>' for t, c in spans if t)
    return (
        f'<text x="{x:g}" y="{y:g}" font-family="{MONO}" font-size="{size}" '
        f'xml:space="preserve" style="white-space:pre">{parts}</text>'
    )


def serve_demo_lines(rec: dict) -> list[list[tuple[str, str]]]:
    """The recorded session as terminal lines of (text, colour) spans."""
    lines = []
    for step in rec["session"]:
        if "comment" in step:
            lines.append([(f"# {step['comment']}", TERM["comment"])])
            continue
        cmd = step["cmd"].split("\n")
        first = [("$ ", TERM["prompt"]), (cmd[0], TERM["text"])]
        if step.get("note"):
            first.append((f"   # {step['note']}", TERM["comment"]))
        lines.append(first)
        lines += [[(c, TERM["text"])] for c in cmd[1:]]
        out = step.get("out", "")
        if out:
            lines += [[(o, TERM["out"])] for o in out.rstrip("\n").split("\n")]
    return lines


def serve_demo() -> str:
    rec = json.loads(SERVE_DEMO.read_text())
    lines = serve_demo_lines(rec)
    size, line_h, char_w, pad = 13, 20, 7.9, 24
    longest = max(sum(len(t) for t, _ in spans) for spans in lines)
    W = max(880, int(2 * pad + char_w * longest + 0.999))
    top = 44 + 26
    H = top + line_h * len(lines) + 30
    plat = rec["platform"]
    caption = (
        f"Recorded {rec['recorded']} on {plat['soc']}, macOS {plat['macos']}, laya-apple {rec['laya_apple']}, "
        f"{rec['client']['name']} {rec['client']['version']}"
    )
    b = [
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="10" fill="{TERM["bg"]}" stroke="{TERM["edge"]}"/>',
        f'<path d="M0.5,44 V10.5 a10,10 0 0 1 10,-10 H{W - 10.5} a10,10 0 0 1 10,10 V44 z" fill="{TERM["bar"]}"/>',
        f'<line x1="0.5" x2="{W - 0.5}" y1="44" y2="44" stroke="{TERM["edge"]}"/>',
        *(
            f'<circle cx="{22 + 20 * k}" cy="22" r="6" fill="{c}"/>'
            for k, c in enumerate(("#ff5f57", "#febc2e", "#28c840"))
        ),
        f'<text x="{W / 2:g}" y="27" fill="{TERM["comment"]}" font-size="13" text-anchor="middle">'
        f"{escape(caption)}</text>",
    ]
    for i, spans in enumerate(lines):
        b.append(_mono(pad, top + i * line_h, spans, size))
    client = rec["client"]
    answer = rec["response"]["answers"]
    (qid,) = answer
    la = rec["response"]["laya_apple"]
    title = (
        f"Terminal: laya-apple serve runs locally; {client['name']} {client['version']}, unmodified, "
        f"is pointed at it with TYPESAFE_BASE_URL and gets the answer {answer[qid]['choice']!r}, "
        f"which laya-apple reports was answered by the {la['model']} checkpoint on the {la['device'].upper()}"
    )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="title" font-family="{FONT}">',
            f'<title id="title">{escape(title)}</title>',
            *b,
            "</svg>",
            "",
        ]
    )


SERVE_LLM_RUN = ROOT / "benchmarks" / "serve" / "m4-max-r2"


def serve_llm_data() -> dict:
    """Short-decision P99 and LLM tok/s per cell of the serve-beside-an-LLM run. Aborts if the
    run is not valid under its own checks: an invalid run draws no result, so no figure."""
    res = json.loads((SERVE_LLM_RUN / "results.json").read_text())
    if res.get("smoke") or not res.get("valid"):
        sys.exit(f"{SERVE_LLM_RUN.name}: not a valid campaign; the figure draws only valid runs")
    camp = json.loads((SERVE_LLM_RUN / "raw" / "campaign.json").read_text())
    cells = res["cells"]

    def p99(cell):
        return cells[cell]["classes"]["short_1q"]["p99_ms_median"]

    return {
        "run": res["campaign"],
        "platform": camp["platform"],
        "llm_model": res["llm_model"],
        "offered_req_s": res["offered_req_s"],
        "windows": cells["auto/decisions_llm"]["windows"],
        "alone_windows": cells["llm_alone"]["windows"],
        "p99": [
            ("serve --device gpu", "LLM idle", p99("gpu/decisions"), "gpu"),
            ("serve --device gpu", "LLM busy", p99("gpu/decisions_llm"), "gpu"),
            ("serve auto", "LLM idle", p99("auto/decisions"), "acc"),
            ("serve auto", "LLM busy", p99("auto/decisions_llm"), "acc"),
        ],
        "tok_s": [
            ("LLM alone", cells["llm_alone"]["llm_tok_s_median"], None, "m"),
            (
                "LLM + serve --device gpu",
                cells["gpu/decisions_llm"]["llm_tok_s_median"],
                res["llm_tok_s_drop"]["gpu"],
                "gpu",
            ),
            ("LLM + serve auto", cells["auto/decisions_llm"]["llm_tok_s_median"], res["llm_tok_s_drop"]["auto"], "acc"),
        ],
    }


def serve_llm() -> str:
    d = serve_llm_data()
    W, x_label, x0, bar_max, bar_h, row_h = 880, 24, 260, 430, 16, 28
    b = [
        _text(24, 34, "laya-apple serve beside a busy local LLM", size=18, weight=600),
        _text(
            24,
            58,
            f"Decisions at {d['offered_req_s']:g} req/s while the LLM generates at saturation · "
            "serve auto sends short decisions to the Neural Engine",
            cls="m",
        ),
    ]
    y = 96
    b.append(_text(24, y, "Short-decision P99 latency, ms (lower is better)", size=14, weight=600))
    y += 16
    scale = bar_max / max(v for _, _, v, _ in d["p99"])
    for cfg, state, v, cls in d["p99"]:
        busy = state == "LLM busy"
        b.append(_text(x_label, y + 12.5, f"{cfg} · {state}", cls="t" if busy else "m", size=12))
        opacity = "" if busy else ' fill-opacity="0.45"'
        b.append(f'<rect x="{x0}" y="{y}" width="{v * scale:.1f}" height="{bar_h}" rx="2" class="{cls}"{opacity}/>')
        b.append(_text(x0 + v * scale + 8, y + 12.5, f"{v:.1f} ms", cls="t" if busy else "m", size=12))
        y += row_h
    y += 16
    b.append(f'<line x1="24" x2="{W - 24}" y1="{y}" y2="{y}" class="grid"/>')
    y += 28
    b.append(
        _text(24, y, "Local LLM generation throughput, tok/s (higher is better; bars start at 0)", size=14, weight=600)
    )
    y += 16
    scale = bar_max / max(v for _, v, _, _ in d["tok_s"])
    for label, v, drop, cls in d["tok_s"]:
        b.append(_text(x_label, y + 12.5, label, size=12))
        b.append(f'<rect x="{x0}" y="{y}" width="{v * scale:.1f}" height="{bar_h}" rx="2" class="{cls}"/>')
        note = f"{v:.1f} tok/s" + ("" if drop is None else f" (−{drop * 100:.1f}%)")
        b.append(_text(x0 + v * scale + 8, y + 12.5, note, size=12))
        y += row_h
    y += 8
    b.append(f'<line x1="24" x2="{W - 24}" y1="{y}" y2="{y}" class="grid"/>')
    plat = d["platform"]
    b.append(
        _text(
            24,
            y + 24,
            f"One run ({d['run']}) on {plat['soc']} · macOS {plat['macos']} · LLM {d['llm_model']} · "
            f"serve --model laya",
            cls="m",
            size=12,
        )
    )
    b.append(
        _text(
            24,
            y + 44,
            f"Medians over {d['windows']} windows of 60 s per cell ({d['alone_windows']} for the LLM alone); "
            "P99 from scheduled arrival, queueing included",
            cls="m",
            size=12,
        )
    )
    H = int(y + 64)
    (gi, gb, ai, ab) = (v for _, _, v, _ in d["p99"])
    alone, gtok, atok = (v for _, v, _, _ in d["tok_s"])
    title = (
        f"laya-apple serve beside a busy local LLM, one run on {plat['soc']} with {d['llm_model']}: "
        f"short-decision P99 {ab:.1f} ms with serve auto against {gb:.1f} ms with --device gpu while the LLM "
        f"generates ({ai:.1f} and {gi:.1f} ms with the LLM idle); LLM throughput {alone:.1f} tok/s alone, "
        f"{gtok:.1f} beside serve --device gpu and {atok:.1f} beside serve auto"
    )
    return _svg(W, H, title, b)


FIGURES = {
    "hero-throughput.svg": hero,
    "architecture.svg": architecture,
    "serve-demo.svg": serve_demo,
    "serve-llm-load.svg": serve_llm,
}

SOCIAL_W, SOCIAL_H = 1280, 640
CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
)


def hard_mismatches() -> int:
    """Hard decision mismatches of laya-apple MLX FP16 and ANE FP16 against the upstream
    goldens, summed over the three models (benchmarks/v1.0/parity)."""
    total = 0
    for m in MODELS:
        for name in ("laya-apple-gpu-float16", "laya-apple-ane-float16"):
            p = _parity(m, name)
            if p is None or not p.get("passed"):
                sys.exit(f"{m} {name}: parity result missing or failed")
            total += _mismatches(p)[0]
    return total


def social_preview() -> str:
    """The 1280x640 link preview. A fixed light palette: it is a PNG shown on any background."""
    rows, (soc, macos) = throughput_data()
    best = max(rows, key=lambda r: r[3])
    hard = hard_mismatches()
    W, H = SOCIAL_W, SOCIAL_H
    ink, muted, acc, gpu = "#1f2328", "#59636e", "#0969da", "#8c959f"

    def t(x, y, s, fill, size, weight=None, anchor=None):
        return _text(x, y, s, size=size, weight=weight, anchor=anchor).replace('class="t"', f'fill="{fill}"')

    # One pair of bars for the model with the largest gain, on the right.
    bx, base, top = 930, 500, 250
    scale = (base - top) / best[2]
    b = [
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<rect x="0" y="0" width="{W}" height="10" fill="{acc}"/>',
        t(72, 138, "laya-apple", ink, 84, 700),
        t(72, 190, "Correctness-validated heterogeneous Laya runtime for Apple Silicon", muted, 27),
        f'<rect x="72" y="224" width="420" height="52" rx="26" fill="#ddf4ff" stroke="{acc}" stroke-width="1.5"/>',
        t(282, 259, "MLX GPU + Apple Neural Engine", acc, 25, 600, "middle"),
        t(72, 402, f"Up to {best[3]:.2f}×", acc, 104, 700),
        t(72, 450, "mixed-workload throughput vs GPU-only serving", ink, 29),
        '<circle cx="84" cy="499" r="9" fill="#1a7f37"/>',
        t(104, 508, f"{hard} hard decision mismatches against upstream Laya", ink, 27, 600),
        t(72, 590, f"Measured on {soc} · macOS {macos}", muted, 22),
    ]
    for k, (label, v, fill) in enumerate((("GPU-only", best[1], gpu), ("GPU + ANE", best[2], acc))):
        x, h = bx + k * 140, best[1 + k] * scale
        b.append(f'<rect x="{x}" y="{base - h:.1f}" width="96" height="{h:.1f}" rx="4" fill="{fill}"/>')
        b.append(t(x + 48, base - h - 14, f"{v:.1f}", ink, 24, 600, "middle"))
        b.append(t(x + 48, base + 34, label, muted, 22, None, "middle"))
    b.append(f'<line x1="{bx - 20}" x2="{bx + 256}" y1="{base}" y2="{base}" stroke="#d1d9e0" stroke-width="2"/>')
    b.append(t(bx + 118, base + 76, f"{best[0]}, req/s", muted, 22, None, "middle"))
    body = "\n".join(b)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'font-family="{FONT}">\n{body}\n</svg>\n'
    )


def render_social_png(out: Path) -> None:
    chrome = next((c for c in CHROME_CANDIDATES if Path(c).exists() or shutil.which(c)), None)
    if chrome is None:
        sys.exit("--social-png needs Google Chrome or Chromium")
    with tempfile.TemporaryDirectory() as d:
        svg = Path(d) / "social-preview.svg"
        svg.write_text(social_preview())
        page = Path(d) / "page.html"
        page.write_text(
            f'<html><body style="margin:0"><img src="{svg.as_uri()}" width="{SOCIAL_W}" height="{SOCIAL_H}"></body></html>'
        )
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                "--allow-file-access-from-files",
                f"--window-size={SOCIAL_W},{SOCIAL_H}",
                f"--screenshot={out}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    print(f"wrote {out.relative_to(ROOT)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if a committed figure differs from the data")
    ap.add_argument(
        "--social-png", action="store_true", help="also render docs/readme/social-preview.png (needs Chrome)"
    )
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
    if args.social_png and not args.check:
        render_social_png(OUT / "social-preview.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
