"""Render the switchyard-v1 benchmark report (Markdown) for one campaign directory
(`benchmarks/switchyard/run.sh <campaign-dir>`, e.g. `benchmarks/switchyard/v1-m4-max/`,
holding `raw/run-001/`, `raw/run-002/`, `raw/run-003/`).

    uv run python scripts/switchyard_report.py benchmarks/switchyard/v1-m4-max
    uv run python scripts/switchyard_report.py --check benchmarks/switchyard/v1-m4-max

Each positional argument is a run directory containing `result.json`, a `result.json` file
itself, or a campaign directory (its `raw/run-*/result.json` files are used). `--check
CAMPAIGN_DIR` discovers `CAMPAIGN_DIR/raw/run-00N/result.json` itself and compares the
regenerated report against the `<!-- switchyard-report:begin <label> -->` /
`<!-- switchyard-report:end <label> -->` block for that campaign (`label` = the campaign
directory's name) already committed in `benchmarks/switchyard/README.md` (`CAMPAIGN_DIR`'s
parent) — one such labelled section per campaign directory, the way the repo's other `--check`
scripts compare regenerated output with a committed file (see
`benchmarks/ane-process-isolation/analyze.py`). `--update-readme CAMPAIGN_DIR` writes that
campaign's section the same way.

Every `result.json` must validate against `laya_apple/data/switchyard-result.schema.json`
(`laya_apple.schema`, docs/switchyard.md); a violation is fatal (non-zero exit), not a warning.

A run only joins the cross-run spread table when it is `standard` and its `laya_apple` version,
`model.revision`, `workload.schedule_sha256` and `ane.state` match the first standard run seen
(the reference fingerprint) — otherwise it is reported on its own per-run table but listed
separately, with the reason, under "excluded from cross-run spread".

Metric order is frozen (docs/switchyard.md): late, P99 decision latency, P99 queue wait,
delivered, P50/P95. There is no composite score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_NAME = "switchyard-result.schema.json"
FINGERPRINT_FIELDS = ("laya_apple", "model.revision", "workload.schedule_sha256", "ane.state")


def markers(label: str) -> tuple[str, str]:
    return f"<!-- switchyard-report:begin {label} -->", f"<!-- switchyard-report:end {label} -->"


def _get(data: dict, dotted: str):
    node = data
    for key in dotted.split("."):
        node = node.get(key, {}) if isinstance(node, dict) else {}
    return node if node != {} else None


def fingerprint(data: dict) -> dict:
    return {f: _get(data, f) for f in FINGERPRINT_FIELDS}


def load_run(path: Path) -> dict:
    """Load and validate `path`. Raises SystemExit on any schema violation."""
    data = json.loads(path.read_text())
    from laya_apple import schema as schema_mod

    errs = schema_mod.errors(schema_mod.load_schema(SCHEMA_NAME), data)
    if errs:
        raise SystemExit(f"{path}: {len(errs)} schema violation(s): " + "; ".join(errs[:10]))
    return data


def discover_runs(path: Path) -> list[Path]:
    """result.json files under `path`: `path` itself if it is one, `path/result.json` if that
    exists, else every `path/raw/run-*/result.json` and `path/run-*/result.json` in order."""
    if path.is_file():
        return [path]
    direct = path / "result.json"
    if direct.exists():
        return [direct]
    for base in (path / "raw", path):
        found = sorted(base.glob("run-*/result.json"))
        if found:
            return found
    return []


def _f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def _heading(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _config_row(label: str, cfg: dict) -> dict:
    game = cfg["game"]
    summary = cfg["summary"]
    late = summary["late"]
    dl = summary["decision_latency"]
    qw = summary["queue_wait"]
    miss = summary["miss_rate_at_ms"]
    return {
        "label": label,
        "rounds": cfg.get("rounds", []),
        "trains": game["trains"],
        "late_count": late["count"],
        "late_rate": late["rate"],
        "p99_decision_ms": dl["p99_ms"],
        "p99_queue_ms": qw["p99_ms"],
        "delivered": game["delivered"],
        "p50_ms": dl["p50_ms"],
        "p95_ms": dl["p95_ms"],
        "misrouted": game["misrouted"],
        "miss_rate_at_ms": miss,
    }


def per_run_section(heading: str, data: dict) -> list[str]:
    lines = [f"### {heading}\n"]
    ane = data.get("ane", {})
    design = data.get("design", {})
    comparison = data.get("comparison", {})
    lines.append(
        f"- standard: `{data.get('standard')}` · ANE state: `{ane.get('state', '?')}`"
        + (f" ({ane.get('reason')})" if ane.get("reason") else "")
        + f" · round order: `{design.get('ordering', '?')}` "
        + str(design.get("sequence", []))
        + f" · counterbalance: `{design.get('counterbalance', '?')}` "
        + f"· decision disagreements: {comparison.get('decision_disagreements', '–')}"
    )
    rows = [_config_row(label, cfg) for label, cfg in data.get("configs", {}).items()]
    lines.append(
        "\n| config | late (count/rate) | P99 decision ms | P99 queue ms | delivered | P50 / P95 ms | "
        "misrouted | miss@25/50/100/250/500 ms |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        miss = "/".join(_f(r["miss_rate_at_ms"].get(k), 3) for k in ("25", "50", "100", "250", "500"))
        lines.append(
            f"| {r['label']} | {r['late_count']} / {_f(r['late_rate'], 3)} | {_f(r['p99_decision_ms'])} "
            f"| {_f(r['p99_queue_ms'])} | {r['delivered']} | {_f(r['p50_ms'])} / {_f(r['p95_ms'])} "
            f"| {r['misrouted']} | {miss} |"
        )
    lines.append("")
    return lines


def spread_section(all_rows: dict[str, list[dict]]) -> list[str]:
    """Cross-run spread (min-max) per config, over the runs that have that config."""
    lines = ["### Cross-run spread (min–max)\n"]
    if not all_rows:
        lines.append("No runs eligible for the cross-run spread.\n")
        return lines
    lines.append("| config | runs | late count | P99 decision ms | P99 queue ms | delivered | P50 ms | P95 ms |")
    lines.append("|---|---:|---|---|---|---|---|---|")
    for label, rows in sorted(all_rows.items()):
        if not rows:
            continue

        def span(key, d=2):
            vals = [r[key] for r in rows if r[key] is not None]
            return (
                "–"
                if not vals
                else (f"{min(vals):.{d}f}" if min(vals) == max(vals) else f"{min(vals):.{d}f}–{max(vals):.{d}f}")
            )

        lines.append(
            f"| {label} | {len(rows)} | {span('late_count', 0)} | {span('p99_decision_ms')} | {span('p99_queue_ms')} "
            f"| {span('delivered', 0)} | {span('p50_ms')} | {span('p95_ms')} |"
        )
    lines.append("")
    return lines


def excluded_section(excluded: list[tuple[str, str]]) -> list[str]:
    lines = ["### Excluded from cross-run spread\n"]
    lines.append("| run | reason |")
    lines.append("|---|---|")
    for heading, reason in excluded:
        lines.append(f"| {heading} | {reason} |")
    lines.append("")
    return lines


def render(paths: list[Path], root: Path | None = None) -> str:
    lines = ["## Switchyard-v1 report\n"]
    all_rows: dict[str, list[dict]] = {}
    excluded: list[tuple[str, str]] = []
    reference = None
    reference_heading = None
    for path in paths:
        data = load_run(path)
        heading = _heading(path, root)
        lines += per_run_section(heading, data)

        fp = fingerprint(data)
        reason = None
        if not data.get("standard"):
            reason = "standard: false"
        elif reference is None:
            reference, reference_heading = fp, heading
        elif fp != reference:
            diffs = ", ".join(
                f"{k}: {fp[k]!r} != {reference[k]!r}" for k in FINGERPRINT_FIELDS if fp[k] != reference[k]
            )
            reason = f"differs from {reference_heading} ({diffs})"
        if reason is None:
            for label, cfg in data.get("configs", {}).items():
                all_rows.setdefault(label, []).append(_config_row(label, cfg))
        else:
            excluded.append((heading, reason))

    if len(paths) > 1:
        lines += spread_section(all_rows)
    if excluded:
        lines += excluded_section(excluded)
    return "\n".join(lines).rstrip() + "\n"


def update_readme(campaign_dir: Path, block: str) -> None:
    """Write `block` into `campaign_dir.parent/README.md`'s labelled section for
    `campaign_dir.name` (the markers for that label must already exist — a new campaign adds
    its own `### <label>` subsection with markers under "## Results" by hand first)."""
    label = campaign_dir.name
    begin, end = markers(label)
    readme_path = campaign_dir.parent / "README.md"
    text = readme_path.read_text()
    if begin not in text or end not in text:
        raise SystemExit(f"{readme_path} is missing the {begin} / {end} markers for campaign {label!r}")
    pre, rest = text.split(begin, 1)
    _, post = rest.split(end, 1)
    readme_path.write_text(pre + begin + "\n\n" + block + "\n" + end + post)


def check(campaign_dir: Path) -> None:
    paths = discover_runs(campaign_dir)
    if not paths:
        raise SystemExit(f"no result.json found under {campaign_dir}")
    report = render(paths, root=campaign_dir)
    label = campaign_dir.name
    begin, end = markers(label)
    readme_path = campaign_dir.parent / "README.md"
    text = readme_path.read_text()
    if begin not in text or end not in text:
        raise SystemExit(f"{readme_path} is missing the {begin} / {end} markers for campaign {label!r}")
    committed = text.split(begin, 1)[1].split(end, 1)[0].strip("\n")
    if committed != report.strip("\n"):
        raise SystemExit(
            f"stale: {readme_path}'s {label!r} section does not match the regenerated report from {campaign_dir}"
        )
    print(f"switchyard report for {label!r} is up to date")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, help="run directories, result.json files, or a campaign directory")
    ap.add_argument("--check", metavar="CAMPAIGN_DIR", type=Path, help="verify README.md's section for this campaign")
    ap.add_argument("--out", type=Path, help="write the report to this file instead of stdout")
    ap.add_argument(
        "--update-readme", metavar="CAMPAIGN_DIR", type=Path, help="write this campaign's section into README.md"
    )
    ap.add_argument("--root", type=Path, help="show run headings relative to this directory (default: cwd)")
    args = ap.parse_args(argv)

    if args.check:
        check(args.check)
        return

    if args.update_readme:
        paths: list[Path] = []
        for p in args.paths or [args.update_readme]:
            found = discover_runs(p)
            if not found:
                raise SystemExit(f"no result.json found for {p}")
            paths += found
        report = render(paths, root=args.update_readme)
        update_readme(args.update_readme, report.strip("\n"))
        if args.out:
            args.out.write_text(report)
        else:
            print(report)
        return

    paths = []
    for p in args.paths:
        found = discover_runs(p)
        if not found:
            raise SystemExit(f"no result.json found for {p}")
        paths += found
    if not paths:
        ap.error("give at least one run directory or result.json, or use --check/--update-readme CAMPAIGN_DIR")

    report = render(paths, root=args.root or Path.cwd())
    if args.out:
        args.out.write_text(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
