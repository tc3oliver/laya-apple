"""v0.3 release gate: run every release check, record status/duration/output per step, and
write a JSON + markdown report.

    LAYA_APPLE_CACHE=/path/to/cache HF_HUB_OFFLINE=1 \\
        uv run python scripts/release_gate.py [--quick] [--out benchmarks/release-gate-X.json]

Every step runs even after an earlier one fails; the process exits non-zero iff any
required step failed. --quick skips the clean-install matrix and slow pytest markers, for
fast local iteration.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAIL_CHARS = 4000
PY_VERSIONS = ["3.11", "3.12", "3.13"]
EXTRAS = ["base", "ane"]
EXCLUDED_ARTIFACT_DIRS = {"quarantine", "rejected", ".staging", ".locks"}


def _tail(s: str, n: int = TAIL_CHARS) -> str:
    return s[-n:] if len(s) > n else s


def _sanitize(text: str) -> str:
    """Replace this machine's repo root and home directory with portable placeholders
    so a committed report never carries the local username or path layout."""
    text = text.replace(str(ROOT), ".")
    text = text.replace(str(Path.home()), "~")
    return text


def run_step(name: str, cmd, *, cwd=None, env=None, required=True) -> dict:
    """Run `cmd` (a list of args or a callable returning a step dict) and record it."""
    t0 = time.monotonic()
    if callable(cmd):
        try:
            ok, output = cmd()
            status = "pass" if ok else "fail"
        except Exception as e:  # noqa: BLE001 - a step failure is data, not a crash
            status = "error"
            output = f"{type(e).__name__}: {e}"
    else:
        try:
            r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=3600)
            status = "pass" if r.returncode == 0 else "fail"
            output = (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired as e:
            status = "error"
            output = f"timed out after {e.timeout}s"
        except Exception as e:  # noqa: BLE001
            status = "error"
            output = f"{type(e).__name__}: {e}"
    duration = time.monotonic() - t0
    step = {
        "name": name,
        "status": status,
        "required": required,
        "duration_s": round(duration, 2),
        "output_tail": _tail(output),
    }
    print(f"[{status.upper():5}] {name} ({step['duration_s']}s)", flush=True)
    return step


def env_with_cache():
    import os

    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def _du_bytes(path: Path) -> int:
    r = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        return -1
    return int(r.stdout.split()[0]) * 1024


def _top_distributions(site_packages: Path, n: int = 10):
    sizes = []
    for dist_info in site_packages.glob("*.dist-info"):
        pkg_name = dist_info.name.split("-")[0]
        size = 0
        record = dist_info / "RECORD"
        if record.exists():
            for line in record.read_text(errors="ignore").splitlines():
                rel = line.split(",")[0]
                p = site_packages / rel
                if p.is_file():
                    try:
                        size += p.stat().st_size
                    except OSError:
                        pass
        else:
            size = _du_bytes(dist_info.parent / pkg_name) if (dist_info.parent / pkg_name).exists() else 0
        sizes.append({"name": pkg_name, "bytes": size})
    sizes.sort(key=lambda x: -x["bytes"])
    return sizes[:n]


def _quickstart_snippet() -> str:
    text = (ROOT / "README.md").read_text()
    marker = "## Quickstart"
    idx = text.index(marker)
    fence_start = text.index("```python", idx) + len("```python")
    fence_end = text.index("```", fence_start)
    return text[fence_start:fence_end].strip("\n")


def install_matrix_step():
    """Clean-install laya-apple[extra] for each Python version x extra, run the README
    quickstart offline plus `laya-apple --offline info`, and record installed sizes."""
    results = []
    quickstart = _quickstart_snippet()
    for py in PY_VERSIONS:
        for extra in EXTRAS:
            tmp = Path(tempfile.mkdtemp(prefix=f"laya-apple-gate-{py}-{extra}-"))
            entry = {"python": py, "extra": extra, "tmpdir": str(tmp)}
            try:
                venv_dir = tmp / "venv"
                r = subprocess.run(
                    ["uv", "venv", "--python", py, str(venv_dir)],
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                )
                if r.returncode != 0:
                    entry.update(ok=False, stage="uv venv", output=_tail(r.stdout + r.stderr))
                    results.append(entry)
                    continue
                venv_python = venv_dir / "bin" / "python"
                pkg_spec = str(ROOT) if extra == "base" else f"{ROOT}[{extra}]"
                r = subprocess.run(
                    ["uv", "pip", "install", "--python", str(venv_python), pkg_spec],
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                )
                if r.returncode != 0:
                    entry.update(ok=False, stage="uv pip install", output=_tail(r.stdout + r.stderr))
                    results.append(entry)
                    continue
                env = env_with_cache()
                r = subprocess.run(
                    [str(venv_python), "-c", quickstart],
                    capture_output=True,
                    text=True,
                    cwd=str(tmp),
                    env=env,
                )
                if r.returncode != 0:
                    entry.update(ok=False, stage="quickstart", output=_tail(r.stdout + r.stderr))
                    results.append(entry)
                    continue
                laya_apple_cli = venv_dir / "bin" / "laya-apple"
                r = subprocess.run(
                    [str(laya_apple_cli), "--offline", "info"],
                    capture_output=True,
                    text=True,
                    cwd=str(tmp),
                    env=env,
                )
                if r.returncode != 0:
                    entry.update(ok=False, stage="cli info", output=_tail(r.stdout + r.stderr))
                    results.append(entry)
                    continue
                r = subprocess.run(
                    [str(laya_apple_cli), "switchyard", "--help"],
                    capture_output=True,
                    text=True,
                    cwd=str(tmp),
                    env=env,
                )
                if r.returncode != 0:
                    entry.update(ok=False, stage="switchyard --help", output=_tail(r.stdout + r.stderr))
                    results.append(entry)
                    continue
                site_packages = next(venv_dir.glob("lib/python*/site-packages"), None)
                entry.update(
                    ok=True,
                    installed_size_bytes=_du_bytes(site_packages) if site_packages else -1,
                    top_distributions=_top_distributions(site_packages) if site_packages else [],
                )
                results.append(entry)
            finally:
                if tmp.exists() and str(tmp).startswith(tempfile.gettempdir()):
                    shutil.rmtree(tmp, ignore_errors=True)
                entry.pop("tmpdir", None)
    ok = all(e.get("ok") for e in results)
    return ok, json.dumps(results, indent=1)


def git_revision():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    rev = r.stdout.strip() if r.returncode == 0 else "unknown"
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return rev, bool(dirty.stdout.strip())


def soak_step(seconds: int):
    """Run the opt-in stress/soak suite for `seconds` (LAYA_APPLE_STRESS=1)."""
    env = env_with_cache()
    env["LAYA_APPLE_STRESS"] = "1"
    env["LAYA_APPLE_STRESS_SECONDS"] = str(seconds)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/stress", "-q"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=max(3600, seconds + 600),
    )
    return r.returncode == 0, (r.stdout or "") + (r.stderr or "")


def find_manifest_paths(artifacts_root: Path) -> list[Path]:
    """Every `*/*/*/manifest.json` under `artifacts_root`, excluding non-artifact subtrees."""
    if not artifacts_root.exists():
        return []
    out = []
    for path in artifacts_root.glob("*/*/*/manifest.json"):
        if any(part in EXCLUDED_ARTIFACT_DIRS for part in path.relative_to(artifacts_root).parts):
            continue
        out.append(path)
    return sorted(out)


def check_manifests(artifacts_root: Path):
    """Validate every artifact manifest under `artifacts_root` against the shipped JSON
    Schema. Returns (ok, message)."""
    from laya_apple.schema import manifest_errors

    paths = find_manifest_paths(artifacts_root)
    problems = []
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001 - an unreadable manifest is a finding, not a crash
            problems.append(f"{path}: {type(e).__name__}: {e}")
            continue
        errs = manifest_errors(data)
        if errs:
            problems.append(f"{path}: {'; '.join(errs)}")
    if problems:
        return False, f"{len(problems)} invalid manifest(s):\n" + "\n".join(problems)
    return True, f"{len(paths)} manifest(s) validated"


def manifest_schema_step():
    """Validate every cached artifact manifest against the shipped JSON Schema."""
    from laya_apple.hub import cache_root

    return check_manifests(cache_root() / "artifacts")


_DOC_REF_RE = re.compile(r"`([\w./]+\.py)::([\w*]+)`")


def parse_doc_test_refs(markdown: str) -> list[tuple[str, str]]:
    """Every `file.py::name` reference in `markdown`, as (file, name) pairs."""
    return _DOC_REF_RE.findall(markdown)


def resolve_doc_test_refs(refs: list[tuple[str, str]], tests_root: Path) -> list[str]:
    """Every ref in `refs` that does not resolve to a `def test_...` (or file, for
    non-test-function refs) under `tests_root`. A trailing `*` in the name is a prefix
    match against `def test_<prefix>...` in that file."""
    unresolved = []
    cache: dict[Path, str] = {}
    for file, name in refs:
        candidates = list(tests_root.glob(f"**/{file}"))
        if not candidates:
            unresolved.append(f"{file}::{name}")
            continue
        found = False
        for candidate in candidates:
            if candidate not in cache:
                try:
                    cache[candidate] = candidate.read_text()
                except OSError:
                    cache[candidate] = ""
            text = cache[candidate]
            if name.endswith("*"):
                prefix = name[:-1]
                if re.search(rf"^def {re.escape(prefix)}\w*", text, re.MULTILINE):
                    found = True
                    break
            else:
                if re.search(rf"^def {re.escape(name)}\b", text, re.MULTILINE):
                    found = True
                    break
        if not found:
            unresolved.append(f"{file}::{name}")
    return unresolved


def no_silent_fallback_docs_step():
    """docs/no-silent-fallback.md's `file.py::name` references all resolve under tests/."""
    doc_path = ROOT / "docs" / "no-silent-fallback.md"
    if not doc_path.exists():
        return False, f"{doc_path} does not exist"
    refs = parse_doc_test_refs(doc_path.read_text())
    if not refs:
        return False, f"no `file.py::name` references found in {doc_path}"
    unresolved = resolve_doc_test_refs(refs, ROOT / "tests")
    if unresolved:
        return False, f"{len(unresolved)} unresolved reference(s):\n" + "\n".join(unresolved)
    return True, f"{len(refs)} reference(s) all resolved"


def docs_versions_step():
    """pyproject.toml's version matches `laya_apple.__version__`, and CHANGELOG.md has a
    matching `## [<version>]` heading."""
    from laya_apple import __version__

    pyproject_text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject_text)
    if not m:
        return False, "no version= line found in pyproject.toml"
    pyproject_version = m.group(1)
    if pyproject_version != __version__:
        return False, f"pyproject.toml version {pyproject_version!r} != laya_apple.__version__ {__version__!r}"
    changelog = (ROOT / "CHANGELOG.md").read_text()
    heading = f"## [{__version__}]"
    if heading not in changelog:
        return False, f"CHANGELOG.md has no {heading!r} heading"
    return True, f"pyproject.toml and CHANGELOG.md agree on version {__version__!r}"


def build_report(quick: bool, soak: int = 0) -> dict:
    from laya_apple import __version__
    from laya_apple.artifacts import platform_profile

    env = env_with_cache()
    steps = []

    steps.append(run_step("ruff check", [sys.executable, "-m", "ruff", "check", "."], cwd=ROOT, env=env))
    steps.append(
        run_step(
            "ruff format --check",
            [sys.executable, "-m", "ruff", "format", "--check", "laya_apple", "scripts", "tests"],
            cwd=ROOT,
            env=env,
        )
    )
    steps.append(
        run_step("derive_routing --check", [sys.executable, "scripts/derive_routing.py", "--check"], cwd=ROOT, env=env)
    )
    steps.append(
        run_step("make_goldens --check", [sys.executable, "scripts/make_goldens.py", "--check"], cwd=ROOT, env=env)
    )
    steps.append(
        run_step(
            "derive_placement --check", [sys.executable, "scripts/derive_placement.py", "--check"], cwd=ROOT, env=env
        )
    )

    pytest_cmd = [sys.executable, "-m", "pytest", "-q"]
    if quick:
        pytest_cmd += ["-m", "not stress and not parity"]
    steps.append(run_step("pytest", pytest_cmd, cwd=ROOT, env=env))

    if not quick:
        steps.append(run_step("clean install matrix", install_matrix_step, required=True))
    else:
        steps.append(
            {
                "name": "clean install matrix",
                "status": "skipped",
                "required": False,
                "duration_s": 0.0,
                "output_tail": "skipped by --quick",
            }
        )

    steps.append(
        run_step(
            "artifacts verify",
            [sys.executable, "-m", "laya_apple.cli", "--offline", "artifacts", "verify"],
            cwd=ROOT,
            env=env,
        )
    )

    steps.append(run_step("manifest schema", manifest_schema_step, required=True))
    steps.append(run_step("no silent fallback tests", no_silent_fallback_docs_step, required=True))
    steps.append(run_step("docs versions", docs_versions_step, required=True))

    if soak > 0:
        steps.append(run_step("soak", lambda: soak_step(soak), required=True))
    else:
        steps.append(
            {
                "name": "soak",
                "status": "skipped",
                "required": False,
                "duration_s": 0.0,
                "output_tail": "skipped: --soak not given (0 seconds)",
            }
        )

    rev, dirty = git_revision()
    required_failed = [s for s in steps if s.get("required") and s["status"] not in ("pass", "skipped")]
    return {
        "version": __version__,
        "quick": quick,
        "soak_seconds": soak,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_revision": rev,
        "git_dirty": dirty,
        "python": sys.version,
        "platform_profile": platform_profile(),
        "steps": steps,
        "passed": not required_failed,
    }


def to_markdown(report: dict) -> str:
    lines = [
        f"# Release gate — laya-apple {report['version']}",
        "",
        f"- generated: {report['generated_at']}",
        f"- git revision: `{report['git_revision']}`{' (dirty)' if report['git_dirty'] else ''}",
        f"- python: {report['python'].splitlines()[0]}",
        f"- platform: {json.dumps(report['platform_profile'])}",
        f"- quick mode: {report['quick']}",
        f"- soak seconds: {report.get('soak_seconds', 0)}",
        f"- **result: {'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "| step | status | duration (s) | required |",
        "|---|---|---|---|",
    ]
    for s in report["steps"]:
        lines.append(f"| {s['name']} | {s['status']} | {s['duration_s']} | {s['required']} |")
    lines.append("")
    for s in report["steps"]:
        if s["status"] not in ("pass", "skipped"):
            lines += [f"## {s['name']} — {s['status']}", "", "```", s["output_tail"], "```", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="skip the install matrix and slow pytest markers")
    p.add_argument(
        "--soak",
        type=int,
        default=0,
        metavar="SECONDS",
        help="run tests/stress for SECONDS (LAYA_APPLE_STRESS=1); 0 (default) skips it",
    )
    p.add_argument("--out", help="output path stem (writes <stem>.json and <stem>.md)")
    a = p.parse_args(argv)

    report = build_report(a.quick, a.soak)

    if a.out:
        stem = Path(a.out)
        if stem.suffix in (".json", ".md"):  # a version like release-gate-0.3.0 keeps its dots
            stem = stem.with_suffix("")
    else:
        stem = ROOT / "benchmarks" / f"release-gate-{report['version']}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = stem.with_name(stem.name + ".json"), stem.with_name(stem.name + ".md")
    json_path.write_text(_sanitize(json.dumps(report, indent=1)))
    md_path.write_text(_sanitize(to_markdown(report)))
    print(f"wrote {json_path} and {md_path}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
