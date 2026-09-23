"""Disk preflight for large benchmark matrices: free space and Core ML E5 cache size.

    uv run python scripts/bench_preflight.py [--workspace benchmarks/v1.0]

Every Core ML model a process loads is compiled by macOS into an E5 bundle and kept under
~/Library/Caches/<process name>/com.apple.e5rt.e5bundlecache, one bundle per distinct
model and compute-unit configuration. Nothing evicts it, so a benchmark matrix that loads
many Core ML variants (scripts/bench_v1.sh) can leave tens of GiB behind. This check
reports free space and those caches, and exits 1 when either threshold is exceeded,
before a run starts rather than when the disk fills halfway through it.

It never deletes anything: a cache named after the Python executable is shared with every
other Core ML workload run by that Python. Cleanup is manual; see docs/benchmarks.md,
"Disk space and the Core ML E5 cache".

Thresholds (GiB) can be changed for a machine with a smaller disk, and 0 disables one:
LAYA_APPLE_PREFLIGHT_MIN_FREE_GIB (default 200), LAYA_APPLE_PREFLIGHT_MAX_E5_GIB (default 50).
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3
E5_CACHE_DIRNAME = "com.apple.e5rt.e5bundlecache"
MIN_FREE_GIB = 200.0
MAX_E5_GIB = 50.0
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Report:
    workspace: Path
    free_bytes: int
    caches: list[tuple[Path, int]]
    unreadable: int = 0

    @property
    def e5_bytes(self) -> int:
        return sum(size for _, size in self.caches)


def e5_cache_dirs(caches_root: Path) -> list[Path]:
    """Every E5 bundle cache under a Library/Caches directory, one per process name."""
    return sorted(p for p in caches_root.glob(f"*/{E5_CACHE_DIRNAME}") if p.is_dir() and not p.is_symlink())


def tree_bytes(path: Path) -> tuple[int, int]:
    """(total file size, entries that could not be read) under path, without following symlinks."""
    total = unreadable = 0

    def skipped(_):
        nonlocal unreadable
        unreadable += 1

    for dirpath, _, filenames in os.walk(path, onerror=skipped):
        for name in filenames:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                unreadable += 1
                continue
            total += st.st_size
    return total, unreadable


def measure(workspace: Path, caches_root: Path) -> Report:
    # The workspace may not exist yet (first run of a new output directory).
    probe = workspace
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    caches, unreadable = [], 0
    for d in e5_cache_dirs(caches_root):
        size, skipped = tree_bytes(d)
        caches.append((d, size))
        unreadable += skipped
    return Report(workspace, shutil.disk_usage(probe).free, caches, unreadable)


def thresholds_from_env(env=os.environ) -> tuple[float, float]:
    def read(name: str, default: float) -> float:
        raw = env.get(name)
        if raw is None or raw == "":
            return default
        try:
            value = float(raw)
        except ValueError:
            raise SystemExit(f"{name} must be a number of GiB, got {raw!r}") from None
        if not math.isfinite(value) or value < 0:
            raise SystemExit(f"{name} must be a finite number >= 0, got {raw!r}")
        return value

    return (
        read("LAYA_APPLE_PREFLIGHT_MIN_FREE_GIB", MIN_FREE_GIB),
        read("LAYA_APPLE_PREFLIGHT_MAX_E5_GIB", MAX_E5_GIB),
    )


def problems(report: Report, min_free_gib: float, max_e5_gib: float) -> list[str]:
    """Human-readable threshold violations; empty when the run may start. 0 disables a check."""
    out = []
    if min_free_gib and report.free_bytes < min_free_gib * GIB:
        out.append(
            f"free disk {report.free_bytes / GIB:.1f} GiB on the filesystem holding {report.workspace} "
            f"is below the {min_free_gib:g} GiB minimum"
        )
    if max_e5_gib and report.e5_bytes > max_e5_gib * GIB:
        out.append(f"Core ML E5 caches total {report.e5_bytes / GIB:.1f} GiB, above the {max_e5_gib:g} GiB maximum")
    return out


def format_report(report: Report) -> str:
    lines = [
        f"workspace: {report.workspace}",
        f"free disk: {report.free_bytes / GIB:.1f} GiB",
        f"Core ML E5 caches: {report.e5_bytes / GIB:.1f} GiB in {len(report.caches)} director"
        + ("y" if len(report.caches) == 1 else "ies"),
    ]
    for path, size in sorted(report.caches, key=lambda c: -c[1]):
        lines.append(f"  {size / GIB:8.1f} GiB  {path}")
    if report.unreadable:
        lines.append(f"  ({report.unreadable} entries could not be read; the total may be low)")
    return "\n".join(lines)


ADVICE = (
    "Not starting the benchmark. Nothing was deleted.\n"
    "Free space before retrying. The E5 caches above are safe to delete only by hand:\n"
    "  1. stop every process that uses Core ML under that cache's name (a cache named after a\n"
    "     Python executable is shared by other benchmarks, tests and unrelated projects;\n"
    "     leave caches of macOS services alone);\n"
    "  2. delete the cache directory yourself; the next model load recompiles it.\n"
    "See docs/benchmarks.md, 'Disk space and the Core ML E5 cache'. On a machine with a smaller\n"
    "disk, set LAYA_APPLE_PREFLIGHT_MIN_FREE_GIB / LAYA_APPLE_PREFLIGHT_MAX_E5_GIB."
)


def check(workspace: Path, caches_root: Path | None = None, env=os.environ) -> int:
    """Print the report; return 0 if the run may start, 1 if a threshold is exceeded."""
    caches_root = caches_root or Path.home() / "Library" / "Caches"
    min_free_gib, max_e5_gib = thresholds_from_env(env)
    report = measure(workspace, caches_root)
    print("benchmark preflight\n" + format_report(report), flush=True)
    found = problems(report, min_free_gib, max_e5_gib)
    if not found:
        print("preflight: ok", flush=True)
        return 0
    for p in found:
        print(f"preflight FAILED: {p}", file=sys.stderr)
    print(ADVICE, file=sys.stderr, flush=True)
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", type=Path, default=ROOT / "benchmarks", help="where the benchmark writes its output")
    a = p.parse_args(argv)
    return check(a.workspace.resolve())


if __name__ == "__main__":
    sys.exit(main())
