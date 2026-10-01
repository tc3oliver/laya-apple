"""Fail when a tracked file is over the size cap and is not grandfathered.

    uv run python scripts/check_file_sizes.py

Raw research and benchmark data is the one thing that grows without bound in this repository
(see docs/evidence.md). The check reads the blob size of every file in the git index, so it
sees what a commit would add and does not depend on the working tree, line endings or the
checkout. A file over MAX_BYTES passes only if scripts/file_size_allowlist.txt lists it, and
then only up to the size recorded there. The allowlist is read from the index too (the working
tree copy only if the index has none), so a local run and CI agree. It is the set of files that
were already over the cap when the policy started. Never add research or benchmark data to it.
Any other addition needs explicit maintainer agreement in the pull request. An entry whose file
is gone or is now under the cap is an error, so the list never grows for evidence data.

Exit status: 0 all within the cap, 1 a file or an allowlist line broke the rule, 2 the check
could not run (git failed, or the allowlist is malformed).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = Path("scripts/file_size_allowlist.txt")
MAX_BYTES = 1024 * 1024  # 1 MiB; a file of exactly this size is allowed
GITLINK_MODE = "160000"


def tracked_sizes(root: Path) -> dict[str, int]:
    """Blob size in bytes of every file in the git index, keyed by repo-relative path."""
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-s", "-z"], capture_output=True, check=True
    ).stdout.decode()
    entries = []  # (blob id, path)
    for record in listing.split("\0"):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        mode, blob, _stage = meta.split()
        if mode != GITLINK_MODE:  # a submodule has no blob to measure
            entries.append((blob, path))
    if not entries:
        return {}
    batch = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch-check=%(objectsize)"],
        input="".join(f"{blob}\n" for blob, _ in entries).encode(),
        capture_output=True,
        check=True,
    ).stdout.split()
    return {path: int(size) for (_, path), size in zip(entries, batch, strict=True)}


def parse_allowlist(text: str, source: str) -> dict[str, int]:
    """Parse `<max bytes> <path>` lines; blank lines and `#` comments are ignored."""
    entries: dict[str, int] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        size, _, rel = line.partition(" ")
        rel = rel.strip()
        if not (size.isascii() and size.isdigit()) or not rel:
            raise ValueError(f"{source}:{number}: expected '<max bytes> <path>', got {raw!r}")
        if rel in entries:
            raise ValueError(f"{source}:{number}: {rel} is listed twice")
        entries[rel] = int(size)
    return entries


def read_allowlist(root: Path) -> dict[str, int]:
    """The allowlist as staged in the index, else the working tree file, else empty."""
    staged = subprocess.run(["git", "-C", str(root), "show", f":{ALLOWLIST.as_posix()}"], capture_output=True)
    if staged.returncode == 0:
        return parse_allowlist(staged.stdout.decode(), f":{ALLOWLIST}")
    path = root / ALLOWLIST
    return parse_allowlist(path.read_text(), str(path)) if path.exists() else {}


def find_problems(sizes: dict[str, int], allowlist: dict[str, int], limit: int = MAX_BYTES) -> list[str]:
    """One message per violation: an over-cap file, a grown grandfathered file, a stale entry."""
    problems = []
    for path, size in sorted(sizes.items()):
        if size <= limit:
            continue
        ceiling = allowlist.get(path)
        if ceiling is None:
            problems.append(f"{path}: {size} bytes is over the {limit}-byte cap and is not grandfathered")
        elif size > ceiling:
            problems.append(f"{path}: grew from {ceiling} to {size} bytes; a grandfathered file may not grow")
    for path in sorted(allowlist):
        size = sizes.get(path)
        if size is None:
            problems.append(f"{path}: allowlisted but no longer tracked; remove its allowlist line")
        elif size <= limit:
            problems.append(f"{path}: allowlisted but now {size} bytes, under the cap; remove its allowlist line")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (default: this checkout)")
    args = parser.parse_args(argv)

    try:
        sizes = tracked_sizes(args.root)
        problems = find_problems(sizes, read_allowlist(args.root))
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip().splitlines()
        print(f"error: git exited {exc.returncode}: {detail[-1] if detail else 'no output'}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(
            f"\n{len(problems)} problem(s). Raw research and benchmark data over {MAX_BYTES / 2**20:g} MiB "
            "belongs in an external evidence release referenced from an in-repo manifest, "
            "not in Git: see docs/evidence.md.",
            file=sys.stderr,
        )
        return 1
    print(f"ok: {len(sizes)} tracked files, none over {MAX_BYTES} bytes except the grandfathered ones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
