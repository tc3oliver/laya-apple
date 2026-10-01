"""Unit tests for scripts/check_file_sizes.py (the cap on tracked file size)."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cfs():
    spec = importlib.util.spec_from_file_location("check_file_sizes", ROOT / "scripts" / "check_file_sizes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    return tmp_path


def _track(repo: Path, rel: str, size: int) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    _git(repo, "add", "--", rel)


def _stage_allowlist(cfs, repo: Path, text: str) -> None:
    path = repo / cfs.ALLOWLIST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    _git(repo, "add", "--", cfs.ALLOWLIST.as_posix())


# ------------------------------------------------------------------------------- the cap


def test_cap_is_one_mebibyte(cfs):
    assert cfs.MAX_BYTES == 1024 * 1024


def test_files_at_or_under_the_cap_pass(cfs):
    assert cfs.find_problems({"a": 0, "b": 99, "c": 100}, {}, limit=100) == []


def test_file_over_the_cap_fails(cfs):
    problems = cfs.find_problems({"research/x/raw/trace.jsonl": 101}, {}, limit=100)
    assert len(problems) == 1
    assert "research/x/raw/trace.jsonl" in problems[0]
    assert "not grandfathered" in problems[0]


# ------------------------------------------------------------------------------ allowlist


def test_allowlisted_file_passes_up_to_its_recorded_size(cfs):
    allowlist = {"old.json": 150}
    assert cfs.find_problems({"old.json": 150}, allowlist, limit=100) == []
    assert cfs.find_problems({"old.json": 120}, allowlist, limit=100) == []


def test_allowlisted_file_may_not_grow(cfs):
    problems = cfs.find_problems({"old.json": 151}, {"old.json": 150}, limit=100)
    assert len(problems) == 1
    assert "grew from 150 to 151" in problems[0]


def test_allowlist_covers_only_the_listed_path(cfs):
    problems = cfs.find_problems({"old.json": 150, "new.json": 150}, {"old.json": 150}, limit=100)
    assert len(problems) == 1
    assert problems[0].startswith("new.json")


def test_renamed_grandfathered_file_fails_until_its_line_is_updated(cfs):
    problems = cfs.find_problems({"research/moved.json": 150}, {"research/old.json": 150}, limit=100)
    assert len(problems) == 2
    assert any(p.startswith("research/moved.json") and "not grandfathered" in p for p in problems)
    assert any(p.startswith("research/old.json") and "no longer tracked" in p for p in problems)


def test_stale_entry_for_a_removed_file_fails(cfs):
    problems = cfs.find_problems({"other": 1}, {"gone.json": 150}, limit=100)
    assert len(problems) == 1
    assert "gone.json" in problems[0]
    assert "no longer tracked" in problems[0]


def test_stale_entry_for_a_file_now_under_the_cap_fails(cfs):
    problems = cfs.find_problems({"shrunk.json": 100}, {"shrunk.json": 150}, limit=100)
    assert len(problems) == 1
    assert "under the cap" in problems[0]


def test_parse_allowlist_reads_entries_comments_and_spaces(cfs):
    text = "# header\n\n150 research/a/raw.json\n  200   research/b/with space [1].json  \n"
    assert cfs.parse_allowlist(text, "allow.txt") == {"research/a/raw.json": 150, "research/b/with space [1].json": 200}


@pytest.mark.parametrize(
    "line",
    [
        "research/a.json",
        "abc research/a.json",
        "150",
        "150 ",
        "-5 research/a.json",
        "² research/a.json",  # superscript two: str.isdigit() is true, int() raises
        "١٥٠ research/a.json",  # Arabic-Indic digits: int() would accept them
    ],
)
def test_parse_allowlist_rejects_malformed_lines(cfs, line):
    with pytest.raises(ValueError, match="allow.txt:1"):
        cfs.parse_allowlist(line + "\n", "allow.txt")


def test_parse_allowlist_rejects_duplicates(cfs):
    with pytest.raises(ValueError, match="allow.txt:2.*listed twice"):
        cfs.parse_allowlist("150 a.json\n160 a.json\n", "allow.txt")


def test_committed_allowlist_parses_and_is_all_over_the_cap(cfs):
    entries = cfs.parse_allowlist((ROOT / cfs.ALLOWLIST).read_text(), str(cfs.ALLOWLIST))
    assert entries
    assert all(size > cfs.MAX_BYTES for size in entries.values())


def test_read_allowlist_without_a_file_is_empty(cfs, repo):
    assert cfs.read_allowlist(repo) == {}


def test_read_allowlist_falls_back_to_the_working_tree(cfs, repo):
    (repo / cfs.ALLOWLIST).parent.mkdir(parents=True)
    (repo / cfs.ALLOWLIST).write_text("150 a.json\n")  # never staged
    assert cfs.read_allowlist(repo) == {"a.json": 150}


def test_read_allowlist_prefers_the_staged_copy(cfs, repo):
    _stage_allowlist(cfs, repo, "150 staged.json\n")
    (repo / cfs.ALLOWLIST).write_text("160 edited-after-staging.json\n")
    assert cfs.read_allowlist(repo) == {"staged.json": 150}


# ------------------------------------------------------------------------- git index sizes


def test_tracked_sizes_reads_index_blobs(cfs, repo):
    _track(repo, "small.txt", 10)
    _track(repo, "research/raw/with space.json", 2000)
    assert cfs.tracked_sizes(repo) == {"small.txt": 10, "research/raw/with space.json": 2000}


def test_tracked_sizes_of_an_empty_index_is_empty(cfs, repo):
    assert cfs.tracked_sizes(repo) == {}


def test_tracked_sizes_ignores_untracked_files_and_unstaged_edits(cfs, repo):
    _track(repo, "tracked.bin", 10)
    (repo / "untracked.bin").write_bytes(b"x" * 5000)
    (repo / "tracked.bin").write_bytes(b"x" * 5000)  # edited after `git add`, not staged
    assert cfs.tracked_sizes(repo) == {"tracked.bin": 10}


# ------------------------------------------------------------------------------------ CLI


def test_main_fails_on_an_unlisted_large_file(cfs, repo, capsys):
    _track(repo, "research/new/raw/trace.jsonl", cfs.MAX_BYTES + 1)
    assert cfs.main(["--root", str(repo)]) == 1
    err = capsys.readouterr().err
    assert "research/new/raw/trace.jsonl" in err
    assert "docs/evidence.md" in err


def test_main_passes_a_file_of_exactly_the_cap(cfs, repo, capsys):
    _track(repo, "research/new/raw/trace.jsonl", cfs.MAX_BYTES)
    assert cfs.main(["--root", str(repo)]) == 0
    assert "ok:" in capsys.readouterr().out


def test_main_passes_a_grandfathered_file(cfs, repo):
    size = cfs.MAX_BYTES + 1
    _track(repo, "research/old/raw/trace.jsonl", size)
    _stage_allowlist(cfs, repo, f"# grandfathered\n{size} research/old/raw/trace.jsonl\n")
    assert cfs.main(["--root", str(repo)]) == 0


def test_main_takes_the_allowlist_from_the_index_not_an_unstaged_edit(cfs, repo):
    size = cfs.MAX_BYTES + 1
    _track(repo, "research/old/raw/trace.jsonl", size)
    _stage_allowlist(cfs, repo, f"{size} research/old/raw/trace.jsonl\n")
    (repo / cfs.ALLOWLIST).write_text("")  # an unstaged edit that drops the entry
    assert cfs.main(["--root", str(repo)]) == 0


def test_main_does_not_honour_an_unstaged_allowlist_entry_once_the_index_has_one(cfs, repo):
    size = cfs.MAX_BYTES + 1
    _track(repo, "research/new/raw/trace.jsonl", size)
    _stage_allowlist(cfs, repo, "")
    (repo / cfs.ALLOWLIST).write_text(f"{size} research/new/raw/trace.jsonl\n")  # unstaged edit
    assert cfs.main(["--root", str(repo)]) == 1


def test_main_fails_on_a_stale_allowlist_entry(cfs, repo, capsys):
    _track(repo, "README.md", 10)
    _stage_allowlist(cfs, repo, f"{cfs.MAX_BYTES + 1} research/old/raw/trace.jsonl\n")
    assert cfs.main(["--root", str(repo)]) == 1
    assert "no longer tracked" in capsys.readouterr().err


def test_main_reports_a_malformed_allowlist_in_one_line_and_exits_2(cfs, repo, capsys):
    _track(repo, "README.md", 10)
    _stage_allowlist(cfs, repo, "not-a-number research/a.json\n")
    assert cfs.main(["--root", str(repo)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "file_size_allowlist.txt:1" in captured.err
    assert len(captured.err.strip().splitlines()) == 1


def test_main_reports_a_git_failure_in_one_line_and_exits_2(cfs, tmp_path, capsys):
    # tmp_path is not a git repository
    assert cfs.main(["--root", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("error: git exited ")
    assert len(captured.err.strip().splitlines()) == 1
