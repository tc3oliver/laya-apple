"""Unit tests for scripts/bench_preflight.py: disk and Core ML E5 cache thresholds."""

from __future__ import annotations

import importlib.util
import sys
from collections import namedtuple
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3
Usage = namedtuple("Usage", "total used free")


@pytest.fixture(scope="module")
def bp():
    spec = importlib.util.spec_from_file_location("bench_preflight", ROOT / "scripts" / "bench_preflight.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass looks its module up in sys.modules
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


@pytest.fixture
def caches(tmp_path):
    """A fake ~/Library/Caches: two E5 caches and one unrelated cache."""
    root = tmp_path / "Caches"
    _write(root / "python3/com.apple.e5rt.e5bundlecache/25G83/AAA/a.bundle/H16C.e5", 3000)
    _write(root / "python3/com.apple.e5rt.e5bundlecache/25G83/BBB/weights", 2000)
    _write(root / "org.python.python/com.apple.e5rt.e5bundlecache/25G83/CCC/H16C.e5", 500)
    _write(root / "com.example.app/other-cache/big.bin", 10_000)
    return root


def _fixed_free(monkeypatch, bp, free_bytes):
    monkeypatch.setattr(bp.shutil, "disk_usage", lambda _: Usage(0, 0, free_bytes))


def test_finds_only_e5_cache_dirs(bp, caches):
    assert [p.parent.name for p in bp.e5_cache_dirs(caches)] == ["org.python.python", "python3"]


def test_missing_caches_root_is_empty(bp, tmp_path):
    assert bp.e5_cache_dirs(tmp_path / "nope") == []


def test_tree_bytes_sums_files_without_following_symlinks(bp, caches, tmp_path):
    e5 = caches / "python3/com.apple.e5rt.e5bundlecache"
    outside = tmp_path / "outside"
    _write(outside / "huge.bin", 50_000)
    (e5 / "link").symlink_to(outside, target_is_directory=True)
    assert bp.tree_bytes(e5) == (5000, 0)


def test_symlinked_cache_dir_is_not_counted(bp, caches, tmp_path):
    target = tmp_path / "elsewhere" / "com.apple.e5rt.e5bundlecache"
    _write(target / "x", 100)
    (caches / "linked").mkdir()
    (caches / "linked" / "com.apple.e5rt.e5bundlecache").symlink_to(target, target_is_directory=True)
    assert "linked" not in [p.parent.name for p in bp.e5_cache_dirs(caches)]


def test_unreadable_entries_are_counted_and_reported(bp, caches, tmp_path, monkeypatch, capsys):
    locked = caches / "python3/com.apple.e5rt.e5bundlecache/25G83/AAA"
    locked.chmod(0)
    try:
        size, unreadable = bp.tree_bytes(caches / "python3/com.apple.e5rt.e5bundlecache")
        _fixed_free(monkeypatch, bp, 500 * GIB)
        assert bp.check(tmp_path, caches, env={}) == 0
    finally:
        locked.chmod(0o755)
    assert (size, unreadable) == (2000, 1)
    assert "1 entries could not be read" in capsys.readouterr().out


def test_measure_reports_each_cache_and_total(bp, caches, tmp_path, monkeypatch):
    _fixed_free(monkeypatch, bp, 300 * GIB)
    r = bp.measure(tmp_path, caches)
    assert r.free_bytes == 300 * GIB
    assert {p.parent.name: size for p, size in r.caches} == {"python3": 5000, "org.python.python": 500}
    assert r.e5_bytes == 5500


def test_measure_accepts_a_workspace_that_does_not_exist_yet(bp, caches, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(bp.shutil, "disk_usage", lambda p: seen.append(Path(p)) or Usage(0, 0, GIB))
    bp.measure(tmp_path / "benchmarks" / "v9.9", caches)
    assert seen == [tmp_path]


@pytest.mark.parametrize(
    "free_gib, e5_gib, fails, warns",
    [
        (250, 10, False, False),
        (200, 50, False, False),  # exactly at both levels is allowed
        (199.9, 10, True, False),
        (250, 50.1, False, True),
        (100, 80, True, True),
    ],
)
def test_free_disk_fails_and_e5_size_only_warns(bp, free_gib, e5_gib, fails, warns):
    r = bp.Report(Path("/w"), int(free_gib * GIB), [(Path("/c"), int(e5_gib * GIB))])
    assert (bp.failure(r, 200) is not None) == fails
    assert (bp.warning(r, 50) is not None) == warns
    if fails:
        assert "free disk" in bp.failure(r, 200)
    if warns:
        assert "E5 caches" in bp.warning(r, 50)


def test_zero_disables_each_check(bp):
    r = bp.Report(Path("/w"), 0, [(Path("/c"), 999 * GIB)])
    assert bp.failure(r, 0) is None
    assert bp.warning(r, 0) is None


def test_thresholds_from_env(bp):
    assert bp.thresholds_from_env({}) == (200.0, 50.0)
    env = {"LAYA_APPLE_PREFLIGHT_MIN_FREE_GIB": "40", "LAYA_APPLE_PREFLIGHT_WARN_E5_GIB": "0"}
    assert bp.thresholds_from_env(env) == (40.0, 0.0)
    with pytest.raises(SystemExit, match="number of GiB"):
        bp.thresholds_from_env({"LAYA_APPLE_PREFLIGHT_WARN_E5_GIB": "lots"})
    for bad in ("-1", "nan", "inf"):
        with pytest.raises(SystemExit, match="finite number >= 0"):
            bp.thresholds_from_env({"LAYA_APPLE_PREFLIGHT_MIN_FREE_GIB": bad})


def test_check_passes_and_reports(bp, caches, tmp_path, monkeypatch, capsys):
    _fixed_free(monkeypatch, bp, 500 * GIB)
    assert bp.check(tmp_path, caches, env={}) == 0
    out, err = capsys.readouterr()
    assert "free disk: 500.0 GiB" in out
    assert str(caches / "python3/com.apple.e5rt.e5bundlecache") in out
    assert "preflight: ok" in out
    assert err == ""


def test_large_e5_cache_warns_but_the_run_may_start(bp, caches, tmp_path, monkeypatch, capsys):
    # The current machine's case: ~94 GiB of E5 caches, ~330 GiB free.
    _fixed_free(monkeypatch, bp, 330 * GIB)
    before = sorted(caches.rglob("*"))
    assert bp.check(tmp_path, caches, env={"LAYA_APPLE_PREFLIGHT_WARN_E5_GIB": "0.000001"}) == 0
    out, err = capsys.readouterr()
    assert "preflight WARNING" in err and "E5 caches total" in err
    assert str(caches / "python3/com.apple.e5rt.e5bundlecache") in out
    assert "FAILED" not in err and "Not starting" not in err
    assert "preflight: ok (with warning)" in out
    assert sorted(caches.rglob("*")) == before


def test_low_free_disk_fails_fast_with_actionable_message_and_deletes_nothing(
    bp, caches, tmp_path, monkeypatch, capsys
):
    _fixed_free(monkeypatch, bp, 10 * GIB)
    before = sorted(caches.rglob("*"))
    assert bp.check(tmp_path, caches, env={"LAYA_APPLE_PREFLIGHT_WARN_E5_GIB": "0.000001"}) == 1
    err = capsys.readouterr().err
    assert "preflight FAILED" in err and "below the 200 GiB minimum" in err
    assert "preflight WARNING" in err
    assert "Nothing was deleted" in err
    assert "docs/benchmarks.md" in err
    assert sorted(caches.rglob("*")) == before


def test_main_exit_code_follows_the_check(bp, caches, tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "Library").mkdir(parents=True)
    caches.rename(home / "Library" / "Caches")
    monkeypatch.setattr(bp.Path, "home", lambda: home)
    monkeypatch.delenv("LAYA_APPLE_PREFLIGHT_MIN_FREE_GIB", raising=False)
    monkeypatch.delenv("LAYA_APPLE_PREFLIGHT_WARN_E5_GIB", raising=False)
    _fixed_free(monkeypatch, bp, 500 * GIB)
    assert bp.main(["--workspace", str(tmp_path / "out")]) == 0
    _fixed_free(monkeypatch, bp, GIB)
    assert bp.main(["--workspace", str(tmp_path / "out")]) == 1
