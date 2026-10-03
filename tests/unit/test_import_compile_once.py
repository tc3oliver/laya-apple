"""Unit tests for research/import-compile-once/ (research only, #162): analyze.py's guard and
preregistered thresholds on synthetic rows, the harness's e5rt listing diff and the path safety of
its cleanup, and proto_import.py's mechanism A with Core ML stubbed (a stand-in coremltools, a
clean compute plan, a stubbed parity gate and probe), so the real manifest, profile, hash and
lock code runs."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import types
from pathlib import Path

import pytest

import laya_apple.artifacts as A
import laya_apple.hub as hub
import laya_apple.parity.ane as parity_ane
from laya_apple.artifacts import COMPILED, artifact_dir, load_verified, tree_sha256
from laya_apple.backends.coreml_ane import PROBE_MAX_RATIO
from laya_apple.errors import ArtifactError, ArtifactMissingError, ArtifactParityError, ComputeUnitMismatchError
from laya_apple.registry import models

DIR = Path(__file__).resolve().parents[2] / "research" / "import-compile-once"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


analyze = _load("ico_analyze_for_tests", DIR / "analyze.py")
harness = _load("ico_harness_for_tests", DIR / "harness.py")
proto = _load("ico_proto_for_tests", DIR / "proto_import.py")

PARITY = {"passed": True, "hard_mismatches": 0, "prob_max_abs": 0.004, "rows": 40, "near_tie_flips": []}
PREDS = {"logits": {"dtype": "float16", "shape": [1, 8], "sha256": "ab" * 32}}
GB = 10**9


def row(label, arm, *, T, E, F=10.0, W2=1.0, inject="none", bucket=128, published=100.0, **over):
    r = {
        "label": label,
        "arm": arm,
        "bucket": bucket,
        "inject": inject,
        "import": {
            "ok": True,
            "T": T,
            "phases": {"S": 1.0, "P": 20.0, "Q": 2.0},
            "probes": {"probe:staged": {"ratio": 0.4}},
            "events": [{"t": published, "msg": "laya-typed-decisions L128: publishing"}],
            "inject_fired": [],
        },
        "e5rt_import": {"count": 2, "bytes": E},
        "post": {"parity": dict(PARITY), "quarantine": [], "manifest_exists": True, "final_exists": True},
        "reader": {
            "ok": True,
            "F": F,
            "W2": W2,
            "compute_unit": "ComputeUnit.CPU_AND_NE",
            "probe": {"ratio": 0.4},
            "predictions": PREDS,
        },
        "poller": None,
    }
    if arm == "final-path":
        r["poller"] = [
            {"event": "ready"},
            {"event": "attempt", "t0": 1.0, "t1": 1.2, "outcome": "ArtifactMissingError", "quarantine": []},
        ]
    r.update(over)
    return r


def inj_i(**over):
    r = row("inject-i-L64", "final-path", T=30, E=GB, inject="parity-fail", bucket=64)
    r["import"].update(ok=False, error={"type": "ArtifactParityError"}, inject_fired=["parity-fail"])
    r["post"] = {"final_exists": False, "manifest_exists": False, "quarantine": []}
    r.update(over)
    return r


def screen(base_T=100.0, base_E=2 * GB, T=60.0, E=GB, F=10.0, base_F=10.0):
    return [
        row("baseline-L128-1", "baseline", T=base_T, E=base_E, F=base_F),
        row("final-path-L128-1", "final-path", T=T, E=E, F=F),
        inj_i(),
    ]


# ----------------------------------------------------------------------------- analyze


def test_thresholds_are_the_preregistered_ones():
    assert analyze.PROBE_MAX_RATIO == PROBE_MAX_RATIO == 0.8
    assert (analyze.SAVES_T_RATIO, analyze.SAVES_MIN_SAVING_S, analyze.SAVES_E_RATIO, analyze.SAVES_F_SLACK_S) == (
        0.80,
        15.0,
        0.6,
        5.0,
    )
    assert (analyze.NO_EFFECT_T_RATIO, analyze.NO_EFFECT_E_RATIO, analyze.REUSE_SLACK_S) == (0.90, 0.9, 10.0)


def test_screen_saves_at_the_inclusive_edges():
    # T = 0.80 B, B - T = 20 >= 15, E = 0.6 Eb, F = Fb + 5: every bound is inclusive
    res = analyze.analyze(screen(base_T=100.0, T=80.0, base_E=10 * GB, E=6 * GB, F=15.0), "screen")
    assert res["label"] == "screen-SAVES"


@pytest.mark.parametrize(
    "kw",
    [
        {"T": 80.1},  # T > 0.80 B
        {"base_T": 70.0, "T": 56.0},  # T = 0.80 B, but B - T = 14 s < 15 s
        {"E": 6.1 * GB},  # E > 0.6 Eb
        {"F": 15.1},  # F > Fb + 5
    ],
)
def test_screen_inconclusive_when_one_saves_bound_misses(kw):
    args = {"base_T": 100.0, "T": 80.0, "base_E": 10 * GB, "E": 6 * GB, "F": 15.0} | kw
    assert analyze.analyze(screen(**args), "screen")["label"] == "screen-INCONCLUSIVE"


@pytest.mark.parametrize("kw", [{"T": 90.0, "E": GB}, {"T": 50.0, "E": 9 * GB}])
def test_screen_no_effect_on_time_or_on_e5rt(kw):
    res = analyze.analyze(screen(base_T=100.0, base_E=10 * GB, **kw), "screen")
    assert res["label"] == "screen-NO-EFFECT"


def test_both_rules_holding_is_reported_not_resolved():
    res = analyze.analyze(screen(base_T=100.0, T=50.0, base_E=0, E=0), "screen")
    assert res["label"] == "screen-UNDEFINED"


def test_full_needs_every_repeat():
    base = [row(f"baseline-L128-{i}", "baseline", T=100.0, E=10 * GB) for i in (1, 2, 3)]
    fin = [row(f"final-path-L128-{i}", "final-path", T=t, E=GB) for i, t in ((1, 50.0), (2, 60.0), (3, 81.0))]
    inj = [inj_i()]
    for k in ("probe-fail",):
        r = inj_i(label="inject-ii-L64", inject=k)
        r["import"]["inject_fired"] = [k]
        inj.append(r)
    iii = row("inject-iii-L64", "final-path", T=1, E=0, inject="force-replace-probe-fail", bucket=64)
    iii["import"].update(ok=False, inject_fired=["force-replace-probe-fail"])
    iii["setup"] = {"ok": True}
    iii["before"] = {"manifest_exists": True, "manifest_sha256": "m", "tree_sha256": "t"}
    iii["post"] = {"manifest_sha256": "m", "tree_sha256": "t"}
    iv = row("inject-iv-L64", "final-path", T=1, E=0, inject="sigkill", bucket=64)
    iv["import"]["killed"] = {"after_gate_start_s": 3.0}
    iv["post"] = {"manifest_exists": False}
    iv["reader"] = {"ok": False, "error": {"type": "ArtifactMissingError"}}
    iv["reimport"] = {"ok": True}
    iv["reimport_post"] = {
        "manifest_exists": True,
        "leftovers": [],
        "pending_files": [],
        "staging": [],
        "quarantine": [],
    }
    res = analyze.analyze(base + fin + inj + [iii, iv], "full")
    assert res["guard"] and not any(res["guard"].values())
    assert res["label"] == "INCONCLUSIVE"  # repeat 3 misses T <= 0.80 B: repeats disagree
    fin[2]["import"]["T"] = 80.0
    assert analyze.analyze(base + fin + inj + [iii, iv], "full")["label"] == "SAVES"
    assert analyze.analyze(base + fin + inj + [iii], "full")["label"] == "INCOMPLETE"
    iii["post"]["tree_sha256"] = "changed"
    assert analyze.analyze(base + fin + inj + [iii, iv], "full")["label"] == "INVALID"


def test_guard_failures_make_the_screen_invalid():
    rows = screen(T=60.0)
    rows[1]["post"]["parity"]["prob_max_abs"] = 0.005  # differs from baseline's summary
    assert analyze.analyze(rows, "screen")["label"] == "screen-INVALID"
    rows = screen(T=60.0)
    rows[1]["import"]["probes"]["probe:registered"] = {"ratio": 0.81}
    assert analyze.analyze(rows, "screen")["label"] == "screen-INVALID"
    rows = screen(T=60.0)
    rows[1]["reader"]["predictions"] = {"logits": dict(PREDS["logits"], sha256="cd" * 32)}
    assert analyze.analyze(rows, "screen")["label"] == "screen-INVALID"
    rows = screen(T=60.0)
    rows[2]["poller"].append({"event": "manifest_seen", "t0": 5.0})
    assert analyze.analyze(rows, "screen")["label"] == "screen-INVALID"
    rows = screen(T=60.0)
    rows[0]["reader"]["probe"] = {"ratio": 0.9}
    assert analyze.analyze(rows, "screen")["label"] == "screen-INVALID (baseline arm)"


def test_poller_guard():
    r = row("f", "final-path", T=1, E=1, published=100.0)
    assert analyze.guard_poller(r) == []
    r["poller"].append({"event": "attempt", "t0": 99.0, "t1": 100.5, "outcome": "model", "quarantine": []})
    assert analyze.guard_poller(r) == []  # finished loading after the manifest was published
    r["poller"].append({"event": "attempt", "t0": 98.0, "t1": 99.0, "outcome": "model", "quarantine": []})
    assert analyze.guard_poller(r)  # a model before publishing
    r = row("f", "final-path", T=1, E=1)
    r["poller"].append({"event": "attempt", "t0": 2, "t1": 3, "outcome": "ArtifactIntegrityError", "quarantine": []})
    assert analyze.guard_poller(r)
    r = row("f", "final-path", T=1, E=1)
    r["poller"][1]["quarantine"] = ["x"]
    assert analyze.guard_poller(r)
    r = row("f", "final-path", T=1, E=1)
    r["poller"] = [{"event": "ready"}]
    assert analyze.guard_poller(r) == ["the poller made no attempt during the import"]


# ----------------------------------------------------------------------------- e5rt listing and cleanup

HEX = "FDF653365C60E8E5325AB7E15954662643231E429A0D8EFBC51299B49F05838A"


def _bundle(caches: Path, name: str, payload=b"x" * 100) -> Path:
    d = harness.entry_path(name, caches)
    d.mkdir(parents=True)
    (d / "bundle.bin").write_bytes(payload)
    return d


def test_e5rt_listing_and_diff(tmp_path):
    caches = tmp_path / "Caches"
    _bundle(caches, f"python/26A428/{HEX}")
    (caches / "com.apple.Spotlight" / harness.E5RT / "26A428" / "AA").mkdir(parents=True)  # not python*: ignored
    before = harness.e5rt_names(caches)
    assert before == [f"python/26A428/{HEX}"]
    new = "B" * 64
    _bundle(caches, f"python3/26A428/{new}", b"y" * 300)
    d = harness.e5rt_diff(before, harness.e5rt_names(caches), caches)
    assert (d["count"], d["bytes"], d["removed_count"]) == (1, 300, 0)
    assert d["new"][0]["name"] == f"python3/26A428/{new}"


def test_safe_e5rt_target_refuses_anything_but_a_hex_entry_in_a_build_dir(tmp_path):
    caches = tmp_path / "Caches"
    good = _bundle(caches, f"python/26A428/{HEX}")
    assert harness.safe_e5rt_target(f"python/26A428/{HEX}", caches) == good.resolve()
    for bad in (
        f"python/26A428/../{HEX}",
        f"python/../{HEX}",
        "python/26A428/not-hex",
        f"Spotlight/26A428/{HEX}",
        f"python/26A428/{HEX}/sub",
        f"python/26A428/{'C' * 64}",
    ):  # the last one does not exist
        assert harness.safe_e5rt_target(bad, caches) is None, bad
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    link = harness.entry_path(f"python/26A428/{'D' * 64}", caches)
    link.symlink_to(outside)
    assert harness.safe_e5rt_target(f"python/26A428/{'D' * 64}", caches) is None


def test_cleanup_removes_only_new_safe_entries_and_ledger_temp_dirs(tmp_path):
    caches, tmp_root = tmp_path / "Caches", tmp_path / "T"
    tmp_root.mkdir()
    old = _bundle(caches, f"python/26A428/{HEX}")
    before = harness.e5rt_names(caches)
    new = _bundle(caches, f"python/26A428/{'E' * 64}")
    odd = harness.entry_path("python/26A428/readme", caches)
    odd.mkdir()
    outside = tmp_path / "keep"
    outside.mkdir()
    harness.entry_path(f"python/26A428/{'F' * 64}", caches).symlink_to(outside)
    mine = tmp_root / (harness.TMP_PREFIX + "cache-1")
    mine.mkdir()
    foreign = tmp_root / "someone-else"
    foreign.mkdir()
    nested = mine / (harness.TMP_PREFIX + "inner")
    nested.mkdir()
    ledger = [str(nested), str(foreign), str(mine), str(tmp_path / "gone"), ""]
    dry = harness.cleanup(before, ledger, caches=caches, tmp_root=tmp_root, dry_run=True)
    assert new.exists() and mine.exists() and [r["name"] for r in dry["e5rt_removed"]] == [f"python/26A428/{'E' * 64}"]
    rep = harness.cleanup(before, ledger, caches=caches, tmp_root=tmp_root)
    assert old.exists() and not new.exists() and odd.exists() and outside.exists()
    assert sorted(rep["e5rt_skipped"]) == [f"python/26A428/{'F' * 64}", "python/26A428/readme"]
    assert not mine.exists() and foreign.exists()
    assert rep["tmp_removed"] == [str(mine.resolve())]
    assert sorted(rep["tmp_skipped"]) == sorted([str(nested), str(foreign)])


def test_redaction_and_phase_timer():
    pairs = harness.redactions("/Volumes/X/cache")
    text = f"/Volumes/X/cache/artifacts {Path.home()}/Library/Caches {harness.REPO}/laya_apple"
    assert "/Volumes/X" not in harness.redact_text(text, pairs)
    assert str(Path.home()) not in harness.redact_text(text, pairs)
    t = harness.PhaseTimer()
    inner = t.wrap("S", lambda: 1)
    outer = t.wrap("R", lambda: inner() + 1)
    assert outer() == 2 and inner() == 1
    assert set(t.totals) == {"R", "S"}  # S counted once, for the outer-less call only


# ----------------------------------------------------------------------------- proto_import

SPEC = models()["laya-typed-decisions"]
HERE = {"soc": "Apple M4 Max", "macos": "27.0", "macos_build": "26A428", "coremltools": "9.0"}
CLEAN_PLAN = {"compute_units": "CPU_AND_NE", "ops": {"ane": 1, "cpu": 0, "gpu": 0}, "transitions": 0}


@pytest.fixture
def stubbed(tmp_path, monkeypatch, manifest_factory):
    """The real proto_import / import_artifact with Core ML stubbed. Returns (archive, calls);
    the parity stub and the probe record whether a manifest was visible to the runtime."""
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(A, "platform_profile", lambda: dict(HERE))
    ct = types.ModuleType("coremltools")
    ct.ComputeUnit = types.SimpleNamespace(CPU_AND_NE="CPU_AND_NE", CPU_ONLY="CPU_ONLY")
    ct.models = types.SimpleNamespace(CompiledMLModel=lambda *a, **k: object())
    monkeypatch.setitem(sys.modules, "coremltools", ct)
    monkeypatch.setattr(A, "compute_plan_summary", lambda *a, **k: dict(CLEAN_PLAN))
    monkeypatch.setattr(hub, "checkpoint_path", lambda spec, local_files_only=False: tmp_path)
    monkeypatch.setattr(hub, "verify_weights", lambda spec, path: "ok")
    calls = {"parity": [], "probe": [], "parity_passed": True}

    def visible():
        try:
            load_verified(SPEC, 64)
            return True
        except ArtifactMissingError:
            return False

    def parity(spec, compiled, length, ckpt):
        calls["parity"].append((Path(compiled), visible()))
        return {"passed": calls["parity_passed"], "hard_mismatches": 0}

    monkeypatch.setattr(parity_ane, "ane_parity", parity)
    data = manifest_factory(SPEC, 64, platform=HERE)
    weights = b"weights"
    h = hashlib.sha256()
    h.update(b"weights.bin")
    h.update(hashlib.sha256(weights).digest())
    data["integrity"]["artifact_sha256"] = h.hexdigest()
    archive = tmp_path / "laya-typed-decisions-L64.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, payload in (("manifest.json", json.dumps(data).encode()), (f"{COMPILED}/weights.bin", weights)):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return archive, calls


def _probe(calls, fail=False):
    def probe(spec, bucket, compiled):
        calls["probe"].append(Path(compiled))
        if fail:
            raise ComputeUnitMismatchError("simulated")
        return {"ratio": 0.4}

    return probe


def test_proto_validates_at_the_registered_path_with_the_manifest_withheld(stubbed):
    archive, calls = stubbed
    final = proto.proto_import_artifact(
        archive, local_files_only=True, log=lambda m: None, probe=_probe(calls), registered_probe=_probe(calls)
    )
    assert final == artifact_dir(SPEC, 64)
    assert calls["parity"] == [(final / COMPILED, False)]  # at the registered path, not visible yet
    assert calls["probe"] == [final / COMPILED]  # one probe, at the registered path
    assert sorted(p.name for p in final.iterdir()) == ["manifest.json", COMPILED]
    assert json.loads((final / "manifest.json").read_text())["imported"]["artifact_sha256_check"] == tree_sha256(
        final / COMPILED
    )
    load_verified(SPEC, 64)  # registered; the stamp the prototype wrote matches load_verified's key


@pytest.mark.parametrize("fail", ["parity", "probe"])
def test_proto_failure_leaves_nothing_registered(stubbed, fail):
    archive, calls = stubbed
    calls["parity_passed"] = fail != "parity"
    with pytest.raises(ArtifactParityError if fail == "parity" else ComputeUnitMismatchError):
        proto.proto_import_artifact(
            archive, local_files_only=True, log=lambda m: None, probe=_probe(calls, fail=fail == "probe")
        )
    final = artifact_dir(SPEC, 64)
    assert not final.exists() and list(final.parent.iterdir()) == []
    assert not (A.artifacts_root() / "quarantine").exists()


def test_proto_recovers_an_unpublished_install_and_refuses_a_registered_one(stubbed):
    archive, calls = stubbed
    final = artifact_dir(SPEC, 64)
    (final / COMPILED).mkdir(parents=True)
    (final / proto.PENDING).write_text(json.dumps({"pid": 999999}))
    (final / proto.PENDING_MANIFEST).write_text("{}")
    with pytest.raises(ArtifactMissingError):
        load_verified(SPEC, 64)  # what a reader sees after a SIGKILL during the gate
    proto.proto_import_artifact(archive, local_files_only=True, log=lambda m: None, probe=_probe(calls))
    assert sorted(p.name for p in final.parent.iterdir()) == [final.name]
    assert not (final / proto.PENDING).exists()
    with pytest.raises(ArtifactError, match="already exists"):
        proto.proto_import_artifact(archive, local_files_only=True, log=lambda m: None, probe=_probe(calls))


def test_proto_force_delegates_to_main(stubbed, monkeypatch):
    archive, calls = stubbed
    seen = []
    import laya_apple.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "import_artifact", lambda *a, **k: seen.append(k["force"]) or "main")
    assert proto.proto_import_artifact(archive, force=True, log=lambda m: None) == "main"
    assert seen == [True]
    assert os.environ["LAYA_APPLE_CACHE"]
