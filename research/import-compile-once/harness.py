"""Harness for #162 (research/import-compile-once): one recorded step per call.

    python harness.py preflight  --out-dir OUT
    python harness.py e5rt-snapshot --out OUT/e5rt-run-before.json
    python harness.py step --label baseline-L128-1 --arm baseline --bucket 128 --archive A.tar.gz --out-dir OUT
    python harness.py step --label final-path-L128-1 --arm final-path --bucket 128 --archive A.tar.gz --poller ...
    python harness.py step --label inject-i-L64 --arm final-path --bucket 64 --archive B.tar.gz --poller \
        --inject parity-fail ...
    python harness.py cleanup --before OUT/e5rt-run-before.json --ledger OUT/created-paths.txt --out OUT/cleanup.json

The parent (`step`, `preflight`, `e5rt-snapshot`, `cleanup`) uses only the standard library and
never imports laya_apple or coremltools. Everything that touches Core ML runs in a child process
of the same interpreter (`sys.executable`, so the same e5rt process name):
- `_import`: one import call (main's `import_artifact` or `proto_import_artifact`), with
  sub-phase timers wrapped around the laya_apple helpers both call, and the injections;
- `_reader`: `load_verified` + the runtime placement probe at the registered path, then a second
  `load_verified` in the same process;
- `_poll`: `load_verified(spec, bucket)` every 0.5 s while an import runs;
- `_inspect`: the on-disk state of the cache after a step.

The parent records raw facts only; analyze.py applies the preregistered guard and classification.
Each step uses a new empty LAYA_APPLE_CACHE under the internal-disk temp dir and removes it at the
end. `cleanup` removes only e5rt bundle directories that are new since the run's first snapshot,
after checking each one sits directly inside an e5rt build directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL = "laya-typed-decisions"
E5RT = "com.apple.e5rt.e5bundlecache"
BUNDLE_NAME = re.compile(r"^[0-9A-Fa-f]{16,128}$")
TMP_PREFIX = "laya-ic-"
POLL_INTERVAL_S = 0.5
INJECTIONS = ("none", "parity-fail", "probe-fail", "force-replace-probe-fail", "sigkill")
MIN_FREE_GB = 200
DATA_VOLUME_ENV = "LAYA_IC_DATA_VOLUME"  # the second volume whose free space is checked (optional)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # a line cut by SIGKILL
    return out


# ----------------------------------------------------------------------------- e5rt listing


def caches_dir() -> Path:
    return Path.home() / "Library" / "Caches"


def e5rt_build_dirs(caches: Path | None = None) -> list[Path]:
    """Every `<caches>/python*/com.apple.e5rt.e5bundlecache/<build>/` directory."""
    caches = caches or caches_dir()
    out = []
    for proc in sorted(caches.glob("python*")):
        root = proc / E5RT
        if root.is_dir() and not root.is_symlink():
            out.extend(sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()))
    return out


def e5rt_names(caches: Path | None = None) -> list[str]:
    """Names (`<proc>/<build>/<entry>`) of every entry in every e5rt build directory. No sizes,
    no contents: listing only."""
    caches = caches or caches_dir()
    names = []
    for build in e5rt_build_dirs(caches):
        rel = build.relative_to(caches)
        names.extend(f"{rel.parts[0]}/{rel.parts[2]}/{p.name}" for p in build.iterdir())
    return sorted(names)


def entry_path(name: str, caches: Path | None = None) -> Path:
    proc, build, entry = name.split("/")
    return (caches or caches_dir()) / proc / E5RT / build / entry


def entry_size(path: Path) -> dict:
    """Apparent bytes and allocated KiB (as `du -sk`) of one e5rt entry."""
    nbytes = blocks = 0
    paths = [path] if not path.is_dir() else [Path(r) / f for r, _, fs in os.walk(path) for f in fs]
    for p in paths:
        try:
            st = os.lstat(p)
        except OSError:
            continue
        nbytes += st.st_size
        blocks += st.st_blocks
    return {"bytes": nbytes, "kb": blocks * 512 // 1024}


def e5rt_diff(before: list[str], after: list[str], caches: Path | None = None) -> dict:
    """New entries between two listings, each with its size (only new entries are sized)."""
    new = sorted(set(after) - set(before))
    rows = [{"name": n, **entry_size(entry_path(n, caches))} for n in new]
    return {
        "count": len(rows),
        "bytes": sum(r["bytes"] for r in rows),
        "kb": sum(r["kb"] for r in rows),
        "removed_count": len(set(before) - set(after)),
        "new": rows,
    }


def safe_e5rt_target(name: str, caches: Path | None = None) -> Path | None:
    """The path of e5rt entry `name` if it is safe to delete: a hex-named entry directly inside
    `<caches>/python*/com.apple.e5rt.e5bundlecache/<build>/`, not a symlink, after resolving.
    Else None."""
    caches = (caches or caches_dir()).resolve()
    parts = name.split("/")
    if len(parts) != 3 or any(p in ("", ".", "..") for p in parts):
        return None
    proc, build, entry = parts
    if not proc.startswith("python") or not BUNDLE_NAME.match(entry):
        return None
    build_dir = caches / proc / E5RT / build
    path = build_dir / entry
    if path.is_symlink() or build_dir.is_symlink() or (caches / proc / E5RT).is_symlink():
        return None
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if resolved.parent != build_dir.resolve() or not resolved.is_relative_to(caches):
        return None
    return resolved


def safe_tmp_target(path: str, tmp_root: Path | None = None) -> Path | None:
    """A temp directory this harness created (TMP_PREFIX, directly inside the temp dir), or None."""
    tmp_root = (tmp_root or Path(tempfile.gettempdir())).resolve()
    p = Path(path)
    if p.is_symlink():
        return None
    try:
        resolved = p.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if resolved.parent != tmp_root or not resolved.name.startswith(TMP_PREFIX) or not resolved.is_dir():
        return None
    return resolved


# ----------------------------------------------------------------------------- machine snapshot


def _cmd(*cmd, cwd=None) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=cwd).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"<{type(e).__name__}: {e}>"


def _version(dist: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def machine_snapshot() -> dict:
    data = os.environ.get(DATA_VOLUME_ENV)
    vols = ["/"] + ([data] if data and Path(data).is_dir() else [])
    e5 = {}
    for build in e5rt_build_dirs():
        du = _cmd("du", "-sk", str(build)).split()
        e5[str(build.relative_to(caches_dir()))] = {
            "entries": sum(1 for _ in build.iterdir()),
            "kb": int(du[0]) if du and du[0].isdigit() else None,
        }
    return {
        "at": now(),
        "sw_vers": _cmd("sw_vers"),
        "uptime": _cmd("uptime"),
        "pmset_therm": _cmd("pmset", "-g", "therm"),
        "pmset_batt": _cmd("pmset", "-g", "batt"),
        "df": _cmd("df", "-k", *vols),
        "e5rt": e5,
        "git_sha": _cmd("git", "rev-parse", "HEAD", cwd=REPO),
        "git_dirty": bool(_cmd("git", "status", "--porcelain", cwd=REPO)),
        "coremltools": _version("coremltools"),
        "laya_apple": _version("laya-apple"),
        "python": sys.version.split()[0],
        "executable": Path(sys.executable).name,
    }


def free_gb(df_k: str) -> dict:
    """{mount: free GB} from `df -k` output."""
    out = {}
    for line in df_k.splitlines()[1:]:
        f = line.split()
        if len(f) >= 6 and f[3].isdigit():
            out[f[-1]] = int(f[3]) * 1024 / 1e9
    return out


# ----------------------------------------------------------------------------- children


class PhaseTimer:
    """Wall time per sub-phase. A call made inside a timed call counts toward the outer one
    (load_verified's own hash and compute plan are part of R, not S)."""

    def __init__(self):
        self.totals: dict[str, float] = {}
        self.depth = 0

    def wrap(self, label: str, fn):
        def timed(*a, **k):
            if self.depth:
                return fn(*a, **k)
            self.depth += 1
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.depth -= 1
                self.totals[label] = self.totals.get(label, 0.0) + time.perf_counter() - t

        return timed


def _events(path: Path):
    fh = open(path, "a", buffering=1)

    def log(msg, **extra):
        fh.write(json.dumps({"t": time.time(), "msg": str(msg), **extra}) + "\n")
        fh.flush()
        print(msg, file=sys.stderr, flush=True)

    return log


def child_import(a) -> int:
    """One import call in this process; writes its result JSON (a SIGKILL leaves only events)."""
    sys.path.insert(0, str(HERE))
    from proto_import import proto_import_artifact

    import laya_apple.artifacts as A
    import laya_apple.parity.ane as PA
    from laya_apple.artifacts import COMPILED, artifact_dir
    from laya_apple.errors import ComputeUnitMismatchError
    from laya_apple.lifecycle import import_artifact
    from laya_apple.prebuilt import _probe
    from laya_apple.registry import resolve

    spec = resolve(a.model)
    final = artifact_dir(spec, a.bucket)
    final_compiled = final / COMPILED
    log = _events(Path(a.events))
    log("start", final=str(final), arm=a.arm, inject=a.inject, pid=os.getpid())
    timer = PhaseTimer()
    fired: list[str] = []
    probes: dict[str, dict] = {}

    real_parity = PA.ane_parity

    def parity(spec_, compiled, length, ckpt):
        out = real_parity(spec_, compiled, length, ckpt)
        if a.inject == "parity-fail" and Path(compiled) == final_compiled:
            fired.append("parity-fail")
            out = dict(out, passed=False)
        return out

    def hook(role):
        def run(spec_, bucket, compiled):
            result = _probe(spec_, bucket, compiled, local_files_only=True)
            probes[f"{role}:{'registered' if Path(compiled) == final_compiled else 'staged'}"] = result
            if a.inject in ("probe-fail", "force-replace-probe-fail") and Path(compiled) == final_compiled:
                fired.append(a.inject)
                raise ComputeUnitMismatchError(f"injected: the probe at the registered path failed ({result})")
            return result

        return run

    for name, label in (("verify_files", "S"), ("tree_sha256", "S"), ("compute_plan_summary", "S")):
        setattr(A, name, timer.wrap(label, getattr(A, name)))
    A.load_verified = timer.wrap("R", A.load_verified)
    PA.ane_parity = timer.wrap("P", parity)
    probe = timer.wrap("Q", hook("probe"))
    # main: the registered-path load and probe are R. final-path: its one probe at the
    # registered path is Q (the prototype calls registered_probe there), and there is no R.
    registered_probe = timer.wrap("R" if a.arm == "baseline" else "Q", hook("registered_probe"))
    fn = import_artifact if a.arm == "baseline" else proto_import_artifact
    result = {"arm": a.arm, "bucket": a.bucket, "inject": a.inject, "force": a.force, "final": str(final)}
    t = time.perf_counter()
    try:
        fn(
            Path(a.archive),
            local_files_only=True,
            force=a.force,
            log=log,
            probe=probe,
            registered_probe=registered_probe,
            expect=(spec.name, a.bucket),
        )
        result.update(ok=True, error=None)
    except BaseException as e:  # recorded, including the injected failures
        result.update(ok=False, error={"type": type(e).__name__, "message": str(e)[:2000]})
    result["T"] = time.perf_counter() - t
    result["phases"] = timer.totals
    result["probes"] = probes
    result["inject_fired"] = fired
    write_json(Path(a.out), result)
    return 0


def _digest(outputs: dict) -> dict:
    import numpy as np

    out = {}
    for k in sorted(outputs):
        v = np.ascontiguousarray(outputs[k])
        out[k] = {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": hashlib.sha256(v.tobytes()).hexdigest()}
    return out


def child_reader(a) -> int:
    """A new process: load_verified + the runtime probe at the registered path, predictions on
    the probe features, then a second load_verified in this process."""
    from laya_apple.artifacts import COMPILED, artifact_dir, load_verified
    from laya_apple.backends.coreml_ane import HostWeights, probe_features, probe_placement
    from laya_apple.hub import checkpoint_path
    from laya_apple.prompt import Tokenizer
    from laya_apple.registry import ANE_COMPUTE_UNITS, resolve

    spec = resolve(a.model)
    res: dict = {"bucket": a.bucket, "requested_compute_units": ANE_COMPUTE_UNITS}
    try:
        ckpt = checkpoint_path(spec, local_files_only=True)
        local_attention = int(json.loads((ckpt / "encoder/config.json").read_text())["local_attention"])
        pad_id = Tokenizer(ckpt / "tokenizer").pad_token_id
        feats = probe_features(HostWeights(ckpt, local_attention), pad_id, a.bucket)
        t = time.perf_counter()
        model, _ = load_verified(spec, a.bucket)
        res["F"] = time.perf_counter() - t
        res["compute_unit"] = str(getattr(model, "compute_unit", None))
        try:
            res["probe"] = probe_placement(spec, a.bucket, model, artifact_dir(spec, a.bucket) / COMPILED, feats)
        except Exception as e:
            res["probe_error"] = {"type": type(e).__name__, "message": str(e)[:2000]}
        res["predictions"] = _digest(model.predict(feats))
        del model
        t = time.perf_counter()
        load_verified(spec, a.bucket)
        res["W2"] = time.perf_counter() - t
        res.update(ok=True, error=None)
    except BaseException as e:
        res.update(ok=False, error={"type": type(e).__name__, "message": str(e)[:2000]})
    write_json(Path(a.out), res)
    return 0


def child_poll(a) -> int:
    """load_verified(spec, bucket) every 0.5 s until the stop file exists; one JSON line per
    attempt. Stops before loading once manifest.json exists, and after one successful load, so
    its own loads add as little as possible to the import's e5rt listing."""
    from laya_apple.artifacts import artifact_dir, artifacts_root, load_verified
    from laya_apple.errors import ArtifactMissingError
    from laya_apple.registry import resolve

    spec = resolve(a.model)
    final = artifact_dir(spec, a.bucket)
    quarantine = artifacts_root() / "quarantine"
    stop = Path(a.stop)
    with open(a.out, "a", buffering=1) as fh:

        def emit(**rec):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

        emit(event="ready", t=time.time(), final=str(final))
        while not stop.exists():
            t0 = time.time()
            if (final / "manifest.json").exists():
                emit(event="manifest_seen", t0=t0)
                break
            try:
                load_verified(spec, a.bucket)
                outcome, msg = "model", None
            except ArtifactMissingError:
                outcome, msg = "ArtifactMissingError", None
            except BaseException as e:
                outcome, msg = type(e).__name__, str(e)[:500]
            q = sorted(p.name for p in quarantine.iterdir()) if quarantine.is_dir() else []
            emit(event="attempt", t0=t0, t1=time.time(), outcome=outcome, message=msg, quarantine=q)
            if outcome == "model":
                break
            time.sleep(max(0.0, t0 + POLL_INTERVAL_S - time.time()))
    return 0


def child_inspect(a) -> int:
    """The on-disk state that the guard reads, for one artifact key in this cache."""
    from laya_apple.artifacts import COMPILED, artifact_dir, artifacts_root, tree_sha256
    from laya_apple.registry import resolve

    spec = resolve(a.model)
    final = artifact_dir(spec, a.bucket)
    root = artifacts_root()
    mf = final / "manifest.json"
    out = {
        "final": str(final),
        "final_exists": final.exists(),
        "manifest_exists": mf.exists(),
        "pending_files": sorted(p for p in ("PENDING.json", "manifest.pending.json") if (final / p).exists()),
        "leftovers": sorted(p.name for p in final.parent.glob(final.name + ".*")) if final.parent.exists() else [],
        "staging": sorted(p.name for p in (root / ".staging").iterdir()) if (root / ".staging").is_dir() else [],
        "quarantine": sorted(p.name for p in (root / "quarantine").iterdir()) if (root / "quarantine").is_dir() else [],
        "manifest_sha256": hashlib.sha256(mf.read_bytes()).hexdigest() if mf.exists() else None,
        "tree_sha256": tree_sha256(final / COMPILED) if (final / COMPILED).exists() else None,
    }
    if mf.exists():
        try:
            imported = json.loads(mf.read_text()).get("imported") or {}
            out["parity"], out["placement"] = imported.get("parity"), imported.get("placement")
        except ValueError:
            out["parity"] = out["placement"] = None
    write_json(Path(a.out), out)
    return 0


# ----------------------------------------------------------------------------- parent


def _child(sub: str, *args: str) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), sub, *args]


def _run_child(sub: str, env: dict, *args: str, timeout: float = 3600) -> int:
    return subprocess.run(_child(sub, *args), env=env, timeout=timeout).returncode


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _new_cache(ledger: Path) -> Path:
    d = Path(tempfile.mkdtemp(prefix=TMP_PREFIX + "cache-"))
    with open(ledger, "a") as fh:
        fh.write(str(d) + "\n")
    return d


def _remove_tmp(path: Path) -> bool:
    target = safe_tmp_target(str(path))
    if target is None:
        return False
    shutil.rmtree(target)
    return True


class Poller:
    def __init__(self, env: dict, model: str, bucket: int, work: Path, label: str):
        self.out, self.stop = work / f"{label}.poll.jsonl", work / f"{label}.poll.stop"
        self.proc = subprocess.Popen(
            _child(
                "_poll", "--model", model, "--bucket", str(bucket), "--out", str(self.out), "--stop", str(self.stop)
            ),
            env=env,
        )
        deadline = time.time() + 300
        while not any(r.get("event") == "ready" for r in read_jsonl(self.out)):
            if self.proc.poll() is not None or time.time() > deadline:
                raise RuntimeError(f"poller did not start (rc {self.proc.poll()})")
            time.sleep(0.1)

    def finish(self) -> list[dict]:
        self.stop.touch()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()  # mid-load; its attempts so far are already on disk
            self.proc.wait()
        self.stop.unlink(missing_ok=True)
        return read_jsonl(self.out)


def _import(env, a, work: Path, label: str, *, force=False, inject="none", arm=None, kill_delay=None) -> dict:
    """Run one import child; with kill_delay, SIGKILL it that many seconds after its parity gate
    starts. Returns the child's result, or what its events show when it was killed."""
    out, events = work / f"{label}.import.json", work / f"{label}.events.jsonl"
    args = [
        "--arm",
        arm or a.arm,
        "--archive",
        a.archive,
        "--bucket",
        str(a.bucket),
        "--model",
        a.model,
        "--inject",
        inject,
        "--out",
        str(out),
        "--events",
        str(events),
    ] + (["--force"] if force else [])
    proc = subprocess.Popen(_child("_import", *args), env=env)
    killed = None
    if kill_delay is not None:
        while proc.poll() is None:
            gate = [e for e in read_jsonl(events) if e["msg"].startswith("parity gate for imported")]
            if gate:
                time.sleep(max(0.0, gate[0]["t"] + kill_delay - time.time()))
                if proc.poll() is None:
                    os.kill(proc.pid, signal.SIGKILL)
                    killed = {"at": time.time(), "after_gate_start_s": time.time() - gate[0]["t"]}
                break
            time.sleep(0.2)
    proc.wait()
    res = _read(out) or {"ok": None, "error": {"type": "NoResult", "message": f"rc {proc.returncode}"}}
    res["returncode"], res["killed"], res["events"] = proc.returncode, killed, read_jsonl(events)
    return res


def _inspect(env, a, work: Path, label: str) -> dict:
    out = work / f"{label}.inspect.json"
    _run_child("_inspect", env, "--model", a.model, "--bucket", str(a.bucket), "--out", str(out))
    return _read(out) or {}


def _reader(env, a, work: Path, label: str) -> tuple[dict, dict]:
    out = work / f"{label}.reader.json"
    before = e5rt_names()
    _run_child("_reader", env, "--model", a.model, "--bucket", str(a.bucket), "--out", str(out))
    return _read(out) or {"ok": None}, e5rt_diff(before, e5rt_names())


def cmd_step(a) -> int:
    out_dir = Path(a.out_dir)
    work = out_dir / "steps"
    work.mkdir(parents=True, exist_ok=True)
    ledger = out_dir / "created-paths.txt"
    cache = _new_cache(ledger)
    env = dict(os.environ, LAYA_APPLE_CACHE=str(cache), HF_HUB_OFFLINE="1")
    rec: dict = {
        "label": a.label,
        "arm": a.arm,
        "bucket": a.bucket,
        "inject": a.inject,
        "model": a.model,
        "archive": Path(a.archive).name,
        "poller_enabled": a.poller,
        "started": now(),
    }
    rec["machine_before"] = machine_snapshot()
    try:
        if a.inject == "force-replace-probe-fail":
            rec["setup"] = _import(env, a, work, a.label + ".setup", arm="final-path")
            rec["before"] = _inspect(env, a, work, a.label + ".before")
        if a.inject == "sigkill":
            e5 = e5rt_names()
            rec["import"] = _import(env, a, work, a.label, kill_delay=a.kill_delay)
            rec["e5rt_import"] = e5rt_diff(e5, e5rt_names())
            rec["post"] = _inspect(env, a, work, a.label + ".post")
            rec["reader"], rec["e5rt_reader"] = _reader(env, a, work, a.label)
            rec["reimport"] = _import(env, a, work, a.label + ".reimport")
            rec["reimport_post"] = _inspect(env, a, work, a.label + ".reimport-post")
        else:
            e5 = e5rt_names()
            poller = Poller(env, a.model, a.bucket, work, a.label) if a.poller else None
            try:
                rec["import"] = _import(
                    env, a, work, a.label, force=a.inject == "force-replace-probe-fail", inject=a.inject
                )
            finally:
                rec["poller"] = poller.finish() if poller else None
            rec["e5rt_import"] = e5rt_diff(e5, e5rt_names())
            rec["post"] = _inspect(env, a, work, a.label + ".post")
            if a.inject == "none" and rec["import"].get("ok"):
                rec["reader"], rec["e5rt_reader"] = _reader(env, a, work, a.label)
    finally:
        rec["machine_after"] = machine_snapshot()
        rec["finished"] = now()
        rec["cache_removed"] = _remove_tmp(cache)
        write_json(out_dir / f"{a.label}.json", rec)
    if a.inject == "none" and not rec.get("import", {}).get("ok"):
        print(f"STOP: {a.label} import failed: {rec.get('import', {}).get('error')}", file=sys.stderr)
        return 3
    return 0


def cmd_e5rt_snapshot(a) -> int:
    write_json(Path(a.out), {"at": now(), "names": e5rt_names()})
    return 0


def cleanup(
    before: list[str], ledger_paths: list[str], *, caches: Path | None = None, tmp_root=None, dry_run=False
) -> dict:
    """Delete e5rt entries new since `before` (each checked with safe_e5rt_target) and the temp
    directories in the ledger (each checked with safe_tmp_target). Everything else is reported."""
    new = sorted(set(e5rt_names(caches)) - set(before))
    report = {"e5rt_removed": [], "e5rt_skipped": [], "tmp_removed": [], "tmp_skipped": [], "dry_run": dry_run}
    for name in new:
        target = safe_e5rt_target(name, caches)
        if target is None:
            report["e5rt_skipped"].append(name)
            continue
        size = entry_size(target)
        if not dry_run:
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        report["e5rt_removed"].append({"name": name, **size})
    for p in ledger_paths:
        if not p.strip():
            continue
        target = safe_tmp_target(p.strip(), tmp_root)
        if target is None:
            if Path(p.strip()).exists():
                report["tmp_skipped"].append(p.strip())
            continue
        if not dry_run:
            shutil.rmtree(target)
        report["tmp_removed"].append(str(target))
    return report


def cmd_cleanup(a) -> int:
    before = (_read(Path(a.before)) or {}).get("names")
    if before is None:
        print("no run-start e5rt listing: nothing removed from the e5rt cache", file=sys.stderr)
        return 2
    ledger = Path(a.ledger).read_text().splitlines() if Path(a.ledger).exists() else []
    report = cleanup(before, ledger, dry_run=a.dry_run)
    report["at"] = now()
    write_json(Path(a.out), report)
    print(
        f"cleanup: removed {len(report['e5rt_removed'])} e5rt entries "
        f"({sum(r['bytes'] for r in report['e5rt_removed']) / 1e9:.2f} GB), {len(report['tmp_removed'])} temp dirs; "
        f"skipped {len(report['e5rt_skipped'])} e5rt, {len(report['tmp_skipped'])} temp",
        file=sys.stderr,
    )
    return 0


def redactions(source_cache: str | None) -> list[tuple[str, str]]:
    """(local prefix, placeholder), longest first: no machine-specific absolute path is kept."""
    pairs = []
    for raw, tag in (
        (source_cache, "<cache>"),
        (os.environ.get(DATA_VOLUME_ENV), "<data>"),
        (tempfile.gettempdir(), "<tmp>"),
        (str(REPO), "<repo>"),
        (str(Path.home()), "~"),
    ):
        if raw:
            for p in {str(raw).rstrip("/"), os.path.realpath(raw)}:
                if len(p) > 1:
                    pairs.append((p, tag))
    return sorted(set(pairs), key=lambda x: -len(x[0]))


def redact_text(text: str, pairs: list[tuple[str, str]]) -> str:
    for p, tag in pairs:
        text = text.replace(p, tag)
    return text


def cmd_redact(a) -> int:
    pairs = redactions(a.source_cache)
    for f in sorted(Path(a.out_dir).rglob("*")):
        if f.is_file() and f.suffix in (".json", ".jsonl", ".txt", ".log"):
            text = f.read_text(errors="replace")
            new = redact_text(text, pairs)
            if new != text:
                f.write_text(new)
    return 0


def cmd_preflight(a) -> int:
    snap = machine_snapshot()
    write_json(Path(a.out_dir) / "preflight.json", snap)
    problems = [f"{m}: {gb:.0f} GB free < {MIN_FREE_GB}" for m, gb in free_gb(snap["df"]).items() if gb < MIN_FREE_GB]
    data = os.environ.get(DATA_VOLUME_ENV)
    if data and not Path(data).is_dir():
        problems.append(f"${DATA_VOLUME_ENV} is not a mounted directory")
    if "No thermal warning" not in snap["pmset_therm"]:
        problems.append("pmset -g therm reports a thermal warning (or none was readable)")
    if "AC Power" not in snap["pmset_batt"]:
        problems.append("not on mains power (pmset -g batt)")
    for p in problems:
        print(f"STOP: {p}", file=sys.stderr)
    return 2 if problems else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("step", help="one recorded import step (parent)")
    s.add_argument("--label", required=True)
    s.add_argument("--arm", choices=["baseline", "final-path"], required=True)
    s.add_argument("--archive", required=True)
    s.add_argument("--bucket", type=int, required=True)
    s.add_argument("--model", default=MODEL)
    s.add_argument("--poller", action="store_true")
    s.add_argument("--inject", choices=INJECTIONS, default="none")
    s.add_argument("--kill-delay", type=float, default=3.0, help="sigkill: seconds after the parity gate starts")
    s.add_argument("--out-dir", required=True)
    s.set_defaults(fn=cmd_step)

    s = sub.add_parser("e5rt-snapshot")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_e5rt_snapshot)

    s = sub.add_parser("cleanup")
    s.add_argument("--before", required=True)
    s.add_argument("--ledger", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_cleanup)

    s = sub.add_parser("redact", help="replace local absolute paths in OUT's text files with placeholders")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--source-cache")
    s.set_defaults(fn=cmd_redact)

    s = sub.add_parser("preflight")
    s.add_argument("--out-dir", required=True)
    s.set_defaults(fn=cmd_preflight)

    s = sub.add_parser("_import")
    for k in ("--arm", "--archive", "--model", "--out", "--events"):
        s.add_argument(k, required=True)
    s.add_argument("--bucket", type=int, required=True)
    s.add_argument("--inject", choices=INJECTIONS, default="none")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=child_import)

    for name, fn in (("_reader", child_reader), ("_inspect", child_inspect)):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--bucket", type=int, required=True)
        s.add_argument("--out", required=True)
        s.set_defaults(fn=fn)

    s = sub.add_parser("_poll")
    for k in ("--model", "--out", "--stop"):
        s.add_argument(k, required=True)
    s.add_argument("--bucket", type=int, required=True)
    s.set_defaults(fn=child_poll)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
