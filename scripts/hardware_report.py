"""Community hardware report: one command that measures this Mac for the results matrix.

    uv run python scripts/hardware_report.py [--models M ...] [--quick] [--offline] [--json PATH]
    uv run python scripts/hardware_report.py --render hardware-results/<dir>/bundle.json [--json PATH]

Records the environment automatically (SoC, memory, macOS, package versions, laya-apple
revision, pinned model revisions and weight hashes, whether a shipped routing profile
matches), then per model:

  a) MLX FP16 parity against the shipped goldens (as `laya-apple parity`);
  b) ANE parity, if validated artifacts for this machine are registered. Missing or
     rejected artifacts are recorded with the build command; nothing is built here;
  c) warm latency (forward and predict P50/P95/P99) via laya_apple.benchmark, measured
     in this process (unlike scripts/release_bench.py, which uses a fresh process per
     configuration);
  d) the device and routing reason `device="auto"` picks for a set of request shapes;
  e) a short closed-loop GPU-only vs GPU+ANE mix (scripts/bench_concurrency.py), when
     the ANE is available and auto routes to it.

Writes hardware-results/<soc>-macos<major>/bundle.json (everything, raw samples
included) and summary.md. Home-directory paths are replaced with `~`.

--json PATH also writes a compact machine-readable summary of the bundle (platform, forward
P50 per model, device, length and question count, parity per model) with a schema_version;
the schema is in docs/community-benchmarks.md. With --render it summarises an existing
bundle without measuring anything.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from argparse import Namespace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_FORMAT = "laya-apple-hardware-summary"
SUMMARY_SCHEMA_VERSION = 1  # bump on any change to the keys or meaning of build_summary()'s output
QUICK_MODELS = ["laya-typed-decisions"]
MLX_CONFIGS = [((64, 128, 512), 1), ((128,), 4)]  # (lengths, questions)
ROUTING_SHAPES = [(64, 1), (96, 1), (128, 1), (256, 1), (512, 1), (1024, 1), (64, 4), (128, 4)]


# ----------------------------------------------------------------------------- pure helpers


def _sh(*cmd) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def slug(soc: str, macos: str) -> str:
    """'Apple M4 Max', '26.6.2' -> 'apple-m4-max-macos26'."""
    base = re.sub(r"[^a-z0-9]+", "-", (soc or "unknown-soc").lower()).strip("-") or "unknown-soc"
    major = (macos or "").split(".")[0] or "unknown"
    return f"{base}-macos{major}"


def output_dir(root: Path, name: str, now: datetime) -> Path:
    """root/name, or root/name-<timestamp> if that already exists (never overwrite)."""
    path = root / name
    if path.exists():
        path = root / f"{name}-{now.strftime('%Y%m%d-%H%M%S')}"
    return path


def sanitize(obj, home: str | None = None, root: str | None = None, cache: str | None = None):
    """Replace the laya-apple cache directory with <cache>, the checkout root (if given) with
    . and the home directory with ~ in every string (keys included), recursively."""
    home = home or str(Path.home())
    if cache is None:
        from laya_apple.hub import cache_root

        cache = str(cache_root())
    if isinstance(obj, str):
        s = obj.replace(cache, "<cache>") if cache else obj
        return (s.replace(root, ".") if root else s).replace(home, "~")
    if isinstance(obj, dict):
        return {sanitize(k, home, root, cache): sanitize(v, home, root, cache) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v, home, root, cache) for v in obj]
    return obj


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def git_info(root: Path) -> dict | None:
    """Revision and dirty flag of the checkout, or None when not in a git checkout."""
    if not (root / ".git").exists():
        return None
    rev = _sh("git", "-C", str(root), "rev-parse", "HEAD")
    if not rev:
        return None
    status = _sh("git", "-C", str(root), "status", "--porcelain", "--untracked-files=no")
    return {"revision": rev, "dirty": bool(status)}


def collect_environment() -> dict:
    import laya_apple

    memsize = _sh("sysctl", "-n", "hw.memsize")
    source = Path(laya_apple.__file__).resolve().parent
    return {
        "soc": _sh("sysctl", "-n", "machdep.cpu.brand_string") or None,
        "hw_model": _sh("sysctl", "-n", "hw.model") or None,
        "memory_bytes": int(memsize) if memsize.isdigit() else None,
        "memory_gb": round(int(memsize) / 2**30) if memsize.isdigit() else None,
        "macos": _sh("sw_vers", "-productVersion") or None,
        "macos_build": _sh("sw_vers", "-buildVersion") or None,
        "python": platform.python_version(),
        "packages": {d: _version(d) for d in ("mlx", "coremltools", "numpy")},
        "laya_apple": {
            "version": laya_apple.__version__,
            "source": str(source),
            "git": git_info(ROOT) if source.parent == ROOT else None,
        },
    }


def _ok(section: dict | None) -> bool | None:
    """True if it ran and passed, False if it failed or errored, None if it did not run
    (skipped, or ANE artifacts unavailable)."""
    if not section or section.get("status") not in ("ok", "error"):
        return None
    return section.get("status") == "ok" and bool(section.get("passed"))


def matrix_cells(models: dict) -> dict:
    """The four matrix cells from per-model results.

    MLX / ANE / Heterogeneous: ✓ every model that ran passed, ✗ any failed or errored,
    untested if none ran. Auto: yes if auto routed any request to the ANE on any model."""

    def cell(values):
        ran = [v for v in values if v is not None]
        if not ran:
            return "untested"
        return "✓" if all(ran) else "✗"

    results = list(models.values())
    auto = [r.get("routing") or {} for r in results]
    return {
        "mlx": cell([_ok(r.get("mlx_parity")) for r in results]),
        "ane": cell([_ok(r.get("ane_parity")) for r in results]),
        "auto_uses_ane": "yes" if any(a.get("uses_ane") for a in auto) else "no",
        "heterogeneous": cell([_ok(r.get("heterogeneous")) for r in results]),
    }


def matrix_row(env: dict, models: dict) -> str:
    c = matrix_cells(models)
    label = f"{env.get('soc') or 'unknown SoC'} ({env.get('memory_gb')} GB, macOS {env.get('macos')})"
    return f"| {label} | {c['mlx']} | {c['ane']} | {c['auto_uses_ane']} | {c['heterogeneous']} |"


# ----------------------------------------------------------------------------- measurements


def _error(e: BaseException) -> dict:
    return {"status": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def _parity(laya, spec, device: str) -> dict:
    """The same evaluation `laya-apple parity` runs (laya_apple.cli.cmd_parity)."""
    from laya_apple.parity import evaluate

    backend = laya.ane if device == "ane" else laya.mlx
    summary = evaluate(
        spec.name,
        laya.config,
        backend.forward,
        precision="float16",
        max_len=max(backend.buckets) if device == "ane" else None,
        prepare=lambda s, q: laya.prepare(s, q).items,
    )
    if device == "ane":
        summary["artifact_revision"] = {str(b): m.artifact_sha256 for b, m in backend.manifests.items()}
    return {"status": "ok", **summary}


def _latency(laya, spec, device: str, configs, warmup: int, iters: int) -> list:
    from laya_apple.benchmark import run

    out = []
    for lengths, questions in configs:
        out += run(
            spec.name,
            device=device,
            lengths=[L for L in lengths if L <= spec.max_len],
            questions=questions,
            warmup=warmup,
            iters=iters,
            laya=laya,
        )
    return out


def _routing(laya, spec) -> dict:
    from laya_apple.workload import make_request

    shapes = [(L, q) for L, q in ROUTING_SHAPES if L <= spec.max_len]
    reqs = [(L, q, *make_request(laya.tokenizer, laya.config, L, n_questions=q)) for L, q in shapes]
    rows = []
    for L, q, state, qs in reqs:
        rt = laya.predict(context=state, questions=qs).runtime
        rows.append(
            {
                "length": L,
                "questions": q,
                "backend": rt.backend,
                "device": rt.device,
                "routing_reason": rt.routing_reason,
            }
        )
    info = laya.info()
    return {
        "status": "ok",
        "routing_profile": info["routing_profile"],
        "auto_ane": info["auto_ane"],
        "requests": rows,
        "uses_ane": any(r["device"] == "ane" for r in rows),
    }


def _load_bench_concurrency():
    spec = importlib.util.spec_from_file_location("bench_concurrency", ROOT / "scripts" / "bench_concurrency.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _heterogeneous(spec, seconds: float, local_files_only: bool) -> dict:
    """Part A of scripts/bench_concurrency.py, one cycle: solo, GPU+ANE and GPU-only windows."""
    from laya_apple import Laya, LayaAppleError
    from laya_apple.workload import make_request

    bc = _load_bench_concurrency()
    short, long_ = 128, min(1024, spec.max_len)
    kw = dict(execution="workers", local_files_only=local_files_only)
    lays = []
    try:
        laya_auto = Laya.from_pretrained(spec.name, device="auto", **kw)
        lays.append(laya_auto)
        laya_gpu = Laya.from_pretrained(spec.name, device="gpu", **kw)
        lays.append(laya_gpu)
        ref_gpu = Laya.from_pretrained(spec.name, device="gpu", local_files_only=local_files_only)
        lays.append(ref_gpu)
        ref_ane = Laya.from_pretrained(spec.name, device="ane", local_files_only=local_files_only)
        lays.append(ref_ane)
        reqs = {}
        for name, L in (("short", short), ("long", long_)):
            state, qs = make_request(laya_auto.tokenizer, laya_auto.config, L, n_questions=1)
            reqs[name] = {"state": state, "questions": qs, "length": L, "q": 1}
        refs = {}
        for name, r in reqs.items():
            refs[name] = {"gpu": ref_gpu.predict(context=r["state"], questions=r["questions"]).answers}
            try:
                refs[name]["ane"] = ref_ane.predict(context=r["state"], questions=r["questions"]).answers
            except LayaAppleError:
                refs[name]["ane"] = None
        for laya in (laya_auto, laya_gpu):  # warm both worker paths
            for r in reqs.values():
                for _ in range(5):
                    laya.predict(context=r["state"], questions=r["questions"])
        part = bc.part_a(Namespace(seconds=seconds, cycles=1), laya_auto, laya_gpu, reqs, refs)
    finally:
        for laya in lays:
            laya.close()
    gate = part["gate"]
    return {
        "status": "ok",
        "method": {"source": "scripts/bench_concurrency.py part_a", "seconds": seconds, "cycles": 1},
        "short_tokens": short,
        "long_tokens": long_,
        "aggregate_gpu_only_req_s": gate["aggregate_gpu_only_req_s"],
        "aggregate_gpu_ane_req_s": gate["aggregate_hetero_req_s"],
        "ratio": gate["ratio"],
        "mismatches": gate["mismatches"],
        # Here "passed" means correct and faster than GPU-only; the 2.5x release gate is
        # specific to the tested profile and kept below as release_gate.
        "passed": gate["mismatches"] == 0 and gate["ratio"] > 1.0,
        "release_gate": gate,
        "part_a": part,
    }


def measure_model(name: str, *, quick: bool, local_files_only: bool, log) -> dict:
    from laya_apple import ArtifactError, BackendUnavailableError, Laya
    from laya_apple.hub import checkpoint_path, verify_weights
    from laya_apple.registry import resolve

    spec = resolve(name)
    warmup, iters = (3, 20) if quick else (5, 30)
    out: dict = {
        "model": spec.name,
        "repo": spec.repo,
        "revision": spec.revision,
        "weights_sha256_pinned": spec.weights_sha256,
        "ane_buckets": list(spec.ane_buckets),
        "shipped_auto_ane_buckets": list(spec.auto_ane_buckets),
    }
    try:
        out["weights_sha256_verified"] = verify_weights(spec, checkpoint_path(spec, local_files_only=local_files_only))
    except Exception as e:
        out["weights"] = _error(e)
        return out

    log(f"{spec.name}: MLX parity and latency")
    try:
        with Laya.from_pretrained(spec.name, device="gpu", local_files_only=local_files_only) as gpu:
            out["mlx_parity"] = _parity(gpu, spec, "gpu")
            out["mlx_latency"] = _latency(gpu, spec, "gpu", MLX_CONFIGS, warmup, iters)
    except Exception as e:
        out["mlx_latency" if "mlx_parity" in out else "mlx_parity"] = _error(e)

    log(f"{spec.name}: ANE parity and latency")
    try:
        ane = Laya.from_pretrained(spec.name, device="ane", local_files_only=local_files_only)
    except (ArtifactError, BackendUnavailableError) as e:
        ane = None
        out["ane_parity"] = {
            "status": "unavailable",
            "reason": f"{type(e).__name__}: {e}",
            "build_command": f"uv run laya-apple artifacts build {spec.name}",
        }
    except Exception as e:
        ane = None
        out["ane_parity"] = _error(e)
    if ane is not None:
        try:
            out["ane_parity"] = _parity(ane, spec, "ane")
            out["ane_latency"] = _latency(ane, spec, "ane", [(tuple(ane.ane.buckets), 1)], warmup, iters)
        except Exception as e:
            out["ane_latency" if "ane_parity" in out else "ane_parity"] = _error(e)
        finally:
            ane.close()

    log(f"{spec.name}: auto routing")
    try:
        with Laya.from_pretrained(spec.name, device="auto", local_files_only=local_files_only) as auto:
            out["routing"] = _routing(auto, spec)
    except Exception as e:
        out["routing"] = _error(e)

    ane_ok = _ok(out.get("ane_parity"))
    if not ane_ok:
        reason = "no validated ANE artifacts on this machine" if ane_ok is None else "ANE parity did not pass"
        out["heterogeneous"] = {"status": "skipped", "reason": reason}
    elif not (out.get("routing") or {}).get("uses_ane"):
        reasons = sorted({r["routing_reason"] for r in out["routing"].get("requests", [])})
        out["heterogeneous"] = {
            "status": "skipped",
            "reason": f"device='auto' does not route to the ANE here ({', '.join(reasons)}); "
            f"`uv run laya-apple calibrate {spec.name}` writes a local profile that enables it",
        }
    else:
        seconds = 5.0 if quick else 10.0
        log(f"{spec.name}: heterogeneous closed-loop mix, 4 windows x {seconds:g} s")
        try:
            out["heterogeneous"] = _heterogeneous(spec, seconds, local_files_only)
        except Exception as e:
            out["heterogeneous"] = _error(e)
    return out


# ----------------------------------------------------------------------------- report


def _fmt_latency(records: list) -> list[str]:
    lines = [
        "Warm latency, in-process:",
        "",
        "| L | q | device | forward P50/P95/P99 ms | predict P50/P95/P99 ms |",
        "|---:|---:|---|---|---|",
    ]
    for r in records:
        if r.get("status") != "ok":
            lines.append(
                f"| {r.get('length')} | {r.get('questions')} | {r.get('device_requested')} | {r.get('status')} | |"
            )
            continue
        f, p = r["forward"], r["predict"]
        lines.append(
            f"| {r['length']} | {r['questions']} | {r['device']} | "
            f"{f['p50_ms']:.2f} / {f['p95_ms']:.2f} / {f['p99_ms']:.2f} | "
            f"{p['p50_ms']:.2f} / {p['p95_ms']:.2f} / {p['p99_ms']:.2f} |"
        )
    return lines


def _fmt_status(section: dict | None) -> str:
    if not section:
        return "not run"
    if section.get("status") == "ok":
        return "passed" if section.get("passed") else "FAILED"
    return f"{section.get('status')}: {section.get('reason') or section.get('error')}"


def render_summary(bundle: dict) -> str:
    env, cfg = bundle["environment"], bundle["configuration"]
    git = env["laya_apple"].get("git") or {}
    lines = [
        f"# Hardware report: {env['soc']}, macOS {env['macos']}",
        "",
        "Matrix row (docs/community-benchmarks.md):",
        "",
        "| Mac | MLX | ANE | Auto uses ANE | Heterogeneous |",
        "|---|---|---|---|---|",
        bundle["matrix_row"],
        "",
        "## Environment",
        "",
        "| | |",
        "|---|---|",
        f"| SoC | {env['soc']} ({env['hw_model']}) |",
        f"| Memory | {env['memory_gb']} GB |",
        f"| macOS | {env['macos']} ({env['macos_build']}) |",
        f"| Python | {env['python']} |",
        *(f"| {k} | {v} |" for k, v in env["packages"].items()),
        f"| laya-apple | {env['laya_apple']['version']}"
        + (f" @ {git['revision'][:12]}{' (dirty)' if git['dirty'] else ''}" if git else "")
        + " |",
        f"| Shipped routing profile matches | {'yes' if bundle['platform']['shipped_profile_matches'] else 'no'} |",
        "",
        f"Configuration: {'quick' if cfg['quick'] else 'full'}, warmup {cfg['warmup']}, iters {cfg['iters']}, "
        f"latency measured {cfg['latency_process']}. Wall time {bundle['wall_time_s']:.0f} s.",
    ]
    for name, m in bundle["models"].items():
        lines += [
            "",
            f"## {name}",
            "",
            f"Revision `{m['revision']}`, weights sha256 `{m['weights_sha256_pinned']}`.",
            "",
        ]
        if "weights" in m:
            lines.append(f"- weights: {_fmt_status(m['weights'])}")
            continue
        lines += [
            f"- MLX FP16 parity: {_fmt_status(m.get('mlx_parity'))}",
            f"- ANE parity: {_fmt_status(m.get('ane_parity'))}",
        ]
        if (m.get("ane_parity") or {}).get("status") == "unavailable":
            lines.append(f"  - to build and validate here: `{m['ane_parity']['build_command']}`")
        routing = m.get("routing") or {}
        if routing.get("status") == "ok":
            lines.append(f"- auto routing (profile: {routing['routing_profile']}):")
            lines += [
                f"  - L{r['length']} q{r['questions']}: {r['device']} ({r['routing_reason']})"
                for r in routing["requests"]
            ]
        else:
            lines.append(f"- auto routing: {_fmt_status(routing)}")
        het = m.get("heterogeneous") or {}
        if het.get("status") == "ok":
            lines.append(
                f"- heterogeneous: GPU-only {het['aggregate_gpu_only_req_s']:.1f} req/s, "
                f"GPU+ANE {het['aggregate_gpu_ane_req_s']:.1f} req/s ({het['ratio']:.2f}x), "
                f"{het['mismatches']} mismatches"
            )
        else:
            lines.append(f"- heterogeneous: {_fmt_status(het)}")
        latency = []
        for key in ("mlx_latency", "ane_latency"):
            if isinstance(m.get(key), list):
                latency += m[key]
            elif m.get(key):
                lines.append(f"- {key}: {_fmt_status(m[key])}")
        if latency:
            lines += ["", *_fmt_latency(latency)]
    return "\n".join(lines) + "\n"


def build_summary(bundle: dict) -> dict:
    """The compact machine-readable summary of a bundle (docs/community-benchmarks.md).

    Everything is copied from `bundle`, so the numbers are the bundle's own. A parity entry is
    true if the gate passed, false if it failed or errored and null if it did not run; latency
    steps that were skipped or errored have no entry."""
    env = bundle["environment"]
    models = {}
    for name, m in bundle["models"].items():
        forward = []
        for key in ("mlx_latency", "ane_latency"):
            records = m.get(key)
            for r in records if isinstance(records, list) else []:
                if r.get("status") == "ok":
                    forward.append(
                        {
                            "device": r["device"],
                            "length": r["length"],
                            "questions": r["questions"],
                            "p50_ms": r["forward"]["p50_ms"],
                        }
                    )
        models[name] = {
            "revision": m["revision"],
            "parity": {"mlx": _ok(m.get("mlx_parity")), "ane": _ok(m.get("ane_parity"))},
            "forward_p50_ms": forward,
        }
    return {
        "format": SUMMARY_FORMAT,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "laya_apple": env["laya_apple"]["version"],
        "started_at": bundle["started_at"],
        "quick": bundle["configuration"]["quick"],
        "platform": dict(bundle["platform"]["profile"]),
        "hardware": {"hw_model": env["hw_model"], "memory_gb": env["memory_gb"]},
        "models": models,
    }


def json_path_problem(path: Path, render: Path | None) -> str | None:
    """Why `--json path` cannot be written (None if it can). `render` is the bundle `--render` reads:
    writing the summary over it would destroy the bundle."""
    if path.is_dir():
        return "is a directory"
    if not path.parent.is_dir():
        return "its directory does not exist"
    if render is not None and path.exists() and render.exists() and path.samefile(render):
        return "is the --render bundle, which the summary would overwrite"
    if not os.access(path if path.exists() else path.parent, os.W_OK):
        return "is not writable"
    return None


def write_summary(bundle: dict, path: Path) -> None:
    path.write_text(json.dumps(build_summary(bundle), indent=1, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", help="default: laya-typed-decisions with --quick, every model otherwise")
    ap.add_argument("--quick", action="store_true", help="one model, fewer iterations, shorter mix")
    ap.add_argument("--offline", action="store_true", help="never touch the network (local_files_only)")
    ap.add_argument("--out-root", type=Path, default=ROOT / "hardware-results")
    ap.add_argument("--render", type=Path, metavar="BUNDLE", help="print summary.md for an existing bundle.json")
    ap.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="also write the compact JSON summary here (overwrites PATH; with --render, from that bundle)",
    )
    a = ap.parse_args(argv)
    if a.json:  # fail now, not after a 10-25 minute run
        problem = json_path_problem(a.json, a.render)
        if problem:
            ap.error(f"--json {a.json}: {problem}")
    if a.render:
        bundle = json.loads(a.render.read_text())
        print(render_summary(bundle), end="")
        if a.json:
            write_summary(bundle, a.json)
        return 0

    from laya_apple.artifacts import platform_profile
    from laya_apple.profiles import local_profile_path, shipped_profile_matches
    from laya_apple.registry import models

    t0 = time.perf_counter()
    started = datetime.now(timezone.utc)
    names = a.models or (QUICK_MODELS if a.quick else list(models()))
    env = collect_environment()
    warmup, iters = (3, 20) if a.quick else (5, 30)
    bundle = {
        "format": "laya-apple-hardware-report",
        "format_version": 1,
        "started_at": started.isoformat(),
        "environment": env,
        "platform": {
            "profile": platform_profile(),
            "shipped_profile_matches": shipped_profile_matches(),
            "local_profile": str(local_profile_path()) if local_profile_path().exists() else None,
        },
        "configuration": {
            "models": names,
            "quick": a.quick,
            "offline": a.offline,
            "warmup": warmup,
            "iters": iters,
            "mlx_latency": [{"lengths": list(ls), "questions": q} for ls, q in MLX_CONFIGS],
            "ane_latency": "every registered ANE bucket, 1 question",
            "routing_shapes": [{"length": L, "questions": q} for L, q in ROUTING_SHAPES],
            "heterogeneous_seconds": 5.0 if a.quick else 10.0,
            "latency_process": "in-process (one process for every configuration)",
        },
        "models": {},
    }

    def log(msg):
        print(f"[{time.perf_counter() - t0:7.1f} s] {msg}", flush=True)

    for name in names:
        bundle["models"][name] = measure_model(name, quick=a.quick, local_files_only=a.offline, log=log)
    bundle["matrix"] = matrix_cells(bundle["models"])
    bundle["matrix_row"] = matrix_row(env, bundle["models"])
    bundle["wall_time_s"] = time.perf_counter() - t0
    bundle = sanitize(bundle, root=str(ROOT))

    out = output_dir(a.out_root, slug(env["soc"], env["macos"]), started)
    out.mkdir(parents=True)
    (out / "bundle.json").write_text(json.dumps(bundle, indent=1, ensure_ascii=False) + "\n")
    (out / "summary.md").write_text(render_summary(bundle))
    shown = sanitize(str(out.resolve().relative_to(ROOT)) if out.resolve().is_relative_to(ROOT) else str(out))
    print()
    print(bundle["matrix_row"])
    print(f"\nwrote {shown}/bundle.json and {shown}/summary.md ({bundle['wall_time_s']:.0f} s)")
    if a.json:
        write_summary(bundle, a.json)
        print(f"wrote {a.json} (schema_version {SUMMARY_SCHEMA_VERSION})")
    print(f"open a PR adding this directory: {shown}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
