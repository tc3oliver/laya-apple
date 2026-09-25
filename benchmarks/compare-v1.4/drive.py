"""Same-machine comparison of Apple-silicon Laya runtimes: setup, smoke, campaign, teardown.

    uv run python benchmarks/compare-v1.4/drive.py plan                  # matrix + estimated runtime
    uv run python benchmarks/compare-v1.4/drive.py setup                 # envs, sources, model files
    uv run python benchmarks/compare-v1.4/drive.py smoke                 # 1 golden case + 1 window each
    uv run python benchmarks/compare-v1.4/drive.py run --run-id ID       # the campaign -> raw/ID/
    uv run python benchmarks/compare-v1.4/drive.py teardown [--models]   # remove what setup created

Method: README.md in this directory (written before any campaign run). This driver runs in
the laya-apple environment and needs only the standard library plus laya_apple.registry.
Each runtime runs in its own uv environment under --base (default
~/Developer/scratch/compare-v1.4/<runtime>), through adapter.py, invoked with that
environment's python. laya-apple itself runs in this checkout's `.venv`.

Machine-specific paths (environments, snapshots, converted models) go to a config file under
--base, never into raw/. raw/ holds measurements, versions and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = json.loads((HERE / "runtimes.json").read_text())
SHAPES = HERE / "shapes.json"
ADAPTER = HERE / "adapter.py"
RAW = HERE / "raw"
MARKER = ".created-by-compare-v1.4"
CODELOAD = "https://codeload." + "github.com"
HF_API = "https://huggingface.co/api/models"
LATENCY_SHAPES = ("short", "long", "mixed3", "uniform3", "batch16")
SMOKE_CASE = {"laya": "lang-en", "laya-multilingual": "lang-en", "laya-typed-decisions": "lang-en"}
LAYA_FAST_PINNED_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"

# Planning estimates (seconds) for `plan`: process start + load, from the smoke run on the
# reference machine rounded up (laya-fast "fast" and laya-coreml "ane" compile Core ML on
# first load; a cold compile of all six laya-fast buckets can take several minutes).
EST_LOAD_S = {
    ("laya-apple", "auto"): 10,
    ("laya-apple", "gpu"): 5,
    ("laya-mlx", "fp16"): 5,
    ("laya-mlx", "fp16-opt"): 5,
    ("laya-coreml", "default"): 15,
    ("laya-coreml", "ane"): 35,
    ("laya-fast", "mlx"): 5,
    ("laya-fast", "fast"): 120,
    ("upstream", "cpu-fp32"): 20,
}
EST_PARITY_S_PER_SET = 60  # laya-coreml default answers mixed and long calls in 0.5-3 s each
EST_SETUP_S = {"envs": 300, "downloads": 600, "laya_fast_convert": 120, "laya_fast_ane_export_per_bucket": 180}


# ------------------------------------------------------------------------------ paths
def checkpoints() -> dict:
    sys.path.insert(0, str(ROOT))
    from laya_apple.registry import models

    return {n: {"repo": s.repo, "revision": s.revision} for n, s in models().items()}


def env_python(base: Path, runtime: str) -> Path:
    if PINS["runtimes"][runtime].get("env") == "repo":
        return ROOT / ".venv/bin/python"
    return base / runtime / "bin/python"


def source_dir(base: Path, runtime: str) -> Path | None:
    s = PINS["runtimes"][runtime].get("source")
    if not s:
        return None
    return base / "src" / f"{s['repo'].split('/')[1]}-{s['commit']}"


def snapshots() -> dict:
    """Local snapshot directories of the pinned checkpoints, resolved as laya-apple resolves them."""
    sys.path.insert(0, str(ROOT))
    from laya_apple.hub import checkpoint_path
    from laya_apple.registry import models

    return {n: str(checkpoint_path(s, local_files_only=True)) for n, s in models().items()}


def adapter_config(a) -> Path:
    base, mroot = Path(a.base), Path(a.model_root)
    cfg = {
        "checkpoints": checkpoints(),
        "snapshots": snapshots(),
        "runtimes": {
            "laya-coreml": {"bundles": PINS["runtimes"]["laya-coreml"]["bundles"]},
            "laya-fast": {
                "source_dir": str(source_dir(base, "laya-fast")),
                "converted_dir": str(mroot / "laya-fast/converted-fp16"),
                "ane_dir": str(mroot / "laya-fast/ane"),
            },
        },
    }
    p = base / "config.json"
    p.write_text(json.dumps(cfg, indent=1) + "\n")
    return p


def child_env(offline: bool = True) -> dict:
    env = dict(os.environ)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def sh(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def configs(runtimes=None, models=None, include_reference=False):
    """[(runtime, variant, model)] in a fixed order."""
    out = []
    for rt, pin in PINS["runtimes"].items():
        if rt == "upstream" and not include_reference:
            continue
        if runtimes and rt not in runtimes:
            continue
        for var, ms in pin["variants"].items():
            for m in ms:
                if models and m not in models:
                    continue
                out.append((rt, var, m))
    return out


# ------------------------------------------------------------------------------ setup
def fetch_source(base: Path, runtime: str):
    s = PINS["runtimes"][runtime]["source"]
    dest = source_dir(base, runtime)
    if dest.exists():
        return dest
    data = urllib.request.urlopen(f"{CODELOAD}/{s['repo']}/tar.gz/{s['commit']}", timeout=120).read()
    (base / "src").mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(base / "src", filter="data")
    print(f"fetched {s['repo']}@{s['commit'][:12]} ({len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()[:16]})")
    return dest


def hf_tree(repo: str, rev: str) -> dict:
    out = {}
    for d in ("", "encoder", "tokenizer"):
        with urllib.request.urlopen(f"{HF_API}/{repo}/tree/{rev}/{d}", timeout=60) as r:
            for f in json.load(r):
                if f["type"] == "file":
                    out[f["path"]] = (f.get("lfs") or {}).get("oid") or f["oid"]
    return out


def check_laya_fast_checkpoint() -> dict:
    """laya-fast pins convaiinnovations/laya@1c5edc17; laya-apple pins another revision."""
    pin = checkpoints()["laya"]
    a, b = hf_tree(pin["repo"], LAYA_FAST_PINNED_REVISION), hf_tree(pin["repo"], pin["revision"])
    differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    used_differ = [k for k in differ if k != "README.md"]
    return {
        "laya_fast_revision": LAYA_FAST_PINNED_REVISION,
        "pinned_revision": pin["revision"],
        "differing_files": differ,
        "same_inputs": not used_differ,
    }


def setup(a):
    base, mroot = Path(a.base), Path(a.model_root)
    base.mkdir(parents=True, exist_ok=True)
    (base / MARKER).write_text("created by benchmarks/compare-v1.4/drive.py setup\n")
    want = a.runtimes or [r for r in PINS["runtimes"]]
    py = PINS["python"]
    for rt in want:
        pin = PINS["runtimes"][rt]
        if pin.get("source"):
            fetch_source(base, rt)
        if pin.get("env") == "repo":
            sh(["uv", "sync", "--inexact", "--extra", "ane"], cwd=ROOT)
            continue
        env = base / rt
        if not env.exists():
            sh(["uv", "venv", "--python", py, env])
        spec = list(pin.get("pip", []))
        for r in pin.get("pip_from_source", []):
            flag, path = r.split(" ", 1)
            spec += [flag, source_dir(base, rt) / path]
        sh(["uv", "pip", "install", "--python", env / "bin/python", *spec])
        freeze = subprocess.run(
            ["uv", "pip", "freeze", "--python", env / "bin/python"], capture_output=True, text=True, check=True
        )
        (base / f"{rt}.freeze.txt").write_text(freeze.stdout)
    cfg = adapter_config(a)
    if "laya-coreml" in want:
        for var, bundles in PINS["runtimes"]["laya-coreml"]["bundles"].items():
            for m, b in bundles.items():
                if a.models and m not in a.models:
                    continue
                if a.variants and var not in a.variants:
                    continue
                code = f"from huggingface_hub import snapshot_download as s; print(s({b['repo']!r}, revision={b['revision']!r}))"
                sh([env_python(base, "laya-coreml"), "-c", code], env=child_env(offline=False))
    if "laya-fast" in want:
        chk = check_laya_fast_checkpoint()
        (base / "laya-fast.checkpoint-check.json").write_text(json.dumps(chk, indent=1) + "\n")
        print(f"laya-fast checkpoint check: {chk}")
        if not chk["same_inputs"]:
            raise SystemExit(
                "laya-fast's pinned checkpoint differs from laya-apple's pin in model files; stop and record it"
            )
        src = source_dir(base, "laya-fast")
        conv = mroot / "laya-fast/converted-fp16"
        (mroot / "laya-fast").mkdir(parents=True, exist_ok=True)
        (mroot / MARKER).write_text("created by benchmarks/compare-v1.4/drive.py setup\n")
        if not conv.exists():
            snap = snapshots()["laya"]
            sh(
                [
                    env_python(base, "laya-fast"),
                    src / "convert.py",
                    "--source",
                    snap,
                    "--output",
                    conv,
                    "--dtype",
                    "float16",
                ],
                cwd=src,
                env=child_env(),
            )
        buckets = a.ane_buckets if a.ane_buckets is not None else PINS["runtimes"]["laya-fast"]["ane_buckets"]
        for L in buckets:
            out = mroot / f"laya-fast/ane/body{L}"
            if (out / "model.mlmodelc").exists() or (out / "model.mlpackage").exists():
                continue
            t = time.time()
            sh(
                [
                    env_python(base, "laya-fast"),
                    src / "ane/export.py",
                    "--model",
                    conv,
                    "--length",
                    L,
                    "--output",
                    out,
                    "--force",
                ],
                cwd=src,
                env=child_env(),
            )
            print(f"exported laya-fast ANE body L{L} in {time.time() - t:.0f} s")
    print(f"setup done; adapter config {cfg}")


# ------------------------------------------------------------------------------ fixtures
def fixture_sets(a, run_dir: Path) -> dict:
    """{(model, set name): path}. Written under --base (third-party text is not copied into raw/)."""
    sys.path.insert(0, str(ROOT))
    from laya_apple.parity import load_goldens

    base = Path(a.base)
    fdir = base / "fixtures"
    fdir.mkdir(parents=True, exist_ok=True)
    out = {}

    def put(model, name, cases):
        p = fdir / f"{name}-{model}.json"
        p.write_text(json.dumps({"name": name, "cases": cases}, ensure_ascii=False) + "\n")
        out[(model, name)] = p

    shapes = json.loads(SHAPES.read_text())["shapes"]
    ref = source_dir(base, "laya-coreml") / "benchmarks/results/reference.json"
    published = json.loads(ref.read_text())["models"] if ref.exists() else {}
    fast = None
    for m in checkpoints():
        g = load_goldens(m)["cases"]
        put(m, "goldens", [{k: c[k] for k in ("name", "state", "questions")} for c in g])
        put(m, "shapes", [{"name": k, "state": v["state"], "questions": v["questions"]} for k, v in shapes.items()])
        if m in published:
            put(
                m,
                "published-laya-coreml",
                [{k: c[k] for k in ("name", "state", "questions")} for c in published[m]["cases"]],
            )
    if source_dir(base, "laya-fast").exists():
        p = fdir / "laya-fast-dump.json"
        subprocess.run(
            [
                env_python(base, "laya-fast"),
                ADAPTER,
                "fixtures",
                "--runtime",
                "laya-fast",
                "--config",
                base / "config.json",
                "--out",
                p,
            ],
            check=True,
            env=child_env(),
            cwd=base,
        )
        fast = json.loads(p.read_text())["cases"]
        put("laya", "published-laya-fast", fast)
    record = {f"{m}/{n}": hashlib.sha256(p.read_bytes()).hexdigest() for (m, n), p in out.items()}
    (run_dir / "fixtures.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return out


# ------------------------------------------------------------------------------ adapter calls
def adapter(a, task, rt, var, model, *args, timeout=None):
    base = Path(a.base)
    cmd = [
        env_python(base, rt),
        ADAPTER,
        task,
        "--runtime",
        rt,
        "--variant",
        var,
        "--model",
        model,
        "--config",
        base / "config.json",
        *args,
    ]
    print("+ " + " ".join(str(c) for c in cmd[1:]), flush=True)
    return subprocess.run([str(c) for c in cmd], env=child_env(), cwd=base, timeout=timeout).returncode


def run_parity(a, rt, var, m, sets, out_dir: Path):
    rc = {}
    for (fm, name), path in sets.items():
        if fm != m:
            continue
        out = out_dir / f"{rt}-{var}-{m}-{name}.json"
        rc[name] = adapter(a, "parity", rt, var, m, "--fixtures", path, "--out", out)
    return rc


# ------------------------------------------------------------------------------ smoke
def smoke(a):
    base = Path(a.base)
    cfg = adapter_config(a)
    sdir = base / "smoke"
    sdir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import analyze

    from laya_apple.parity import load_goldens

    report = []
    todo = [c for c in configs(a.runtimes, include_reference=True)]
    first = {}
    for rt, var, m in todo:  # one model per (runtime, variant): English unless the variant lacks it
        first.setdefault((rt, var), m)
    for (rt, var), m in first.items():
        case = next(c for c in load_goldens(m)["cases"] if c["name"] == SMOKE_CASE[m])
        fx = sdir / f"fixture-{m}.json"
        fx.write_text(
            json.dumps(
                {"name": "goldens-smoke", "cases": [{k: case[k] for k in ("name", "state", "questions")}]},
                ensure_ascii=False,
            )
        )
        pout, lout = sdir / f"{rt}-{var}-{m}-parity.json", sdir / f"{rt}-{var}-{m}-latency.json"
        row = {"runtime": rt, "variant": var, "model": m}
        rc = adapter(a, "parity", rt, var, m, "--fixtures", fx, "--out", pout, timeout=3600)
        if rc or not pout.exists():
            row["parity"] = f"adapter exit {rc}"
        else:
            p = json.loads(pout.read_text())
            row["load_s"] = p["load_s"]
            row["packages"] = p["runtime_info"].get("packages")
            c = p["cases"][0]
            if "answers" not in c:
                row["parity"] = c.get("error")
            else:
                rows = analyze.compare_answers(analyze.golden_reference(m)[case["name"]], c["answers"])
                g = analyze.gate(rows, repeat_identical=p.get("repeat_identical"))
                row["parity"] = {
                    k: g[k] for k in ("rows", "prob_max_abs", "act_prob_max_abs", "hard_mismatches", "passed")
                }
                row["parity"]["near_tie_flips"] = len(g["near_tie_flips"])
        if rt != "upstream":
            rc = adapter(
                a,
                "latency",
                rt,
                var,
                m,
                "--shapes",
                SHAPES,
                "--order",
                "short",
                "--warmup",
                "5",
                "--window-s",
                str(a.window_s),
                "--windows",
                "1",
                "--out",
                lout,
                timeout=3600,
            )
            if rc or not lout.exists():
                row["latency"] = f"adapter exit {rc}"
            else:
                w = json.loads(lout.read_text())["windows"][0]
                row["latency"] = w.get("error") or {
                    "calls": w["calls"],
                    "p50_ms": round(analyze.pct(w["latency_ms"], 0.5), 2),
                    "devices": w["devices"],
                }
        report.append(row)
        print(json.dumps(row), flush=True)
    (sdir / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nsmoke summary (not committed): {sdir / 'summary.json'}; config {cfg}")


# ------------------------------------------------------------------------------ campaign
def omlx_running() -> bool:
    r = subprocess.run(["pgrep", "-fil", "omlx"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def machine_state() -> dict:
    def out(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.stdout.strip()

    return {
        "loadavg": os.getloadavg(),
        "power_source": out(["pmset", "-g", "ps"]).splitlines()[0] if out(["pmset", "-g", "ps"]) else None,
        "thermal": out(["pmset", "-g", "therm"]),
        "omlx_running": omlx_running(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def platform_info() -> dict:
    soc = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip() or 0)
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    return {"soc": soc, "memory_gb": round(mem / 2**30), "macos": platform.mac_ver()[0], "macos_build": build}


def idle_window(a, path: Path, rnd: int):
    if not a.energy_cmd:
        return
    tmp = path.with_suffix(".sampler.json")
    proc = subprocess.Popen(a.energy_cmd.format(out=str(tmp)), shell=True, start_new_session=True)
    time.sleep(a.window_s)
    os.killpg(proc.pid, 2)
    proc.wait(timeout=30)
    d = json.loads(tmp.read_text()) if tmp.exists() else {}
    path.write_text(json.dumps({"round": rnd, "mean_power_w": d.get("mean_power_w"), "sampler": d}, indent=1) + "\n")


def weights_record(a) -> dict:
    """What each configuration loads, and whether it is the laya-apple pinned checkpoint."""
    base = Path(a.base)
    ck = checkpoints()
    out = {}
    for rt, var, m in configs(include_reference=True):
        key = f"{rt}/{var}/{m}"
        if rt == "laya-coreml":
            b = PINS["runtimes"]["laya-coreml"]["bundles"][var][m]
            out[key] = {"source": f"{b['repo']}@{b['revision'][:12]} (Core ML bundle)", "same_as_pin": None}
        elif rt == "laya-fast":
            chk = base / "laya-fast.checkpoint-check.json"
            same = json.loads(chk.read_text())["same_inputs"] if chk.exists() else None
            out[key] = {
                "source": f"convert.py fp16 of {ck[m]['repo']}@{ck[m]['revision'][:12]} (README pin {LAYA_FAST_PINNED_REVISION[:12]})",
                "same_as_pin": same,
            }
        else:
            out[key] = {"source": f"{ck[m]['repo']}@{ck[m]['revision'][:12]}", "same_as_pin": True}
    return out


def run(a):
    if omlx_running() and not a.allow_omlx:
        raise SystemExit(
            "oMLX is running. Stop it for the campaign (see README), or pass --allow-omlx to record a non-standard run."
        )
    run_dir = Path(a.raw) / a.run_id
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; raw data is never overwritten. Choose a new --run-id.")
    (run_dir / "parity").mkdir(parents=True)
    (run_dir / "latency").mkdir()
    adapter_config(a)
    base = Path(a.base)
    manifest = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform_info(),
        "machine_state": machine_state(),
        "method": {
            "rounds": a.rounds,
            "warmup_calls": a.warmup,
            "window_s": a.window_s,
            "shapes": list(LATENCY_SHAPES),
            "cooldown_s": a.cooldown_s,
            "energy_cmd": bool(a.energy_cmd),
            "rotation": "configuration order and shape order rotate by one position per round",
        },
        "pins": PINS,
        "checkpoints": checkpoints(),
        "weights": weights_record(a),
        "laya_apple_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "env_freeze": {p.stem.replace(".freeze", ""): p.read_text().splitlines() for p in base.glob("*.freeze.txt")},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    sets = fixture_sets(a, run_dir)
    cfgs = configs(a.runtimes, a.models)
    # 1. reference answers for every fixture set without committed goldens
    for m in sorted({m for _, _, m in cfgs}):
        run_parity(a, "upstream", "cpu-fp32", m, sets, run_dir / "parity")
    # 2. parity per configuration
    for rt, var, m in cfgs:
        run_parity(a, rt, var, m, sets, run_dir / "parity")
    # 3. latency rounds, rotated
    for r in range(a.rounds):
        idle_window(a, run_dir / "latency" / f"idle-{r}.json", r)
        order = cfgs[r % len(cfgs) :] + cfgs[: r % len(cfgs)]
        for i, (rt, var, m) in enumerate(order):
            k = (r + i) % len(LATENCY_SHAPES)
            shapes = LATENCY_SHAPES[k:] + LATENCY_SHAPES[:k]
            out = run_dir / "latency" / f"r{r}-{rt}-{var}-{m}.json"
            extra = ["--energy-cmd", a.energy_cmd] if a.energy_cmd else []
            adapter(
                a,
                "latency",
                rt,
                var,
                m,
                "--shapes",
                SHAPES,
                "--order",
                ",".join(shapes),
                "--warmup",
                str(a.warmup),
                "--window-s",
                str(a.window_s),
                "--windows",
                "1",
                "--out",
                out,
                *extra,
            )
            time.sleep(a.cooldown_s)
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["machine_state_end"] = machine_state()
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"campaign done: {run_dir}; now run analyze.py")


# ------------------------------------------------------------------------------ plan / teardown
def estimate(a) -> dict:
    cfgs = configs(a.runtimes, a.models)
    per_round = 0.0
    for rt, var, m in cfgs:
        per_round += EST_LOAD_S[(rt, var)] + len(LATENCY_SHAPES) * a.window_s + a.cooldown_s
        per_round += len(LATENCY_SHAPES) * a.warmup * 0.1  # warm-up: ~100 ms per call, upper side
        per_round += a.window_s if a.energy_cmd else 0
    idle = a.window_s * a.rounds if a.energy_cmd else 0
    sets_per_model = {"laya": 4, "laya-multilingual": 3, "laya-typed-decisions": 3}
    parity = sum(EST_LOAD_S[(rt, var)] + sets_per_model[m] * EST_PARITY_S_PER_SET for rt, var, m in cfgs)
    parity += sum(
        EST_LOAD_S[("upstream", "cpu-fp32")] + sets_per_model[m] * EST_PARITY_S_PER_SET * 2
        for m in {m for _, _, m in cfgs}
    )
    campaign = per_round * a.rounds + idle + parity
    setup_s = EST_SETUP_S["envs"] + EST_SETUP_S["downloads"] + EST_SETUP_S["laya_fast_convert"]
    setup_s += EST_SETUP_S["laya_fast_ane_export_per_bucket"] * len(PINS["runtimes"]["laya-fast"]["ane_buckets"])
    return {
        "configurations": len(cfgs),
        "latency_round_s": round(per_round),
        "parity_s": round(parity),
        "campaign_s": round(campaign),
        "setup_s": setup_s,
    }


def plan(a):
    for rt, var, m in configs(a.runtimes, a.models):
        print(f"{rt:12s} {var:9s} {m}")
    e = estimate(a)
    print(json.dumps(e, indent=1))
    print(f"estimated: setup ~{e['setup_s'] / 60:.0f} min (first time), campaign ~{e['campaign_s'] / 3600:.1f} h")


def teardown(a):
    base, mroot = Path(a.base).resolve(), Path(a.model_root).resolve()
    if not (base / MARKER).exists():
        raise SystemExit(f"{base} has no {MARKER}; refusing to delete anything there")
    for name in [*PINS["runtimes"], "src", "fixtures", "smoke"]:
        p = base / name
        if p.is_dir() and p.resolve().parent == base:
            shutil.rmtree(p)
            print(f"removed {p}")
    for p in [
        *base.glob("*.freeze.txt"),
        base / "config.json",
        base / "laya-fast.checkpoint-check.json",
        base / MARKER,
    ]:
        if p.exists():
            p.unlink()
            print(f"removed {p}")
    if not any(base.iterdir()):
        base.rmdir()
        print(f"removed {base}")
    if a.models:
        if (mroot / MARKER).exists():
            shutil.rmtree(mroot / "laya-fast", ignore_errors=True)
            (mroot / MARKER).unlink()
            if not any(mroot.iterdir()):
                mroot.rmdir()
            print(f"removed {mroot}")
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
        for var in PINS["runtimes"]["laya-coreml"]["bundles"].values():
            for b in var.values():
                d = hub / ("models--" + b["repo"].replace("/", "--"))
                if d.is_dir() and d.name.startswith("models--aac6fef--laya-"):
                    shutil.rmtree(d)
                    print(f"removed {d}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["plan", "setup", "smoke", "run", "teardown"])
    ap.add_argument(
        "--base", default=os.environ.get("COMPARE_BASE", str(Path.home() / "Developer/scratch/compare-v1.4"))
    )
    ap.add_argument(
        "--model-root",
        default=os.environ.get(
            "COMPARE_MODEL_ROOT",
            str(Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).parent / "compare-v1.4"),
        ),
        help="converted models and ANE bodies (default: next to HF_HOME, i.e. on the model volume)",
    )
    ap.add_argument("--runtimes", nargs="*", choices=sorted(PINS["runtimes"]))
    ap.add_argument(
        "--models", nargs="*", help="setup/run/plan: restrict models; teardown: flag, also remove model files"
    )
    ap.add_argument("--variants", nargs="*", help="setup: restrict laya-coreml bundle downloads")
    ap.add_argument("--ane-buckets", nargs="*", type=int, help="setup: laya-fast ANE buckets to export (default: all)")
    ap.add_argument("--run-id")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--window-s", type=float, default=20.0)
    ap.add_argument("--cooldown-s", type=float, default=15.0)
    ap.add_argument("--energy-cmd", default=os.environ.get("COMPARE_ENERGY_CMD"), help="sampler command with {out}")
    ap.add_argument("--allow-omlx", action="store_true")
    ap.add_argument("--raw", default=str(RAW), help="run: output root (default: raw/ next to this file)")
    a = ap.parse_args(argv)
    if a.command == "teardown":
        a.models = a.models is not None
    if a.command == "run" and not a.run_id:
        ap.error("run needs --run-id")
    {"plan": plan, "setup": setup, "smoke": smoke, "run": run, "teardown": teardown}[a.command](a)


if __name__ == "__main__":
    main()
