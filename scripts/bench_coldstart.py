"""Cold start: ane_startup="wait" against "background" (v0.3), and where Core ML's on-device
ANE compile cache is reused (research/coreml-compile-cache/).

Core ML caches its on-device ANE compile per artifact *location*. To measure a genuinely
cold start, each run copies the model's registered artifacts into a fresh cache root, then
measures in a new process:

- `ready_s`: how long `Laya.from_pretrained(..., execution="workers")` takes to return;
- `first_request_s`: the first short request's latency, and the device it used;
- `ane_ready_s`: time until the ANE serves requests;
- the same process then re-opens the model at the now-warm location (`warm_ready_s`).

    LAYA_APPLE_CACHE=<cache> HF_HUB_OFFLINE=1 \\
        .venv/bin/python scripts/bench_coldstart.py [MODEL ...] [--out benchmarks/v0.3/coldstart.json]

`--location` chooses what happens to the fresh copy before the measured process opens it
(the default, `fresh-copy`, is the v0.3 method above):

- `fresh-copy`: nothing; a location Core ML has never compiled.
- `move`: a child process loads the copy once (paying its compile), then the whole cache
  root is renamed to a new path (same files, same inodes, new path).
- `same-path-recopy`: loaded once, then its artifacts are copied aside, the originals
  deleted and the copies renamed back (same path and bytes, new inodes and mtimes).
- `touch`: loaded once, then every artifact file's mtime is set to now (same path, inodes
  and bytes).

`--modes` limits the start-up modes (default: wait and background) and `--repeats` repeats
each (model, mode) with a new copy. The copies go under <cache>/../laya-apple-coldstart-<pid>/
and are removed afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("laya-typed-decisions", "laya", "laya-multilingual")
MODES = ("wait", "background")
LOCATIONS = ("fresh-copy", "move", "same-path-recopy", "touch")


def _child(model: str, mode: str) -> dict:
    from laya_apple import Laya
    from laya_apple.workload import make_request

    t0 = time.perf_counter()
    laya = Laya.from_pretrained(model, execution="workers", ane_startup=mode, local_files_only=True)
    ready = time.perf_counter() - t0
    state, qs = make_request(laya.tokenizer, laya.config, 64, 1, seed=3)
    t1 = time.perf_counter()
    r = laya.predict(context=state, questions=qs)
    first = time.perf_counter() - t1
    laya.wait_for_ane()
    ane_ready = time.perf_counter() - t0
    after = laya.predict(context=state, questions=qs)
    probes = laya.info().get("ane_probes", {})
    laya.close()
    t2 = time.perf_counter()
    with Laya.from_pretrained(model, execution="workers", ane_startup=mode, local_files_only=True):
        warm_ready = time.perf_counter() - t2
    return {
        "model": model,
        "mode": mode,
        "ready_s": round(ready, 3),
        "first_request_s": round(first, 4),
        "first_request_device": r.runtime.device,
        "first_request_reason": r.runtime.routing_reason,
        "ane_ready_s": round(ane_ready, 3),
        "after_ready_device": after.runtime.device,
        "after_ready_answers": after.answers,
        "same_answers": r.answers == after.answers,
        "warm_ready_s": round(warm_ready, 3),
        "ane_probes": probes,
    }


def _child_warm(model: str) -> dict:
    """Load once with every offered bucket (pays this location's compile), then close."""
    from laya_apple import Laya

    t0 = time.perf_counter()
    with Laya.from_pretrained(model, execution="workers", ane_startup="wait", local_files_only=True) as laya:
        buckets = list(laya.ane_state.buckets) + list(laya.info()["auto_ane"]["tie_buckets"])
    return {"model": model, "prep_load_s": round(time.perf_counter() - t0, 3), "prep_buckets": sorted(buckets)}


def _fresh_cache(src_cache: Path, model: str, dest: Path) -> None:
    from laya_apple.registry import resolve

    spec = resolve(model)
    src = src_cache / "artifacts" / spec.name
    shutil.copytree(src, dest / "artifacts" / spec.name, symlinks=True)


def _run(args: list, cache: Path) -> dict:
    env = dict(os.environ, LAYA_APPLE_CACHE=str(cache), HF_HUB_OFFLINE="1")
    r = subprocess.run([sys.executable, __file__, *args], env=env, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"{' '.join(args)} failed:\n{r.stderr[-3000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def _prepare(location: str, model: str, dest: Path) -> tuple[Path, dict]:
    """Apply `location` to the fresh copy at `dest`; returns the cache root to measure and
    what the preparation recorded."""
    if location == "fresh-copy":
        return dest, {}
    prep = _run(["--child-warm", model], dest)
    files = [f for f in (dest / "artifacts").rglob("*") if f.is_file()]
    if location == "move":
        moved = dest.with_name(dest.name + "-moved")
        os.rename(dest, moved)
        return moved, prep
    if location == "same-path-recopy":
        aside = dest.with_name(dest.name + "-aside")
        shutil.copytree(dest / "artifacts", aside, symlinks=True)
        shutil.rmtree(dest / "artifacts")
        os.rename(aside, dest / "artifacts")
        return dest, prep
    if location == "touch":
        now = time.time()
        for f in files:
            os.utime(f, (now, now))
        return dest, prep
    raise ValueError(location)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", default=list(MODELS))
    p.add_argument("--out", default=str(ROOT / "benchmarks" / "v0.3" / "coldstart.json"))
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--location", choices=LOCATIONS, default="fresh-copy")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--child", nargs=2, metavar=("MODEL", "MODE"), help=argparse.SUPPRESS)
    p.add_argument("--child-warm", metavar="MODEL", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.child:
        print(json.dumps(_child(*a.child)))
        return 0
    if a.child_warm:
        print(json.dumps(_child_warm(a.child_warm)))
        return 0
    if a.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    from laya_apple.artifacts import platform_profile
    from laya_apple.hub import cache_root

    src_cache = cache_root()
    work = src_cache.parent / f"laya-apple-coldstart-{os.getpid()}"
    rows = []
    try:
        for model in a.models:
            for mode in a.modes:
                for repeat in range(a.repeats):
                    dest = work / f"{model}-{mode}-{repeat}"
                    _fresh_cache(src_cache, model, dest)
                    cache, prep = _prepare(a.location, model, dest)
                    row = {"location": a.location, "repeat": repeat, **prep, **_run(["--child", model, mode], cache)}
                    rows.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != "after_ready_answers"}), flush=True)
                    shutil.rmtree(cache)
                    if dest.exists():
                        shutil.rmtree(dest)
    finally:
        if work.exists() and work.name.startswith("laya-apple-coldstart-"):
            shutil.rmtree(work)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "location": a.location,
        "modes": a.modes,
        "repeats": a.repeats,
        "platform": platform_profile(),
        "python": platform.python_version(),
    }
    out.write_text(json.dumps({**meta, "rows": rows}, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
