"""Per-runtime adapter for the same-machine runtime comparison (benchmarks/compare-v1.4).

This file runs *inside each runtime's own environment*, started by drive.py with that
environment's python. It imports only the standard library, numpy and the runtime under
test, so the runtimes never share an environment or a module namespace (laya-fast ships a
top-level `laya_mlx` module that would shadow the `laya_mlx` package from PyPI).

    <env>/bin/python adapter.py info     --runtime R --variant V --model M --config C.json
    <env>/bin/python adapter.py parity   --runtime R --variant V --model M --config C.json \
        --fixtures F.json --out OUT.json
    <env>/bin/python adapter.py latency  --runtime R --variant V --model M --config C.json \
        --shapes S.json --out OUT.json [--order a,b,c] [--warmup N] [--window-s S] [--windows W]
    <env>/bin/python adapter.py fixtures --runtime laya-fast --config C.json --out OUT.json

C.json is written by drive.py: runtime paths and checkpoint locations (see drive.py
`adapter_config`). Every answer is normalised to the upstream public answer schema before it
is written, so analyze.py can compare any runtime with any reference:
  {qid: {"type", "labels": [...], "probs": [...], "choice", "act_probability", "raw"}}
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ADAPTER_SCHEMA = 1


# ------------------------------------------------------------------------------ helpers
def _versions(names):
    out = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


def platform_info():
    soc = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
    return {
        "soc": soc.stdout.strip() or None,
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
    }


def normalize_answer(ans: dict) -> dict:
    """One public answer (upstream / Jev schema, or laya-fast's rl_agent variant) -> common form."""
    t = ans.get("type")
    if t == "noul" or ("noul" in ans and "probabilities" not in ans):
        p = float(ans["noul"])
        labels, probs = ["false", "true"], [1.0 - p, p]
        t = "noul"
    else:
        probs_map = ans.get("probabilities") or {}
        labels, probs = list(probs_map), [float(v) for v in probs_map.values()]
    act = (ans.get("action") or {}).get("act_probability")
    if act is None:
        act = (ans.get("rl_agent") or {}).get("act_probability")
    return {
        "type": t,
        "labels": labels,
        "probs": probs,
        "choice": ans.get("choice"),
        "act_probability": None if act is None else float(act),
        "raw": ans,
    }


def normalize_result(answers: dict) -> dict:
    return {qid: normalize_answer(a) for qid, a in answers.items()}


# ------------------------------------------------------------------------------ runtimes
class Runtime:
    """predict(state, questions) -> (answers dict in public schema, extra per-call info)."""

    packages: tuple = ()

    def __init__(self, variant: str, model: str, cfg: dict):
        self.variant, self.model, self.cfg = variant, model, cfg

    def predict(self, state, questions):
        raise NotImplementedError

    def info(self) -> dict:
        return {"packages": _versions(self.packages)}


def _snapshot(cfg: dict, model: str) -> str:
    """Local snapshot of the pinned upstream checkpoint, as drive.py resolved it (never downloads)."""
    return cfg["snapshots"][model]


class LayaApple(Runtime):
    packages = ("laya-apple", "mlx", "coremltools", "numpy", "tokenizers")

    def __init__(self, variant, model, cfg):
        super().__init__(variant, model, cfg)
        from laya_apple import Laya

        device = {"auto": "auto", "gpu": "gpu"}[variant]
        self.m = Laya.from_pretrained(f"convaiinnovations/{model}", device=device, local_files_only=True)

    def predict(self, state, questions):
        r = self.m.predict(state=state, questions=questions)
        rt = r.runtime
        return r.answers, {"device": rt.device, "backend": rt.backend, "reason": rt.routing_reason}


class LayaMlx(Runtime):
    packages = ("laya-mlx", "mlx", "numpy", "tokenizers", "huggingface-hub")

    def __init__(self, variant, model, cfg):
        super().__init__(variant, model, cfg)
        import laya_mlx

        kw = {"dtype": "float16"}
        if variant == "fp16-opt":  # the documented opt-in path (README "For repeated workloads")
            kw.update(compile=True, pad_to_multiple=16, cache_prompts=True)
        elif variant != "fp16":
            raise ValueError(variant)
        self.m = laya_mlx.load(_snapshot(cfg, model), **kw)

    def predict(self, state, questions):
        return self.m.predict(state, questions)["answers"], {"device": "gpu"}


class LayaCoreml(Runtime):
    packages = ("laya-coreml", "coremltools", "numpy", "tokenizers", "huggingface-hub")

    def __init__(self, variant, model, cfg):
        super().__init__(variant, model, cfg)
        import laya_coreml

        b = cfg["runtimes"]["laya-coreml"]["bundles"][variant][model]
        self.bundle = b
        self.m = laya_coreml.load(b["repo"], revision=b["revision"], local_files_only=True)

    def predict(self, state, questions):
        return self.m.predict(state, questions)["answers"], {"device": getattr(self.m, "compute_units", None)}

    def info(self):
        """Bundle pin plus the source weight hash its own manifest records (provenance check)."""
        from laya_coreml.hub import resolve_checkpoint

        d = Path(resolve_checkpoint(self.bundle["repo"], revision=self.bundle["revision"], local_files_only=True))
        src = {}
        for f in sorted(d.glob("*.json")):
            try:
                text = f.read_text()
            except OSError:
                continue
            if "source_weights_sha256" in text:
                j = json.loads(text)
                stack = [j]
                while stack:
                    o = stack.pop()
                    if isinstance(o, dict):
                        for k, v in o.items():
                            if k in ("source_weights_sha256", "source_revision", "revision") and isinstance(v, str):
                                src.setdefault(k, v)
                            else:
                                stack.append(v)
                    elif isinstance(o, list):
                        stack.extend(o)
                src["manifest"] = f.name
                break
        return {**super().info(), "bundle": self.bundle, "bundle_source": src}


class LayaFast(Runtime):
    packages = ("mlx", "coremltools", "torch", "numpy", "tokenizers")

    def __init__(self, variant, model, cfg):
        super().__init__(variant, model, cfg)
        rc = cfg["runtimes"]["laya-fast"]
        sys.path.insert(0, rc["source_dir"])
        if model != "laya":
            raise ValueError("laya-fast supports the English laya checkpoint only")
        if variant == "mlx":
            from laya_api import LayaMLX

            self.m = LayaMLX(rc["converted_dir"], dtype="float16", compile=True)
        elif variant == "fast":
            from laya_fast import LayaFast as LF

            self.m = LF(rc["converted_dir"], ane_dir=rc["ane_dir"])
            self.ane_buckets = list(getattr(self.m.ane, "buckets", []) or [])
            # observe (not change) the router: which engine served each call
            self._used = set()
            for attr, dev in (("_ane_rows", "ane"), ("_mlx_rows", "gpu")):
                fn = getattr(self.m, attr)

                def wrapped(*args, _fn=fn, _dev=dev, **kw):
                    self._used.add(_dev)
                    return _fn(*args, **kw)

                setattr(self.m, attr, wrapped)
        else:
            raise ValueError(variant)

    def predict(self, state, questions):
        if self.variant == "fast":
            self._used = set()
            answers = self.m.system_one(state, questions)["answers"]
            return answers, {"device": "+".join(sorted(self._used))}
        return self.m.system_one(state, questions)["answers"], {"device": "gpu"}

    def info(self):
        i = super().info()
        if self.variant == "fast":
            i["ane_buckets_present"] = self.ane_buckets
        return i


class Upstream(Runtime):
    """Unmodified upstream Laya (PyTorch CPU FP32): the reference for fixtures without goldens."""

    packages = ("laya", "torch", "transformers", "tokenizers", "numpy", "safetensors")

    def __init__(self, variant, model, cfg):
        super().__init__(variant, model, cfg)
        import shutil
        import tempfile

        import laya
        import torch

        torch.set_grad_enabled(False)
        src = Path(_snapshot(cfg, model))
        # upstream may rewrite tokenizer_config.json in place; keep the shared HF cache untouched
        self._tmp = tempfile.TemporaryDirectory()
        dest = Path(self._tmp.name) / "checkpoint"
        dest.mkdir()
        (dest / "model.safetensors").symlink_to((src / "model.safetensors").resolve())
        shutil.copy(src / "rl_agent_config.json", dest)
        shutil.copytree(src / "encoder", dest / "encoder")
        shutil.copytree(src / "tokenizer", dest / "tokenizer")
        self.m = laya.load(str(dest), device="cpu")
        if self.m.device.type != "cpu" or self.m.amp_enabled:
            raise SystemExit(f"upstream did not select CPU FP32: {self.m.device} amp={self.m.amp_enabled}")

    def predict(self, state, questions):
        return self.m.system_one(state, questions)["answers"], {"device": "cpu"}


RUNTIMES = {
    "laya-apple": LayaApple,
    "laya-mlx": LayaMlx,
    "laya-coreml": LayaCoreml,
    "laya-fast": LayaFast,
    "upstream": Upstream,
}


# ------------------------------------------------------------------------------ energy hook
class EnergyWindow:
    """Optional: wrap one latency window in an external power sampler.

    `cmd` is a command template with an `{out}` placeholder (drive.py --energy-cmd). The
    sampler runs until SIGINT and then writes {out} (JSON: energy_j, mean_power_w,
    duration_s, source, ...). Without a command this does nothing, and analyze.py reports
    the energy column as not measured.
    """

    def __init__(self, cmd: str | None, out: Path | None, lead_s: float = 1.0):
        self.cmd, self.out, self.lead_s, self.proc = cmd, out, lead_s, None

    def __enter__(self):
        if self.cmd and self.out:
            self.proc = subprocess.Popen(self.cmd.format(out=str(self.out)), shell=True, start_new_session=True)
            time.sleep(self.lead_s)
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        return False

    def result(self):
        if self.proc is None or not self.out or not self.out.exists():
            return None
        try:
            return json.loads(self.out.read_text())
        except (OSError, ValueError):
            return None


# ------------------------------------------------------------------------------ tasks
def load_runtime(a, cfg):
    t0 = time.perf_counter()
    rt = RUNTIMES[a.runtime](a.variant, a.model, cfg)
    return rt, time.perf_counter() - t0


def header(a, rt, load_s):
    return {
        "schema": ADAPTER_SCHEMA,
        "runtime": a.runtime,
        "variant": a.variant,
        "model": a.model,
        "load_s": round(load_s, 3),
        "platform": platform_info(),
        "runtime_info": rt.info(),
        "started_unix": time.time(),
    }


def task_parity(a, cfg):
    fixtures = json.loads(Path(a.fixtures).read_text())
    rt, load_s = load_runtime(a, cfg)
    out = header(a, rt, load_s)
    out["fixture_set"] = fixtures["name"]
    cases = []
    for case in fixtures["cases"]:
        rec = {"name": case["name"]}
        try:
            t = time.perf_counter()
            answers, extra = rt.predict(case["state"], case["questions"])
            rec["ms"] = (time.perf_counter() - t) * 1e3
            rec["answers"] = normalize_result(answers)
            rec["extra"] = extra
        except Exception as e:  # recorded, never hidden: an unsupported input is a result
            rec["error"] = f"{type(e).__name__}: {e}"
        cases.append(rec)
    out["cases"] = cases
    # determinism: the first answered case twice more, compared on the public answers
    first = next((c for c, r in zip(fixtures["cases"], cases) if "answers" in r), None)
    if first is not None:
        r1, _ = rt.predict(first["state"], first["questions"])
        r2, _ = rt.predict(first["state"], first["questions"])
        out["repeat_identical"] = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
        out["repeat_case"] = first["name"]
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    ok = sum("answers" in c for c in cases)
    print(f"{a.runtime}/{a.variant}/{a.model} parity {fixtures['name']}: {ok}/{len(cases)} answered", flush=True)


def _pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def task_latency(a, cfg):
    shapes = json.loads(Path(a.shapes).read_text())["shapes"]
    order = a.order.split(",") if a.order else list(shapes)
    rt, load_s = load_runtime(a, cfg)
    out = header(a, rt, load_s)
    out["method"] = {"warmup_calls": a.warmup, "window_s": a.window_s, "windows": a.windows, "order": order}
    out["windows"] = []
    energy_dir = Path(a.out).with_suffix("")
    for name in order:
        shape = shapes[name]
        state, questions = shape["state"], shape["questions"]
        try:
            _, extra = rt.predict(state, questions)
        except Exception as e:
            out["windows"].append({"shape": name, "error": f"{type(e).__name__}: {e}"})
            print(f"  {name}: unsupported ({type(e).__name__})", flush=True)
            continue
        t = time.perf_counter()
        for _ in range(a.warmup):
            rt.predict(state, questions)
        warm_s = time.perf_counter() - t
        for w in range(a.windows):
            ef = energy_dir / f"energy-{name}-{w}.json" if a.energy_cmd else None
            if ef:
                ef.parent.mkdir(parents=True, exist_ok=True)
            devices: dict = {}
            lat = []
            with EnergyWindow(a.energy_cmd, ef) as ew:
                up0 = time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)  # the energy sampler's clock
                t_start, wall0 = time.time(), time.perf_counter()
                while True:
                    t0 = time.perf_counter()
                    _, extra = rt.predict(state, questions)
                    t1 = time.perf_counter()
                    lat.append((t1 - t0) * 1e3)
                    d = (extra or {}).get("device")
                    devices[str(d)] = devices.get(str(d), 0) + 1
                    if t1 - wall0 >= a.window_s:
                        break
                wall = time.perf_counter() - wall0
                t_end = time.time()
                up1 = time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)
            rec = {
                "shape": name,
                "window": w,
                "questions_per_call": len(questions),
                "calls": len(lat),
                "wall_s": wall,
                "t_start_unix": t_start,
                "t_end_unix": t_end,
                "t_start_uptime_ns": up0,
                "t_end_uptime_ns": up1,
                "warmup_s": warm_s,
                "devices": devices,
                "latency_ms": [round(x, 4) for x in lat],
            }
            if a.energy_cmd:
                rec["energy"] = ew.result()
            out["windows"].append(rec)
            print(
                f"  {name}[{w}] n={len(lat)} p50={_pct(lat, 0.5):.2f} p99={_pct(lat, 0.99):.2f} ms devices={devices}",
                flush=True,
            )
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False) + "\n")


def task_info(a, cfg):
    rt, load_s = load_runtime(a, cfg)
    print(json.dumps(header(a, rt, load_s), indent=1))


def task_fixtures(a, cfg):
    """laya-fast's published fixtures (benchmarks/benchmark.py make_fixtures) as JSON cases."""
    rc = cfg["runtimes"]["laya-fast"]
    sys.path.insert(0, rc["source_dir"])
    sys.argv = [sys.argv[0]]
    import importlib.util

    spec = importlib.util.spec_from_file_location("lf_benchmark", Path(rc["source_dir"]) / "benchmarks/benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = [{"name": k, "state": s, "questions": q} for k, (s, q) in mod.make_fixtures().items()]
    Path(a.out).write_text(json.dumps({"name": "laya-fast-fixtures", "cases": cases}, indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", choices=["info", "parity", "latency", "fixtures"])
    ap.add_argument("--runtime", required=True, choices=sorted(RUNTIMES))
    ap.add_argument("--variant", default="")
    ap.add_argument("--model", default="laya")
    ap.add_argument("--config", required=True)
    ap.add_argument("--fixtures")
    ap.add_argument("--shapes")
    ap.add_argument("--order", default="")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--window-s", type=float, default=20.0)
    ap.add_argument("--windows", type=int, default=1)
    ap.add_argument("--energy-cmd", default=None)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    try:
        {"info": task_info, "parity": task_parity, "latency": task_latency, "fixtures": task_fixtures}[a.task](a, cfg)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
