"""Which property of a thread-placed ANE call delays the GPU reply: Core ML, or the GIL?

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML python \
        research/coreml-gil-completion-path/scripts/completion_probe.py \
        --out research/coreml-gil-completion-path/raw/completion-probe-laya-typed-decisions.json

One runtime GPU instance (`Laya(device="gpu", execution="workers", trace=...)`, MLX in its
worker process, as in the product) serves GPU L128 requests in a closed loop from the main
thread. Meanwhile a load thread in the same interpreter loops one call. The loads form a
2x2 of {Core ML, not Core ML} x {GIL held, GIL released}, plus an idle baseline:

  idle          nothing
  ct_predict    Core ML predict through coremltools (the product path): Core ML, GIL held
  objc_predict  the same compiled model and inputs through PyObjC: Core ML, GIL released
  held_sleep    usleep(T) through ctypes.PyDLL: no Core ML, GIL held for T
  free_sleep    usleep(T) through ctypes.CDLL: no Core ML, GIL released for T
                (T = the median ct_predict call, measured first)

Per load: the GPU reply leg from the runtime trace (service_end -> received) and its parent
side split by boundary.py (header = read returned and GIL re-acquired). Loads interleave over
--repeats rounds of --seconds each.

--sample-seconds: afterwards, for ct_predict and objc_predict, macOS `sample` records this
process's native stacks (1 ms interval) while the loop runs. The GPU dispatcher is the thread
reading the worker socket (`os_read`); its samples are counted by where that read is: in the
`read` syscall, or back from it and waiting for the GIL (`take_gil`). Sampling perturbs
timing, so those windows are not used for the latency results.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import boundary  # noqa: E402
import nogil_predict  # noqa: E402

now = time.monotonic_ns


def pct(a) -> dict:
    a = np.asarray(a, float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


_LINE = re.compile(r"^(?P<lead>[\s+!:|]*?)(?P<n>\d+) (?P<frame>\S+)")


def classify(sample_text: str) -> dict:
    """Count the socket reader's samples by where its os.read / os.write is.

    `sample` prints each thread as a call tree of "<count> <frame>" lines, indented by depth.
    Under the dispatcher's os_read (os_write), a sample is either in the syscall (`read`,
    `write`: waiting for the worker's reply, or writing the job) or in `take_gil`: the syscall
    has returned and the thread is waiting to re-acquire the GIL.
    """
    out = {
        "read_syscall": 0,
        "read_then_take_gil": 0,
        "write_syscall": 0,
        "write_then_take_gil": 0,
        "reader_threads": 0,
    }
    for block in re.split(r"\n(?=\s+\d+ Thread_)", sample_text):
        if "os_read" not in block:
            continue
        out["reader_threads"] += 1
        lines = block.splitlines()
        for i, line in enumerate(lines):
            m = _LINE.match(line)
            if not m or m.group("frame") not in ("os_read", "os_write"):
                continue
            side, depth = m.group("frame")[3:], m.start("n")
            for sub in lines[i + 1 :]:
                k = _LINE.match(sub)
                if not k or k.start("n") <= depth:
                    break
                frame = k.group("frame").split(".llvm.")[0]
                if frame in ("read", "write", "__read_nocancel", "__write_nocancel"):
                    out[f"{side}_syscall"] += int(k.group("n"))
                elif frame == "take_gil":
                    out[f"{side}_then_take_gil"] += int(k.group("n"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-typed-decisions")
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sample-seconds", type=int, default=3)
    ap.add_argument("--keep-samples", action="store_true", help="keep the raw `sample` text (not committed)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import coremltools
    import mlx.core as mx

    import laya_apple
    from laya_apple import Laya
    from laya_apple.artifacts import COMPILED, artifact_dir, platform_profile
    from laya_apple.backends.coreml_ane import ANEBackend, ane_features
    from laya_apple.hub import checkpoint_path
    from laya_apple.prompt import Tokenizer, prepare
    from laya_apple.registry import resolve
    from laya_apple.workload import make_request

    traces: list = []
    stamps: list = []
    gpu = Laya.from_pretrained(
        args.model, device="gpu", execution="workers", local_files_only=True, trace=traces.append
    )
    assert boundary.attach(gpu, stamps)
    spec = resolve(args.model)
    ckpt = checkpoint_path(spec, local_files_only=True)
    cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
    enc = json.loads((ckpt / "encoder/config.json").read_text())
    tok = Tokenizer(ckpt / "tokenizer")
    B = args.length
    ane = ANEBackend(spec, ckpt, tok.pad_token_id, int(enc["local_attention"]), [B], strict=True)
    objc = nogil_predict.NoGilModel(artifact_dir(spec, B) / COMPILED)
    rows = prepare(tok, cfg, *make_request(tok, cfg, B, n_questions=1, seed=1)).items
    feats = ane_features(rows, B, 1, ane.host.embedding, ane.host.type_embedding, ane.host.window(B), ane.pad_id)
    state, questions = make_request(gpu.tokenizer, gpu.config, args.length, n_questions=1, seed=100)
    reference = gpu.predict(context=state, questions=questions).answers
    for _ in range(20):
        ane.models[B].predict(feats)
        objc.predict(feats)
        gpu.predict(context=state, questions=questions)
    ct = []
    for _ in range(100):
        t = now()
        ane.models[B].predict(feats)
        ct.append(now() - t)
    T_us = int(np.median(ct) / 1000)
    held, free = ctypes.PyDLL(None).usleep, ctypes.CDLL(None).usleep
    loads = {
        "idle": None,
        "ct_predict": lambda: ane.models[B].predict(feats),
        "objc_predict": lambda: objc.predict(feats),
        "held_sleep": lambda: held(T_us),
        "free_sleep": lambda: free(T_us),
    }

    def window(name, seconds):
        fn, stop, calls = loads[name], threading.Event(), []

        def loop():
            while not stop.is_set():
                t = now()
                fn()
                calls.append((t, now()))

        th = None
        if fn is not None:
            th = threading.Thread(target=loop, daemon=True)
            th.start()
            time.sleep(0.3)
        del traces[:], stamps[:]
        end, mismatches = now() + int(seconds * 1e9), 0
        while now() < end:
            mismatches += gpu.predict(context=state, questions=questions).answers != reference
        stop.set()
        if th is not None:
            th.join()
        return list(traces), list(stamps), calls, mismatches

    results = []
    for rep in range(args.repeats):
        for name in loads:
            tr, st, calls, mism = window(name, args.seconds)
            by_id = {s[0]: s for s in st}
            ends = np.array([c[1] for c in calls], np.int64)
            leg, hdr, after_hdr, predict_end_to_hdr = [], [], [], []
            for t in tr:
                s = by_id.get(t.request_id)
                if s is None:
                    continue
                _, enter, header, body, loaded = s
                leg.append((t.received_ns - t.service_end_ns) / 1e6)
                hdr.append((header - t.service_end_ns) / 1e6)
                after_hdr.append((t.received_ns - header) / 1e6)
                if ends.size:  # the first load call to end after the GPU forward ended
                    i = np.searchsorted(ends, t.service_end_ns)
                    if i < ends.size:
                        predict_end_to_hdr.append((header - ends[i]) / 1e6)
            res = {
                "repeat": rep,
                "load": name,
                "gpu_requests": len(tr),
                "mismatches": int(mism),
                "gpu_req_s": len(tr) / args.seconds,
                "return_ms": pct(leg),
                "service_end_to_header_ms": pct(hdr),
                "header_to_received_ms": pct(after_hdr),
                "load_call_end_to_header_ms": pct(predict_end_to_hdr),
                "gpu_service_ms": pct([(t.service_end_ns - t.service_start_ns) / 1e6 for t in tr]),
                "gpu_occupancy_ms": pct([(t.received_ns - t.dispatch_ns) / 1e6 for t in tr]),
                "gpu_e2e_ms": pct([(t.response_ns - t.submit_ns) / 1e6 for t in tr]),
                "load_call_ms": pct([(b - a) / 1e6 for a, b in calls]) if calls else None,
                "load_calls_s": len(calls) / args.seconds,
            }
            results.append(res)
            print(
                rep,
                name,
                f"{res['gpu_req_s']:.1f} req/s",
                "return p50",
                round(res["return_ms"]["p50"], 3),
                "p99",
                round(res["return_ms"]["p99"], 3),
                "hdr p50",
                round(res["service_end_to_header_ms"]["p50"], 3),
                flush=True,
            )

    sampled = {}
    for name in ("ct_predict", "objc_predict"):
        path = Path(args.out).with_suffix(f".{name}.sample.txt")
        proc = subprocess.Popen(
            [
                "/usr/bin/sample",
                str(__import__("os").getpid()),
                str(args.sample_seconds),
                "1",
                "-mayDie",
                "-file",
                str(path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        window(name, args.sample_seconds + 1.5)
        proc.wait()
        text = path.read_text()
        sampled[name] = classify(text)
        if not args.keep_samples:
            path.unlink()  # machine-specific stacks and addresses; the counts are what is kept
        print("sample", name, sampled[name], flush=True)

    gpu.close()
    out = {
        "experiment": "coreml-gil-completion-probe",
        "args": vars(args),
        "environment": {
            "laya_apple": laya_apple.__version__,
            "platform": platform_profile(),
            "python": platform.python_version(),
            "mlx": mx.__version__,
            "coremltools": coremltools.__version__,
            "pyobjc": __import__("objc").__version__,
            "switch_interval_s": sys.getswitchinterval(),
            "machine": platform.machine(),
        },
        "sleep_us": T_us,
        "results": results,
        "stack_samples": sampled,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
