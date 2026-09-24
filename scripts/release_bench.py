"""Release benchmark: warm latency per model x device x length.

    LAYA_APPLE_CACHE=... uv run python scripts/release_bench.py benchmarks/v0.1/raw.jsonl

One fresh `laya-apple benchmark` process per configuration; two passes, the second in
reversed order, so order effects show up as pass-to-pass differences. Every record keeps
its raw samples. Requests are exact-length (laya_apple.workload).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import bench_preflight  # scripts/ is on sys.path when this file runs as a script

ROOT = Path(__file__).resolve().parents[1]
CLI = [str(ROOT / ".venv/bin/laya-apple"), "--offline", "benchmark"]
LENGTHS = {
    "laya": [64, 96, 128, 256, 512],
    "laya-multilingual": [64, 96, 128, 256, 512, 1024],
    "laya-typed-decisions": [64, 96, 128, 256, 512, 1024],
}


def plan():
    from laya_apple.registry import models

    out = []
    for name, spec in models().items():
        for L in LENGTHS[name]:
            out.append((name, "gpu", L, 1))
            out.append((name, "auto", L, 1))
            if L in spec.ane_buckets:
                out.append((name, "ane", L, 1))
        out.append((name, "gpu", 128, 4))
        out.append((name, "auto", 128, 4))
    return out


def main():
    target = Path(sys.argv[1])
    iters = sys.argv[2] if len(sys.argv) > 2 else "50"
    if bench_preflight.check(target.parent.resolve()) != 0:
        sys.exit(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    configs = plan()
    for pass_no, order in ((1, configs), (2, list(reversed(configs)))):
        for model, device, length, q in order:
            tmp = target.with_suffix(".tmp.jsonl")
            tmp.unlink(missing_ok=True)
            cmd = CLI + [
                model,
                "--device",
                device,
                "--lengths",
                str(length),
                "--questions",
                str(q),
                "--iters",
                iters,
                "--warmup",
                "10",
                "--output",
                str(tmp),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not tmp.exists():
                print(f"FAILED {model} {device} L{length} q{q}: {r.stderr[-500:]}", flush=True)
                continue
            with target.open("a") as f:
                for line in tmp.read_text().splitlines():
                    rec = json.loads(line)
                    rec["pass"] = pass_no
                    f.write(json.dumps(rec) + "\n")
                    fw = rec.get("forward", {})
                    print(
                        pass_no,
                        model,
                        device,
                        length,
                        q,
                        rec.get("device"),
                        rec.get("routing_reason"),
                        round(fw.get("p50_ms", float("nan")), 2),
                        rec["status"][:60],
                        flush=True,
                    )
            tmp.unlink()


if __name__ == "__main__":
    main()
