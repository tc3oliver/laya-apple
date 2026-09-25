"""Cross-check the no-sudo sampler against `sudo powermetrics` on one fixed workload.

Terminal 1 (the operator runs this; nothing in this repository runs sudo):

    sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 200 \\
      -o research/energy-sampler/raw/crosscheck-powermetrics.txt

Terminal 2, within ~20 s of starting terminal 1:

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 \\
      uv run python research/energy-sampler/scripts/crosscheck.py run \\
      --out research/energy-sampler/raw/crosscheck-sampler.json.gz

Then `uv run python research/energy-sampler/scripts/analyze.py` reports the agreement
(results.json "crosscheck"); `crosscheck.py compare` prints it without writing anything.

Fixed workload (phases, each closed loop, one request at a time, laya-typed-decisions):
idle 20 s, GPU L128 single-question 40 s, idle 20 s, ANE L128 single-question 40 s, idle 20 s.
Both tools read the same IOReport energy model, so this checks the sampler's reading, unit
scaling, interval arithmetic and clock alignment, not the physical accuracy of the model.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import PROTOCOL, SamplerProcess, machine_meta  # noqa: E402
from sampler import uptime_ns  # noqa: E402

PHASES = [("idle", None, 20.0), ("gpu", "gpu", 40.0), ("idle", None, 20.0), ("ane", "ane", 40.0), ("idle", None, 20.0)]


def run(out: Path, model: str, interval: float) -> None:
    from laya_apple import Laya
    from laya_apple.workload import make_request

    lay = {d: Laya.from_pretrained(model, device=d, local_files_only=True) for d in ("gpu", "ane")}
    ctx, qs = make_request(lay["gpu"].tokenizer, lay["gpu"].config, 128, n_questions=1, seed=2)
    for m in lay.values():
        m.predict(context=ctx, questions=qs)
    sampler = SamplerProcess(interval)
    phases = []
    try:
        for name, dev, secs in PHASES:
            t0, u0 = uptime_ns(), time.time_ns()
            n = 0
            end = t0 + int(secs * 1e9)
            while uptime_ns() < end:
                if dev is None:
                    time.sleep(0.2)
                else:
                    lay[dev].predict(context=ctx, questions=qs)
                    n += 1
            phases.append(
                {"phase": name, "uptime_ns": [t0, uptime_ns()], "unix_ns": [u0, time.time_ns()], "requests": n}
            )
            print(f"{name:5s} {secs:.0f}s requests={n}", file=sys.stderr, flush=True)
    finally:
        samples = sampler.stop()
        for m in lay.values():
            m.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump({"meta": machine_meta(), "model": model, "phases": phases, "sampler": samples}, f)
    print(f"wrote {out}", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, default=HERE.parent / "raw" / "crosscheck-sampler.json.gz")
    r.add_argument("--model", default="laya-typed-decisions")
    r.add_argument("--interval", type=float, default=PROTOCOL["sampler_interval_s"])
    c = sub.add_parser("compare")
    c.add_argument("--sampler", type=Path, default=HERE.parent / "raw" / "crosscheck-sampler.json.gz")
    c.add_argument("--powermetrics", type=Path, default=HERE.parent / "raw" / "crosscheck-powermetrics.txt")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.out, a.model, a.interval)
    else:
        from analyze import crosscheck

        print(json.dumps(crosscheck(a.sampler, a.powermetrics), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
