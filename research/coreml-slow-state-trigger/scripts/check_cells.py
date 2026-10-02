"""The pre-campaign check: PB-R and PB-H bit-identical to coremltools, and PB-R / PB-H releasing /
holding the GIL, on laya's ANE buckets.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
        research/coreml-slow-state-trigger/scripts/check_cells.py [--out research/coreml-slow-state-trigger/raw/check.json]

The ANE backend loads through the runtime (Laya.from_pretrained("laya", device="ane"): the
verified artifacts, coremltools, every bucket). Per bucket, #83's PrebindModel is built twice on
the same model.mlmodelc: PB-R as #83 ships it (shim through ctypes.CDLL) and PB-H with its _call
replaced by run_config.gil_holding(prebind._shim()), exactly as run_config.py's PB-H cell does.
  1. Bit identity (#83's check_prebind.py row generator and `same`, reused): the warm-up row plus
     ROWS generated rows through coremltools, PB-R and PB-H: every output bytes-equal at the
     model level, and ANEBackend.forward's logits and actions bytes-equal with each swapped in.
  2. GIL behaviour, on the L128 bucket: a background thread runs N_PREDICTS predicts of one
     fixed row back to back while the main thread sleeps to a 1 ms grid and records, per tick,
     how late it ran Python again (#83's GilProbe logic, on the main thread). Lateness of the
     ticks scheduled between the first predict's entry and the last predict's exit is kept.
     PB-R passes if its P50 lateness is < 1 ms; PB-H passes if its P50 lateness is at least
     0.5 x the median native predict time (the shim's own stamps).
  3. Crossings of one predict per binding (#83's crossings.count), recorded only: PYFUNCTYPE
     functions are ctypes foreign functions too, so PB-H is expected to count as "ctypes".
Exit status 1 unless all four flags hold: the campaign does not start.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PB83 = ROOT / "research" / "coreml-prebind-predict" / "scripts"
sys.path.insert(0, str(PB83))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_config = _load("slow_state_run_config", HERE / "run_config.py")  # gil_holding, as the PB-H cell uses it

import check_prebind  # noqa: E402  #83's: rows_for, same
import crossings  # noqa: E402
import prebind  # noqa: E402
import probe  # noqa: E402

MODEL = "laya"
GIL_BUCKET = 128
N_PREDICTS = 50
PERIOD_NS = 1_000_000
RELEASE_MAX_MS = 1.0  # PB-R: P50 lateness below this
HOLD_MIN_FRACTION = 0.5  # PB-H: P50 lateness at least this fraction of the median native predict
SEED = 0


def gil_test(model, feats) -> dict:
    """N_PREDICTS predicts on a background thread; a 1 ms sleep-grid probe on this thread."""
    n0 = len(prebind.STAMPS)
    span = {}
    done = threading.Event()

    def work():
        span["start"] = time.monotonic_ns()
        for _ in range(N_PREDICTS):
            model.predict(feats)
        span["end"] = time.monotonic_ns()
        done.set()

    ticks, late = [], []
    t = threading.Thread(target=work, name="predict-loop")
    tick = time.monotonic_ns() + PERIOD_NS
    t.start()
    while not done.is_set():
        d = tick - time.monotonic_ns()
        if d > 0:
            time.sleep(d / 1e9)
        now = time.monotonic_ns()
        ticks.append(tick)
        late.append(now - tick)
        tick = probe.next_tick(tick, now, PERIOD_NS)
    t.join()
    stamps = prebind.STAMPS[n0:]
    native = [s[2] - s[1] for s in stamps]
    sel = [x for tk, x in zip(ticks, late) if span["start"] <= tk < span["end"]]
    lat = np.asarray(sel, np.float64) / 1e6
    return {
        "predicts": len(stamps),
        "ticks": len(sel),
        "lateness_p50_ms": float(np.percentile(lat, 50)) if len(lat) else None,
        "lateness_p99_ms": float(np.percentile(lat, 99)) if len(lat) else None,
        "lateness_mean_ms": float(lat.mean()) if len(lat) else None,
        "native_predict_p50_ms": float(np.median(native)) / 1e6 if native else None,
        "loop_ms": (span["end"] - span["start"]) / 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE.parent / "raw" / "check.json")
    a = ap.parse_args()
    from importlib import metadata

    import laya_apple
    from laya_apple import Laya
    from laya_apple.artifacts import COMPILED, artifact_dir
    from laya_apple.backends.coreml_ane import ane_features

    rng = np.random.default_rng(SEED)
    shim = prebind._shim()
    held = run_config.gil_holding(shim)
    laya = Laya.from_pretrained(MODEL, device="ane")
    be = laya.ane
    host, pad = be.host, be.pad_id
    vocab, qtypes = host.embedding.shape[0], host.type_embedding.shape[0]
    buckets, gil = {}, {}
    lower = 0
    for b in be.buckets:
        path = artifact_dir(be.spec, b) / COMPILED
        ct = be.models[b]
        pbr, pbh = prebind.PrebindModel(path), prebind.PrebindModel(path)
        pbh._call = held
        rows = check_prebind.rows_for(b, lower, pad, vocab, qtypes, rng)
        lower = b
        ok = {"PB-R": [True, True], "PB-H": [True, True]}  # model level, forward level
        for row in rows:
            feats = ane_features([row], b, 1, host.embedding, host.type_embedding, host.window(b), pad)
            ref = ct.predict(feats)
            be.models[b] = ct
            logits_ref, act_ref = be.forward([row])
            for name, m in (("PB-R", pbr), ("PB-H", pbh)):
                ok[name][0] &= check_prebind.same(ref, m.predict(feats))
                be.models[b] = m
                logits, act = be.forward([row])
                be.models[b] = ct
                ok[name][1] &= check_prebind.same({"l": logits_ref, "a": act_ref}, {"l": logits, "a": act})
        feats = ane_features([rows[1]], b, 1, host.embedding, host.type_embedding, host.window(b), pad)
        buckets[str(b)] = {
            "rows": len(rows),
            **{f"{n}_bit_identical": bool(v[0]) for n, v in ok.items()},
            **{f"{n}_forward_bit_identical": bool(v[1]) for n, v in ok.items()},
            "call_flags": {"PB-R": int(pbr._call._flags_), "PB-H": int(pbh._call._flags_)},
            "crossings": {
                "PB-R": crossings.count(pbr.predict, feats)[1],
                "PB-H": crossings.count(pbh.predict, feats)[1],
            },
            "PB_backings": pbr.modes,
        }
        print(MODEL, f"L{b}", json.dumps(buckets[str(b)]), flush=True)
        if b == GIL_BUCKET:
            for name, m in (("PB-R", pbr), ("PB-H", pbh)):
                gil[name] = gil_test(m, feats)
                print(MODEL, f"L{b}", name, "GIL test", json.dumps(gil[name]), flush=True)

    if set(gil) != {"PB-R", "PB-H"}:
        raise RuntimeError(f"{MODEL} has no L{GIL_BUCKET} ANE bucket")
    r, h = gil["PB-R"], gil["PB-H"]
    res = {
        "model": MODEL,
        "seed": SEED,
        "rows_per_bucket": check_prebind.ROWS + 1,
        "laya_apple": laya_apple.__version__,
        "pyobjc": metadata.version("pyobjc-framework-CoreML"),
        "coremltools": metadata.version("coremltools"),
        "switch_interval_s": sys.getswitchinterval(),
        "buckets": buckets,
        "gil_test": {
            "bucket": GIL_BUCKET,
            "predicts": N_PREDICTS,
            "period_ns": PERIOD_NS,
            "release_max_ms": RELEASE_MAX_MS,
            "hold_min_fraction_of_native": HOLD_MIN_FRACTION,
            **gil,
        },
    }
    res["PB_R_bit_identical"] = all(
        x["PB-R_bit_identical"] and x["PB-R_forward_bit_identical"] for x in buckets.values()
    )
    res["PB_H_bit_identical"] = all(
        x["PB-H_bit_identical"] and x["PB-H_forward_bit_identical"] for x in buckets.values()
    )
    res["PB_R_releases_gil"] = r["lateness_p50_ms"] is not None and r["lateness_p50_ms"] < RELEASE_MAX_MS
    res["PB_H_holds_gil"] = (
        h["lateness_p50_ms"] is not None
        and h["native_predict_p50_ms"] is not None
        and h["lateness_p50_ms"] >= HOLD_MIN_FRACTION * h["native_predict_p50_ms"]
    )
    flags = ("PB_R_bit_identical", "PB_H_bit_identical", "PB_R_releases_gil", "PB_H_holds_gil")
    res["all_ok"] = all(res[k] for k in flags)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for k in (*flags, "all_ok"):
        print(f"{k}: {res[k]}")
    sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
