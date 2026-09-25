"""Phase 0: is the asynchronous prebound Core ML predict (PB-ASYNC) feasible? Gates 1-8 of
../criteria.md, and the non-gating records.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
        research/coreml-async-predict/scripts/phase0.py [--out research/coreml-async-predict/raw/phase0.json]

The ANE backend loads through the runtime (Laya.from_pretrained("laya", device="ane"): the
verified artifacts, coremltools, every bucket). Per bucket, on the same model.mlmodelc: #83's
PrebindModel (PB-SYNC) and prebind_async.AsyncPrebindModel (PB-ASYNC).
  1 callable     the async API responds, and every submit of Phase 0 completes within 5 s
  2 correctness  #83's check_prebind rows (the warm-up row plus 10 generated rows) through
                 coremltools, PB-SYNC and PB-ASYNC: every output bytes-equal (check_prebind.same),
                 and ANEBackend.forward's logits and actions bytes-equal with each swapped in
  3 backings     every PB-ASYNC output's mode is "backed" (#83's sentinel check, made through the
                 async call at load; the evidence is recorded)
  4 callbacks    over every Phase-0 submit: exactly one callback each, no error, no late or
                 duplicate callback (checked after a 0.5 s grace period)
  5 repeat       per bucket, 1000 ANEBackend.forward calls through PB-ASYNC on generated rows
                 (check_prebind.rows_for, so the inputs change): no hang, timeout or exception;
                 RSS growth (ps -o rss) over the 1000 <= 64 MB; every 100th forward bytes-equal to
                 the coremltools forward of the same row
  6 threading    no callback ran on a thread that submitted (callback thread ids and dispatch
                 queue labels recorded)
  7 early return on L128's repeat: P50(submit_after - submit_before) < 1 ms and < 0.2 x
                 P50(callback_entry - submit_before)
  8 GIL          a background thread runs 200 PB-ASYNC L128 predicts while this thread sleeps to a
                 1 ms grid (#89's check_cells method): probe lateness P50 < 1 ms
Non-gating: callback_entry - submit_before (native plus handoff), wake - callback_entry (waiter
handoff) and the submit call's own time; crossings of one forward on the calling thread (#83's
crossings.count) and callbacks per forward; PB-ASYNC-STAMPED's native completion ->
callback_entry (200 L128 predicts through async_stamper.m's C block), or the reason it could
not be built: a semantics check of the completion path, not performance evidence (the C block
adds its own frame and is never used in the screen).
Exit status 1 unless every gate passes: the screen does not start.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PB83 = ROOT / "research" / "coreml-prebind-predict" / "scripts"
sys.path.insert(0, str(PB83))
sys.path.insert(0, str(HERE))

import check_prebind  # noqa: E402  #83's: rows_for, same
import crossings  # noqa: E402
import prebind  # noqa: E402
import prebind_async  # noqa: E402
import probe  # noqa: E402

MODEL = "laya"
REPEAT = 1000
CHECK_EVERY = 100
RSS_MAX_MB = 64.0
GATE7_BUCKET = 128
SUBMIT_MAX_MS = 1.0
SUBMIT_MAX_FRACTION = 0.2
GIL_PREDICTS = 200
GIL_MAX_MS = 1.0
PERIOD_NS = 1_000_000
STAMPED_PREDICTS = 200
GRACE_S = 0.5
STAMPED_LABEL = "semantics check, not performance evidence"  # the C block is Phase 0 only, never the screen
SEED = 0


def rss_mb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0]) / 1024


def dist_ms(values_ns) -> dict:
    x = np.asarray(values_ns, np.float64) / 1e6
    if not len(x):
        return {"n": 0}
    return {
        "n": int(len(x)),
        "p50_ms": float(np.percentile(x, 50)),
        "p95_ms": float(np.percentile(x, 95)),
        "p99_ms": float(np.percentile(x, 99)),
        "max_ms": float(x.max()),
        "mean_ms": float(x.mean()),
    }


def stamp_records(stamps) -> dict:
    s = np.asarray(stamps, np.int64).reshape(-1, 4)
    return {
        "submit": dist_ms(s[:, 1] - s[:, 0]),
        "callback_entry_after_submit": dist_ms(s[:, 2] - s[:, 0]),
        "waiter_handoff": dist_ms(s[:, 3] - s[:, 2]),
    }


def gil_test(model, feats, n: int) -> dict:
    """n predicts on a background thread; a 1 ms sleep-grid probe on this thread (#89's method)."""
    span, done, tid = {}, threading.Event(), {}

    def work():
        tid["id"] = threading.get_native_id()
        span["start"] = time.monotonic_ns()
        for _ in range(n):
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
    sel = [x for tk, x in zip(ticks, late) if span["start"] <= tk < span["end"]]
    lat = dist_ms(sel)
    return {"predicts": n, "ticks": len(sel), "lateness": lat, "loop_ms": (span["end"] - span["start"]) / 1e6}, tid[
        "id"
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE.parent / "raw" / "phase0.json")
    a = ap.parse_args()
    from importlib import metadata

    import laya_apple
    from laya_apple import Laya
    from laya_apple.artifacts import COMPILED, artifact_dir
    from laya_apple.backends.coreml_ane import ane_features

    rng = np.random.default_rng(SEED)
    laya = Laya.from_pretrained(MODEL, device="ane")
    be = laya.ane
    host, pad = be.host, be.pad_id
    vocab, qtypes = host.embedding.shape[0], host.type_embedding.shape[0]
    calling = {threading.get_native_id()}
    models: list = []
    buckets: dict = {}
    errors: list[str] = []
    timeouts = 0
    lower = 0

    def feats_of(row, b):
        return ane_features([row], b, 1, host.embedding, host.type_embedding, host.window(b), pad)

    for b in be.buckets:
        path = artifact_dir(be.spec, b) / COMPILED
        ct = be.models[b]
        rec: dict = {}
        try:
            sync = prebind.PrebindModel(path)
            asy = prebind_async.AsyncPrebindModel(path)
            models.append(asy)
        except Exception as e:  # the API did not respond at load (the backing check submits once)
            errors.append(f"L{b} load: {e!r}")
            timeouts += isinstance(e, TimeoutError)
            buckets[str(b)] = {"error": repr(e)}
            continue
        rec["modes"], rec["backing_evidence"] = dict(asy.modes), asy.backing_evidence
        rows = check_prebind.rows_for(b, lower, pad, vocab, qtypes, rng)
        ok = {"PB-SYNC": [True, True], "PB-ASYNC": [True, True]}
        try:
            for row in rows:
                feats = feats_of(row, b)
                ref = ct.predict(feats)
                be.models[b] = ct
                logits_ref, act_ref = be.forward([row])
                for name, m in (("PB-SYNC", sync), ("PB-ASYNC", asy)):
                    ok[name][0] &= check_prebind.same(ref, m.predict(feats))
                    be.models[b] = m
                    logits, act = be.forward([row])
                    be.models[b] = ct
                    ok[name][1] &= check_prebind.same({"l": logits_ref, "a": act_ref}, {"l": logits, "a": act})
        except Exception as e:
            errors.append(f"L{b} correctness: {e!r}")
            timeouts += isinstance(e, TimeoutError)
            ok = {k: [False, False] for k in ok}
        be.models[b] = ct
        rec.update({f"{n}_bit_identical": bool(v[0]) for n, v in ok.items()})
        rec.update({f"{n}_forward_bit_identical": bool(v[1]) for n, v in ok.items()})
        feats = feats_of(rows[1], b)
        n_cb = asy._completion().callbacks
        rec["crossings_per_forward"] = crossings.count(asy.predict, feats)[1]
        rec["callbacks_per_forward"] = asy._completion().callbacks - n_cb

        # gate 5: 1000 forwards on generated rows
        rep = [
            r
            for _ in range(REPEAT // check_prebind.ROWS)
            for r in check_prebind.rows_for(b, lower, pad, vocab, qtypes, rng)[1:]
        ]
        lower = b
        n0 = len(prebind_async.STAMPS)
        rss = [rss_mb()]
        checked, identical, done = 0, True, 0
        t0 = time.monotonic()
        try:
            for i, row in enumerate(rep[:REPEAT]):
                be.models[b] = asy
                logits, act = be.forward([row])
                done += 1
                if (i + 1) % CHECK_EVERY == 0:
                    be.models[b] = ct
                    logits_ref, act_ref = be.forward([row])
                    identical &= check_prebind.same({"l": logits_ref, "a": act_ref}, {"l": logits, "a": act})
                    checked += 1
                    rss.append(rss_mb())
        except Exception as e:
            errors.append(f"L{b} repeat forward {done}: {e!r}")
            timeouts += isinstance(e, TimeoutError)
        be.models[b] = ct
        rec["repeat"] = {
            "forwards": done,
            "seconds": time.monotonic() - t0,
            "checked": checked,
            "checked_bit_identical": bool(identical and checked == REPEAT // CHECK_EVERY),
            "rss_mb": rss,
            "rss_growth_mb": rss[-1] - rss[0],
            "stamps": stamp_records(prebind_async.STAMPS[n0:]),
        }
        if b == GATE7_BUCKET:
            rec["repeat"]["_stamps_range"] = (n0, len(prebind_async.STAMPS))
        buckets[str(b)] = rec
        print(MODEL, f"L{b}", json.dumps({k: v for k, v in rec.items() if k != "backing_evidence"}), flush=True)
        if b == GATE7_BUCKET:
            gil_model, gil_feats = asy, feats

    gil, stamped = None, None
    if str(GATE7_BUCKET) in buckets and "repeat" in buckets[str(GATE7_BUCKET)]:
        try:
            gil, tid = gil_test(gil_model, gil_feats, GIL_PREDICTS)
            calling.add(tid)
        except Exception as e:
            errors.append(f"GIL test: {e!r}")
            timeouts += isinstance(e, TimeoutError)
        try:  # PB-ASYNC-STAMPED, non-gating
            path = artifact_dir(be.spec, GATE7_BUCKET) / COMPILED
            sm = prebind_async.StampedAsyncPrebindModel(path)
            models.append(sm)
            ns = len(prebind_async.StampedAsyncPrebindModel.NATIVE)
            for _ in range(STAMPED_PREDICTS):
                sm.predict(gil_feats)
            nat = np.asarray(prebind_async.StampedAsyncPrebindModel.NATIVE[ns:], np.int64).reshape(-1, 2)
            stamped = {
                "label": STAMPED_LABEL,
                "built": True,
                "predicts": int(len(nat)),
                "native_completion_to_callback_entry": dist_ms(nat[:, 1] - nat[:, 0]),
                "ordered": bool((nat[:, 1] >= nat[:, 0]).all()),
            }
        except Exception as e:
            stamped = {
                "label": STAMPED_LABEL,
                "built": False,
                "reason": f"native completion could not be stamped separately: {e!r}",
            }
    time.sleep(GRACE_S)  # a late or duplicate callback lands here at the latest

    submits = sum(m._completion().submits for m in models)
    cbs = prebind_async.CALLBACKS
    cb_errors = sum(e for _, _, e in cbs)
    cb_tids = sorted({t for _, t, _ in cbs})
    g7 = buckets.get(str(GATE7_BUCKET), {}).get("repeat")
    g7_submit = g7_ratio = None
    if g7:
        lo, hi = g7.pop("_stamps_range")
        s = np.asarray(prebind_async.STAMPS[lo:hi], np.int64).reshape(-1, 4)
        g7_submit = float(np.median(s[:, 1] - s[:, 0])) / 1e6
        g7_cb = float(np.median(s[:, 2] - s[:, 0])) / 1e6
        g7_ratio = g7_submit / g7_cb if g7_cb else None
    ok_b = [x for x in buckets.values() if "error" not in x]
    gates = {
        "1_callable": bool(models) and timeouts == 0 and len(ok_b) == len(buckets) and not errors,
        "2_correctness": len(ok_b) == len(buckets)
        and all(
            x[f"{n}_bit_identical"] and x[f"{n}_forward_bit_identical"] for x in ok_b for n in ("PB-SYNC", "PB-ASYNC")
        ),
        "3_backings": len(ok_b) == len(buckets) and all(set(x["modes"].values()) == {"backed"} for x in ok_b),
        "4_callbacks": submits == len(cbs) and cb_errors == 0 and not prebind_async.ANOMALIES and submits > 0,
        "5_repeat": len(ok_b) == len(buckets)
        and all(
            x["repeat"]["forwards"] == REPEAT
            and x["repeat"]["checked_bit_identical"]
            and x["repeat"]["rss_growth_mb"] <= RSS_MAX_MB
            for x in ok_b
        ),
        "6_threading": bool(cb_tids) and not (set(cb_tids) & calling),
        "7_submit_returns_early": g7_submit is not None
        and g7_ratio is not None
        and g7_submit < SUBMIT_MAX_MS
        and g7_ratio < SUBMIT_MAX_FRACTION,
        "8_gil": gil is not None
        and gil["lateness"].get("p50_ms") is not None
        and gil["lateness"]["p50_ms"] < GIL_MAX_MS,
    }
    res = {
        "model": MODEL,
        "seed": SEED,
        "laya_apple": laya_apple.__version__,
        "pyobjc": metadata.version("pyobjc-framework-CoreML"),
        "coremltools": metadata.version("coremltools"),
        "macos": subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip(),
        "limits": {
            "timeout_s": prebind_async.TIMEOUT_S,
            "repeat": REPEAT,
            "check_every": CHECK_EVERY,
            "rss_max_mb": RSS_MAX_MB,
            "submit_max_ms": SUBMIT_MAX_MS,
            "submit_max_fraction": SUBMIT_MAX_FRACTION,
            "gil_predicts": GIL_PREDICTS,
            "gil_max_ms": GIL_MAX_MS,
            "grace_s": GRACE_S,
        },
        "buckets": buckets,
        "errors": errors,
        "timeouts": timeouts,
        "submits": submits,
        "callbacks": len(cbs),
        "callback_errors": cb_errors,
        "anomalies": prebind_async.ANOMALIES,
        "calling_thread_ids": sorted(calling),
        "callback_thread_ids": cb_tids,
        "callback_queue_labels": dict(prebind_async.LABELS),
        "gate7": {"bucket": GATE7_BUCKET, "submit_p50_ms": g7_submit, "submit_over_callback_entry_p50": g7_ratio},
        "gil_test": gil,
        "stamped": stamped,
        "all_stamps": stamp_records(prebind_async.STAMPS),
        "gates": gates,
        "all_pass": all(gates.values()),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for k, v in gates.items():
        print(f"gate {k}: {'PASS' if v else 'FAIL'}")
    print("all gates pass:", res["all_pass"])
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
