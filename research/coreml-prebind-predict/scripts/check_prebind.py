"""The pre-campaign check (criteria.md): PB and C against coremltools, bit for bit, on every ANE
bucket of all three models; each PB output's backing mode; crossings per forward for C and PB.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
        research/coreml-prebind-predict/scripts/check_prebind.py [--out research/coreml-prebind-predict/raw/check.json]

Per model, the ANE backend loads through the runtime (Laya.from_pretrained(device="ane"): the
verified artifacts, coremltools, every bucket in the model's ane_buckets). Per bucket, the
warm-up row plus ROWS generated rows (different ids, lengths, markers and question types, so
PB's reused buffers must change between calls) are run through coremltools, C (#77's binding,
unchanged) and PB on the same model.mlmodelc:
  - model level: every output, same names, dtype and shape, bytes equal;
  - forward level: ANEBackend.forward's logits and actions with PB swapped in, bytes equal;
  - stamps ordered: python_before <= native_before <= native_after <= python_after.
Exit status 1 if PB is not bit-identical everywhere: the campaign does not start.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MIX77 = ROOT / "research" / "coreml-nogil-product-mix" / "scripts"
sys.path.insert(0, str(MIX77))
sys.path.insert(0, str(HERE))

import crossings  # noqa: E402
import nogil  # noqa: E402  #77's binding, unchanged
import prebind  # noqa: E402

MODELS = ("laya", "laya-typed-decisions", "laya-multilingual")
ROWS = 10
SEED = 0


def same(x: dict, y: dict) -> bool:
    if set(x) != set(y):
        return False
    for k in x:
        a, b = np.asarray(x[k]), np.asarray(y[k])
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if not np.array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8)):
            return False
    return True


def rows_for(b: int, lower: int, pad: int, vocab: int, qtypes: int, rng) -> list[dict]:
    """The warm-up row, then ROWS rows whose length selects bucket b (lower < length <= b)."""
    from laya_apple.registry import ANE_MAX_OPTIONS

    rows = [{"ids": [pad] * b, "markers": [1, 2], "qtype": 0}]
    for _ in range(ROWS):
        n = int(rng.integers(max(lower + 1, 3), b + 1))
        k = int(rng.integers(2, min(ANE_MAX_OPTIONS, n - 1) + 1))
        ids = rng.integers(0, vocab, n).tolist()
        markers = sorted(rng.choice(np.arange(1, n), size=k, replace=False).tolist())
        rows.append({"ids": ids, "markers": markers, "qtype": int(rng.integers(0, qtypes))})
    return rows


def check_model(model_id: str, rng) -> dict:
    from laya_apple import Laya
    from laya_apple.artifacts import COMPILED, artifact_dir
    from laya_apple.backends.coreml_ane import ane_features

    laya = Laya.from_pretrained(model_id, device="ane")
    be = laya.ane
    host, pad = be.host, be.pad_id
    vocab, qtypes = host.embedding.shape[0], host.type_embedding.shape[0]
    res = {"buckets": {}}
    lower = 0
    for b in be.buckets:
        path = artifact_dir(be.spec, b) / COMPILED
        ct, c, pb = be.models[b], nogil.StampedNoGilModel(path), prebind.PrebindModel(path)
        rows = rows_for(b, lower, pad, vocab, qtypes, rng)
        lower = b
        model_c = model_pb = fwd_pb = True
        count_c = count_pb = None
        for i, row in enumerate(rows):
            feats = ane_features([row], b, 1, host.embedding, host.type_embedding, host.window(b), pad)
            ref = ct.predict(feats)
            if i == 1:  # a generated row, after the warm-up row
                out_c, count_c = crossings.count(c.predict, feats)
                out_pb, count_pb = crossings.count(pb.predict, feats)
            else:
                out_c, out_pb = c.predict(feats), pb.predict(feats)
            model_c &= same(ref, out_c)
            model_pb &= same(ref, out_pb)
            be.models[b] = ct
            logits_ref, act_ref = be.forward([row])
            be.models[b] = pb
            logits_pb, act_pb = be.forward([row])
            be.models[b] = ct
            fwd_pb &= same({"l": logits_ref, "a": act_ref}, {"l": logits_pb, "a": act_pb})
        ordered = {
            name: all(s[0] <= s[1] <= s[2] <= s[3] for s in mod.STAMPS) for name, mod in (("C", nogil), ("PB", prebind))
        }
        res["buckets"][str(b)] = {
            "rows": len(rows),
            "C_bit_identical": bool(model_c),
            "PB_bit_identical": bool(model_pb),
            "PB_forward_bit_identical": bool(fwd_pb),
            "stamps_ordered": ordered,
            "crossings": {"C": count_c, "PB": count_pb},
            "PB_backings": {"modes": pb.modes, "evidence": pb.backing_evidence},
        }
        print(model_id, f"L{b}", json.dumps(res["buckets"][str(b)]), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE.parent / "raw" / "check.json")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    a = ap.parse_args()
    from importlib import metadata

    import laya_apple

    rng = np.random.default_rng(SEED)
    res = {
        "seed": SEED,
        "rows_per_bucket": ROWS + 1,
        "laya_apple": laya_apple.__version__,
        "pyobjc": metadata.version("pyobjc-framework-CoreML"),
        "coremltools": metadata.version("coremltools"),
        "models": {m: check_model(m, rng) for m in a.models},
    }
    buckets = [x for m in res["models"].values() for x in m["buckets"].values()]
    res["PB_bit_identical_everywhere"] = all(x["PB_bit_identical"] and x["PB_forward_bit_identical"] for x in buckets)
    res["C_bit_identical_everywhere"] = all(x["C_bit_identical"] for x in buckets)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print("PB bit-identical everywhere:", res["PB_bit_identical_everywhere"])
    print("C bit-identical everywhere:", res["C_bit_identical_everywhere"])
    sys.exit(0 if res["PB_bit_identical_everywhere"] else 1)


if __name__ == "__main__":
    main()
