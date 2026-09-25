"""Pure derivations for the non-gating records (criteria.md). NumPy only, no I/O; unit-tested
in tests/unit/test_prebind_predict.py and used by analyze.py.

Time values are integer time.monotonic_ns() throughout.
"""

from __future__ import annotations

import numpy as np

# The fixed exposure window of the collision analysis (criteria.md, "Prior evidence"): an ANE
# request collides if a GPU completion (parent-side received_ns) falls in
# [service_start - BEFORE_NS, service_start + AFTER_NS], both ends included.
BEFORE_NS = 1_000_000
AFTER_NS = 300_000

STAGES = ("features", "pre", "native", "reacquire", "post", "tail")


def in_windows(t, bounds) -> np.ndarray:
    """Boolean mask: t inside any [start, end) of bounds."""
    t = np.asarray(t, dtype=np.int64)
    mask = np.zeros(t.shape, bool)
    for lo, hi in bounds:
        mask |= (t >= lo) & (t < hi)
    return mask


def window_index(t, bounds) -> np.ndarray:
    """Index into bounds of the [start, end) window containing each t, or -1."""
    t = np.asarray(t, dtype=np.int64)
    idx = np.full(t.shape, -1, np.int64)
    for i, (lo, hi) in enumerate(bounds):
        idx[(t >= lo) & (t < hi)] = i
    return idx


def collide_fixed(starts, completions, before_ns: int = BEFORE_NS, after_ns: int = AFTER_NS) -> np.ndarray:
    """For each ANE service start: is any GPU completion in [start - before, start + after]?"""
    s = np.asarray(starts, dtype=np.int64)
    c = np.sort(np.asarray(completions, dtype=np.int64))
    lo = np.searchsorted(c, s - before_ns, side="left")
    hi = np.searchsorted(c, s + after_ns, side="right")
    return hi > lo


def overlap_counts(spans, intervals) -> np.ndarray:
    """For each span [a, b): how many intervals [c, d] overlap it (c < b and d > a).

    Every interval that starts before b either ends by a (no overlap) or overlaps, so the
    count is #(c < b) - #(d <= a). Intervals must have c <= d.
    """
    spans = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    iv = np.asarray(intervals, dtype=np.int64).reshape(-1, 2)
    if (iv[:, 0] > iv[:, 1]).any():
        raise ValueError("interval ends before it starts")
    starts, ends = np.sort(iv[:, 0]), np.sort(iv[:, 1])
    return np.searchsorted(starts, spans[:, 1], side="left") - np.searchsorted(ends, spans[:, 0], side="right")


def odds_ratio(exposed, outcome) -> float:
    """Odds ratio of outcome given exposure, 0.5 added to every cell of the 2x2 table."""
    e, o = np.asarray(exposed, bool), np.asarray(outcome, bool)
    a = np.sum(e & o) + 0.5
    b = np.sum(e & ~o) + 0.5
    c = np.sum(~e & o) + 0.5
    d = np.sum(~e & ~o) + 0.5
    return float((a * d) / (b * c))


def tail_association(latency_ns, collide, threshold_ns) -> dict:
    """Tail = latency above the threshold (the production configuration's P99)."""
    lat = np.asarray(latency_ns, dtype=np.int64)
    col = np.asarray(collide, bool)
    tail = lat > threshold_ns
    n, nt = len(lat), int(tail.sum())
    return {
        "n": n,
        "tail_n": nt,
        "tail_fraction": nt / n if n else None,
        "collide_fraction": float(col.mean()) if n else None,
        "tail_colliding_fraction": float(col[tail].mean()) if nt else None,
        "odds_ratio": odds_ratio(col, tail) if n else None,
    }


def join_predicts(forwards, predicts) -> np.ndarray:
    """For each forward [service_start, service_end], the index of the first predict record
    whose entry stamp lies inside it, or -1. predicts: rows whose column 0 is the entry stamp,
    in time order (one ANE thread appends them)."""
    f = np.asarray(forwards, dtype=np.int64).reshape(-1, 2)
    p = np.asarray(predicts, dtype=np.int64)
    if not len(p):
        return np.full(len(f), -1, np.int64)
    entry = p[:, 0]
    i = np.searchsorted(entry, f[:, 0], side="left")
    ok = (i < len(entry)) & (entry[np.minimum(i, len(entry) - 1)] <= f[:, 1])
    return np.where(ok, i, -1)


def stages(forward, predict) -> dict:
    """Stage durations (ns) of one forward.

    forward: (service_start, service_end). predict: (entry, python_before, native_before,
    native_after, python_after, exit); the four middle stamps are 0 when the binding has no
    native stamps (coremltools, which holds the GIL), and then pre/native/reacquire/post are
    None and `predict` is the whole call.
    """
    s0, s1 = int(forward[0]), int(forward[1])
    entry, _pb, nb, na, pa, exit_ = (int(x) for x in predict)
    out = {"features": entry - s0, "tail": s1 - exit_, "predict": exit_ - entry}
    if nb and na and pa:
        out.update(pre=nb - entry, native=na - nb, reacquire=pa - na, post=exit_ - pa)
    else:
        out.update(pre=None, native=None, reacquire=None, post=None)
    return out


def lateness_summary(late_ns) -> dict:
    x = np.asarray(late_ns, dtype=np.float64) / 1e6
    if not len(x):
        return {"n": 0, "p50_ms": None, "p99_ms": None, "mean_ms": None, "over_1ms_fraction": None}
    return {
        "n": int(len(x)),
        "p50_ms": float(np.percentile(x, 50)),
        "p99_ms": float(np.percentile(x, 99)),
        "mean_ms": float(x.mean()),
        "over_1ms_fraction": float((x > 1.0).mean()),
    }


SLOW_CPU_RATIO = 1.25


def slow_cpu(window_cpu_ns, reference_cpu_ns, limit: float = SLOW_CPU_RATIO) -> dict:
    """Median CPU per forward in a window over the median of the reference (solo_short) windows."""
    w, r = np.asarray(window_cpu_ns, np.float64), np.asarray(reference_cpu_ns, np.float64)
    if not len(w) or not len(r) or np.median(r) <= 0:
        return {"ratio": None, "flag": None}
    ratio = float(np.median(w) / np.median(r))
    return {"ratio": ratio, "flag": ratio > limit}


def thread_cpu_delta(before: dict, after: dict) -> dict:
    """Per-thread-name CPU ns between two probe.snapshot() results (a thread that appears only
    after counts from 0; a thread that exits in between is dropped)."""
    b = before["threads"]
    return {k: v - b.get(k, 0) for k, v in after["threads"].items()}
