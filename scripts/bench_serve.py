"""`laya-apple serve` decisions beside a local LLM that generates at saturation.

    # the whole campaign (benchmarks/serve/run.sh wraps this)
    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 OMLX_SETTINGS=<oMLX base path>/settings.json \
        uv run python scripts/bench_serve.py campaign --out benchmarks/serve/<campaign>

    # a few minutes, low load, one short window per cell; never committed
    ... uv run python scripts/bench_serve.py campaign --smoke --out <scratch dir>

Methodology, cells and criteria: benchmarks/serve/README.md and benchmarks/serve/criteria.json.

What runs:
- **Decisions.** N plugin-like clients, each its own HTTP client, POST /v1/systemone to a
  `laya-apple serve` this script starts itself (`laya-apple --offline serve`) on a free
  loopback port. Arrivals are open loop: every client has its own seeded Poisson timetable,
  identical in every window. Each arrival is a short single-question decision (ANE-eligible
  in `--device auto`) or a mixed choice/score/noul call (three questions, always the GPU).
- **LLM load.** K closed-loop streaming chat completions with a fixed prompt, fixed
  `max_tokens` and temperature 0 against an OpenAI-compatible server (oMLX), run in a
  separate process so its stream parsing never delays the decision clients.
- **Cells.** `llm_alone` (no serve process), and for serve `--device gpu` and `--device auto`:
  `decisions` (LLM idle) and `decisions_llm` (LLM at saturation). Serve blocks run in ABBA
  order per round; the window order inside a block alternates.

Correctness: every decision is compared with the answer the same serve process gave the same
request unloaded, before the block's measured windows (FP16 gate: probability error <= 0.02,
0 hard mismatches, near-tie flips listed).

The LLM server's API key, when it needs one, is read at run time from `$LLM_API_KEY` or from
the `auth.api_key` field of the JSON settings file named by `--llm-settings` /
`$OMLX_SETTINGS`. It is sent only in the Authorization header and never written or printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FP16_TOL = 0.02
CONFIGS = ("gpu", "auto")
WINDOW_KINDS = ("llm_alone", "decisions", "decisions_llm")
CLASSES = ("short_1q", "mixed_3q")
LLM_PROMPT = (
    "Write a step-by-step technical explanation of how an open-addressing hash table handles "
    "insertion, lookup, deletion with tombstones, and resizing. Use numbered sections and "
    "include small worked examples with integer keys."
)

# ---------------------------------------------------------------------------------------
# Pure logic (unit-tested in tests/unit/test_bench_serve.py)
# ---------------------------------------------------------------------------------------


def percentile(values, q: float) -> float | None:
    """numpy's linear-interpolation percentile, as every other benchmark here; None if empty."""
    a = np.asarray(values, np.float64)
    return float(np.percentile(a, q)) if a.size else None


def latency_stats(values_ms) -> dict:
    a = np.asarray(values_ms, np.float64)
    if a.size == 0:
        return {"n": 0, "p50_ms": None, "p99_ms": None, "max_ms": None}
    return {"n": int(a.size), "p50_ms": percentile(a, 50), "p99_ms": percentile(a, 99), "max_ms": float(a.max())}


def client_schedule(
    clients: int, rate_per_client: float, seconds: float, seed: int, mix: dict[str, float], variants: dict[str, int]
) -> list[dict]:
    """Open-loop timetable: every client's own Poisson arrivals (seed + client), each arrival
    assigned a class by `mix` and a request variant, from that client's own seeded RNG.
    Sorted by time. Deterministic for a seed, so every window replays the same timetable."""
    from laya_apple.workload import arrivals

    names = list(mix)
    weights = [mix[n] for n in names]
    out = []
    for c in range(clients):
        rng = random.Random(seed * 1000 + c)
        for t in arrivals(rate_per_client, seconds, seed=seed * 1000 + c, bursty=False):
            cls = rng.choices(names, weights)[0]
            out.append({"t": t, "client": c, "cls": cls, "variant": rng.randrange(variants[cls])})
    out.sort(key=lambda a: (a["t"], a["client"]))
    return out


def answer_vector(ans: dict) -> tuple[list[float], int]:
    """(probability vector, decided index) of one upstream-format answer."""
    kind = ans.get("type")
    if kind in ("choice", "score"):
        probs = [float(v) for v in ans["probabilities"].values()]
        if kind == "choice":
            return probs, list(ans["probabilities"]).index(ans["choice"])
        return probs, int(np.argmax(probs))
    if kind == "noul":
        p = float(ans["noul"])
        v = [1.0 - p, p]
        return v, int(np.argmax(v))
    raise ValueError(f"unknown answer type {kind!r}")


def compare_answers(ref: dict, got: dict, tol: float = FP16_TOL) -> dict:
    """The FP16 gate between two `answers` objects: max probability error (answer
    probabilities and act_probability), hard mismatches (decision differs and the reference
    top-1/top-2 margin >= 2 * tol) and near-tie flips (differs inside that band). A missing
    question or a changed answer type is a hard mismatch."""
    prob_err = act_err = 0.0
    hard, flips = [], []
    for qid, r in ref.items():
        g = got.get(qid) if isinstance(got, dict) else None
        if not isinstance(g, dict) or g.get("type") != r.get("type"):
            hard.append(qid)
            continue
        rv, rd = answer_vector(r)
        gv, gd = answer_vector(g)
        if len(rv) != len(gv):
            hard.append(qid)
            continue
        prob_err = max(prob_err, float(np.max(np.abs(np.subtract(rv, gv)))))
        ra, ga = (r.get("action") or {}).get("act_probability"), (g.get("action") or {}).get("act_probability")
        if ra is not None and ga is not None:
            act_err = max(act_err, abs(float(ra) - float(ga)))
        if rd != gd:
            srt = sorted(rv)
            margin = srt[-1] - srt[-2] if len(srt) > 1 else 1.0
            (hard if margin >= 2 * tol else flips).append(qid)
    extra = sorted(set(got or {}) - set(ref))
    hard += extra
    return {"prob_err": prob_err, "act_err": act_err, "hard": hard, "flips": flips}


def tokens_in_window(chunks: list[tuple[int, int]], completion_tokens: int | None, start_ns: int, end_ns: int) -> float:
    """Completion tokens of one streamed request attributed to [start_ns, end_ns).

    The server batches several tokens into one SSE chunk, so each chunk's share of the
    request's `completion_tokens` is its share of the streamed text (content + reasoning)."""
    total = sum(n for _, n in chunks)
    if not total or not completion_tokens:
        return 0.0
    inside = sum(n for t, n in chunks if start_ns <= t < end_ns)
    return completion_tokens * inside / total


def coverage(intervals: list[tuple[int, int]], start_ns: int, end_ns: int) -> float:
    """Fraction of [start_ns, end_ns) covered by the union of the intervals."""
    span = end_ns - start_ns
    if span <= 0:
        return 0.0
    clipped = sorted((max(a, start_ns), min(b, end_ns)) for a, b in intervals if b > start_ns and a < end_ns)
    covered, cur_a, cur_b = 0, None, None
    for a, b in clipped:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                covered += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        covered += cur_b - cur_a
    return covered / span


def plan(rounds: int, smoke: bool = False) -> list[dict]:
    """The campaign's step list. Full: per round an ABBA order of serve blocks (round 0:
    gpu, auto, auto, gpu; round 1: auto, gpu, gpu, auto; ...), an `llm_alone` window before,
    between and after the blocks, and inside a block the window order alternates
    (decisions, decisions_llm) / (decisions_llm, decisions) with the block's position.
    Smoke: one window per cell (llm_alone, gpu x 2, auto x 2)."""
    if smoke:
        return [
            {"kind": "llm_alone"},
            {"kind": "block", "config": "gpu", "windows": ["decisions", "decisions_llm"]},
            {"kind": "block", "config": "auto", "windows": ["decisions", "decisions_llm"]},
        ]
    steps: list[dict] = [{"kind": "llm_alone"}]
    for r in range(rounds):
        order = ["gpu", "auto", "auto", "gpu"] if r % 2 == 0 else ["auto", "gpu", "gpu", "auto"]
        for i, cfg in enumerate(order):
            windows = ["decisions", "decisions_llm"] if i % 2 == 0 else ["decisions_llm", "decisions"]
            steps.append({"kind": "block", "config": cfg, "windows": windows})
            steps.append({"kind": "llm_alone"})
    return steps


def estimate_seconds(steps: list[dict], a) -> float:
    """Rough wall time of a plan: windows, warm-ups, LLM drain and serve start-up."""
    llm_req_s = a.max_tokens / 25.0  # a conservative decode rate for a local 27B-class model
    llm_window = a.llm_warmup + a.window + a.llm_streams * llm_req_s + a.cooldown
    dec_window = a.decision_warmup + a.window + 2 + a.cooldown
    total = 0.0
    for s in steps:
        if s["kind"] == "llm_alone":
            total += llm_window
        else:
            total += (90 if s["config"] == "auto" else 45) + 15  # start-up, warm-up + references
            total += sum(llm_window if w == "decisions_llm" else dec_window for w in s["windows"])
    return total


def read_api_key(settings: str | None, env: dict | None = None) -> str | None:
    """The LLM server's key: $LLM_API_KEY, else `auth.api_key` of a JSON settings file."""
    env = os.environ if env is None else env
    if env.get("LLM_API_KEY"):
        return env["LLM_API_KEY"]
    if settings:
        data = json.loads(Path(settings).expanduser().read_text())
        key = (data.get("auth") or {}).get("api_key")
        return key or None
    return None


# ---------------------------------------------------------------------------------------
# Machine state
# ---------------------------------------------------------------------------------------


def conditions() -> dict:
    env = dict(os.environ, LC_ALL="C")
    top = subprocess.run(["ps", "-Ao", "%cpu=,comm="], capture_output=True, text=True, env=env).stdout.splitlines()
    busy = sorted((line.split(None, 1) for line in top if line.strip()), key=lambda x: -float(x[0]))[:5]
    therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout.strip()
    return {
        "loadavg": os.getloadavg(),
        "top_cpu": [(float(c), n.rsplit("/", 1)[-1]) for c, n in busy],
        "therm": therm,
    }


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------------------
# LLM load (its own process: `bench_serve.py llm-load`)
# ---------------------------------------------------------------------------------------


def _llm_headers(key: str | None) -> dict:
    return {"Authorization": f"Bearer {key}"} if key else {}


async def _llm_stream(client, url: str, body: dict, rec: dict) -> None:
    rec["start_ns"] = time.monotonic_ns()
    chunks = rec["chunks"] = []
    try:
        async with client.stream("POST", url, json=body) as r:
            rec["status"] = r.status_code
            if r.status_code != 200:
                await r.aread()
                return
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                j = json.loads(data)
                if j.get("usage"):
                    u = j["usage"]
                    rec["usage"] = {
                        k: u.get(k)
                        for k in (
                            "prompt_tokens",
                            "completion_tokens",
                            "time_to_first_token",
                            "generation_duration",
                            "generation_tokens_per_second",
                        )
                    }
                    rec["usage"]["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    n = sum(len(d.get(k) or "") for k in ("content", "reasoning_content", "reasoning"))
                    if n:
                        chunks.append((time.monotonic_ns(), n))
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
    except asyncio.CancelledError:
        rec["aborted"] = True
        raise
    except Exception as e:  # recorded, never raised: an LLM error is a result
        rec["error"] = type(e).__name__
    finally:
        rec["end_ns"] = time.monotonic_ns()


async def llm_load(a) -> dict:
    import httpx

    key = read_api_key(a.llm_settings)
    body = {
        "model": a.llm_model,
        "messages": [{"role": "user", "content": LLM_PROMPT}],
        "max_tokens": a.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = a.llm_url.rstrip("/") + "/v1/chat/completions"
    requests: list[dict] = []
    stop_ns = a.stop_at_ns
    timeout = httpx.Timeout(a.drain_timeout + 600, connect=10)

    async def stream_loop(i: int, client) -> None:
        first = True  # every stream sends at least one request (the unmeasured warm-up relies on it)
        while first or time.monotonic_ns() < stop_ns:
            first = False
            rec = {"stream": i}
            requests.append(rec)
            await _llm_stream(client, url, body, rec)
            if rec.get("status") != 200 or rec.get("error"):
                await asyncio.sleep(1.0)  # do not hammer a failing server

    async with httpx.AsyncClient(headers=_llm_headers(key), timeout=timeout) as client:
        tasks = [asyncio.create_task(stream_loop(i, client)) for i in range(a.llm_streams)]
        # no new request starts after stop_ns; in-flight ones finish (drain), so the server is
        # never left generating for a client that went away
        drain_deadline = stop_ns / 1e9 + a.drain_timeout
        done, pending = await asyncio.wait(tasks, timeout=max(1.0, drain_deadline - time.monotonic()))
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    return {"requests": requests, "drain_aborted": len(pending)}


def cmd_llm_load(a) -> None:
    out = asyncio.run(llm_load(a))
    Path(a.out).write_text(json.dumps(out) + "\n")


# ---------------------------------------------------------------------------------------
# LLM server status (oMLX /api/status): idle gate and foreign-traffic check
# ---------------------------------------------------------------------------------------


def llm_status(url: str, key: str | None) -> dict | None:
    import httpx

    try:
        r = httpx.get(url.rstrip("/") + "/api/status", headers=_llm_headers(key), timeout=5)
        if r.status_code != 200:
            return {"http_status": r.status_code}
        j = r.json()
        return {
            k: j.get(k)
            for k in ("version", "active_requests", "waiting_requests", "total_requests", "total_completion_tokens")
        }
    except Exception as e:
        return {"error": type(e).__name__}


def wait_llm_idle(url: str, key: str | None, max_wait: float) -> dict:
    """Wait until the LLM server reports no active or waiting request (someone else's
    generation would contaminate the window); give up after max_wait seconds."""
    t0 = time.monotonic()
    while True:
        s = llm_status(url, key) or {}
        if s.get("active_requests") == 0 and s.get("waiting_requests") == 0:
            s["waited_s"] = time.monotonic() - t0
            return s
        if "active_requests" not in s:  # no status endpoint: record and go on
            s["waited_s"] = time.monotonic() - t0
            s["idle_unknown"] = True
            return s
        if time.monotonic() - t0 > max_wait:
            raise SystemExit(f"LLM server busy for {max_wait:.0f} s with other requests: {s}; not measuring")
        print(f"  LLM server busy ({s.get('active_requests')} active, {s.get('waiting_requests')} waiting); waiting")
        time.sleep(5)


# ---------------------------------------------------------------------------------------
# laya-apple serve
# ---------------------------------------------------------------------------------------


def serve_command() -> list[str]:
    exe = Path(sys.executable).with_name("laya-apple")
    if exe.exists():
        return [str(exe)]
    found = shutil.which("laya-apple")
    if not found:
        raise SystemExit("cannot find the laya-apple CLI; run through `uv run` with the [serve] extra")
    return [found]


class Serve:
    """One `laya-apple --offline serve` child on a free loopback port."""

    def __init__(self, device: str, model: str, log_path: Path):
        self.device, self.model, self.port = device, model, free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        env = {k: v for k, v in os.environ.items() if k not in ("LAYA_API_KEY", "LLM_API_KEY")}
        cmd = serve_command() + [
            "--offline",
            "serve",
            "--model",
            model,
            "--device",
            device,
            "--port",
            str(self.port),
            "--log-level",
            "warning",
        ]
        self.log = open(log_path, "w")
        self.t0 = time.monotonic()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)

    def wait_ready(self, timeout: float = 600) -> dict:
        """/health up and, for --device auto, the model's ANE path `ready`."""
        import httpx

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"serve exited with {self.proc.returncode}; see {self.log.name}")
            try:
                h = httpx.get(self.base + "/health", timeout=2).json()
                ane = h.get("ane", {}).get(self.model, {})
                if self.device != "auto" or ane.get("status") == "ready":
                    h["startup_s"] = time.monotonic() - self.t0
                    return h
                if ane.get("status") == "unavailable":
                    raise RuntimeError(f"serve --device auto has no ANE path: {ane}")
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.5)
        raise RuntimeError(f"serve not ready after {timeout:.0f} s")

    def stop(self) -> int | None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)  # uvicorn shuts down gracefully, closing the models
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        self.log.close()
        return self.proc.returncode


# ---------------------------------------------------------------------------------------
# Decision clients
# ---------------------------------------------------------------------------------------


def build_requests(model: str, short_len: int, mixed_len: int, short_variants: int, mixed_variants: int) -> dict:
    """Exact-length requests from the runtime's own generator (no third-party text).
    short_1q: one question, rotating choice / score / noul across variants.
    mixed_3q: three questions of mixed types in one call."""
    from laya_apple.hub import checkpoint_path
    from laya_apple.prompt import Tokenizer
    from laya_apple.registry import resolve
    from laya_apple.workload import make_request, questions_for

    ckpt = checkpoint_path(resolve(model), local_files_only=True)
    tok = Tokenizer(ckpt / "tokenizer")
    cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
    out = {"short_1q": [], "mixed_3q": []}
    for v in range(short_variants):
        state, qs = make_request(tok, cfg, short_len, n_questions=1, seed=v)
        out["short_1q"].append({"state": state, "questions": qs, "length": short_len})
    seed = short_variants  # the question pool rotates with the seed: keep calls that mix all three types
    while len(out["mixed_3q"]) < mixed_variants:
        if {q["type"] for q in questions_for(3, offset=seed).values()} == {"choice", "score", "noul"}:
            state, qs = make_request(tok, cfg, mixed_len, n_questions=3, seed=seed)
            out["mixed_3q"].append({"state": state, "questions": qs, "length": mixed_len})
        seed += 1
    return out


HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer local-placeholder"}


def _body(req: dict) -> dict:
    return {"model": "jev-latest", "state": req["state"], "questions": req["questions"]}


async def _post(client, base: str, req: dict) -> tuple[int | None, dict | None, str | None]:
    try:
        r = await client.post(base + "/v1/systemone", json=_body(req), headers=HEADERS)
        return r.status_code, (r.json() if r.status_code == 200 else None), None
    except Exception as e:
        return None, None, type(e).__name__


async def references(base: str, reqs: dict, warm: int) -> dict:
    """Unloaded, sequential: `warm` passes over every variant, then the reference pass."""
    import httpx

    refs: dict = {cls: [] for cls in reqs}
    async with httpx.AsyncClient(timeout=60) as client:
        for _ in range(warm):
            for cls, vs in reqs.items():
                for req in vs:
                    await _post(client, base, req)
        for cls, vs in reqs.items():
            for req in vs:
                status, body, err = await _post(client, base, req)
                if status != 200:
                    raise RuntimeError(f"reference request failed: {status} {err}")
                la = body["laya_apple"]
                refs[cls].append({"answers": body["answers"], "device": la["device"], "model": la["model"]})
    return refs


async def run_decisions(base: str, clients: int, schedule: list[dict], start_ns: int, reqs: dict, refs: dict) -> list:
    """Fire the timetable open loop from start_ns; one HTTP client per plugin-like client."""
    import httpx

    records: list[dict] = []
    pool = [httpx.AsyncClient(timeout=30) for _ in range(clients)]

    async def one(a: dict, sched_ns: int) -> None:
        rec = {"client": a["client"], "cls": a["cls"], "variant": a["variant"], "sched_ns": sched_ns, "t": a["t"]}
        rec["sent_ns"] = time.monotonic_ns()
        status, body, err = await _post(pool[a["client"]], base, reqs[a["cls"]][a["variant"]])
        rec["done_ns"] = time.monotonic_ns()
        rec["status"], rec["error"] = status, err
        if body is not None:
            la = body.get("laya_apple") or {}
            ref = refs[a["cls"]][a["variant"]]
            cmp = compare_answers(ref["answers"], body.get("answers") or {})
            rec.update(
                device=la.get("device"),
                reason=la.get("routing_reason"),
                server_ms=la.get("latency_ms"),
                ref_device=ref["device"],
                prob_err=cmp["prob_err"],
                act_err=cmp["act_err"],
                hard=cmp["hard"],
                flips=cmp["flips"],
            )
        records.append(rec)

    tasks = []
    try:
        for a in schedule:
            sched_ns = start_ns + int(a["t"] * 1e9)
            delay = (sched_ns - time.monotonic_ns()) / 1e9
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(a, sched_ns)))
        await asyncio.gather(*tasks)
    finally:
        for c in pool:
            await c.aclose()
    records.sort(key=lambda r: r["sched_ns"])
    return records


def summarize_decisions(records: list[dict]) -> dict:
    out = {}
    for cls in CLASSES:
        rs = [r for r in records if r["cls"] == cls]
        ok = [r for r in rs if r.get("status") == 200]
        s = latency_stats([(r["done_ns"] - r["sched_ns"]) / 1e6 for r in ok])
        s["errors"] = len(rs) - len(ok)
        s["hard"] = sum(len(r.get("hard") or []) for r in ok)
        s["devices"] = {d: sum(r.get("device") == d for r in ok) for d in {r.get("device") for r in ok}}
        out[cls] = s
    return out


# ---------------------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------------------


def llm_process(a, stop_at_ns: int, out: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "llm-load",
        "--llm-url",
        a.llm_url,
        "--llm-model",
        a.llm_model,
        "--llm-streams",
        str(a.llm_streams),
        "--max-tokens",
        str(a.max_tokens),
        "--stop-at-ns",
        str(stop_at_ns),
        "--drain-timeout",
        str(a.drain_timeout),
        "--out",
        str(out),
    ]
    if a.llm_settings:
        cmd += ["--llm-settings", a.llm_settings]
    return subprocess.Popen(cmd)


def llm_window_summary(llm: dict, start_ns: int, end_ns: int) -> dict:
    reqs = llm["requests"]
    tokens = sum(
        tokens_in_window(r.get("chunks") or [], (r.get("usage") or {}).get("completion_tokens"), start_ns, end_ns)
        for r in reqs
    )
    gen = [(r["chunks"][0][0], r["chunks"][-1][0]) for r in reqs if r.get("chunks")]
    return {
        "tok_s": tokens / ((end_ns - start_ns) / 1e9),
        "generating_coverage": coverage(gen, start_ns, end_ns),
        "requests": len(reqs),
        "errors": sum(1 for r in reqs if r.get("error") or (r.get("status") not in (None, 200))),
        "no_usage": sum(1 for r in reqs if r.get("chunks") and not r.get("usage")),
    }


def run_window(a, kind: str, serve: Serve | None, ctx: dict, key: str | None) -> dict:
    """One measured window; returns the raw record (written by the caller)."""
    w: dict = {"kind": kind, "config": serve.device if serve else None}
    w["llm_status_before"] = wait_llm_idle(a.llm_url, key, a.idle_wait)
    w["conditions_before"] = conditions()
    schedule = ctx.get("schedule")
    llm_out = ctx["raw_dir"] / f".llm-{ctx['index']:03d}.json"
    proc = None
    now = time.monotonic_ns()
    if kind in ("llm_alone", "decisions_llm"):
        start_ns = now + int(a.llm_warmup * 1e9)
        end_ns = start_ns + int(a.window * 1e9)
        proc = llm_process(a, end_ns, llm_out)
    else:
        start_ns = now + int((a.decision_warmup + 0.5) * 1e9)
        end_ns = start_ns + int(a.window * 1e9)
    w["window_ns"] = [start_ns, end_ns]
    try:
        if kind != "llm_alone":
            # decisions warm up for decision_warmup s right before the window (excluded)
            warm = [
                dict(x, t=x["t"] - a.decision_warmup)
                for x in client_schedule(a.clients, a.rate, a.decision_warmup, a.seed + 1, ctx["mix"], ctx["variants"])
            ]
            full = warm + schedule
            recs = asyncio.run(run_decisions(serve.base, a.clients, full, start_ns, ctx["reqs"], ctx["refs"]))
            for r in recs:
                r["warmup"] = r["t"] < 0
            w["decisions"] = recs
            w["decision_summary"] = summarize_decisions([r for r in recs if not r["warmup"]])
        else:
            while time.monotonic_ns() < end_ns:
                time.sleep(0.2)
    finally:
        if proc is not None:
            try:
                proc.wait(timeout=a.window + a.llm_warmup + a.drain_timeout + 120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    if proc is not None:
        llm = json.loads(llm_out.read_text())
        llm_out.unlink()
        w["llm"] = llm
        w["llm_summary"] = llm_window_summary(llm, start_ns, end_ns)
    w["llm_status_after"] = llm_status(a.llm_url, key)
    w["conditions_after"] = conditions()
    time.sleep(a.cooldown)
    return w


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=None, separators=(",", ":")) + "\n")


def cmd_campaign(a) -> None:
    import laya_apple
    from laya_apple.artifacts import platform_profile

    if a.smoke:
        a.window, a.llm_warmup, a.decision_warmup, a.cooldown = 10.0, 8.0, 2.0, 2.0
        a.clients, a.rate, a.max_tokens = 2, 1.0, 96
    out = Path(a.out)
    raw = out / "raw"
    if raw.exists() and any(raw.iterdir()):
        raise SystemExit(f"{raw} already has data; every campaign is a new directory")
    raw.mkdir(parents=True, exist_ok=True)
    key = read_api_key(a.llm_settings)
    if not a.llm_model:
        import httpx

        a.llm_model = httpx.get(a.llm_url.rstrip("/") + "/health", timeout=5).json()["default_model"]
    steps = plan(a.rounds, a.smoke)
    est = estimate_seconds(steps, a)
    print(f"plan: {len(steps)} steps, about {est / 60:.0f} min; LLM model {a.llm_model}", flush=True)

    mix = {"short_1q": a.short_share, "mixed_3q": 1.0 - a.short_share}
    reqs = build_requests(a.model, a.short_len, a.mixed_len, a.short_variants, a.mixed_variants)
    variants = {k: len(v) for k, v in reqs.items()}
    schedule = client_schedule(a.clients, a.rate, a.window, a.seed, mix, variants)
    meta = {
        "experiment": "laya-apple serve decisions beside a local LLM at saturation",
        "smoke": a.smoke,
        "args": {k: v for k, v in vars(a).items() if k not in ("fn", "llm_settings")},
        "llm_key_source": "env LLM_API_KEY" if os.environ.get("LLM_API_KEY") else ("settings file" if key else None),
        "laya_apple": laya_apple.__version__,
        "platform": platform_profile() | {"python": platform.python_version()},
        "time": datetime.now(timezone.utc).isoformat(),
        "plan": steps,
        "estimated_s": est,
        "arrivals": "open loop: per-client seeded Poisson timetables, identical in every window",
        "schedule_n": {c: sum(x["cls"] == c for x in schedule) for c in CLASSES},
        "requests": {c: [{"length": r["length"], "questions": r["questions"]} for r in v] for c, v in reqs.items()},
        "llm_prompt": LLM_PROMPT,
        "llm_status_start": llm_status(a.llm_url, key),
    }
    write_json(raw / "campaign.json", meta)

    # one unmeasured LLM request first: a model that sat idle compiles/prefills cold
    wait_llm_idle(a.llm_url, key, a.idle_wait)
    warm_path = raw / ".llm-warm.json"
    p = llm_process(argparse.Namespace(**{**vars(a), "llm_streams": 1}), time.monotonic_ns() + 1, warm_path)
    p.wait()
    warm_path.unlink(missing_ok=True)

    t_start = time.monotonic()
    index = 0
    for bi, step in enumerate(steps):
        if step["kind"] == "llm_alone":
            ctx = {"raw_dir": raw, "index": index}
            w = run_window(a, "llm_alone", None, ctx, key)
            w.update(index=index, step=bi)
            write_json(raw / f"{index:03d}-llm_alone.json", w)
            print(f"[{index:03d}] llm_alone  tok/s {w['llm_summary']['tok_s']:.1f}", flush=True)
            index += 1
            continue
        cfg = step["config"]
        serve = Serve(cfg, a.model, raw / f"serve-step{bi:02d}-{cfg}.log")
        try:
            health = serve.wait_ready()
            refs = asyncio.run(references(serve.base, reqs, a.reference_warm))
            block = {"step": bi, "config": cfg, "health": health, "references": refs, "port": serve.port}
            write_json(raw / f"refs-step{bi:02d}-{cfg}.json", block)
            ref_dev = {c: sorted({r["device"] for r in v}) for c, v in refs.items()}
            print(f"serve {cfg} ready in {health['startup_s']:.1f} s; reference devices {ref_dev}", flush=True)
            for kind in step["windows"]:
                ctx = {
                    "raw_dir": raw,
                    "index": index,
                    "schedule": schedule,
                    "mix": mix,
                    "variants": variants,
                    "reqs": reqs,
                    "refs": refs,
                }
                w = run_window(a, kind, serve, ctx, key)
                w.update(index=index, step=bi)
                write_json(raw / f"{index:03d}-{kind}-{cfg}.json", w)
                ds = w["decision_summary"]
                line = " ".join(
                    f"{c}: P50 {ds[c]['p50_ms'] or 0:.1f} P99 {ds[c]['p99_ms'] or 0:.1f} ms "
                    f"err {ds[c]['errors']} hard {ds[c]['hard']} dev {ds[c]['devices']}"
                    for c in CLASSES
                )
                tok = f" tok/s {w['llm_summary']['tok_s']:.1f}" if "llm_summary" in w else ""
                print(f"[{index:03d}] {kind:<13} {cfg:<4} {line}{tok}", flush=True)
                index += 1
        finally:
            rc = serve.stop()
            print(f"serve {cfg} stopped ({rc})", flush=True)
    meta["elapsed_s"] = time.monotonic() - t_start
    meta["llm_status_end"] = llm_status(a.llm_url, key)
    write_json(raw / "campaign.json", meta)
    print(f"done in {meta['elapsed_s'] / 60:.1f} min: {raw}", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def llm_args(s):
        s.add_argument("--llm-url", default="http://127.0.0.1:8000")
        s.add_argument("--llm-model", default=None, help="default: the server's default_model from /health")
        s.add_argument(
            "--llm-settings",
            default=os.environ.get("OMLX_SETTINGS"),
            help="JSON file whose auth.api_key is the LLM server's key (default $OMLX_SETTINGS); or set $LLM_API_KEY",
        )
        s.add_argument("--llm-streams", type=int, default=2, help="closed-loop streams (K)")
        s.add_argument("--max-tokens", type=int, default=256)
        s.add_argument("--drain-timeout", type=float, default=120.0)

    c = sub.add_parser("campaign", help="run the plan: serve blocks, LLM load, raw JSON per window")
    llm_args(c)
    c.add_argument("--out", required=True, help="campaign directory; raw/ goes inside")
    c.add_argument("--smoke", action="store_true", help="one short low-load window per cell; not a result")
    c.add_argument("--rounds", type=int, default=2)
    c.add_argument("--model", default="laya", help="serve --model (the checkpoint the clients' requests get)")
    c.add_argument("--clients", type=int, default=8, help="plugin-like clients (N)")
    c.add_argument("--rate", type=float, default=1.0, help="Poisson arrivals per second per client")
    c.add_argument("--short-share", type=float, default=0.8)
    c.add_argument("--short-len", type=int, default=96)
    c.add_argument("--mixed-len", type=int, default=128)
    c.add_argument("--short-variants", type=int, default=7)
    c.add_argument("--mixed-variants", type=int, default=4)
    c.add_argument("--window", type=float, default=60.0)
    c.add_argument("--llm-warmup", type=float, default=15.0)
    c.add_argument("--decision-warmup", type=float, default=3.0)
    c.add_argument("--reference-warm", type=int, default=3)
    c.add_argument("--cooldown", type=float, default=5.0)
    c.add_argument("--idle-wait", type=float, default=600.0)
    c.add_argument("--seed", type=int, default=11)
    c.set_defaults(fn=cmd_campaign)

    s = sub.add_parser("llm-load", help="internal: K closed-loop streams until --stop-at-ns, then drain")
    llm_args(s)
    s.add_argument("--stop-at-ns", type=int, required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_llm_load)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
