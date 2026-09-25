"""Run unmodified upstream `laya.serve` on the checkpoint revisions laya-apple pins.

Upstream's router loads the Hub bundle's latest revision; this launcher hands it the local
snapshot directories of the pinned standalone checkpoints instead, so both servers answer
with the same weights. It runs in upstream's own environment, not laya-apple's:

    uv run --no-project --with 'laya[serve]==0.3.20' python scripts/upstream_serve_pinned.py \
        --port 8643 english=DIR multilingual=DIR typed-decisions=DIR

Each DIR is the path `laya-apple download MODEL` prints. Only upstream's `create_app` and
`Router` are used; nothing in upstream is patched.
"""

import argparse

import uvicorn
from laya.router import Router
from laya.serve import create_app


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="+", help="NAME=DIR, NAME in english, multilingual, typed-decisions")
    ap.add_argument("--port", type=int, default=8643)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    models = dict(m.split("=", 1) for m in a.models)
    router = Router(models=models, device=a.device, max_loaded=len(models))
    router.preload(list(models))
    uvicorn.run(create_app(router=router), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
