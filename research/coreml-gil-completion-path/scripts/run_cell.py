"""The gpu-ane-interference harness with two research-only additions (nothing else changes):

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML python \
        research/coreml-gil-completion-path/scripts/run_cell.py --ane-predict pyobjc \
        --model laya-typed-decisions --ane-placement thread --plan device --parts \
        --cells solo:gpu_M solo:ane_B matrix:gpu_M+ane_B --out ...

  --ane-predict coremltools   the product path (default)
  --ane-predict pyobjc        a thread-placed ANE backend predicts through nogil_predict.py:
                              the same compiled model and inputs, with the GIL released
  (always)                    the GPU worker's reply is stamped in the parent (boundary.py)

All other arguments go to research/gpu-ane-interference/scripts/interference.py unchanged, and
the output is that harness's raw format plus a `completion_path` section:
{"ane_predict": ..., "stamps": {"request_id", "recv_enter_us", "header_us", "body_us",
"loaded_us"}} on the run's t_ref_ns axis.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "gpu-ane-interference" / "scripts"))

import boundary  # noqa: E402
import interference  # noqa: E402


def main():
    argv, variant = list(sys.argv[1:]), "coremltools"
    if "--ane-predict" in argv:
        i = argv.index("--ane-predict")
        variant = argv[i + 1]
        del argv[i : i + 2]
    if variant not in ("coremltools", "pyobjc"):
        raise SystemExit("--ane-predict must be coremltools or pyobjc")
    sys.argv = [sys.argv[0], *argv]
    stamps: list = []
    init, finish = interference.Run.__init__, interference.Run.finish

    def run_init(self, args):
        init(self, args)
        attached = sum(boundary.attach(laya, stamps) for laya in self.lays.values())
        if attached != 1:
            raise SystemExit(f"expected one process-placed GPU worker, found {attached}")
        if variant == "pyobjc":
            import nogil_predict

            ane = [w for laya in self.lays.values() for w in laya._workers.values() if w.kind == "ane"]
            if len(ane) != 1 or ane[0].placement != "thread":
                raise SystemExit("--ane-predict pyobjc needs exactly one thread-placed ANE worker")
            nogil_predict.install(ane[0]._backend)

    def run_finish(self):
        record = finish(self)
        us = self._us
        rows = [(j, us(a), us(b), us(c), us(d)) for j, a, b, c, d in stamps]
        keys = ("request_id", "recv_enter_us", "header_us", "body_us", "loaded_us")
        record["completion_path"] = {
            "ane_predict": variant,
            "stamps": {k: [r[i] for r in rows] for i, k in enumerate(keys)},
        }
        return record

    interference.Run.__init__, interference.Run.finish = run_init, run_finish
    interference.main()


if __name__ == "__main__":
    main()
