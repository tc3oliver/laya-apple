"""Run a research/ane-w8 script with its RAW output redirected to raw/build/ here.

    run_w8.py build_w8|latency [script args]

ane-w8's scripts write build.json / read it from common.RAW (research/ane-w8/raw, historical data).
This patches common.RAW before the script runs so that directory is never touched.
"""

import runpy
import sys
from pathlib import Path

W8 = Path(__file__).resolve().parents[2] / "ane-w8" / "scripts"
sys.path.insert(0, str(W8))
import common  # noqa: E402

common.RAW = Path(__file__).resolve().parents[1] / "raw" / "build"
name = sys.argv.pop(1)
sys.argv[0] = str(W8 / f"{name}.py")
runpy.run_path(sys.argv[0], run_name="__main__")
