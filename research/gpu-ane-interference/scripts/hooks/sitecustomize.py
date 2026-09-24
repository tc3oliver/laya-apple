"""Installs jobtrace.py's backend phase hooks in laya-apple worker processes.

Python imports `sitecustomize` at start-up from sys.path. interference.py puts this
directory on PYTHONPATH and sets LAYA_TRACE_DIR before it starts any worker, so only the
benchmark's own worker processes are affected; without LAYA_TRACE_DIR this does nothing.
"""

import os
import sys

if os.environ.get("LAYA_TRACE_DIR"):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import jobtrace  # noqa: E402  (research/gpu-ane-interference/scripts/jobtrace.py)

    jobtrace.install_in_worker()
