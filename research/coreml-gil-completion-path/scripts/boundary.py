"""Research-only stamps on the parent side of the GPU worker's reply.

The runtime trace brackets the GPU reply leg with `service_end_ns` (end of the forward, in the
worker) and `received_ns` (the parent's done-callback). `StampedConnection` splits the parent
side of that leg without changing it. It replaces the GPU DeviceWorker's connection (an
instance attribute the dispatcher looks up per job) and repeats `Connection.recv` with a clock
read between its steps:

  recv_enter   the dispatcher calls recv (it has just sent the job)
  header       the 4-byte length header was read: the read syscall returned and this thread
               holds the GIL again (os.read releases it while blocked)
  body         the pickled reply was read
  loaded       the reply was unpickled; recv returns

received_ns follows after `Future.set_result` runs the runtime's done-callback. Each record is
(job_id, recv_enter, header, body, loaded) in time.monotonic_ns(), the runtime trace's clock.
"""

from __future__ import annotations

import struct
import threading
import time
from multiprocessing.reduction import ForkingPickler

now = time.monotonic_ns


class StampedConnection:
    def __init__(self, conn, records: list):
        self._conn, self.records = conn, records
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def send(self, obj):
        self._conn.send(obj)

    def recv(self):
        c = self._conn
        c._check_closed()
        c._check_readable()
        t0 = now()
        buf = c._recv(4)
        t1 = now()
        (size,) = struct.unpack("!i", buf.getvalue())
        if size == -1:
            (size,) = struct.unpack("!Q", c._recv(8).getvalue())
        buf = c._recv(size)
        t2 = now()
        obj = ForkingPickler.loads(buf.getbuffer())
        t3 = now()
        job = obj[0] if isinstance(obj, tuple) and obj and isinstance(obj[0], int) else None
        with self._lock:
            self.records.append((job, t0, t1, t2, t3))
        return obj


def attach(laya, records: list) -> bool:
    """Stamp the process-placed GPU worker of one Laya instance; False if it has none."""
    w = laya._workers.get("gpu")
    if w is None or w.placement != "process":
        return False
    w._conn = StampedConnection(w._conn, records)
    return True
