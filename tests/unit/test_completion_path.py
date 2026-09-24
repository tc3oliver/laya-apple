"""Unit tests for the research-only completion-path probes (research/coreml-gil-completion-path/)."""

from __future__ import annotations

import sys
import threading
from multiprocessing import Pipe
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "research" / "coreml-gil-completion-path" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import boundary  # noqa: E402
import completion_probe  # noqa: E402

SAMPLE = """\
Call graph:
    1552 Thread_1   DispatchQueue_1: com.apple.main-thread  (serial)
    + 1552 start  (in dyld) + 6076  [0x1]
    +   1552 lock_PyThread_acquire_lock  (in python3.12) + 56  [0x2]
    1552 Thread_2
    + 1552 thread_run  (in python3.12) + 1  [0x3]
    +   1210 os_read  (in python3.12) + 244  [0x4]
    +   ! 615 _Py_read  (in python3.12) + 132  [0x5]
    +   ! : 615 read  (in libsystem_kernel.dylib) + 8  [0x6]
    +   ! 552 _Py_read  (in python3.12) + 152  [0x7]
    +   !   552 take_gil.llvm.9825546073713290718  (in python3.12) + 488  [0x8]
    +   !     552 _pthread_cond_wait  (in libsystem_pthread.dylib) + 980  [0x9]
    +   43 os_write  (in python3.12) + 144  [0xa]
    +     43 take_gil.llvm.9825546073713290718  (in python3.12) + 488  [0xb]
    +   299 lock_PyThread_acquire_lock  (in python3.12) + 56  [0xc]
    +     299 take_gil.llvm.9825546073713290718  (in python3.12) + 488  [0xd]
    1552 Thread_3: ANEServicesThread
    + 1552 take_gil.llvm.1  (in python3.12) + 1  [0xe]
"""


def test_classify_counts_only_the_socket_readers_read_and_write_legs():
    got = completion_probe.classify(SAMPLE)
    assert got == {
        "read_syscall": 615,
        "read_then_take_gil": 552,
        "write_syscall": 0,
        "write_then_take_gil": 43,
        "reader_threads": 1,
    }  # the lock wait on the same thread and other threads' take_gil are not counted


def test_classify_without_a_reader_thread_counts_nothing():
    assert (
        completion_probe.classify("    10 Thread_9\n    + 10 read  (in libsystem_kernel.dylib)\n")["reader_threads"]
        == 0
    )


def test_stamped_connection_returns_the_same_objects_and_stamps_in_order():
    parent, child = Pipe()
    records: list = []
    conn = boundary.StampedConnection(parent, records)
    payload = (17, "ok", (np.arange(5, dtype=np.float32), None, 1, 2))
    big = (18, "ok", b"x" * (1 << 20))  # longer than one read

    def reply():
        assert child.recv() == ("job", 1)
        child.send(payload)
        child.send(big)

    t = threading.Thread(target=reply)
    t.start()
    conn.send(("job", 1))
    got = conn.recv()
    assert got[0] == 17 and got[1] == "ok" and np.array_equal(got[2][0], payload[2][0])
    assert conn.recv() == big
    t.join()
    assert [r[0] for r in records] == [17, 18]
    for _, enter, header, body, loaded in records:
        assert enter <= header <= body <= loaded
    assert conn.poll() is False  # other Connection methods pass through


class _Worker:
    def __init__(self, placement):
        self.placement, self._conn = placement, object()


class _Laya:
    def __init__(self, workers):
        self._workers = workers


@pytest.mark.parametrize(
    ("workers", "attached"),
    [({"gpu": _Worker("process")}, True), ({"ane": _Worker("thread")}, False), ({}, False)],
)
def test_attach_only_wraps_a_process_placed_gpu_worker(workers, attached):
    laya = _Laya(workers)
    assert boundary.attach(laya, []) is attached
    if attached:
        assert isinstance(laya._workers["gpu"]._conn, boundary.StampedConnection)
