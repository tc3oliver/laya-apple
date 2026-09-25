"""The phys_footprint reader the ANE soak test uses (tests/stress/_footprint.py)."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("_footprint", Path(__file__).parents[1] / "stress" / "_footprint.py")
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


def _task_vm_info(phys_footprint: int, count: int = fp.BUFFER_COUNT) -> bytes:
    """A task_vm_info buffer with distinct values in every 8-byte field around phys_footprint."""
    buf = bytearray(fp.BUFFER_COUNT * 4)
    for off in range(16, len(buf) - 8, 8):
        struct.pack_into("<Q", buf, off, 1000 + off)
    struct.pack_into("<Q", buf, fp.PHYS_FOOTPRINT_OFFSET, phys_footprint)
    return bytes(buf)


def test_parser_reads_phys_footprint_at_its_rev1_offset():
    assert fp.PHYS_FOOTPRINT_OFFSET == 144 and fp.REV1_COUNT == 38
    assert fp.parse_phys_footprint(_task_vm_info(123_456_789), fp.BUFFER_COUNT) == 123_456_789


def test_parser_refuses_a_struct_older_than_rev1():
    with pytest.raises(ValueError, match="needs 38"):
        fp.parse_phys_footprint(_task_vm_info(1), fp.REV1_COUNT - 1)
    with pytest.raises(ValueError, match="buffer holds"):
        fp.parse_phys_footprint(b"\0" * 100, fp.REV1_COUNT)


@pytest.mark.skipif(sys.platform != "darwin", reason="task_info is a Mach call")
def test_reads_this_process_footprint():
    first = fp.phys_footprint()
    held = bytearray(64 * 1024 * 1024)  # 64 MB, dirtied
    for i in range(0, len(held), 16384):
        held[i] = 1
    assert fp.phys_footprint() - first > 48 * 1024 * 1024
    del held
