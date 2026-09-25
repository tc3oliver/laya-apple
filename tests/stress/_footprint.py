"""This process's phys_footprint (macOS), read with task_info(TASK_VM_INFO) through ctypes.

phys_footprint is the memory the OS charges the process (dirty + compressed, including
IOSurface and other IOKit memory mapped into it), which RSS can miss. Test helper only.
"""

from __future__ import annotations

import ctypes
import struct

TASK_VM_INFO = 22
# struct task_vm_info (mach/task_info.h, #pragma pack(4)): virtual_size (8), region_count (4),
# page_size (4), then 16 mach_vm_size_t fields from resident_size to compressed_lifetime, then
# phys_footprint (rev1).
PHYS_FOOTPRINT_OFFSET = 16 + 16 * 8
REV1_COUNT = (PHYS_FOOTPRINT_OFFSET + 8) // 4  # in natural_t (4-byte) units
BUFFER_COUNT = 128  # room for later revisions of the struct


def parse_phys_footprint(buf: bytes, count: int) -> int:
    """phys_footprint from a task_vm_info buffer that task_info filled with `count` natural_t."""
    if count < REV1_COUNT:
        raise ValueError(f"task_info returned {count} natural_t; phys_footprint needs {REV1_COUNT}")
    if len(buf) < PHYS_FOOTPRINT_OFFSET + 8:
        raise ValueError(f"task_vm_info buffer holds {len(buf)} bytes")
    return struct.unpack_from("<Q", buf, PHYS_FOOTPRINT_OFFSET)[0]


def phys_footprint() -> int:
    """Bytes charged to this process now."""
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    task = ctypes.c_uint.in_dll(libc, "mach_task_self_")  # what mach_task_self() returns
    buf = (ctypes.c_uint32 * BUFFER_COUNT)()
    count = ctypes.c_uint32(BUFFER_COUNT)
    kr = libc.task_info(task, TASK_VM_INFO, ctypes.byref(buf), ctypes.byref(count))
    if kr != 0:
        raise OSError(f"task_info(TASK_VM_INFO) failed: kern_return_t {kr}")
    return parse_phys_footprint(bytes(buf), count.value)
