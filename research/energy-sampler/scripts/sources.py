"""No-sudo power sources on Apple silicon, through ctypes only (no third-party packages).

Two sources, both read as the logged-in user without sudo:

* ``SMC``      - the System Management Controller key ``PSTR`` (total system power, W, a
                 32-bit float) read through the ``AppleSMC`` IOKit user client. An
                 instantaneous reading that the SMC itself filters; it has no energy
                 counter, so energy is the integral of polled readings.
* ``IOReport`` - the private ``libIOReport`` "Energy Model" group, the same source that
                 ``powermetrics`` and the sudoless tools (macmon, asitop successors) use.
                 Cumulative per-rail energy counters (CPU, GPU, ANE, DRAM, ...). The
                 energy of an interval is the delta of two samples, so no polling-rate
                 aliasing: a sample pair covers everything that happened between them.

Channel names, key names and units differ by SoC and macOS version. ``describe()``
prints what this machine exposes; README.md records what was found on the tested SoC.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct
import time
from ctypes import POINTER, Structure, byref, c_char, c_int32, c_uint8, c_uint16, c_uint32, c_uint64, c_void_p

_cf = None
_iokit = None
_ior = None

kCFStringEncodingUTF8 = 0x08000100


def _libs():
    global _cf, _iokit, _ior
    if _cf is None:
        _cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        _iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
        _ior = ctypes.CDLL("/usr/lib/libIOReport.dylib")  # in the dyld shared cache

        _cf.CFStringCreateWithCString.restype = c_void_p
        _cf.CFStringCreateWithCString.argtypes = [c_void_p, ctypes.c_char_p, c_uint32]
        _cf.CFStringGetCString.restype = ctypes.c_bool
        _cf.CFStringGetCString.argtypes = [c_void_p, ctypes.c_char_p, ctypes.c_long, c_uint32]
        _cf.CFDictionaryGetValue.restype = c_void_p
        _cf.CFDictionaryGetValue.argtypes = [c_void_p, c_void_p]
        _cf.CFArrayGetCount.restype = ctypes.c_long
        _cf.CFArrayGetCount.argtypes = [c_void_p]
        _cf.CFArrayGetValueAtIndex.restype = c_void_p
        _cf.CFArrayGetValueAtIndex.argtypes = [c_void_p, ctypes.c_long]
        _cf.CFRelease.argtypes = [c_void_p]
        _cf.CFRelease.restype = None
        _cf.CFDictionaryCreateMutableCopy.restype = c_void_p
        _cf.CFDictionaryCreateMutableCopy.argtypes = [c_void_p, ctypes.c_long, c_void_p]
        _iokit.IOServiceMatching.restype = c_void_p
        _iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
        _iokit.IOServiceGetMatchingService.restype = c_uint32
        _iokit.IOServiceGetMatchingService.argtypes = [c_uint32, c_void_p]
        _iokit.IOServiceOpen.restype = c_int32
        _iokit.IOServiceOpen.argtypes = [c_uint32, c_uint32, c_uint32, POINTER(c_uint32)]
        _iokit.IOServiceClose.argtypes = [c_uint32]
        _iokit.IOObjectRelease.argtypes = [c_uint32]
        _iokit.IOConnectCallStructMethod.restype = c_int32
        _iokit.IOConnectCallStructMethod.argtypes = [
            c_uint32,
            c_uint32,
            c_void_p,
            ctypes.c_size_t,
            c_void_p,
            POINTER(ctypes.c_size_t),
        ]

        _ior.IOReportCopyChannelsInGroup.restype = c_void_p
        _ior.IOReportCopyChannelsInGroup.argtypes = [c_void_p, c_void_p, c_uint64, c_uint64, c_uint64]
        _ior.IOReportCreateSubscription.restype = c_void_p
        _ior.IOReportCreateSubscription.argtypes = [c_void_p, c_void_p, POINTER(c_void_p), c_uint64, c_void_p]
        _ior.IOReportCreateSamples.restype = c_void_p
        _ior.IOReportCreateSamples.argtypes = [c_void_p, c_void_p, c_void_p]
        _ior.IOReportCreateSamplesDelta.restype = c_void_p
        _ior.IOReportCreateSamplesDelta.argtypes = [c_void_p, c_void_p, c_void_p]
        for fn in (
            "IOReportChannelGetGroup",
            "IOReportChannelGetSubGroup",
            "IOReportChannelGetChannelName",
            "IOReportChannelGetUnitLabel",
        ):
            getattr(_ior, fn).restype = c_void_p
            getattr(_ior, fn).argtypes = [c_void_p]
        _ior.IOReportSimpleGetIntegerValue.restype = ctypes.c_int64
        _ior.IOReportSimpleGetIntegerValue.argtypes = [c_void_p, c_int32]
        _ior.IOReportChannelGetFormat.restype = c_uint8
        _ior.IOReportChannelGetFormat.argtypes = [c_void_p]
    return _cf, _iokit, _ior


def _cfstr(s: str) -> int:
    cf, _, _ = _libs()
    return cf.CFStringCreateWithCString(None, s.encode(), kCFStringEncodingUTF8)


def _pystr(ref) -> str | None:
    if not ref:
        return None
    cf, _, _ = _libs()
    buf = ctypes.create_string_buffer(256)
    return buf.value.decode() if cf.CFStringGetCString(ref, buf, 256, kCFStringEncodingUTF8) else None


# --------------------------------------------------------------------------------------- SMC


class _Vers(Structure):
    _fields_ = [("major", c_char), ("minor", c_char), ("build", c_char), ("reserved", c_char), ("release", c_uint16)]


class _PLimit(Structure):
    _fields_ = [
        ("version", c_uint16),
        ("length", c_uint16),
        ("cpu", c_uint32),
        ("gpu", c_uint32),
        ("mem", c_uint32),
    ]


class _KeyInfo(Structure):
    _fields_ = [("dataSize", c_uint32), ("dataType", c_uint32), ("dataAttributes", c_uint8)]


class _SMCKeyData(Structure):
    _fields_ = [
        ("key", c_uint32),
        ("vers", _Vers),
        ("pLimitData", _PLimit),
        ("keyInfo", _KeyInfo),
        ("result", c_uint8),
        ("status", c_uint8),
        ("data8", c_uint8),
        ("data32", c_uint32),
        ("bytes", c_uint8 * 32),
    ]


assert ctypes.sizeof(_SMCKeyData) == 80, ctypes.sizeof(_SMCKeyData)

_KERNEL_INDEX_SMC = 2
_SMC_CMD_READ_BYTES = 5
_SMC_CMD_READ_KEYINFO = 9


def _fourcc(s: str) -> int:
    return struct.unpack(">I", s.encode())[0]


def _unfourcc(v: int) -> str:
    return struct.pack(">I", v).decode(errors="replace")


class SMC:
    """Read SMC keys through the AppleSMC user client (no sudo on macOS 26 / M4 Max)."""

    def __init__(self) -> None:
        _, iokit, _ = _libs()
        svc = iokit.IOServiceGetMatchingService(0, iokit.IOServiceMatching(b"AppleSMC"))
        if not svc:
            raise OSError("AppleSMC service not found")
        conn = c_uint32(0)
        kr = iokit.IOServiceOpen(svc, _libc_task_self(), 0, byref(conn))
        iokit.IOObjectRelease(svc)
        if kr != 0:
            raise OSError(f"IOServiceOpen(AppleSMC) failed: {kr:#x}")
        self._conn = conn.value
        self._info: dict[str, tuple[int, str]] = {}

    def close(self) -> None:
        if self._conn:
            _libs()[1].IOServiceClose(self._conn)
            self._conn = 0

    def _call(self, inp: _SMCKeyData) -> _SMCKeyData:
        out = _SMCKeyData()
        size = ctypes.c_size_t(ctypes.sizeof(out))
        kr = _libs()[1].IOConnectCallStructMethod(
            self._conn, _KERNEL_INDEX_SMC, byref(inp), ctypes.sizeof(inp), byref(out), byref(size)
        )
        if kr != 0:
            raise OSError(f"IOConnectCallStructMethod failed: {kr:#x}")
        if out.result != 0:
            raise KeyError(f"SMC result {out.result}")
        return out

    def key_info(self, key: str) -> tuple[int, str]:
        if key not in self._info:
            inp = _SMCKeyData(key=_fourcc(key), data8=_SMC_CMD_READ_KEYINFO)
            out = self._call(inp)
            self._info[key] = (out.keyInfo.dataSize, _unfourcc(out.keyInfo.dataType))
        return self._info[key]

    def read_raw(self, key: str) -> tuple[str, bytes]:
        size, typ = self.key_info(key)
        inp = _SMCKeyData(key=_fourcc(key), data8=_SMC_CMD_READ_BYTES)
        inp.keyInfo.dataSize = size
        out = self._call(inp)
        return typ, bytes(out.bytes[:size])

    def read(self, key: str) -> float:
        typ, raw = self.read_raw(key)
        return decode_smc(typ, raw)

    def key_count(self) -> int:
        return int(struct.unpack(">I", self.read_raw("#KEY")[1])[0])

    def key_at(self, i: int) -> str:
        inp = _SMCKeyData(data8=8, data32=i)  # kSMCGetKeyFromIndex
        return _unfourcc(self._call(inp).key)


def decode_smc(typ: str, raw: bytes) -> float:
    """Decode an SMC value. Apple silicon power keys are 'flt ' (little-endian float32)."""
    t = typ.strip()
    if t == "flt":
        return struct.unpack("<f", raw[:4])[0]
    if t == "ui8":
        return float(raw[0])
    if t == "ui16":
        return float(struct.unpack(">H", raw[:2])[0])
    if t == "ui32":
        return float(struct.unpack(">I", raw[:4])[0])
    if t == "sp78":
        return struct.unpack(">h", raw[:2])[0] / 256.0
    if t == "fpe2":
        return struct.unpack(">H", raw[:2])[0] / 4.0
    raise ValueError(f"unsupported SMC type {typ!r}")


def _libc_task_self() -> int:
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    return c_uint32.in_dll(libc, "mach_task_self_").value


# ---------------------------------------------------------------------------------- IOReport

_ENERGY_UNITS = {"mJ": 1e-3, "uJ": 1e-6, "µJ": 1e-6, "nJ": 1e-9}


class IOReportEnergy:
    """Cumulative per-rail energy from the IOReport "Energy Model" group.

    ``sample()`` returns an opaque handle; ``delta(a, b)`` returns {channel: joules} for the
    interval between two samples.
    """

    GROUP = "Energy Model"

    def __init__(self) -> None:
        # One sample costs ~3.3 ms of CPU on M4 Max / macOS 26.6.2 whether the subscription
        # holds all ~330 channels or only the four rails used (measured), so the whole group
        # is subscribed and the rails are picked in delta().
        cf, _, ior = _libs()
        grp = _cfstr(self.GROUP)
        chans = ior.IOReportCopyChannelsInGroup(grp, None, 0, 0, 0)
        cf.CFRelease(grp)
        if not chans:
            raise OSError("IOReport: no 'Energy Model' channels")
        self._chans = cf.CFDictionaryCreateMutableCopy(None, 0, chans)
        cf.CFRelease(chans)
        self._key = _cfstr("IOReportChannels")
        subbed = c_void_p()
        self._sub = ior.IOReportCreateSubscription(None, self._chans, byref(subbed), 0, None)
        if not self._sub:
            raise OSError("IOReportCreateSubscription failed")
        self._subbed = subbed.value

    def sample(self) -> int:
        s = _libs()[2].IOReportCreateSamples(self._sub, self._subbed, None)
        if not s:
            raise OSError("IOReportCreateSamples failed")
        return s

    def release(self, s: int) -> None:
        _libs()[0].CFRelease(s)

    def delta(self, a: int, b: int, only: tuple[str, ...] | None = None) -> dict[str, float]:
        """{channel: joules} between samples a and b. With `only`, the channel positions are
        found once and then re-verified by name on every call (cheaper than walking all ~330
        channels; a reordering falls back to a full walk)."""
        cf, _, ior = _libs()
        d = ior.IOReportCreateSamplesDelta(a, b, None)
        try:
            arr = cf.CFDictionaryGetValue(d, self._key)
            n = cf.CFArrayGetCount(arr)
            if only is not None:
                idx = getattr(self, "_idx", {}).get(only)
                if idx is not None:
                    out = self._read_idx(arr, n, idx)
                    if out is not None:
                        return out
            out: dict[str, float] = {}
            found: list[tuple[int, str, float]] = []
            for i in range(n):
                item = cf.CFArrayGetValueAtIndex(arr, i)
                name = _pystr(ior.IOReportChannelGetChannelName(item))
                unit = (_pystr(ior.IOReportChannelGetUnitLabel(item)) or "").strip()
                if name is None or unit not in _ENERGY_UNITS or (only is not None and name not in only):
                    continue
                found.append((i, name, _ENERGY_UNITS[unit]))
                out[name] = out.get(name, 0.0) + ior.IOReportSimpleGetIntegerValue(item, 0) * _ENERGY_UNITS[unit]
            if only is not None:
                self._idx = {**getattr(self, "_idx", {}), only: found}
            return out
        finally:
            cf.CFRelease(d)

    def _read_idx(self, arr, n, idx):
        cf, _, ior = _libs()
        out: dict[str, float] = {}
        for i, name, scale in idx:
            if i >= n:
                return None
            item = cf.CFArrayGetValueAtIndex(arr, i)
            if _pystr(ior.IOReportChannelGetChannelName(item)) != name:
                return None
            out[name] = out.get(name, 0.0) + ior.IOReportSimpleGetIntegerValue(item, 0) * scale
        return out

    def channels(self) -> list[tuple[str | None, str | None, str | None]]:
        cf, _, ior = _libs()
        arr = cf.CFDictionaryGetValue(self._chans, self._key)
        return [
            (
                _pystr(ior.IOReportChannelGetSubGroup(item)),
                _pystr(ior.IOReportChannelGetChannelName(item)),
                _pystr(ior.IOReportChannelGetUnitLabel(item)),
            )
            for item in (cf.CFArrayGetValueAtIndex(arr, i) for i in range(cf.CFArrayGetCount(arr)))
        ]


def describe(seconds: float = 1.0) -> dict:
    """What this machine exposes, for README.md and the run metadata."""
    out: dict = {}
    try:
        smc = SMC()
        keys = {}
        for k in ("PSTR", "PDTR", "PPBR", "PHPC", "PMVC", "PZC0"):
            try:
                keys[k] = {"type": smc.key_info(k)[1], "value": smc.read(k)}
            except (KeyError, OSError, ValueError) as e:
                keys[k] = {"error": str(e)}
        out["smc"] = {"ok": True, "keys": keys}
        smc.close()
    except OSError as e:
        out["smc"] = {"ok": False, "error": str(e)}
    try:
        ior = IOReportEnergy()
        a = ior.sample()
        t0 = time.perf_counter()
        time.sleep(seconds)
        b = ior.sample()
        dt = time.perf_counter() - t0
        watts = {k: v / dt for k, v in ior.delta(a, b).items() if v}
        ior.release(a)
        ior.release(b)
        out["ioreport"] = {"ok": True, "channels": ior.channels(), "watts_over_1s": watts}
    except OSError as e:
        out["ioreport"] = {"ok": False, "error": str(e)}
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(describe(), indent=1))
