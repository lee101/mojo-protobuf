from __future__ import annotations

import ctypes
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-protobuf.so")
I = ctypes.c_int64

_SIGNATURES = {
    "mpb_encode_varints": ([I, I, I, I], I),
    "mpb_encode_signed_varints": ([I, I, I, I], I),
    "mpb_encode_zigzag_varints": ([I, I, I, I], I),
    "mpb_decode_varints": ([I, I, I, I], I),
    "mpb_decode_zigzag_varints": ([I, I, I, I], I),
    "mpb_scan_fields": ([I, I, I, I, I, I, I], I),
}

_handle: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _handle
    if _handle is None:
        if not os.path.exists(LIB):
            raise RuntimeError("Mojo library is missing; run `pixi run build`")
        _handle = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_handle, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _handle


def addr(array) -> int:
    return int(array.ctypes.data)
