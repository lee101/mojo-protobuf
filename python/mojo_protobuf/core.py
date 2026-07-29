from __future__ import annotations

from numbers import Integral
import struct
from typing import NamedTuple

import numpy as np
from google.protobuf import message

from ._lib import addr, lib


def _bytes_array(data) -> np.ndarray:
    view = memoryview(data)
    if not view.c_contiguous:
        view = memoryview(bytes(view))
    return np.frombuffer(view.cast("B"), dtype=np.uint8)


def _byte_view(data) -> memoryview:
    view = memoryview(data)
    if not view.c_contiguous:
        view = memoryview(bytes(view))
    return view.cast("B")


def encode_varints(values, *, signed=False, zigzag=False, bits=64) -> bytes:
    if bits not in (32, 64):
        raise ValueError("bits must be 32 or 64")
    array = np.asarray(values)
    if not array.size:
        return b""
    if array.dtype.kind not in "iub":
        exact = np.asarray(values, dtype=object).reshape(-1)
        if not all(isinstance(value, Integral) for value in exact):
            raise TypeError("varint values must be integers")
        array = exact
    if signed or zigzag:
        lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        if np.any(array < lower) or np.any(array > upper):
            raise OverflowError(f"value does not fit int{bits}")
        wide = np.ascontiguousarray(array, dtype=np.int64).reshape(-1)
        dst = np.empty(wide.size * 10, dtype=np.uint8)
        fn = lib().mpb_encode_zigzag_varints if zigzag else lib().mpb_encode_signed_varints
        size = fn(addr(wide), wide.size, addr(dst), dst.size) if wide.size else 0
    else:
        upper = (1 << bits) - 1
        if np.any(array < 0) or np.any(array > upper):
            raise OverflowError(f"value does not fit uint{bits}")
        wide = np.ascontiguousarray(array, dtype=np.uint64).reshape(-1)
        dst = np.empty(wide.size * (5 if bits == 32 else 10), dtype=np.uint8)
        size = (
            lib().mpb_encode_varints(addr(wide), wide.size, addr(dst), dst.size)
            if wide.size else 0
        )
    if size < 0:
        raise RuntimeError("Mojo varint encoder rejected its input or output buffer")
    return dst[:size].tobytes()


def decode_varints(data, *, signed=False, zigzag=False, bits=64, count=None) -> np.ndarray:
    if bits not in (32, 64):
        raise ValueError("bits must be 32 or 64")
    src = _bytes_array(data)
    limit = src.size if count is None else int(count)
    if limit < 0:
        raise ValueError("count must be non-negative")
    dtype = np.int64 if signed or zigzag else np.uint64
    dst = np.empty(limit, dtype=dtype)
    if not src.size:
        return dst[:0]
    if limit == 0:
        raise ValueError("count is smaller than the number of encoded values")
    fn = lib().mpb_decode_zigzag_varints if zigzag else lib().mpb_decode_varints
    status = fn(addr(src), src.size, addr(dst), limit)
    if status == -1:
        raise message.DecodeError("Truncated varint.")
    if status == -2:
        raise message.DecodeError("Too many bytes when decoding varint.")
    if status == -3:
        raise ValueError("count is smaller than the number of encoded values")
    if status < 0:
        raise RuntimeError("Mojo varint decoder rejected its buffers")
    if signed and not zigzag:
        dst = dst.view(np.int64)
    if bits == 32:
        dst = dst.astype(np.int32 if signed or zigzag else np.uint32)
    return dst[:status].copy()


_FIXED = {
    "fixed32": ("<u4", 4),
    "sfixed32": ("<i4", 4),
    "float": ("<f4", 4),
    "fixed64": ("<u8", 8),
    "sfixed64": ("<i8", 8),
    "double": ("<f8", 8),
}


def encode_fixed(values, kind) -> bytes:
    try:
        dtype, _ = _FIXED[kind]
    except KeyError:
        raise ValueError(f"unknown fixed kind: {kind}") from None
    target = np.dtype(dtype)
    source = np.asarray(values)
    if target.kind in "iu":
        if source.dtype.kind not in "iub":
            exact = np.asarray(values, dtype=object)
            if not all(isinstance(value, Integral) for value in exact.reshape(-1)):
                raise TypeError(f"{kind} values must be integers")
            source = exact
        limits = np.iinfo(target)
        if np.any(source < limits.min) or np.any(source > limits.max):
            raise OverflowError(f"value does not fit {kind}")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.ascontiguousarray(source, dtype=target)
    if target.kind == "f" and source.dtype.kind == "f":
        if np.any(np.isfinite(source) & ~np.isfinite(converted)):
            raise OverflowError(f"value does not fit {kind}")
    return converted.tobytes()


def decode_fixed(data, kind) -> np.ndarray:
    try:
        dtype, width = _FIXED[kind]
    except KeyError:
        raise ValueError(f"unknown fixed kind: {kind}") from None
    view = _byte_view(data)
    if view.nbytes % width:
        raise message.DecodeError("Truncated fixed-width value.")
    return np.frombuffer(view, dtype=dtype).copy()


def encode_packed(values, kind) -> bytes:
    if kind in ("int32", "int64"):
        payload = encode_varints(values, signed=True, bits=32 if kind == "int32" else 64)
    elif kind == "enum":
        payload = encode_varints(values, signed=True, bits=32)
    elif kind in ("uint32", "uint64", "bool"):
        payload = encode_varints(values, bits=32 if kind in ("uint32", "bool") else 64)
    elif kind in ("sint32", "sint64"):
        payload = encode_varints(values, signed=True, zigzag=True, bits=32 if kind == "sint32" else 64)
    else:
        payload = encode_fixed(values, kind)
    return encode_varints([len(payload)]) + payload


def decode_packed(data, kind) -> np.ndarray:
    raw = _byte_view(data)
    length, pos = _decode_one(raw, 0, 64, False)
    end = pos + length
    if end != len(raw):
        raise message.DecodeError("Packed payload length does not match buffer.")
    payload = raw[pos:end]
    if kind in ("int32", "int64"):
        return decode_varints(payload, signed=True, bits=32 if kind == "int32" else 64)
    if kind == "enum":
        return decode_varints(payload, signed=True, bits=32)
    if kind in ("uint32", "uint64", "bool"):
        values = decode_varints(payload, bits=32 if kind in ("uint32", "bool") else 64)
        return values.astype(bool) if kind == "bool" else values
    if kind in ("sint32", "sint64"):
        return decode_varints(payload, signed=True, zigzag=True, bits=32 if kind == "sint32" else 64)
    return decode_fixed(payload, kind)


def _decode_one(buffer, pos, bits=64, signed=False):
    result = 0
    shift = 0
    while shift < 64:
        try:
            byte = buffer[pos]
        except IndexError:
            raise message.DecodeError("Truncated varint.") from None
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            mask = (1 << bits) - 1
            result &= mask
            if signed and result & (1 << (bits - 1)):
                result -= 1 << bits
            return result, pos
        shift += 7
    raise message.DecodeError("Too many bytes when decoding varint.")


class Field(NamedTuple):
    number: int
    wire_type: int
    value: memoryview

    def as_varint(self, *, signed=False, zigzag=False, bits=64):
        value, pos = _decode_one(self.value, 0, bits, signed)
        if pos != len(self.value):
            raise message.DecodeError("Invalid varint field.")
        if zigzag:
            return value >> 1 if not value & 1 else (value >> 1) ^ -1
        return value

    def as_fixed32(self):
        return struct.unpack("<I", self.value)[0]

    def as_fixed64(self):
        return struct.unpack("<Q", self.value)[0]

    def as_bytes(self):
        return self.value.tobytes()


def scan_fields(data) -> list[Field]:
    raw = _byte_view(data)
    src = _bytes_array(raw)
    if not src.size:
        return []
    maximum = (src.size + 1) // 2
    numbers = np.empty(maximum, dtype=np.uint32)
    wires = np.empty(maximum, dtype=np.uint8)
    offsets = np.empty(maximum, dtype=np.int64)
    lengths = np.empty(maximum, dtype=np.int64)
    status = lib().mpb_scan_fields(
        addr(src), src.size, addr(numbers), addr(wires), addr(offsets), addr(lengths), maximum
    )
    if status == -1:
        raise message.DecodeError("Truncated message.")
    if status == -2:
        raise message.DecodeError("Too many bytes when decoding varint.")
    if status == -4:
        raise message.DecodeError("Invalid field number or wire type.")
    if status == -5:
        raise NotImplementedError("group wire types are not supported by scan_fields")
    if status < 0:
        raise message.DecodeError("Could not scan wire fields.")
    records = zip(
        numbers[:status].tolist(),
        wires[:status].tolist(),
        offsets[:status].tolist(),
        lengths[:status].tolist(),
    )
    field_new = tuple.__new__
    return [
        field_new(Field, (number, wire_type, raw[offset:offset + length]))
        for number, wire_type, offset, length in records
    ]
