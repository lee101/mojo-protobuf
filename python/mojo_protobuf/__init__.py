"""Mojo-accelerated Protocol Buffers wire-format primitives."""

from . import decoder, encoder, wire_format
from .core import (
    Field,
    decode_fixed,
    decode_packed,
    decode_varints,
    encode_fixed,
    encode_packed,
    encode_varints,
    scan_fields,
)

__all__ = [
    "Field",
    "decoder",
    "encoder",
    "wire_format",
    "encode_varints",
    "decode_varints",
    "encode_fixed",
    "decode_fixed",
    "encode_packed",
    "decode_packed",
    "scan_fields",
]
