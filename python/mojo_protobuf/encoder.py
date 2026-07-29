from __future__ import annotations

import struct

from . import wire_format
from .core import encode_fixed, encode_varints


def _VarintSize(value):
    return max(1, (int(value).bit_length() + 6) // 7)


def _SignedVarintSize(value):
    return 10 if value < 0 else _VarintSize(value)


def _TagSize(field_number):
    return _VarintSize(wire_format.PackTag(field_number, 0))


def _VarintEncoder():
    def EncodeVarint(write, value, unused_deterministic=None):
        return write(encode_varints([value]))
    return EncodeVarint


def _SignedVarintEncoder():
    def EncodeSignedVarint(write, value, unused_deterministic=None):
        return write(encode_varints([value], signed=True))
    return EncodeSignedVarint


_EncodeVarint = _VarintEncoder()
_EncodeSignedVarint = _SignedVarintEncoder()


def _VarintBytes(value):
    return encode_varints([value])


def TagBytes(field_number, wire_type):
    return _VarintBytes(wire_format.PackTag(field_number, wire_type))


def _varint_encoder(kind):
    signed = kind.startswith("int")
    zigzag = kind.startswith("sint")
    bits = 32 if kind.endswith("32") else 64

    def SpecificEncoder(field_number, is_repeated, is_packed):
        tag = TagBytes(
            field_number,
            wire_format.WIRETYPE_LENGTH_DELIMITED if is_packed else wire_format.WIRETYPE_VARINT,
        )

        def payload(value):
            return encode_varints(value, signed=signed or zigzag, zigzag=zigzag, bits=bits)

        if is_packed:
            def EncodePackedField(write, value, deterministic):
                body = payload(value)
                write(tag)
                write(encode_varints([len(body)]))
                return write(body)
            return EncodePackedField
        if is_repeated:
            def EncodeRepeatedField(write, value, deterministic=None):
                for element in value:
                    write(tag)
                    write(payload([element]))
            return EncodeRepeatedField

        def EncodeField(write, value, deterministic=None):
            write(tag)
            return write(payload([value]))
        return EncodeField
    return SpecificEncoder


Int32Encoder = _varint_encoder("int32")
Int64Encoder = _varint_encoder("int64")
UInt32Encoder = _varint_encoder("uint32")
UInt64Encoder = _varint_encoder("uint64")
SInt32Encoder = _varint_encoder("sint32")
SInt64Encoder = _varint_encoder("sint64")
EnumEncoder = Int32Encoder


def _fixed_encoder(kind, wire_type, fmt):
    def SpecificEncoder(field_number, is_repeated, is_packed):
        tag = TagBytes(
            field_number,
            wire_format.WIRETYPE_LENGTH_DELIMITED if is_packed else wire_type,
        )
        if is_packed:
            def EncodePackedField(write, value, deterministic):
                body = encode_fixed(value, kind)
                write(tag)
                write(encode_varints([len(body)]))
                return write(body)
            return EncodePackedField
        if is_repeated:
            def EncodeRepeatedField(write, value, deterministic=None):
                for element in value:
                    write(tag)
                    write(struct.pack(fmt, element))
            return EncodeRepeatedField

        def EncodeField(write, value, deterministic=None):
            write(tag)
            return write(struct.pack(fmt, value))
        return EncodeField
    return SpecificEncoder


Fixed32Encoder = _fixed_encoder("fixed32", wire_format.WIRETYPE_FIXED32, "<I")
SFixed32Encoder = _fixed_encoder("sfixed32", wire_format.WIRETYPE_FIXED32, "<i")
FloatEncoder = _fixed_encoder("float", wire_format.WIRETYPE_FIXED32, "<f")
Fixed64Encoder = _fixed_encoder("fixed64", wire_format.WIRETYPE_FIXED64, "<Q")
SFixed64Encoder = _fixed_encoder("sfixed64", wire_format.WIRETYPE_FIXED64, "<q")
DoubleEncoder = _fixed_encoder("double", wire_format.WIRETYPE_FIXED64, "<d")


def BoolEncoder(field_number, is_repeated, is_packed):
    tag = TagBytes(
        field_number,
        wire_format.WIRETYPE_LENGTH_DELIMITED if is_packed else wire_format.WIRETYPE_VARINT,
    )
    if is_packed:
        def EncodePackedField(write, value, deterministic):
            body = bytes(1 if element else 0 for element in value)
            write(tag)
            write(encode_varints([len(body)]))
            return write(body)
        return EncodePackedField
    if is_repeated:
        def EncodeRepeatedField(write, value, deterministic=None):
            for element in value:
                write(tag)
                write(b"\x01" if element else b"\x00")
        return EncodeRepeatedField

    def EncodeField(write, value, deterministic=None):
        write(tag)
        return write(b"\x01" if value else b"\x00")
    return EncodeField


def _length_delimited_encoder(to_bytes):
    def SpecificEncoder(field_number, is_repeated, is_packed):
        assert not is_packed
        tag = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)

        def emit(write, element):
            body = to_bytes(element)
            write(tag)
            write(encode_varints([len(body)]))
            return write(body)

        if is_repeated:
            def EncodeRepeatedField(write, value, deterministic=None):
                for element in value:
                    emit(write, element)
            return EncodeRepeatedField

        def EncodeField(write, value, deterministic=None):
            return emit(write, value)
        return EncodeField
    return SpecificEncoder


StringEncoder = _length_delimited_encoder(lambda value: value.encode("utf-8"))
BytesEncoder = _length_delimited_encoder(bytes)


def _simple_sizer(value_size):
    def SpecificSizer(field_number, is_repeated, is_packed):
        tag_size = _TagSize(field_number)
        if is_packed:
            def PackedFieldSize(value):
                body = sum(value_size(x) for x in value)
                return tag_size + _VarintSize(body) + body
            return PackedFieldSize
        if is_repeated:
            return lambda value: tag_size * len(value) + sum(value_size(x) for x in value)
        return lambda value: tag_size + value_size(value)
    return SpecificSizer


Int32Sizer = Int64Sizer = EnumSizer = _simple_sizer(_SignedVarintSize)
UInt32Sizer = UInt64Sizer = _simple_sizer(_VarintSize)
SInt32Sizer = SInt64Sizer = _simple_sizer(lambda x: _VarintSize(wire_format.ZigZagEncode(x)))
Fixed32Sizer = SFixed32Sizer = FloatSizer = _simple_sizer(lambda unused: 4)
Fixed64Sizer = SFixed64Sizer = DoubleSizer = _simple_sizer(lambda unused: 8)
BoolSizer = _simple_sizer(lambda unused: 1)


def StringSizer(field_number, is_repeated, is_packed):
    return _length_sizer(field_number, is_repeated, is_packed, lambda x: len(x.encode("utf-8")))


def BytesSizer(field_number, is_repeated, is_packed):
    return _length_sizer(field_number, is_repeated, is_packed, len)


def _length_sizer(field_number, is_repeated, is_packed, length):
    assert not is_packed
    tag_size = _TagSize(field_number)

    def one(value):
        size = length(value)
        return tag_size + _VarintSize(size) + size
    if is_repeated:
        return lambda value: sum(one(x) for x in value)
    return one
