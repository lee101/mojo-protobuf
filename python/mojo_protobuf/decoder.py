from __future__ import annotations

import struct

from google.protobuf import message

from . import wire_format
from .core import _decode_one, decode_fixed, decode_varints
from .encoder import TagBytes

_DecodeError = message.DecodeError


def _DecodeVarint(buffer, pos: int = None):
    if pos is None:
        result = 0
        shift = 0
        while shift < 64:
            part = buffer.read(1)
            if not part:
                if shift == 0:
                    return None
                raise ValueError("Fail to read varint")
            byte = part[0]
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result & ((1 << 64) - 1)
            shift += 7
        raise _DecodeError("Too many bytes when decoding varint.")
    return _decode_one(buffer, pos, 64, False)


def _DecodeVarint32(buffer, pos: int = None):
    if pos is None:
        value = _DecodeVarint(buffer, None)
        return None if value is None else value & ((1 << 32) - 1)
    return _decode_one(buffer, pos, 32, False)


def _DecodeSignedVarint(buffer, pos):
    return _decode_one(buffer, pos, 64, True)


def _DecodeSignedVarint32(buffer, pos):
    return _decode_one(buffer, pos, 32, True)


def ReadTag(buffer, pos):
    start = pos
    limit = len(buffer)
    while pos < limit and buffer[pos] & 0x80:
        pos += 1
    if pos >= limit:
        raise _DecodeError("Truncated message.")
    pos += 1
    piece = buffer[start:pos]
    return piece.tobytes() if hasattr(piece, "tobytes") else bytes(piece), pos


def DecodeTag(tag_bytes):
    return wire_format.UnpackTag(_DecodeVarint(tag_bytes, 0)[0])


def IsDefaultScalarValue(value):
    if isinstance(value, float) and value == 0.0:
        return struct.pack("<d", value) == b"\0" * 8
    return not value


def _varint_decoder(kind):
    signed = kind.startswith("int")
    zigzag = kind.startswith("sint")
    bits = 32 if kind.endswith("32") else 64

    def decode_one(buffer, pos):
        value, pos = _decode_one(buffer, pos, bits, signed)
        return (wire_format.ZigZagDecode(value), pos) if zigzag else (value, pos)

    def decode_many(buffer):
        return decode_varints(buffer, signed=signed or zigzag, zigzag=zigzag, bits=bits).tolist()

    return _decoder_constructor(wire_format.WIRETYPE_VARINT, decode_one, decode_many)


def _decoder_constructor(wire_type, decode_one_value, decode_many=None):
    def SpecificDecoder(
        field_number, is_repeated, is_packed, key, new_default, clear_if_default=False
    ):
        if is_packed:
            def DecodePackedField(buffer, pos, end, message_obj, field_dict, current_depth=0):
                endpoint, pos = _DecodeVarint(buffer, pos)
                endpoint += pos
                if endpoint > end:
                    raise _DecodeError("Truncated message.")
                value = field_dict.get(key)
                if value is None:
                    value = field_dict.setdefault(key, new_default(message_obj))
                if decode_many is not None:
                    value.extend(decode_many(buffer[pos:endpoint]))
                    return endpoint
                while pos < endpoint:
                    element, pos = decode_one_value(buffer, pos)
                    value.append(element)
                if pos != endpoint:
                    del value[-1]
                    raise _DecodeError("Packed element was truncated.")
                return pos
            return DecodePackedField
        if is_repeated:
            tag = TagBytes(field_number, wire_type)
            tag_len = len(tag)

            def DecodeRepeatedField(buffer, pos, end, message_obj, field_dict, current_depth=0):
                value = field_dict.get(key)
                if value is None:
                    value = field_dict.setdefault(key, new_default(message_obj))
                while True:
                    element, new_pos = decode_one_value(buffer, pos)
                    value.append(element)
                    next_pos = new_pos + tag_len
                    if new_pos >= end or bytes(buffer[new_pos:next_pos]) != tag:
                        if new_pos > end:
                            raise _DecodeError("Truncated message.")
                        return new_pos
                    pos = next_pos
            return DecodeRepeatedField

        def DecodeField(buffer, pos, end, message_obj, field_dict, current_depth=0):
            value, pos = decode_one_value(buffer, pos)
            if pos > end:
                raise _DecodeError("Truncated message.")
            if clear_if_default and IsDefaultScalarValue(value):
                field_dict.pop(key, None)
            else:
                field_dict[key] = value
            return pos
        return DecodeField
    return SpecificDecoder


Int32Decoder = _varint_decoder("int32")
Int64Decoder = _varint_decoder("int64")
UInt32Decoder = _varint_decoder("uint32")
UInt64Decoder = _varint_decoder("uint64")
SInt32Decoder = _varint_decoder("sint32")
SInt64Decoder = _varint_decoder("sint64")
EnumDecoder = Int32Decoder


def _fixed_decoder(kind, wire_type, fmt):
    width = struct.calcsize(fmt)

    def decode_one_value(buffer, pos):
        new_pos = pos + width
        if new_pos > len(buffer):
            raise _DecodeError("Truncated message.")
        return struct.unpack(fmt, buffer[pos:new_pos])[0], new_pos

    def decode_many(buffer):
        return decode_fixed(buffer, kind).tolist()

    return _decoder_constructor(wire_type, decode_one_value, decode_many)


Fixed32Decoder = _fixed_decoder("fixed32", wire_format.WIRETYPE_FIXED32, "<I")
SFixed32Decoder = _fixed_decoder("sfixed32", wire_format.WIRETYPE_FIXED32, "<i")
FloatDecoder = _fixed_decoder("float", wire_format.WIRETYPE_FIXED32, "<f")
Fixed64Decoder = _fixed_decoder("fixed64", wire_format.WIRETYPE_FIXED64, "<Q")
SFixed64Decoder = _fixed_decoder("sfixed64", wire_format.WIRETYPE_FIXED64, "<q")
DoubleDecoder = _fixed_decoder("double", wire_format.WIRETYPE_FIXED64, "<d")


def _decode_bool(buffer, pos):
    value, pos = _DecodeVarint(buffer, pos)
    return bool(value), pos


BoolDecoder = _decoder_constructor(
    wire_format.WIRETYPE_VARINT,
    _decode_bool,
    lambda buffer: decode_varints(buffer).astype(bool).tolist(),
)


def _length_decoder(to_value):
    def SpecificDecoder(
        field_number, is_repeated, is_packed, key, new_default, clear_if_default=False
    ):
        assert not is_packed
        wire_type = wire_format.WIRETYPE_LENGTH_DELIMITED

        def one(buffer, pos):
            size, pos = _DecodeVarint(buffer, pos)
            new_pos = pos + size
            if new_pos > len(buffer):
                raise _DecodeError("Truncated string or bytes field.")
            return to_value(buffer[pos:new_pos]), new_pos
        return _decoder_constructor(wire_type, one)(
            field_number, is_repeated, False, key, new_default, clear_if_default
        )
    return SpecificDecoder


StringDecoder = _length_decoder(lambda value: bytes(value).decode("utf-8"))
BytesDecoder = _length_decoder(bytes)
