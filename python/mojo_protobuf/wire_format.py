from __future__ import annotations

from google.protobuf import message

TAG_TYPE_BITS = 3
TAG_TYPE_MASK = 7
WIRETYPE_VARINT = 0
WIRETYPE_FIXED64 = 1
WIRETYPE_LENGTH_DELIMITED = 2
WIRETYPE_START_GROUP = 3
WIRETYPE_END_GROUP = 4
WIRETYPE_FIXED32 = 5

INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)
UINT32_MAX = (1 << 32) - 1
INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)
UINT64_MAX = (1 << 64) - 1


def PackTag(field_number, wire_type):
    if not 0 <= wire_type <= WIRETYPE_FIXED32:
        raise message.EncodeError(f"Unknown wire type: {wire_type}")
    return (field_number << TAG_TYPE_BITS) | wire_type


def UnpackTag(tag):
    return tag >> TAG_TYPE_BITS, tag & TAG_TYPE_MASK


def ZigZagEncode(value):
    return value << 1 if value >= 0 else (value << 1) ^ -1


def ZigZagDecode(value):
    return value >> 1 if not value & 1 else (value >> 1) ^ -1


def _VarUInt64ByteSizeNoTag(value):
    return max(1, (int(value).bit_length() + 6) // 7)


def TagByteSize(field_number):
    return _VarUInt64ByteSizeNoTag(PackTag(field_number, 0))


def Int32ByteSizeNoTag(value):
    return _VarUInt64ByteSizeNoTag(value & UINT64_MAX)


def Int64ByteSize(field_number, value):
    return TagByteSize(field_number) + _VarUInt64ByteSizeNoTag(value & UINT64_MAX)


def Int32ByteSize(field_number, value):
    return Int64ByteSize(field_number, value)


def UInt64ByteSize(field_number, value):
    return TagByteSize(field_number) + _VarUInt64ByteSizeNoTag(value)


UInt32ByteSize = UInt64ByteSize


def SInt32ByteSize(field_number, value):
    return UInt64ByteSize(field_number, ZigZagEncode(value))


SInt64ByteSize = SInt32ByteSize


def Fixed32ByteSize(field_number, unused):
    return TagByteSize(field_number) + 4


SFixed32ByteSize = FloatByteSize = Fixed32ByteSize


def Fixed64ByteSize(field_number, unused):
    return TagByteSize(field_number) + 8


SFixed64ByteSize = DoubleByteSize = Fixed64ByteSize


def BoolByteSize(field_number, unused):
    return TagByteSize(field_number) + 1


EnumByteSize = UInt32ByteSize


def BytesByteSize(field_number, value):
    return TagByteSize(field_number) + _VarUInt64ByteSizeNoTag(len(value)) + len(value)


def StringByteSize(field_number, value):
    return BytesByteSize(field_number, value.encode("utf-8"))
