import io

import numpy as np
import pytest
from google.protobuf.internal import decoder as upstream_decoder
from google.protobuf.internal import encoder as upstream_encoder
from google.protobuf.internal import wire_format as upstream_wire

from mojo_protobuf import decoder, encoder, wire_format


VARINT_CASES = [
    ("Int32", [0, -1, 127, -(2**31)]),
    ("UInt32", [0, 1, 300, 2**32 - 1]),
    ("SInt32", [0, -1, 1, -(2**31)]),
    ("Int64", [0, -1, 2**40, -(2**63)]),
    ("UInt64", [0, 300, 2**40, 2**64 - 1]),
    ("SInt64", [0, -1, 2**40, -(2**63)]),
    ("Bool", [False, True, True, False]),
]

FIXED_CASES = [
    ("Fixed32", [0, 1, 2**32 - 1]),
    ("SFixed32", [-(2**31), -1, 2**31 - 1]),
    ("Float", [0.0, -1.25, 3.5]),
    ("Fixed64", [0, 1, 2**64 - 1]),
    ("SFixed64", [-(2**63), -1, 2**63 - 1]),
    ("Double", [0.0, -1.25, 1e100]),
]


def encoded(factory, field, repeated, packed, value):
    sink = io.BytesIO()
    factory(field, repeated, packed)(sink.write, value, True)
    return sink.getvalue()


@pytest.mark.parametrize("name,values", VARINT_CASES + FIXED_CASES)
@pytest.mark.parametrize("packed", [False, True])
def test_numeric_encoder_factories_match_upstream(name, values, packed):
    ours = getattr(encoder, name + "Encoder")
    theirs = getattr(upstream_encoder, name + "Encoder")
    value = values if packed else values[1]
    assert encoded(ours, 19, packed, packed, value) == encoded(theirs, 19, packed, packed, value)


@pytest.mark.parametrize("name,values", VARINT_CASES + FIXED_CASES)
def test_repeated_encoder_factories_match_upstream(name, values):
    ours = getattr(encoder, name + "Encoder")
    theirs = getattr(upstream_encoder, name + "Encoder")
    assert encoded(ours, 7, True, False, values) == encoded(theirs, 7, True, False, values)


@pytest.mark.parametrize(
    "name,value",
    [("String", "héllo protobuf"), ("Bytes", b"\x00\x01payload\xff")],
)
@pytest.mark.parametrize("repeated", [False, True])
def test_length_delimited_encoders_match_upstream(name, value, repeated):
    ours = getattr(encoder, name + "Encoder")
    theirs = getattr(upstream_encoder, name + "Encoder")
    actual = [value, value] if repeated else value
    assert encoded(ours, 300, repeated, False, actual) == encoded(
        theirs, 300, repeated, False, actual
    )


@pytest.mark.parametrize("name,values", VARINT_CASES + FIXED_CASES)
def test_packed_decoder_factories_match_upstream(name, values):
    data = encoded(getattr(upstream_encoder, name + "Encoder"), 11, True, True, values)
    tag, pos = upstream_decoder.ReadTag(memoryview(data), 0)
    assert tag == upstream_encoder.TagBytes(11, upstream_wire.WIRETYPE_LENGTH_DELIMITED)
    ours_dict = {}
    theirs_dict = {}
    new = lambda unused: []
    ours_pos = getattr(decoder, name + "Decoder")(11, True, True, "x", new)(
        memoryview(data), pos, len(data), None, ours_dict
    )
    theirs_pos = getattr(upstream_decoder, name + "Decoder")(11, True, True, "x", new)(
        memoryview(data), pos, len(data), None, theirs_dict
    )
    assert ours_pos == theirs_pos == len(data)
    if name in ("Float", "Double"):
        assert ours_dict["x"] == pytest.approx(theirs_dict["x"])
    else:
        assert ours_dict == theirs_dict


@pytest.mark.parametrize("name,values", VARINT_CASES + FIXED_CASES)
@pytest.mark.parametrize("repeated", [False, True])
def test_unpacked_decoder_factories_match_upstream(name, values, repeated):
    value = values if repeated else values[1]
    data = encoded(getattr(upstream_encoder, name + "Encoder"), 11, repeated, False, value)
    _, pos = upstream_decoder.ReadTag(memoryview(data), 0)
    ours_dict = {}
    theirs_dict = {}
    new = lambda unused: []
    ours_pos = getattr(decoder, name + "Decoder")(11, repeated, False, "x", new)(
        memoryview(data), pos, len(data), None, ours_dict
    )
    theirs_pos = getattr(upstream_decoder, name + "Decoder")(11, repeated, False, "x", new)(
        memoryview(data), pos, len(data), None, theirs_dict
    )
    assert ours_pos == theirs_pos == len(data)
    if name in ("Float", "Double"):
        if repeated:
            assert ours_dict["x"] == pytest.approx(theirs_dict["x"])
        else:
            assert ours_dict["x"] == pytest.approx(theirs_dict["x"])
    else:
        assert ours_dict == theirs_dict


@pytest.mark.parametrize("name,value", [("String", "wire text"), ("Bytes", b"wire\x00bytes")])
@pytest.mark.parametrize("repeated", [False, True])
def test_length_delimited_decoders_match_upstream(name, value, repeated):
    actual = [value, value] if repeated else value
    data = encoded(getattr(upstream_encoder, name + "Encoder"), 2, repeated, False, actual)
    _, pos = upstream_decoder.ReadTag(memoryview(data), 0)
    ours_dict = {}
    theirs_dict = {}
    new = (lambda unused: []) if repeated else (lambda unused: None)
    ours_pos = getattr(decoder, name + "Decoder")(2, repeated, False, "x", new)(
        memoryview(data), pos, len(data), None, ours_dict
    )
    theirs_pos = getattr(upstream_decoder, name + "Decoder")(2, repeated, False, "x", new)(
        memoryview(data), pos, len(data), None, theirs_dict
    )
    assert ours_pos == theirs_pos == len(data)
    assert ours_dict == theirs_dict


@pytest.mark.parametrize("field", [1, 15, 16, 2048, (1 << 29) - 1])
@pytest.mark.parametrize("wire", range(6))
def test_tags_match_upstream(field, wire):
    assert wire_format.PackTag(field, wire) == upstream_wire.PackTag(field, wire)
    assert wire_format.UnpackTag(wire_format.PackTag(field, wire)) == (field, wire)
    assert encoder.TagBytes(field, wire) == upstream_encoder.TagBytes(field, wire)


@pytest.mark.parametrize("value", [0, -1, 1, -2, 2**31 - 1, -(2**63)])
def test_zigzag_matches_upstream(value):
    encoded = wire_format.ZigZagEncode(value)
    assert encoded == upstream_wire.ZigZagEncode(value)
    assert wire_format.ZigZagDecode(encoded) == value


@pytest.mark.parametrize("name,values", VARINT_CASES + FIXED_CASES)
def test_sizers_match_upstream(name, values):
    ours = getattr(encoder, name + "Sizer")
    theirs = getattr(upstream_encoder, name + "Sizer")
    for repeated, packed, value in [
        (False, False, values[1]),
        (True, False, values),
        (True, True, values),
    ]:
        assert ours(23, repeated, packed)(value) == theirs(23, repeated, packed)(value)


@pytest.mark.parametrize("name,value", [("String", "héllo"), ("Bytes", b"\x00wire")])
def test_length_delimited_sizers_match_upstream(name, value):
    ours = getattr(encoder, name + "Sizer")
    theirs = getattr(upstream_encoder, name + "Sizer")
    for repeated, actual in [(False, value), (True, [value, value])]:
        assert ours(23, repeated, False)(actual) == theirs(23, repeated, False)(actual)


def test_wire_format_byte_size_helpers_match_upstream():
    cases = [
        ("Int32ByteSize", -1),
        ("Int64ByteSize", -(2**63)),
        ("UInt32ByteSize", 2**32 - 1),
        ("UInt64ByteSize", 2**64 - 1),
        ("SInt32ByteSize", -(2**31)),
        ("SInt64ByteSize", -(2**63)),
        ("Fixed32ByteSize", 1),
        ("SFixed32ByteSize", -1),
        ("FloatByteSize", 1.25),
        ("Fixed64ByteSize", 1),
        ("SFixed64ByteSize", -1),
        ("DoubleByteSize", 1.25),
        ("BoolByteSize", True),
        ("EnumByteSize", -1),
        ("BytesByteSize", b"wire"),
        ("StringByteSize", "héllo"),
    ]
    for name, value in cases:
        assert getattr(wire_format, name)(19, value) == getattr(upstream_wire, name)(19, value)
