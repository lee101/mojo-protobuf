import struct

import numpy as np
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message, message_factory
from google.protobuf.internal import encoder as upstream_encoder

import mojo_protobuf as mpb


FIXED_CASES = [
    ("fixed32", np.array([0, 1, 2**32 - 1], dtype=np.uint32), "<I"),
    ("sfixed32", np.array([-(2**31), -1, 2**31 - 1], dtype=np.int32), "<i"),
    ("float", np.array([0.0, -1.25, np.inf, -np.inf], dtype=np.float32), "<f"),
    ("fixed64", np.array([0, 1, 2**64 - 1], dtype=np.uint64), "<Q"),
    ("sfixed64", np.array([-(2**63), -1, 2**63 - 1], dtype=np.int64), "<q"),
    ("double", np.array([0.0, -1.25, 1e200], dtype=np.float64), "<d"),
]


@pytest.mark.parametrize("kind,values,fmt", FIXED_CASES)
def test_fixed_arrays_match_struct(kind, values, fmt):
    expected = b"".join(struct.pack(fmt, x) for x in values)
    encoded = mpb.encode_fixed(values, kind)
    assert encoded == expected
    np.testing.assert_equal(mpb.decode_fixed(encoded, kind), values)


@pytest.mark.parametrize(
    "kind,values",
    [
        ("int32", [-1, 0, 1, 2**31 - 1]),
        ("uint64", [0, 127, 128, 2**63]),
        ("sint64", [-(2**63), -1, 0, 2**63 - 1]),
        ("bool", [False, True, True]),
        ("fixed32", [0, 1, 2**32 - 1]),
        ("double", [-1.0, 0.0, 1e50]),
    ],
)
def test_packed_payload_roundtrip(kind, values):
    encoded = mpb.encode_packed(values, kind)
    result = mpb.decode_packed(encoded, kind)
    if kind == "double":
        assert result == pytest.approx(values)
    else:
        assert result.tolist() == values


def test_signed_enum_packed_payload_matches_int32():
    values = [-1, 0, 2**31 - 1]
    encoded = mpb.encode_packed(values, "enum")
    assert encoded == mpb.encode_packed(values, "int32")
    assert mpb.decode_packed(encoded, "enum").tolist() == values


@pytest.mark.parametrize(
    "kind,values,error",
    [
        ("fixed32", [-1], OverflowError),
        ("sfixed32", [2**31], OverflowError),
        ("fixed64", [2**64], OverflowError),
        ("fixed32", [1.5], TypeError),
    ],
)
def test_fixed_encoding_rejects_lossy_integer_conversions(kind, values, error):
    with pytest.raises(error):
        mpb.encode_fixed(values, kind)


def test_float32_encoding_rejects_overflow_but_accepts_infinity():
    with pytest.raises(OverflowError):
        mpb.encode_fixed([1e300], "float")
    assert mpb.decode_fixed(mpb.encode_fixed([np.inf], "float"), "float")[0] == np.inf


def test_fixed_decode_accepts_noncontiguous_byte_view():
    storage = np.zeros(8, dtype=np.uint8)
    storage[::2] = np.frombuffer(struct.pack("<I", 0xDEADBEEF), dtype=np.uint8)
    assert mpb.decode_fixed(storage[::2], "fixed32").tolist() == [0xDEADBEEF]


def test_packed_length_mismatch_is_rejected():
    with pytest.raises(message.DecodeError, match="length"):
        mpb.decode_packed(b"\x03\x01\x02", "uint32")


def build_message_class():
    file_proto = descriptor_pb2.FileDescriptorProto(name="wire_test.proto", package="wire")
    msg = file_proto.message_type.add(name="Record")
    fields = [
        ("counter", 1, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64, False),
        ("signed", 2, descriptor_pb2.FieldDescriptorProto.TYPE_SINT64, False),
        ("fixed", 3, descriptor_pb2.FieldDescriptorProto.TYPE_FIXED32, False),
        ("ratio", 4, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE, False),
        ("payload", 5, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES, False),
        ("samples", 6, descriptor_pb2.FieldDescriptorProto.TYPE_INT32, True),
    ]
    for name, number, kind, repeated in fields:
        field = msg.field.add(name=name, number=number, type=kind)
        if repeated:
            field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
            field.options.packed = True
    descriptor = descriptor_pool.DescriptorPool().Add(file_proto).message_types_by_name["Record"]
    return message_factory.GetMessageClass(descriptor)


def test_scan_fields_matches_generated_protobuf_message():
    Record = build_message_class()
    record = Record(
        counter=2**63 + 9,
        signed=-123456789,
        fixed=0xDEADBEEF,
        ratio=1.25,
        payload=b"hello\x00wire",
        samples=[-1, 0, 1, 300],
    )
    data = record.SerializeToString(deterministic=True)
    fields = mpb.scan_fields(data)
    assert [field.number for field in fields] == [1, 2, 3, 4, 5, 6]
    assert [field.wire_type for field in fields] == [0, 0, 5, 1, 2, 2]
    assert fields[0].as_varint() == record.counter
    assert fields[1].as_varint(zigzag=True) == record.signed
    assert fields[2].as_fixed32() == record.fixed
    assert struct.unpack("<d", fields[3].value)[0] == record.ratio
    assert fields[4].as_bytes() == record.payload
    assert mpb.decode_varints(fields[5].value, signed=True, bits=32).tolist() == list(record.samples)


def test_scan_fields_returns_zero_copy_views():
    data = bytearray(upstream_encoder.TagBytes(1, 2) + b"\x03abc")
    field = mpb.scan_fields(data)[0]
    data[-1] = ord("z")
    assert field.as_bytes() == b"abz"


def test_scan_fields_uses_byte_offsets_for_typed_buffers():
    data = np.frombuffer(b"\x0a\x02ab", dtype=np.uint16)
    field = mpb.scan_fields(data)[0]
    assert (field.number, field.as_bytes()) == (1, b"ab")


def test_scan_fields_copies_noncontiguous_input_safely():
    storage = np.zeros(8, dtype=np.uint8)
    storage[::2] = np.frombuffer(b"\x0a\x02ab", dtype=np.uint8)
    field = mpb.scan_fields(storage[::2])[0]
    storage[-2] = ord("z")
    assert field.as_bytes() == b"ab"


def test_scan_fields_handles_dense_scalar_fields():
    data = (upstream_encoder.TagBytes(1, 0) + b"\x00") * 257
    fields = mpb.scan_fields(data)
    assert len(fields) == 257
    assert all(field.number == 1 and field.as_varint() == 0 for field in fields)


@pytest.mark.parametrize(
    "data,error",
    [
        (b"\x08\x80", "Truncated"),
        (b"\x0a\x05abc", "Truncated"),
        (b"\x0f", "Invalid"),
        (b"\x00", "Invalid"),
    ],
)
def test_scan_fields_rejects_malformed_messages(data, error):
    with pytest.raises(message.DecodeError, match=error):
        mpb.scan_fields(data)


def test_scan_fields_reports_groups_as_unsupported():
    with pytest.raises(NotImplementedError, match="group"):
        mpb.scan_fields(b"\x0b\x0c")
