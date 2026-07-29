import io

import numpy as np
import pytest
from google.protobuf import message
from google.protobuf.internal import decoder as upstream_decoder
from google.protobuf.internal import encoder as upstream_encoder
from google.protobuf.internal import wire_format as upstream_wire

import mojo_protobuf as mpb


def upstream_encode(values, signed=False, zigzag=False):
    write = io.BytesIO()
    fn = upstream_encoder._EncodeSignedVarint if signed and not zigzag else upstream_encoder._EncodeVarint
    for value in values:
        value = int(value)
        if zigzag:
            value = upstream_wire.ZigZagEncode(value)
        fn(write.write, value)
    return write.getvalue()


def upstream_decode(data, signed=False, zigzag=False, bits=64):
    fn = {
        (False, 64): upstream_decoder._DecodeVarint,
        (False, 32): upstream_decoder._DecodeVarint32,
        (True, 64): upstream_decoder._DecodeSignedVarint,
        (True, 32): upstream_decoder._DecodeSignedVarint32,
    }[signed and not zigzag, bits]
    values = []
    pos = 0
    while pos < len(data):
        value, pos = fn(data, pos)
        values.append(upstream_wire.ZigZagDecode(value) if zigzag else value)
    return values


def test_unsigned_varint_boundaries_match_upstream():
    values = np.array(
        [0, 1, 127, 128, 255, 300, 16383, 16384, 2**32 - 1, 2**63, 2**64 - 1],
        dtype=np.uint64,
    )
    encoded = mpb.encode_varints(values)
    assert encoded == upstream_encode(values)
    assert mpb.decode_varints(encoded).tolist() == values.tolist()


def test_signed_varint_boundaries_match_upstream():
    values = np.array(
        [0, -1, 1, -2, 2**31 - 1, -(2**31), 2**63 - 1, -(2**63)],
        dtype=np.int64,
    )
    encoded = mpb.encode_varints(values, signed=True)
    assert encoded == upstream_encode(values, signed=True)
    assert mpb.decode_varints(encoded, signed=True).tolist() == values.tolist()


@pytest.mark.parametrize("bits", [32, 64])
def test_random_unsigned_varints_match_upstream(bits):
    rng = np.random.default_rng(42 + bits)
    high = 2**32 if bits == 32 else 2**64
    values = rng.integers(0, high, 10_000, dtype=np.uint32 if bits == 32 else np.uint64)
    encoded = mpb.encode_varints(values, bits=bits)
    assert encoded == upstream_encode(values)
    assert mpb.decode_varints(encoded, bits=bits).tolist() == upstream_decode(encoded, bits=bits)


@pytest.mark.parametrize("bits", [32, 64])
def test_random_zigzag_varints_match_upstream(bits):
    rng = np.random.default_rng(100 + bits)
    dtype = np.int32 if bits == 32 else np.int64
    values = rng.integers(np.iinfo(dtype).min, np.iinfo(dtype).max, 10_000, dtype=dtype)
    encoded = mpb.encode_varints(values, signed=True, zigzag=True, bits=bits)
    assert encoded == upstream_encode(values, zigzag=True)
    assert mpb.decode_varints(encoded, signed=True, zigzag=True, bits=bits).tolist() == values.tolist()


def test_empty_varint_stream_roundtrips():
    assert mpb.encode_varints([], signed=True) == b""
    assert mpb.decode_varints(b"").size == 0


def test_truncated_varint_raises_decode_error():
    with pytest.raises(message.DecodeError, match="Truncated"):
        mpb.decode_varints(b"\x80")


def test_overlong_varint_raises_decode_error():
    with pytest.raises(message.DecodeError, match="Too many"):
        mpb.decode_varints(b"\x80" * 10 + b"\x00")


def test_count_limit_is_enforced():
    data = mpb.encode_varints([1, 2, 3])
    with pytest.raises(ValueError, match="count"):
        mpb.decode_varints(data, count=2)
    with pytest.raises(ValueError, match="count"):
        mpb.decode_varints(data, count=0)


def test_low_level_varint_decoder_matches_upstream():
    data = b"prefix" + mpb.encode_varints([2**63 + 19]) + b"suffix"
    assert mpb.decoder._DecodeVarint(data, 6) == upstream_decoder._DecodeVarint(data, 6)
    signed = upstream_encode([-12345], signed=True)
    assert mpb.decoder._DecodeSignedVarint(signed, 0) == upstream_decoder._DecodeSignedVarint(signed, 0)


def test_bytesio_varint_decoder_matches_upstream():
    data = mpb.encode_varints([300])
    assert mpb.decoder._DecodeVarint(io.BytesIO(data)) == 300
    assert mpb.decoder._DecodeVarint(io.BytesIO(b"")) is None


def test_varint_input_validation():
    with pytest.raises(OverflowError):
        mpb.encode_varints([-1])
    with pytest.raises(OverflowError):
        mpb.encode_varints([2**32], bits=32)
    with pytest.raises(OverflowError):
        mpb.encode_varints([2**64])
    with pytest.raises(TypeError):
        mpb.encode_varints([1.5])
    with pytest.raises(ValueError):
        mpb.encode_varints([1], bits=16)
