from __future__ import annotations

import io
import os
import platform
import sys
import time

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
import google.protobuf
from google.protobuf.internal import decoder as upstream_decoder
from google.protobuf.internal import encoder as upstream_encoder

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_protobuf as mpb  # noqa: E402


def best(fn, repeat=3):
    result = None
    elapsed = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        elapsed = min(elapsed, time.perf_counter() - start)
    return elapsed, result


def factory_encode(factory, values, kind="UInt64"):
    sink = io.BytesIO()
    factory(1, True, True)(sink.write, values, True)
    return sink.getvalue()


def factory_decode(factory, data, kind="UInt64"):
    _, pos = upstream_decoder.ReadTag(memoryview(data), 0)
    fields = {}
    factory(1, True, True, "x", lambda unused: [])(
        memoryview(data), pos, len(data), None, fields
    )
    return fields["x"]


def message_class():
    file_proto = descriptor_pb2.FileDescriptorProto(name="bench.proto", package="bench")
    msg = file_proto.message_type.add(name="Packed")
    field = msg.field.add(
        name="value",
        number=1,
        type=descriptor_pb2.FieldDescriptorProto.TYPE_UINT64,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    field.options.packed = True
    descriptor = descriptor_pool.DescriptorPool().Add(file_proto).message_types_by_name["Packed"]
    return message_factory.GetMessageClass(descriptor)


def python_scan_varints(data):
    view = memoryview(data)
    result = []
    pos = 0
    while pos < len(view):
        tag, pos = upstream_decoder._DecodeVarint(view, pos)
        start = pos
        _, pos = upstream_decoder._DecodeVarint(view, pos)
        result.append((tag >> 3, tag & 7, start, pos - start))
    return result


def main():
    rng = np.random.default_rng(0)
    values = rng.integers(0, 1 << 28, 1_000_000, dtype=np.uint64)
    signed = rng.integers(-(1 << 27), 1 << 27, 1_000_000, dtype=np.int64)

    ours_uint = lambda: factory_encode(mpb.encoder.UInt64Encoder, values)
    upstream_uint = lambda: factory_encode(upstream_encoder.UInt64Encoder, values)
    encoded_uint = ours_uint()
    assert encoded_uint == upstream_uint()

    ours_signed = lambda: factory_encode(mpb.encoder.SInt64Encoder, signed)
    upstream_signed = lambda: factory_encode(upstream_encoder.SInt64Encoder, signed)
    encoded_signed = ours_signed()
    assert encoded_signed == upstream_signed()

    ours_decode = lambda: factory_decode(mpb.decoder.UInt64Decoder, encoded_uint)
    upstream_decode = lambda: factory_decode(upstream_decoder.UInt64Decoder, encoded_uint)
    assert ours_decode() == upstream_decode()

    Packed = message_class()
    record = Packed(value=values.tolist())
    assert record.SerializeToString() == encoded_uint
    ours_full = ours_uint
    upstream_full = record.SerializeToString

    scalar_data = (upstream_encoder.TagBytes(1, 0) + upstream_encoder._VarintBytes(150)) * 250_000
    assert len(mpb.scan_fields(scalar_data)) == len(python_scan_varints(scalar_data))

    cases = [
        ("encode packed uint64 (1M)", ours_uint, upstream_uint, "Python encoder"),
        ("encode packed sint64 (1M)", ours_signed, upstream_signed, "Python encoder"),
        ("decode packed uint64 (1M)", ours_decode, upstream_decode, "Python decoder"),
        ("serialize packed message (1M)", ours_full, upstream_full, "upb runtime"),
        (
            "scan 250k varint fields",
            lambda: mpb.scan_fields(scalar_data),
            lambda: python_scan_varints(scalar_data),
            "Python reference",
        ),
    ]

    print(
        f"Machine: {platform.processor() or 'x86_64'}; {platform.system()} "
        f"{platform.release()}; protobuf {google.protobuf.__version__}"
    )
    print()
    print("| case | mojo-protobuf | reference | speedup | comparison |")
    print("|---|---:|---:|---:|---|")
    for name, ours, reference, label in cases:
        ours()
        reference()
        ours_time, _ = best(ours)
        reference_time, _ = best(reference)
        ratio = reference_time / ours_time
        print(
            f"| {name} | {ours_time * 1000:.2f} ms | {reference_time * 1000:.2f} ms "
            f"| {ratio:.2f}x | {label} |"
        )


if __name__ == "__main__":
    main()
