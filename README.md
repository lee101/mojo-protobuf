# mojo-protobuf

`mojo-protobuf` is a Mojo port of the compute-heavy parts of Python
Protocol Buffers wire-format encoding and decoding. It batches work that the
pure-Python protobuf internals perform one integer at a time, while retaining
the same canonical wire representation.

This is a focused wire library, not another message implementation. It is
useful for columnar data, packed fields, wire inspection, and code that already
uses the low-level functions in `google.protobuf.internal.encoder`,
`decoder`, or `wire_format`.

## Coverage

The Mojo kernels implement:

- unsigned, signed two's-complement, and ZigZag varint streams for 32- and
  64-bit values;
- packed scalar payloads for `int32`, `uint32`, `sint32`, `int64`, `uint64`,
  `sint64`, enum, bool, fixed/sfixed 32- and 64-bit integers, float, and double;
- little-endian fixed-width array encoding and decoding;
- zero-copy views when scanning contiguous varint, fixed32, fixed64, and
  length-delimited input;
- tags, ZigZag transforms, byte-size helpers, and upstream-compatible encoder
  and decoder factories for all scalar, string, and bytes field types.

The covered Python factories preserve protobuf 7.35.1's names and signatures,
including `Int32Encoder(field_number, is_repeated, is_packed)` and
`Int32Decoder(field_number, is_repeated, is_packed, key, new_default,
clear_if_default=False)`. Import the matching module from this package:

```python
from mojo_protobuf import encoder, decoder, wire_format
```

Not covered: descriptors, generated classes, reflection, JSON/text formats,
maps, embedded-message factories, extensions, unknown-field storage, or group
wire types. `scan_fields` explicitly rejects deprecated groups. The official
`protobuf` package remains the message runtime and the parity reference.

## Install

The repository pins the tested Mojo nightly and all Python dependencies:

```bash
pixi install
pixi run build
```

The build creates `dist/libmojo-protobuf.so`. Run validation and benchmarks
with:

```bash
pixi run test
pixi run bench
```

## Usage

This example batch-encodes signed values, decodes them, and inspects a small
wire message:

```python
import numpy as np
import mojo_protobuf as mpb

values = np.array([-1, 0, 1, 300], dtype=np.int64)
payload = mpb.encode_varints(values, signed=True, zigzag=True)
assert mpb.decode_varints(
    payload, signed=True, zigzag=True
).tolist() == values.tolist()

message = mpb.encoder.TagBytes(3, mpb.wire_format.WIRETYPE_LENGTH_DELIMITED)
message += mpb.encode_varints([len(payload)]) + payload
field = mpb.scan_fields(message)[0]
assert (field.number, field.as_bytes()) == (3, payload)
```

Run it after building with `pixi run python example.py`, or paste it into
`pixi run python`.

## Benchmarks

Measured with `pixi run bench` on this x86_64 machine, Linux
6.8.0-136-generic, using protobuf 7.35.1. Times are the best of three warm
runs. A ratio above 1 means mojo-protobuf was faster.

| case | mojo-protobuf | reference | speedup | comparison |
|---|---:|---:|---:|---|
| encode packed uint64 (1M) | 8.79 ms | 1711.49 ms | 194.68x | Python encoder |
| encode packed sint64 (1M) | 7.14 ms | 2069.21 ms | 289.74x | Python encoder |
| decode packed uint64 (1M) | 52.26 ms | 1029.19 ms | 19.69x | Python decoder |
| serialize packed message (1M) | 10.71 ms | 16.85 ms | 1.57x | upb runtime |
| scan 250k varint fields | 182.61 ms | 285.39 ms | 1.56x | Python reference |

Materializing 250,000 Python `Field` records dominates the scanner benchmark.
Payload memoryviews are created lazily and cached on first access, avoiding an
eager allocation per field while keeping contiguous inputs zero-copy.

There is no GPU path. Protobuf wire kernels have low arithmetic intensity and
data-dependent field boundaries, so device transfers and launch overhead cost
more than these byte-oriented kernels can recover.

## How it works

All Mojo code is compiled as one shared-library unit. NumPy owns every input
and output allocation. ctypes passes only integer addresses and lengths across
the C ABI; the Mojo exports reconstruct mutable `UnsafePointer` values with
`AnyOrigin[mut=True]`. There are no cross-language allocations to free.

Varint encoders write into a caller-sized byte array (at most ten bytes per
64-bit input). Decoders write into contiguous integer arrays. Fixed-width
values use protobuf's required little-endian layout. `scan_fields` records
offsets into the original contiguous byte buffer, and the Python `Field.value`
is a memoryview, so length-delimited payload inspection does not copy bytes.
Noncontiguous inputs are copied once into a safe contiguous buffer.

The tests compare encoded bytes, decoded values, error cases, tags, factory
behavior, and a dynamically generated message directly against the installed
upstream protobuf package.
