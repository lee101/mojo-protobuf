comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def encode_one(value: UInt64, dst: BPtr, pos: Int) -> Int:
    var v = value
    var p = pos
    while v >= 128:
        dst[p] = UInt8((v & 127) | 128)
        v >>= 7
        p += 1
    dst[p] = UInt8(v)
    return p + 1


def encoded_size(value: UInt64) -> Int:
    var v = value
    var size = 1
    while v >= 128:
        v >>= 7
        size += 1
    return size


@export("mpb_encode_varints")
def mpb_encode_varints(
    src_addr: Int, n: Int, dst_addr: Int, dst_capacity: Int
) abi("C") -> Int:
    if n <= 0:
        return 0
    if src_addr == 0 or dst_addr == 0 or dst_capacity < n:
        return -6
    var src = U64Ptr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var pos = 0
    for i in range(n):
        if pos + encoded_size(src[i]) > dst_capacity:
            return -6
        pos = encode_one(src[i], dst, pos)
    return pos


@export("mpb_encode_signed_varints")
def mpb_encode_signed_varints(
    src_addr: Int, n: Int, dst_addr: Int, dst_capacity: Int
) abi("C") -> Int:
    if n <= 0:
        return 0
    if src_addr == 0 or dst_addr == 0 or dst_capacity < n:
        return -6
    var src = I64Ptr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var pos = 0
    for i in range(n):
        if pos + 10 > dst_capacity:
            return -6
        pos = encode_one(UInt64(src[i]), dst, pos)
    return pos


@export("mpb_encode_zigzag_varints")
def mpb_encode_zigzag_varints(
    src_addr: Int, n: Int, dst_addr: Int, dst_capacity: Int
) abi("C") -> Int:
    if n <= 0:
        return 0
    if src_addr == 0 or dst_addr == 0 or dst_capacity < n:
        return -6
    var src = I64Ptr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var pos = 0
    for i in range(n):
        if pos + 10 > dst_capacity:
            return -6
        var value = src[i]
        var encoded = (UInt64(value) << 1) ^ UInt64(value >> 63)
        pos = encode_one(encoded, dst, pos)
    return pos


def decode_one(src: BPtr, n: Int, start: Int, value: U64Ptr) -> Int:
    var pos = start
    var shift = 0
    var result = UInt64(0)
    while shift < 64:
        if pos >= n:
            return -1
        var byte = src[pos]
        pos += 1
        result |= UInt64(byte & 127) << UInt64(shift)
        if (byte & 128) == 0:
            value[0] = result
            return pos
        shift += 7
    return -2


@export("mpb_decode_varints")
def mpb_decode_varints(
    src_addr: Int, n: Int, dst_addr: Int, max_values: Int
) abi("C") -> Int:
    if n <= 0 or max_values <= 0:
        return 0
    if src_addr == 0 or dst_addr == 0:
        return -6
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = U64Ptr(unsafe_from_address=dst_addr)
    var pos = 0
    var count = 0
    while pos < n and count < max_values:
        var next_pos = decode_one(src, n, pos, dst + count)
        if next_pos < 0:
            return next_pos
        pos = next_pos
        count += 1
    if pos != n:
        return -3
    return count


@export("mpb_decode_zigzag_varints")
def mpb_decode_zigzag_varints(
    src_addr: Int, n: Int, dst_addr: Int, max_values: Int
) abi("C") -> Int:
    if n <= 0 or max_values <= 0:
        return 0
    if src_addr == 0 or dst_addr == 0:
        return -6
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = I64Ptr(unsafe_from_address=dst_addr)
    var raw_dst = U64Ptr(unsafe_from_address=dst_addr)
    var pos = 0
    var count = 0
    while pos < n and count < max_values:
        var next_pos = decode_one(src, n, pos, raw_dst + count)
        if next_pos < 0:
            return next_pos
        var encoded = raw_dst[count]
        dst[count] = Int64((encoded >> 1) ^ (UInt64(0) - (encoded & 1)))
        pos = next_pos
        count += 1
    if pos != n:
        return -3
    return count


@export("mpb_scan_fields")
def mpb_scan_fields(
    src_addr: Int,
    n: Int,
    field_addr: Int,
    wire_addr: Int,
    offset_addr: Int,
    length_addr: Int,
    max_fields: Int,
) abi("C") -> Int:
    if n <= 0:
        return 0
    if max_fields <= 0:
        return -3
    if (
        src_addr == 0
        or field_addr == 0
        or wire_addr == 0
        or offset_addr == 0
        or length_addr == 0
    ):
        return -6
    var src = BPtr(unsafe_from_address=src_addr)
    var fields = U32Ptr(unsafe_from_address=field_addr)
    var wires = BPtr(unsafe_from_address=wire_addr)
    var offsets = I64Ptr(unsafe_from_address=offset_addr)
    var scratch_values = U64Ptr(unsafe_from_address=offset_addr)
    var lengths = I64Ptr(unsafe_from_address=length_addr)
    var pos = 0
    var count = 0
    while pos < n:
        if count >= max_fields:
            return -3
        var next_pos = decode_one(src, n, pos, scratch_values + count)
        if next_pos < 0:
            return next_pos
        var tag = scratch_values[count]
        var field_number = tag >> 3
        var wire_type = Int(tag & 7)
        if field_number == 0 or field_number >= (UInt64(1) << 29):
            return -4
        pos = next_pos
        fields[count] = UInt32(field_number)
        wires[count] = UInt8(wire_type)
        if wire_type == 0:
            var value_start = pos
            next_pos = decode_one(src, n, pos, scratch_values + count)
            if next_pos < 0:
                return next_pos
            offsets[count] = Int64(value_start)
            lengths[count] = Int64(next_pos - value_start)
            pos = next_pos
        elif wire_type == 1:
            if pos + 8 > n:
                return -1
            offsets[count] = Int64(pos)
            lengths[count] = 8
            pos += 8
        elif wire_type == 2:
            next_pos = decode_one(src, n, pos, scratch_values + count)
            if next_pos < 0:
                return next_pos
            var payload_length = scratch_values[count]
            if payload_length > UInt64(n - next_pos):
                return -1
            pos = next_pos
            offsets[count] = Int64(pos)
            lengths[count] = Int64(payload_length)
            pos += Int(payload_length)
        elif wire_type == 5:
            if pos + 4 > n:
                return -1
            offsets[count] = Int64(pos)
            lengths[count] = 4
            pos += 4
        elif wire_type == 3 or wire_type == 4:
            return -5
        else:
            return -4
        count += 1
    return count
