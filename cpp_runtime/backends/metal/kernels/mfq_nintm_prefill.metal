#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;

namespace {

inline uint read_bits(
    const device uchar* stream,
    uint value_index,
    uint bits) {
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index =
        (value_index >> 3u) * bits + (residual_bits >> 3u);
    uint shift = residual_bits & 7u;
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) {
        packed |= uint(stream[byte_index + 1u]) << 8u;
    }
    if (shift + bits > 16u) {
        packed |= uint(stream[byte_index + 2u]) << 16u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

#if defined(MFQ_ENABLE_LEGACY_VQ_VECTOR) \
    || defined(MFQ_ENABLE_JSC_EXTENDED_VECTOR)
template <uint BITS>
inline uint3 read_vq_group_indices(
    const device uchar* stream,
    uint row,
    uint group,
    uint vectors) {
    uint first = group * 3u;
    if (first + 2u >= vectors) {
        return uint3(
            first < vectors
                ? read_bits(stream, row * vectors + first, BITS)
                : 0u,
            first + 1u < vectors
                ? read_bits(stream, row * vectors + first + 1u, BITS)
                : 0u,
            0u);
    }
    uint bit = (row * vectors + first) * BITS;
    uint byte = bit >> 3u;
    uint shift = bit & 7u;
    ulong packed = ulong(stream[byte])
        | (ulong(stream[byte + 1u]) << 8u)
        | (ulong(stream[byte + 2u]) << 16u);
    if (shift + 3u * BITS > 24u) {
        packed |= ulong(stream[byte + 3u]) << 24u;
    }
    if (shift + 3u * BITS > 32u) {
        packed |= ulong(stream[byte + 4u]) << 32u;
    }
    ulong mask = (1ul << BITS) - 1ul;
    return uint3(
        uint((packed >> shift) & mask),
        uint((packed >> (shift + BITS)) & mask),
        uint((packed >> (shift + 2u * BITS)) & mask));
}
#endif

inline uint read_nint_value(
    const device uchar* stream,
    uint value_index,
    uint bits,
    uint group_size,
    uint q5_execution) {
    if (q5_execution != 0u && bits == 5u) {
        uint metadata = value_index / group_size;
        uint element = value_index - metadata * group_size;
        uint low_bytes = (group_size + 1u) >> 1u;
        uint high_bytes = (group_size + 7u) >> 3u;
        uint base = metadata * (low_bytes + high_bytes);
        uint low = (uint(stream[base + (element >> 1u)])
            >> ((element & 1u) * 4u)) & 15u;
        uint high = (uint(stream[base + low_bytes + (element >> 3u)])
            >> (element & 7u)) & 1u;
        return low | (high << 4u);
    }
    return read_bits(stream, value_index, bits);
}

inline ushort4 read_nint_quad(
    const device uchar* stream,
    uint value_index,
    uint bits,
    uint group_size,
    uint q5_execution) {
    // All production NINT group sizes are multiples of four and callers feed
    // a four-value-aligned index.  Decode the packed bytes once per quad
    // instead of repeating overlapping scalar loads.
    if (q5_execution != 0u && bits == 5u) {
        uint metadata = value_index / group_size;
        uint element = value_index - metadata * group_size;
        uint low_bytes = (group_size + 1u) >> 1u;
        uint high_bytes = (group_size + 7u) >> 3u;
        uint base = metadata * (low_bytes + high_bytes);
        uchar2 low = *reinterpret_cast<device const uchar2*>(
            stream + base + (element >> 1u));
        uchar high = stream[base + low_bytes + (element >> 3u)];
        uint high_shift = element & 7u;
        return ushort4(
            uint(low.x & 15u) | (((uint(high) >> high_shift) & 1u) << 4u),
            uint(low.x >> 4u) | (((uint(high) >> (high_shift + 1u)) & 1u) << 4u),
            uint(low.y & 15u) | (((uint(high) >> (high_shift + 2u)) & 1u) << 4u),
            uint(low.y >> 4u) | (((uint(high) >> (high_shift + 3u)) & 1u) << 4u));
    }

    uint bit_offset = value_index * bits;
    uint byte_index = bit_offset >> 3u;
    uint shift = bit_offset & 7u;
    if (bits == 2u) {
        uint packed = uint(stream[byte_index]);
        return ushort4(
            packed & 3u,
            (packed >> 2u) & 3u,
            (packed >> 4u) & 3u,
            (packed >> 6u) & 3u);
    }
    if (bits == 3u) {
        uint packed = uint(stream[byte_index])
            | (uint(stream[byte_index + 1u]) << 8u);
        packed >>= shift;
        return ushort4(
            packed & 7u,
            (packed >> 3u) & 7u,
            (packed >> 6u) & 7u,
            (packed >> 9u) & 7u);
    }
    if (bits == 4u) {
        uchar2 packed = *reinterpret_cast<device const uchar2*>(
            stream + byte_index);
        return ushort4(
            packed.x & 15u,
            packed.x >> 4u,
            packed.y & 15u,
            packed.y >> 4u);
    }
    if (bits == 5u) {
        uint packed = uint(stream[byte_index])
            | (uint(stream[byte_index + 1u]) << 8u)
            | (uint(stream[byte_index + 2u]) << 16u);
        packed >>= shift;
        return ushort4(
            packed & 31u,
            (packed >> 5u) & 31u,
            (packed >> 10u) & 31u,
            (packed >> 15u) & 31u);
    }
    if (bits == 6u) {
        uint packed = uint(stream[byte_index])
            | (uint(stream[byte_index + 1u]) << 8u)
            | (uint(stream[byte_index + 2u]) << 16u);
        return ushort4(
            packed & 63u,
            (packed >> 6u) & 63u,
            (packed >> 12u) & 63u,
            (packed >> 18u) & 63u);
    }
    uchar4 packed = *reinterpret_cast<device const uchar4*>(
        stream + byte_index);
    return ushort4(packed.x, packed.y, packed.z, packed.w);
}

inline float decode_mxfp4_value(uchar raw) {
    uchar magnitude = raw & 7u;
    float value = magnitude == 0u ? 0.0f
        : (magnitude == 1u ? 0.5f
        : (magnitude == 2u ? 1.0f
        : (magnitude == 3u ? 1.5f
        : (magnitude == 4u ? 2.0f
        : (magnitude == 5u ? 3.0f
        : (magnitude == 6u ? 4.0f : 6.0f))))));
    return (raw & 8u) == 0u ? value : -value;
}

constant float kMxfp4DecodeLut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

inline float decode_e8m0(uchar raw) {
    if (raw == 255u) {
        return NAN;
    }
    uint bits = raw == 0u ? 0x00400000u : uint(raw) << 23u;
    return as_type<float>(bits);
}

inline float decode_mxfp8_value(uchar raw) {
    uint magnitude = uint(raw & 0x7fu);
    uint exponent = magnitude >> 3u;
    uint mantissa = magnitude & 7u;
    if (exponent == 15u && mantissa == 7u) {
        return NAN;
    }
    float value = exponent == 0u
        ? float(mantissa) * 0.001953125f
        : as_type<float>((exponent + 120u) << 23u)
            * (1.0f + float(mantissa) * 0.125f);
    return (raw & 0x80u) == 0u ? value : -value;
}

inline void decode_mxfp4_group32(
    const device int* d,
    const device uchar* values,
    const device uchar* scales,
    threadgroup half* target,
    uint row,
    uint group,
    uint k_size) {
    uint groups = uint(d[4]);
    uint value_offset = uint(d[5]);
    uint scale_offset = uint(d[6]);
    float scale = decode_e8m0(
        scales[scale_offset + row * groups + group]);
    uint packed_row = value_offset + row * (k_size >> 1u);
    uint packed_column = group * 16u;
    // Four vector loads replace sixteen scalar loads.  The constant E2M1
    // table is exact, so this preserves the previous FP32 multiply -> FP16
    // conversion while avoiding the nested magnitude branch per nibble.
#pragma clang loop unroll(full)
    for (uint chunk = 0u; chunk < 4u; ++chunk) {
        uchar4 packed = *reinterpret_cast<device const uchar4*>(
            values + packed_row + packed_column + chunk * 4u);
        half4 first = half4(
            scale * kMxfp4DecodeLut[packed.x & 15u],
            scale * kMxfp4DecodeLut[packed.x >> 4u],
            scale * kMxfp4DecodeLut[packed.y & 15u],
            scale * kMxfp4DecodeLut[packed.y >> 4u]);
        half4 second = half4(
            scale * kMxfp4DecodeLut[packed.z & 15u],
            scale * kMxfp4DecodeLut[packed.z >> 4u],
            scale * kMxfp4DecodeLut[packed.w & 15u],
            scale * kMxfp4DecodeLut[packed.w >> 4u]);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u) = first;
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u + 4u) = second;
    }
}

inline half decode_mxfp8_value_at(
    const device int* d,
    const device uchar* values,
    const device uchar* scales,
    uint row,
    uint column,
    uint k_size) {
    uint groups = uint(d[4]);
    uint value_offset = uint(d[5]);
    uint scale_offset = uint(d[6]);
    uint group = column >> 7u;
    float scale = decode_e8m0(
        scales[scale_offset + (row >> 7u) * groups + group]);
    return half(scale * decode_mxfp8_value(
        values[value_offset + row * k_size + column]));
}

inline half4 decode_mxfp8_quad_at(
    const device int* d,
    const device uchar* values,
    const device uchar* scales,
    uint row,
    uint column,
    uint k_size) {
    uint groups = uint(d[4]);
    uint value_offset = uint(d[5]);
    uint scale_offset = uint(d[6]);
    uint group = column >> 7u;
    float scale = decode_e8m0(
        scales[scale_offset + (row >> 7u) * groups + group]);
    uchar4 packed = *reinterpret_cast<device const uchar4*>(
        values + value_offset + row * k_size + column);
    float4 decoded = float4(
        decode_mxfp8_value(packed.x),
        decode_mxfp8_value(packed.y),
        decode_mxfp8_value(packed.z),
        decode_mxfp8_value(packed.w));
    return half4(scale * decoded);
}

inline half decode_dense_value_at(
    const device int* d,
    const device int8_t* values,
    uint family,
    uint row,
    uint column,
    uint k_size) {
    ulong byte_offset = ulong(uint(d[4]))
        + (ulong(row) * ulong(k_size) + ulong(column)) * 2u;
    ushort raw = *reinterpret_cast<device const ushort*>(
        values + byte_offset);
    return family == 5u
        ? half(as_type<float>(uint(raw) << 16u))
        : as_type<half>(raw);
}

inline half4 decode_dense_quad_at(
    const device int* d,
    const device int8_t* values,
    uint family,
    uint row,
    uint column,
    uint k_size) {
    ulong byte_offset = ulong(uint(d[4]))
        + (ulong(row) * ulong(k_size) + ulong(column)) * 2u;
    if (family == 5u) {
        ushort4 raw = *reinterpret_cast<device const ushort4*>(
            values + byte_offset);
        return half4(as_type<float4>(uint4(raw) << 16u));
    }
    return *reinterpret_cast<device const half4*>(values + byte_offset);
}

inline void decode_nint4_group24(
    const device int* d,
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    threadgroup half* target,
    uint row,
    uint group) {
    uint groups = uint(d[6]);
    uint q_offset = uint(d[7]);
    uint sub_offset = uint(d[8]);
    uint anchor_offset = uint(d[9]);
    uint metadata = row * groups + group;
    float scale = anchor_scales[anchor_offset + row]
        * float(sub_scales[sub_offset + metadata]);
    float minimum = anchor_mins[anchor_offset + row]
        * float(sub_mins[sub_offset + metadata]);
    uint packed_base = q_offset + metadata * 12u;
    for (uint chunk = 0u; chunk < 3u; ++chunk) {
        uchar4 packed = *reinterpret_cast<device const uchar4*>(
            values + packed_base + chunk * 4u);
        half4 first = half4(
            scale * float(packed.x & 15u) - minimum,
            scale * float(packed.x >> 4u) - minimum,
            scale * float(packed.y & 15u) - minimum,
            scale * float(packed.y >> 4u) - minimum);
        half4 second = half4(
            scale * float(packed.z & 15u) - minimum,
            scale * float(packed.z >> 4u) - minimum,
            scale * float(packed.w & 15u) - minimum,
            scale * float(packed.w >> 4u) - minimum);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u) = first;
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u + 4u) = second;
    }
}

inline void decode_nint3_group24(
    const device int* d,
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    threadgroup half* target,
    uint row,
    uint group) {
    uint groups = uint(d[6]);
    uint q_offset = uint(d[7]);
    uint sub_offset = uint(d[8]);
    uint anchor_offset = uint(d[9]);
    uint metadata = row * groups + group;
    float scale = anchor_scales[anchor_offset + row]
        * float(sub_scales[sub_offset + metadata]);
    float minimum = anchor_mins[anchor_offset + row]
        * float(sub_mins[sub_offset + metadata]);
    uint packed_base = q_offset + metadata * 9u;
#pragma clang loop unroll(full)
    for (uint chunk = 0u; chunk < 3u; ++chunk) {
        uint chunk_base = packed_base + chunk * 3u;
        uint packed = uint(values[chunk_base])
            | (uint(values[chunk_base + 1u]) << 8u)
            | (uint(values[chunk_base + 2u]) << 16u);
        half4 first = half4(
            scale * float(packed & 7u) - minimum,
            scale * float((packed >> 3u) & 7u) - minimum,
            scale * float((packed >> 6u) & 7u) - minimum,
            scale * float((packed >> 9u) & 7u) - minimum);
        half4 second = half4(
            scale * float((packed >> 12u) & 7u) - minimum,
            scale * float((packed >> 15u) & 7u) - minimum,
            scale * float((packed >> 18u) & 7u) - minimum,
            scale * float((packed >> 21u) & 7u) - minimum);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u) = first;
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u + 4u) = second;
    }
}

inline void decode_nint6_group24(
    const device int* d,
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    threadgroup half* target,
    uint row,
    uint group) {
    uint groups = uint(d[6]);
    uint q_offset = uint(d[7]);
    uint sub_offset = uint(d[8]);
    uint anchor_offset = uint(d[9]);
    uint metadata = row * groups + group;
    float scale = anchor_scales[anchor_offset + row]
        * float(sub_scales[sub_offset + metadata]);
    float minimum = anchor_mins[anchor_offset + row]
        * float(sub_mins[sub_offset + metadata]);
    uint packed_base = q_offset + metadata * 18u;
#pragma clang loop unroll(full)
    for (uint chunk = 0u; chunk < 6u; ++chunk) {
        uint chunk_base = packed_base + chunk * 3u;
        uint packed = uint(values[chunk_base])
            | (uint(values[chunk_base + 1u]) << 8u)
            | (uint(values[chunk_base + 2u]) << 16u);
        half4 decoded = half4(
            scale * float(packed & 63u) - minimum,
            scale * float((packed >> 6u) & 63u) - minimum,
            scale * float((packed >> 12u) & 63u) - minimum,
            scale * float((packed >> 18u) & 63u) - minimum);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 4u) = decoded;
    }
}

inline void decode_nint2_packed16(
    uchar4 packed,
    float scale,
    float minimum,
    threadgroup half* target) {
#pragma clang loop unroll(full)
    for (uint chunk = 0u; chunk < 4u; ++chunk) {
        uint value = uint(packed[chunk]);
        half4 decoded = half4(
            scale * float(value & 3u) - minimum,
            scale * float((value >> 2u) & 3u) - minimum,
            scale * float((value >> 4u) & 3u) - minimum,
            scale * float((value >> 6u) & 3u) - minimum);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 4u) = decoded;
    }
}

inline void decode_nint2_group16(
    const device int* d,
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    threadgroup half* target,
    uint row,
    uint group) {
    uint groups = uint(d[6]);
    uint q_offset = uint(d[7]);
    uint sub_offset = uint(d[8]);
    uint anchor_offset = uint(d[9]);
    uint metadata = row * groups + group;
    float scale = anchor_scales[anchor_offset + row]
        * float(sub_scales[sub_offset + metadata]);
    float minimum = anchor_mins[anchor_offset + row]
        * float(sub_mins[sub_offset + metadata]);
    uchar4 packed = *reinterpret_cast<device const uchar4*>(
        values + q_offset + metadata * 4u);
    decode_nint2_packed16(packed, scale, minimum, target);
}

inline void decode_nint2_pair32(
    const device int* d,
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    threadgroup half* target,
    uint row,
    uint first_group) {
    uint groups = uint(d[6]);
    uint q_offset = uint(d[7]);
    uint sub_offset = uint(d[8]);
    uint anchor_offset = uint(d[9]);
    uint first_metadata = row * groups + first_group;
    float anchor_scale = anchor_scales[anchor_offset + row];
    float anchor_minimum = anchor_mins[anchor_offset + row];
    float first_scale = anchor_scale
        * float(sub_scales[sub_offset + first_metadata]);
    float first_minimum = anchor_minimum
        * float(sub_mins[sub_offset + first_metadata]);
    float second_scale = anchor_scale
        * float(sub_scales[sub_offset + first_metadata + 1u]);
    float second_minimum = anchor_minimum
        * float(sub_mins[sub_offset + first_metadata + 1u]);
    uchar4 first_packed = *reinterpret_cast<device const uchar4*>(
        values + q_offset + first_metadata * 4u);
    uchar4 second_packed = *reinterpret_cast<device const uchar4*>(
        values + q_offset + (first_metadata + 1u) * 4u);
    decode_nint2_packed16(
        first_packed, first_scale, first_minimum, target);
    decode_nint2_packed16(
        second_packed, second_scale, second_minimum, target + 16u);
}

template <uint BITS, uint GROUP_SIZE>
inline void decode_nint_aligned_group(
    const device int* d,
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    threadgroup half* target,
    uint row,
    uint group,
    uint neuron_len) {
    uint groups = uint(d[6]);
    uint q_offset = uint(d[7]);
    uint sub_offset = uint(d[8]);
    uint anchor_offset = uint(d[9]);
    uint metadata = row * groups + group;
    float scale = anchor_scales[anchor_offset + row]
        * float(sub_scales[sub_offset + metadata]);
    float minimum = anchor_mins[anchor_offset + row]
        * float(sub_mins[sub_offset + metadata]);
    uint first_column = group * GROUP_SIZE;
#pragma clang loop unroll(full)
    for (uint column = 0; column < GROUP_SIZE; column += 4u) {
        ushort4 quantized = read_nint_quad(
            values + q_offset,
            metadata * GROUP_SIZE + column,
            BITS,
            GROUP_SIZE,
            0u);
#pragma clang loop unroll(full)
        for (uint element = 0; element < 4u; ++element) {
            if (first_column + column + element < neuron_len) {
                target[column + element] = half(
                    scale * float(quantized[element]) - minimum);
            }
        }
    }
}

inline void decode_nint8_group48(
    const device int* d,
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    threadgroup half* target,
    uint row,
    uint group,
    uint neuron_len) {
    constexpr uint GROUP_SIZE = 48u;
    uint groups = uint(d[6]);
    uint q_offset = uint(d[7]);
    uint sub_offset = uint(d[8]);
    uint anchor_offset = uint(d[9]);
    uint metadata = row * groups + group;
    float scale = anchor_scales[anchor_offset + row]
        * float(sub_scales[sub_offset + metadata]);
    float minimum = anchor_mins[anchor_offset + row]
        * float(sub_mins[sub_offset + metadata]);
    uint first_column = group * GROUP_SIZE;
    uint valid = first_column < neuron_len
        ? min(GROUP_SIZE, neuron_len - first_column)
        : 0u;
    uint packed_base = q_offset + metadata * GROUP_SIZE;
#pragma clang loop unroll(full)
    for (uint column = 0u; column < GROUP_SIZE; column += 4u) {
        uchar4 packed = *reinterpret_cast<device const uchar4*>(
            values + packed_base + column);
        half4 decoded = half4(
            scale * float4(packed) - float4(minimum));
        if (column + 4u <= valid) {
            *reinterpret_cast<threadgroup half4*>(target + column) = decoded;
        } else {
#pragma clang loop unroll(full)
            for (uint lane = 0u; lane < 4u; ++lane) {
                if (column + lane < valid) {
                    target[column + lane] = decoded[lane];
                }
            }
        }
    }
}

template <uint BITS, uint GROUP_SIZE>
inline void decode_nint_group_slice(
    const device int* d,
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    threadgroup half* target,
    uint row,
    uint group,
    uint first_element,
    uint element_count) {
    uint groups = uint(d[6]);
    uint q_offset = uint(d[7]);
    uint sub_offset = uint(d[8]);
    uint anchor_offset = uint(d[9]);
    uint q5_execution = uint(d[10]);
    uint metadata = row * groups + group;
    float scale = anchor_scales[anchor_offset + row]
        * float(sub_scales[sub_offset + metadata]);
    float minimum = anchor_mins[anchor_offset + row]
        * float(sub_mins[sub_offset + metadata]);
#pragma clang loop unroll(full)
    for (uint element = 0; element < GROUP_SIZE; element += 4u) {
        if (element >= element_count) {
            continue;
        }
        ushort4 quantized = read_nint_quad(
            values + q_offset,
            metadata * GROUP_SIZE + first_element + element,
            BITS,
            GROUP_SIZE,
            q5_execution);
        half4 decoded = half4(
            scale * float(quantized.x) - minimum,
            scale * float(quantized.y) - minimum,
            scale * float(quantized.z) - minimum,
            scale * float(quantized.w) - minimum);
        if (element + 4u <= element_count) {
            *reinterpret_cast<threadgroup half4*>(
                target + element) = decoded;
        } else {
#pragma clang loop unroll(full)
            for (uint lane = 0u; lane < 4u; ++lane) {
                if (element + lane < element_count) {
                    target[element + lane] = decoded[lane];
                }
            }
        }
    }
}

inline half decode_nint_value(
    const device int* d,
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    uint row,
    uint column) {
    uint bits = uint(d[4]);
    uint group_size = uint(d[5]);
    uint groups = uint(d[6]);
    uint q_offset = uint(d[7]);
    uint sub_offset = uint(d[8]);
    uint anchor_offset = uint(d[9]);
    uint q5_execution = uint(d[10]);
    uint group = column / group_size;
    uint metadata = row * groups + group;
    uint value_index = metadata * group_size
        + column - group * group_size;
    uint quantized = read_nint_value(
        values + q_offset,
        value_index,
        bits,
        group_size,
        q5_execution);
    float scale = anchor_scales[anchor_offset + row]
        * float(sub_scales[sub_offset + metadata]);
    float minimum = anchor_mins[anchor_offset + row]
        * float(sub_mins[sub_offset + metadata]);
    return half(scale * float(quantized) - minimum);
}

inline void decode_nint8_zero_group32(
    const device int* d,
    const device int8_t* values,
    const device half* scales,
    threadgroup half* target,
    uint row,
    uint group) {
    uint groups = uint(d[4]);
    uint q_offset = uint(d[5]);
    uint scale_offset = uint(d[6]);
    float scale = float(scales[scale_offset + row * groups + group]);
    uint value_base = q_offset + (row * groups + group) * 32u;
#pragma clang loop unroll(full)
    for (uint column = 0u; column < 32u; column += 4u) {
        char4 packed = *reinterpret_cast<device const char4*>(
            values + value_base + column);
        *reinterpret_cast<threadgroup half4*>(target + column) =
            half4(scale * float4(packed));
    }
}

template <uint JSC_VECTOR, uint BYTES_PER_SIGN>
inline void decode_jsc_group24(
    const device int* d,
    const device uchar* indices,
    const device uchar* state_stream,
    const device uchar* aux,
    const device float* anchors,
    const device int8_t* codebooks,
    const device float* scales,
    const device uchar* state_to_bank,
    threadgroup half* target,
    uint row,
    uint group,
    uint k_size) {
    constexpr bool DUAL_INDEX = JSC_VECTOR == 4u;
    uint groups = uint(d[5]);
    uint vectors = uint(d[7]);
    uint indices_offset = uint(d[18]);
    uint state_offset = uint(d[19]);
    uint aux_offset = uint(d[20]);
    uint anchor_offset = uint(d[21]);
    uint codebook_offset = uint(d[22]);
    uint scale_offset = uint(d[23]);
    uint state_bank_offset = uint(d[24]);
    uint execution = uint(d[29]);
    uint state_index = row * groups + group;
    uint signs = (k_size + 7u) / 8u;
    uint packed_state =
        uint(state_stream[state_offset + (state_index >> 1u)]);
    uint state =
        (packed_state >> ((state_index & 1u) * 4u)) & 15u;
    uint selected_bank =
        uint(state_to_bank[state_bank_offset + state]);
    float scale = anchors[anchor_offset + row]
        * scales[scale_offset + state];
    uint sign_base = row * signs + group * 3u;
    uint vector_base = row * vectors + group * (24u / JSC_VECTOR);

#pragma clang loop unroll(full)
    for (uint chunk = 0u; chunk < 3u; ++chunk) {
        uint sign_index = sign_base + chunk;
        uint first_vector = vector_base + chunk * (8u / JSC_VECTOR);
        uint index0 = 0u;
        uint index1 = 0u;
        uint sign_value = 0u;
        if (execution != 0u) {
            uint offset = indices_offset
                + sign_index * BYTES_PER_SIGN;
            if constexpr (DUAL_INDEX) {
                index0 = uint(indices[offset]);
                index1 = uint(indices[offset + 1u]);
                sign_value = uint(indices[offset + 2u]);
            } else {
                uchar2 packed = *reinterpret_cast<device const uchar2*>(
                    indices + offset);
                index0 = uint(packed.x);
                sign_value = uint(packed.y);
            }
        } else {
            index0 = uint(indices[indices_offset + first_vector]);
            if constexpr (DUAL_INDEX) {
                index1 = uint(indices[indices_offset + first_vector + 1u]);
            }
            sign_value = read_bits(
                aux + aux_offset,
                sign_index,
                7u);
        }
        uint sign_bits = sign_value;
        if (execution == 0u) {
            sign_bits |= (popcount(sign_value) & 1u) << 7u;
        }
        uint first_code_base = codebook_offset
            + (selected_bank * 256u + index0) * JSC_VECTOR;
        uint second_code_base = 0u;
        if constexpr (DUAL_INDEX) {
            second_code_base = codebook_offset
                + (selected_bank * 256u + index1) * JSC_VECTOR;
        } else {
            second_code_base = first_code_base + 4u;
        }
        char4 first_code = *reinterpret_cast<device const char4*>(
            codebooks + first_code_base);
        char4 second_code = *reinterpret_cast<device const char4*>(
            codebooks + second_code_base);
        float4 first_value = float4(first_code);
        float4 second_value = float4(second_code);
        first_value = select(
            first_value,
            -first_value,
            (uint4(sign_bits) & uint4(1u, 2u, 4u, 8u))
                != uint4(0u));
        second_value = select(
            second_value,
            -second_value,
            (uint4(sign_bits) & uint4(16u, 32u, 64u, 128u))
                != uint4(0u));
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u) = half4(scale * first_value);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u + 4u) = half4(scale * second_value);
    }
}

#ifdef MFQ_ENABLE_JSC_EXTENDED_VECTOR
inline void decode_jsc_extended_group24(
    const device int* d,
    const device uchar* indices,
    const device uchar* state_stream,
    const device uchar* aux,
    const device float* anchors,
    const device int8_t* codebooks,
    const device float* scales,
    const device uchar* state_to_bank,
    threadgroup half* target,
    uint row,
    uint group,
    uint k_size) {
    uint groups = uint(d[5]);
    uint vectors = uint(d[7]);
    uint index_bits = uint(d[8]);
    uint entries = uint(d[11]);
    uint indices_offset = uint(d[18]);
    uint state_offset = uint(d[19]);
    uint aux_offset = uint(d[20]);
    uint anchor_offset = uint(d[21]);
    uint codebook_offset = uint(d[22]);
    uint scale_offset = uint(d[23]);
    uint state_bank_offset = uint(d[24]);
    uint execution = uint(d[29]);
    uint state_index = row * groups + group;
    uint signs = (k_size + 7u) / 8u;
    uint state;
    uint3 group_indices;
    uint3 sign_values;
    if (execution == 2u) {
        ulong record = *reinterpret_cast<device const ulong*>(
            indices + indices_offset + state_index * 8u);
        group_indices = uint3(
            uint(record & 4095ul),
            uint((record >> 20u) & 4095ul),
            uint((record >> 40u) & 4095ul));
        sign_values = uint3(
            uint((record >> 12u) & 255ul),
            uint((record >> 32u) & 255ul),
            uint((record >> 52u) & 255ul));
        state = uint(record >> 60u);
    } else {
        group_indices = index_bits == 10u
            ? read_vq_group_indices<10u>(
                indices + indices_offset, row, group, vectors)
            : read_vq_group_indices<12u>(
                indices + indices_offset, row, group, vectors);
        uint sign_base = row * signs + group * 3u;
        sign_values = uint3(
            read_bits(aux + aux_offset, sign_base, 7u),
            read_bits(aux + aux_offset, sign_base + 1u, 7u),
            read_bits(aux + aux_offset, sign_base + 2u, 7u));
        state = read_bits(
            state_stream + state_offset,
            state_index,
            4u);
    }
    uint selected_bank = uint(
        state_to_bank[state_bank_offset + state]);
    float scale = anchors[anchor_offset + row]
        * scales[scale_offset + state];

#pragma clang loop unroll(full)
    for (uint chunk = 0u; chunk < 3u; ++chunk) {
        uint column_base = group * 24u + chunk * 8u;
        if (column_base >= k_size) {
            *reinterpret_cast<threadgroup half4*>(
                target + chunk * 8u) = half4(0.0h);
            *reinterpret_cast<threadgroup half4*>(
                target + chunk * 8u + 4u) = half4(0.0h);
            continue;
        }
        uint sign_bits = sign_values[chunk];
        if (execution == 0u) {
            sign_bits |= (popcount(sign_bits) & 1u) << 7u;
        }
        uint code_base = codebook_offset
            + (selected_bank * entries + group_indices[chunk]) * 8u;
        float4 first_code = float4(
            *reinterpret_cast<device const char4*>(
                codebooks + code_base));
        float4 second_code = float4(
            *reinterpret_cast<device const char4*>(
                codebooks + code_base + 4u));
        first_code = select(
            first_code,
            -first_code,
            (uint4(sign_bits) & uint4(1u, 2u, 4u, 8u))
                != uint4(0u));
        second_code = select(
            second_code,
            -second_code,
            (uint4(sign_bits) & uint4(16u, 32u, 64u, 128u))
                != uint4(0u));
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u) = half4(scale * first_code);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u + 4u) = half4(scale * second_code);
    }
}
#endif

#ifdef MFQ_ENABLE_LEGACY_VQ_VECTOR
template <uint STATE_WIDTH, uint INDEX_WIDTH, uint TABLE_SIZE>
inline void decode_npq_group24(
    const device int* d,
    const device uchar* indices,
    const device uchar* state_stream,
    const device float* anchors,
    const device int8_t* codebooks,
    const device float* scales,
    threadgroup half* target,
    uint row,
    uint group) {
    uint groups = uint(d[5]);
    uint vectors = uint(d[7]);
    uint indices_offset = uint(d[18]);
    uint state_offset = uint(d[19]);
    uint anchor_offset = uint(d[21]);
    uint codebook_offset = uint(d[22]);
    uint scale_offset = uint(d[23]);
    uint state = read_bits(
        state_stream + state_offset,
        row * groups + group,
        STATE_WIDTH);
    float scale = anchors[anchor_offset + row]
        * scales[scale_offset + state];
    uint3 group_indices = read_vq_group_indices<INDEX_WIDTH>(
        indices + indices_offset, row, group, vectors);

#pragma clang loop unroll(full)
    for (uint chunk = 0u; chunk < 3u; ++chunk) {
        uint index = group_indices[chunk];
        uint code_base = codebook_offset
            + (state * TABLE_SIZE + index) * 8u;
        char4 first_code = *reinterpret_cast<device const char4*>(
            codebooks + code_base);
        char4 second_code = *reinterpret_cast<device const char4*>(
            codebooks + code_base + 4u);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u) = half4(
                scale * float4(first_code));
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u + 4u) = half4(
                scale * float4(second_code));
    }
}

template <
    uint STATE_WIDTH,
    uint INDEX_WIDTH,
    uint ENTRIES_PER_BANK,
    bool DUAL_BANK>
inline void decode_nvq1_group24(
    const device int* d,
    const device uchar* indices,
    const device uchar* state_stream,
    const device uchar* aux,
    const device float* anchors,
    const device int8_t* codebooks,
    const device float* scales,
    const device float* parameters,
    threadgroup half* target,
    uint row,
    uint group) {
    uint groups = uint(d[5]);
    uint vectors = uint(d[7]);
    uint indices_offset = uint(d[18]);
    uint state_offset = uint(d[19]);
    uint aux_offset = uint(d[20]);
    uint anchor_offset = uint(d[21]);
    uint codebook_offset = uint(d[22]);
    uint scale_offset = uint(d[23]);
    uint parameter_offset = uint(d[26]);
    uint state_index = row * groups + group;
    uint state = read_bits(
        state_stream + state_offset, state_index, STATE_WIDTH);
    uint bank = read_bits(aux + aux_offset, state_index, 1u);
    float delta = parameters[parameter_offset];
    float signed_delta = bank != 0u ? -delta : delta;
    float scale = anchors[anchor_offset + row]
        * scales[scale_offset + state];
    uint3 group_indices = read_vq_group_indices<INDEX_WIDTH>(
        indices + indices_offset, row, group, vectors);

#pragma clang loop unroll(full)
    for (uint chunk = 0u; chunk < 3u; ++chunk) {
        uint entry = group_indices[chunk];
        if constexpr (DUAL_BANK) {
            entry += bank * ENTRIES_PER_BANK;
        }
        uint code_base = codebook_offset + entry * 8u;
        float4 first_code = float4(
            *reinterpret_cast<device const char4*>(
                codebooks + code_base)) + signed_delta;
        float4 second_code = float4(
            *reinterpret_cast<device const char4*>(
                codebooks + code_base + 4u)) + signed_delta;
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u) = half4(scale * first_code);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u + 4u) = half4(scale * second_code);
    }
}
#endif

inline void decode_vq_group24(
    const device int* d,
    const device uchar* indices,
    const device uchar* state_stream,
    const device uchar* aux,
    const device float* anchors,
    const device int8_t* codebooks,
    const device float* scales,
    const device uchar* state_to_bank,
    const device uchar* banks,
    const device float* parameters,
    threadgroup half* target,
    uint row,
    uint group,
    uint k_size) {
    uint groups = uint(d[5]);
    uint vector_size = uint(d[6]);
    uint vectors = uint(d[7]);
    uint index_bits = uint(d[8]);
    uint state_bits = uint(d[9]);
    uint state_count = uint(d[10]);
    uint entries = uint(d[11]);
    uint code_banks = uint(d[12]);
    uint aux_mode = uint(d[13]);
    uint code_bank_mode = uint(d[14]);
    uint has_table_banks = uint(d[15]);
    uint groups_per_super = uint(d[16]);
    uint supergroups = uint(d[17]);
    uint indices_offset = uint(d[18]);
    uint state_offset = uint(d[19]);
    uint aux_offset = uint(d[20]);
    uint anchor_offset = uint(d[21]);
    uint codebook_offset = uint(d[22]);
    uint scale_offset = uint(d[23]);
    uint state_bank_offset = uint(d[24]);
    uint bank_offset = uint(d[25]);
    uint parameter_offset = uint(d[26]);
    uint execution = uint(d[29]);
    uint profile = uint(d[28]);
    uint state_index = row * groups + group;
    uint signs = (k_size + 7u) / 8u;
    float anchor = anchors[anchor_offset + row];

    if (profile == 1u) {
        decode_jsc_group24<4u, 3u>(
            d, indices, state_stream, aux, anchors, codebooks, scales,
            state_to_bank, target, row, group, k_size);
        return;
    }
    if (profile == 4u) {
        decode_jsc_group24<8u, 2u>(
            d, indices, state_stream, aux, anchors, codebooks, scales,
            state_to_bank, target, row, group, k_size);
        return;
    }
#ifdef MFQ_ENABLE_JSC_EXTENDED_VECTOR
    if (profile == 7u) {
        decode_jsc_extended_group24(
            d, indices, state_stream, aux, anchors, codebooks, scales,
            state_to_bank, target, row, group, k_size);
        return;
    }
#endif

#ifdef MFQ_ENABLE_LEGACY_VQ_VECTOR
    if (profile == 2u) {
        decode_npq_group24<2u, 6u, 64u>(
            d, indices, state_stream, anchors, codebooks, scales,
            target, row, group);
        return;
    }
    if (profile == 5u) {
        decode_npq_group24<3u, 7u, 128u>(
            d, indices, state_stream, anchors, codebooks, scales,
            target, row, group);
        return;
    }

    if (profile == 3u) {
        decode_nvq1_group24<3u, 11u, 2048u, false>(
            d, indices, state_stream, aux, anchors, codebooks, scales,
            parameters, target, row, group);
        return;
    }
    if (profile == 6u) {
        decode_nvq1_group24<4u, 9u, 512u, true>(
            d, indices, state_stream, aux, anchors, codebooks, scales,
            parameters, target, row, group);
        return;
    }
#else
    if (profile == 2u || profile == 5u) {
        uint state_width = profile == 2u ? 2u : 3u;
        uint index_width = profile == 2u ? 6u : 7u;
        uint state = read_bits(
            state_stream + state_offset,
            state_index,
            state_width);
        uint table_size = profile == 2u ? 64u : 128u;
        float scale = anchor * scales[scale_offset + state];
        for (uint chunk = 0u; chunk < 3u; ++chunk) {
            uint index = read_bits(
                indices + indices_offset,
                row * vectors + group * 3u + chunk,
                index_width);
            uint code_base = codebook_offset
                + (state * table_size + index) * 8u;
            for (uint inner = 0u; inner < 8u; ++inner) {
                target[chunk * 8u + inner] = half(
                    scale * float(codebooks[code_base + inner]));
            }
        }
        return;
    }

    if (profile == 3u) {
        uint state = read_bits(
            state_stream + state_offset,
            state_index,
            state_bits);
        uint sign = read_bits(
            aux + aux_offset,
            state_index,
            1u);
        float delta = parameters[parameter_offset];
        float scale = anchor * scales[scale_offset + state];
        for (uint chunk = 0u; chunk < 3u; ++chunk) {
            uint index = read_bits(
                indices + indices_offset,
                row * vectors + group * 3u + chunk,
                11u);
            uint code_base = codebook_offset + index * 8u;
            for (uint inner = 0u; inner < 8u; ++inner) {
                float code = float(codebooks[code_base + inner]);
                code += sign != 0u ? -delta : delta;
                target[chunk * 8u + inner] = half(scale * code);
            }
        }
        return;
    }
#endif

    uint state = read_bits(
        state_stream + state_offset,
        state_index,
        state_bits);
    uint table_bank = has_table_banks != 0u
        ? uint(banks[
              bank_offset + row * supergroups
              + group / groups_per_super])
        : 0u;
    uint delta_value = aux_mode == 3u
        ? read_bits(aux + aux_offset, state_index, 1u)
        : 0u;
    uint selected_bank = code_bank_mode == 1u
        ? uint(state_to_bank[state_bank_offset + state])
        : (code_bank_mode == 2u ? delta_value : 0u);
    float scale = anchor * scales[
        scale_offset + table_bank * state_count + state];
#ifdef MFQ_ENABLE_LEGACY_VQ_VECTOR
    if (profile == 0u && vector_size == 8u) {
#pragma clang loop unroll(full)
        for (uint chunk = 0u; chunk < 3u; ++chunk) {
            uint column_base = group * 24u + chunk * 8u;
            if (column_base >= k_size) {
                *reinterpret_cast<threadgroup half4*>(
                    target + chunk * 8u) = half4(0.0h);
                *reinterpret_cast<threadgroup half4*>(
                    target + chunk * 8u + 4u) = half4(0.0h);
                continue;
            }
            uint sign_value = 0u;
            if (aux_mode == 1u || aux_mode == 2u) {
                sign_value = read_bits(
                    aux + aux_offset,
                    row * signs + group * 3u + chunk,
                    7u);
            }
            uint vector = group * 3u + chunk;
            uint index = read_bits(
                indices + indices_offset,
                row * vectors + vector,
                index_bits);
            uint code_base = codebook_offset
                + (((table_bank * code_banks + selected_bank) * entries
                  + index) * 8u);
            float4 first_code = float4(
                *reinterpret_cast<device const char4*>(
                    codebooks + code_base));
            float4 second_code = float4(
                *reinterpret_cast<device const char4*>(
                    codebooks + code_base + 4u));
            if (aux_mode == 1u || aux_mode == 2u) {
                first_code = select(
                    first_code,
                    -first_code,
                    (uint4(sign_value) & uint4(1u, 2u, 4u, 8u))
                        != uint4(0u));
                uint parity = popcount(sign_value) & 1u;
                if (aux_mode == 2u) {
                    parity ^= (index >> 7u) & 1u;
                }
                second_code = select(
                    second_code,
                    -second_code,
                    bool4(
                        (sign_value & 16u) != 0u,
                        (sign_value & 32u) != 0u,
                        (sign_value & 64u) != 0u,
                        parity != 0u));
            } else if (aux_mode == 3u) {
                float delta = parameters[parameter_offset];
                float signed_delta = delta_value != 0u
                    ? -delta : delta;
                first_code += signed_delta;
                second_code += signed_delta;
            }
            *reinterpret_cast<threadgroup half4*>(
                target + chunk * 8u) = half4(scale * first_code);
            *reinterpret_cast<threadgroup half4*>(
                target + chunk * 8u + 4u) = half4(
                    scale * second_code);
        }
        return;
    }
    if (profile == 0u && vector_size == 4u) {
#pragma clang loop unroll(full)
        for (uint chunk = 0u; chunk < 3u; ++chunk) {
            uint column_base = group * 24u + chunk * 8u;
            if (column_base >= k_size) {
                *reinterpret_cast<threadgroup half4*>(
                    target + chunk * 8u) = half4(0.0h);
                *reinterpret_cast<threadgroup half4*>(
                    target + chunk * 8u + 4u) = half4(0.0h);
                continue;
            }
            uint sign_value = 0u;
            if (aux_mode == 1u || aux_mode == 2u) {
                sign_value = read_bits(
                    aux + aux_offset,
                    row * signs + group * 3u + chunk,
                    7u);
            }
#pragma clang loop unroll(full)
            for (uint local = 0u; local < 2u; ++local) {
                uint vector = group * 6u + chunk * 2u + local;
                if (vector >= vectors) {
                    *reinterpret_cast<threadgroup half4*>(
                        target + chunk * 8u + local * 4u) = half4(0.0h);
                    continue;
                }
                uint index = read_bits(
                    indices + indices_offset,
                    row * vectors + vector,
                    index_bits);
                uint code_base = codebook_offset
                    + (((table_bank * code_banks + selected_bank) * entries
                      + index) * 4u);
                float4 code = float4(
                    *reinterpret_cast<device const char4*>(
                        codebooks + code_base));
                if (aux_mode == 1u || aux_mode == 2u) {
                    if (local == 0u) {
                        code = select(
                            code,
                            -code,
                            (uint4(sign_value)
                                & uint4(1u, 2u, 4u, 8u))
                                != uint4(0u));
                    } else {
                        uint parity = popcount(sign_value) & 1u;
                        if (aux_mode == 2u) {
                            parity ^= (index >> 7u) & 1u;
                        }
                        code = select(
                            code,
                            -code,
                            bool4(
                                (sign_value & 16u) != 0u,
                                (sign_value & 32u) != 0u,
                                (sign_value & 64u) != 0u,
                                parity != 0u));
                    }
                } else if (aux_mode == 3u) {
                    float delta = parameters[parameter_offset];
                    code += delta_value != 0u ? -delta : delta;
                }
                *reinterpret_cast<threadgroup half4*>(
                    target + chunk * 8u + local * 4u) = half4(
                        scale * code);
            }
        }
        return;
    }
#endif
    uint vectors_per_chunk = 8u / vector_size;
    for (uint chunk = 0u; chunk < 3u; ++chunk) {
        uint sign_value = 0u;
        if (aux_mode == 1u || aux_mode == 2u) {
            sign_value = read_bits(
                aux + aux_offset,
                row * signs + group * 3u + chunk,
                7u);
        }
        for (uint local = 0u; local < vectors_per_chunk; ++local) {
            uint vector =
                group * (24u / vector_size)
                + chunk * vectors_per_chunk + local;
            uint index = read_bits(
                indices + indices_offset,
                row * vectors + vector,
                index_bits);
            uint code_base = codebook_offset +
                (((table_bank * code_banks + selected_bank) * entries
                  + index) * vector_size);
            for (uint component = 0u;
                 component < vector_size;
                 ++component) {
                uint inner = local * vector_size + component;
                float code = float(codebooks[code_base + component]);
                uint negative = inner < 7u
                    ? ((sign_value >> inner) & 1u)
                    : (popcount(sign_value) & 1u);
                if (aux_mode == 2u && inner == 7u) {
                    negative ^= (index >> 7u) & 1u;
                }
                if (aux_mode == 1u || aux_mode == 2u) {
                    if (negative != 0u) {
                        code = -code;
                    }
                } else if (aux_mode == 3u) {
                    float delta = parameters[parameter_offset];
                    code += delta_value != 0u ? -delta : delta;
                }
                target[chunk * 8u + inner] = half(scale * code);
            }
        }
    }
}

inline void add_nepq_residual_group24(
    const device int* d,
    const device float* codebooks,
    const device short* first_records,
    const device short* second_records,
    threadgroup half* target,
    uint row,
    uint group,
    uint vectors) {
    uint residual_profile = uint(d[28]);
    uint position_bits = (residual_profile >> 8u) & 255u;
    uint block_vectors = (residual_profile >> 16u) & 255u;
    if (position_bits == 0u || block_vectors == 0u) {
        return;
    }
    uint residual_blocks =
        (vectors + block_vectors - 1u) / block_vectors;
    uint codebook_offset = uint(d[30]);
    uint record_offset = uint(d[31]);
    uint position_mask = (1u << position_bits) - 1u;
    uint first_vector = group * 3u;
    uint last_vector = min(first_vector + 2u, vectors - 1u);
    uint first_block = first_vector / block_vectors;
    uint last_block = last_vector / block_vectors;
    for (uint block = first_block; block <= last_block; ++block) {
        uint record_index =
            record_offset + row * residual_blocks + block;
        short records[2] = {
            first_records[record_index],
            second_records[record_index],
        };
        for (uint stream = 0u; stream < 2u; ++stream) {
            int record = int(records[stream]);
            if (record < 0) {
                continue;
            }
            uint position = uint(record) & position_mask;
            uint dictionary_id = uint(record) >> position_bits;
            uint vector = block * block_vectors + position;
            if (
                dictionary_id >= 1024u
                || vector < first_vector
                || vector > last_vector
                || vector >= vectors
            ) {
                continue;
            }
            uint target_offset = (vector - first_vector) * 8u;
            uint dictionary_offset =
                codebook_offset + dictionary_id * 8u;
            for (uint component = 0u; component < 8u; ++component) {
                target[target_offset + component] += half(
                    codebooks[dictionary_offset + component]);
            }
        }
    }
}

} // namespace

struct MfqGroupedMmqParams {
    int route_count;
    int tokens;
    int routes;
    int experts;
    int output_width;
    int matrix_output_width;
    int input_width;
    int descriptor_size;
    int variant_stride;
    int shared_input;
    int input_sorted;
    float swiglu_limit;
};

template <bool FUSED_SWIGLU, bool HAS_NEPQ_RESIDUAL>
[[kernel]] void mfq_grouped_mmq_f16_bm32_bn64_bk96(
    const device int* descriptors [[buffer(0)]],
    const device uchar* vq_indices [[buffer(1)]],
    const device uchar* vq_state [[buffer(2)]],
    const device uchar* vq_aux [[buffer(3)]],
    const device float* vq_anchors [[buffer(4)]],
    const device int8_t* vq_codebooks [[buffer(5)]],
    const device float* vq_scales [[buffer(6)]],
    const device uchar* vq_state_to_bank [[buffer(7)]],
    const device uchar* vq_banks [[buffer(8)]],
    const device float* vq_parameters [[buffer(9)]],
    const device half* x [[buffer(10)]],
    const device int* expert_ids [[buffer(11)]],
    const device int* route_order [[buffer(12)]],
    device half* y [[buffer(13)]],
    constant MfqGroupedMmqParams& params [[buffer(14)]],
    const device float* vq_residual_codebooks [[buffer(15)]],
    const device short* vq_residual_first [[buffer(16)]],
    const device short* vq_residual_second [[buffer(17)]],
    const device int* block_meta [[buffer(18)]],
    const device int* block_count [[buffer(19)]],
    const device uchar* mx_values [[buffer(20)]],
    const device uchar* mx_scales [[buffer(21)]],
    const device uchar* nint_q [[buffer(22)]],
    const device uchar* nint_sub_scale [[buffer(23)]],
    const device uchar* nint_sub_min [[buffer(24)]],
    const device float* nint_anchor_scale [[buffer(25)]],
    const device float* nint_anchor_min [[buffer(26)]],
    const device int8_t* q8_q [[buffer(27)]],
    const device half* q8_scales [[buffer(28)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    constexpr int BM = 32;
    constexpr int BN = 64;
    constexpr int BK = 96;
    constexpr int BK_padded = 104;
    constexpr uint TGP_SIZE = 256u;
    const int route_count = params.route_count;
    const int tokens = params.tokens;
    const int routes = params.routes;
    const int experts = params.experts;
    const int output_width = params.output_width;
    const int matrix_output_width = params.matrix_output_width;
    const int input_width = params.input_width;
    const int descriptor_size = params.descriptor_size;
    const int variant_stride = params.variant_stride;
    const int shared_input = params.shared_input;
    const int input_sorted = params.input_sorted;
    const float swiglu_limit = params.swiglu_limit;
    using mma_t = mlx::steel::BlockMMA<
        half,
        half,
        BM,
        BN,
        BK,
        2,
        4,
        false,
        true,
        BK_padded,
        BK_padded>;

    int output_base = int(tid.x) * BN;
    int block_id = int(tid.y);
    int nblocks = block_count[0];
    if (output_base >= output_width || block_id >= nblocks) {
        return;
    }

    int row_base = block_meta[block_id * 3 + 0];
    int expert = block_meta[block_id * 3 + 1];
    int row_count = block_meta[block_id * 3 + 2];
    if (row_count <= 0 || expert < 0 || expert >= experts) {
        return;
    }

    const device int* descriptor =
        descriptors + expert * descriptor_size;
    uint family = uint(descriptor[0]);
    uint local_expert = uint(descriptor[1]);
    uint rotation = uint(descriptor[27]);
    short valid_n = short(min(BN, output_width - output_base));
    threadgroup half Xs[BM * BK_padded];
    threadgroup half Ws[BN * BK_padded];

    thread mma_t gate_mma(simd_group_id, simd_lane_id);
    thread mma_t up_mma(simd_group_id, simd_lane_id);
        for (int k_base = 0; k_base < input_width; k_base += BK) {
            for (uint item = thread_id;
                 item < uint(BM * BK_padded);
                 item += TGP_SIZE) {
                uint row = item / uint(BK_padded);
                uint column = item - row * uint(BK_padded);
                half value = half(0.0f);
                int input_column = k_base + int(column);
                if (int(row) < row_count && column < uint(BK)
                    && input_column < input_width) {
                    uint route_index =
                        uint(route_order[row_base + int(row)]);
                    uint source_row = input_sorted != 0
                        ? uint(row_base) + row
                        : (shared_input != 0
                            ? route_index / uint(routes)
                            : route_index);
                    uint source_offset =
                        (rotation * uint(variant_stride) + source_row)
                        * uint(input_width);
                    value = x[source_offset + uint(input_column)];
                }
                Xs[item] = value;
            }
            constexpr int PROJECTIONS = FUSED_SWIGLU ? 2 : 1;
            for (int projection = 0;
                 projection < PROJECTIONS;
                 ++projection) {
                for (uint item = thread_id;
                     item < uint(BN * BK_padded);
                     item += TGP_SIZE) {
                    Ws[item] = half(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                uint nint_bits = uint(descriptor[4]);
                uint nint_group_size = uint(descriptor[5]);
                bool aligned_nint = family == 0u && (
                    (nint_bits == 2u && nint_group_size == 16u)
                    || ((nint_bits == 3u || nint_bits == 4u
                         || nint_bits == 6u)
                        && nint_group_size == 24u)
                    || (nint_bits == 8u && nint_group_size == 48u));
                bool sliced_nint5 = family == 0u
                    && nint_bits == 5u && nint_group_size == 28u;
                bool scalar_decode = family == 0u && !aligned_nint;
                if (aligned_nint) {
                    uint groups_per_tile = uint(BK) / nint_group_size;
                    uint decode_units_per_tile = nint_bits == 2u
                        ? (groups_per_tile + 1u) / 2u
                        : groups_per_tile;
                    for (uint item = thread_id;
                         item < uint(BN) * decode_units_per_tile;
                         item += TGP_SIZE) {
                        uint output_row = item / decode_units_per_tile;
                        uint decode_unit =
                            item - output_row * decode_units_per_tile;
                        uint local_group = nint_bits == 2u
                            ? decode_unit * 2u
                            : decode_unit;
                        uint input_column =
                            uint(k_base) + local_group * nint_group_size;
                        if (output_row >= uint(valid_n)
                            || input_column >= uint(input_width)) {
                            continue;
                        }
                        uint group = input_column / nint_group_size;
                        uint pool_row =
                            local_expert * uint(matrix_output_width)
                            + uint(output_base) + output_row
                            + uint(projection * output_width);
                        threadgroup half* target =
                            Ws + output_row * uint(BK_padded)
                            + local_group * nint_group_size;
                        if (nint_bits == 2u) {
                            if (local_group + 1u < groups_per_tile
                                && input_column + nint_group_size
                                    < uint(input_width)) {
                                decode_nint2_pair32(
                                    descriptor, nint_q, nint_sub_scale,
                                    nint_sub_min, nint_anchor_scale,
                                    nint_anchor_min, target, pool_row, group);
                            } else {
                                decode_nint2_group16(
                                    descriptor, nint_q, nint_sub_scale,
                                    nint_sub_min, nint_anchor_scale,
                                    nint_anchor_min, target, pool_row, group);
                            }
                        } else if (nint_bits == 3u) {
                            decode_nint3_group24(
                                descriptor, nint_q, nint_sub_scale,
                                nint_sub_min, nint_anchor_scale,
                                nint_anchor_min, target, pool_row, group);
                        } else if (nint_bits == 4u) {
                            decode_nint4_group24(
                                descriptor, nint_q, nint_sub_scale,
                                nint_sub_min, nint_anchor_scale,
                                nint_anchor_min, target, pool_row, group);
                        } else if (nint_bits == 6u) {
                            decode_nint6_group24(
                                descriptor, nint_q, nint_sub_scale,
                                nint_sub_min, nint_anchor_scale,
                                nint_anchor_min, target, pool_row, group);
                        } else {
                            decode_nint8_group48(
                                descriptor, nint_q, nint_sub_scale,
                                nint_sub_min, nint_anchor_scale,
                                nint_anchor_min, target, pool_row, group,
                                uint(input_width));
                        }
                    }
                } else if (sliced_nint5) {
                    constexpr uint GROUP_SIZE = 28u;
                    constexpr uint MAX_GROUPS_PER_TILE = 5u;
                    uint first_group = uint(k_base) / GROUP_SIZE;
                    uint tile_end = min(
                        uint(k_base + BK),
                        uint(input_width));
                    uint last_group = (tile_end - 1u) / GROUP_SIZE;
                    uint tile_groups = last_group - first_group + 1u;
                    for (uint item = thread_id;
                         item < uint(BN) * MAX_GROUPS_PER_TILE;
                         item += TGP_SIZE) {
                        uint output_row = item / MAX_GROUPS_PER_TILE;
                        uint group_slot =
                            item - output_row * MAX_GROUPS_PER_TILE;
                        if (output_row >= uint(valid_n)
                            || group_slot >= tile_groups) {
                            continue;
                        }
                        uint group = first_group + group_slot;
                        uint group_start = group * GROUP_SIZE;
                        uint slice_start = max(uint(k_base), group_start);
                        uint slice_end = min(tile_end, group_start + GROUP_SIZE);
                        uint pool_row =
                            local_expert * uint(matrix_output_width)
                            + uint(output_base) + output_row
                            + uint(projection * output_width);
                        decode_nint_group_slice<5u, GROUP_SIZE>(
                            descriptor, nint_q, nint_sub_scale,
                            nint_sub_min, nint_anchor_scale,
                            nint_anchor_min,
                            Ws + output_row * uint(BK_padded)
                                + slice_start - uint(k_base),
                            pool_row, group, slice_start - group_start,
                            slice_end - slice_start);
                    }
                } else if (family == 4u) {
                    constexpr uint VALUES_PER_ITEM = 4u;
                    constexpr uint ITEMS_PER_ROW = uint(BK) / VALUES_PER_ITEM;
                    for (uint item = thread_id;
                         item < uint(BN) * ITEMS_PER_ROW;
                         item += TGP_SIZE) {
                        uint output_row = item / ITEMS_PER_ROW;
                        uint local_item = item - output_row * ITEMS_PER_ROW;
                        uint local_column = local_item * VALUES_PER_ITEM;
                        uint input_column = uint(k_base) + local_column;
                        if (output_row < uint(valid_n)
                            && input_column < uint(input_width)) {
                            uint pool_row =
                                local_expert * uint(matrix_output_width)
                                + uint(output_base) + output_row
                                + uint(projection * output_width);
                            *reinterpret_cast<threadgroup half4*>(
                                Ws + output_row * uint(BK_padded)
                                    + local_column) = decode_mxfp8_quad_at(
                                descriptor, mx_values, mx_scales,
                                pool_row, input_column, uint(input_width));
                        }
                    }
                } else if (family == 5u || family == 6u) {
                    constexpr uint VALUES_PER_ITEM = 4u;
                    constexpr uint ITEMS_PER_ROW = uint(BK) / VALUES_PER_ITEM;
                    for (uint item = thread_id;
                         item < uint(BN) * ITEMS_PER_ROW;
                         item += TGP_SIZE) {
                        uint output_row = item / ITEMS_PER_ROW;
                        uint local_item = item - output_row * ITEMS_PER_ROW;
                        uint local_column = local_item * VALUES_PER_ITEM;
                        uint input_column = uint(k_base) + local_column;
                        if (output_row >= uint(valid_n)
                            || input_column >= uint(input_width)) {
                            continue;
                        }
                        uint pool_row =
                            local_expert * uint(matrix_output_width)
                            + uint(output_base) + output_row
                            + uint(projection * output_width);
                        threadgroup half* target =
                            Ws + output_row * uint(BK_padded) + local_column;
                        if (input_column + VALUES_PER_ITEM
                            <= uint(input_width)) {
                            *reinterpret_cast<threadgroup half4*>(target) =
                                decode_dense_quad_at(
                                    descriptor, q8_q, family, pool_row,
                                    input_column, uint(input_width));
                        } else {
#pragma clang loop unroll(full)
                            for (uint lane = 0u;
                                 lane < VALUES_PER_ITEM;
                                 ++lane) {
                                if (input_column + lane < uint(input_width)) {
                                    target[lane] = decode_dense_value_at(
                                        descriptor, q8_q, family, pool_row,
                                        input_column + lane,
                                        uint(input_width));
                                }
                            }
                        }
                    }
                } else if (scalar_decode) {
                    for (uint item = thread_id;
                         item < uint(BN * BK);
                         item += TGP_SIZE) {
                        uint output_row = item / uint(BK);
                        uint local_column = item - output_row * uint(BK);
                        uint input_column = uint(k_base) + local_column;
                        if (output_row < uint(valid_n)
                            && input_column < uint(input_width)) {
                            uint pool_row =
                                local_expert * uint(matrix_output_width)
                                + uint(output_base) + output_row
                                + uint(projection * output_width);
                            half value;
                            if (family == 0u) {
                                value = decode_nint_value(
                                    descriptor,
                                    nint_q,
                                    nint_sub_scale,
                                    nint_sub_min,
                                    nint_anchor_scale,
                                    nint_anchor_min,
                                    pool_row,
                                    input_column);
                            } else if (family == 4u) {
                                value = decode_mxfp8_value_at(
                                    descriptor,
                                    mx_values,
                                    mx_scales,
                                    pool_row,
                                    input_column,
                                    uint(input_width));
                            } else {
                                value = decode_dense_value_at(
                                    descriptor,
                                    q8_q,
                                    family,
                                    pool_row,
                                    input_column,
                                    uint(input_width));
                            }
                            Ws[output_row * uint(BK_padded) + local_column] =
                                value;
                        }
                    }
                } else {
                constexpr uint GROUPS_PER_TILE = BK / 24;
                for (uint group_item = thread_id;
                     group_item < uint(BN) * GROUPS_PER_TILE;
                     group_item += TGP_SIZE) {
                    uint output_row = group_item / GROUPS_PER_TILE;
                    uint local_group =
                        group_item - output_row * GROUPS_PER_TILE;
                    if (output_row < uint(valid_n) && family == 2u
                        && local_group < uint(BK / 32)) {
                        uint input_column =
                            uint(k_base) + local_group * 32u;
                        if (input_column >= uint(input_width)) {
                            continue;
                        }
                        uint group = input_column / 32u;
                        uint pool_row =
                            local_expert * uint(matrix_output_width)
                            + uint(output_base) + output_row
                            + uint(projection * output_width);
                        decode_nint8_zero_group32(
                            descriptor,
                            q8_q,
                            q8_scales,
                            Ws + output_row * uint(BK_padded)
                                + local_group * 32u,
                            pool_row,
                            group);
                    } else if (output_row < uint(valid_n) && family == 3u
                        && local_group < uint(BK / 32)) {
                        uint input_column =
                            uint(k_base) + local_group * 32u;
                        if (input_column >= uint(input_width)) {
                            continue;
                        }
                        uint group = input_column / 32u;
                        uint pool_row =
                            local_expert * uint(matrix_output_width)
                            + uint(output_base) + output_row
                            + uint(projection * output_width);
                        decode_mxfp4_group32(
                            descriptor,
                            mx_values,
                            mx_scales,
                            Ws + output_row * uint(BK_padded)
                                + local_group * 32u,
                            pool_row,
                            group,
                            uint(input_width));
                    } else if (output_row < uint(valid_n)
                        && family == 1u) {
                        uint input_column =
                            uint(k_base) + local_group * 24u;
                        if (input_column >= uint(input_width)) {
                            continue;
                        }
                        uint group = input_column / 24u;
                        uint pool_row =
                            local_expert * uint(matrix_output_width)
                            + uint(output_base) + output_row
                            + uint(projection * output_width);
                        decode_vq_group24(
                            descriptor,
                            vq_indices,
                            vq_state,
                            vq_aux,
                            vq_anchors,
                            vq_codebooks,
                            vq_scales,
                            vq_state_to_bank,
                            vq_banks,
                            vq_parameters,
                            Ws + output_row * uint(BK_padded)
                                + local_group * 24u,
                            pool_row,
                            group,
                            uint(input_width));
                        if constexpr (HAS_NEPQ_RESIDUAL) {
                            add_nepq_residual_group24(
                                descriptor,
                                vq_residual_codebooks,
                                vq_residual_first,
                                vq_residual_second,
                                Ws + output_row * uint(BK_padded)
                                    + local_group * 24u,
                                pool_row,
                                group,
                                uint(descriptor[7]));
                        }
                    }
                }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (projection == 0) {
                    gate_mma.mma(Xs, Ws);
                } else {
                    up_mma.mma(Xs, Ws);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if constexpr (FUSED_SWIGLU) {
            for (short item = 0;
                 item < decltype(gate_mma.Ctile)::kElemsPerTile;
                 ++item) {
                float gate = gate_mma.Ctile.elems()[item];
                float up = up_mma.Ctile.elems()[item];
                if (swiglu_limit > 0.0f) {
                    gate = min(gate, swiglu_limit);
                    up = clamp(up, -swiglu_limit, swiglu_limit);
                }
                gate_mma.Ctile.elems()[item] =
                    gate / (1.0f + exp(-gate)) * up;
            }
        }
        device half* destination =
            y + row_base * output_width + output_base;
        gate_mma.store_result_slice(
            destination,
            output_width,
            short2(0, 0),
            short2(valid_n, short(row_count)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#define instantiate_mfq_grouped_mmq(name, fused, residual) \
    template [[host_name(name)]] [[kernel]] \
    decltype(mfq_grouped_mmq_f16_bm32_bn64_bk96<fused, residual>) \
    mfq_grouped_mmq_f16_bm32_bn64_bk96<fused, residual>;

instantiate_mfq_grouped_mmq(
    "mfq_grouped_mmq_f16_bm32_bn64_bk96",
    false,
    false)
instantiate_mfq_grouped_mmq(
    "mfq_grouped_mmq_swiglu_f16_bm32_bn64_bk96",
    true,
    false)
instantiate_mfq_grouped_mmq(
    "mfq_grouped_mmq_f16_bm32_bn64_bk96_nr",
    false,
    true)
instantiate_mfq_grouped_mmq(
    "mfq_grouped_mmq_swiglu_f16_bm32_bn64_bk96_nr",
    true,
    true)

#ifdef MFQ_ENABLE_NAX
// Homogeneous NINT4/GS24 expert prefill for M5.  Routes have already been
// sorted into expert-contiguous blocks.  Activations are gathered once into
// threadgroup memory and both operands are consumed by NAX without ever
// materializing an expert matrix outside the tile.
template <
    int BM,
    int BN,
    int X_STRIDE,
    int W_STRIDE,
    bool FUSED_SWIGLU,
    bool DIRECT_PACKED>
[[kernel]] void mfq_grouped_nint4_nax_f16_bk96(
    const device int* descriptors [[buffer(0)]],
    const device uchar* vq_indices [[buffer(1)]],
    const device uchar* vq_state [[buffer(2)]],
    const device uchar* vq_aux [[buffer(3)]],
    const device float* vq_anchors [[buffer(4)]],
    const device int8_t* vq_codebooks [[buffer(5)]],
    const device float* vq_scales [[buffer(6)]],
    const device uchar* vq_state_to_bank [[buffer(7)]],
    const device uchar* vq_banks [[buffer(8)]],
    const device float* vq_parameters [[buffer(9)]],
    const device half* x [[buffer(10)]],
    const device int* expert_ids [[buffer(11)]],
    const device int* route_order [[buffer(12)]],
    device half* y [[buffer(13)]],
    constant MfqGroupedMmqParams& params [[buffer(14)]],
    const device float* vq_residual_codebooks [[buffer(15)]],
    const device short* vq_residual_first [[buffer(16)]],
    const device short* vq_residual_second [[buffer(17)]],
    const device int* block_meta [[buffer(18)]],
    const device int* block_count [[buffer(19)]],
    const device uchar* mx_values [[buffer(20)]],
    const device uchar* mx_scales [[buffer(21)]],
    const device uchar* nint_q [[buffer(22)]],
    const device uchar* nint_sub_scale [[buffer(23)]],
    const device uchar* nint_sub_min [[buffer(24)]],
    const device float* nint_anchor_scale [[buffer(25)]],
    const device float* nint_anchor_min [[buffer(26)]],
    const device int8_t* q8_q [[buffer(27)]],
    const device half* q8_scales [[buffer(28)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    constexpr int BK = 96;
    // BM64 uses twice as many SIMD groups in the row dimension.  This keeps
    // every SIMD group's accumulator geometry identical to BM32 (16x32)
    // instead of doubling TM and spilling the fused gate/up accumulator.
    static_assert(BM % 16 == 0);
    constexpr int WM = BM / 16;
    constexpr int WN = BN / 32;
    constexpr uint TGP_SIZE = uint(WM * WN * 32);
    constexpr short SM = BM / WM;
    constexpr short SN = BN / WN;
    constexpr short SK = 32;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;
    constexpr short TK = SK / 16;
    constexpr int PROJECTIONS = FUSED_SWIGLU ? 2 : 1;

    int output_base = int(tid.x) * BN;
    int block_id = int(tid.y);
    int nblocks = block_count[0];
    if (output_base >= params.output_width || block_id >= nblocks) {
        return;
    }
    int row_base = block_meta[block_id * 3 + 0];
    int expert = block_meta[block_id * 3 + 1];
    int row_count = block_meta[block_id * 3 + 2];
    if (row_count <= 0 || expert < 0 || expert >= params.experts) {
        return;
    }

    const device int* descriptor =
        descriptors + expert * params.descriptor_size;
    uint family = uint(descriptor[0]);
    uint local_expert = uint(descriptor[1]);
    uint rotation = uint(descriptor[27]);
    short valid_n = short(min(BN, params.output_width - output_base));
    short tm = short(SM * int(simd_group_id / WN));
    short tn = short(SN * int(simd_group_id % WN));
    short simd_m = short(max(0, min(int(SM), row_count - int(tm))));
    short simd_n = short(max(0, min(int(SN), int(valid_n) - int(tn))));

    threadgroup half Xs[BM * X_STRIDE];
    // The direct path never materializes a decoded FP16 weight tile.  A
    // one-element declaration keeps the template valid without reserving the
    // 13 KiB staging allocation used by the compatibility path.
    threadgroup half Ws[DIRECT_PACKED ? 1 : BN * W_STRIDE];
    mlx::steel::NAXTile<float, TM, TN> gate_tile;
    mlx::steel::NAXTile<float, TM, TN> up_tile;
    gate_tile.clear();
    up_tile.clear();

    for (int k_base = 0; k_base < params.input_width; k_base += BK) {
        for (uint item = thread_id;
             item < uint(BM * BK);
             item += TGP_SIZE) {
            uint row = item / uint(BK);
            uint column = item - row * uint(BK);
            half value = half(0.0f);
            int input_column = k_base + int(column);
            if (int(row) < row_count && column < uint(BK)
                && input_column < params.input_width) {
                uint route_index = uint(route_order[row_base + int(row)]);
                uint source_row = params.input_sorted != 0
                    ? uint(row_base) + row
                    : (params.shared_input != 0
                        ? route_index / uint(params.routes)
                        : route_index);
                uint source_offset =
                    (rotation * uint(params.variant_stride) + source_row)
                    * uint(params.input_width);
                value = x[source_offset + uint(input_column)];
            }
            Xs[row * uint(X_STRIDE) + column] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int projection = 0; projection < PROJECTIONS; ++projection) {
            if constexpr (!DIRECT_PACKED) {
                // Complete tiles overwrite all weight elements. Clear only a
                // boundary tile, where an absent tail would otherwise leave
                // undefined threadgroup data multiplied by padded zeros.
                if (k_base + BK > params.input_width || valid_n < BN) {
                    for (uint item = thread_id;
                         item < uint(BN * W_STRIDE);
                         item += TGP_SIZE) {
                        Ws[item] = half(0.0f);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                uint nint_bits = uint(descriptor[4]);
                uint nint_group_size = uint(descriptor[5]);
                bool aligned_nint = family == 0u && (
                    (nint_bits == 2u && nint_group_size == 16u)
                    || ((nint_bits == 3u || nint_bits == 4u
                         || nint_bits == 6u)
                        && nint_group_size == 24u)
                    || (nint_bits == 8u && nint_group_size == 48u));
                bool sliced_nint5 = family == 0u
                    && nint_bits == 5u && nint_group_size == 28u;
                bool scalar_decode = family == 0u && !aligned_nint;
                if (aligned_nint) {
                    uint groups_per_tile = uint(BK) / nint_group_size;
                    uint decode_units_per_tile = nint_bits == 2u
                        ? (groups_per_tile + 1u) / 2u
                        : groups_per_tile;
                    for (uint item = thread_id;
                         item < uint(BN) * decode_units_per_tile;
                         item += TGP_SIZE) {
                        uint output_row = item / decode_units_per_tile;
                        uint decode_unit =
                            item - output_row * decode_units_per_tile;
                        uint local_group = nint_bits == 2u
                            ? decode_unit * 2u
                            : decode_unit;
                        uint input_column =
                            uint(k_base) + local_group * nint_group_size;
                        if (output_row >= uint(valid_n)
                            || input_column >= uint(params.input_width)) {
                            continue;
                        }
                        uint group = input_column / nint_group_size;
                        uint pool_row =
                            local_expert * uint(params.matrix_output_width)
                            + uint(output_base) + output_row
                            + uint(projection * params.output_width);
                        threadgroup half* target =
                            Ws + output_row * uint(W_STRIDE)
                            + local_group * nint_group_size;
                        if (nint_bits == 2u) {
                            if (local_group + 1u < groups_per_tile
                                && input_column + nint_group_size
                                    < uint(params.input_width)) {
                                decode_nint2_pair32(
                                    descriptor, nint_q, nint_sub_scale,
                                    nint_sub_min, nint_anchor_scale,
                                    nint_anchor_min, target, pool_row, group);
                            } else {
                                decode_nint2_group16(
                                    descriptor, nint_q, nint_sub_scale,
                                    nint_sub_min, nint_anchor_scale,
                                    nint_anchor_min, target, pool_row, group);
                            }
                        } else if (nint_bits == 3u) {
                            decode_nint3_group24(
                                descriptor, nint_q, nint_sub_scale,
                                nint_sub_min, nint_anchor_scale,
                                nint_anchor_min, target, pool_row, group);
                        } else if (nint_bits == 4u) {
                            decode_nint4_group24(
                                descriptor, nint_q, nint_sub_scale,
                                nint_sub_min, nint_anchor_scale,
                                nint_anchor_min, target, pool_row, group);
                        } else if (nint_bits == 6u) {
                            decode_nint6_group24(
                                descriptor, nint_q, nint_sub_scale,
                                nint_sub_min, nint_anchor_scale,
                                nint_anchor_min, target, pool_row, group);
                        } else {
                            decode_nint8_group48(
                                descriptor, nint_q, nint_sub_scale,
                                nint_sub_min, nint_anchor_scale,
                                nint_anchor_min, target, pool_row, group,
                                uint(params.input_width));
                        }
                    }
                } else if (sliced_nint5) {
                    constexpr uint GROUP_SIZE = 28u;
                    constexpr uint MAX_GROUPS_PER_TILE = 5u;
                    uint first_group = uint(k_base) / GROUP_SIZE;
                    uint tile_end = min(
                        uint(k_base + BK),
                        uint(params.input_width));
                    uint last_group = (tile_end - 1u) / GROUP_SIZE;
                    uint tile_groups = last_group - first_group + 1u;
                    for (uint item = thread_id;
                         item < uint(BN) * MAX_GROUPS_PER_TILE;
                         item += TGP_SIZE) {
                        uint output_row = item / MAX_GROUPS_PER_TILE;
                        uint group_slot =
                            item - output_row * MAX_GROUPS_PER_TILE;
                        if (output_row >= uint(valid_n)
                            || group_slot >= tile_groups) {
                            continue;
                        }
                        uint group = first_group + group_slot;
                        uint group_start = group * GROUP_SIZE;
                        uint slice_start = max(uint(k_base), group_start);
                        uint slice_end = min(tile_end, group_start + GROUP_SIZE);
                        uint pool_row = local_expert
                                * uint(params.matrix_output_width)
                            + uint(output_base) + output_row
                            + uint(projection * params.output_width);
                        decode_nint_group_slice<5u, GROUP_SIZE>(
                            descriptor, nint_q, nint_sub_scale,
                            nint_sub_min, nint_anchor_scale,
                            nint_anchor_min,
                            Ws + output_row * uint(W_STRIDE)
                                + slice_start - uint(k_base),
                            pool_row, group, slice_start - group_start,
                            slice_end - slice_start);
                    }
                } else if (family == 4u) {
                    constexpr uint VALUES_PER_ITEM = 4u;
                    constexpr uint ITEMS_PER_ROW = uint(BK) / VALUES_PER_ITEM;
                    for (uint item = thread_id;
                         item < uint(BN) * ITEMS_PER_ROW;
                         item += TGP_SIZE) {
                        uint output_row = item / ITEMS_PER_ROW;
                        uint local_item = item - output_row * ITEMS_PER_ROW;
                        uint local_column = local_item * VALUES_PER_ITEM;
                        uint input_column = uint(k_base) + local_column;
                        if (output_row < uint(valid_n)
                            && input_column < uint(params.input_width)) {
                            uint pool_row = local_expert
                                    * uint(params.matrix_output_width)
                                + uint(output_base) + output_row
                                + uint(projection * params.output_width);
                            *reinterpret_cast<threadgroup half4*>(
                                Ws + output_row * uint(W_STRIDE)
                                    + local_column) = decode_mxfp8_quad_at(
                                descriptor, mx_values, mx_scales,
                                pool_row, input_column,
                                uint(params.input_width));
                        }
                    }
                } else if (family == 5u || family == 6u) {
                    constexpr uint VALUES_PER_ITEM = 4u;
                    constexpr uint ITEMS_PER_ROW = uint(BK) / VALUES_PER_ITEM;
                    for (uint item = thread_id;
                         item < uint(BN) * ITEMS_PER_ROW;
                         item += TGP_SIZE) {
                        uint output_row = item / ITEMS_PER_ROW;
                        uint local_item = item - output_row * ITEMS_PER_ROW;
                        uint local_column = local_item * VALUES_PER_ITEM;
                        uint input_column = uint(k_base) + local_column;
                        if (output_row >= uint(valid_n)
                            || input_column >= uint(params.input_width)) {
                            continue;
                        }
                        uint pool_row = local_expert
                                * uint(params.matrix_output_width)
                            + uint(output_base) + output_row
                            + uint(projection * params.output_width);
                        threadgroup half* target =
                            Ws + output_row * uint(W_STRIDE) + local_column;
                        if (input_column + VALUES_PER_ITEM
                            <= uint(params.input_width)) {
                            *reinterpret_cast<threadgroup half4*>(target) =
                                decode_dense_quad_at(
                                    descriptor, q8_q, family, pool_row,
                                    input_column, uint(params.input_width));
                        } else {
#pragma clang loop unroll(full)
                            for (uint lane = 0u;
                                 lane < VALUES_PER_ITEM;
                                 ++lane) {
                                if (input_column + lane
                                    < uint(params.input_width)) {
                                    target[lane] = decode_dense_value_at(
                                        descriptor, q8_q, family, pool_row,
                                        input_column + lane,
                                        uint(params.input_width));
                                }
                            }
                        }
                    }
                } else if (scalar_decode) {
                    for (uint item = thread_id;
                         item < uint(BN * BK);
                         item += TGP_SIZE) {
                        uint output_row = item / uint(BK);
                        uint local_column = item - output_row * uint(BK);
                        uint input_column = uint(k_base) + local_column;
                        if (output_row < uint(valid_n)
                            && input_column < uint(params.input_width)) {
                            uint pool_row =
                                local_expert * uint(params.matrix_output_width)
                                + uint(output_base) + output_row
                                + uint(projection * params.output_width);
                            half value;
                            if (family == 0u) {
                                value = decode_nint_value(
                                    descriptor, nint_q, nint_sub_scale,
                                    nint_sub_min, nint_anchor_scale,
                                    nint_anchor_min, pool_row, input_column);
                            } else if (family == 4u) {
                                value = decode_mxfp8_value_at(
                                    descriptor, mx_values, mx_scales,
                                    pool_row, input_column,
                                    uint(params.input_width));
                            } else {
                                value = decode_dense_value_at(
                                    descriptor, q8_q, family, pool_row,
                                    input_column,
                                    uint(params.input_width));
                            }
                            Ws[output_row * uint(W_STRIDE) + local_column] =
                                value;
                        }
                    }
                } else {
                    constexpr uint GROUPS_PER_TILE = BK / 24;
                    for (uint item = thread_id;
                         item < uint(BN) * GROUPS_PER_TILE;
                         item += TGP_SIZE) {
                        uint output_row = item / GROUPS_PER_TILE;
                        uint local_group = item - output_row * GROUPS_PER_TILE;
                        if (output_row < uint(valid_n) && family == 2u
                            && local_group < uint(BK / 32)) {
                            uint input_column =
                                uint(k_base) + local_group * 32u;
                            if (input_column >= uint(params.input_width)) {
                                continue;
                            }
                            uint group = input_column / 32u;
                            uint pool_row = local_expert
                                    * uint(params.matrix_output_width)
                                + uint(output_base) + output_row
                                + uint(projection * params.output_width);
                            decode_nint8_zero_group32(
                                descriptor, q8_q, q8_scales,
                                Ws + output_row * uint(W_STRIDE)
                                    + local_group * 32u,
                                pool_row, group);
                        } else if (output_row < uint(valid_n)
                            && family == 3u
                            && local_group < uint(BK / 32)) {
                            uint input_column =
                                uint(k_base) + local_group * 32u;
                            if (input_column >= uint(params.input_width)) {
                                continue;
                            }
                            uint group = input_column / 32u;
                            uint pool_row = local_expert
                                    * uint(params.matrix_output_width)
                                + uint(output_base) + output_row
                                + uint(projection * params.output_width);
                            decode_mxfp4_group32(
                                descriptor, mx_values, mx_scales,
                                Ws + output_row * uint(W_STRIDE)
                                    + local_group * 32u,
                                pool_row, group,
                                uint(params.input_width));
                        } else if (output_row < uint(valid_n)
                            && family == 1u) {
                            uint input_column =
                                uint(k_base) + local_group * 24u;
                            if (input_column >= uint(params.input_width)) {
                                continue;
                            }
                            uint group = input_column / 24u;
                            uint pool_row = local_expert
                                    * uint(params.matrix_output_width)
                                + uint(output_base) + output_row
                                + uint(projection * params.output_width);
                            decode_vq_group24(
                                descriptor, vq_indices, vq_state, vq_aux,
                                vq_anchors, vq_codebooks, vq_scales,
                                vq_state_to_bank, vq_banks, vq_parameters,
                                Ws + output_row * uint(W_STRIDE)
                                    + local_group * 24u,
                                pool_row, group,
                                uint(params.input_width));
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            for (int kk = 0; kk < BK; kk += SK) {
                mlx::steel::NAXTile<half, TM, TK> input_tile;
                mlx::steel::NAXTile<half, TN, TK> weight_tile;
                input_tile.template load<half, X_STRIDE, 1>(
                    Xs + int(tm) * X_STRIDE + kk);
                if constexpr (DIRECT_PACKED) {
                    // BaseNAXFrag assigns every SIMD lane two rows and four
                    // contiguous columns from each 16x16 fragment.  Decode
                    // those packed NINT values straight into the fragment;
                    // no FP16 weight tile or threadgroup round trip exists.
                    short2 fragment_coord =
                        mlx::steel::BaseNAXFrag::get_coord();
                    uint bits = uint(descriptor[4]);
                    uint group_size = uint(descriptor[5]);
                    uint groups = uint(descriptor[6]);
                    uint q_offset = uint(descriptor[7]);
                    uint sub_offset = uint(descriptor[8]);
                    uint anchor_offset = uint(descriptor[9]);
                    uint q5_execution = uint(descriptor[10]);
#pragma clang loop unroll(full)
                    for (short fragment_n = 0;
                         fragment_n < TN;
                         ++fragment_n) {
#pragma clang loop unroll(full)
                    for (short fragment_k = 0;
                         fragment_k < TK;
                         ++fragment_k) {
                        thread auto& weight_fragment =
                            weight_tile.frag_at(fragment_n, fragment_k);
#pragma clang loop unroll(full)
                        for (short fragment_row = 0;
                             fragment_row < 2;
                             ++fragment_row) {
                            int local_output = int(tn)
                                + int(fragment_n) * 16
                                + int(fragment_coord.y)
                                + int(fragment_row) * 8;
                            int input_column = k_base + kk
                                + int(fragment_k) * 16
                                + int(fragment_coord.x);
                            half4 decoded = half4(0.0h);
                            if (local_output < int(valid_n)
                                && input_column < params.input_width) {
                                uint pool_row = local_expert
                                        * uint(params.matrix_output_width)
                                    + uint(output_base + local_output)
                                    + uint(projection * params.output_width);
                                uint first_group =
                                    uint(input_column) / group_size;
                                bool complete_quad = input_column + 3
                                    < params.input_width;
                                bool one_group = complete_quad
                                    && uint(input_column + 3) / group_size
                                        == first_group;
                                if (one_group) {
                                    uint within_group = uint(input_column)
                                        - first_group * group_size;
                                    uint metadata =
                                        pool_row * groups + first_group;
                                    float scale = nint_anchor_scale[
                                            anchor_offset + pool_row]
                                        * float(nint_sub_scale[
                                            sub_offset + metadata]);
                                    float minimum = nint_anchor_min[
                                            anchor_offset + pool_row]
                                        * float(nint_sub_min[
                                            sub_offset + metadata]);
                                    ushort4 quantized = read_nint_quad(
                                        nint_q + q_offset,
                                        metadata * group_size + within_group,
                                        bits,
                                        group_size,
                                        q5_execution);
#pragma clang loop unroll(full)
                                    for (short element = 0;
                                         element < 4;
                                         ++element) {
                                        decoded[element] = half(
                                            scale * float(quantized[element])
                                            - minimum);
                                    }
                                } else {
#pragma clang loop unroll(full)
                                    for (short element = 0;
                                         element < 4;
                                         ++element) {
                                        int column =
                                            input_column + int(element);
                                        if (column < params.input_width) {
                                            decoded[element] =
                                                decode_nint_value(
                                                    descriptor, nint_q,
                                                    nint_sub_scale,
                                                    nint_sub_min,
                                                    nint_anchor_scale,
                                                    nint_anchor_min,
                                                    pool_row,
                                                    uint(column));
                                        }
                                    }
                                }
                            }
                            weight_fragment[fragment_row * 4] = decoded[0];
                            weight_fragment[fragment_row * 4 + 1] = decoded[1];
                            weight_fragment[fragment_row * 4 + 2] = decoded[2];
                            weight_fragment[fragment_row * 4 + 3] = decoded[3];
                        }
                    }
                    }
                } else {
                    weight_tile.template load<half, W_STRIDE, 1>(
                        Ws + int(tn) * W_STRIDE + kk);
                }
                if (projection == 0) {
                    mlx::steel::tile_matmad_nax(
                        gate_tile,
                        input_tile,
                        metal::bool_constant<false>{},
                        weight_tile,
                        metal::bool_constant<true>{});
                } else {
                    mlx::steel::tile_matmad_nax(
                        up_tile,
                        input_tile,
                        metal::bool_constant<false>{},
                        weight_tile,
                        metal::bool_constant<true>{});
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if constexpr (FUSED_SWIGLU) {
        for (short item = 0;
             item < decltype(gate_tile)::kElemsPerTile;
             ++item) {
            float gate = gate_tile.elems()[item];
            float up = up_tile.elems()[item];
            if (params.swiglu_limit > 0.0f) {
                gate = min(gate, params.swiglu_limit);
                up = clamp(up, -params.swiglu_limit, params.swiglu_limit);
            }
            gate_tile.elems()[item] =
                gate / (1.0f + exp(-gate)) * up;
        }
    }
    if (simd_m > 0 && simd_n > 0) {
        device half* destination =
            y + (row_base + int(tm)) * params.output_width
                + output_base + int(tn);
        gate_tile.store_safe(
            destination,
            params.output_width,
            short2(simd_n, simd_m));
    }
}

#define instantiate_mfq_grouped_nint4_nax( \
    name, bm, bn, x_stride, w_stride, fused, direct) \
    template [[host_name(name)]] [[kernel]] \
    decltype(mfq_grouped_nint4_nax_f16_bk96< \
        bm, bn, x_stride, w_stride, fused, direct>) \
    mfq_grouped_nint4_nax_f16_bk96< \
        bm, bn, x_stride, w_stride, fused, direct>;

instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_f16_bm32_bn64_bk96",
    32,
    64,
    104,
    104,
    false,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_swiglu_f16_bm32_bn64_bk96",
    32,
    64,
    104,
    104,
    true,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_f16_bm32_bn128_bk96",
    32,
    128,
    100,
    100,
    false,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_swiglu_f16_bm32_bn128_bk96",
    32,
    128,
    100,
    100,
    true,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_direct_f16_bm32_bn64_bk96",
    32,
    64,
    104,
    104,
    false,
    true)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_direct_swiglu_f16_bm32_bn64_bk96",
    32,
    64,
    104,
    104,
    true,
    true)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_f16_bm64_bn64_bk96",
    64,
    64,
    104,
    104,
    false,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_swiglu_f16_bm64_bn64_bk96",
    64,
    64,
    104,
    104,
    true,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_f16_bm64_bn96_bk96",
    64,
    96,
    104,
    100,
    false,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_swiglu_f16_bm64_bn96_bk96",
    64,
    96,
    104,
    100,
    true,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_f16_bm48_bn64_bk96",
    48,
    64,
    100,
    100,
    false,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_swiglu_f16_bm48_bn64_bk96",
    48,
    64,
    100,
    100,
    true,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_f16_bm48_bn96_bk96",
    48,
    96,
    100,
    100,
    false,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_swiglu_f16_bm48_bn96_bk96",
    48,
    96,
    100,
    100,
    true,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_f16_bm96_bn64_bk96",
    96,
    64,
    100,
    100,
    false,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_swiglu_f16_bm96_bn64_bk96",
    96,
    64,
    100,
    100,
    true,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_f16_bm80_bn64_bk96",
    80,
    64,
    100,
    100,
    false,
    false)
instantiate_mfq_grouped_nint4_nax(
    "mfq_grouped_nint4_nax_swiglu_f16_bm80_bn64_bk96",
    80,
    64,
    100,
    100,
    true,
    false)
#endif

// Ordinary dense NINT projections use the same packed Steel path as the
// heterogeneous expert pool.  Keeping weights packed here avoids materializing
// a full FP16 matrix before every prefill projection.
struct MfqDenseNintMmqParams {
    int rows;
    int output_width;
    int input_width;
    int bits;
    int group_size;
    int groups;
    int q5_execution;
};

inline half decode_dense_nint_value(
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    constant MfqDenseNintMmqParams& params,
    uint row,
    uint column) {
    uint group_size = uint(params.group_size);
    uint groups = uint(params.groups);
    uint group = column / group_size;
    uint metadata = row * groups + group;
    uint value_index = metadata * group_size
        + column - group * group_size;
    uint quantized = read_nint_value(
        values,
        value_index,
        uint(params.bits),
        group_size,
        uint(params.q5_execution));
    float scale = anchor_scales[row] * float(sub_scales[metadata]);
    float minimum = anchor_mins[row] * float(sub_mins[metadata]);
    return half(scale * float(quantized) - minimum);
}

inline void decode_dense_nint4_group24(
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    threadgroup half* target,
    uint row,
    uint group,
    uint groups) {
    uint metadata = row * groups + group;
    float scale = anchor_scales[row] * float(sub_scales[metadata]);
    float minimum = anchor_mins[row] * float(sub_mins[metadata]);
    uint packed_base = metadata * 12u;
    for (uint pair = 0u; pair < 12u; ++pair) {
        uint packed = uint(values[packed_base + pair]);
        target[pair * 2u] = half(
            scale * float(packed & 15u) - minimum);
        target[pair * 2u + 1u] = half(
            scale * float(packed >> 4u) - minimum);
    }
}

inline void decode_dense_nint6_group24(
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    threadgroup half* target,
    uint row,
    uint group,
    uint groups) {
    uint metadata = row * groups + group;
    float scale = anchor_scales[row] * float(sub_scales[metadata]);
    float minimum = anchor_mins[row] * float(sub_mins[metadata]);
    uint packed_base = metadata * 18u;
    for (uint quartet = 0u; quartet < 6u; ++quartet) {
        uint byte_base = packed_base + quartet * 3u;
        uint packed = uint(values[byte_base])
            | (uint(values[byte_base + 1u]) << 8u)
            | (uint(values[byte_base + 2u]) << 16u);
        uint target_base = quartet * 4u;
        target[target_base] = half(
            scale * float(packed & 63u) - minimum);
        target[target_base + 1u] = half(
            scale * float((packed >> 6u) & 63u) - minimum);
        target[target_base + 2u] = half(
            scale * float((packed >> 12u) & 63u) - minimum);
        target[target_base + 3u] = half(
            scale * float((packed >> 18u) & 63u) - minimum);
    }
}

[[kernel]] void mfq_dense_nint_mmq_f16_bm128_bn64_bk48(
    const device uchar* q [[buffer(0)]],
    const device uchar* sub_scale [[buffer(1)]],
    const device uchar* sub_min [[buffer(2)]],
    const device float* anchor_scale [[buffer(3)]],
    const device float* anchor_min [[buffer(4)]],
    const device half* x [[buffer(5)]],
    device half* y [[buffer(6)]],
    constant MfqDenseNintMmqParams& params [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 48;
    constexpr int BK_padded = 56;
    constexpr uint TGP_SIZE = 256u;
    using mma_t = mlx::steel::BlockMMA<
        half,
        half,
        BM,
        BN,
        BK,
        4,
        2,
        false,
        true,
        BK_padded,
        BK_padded>;

    int output_base = int(tid.x) * BN;
    int row_base = int(tid.y) * BM;
    if (output_base >= params.output_width || row_base >= params.rows) {
        return;
    }
    short valid_n = short(min(BN, params.output_width - output_base));
    short valid_m = short(min(BM, params.rows - row_base));
    threadgroup half Xs[BM * BK_padded];
    threadgroup half Ws[BN * BK_padded];
    thread mma_t mma(simd_group_id, simd_lane_id);

    for (int k_base = 0; k_base < params.input_width; k_base += BK) {
        for (uint item = thread_id;
             item < uint(BM * BK_padded);
             item += TGP_SIZE) {
            uint row = item / uint(BK_padded);
            uint column = item - row * uint(BK_padded);
            half value = half(0.0f);
            int input_column = k_base + int(column);
            if (row < uint(valid_m) && column < uint(BK)
                && input_column < params.input_width) {
                value = x[(uint(row_base) + row)
                    * uint(params.input_width) + uint(input_column)];
            }
            Xs[item] = value;
        }
        for (uint item = thread_id;
             item < uint(BN * BK_padded);
             item += TGP_SIZE) {
            Ws[item] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool group24 = (params.bits == 4 || params.bits == 6)
            && params.group_size == 24
            && params.q5_execution == 0;
        if (group24) {
            constexpr uint GROUPS_PER_TILE = BK / 24;
            for (uint item = thread_id;
                 item < uint(BN) * GROUPS_PER_TILE;
                 item += TGP_SIZE) {
                uint output_row = item / GROUPS_PER_TILE;
                uint local_group = item - output_row * GROUPS_PER_TILE;
                uint input_column = uint(k_base) + local_group * 24u;
                if (output_row < uint(valid_n)
                    && input_column < uint(params.input_width)) {
                    if (params.bits == 4) {
                        decode_dense_nint4_group24(
                            q,
                            sub_scale,
                            sub_min,
                            anchor_scale,
                            anchor_min,
                            Ws + output_row * uint(BK_padded)
                                + local_group * 24u,
                            uint(output_base) + output_row,
                            input_column / 24u,
                            uint(params.groups));
                    } else {
                        decode_dense_nint6_group24(
                            q,
                            sub_scale,
                            sub_min,
                            anchor_scale,
                            anchor_min,
                            Ws + output_row * uint(BK_padded)
                                + local_group * 24u,
                            uint(output_base) + output_row,
                            input_column / 24u,
                            uint(params.groups));
                    }
                }
            }
        } else {
            for (uint item = thread_id;
                 item < uint(BN * BK);
                 item += TGP_SIZE) {
                uint output_row = item / uint(BK);
                uint local_column = item - output_row * uint(BK);
                uint input_column = uint(k_base) + local_column;
                if (output_row < uint(valid_n)
                    && input_column < uint(params.input_width)) {
                    Ws[output_row * uint(BK_padded) + local_column] =
                        decode_dense_nint_value(
                            q,
                            sub_scale,
                            sub_min,
                            anchor_scale,
                            anchor_min,
                            params,
                            uint(output_base) + output_row,
                            input_column);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma.mma(Xs, Ws);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mma.store_result_slice(
        y + row_base * params.output_width + output_base,
        params.output_width,
        short2(0, 0),
        short2(valid_n, valid_m));
}

#ifdef MFQ_ENABLE_NAX
// M5-family large-M path.  Unlike BlockMMA, NAX loads activation fragments
// directly into cooperative registers, so the threadgroup memory budget is
// spent only on one decoded weight tile and occupancy stays high.
[[kernel]] void mfq_dense_nint_nax_f16_bm64_bn64_bk96(
    const device uchar* q [[buffer(0)]],
    const device uchar* sub_scale [[buffer(1)]],
    const device uchar* sub_min [[buffer(2)]],
    const device float* anchor_scale [[buffer(3)]],
    const device float* anchor_min [[buffer(4)]],
    const device half* x [[buffer(5)]],
    device half* y [[buffer(6)]],
    constant MfqDenseNintMmqParams& params [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 96;
    constexpr int BK_padded = 104;
    constexpr int WM = 2;
    constexpr int WN = 2;
    constexpr uint TGP_SIZE = uint(WM * WN * 32);
    constexpr short SM = BM / WM;
    constexpr short SN = BN / WN;
    constexpr short SK = 32;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;
    constexpr short TK = SK / 16;

    int output_base = int(tid.x) * BN;
    int row_base = int(tid.y) * BM;
    if (output_base >= params.output_width || row_base >= params.rows) {
        return;
    }
    short valid_n = short(min(BN, params.output_width - output_base));
    short valid_m = short(min(BM, params.rows - row_base));
    short tm = short(SM * int(simd_group_id / WN));
    short tn = short(SN * int(simd_group_id % WN));
    short simd_m = short(max(0, min(int(SM), int(valid_m) - int(tm))));
    short simd_n = short(max(0, min(int(SN), int(valid_n) - int(tn))));

    threadgroup half Ws[BN * BK_padded];
    mlx::steel::NAXTile<float, TM, TN> output_tile;
    output_tile.clear();

    for (int k_base = 0; k_base < params.input_width; k_base += BK) {
        for (uint item = thread_id;
             item < uint(BN * BK_padded);
             item += TGP_SIZE) {
            Ws[item] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool group24 = (params.bits == 4 || params.bits == 6)
            && params.group_size == 24
            && params.q5_execution == 0;
        if (group24) {
            constexpr uint GROUPS_PER_TILE = BK / 24;
            for (uint item = thread_id;
                 item < uint(BN) * GROUPS_PER_TILE;
                 item += TGP_SIZE) {
                uint output_row = item / GROUPS_PER_TILE;
                uint local_group = item - output_row * GROUPS_PER_TILE;
                uint input_column = uint(k_base) + local_group * 24u;
                if (output_row < uint(valid_n)
                    && input_column < uint(params.input_width)) {
                    threadgroup half* target =
                        Ws + output_row * uint(BK_padded)
                            + local_group * 24u;
                    if (params.bits == 4) {
                        decode_dense_nint4_group24(
                            q,
                            sub_scale,
                            sub_min,
                            anchor_scale,
                            anchor_min,
                            target,
                            uint(output_base) + output_row,
                            input_column / 24u,
                            uint(params.groups));
                    } else {
                        decode_dense_nint6_group24(
                            q,
                            sub_scale,
                            sub_min,
                            anchor_scale,
                            anchor_min,
                            target,
                            uint(output_base) + output_row,
                            input_column / 24u,
                            uint(params.groups));
                    }
                }
            }
        } else {
            for (uint item = thread_id;
                 item < uint(BN * BK);
                 item += TGP_SIZE) {
                uint output_row = item / uint(BK);
                uint local_column = item - output_row * uint(BK);
                uint input_column = uint(k_base) + local_column;
                if (output_row < uint(valid_n)
                    && input_column < uint(params.input_width)) {
                    Ws[output_row * uint(BK_padded) + local_column] =
                        decode_dense_nint_value(
                            q,
                            sub_scale,
                            sub_min,
                            anchor_scale,
                            anchor_min,
                            params,
                            uint(output_base) + output_row,
                            input_column);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < BK; kk += SK) {
            int input_column = k_base + kk;
            short valid_k = short(max(
                0,
                min(int(SK), params.input_width - input_column)));
            mlx::steel::NAXTile<half, TM, TK> input_tile;
            mlx::steel::NAXTile<half, TN, TK> weight_tile;
            const device half* input_base =
                x + (row_base + int(tm)) * params.input_width
                    + input_column;
            if (simd_m == SM && valid_k == SK) {
                input_tile.load(input_base, params.input_width);
            } else {
                input_tile.load_safe(
                    input_base,
                    params.input_width,
                    short2(valid_k, simd_m));
            }
            weight_tile.template load<half, BK_padded, 1>(
                Ws + int(tn) * BK_padded + kk);
            mlx::steel::tile_matmad_nax(
                output_tile,
                input_tile,
                metal::bool_constant<false>{},
                weight_tile,
                metal::bool_constant<true>{});
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device half* destination =
        y + (row_base + int(tm)) * params.output_width
            + output_base + int(tn);
    if (simd_m == SM && simd_n == SN) {
        output_tile.store(destination, params.output_width);
    } else {
        output_tile.store_safe(
            destination,
            params.output_width,
            short2(simd_n, simd_m));
    }
}
#endif
