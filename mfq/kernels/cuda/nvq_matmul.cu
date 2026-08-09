// Compact NPQ/NVQ inference kernels.
//
// The persistent GPU representation keeps every variable-width stream packed:
//   NVQ1-L: 11-bit ternary codebook indices, 1-bit group delta signs
//   NVQ2: 8-bit E8 indices, 7-bit even-parity signs per 8 weights
//   NVQ3: 8-bit D4 indices per 4 weights, 7-bit even-parity signs per 8 weights
// All profiles share one fp32 neuron anchor and one packed relative scale per
// gs=24 group. Decode/small-batch matmul quantizes activations per group and
// performs signed int8 dot products with __dp4a or integer MMA. The medium-M
// path expands only one 64x96 fp16 weight tile at a time for Tensor Core GEMM.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <type_traits>

using namespace nvcuda;

namespace {

constexpr int kNvq1L = 1;
constexpr int kNvq2 = 2;
constexpr int kNvq3 = 3;
constexpr int kNvq2Exec = 4;
constexpr int kNvq2Jsc = 5;
constexpr int kNvq2JscExec = 6;
constexpr int kNpq0L = 7;
constexpr int kNvq1S = 8;
constexpr int kNpq0S = 9;
constexpr int kNvq3Jsc = 10;
constexpr int kNvq3Jsc2 = 11;
constexpr int kNvq3Jsc512 = 12;
constexpr int kNvq2JscL = 13;
constexpr int kNvq2JscXL = 14;
constexpr int kNvq3JscL = 15;
constexpr int kNvq2JscXLGroupExec = 16;
constexpr int kNvq3JscLGroupExec = 17;
constexpr int kGroupSize = 24;
constexpr int kChunksPerGroup = 6;  // six dp4a chunks of four values
constexpr int kGroupsPerWarp = 5;   // 5 * 6 <= 32
constexpr int kJscMetadataBytes = 64;
constexpr int kJscLutOffset = 4;
constexpr int kJscBankMapOffset = 36;
constexpr int kJscE8CodebookBytes = 256 * 8;
constexpr int kJscE8Codebook1024Bytes = 1024 * 8;
constexpr int kJscE8Codebook4096Bytes = 4096 * 8;
constexpr int kJscE8PartialEntries = 384;
constexpr int kJscE8PartialBytes = kJscE8PartialEntries * 8;
constexpr int kJscD4CodebookBytes = 256 * 4;
constexpr int kJscD4Codebook512Bytes = 512 * 4;
constexpr int kJscD4Codebook1024Bytes = 1024 * 4;
constexpr int kNpq0LMetadataBytes = 64;
constexpr int kNpq0LLutOffset = 8;
constexpr int kNpq0LStates = 8;
constexpr int kNpq0LFirstEntries = 8;
constexpr int kNpq0LSecondEntries = 16;
constexpr int kNpq0LFirstBytesPerState = kNpq0LFirstEntries * 4;
constexpr int kNpq0LSecondBytesPerState = kNpq0LSecondEntries * 4;
constexpr int kNpq0LSecondOffset =
    kNpq0LMetadataBytes + kNpq0LStates * kNpq0LFirstBytesPerState;
constexpr int kNpq0LTableBytes =
    kNpq0LSecondOffset + kNpq0LStates * kNpq0LSecondBytesPerState;
constexpr int kNvq1SBankEntries = 512;
constexpr int kNvq1SBankBytes = kNvq1SBankEntries * 8;
constexpr int kNpq0SMetadataBytes = 64;
constexpr int kNpq0SLutOffset = 8;
constexpr int kNpq0SStates = 4;
constexpr int kNpq0SEntries = 64;
constexpr int kNpq0SCodebookBytes = kNpq0SStates * kNpq0SEntries * 8;
constexpr int kNpq0STableBytes = kNpq0SMetadataBytes + kNpq0SCodebookBytes;
constexpr int kNepq0SEntriesPerHalf = 8;
constexpr int kNepq0SBytesPerStateHalf = kNepq0SEntriesPerHalf * 4;
constexpr int kNepq0SSecondOffset =
    kNpq0SMetadataBytes + kNpq0SStates * kNepq0SBytesPerStateHalf;
constexpr int kNepq0SCompactTableBytes =
    kNepq0SSecondOffset + kNpq0SStates * kNepq0SBytesPerStateHalf;
constexpr int kNepqGroupsPerSupergroup = 4;

__device__ __forceinline__ const int8_t * nepq_active_table(
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    int row,
    int group,
    int supergroups,
    int table_stride) {
    const int selector = group / kNepqGroupsPerSupergroup;
    const uint32_t bank = bank_ids[static_cast<int64_t>(row) * supergroups + selector];
    return table_pool + static_cast<int64_t>(bank) * table_stride;
}

__host__ __device__ constexpr bool is_d4_format(int format) {
    return format == kNvq3 || format == kNvq3Jsc ||
           format == kNvq3Jsc2 || format == kNvq3Jsc512 ||
           format == kNvq3JscL || format == kNvq3JscLGroupExec;
}

__host__ __device__ constexpr bool is_e8_format(int format) {
    return format == kNvq2 || format == kNvq2Exec ||
           format == kNvq2Jsc || format == kNvq2JscExec ||
           format == kNvq2JscL || format == kNvq2JscXL ||
           format == kNvq2JscXLGroupExec;
}

__host__ __device__ constexpr int format_index_bits(int format) {
    if (format == kNvq1L) return 11;
    if (format == kNvq1S || format == kNvq3Jsc512) return 9;
    if (format == kNpq0L) return 7;
    if (format == kNpq0S) return 6;
    if (format == kNvq2JscL || format == kNvq3JscL ||
        format == kNvq3JscLGroupExec) return 10;
    if (format == kNvq2JscXL || format == kNvq2JscXLGroupExec) return 12;
    if (format == kNvq2Exec || format == kNvq2JscExec) return 16;
    return 8;
}

template <int FORMAT>
__device__ __forceinline__ const int8_t * active_codebook(
    const int8_t * metadata,
    uint32_t state) {
    if constexpr (FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscE8CodebookBytes;
    }
    if constexpr (FORMAT == kNvq2JscL) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscE8Codebook1024Bytes;
    }
    if constexpr (FORMAT == kNvq2JscXL) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscE8Codebook4096Bytes;
    }
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        const uint32_t bank = state & 3u;
        return metadata + kJscMetadataBytes + bank * kJscE8Codebook4096Bytes;
    }
    if constexpr (FORMAT == kNvq3Jsc) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscD4CodebookBytes;
    }
    if constexpr (FORMAT == kNvq3Jsc2) {
        const uint32_t bank = state & 1u;
        return metadata + kJscMetadataBytes + bank * kJscD4CodebookBytes;
    }
    if constexpr (FORMAT == kNvq3Jsc512) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscD4Codebook512Bytes;
    }
    if constexpr (FORMAT == kNvq3JscL) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscD4Codebook1024Bytes;
    }
    if constexpr (FORMAT == kNvq3JscLGroupExec) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscD4Codebook1024Bytes;
    }
    return metadata;
}

__device__ __forceinline__ uint32_t load_packed_bits(
    const uint8_t * data, int64_t bit, int bits, int64_t nbytes) {
    const int64_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    uint32_t word = data[byte];
    if (byte + 1 < nbytes) word |= static_cast<uint32_t>(data[byte + 1]) << 8;
    if (byte + 2 < nbytes) word |= static_cast<uint32_t>(data[byte + 2]) << 16;
    return (word >> shift) & ((1u << bits) - 1u);
}

__device__ __forceinline__ uint32_t load_packed_4(
    const uint8_t * data, int64_t linear) {
    return (data[linear >> 1] >> ((linear & 1) * 4)) & 0x0fu;
}

__device__ __forceinline__ uint64_t load_group_exec64(
    const uint8_t * data, int row, int group, int ng) {
    return reinterpret_cast<const uint64_t *>(data)[
        static_cast<int64_t>(row) * ng + group];
}

__device__ __forceinline__ void load_group_exec96_words(
    const uint8_t * data,
    int row,
    int group,
    int ng,
    uint32_t (&words)[3]) {
    const uint32_t * source = reinterpret_cast<const uint32_t *>(data) +
        (static_cast<int64_t>(row) * ng + group) * 3;
    words[0] = source[0];
    words[1] = source[1];
    words[2] = source[2];
}

__device__ __forceinline__ uint32_t load_group_exec96_bits(
    const uint8_t * data, int row, int group, int ng, int bit, int bits) {
    uint32_t words[3];
    load_group_exec96_words(data, row, group, ng, words);
    const int word = bit >> 5;
    const int shift = bit & 31;
    uint32_t value = words[word] >> shift;
    if (shift + bits > 32) value |= words[word + 1] << (32 - shift);
    return value & ((1u << bits) - 1u);
}

__device__ __forceinline__ int load_i8x4(const int8_t * p) {
    const uint8_t * u = reinterpret_cast<const uint8_t *>(p);
    return static_cast<int>(u[0]) |
           (static_cast<int>(u[1]) << 8) |
           (static_cast<int>(u[2]) << 16) |
           (static_cast<int>(u[3]) << 24);
}

__device__ __forceinline__ int parity7(uint32_t mask) {
    return __popc(mask & 0x7fu) & 1;
}

__device__ __forceinline__ int sign_bytes4(uint32_t mask8, int base) {
    int result = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int negative = (mask8 >> (base + i)) & 1u;
        result |= (negative ? 0xff : 0x00) << (8 * i);
    }
    return result;
}

__device__ __forceinline__ int apply_sign4(int values, uint32_t mask8, int base) {
    const int signs = sign_bytes4(mask8, base);
    return __vsub4(values ^ signs, signs);
}

__device__ __forceinline__ int nvq1_l_scale_delta4(int values, int delta) {
    int result = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int v = static_cast<int>(static_cast<int8_t>((values >> (8 * i)) & 0xff));
        const int scaled = 8 * v + delta;
        result |= (scaled & 0xff) << (8 * i);
    }
    return result;
}

__device__ __forceinline__ int nvq1_s_scale_delta4(int values, int delta) {
    int result = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int v = static_cast<int>(static_cast<int8_t>((values >> (8 * i)) & 0xff));
        const int scaled = 32 * v + 5 * delta;
        result |= (scaled & 0xff) << (8 * i);
    }
    return result;
}

__device__ __forceinline__ int2 load_npq0_l_vec8(
    const int8_t * metadata,
    uint32_t state,
    uint32_t index) {
    const int8_t * first = metadata + kNpq0LMetadataBytes
        + state * kNpq0LFirstBytesPerState + (index & 7u) * 4;
    const int8_t * second = metadata + kNpq0LSecondOffset
        + state * kNpq0LSecondBytesPerState + (index >> 3) * 4;
    return make_int2(load_i8x4(first), load_i8x4(second));
}

__device__ __forceinline__ int2 load_npq0_s_vec8(
    const int8_t * metadata,
    uint32_t state,
    uint32_t index) {
    const int8_t * code = metadata + kNpq0SMetadataBytes
        + (state * kNpq0SEntries + index) * 8;
    return reinterpret_cast<const int2 *>(code)[0];
}

__device__ __forceinline__ int2 load_nepq0_s_vec8(
    const int8_t * metadata,
    uint32_t state,
    uint32_t index) {
    const int8_t * first = metadata + kNpq0SMetadataBytes
        + state * kNepq0SBytesPerStateHalf + (index & 7u) * 4;
    const int8_t * second = metadata + kNepq0SSecondOffset
        + state * kNepq0SBytesPerStateHalf + (index >> 3) * 4;
    return make_int2(load_i8x4(first), load_i8x4(second));
}

template <int FORMAT>
__device__ __forceinline__ int decode_chunk4(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const int8_t * codebook,
    int row,
    int group,
    int chunk,
    int nvec,
    int nsign,
    int ng,
    int sign_mode,
    uint32_t state) {
    const int vector8 = group * 3 + (chunk >> 1);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        if (vector8 >= nvec || vector8 >= nsign) return 0;
        const int local = vector8 - group * 3;
        const uint64_t metadata = load_group_exec64(indices, row, group, ng);
        const uint32_t segment_metadata =
            (metadata >> (local * 20)) & 0xfffffu;
        const uint32_t index = segment_metadata & 0xfffu;
        const uint32_t mask8 = segment_metadata >> 12;
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return apply_sign4(
            load_i8x4(bank + index * 8 + (chunk & 1) * 4),
            mask8, (chunk & 1) * 4);
    } else if constexpr (FORMAT == kNvq3JscLGroupExec) {
        const int vector4 = group * 6 + chunk;
        if (vector4 >= nvec || vector8 >= nsign) return 0;
        const uint32_t index = load_group_exec96_bits(
            indices, row, group, ng, chunk * 10, 10);
        const int local = vector8 - group * 3;
        const uint32_t mask8 = load_group_exec96_bits(
            indices, row, group, ng, 60 + local * 8, 8);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return apply_sign4(
            load_i8x4(bank + index * 4), mask8, (chunk & 1) * 4);
    } else if constexpr (
        FORMAT == kNvq3 || FORMAT == kNvq3Jsc ||
        FORMAT == kNvq3Jsc2 || FORMAT == kNvq3Jsc512 ||
        FORMAT == kNvq3JscL) {
        const int vector4 = group * 6 + chunk;
        if (vector4 >= nvec || vector8 >= nsign) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector4;
        constexpr int INDEX_BITS = format_index_bits(FORMAT);
        const uint32_t index = INDEX_BITS == 8
            ? indices[index_linear]
            : load_packed_bits(
                indices, index_linear * INDEX_BITS, INDEX_BITS, indices_nbytes);
        const int64_t sign_linear = static_cast<int64_t>(row) * nsign + vector8;
        const uint32_t mask7 = load_packed_bits(aux, sign_linear * 7, 7, aux_nbytes);
        const uint32_t mask8 = mask7 | (static_cast<uint32_t>(parity7(mask7)) << 7);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return apply_sign4(load_i8x4(bank + index * 4), mask8, (chunk & 1) * 4);
    } else if constexpr (FORMAT == kNpq0L) {
        if (vector8 >= nvec) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector8;
        const uint32_t index = load_packed_bits(
            indices, index_linear * 7, 7, indices_nbytes);
        const int2 values = load_npq0_l_vec8(codebook, state, index);
        return chunk & 1 ? values.y : values.x;
    } else if constexpr (FORMAT == kNpq0S) {
        if (vector8 >= nvec) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector8;
        const uint32_t index = load_packed_bits(
            indices, index_linear * 6, 6, indices_nbytes);
        const int2 values = load_npq0_s_vec8(codebook, state, index);
        return chunk & 1 ? values.y : values.x;
    } else if constexpr (
        FORMAT == kNvq2 || FORMAT == kNvq2Exec ||
        FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec ||
        FORMAT == kNvq2JscL || FORMAT == kNvq2JscXL) {
        if (vector8 >= nvec || vector8 >= nsign) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector8;
        uint32_t index;
        uint32_t mask8;
        if constexpr (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec) {
            const uint16_t metadata = reinterpret_cast<const uint16_t *>(indices)[index_linear];
            index = metadata & 0xffu;
            mask8 = metadata >> 8;
        } else {
            constexpr int INDEX_BITS = format_index_bits(FORMAT);
            index = INDEX_BITS == 8
                ? indices[index_linear]
                : load_packed_bits(
                    indices, index_linear * INDEX_BITS, INDEX_BITS, indices_nbytes);
            const int64_t sign_linear = static_cast<int64_t>(row) * nsign + vector8;
            const uint32_t mask7 = load_packed_bits(aux, sign_linear * 7, 7, aux_nbytes);
            const int last = parity7(mask7) ^ (sign_mode ? ((index >> 7) & 1u) : 0u);
            mask8 = mask7 | (static_cast<uint32_t>(last) << 7);
        }
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return apply_sign4(load_i8x4(bank + index * 8 + (chunk & 1) * 4),
                           mask8, (chunk & 1) * 4);
    } else if constexpr (FORMAT == kNvq1S) {
        if (vector8 >= nvec) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector8;
        const uint32_t index = load_packed_bits(indices, index_linear * 9, 9, indices_nbytes);
        const int64_t delta_linear = static_cast<int64_t>(row) * ng + group;
        const int negative_delta = static_cast<int>(
            load_packed_bits(aux, delta_linear, 1, aux_nbytes));
        const int delta = negative_delta ? -1 : 1;
        const int8_t * bank = codebook + negative_delta * kNvq1SBankBytes;
        return nvq1_s_scale_delta4(
            load_i8x4(bank + index * 8 + (chunk & 1) * 4), delta);
    } else {
        if (vector8 >= nvec) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector8;
        const uint32_t index = load_packed_bits(indices, index_linear * 11, 11, indices_nbytes);
        const int64_t delta_linear = static_cast<int64_t>(row) * ng + group;
        const int negative_delta = static_cast<int>(
            load_packed_bits(aux, delta_linear, 1, aux_nbytes));
        const int delta = negative_delta ? -1 : 1;
        return nvq1_l_scale_delta4(
            load_i8x4(codebook + index * 8 + (chunk & 1) * 4), delta);
    }
}

template <int FORMAT>
__device__ __forceinline__ int decode_nepq_chunk4(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const int8_t * codebook,
    int row,
    int group,
    int chunk,
    int nvec,
    int nsign,
    int ng,
    int sign_mode,
    uint32_t state) {
    if constexpr (FORMAT == kNpq0S) {
        const int vector8 = group * 3 + (chunk >> 1);
        if (vector8 >= nvec) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector8;
        const uint32_t index = load_packed_bits(
            indices, index_linear * 6, 6, indices_nbytes);
        const uint32_t half_index = chunk & 1 ? index >> 3 : index & 7u;
        const int8_t * half = chunk & 1
            ? codebook + kNepq0SSecondOffset
            : codebook + kNpq0SMetadataBytes;
        return load_i8x4(
            half + state * kNepq0SBytesPerStateHalf + half_index * 4);
    }
    return decode_chunk4<FORMAT>(
        indices, indices_nbytes, aux, aux_nbytes, codebook,
        row, group, chunk, nvec, nsign, ng, sign_mode, state);
}

template <int FORMAT>
__device__ __forceinline__ float format_scale(
    float anchor,
    uint32_t sub_scale,
    const int8_t * codebook) {
    if constexpr (FORMAT == kNvq3Jsc2) {
        const uint32_t rank = sub_scale >> 1;
        return anchor * static_cast<float>(rank + 1);
    }
    if constexpr (
        FORMAT == kNvq3Jsc || FORMAT == kNvq3Jsc512 ||
        FORMAT == kNvq3JscL || FORMAT == kNvq3JscLGroupExec) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(codebook);
        const __half * lut = reinterpret_cast<const __half *>(bytes + kJscLutOffset);
        return anchor * __half2float(lut[sub_scale]);
    }
    if constexpr (
        FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec ||
        FORMAT == kNvq2JscL || FORMAT == kNvq2JscXL ||
        FORMAT == kNvq2JscXLGroupExec) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(codebook);
        const __half * lut = reinterpret_cast<const __half *>(bytes + kJscLutOffset);
        return anchor * __half2float(lut[sub_scale]);
    }
    if constexpr (FORMAT == kNpq0L) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(codebook);
        const __half * lut = reinterpret_cast<const __half *>(bytes + kNpq0LLutOffset);
        return anchor * __half2float(lut[sub_scale]);
    }
    if constexpr (FORMAT == kNpq0S) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(codebook);
        const __half * lut = reinterpret_cast<const __half *>(bytes + kNpq0SLutOffset);
        return anchor * __half2float(lut[sub_scale]);
    }
    const float scale = anchor * static_cast<float>(sub_scale);
    if constexpr (FORMAT == kNvq1L) return scale * 0.125f;
    if constexpr (FORMAT == kNvq1S) return scale * 0.03125f;
    return scale;
}

template <int FORMAT>
__global__ void nvq_dequant_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    __half * weight,
    int N,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    const int segments = (K + 7) / 8;
    const int64_t total = static_cast<int64_t>(N) * segments;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(linear / segments);
        const int segment = static_cast<int>(linear - static_cast<int64_t>(row) * segments);
        const int k0 = segment * 8;
        const int group = k0 / kGroupSize;
        const int chunk0 = (k0 - group * kGroupSize) / 4;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_packed_bits(
            sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
        const float scale = format_scale<FORMAT>(neuron_scale[row], sub, codebook);
        const int lo = decode_chunk4<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, group, chunk0, nvec, nsign, ng, sign_mode, sub);
        const int hi = decode_chunk4<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, group, chunk0 + 1, nvec, nsign, ng, sign_mode, sub);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k = k0 + i;
            if (k < K) {
                const int packed = i < 4 ? lo : hi;
                const int byte = (packed >> (8 * (i & 3))) & 0xff;
                const int value = static_cast<int>(static_cast<int8_t>(byte));
                weight[static_cast<int64_t>(row) * K + k] =
                    __float2half(scale * static_cast<float>(value));
            }
        }
    }
}

__global__ void nvq_quantize_x_gs24_kernel(
    const __half * x,
    int8_t * qx,
    float * xscale,
    int M,
    int K,
    int ng) {
    const int m = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    if (group >= ng) return;
    const int k = group * kGroupSize + lane;
    const bool valid = lane < kGroupSize && k < K;
    const float value = valid ? __half2float(x[static_cast<int64_t>(m) * K + k]) : 0.0f;
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    int q = 0;
    if (valid) {
        q = static_cast<int>(roundf(value / scale));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    if (lane == 0) xscale[static_cast<int64_t>(m) * ng + group] = scale;
    if (lane < kGroupSize) {
        qx[(static_cast<int64_t>(m) * ng + group) * kGroupSize + lane] =
            valid ? static_cast<int8_t>(q) : static_cast<int8_t>(0);
    }
}

template <int MODE>
__global__ void nvq_quantize_x_gate_gs24_kernel(
    const __half * x,
    const __half * gate,
    int8_t * qx,
    float * xscale,
    int M,
    int K,
    int ng) {
    const int m = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    const int k = group * kGroupSize + lane;
    const bool valid = lane < kGroupSize && k < K;
    float value = 0.0f;
    if (valid) {
        const int64_t offset = static_cast<int64_t>(m) * K + k;
        const float gate_value = __half2float(gate[offset]);
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        const float multiplier = MODE == 1 ? sigmoid : gate_value * sigmoid;
        value = __half2float(x[offset]) * multiplier;
    }
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    int q = 0;
    if (valid) {
        q = static_cast<int>(roundf(value / scale));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    if (lane == 0) xscale[static_cast<int64_t>(m) * ng + group] = scale;
    if (lane < kGroupSize) {
        qx[(static_cast<int64_t>(m) * ng + group) * kGroupSize + lane] =
            valid ? static_cast<int8_t>(q) : static_cast<int8_t>(0);
    }
}

template <int FORMAT, int MAX_M, int M_SPLIT>
__global__ void __launch_bounds__(256) nvq_gemv_gs24_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    constexpr int kRowsPerBlock = 4;
    constexpr int kRowsPerSplit = (MAX_M + M_SPLIT - 1) / M_SPLIT;
    const int warp = threadIdx.y;
    const int row = blockIdx.x * kRowsPerBlock + warp / M_SPLIT;
    const int msplit = warp % M_SPLIT;
    const int lane = threadIdx.x;
    if (row >= N) return;

    float acc[kRowsPerSplit];
#pragma unroll
    for (int i = 0; i < kRowsPerSplit; ++i) acc[i] = 0.0f;

    const int relative_group = lane / kChunksPerGroup;
    const int chunk = lane - relative_group * kChunksPerGroup;
    const bool active = relative_group < kGroupsPerWarp;
    for (int group_base = 0; group_base < ng; group_base += kGroupsPerWarp) {
        const int group = group_base + relative_group;
        if (!active || group >= ng) continue;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_packed_bits(
            sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
        const int weights = decode_chunk4<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, group, chunk, nvec, nsign, ng, sign_mode, sub);
        const float weight_scale = format_scale<FORMAT>(neuron_scale[row], sub, codebook);
        const int k = group * kGroupSize + chunk * 4;
#pragma unroll
        for (int i = 0; i < kRowsPerSplit; ++i) {
            const int m = msplit * kRowsPerSplit + i;
            if (m < M) {
                const int activations = load_i8x4(qx + static_cast<int64_t>(m) * ng * kGroupSize + k);
                const int dot = __dp4a(weights, activations, 0);
                acc[i] += weight_scale * xscale[static_cast<int64_t>(m) * ng + group]
                          * static_cast<float>(dot);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < kRowsPerSplit; ++i) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[i] += __shfl_xor_sync(0xffffffff, acc[i], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < kRowsPerSplit; ++i) {
            const int m = msplit * kRowsPerSplit + i;
            if (m < M) output[static_cast<int64_t>(m) * N + row] = __float2half(acc[i]);
        }
    }
}

__device__ __forceinline__ int2 apply_sign8(int2 values, uint32_t mask8) {
    const uint32_t broadcast = mask8 * 0x01010101u;
    const int signs0 = __vcmpne4(broadcast & 0x08040201u, 0);
    const int signs1 = __vcmpne4(broadcast & 0x80402010u, 0);
    return make_int2(
        __vsub4(values.x ^ signs0, signs0),
        __vsub4(values.y ^ signs1, signs1));
}

__device__ uint32_t nvq_sign_expand4[16] = {
    0x00000000u, 0x000000ffu, 0x0000ff00u, 0x0000ffffu,
    0x00ff0000u, 0x00ff00ffu, 0x00ffff00u, 0x00ffffffu,
    0xff000000u, 0xff0000ffu, 0xff00ff00u, 0xff00ffffu,
    0xffff0000u, 0xffff00ffu, 0xffffff00u, 0xffffffffu,
};

__device__ __forceinline__ int2 apply_sign8_exec(int2 values, uint32_t mask8) {
    const int signs0 = static_cast<int>(__ldg(nvq_sign_expand4 + (mask8 & 0x0fu)));
    const int signs1 = static_cast<int>(__ldg(nvq_sign_expand4 + (mask8 >> 4)));
    return make_int2(
        __vsub4(values.x ^ signs0, signs0),
        __vsub4(values.y ^ signs1, signs1));
}

template <int FORMAT>
struct NvqVec8Values {
    int2 values;
    int delta;
    bool valid;
};

struct NvqDeviceWeight {
    const uint8_t * indices;
    int64_t indices_nbytes;
    const uint8_t * aux;
    int64_t aux_nbytes;
    const uint8_t * sub_scale;
    int64_t sub_scale_nbytes;
    const float * neuron_scale;
    const int8_t * codebook;
    int codebook_nbytes;
    int N;
    int ng;
    int nvec;
    int nsign;
    int sub_bits;
    int sign_mode;
};

template <int FORMAT>
__device__ __forceinline__ NvqVec8Values<FORMAT> load_nvq_vec8(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const int8_t * codebook,
    int row,
    int segment,
    int group,
    int ng,
    int nvec,
    int nsign,
    int sign_mode,
    uint32_t state) {
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        if (segment >= nsign) return {make_int2(0, 0), 0, false};
        const int local = segment - group * 3;
        const uint64_t metadata = load_group_exec64(indices, row, group, ng);
        const uint32_t segment_metadata =
            (metadata >> (local * 20)) & 0xfffffu;
        const uint32_t index = segment_metadata & 0xfffu;
        const uint32_t mask8 = segment_metadata >> 12;
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return {apply_sign8(
            reinterpret_cast<const int2 *>(bank)[index], mask8), 0, true};
    } else if constexpr (FORMAT == kNvq3JscLGroupExec) {
        const int vector4 = segment * 2;
        if (vector4 >= nvec || segment >= nsign) {
            return {make_int2(0, 0), 0, false};
        }
        const int local = segment - group * 3;
        const uint32_t index0 = load_group_exec96_bits(
            indices, row, group, ng, local * 20, 10);
        const uint32_t index1 = vector4 + 1 < nvec
            ? load_group_exec96_bits(
                indices, row, group, ng, local * 20 + 10, 10)
            : 0;
        const uint32_t mask8 = load_group_exec96_bits(
            indices, row, group, ng, 60 + local * 8, 8);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return {apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index0],
                reinterpret_cast<const int *>(bank)[index1]),
            mask8), 0, true};
    } else if constexpr (
        FORMAT == kNvq3 || FORMAT == kNvq3Jsc ||
        FORMAT == kNvq3Jsc2 || FORMAT == kNvq3Jsc512 ||
        FORMAT == kNvq3JscL) {
        const int vector4 = segment * 2;
        if (vector4 >= nvec) return {make_int2(0, 0), 0, false};
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector4;
        constexpr int INDEX_BITS = format_index_bits(FORMAT);
        const uint32_t index0 = INDEX_BITS == 8
            ? indices[index_linear]
            : load_packed_bits(
                indices, index_linear * INDEX_BITS, INDEX_BITS, indices_nbytes);
        const uint32_t index1 = vector4 + 1 < nvec
            ? (INDEX_BITS != 8
                ? load_packed_bits(
                    indices, (index_linear + 1) * INDEX_BITS,
                    INDEX_BITS, indices_nbytes)
                : indices[index_linear + 1])
            : 0;
        const int64_t sign_linear = static_cast<int64_t>(row) * nsign + segment;
        const uint32_t mask7 = load_packed_bits(aux, sign_linear * 7, 7, aux_nbytes);
        const uint32_t mask8 = mask7 | (static_cast<uint32_t>(parity7(mask7)) << 7);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return {apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index0],
                reinterpret_cast<const int *>(bank)[index1]),
            mask8), 0, true};
    } else {
        if (segment >= nvec) return {make_int2(0, 0), 0, false};
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + segment;
        if constexpr (FORMAT == kNpq0L) {
            const uint32_t index = load_packed_bits(
                indices, index_linear * 7, 7, indices_nbytes);
            return {load_npq0_l_vec8(codebook, state, index), 0, true};
        } else if constexpr (FORMAT == kNpq0S) {
            const uint32_t index = load_packed_bits(
                indices, index_linear * 6, 6, indices_nbytes);
            return {load_npq0_s_vec8(codebook, state, index), 0, true};
        } else if constexpr (
            FORMAT == kNvq2 || FORMAT == kNvq2Exec ||
            FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec ||
            FORMAT == kNvq2JscL || FORMAT == kNvq2JscXL) {
            uint32_t index;
            uint32_t mask8;
            if constexpr (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec) {
                const uint16_t metadata = reinterpret_cast<const uint16_t *>(indices)[index_linear];
                index = metadata & 0xffu;
                mask8 = metadata >> 8;
            } else {
                constexpr int INDEX_BITS = format_index_bits(FORMAT);
                index = INDEX_BITS == 8
                    ? indices[index_linear]
                    : load_packed_bits(
                        indices, index_linear * INDEX_BITS,
                        INDEX_BITS, indices_nbytes);
                const int64_t sign_linear = static_cast<int64_t>(row) * nsign + segment;
                const uint32_t mask7 = load_packed_bits(aux, sign_linear * 7, 7, aux_nbytes);
                const int last = parity7(mask7) ^ (sign_mode ? ((index >> 7) & 1u) : 0u);
                mask8 = mask7 | (static_cast<uint32_t>(last) << 7);
            }
            const int8_t * bank = active_codebook<FORMAT>(codebook, state);
            const int2 values = reinterpret_cast<const int2 *>(bank)[index];
            if constexpr (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec) {
                return {apply_sign8_exec(values, mask8), 0, true};
            }
            return {apply_sign8(values, mask8), 0, true};
        } else if constexpr (FORMAT == kNvq1S) {
            const uint32_t index = load_packed_bits(
                indices, index_linear * 9, 9, indices_nbytes);
            const int64_t delta_linear = static_cast<int64_t>(row) * ng + group;
            const int negative_delta = static_cast<int>(
                load_packed_bits(aux, delta_linear, 1, aux_nbytes));
            const int delta = negative_delta ? -1 : 1;
            const int8_t * bank = codebook + negative_delta * kNvq1SBankBytes;
            return {reinterpret_cast<const int2 *>(bank)[index], delta, true};
        } else {
            const uint32_t index = load_packed_bits(indices, index_linear * 11, 11, indices_nbytes);
            const int64_t delta_linear = static_cast<int64_t>(row) * ng + group;
            const int delta = load_packed_bits(aux, delta_linear, 1, aux_nbytes) ? -1 : 1;
            return {reinterpret_cast<const int2 *>(codebook)[index], delta, true};
        }
    }
}

template <int FORMAT>
__device__ __forceinline__ NvqVec8Values<FORMAT> load_nepq_vec8(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const int8_t * codebook,
    int row,
    int segment,
    int group,
    int ng,
    int nvec,
    int nsign,
    int sign_mode,
    uint32_t state) {
    if constexpr (FORMAT == kNpq0S) {
        if (segment >= nvec) return {make_int2(0, 0), 0, false};
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + segment;
        const uint32_t index = load_packed_bits(
            indices, index_linear * 6, 6, indices_nbytes);
        return {load_nepq0_s_vec8(codebook, state, index), 0, true};
    }
    return load_nvq_vec8<FORMAT>(
        indices, indices_nbytes, aux, aux_nbytes, codebook,
        row, segment, group, ng, nvec, nsign, sign_mode, state);
}

template <int FORMAT>
__device__ __forceinline__ float dot_nvq_vec8(
    const NvqVec8Values<FORMAT> & weight,
    const int8_t * activation) {
    if (!weight.valid) return 0.0f;
    const int2 x = *reinterpret_cast<const int2 *>(activation);
    const int dot = __dp4a(weight.values.y, x.y, __dp4a(weight.values.x, x.x, 0));
    if constexpr (FORMAT == kNvq1L) {
        const int sum = __dp4a(0x01010101, x.y, __dp4a(0x01010101, x.x, 0));
        return static_cast<float>(dot) + 0.125f * static_cast<float>(weight.delta * sum);
    }
    if constexpr (FORMAT == kNvq1S) {
        const int sum = __dp4a(0x01010101, x.y, __dp4a(0x01010101, x.x, 0));
        return static_cast<float>(dot) + 0.15625f * static_cast<float>(weight.delta * sum);
    }
    return static_cast<float>(dot);
}

template <int FORMAT>
__device__ __forceinline__ float nvq_vec8_dot(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const int8_t * codebook,
    const int8_t * activation,
    int row,
    int segment,
    int group,
    int ng,
    int nvec,
    int nsign,
    int sign_mode,
    uint32_t state) {
    return dot_nvq_vec8<FORMAT>(
        load_nvq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, segment, group, ng, nvec, nsign, sign_mode, state),
        activation);
}

template <int FORMAT>
__device__ __forceinline__ float nvq_vec8_scaled_fma(
    const NvqDeviceWeight & weight,
    const int8_t * activation,
    const float * activation_scale,
    int row,
    int segment,
    float acc) {
    const int group = segment / 3;
    const int k = group * kGroupSize + (segment - group * 3) * 8;
    const int64_t sub_linear = static_cast<int64_t>(row) * weight.ng + group;
    const uint32_t sub = load_packed_bits(
        weight.sub_scale, sub_linear * weight.sub_bits,
        weight.sub_bits, weight.sub_scale_nbytes);
    const float dot = nvq_vec8_dot<FORMAT>(
        weight.indices, weight.indices_nbytes,
        weight.aux, weight.aux_nbytes,
        weight.codebook, activation + k,
        row, segment, group, weight.ng, weight.nvec, weight.nsign,
        weight.sign_mode, sub);
    const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
        ? weight.neuron_scale[row] * static_cast<float>(sub)
        : format_scale<FORMAT>(weight.neuron_scale[row], sub, weight.codebook);
    return fmaf(weight_scale * activation_scale[group], dot, acc);
}

template <bool EXEC_LAYOUT = false, bool JSC = false>
__device__ __forceinline__ float nvq2_vec8_scaled_fma_s4(
    const NvqDeviceWeight & weight,
    const int8_t * activation,
    const float * activation_scale,
    int64_t index_row,
    int64_t sign_row,
    int64_t sub_row,
    float anchor,
    int segment,
    float acc) {
    const int group = segment / 3;
    const int k = group * kGroupSize + (segment - group * 3) * 8;
    uint32_t index;
    uint32_t mask8;
    if constexpr (EXEC_LAYOUT) {
        const uint16_t metadata =
            reinterpret_cast<const uint16_t *>(weight.indices)[index_row + segment];
        index = metadata & 0xffu;
        mask8 = metadata >> 8;
    } else {
        index = weight.indices[index_row + segment];
        const int64_t sign_linear = sign_row + segment;
        const int64_t sign_bit = sign_linear * 7;
        const int64_t sign_byte = sign_bit >> 3;
        const int sign_shift = static_cast<int>(sign_bit & 7);
        uint32_t sign_word = weight.aux[sign_byte];
        if (sign_byte + 1 < weight.aux_nbytes) {
            sign_word |= static_cast<uint32_t>(weight.aux[sign_byte + 1]) << 8;
        }
        const uint32_t mask7 = (sign_word >> sign_shift) & 0x7fu;
        const int last = parity7(mask7) ^
            (weight.sign_mode ? ((index >> 7) & 1u) : 0u);
        mask8 = mask7 | (static_cast<uint32_t>(last) << 7);
    }
    const uint32_t sub = load_packed_4(weight.sub_scale, sub_row + group);
    const int8_t * bank = JSC
        ? active_codebook<kNvq2Jsc>(weight.codebook, sub)
        : weight.codebook;
    const int2 code = reinterpret_cast<const int2 *>(bank)[index];
    const int2 values = EXEC_LAYOUT ? apply_sign8_exec(code, mask8) : apply_sign8(code, mask8);
    const int2 x = *reinterpret_cast<const int2 *>(activation + k);
    const int dot = __dp4a(values.y, x.y, __dp4a(values.x, x.x, 0));
    const float weight_scale = JSC
        ? format_scale<kNvq2Jsc>(anchor, sub, weight.codebook)
        : anchor * static_cast<float>(sub);
    return fmaf(weight_scale * activation_scale[group], static_cast<float>(dot), acc);
}

__device__ __forceinline__ uint32_t load_nvq_sign7(
    const uint8_t * aux,
    int64_t aux_nbytes,
    int64_t sign_linear) {
    const int64_t sign_bit = sign_linear * 7;
    const int64_t sign_byte = sign_bit >> 3;
    const int sign_shift = static_cast<int>(sign_bit & 7);
    uint32_t word = aux[sign_byte];
    if (sign_byte + 1 < aux_nbytes) {
        word |= static_cast<uint32_t>(aux[sign_byte + 1]) << 8;
    }
    return (word >> sign_shift) & 0x7fu;
}

template <bool JSC = false, bool ANALYTIC2 = false>
__device__ __forceinline__ float nvq3_vec8_scaled_fma_s4(
    const NvqDeviceWeight & weight,
    const int8_t * activation,
    const float * activation_scale,
    int64_t index_row,
    int64_t sign_row,
    int64_t sub_row,
    float anchor,
    int segment,
    float acc) {
    const int group = segment / 3;
    const int k = group * kGroupSize + (segment - group * 3) * 8;
    const int vector4 = segment * 2;
    const uint32_t index0 = weight.indices[index_row + vector4];
    const uint32_t index1 = vector4 + 1 < weight.nvec
        ? weight.indices[index_row + vector4 + 1]
        : 0;
    const uint32_t mask7 = load_nvq_sign7(weight.aux, weight.aux_nbytes, sign_row + segment);
    const uint32_t mask8 = mask7 | (static_cast<uint32_t>(parity7(mask7)) << 7);
    const uint32_t sub = load_packed_4(weight.sub_scale, sub_row + group);
    const uint8_t * metadata = reinterpret_cast<const uint8_t *>(weight.codebook);
    const int8_t * bank = weight.codebook;
    if constexpr (JSC) {
        const uint32_t bank_id = ANALYTIC2
            ? sub & 1u
            : metadata[kJscBankMapOffset + sub];
        bank = weight.codebook + kJscMetadataBytes + bank_id * kJscD4CodebookBytes;
    }
    const int2 values = apply_sign8(
        make_int2(
            reinterpret_cast<const int *>(bank)[index0],
            reinterpret_cast<const int *>(bank)[index1]),
        mask8);
    const int2 x = *reinterpret_cast<const int2 *>(activation + k);
    const int dot = __dp4a(values.y, x.y, __dp4a(values.x, x.x, 0));
    float weight_scale;
    if constexpr (ANALYTIC2) {
        weight_scale = anchor * static_cast<float>((sub >> 1) + 1);
    } else if constexpr (JSC) {
        const __half * lut = reinterpret_cast<const __half *>(metadata + kJscLutOffset);
        weight_scale = anchor * __half2float(lut[sub]);
    } else {
        weight_scale = anchor * static_cast<float>(sub);
    }
    return fmaf(weight_scale * activation_scale[group], static_cast<float>(dot), acc);
}

template <bool EXEC_LAYOUT = false, bool JSC = false>
__device__ __forceinline__ void nvq2_vec8_pair_scaled_fma_s4(
    const NvqDeviceWeight & gate,
    const NvqDeviceWeight & up,
    const int8_t * activation,
    const float * activation_scale,
    int64_t index_row,
    int64_t sign_row,
    int64_t sub_row,
    float gate_anchor,
    float up_anchor,
    int segment,
    float & gate_acc,
    float & up_acc) {
    const int group = segment / 3;
    const int k = group * kGroupSize + (segment - group * 3) * 8;
    const int2 x = *reinterpret_cast<const int2 *>(activation + k);
    const float x_scale = activation_scale[group];
    uint32_t gate_index;
    uint32_t up_index;
    uint32_t gate_mask8;
    uint32_t up_mask8;
    if constexpr (EXEC_LAYOUT) {
        const uint16_t gate_metadata =
            reinterpret_cast<const uint16_t *>(gate.indices)[index_row + segment];
        const uint16_t up_metadata =
            reinterpret_cast<const uint16_t *>(up.indices)[index_row + segment];
        gate_index = gate_metadata & 0xffu;
        up_index = up_metadata & 0xffu;
        gate_mask8 = gate_metadata >> 8;
        up_mask8 = up_metadata >> 8;
    } else {
        const int64_t sign_linear = sign_row + segment;
        const int64_t sign_bit = sign_linear * 7;
        const int64_t sign_byte = sign_bit >> 3;
        const int sign_shift = static_cast<int>(sign_bit & 7);
        uint32_t gate_sign_word = gate.aux[sign_byte];
        uint32_t up_sign_word = up.aux[sign_byte];
        if (sign_byte + 1 < gate.aux_nbytes) {
            gate_sign_word |= static_cast<uint32_t>(gate.aux[sign_byte + 1]) << 8;
            up_sign_word |= static_cast<uint32_t>(up.aux[sign_byte + 1]) << 8;
        }
        const uint32_t gate_mask7 = (gate_sign_word >> sign_shift) & 0x7fu;
        const uint32_t up_mask7 = (up_sign_word >> sign_shift) & 0x7fu;
        gate_index = gate.indices[index_row + segment];
        up_index = up.indices[index_row + segment];
        const int gate_last = parity7(gate_mask7) ^
            (gate.sign_mode ? ((gate_index >> 7) & 1u) : 0u);
        const int up_last = parity7(up_mask7) ^
            (up.sign_mode ? ((up_index >> 7) & 1u) : 0u);
        gate_mask8 = gate_mask7 | (static_cast<uint32_t>(gate_last) << 7);
        up_mask8 = up_mask7 | (static_cast<uint32_t>(up_last) << 7);
    }
    const uint32_t gate_sub = load_packed_4(gate.sub_scale, sub_row + group);
    const uint32_t up_sub = load_packed_4(up.sub_scale, sub_row + group);
    const int8_t * gate_bank = JSC
        ? active_codebook<kNvq2Jsc>(gate.codebook, gate_sub)
        : gate.codebook;
    const int8_t * up_bank = JSC
        ? active_codebook<kNvq2Jsc>(up.codebook, up_sub)
        : up.codebook;
    const int2 gate_code = reinterpret_cast<const int2 *>(gate_bank)[gate_index];
    const int2 up_code = reinterpret_cast<const int2 *>(up_bank)[up_index];
    const int2 gate_values = EXEC_LAYOUT
        ? apply_sign8_exec(gate_code, gate_mask8)
        : apply_sign8(gate_code, gate_mask8);
    const int2 up_values = EXEC_LAYOUT
        ? apply_sign8_exec(up_code, up_mask8)
        : apply_sign8(up_code, up_mask8);
    const int gate_dot = __dp4a(
        gate_values.y, x.y, __dp4a(gate_values.x, x.x, 0));
    const int up_dot = __dp4a(
        up_values.y, x.y, __dp4a(up_values.x, x.x, 0));
    const float gate_scale = JSC
        ? format_scale<kNvq2Jsc>(gate_anchor, gate_sub, gate.codebook)
        : gate_anchor * static_cast<float>(gate_sub);
    const float up_scale = JSC
        ? format_scale<kNvq2Jsc>(up_anchor, up_sub, up.codebook)
        : up_anchor * static_cast<float>(up_sub);
    gate_acc = fmaf(gate_scale * x_scale, static_cast<float>(gate_dot), gate_acc);
    up_acc = fmaf(up_scale * x_scale, static_cast<float>(up_dot), up_acc);
}

// llama.cpp-style MMVQ schedule: one output row per block, multiple warps
// split K, and each lane consumes one complete 8-value VQ segment.
template <int FORMAT, int NWARPS, bool FAST_S4 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq_gemv_m1_vec8_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    float acc = 0.0f;
    [[maybe_unused]] const NvqDeviceWeight fast_weight{
        indices, indices_nbytes, aux, aux_nbytes,
        sub_scale, sub_scale_nbytes, neuron_scale, codebook,
        0, N, ng, nvec, nsign, 4, sign_mode};
    [[maybe_unused]] const int64_t fast_index_row = static_cast<int64_t>(row) * nvec;
    [[maybe_unused]] const int64_t fast_sign_row = static_cast<int64_t>(row) * nsign;
    [[maybe_unused]] const int64_t fast_sub_row = static_cast<int64_t>(row) * ng;
    [[maybe_unused]] const float fast_anchor = neuron_scale[row];
    for (int segment = warp * 32 + lane;
         segment < nsign;
         segment += NWARPS * 32) {
        if constexpr (FAST_S4) {
            if constexpr (
                FORMAT == kNvq2 || FORMAT == kNvq2Exec ||
                FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec) {
                acc = nvq2_vec8_scaled_fma_s4<
                    FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec,
                    FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec>(
                    fast_weight, qx, xscale,
                    fast_index_row, fast_sign_row, fast_sub_row, fast_anchor,
                    segment, acc);
            } else {
                acc = nvq3_vec8_scaled_fma_s4<
                    FORMAT == kNvq3Jsc || FORMAT == kNvq3Jsc2,
                    FORMAT == kNvq3Jsc2>(
                    fast_weight, qx, xscale,
                    fast_index_row, fast_sign_row, fast_sub_row, fast_anchor,
                    segment, acc);
            }
        } else {
            const int group = segment / 3;
            const int segment_in_group = segment - group * 3;
            const int k = group * kGroupSize + segment_in_group * 8;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            const uint32_t sub = load_packed_bits(
                sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
            const float dot = nvq_vec8_dot<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, codebook, qx + k,
                row, segment, group, ng, nvec, nsign, sign_mode, sub);
            const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                ? neuron_scale[row] * static_cast<float>(sub)
                : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
            acc = fmaf(weight_scale * xscale[group], dot, acc);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float partial[NWARPS];
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) output[row] = __float2half(acc);
    }
}

template <int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq3j2_gemv_m1_group_kernel(
    const uint8_t * indices,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    const float * neuron_scale,
    const int8_t * metadata,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int N,
    int ng,
    int nvec,
    int nsign) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    const int64_t index_row = static_cast<int64_t>(row) * nvec;
    const int64_t sign_row = static_cast<int64_t>(row) * nsign;
    const int64_t sub_row = static_cast<int64_t>(row) * ng;
    const float anchor = neuron_scale[row];
    float acc = 0.0f;
    for (int group = warp * 32 + lane; group < ng; group += NWARPS * 32) {
        const uint32_t state = load_packed_4(sub_scale, sub_row + group);
        const int8_t * bank = metadata + kJscMetadataBytes
            + (state & 1u) * kJscD4CodebookBytes;
        int group_dot = 0;
#pragma unroll
        for (int local = 0; local < 3; ++local) {
            const int segment = group * 3 + local;
            if (segment >= nsign) continue;
            const int vector4 = segment * 2;
            const uint32_t index0 = indices[index_row + vector4];
            const uint32_t index1 = vector4 + 1 < nvec
                ? indices[index_row + vector4 + 1]
                : 0;
            const uint32_t mask7 = load_nvq_sign7(aux, aux_nbytes, sign_row + segment);
            const uint32_t mask8 = mask7
                | (static_cast<uint32_t>(parity7(mask7)) << 7);
            const int2 values = apply_sign8(
                make_int2(
                    reinterpret_cast<const int *>(bank)[index0],
                    reinterpret_cast<const int *>(bank)[index1]),
                mask8);
            const int2 x = *reinterpret_cast<const int2 *>(
                qx + group * kGroupSize + local * 8);
            group_dot = __dp4a(
                values.y, x.y, __dp4a(values.x, x.x, group_dot));
        }
        const float weight_scale = anchor * static_cast<float>((state >> 1) + 1);
        acc = fmaf(
            weight_scale * xscale[group], static_cast<float>(group_dot), acc);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float partial[NWARPS];
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) output[row] = __float2half(acc);
    }
}

__device__ __forceinline__ uint32_t extract_group_exec96_bits(
    const uint32_t (&words)[3], int bit, int bits) {
    const int word = bit >> 5;
    const int shift = bit & 31;
    uint32_t value = words[word] >> shift;
    if (shift + bits > 32) value |= words[word + 1] << (32 - shift);
    return value & ((1u << bits) - 1u);
}

__device__ __forceinline__ int aligned_group_d4_dot_words(
    const NvqDeviceWeight & weight,
    int group,
    const int8_t * qx,
    const uint32_t (&words)[3],
    uint32_t & state) {
    state = extract_group_exec96_bits(words, 84, 4);
    const int8_t * bank = active_codebook<kNvq3JscLGroupExec>(
        weight.codebook, state);
    const uint32_t index00 = extract_group_exec96_bits(words, 0, 10);
    const uint32_t index01 = extract_group_exec96_bits(words, 10, 10);
    const uint32_t index10 = extract_group_exec96_bits(words, 20, 10);
    const uint32_t index11 = extract_group_exec96_bits(words, 30, 10);
    const uint32_t index20 = extract_group_exec96_bits(words, 40, 10);
    const uint32_t index21 = extract_group_exec96_bits(words, 50, 10);
    const uint32_t mask0 = extract_group_exec96_bits(words, 60, 8);
    const uint32_t mask1 = extract_group_exec96_bits(words, 68, 8);
    const uint32_t mask2 = extract_group_exec96_bits(words, 76, 8);
    const int2 values0 = apply_sign8(
        make_int2(
            reinterpret_cast<const int *>(bank)[index00],
            reinterpret_cast<const int *>(bank)[index01]),
        mask0);
    const int2 values1 = apply_sign8(
        make_int2(
            reinterpret_cast<const int *>(bank)[index10],
            reinterpret_cast<const int *>(bank)[index11]),
        mask1);
    const int2 values2 = apply_sign8(
        make_int2(
            reinterpret_cast<const int *>(bank)[index20],
            reinterpret_cast<const int *>(bank)[index21]),
        mask2);
    const int8_t * activation = qx + group * kGroupSize;
    const int2 x0 = *reinterpret_cast<const int2 *>(activation);
    const int2 x1 = *reinterpret_cast<const int2 *>(activation + 8);
    const int2 x2 = *reinterpret_cast<const int2 *>(activation + 16);
    const int dot0 = __dp4a(values0.y, x0.y, __dp4a(values0.x, x0.x, 0));
    const int dot1 = __dp4a(values1.y, x1.y, __dp4a(values1.x, x1.x, 0));
    const int dot2 = __dp4a(values2.y, x2.y, __dp4a(values2.x, x2.x, 0));
    return dot0 + dot1 + dot2;
}

template <int FORMAT>
__device__ __forceinline__ int aligned_group_dot(
    const NvqDeviceWeight & weight,
    int row,
    int group,
    const int8_t * qx,
    uint32_t & state) {
    static_assert(
        FORMAT == kNvq2JscXLGroupExec ||
        FORMAT == kNvq3JscLGroupExec);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        const uint64_t metadata = load_group_exec64(
            weight.indices, row, group, weight.ng);
        state = static_cast<uint32_t>(metadata >> 60);
        const int8_t * bank = active_codebook<FORMAT>(weight.codebook, state);
        int dot = 0;
#pragma unroll
        for (int local = 0; local < 3; ++local) {
            const int segment = group * 3 + local;
            if (segment >= weight.nsign) continue;
            const uint32_t segment_metadata =
                (metadata >> (local * 20)) & 0xfffffu;
            const uint32_t index = segment_metadata & 0xfffu;
            const uint32_t mask8 = segment_metadata >> 12;
            const int2 values = apply_sign8(
                reinterpret_cast<const int2 *>(bank)[index], mask8);
            const int2 x = *reinterpret_cast<const int2 *>(
                qx + group * kGroupSize + local * 8);
            dot = __dp4a(values.y, x.y, __dp4a(values.x, x.x, dot));
        }
        return dot;
    } else {
        uint32_t words[3];
        load_group_exec96_words(
            weight.indices, row, group, weight.ng, words);
        return aligned_group_d4_dot_words(weight, group, qx, words, state);
    }
}

template <int FORMAT>
__device__ __forceinline__ void aligned_group_pair_dot(
    const NvqDeviceWeight & gate,
    const NvqDeviceWeight & up,
    int row,
    int group,
    const int8_t * qx,
    uint32_t & gate_state,
    uint32_t & up_state,
    int & gate_dot,
    int & up_dot) {
    static_assert(
        FORMAT == kNvq2JscXLGroupExec ||
        FORMAT == kNvq3JscLGroupExec);
    gate_dot = 0;
    up_dot = 0;
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        const uint64_t gate_metadata = load_group_exec64(
            gate.indices, row, group, gate.ng);
        const uint64_t up_metadata = load_group_exec64(
            up.indices, row, group, up.ng);
        gate_state = static_cast<uint32_t>(gate_metadata >> 60);
        up_state = static_cast<uint32_t>(up_metadata >> 60);
        const int8_t * gate_bank = active_codebook<FORMAT>(
            gate.codebook, gate_state);
        const int8_t * up_bank = active_codebook<FORMAT>(
            up.codebook, up_state);
#pragma unroll
        for (int local = 0; local < 3; ++local) {
            const int segment = group * 3 + local;
            if (segment >= gate.nsign) continue;
            const uint32_t gate_segment =
                (gate_metadata >> (local * 20)) & 0xfffffu;
            const uint32_t up_segment =
                (up_metadata >> (local * 20)) & 0xfffffu;
            const uint32_t gate_index = gate_segment & 0xfffu;
            const uint32_t up_index = up_segment & 0xfffu;
            const uint32_t gate_mask = gate_segment >> 12;
            const uint32_t up_mask = up_segment >> 12;
            const int2 gate_values = apply_sign8(
                reinterpret_cast<const int2 *>(gate_bank)[gate_index],
                gate_mask);
            const int2 up_values = apply_sign8(
                reinterpret_cast<const int2 *>(up_bank)[up_index],
                up_mask);
            const int2 x = *reinterpret_cast<const int2 *>(
                qx + group * kGroupSize + local * 8);
            gate_dot = __dp4a(
                gate_values.y, x.y,
                __dp4a(gate_values.x, x.x, gate_dot));
            up_dot = __dp4a(
                up_values.y, x.y,
                __dp4a(up_values.x, x.x, up_dot));
        }
    } else {
        uint32_t gate_words[3];
        uint32_t up_words[3];
        load_group_exec96_words(
            gate.indices, row, group, gate.ng, gate_words);
        load_group_exec96_words(
            up.indices, row, group, up.ng, up_words);
        gate_state = extract_group_exec96_bits(gate_words, 84, 4);
        up_state = extract_group_exec96_bits(up_words, 84, 4);
        const int8_t * gate_bank = active_codebook<FORMAT>(
            gate.codebook, gate_state);
        const int8_t * up_bank = active_codebook<FORMAT>(
            up.codebook, up_state);
#pragma unroll
        for (int local = 0; local < 3; ++local) {
            const int segment = group * 3 + local;
            if (segment >= gate.nsign) continue;
            const int vector4 = segment * 2;
            const uint32_t gate_index0 = extract_group_exec96_bits(
                gate_words, local * 20, 10);
            const uint32_t gate_index1 = vector4 + 1 < gate.nvec
                ? extract_group_exec96_bits(
                    gate_words, local * 20 + 10, 10)
                : 0;
            const uint32_t up_index0 = extract_group_exec96_bits(
                up_words, local * 20, 10);
            const uint32_t up_index1 = vector4 + 1 < up.nvec
                ? extract_group_exec96_bits(
                    up_words, local * 20 + 10, 10)
                : 0;
            const uint32_t gate_mask = extract_group_exec96_bits(
                gate_words, 60 + local * 8, 8);
            const uint32_t up_mask = extract_group_exec96_bits(
                up_words, 60 + local * 8, 8);
            const int2 gate_values = apply_sign8(
                make_int2(
                    reinterpret_cast<const int *>(gate_bank)[gate_index0],
                    reinterpret_cast<const int *>(gate_bank)[gate_index1]),
                gate_mask);
            const int2 up_values = apply_sign8(
                make_int2(
                    reinterpret_cast<const int *>(up_bank)[up_index0],
                    reinterpret_cast<const int *>(up_bank)[up_index1]),
                up_mask);
            const int2 x = *reinterpret_cast<const int2 *>(
                qx + group * kGroupSize + local * 8);
            gate_dot = __dp4a(
                gate_values.y, x.y,
                __dp4a(gate_values.x, x.x, gate_dot));
            up_dot = __dp4a(
                up_values.y, x.y,
                __dp4a(up_values.x, x.x, up_dot));
        }
    }
}

template <
    int FORMAT, int NWARPS, int ROWS_PER_BLOCK = 1,
    bool ADD_RESIDUAL = false>
__global__ void __launch_bounds__(NWARPS * ROWS_PER_BLOCK * 32, 1)
aligned_group_gemv_m1_kernel(
    NvqDeviceWeight weight,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    const __half * residual) {
    const int row_slot = threadIdx.y / NWARPS;
    const int warp = threadIdx.y - row_slot * NWARPS;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_slot;
    const int lane = threadIdx.x;
    const bool valid = row < weight.N;

    const float anchor = valid ? weight.neuron_scale[row] : 0.0f;
    float acc = 0.0f;
    if (valid) {
        for (int group = warp * 32 + lane;
             group < weight.ng;
             group += NWARPS * 32) {
            uint32_t state = 0;
            const int dot = aligned_group_dot<FORMAT>(
                weight, row, group, qx, state);
            acc = fmaf(
                format_scale<FORMAT>(anchor, state, weight.codebook) *
                    xscale[group],
                static_cast<float>(dot), acc);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];
    if (lane == 0) partial[row_slot][warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[row_slot][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0 && valid) {
            const __half projected = __float2half(acc);
            if constexpr (ADD_RESIDUAL) {
                output[row] = __hadd(residual[row], projected);
            } else {
                output[row] = projected;
            }
        }
    }
}

template <int FORMAT, int NWARPS, int ROWS_PER_BLOCK = 1>
__global__ void __launch_bounds__(NWARPS * ROWS_PER_BLOCK * 32, 1)
aligned_group_multi2_m1_kernel(
    NvqDeviceWeight first,
    NvqDeviceWeight second,
    const int8_t * qx,
    const float * xscale,
    __half * output) {
    const int row_slot = threadIdx.y / NWARPS;
    const int warp = threadIdx.y - row_slot * NWARPS;
    const int combined_row = blockIdx.x * ROWS_PER_BLOCK + row_slot;
    const int total_rows = first.N + second.N;
    const bool valid = combined_row < total_rows;
    const bool use_second = combined_row >= first.N;
    const NvqDeviceWeight weight = use_second ? second : first;
    const int row = use_second ? combined_row - first.N : combined_row;
    const int lane = threadIdx.x;

    const float anchor = valid ? weight.neuron_scale[row] : 0.0f;
    float acc = 0.0f;
    if (valid) {
        for (int group = warp * 32 + lane;
             group < weight.ng;
             group += NWARPS * 32) {
            uint32_t state = 0;
            const int dot = aligned_group_dot<FORMAT>(weight, row, group, qx, state);
            acc = fmaf(
                format_scale<FORMAT>(anchor, state, weight.codebook) * xscale[group],
                static_cast<float>(dot), acc);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];
    if (lane == 0) partial[row_slot][warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[row_slot][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0 && valid) output[combined_row] = __float2half(acc);
    }
}

template <int FORMAT>
__device__ __forceinline__ int aligned_group_dot_preloaded(
    const NvqDeviceWeight & weight,
    const int8_t * codebook,
    int row,
    int group,
    const int2 & x0,
    const int2 & x1,
    const int2 & x2,
    uint32_t & state) {
    static_assert(
        FORMAT == kNvq2JscXLGroupExec ||
        FORMAT == kNvq3JscLGroupExec);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        const uint64_t metadata = load_group_exec64(
            weight.indices, row, group, weight.ng);
        state = static_cast<uint32_t>(metadata >> 60);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        const uint32_t segment0 = static_cast<uint32_t>(metadata) & 0xfffffu;
        const uint32_t segment1 = static_cast<uint32_t>(metadata >> 20) & 0xfffffu;
        const uint32_t segment2 = static_cast<uint32_t>(metadata >> 40) & 0xfffffu;
        const int2 values0 = apply_sign8(
            reinterpret_cast<const int2 *>(bank)[segment0 & 0xfffu],
            segment0 >> 12);
        const int2 values1 = apply_sign8(
            reinterpret_cast<const int2 *>(bank)[segment1 & 0xfffu],
            segment1 >> 12);
        const int2 values2 = apply_sign8(
            reinterpret_cast<const int2 *>(bank)[segment2 & 0xfffu],
            segment2 >> 12);
        int dot = __dp4a(values0.y, x0.y, __dp4a(values0.x, x0.x, 0));
        dot = __dp4a(values1.y, x1.y, __dp4a(values1.x, x1.x, dot));
        return __dp4a(values2.y, x2.y, __dp4a(values2.x, x2.x, dot));
    } else {
        uint32_t words[3];
        load_group_exec96_words(
            weight.indices, row, group, weight.ng, words);
        state = extract_group_exec96_bits(words, 84, 4);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        const uint32_t index00 = extract_group_exec96_bits(words, 0, 10);
        const uint32_t index01 = extract_group_exec96_bits(words, 10, 10);
        const uint32_t index10 = extract_group_exec96_bits(words, 20, 10);
        const uint32_t index11 = extract_group_exec96_bits(words, 30, 10);
        const uint32_t index20 = extract_group_exec96_bits(words, 40, 10);
        const uint32_t index21 = extract_group_exec96_bits(words, 50, 10);
        const int2 values0 = apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index00],
                reinterpret_cast<const int *>(bank)[index01]),
            extract_group_exec96_bits(words, 60, 8));
        const int2 values1 = apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index10],
                reinterpret_cast<const int *>(bank)[index11]),
            extract_group_exec96_bits(words, 68, 8));
        const int2 values2 = apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index20],
                reinterpret_cast<const int *>(bank)[index21]),
            extract_group_exec96_bits(words, 76, 8));
        const int dot0 = __dp4a(values0.y, x0.y, __dp4a(values0.x, x0.x, 0));
        const int dot1 = __dp4a(values1.y, x1.y, __dp4a(values1.x, x1.x, 0));
        const int dot2 = __dp4a(values2.y, x2.y, __dp4a(values2.x, x2.x, 0));
        return dot0 + dot1 + dot2;
    }
}

__device__ __forceinline__ int aligned_group_dot_preloaded_e8_stage3_metadata(
    const int8_t * global_codebook,
    const int8_t * shared_codebook,
    bool table_is_staged,
    uint64_t metadata,
    const int2 & x0,
    const int2 & x1,
    const int2 & x2,
    uint32_t & state) {
    state = static_cast<uint32_t>(metadata >> 60);
    const uint32_t bank_id = state & 3u;
    const bool use_staged_bank = table_is_staged && bank_id < 3;
    const int8_t * bank = use_staged_bank
        ? shared_codebook + bank_id * kJscE8Codebook4096Bytes
        : global_codebook + kJscMetadataBytes +
            bank_id * kJscE8Codebook4096Bytes;
    const uint32_t segment0 = static_cast<uint32_t>(metadata) & 0xfffffu;
    const uint32_t segment1 = static_cast<uint32_t>(metadata >> 20) & 0xfffffu;
    const uint32_t segment2 = static_cast<uint32_t>(metadata >> 40) & 0xfffffu;
    const uint32_t index0 = segment0 & 0xfffu;
    const uint32_t index1 = segment1 & 0xfffu;
    const uint32_t index2 = segment2 & 0xfffu;
    int2 raw0;
    int2 raw1;
    int2 raw2;
    if (use_staged_bank) {
        const int * plane = reinterpret_cast<const int *>(bank);
        constexpr int plane_entries = 4096;
        raw0 = make_int2(plane[index0], plane[plane_entries + index0]);
        raw1 = make_int2(plane[index1], plane[plane_entries + index1]);
        raw2 = make_int2(plane[index2], plane[plane_entries + index2]);
    } else if (table_is_staged && bank_id == 3) {
        const int * partial = reinterpret_cast<const int *>(
            shared_codebook + 3 * kJscE8Codebook4096Bytes);
        const int2 * global = reinterpret_cast<const int2 *>(bank);
        raw0 = index0 < kJscE8PartialEntries
            ? make_int2(
                partial[index0], partial[kJscE8PartialEntries + index0])
            : global[index0];
        raw1 = index1 < kJscE8PartialEntries
            ? make_int2(
                partial[index1], partial[kJscE8PartialEntries + index1])
            : global[index1];
        raw2 = index2 < kJscE8PartialEntries
            ? make_int2(
                partial[index2], partial[kJscE8PartialEntries + index2])
            : global[index2];
    } else {
        raw0 = reinterpret_cast<const int2 *>(bank)[index0];
        raw1 = reinterpret_cast<const int2 *>(bank)[index1];
        raw2 = reinterpret_cast<const int2 *>(bank)[index2];
    }
    const int2 values0 = apply_sign8(raw0, segment0 >> 12);
    const int2 values1 = apply_sign8(raw1, segment1 >> 12);
    const int2 values2 = apply_sign8(raw2, segment2 >> 12);
    int dot = __dp4a(values0.y, x0.y, __dp4a(values0.x, x0.x, 0));
    dot = __dp4a(values1.y, x1.y, __dp4a(values1.x, x1.x, dot));
    return __dp4a(values2.y, x2.y, __dp4a(values2.x, x2.x, dot));
}

__device__ __forceinline__ int aligned_group_dot_preloaded_e8_stage3(
    const NvqDeviceWeight & weight,
    const int8_t * global_codebook,
    const int8_t * shared_codebook,
    bool table_is_staged,
    int row,
    int group,
    const int2 & x0,
    const int2 & x1,
    const int2 & x2,
    uint32_t & state) {
    const uint64_t metadata = load_group_exec64(
        weight.indices, row, group, weight.ng);
    return aligned_group_dot_preloaded_e8_stage3_metadata(
        global_codebook, shared_codebook, table_is_staged, metadata,
        x0, x1, x2, state);
}

// TPQ5-style decode geometry: every warp owns several output rows and reuses
// the same quantized activation values across those rows.  The compact D4
// tables fit in shared memory and are staged once per CTA; the 128 KiB E8
// tables remain in the read-only cache path.
template <
    int FORMAT, int NWARPS, int ROWS_PER_WARP, bool MULTI2,
    bool STAGE_E8_THREE_BANKS = false, bool ADD_RESIDUAL = false>
__global__ void __launch_bounds__(NWARPS * 32, 1)
aligned_group_row_tiled_m1_kernel(
    NvqDeviceWeight first,
    NvqDeviceWeight second,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    const __half * residual) {
    static_assert(!ADD_RESIDUAL || !MULTI2);
    static_assert(NWARPS == 16 || NWARPS == 32);
    static_assert(
        ROWS_PER_WARP == 1 || ROWS_PER_WARP == 2 || ROWS_PER_WARP == 4);
    static_assert(
        !STAGE_E8_THREE_BANKS ||
        (FORMAT == kNvq2JscXLGroupExec && NWARPS == 32 &&
         (ROWS_PER_WARP == 1 || ROWS_PER_WARP == 4)));
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int linear_thread = warp * 32 + lane;
    const int block_threads = NWARPS * 32;
    const int k_pad = first.ng * kGroupSize;
    const int scale_offset = (k_pad + 3) & ~3;
    const int codebook_offset = scale_offset + first.ng * sizeof(float);
    extern __shared__ uint8_t shared_storage[];
    int8_t * shared_qx = reinterpret_cast<int8_t *>(shared_storage);
    float * shared_xscale = reinterpret_cast<float *>(
        shared_storage + scale_offset);

    if constexpr (!STAGE_E8_THREE_BANKS) {
        for (int item = linear_thread; item < k_pad; item += block_threads) {
            shared_qx[item] = qx[item];
        }
        for (int item = linear_thread; item < first.ng; item += block_threads) {
            shared_xscale[item] = xscale[item];
        }
    }

    const int8_t * first_codebook = first.codebook;
    const int8_t * second_codebook = second.codebook;
    const int rows_per_block = NWARPS * ROWS_PER_WARP;
    const int block_row_base = blockIdx.x * rows_per_block;
    const bool stage_second_table = MULTI2 && block_row_base >= first.N;
    const int8_t * staged_global_codebook = stage_second_table
        ? second.codebook
        : first.codebook;
    constexpr int stage_min_rows = ROWS_PER_WARP == 1 ? 4096 : 12288;
    const bool stage_active_table = STAGE_E8_THREE_BANKS &&
        (stage_second_table
            ? second.N >= stage_min_rows
            : first.N >= stage_min_rows);
    const int8_t * shared_e8_codebook = nullptr;
    if constexpr (STAGE_E8_THREE_BANKS) {
        int8_t * staged = reinterpret_cast<int8_t *>(shared_storage);
        constexpr int entries_per_bank = 4096;
        constexpr int staged_entries = 3 * entries_per_bank;
        if (stage_active_table) {
            for (int item = linear_thread;
                 item < staged_entries;
                 item += block_threads) {
                const int bank = item / entries_per_bank;
                const int entry = item - bank * entries_per_bank;
                const int2 value = reinterpret_cast<const int2 *>(
                    staged_global_codebook + kJscMetadataBytes +
                    bank * kJscE8Codebook4096Bytes)[entry];
                int * destination = reinterpret_cast<int *>(
                    staged + bank * kJscE8Codebook4096Bytes);
                destination[entry] = value.x;
                destination[entries_per_bank + entry] = value.y;
            }
            int * partial = reinterpret_cast<int *>(
                staged + 3 * kJscE8Codebook4096Bytes);
            for (int entry = linear_thread;
                 entry < kJscE8PartialEntries;
                 entry += block_threads) {
                const int2 value = reinterpret_cast<const int2 *>(
                    staged_global_codebook + kJscMetadataBytes +
                    3 * kJscE8Codebook4096Bytes)[entry];
                partial[entry] = value.x;
                partial[kJscE8PartialEntries + entry] = value.y;
            }
        }
        shared_e8_codebook = staged;
    }
    if constexpr (FORMAT == kNvq3JscLGroupExec) {
        int8_t * shared_first_codebook = reinterpret_cast<int8_t *>(
            shared_storage + codebook_offset);
        for (int item = linear_thread;
             item < first.codebook_nbytes;
             item += block_threads) {
            shared_first_codebook[item] = first.codebook[item];
        }
        first_codebook = shared_first_codebook;
        if constexpr (MULTI2) {
            int8_t * shared_second_codebook =
                shared_first_codebook + first.codebook_nbytes;
            for (int item = linear_thread;
                 item < second.codebook_nbytes;
                 item += block_threads) {
                shared_second_codebook[item] = second.codebook[item];
            }
            second_codebook = shared_second_codebook;
        }
    }
    __syncthreads();

    const int total_rows = MULTI2 ? first.N + second.N : first.N;
    const int row_base = blockIdx.x * rows_per_block + warp;
    float accumulators[ROWS_PER_WARP] = {};
    float anchors[ROWS_PER_WARP] = {};
#pragma unroll
    for (int item = 0; item < ROWS_PER_WARP; ++item) {
        const int combined_row = row_base + item * NWARPS;
        if (combined_row < total_rows) {
            if constexpr (MULTI2) {
                anchors[item] = combined_row < first.N
                    ? first.neuron_scale[combined_row]
                    : second.neuron_scale[combined_row - first.N];
            } else {
                anchors[item] = first.neuron_scale[combined_row];
            }
        }
    }

    for (int group = lane; group < first.ng; group += 32) {
        const int8_t * activation =
            (STAGE_E8_THREE_BANKS ? qx : shared_qx) + group * kGroupSize;
        const int2 x0 = *reinterpret_cast<const int2 *>(activation);
        const int2 x1 = *reinterpret_cast<const int2 *>(activation + 8);
        const int2 x2 = *reinterpret_cast<const int2 *>(activation + 16);
        const float activation_scale = STAGE_E8_THREE_BANKS
            ? xscale[group]
            : shared_xscale[group];
        if constexpr (STAGE_E8_THREE_BANKS) {
            {
                uint64_t metadata[ROWS_PER_WARP] = {};
                int rows[ROWS_PER_WARP] = {};
                bool use_second[ROWS_PER_WARP] = {};
                bool valid[ROWS_PER_WARP] = {};
#pragma unroll
                for (int item = 0; item < ROWS_PER_WARP; ++item) {
                    const int combined_row = row_base + item * NWARPS;
                    valid[item] = combined_row < total_rows;
                    use_second[item] = combined_row >= first.N;
                    rows[item] = use_second[item]
                        ? combined_row - first.N
                        : combined_row;
                    if (valid[item]) {
                        const uint8_t * indices = use_second[item]
                            ? second.indices
                            : first.indices;
                        metadata[item] = load_group_exec64(
                            indices, rows[item], group, first.ng);
                    }
                }
#pragma unroll
                for (int item = 0; item < ROWS_PER_WARP; ++item) {
                    if (!valid[item]) continue;
                    const int8_t * active_table = use_second[item]
                        ? second_codebook
                        : first_codebook;
                    uint32_t state = 0;
                    const int dot =
                        aligned_group_dot_preloaded_e8_stage3_metadata(
                            active_table, shared_e8_codebook,
                            stage_active_table &&
                                active_table == staged_global_codebook,
                            metadata[item], x0, x1, x2, state);
                    accumulators[item] = fmaf(
                        format_scale<FORMAT>(
                            anchors[item], state, active_table) *
                            activation_scale,
                        static_cast<float>(dot), accumulators[item]);
                }
            }
        } else {
#pragma unroll
            for (int item = 0; item < ROWS_PER_WARP; ++item) {
                const int combined_row = row_base + item * NWARPS;
                if (combined_row >= total_rows) continue;
                uint32_t state = 0;
                int dot = 0;
                const int8_t * active_table = first_codebook;
                if constexpr (MULTI2) {
                    if (combined_row < first.N) {
                        if constexpr (STAGE_E8_THREE_BANKS) {
                            dot = aligned_group_dot_preloaded_e8_stage3(
                                first, first_codebook, shared_e8_codebook,
                                stage_active_table &&
                                    first_codebook == staged_global_codebook,
                                combined_row, group, x0, x1, x2, state);
                        } else {
                            dot = aligned_group_dot_preloaded<FORMAT>(
                                first, first_codebook, combined_row, group,
                                x0, x1, x2, state);
                        }
                    } else {
                        const int row = combined_row - first.N;
                        active_table = second_codebook;
                        if constexpr (STAGE_E8_THREE_BANKS) {
                            dot = aligned_group_dot_preloaded_e8_stage3(
                                second, second_codebook, shared_e8_codebook,
                                stage_active_table &&
                                    second_codebook == staged_global_codebook,
                                row, group, x0, x1, x2, state);
                        } else {
                            dot = aligned_group_dot_preloaded<FORMAT>(
                                second, second_codebook, row, group,
                                x0, x1, x2, state);
                        }
                    }
                } else {
                    if constexpr (STAGE_E8_THREE_BANKS) {
                        dot = aligned_group_dot_preloaded_e8_stage3(
                            first, first_codebook, shared_e8_codebook,
                            stage_active_table,
                            combined_row, group, x0, x1, x2, state);
                    } else {
                        dot = aligned_group_dot_preloaded<FORMAT>(
                            first, first_codebook, combined_row, group,
                            x0, x1, x2, state);
                    }
                }
                accumulators[item] = fmaf(
                    format_scale<FORMAT>(anchors[item], state, active_table) *
                        activation_scale,
                    static_cast<float>(dot), accumulators[item]);
            }
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            accumulators[item] += __shfl_xor_sync(
                0xffffffff, accumulators[item], offset);
        }
    }
    if constexpr (ADD_RESIDUAL) {
        auto * projected_rows = reinterpret_cast<__half *>(shared_storage);
        // Every warp must finish its codebook reads before this storage is
        // reused for the coalesced residual epilogue.
        __syncthreads();
        if (lane == 0) {
#pragma unroll
            for (int item = 0; item < ROWS_PER_WARP; ++item) {
                const int local_row = warp + item * NWARPS;
                const int combined_row = block_row_base + local_row;
                if (combined_row < total_rows) {
                    projected_rows[local_row] =
                        __float2half(accumulators[item]);
                }
            }
        }
        __syncthreads();
        if (linear_thread < rows_per_block) {
            const int combined_row = block_row_base + linear_thread;
            if (combined_row < total_rows) {
                output[combined_row] = __hadd(
                    residual[combined_row], projected_rows[linear_thread]);
            }
        }
    } else if (lane == 0) {
#pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            const int combined_row = row_base + item * NWARPS;
            if (combined_row < total_rows) {
                output[combined_row] = __float2half(accumulators[item]);
            }
        }
    }
}

template <int FORMAT, int NWARPS, bool OUTPUT_F32 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) aligned_group_swiglu_pair_kernel(
    NvqDeviceWeight gate,
    NvqDeviceWeight up,
    const int8_t * qx,
    const float * xscale,
    __half * output_h,
    float * output_f) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= gate.N) return;

    const float gate_anchor = gate.neuron_scale[row];
    const float up_anchor = up.neuron_scale[row];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int group = warp * 32 + lane;
         group < gate.ng;
         group += NWARPS * 32) {
        uint32_t gate_state = 0;
        uint32_t up_state = 0;
        int gate_dot = 0;
        int up_dot = 0;
        aligned_group_pair_dot<FORMAT>(
            gate, up, row, group, qx,
            gate_state, up_state, gate_dot, up_dot);
        const float activation_scale = xscale[group];
        gate_acc = fmaf(
            format_scale<FORMAT>(gate_anchor, gate_state, gate.codebook) *
                activation_scale,
            static_cast<float>(gate_dot), gate_acc);
        up_acc = fmaf(
            format_scale<FORMAT>(up_anchor, up_state, up.codebook) *
                activation_scale,
            static_cast<float>(up_dot), up_acc);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
        up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
    }
    __shared__ float partial[2][NWARPS];
    if (lane == 0) {
        partial[0][warp] = gate_acc;
        partial[1][warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0) {
        gate_acc = lane < NWARPS ? partial[0][lane] : 0.0f;
        up_acc = lane < NWARPS ? partial[1][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
        }
        if (lane == 0) {
            const float gate_value = __half2float(__float2half(gate_acc));
            const float up_value = __half2float(__float2half(up_acc));
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            const float value = up_value * (gate_value * sigmoid);
            if constexpr (OUTPUT_F32) output_f[row] = value;
            else output_h[row] = __float2half(value);
        }
    }
}

template <int FORMAT, int NWARPS, int MAX_M>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq_gemv_batch_vec8_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    float acc[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;
    for (int segment = warp * 32 + lane; segment < nsign; segment += NWARPS * 32) {
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const int k = group * kGroupSize + segment_in_group * 8;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_packed_bits(
            sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
        const auto weight = load_nvq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, segment, group, ng, nvec, nsign, sign_mode, sub);
        const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
            ? neuron_scale[row] * static_cast<float>(sub)
            : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float dot = dot_nvq_vec8<FORMAT>(
                    weight, qx + (static_cast<int64_t>(m) * ng * kGroupSize + k));
                acc[m] = fmaf(
                    weight_scale * xscale[static_cast<int64_t>(m) * ng + group], dot, acc[m]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
        }
    }
    __shared__ float partial[MAX_M][NWARPS];
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) partial[m][warp] = acc[m];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            acc[m] = lane < NWARPS ? partial[m][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
            }
            if (lane == 0 && m < M) {
                output[static_cast<int64_t>(m) * N + row] = __float2half(acc[m]);
            }
        }
    }
}

// One grid covers two projections with independent packed streams/codebooks.
// The activation is quantized once before this kernel.
template <int FORMAT, int NWARPS, int MAX_M, bool FAST_NVQ2_S4 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq_gemv_multi2_vec8_kernel(
    NvqDeviceWeight first,
    NvqDeviceWeight second,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M) {
    const int combined_row = blockIdx.x;
    const bool use_second = combined_row >= first.N;
    const NvqDeviceWeight weight = use_second ? second : first;
    const int row = use_second ? combined_row - first.N : combined_row;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= weight.N) return;

    float acc[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;
    [[maybe_unused]] const int64_t fast_index_row = static_cast<int64_t>(row) * weight.nvec;
    [[maybe_unused]] const int64_t fast_sign_row = static_cast<int64_t>(row) * weight.nsign;
    [[maybe_unused]] const int64_t fast_sub_row = static_cast<int64_t>(row) * weight.ng;
    [[maybe_unused]] const float fast_anchor = weight.neuron_scale[row];
    for (int segment = warp * 32 + lane;
         segment < weight.nsign;
         segment += NWARPS * 32) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const int8_t * qrow = qx + static_cast<int64_t>(m) * weight.ng * kGroupSize;
                const float * srow = xscale + static_cast<int64_t>(m) * weight.ng;
                if constexpr (FAST_NVQ2_S4) {
                    acc[m] = nvq2_vec8_scaled_fma_s4<
                        FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec,
                        FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec>(
                        weight, qrow, srow,
                        fast_index_row, fast_sign_row, fast_sub_row, fast_anchor,
                        segment, acc[m]);
                } else {
                    acc[m] = nvq_vec8_scaled_fma<FORMAT>(
                        weight, qrow, srow, row, segment, acc[m]);
                }
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
        }
    }
    __shared__ float partial[MAX_M][NWARPS];
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) partial[m][warp] = acc[m];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            acc[m] = lane < NWARPS ? partial[m][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
            }
            if (lane == 0 && m < M) {
                output[static_cast<int64_t>(m) * (first.N + second.N) + combined_row] =
                    __float2half(acc[m]);
            }
        }
    }
}

// NINT-style paired projection: each block computes matching gate/up rows and
// applies SwiGLU after the same fp16 rounding as the materialized path.
template <int FORMAT, int NWARPS, bool OUTPUT_F32 = false, bool FAST_NVQ2_S4 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq_gemv_swiglu_pair_kernel(
    NvqDeviceWeight gate,
    NvqDeviceWeight up,
    const int8_t * qx,
    const float * xscale,
    __half * output_h,
    float * output_f) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= gate.N) return;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    [[maybe_unused]] const int64_t fast_index_row = static_cast<int64_t>(row) * gate.nvec;
    [[maybe_unused]] const int64_t fast_sign_row = static_cast<int64_t>(row) * gate.nsign;
    [[maybe_unused]] const int64_t fast_sub_row = static_cast<int64_t>(row) * gate.ng;
    const float fast_gate_anchor = gate.neuron_scale[row];
    const float fast_up_anchor = up.neuron_scale[row];
    for (int segment = warp * 32 + lane;
        segment < gate.nsign;
         segment += NWARPS * 32) {
        if constexpr (FAST_NVQ2_S4) {
            nvq2_vec8_pair_scaled_fma_s4<
                FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec,
                FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec>(
                gate, up, qx, xscale,
                fast_index_row, fast_sign_row, fast_sub_row,
                fast_gate_anchor, fast_up_anchor, segment, gate_acc, up_acc);
        } else {
            gate_acc = nvq_vec8_scaled_fma<FORMAT>(
                gate, qx, xscale, row, segment, gate_acc);
            up_acc = nvq_vec8_scaled_fma<FORMAT>(
                up, qx, xscale, row, segment, up_acc);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
        up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
    }
    __shared__ float partial[2][NWARPS];
    if (lane == 0) {
        partial[0][warp] = gate_acc;
        partial[1][warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0) {
        gate_acc = lane < NWARPS ? partial[0][lane] : 0.0f;
        up_acc = lane < NWARPS ? partial[1][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
        }
        if (lane == 0) {
            const float gate_value = __half2float(__float2half(gate_acc));
            const float up_value = __half2float(__float2half(up_acc));
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            const float value = up_value * (gate_value * sigmoid);
            if constexpr (OUTPUT_F32) {
                output_f[row] = value;
            } else {
                output_h[row] = __float2half(value);
            }
        }
    }
}

__device__ __forceinline__ uint32_t load_u8x4(const uint8_t * data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

// Decode four adjacent vectors per thread, then replay the original MMVQ FMA
// order from shared memory. This keeps the packed-load savings without changing
// the quantized activation produced for Wdown.
template <int NWARPS, bool OUTPUT_F32 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq2_gemv_swiglu_vec4_ordered_kernel(
    NvqDeviceWeight gate,
    NvqDeviceWeight up,
    const int8_t * qx,
    const float * xscale,
    __half * output_h,
    float * output_f) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int thread = warp * 32 + lane;
    if (row >= gate.N) return;

    extern __shared__ uint32_t metadata[];
    const int64_t index_row = static_cast<int64_t>(row) * gate.nvec;
    const int64_t sign_row = static_cast<int64_t>(row) * gate.nsign;
    const int units = gate.nsign / 4;
    for (int unit = thread; unit < units; unit += NWARPS * 32) {
        const int segment_base = unit * 4;
        const uint32_t gate_indices = *reinterpret_cast<const uint32_t *>(
            gate.indices + index_row + segment_base);
        const uint32_t up_indices = *reinterpret_cast<const uint32_t *>(
            up.indices + index_row + segment_base);
        const int64_t sign_bit = (sign_row + segment_base) * 7;
        const int64_t sign_byte = sign_bit >> 3;
        const int sign_shift = static_cast<int>(sign_bit & 7);
        const uint32_t gate_signs = load_u8x4(gate.aux + sign_byte) >> sign_shift;
        const uint32_t up_signs = load_u8x4(up.aux + sign_byte) >> sign_shift;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t gate_index = (gate_indices >> (8 * i)) & 0xffu;
            const uint32_t up_index = (up_indices >> (8 * i)) & 0xffu;
            const uint32_t gate_mask7 = (gate_signs >> (7 * i)) & 0x7fu;
            const uint32_t up_mask7 = (up_signs >> (7 * i)) & 0x7fu;
            metadata[segment_base + i] = gate_index | (up_index << 8) |
                (gate_mask7 << 16) | (up_mask7 << 23);
        }
    }
    __syncthreads();

    const int64_t sub_row = static_cast<int64_t>(row) * gate.ng;
    const float gate_anchor = gate.neuron_scale[row];
    const float up_anchor = up.neuron_scale[row];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int segment = thread; segment < gate.nsign; segment += NWARPS * 32) {
        const uint32_t meta = metadata[segment];
        const uint32_t gate_index = meta & 0xffu;
        const uint32_t up_index = (meta >> 8) & 0xffu;
        const uint32_t gate_mask7 = (meta >> 16) & 0x7fu;
        const uint32_t up_mask7 = (meta >> 23) & 0x7fu;
        const int gate_last = parity7(gate_mask7) ^
            (gate.sign_mode ? ((gate_index >> 7) & 1u) : 0u);
        const int up_last = parity7(up_mask7) ^
            (up.sign_mode ? ((up_index >> 7) & 1u) : 0u);
        const int2 gate_values = apply_sign8(
            reinterpret_cast<const int2 *>(gate.codebook)[gate_index],
            gate_mask7 | (static_cast<uint32_t>(gate_last) << 7));
        const int2 up_values = apply_sign8(
            reinterpret_cast<const int2 *>(up.codebook)[up_index],
            up_mask7 | (static_cast<uint32_t>(up_last) << 7));
        const int2 x = *reinterpret_cast<const int2 *>(qx + segment * 8);
        const int gate_dot = __dp4a(
            gate_values.y, x.y, __dp4a(gate_values.x, x.x, 0));
        const int up_dot = __dp4a(
            up_values.y, x.y, __dp4a(up_values.x, x.x, 0));
        const int group = segment / 3;
        const float activation_scale = xscale[group];
        const uint32_t gate_sub = load_packed_4(gate.sub_scale, sub_row + group);
        const uint32_t up_sub = load_packed_4(up.sub_scale, sub_row + group);
        const float gate_scale = gate_anchor * static_cast<float>(gate_sub);
        const float up_scale = up_anchor * static_cast<float>(up_sub);
        gate_acc = fmaf(
            gate_scale * activation_scale, static_cast<float>(gate_dot), gate_acc);
        up_acc = fmaf(
            up_scale * activation_scale, static_cast<float>(up_dot), up_acc);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
        up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
    }
    __shared__ float partial[2][NWARPS];
    if (lane == 0) {
        partial[0][warp] = gate_acc;
        partial[1][warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0) {
        gate_acc = lane < NWARPS ? partial[0][lane] : 0.0f;
        up_acc = lane < NWARPS ? partial[1][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
        }
        if (lane == 0) {
            const float gate_value = __half2float(__float2half(gate_acc));
            const float up_value = __half2float(__float2half(up_acc));
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            const float value = up_value * (gate_value * sigmoid);
            if constexpr (OUTPUT_F32) {
                output_f[row] = value;
            } else {
                output_h[row] = __float2half(value);
            }
        }
    }
}

template <int GS>
__global__ void nvq_quantize_f32_kernel(
    const float * input,
    int8_t * output_qx,
    float * output_xscale,
    int K,
    int K_pad) {
    const int group = blockIdx.x;
    const int lane = threadIdx.x;
    const int k = group * GS + lane;
    const bool valid = lane < GS && k < K;
    const float value = valid ? input[k] : 0.0f;
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    int quant = 0;
    if (valid) {
        quant = static_cast<int>(roundf(value / scale));
        quant = quant < -127 ? -127 : (quant > 127 ? 127 : quant);
    }
    if (lane == 0) output_xscale[group] = scale;
    if (lane < GS && k < K_pad) output_qx[k] = static_cast<int8_t>(quant);
}

// The aligned group layout is faster when gate/up rows remain independent:
// 2N blocks expose both projections to the scheduler and keep the GEMV at the
// lower single-projection register count.  The existing N-float workspace is
// exactly large enough for the two fp16 projections, so this adds no workspace.
template <int GS>
__global__ void nvq_swiglu_quantize_f16_pair_kernel(
    const __half * gate_up,
    int8_t * output_qx,
    float * output_xscale,
    int K,
    int K_pad) {
    const int group = blockIdx.x;
    const int lane = threadIdx.x;
    if (group * GS >= K_pad) return;
    const int k = group * GS + lane;
    const bool valid = lane < GS && k < K;
    float value = 0.0f;
    if (valid) {
        const float gate_value = __half2float(gate_up[k]);
        const float up_value = __half2float(gate_up[K + k]);
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        value = up_value * (gate_value * sigmoid);
    }
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    int quant = 0;
    if (valid) {
        quant = static_cast<int>(roundf(value / scale));
        quant = quant < -127 ? -127 : (quant > 127 ? 127 : quant);
    }
    if (lane == 0) output_xscale[group] = scale;
    if (lane < GS && k < K_pad) output_qx[k] = static_cast<int8_t>(quant);
}

// Computes gate/up, applies SwiGLU, and writes the q8 activation layout used
// directly by the following NINT/NVQ down projection.
template <int FORMAT, int DOWN_GS>
__global__ void __launch_bounds__(1024, 1) nvq_ffn_swiglu_quant_kernel(
    NvqDeviceWeight gate,
    NvqDeviceWeight up,
    const int8_t * input_qx,
    const float * input_xscale,
    int8_t * output_qx,
    float * output_xscale,
    int down_k_pad) {
    const int output_group = blockIdx.x;
    const int element = threadIdx.y;
    const int lane = threadIdx.x;
    const int row = output_group * DOWN_GS + element;

    __shared__ float values[DOWN_GS];
    __shared__ float scale;

    float value = 0.0f;
    if (row < gate.N) {
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (int segment = lane; segment < gate.nsign; segment += 32) {
            gate_acc = nvq_vec8_scaled_fma<FORMAT>(
                gate, input_qx, input_xscale, row, segment, gate_acc);
            up_acc = nvq_vec8_scaled_fma<FORMAT>(
                up, input_qx, input_xscale, row, segment, up_acc);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
        }
        if (lane == 0) {
            const float gate_value = __half2float(__float2half(gate_acc));
            const float up_value = __half2float(__float2half(up_acc));
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            value = up_value * (gate_value * sigmoid);
        }
    }
    if (lane == 0) values[element] = value;
    __syncthreads();

    if (element == 0) {
        float amax = lane < DOWN_GS ? fabsf(values[lane]) : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
        }
        if (lane == 0) {
            scale = amax > 0.0f ? amax / 127.0f : 1.0f;
            output_xscale[output_group] = scale;
        }
    }
    __syncthreads();

    if (lane == 0) {
        int quant = 0;
        if (row < gate.N) {
            quant = static_cast<int>(roundf(values[element] / scale));
            quant = max(-127, min(127, quant));
        }
        const int offset = output_group * DOWN_GS + element;
        if (offset < down_k_pad) output_qx[offset] = static_cast<int8_t>(quant);
    }
}

template <int FORMAT>
__global__ void nvq_embedding_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int64_t * token_ids,
    __half * output,
    int tokens,
    int vocab,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    const int segments = (K + 7) / 8;
    const int64_t total = static_cast<int64_t>(tokens) * segments;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int token = static_cast<int>(linear / segments);
        const int segment = static_cast<int>(linear - static_cast<int64_t>(token) * segments);
        const int row = static_cast<int>(token_ids[token]);
        if (row < 0 || row >= vocab) continue;
        const int k0 = segment * 8;
        const int group = k0 / kGroupSize;
        const int chunk0 = (k0 - group * kGroupSize) / 4;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_packed_bits(
            sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
        const float scale = format_scale<FORMAT>(neuron_scale[row], sub, codebook);
        const int lo = decode_chunk4<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, group, chunk0, nvec, nsign, ng, sign_mode, sub);
        const int hi = decode_chunk4<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, group, chunk0 + 1, nvec, nsign, ng, sign_mode, sub);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k = k0 + i;
            if (k < K) {
                const int packed = i < 4 ? lo : hi;
                const int byte = (packed >> (8 * (i & 3))) & 0xff;
                const int value = static_cast<int>(static_cast<int8_t>(byte));
                output[static_cast<int64_t>(token) * K + k] =
                    __float2half(scale * static_cast<float>(value));
            }
        }
    }
}

__device__ __forceinline__ int mma168_i(int fragment) {
    return ((fragment / 2) * 8) + (threadIdx.x / 4);
}

__device__ __forceinline__ int mma168_j(int fragment) {
    return ((threadIdx.x % 4) * 2) + (fragment % 2);
}

__device__ __forceinline__ void load_mma_a_m16n8k32(int (&a)[4], const int * ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "l"(ptr));
}

__device__ __forceinline__ void load_mma_b_m16n8k32(int (&b)[2], const int * ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(b[0]), "=r"(b[1])
        : "l"(ptr));
}

__device__ __forceinline__ void mma_m16n8k32_s8(
    int (&d)[4], const int (&a)[4], const int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Integer Tensor Core path for M=4..64. Each gs24 quantization group is
// represented as one transient K32 tile in shared memory. The MMA dot is
// rescaled per group before accumulation, preserving NVQ's group scales.
template <int FORMAT, int MTILES, bool NEED_CHECK>
__global__ void __launch_bounds__(256) nvq_mmq_mma24_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    constexpr int kChunkGroups = 8;
    constexpr int kGroupInts = 8;
    constexpr int kStride = kChunkGroups * kGroupInts + 4;
    constexpr int kTileM = 16 * MTILES;
    constexpr int kRowsPerWarp = 8;
    constexpr int kWarps = 8;
    constexpr int kTileN = kRowsPerWarp * kWarps;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * kTileM;
    const int n0 = blockIdx.x * kTileN;

    __shared__ int activation[kTileM][kStride];
    __shared__ int weight[kWarps][kRowsPerWarp][kStride];
    __shared__ __half weight_scale[kChunkGroups][kTileN];
    __shared__ float activation_scale[kChunkGroups][kTileM];

    float acc[MTILES][4];
#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
#pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) acc[mt][fragment] = 0.0f;
    }

    for (int group_base = 0; group_base < ng; group_base += kChunkGroups) {
        const int active_groups = min(kChunkGroups, ng - group_base);

        for (int index = tid; index < kTileM * kChunkGroups * kGroupInts;
             index += kWarps * 32) {
            const int m_local = index / (kChunkGroups * kGroupInts);
            const int rest = index - m_local * kChunkGroups * kGroupInts;
            const int group_local = rest / kGroupInts;
            const int quartet = rest - group_local * kGroupInts;
            const int m = m0 + m_local;
            int value = 0;
            if (group_local < active_groups && quartet < kChunksPerGroup &&
                (!NEED_CHECK || m < M)) {
                const int group = group_base + group_local;
                value = load_i8x4(
                    qx + (static_cast<int64_t>(m) * ng + group) * kGroupSize + quartet * 4);
            }
            activation[m_local][group_local * kGroupInts + quartet] = value;
        }

        constexpr int kWeightTasks = kRowsPerWarp * kChunkGroups;
        for (int task = lane; task < kWeightTasks; task += 32) {
            const int row_local = task / kChunkGroups;
            const int group_local = task - row_local * kChunkGroups;
            const int row = n0 + warp * kRowsPerWarp + row_local;
            const int group = group_base + group_local;
            const bool valid = group_local < active_groups && (!NEED_CHECK || row < N);
            uint32_t sub = 0;
            if (valid) {
                const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
                sub = load_packed_bits(
                    sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
            }
#pragma unroll
            for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
                weight[warp][row_local][group_local * kGroupInts + quartet] = valid
                    ? decode_chunk4<FORMAT>(
                          indices, indices_nbytes, aux, aux_nbytes, codebook,
                          row, group, quartet, nvec, nsign, ng, sign_mode, sub)
                    : 0;
            }
            weight[warp][row_local][group_local * kGroupInts + 6] = 0;
            weight[warp][row_local][group_local * kGroupInts + 7] = 0;
            float scale = 0.0f;
            if (valid) {
                scale = format_scale<FORMAT>(neuron_scale[row], sub, codebook);
            }
            weight_scale[group_local][warp * kRowsPerWarp + row_local] = __float2half(scale);
        }

        for (int index = tid; index < kTileM * kChunkGroups; index += kWarps * 32) {
            const int m_local = index / kChunkGroups;
            const int group_local = index - m_local * kChunkGroups;
            const int m = m0 + m_local;
            activation_scale[group_local][m_local] =
                group_local < active_groups && (!NEED_CHECK || m < M)
                ? xscale[static_cast<int64_t>(m) * ng + group_base + group_local]
                : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int group_local = 0; group_local < kChunkGroups; ++group_local) {
            int a[MTILES][4];
#pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                load_mma_a_m16n8k32(
                    a[mt],
                    &activation[mt * 16 + lane % 16]
                               [group_local * kGroupInts + (lane / 16) * 4]);
            }
            int b[2];
            load_mma_b_m16n8k32(
                b,
                &weight[warp][lane % 8]
                       [group_local * kGroupInts + (((lane / 8) * 4) & 7)]);
#pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                int dot[4] = {0, 0, 0, 0};
                mma_m16n8k32_s8(dot, a[mt], b);
#pragma unroll
                for (int fragment = 0; fragment < 4; ++fragment) {
                    const int m_local = mt * 16 + mma168_i(fragment);
                    const int row_local = warp * kRowsPerWarp + mma168_j(fragment);
                    const int m = m0 + m_local;
                    const int row = n0 + row_local;
                    if (!NEED_CHECK || (m < M && row < N)) {
                        const float scale = activation_scale[group_local][m_local]
                            * __half2float(weight_scale[group_local][row_local]);
                        acc[mt][fragment] = fmaf(scale, static_cast<float>(dot[fragment]),
                                                acc[mt][fragment]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
#pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) {
            const int m = m0 + mt * 16 + mma168_i(fragment);
            const int row = n0 + warp * kRowsPerWarp + mma168_j(fragment);
            if (!NEED_CHECK || (m < M && row < N)) {
                output[static_cast<int64_t>(m) * N + row] = __float2half(acc[mt][fragment]);
            }
        }
    }
}

// Online-dequant FP16 Tensor Core path for M=16..256. Four gs24 groups form
// one K=96 chunk. A block expands a 64-row weight tile into shared memory and
// reuses it across 16..128 activation rows without materializing full weights.
template <int FORMAT, int MTILES, bool NEED_CHECK>
__global__ void __launch_bounds__(256) nvq_gemm_f16_gs24_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const __half * x,
    __half * output,
    int M,
    int N,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    constexpr int kGroupsPerChunk = 4;
    constexpr int kTileK = kGroupSize * kGroupsPerChunk;
    constexpr int kStrideK = kTileK + 8;
    constexpr int kTileM = 16 * MTILES;
    constexpr int kTileN = 64;
    constexpr int kWarps = 8;
    constexpr int kWmmaM = 16;
    constexpr int kWmmaN = 16;
    constexpr int kWmmaK = 16;
    constexpr int kNFragments = kTileN / kWmmaN;
    constexpr int kAccumulatorsPerWarp = (MTILES + 1) / 2;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * kTileM;
    const int n0 = blockIdx.x * kTileN;
    const int warp_m0 = warp / kNFragments;
    const int warp_n = warp % kNFragments;

    __shared__ __half weight_tile[kTileN][kStrideK];
    __shared__ __half activation_tile[kTileM][kStrideK];
    __shared__ float output_tile[kNFragments][kWmmaM][kWmmaN];

    using FragmentA = wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                                     __half, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                                     __half, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>;
    FragmentC accumulators[kAccumulatorsPerWarp];
#pragma unroll
    for (int accumulator = 0; accumulator < kAccumulatorsPerWarp; ++accumulator) {
        wmma::fill_fragment(accumulators[accumulator], 0.0f);
    }

    const int chunks = (ng + kGroupsPerChunk - 1) / kGroupsPerChunk;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int group_base = chunk * kGroupsPerChunk;
        const int k_base = group_base * kGroupSize;

        // There are exactly 64 * 4 row/group tasks, one per block thread.
        const int row_local = tid / kGroupsPerChunk;
        const int group_local = tid - row_local * kGroupsPerChunk;
        const int row = n0 + row_local;
        const int group = group_base + group_local;
        const bool valid_weight = group < ng && (!NEED_CHECK || row < N);
        uint32_t state = 0;
        float scale = 0.0f;
        if (valid_weight) {
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            state = load_packed_bits(
                sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
            scale = format_scale<FORMAT>(neuron_scale[row], state, codebook);
        }
#pragma unroll
        for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
            const int packed = valid_weight
                ? decode_chunk4<FORMAT>(
                      indices, indices_nbytes, aux, aux_nbytes, codebook,
                      row, group, quartet, nvec, nsign, ng, sign_mode, state)
                : 0;
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const int shift = pair * 16;
                const int value0 = static_cast<int>(
                    static_cast<int8_t>((packed >> shift) & 0xff));
                const int value1 = static_cast<int>(
                    static_cast<int8_t>((packed >> (shift + 8)) & 0xff));
                const __half2 values = __halves2half2(
                    __float2half(scale * static_cast<float>(value0)),
                    __float2half(scale * static_cast<float>(value1)));
                *reinterpret_cast<__half2 *>(
                    &weight_tile[row_local]
                                [group_local * kGroupSize + quartet * 4 + pair * 2]) = values;
            }
        }

        constexpr int kActivationPairs = kTileM * (kTileK / 2);
        for (int index = tid; index < kActivationPairs; index += kWarps * 32) {
            const int m_local = index / (kTileK / 2);
            const int pair = index - m_local * (kTileK / 2);
            const int m = m0 + m_local;
            const int k = k_base + pair * 2;
            __half2 values = __float2half2_rn(0.0f);
            if ((!NEED_CHECK || m < M) && k + 1 < K && (K & 1) == 0) {
                values = *reinterpret_cast<const __half2 *>(
                    x + static_cast<int64_t>(m) * K + k);
            } else if ((!NEED_CHECK || m < M) && k < K) {
                const __half value0 = x[static_cast<int64_t>(m) * K + k];
                const __half value1 = k + 1 < K
                    ? x[static_cast<int64_t>(m) * K + k + 1]
                    : __float2half(0.0f);
                values = __halves2half2(value0, value1);
            }
            *reinterpret_cast<__half2 *>(
                &activation_tile[m_local][pair * 2]) = values;
        }
        __syncthreads();

        const bool warp_active = warp_m0 < MTILES;
#pragma unroll
        for (int k_local = 0; k_local < kTileK; k_local += kWmmaK) {
            if (warp_active) {
                FragmentB weight_fragment;
                wmma::load_matrix_sync(
                    weight_fragment,
                    &weight_tile[warp_n * kWmmaN][k_local],
                    kStrideK);
#pragma unroll
                for (int accumulator = 0;
                     accumulator < kAccumulatorsPerWarp;
                     ++accumulator) {
                    const int m_fragment = warp_m0 + accumulator * 2;
                    if (m_fragment < MTILES) {
                        FragmentA activation_fragment;
                        wmma::load_matrix_sync(
                            activation_fragment,
                            &activation_tile[m_fragment * kWmmaM][k_local],
                            kStrideK);
                        wmma::mma_sync(
                            accumulators[accumulator], activation_fragment,
                            weight_fragment, accumulators[accumulator]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int accumulator = 0;
         accumulator < kAccumulatorsPerWarp;
         ++accumulator) {
#pragma unroll
        for (int warp_row = 0; warp_row < 2; ++warp_row) {
            const int m_fragment = warp_m0 + accumulator * 2;
            const bool owns_fragment = warp_m0 == warp_row && m_fragment < MTILES;
            if (owns_fragment) {
                wmma::store_matrix_sync(
                    &output_tile[warp_n][0][0], accumulators[accumulator],
                    kWmmaN, wmma::mem_row_major);
            }
            __syncthreads();
            if (owns_fragment) {
                const int output_m0 = m0 + m_fragment * kWmmaM;
                const int output_n0 = n0 + warp_n * kWmmaN;
#pragma unroll
                for (int element = lane; element < kWmmaM * kWmmaN; element += 32) {
                    const int local_m = element / kWmmaN;
                    const int local_n = element - local_m * kWmmaN;
                    const int m = output_m0 + local_m;
                    const int n = output_n0 + local_n;
                    if (!NEED_CHECK || (m < M && n < N)) {
                        output[static_cast<int64_t>(m) * N + n] =
                            __float2half(output_tile[warp_n][local_m][local_n]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <int FORMAT>
__global__ void nepq_dequant_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    __half * weight,
    int N,
    int K,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int sign_mode) {
    const int64_t total = static_cast<int64_t>(N) * nsign;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(linear / nsign);
        const int segment = static_cast<int>(linear - static_cast<int64_t>(row) * nsign);
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const int k0 = group * kGroupSize + segment_in_group * 8;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_packed_bits(
            sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
        const int8_t * table = nepq_active_table(
            table_pool, bank_ids, row, group, nsuper, table_stride);
        const auto values = load_nepq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, table,
            row, segment, group, ng, nvec, nsign, sign_mode, sub);
        const float scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
            ? neuron_scale[row] * static_cast<float>(sub)
            : format_scale<FORMAT>(neuron_scale[row], sub, table);
        const int2 packed = values.values;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k = k0 + i;
            if (k >= K) continue;
            const int word = i < 4 ? packed.x : packed.y;
            float value = static_cast<float>(
                static_cast<int8_t>((word >> (8 * (i & 3))) & 0xff));
            if constexpr (FORMAT == kNvq1L) {
                value += 0.125f * static_cast<float>(values.delta);
            }
            if constexpr (FORMAT == kNvq1S) {
                value += 0.15625f * static_cast<float>(values.delta);
            }
            weight[static_cast<int64_t>(row) * K + k] = __float2half(scale * value);
        }
    }
}

template <int FORMAT, int NWARPS, int MAX_M>
__global__ void __launch_bounds__(NWARPS * 32, 1) nepq_gemv_batch_vec8_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int sign_mode) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    float acc[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;
    for (int segment = warp * 32 + lane;
         segment < nsign;
         segment += NWARPS * 32) {
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const int k = group * kGroupSize + segment_in_group * 8;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_packed_bits(
            sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
        const int8_t * table = nepq_active_table(
            table_pool, bank_ids, row, group, nsuper, table_stride);
        const auto weight = load_nepq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, table,
            row, segment, group, ng, nvec, nsign, sign_mode, sub);
        const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
            ? neuron_scale[row] * static_cast<float>(sub)
            : format_scale<FORMAT>(neuron_scale[row], sub, table);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float dot = dot_nvq_vec8<FORMAT>(
                    weight, qx + (static_cast<int64_t>(m) * ng * kGroupSize + k));
                acc[m] = fmaf(
                    weight_scale * xscale[static_cast<int64_t>(m) * ng + group],
                    dot,
                    acc[m]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
        }
    }
    __shared__ float partial[MAX_M][NWARPS];
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) partial[m][warp] = acc[m];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            acc[m] = lane < NWARPS ? partial[m][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
            }
            if (lane == 0 && m < M) {
                output[static_cast<int64_t>(m) * N + row] = __float2half(acc[m]);
            }
        }
    }
}

template <int FORMAT, int MTILES, bool NEED_CHECK>
__global__ void __launch_bounds__(256) nepq_mmq_mma24_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int sign_mode) {
    constexpr int kChunkGroups = 8;
    constexpr int kGroupInts = 8;
    constexpr int kStride = kChunkGroups * kGroupInts + 4;
    constexpr int kTileM = 16 * MTILES;
    constexpr int kRowsPerWarp = 8;
    constexpr int kWarps = 8;
    constexpr int kTileN = kRowsPerWarp * kWarps;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * kTileM;
    const int n0 = blockIdx.x * kTileN;

    __shared__ int activation[kTileM][kStride];
    __shared__ int weight[kWarps][kRowsPerWarp][kStride];
    __shared__ __half weight_scale[kChunkGroups][kTileN];
    __shared__ float activation_scale[kChunkGroups][kTileM];

    float acc[MTILES][4];
#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
#pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) acc[mt][fragment] = 0.0f;
    }

    for (int group_base = 0; group_base < ng; group_base += kChunkGroups) {
        const int active_groups = min(kChunkGroups, ng - group_base);
        for (int index = tid; index < kTileM * kChunkGroups * kGroupInts;
             index += kWarps * 32) {
            const int m_local = index / (kChunkGroups * kGroupInts);
            const int rest = index - m_local * kChunkGroups * kGroupInts;
            const int group_local = rest / kGroupInts;
            const int quartet = rest - group_local * kGroupInts;
            const int m = m0 + m_local;
            int value = 0;
            if (group_local < active_groups && quartet < kChunksPerGroup &&
                (!NEED_CHECK || m < M)) {
                const int group = group_base + group_local;
                value = load_i8x4(
                    qx + (static_cast<int64_t>(m) * ng + group) * kGroupSize
                    + quartet * 4);
            }
            activation[m_local][group_local * kGroupInts + quartet] = value;
        }

        constexpr int kWeightTasks = kRowsPerWarp * kChunkGroups;
        for (int task = lane; task < kWeightTasks; task += 32) {
            const int row_local = task / kChunkGroups;
            const int group_local = task - row_local * kChunkGroups;
            const int row = n0 + warp * kRowsPerWarp + row_local;
            const int group = group_base + group_local;
            const bool valid = group_local < active_groups && (!NEED_CHECK || row < N);
            uint32_t sub = 0;
            const int8_t * table = table_pool;
            if (valid) {
                const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
                sub = load_packed_bits(
                    sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
                table = nepq_active_table(
                    table_pool, bank_ids, row, group, nsuper, table_stride);
            }
#pragma unroll
            for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
                weight[warp][row_local][group_local * kGroupInts + quartet] = valid
                    ? decode_nepq_chunk4<FORMAT>(
                          indices, indices_nbytes, aux, aux_nbytes, table,
                          row, group, quartet, nvec, nsign, ng, sign_mode, sub)
                    : 0;
            }
            weight[warp][row_local][group_local * kGroupInts + 6] = 0;
            weight[warp][row_local][group_local * kGroupInts + 7] = 0;
            const float scale = valid
                ? format_scale<FORMAT>(neuron_scale[row], sub, table)
                : 0.0f;
            weight_scale[group_local][warp * kRowsPerWarp + row_local] = __float2half(scale);
        }

        for (int index = tid; index < kTileM * kChunkGroups; index += kWarps * 32) {
            const int m_local = index / kChunkGroups;
            const int group_local = index - m_local * kChunkGroups;
            const int m = m0 + m_local;
            activation_scale[group_local][m_local] =
                group_local < active_groups && (!NEED_CHECK || m < M)
                ? xscale[static_cast<int64_t>(m) * ng + group_base + group_local]
                : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int group_local = 0; group_local < kChunkGroups; ++group_local) {
            int a[MTILES][4];
#pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                load_mma_a_m16n8k32(
                    a[mt],
                    &activation[mt * 16 + lane % 16]
                               [group_local * kGroupInts + (lane / 16) * 4]);
            }
            int b[2];
            load_mma_b_m16n8k32(
                b,
                &weight[warp][lane % 8]
                       [group_local * kGroupInts + (((lane / 8) * 4) & 7)]);
#pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                int dot[4] = {0, 0, 0, 0};
                mma_m16n8k32_s8(dot, a[mt], b);
#pragma unroll
                for (int fragment = 0; fragment < 4; ++fragment) {
                    const int m_local = mt * 16 + mma168_i(fragment);
                    const int row_local = warp * kRowsPerWarp + mma168_j(fragment);
                    const int m = m0 + m_local;
                    const int row = n0 + row_local;
                    if (!NEED_CHECK || (m < M && row < N)) {
                        const float scale = activation_scale[group_local][m_local]
                            * __half2float(weight_scale[group_local][row_local]);
                        acc[mt][fragment] = fmaf(
                            scale, static_cast<float>(dot[fragment]), acc[mt][fragment]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
#pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) {
            const int m = m0 + mt * 16 + mma168_i(fragment);
            const int row = n0 + warp * kRowsPerWarp + mma168_j(fragment);
            if (!NEED_CHECK || (m < M && row < N)) {
                output[static_cast<int64_t>(m) * N + row] = __float2half(acc[mt][fragment]);
            }
        }
    }
}

template <int FORMAT, int MTILES, bool NEED_CHECK>
__global__ void __launch_bounds__(256) nepq_gemm_f16_gs24_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const __half * x,
    __half * output,
    int M,
    int N,
    int K,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int sign_mode) {
    constexpr int kGroupsPerChunk = kNepqGroupsPerSupergroup;
    constexpr int kTileK = kGroupSize * kGroupsPerChunk;
    constexpr int kStrideK = kTileK + 8;
    constexpr int kTileM = 16 * MTILES;
    constexpr int kTileN = 64;
    constexpr int kWarps = 8;
    constexpr int kWmmaM = 16;
    constexpr int kWmmaN = 16;
    constexpr int kWmmaK = 16;
    constexpr int kNFragments = kTileN / kWmmaN;
    constexpr int kAccumulatorsPerWarp = (MTILES + 1) / 2;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * kTileM;
    const int n0 = blockIdx.x * kTileN;
    const int warp_m0 = warp / kNFragments;
    const int warp_n = warp % kNFragments;

    __shared__ __half weight_tile[kTileN][kStrideK];
    __shared__ __half activation_tile[kTileM][kStrideK];
    __shared__ float output_tile[kNFragments][kWmmaM][kWmmaN];

    using FragmentA = wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                                     __half, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                                     __half, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>;
    FragmentC accumulators[kAccumulatorsPerWarp];
#pragma unroll
    for (int accumulator = 0; accumulator < kAccumulatorsPerWarp; ++accumulator) {
        wmma::fill_fragment(accumulators[accumulator], 0.0f);
    }

    const int chunks = (ng + kGroupsPerChunk - 1) / kGroupsPerChunk;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int group_base = chunk * kGroupsPerChunk;
        const int k_base = group_base * kGroupSize;
        const int row_local = tid / kGroupsPerChunk;
        const int group_local = tid - row_local * kGroupsPerChunk;
        const int row = n0 + row_local;
        const int group = group_base + group_local;
        const bool valid_weight = group < ng && (!NEED_CHECK || row < N);
        uint32_t state = 0;
        float scale = 0.0f;
        const int8_t * table = table_pool;
        if (valid_weight) {
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            state = load_packed_bits(
                sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
            table = nepq_active_table(
                table_pool, bank_ids, row, group, nsuper, table_stride);
            scale = format_scale<FORMAT>(neuron_scale[row], state, table);
        }
#pragma unroll
        for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
            const int packed = valid_weight
                ? decode_nepq_chunk4<FORMAT>(
                      indices, indices_nbytes, aux, aux_nbytes, table,
                      row, group, quartet, nvec, nsign, ng, sign_mode, state)
                : 0;
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const int shift = pair * 16;
                float value0 = static_cast<float>(
                    static_cast<int8_t>((packed >> shift) & 0xff));
                float value1 = static_cast<float>(
                    static_cast<int8_t>((packed >> (shift + 8)) & 0xff));
                const __half2 values = __halves2half2(
                    __float2half(scale * value0), __float2half(scale * value1));
                *reinterpret_cast<__half2 *>(
                    &weight_tile[row_local]
                                [group_local * kGroupSize + quartet * 4 + pair * 2]) = values;
            }
        }

        constexpr int kActivationPairs = kTileM * (kTileK / 2);
        for (int index = tid; index < kActivationPairs; index += kWarps * 32) {
            const int m_local = index / (kTileK / 2);
            const int pair = index - m_local * (kTileK / 2);
            const int m = m0 + m_local;
            const int k = k_base + pair * 2;
            __half2 values = __float2half2_rn(0.0f);
            if ((!NEED_CHECK || m < M) && k + 1 < K && (K & 1) == 0) {
                values = *reinterpret_cast<const __half2 *>(
                    x + static_cast<int64_t>(m) * K + k);
            } else if ((!NEED_CHECK || m < M) && k < K) {
                const __half value0 = x[static_cast<int64_t>(m) * K + k];
                const __half value1 = k + 1 < K
                    ? x[static_cast<int64_t>(m) * K + k + 1]
                    : __float2half(0.0f);
                values = __halves2half2(value0, value1);
            }
            *reinterpret_cast<__half2 *>(
                &activation_tile[m_local][pair * 2]) = values;
        }
        __syncthreads();

        const bool warp_active = warp_m0 < MTILES;
#pragma unroll
        for (int k_local = 0; k_local < kTileK; k_local += kWmmaK) {
            if (warp_active) {
                FragmentB weight_fragment;
                wmma::load_matrix_sync(
                    weight_fragment, &weight_tile[warp_n * kWmmaN][k_local], kStrideK);
#pragma unroll
                for (int accumulator = 0;
                     accumulator < kAccumulatorsPerWarp;
                     ++accumulator) {
                    const int m_fragment = warp_m0 + accumulator * 2;
                    if (m_fragment < MTILES) {
                        FragmentA activation_fragment;
                        wmma::load_matrix_sync(
                            activation_fragment,
                            &activation_tile[m_fragment * kWmmaM][k_local],
                            kStrideK);
                        wmma::mma_sync(
                            accumulators[accumulator], activation_fragment,
                            weight_fragment, accumulators[accumulator]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int accumulator = 0; accumulator < kAccumulatorsPerWarp; ++accumulator) {
#pragma unroll
        for (int warp_row = 0; warp_row < 2; ++warp_row) {
            const int m_fragment = warp_m0 + accumulator * 2;
            const bool owns_fragment = warp_m0 == warp_row && m_fragment < MTILES;
            if (owns_fragment) {
                wmma::store_matrix_sync(
                    &output_tile[warp_n][0][0], accumulators[accumulator],
                    kWmmaN, wmma::mem_row_major);
            }
            __syncthreads();
            if (owns_fragment) {
                const int output_m0 = m0 + m_fragment * kWmmaM;
                const int output_n0 = n0 + warp_n * kWmmaN;
#pragma unroll
                for (int element = lane; element < kWmmaM * kWmmaN; element += 32) {
                    const int local_m = element / kWmmaN;
                    const int local_n = element - local_m * kWmmaN;
                    const int m = output_m0 + local_m;
                    const int n = output_n0 + local_n;
                    if (!NEED_CHECK || (m < M && n < N)) {
                        output[static_cast<int64_t>(m) * N + n] =
                            __float2half(output_tile[warp_n][local_m][local_n]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <int FORMAT, int NWARPS, int ROWS_PER_BLOCK, bool SHARE_GROUP_STATE>
__global__ void __launch_bounds__(NWARPS * 32) nvq_moe_mmvq_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    const int32_t * ids,
    const int32_t * expert_local,
    __half * output,
    int pairs,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    bool routed_input) {
    const int pair = blockIdx.y;
    const int row0 = blockIdx.x * ROWS_PER_BLOCK;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (pair >= pairs) return;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >=
        static_cast<unsigned int>(global_experts)) {
        return;
    }
    const int local_expert = expert_local[expert];
    if (static_cast<unsigned int>(local_expert) >=
        static_cast<unsigned int>(pool_experts)) {
        return;
    }
    const int source_row = routed_input ? pair : pair / routes;

    float acc[ROWS_PER_BLOCK];
#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
        acc[row_local] = 0.0f;
    }
    for (int segment = warp * 32 + lane;
         segment < nsign;
         segment += NWARPS * 32) {
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const unsigned int active_mask = __activemask();
        const int group_leader = lane - segment_in_group;
        const int source_lane = group_leader >= 0 ? group_leader : lane;
        const int k = group * kGroupSize + segment_in_group * 8;
        const int8_t * activation =
            qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
        float activation_scale;
        if constexpr (
            SHARE_GROUP_STATE &&
            (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec)) {
            const float loaded_scale = lane == source_lane
                ? xscale[static_cast<int64_t>(source_row) * ng + group]
                : 0.0f;
            activation_scale = __shfl_sync(
                active_mask, loaded_scale, source_lane);
        } else {
            activation_scale =
                xscale[static_cast<int64_t>(source_row) * ng + group];
        }
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            const int local_row = row0 + row_local;
            if (local_row >= out_per_expert) continue;
            const int row = local_expert * out_per_expert + local_row;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            uint32_t sub;
            float weight_scale;
            if constexpr (
                SHARE_GROUP_STATE &&
                (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec)) {
                const uint32_t loaded_sub = lane == source_lane
                    ? load_packed_bits(
                        sub_scale, sub_linear * sub_bits, sub_bits,
                        sub_scale_nbytes)
                    : 0;
                sub = __shfl_sync(active_mask, loaded_sub, source_lane);
                const float loaded_weight_scale = lane == source_lane
                    ? format_scale<FORMAT>(
                        neuron_scale[row], loaded_sub, codebook)
                    : 0.0f;
                weight_scale = __shfl_sync(
                    active_mask, loaded_weight_scale, source_lane);
            } else {
                sub = load_packed_bits(
                    sub_scale, sub_linear * sub_bits, sub_bits,
                    sub_scale_nbytes);
                weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                    ? neuron_scale[row] * static_cast<float>(sub)
                    : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
            }
            const auto weight = load_nvq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, codebook,
                row, segment, group, ng, nvec, nsign, sign_mode, sub);
            acc[row_local] = fmaf(
                weight_scale * activation_scale,
                dot_nvq_vec8<FORMAT>(weight, activation),
                acc[row_local]);
        }
    }

#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[row_local] += __shfl_xor_sync(0xffffffff, acc[row_local], offset);
        }
    }
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];
    if (lane == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            partial[row_local][warp] = acc[row_local];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            float value = lane < NWARPS ? partial[row_local][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_xor_sync(0xffffffff, value, offset);
            }
            const int local_row = row0 + row_local;
            if (lane == 0 && local_row < out_per_expert) {
                output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                    __float2half(value);
            }
        }
    }
}

template <int FORMAT, int PHYSICAL_WARPS, int LOGICAL_WARPS, int ROWS_PER_BLOCK>
__global__ void __launch_bounds__(PHYSICAL_WARPS * 32)
nvq_moe_mmvq_exact_reduction_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    const int32_t * ids,
    const int32_t * expert_local,
    __half * output,
    int pairs,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    bool routed_input) {
    static_assert(LOGICAL_WARPS % PHYSICAL_WARPS == 0);
    constexpr int kAccumulators = LOGICAL_WARPS / PHYSICAL_WARPS;
    const int pair = blockIdx.y;
    const int row0 = blockIdx.x * ROWS_PER_BLOCK;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (pair >= pairs) return;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >=
        static_cast<unsigned int>(global_experts)) {
        return;
    }
    const int local_expert = expert_local[expert];
    if (static_cast<unsigned int>(local_expert) >=
        static_cast<unsigned int>(pool_experts)) {
        return;
    }
    const int source_row = routed_input ? pair : pair / routes;

    float acc[ROWS_PER_BLOCK][kAccumulators];
#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
        for (int accumulator = 0; accumulator < kAccumulators; ++accumulator) {
            acc[row_local][accumulator] = 0.0f;
        }
    }
#pragma unroll
    for (int accumulator = 0; accumulator < kAccumulators; ++accumulator) {
        const int logical_warp = warp + accumulator * PHYSICAL_WARPS;
        for (int segment = logical_warp * 32 + lane;
             segment < nsign;
             segment += LOGICAL_WARPS * 32) {
            const int group = segment / 3;
            const int segment_in_group = segment - group * 3;
            const int k = group * kGroupSize + segment_in_group * 8;
            const int8_t * activation =
                qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
            const float activation_scale =
                xscale[static_cast<int64_t>(source_row) * ng + group];
#pragma unroll
            for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
                const int local_row = row0 + row_local;
                if (local_row >= out_per_expert) continue;
                const int row = local_expert * out_per_expert + local_row;
                const int64_t sub_linear =
                    static_cast<int64_t>(row) * ng + group;
                const uint32_t sub = load_packed_bits(
                    sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
                const auto weight = load_nvq_vec8<FORMAT>(
                    indices, indices_nbytes, aux, aux_nbytes, codebook,
                    row, segment, group, ng, nvec, nsign, sign_mode, sub);
                const float weight_scale =
                    (FORMAT == kNvq1L || FORMAT == kNvq1S)
                    ? neuron_scale[row] * static_cast<float>(sub)
                    : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
                acc[row_local][accumulator] = fmaf(
                    weight_scale * activation_scale,
                    dot_nvq_vec8<FORMAT>(weight, activation),
                    acc[row_local][accumulator]);
            }
        }
    }

#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
        for (int accumulator = 0; accumulator < kAccumulators; ++accumulator) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[row_local][accumulator] += __shfl_xor_sync(
                    0xffffffff, acc[row_local][accumulator], offset);
            }
        }
    }
    __shared__ float partial[ROWS_PER_BLOCK][LOGICAL_WARPS];
    if (lane == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
            for (int accumulator = 0; accumulator < kAccumulators; ++accumulator) {
                const int logical_warp = warp + accumulator * PHYSICAL_WARPS;
                partial[row_local][logical_warp] =
                    acc[row_local][accumulator];
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            float value = lane < LOGICAL_WARPS
                ? partial[row_local][lane]
                : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_xor_sync(0xffffffff, value, offset);
            }
            const int local_row = row0 + row_local;
            if (lane == 0 && local_row < out_per_expert) {
                output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                    __float2half(value);
            }
        }
    }
}

template <int FORMAT, int TILE_M>
__global__ void __launch_bounds__(128) nvq_moe_grouped_tile_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    const int32_t * expert_local,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    const int32_t * tile_bounds,
    const int32_t * tile_experts,
    __half * output,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    int max_tiles,
    int row_tiles,
    bool routed_input) {
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int row_tile = static_cast<int>(task % row_tiles);
        const int tile = static_cast<int>(task / row_tiles);
        if (tile >= tile_bounds[global_experts]) continue;
        const int expert = tile_experts[tile];
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned int>(local_expert) >=
            static_cast<unsigned int>(pool_experts)) {
            continue;
        }
        const int local_tile = tile - tile_bounds[expert];
        const int first = expert_bounds[expert] + local_tile * TILE_M;
        const int last = min(first + TILE_M, expert_bounds[expert + 1]);
        const int local_row = row_tile * 4 + warp;
        if (local_row >= out_per_expert) continue;
        const int row = local_expert * out_per_expert + local_row;

        float acc[TILE_M];
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) acc[item] = 0.0f;
        for (int segment = lane; segment < nsign; segment += 32) {
            const int group = segment / 3;
            const int segment_in_group = segment - group * 3;
            const int k = group * kGroupSize + segment_in_group * 8;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            const uint32_t sub = load_packed_bits(
                sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
            const auto weight = load_nvq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, codebook,
                row, segment, group, ng, nvec, nsign, sign_mode, sub);
            const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                ? neuron_scale[row] * static_cast<float>(sub)
                : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
#pragma unroll
            for (int item = 0; item < TILE_M; ++item) {
                const int compact = first + item;
                if (compact >= last) continue;
                const int pair = ids_dst[compact];
                const int source_row = routed_input ? pair : pair / routes;
                const int8_t * activation =
                    qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
                acc[item] = fmaf(
                    weight_scale *
                        xscale[static_cast<int64_t>(source_row) * ng + group],
                    dot_nvq_vec8<FORMAT>(weight, activation),
                    acc[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[item] += __shfl_xor_sync(0xffffffff, acc[item], offset);
            }
            if (lane == 0 && first + item < last) {
                const int pair = ids_dst[first + item];
                output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                    __float2half(acc[item]);
            }
        }
    }
}

template <int FORMAT, int NWARPS, int ROWS_PER_BLOCK>
__global__ void __launch_bounds__(NWARPS * 32) nepq_moe_mmvq_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const int8_t * qx,
    const float * xscale,
    const int32_t * ids,
    const int32_t * expert_local,
    __half * output,
    int pairs,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    bool routed_input) {
    const int pair = blockIdx.y;
    const int row0 = blockIdx.x * ROWS_PER_BLOCK;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (pair >= pairs) return;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >=
        static_cast<unsigned int>(global_experts)) {
        return;
    }
    const int local_expert = expert_local == nullptr ? expert : expert_local[expert];
    if (static_cast<unsigned int>(local_expert) >=
        static_cast<unsigned int>(pool_experts)) {
        return;
    }
    const int source_row = routed_input ? pair : pair / routes;

    float acc[ROWS_PER_BLOCK];
#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
        acc[row_local] = 0.0f;
    }
    for (int segment = warp * 32 + lane;
         segment < nsign;
         segment += NWARPS * 32) {
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const int k = group * kGroupSize + segment_in_group * 8;
        const int8_t * activation =
            qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
        const float activation_scale =
            xscale[static_cast<int64_t>(source_row) * ng + group];
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            const int local_row = row0 + row_local;
            if (local_row >= out_per_expert) continue;
            const int row = local_expert * out_per_expert + local_row;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            const uint32_t sub = load_packed_bits(
                sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
            const int8_t * table = nepq_active_table(
                table_pool, bank_ids, row, group, nsuper, table_stride);
            const auto weight = load_nepq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, table,
                row, segment, group, ng, nvec, nsign, 0, sub);
            const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                ? neuron_scale[row] * static_cast<float>(sub)
                : format_scale<FORMAT>(neuron_scale[row], sub, table);
            acc[row_local] = fmaf(
                weight_scale * activation_scale,
                dot_nvq_vec8<FORMAT>(weight, activation),
                acc[row_local]);
        }
    }

#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[row_local] += __shfl_xor_sync(0xffffffff, acc[row_local], offset);
        }
    }
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];
    if (lane == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            partial[row_local][warp] = acc[row_local];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            float value = lane < NWARPS ? partial[row_local][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_xor_sync(0xffffffff, value, offset);
            }
            const int local_row = row0 + row_local;
            if (lane == 0 && local_row < out_per_expert) {
                output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                    __float2half(value);
            }
        }
    }
}

template <int FORMAT, int TILE_M>
__global__ void __launch_bounds__(128) nepq_moe_grouped_tile_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const int8_t * qx,
    const float * xscale,
    const int32_t * expert_local,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    const int32_t * tile_bounds,
    const int32_t * tile_experts,
    __half * output,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int max_tiles,
    int row_tiles,
    bool routed_input) {
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int row_tile = static_cast<int>(task % row_tiles);
        const int tile = static_cast<int>(task / row_tiles);
        if (tile >= tile_bounds[global_experts]) continue;
        const int expert = tile_experts[tile];
        const int local_expert =
            expert_local == nullptr ? expert : expert_local[expert];
        if (static_cast<unsigned int>(local_expert) >=
            static_cast<unsigned int>(pool_experts)) {
            continue;
        }
        const int local_tile = tile - tile_bounds[expert];
        const int first = expert_bounds[expert] + local_tile * TILE_M;
        const int last = min(first + TILE_M, expert_bounds[expert + 1]);
        const int local_row = row_tile * 4 + warp;
        if (local_row >= out_per_expert) continue;
        const int row = local_expert * out_per_expert + local_row;

        float acc[TILE_M];
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) acc[item] = 0.0f;
        for (int segment = lane; segment < nsign; segment += 32) {
            const int group = segment / 3;
            const int segment_in_group = segment - group * 3;
            const int k = group * kGroupSize + segment_in_group * 8;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            const uint32_t sub = load_packed_bits(
                sub_scale, sub_linear * sub_bits, sub_bits, sub_scale_nbytes);
            const int8_t * table = nepq_active_table(
                table_pool, bank_ids, row, group, nsuper, table_stride);
            const auto weight = load_nvq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, table,
                row, segment, group, ng, nvec, nsign, 0, sub);
            const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                ? neuron_scale[row] * static_cast<float>(sub)
                : format_scale<FORMAT>(neuron_scale[row], sub, table);
#pragma unroll
            for (int item = 0; item < TILE_M; ++item) {
                const int compact = first + item;
                if (compact >= last) continue;
                const int pair = ids_dst[compact];
                const int source_row = routed_input ? pair : pair / routes;
                const int8_t * activation =
                    qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
                acc[item] = fmaf(
                    weight_scale * xscale[static_cast<int64_t>(source_row) * ng + group],
                    dot_nvq_vec8<FORMAT>(weight, activation),
                    acc[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[item] += __shfl_xor_sync(0xffffffff, acc[item], offset);
            }
            if (lane == 0 && first + item < last) {
                const int pair = ids_dst[first + item];
                output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                    __float2half(acc[item]);
            }
        }
    }
}

void check_nepq_common(
    const torch::Tensor & indices,
    const torch::Tensor & aux,
    const torch::Tensor & sub_scale,
    const torch::Tensor & neuron_scale,
    const torch::Tensor & table_pool,
    const torch::Tensor & bank_ids,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format) {
    TORCH_CHECK(
        format == kNvq1L || format == kNpq0L ||
        format == kNvq1S || format == kNpq0S,
        "NEPQ base format must be NVQ1-L/NPQ0-L/NVQ1-S/NPQ0-S");
    TORCH_CHECK(neuron_len > 0 && neuron_len % 8 == 0,
                "NEPQ neuron_len must be a positive multiple of 8");
    const int expected_sub_bits = format == kNvq1S ? 4 :
        (format == kNpq0S ? 2 : 3);
    TORCH_CHECK(sub_bits == expected_sub_bits, "NEPQ state width mismatch");
    for (const auto * tensor : {&indices, &aux, &sub_scale, &bank_ids}) {
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == torch::kUInt8 &&
                    tensor->is_contiguous(),
                    "NEPQ packed streams must be CUDA contiguous uint8");
    }
    TORCH_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == torch::kFloat32 &&
                neuron_scale.is_contiguous() && neuron_scale.dim() == 1,
                "NEPQ neuron_scale must be CUDA contiguous float32 rank-1");
    TORCH_CHECK(table_pool.is_cuda() && table_pool.scalar_type() == torch::kInt8 &&
                table_pool.is_contiguous() && table_pool.dim() == 2 &&
                table_pool.size(0) >= 1 && table_pool.size(0) <= 256,
                "NEPQ table pool must be CUDA contiguous int8 [1..256,stride]");
    const int table_stride = format == kNvq1L ? 2048 * 8 :
        (format == kNvq1S ? 1024 * 8 :
         (format == kNpq0L ? kNpq0LTableBytes : kNepq0SCompactTableBytes));
    TORCH_CHECK(table_pool.size(1) == table_stride, "NEPQ table stride mismatch");
    const int64_t N = neuron_scale.numel();
    const int64_t ng = (neuron_len + kGroupSize - 1) / kGroupSize;
    const int64_t nvec = neuron_len / 8;
    const int64_t nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    TORCH_CHECK(bank_ids.dim() == 2 && bank_ids.size(0) == N &&
                bank_ids.size(1) == nsuper,
                "NEPQ bank selectors must have shape [rows,ceil(groups/4)]");
    const int index_bits = format == kNvq1L ? 11 :
        (format == kNvq1S ? 9 : (format == kNpq0L ? 7 : 6));
    const int64_t index_bytes = (N * nvec * index_bits + 7) / 8;
    const bool delta_format = format == kNvq1L || format == kNvq1S;
    const int64_t aux_bytes = delta_format ? (N * ng + 7) / 8 : 0;
    const int64_t sub_bytes = (N * ng * sub_bits + 7) / 8;
    TORCH_CHECK(indices.numel() == index_bytes, "NEPQ index stream length mismatch");
    TORCH_CHECK(aux.numel() == aux_bytes, "NEPQ aux stream length mismatch");
    TORCH_CHECK(sub_scale.numel() == sub_bytes, "NEPQ state stream length mismatch");
}

void check_common(
    const torch::Tensor & indices,
    const torch::Tensor & aux,
    const torch::Tensor & sub_scale,
    const torch::Tensor & neuron_scale,
    const torch::Tensor & codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode) {
    TORCH_CHECK(format >= kNvq1L && format <= kNvq3JscLGroupExec,
                "NVQ format must be in [1,17]");
    TORCH_CHECK(gs == kGroupSize, "NVQ CUDA kernels currently require gs=24");
    TORCH_CHECK(sub_bits >= 1 && sub_bits <= 8, "NVQ sub_bits must be in [1,8]");
    TORCH_CHECK(neuron_len > 0, "NVQ neuron_len must be positive");
    TORCH_CHECK(sign_mode == 0 ||
                (sign_mode == 1 && (format == kNvq2 || format == kNvq2Exec)),
                "index-parity sign mode is valid only for NVQ2");
    for (const auto * tensor : {&indices, &aux, &sub_scale}) {
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == torch::kUInt8 && tensor->is_contiguous(),
                    "NVQ packed streams must be CUDA contiguous uint8");
    }
    TORCH_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == torch::kFloat32 &&
                neuron_scale.is_contiguous(), "NVQ neuron_scale must be CUDA contiguous float32");
    TORCH_CHECK(codebook.is_cuda() && codebook.scalar_type() == torch::kInt8 && codebook.is_contiguous(),
                "NVQ codebook must be CUDA contiguous int8");
    if (format == kNvq1L) {
        TORCH_CHECK(codebook.dim() == 2 && codebook.size(0) == 2048 && codebook.size(1) == 8,
                    "NVQ1-L codebook must have shape [2048,8]");
    } else if (format == kNvq1S) {
        TORCH_CHECK(codebook.dim() == 2 && codebook.size(0) == 1024 && codebook.size(1) == 8,
                    "NVQ1-S codebook must have shape [1024,8]");
        TORCH_CHECK(sub_bits == 4 && sign_mode == 0,
                    "NVQ1-S requires a 4-bit scale and no sign mode");
    } else if (format == kNvq2 || format == kNvq2Exec) {
        TORCH_CHECK(codebook.dim() == 2 && codebook.size(0) == 256 && codebook.size(1) == 8,
                    "NVQ2 codebook must have shape [256,8]");
    } else if (format == kNvq2Jsc || format == kNvq2JscExec) {
        TORCH_CHECK(codebook.dim() == 1 &&
                    (codebook.numel() == kJscMetadataBytes + kJscE8CodebookBytes ||
                     codebook.numel() == kJscMetadataBytes + 2 * kJscE8CodebookBytes ||
                     codebook.numel() == kJscMetadataBytes + 4 * kJscE8CodebookBytes),
                    "NVQ2J codebook metadata has an invalid size");
        TORCH_CHECK(sub_bits == 4 && sign_mode == 0,
                    "NVQ2J requires a 4-bit state and even-parity signs");
    } else if (
        format == kNvq2JscL || format == kNvq2JscXL ||
        format == kNvq2JscXLGroupExec) {
        const int e8_bank_bytes = format == kNvq2JscL
            ? kJscE8Codebook1024Bytes
            : kJscE8Codebook4096Bytes;
        TORCH_CHECK(codebook.dim() == 1 &&
                    (codebook.numel() == kJscMetadataBytes + e8_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 2 * e8_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 4 * e8_bank_bytes),
                    "extended NVQ2J codebook metadata has an invalid size");
        TORCH_CHECK(sub_bits == 4 && sign_mode == 0,
                    "extended NVQ2J requires a 4-bit state and even-parity signs");
    } else if (
        format == kNvq3Jsc || format == kNvq3Jsc2 ||
        format == kNvq3Jsc512 || format == kNvq3JscL ||
        format == kNvq3JscLGroupExec) {
        const int d4_bank_bytes = format == kNvq3Jsc512
            ? kJscD4Codebook512Bytes
            : (format == kNvq3JscL || format == kNvq3JscLGroupExec
               ? kJscD4Codebook1024Bytes
               : kJscD4CodebookBytes);
        TORCH_CHECK(codebook.dim() == 1 &&
                    (codebook.numel() == kJscMetadataBytes + d4_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 2 * d4_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 4 * d4_bank_bytes),
                    "NVQ3J codebook metadata has an invalid size");
        if (format == kNvq3Jsc2) {
            TORCH_CHECK(
                codebook.numel() == kJscMetadataBytes + 2 * kJscD4CodebookBytes,
                "NVQ3J analytic-2 requires exactly two D4 banks");
        }
        TORCH_CHECK(sub_bits == 4 && sign_mode == 0,
                    "NVQ3J requires a 4-bit state and even-parity signs");
    } else if (format == kNpq0L) {
        TORCH_CHECK(codebook.dim() == 1 && codebook.numel() == kNpq0LTableBytes,
                    "NPQ0-L codebook metadata must contain 832 bytes");
        TORCH_CHECK(sub_bits == 3 && sign_mode == 0,
                    "NPQ0-L requires a 3-bit state and no sign stream");
    } else if (format == kNpq0S) {
        TORCH_CHECK(codebook.dim() == 1 && codebook.numel() == kNpq0STableBytes,
                    "NPQ0-S expanded PQ3+3 runtime LUT must contain 2112 bytes");
        TORCH_CHECK(sub_bits == 2 && sign_mode == 0,
                    "NPQ0-S requires a 2-bit state and no sign stream");
    } else {
        TORCH_CHECK(codebook.dim() == 2 && codebook.size(0) == 256 && codebook.size(1) == 4,
                    "NVQ3 codebook must have shape [256,4]");
    }

    const int64_t N = neuron_scale.numel();
    const int64_t ng = (neuron_len + gs - 1) / gs;
    const int64_t nvec = (neuron_len + (is_d4_format(format) ? 3 : 7)) /
                         (is_d4_format(format) ? 4 : 8);
    const int64_t nsign = (neuron_len + 7) / 8;
    const bool exec_layout = format == kNvq2Exec || format == kNvq2JscExec;
    const bool group_exec_layout =
        format == kNvq2JscXLGroupExec || format == kNvq3JscLGroupExec;
    const int index_bits = format_index_bits(format);
    const int64_t index_bytes = group_exec_layout
        ? N * ng * (format == kNvq2JscXLGroupExec ? 8 : 12)
        : (N * nvec * index_bits + 7) / 8;
    const bool delta_format = format == kNvq1L || format == kNvq1S;
    const bool no_aux_format = format == kNpq0L || format == kNpq0S;
    const int aux_bits = delta_format ? 1 : (no_aux_format ? 0 : 7);
    const int64_t aux_count = delta_format ? N * ng :
        (no_aux_format ? 0 : N * nsign);
    const int64_t aux_bytes = exec_layout || group_exec_layout
        ? 0
        : (aux_count * aux_bits + 7) / 8;
    const int64_t sub_bytes = (N * ng * sub_bits + 7) / 8;
    TORCH_CHECK(indices.numel() == index_bytes, "NVQ index stream length mismatch");
    TORCH_CHECK(aux.numel() == aux_bytes, "NVQ aux stream length mismatch");
    TORCH_CHECK(sub_scale.numel() == sub_bytes, "NVQ sub-scale stream length mismatch");
}

template <typename Launch>
void launch_by_format(int format, Launch && launch) {
    switch (format) {
        case kNvq1L: launch(std::integral_constant<int, kNvq1L>{}); break;
        case kNvq2: launch(std::integral_constant<int, kNvq2>{}); break;
        case kNvq3: launch(std::integral_constant<int, kNvq3>{}); break;
        case kNvq2Exec: launch(std::integral_constant<int, kNvq2Exec>{}); break;
        case kNvq2Jsc: launch(std::integral_constant<int, kNvq2Jsc>{}); break;
        case kNvq2JscExec: launch(std::integral_constant<int, kNvq2JscExec>{}); break;
        case kNpq0L: launch(std::integral_constant<int, kNpq0L>{}); break;
        case kNvq1S: launch(std::integral_constant<int, kNvq1S>{}); break;
        case kNpq0S: launch(std::integral_constant<int, kNpq0S>{}); break;
        case kNvq3Jsc: launch(std::integral_constant<int, kNvq3Jsc>{}); break;
        case kNvq3Jsc2: launch(std::integral_constant<int, kNvq3Jsc2>{}); break;
        case kNvq3Jsc512:
            launch(std::integral_constant<int, kNvq3Jsc512>{});
            break;
        case kNvq2JscL:
            launch(std::integral_constant<int, kNvq2JscL>{});
            break;
        case kNvq2JscXL:
            launch(std::integral_constant<int, kNvq2JscXL>{});
            break;
        case kNvq3JscL:
            launch(std::integral_constant<int, kNvq3JscL>{});
            break;
        case kNvq2JscXLGroupExec:
            launch(std::integral_constant<int, kNvq2JscXLGroupExec>{});
            break;
        case kNvq3JscLGroupExec:
            launch(std::integral_constant<int, kNvq3JscLGroupExec>{});
            break;
        default: TORCH_CHECK(false, "unsupported NVQ format");
    }
}

template <typename Launch>
void launch_nepq_by_format(int format, Launch && launch) {
    switch (format) {
        case kNvq1L: launch(std::integral_constant<int, kNvq1L>{}); break;
        case kNpq0L: launch(std::integral_constant<int, kNpq0L>{}); break;
        case kNvq1S: launch(std::integral_constant<int, kNvq1S>{}); break;
        case kNpq0S: launch(std::integral_constant<int, kNpq0S>{}); break;
        default: TORCH_CHECK(false, "unsupported NEPQ base format");
    }
}

int aligned_group_rows_per_block(int format) {
    const char * value = std::getenv(
        format == kNvq2JscXLGroupExec
            ? "MFQ_NVQ_GROUP_ROWS_E8"
            : "MFQ_NVQ_GROUP_ROWS_D4");
    if (value == nullptr) value = std::getenv("MFQ_NVQ_GROUP_ROWS");
    if (value != nullptr && value[0] == '4') return 4;
    if (value != nullptr && value[0] == '2') return 2;
    return 1;
}

int aligned_group_row_tile(int format) {
    const char * value = std::getenv(
        format == kNvq2JscXLGroupExec
            ? "MFQ_NVQ_ROW_TILE_E8"
            : "MFQ_NVQ_ROW_TILE_D4");
    if (value == nullptr) value = std::getenv("MFQ_NVQ_ROW_TILE");
    return value == nullptr ? 0 : std::atoi(value);
}

bool aligned_group_e8_down_w32r1_enabled() {
    static const bool value = [] {
        const char * text = std::getenv("MFQ_NVQ_E8_STAGE_DOWN_W32R1");
        return text == nullptr || text[0] != '0';
    }();
    return value;
}

template <
    int FORMAT, int NWARPS, int ROWS_PER_WARP, bool MULTI2,
    bool STAGE_E8_THREE_BANKS = false, bool ADD_RESIDUAL = false>
void launch_aligned_group_row_tiled_m1(
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    cudaStream_t stream,
    const __half * residual = nullptr) {
    static_assert(!ADD_RESIDUAL || !MULTI2);
    const int total_rows = MULTI2 ? first.N + second.N : first.N;
    const int rows_per_block = NWARPS * ROWS_PER_WARP;
    size_t shared_bytes = 0;
    if constexpr (STAGE_E8_THREE_BANKS) {
        shared_bytes =
            3 * kJscE8Codebook4096Bytes + kJscE8PartialBytes;
        static const cudaError_t attribute_status = cudaFuncSetAttribute(
            aligned_group_row_tiled_m1_kernel<
                FORMAT, NWARPS, ROWS_PER_WARP, MULTI2, true, ADD_RESIDUAL>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            attribute_status == cudaSuccess,
            "failed to opt in to the E8 row-tiled shared-memory size");
    } else {
        const int k_pad = first.ng * kGroupSize;
        const int scale_offset = (k_pad + 3) & ~3;
        shared_bytes = static_cast<size_t>(scale_offset) +
            static_cast<size_t>(first.ng) * sizeof(float);
        if constexpr (FORMAT == kNvq3JscLGroupExec) {
            shared_bytes += static_cast<size_t>(first.codebook_nbytes);
            if constexpr (MULTI2) {
                shared_bytes += static_cast<size_t>(second.codebook_nbytes);
            }
        }
    }
    aligned_group_row_tiled_m1_kernel<
        FORMAT, NWARPS, ROWS_PER_WARP, MULTI2,
        STAGE_E8_THREE_BANKS, ADD_RESIDUAL>
        <<<(total_rows + rows_per_block - 1) / rows_per_block,
           dim3(32, NWARPS), shared_bytes, stream>>>(
            first, second, qx, xscale, output, residual);
}

template <int FORMAT, bool ADD_RESIDUAL = false>
void launch_aligned_group_m1(
    const NvqDeviceWeight & weight,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    cudaStream_t stream,
    const __half * residual = nullptr) {
    const int tile = aligned_group_row_tile(FORMAT);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        if (tile == 3243 && weight.codebook_nbytes >=
                kJscMetadataBytes + 4 * kJscE8Codebook4096Bytes) {
            if (weight.N >= 12288) {
                launch_aligned_group_row_tiled_m1<
                    FORMAT, 32, 4, false, true, ADD_RESIDUAL>(
                    weight, weight, qx, xscale, output, stream, residual);
                return;
            }
            if (weight.N >= 4096 && aligned_group_e8_down_w32r1_enabled()) {
                launch_aligned_group_row_tiled_m1<
                    FORMAT, 32, 1, false, true, ADD_RESIDUAL>(
                    weight, weight, qx, xscale, output, stream, residual);
                return;
            }
        }
        if (tile == 164) {
            launch_aligned_group_row_tiled_m1<
                FORMAT, 16, 4, false, false, ADD_RESIDUAL>(
                weight, weight, qx, xscale, output, stream, residual);
            return;
        }
        if (tile == 324) {
            launch_aligned_group_row_tiled_m1<
                FORMAT, 32, 4, false, false, ADD_RESIDUAL>(
                weight, weight, qx, xscale, output, stream, residual);
            return;
        }
    } else {
        if (tile == 162) {
            launch_aligned_group_row_tiled_m1<
                FORMAT, 16, 2, false, false, ADD_RESIDUAL>(
                weight, weight, qx, xscale, output, stream, residual);
            return;
        }
        if (tile == 322) {
            launch_aligned_group_row_tiled_m1<
                FORMAT, 32, 2, false, false, ADD_RESIDUAL>(
                weight, weight, qx, xscale, output, stream, residual);
            return;
        }
    }
    const int rows = aligned_group_rows_per_block(FORMAT);
    if (rows == 4) {
        aligned_group_gemv_m1_kernel<FORMAT, 4, 4, ADD_RESIDUAL>
            <<<(weight.N + 3) / 4, dim3(32, 16), 0, stream>>>(
                weight, qx, xscale, output, residual);
    } else if (rows == 2) {
        aligned_group_gemv_m1_kernel<FORMAT, 4, 2, ADD_RESIDUAL>
            <<<(weight.N + 1) / 2, dim3(32, 8), 0, stream>>>(
                weight, qx, xscale, output, residual);
    } else {
        aligned_group_gemv_m1_kernel<FORMAT, 4, 1, ADD_RESIDUAL>
            <<<weight.N, dim3(32, 4), 0, stream>>>(
                weight, qx, xscale, output, residual);
    }
}

template <int FORMAT>
void launch_aligned_group_multi2_m1(
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    cudaStream_t stream) {
    const int blocks = first.N + second.N;
    const int tile = aligned_group_row_tile(FORMAT);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        if (tile == 3243 &&
            (first.N >= 12288 || second.N >= 12288) &&
            first.codebook_nbytes >=
                kJscMetadataBytes + 4 * kJscE8Codebook4096Bytes &&
            second.codebook_nbytes >=
                kJscMetadataBytes + 4 * kJscE8Codebook4096Bytes) {
            launch_aligned_group_row_tiled_m1<FORMAT, 32, 4, true, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
        if (tile == 164) {
            launch_aligned_group_row_tiled_m1<FORMAT, 16, 4, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
        if (tile == 324) {
            launch_aligned_group_row_tiled_m1<FORMAT, 32, 4, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
    } else {
        if (tile == 162) {
            launch_aligned_group_row_tiled_m1<FORMAT, 16, 2, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
        if (tile == 322) {
            launch_aligned_group_row_tiled_m1<FORMAT, 32, 2, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
    }
    const int rows = aligned_group_rows_per_block(FORMAT);
    if (rows == 4) {
        aligned_group_multi2_m1_kernel<FORMAT, 4, 4>
            <<<(blocks + 3) / 4, dim3(32, 16), 0, stream>>>(
                first, second, qx, xscale, output);
    } else if (rows == 2) {
        aligned_group_multi2_m1_kernel<FORMAT, 4, 2>
            <<<(blocks + 1) / 2, dim3(32, 8), 0, stream>>>(
                first, second, qx, xscale, output);
    } else {
        aligned_group_multi2_m1_kernel<FORMAT, 4, 1>
            <<<blocks, dim3(32, 4), 0, stream>>>(
                first, second, qx, xscale, output);
    }
}

template <int NWARPS>
void launch_m1_vec8_by_format(
    int format,
    const torch::Tensor & indices,
    const torch::Tensor & aux,
    const torch::Tensor & sub_scale,
    const torch::Tensor & neuron_scale,
    const torch::Tensor & codebook,
    const torch::Tensor & qx,
    const torch::Tensor & xscale,
    torch::Tensor & output,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    cudaStream_t stream) {
    if (format == kNvq2 && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq2, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq2Exec && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq2Exec, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq2Jsc && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq2Jsc, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq2JscExec && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq2JscExec, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq3 && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq3, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq3Jsc && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq3Jsc, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq3Jsc2 && sub_bits == 4) {
        nvq3j2_gemv_m1_group_kernel<NWARPS><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            N, ng, nvec, nsign);
        return;
    }
    if (format == kNvq2JscXLGroupExec || format == kNvq3JscLGroupExec) {
        const NvqDeviceWeight weight{
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
            static_cast<int>(codebook.numel()),
            N, ng, nvec, nsign, sub_bits, sign_mode};
        if (format == kNvq2JscXLGroupExec) {
            launch_aligned_group_m1<kNvq2JscXLGroupExec>(
                weight, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<at::Half>()), stream);
        } else {
            launch_aligned_group_m1<kNvq3JscLGroupExec>(
                weight, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<at::Half>()), stream);
        }
        return;
    }
    launch_by_format(format, [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nvq_gemv_m1_vec8_kernel<F, NWARPS><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            N, ng, nvec, nsign, sub_bits, sign_mode);
    });
}

template <int NWARPS, int MAX_M>
void launch_batch_vec8_by_format(
    int format,
    const torch::Tensor & indices,
    const torch::Tensor & aux,
    const torch::Tensor & sub_scale,
    const torch::Tensor & neuron_scale,
    const torch::Tensor & codebook,
    const torch::Tensor & qx,
    const torch::Tensor & xscale,
    torch::Tensor & output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    cudaStream_t stream) {
    launch_by_format(format, [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nvq_gemv_batch_vec8_kernel<F, NWARPS, MAX_M><<<N, dim3(32, NWARPS), 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            M, N, ng, nvec, nsign, sub_bits, sign_mode);
    });
}

void launch_selected_batch_vec8(
    int format,
    const torch::Tensor & indices,
    const torch::Tensor & aux,
    const torch::Tensor & sub_scale,
    const torch::Tensor & neuron_scale,
    const torch::Tensor & codebook,
    const torch::Tensor & qx,
    const torch::Tensor & xscale,
    torch::Tensor & output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    cudaStream_t stream) {
#define NVQ_SELECTED_BATCH(NW, MAX_M)                                                   \
    launch_batch_vec8_by_format<NW, MAX_M>(                                            \
        format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale, output,  \
        M, N, ng, nvec, nsign, sub_bits, sign_mode, stream)
    const bool compact_s = format == kNvq1S || format == kNpq0S;
    const bool compact_s_small = compact_s && N <= 1024;
    if (M <= 2) {
        if (compact_s_small) {
            NVQ_SELECTED_BATCH(8, 2);
        } else if (compact_s) {
            NVQ_SELECTED_BATCH(4, 2);
        } else if (is_e8_format(format)) {
            if (N <= 1024) {
                NVQ_SELECTED_BATCH(8, 2);
            } else {
                NVQ_SELECTED_BATCH(4, 2);
            }
        } else if (is_d4_format(format)) {
            NVQ_SELECTED_BATCH(2, 2);
        } else {
            NVQ_SELECTED_BATCH(8, 2);
        }
    } else if (M <= 4) {
        if (compact_s_small) {
            NVQ_SELECTED_BATCH(8, 4);
        } else if (compact_s) {
            NVQ_SELECTED_BATCH(4, 4);
        } else if (is_e8_format(format)) {
            if (N <= 1024) {
                NVQ_SELECTED_BATCH(8, 4);
            } else {
                NVQ_SELECTED_BATCH(4, 4);
            }
        } else if (is_d4_format(format)) {
            NVQ_SELECTED_BATCH(4, 4);
        } else {
            NVQ_SELECTED_BATCH(8, 4);
        }
    } else if (M <= 8 && compact_s && (compact_s_small || ng >= 384)) {
        NVQ_SELECTED_BATCH(8, 8);
    } else if (M <= 8 && compact_s) {
        NVQ_SELECTED_BATCH(4, 8);
    } else if (M <= 8 && is_e8_format(format)) {
        if (N <= 1024) {
            NVQ_SELECTED_BATCH(8, 8);
        } else {
            NVQ_SELECTED_BATCH(4, 8);
        }
    } else if (M <= 8) {
        NVQ_SELECTED_BATCH(8, 8);
    } else if (M <= 12 && compact_s_small) {
        NVQ_SELECTED_BATCH(8, 12);
    } else if (M <= 12 && compact_s) {
        NVQ_SELECTED_BATCH(4, 12);
    } else if (M <= 12 && is_e8_format(format)) {
        if (M == 12 && N <= 1024) {
            NVQ_SELECTED_BATCH(8, 12);
        } else {
            NVQ_SELECTED_BATCH(4, 12);
        }
    } else if (M <= 12) {
        NVQ_SELECTED_BATCH(8, 12);
    } else if (compact_s_small) {
        NVQ_SELECTED_BATCH(8, 16);
    } else if (compact_s) {
        NVQ_SELECTED_BATCH(4, 16);
    } else if (is_e8_format(format)) {
        if (N <= 1024) {
            NVQ_SELECTED_BATCH(8, 16);
        } else {
            NVQ_SELECTED_BATCH(4, 16);
        }
    } else {
        NVQ_SELECTED_BATCH(8, 16);
    }
#undef NVQ_SELECTED_BATCH
}

void launch_selected_m1_vec8(
    int format,
    const torch::Tensor & indices,
    const torch::Tensor & aux,
    const torch::Tensor & sub_scale,
    const torch::Tensor & neuron_scale,
    const torch::Tensor & codebook,
    const torch::Tensor & qx,
    const torch::Tensor & xscale,
    torch::Tensor & output,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    cudaStream_t stream) {
    if (format == kNvq1S && N <= 2048) {
        launch_m1_vec8_by_format<8>(
            format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,
            output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
    } else if (format == kNpq0S && N <= 1024) {
        launch_m1_vec8_by_format<2>(
            format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,
            output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
    } else {
        launch_m1_vec8_by_format<4>(
            format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,
            output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
    }
}

NvqDeviceWeight make_device_weight(
    const torch::Tensor & indices,
    const torch::Tensor & aux,
    const torch::Tensor & sub_scale,
    const torch::Tensor & neuron_scale,
    const torch::Tensor & codebook,
    int neuron_len,
    int gs,
    int sub_bits,
    int format,
    int sign_mode) {
    const int ng = (neuron_len + gs - 1) / gs;
    return NvqDeviceWeight{
        indices.data_ptr<uint8_t>(), indices.numel(),
        aux.data_ptr<uint8_t>(), aux.numel(),
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
        neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
        static_cast<int>(codebook.numel()),
        static_cast<int>(neuron_scale.numel()), ng,
        (neuron_len + (is_d4_format(format) ? 3 : 7)) /
            (is_d4_format(format) ? 4 : 8),
        (neuron_len + 7) / 8, sub_bits, sign_mode};
}

template <int FORMAT, int NWARPS, int MAX_M, bool FAST_NVQ2_S4 = false>
void launch_multi2_vec8(
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const torch::Tensor & qx,
    const torch::Tensor & xscale,
    torch::Tensor & output,
    int M,
    cudaStream_t stream) {
    nvq_gemv_multi2_vec8_kernel<FORMAT, NWARPS, MAX_M, FAST_NVQ2_S4>
        <<<first.N + second.N, dim3(32, NWARPS), 0, stream>>>(
            first, second, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()), M);
}

void launch_selected_multi2_vec8(
    int format,
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const torch::Tensor & qx,
    const torch::Tensor & xscale,
    torch::Tensor & output,
    int M,
    cudaStream_t stream) {
#define NVQ_MULTI2(F, NW, MAX_M) \
    launch_multi2_vec8<F, NW, MAX_M>(first, second, qx, xscale, output, M, stream)
#define NVQ_MULTI2_FAST(F, NW, MAX_M) \
    launch_multi2_vec8<F, NW, MAX_M, true>(first, second, qx, xscale, output, M, stream)
    const bool fast_nvq2_s4 =
        (format == kNvq2 || format == kNvq2Exec ||
         format == kNvq2Jsc || format == kNvq2JscExec) &&
        first.sub_bits == 4 && second.sub_bits == 4;
    if (M == 1) {
        switch (format) {
            case kNvq1L: NVQ_MULTI2(kNvq1L, 4, 1); break;
            case kNvq2:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2, 4, 1);
                else NVQ_MULTI2(kNvq2, 4, 1);
                break;
            case kNvq3: NVQ_MULTI2(kNvq3, 4, 1); break;
            case kNvq2Exec:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2Exec, 4, 1);
                else NVQ_MULTI2(kNvq2Exec, 4, 1);
                break;
            case kNvq2Jsc:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2Jsc, 4, 1);
                else NVQ_MULTI2(kNvq2Jsc, 4, 1);
                break;
            case kNvq2JscExec:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2JscExec, 4, 1);
                else NVQ_MULTI2(kNvq2JscExec, 4, 1);
                break;
            case kNpq0L: NVQ_MULTI2(kNpq0L, 4, 1); break;
            case kNvq1S: NVQ_MULTI2(kNvq1S, 4, 1); break;
            case kNpq0S: NVQ_MULTI2(kNpq0S, 4, 1); break;
            case kNvq3Jsc: NVQ_MULTI2(kNvq3Jsc, 4, 1); break;
            case kNvq3Jsc2: NVQ_MULTI2(kNvq3Jsc2, 4, 1); break;
            case kNvq3Jsc512: NVQ_MULTI2(kNvq3Jsc512, 4, 1); break;
            case kNvq2JscL: NVQ_MULTI2(kNvq2JscL, 4, 1); break;
            case kNvq2JscXL: NVQ_MULTI2(kNvq2JscXL, 4, 1); break;
            case kNvq3JscL: NVQ_MULTI2(kNvq3JscL, 4, 1); break;
            case kNvq2JscXLGroupExec:
                launch_aligned_group_multi2_m1<kNvq2JscXLGroupExec>(
                    first, second, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                    reinterpret_cast<__half *>(output.data_ptr<at::Half>()), stream);
                break;
            case kNvq3JscLGroupExec:
                launch_aligned_group_multi2_m1<kNvq3JscLGroupExec>(
                    first, second, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                    reinterpret_cast<__half *>(output.data_ptr<at::Half>()), stream);
                break;
        }
    } else if (M <= 4) {
        if (format == kNvq1L || format == kNvq1S) {
            if (format == kNvq1S) NVQ_MULTI2(kNvq1S, 8, 4);
            else NVQ_MULTI2(kNvq1L, 8, 4);
        } else if (is_e8_format(format)) {
            if (format == kNvq2Exec) NVQ_MULTI2(kNvq2Exec, 4, 4);
            else if (format == kNvq2Jsc) NVQ_MULTI2(kNvq2Jsc, 4, 4);
            else if (format == kNvq2JscExec) NVQ_MULTI2(kNvq2JscExec, 4, 4);
            else if (format == kNvq2JscL) NVQ_MULTI2(kNvq2JscL, 4, 4);
            else if (format == kNvq2JscXL) NVQ_MULTI2(kNvq2JscXL, 4, 4);
            else if (format == kNvq2JscXLGroupExec) {
                NVQ_MULTI2(kNvq2JscXLGroupExec, 4, 4);
            }
            else NVQ_MULTI2(kNvq2, 4, 4);
        } else if (format == kNpq0L || format == kNpq0S) {
            if (format == kNpq0S) NVQ_MULTI2(kNpq0S, 4, 4);
            else NVQ_MULTI2(kNpq0L, 4, 4);
        } else if (M <= 2) {
            if (format == kNvq3Jsc2) NVQ_MULTI2(kNvq3Jsc2, 2, 4);
            else if (format == kNvq3Jsc) NVQ_MULTI2(kNvq3Jsc, 2, 4);
            else if (format == kNvq3Jsc512) NVQ_MULTI2(kNvq3Jsc512, 2, 4);
            else if (format == kNvq3JscL) NVQ_MULTI2(kNvq3JscL, 2, 4);
            else if (format == kNvq3JscLGroupExec) {
                NVQ_MULTI2(kNvq3JscLGroupExec, 2, 4);
            }
            else NVQ_MULTI2(kNvq3, 2, 4);
        } else {
            if (format == kNvq3Jsc2) NVQ_MULTI2(kNvq3Jsc2, 4, 4);
            else if (format == kNvq3Jsc) NVQ_MULTI2(kNvq3Jsc, 4, 4);
            else if (format == kNvq3Jsc512) NVQ_MULTI2(kNvq3Jsc512, 4, 4);
            else if (format == kNvq3JscL) NVQ_MULTI2(kNvq3JscL, 4, 4);
            else if (format == kNvq3JscLGroupExec) {
                NVQ_MULTI2(kNvq3JscLGroupExec, 4, 4);
            }
            else NVQ_MULTI2(kNvq3, 4, 4);
        }
    } else if (format == kNvq1L || format == kNvq1S) {
        if (format == kNvq1S) NVQ_MULTI2(kNvq1S, 8, 8);
        else NVQ_MULTI2(kNvq1L, 8, 8);
    } else if (is_e8_format(format)) {
        if (format == kNvq2Exec) NVQ_MULTI2(kNvq2Exec, 4, 8);
        else if (format == kNvq2Jsc) NVQ_MULTI2(kNvq2Jsc, 4, 8);
        else if (format == kNvq2JscExec) NVQ_MULTI2(kNvq2JscExec, 4, 8);
        else if (format == kNvq2JscL) NVQ_MULTI2(kNvq2JscL, 4, 8);
        else if (format == kNvq2JscXL) NVQ_MULTI2(kNvq2JscXL, 4, 8);
        else if (format == kNvq2JscXLGroupExec) {
            NVQ_MULTI2(kNvq2JscXLGroupExec, 4, 8);
        }
        else NVQ_MULTI2(kNvq2, 4, 8);
    } else if (format == kNpq0L || format == kNpq0S) {
        if (format == kNpq0S) NVQ_MULTI2(kNpq0S, 4, 8);
        else NVQ_MULTI2(kNpq0L, 4, 8);
    } else {
        if (format == kNvq3Jsc2) NVQ_MULTI2(kNvq3Jsc2, 8, 8);
        else if (format == kNvq3Jsc) NVQ_MULTI2(kNvq3Jsc, 8, 8);
        else if (format == kNvq3Jsc512) NVQ_MULTI2(kNvq3Jsc512, 8, 8);
        else if (format == kNvq3JscL) NVQ_MULTI2(kNvq3JscL, 8, 8);
        else if (format == kNvq3JscLGroupExec) {
            NVQ_MULTI2(kNvq3JscLGroupExec, 8, 8);
        }
        else NVQ_MULTI2(kNvq3, 8, 8);
    }
#undef NVQ_MULTI2
#undef NVQ_MULTI2_FAST
}

}  // namespace

torch::Tensor nepq_dequant_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor table_pool,
    torch::Tensor bank_ids,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    const int N = static_cast<int>(neuron_scale.numel());
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    auto output = torch::empty({N, K}, neuron_scale.options().dtype(torch::kFloat16));
    const int block = 256;
    const int grid = static_cast<int>(std::min<int64_t>(
        (static_cast<int64_t>(N) * nsign + block - 1) / block, 65535));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nepq_dequant_kernel<F><<<grid, block, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
            neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),
            bank_ids.data_ptr<uint8_t>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            N, K, ng, nvec, nsign, nsuper, table_stride,
            static_cast<int>(sub_bits), 0);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NEPQ dequant kernel launch failed");
    return output;
}

torch::Tensor nepq_gemv_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor table_pool,
    torch::Tensor bank_ids,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    torch::Tensor qx,
    torch::Tensor xscale) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
                x.is_contiguous() && x.dim() == 2,
                "NEPQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    TORCH_CHECK(x.size(1) == K && M >= 1 && M <= 16,
                "NEPQ GEMV input must be [M,K] with M in [1,16]");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == torch::kInt8 &&
                qx.is_contiguous() && qx.numel() >= static_cast<int64_t>(M) * K_pad,
                "NEPQ qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == torch::kFloat32 &&
                xscale.is_contiguous() && xscale.numel() >= static_cast<int64_t>(M) * ng,
                "NEPQ xscale workspace mismatch");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    auto output = torch::empty({M, N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);

#define NEPQ_GEMV_LAUNCH(NWARPS_VALUE, MAX_M_VALUE)                               \
    nepq_gemv_batch_vec8_kernel<F, NWARPS_VALUE, MAX_M_VALUE><<<                 \
        N, dim3(32, NWARPS_VALUE), 0, stream>>>(                                  \
        indices.data_ptr<uint8_t>(), indices.numel(),                             \
        aux.data_ptr<uint8_t>(), aux.numel(),                                     \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                         \
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),            \
        bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                      \
        xscale.data_ptr<float>(),                                                 \
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()),                  \
        M, N, ng, nvec, nsign, nsuper, table_stride,                              \
        static_cast<int>(sub_bits), 0)
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        const int nwarps = (format == kNvq1S && N <= 2048) ? 8 :
            ((format == kNpq0S && N <= 1024) ? 2 : 4);
        if (M == 1) {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 1);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 1);
            else NEPQ_GEMV_LAUNCH(4, 1);
        } else if (M <= 2) {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 2);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 2);
            else NEPQ_GEMV_LAUNCH(4, 2);
        } else if (M <= 4) {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 4);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 4);
            else NEPQ_GEMV_LAUNCH(4, 4);
        } else if (M <= 8) {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 8);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 8);
            else NEPQ_GEMV_LAUNCH(4, 8);
        } else {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 16);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 16);
            else NEPQ_GEMV_LAUNCH(4, 16);
        }
    });
#undef NEPQ_GEMV_LAUNCH
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NEPQ GEMV kernel launch failed");
    return output;
}

torch::Tensor nepq_mmq_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor table_pool,
    torch::Tensor bank_ids,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    torch::Tensor qx,
    torch::Tensor xscale) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
                x.is_contiguous() && x.dim() == 2,
                "NEPQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    TORCH_CHECK(x.size(1) == K && M >= 4 && M <= 64,
                "NEPQ MMQ input must be [M,K] with M in [4,64]");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == torch::kInt8 &&
                qx.is_contiguous() && qx.numel() >= static_cast<int64_t>(M) * K_pad,
                "NEPQ qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == torch::kFloat32 &&
                xscale.is_contiguous() && xscale.numel() >= static_cast<int64_t>(M) * ng,
                "NEPQ xscale workspace mismatch");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    auto output = torch::empty({M, N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);
    const dim3 block(32, 8);
#define NEPQ_MMQ_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE)                           \
    nepq_mmq_mma24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<<                  \
        dim3((N + 63) / 64, 1), block, 0, stream>>>(                             \
        indices.data_ptr<uint8_t>(), indices.numel(),                            \
        aux.data_ptr<uint8_t>(), aux.numel(),                                    \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                        \
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),           \
        bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                     \
        xscale.data_ptr<float>(),                                                \
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()),                 \
        M, N, ng, nvec, nsign, nsuper, table_stride,                             \
        static_cast<int>(sub_bits), 0)
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M <= 16) {
            if (M == 16 && N % 64 == 0) NEPQ_MMQ_LAUNCH(1, false);
            else NEPQ_MMQ_LAUNCH(1, true);
        } else if (M <= 32) {
            if (M == 32 && N % 64 == 0) NEPQ_MMQ_LAUNCH(2, false);
            else NEPQ_MMQ_LAUNCH(2, true);
        } else if (M <= 48) {
            if (M == 48 && N % 64 == 0) NEPQ_MMQ_LAUNCH(3, false);
            else NEPQ_MMQ_LAUNCH(3, true);
        } else {
            if (M == 64 && N % 64 == 0) NEPQ_MMQ_LAUNCH(4, false);
            else NEPQ_MMQ_LAUNCH(4, true);
        }
    });
#undef NEPQ_MMQ_LAUNCH
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NEPQ MMQ kernel launch failed");
    return output;
}

torch::Tensor nepq_gemm_f16_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor table_pool,
    torch::Tensor bank_ids,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
                x.is_contiguous() && x.dim() == 2,
                "NEPQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    TORCH_CHECK(x.size(1) == K && M >= 16 && M <= 256,
                "NEPQ online GEMM input must be [M,K] with M in [16,256]");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    auto output = torch::empty({M, N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(32, 8);
#define NEPQ_GEMM_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE)                          \
    nepq_gemm_f16_gs24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<<              \
        dim3((N + 63) / 64, (M + 16 * MTILES_VALUE - 1) / (16 * MTILES_VALUE)),  \
        block, 0, stream>>>(                                                       \
        indices.data_ptr<uint8_t>(), indices.numel(),                             \
        aux.data_ptr<uint8_t>(), aux.numel(),                                     \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                         \
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),            \
        bank_ids.data_ptr<uint8_t>(),                                             \
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),                 \
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()),                  \
        M, N, K, ng, nvec, nsign, nsuper, table_stride,                           \
        static_cast<int>(sub_bits), 0)
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M <= 16) {
            if (M == 16 && N % 64 == 0 && ng % 4 == 0) NEPQ_GEMM_LAUNCH(1, false);
            else NEPQ_GEMM_LAUNCH(1, true);
        } else if (M <= 32) {
            if (M == 32 && N % 64 == 0 && ng % 4 == 0) NEPQ_GEMM_LAUNCH(2, false);
            else NEPQ_GEMM_LAUNCH(2, true);
        } else if (M <= 64) {
            if (M == 64 && N % 64 == 0 && ng % 4 == 0) NEPQ_GEMM_LAUNCH(4, false);
            else NEPQ_GEMM_LAUNCH(4, true);
        } else {
            if (M % 128 == 0 && N % 64 == 0 && ng % 4 == 0) {
                NEPQ_GEMM_LAUNCH(8, false);
            } else {
                NEPQ_GEMM_LAUNCH(8, true);
            }
        }
    });
#undef NEPQ_GEMM_LAUNCH
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NEPQ online GEMM kernel launch failed");
    return output;
}

torch::Tensor nepq_moe_grouped_matmul_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor table_pool,
    torch::Tensor bank_ids,
    torch::Tensor grouped_table_pool,
    torch::Tensor x,
    torch::Tensor ids,
    int64_t n_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    torch::Tensor out,
    torch::Tensor qx,
    torch::Tensor xscale,
    torch::Tensor ids_dst,
    torch::Tensor expert_bounds,
    torch::Tensor tile_bounds,
    torch::Tensor tile_experts) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    const int expected_grouped_stride = format == kNpq0S
        ? kNpq0STableBytes
        : static_cast<int>(table_pool.size(1));
    TORCH_CHECK(
        grouped_table_pool.is_cuda() && grouped_table_pool.is_contiguous() &&
        grouped_table_pool.scalar_type() == torch::kInt8 &&
        grouped_table_pool.dim() == 2 &&
        grouped_table_pool.size(0) == table_pool.size(0) &&
        grouped_table_pool.size(1) == expected_grouped_stride,
        "NEPQ grouped table pool shape mismatch");
    TORCH_CHECK(n_experts > 0 && n_experts <= 4096,
                "NEPQ expert count must be in [1,4096]");
    TORCH_CHECK(out_per_expert > 0 &&
                neuron_scale.numel() == n_experts * out_per_expert,
                "NEPQ expert shape does not match the flattened row count");
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() &&
                ids.scalar_type() == torch::kInt32 && ids.dim() == 2,
                "NEPQ route IDs must be CUDA contiguous int32 [T,R]");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == torch::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) && x.size(-1) == neuron_len,
                "NEPQ routed input must be CUDA contiguous fp16 [T,K] or [T,R,K]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    TORCH_CHECK(tokens > 0 && routes > 0, "NEPQ route dimensions must be nonzero");
    TORCH_CHECK((!routed_input && x.size(0) == tokens) ||
                (routed_input && x.size(0) == tokens && x.size(1) == routes),
                "NEPQ input leading dimensions do not match routes");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() &&
                out.scalar_type() == torch::kFloat16 && out.dim() == 3 &&
                out.size(0) == tokens && out.size(1) == routes &&
                out.size(2) == out_per_expert,
                "NEPQ output must be CUDA contiguous fp16 [T,R,O]");

    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    const int input_rows = routed_input ? pairs : tokens;
    TORCH_CHECK(qx.is_cuda() && qx.is_contiguous() &&
                qx.scalar_type() == torch::kInt8 && qx.dim() == 2 &&
                qx.size(0) >= input_rows && qx.size(1) >= K_pad,
                "NEPQ routed qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
                xscale.scalar_type() == torch::kFloat32 && xscale.dim() == 2 &&
                xscale.size(0) >= input_rows && xscale.size(1) >= ng,
                "NEPQ routed xscale workspace mismatch");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    const int grouped_table_stride =
        static_cast<int>(grouped_table_pool.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvq_quantize_x_gs24_kernel<<<dim3(input_rows, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), input_rows, K, ng);

    if (tokens <= 8) {
        constexpr int rows_per_block = 2;
        const int row_blocks = (static_cast<int>(out_per_expert) + rows_per_block - 1) /
            rows_per_block;
#define NEPQ_MOE_SMALL_LAUNCH(NWARPS_VALUE)                                      \
        nepq_moe_mmvq_kernel<F, NWARPS_VALUE, rows_per_block><<<                 \
            dim3(row_blocks, pairs), dim3(32, NWARPS_VALUE), 0, stream>>>(       \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),       \
            bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                 \
            xscale.data_ptr<float>(), ids.data_ptr<int32_t>(),                   \
            nullptr,                                                             \
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(n_experts),                                         \
            static_cast<int>(out_per_expert), ng, nvec, nsign, nsuper,           \
            table_stride, static_cast<int>(sub_bits), routed_input)
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            if (K >= 4096) NEPQ_MOE_SMALL_LAUNCH(8);
            else NEPQ_MOE_SMALL_LAUNCH(4);
        });
#undef NEPQ_MOE_SMALL_LAUNCH
    } else {
        TORCH_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
                    ids_dst.scalar_type() == torch::kInt32 && ids_dst.numel() >= pairs,
                    "NEPQ compact route map is missing");
        TORCH_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
                    expert_bounds.scalar_type() == torch::kInt32 &&
                    expert_bounds.numel() >= n_experts + 1,
                    "NEPQ expert bounds are missing");
        TORCH_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
                    tile_bounds.scalar_type() == torch::kInt32 &&
                    tile_bounds.numel() >= n_experts + 1,
                    "NEPQ tile bounds are missing");
        TORCH_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
                    tile_experts.scalar_type() == torch::kInt32 &&
                    tile_experts.numel() >= pairs,
                    "NEPQ tile expert map is missing");
        constexpr int route_tile = 8;
        const int row_tiles = (static_cast<int>(out_per_expert) + 3) / 4;
        const int max_tiles = (pairs + route_tile - 1) / route_tile +
            static_cast<int>(n_experts);
        const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
        const int blocks = static_cast<int>(std::min<int64_t>(max_tasks, 4096));
#define NEPQ_MOE_GROUPED_LAUNCH()                                                \
        nepq_moe_grouped_tile_kernel<F, route_tile><<<                           \
            blocks, dim3(32, 4), 0, stream>>>(                                   \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), grouped_table_pool.data_ptr<int8_t>(), \
            bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                 \
            xscale.data_ptr<float>(), nullptr, ids_dst.data_ptr<int32_t>(),      \
            expert_bounds.data_ptr<int32_t>(), tile_bounds.data_ptr<int32_t>(),  \
            tile_experts.data_ptr<int32_t>(),                                    \
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()),                \
            routes, static_cast<int>(n_experts),                                 \
            static_cast<int>(n_experts),                                         \
            static_cast<int>(out_per_expert), ng, nvec, nsign, nsuper,           \
            grouped_table_stride, static_cast<int>(sub_bits), max_tiles,         \
            row_tiles,                                                           \
            routed_input)
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            NEPQ_MOE_GROUPED_LAUNCH();
        });
#undef NEPQ_MOE_GROUPED_LAUNCH
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ routed matmul kernel launch failed");
    return out;
}

// Route-compacted online-dequant FP16 Tensor Core path for mixed NVQ pools.
// One task covers up to 16 routed rows and a 64-row output tile.  The route
// map is identical to the production grouped kernels; only the activation
// arithmetic changes from Q8 to FP16 for controlled KLD comparisons.
template <int FORMAT>
__global__ void __launch_bounds__(256, 1) nvq_moe_grouped_f16_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const __half * x,
    const int32_t * expert_local,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    const int32_t * tile_bounds,
    const int32_t * tile_experts,
    __half * output,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    int max_tiles,
    bool routed_input) {
    constexpr int kTileM = 16;
    constexpr int kTileN = 64;
    constexpr int kGroupsPerChunk = 4;
    constexpr int kTileK = kGroupSize * kGroupsPerChunk;
    constexpr int kStrideK = kTileK + 8;
    constexpr int kNFragments = kTileN / 16;
    constexpr int kFineTilesPerTask = kTileM / 8;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int ntiles_n = (out_per_expert + kTileN - 1) / kTileN;
    const int64_t max_tasks = static_cast<int64_t>(max_tiles) * ntiles_n;
    __shared__ __half weight_tile[kTileN][kStrideK];
    __shared__ __half activation_tile[kTileM][kStrideK];
    __shared__ float output_tile[kNFragments][16][16];

    using FragmentA = wmma::fragment<wmma::matrix_a, 16, 16, 16,
                                     __half, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, 16, 16, 16,
                                     __half, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int fine_tile = static_cast<int>(task / ntiles_n);
        const int ntile = static_cast<int>(task -
            static_cast<int64_t>(fine_tile) * ntiles_n);
        if (fine_tile >= tile_bounds[global_experts]) continue;
        const int expert = tile_experts[fine_tile];
        const int local_fine_tile = fine_tile - tile_bounds[expert];
        if (local_fine_tile % kFineTilesPerTask != 0) continue;
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned int>(local_expert) >=
            static_cast<unsigned int>(pool_experts)) continue;
        const int first = expert_bounds[expert] + local_fine_tile * 8;
        const int last = min(first + kTileM, expert_bounds[expert + 1]);
        const int n0 = ntile * kTileN;

        FragmentC accumulator;
        wmma::fill_fragment(accumulator, 0.0f);
        const int chunks = (ng + kGroupsPerChunk - 1) / kGroupsPerChunk;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int group_base = chunk * kGroupsPerChunk;
            const int k_base = group_base * kGroupSize;
            const int row_local = tid / kGroupsPerChunk;
            const int group_local = tid - row_local * kGroupsPerChunk;
            const int local_row = n0 + row_local;
            const int row = local_expert * out_per_expert + local_row;
            const int group = group_base + group_local;
            const bool valid_weight =
                local_row < out_per_expert && group < ng;
            uint32_t state = 0;
            float scale = 0.0f;
            if (valid_weight) {
                const int64_t sub_linear =
                    static_cast<int64_t>(row) * ng + group;
                state = load_packed_bits(
                    sub_scale, sub_linear * sub_bits,
                    sub_bits, sub_scale_nbytes);
                scale = format_scale<FORMAT>(
                    neuron_scale[row], state, codebook);
            }
#pragma unroll
            for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
                const int packed = valid_weight
                    ? decode_chunk4<FORMAT>(
                          indices, indices_nbytes, aux, aux_nbytes, codebook,
                          row, group, quartet, nvec, nsign, ng,
                          sign_mode, state)
                    : 0;
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const int shift = pair * 16;
                    const int value0 = static_cast<int>(
                        static_cast<int8_t>((packed >> shift) & 0xff));
                    const int value1 = static_cast<int>(
                        static_cast<int8_t>((packed >> (shift + 8)) & 0xff));
                    *reinterpret_cast<__half2 *>(
                        &weight_tile[row_local]
                            [group_local * kGroupSize + quartet * 4 + pair * 2]) =
                        __halves2half2(
                            __float2half(scale * static_cast<float>(value0)),
                            __float2half(scale * static_cast<float>(value1)));
                }
            }

            constexpr int kActivationPairs = kTileM * (kTileK / 2);
            for (int index = tid; index < kActivationPairs; index += 256) {
                const int m_local = index / (kTileK / 2);
                const int k_pair = index - m_local * (kTileK / 2);
                const int compact = first + m_local;
                const int k = k_base + k_pair * 2;
                __half2 values = __float2half2_rn(0.0f);
                if (compact < last) {
                    const int pair_index = ids_dst[compact];
                    const int source_row = routed_input
                        ? pair_index : pair_index / routes;
                    if (k + 1 < K) {
                        values = *reinterpret_cast<const __half2 *>(
                            x + static_cast<int64_t>(source_row) * K + k);
                    } else if (k < K) {
                        values = __halves2half2(
                            x[static_cast<int64_t>(source_row) * K + k],
                            __float2half(0.0f));
                    }
                }
                *reinterpret_cast<__half2 *>(
                    &activation_tile[m_local][k_pair * 2]) = values;
            }
            __syncthreads();

            if (warp < kNFragments) {
#pragma unroll
                for (int k_local = 0; k_local < kTileK; k_local += 16) {
                    FragmentA activation_fragment;
                    FragmentB weight_fragment;
                    wmma::load_matrix_sync(
                        activation_fragment, &activation_tile[0][k_local],
                        kStrideK);
                    wmma::load_matrix_sync(
                        weight_fragment, &weight_tile[warp * 16][k_local],
                        kStrideK);
                    wmma::mma_sync(
                        accumulator, activation_fragment,
                        weight_fragment, accumulator);
                }
            }
            __syncthreads();
        }

        if (warp < kNFragments) {
            wmma::store_matrix_sync(
                &output_tile[warp][0][0], accumulator,
                16, wmma::mem_row_major);
        }
        __syncthreads();
        if (warp < kNFragments) {
            for (int element = lane; element < 256; element += 32) {
                const int m_local = element / 16;
                const int n_local = element - m_local * 16;
                const int compact = first + m_local;
                const int local_row = n0 + warp * 16 + n_local;
                if (compact < last && local_row < out_per_expert) {
                    const int pair_index = ids_dst[compact];
                    output[static_cast<int64_t>(pair_index) * out_per_expert +
                           local_row] =
                        __float2half(output_tile[warp][m_local][n_local]);
                }
            }
        }
        __syncthreads();
    }
}

template <int FORMAT>
__global__ void __launch_bounds__(256, 1) nepq_moe_grouped_f16_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const __half * x,
    const int32_t * expert_local,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    const int32_t * tile_bounds,
    const int32_t * tile_experts,
    __half * output,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int K,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int max_tiles,
    bool routed_input) {
    constexpr int kTileM = 16;
    constexpr int kTileN = 64;
    constexpr int kGroupsPerChunk = kNepqGroupsPerSupergroup;
    constexpr int kTileK = kGroupSize * kGroupsPerChunk;
    constexpr int kStrideK = kTileK + 8;
    constexpr int kNFragments = kTileN / 16;
    constexpr int kFineTilesPerTask = kTileM / 8;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int ntiles_n = (out_per_expert + kTileN - 1) / kTileN;
    const int64_t max_tasks = static_cast<int64_t>(max_tiles) * ntiles_n;
    __shared__ __half weight_tile[kTileN][kStrideK];
    __shared__ __half activation_tile[kTileM][kStrideK];
    __shared__ float output_tile[kNFragments][16][16];

    using FragmentA = wmma::fragment<wmma::matrix_a, 16, 16, 16,
                                     __half, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, 16, 16, 16,
                                     __half, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int fine_tile = static_cast<int>(task / ntiles_n);
        const int ntile = static_cast<int>(task -
            static_cast<int64_t>(fine_tile) * ntiles_n);
        if (fine_tile >= tile_bounds[global_experts]) continue;
        const int expert = tile_experts[fine_tile];
        const int local_fine_tile = fine_tile - tile_bounds[expert];
        if (local_fine_tile % kFineTilesPerTask != 0) continue;
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned int>(local_expert) >=
            static_cast<unsigned int>(pool_experts)) continue;
        const int first = expert_bounds[expert] + local_fine_tile * 8;
        const int last = min(first + kTileM, expert_bounds[expert + 1]);
        const int n0 = ntile * kTileN;

        FragmentC accumulator;
        wmma::fill_fragment(accumulator, 0.0f);
        const int chunks = (ng + kGroupsPerChunk - 1) / kGroupsPerChunk;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int group_base = chunk * kGroupsPerChunk;
            const int k_base = group_base * kGroupSize;
            const int row_local = tid / kGroupsPerChunk;
            const int group_local = tid - row_local * kGroupsPerChunk;
            const int local_row = n0 + row_local;
            const int row = local_expert * out_per_expert + local_row;
            const int group = group_base + group_local;
            const bool valid_weight =
                local_row < out_per_expert && group < ng;
            uint32_t state = 0;
            float scale = 0.0f;
            const int8_t * table = table_pool;
            if (valid_weight) {
                const int64_t sub_linear =
                    static_cast<int64_t>(row) * ng + group;
                state = load_packed_bits(
                    sub_scale, sub_linear * sub_bits,
                    sub_bits, sub_scale_nbytes);
                table = nepq_active_table(
                    table_pool, bank_ids, row, group,
                    nsuper, table_stride);
                scale = format_scale<FORMAT>(
                    neuron_scale[row], state, table);
            }
#pragma unroll
            for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
                const int packed = valid_weight
                    ? decode_nepq_chunk4<FORMAT>(
                          indices, indices_nbytes, aux, aux_nbytes, table,
                          row, group, quartet, nvec, nsign, ng,
                          0, state)
                    : 0;
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const int shift = pair * 16;
                    const int value0 = static_cast<int>(
                        static_cast<int8_t>((packed >> shift) & 0xff));
                    const int value1 = static_cast<int>(
                        static_cast<int8_t>((packed >> (shift + 8)) & 0xff));
                    *reinterpret_cast<__half2 *>(
                        &weight_tile[row_local]
                            [group_local * kGroupSize + quartet * 4 + pair * 2]) =
                        __halves2half2(
                            __float2half(scale * static_cast<float>(value0)),
                            __float2half(scale * static_cast<float>(value1)));
                }
            }

            constexpr int kActivationPairs = kTileM * (kTileK / 2);
            for (int index = tid; index < kActivationPairs; index += 256) {
                const int m_local = index / (kTileK / 2);
                const int k_pair = index - m_local * (kTileK / 2);
                const int compact = first + m_local;
                const int k = k_base + k_pair * 2;
                __half2 values = __float2half2_rn(0.0f);
                if (compact < last) {
                    const int pair_index = ids_dst[compact];
                    const int source_row = routed_input
                        ? pair_index : pair_index / routes;
                    if (k + 1 < K) {
                        values = *reinterpret_cast<const __half2 *>(
                            x + static_cast<int64_t>(source_row) * K + k);
                    } else if (k < K) {
                        values = __halves2half2(
                            x[static_cast<int64_t>(source_row) * K + k],
                            __float2half(0.0f));
                    }
                }
                *reinterpret_cast<__half2 *>(
                    &activation_tile[m_local][k_pair * 2]) = values;
            }
            __syncthreads();

            if (warp < kNFragments) {
#pragma unroll
                for (int k_local = 0; k_local < kTileK; k_local += 16) {
                    FragmentA activation_fragment;
                    FragmentB weight_fragment;
                    wmma::load_matrix_sync(
                        activation_fragment, &activation_tile[0][k_local],
                        kStrideK);
                    wmma::load_matrix_sync(
                        weight_fragment, &weight_tile[warp * 16][k_local],
                        kStrideK);
                    wmma::mma_sync(
                        accumulator, activation_fragment,
                        weight_fragment, accumulator);
                }
            }
            __syncthreads();
        }

        if (warp < kNFragments) {
            wmma::store_matrix_sync(
                &output_tile[warp][0][0], accumulator,
                16, wmma::mem_row_major);
        }
        __syncthreads();
        if (warp < kNFragments) {
            for (int element = lane; element < 256; element += 32) {
                const int m_local = element / 16;
                const int n_local = element - m_local * 16;
                const int compact = first + m_local;
                const int local_row = n0 + warp * 16 + n_local;
                if (compact < last && local_row < out_per_expert) {
                    const int pair_index = ids_dst[compact];
                    output[static_cast<int64_t>(pair_index) * out_per_expert +
                           local_row] =
                        __float2half(output_tile[warp][m_local][n_local]);
                }
            }
        }
        __syncthreads();
    }
}

torch::Tensor nvq_moe_grouped_matmul_pool_f16_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook, torch::Tensor x,
    torch::Tensor expert_local, int64_t n_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t gs,
    int64_t sub_bits, int64_t format, int64_t sign_mode,
    torch::Tensor out, torch::Tensor ids_dst, torch::Tensor expert_bounds,
    torch::Tensor tile_bounds, torch::Tensor tile_experts) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(gs == kGroupSize, "NVQ routed FP16 requires gs24");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == torch::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) &&
                x.size(-1) == neuron_len,
                "NVQ routed FP16 input shape mismatch");
    const int routes = static_cast<int>(out.size(1));
    const int pairs = static_cast<int>(out.size(0)) * routes;
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
        (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    const int max_tiles = (pairs + 7) / 8 + static_cast<int>(n_experts);
    const int ntiles_n = (static_cast<int>(out_per_expert) + 63) / 64;
    const int blocks = static_cast<int>(std::min<int64_t>(
        static_cast<int64_t>(max_tiles) * ntiles_n, 4096));
    const dim3 threads(32, 8);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define NVQ_MOE_F16_LAUNCH()                                                     \
    nvq_moe_grouped_f16_kernel<F><<<blocks, threads, 0, stream>>>(               \
        indices.data_ptr<uint8_t>(), indices.numel(),                            \
        aux.data_ptr<uint8_t>(), aux.numel(),                                    \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                        \
        neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),             \
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),                \
        expert_local.data_ptr<int32_t>(), ids_dst.data_ptr<int32_t>(),           \
        expert_bounds.data_ptr<int32_t>(), tile_bounds.data_ptr<int32_t>(),      \
        tile_experts.data_ptr<int32_t>(),                                        \
        reinterpret_cast<__half *>(out.data_ptr<at::Half>()),                    \
        routes, static_cast<int>(n_experts), static_cast<int>(pool_experts),     \
        static_cast<int>(out_per_expert), K, ng, nvec, nsign,                   \
        static_cast<int>(sub_bits), static_cast<int>(sign_mode),                 \
        max_tiles, x.dim() == 3)
    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        NVQ_MOE_F16_LAUNCH();
    });
#undef NVQ_MOE_F16_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nepq_moe_grouped_matmul_pool_f16_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor table_pool,
    torch::Tensor bank_ids, torch::Tensor x, torch::Tensor expert_local,
    int64_t n_experts, int64_t pool_experts, int64_t out_per_expert,
    int64_t neuron_len, int64_t sub_bits, int64_t format,
    torch::Tensor out, torch::Tensor ids_dst, torch::Tensor expert_bounds,
    torch::Tensor tile_bounds, torch::Tensor tile_experts) {
    check_nepq_common(indices, aux, sub_scale, neuron_scale, table_pool,
                      bank_ids, neuron_len, sub_bits, format);
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == torch::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) &&
                x.size(-1) == neuron_len,
                "NEPQ routed FP16 input shape mismatch");
    const int routes = static_cast<int>(out.size(1));
    const int pairs = static_cast<int>(out.size(0)) * routes;
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    const int max_tiles = (pairs + 7) / 8 + static_cast<int>(n_experts);
    const int ntiles_n = (static_cast<int>(out_per_expert) + 63) / 64;
    const int blocks = static_cast<int>(std::min<int64_t>(
        static_cast<int64_t>(max_tiles) * ntiles_n, 4096));
    const dim3 threads(32, 8);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define NEPQ_MOE_F16_LAUNCH()                                                    \
    nepq_moe_grouped_f16_kernel<F><<<blocks, threads, 0, stream>>>(              \
        indices.data_ptr<uint8_t>(), indices.numel(),                            \
        aux.data_ptr<uint8_t>(), aux.numel(),                                    \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                        \
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),           \
        bank_ids.data_ptr<uint8_t>(),                                            \
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),                \
        expert_local.data_ptr<int32_t>(), ids_dst.data_ptr<int32_t>(),           \
        expert_bounds.data_ptr<int32_t>(), tile_bounds.data_ptr<int32_t>(),      \
        tile_experts.data_ptr<int32_t>(),                                        \
        reinterpret_cast<__half *>(out.data_ptr<at::Half>()),                    \
        routes, static_cast<int>(n_experts), static_cast<int>(pool_experts),     \
        static_cast<int>(out_per_expert), K, ng, nvec, nsign, nsuper,           \
        table_stride, static_cast<int>(sub_bits), max_tiles, x.dim() == 3)
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        NEPQ_MOE_F16_LAUNCH();
    });
#undef NEPQ_MOE_F16_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nvq_moe_grouped_matmul_pool_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    torch::Tensor ids,
    torch::Tensor expert_local,
    int64_t n_experts,
    int64_t pool_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    bool input_quantized,
    torch::Tensor out,
    torch::Tensor qx,
    torch::Tensor xscale,
    torch::Tensor ids_dst,
    torch::Tensor expert_bounds,
    torch::Tensor tile_bounds,
    torch::Tensor tile_experts) {
    check_common(
        indices, aux, sub_scale, neuron_scale, codebook,
        neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(n_experts > 0 && n_experts <= 4096,
                "NVQ global expert count must be in [1,4096]");
    TORCH_CHECK(
        pool_experts > 0 &&
        pool_experts <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "NVQ pool storage expert count must fit int32");
    TORCH_CHECK(out_per_expert > 0 &&
                neuron_scale.numel() == pool_experts * out_per_expert,
                "NVQ pool shape does not match the flattened row count");
    TORCH_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() &&
                expert_local.scalar_type() == torch::kInt32 &&
                expert_local.dim() == 1 && expert_local.numel() >= n_experts,
                "NVQ expert-local map must be CUDA contiguous int32");
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() &&
                ids.scalar_type() == torch::kInt32 && ids.dim() == 2,
                "NVQ route IDs must be CUDA contiguous int32 [T,R]");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == torch::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) &&
                (input_quantized || x.size(-1) == neuron_len),
                "NVQ routed input must be CUDA contiguous fp16 [T,K] or [T,R,K]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    TORCH_CHECK(tokens > 0 && routes > 0, "NVQ route dimensions must be nonzero");
    TORCH_CHECK((!routed_input && x.size(0) == tokens) ||
                (routed_input && x.size(0) == tokens && x.size(1) == routes),
                "NVQ input leading dimensions do not match routes");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() &&
                out.scalar_type() == torch::kFloat16 && out.dim() == 3 &&
                out.size(0) == tokens && out.size(1) == routes &&
                out.size(2) == out_per_expert,
                "NVQ output must be CUDA contiguous fp16 [T,R,O]");

    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    const int input_rows = routed_input ? pairs : tokens;
    TORCH_CHECK(qx.is_cuda() && qx.is_contiguous() &&
                qx.scalar_type() == torch::kInt8 && qx.dim() == 2 &&
                qx.size(0) >= input_rows && qx.size(1) >= K_pad,
                "NVQ routed qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
                xscale.scalar_type() == torch::kFloat32 && xscale.dim() == 2 &&
                xscale.size(0) >= input_rows && xscale.size(1) >= ng,
                "NVQ routed xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
        (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (!input_quantized) {
        nvq_quantize_x_gs24_kernel<<<dim3(input_rows, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), input_rows, K, ng);
    }

    if (tokens <= 8) {
        const char * rows_env = std::getenv("MFQ_NVQ_MOE_ROWS_PER_BLOCK");
        int rows_per_block = rows_env == nullptr ? 2 : std::atoi(rows_env);
        if (rows_per_block != 1 && rows_per_block != 2 &&
                rows_per_block != 4 && rows_per_block != 8) {
            rows_per_block = 2;
        }
        const char * warps_env = std::getenv("MFQ_NVQ_MOE_WARPS");
        int warps_override = warps_env == nullptr ? 0 : std::atoi(warps_env);
        if (warps_override != 1 && warps_override != 2 &&
                warps_override != 4 && warps_override != 8) {
            warps_override = 0;
        }
        const char * exact_reduction_env =
            std::getenv("MFQ_NVQ_MOE_EXACT_REDUCTION");
        const bool exact_reduction =
            exact_reduction_env != nullptr &&
            std::atoi(exact_reduction_env) != 0;
        const char * exact_physical_env =
            std::getenv("MFQ_NVQ_MOE_EXACT_PHYSICAL_WARPS");
        const int exact_physical_warps =
            exact_physical_env != nullptr && std::atoi(exact_physical_env) == 4
            ? 4
            : 2;
        const char * share_group_env =
            std::getenv("MFQ_NVQ_MOE_SHARE_GROUP_STATE");
        const bool share_group_state =
            share_group_env != nullptr && std::atoi(share_group_env) != 0;
#define NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, SHARE_VALUE)              \
        nvq_moe_mmvq_kernel<F, NWARPS_VALUE, ROWS_VALUE, SHARE_VALUE><<<         \
            dim3((static_cast<int>(out_per_expert) + ROWS_VALUE - 1) /          \
                     ROWS_VALUE,                                                 \
                 pairs),                                                         \
            dim3(32, NWARPS_VALUE), 0, stream>>>(                                \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),         \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                     \
            ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),           \
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, static_cast<int>(sub_bits),                         \
            static_cast<int>(sign_mode), routed_input)
#define NVQ_MOE_SELECTED_LAUNCH(NWARPS_VALUE, ROWS_VALUE)                        \
        do {                                                                      \
            if constexpr (F == kNvq2Exec || F == kNvq2JscExec) {                 \
                if (share_group_state) {                                          \
                    NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, true);         \
                } else {                                                          \
                    NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, false);        \
                }                                                                 \
            } else {                                                              \
                NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, false);            \
            }                                                                     \
        } while (0)
#define NVQ_MOE_EXACT_LAUNCH(PHYSICAL_WARPS_VALUE, LOGICAL_WARPS_VALUE)          \
        nvq_moe_mmvq_exact_reduction_kernel<                                    \
            F, PHYSICAL_WARPS_VALUE, LOGICAL_WARPS_VALUE, 2><<<                 \
            dim3((static_cast<int>(out_per_expert) + 1) / 2, pairs),            \
            dim3(32, PHYSICAL_WARPS_VALUE), 0, stream>>>(                        \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),         \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                     \
            ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),           \
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, static_cast<int>(sub_bits),                         \
            static_cast<int>(sign_mode), routed_input)
        launch_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            if (exact_reduction) {
                if (K >= 4096) {
                    if (exact_physical_warps == 4) {
                        NVQ_MOE_EXACT_LAUNCH(4, 8);
                    } else {
                        NVQ_MOE_EXACT_LAUNCH(2, 8);
                    }
                } else {
                    NVQ_MOE_EXACT_LAUNCH(2, 4);
                }
                return;
            }
            if (warps_override == 1) {
                NVQ_MOE_SELECTED_LAUNCH(1, 2);
                return;
            }
            if (warps_override == 2) {
                if (rows_per_block == 4) {
                    NVQ_MOE_SELECTED_LAUNCH(2, 4);
                } else {
                    NVQ_MOE_SELECTED_LAUNCH(2, 2);
                }
                return;
            }
            if (warps_override == 4) {
                NVQ_MOE_SELECTED_LAUNCH(4, 2);
                return;
            }
            if (warps_override == 8) {
                NVQ_MOE_SELECTED_LAUNCH(8, 2);
                return;
            }
            if (K >= 4096) {
                if (rows_per_block == 1) NVQ_MOE_SELECTED_LAUNCH(8, 1);
                else if (rows_per_block == 4) NVQ_MOE_SELECTED_LAUNCH(8, 4);
                else if (rows_per_block == 8) NVQ_MOE_SELECTED_LAUNCH(8, 8);
                else NVQ_MOE_SELECTED_LAUNCH(8, 2);
            } else {
                if (rows_per_block == 1) NVQ_MOE_SELECTED_LAUNCH(4, 1);
                else if (rows_per_block == 4) NVQ_MOE_SELECTED_LAUNCH(4, 4);
                else if (rows_per_block == 8) NVQ_MOE_SELECTED_LAUNCH(4, 8);
                else NVQ_MOE_SELECTED_LAUNCH(4, 2);
            }
        });
#undef NVQ_MOE_EXACT_LAUNCH
#undef NVQ_MOE_SELECTED_LAUNCH
#undef NVQ_MOE_SMALL_LAUNCH
    } else {
        TORCH_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
                    ids_dst.scalar_type() == torch::kInt32 &&
                    ids_dst.numel() >= pairs,
                    "NVQ compact route map is missing");
        TORCH_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
                    expert_bounds.scalar_type() == torch::kInt32 &&
                    expert_bounds.numel() >= n_experts + 1,
                    "NVQ expert bounds are missing");
        TORCH_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
                    tile_bounds.scalar_type() == torch::kInt32 &&
                    tile_bounds.numel() >= n_experts + 1,
                    "NVQ tile bounds are missing");
        TORCH_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
                    tile_experts.scalar_type() == torch::kInt32 &&
                    tile_experts.numel() >= pairs,
                    "NVQ tile expert map is missing");
        constexpr int route_tile = 8;
        const int row_tiles = (static_cast<int>(out_per_expert) + 3) / 4;
        const int max_tiles = (pairs + route_tile - 1) / route_tile +
            static_cast<int>(n_experts);
        const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
        const int blocks = static_cast<int>(
            std::min<int64_t>(max_tasks, 4096));
#define NVQ_MOE_GROUPED_LAUNCH()                                                 \
        nvq_moe_grouped_tile_kernel<F, route_tile><<<                            \
            blocks, dim3(32, 4), 0, stream>>>(                                   \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),         \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                     \
            expert_local.data_ptr<int32_t>(), ids_dst.data_ptr<int32_t>(),       \
            expert_bounds.data_ptr<int32_t>(), tile_bounds.data_ptr<int32_t>(),  \
            tile_experts.data_ptr<int32_t>(),                                    \
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()),                \
            routes, static_cast<int>(n_experts),                                 \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, static_cast<int>(sub_bits),                         \
            static_cast<int>(sign_mode), max_tiles, row_tiles, routed_input)
        launch_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            NVQ_MOE_GROUPED_LAUNCH();
        });
#undef NVQ_MOE_GROUPED_LAUNCH
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "NVQ routed pool matmul kernel launch failed");
    return out;
}

torch::Tensor nepq_moe_grouped_matmul_pool_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor table_pool,
    torch::Tensor bank_ids,
    torch::Tensor grouped_table_pool,
    torch::Tensor x,
    torch::Tensor ids,
    torch::Tensor expert_local,
    int64_t n_experts,
    int64_t pool_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    bool input_quantized,
    torch::Tensor out,
    torch::Tensor qx,
    torch::Tensor xscale,
    torch::Tensor ids_dst,
    torch::Tensor expert_bounds,
    torch::Tensor tile_bounds,
    torch::Tensor tile_experts) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    const int expected_grouped_stride = format == kNpq0S
        ? kNpq0STableBytes
        : static_cast<int>(table_pool.size(1));
    TORCH_CHECK(
        grouped_table_pool.is_cuda() && grouped_table_pool.is_contiguous() &&
        grouped_table_pool.scalar_type() == torch::kInt8 &&
        grouped_table_pool.dim() == 2 &&
        grouped_table_pool.size(0) == table_pool.size(0) &&
        grouped_table_pool.size(1) == expected_grouped_stride,
        "NEPQ grouped table pool shape mismatch");
    TORCH_CHECK(n_experts > 0 && n_experts <= 4096,
                "NEPQ global expert count must be in [1,4096]");
    TORCH_CHECK(pool_experts > 0 &&
                pool_experts <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                out_per_expert > 0 &&
                neuron_scale.numel() == pool_experts * out_per_expert,
                "NEPQ pool shape does not match the flattened row count");
    TORCH_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() &&
                expert_local.scalar_type() == torch::kInt32 &&
                expert_local.dim() == 1 && expert_local.numel() >= n_experts,
                "NEPQ expert-local map must be CUDA contiguous int32");
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() &&
                ids.scalar_type() == torch::kInt32 && ids.dim() == 2,
                "NEPQ route IDs must be CUDA contiguous int32 [T,R]");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == torch::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) && x.size(-1) == neuron_len,
                "NEPQ routed input must be CUDA contiguous fp16 [T,K] or [T,R,K]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    TORCH_CHECK(tokens > 0 && routes > 0, "NEPQ route dimensions must be nonzero");
    TORCH_CHECK((!routed_input && x.size(0) == tokens) ||
                (routed_input && x.size(0) == tokens && x.size(1) == routes),
                "NEPQ input leading dimensions do not match routes");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() &&
                out.scalar_type() == torch::kFloat16 && out.dim() == 3 &&
                out.size(0) == tokens && out.size(1) == routes &&
                out.size(2) == out_per_expert,
                "NEPQ output must be CUDA contiguous fp16 [T,R,O]");

    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    const int input_rows = routed_input ? pairs : tokens;
    TORCH_CHECK(qx.is_cuda() && qx.is_contiguous() &&
                qx.scalar_type() == torch::kInt8 && qx.dim() == 2 &&
                qx.size(0) >= input_rows && qx.size(1) >= K_pad,
                "NEPQ routed qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
                xscale.scalar_type() == torch::kFloat32 && xscale.dim() == 2 &&
                xscale.size(0) >= input_rows && xscale.size(1) >= ng,
                "NEPQ routed xscale workspace mismatch");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    const int grouped_table_stride =
        static_cast<int>(grouped_table_pool.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (!input_quantized) {
        nvq_quantize_x_gs24_kernel<<<dim3(input_rows, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), input_rows, K, ng);
    }

    if (tokens <= 8) {
        constexpr int rows_per_block = 2;
        const int row_blocks =
            (static_cast<int>(out_per_expert) + rows_per_block - 1) /
            rows_per_block;
#define NEPQ_MOE_POOL_SMALL_LAUNCH(NWARPS_VALUE)                                 \
        nepq_moe_mmvq_kernel<F, NWARPS_VALUE, rows_per_block><<<                 \
            dim3(row_blocks, pairs), dim3(32, NWARPS_VALUE), 0, stream>>>(       \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),       \
            bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                 \
            xscale.data_ptr<float>(), ids.data_ptr<int32_t>(),                   \
            expert_local.data_ptr<int32_t>(),                                    \
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, nsuper, table_stride,                               \
            static_cast<int>(sub_bits), routed_input)
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            if (K >= 4096) NEPQ_MOE_POOL_SMALL_LAUNCH(8);
            else NEPQ_MOE_POOL_SMALL_LAUNCH(4);
        });
#undef NEPQ_MOE_POOL_SMALL_LAUNCH
    } else {
        TORCH_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
                    ids_dst.scalar_type() == torch::kInt32 &&
                    ids_dst.numel() >= pairs,
                    "NEPQ compact route map is missing");
        TORCH_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
                    expert_bounds.scalar_type() == torch::kInt32 &&
                    expert_bounds.numel() >= n_experts + 1,
                    "NEPQ expert bounds are missing");
        TORCH_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
                    tile_bounds.scalar_type() == torch::kInt32 &&
                    tile_bounds.numel() >= n_experts + 1,
                    "NEPQ tile bounds are missing");
        TORCH_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
                    tile_experts.scalar_type() == torch::kInt32 &&
                    tile_experts.numel() >= pairs,
                    "NEPQ tile expert map is missing");
        constexpr int route_tile = 8;
        const int row_tiles = (static_cast<int>(out_per_expert) + 3) / 4;
        const int max_tiles = (pairs + route_tile - 1) / route_tile +
            static_cast<int>(n_experts);
        const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
        const int blocks = static_cast<int>(
            std::min<int64_t>(max_tasks, 4096));
#define NEPQ_MOE_POOL_GROUPED_LAUNCH()                                           \
        nepq_moe_grouped_tile_kernel<F, route_tile><<<                           \
            blocks, dim3(32, 4), 0, stream>>>(                                   \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), grouped_table_pool.data_ptr<int8_t>(), \
            bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                 \
            xscale.data_ptr<float>(), expert_local.data_ptr<int32_t>(),          \
            ids_dst.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(),      \
            tile_bounds.data_ptr<int32_t>(), tile_experts.data_ptr<int32_t>(),   \
            reinterpret_cast<__half *>(out.data_ptr<at::Half>()),                \
            routes, static_cast<int>(n_experts),                                 \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, nsuper, grouped_table_stride,                       \
            static_cast<int>(sub_bits), max_tiles, row_tiles, routed_input)
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            NEPQ_MOE_POOL_GROUPED_LAUNCH();
        });
#undef NEPQ_MOE_POOL_GROUPED_LAUNCH
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ routed pool matmul kernel launch failed");
    return out;
}

torch::Tensor nvq_dequant_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    const int N = static_cast<int>(neuron_scale.numel());
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = torch::empty({N, K}, neuron_scale.options().dtype(torch::kFloat16));
    const int64_t total = static_cast<int64_t>(N) * nsign;
    const int block = 256;
    const int grid = static_cast<int>(std::min<int64_t>((total + block - 1) / block, 65535));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nvq_dequant_kernel<F><<<grid, block, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            N, K, ng, nvec, nsign, static_cast<int>(sub_bits), static_cast<int>(sign_mode));
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ dequant kernel launch failed");
    return output;
}

torch::Tensor nvq_gemm_f16_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
                x.is_contiguous() && x.dim() == 2,
                "NVQ FP16 GEMM x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    TORCH_CHECK(M >= 16, "NVQ FP16 GEMM requires M >= 16");
    TORCH_CHECK(x.size(1) == K, "NVQ FP16 GEMM x width must equal neuron_len");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = torch::empty({M, N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(32, 8);

#define NVQ_F16_GEMM_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE)                       \
    nvq_gemm_f16_gs24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<<                \
        dim3((N + 63) / 64, (M + 16 * MTILES_VALUE - 1) /                        \
                               (16 * MTILES_VALUE)),                              \
        block, 0, stream>>>(                                                       \
        indices.data_ptr<uint8_t>(), indices.numel(),                             \
        aux.data_ptr<uint8_t>(), aux.numel(),                                     \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                         \
        neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),              \
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),                 \
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()),                  \
        M, N, K, ng, nvec, nsign, static_cast<int>(sub_bits),                     \
        static_cast<int>(sign_mode))

    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M <= 16) {
            if (M == 16 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(1, false);
            else NVQ_F16_GEMM_LAUNCH(1, true);
        } else if (M <= 32) {
            if (M == 32 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(2, false);
            else NVQ_F16_GEMM_LAUNCH(2, true);
        } else if (M <= 48) {
            if (M == 48 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(3, false);
            else NVQ_F16_GEMM_LAUNCH(3, true);
        } else if (M <= 64) {
            if (M == 64 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(4, false);
            else NVQ_F16_GEMM_LAUNCH(4, true);
        } else if (M <= 80) {
            if (M == 80 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(5, false);
            else NVQ_F16_GEMM_LAUNCH(5, true);
        } else if (M <= 96) {
            if (M == 96 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(6, false);
            else NVQ_F16_GEMM_LAUNCH(6, true);
        } else if (M <= 112) {
            if (M == 112 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(7, false);
            else NVQ_F16_GEMM_LAUNCH(7, true);
        } else {
            if (M % 128 == 0 && N % 64 == 0) NVQ_F16_GEMM_LAUNCH(8, false);
            else NVQ_F16_GEMM_LAUNCH(8, true);
        }
    });
#undef NVQ_F16_GEMM_LAUNCH
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ FP16 GEMM kernel launch failed");
    return output;
}

torch::Tensor nvq_gemv_m1_vec8_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t nwarps,
    torch::Tensor qx,
    torch::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1, "NVQ vec8 GEMV x must be CUDA contiguous fp16 [1,K]");
    TORCH_CHECK(nwarps == 1 || nwarps == 2 || nwarps == 4 || nwarps == 8,
                "NVQ vec8 GEMV nwarps must be 1, 2, 4, or 8");
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(x.size(1) == K, "NVQ vec8 GEMV x width must equal neuron_len");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == torch::kInt8 && qx.is_contiguous() &&
                qx.numel() >= K_pad, "NVQ vec8 GEMV qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == torch::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= ng, "NVQ vec8 GEMV xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = torch::empty({1, N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvq_quantize_x_gs24_kernel<<<dim3(1, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), 1, K, ng);
    switch (static_cast<int>(nwarps)) {
        case 1:
            launch_m1_vec8_by_format<1>(format, indices, aux, sub_scale, neuron_scale, codebook,
                                        qx, xscale, output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
            break;
        case 2:
            launch_m1_vec8_by_format<2>(format, indices, aux, sub_scale, neuron_scale, codebook,
                                        qx, xscale, output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
            break;
        case 4:
            launch_m1_vec8_by_format<4>(format, indices, aux, sub_scale, neuron_scale, codebook,
                                        qx, xscale, output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
            break;
        case 8:
            launch_m1_vec8_by_format<8>(format, indices, aux, sub_scale, neuron_scale, codebook,
                                        qx, xscale, output, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
            break;
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ vec8 GEMV kernel launch failed");
    return output;
}

torch::Tensor nvq_gemv_batch_vec8_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t nwarps,
    torch::Tensor qx,
    torch::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ batch vec8 GEMV x must be CUDA contiguous fp16 rank-2");
    TORCH_CHECK(nwarps == 2 || nwarps == 4 || nwarps == 8,
                "NVQ batch vec8 GEMV nwarps must be 2, 4, or 8");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(M >= 2 && M <= 16, "NVQ batch vec8 GEMV supports M in [2,16]");
    TORCH_CHECK(x.size(1) == K, "NVQ batch vec8 GEMV x width must equal neuron_len");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == torch::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ batch vec8 GEMV qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == torch::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ batch vec8 GEMV xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = torch::empty({M, N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), M, K, ng);
#define NVQ_LAUNCH_BATCH_VEC8(NW)                                                       \
    do {                                                                                \
        if (M <= 2) {                                                                   \
            launch_batch_vec8_by_format<NW, 2>(                                        \
                format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,   \
                output, M, N, ng, nvec, nsign, sub_bits, sign_mode, stream);           \
        } else if (M <= 4) {                                                            \
            launch_batch_vec8_by_format<NW, 4>(                                        \
                format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,   \
                output, M, N, ng, nvec, nsign, sub_bits, sign_mode, stream);           \
        } else if (M <= 8) {                                                            \
            launch_batch_vec8_by_format<NW, 8>(                                        \
                format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,   \
                output, M, N, ng, nvec, nsign, sub_bits, sign_mode, stream);           \
        } else if (M <= 12) {                                                           \
            launch_batch_vec8_by_format<NW, 12>(                                       \
                format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,   \
                output, M, N, ng, nvec, nsign, sub_bits, sign_mode, stream);           \
        } else {                                                                        \
            launch_batch_vec8_by_format<NW, 16>(                                       \
                format, indices, aux, sub_scale, neuron_scale, codebook, qx, xscale,   \
                output, M, N, ng, nvec, nsign, sub_bits, sign_mode, stream);           \
        }                                                                               \
    } while (0)
    switch (static_cast<int>(nwarps)) {
        case 2: NVQ_LAUNCH_BATCH_VEC8(2); break;
        case 4: NVQ_LAUNCH_BATCH_VEC8(4); break;
        case 8: NVQ_LAUNCH_BATCH_VEC8(8); break;
    }
#undef NVQ_LAUNCH_BATCH_VEC8
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ batch vec8 GEMV kernel launch failed");
    return output;
}

torch::Tensor nvq_gemv_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    TORCH_CHECK(M >= 1 && M <= 16, "NVQ direct GEMV supports M in [1,16]");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == torch::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == torch::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = torch::empty({M, N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), M, K, ng);
    if (M == 1) {
        launch_selected_m1_vec8(
            static_cast<int>(format), indices, aux, sub_scale, neuron_scale, codebook,
            qx, xscale, output, N, ng, nvec, nsign,
            static_cast<int>(sub_bits), static_cast<int>(sign_mode), stream);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ vec8 GEMV kernel launch failed");
        return output;
    }
    launch_selected_batch_vec8(
        static_cast<int>(format), indices, aux, sub_scale, neuron_scale, codebook,
        qx, xscale, output, M, N, ng, nvec, nsign,
        static_cast<int>(sub_bits), static_cast<int>(sign_mode), stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ batch vec8 GEMV kernel launch failed");
    return output;
}

torch::Tensor nvq_gemv_qx_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == torch::kInt8 && qx.is_contiguous() && qx.dim() == 2,
                "NVQ prequantized qx must be CUDA contiguous int8 rank-2");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == torch::kFloat32 && xscale.is_contiguous() &&
                xscale.dim() == 2, "NVQ prequantized xscale must be CUDA contiguous fp32 rank-2");
    const int M = static_cast<int>(qx.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(M >= 1 && M <= 16, "NVQ prequantized GEMV supports M in [1,16]");
    TORCH_CHECK(qx.size(1) >= K_pad, "NVQ prequantized qx width mismatch");
    TORCH_CHECK(xscale.size(0) >= M && xscale.size(1) >= ng,
                "NVQ prequantized xscale shape mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = torch::empty(
        {M, N}, neuron_scale.options().dtype(torch::kFloat16));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (M == 1) {
        launch_selected_m1_vec8(
            static_cast<int>(format), indices, aux, sub_scale, neuron_scale, codebook,
            qx, xscale, output, N, ng, nvec, nsign,
            static_cast<int>(sub_bits), static_cast<int>(sign_mode), stream);
    } else {
        launch_selected_batch_vec8(
            static_cast<int>(format), indices, aux, sub_scale, neuron_scale, codebook,
            qx, xscale, output, M, N, ng, nvec, nsign,
            static_cast<int>(sub_bits), static_cast<int>(sign_mode), stream);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ prequantized GEMV kernel launch failed");
    return output;
}

torch::Tensor nvq_gemv_qx_residual_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale,
    torch::Tensor residual) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(
        format == kNvq2JscXLGroupExec || format == kNvq3JscLGroupExec,
        "NVQ fused residual requires an aligned group execution format");
    TORCH_CHECK(
        qx.is_cuda() && qx.scalar_type() == torch::kInt8 &&
        qx.is_contiguous() && qx.dim() == 2 && qx.size(0) == 1,
        "NVQ fused residual qx must be CUDA contiguous int8 [1,K]");
    TORCH_CHECK(
        xscale.is_cuda() && xscale.scalar_type() == torch::kFloat32 &&
        xscale.is_contiguous() && xscale.dim() == 2 && xscale.size(0) >= 1,
        "NVQ fused residual xscale must be CUDA contiguous fp32 [1,ng]");
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(qx.size(1) >= K_pad, "NVQ fused residual qx width mismatch");
    TORCH_CHECK(xscale.size(1) >= ng, "NVQ fused residual xscale width mismatch");
    TORCH_CHECK(
        residual.is_cuda() && residual.scalar_type() == torch::kFloat16 &&
        residual.is_contiguous() && residual.dim() == 2 &&
        residual.size(0) == 1 && residual.size(1) == N,
        "NVQ fused residual must be CUDA contiguous fp16 [1,N]");

    const auto weight = make_device_weight(
        indices, aux, sub_scale, neuron_scale, codebook,
        K, static_cast<int>(gs), static_cast<int>(sub_bits),
        static_cast<int>(format), static_cast<int>(sign_mode));
    auto output = torch::empty_like(residual);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto * residual_ptr = reinterpret_cast<const __half *>(
        residual.data_ptr<at::Half>());
    auto * output_ptr = reinterpret_cast<__half *>(
        output.data_ptr<at::Half>());
    if (format == kNvq2JscXLGroupExec) {
        launch_aligned_group_m1<kNvq2JscXLGroupExec, true>(
            weight, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            output_ptr, stream, residual_ptr);
    } else {
        launch_aligned_group_m1<kNvq3JscLGroupExec, true>(
            weight, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            output_ptr, stream, residual_ptr);
    }
    TORCH_CHECK(
        cudaGetLastError() == cudaSuccess,
        "NVQ fused residual GEMV kernel launch failed");
    return output;
}

torch::Tensor nvq_gemv_multi2_ws_cuda(
    torch::Tensor first_indices,
    torch::Tensor first_aux,
    torch::Tensor first_sub_scale,
    torch::Tensor first_neuron_scale,
    torch::Tensor first_codebook,
    torch::Tensor second_indices,
    torch::Tensor second_aux,
    torch::Tensor second_sub_scale,
    torch::Tensor second_neuron_scale,
    torch::Tensor second_codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t first_sub_bits,
    int64_t first_format,
    int64_t first_sign_mode,
    int64_t second_sub_bits,
    int64_t second_format,
    int64_t second_sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    check_common(first_indices, first_aux, first_sub_scale, first_neuron_scale, first_codebook,
                 neuron_len, gs, first_sub_bits, first_format, first_sign_mode);
    check_common(second_indices, second_aux, second_sub_scale, second_neuron_scale, second_codebook,
                 neuron_len, gs, second_sub_bits, second_format, second_sign_mode);
    TORCH_CHECK(first_format == second_format,
                "NVQ multi-projection requires the same NVQ format");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ multi-projection x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(M >= 1 && M <= 8, "NVQ multi-projection supports M in [1,8]");
    TORCH_CHECK(x.size(1) == K, "NVQ multi-projection x width mismatch");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == torch::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad,
                "NVQ multi-projection qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == torch::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng,
                "NVQ multi-projection xscale workspace mismatch");
    const auto first = make_device_weight(
        first_indices, first_aux, first_sub_scale, first_neuron_scale, first_codebook,
        K, static_cast<int>(gs), static_cast<int>(first_sub_bits),
        static_cast<int>(first_format), static_cast<int>(first_sign_mode));
    const auto second = make_device_weight(
        second_indices, second_aux, second_sub_scale, second_neuron_scale, second_codebook,
        K, static_cast<int>(gs), static_cast<int>(second_sub_bits),
        static_cast<int>(second_format), static_cast<int>(second_sign_mode));
    auto output = torch::empty(
        {M, first.N + second.N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), M, K, ng);
    launch_selected_multi2_vec8(
        static_cast<int>(first_format), first, second, qx, xscale, output, M, stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ multi-projection kernel launch failed");
    return output;
}

torch::Tensor nvq_gemv_swiglu_ws_cuda(
    torch::Tensor gate_indices,
    torch::Tensor gate_aux,
    torch::Tensor gate_sub_scale,
    torch::Tensor gate_neuron_scale,
    torch::Tensor gate_codebook,
    torch::Tensor up_indices,
    torch::Tensor up_aux,
    torch::Tensor up_sub_scale,
    torch::Tensor up_neuron_scale,
    torch::Tensor up_codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t gate_sub_bits,
    int64_t gate_format,
    int64_t gate_sign_mode,
    int64_t up_sub_bits,
    int64_t up_format,
    int64_t up_sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    check_common(gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
                 neuron_len, gs, gate_sub_bits, gate_format, gate_sign_mode);
    check_common(up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
                 neuron_len, gs, up_sub_bits, up_format, up_sign_mode);
    TORCH_CHECK(gate_format == up_format, "NVQ SwiGLU requires matching NVQ formats");
    TORCH_CHECK(gate_neuron_scale.numel() == up_neuron_scale.numel(),
                "NVQ SwiGLU gate/up output widths must match");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1,
                "NVQ SwiGLU x must be CUDA contiguous fp16 [1,K]");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(x.size(1) == K, "NVQ SwiGLU x width mismatch");
    TORCH_CHECK(qx.numel() >= K_pad && xscale.numel() >= ng,
                "NVQ SwiGLU workspace mismatch");
    const auto gate = make_device_weight(
        gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
        K, static_cast<int>(gs), static_cast<int>(gate_sub_bits),
        static_cast<int>(gate_format), static_cast<int>(gate_sign_mode));
    const auto up = make_device_weight(
        up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
        K, static_cast<int>(gs), static_cast<int>(up_sub_bits),
        static_cast<int>(up_format), static_cast<int>(up_sign_mode));
    auto output = torch::empty({1, gate.N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvq_quantize_x_gs24_kernel<<<dim3(1, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), 1, K, ng);
    if (gate_format == kNvq2JscXLGroupExec &&
        gate_sub_bits == 4 && up_sub_bits == 4) {
        aligned_group_swiglu_pair_kernel<kNvq2JscXLGroupExec, 4>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<at::Half>()), nullptr);
    } else if (gate_format == kNvq3JscLGroupExec &&
               gate_sub_bits == 4 && up_sub_bits == 4) {
        aligned_group_swiglu_pair_kernel<kNvq3JscLGroupExec, 4>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<at::Half>()), nullptr);
    } else if (gate_format == kNvq2 && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<at::Half>()), nullptr);
    } else if (gate_format == kNvq2Exec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Exec, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<at::Half>()), nullptr);
    } else if (gate_format == kNvq2Jsc && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Jsc, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<at::Half>()), nullptr);
    } else if (gate_format == kNvq2JscExec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2JscExec, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<at::Half>()), nullptr);
    } else {
        launch_by_format(static_cast<int>(gate_format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            nvq_gemv_swiglu_pair_kernel<F, 4><<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<at::Half>()), nullptr);
        });
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ SwiGLU kernel launch failed");
    return output;
}

torch::Tensor nvq2_gemv_swiglu_vec4_ordered_ws_cuda(
    torch::Tensor gate_indices,
    torch::Tensor gate_aux,
    torch::Tensor gate_sub_scale,
    torch::Tensor gate_neuron_scale,
    torch::Tensor gate_codebook,
    torch::Tensor up_indices,
    torch::Tensor up_aux,
    torch::Tensor up_sub_scale,
    torch::Tensor up_neuron_scale,
    torch::Tensor up_codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t gate_sub_bits,
    int64_t gate_format,
    int64_t gate_sign_mode,
    int64_t up_sub_bits,
    int64_t up_format,
    int64_t up_sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    check_common(gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
                 neuron_len, gs, gate_sub_bits, gate_format, gate_sign_mode);
    check_common(up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
                 neuron_len, gs, up_sub_bits, up_format, up_sign_mode);
    TORCH_CHECK(gate_format == kNvq2 && up_format == kNvq2 &&
                gate_sub_bits == 4 && up_sub_bits == 4,
                "NVQ2 ordered vec4 SwiGLU requires NVQ2 sub_bits=4 weights");
    TORCH_CHECK(gate_neuron_scale.numel() == up_neuron_scale.numel(),
                "NVQ2 ordered vec4 SwiGLU gate/up output widths must match");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1 && x.size(1) == neuron_len,
                "NVQ2 ordered vec4 SwiGLU x must be CUDA contiguous fp16 [1,K]");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    const int nsign = (K + 7) / 8;
    TORCH_CHECK(nsign % 4 == 0,
                "NVQ2 ordered vec4 SwiGLU requires a multiple of four 8-value vectors");
    TORCH_CHECK(qx.numel() >= K_pad && xscale.numel() >= ng,
                "NVQ2 ordered vec4 SwiGLU workspace mismatch");
    const auto gate = make_device_weight(
        gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
        K, static_cast<int>(gs), 4, kNvq2, static_cast<int>(gate_sign_mode));
    const auto up = make_device_weight(
        up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
        K, static_cast<int>(gs), 4, kNvq2, static_cast<int>(up_sign_mode));
    auto output = torch::empty({1, gate.N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvq_quantize_x_gs24_kernel<<<dim3(1, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), 1, K, ng);
    const size_t shared_bytes = static_cast<size_t>(nsign) * sizeof(uint32_t);
    nvq2_gemv_swiglu_vec4_ordered_kernel<4>
        <<<gate.N, dim3(32, 4), shared_bytes, stream>>>(
            gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()), nullptr);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "NVQ2 ordered vec4 SwiGLU kernel launch failed");
    return output;
}

void nvq_ffn_swiglu_quant_ws_cuda(
    torch::Tensor gate_indices,
    torch::Tensor gate_aux,
    torch::Tensor gate_sub_scale,
    torch::Tensor gate_neuron_scale,
    torch::Tensor gate_codebook,
    torch::Tensor up_indices,
    torch::Tensor up_aux,
    torch::Tensor up_sub_scale,
    torch::Tensor up_neuron_scale,
    torch::Tensor up_codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t gate_sub_bits,
    int64_t gate_format,
    int64_t gate_sign_mode,
    int64_t up_sub_bits,
    int64_t up_format,
    int64_t up_sign_mode,
    int64_t down_gs,
    torch::Tensor input_qx,
    torch::Tensor input_xscale,
    torch::Tensor output_qx,
    torch::Tensor output_xscale,
    torch::Tensor swiglu_scratch) {
    check_common(gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
                 neuron_len, gs, gate_sub_bits, gate_format, gate_sign_mode);
    check_common(up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
                 neuron_len, gs, up_sub_bits, up_format, up_sign_mode);
    TORCH_CHECK(gate_format == up_format, "NVQ fused FFN requires matching gate/up formats");
    TORCH_CHECK(gate_neuron_scale.numel() == up_neuron_scale.numel(),
                "NVQ fused FFN gate/up widths must match");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1 && x.size(1) == neuron_len,
                "NVQ fused FFN x must be CUDA contiguous fp16 [1,K]");
    TORCH_CHECK(down_gs == 24 || down_gs == 28 || down_gs == 32,
                "NVQ fused FFN down gs must be 24, 28, or 32");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    const int N = static_cast<int>(gate_neuron_scale.numel());
    const int down_ng = (N + static_cast<int>(down_gs) - 1) / static_cast<int>(down_gs);
    const int down_k_pad = down_ng * static_cast<int>(down_gs);
    TORCH_CHECK(input_qx.is_cuda() && input_qx.scalar_type() == torch::kInt8 &&
                input_qx.is_contiguous() && input_qx.numel() >= K_pad,
                "NVQ fused FFN input qx workspace mismatch");
    TORCH_CHECK(input_xscale.is_cuda() && input_xscale.scalar_type() == torch::kFloat32 &&
                input_xscale.is_contiguous() && input_xscale.numel() >= ng,
                "NVQ fused FFN input xscale workspace mismatch");
    TORCH_CHECK(output_qx.is_cuda() && output_qx.scalar_type() == torch::kInt8 &&
                output_qx.is_contiguous() && output_qx.numel() >= down_k_pad,
                "NVQ fused FFN output qx workspace mismatch");
    TORCH_CHECK(output_xscale.is_cuda() && output_xscale.scalar_type() == torch::kFloat32 &&
                output_xscale.is_contiguous() && output_xscale.numel() >= down_ng,
                "NVQ fused FFN output xscale workspace mismatch");
    TORCH_CHECK(swiglu_scratch.is_cuda() && swiglu_scratch.scalar_type() == torch::kFloat32 &&
                swiglu_scratch.is_contiguous() && swiglu_scratch.numel() >= N,
                "NVQ fused FFN SwiGLU scratch mismatch");
    const auto gate = make_device_weight(
        gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
        K, static_cast<int>(gs), static_cast<int>(gate_sub_bits),
        static_cast<int>(gate_format), static_cast<int>(gate_sign_mode));
    const auto up = make_device_weight(
        up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
        K, static_cast<int>(gs), static_cast<int>(up_sub_bits),
        static_cast<int>(up_format), static_cast<int>(up_sign_mode));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvq_quantize_x_gs24_kernel<<<dim3(1, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()), input_qx.data_ptr<int8_t>(),
        input_xscale.data_ptr<float>(), 1, K, ng);
    bool aligned_independent_rows = false;
    const char * vec4_env = std::getenv("MFQ_NVQ_SWIGLU_VEC4");
    if (vec4_env == nullptr) vec4_env = std::getenv("MFQ_NIQ_SWIGLU_VEC4");
    const bool use_vec4 = vec4_env == nullptr || vec4_env[0] != '0';
    if (gate_format == kNvq2JscXLGroupExec &&
        gate_sub_bits == 4 && up_sub_bits == 4) {
        launch_aligned_group_multi2_m1<kNvq2JscXLGroupExec>(
            gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(swiglu_scratch.data_ptr<float>()), stream);
        aligned_independent_rows = true;
    } else if (gate_format == kNvq3JscLGroupExec &&
               gate_sub_bits == 4 && up_sub_bits == 4) {
        launch_aligned_group_multi2_m1<kNvq3JscLGroupExec>(
            gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(swiglu_scratch.data_ptr<float>()), stream);
        aligned_independent_rows = true;
    } else if (gate_format == kNvq2Exec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Exec, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else if (gate_format == kNvq2JscExec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2JscExec, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else if (gate_format == kNvq2Jsc && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Jsc, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else if (use_vec4 && gate_format == kNvq2 && gate_sub_bits == 4 && up_sub_bits == 4 &&
        gate.nsign % 4 == 0 && up.nsign % 4 == 0) {
        const size_t shared_bytes = static_cast<size_t>(gate.nsign) * sizeof(uint32_t);
        nvq2_gemv_swiglu_vec4_ordered_kernel<4, true>
            <<<gate.N, dim3(32, 4), shared_bytes, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else if (gate_format == kNvq2 && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else {
        launch_by_format(static_cast<int>(gate_format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            nvq_gemv_swiglu_pair_kernel<F, 4, true><<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
        });
    }
    switch (static_cast<int>(down_gs)) {
        case 24:
            if (aligned_independent_rows) {
                nvq_swiglu_quantize_f16_pair_kernel<24><<<down_ng, 32, 0, stream>>>(
                    reinterpret_cast<const __half *>(swiglu_scratch.data_ptr<float>()),
                    output_qx.data_ptr<int8_t>(), output_xscale.data_ptr<float>(),
                    N, down_k_pad);
            } else {
                nvq_quantize_f32_kernel<24><<<down_ng, 32, 0, stream>>>(
                    swiglu_scratch.data_ptr<float>(), output_qx.data_ptr<int8_t>(),
                    output_xscale.data_ptr<float>(), N, down_k_pad);
            }
            break;
        case 28:
            if (aligned_independent_rows) {
                nvq_swiglu_quantize_f16_pair_kernel<28><<<down_ng, 32, 0, stream>>>(
                    reinterpret_cast<const __half *>(swiglu_scratch.data_ptr<float>()),
                    output_qx.data_ptr<int8_t>(), output_xscale.data_ptr<float>(),
                    N, down_k_pad);
            } else {
                nvq_quantize_f32_kernel<28><<<down_ng, 32, 0, stream>>>(
                    swiglu_scratch.data_ptr<float>(), output_qx.data_ptr<int8_t>(),
                    output_xscale.data_ptr<float>(), N, down_k_pad);
            }
            break;
        case 32:
            if (aligned_independent_rows) {
                nvq_swiglu_quantize_f16_pair_kernel<32><<<down_ng, 32, 0, stream>>>(
                    reinterpret_cast<const __half *>(swiglu_scratch.data_ptr<float>()),
                    output_qx.data_ptr<int8_t>(), output_xscale.data_ptr<float>(),
                    N, down_k_pad);
            } else {
                nvq_quantize_f32_kernel<32><<<down_ng, 32, 0, stream>>>(
                    swiglu_scratch.data_ptr<float>(), output_qx.data_ptr<int8_t>(),
                    output_xscale.data_ptr<float>(), N, down_k_pad);
            }
            break;
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ fused FFN kernel launch failed");
}

torch::Tensor nvq_gemv_gate_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    torch::Tensor gate,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(mode == 1 || mode == 2, "NVQ gate mode must be 1(sigmoid) or 2(silu)");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == torch::kFloat16 && gate.is_contiguous() &&
                gate.sizes() == x.sizes(), "NVQ gate must be CUDA contiguous fp16 with x shape");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    TORCH_CHECK(M >= 1 && M <= 16, "NVQ gated direct GEMV supports M in [1,16]");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == torch::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == torch::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = torch::empty({M, N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (mode == 1) {
        nvq_quantize_x_gate_gs24_kernel<1><<<dim3(M, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
            reinterpret_cast<const __half *>(gate.data_ptr<at::Half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);
    } else {
        nvq_quantize_x_gate_gs24_kernel<2><<<dim3(M, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
            reinterpret_cast<const __half *>(gate.data_ptr<at::Half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);
    }
    if (M == 1) {
        launch_selected_m1_vec8(
            static_cast<int>(format), indices, aux, sub_scale, neuron_scale, codebook,
            qx, xscale, output, N, ng, nvec, nsign,
            static_cast<int>(sub_bits), static_cast<int>(sign_mode), stream);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ gated vec8 GEMV kernel launch failed");
        return output;
    }
    launch_selected_batch_vec8(
        static_cast<int>(format), indices, aux, sub_scale, neuron_scale, codebook,
        qx, xscale, output, M, N, ng, nvec, nsign,
        static_cast<int>(sub_bits), static_cast<int>(sign_mode), stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ gated batch vec8 GEMV kernel launch failed");
    return output;
}

torch::Tensor nvq_mmq_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    TORCH_CHECK(M >= 4 && M <= 64, "NVQ MMA MMQ supports M in [4,64]");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == torch::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == torch::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = torch::empty({M, N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), M, K, ng);
    const dim3 block(32, 8);
#define NVQ_MMQ_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE)                            \
    nvq_mmq_mma24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<<                   \
        dim3((N + 63) / 64, 1), block, 0, stream>>>(                             \
        indices.data_ptr<uint8_t>(), indices.numel(),                            \
        aux.data_ptr<uint8_t>(), aux.numel(),                                    \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                        \
        neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),             \
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                         \
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()),                 \
        M, N, ng, nvec, nsign, static_cast<int>(sub_bits),                       \
        static_cast<int>(sign_mode))
    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M <= 16) {
            if (M == 16 && N % 64 == 0) NVQ_MMQ_LAUNCH(1, false);
            else NVQ_MMQ_LAUNCH(1, true);
        } else if (M <= 32) {
            if (M == 32 && N % 64 == 0) NVQ_MMQ_LAUNCH(2, false);
            else NVQ_MMQ_LAUNCH(2, true);
        } else if (M <= 48) {
            if (M == 48 && N % 64 == 0) NVQ_MMQ_LAUNCH(3, false);
            else NVQ_MMQ_LAUNCH(3, true);
        } else {
            if (M == 64 && N % 64 == 0) NVQ_MMQ_LAUNCH(4, false);
            else NVQ_MMQ_LAUNCH(4, true);
        }
    });
#undef NVQ_MMQ_LAUNCH
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ MMA MMQ kernel launch failed");
    return output;
}

torch::Tensor nvq_mmq_gate_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    torch::Tensor gate,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(mode == 1 || mode == 2, "NVQ gate mode must be 1(sigmoid) or 2(silu)");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == torch::kFloat16 && gate.is_contiguous() &&
                gate.sizes() == x.sizes(), "NVQ gate must be CUDA contiguous fp16 with x shape");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    TORCH_CHECK(M >= 4 && M <= 64, "NVQ gated MMA MMQ supports M in [4,64]");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == torch::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == torch::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = torch::empty({M, N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (mode == 1) {
        nvq_quantize_x_gate_gs24_kernel<1><<<dim3(M, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
            reinterpret_cast<const __half *>(gate.data_ptr<at::Half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);
    } else {
        nvq_quantize_x_gate_gs24_kernel<2><<<dim3(M, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
            reinterpret_cast<const __half *>(gate.data_ptr<at::Half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);
    }
    const dim3 block(32, 8);
#define NVQ_GATED_MMQ_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE)                      \
    nvq_mmq_mma24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<<                   \
        dim3((N + 63) / 64, 1), block, 0, stream>>>(                             \
        indices.data_ptr<uint8_t>(), indices.numel(),                            \
        aux.data_ptr<uint8_t>(), aux.numel(),                                    \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                        \
        neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),             \
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                         \
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()),                 \
        M, N, ng, nvec, nsign, static_cast<int>(sub_bits),                       \
        static_cast<int>(sign_mode))
    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M <= 16) {
            if (M == 16 && N % 64 == 0) NVQ_GATED_MMQ_LAUNCH(1, false);
            else NVQ_GATED_MMQ_LAUNCH(1, true);
        } else if (M <= 32) {
            if (M == 32 && N % 64 == 0) NVQ_GATED_MMQ_LAUNCH(2, false);
            else NVQ_GATED_MMQ_LAUNCH(2, true);
        } else if (M <= 48) {
            if (M == 48 && N % 64 == 0) NVQ_GATED_MMQ_LAUNCH(3, false);
            else NVQ_GATED_MMQ_LAUNCH(3, true);
        } else {
            if (M == 64 && N % 64 == 0) NVQ_GATED_MMQ_LAUNCH(4, false);
            else NVQ_GATED_MMQ_LAUNCH(4, true);
        }
    });
#undef NVQ_GATED_MMQ_LAUNCH
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ gated MMA MMQ kernel launch failed");
    return output;
}

torch::Tensor nvq_embedding_lookup_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor token_ids,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(token_ids.is_cuda() && token_ids.scalar_type() == torch::kInt64 && token_ids.is_contiguous(),
                "NVQ token_ids must be CUDA contiguous int64");
    const int tokens = static_cast<int>(token_ids.numel());
    const int vocab = static_cast<int>(neuron_scale.numel());
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto shape = token_ids.sizes().vec();
    shape.push_back(K);
    auto output = torch::empty(shape, neuron_scale.options().dtype(torch::kFloat16));
    const int64_t total = static_cast<int64_t>(tokens) * nsign;
    const int block = 256;
    const int grid = static_cast<int>(std::min<int64_t>((total + block - 1) / block, 65535));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nvq_embedding_kernel<F><<<grid, block, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(), aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(), neuron_scale.data_ptr<float>(),
            codebook.data_ptr<int8_t>(), token_ids.data_ptr<int64_t>(),
            reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
            tokens, vocab, K, ng, nvec, nsign, static_cast<int>(sub_bits), static_cast<int>(sign_mode));
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "NVQ embedding kernel launch failed");
    return output;
}
