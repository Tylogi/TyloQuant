#include "mlx_tpq.h"

#include <mlx/allocator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kInt4GroupSize = 64;

constexpr const char* kTpqIndexHeader = R"METAL(
template <typename Stream>
inline uint mfq_tpq_read_index(
    Stream indices,
    uint index,
    uint bits
) {
    uint residual_bits = (index & 7u) * bits;
    uint byte_offset =
        (index >> 3) * bits + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed = uint(indices[byte_offset])
        | (uint(indices[byte_offset + 1u]) << 8u)
        | (uint(indices[byte_offset + 2u]) << 16u);
    return (packed >> shift) & ((1u << bits) - 1u);
}
)METAL";

constexpr const char* kInt4MatmulSource = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5;
    uint output = workgroup % uint(OUT);
    uint row_tile = workgroup / uint(OUT);
    uint first_row = row_tile * uint(TILE_M);
    if (output >= uint(OUT) || first_row >= uint(M)) {
        return;
    }

    float accumulators[TILE_M];
    for (uint row = 0u; row < uint(TILE_M); ++row) {
        accumulators[row] = 0.0f;
    }
    uint packed_base = output * uint(K / 2);
    uint scale_base = output * uint(GROUPS);
    for (
        uint column = lane * 2u;
        column < uint(K);
        column += 64u
    ) {
        uint value = uint(
            packed[packed_base + (column >> 1)]);
        float scale = float(
            scales[
                scale_base
                + column / uint(GROUP_SIZE)
            ]);
        float low =
            float(int(value & 15u) - 8) * scale;
        float high =
            float(int(value >> 4u) - 8) * scale;
        for (uint local = 0u;
             local < uint(TILE_M);
             ++local) {
            uint row = first_row + local;
            if (row < uint(M)) {
                uint input_base = row * uint(K);
                accumulators[local] = fma(
                    low,
                    float(x[input_base + column]),
                    accumulators[local]);
                accumulators[local] = fma(
                    high,
                    float(x[input_base + column + 1u]),
                    accumulators[local]);
            }
        }
    }
    for (uint local = 0u;
         local < uint(TILE_M);
         ++local) {
        float total = simd_sum(accumulators[local]);
        uint row = first_row + local;
        if (lane == 0u && row < uint(M)) {
            y[row * uint(OUT) + output] = T(total);
        }
    }
)METAL";

constexpr const char* kInt4GemvSource = R"METAL(
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group =
        simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;
    float accumulators[ROWS_PER_SIMD] = {0.0f};

    for (
        uint column = lane * 2u;
        column < uint(K);
        column += 64u
    ) {
        float low_input = float(x[column]);
        float high_input = float(x[column + 1u]);
        for (uint local = 0u;
             local < ROWS_PER_SIMD;
             ++local) {
            uint output =
                min(output_base + local, uint(OUT) - 1u);
            uint value = uint(packed[
                output * uint(K / 2)
                + (column >> 1)
            ]);
            float scale = float(scales[
                output * uint(GROUPS)
                + column / uint(GROUP_SIZE)
            ]);
            float low =
                float(int(value & 15u) - 8) * scale;
            float high =
                float(int(value >> 4u) - 8) * scale;
            accumulators[local] = fma(
                low,
                low_input,
                accumulators[local]);
            accumulators[local] = fma(
                high,
                high_input,
                accumulators[local]);
        }
    }

    for (uint local = 0u;
         local < ROWS_PER_SIMD;
         ++local) {
        float total = simd_sum(accumulators[local]);
        uint output = output_base + local;
        if (lane == 0u && output < uint(OUT)) {
            y[output] = T(total);
        }
    }
)METAL";

constexpr const char* kInt4MmqSource = R"METAL(
    constexpr uint K_LANES = 8u;
    constexpr uint ROWS_PER_SIMD =
        32u / K_LANES;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group =
        simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_row = lane / K_LANES;
    uint output_index =
        threadgroup_position_in_grid.y * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD
        + simd_row;
    uint output =
        min(output_index, uint(OUT) - 1u);
    uint first_row =
        threadgroup_position_in_grid.x
        * uint(TILE_M);

    float accumulators[TILE_M];
    for (uint row = 0u;
         row < uint(TILE_M);
         ++row) {
        accumulators[row] = 0.0f;
    }
    uint packed_base = output * uint(K / 2);
    uint scale_base = output * uint(GROUPS);
    for (
        uint column = k_lane * 2u;
        column < uint(K);
        column += 2u * K_LANES
    ) {
        uint value = uint(
            packed[packed_base + (column >> 1)]);
        float scale = float(scales[
            scale_base
            + column / uint(GROUP_SIZE)
        ]);
        float low =
            float(int(value & 15u) - 8) * scale;
        float high =
            float(int(value >> 4u) - 8) * scale;
        for (uint local = 0u;
             local < uint(TILE_M);
             ++local) {
            uint row =
                min(first_row + local, uint(M) - 1u);
            uint input_base = row * uint(K);
            accumulators[local] = fma(
                low,
                float(x[input_base + column]),
                accumulators[local]);
            accumulators[local] = fma(
                high,
                float(x[input_base + column + 1u]),
                accumulators[local]);
        }
    }

    for (uint local = 0u;
         local < uint(TILE_M);
         ++local) {
        accumulators[local] +=
            simd_shuffle_down(accumulators[local], 4);
        accumulators[local] +=
            simd_shuffle_down(accumulators[local], 2);
        accumulators[local] +=
            simd_shuffle_down(accumulators[local], 1);
        uint row = first_row + local;
        if (
            k_lane == 0u
            && output_index < uint(OUT)
            && row < uint(M)
        ) {
            y[row * uint(OUT) + output_index] =
                T(accumulators[local]);
        }
    }
)METAL";

constexpr const char* kInt4DequantizeSource = R"METAL(
    uint linear = thread_position_in_grid.x;
    if (linear >= uint(OUT) * uint(K)) {
        return;
    }
    uint output = linear / uint(K);
    uint column = linear - output * uint(K);
    uint value = uint(packed[
        output * uint(K / 2) + (column >> 1)
    ]);
    uint quantized = (column & 1u) == 0u
        ? value & 15u
        : value >> 4u;
    float scale = float(scales[
        output * uint(GROUPS)
        + column / uint(GROUP_SIZE)
    ]);
    y[linear] =
        T(float(int(quantized) - 8) * scale);
)METAL";

constexpr const char* kInt4EmbeddingSource = R"METAL(
    uint linear = thread_position_in_grid.x;
    if (linear >= uint(COUNT) * uint(K)) {
        return;
    }
    uint item = linear / uint(K);
    uint column = linear - item * uint(K);
    int output = int(token_ids[item]);
    if (output < 0 || output >= int(OUT)) {
        y[linear] = T(0.0f);
        return;
    }
    uint value = uint(packed[
        uint(output) * uint(K / 2)
        + (column >> 1)
    ]);
    uint quantized = (column & 1u) == 0u
        ? value & 15u
        : value >> 4u;
    float scale = float(scales[
        uint(output) * uint(GROUPS)
        + column / uint(GROUP_SIZE)
    ]);
    y[linear] =
        T(float(int(quantized) - 8) * scale);
)METAL";

constexpr const char* kInt4GroupedRowSource = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint task = threadgroup_position_in_grid.x;
    uint local_output =
        task % uint(OUT_PER_GROUP);
    uint group_row =
        task / uint(OUT_PER_GROUP);
    uint group =
        group_row % uint(GROUP_COUNT);
    uint row =
        group_row / uint(GROUP_COUNT);
    if (row >= uint(M)) {
        return;
    }
    uint output =
        group * uint(OUT_PER_GROUP)
        + local_output;
    uint packed_base =
        output * uint(K / 2);
    uint scale_base =
        output * uint(GROUPS);
    uint input_base =
        (row * uint(GROUP_COUNT) + group)
        * uint(K);
    float accumulator = 0.0f;
    for (
        uint column = lane * 2u;
        column < uint(K);
        column += 64u
    ) {
        uint value = uint(
            packed[packed_base + (column >> 1)]);
        float scale = float(scales[
            scale_base
            + column / uint(GROUP_SIZE)
        ]);
        accumulator = fma(
            float(int(value & 15u) - 8) * scale,
            float(x[input_base + column]),
            accumulator);
        accumulator = fma(
            float(int(value >> 4u) - 8) * scale,
            float(x[input_base + column + 1u]),
            accumulator);
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) {
        y[
            (row * uint(GROUP_COUNT) + group)
                * uint(OUT_PER_GROUP)
            + local_output
        ] = T(accumulator);
    }
)METAL";

constexpr const char* kPqMatmulSource = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5;
    uint output = workgroup % uint(OUT);
    uint row_tile = workgroup / uint(OUT);
    uint first_row = row_tile * uint(TILE_M);
    if (output >= uint(OUT) || first_row >= uint(M)) {
        return;
    }

    float accumulators[TILE_M];
    for (uint row = 0u;
         row < uint(TILE_M);
         ++row) {
        accumulators[row] = 0.0f;
    }
    for (uint block = lane;
         block < uint(BLOCKS);
         block += 32u) {
        uint code = mfq_tpq_read_index(
            indices,
            output * uint(BLOCKS) + block,
            uint(INDEX_BITS));
        uint code_base = code * uint(VECTOR_SIZE);
        uint column_base = block * uint(VECTOR_SIZE);
        for (uint component = 0u;
             component < uint(VECTOR_SIZE);
             ++component) {
            float weight =
                float(codebook[code_base + component]);
            uint column = column_base + component;
            for (uint local = 0u;
                 local < uint(TILE_M);
                 ++local) {
                uint row = first_row + local;
                if (row < uint(M)) {
                    accumulators[local] = fma(
                        weight,
                        float(x[row * uint(K) + column]),
                        accumulators[local]);
                }
            }
        }
    }
    for (uint local = 0u;
         local < uint(TILE_M);
         ++local) {
        float total = simd_sum(accumulators[local]);
        uint row = first_row + local;
        if (lane == 0u && row < uint(M)) {
            y[row * uint(OUT) + output] = T(total);
        }
    }
)METAL";

constexpr const char* kPqGemvSource = R"METAL(
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group =
        simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;
    float accumulators[ROWS_PER_SIMD] = {0.0f};

    for (uint block = lane;
         block < uint(BLOCKS);
         block += 32u) {
        uint column_base = block * uint(VECTOR_SIZE);
        float activations[VECTOR_SIZE];
        for (uint component = 0u;
             component < uint(VECTOR_SIZE);
             ++component) {
            activations[component] =
                float(x[column_base + component]);
        }
        for (uint local = 0u;
             local < ROWS_PER_SIMD;
             ++local) {
            uint output =
                min(output_base + local, uint(OUT) - 1u);
            uint code = mfq_tpq_read_index(
                indices,
                output * uint(BLOCKS) + block,
                uint(INDEX_BITS));
            uint code_base = code * uint(VECTOR_SIZE);
            for (uint component = 0u;
                 component < uint(VECTOR_SIZE);
                 ++component) {
                accumulators[local] = fma(
                    float(codebook[
                        code_base + component
                    ]),
                    activations[component],
                    accumulators[local]);
            }
        }
    }
    for (uint local = 0u;
         local < ROWS_PER_SIMD;
         ++local) {
        float total = simd_sum(accumulators[local]);
        uint output = output_base + local;
        if (lane == 0u && output < uint(OUT)) {
            y[output] = T(total);
        }
    }
)METAL";

constexpr const char* kPqMmqSource = R"METAL(
    constexpr uint K_LANES = 8u;
    constexpr uint ROWS_PER_SIMD =
        32u / K_LANES;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group =
        simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_row = lane / K_LANES;
    uint output_index =
        threadgroup_position_in_grid.y * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD
        + simd_row;
    uint output =
        min(output_index, uint(OUT) - 1u);
    uint first_row =
        threadgroup_position_in_grid.x
        * uint(TILE_M);

    float accumulators[TILE_M];
    for (uint row = 0u;
         row < uint(TILE_M);
         ++row) {
        accumulators[row] = 0.0f;
    }
    for (uint block = k_lane;
         block < uint(BLOCKS);
         block += K_LANES) {
        uint code = mfq_tpq_read_index(
            indices,
            output * uint(BLOCKS) + block,
            uint(INDEX_BITS));
        uint code_base = code * uint(VECTOR_SIZE);
        uint column_base = block * uint(VECTOR_SIZE);
        for (uint component = 0u;
             component < uint(VECTOR_SIZE);
             ++component) {
            float weight =
                float(codebook[code_base + component]);
            uint column = column_base + component;
            for (uint local = 0u;
                 local < uint(TILE_M);
                 ++local) {
                uint row =
                    min(first_row + local, uint(M) - 1u);
                accumulators[local] = fma(
                    weight,
                    float(x[row * uint(K) + column]),
                    accumulators[local]);
            }
        }
    }
    for (uint local = 0u;
         local < uint(TILE_M);
         ++local) {
        accumulators[local] +=
            simd_shuffle_down(accumulators[local], 4);
        accumulators[local] +=
            simd_shuffle_down(accumulators[local], 2);
        accumulators[local] +=
            simd_shuffle_down(accumulators[local], 1);
        uint row = first_row + local;
        if (
            k_lane == 0u
            && output_index < uint(OUT)
            && row < uint(M)
        ) {
            y[row * uint(OUT) + output_index] =
                T(accumulators[local]);
        }
    }
)METAL";

constexpr const char* kPqDequantizeSource = R"METAL(
    uint linear = thread_position_in_grid.x;
    if (linear >= uint(OUT) * uint(K)) {
        return;
    }
    uint output = linear / uint(K);
    uint column = linear - output * uint(K);
    uint block = column / uint(VECTOR_SIZE);
    uint component =
        column - block * uint(VECTOR_SIZE);
    uint code = mfq_tpq_read_index(
        indices,
        output * uint(BLOCKS) + block,
        uint(INDEX_BITS));
    y[linear] =
        T(codebook[code * uint(VECTOR_SIZE) + component]);
)METAL";

class BlobCursor {
public:
    explicit BlobCursor(
        const std::vector<std::uint8_t>& blob,
        std::string context)
        : blob_(blob),
          context_(std::move(context)) {}

    template <typename T>
    T scalar(const char* name) {
        require(sizeof(T), name);
        T value{};
        std::memcpy(
            &value,
            blob_.data() + offset_,
            sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    std::vector<std::uint8_t> bytes(
        std::size_t count,
        const char* name) {
        require(count, name);
        std::vector<std::uint8_t> result(
            blob_.begin()
                + static_cast<std::ptrdiff_t>(offset_),
            blob_.begin()
                + static_cast<std::ptrdiff_t>(
                    offset_ + count));
        offset_ += count;
        return result;
    }

    std::size_t offset() const noexcept {
        return offset_;
    }
    std::size_t remaining() const noexcept {
        return blob_.size() - offset_;
    }

private:
    void require(
        std::size_t count,
        const char* name) const {
        if (count > blob_.size() - offset_) {
            throw std::runtime_error(
                "truncated " + context_ + " " + name);
        }
    }

    const std::vector<std::uint8_t>& blob_;
    std::string context_;
    std::size_t offset_ = 0;
};

std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* context) {
    if (right != 0 &&
        left >
            std::numeric_limits<std::size_t>::max()
                / right) {
        throw std::runtime_error(
            std::string(context) + " size overflows");
    }
    return left * right;
}

std::size_t packed_size(
    std::size_t count,
    int bits,
    const char* context) {
    if (
        bits <= 0 || bits > 16 ||
        count >
            (
                std::numeric_limits<std::size_t>::max()
                - 7
            ) / static_cast<std::size_t>(bits)
    ) {
        throw std::runtime_error(
            std::string("invalid ") + context
            + " packed bit count");
    }
    return (
        count * static_cast<std::size_t>(bits) + 7
    ) / 8;
}

int checked_positive_int(
    std::uint64_t value,
    const char* context) {
    if (
        value == 0 ||
        value >
            static_cast<std::uint64_t>(
                std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            std::string("invalid ") + context);
    }
    return static_cast<int>(value);
}

float half_to_float(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const auto exponent =
        static_cast<unsigned>((bits >> 10) & 0x1fu);
    const auto mantissa =
        static_cast<unsigned>(bits & 0x03ffu);
    float value = 0.0f;
    if (exponent == 0) {
        value = std::ldexp(
            static_cast<float>(mantissa),
            -24);
    } else if (exponent == 31) {
        value = mantissa == 0
            ? std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(
            1.0f
                + static_cast<float>(mantissa)
                    / 1024.0f,
            static_cast<int>(exponent) - 15);
    }
    return negative ? -value : value;
}

std::uint32_t packed_index(
    const std::vector<std::uint8_t>& values,
    std::size_t index,
    int bits) {
    const auto bit_offset =
        index * static_cast<std::size_t>(bits);
    const auto byte_offset = bit_offset >> 3;
    const auto shift =
        static_cast<unsigned>(bit_offset & 7);
    std::uint32_t packed = 0;
    for (unsigned byte = 0; byte < 3; ++byte) {
        if (byte_offset + byte < values.size()) {
            packed |=
                static_cast<std::uint32_t>(
                    values[byte_offset + byte])
                << (byte * 8);
        }
    }
    return (
        packed >> shift
    ) & ((1u << bits) - 1u);
}

array make_u8_array(
    std::vector<std::uint8_t> values,
    Shape shape) {
    auto result = array(
        mlx::core::allocator::malloc(values.size()),
        std::move(shape),
        mlx::core::uint8);
    std::memcpy(
        result.data<std::uint8_t>(),
        values.data(),
        values.size());
    return result;
}

array make_float16_array(
    std::vector<std::uint16_t> values,
    Shape shape) {
    const auto bytes = values.size() * sizeof(std::uint16_t);
    auto result = array(
        mlx::core::allocator::malloc(bytes),
        std::move(shape),
        mlx::core::float16);
    std::memcpy(
        result.data<std::uint16_t>(),
        values.data(),
        bytes);
    return result;
}

array make_float16_codebook(
    std::vector<float> values,
    Shape shape) {
    auto source =
        array(values.begin(), shape);
    return mlx::core::contiguous(
        mlx::core::astype(
            source,
            mlx::core::float16));
}

struct PqProfile {
    std::string label;
    int tier;
    int vector_size;
    int entries;
};

PqProfile pq_profile(std::string_view dtype) {
    if (dtype == "TPQ-X") {
        return {std::string(dtype), 1, 8, 256};
    }
    if (dtype == "TPQ-W") {
        return {std::string(dtype), 2, 8, 4096};
    }
    if (dtype == "TPQ-V") {
        return {std::string(dtype), 3, 4, 256};
    }
    if (dtype == "TPQ-VV") {
        return {std::string(dtype), 4, 4, 4096};
    }
    throw std::runtime_error(
        "unsupported native Metal TPQ-PQ dtype: "
        + std::string(dtype));
}

bool valid_storage_bits(
    int entries,
    int bits) {
    if (entries == 256) {
        return bits == 8 ||
            bits == 12 ||
            bits == 14;
    }
    if (entries == 4096) {
        return bits == 12 ||
            bits == 14 ||
            bits == 16;
    }
    return false;
}

mlx::core::fast::CustomKernelFunction make_kernel(
    std::string name,
    std::vector<std::string> inputs,
    const char* source,
    const char* header = "") {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        std::move(name),
        std::move(inputs),
        {"y"},
        source,
        header,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction&
int4_matmul_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_tpq_i4_packed_matmul",
        {"packed", "scales", "x"},
        kInt4MatmulSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
int4_gemv_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_tpq_i4_packed_gemv",
        {"packed", "scales", "x"},
        kInt4GemvSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
int4_mmq_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_tpq_i4_packed_mmq",
        {"packed", "scales", "x"},
        kInt4MmqSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
int4_dequantize_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_tpq_i4_dequantize",
        {"packed", "scales"},
        kInt4DequantizeSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
int4_embedding_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_tpq_i4_embedding",
        {"packed", "scales", "token_ids"},
        kInt4EmbeddingSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
int4_grouped_row_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_tpq_i4_grouped_row",
        {"packed", "scales", "x"},
        kInt4GroupedRowSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
pq_matmul_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_tpq_pq_packed_matmul",
        {"indices", "codebook", "x"},
        kPqMatmulSource,
        kTpqIndexHeader);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
pq_gemv_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_tpq_pq_packed_gemv",
        {"indices", "codebook", "x"},
        kPqGemvSource,
        kTpqIndexHeader);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
pq_mmq_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_tpq_pq_packed_mmq",
        {"indices", "codebook", "x"},
        kPqMmqSource,
        kTpqIndexHeader);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
pq_dequantize_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_tpq_pq_dequantize",
        {"indices", "codebook"},
        kPqDequantizeSource,
        kTpqIndexHeader);
    return kernel;
}

std::vector<std::pair<
    std::string,
    mlx::core::fast::TemplateArg>>
int4_templates(
    Dtype dtype,
    int input_size,
    int output_size,
    int group_size,
    int groups) {
    return {
        {"T", dtype},
        {"OUT", output_size},
        {"K", input_size},
        {"GROUP_SIZE", group_size},
        {"GROUPS", groups},
    };
}

std::vector<std::pair<
    std::string,
    mlx::core::fast::TemplateArg>>
pq_templates(
    Dtype dtype,
    int input_size,
    int output_size,
    int vector_size,
    int blocks,
    int index_bits) {
    return {
        {"T", dtype},
        {"OUT", output_size},
        {"K", input_size},
        {"VECTOR_SIZE", vector_size},
        {"BLOCKS", blocks},
        {"INDEX_BITS", index_bits},
    };
}

std::tuple<
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    int,
    int>
packed_grid(
    int rows,
    int output_size,
    int tile_rows,
    const char* context) {
    const bool gemv = rows == 1 && tile_rows == 1;
    const bool mmq =
        rows >= 2 && rows <= 16 &&
        tile_rows == rows;
    if (gemv) {
        const auto output_tiles =
            (
                static_cast<std::size_t>(output_size)
                + 7
            ) / 8;
        const auto grid_x = checked_product(
            output_tiles,
            std::size_t{64},
            context);
        if (
            grid_x >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max())
        ) {
            throw std::runtime_error(
                std::string(context)
                + " GEMV grid exceeds MLX limits");
        }
        return {
            {
                static_cast<int>(grid_x),
                1,
                1,
            },
            {64, 1, 1},
            tile_rows,
            0,
        };
    }
    if (mmq) {
        const int row_tiles = (rows + 4) / 5;
        const int effective_tile =
            (rows + row_tiles - 1) / row_tiles;
        const auto grid_x = checked_product(
            static_cast<std::size_t>(row_tiles),
            std::size_t{64},
            context);
        const auto grid_y =
            (
                static_cast<std::size_t>(output_size)
                + 7
            ) / 8;
        if (
            grid_x >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max()) ||
            grid_y >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max())
        ) {
            throw std::runtime_error(
                std::string(context)
                + " MMQ grid exceeds MLX limits");
        }
        return {
            {
                static_cast<int>(grid_x),
                static_cast<int>(grid_y),
                1,
            },
            {64, 1, 1},
            effective_tile,
            1,
        };
    }
    const auto row_tiles =
        (
            static_cast<std::size_t>(rows)
            + static_cast<std::size_t>(tile_rows)
            - 1
        ) / static_cast<std::size_t>(tile_rows);
    const auto workgroups = checked_product(
        row_tiles,
        static_cast<std::size_t>(output_size),
        context);
    const auto grid_x = checked_product(
        workgroups,
        std::size_t{32},
        context);
    if (
        grid_x >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            std::string(context)
            + " GEMM grid exceeds MLX limits");
    }
    return {
        {
            static_cast<int>(grid_x),
            1,
            1,
        },
        {32, 1, 1},
        tile_rows,
        2,
    };
}

} // namespace

bool is_tpq_dtype(std::string_view dtype) noexcept {
    return dtype == "TPQ-I4G64" ||
        dtype == "TPQ-X" ||
        dtype == "TPQ-W" ||
        dtype == "TPQ-V" ||
        dtype == "TPQ-VV";
}

MlxTpqInt4Weight::MlxTpqInt4Weight(
    array packed,
    array scales,
    int input_size,
    int output_size,
    int group_size,
    int groups)
    : packed_(std::move(packed)),
      scales_(std::move(scales)),
      input_size_(input_size),
      output_size_(output_size),
      group_size_(group_size),
      groups_(groups) {}

MlxTpqInt4Weight
MlxTpqInt4Weight::from_blob(
    const std::vector<std::uint8_t>& blob) {
    BlobCursor cursor(blob, "TPQ-I4G64");
    const auto magic = cursor.bytes(4, "magic");
    if (
        std::memcmp(magic.data(), "CI41", 4) != 0 ||
        cursor.scalar<std::uint8_t>("version") != 1
    ) {
        throw std::runtime_error(
            "invalid TPQ-I4G64 header");
    }
    const auto padding = cursor.bytes(3, "padding");
    if (
        std::any_of(
            padding.begin(),
            padding.end(),
            [](std::uint8_t value) {
                return value != 0;
            })
    ) {
        throw std::runtime_error(
            "TPQ-I4G64 reserved padding is nonzero");
    }
    const auto group_size =
        cursor.scalar<std::uint32_t>("group size");
    const auto axis =
        cursor.scalar<std::int32_t>("axis");
    const auto neuron_len =
        cursor.scalar<std::int32_t>("neuron length");
    const auto dimensions =
        cursor.scalar<std::uint32_t>(
            "dimension count");
    if (
        group_size != kInt4GroupSize ||
        axis != 0 ||
        neuron_len <= 0 ||
        neuron_len % kInt4GroupSize != 0 ||
        neuron_len % 2 != 0 ||
        dimensions != 2
    ) {
        throw std::runtime_error(
            "TPQ-I4G64 requires a row-major "
            "group-64 rank-two matrix");
    }
    const auto output_shape =
        cursor.scalar<std::int64_t>("output shape");
    const auto input_shape =
        cursor.scalar<std::int64_t>("input shape");
    const auto rows =
        cursor.scalar<std::uint32_t>("row count");
    const auto groups =
        cursor.scalar<std::uint32_t>("group count");
    if (
        output_shape <= 0 ||
        output_shape !=
            static_cast<std::int64_t>(rows) ||
        input_shape != neuron_len ||
        groups !=
            static_cast<std::uint32_t>(
                neuron_len / kInt4GroupSize)
    ) {
        throw std::runtime_error(
            "inconsistent TPQ-I4G64 dimensions");
    }
    const int output_size =
        checked_positive_int(
            rows,
            "TPQ-I4G64 row count");
    const int input_size =
        checked_positive_int(
            static_cast<std::uint64_t>(neuron_len),
            "TPQ-I4G64 input width");
    const int group_count =
        checked_positive_int(
            groups,
            "TPQ-I4G64 group count");
    const auto packed_count = checked_product(
        static_cast<std::size_t>(output_size),
        static_cast<std::size_t>(input_size / 2),
        "TPQ-I4G64 packed values");
    const auto scale_count = checked_product(
        static_cast<std::size_t>(output_size),
        static_cast<std::size_t>(group_count),
        "TPQ-I4G64 scales");
    const auto expected = packed_count +
        checked_product(
            scale_count,
            sizeof(std::uint16_t),
            "TPQ-I4G64 scales");
    if (cursor.remaining() != expected) {
        throw std::runtime_error(
            "invalid TPQ-I4G64 payload length");
    }
    auto packed =
        cursor.bytes(
            packed_count,
            "packed values");
    std::vector<std::uint16_t> scales(
        scale_count);
    for (std::size_t index = 0;
         index < scale_count;
         ++index) {
        scales[index] =
            cursor.scalar<std::uint16_t>("scale");
        const float value =
            half_to_float(scales[index]);
        if (!std::isfinite(value) || value < 0.0f) {
            throw std::runtime_error(
                "TPQ-I4G64 scale must be "
                "finite and nonnegative");
        }
    }
    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "trailing bytes in TPQ-I4G64 tensor");
    }
    return MlxTpqInt4Weight(
        make_u8_array(
            std::move(packed),
            Shape{
                output_size,
                input_size / 2,
            }),
        make_float16_array(
            std::move(scales),
            Shape{
                output_size,
                group_count,
            }),
        input_size,
        output_size,
        kInt4GroupSize,
        group_count);
}

std::size_t
MlxTpqInt4Weight::packed_nbytes() const noexcept {
    return packed_.nbytes() + scales_.nbytes();
}

array MlxTpqInt4Weight::dequantize(
    Dtype dtype) const {
    if (
        dtype != mlx::core::float16 &&
        dtype != mlx::core::float32
    ) {
        throw std::runtime_error(
            "TPQ-I4G64 dequantization requires "
            "float16 or float32 output");
    }
    const auto elements = checked_product(
        static_cast<std::size_t>(output_size_),
        static_cast<std::size_t>(input_size_),
        "TPQ-I4G64 dequantization");
    if (
        elements >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "TPQ-I4G64 dequantization grid "
            "exceeds MLX limits");
    }
    auto outputs = int4_dequantize_kernel()(
        {packed_, scales_},
        {Shape{output_size_, input_size_}},
        {dtype},
        {static_cast<int>(elements), 1, 1},
        {
            std::min(
                256,
                std::max(
                    1,
                    static_cast<int>(elements))),
            1,
            1,
        },
        int4_templates(
            dtype,
            input_size_,
            output_size_,
            group_size_,
            groups_),
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array MlxTpqInt4Weight::embedding(
    const array& token_ids,
    Dtype dtype) const {
    if (
        dtype != mlx::core::float16 &&
        dtype != mlx::core::float32
    ) {
        throw std::runtime_error(
            "TPQ-I4G64 embedding requires "
            "float16 or float32 output");
    }
    auto ids = token_ids;
    if (
        ids.dtype() != mlx::core::int32 &&
        ids.dtype() != mlx::core::uint32
    ) {
        ids = mlx::core::astype(
            ids,
            mlx::core::int32);
    }
    const auto count = ids.size();
    Shape output_shape = ids.shape();
    output_shape.push_back(input_size_);
    if (count == 0) {
        return mlx::core::zeros(
            std::move(output_shape),
            dtype);
    }
    const auto elements = checked_product(
        count,
        static_cast<std::size_t>(input_size_),
        "TPQ-I4G64 embedding");
    if (
        count >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max()) ||
        elements >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "TPQ-I4G64 embedding grid "
            "exceeds MLX limits");
    }
    ids = mlx::core::contiguous(
        mlx::core::reshape(
            ids,
            Shape{static_cast<int>(count)}));
    auto templates = int4_templates(
        dtype,
        input_size_,
        output_size_,
        group_size_,
        groups_);
    templates.emplace_back(
        "COUNT",
        static_cast<int>(count));
    auto outputs = int4_embedding_kernel()(
        {packed_, scales_, ids},
        {
            Shape{
                static_cast<int>(count),
                input_size_,
            },
        },
        {dtype},
        {static_cast<int>(elements), 1, 1},
        {
            std::min(
                256,
                static_cast<int>(elements)),
            1,
            1,
        },
        std::move(templates),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        std::move(outputs.front()),
        std::move(output_shape));
}

array MlxTpqInt4Weight::prepare_input(
    const array& input,
    Shape& prefix,
    int& rows) const {
    if (
        input.ndim() == 0 ||
        input.shape(-1) != input_size_
    ) {
        throw std::runtime_error(
            "TPQ-I4G64 input width does not "
            "match packed weight");
    }
    auto source = input;
    if (
        source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32
    ) {
        source = mlx::core::astype(
            source,
            mlx::core::float16);
    }
    prefix.clear();
    std::size_t row_count = 1;
    for (std::size_t dimension = 0;
         dimension + 1 < source.ndim();
         ++dimension) {
        const int extent =
            source.shape(
                static_cast<int>(dimension));
        if (extent < 0) {
            throw std::runtime_error(
                "invalid TPQ-I4G64 input shape");
        }
        prefix.push_back(extent);
        row_count = checked_product(
            row_count,
            static_cast<std::size_t>(extent),
            "TPQ-I4G64 input rows");
    }
    if (
        row_count >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "TPQ-I4G64 input row count "
            "exceeds MLX limits");
    }
    rows = static_cast<int>(row_count);
    return mlx::core::contiguous(
        mlx::core::reshape(
            source,
            Shape{rows, input_size_}));
}

array MlxTpqInt4Weight::reshape_output(
    array value,
    const Shape& prefix) const {
    Shape output_shape = prefix;
    output_shape.push_back(output_size_);
    return mlx::core::reshape(
        std::move(value),
        std::move(output_shape));
}

array MlxTpqInt4Weight::packed_matmul(
    const array& source,
    const Shape& prefix,
    int rows,
    int tile_rows) const {
    if (rows == 0) {
        Shape output_shape = prefix;
        output_shape.push_back(output_size_);
        return mlx::core::zeros(
            std::move(output_shape),
            source.dtype());
    }
    if (tile_rows <= 0 || tile_rows > 16) {
        throw std::runtime_error(
            "invalid TPQ-I4G64 packed row tile");
    }
    auto [grid, threadgroup, effective_tile, kind] =
        packed_grid(
            rows,
            output_size_,
            tile_rows,
            "TPQ-I4G64");
    auto templates = int4_templates(
        source.dtype(),
        input_size_,
        output_size_,
        group_size_,
        groups_);
    templates.emplace_back("M", rows);
    templates.emplace_back(
        "TILE_M",
        effective_tile);
    const auto& kernel = kind == 0
        ? int4_gemv_kernel()
        : (
            kind == 1
            ? int4_mmq_kernel()
            : int4_matmul_kernel()
        );
    auto outputs = kernel(
        {packed_, scales_, source},
        {Shape{rows, output_size_}},
        {source.dtype()},
        grid,
        threadgroup,
        std::move(templates),
        std::nullopt,
        false,
        {});
    return reshape_output(
        std::move(outputs.front()),
        prefix);
}

array MlxTpqInt4Weight::gemv(
    const array& input) const {
    Shape prefix;
    int rows = 0;
    auto source =
        prepare_input(input, prefix, rows);
    if (rows != 1) {
        throw std::runtime_error(
            "TPQ-I4G64 GEMV requires "
            "exactly one input row");
    }
    return packed_matmul(
        source,
        prefix,
        rows,
        1);
}

array MlxTpqInt4Weight::mmq(
    const array& input) const {
    Shape prefix;
    int rows = 0;
    auto source =
        prepare_input(input, prefix, rows);
    if (rows < 2 || rows > 16) {
        throw std::runtime_error(
            "TPQ-I4G64 MMQ requires "
            "2 to 16 input rows");
    }
    return packed_matmul(
        source,
        prefix,
        rows,
        rows);
}

array MlxTpqInt4Weight::gemm(
    const array& input) const {
    Shape prefix;
    int rows = 0;
    auto source =
        prepare_input(input, prefix, rows);
    return packed_matmul(
        source,
        prefix,
        rows,
        8);
}

array MlxTpqInt4Weight::matmul(
    const array& input) const {
    Shape prefix;
    int rows = 0;
    auto source =
        prepare_input(input, prefix, rows);
    if (rows == 0) {
        return packed_matmul(
            source,
            prefix,
            rows,
            1);
    }
    if (rows >= 64) {
        auto dense = dequantize(source.dtype());
        auto result = mlx::core::matmul(
            source,
            mlx::core::transpose(dense));
        return reshape_output(
            std::move(result),
            prefix);
    }
    if (rows == 1) {
        return packed_matmul(
            source,
            prefix,
            rows,
            1);
    }
    if (rows <= 16) {
        return packed_matmul(
            source,
            prefix,
            rows,
            rows);
    }
    return packed_matmul(
        source,
        prefix,
        rows,
        8);
}

array MlxTpqInt4Weight::grouped_row_matmul(
    const array& input,
    int group_count) const {
    if (
        input.ndim() < 2 ||
        group_count <= 0 ||
        input.shape(-2) != group_count ||
        input.shape(-1) != input_size_ ||
        output_size_ % group_count != 0
    ) {
        throw std::runtime_error(
            "TPQ-I4G64 grouped-row input "
            "or weight shape is incompatible");
    }
    auto source = input;
    if (
        source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32
    ) {
        source = mlx::core::astype(
            source,
            mlx::core::float16);
    }
    Shape prefix(
        source.shape().begin(),
        source.shape().end() - 2);
    std::size_t row_count = 1;
    for (const int extent : prefix) {
        if (extent < 0) {
            throw std::runtime_error(
                "invalid TPQ-I4G64 grouped-row shape");
        }
        row_count = checked_product(
            row_count,
            static_cast<std::size_t>(extent),
            "TPQ-I4G64 grouped-row rows");
    }
    const int out_per_group =
        output_size_ / group_count;
    Shape output_shape = prefix;
    output_shape.push_back(group_count);
    output_shape.push_back(out_per_group);
    if (row_count == 0) {
        return mlx::core::zeros(
            std::move(output_shape),
            source.dtype());
    }
    if (
        row_count >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "TPQ-I4G64 grouped-row count "
            "exceeds MLX limits");
    }
    const int rows =
        static_cast<int>(row_count);
    source = mlx::core::contiguous(
        mlx::core::reshape(
            source,
            Shape{
                rows,
                group_count,
                input_size_,
            }));
    const auto tasks = checked_product(
        checked_product(
            row_count,
            static_cast<std::size_t>(group_count),
            "TPQ-I4G64 grouped-row tasks"),
        static_cast<std::size_t>(out_per_group),
        "TPQ-I4G64 grouped-row tasks");
    const auto grid_x = checked_product(
        tasks,
        std::size_t{32},
        "TPQ-I4G64 grouped-row grid");
    if (
        grid_x >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "TPQ-I4G64 grouped-row grid "
            "exceeds MLX limits");
    }
    auto templates = int4_templates(
        source.dtype(),
        input_size_,
        output_size_,
        group_size_,
        groups_);
    templates.emplace_back("M", rows);
    templates.emplace_back(
        "GROUP_COUNT",
        group_count);
    templates.emplace_back(
        "OUT_PER_GROUP",
        out_per_group);
    auto outputs = int4_grouped_row_kernel()(
        {packed_, scales_, source},
        {
            Shape{
                rows,
                group_count,
                out_per_group,
            },
        },
        {source.dtype()},
        {
            static_cast<int>(grid_x),
            1,
            1,
        },
        {32, 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        std::move(outputs.front()),
        std::move(output_shape));
}

MlxTpqPqWeight::MlxTpqPqWeight(
    array indices,
    array codebook,
    std::string format_label,
    int input_size,
    int output_size,
    int vector_size,
    int blocks,
    int entries,
    int index_bits)
    : indices_(std::move(indices)),
      codebook_(std::move(codebook)),
      format_label_(std::move(format_label)),
      input_size_(input_size),
      output_size_(output_size),
      vector_size_(vector_size),
      blocks_(blocks),
      entries_(entries),
      index_bits_(index_bits) {}

MlxTpqPqWeight MlxTpqPqWeight::from_blob(
    std::string_view dtype,
    const std::vector<std::uint8_t>& blob) {
    const auto profile = pq_profile(dtype);
    BlobCursor cursor(blob, profile.label);
    const auto magic = cursor.bytes(4, "magic");
    if (
        std::memcmp(magic.data(), "CPQ1", 4) != 0 ||
        cursor.scalar<std::uint8_t>("version") != 1
    ) {
        throw std::runtime_error(
            "invalid " + std::string(profile.label)
            + " header");
    }
    const int tier =
        cursor.scalar<std::uint8_t>("tier");
    const int vector_size =
        cursor.scalar<std::uint8_t>("vector size");
    const int index_bits =
        cursor.scalar<std::uint8_t>("index bits");
    const int axis =
        cursor.scalar<std::int32_t>("axis");
    const int neuron_len =
        cursor.scalar<std::int32_t>("neuron length");
    const auto dimensions =
        cursor.scalar<std::uint32_t>(
            "dimension count");
    const auto entries =
        cursor.scalar<std::uint32_t>(
            "codebook entries");
    if (
        tier != profile.tier ||
        vector_size != profile.vector_size ||
        entries !=
            static_cast<std::uint32_t>(
                profile.entries) ||
        !valid_storage_bits(
            profile.entries,
            index_bits) ||
        axis != 0 ||
        neuron_len <= 0 ||
        neuron_len % vector_size != 0 ||
        dimensions != 2
    ) {
        throw std::runtime_error(
            "inconsistent " + std::string(profile.label)
            + " tier metadata");
    }
    const auto output_shape =
        cursor.scalar<std::int64_t>("output shape");
    const auto input_shape =
        cursor.scalar<std::int64_t>("input shape");
    const auto rows =
        cursor.scalar<std::uint32_t>("row count");
    if (
        output_shape <= 0 ||
        output_shape !=
            static_cast<std::int64_t>(rows) ||
        input_shape != neuron_len
    ) {
        throw std::runtime_error(
            "inconsistent " + std::string(profile.label)
            + " matrix dimensions");
    }
    const int output_size =
        checked_positive_int(
            rows,
            "TPQ-PQ row count");
    const int input_size =
        checked_positive_int(
            static_cast<std::uint64_t>(neuron_len),
            "TPQ-PQ input width");
    const int blocks = input_size / vector_size;
    const auto codebook_values = checked_product(
        static_cast<std::size_t>(profile.entries),
        static_cast<std::size_t>(vector_size),
        "TPQ-PQ codebook");
    const auto codebook_bytes = checked_product(
        codebook_values,
        sizeof(float),
        "TPQ-PQ codebook");
    const auto index_count = checked_product(
        static_cast<std::size_t>(output_size),
        static_cast<std::size_t>(blocks),
        "TPQ-PQ indices");
    const auto index_bytes = packed_size(
        index_count,
        index_bits,
        "TPQ-PQ indices");
    if (
        cursor.remaining() !=
            codebook_bytes + index_bytes
    ) {
        throw std::runtime_error(
            "invalid " + std::string(profile.label)
            + " payload length");
    }
    std::vector<float> codebook(codebook_values);
    for (std::size_t index = 0;
         index < codebook_values;
         ++index) {
        codebook[index] =
            cursor.scalar<float>("codebook value");
        if (!std::isfinite(codebook[index])) {
            throw std::runtime_error(
                std::string(profile.label)
                + " codebook must be finite");
        }
    }
    auto indices =
        cursor.bytes(
            index_bytes,
            "index stream");
    for (std::size_t index = 0;
         index < index_count;
         ++index) {
        if (
            packed_index(
                indices,
                index,
                index_bits)
            >= static_cast<std::uint32_t>(
                profile.entries)
        ) {
            throw std::runtime_error(
                std::string(profile.label)
                + " index references a missing codeword");
        }
    }
    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "trailing bytes in "
            + std::string(profile.label)
            + " tensor");
    }
    indices.insert(indices.end(), 2, 0);
    return MlxTpqPqWeight(
        make_u8_array(
            std::move(indices),
            Shape{
                static_cast<int>(index_bytes + 2),
            }),
        make_float16_codebook(
            std::move(codebook),
            Shape{
                profile.entries,
                vector_size,
            }),
        profile.label,
        input_size,
        output_size,
        vector_size,
        blocks,
        profile.entries,
        index_bits);
}

std::size_t
MlxTpqPqWeight::packed_nbytes() const noexcept {
    return indices_.nbytes() + codebook_.nbytes();
}

array MlxTpqPqWeight::dequantize(
    Dtype dtype) const {
    if (
        dtype != mlx::core::float16 &&
        dtype != mlx::core::float32
    ) {
        throw std::runtime_error(
            "TPQ-PQ dequantization requires "
            "float16 or float32 output");
    }
    const auto elements = checked_product(
        static_cast<std::size_t>(output_size_),
        static_cast<std::size_t>(input_size_),
        "TPQ-PQ dequantization");
    if (
        elements >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "TPQ-PQ dequantization grid "
            "exceeds MLX limits");
    }
    auto outputs = pq_dequantize_kernel()(
        {indices_, codebook_},
        {Shape{output_size_, input_size_}},
        {dtype},
        {static_cast<int>(elements), 1, 1},
        {
            std::min(
                256,
                std::max(
                    1,
                    static_cast<int>(elements))),
            1,
            1,
        },
        pq_templates(
            dtype,
            input_size_,
            output_size_,
            vector_size_,
            blocks_,
            index_bits_),
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array MlxTpqPqWeight::prepare_input(
    const array& input,
    Shape& prefix,
    int& rows) const {
    if (
        input.ndim() == 0 ||
        input.shape(-1) != input_size_
    ) {
        throw std::runtime_error(
            "TPQ-PQ input width does not "
            "match packed weight");
    }
    auto source = input;
    if (
        source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32
    ) {
        source = mlx::core::astype(
            source,
            mlx::core::float16);
    }
    prefix.clear();
    std::size_t row_count = 1;
    for (std::size_t dimension = 0;
         dimension + 1 < source.ndim();
         ++dimension) {
        const int extent =
            source.shape(
                static_cast<int>(dimension));
        if (extent < 0) {
            throw std::runtime_error(
                "invalid TPQ-PQ input shape");
        }
        prefix.push_back(extent);
        row_count = checked_product(
            row_count,
            static_cast<std::size_t>(extent),
            "TPQ-PQ input rows");
    }
    if (
        row_count >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max())
    ) {
        throw std::runtime_error(
            "TPQ-PQ input row count "
            "exceeds MLX limits");
    }
    rows = static_cast<int>(row_count);
    return mlx::core::contiguous(
        mlx::core::reshape(
            source,
            Shape{rows, input_size_}));
}

array MlxTpqPqWeight::reshape_output(
    array value,
    const Shape& prefix) const {
    Shape output_shape = prefix;
    output_shape.push_back(output_size_);
    return mlx::core::reshape(
        std::move(value),
        std::move(output_shape));
}

array MlxTpqPqWeight::packed_matmul(
    const array& source,
    const Shape& prefix,
    int rows,
    int tile_rows) const {
    if (rows == 0) {
        Shape output_shape = prefix;
        output_shape.push_back(output_size_);
        return mlx::core::zeros(
            std::move(output_shape),
            source.dtype());
    }
    if (tile_rows <= 0 || tile_rows > 16) {
        throw std::runtime_error(
            "invalid TPQ-PQ packed row tile");
    }
    auto [grid, threadgroup, effective_tile, kind] =
        packed_grid(
            rows,
            output_size_,
            tile_rows,
            "TPQ-PQ");
    auto templates = pq_templates(
        source.dtype(),
        input_size_,
        output_size_,
        vector_size_,
        blocks_,
        index_bits_);
    templates.emplace_back("M", rows);
    templates.emplace_back(
        "TILE_M",
        effective_tile);
    const auto& kernel = kind == 0
        ? pq_gemv_kernel()
        : (
            kind == 1
            ? pq_mmq_kernel()
            : pq_matmul_kernel()
        );
    auto outputs = kernel(
        {indices_, codebook_, source},
        {Shape{rows, output_size_}},
        {source.dtype()},
        grid,
        threadgroup,
        std::move(templates),
        std::nullopt,
        false,
        {});
    return reshape_output(
        std::move(outputs.front()),
        prefix);
}

array MlxTpqPqWeight::gemv(
    const array& input) const {
    Shape prefix;
    int rows = 0;
    auto source =
        prepare_input(input, prefix, rows);
    if (rows != 1) {
        throw std::runtime_error(
            "TPQ-PQ GEMV requires exactly "
            "one input row");
    }
    return packed_matmul(
        source,
        prefix,
        rows,
        1);
}

array MlxTpqPqWeight::mmq(
    const array& input) const {
    Shape prefix;
    int rows = 0;
    auto source =
        prepare_input(input, prefix, rows);
    if (rows < 2 || rows > 16) {
        throw std::runtime_error(
            "TPQ-PQ MMQ requires 2 to 16 "
            "input rows");
    }
    return packed_matmul(
        source,
        prefix,
        rows,
        rows);
}

array MlxTpqPqWeight::gemm(
    const array& input) const {
    Shape prefix;
    int rows = 0;
    auto source =
        prepare_input(input, prefix, rows);
    return packed_matmul(
        source,
        prefix,
        rows,
        8);
}

array MlxTpqPqWeight::matmul(
    const array& input) const {
    Shape prefix;
    int rows = 0;
    auto source =
        prepare_input(input, prefix, rows);
    if (rows == 0) {
        return packed_matmul(
            source,
            prefix,
            rows,
            1);
    }
    if (rows >= 64) {
        auto dense = dequantize(source.dtype());
        auto result = mlx::core::matmul(
            source,
            mlx::core::transpose(dense));
        return reshape_output(
            std::move(result),
            prefix);
    }
    if (rows == 1) {
        return packed_matmul(
            source,
            prefix,
            rows,
            1);
    }
    if (rows <= 16) {
        return packed_matmul(
            source,
            prefix,
            rows,
            rows);
    }
    return packed_matmul(
        source,
        prefix,
        rows,
        8);
}

} // namespace mfq::metal
