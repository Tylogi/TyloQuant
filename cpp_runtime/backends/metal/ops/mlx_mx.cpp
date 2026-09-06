#include "mlx_mx.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr std::array<std::uint8_t, 4> kMagic{'M', 'X', 'T', '1'};
constexpr std::uint8_t kVersion = 1;

class Cursor {
public:
    explicit Cursor(std::span<const std::uint8_t> blob) : blob_(blob) {}

    template <typename T>
    T scalar(const char* name) {
        if (sizeof(T) > remaining()) {
            throw std::runtime_error(std::string("truncated MX ") + name);
        }
        T value{};
        std::memcpy(&value, blob_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    std::span<const std::uint8_t> bytes(std::size_t count, const char* name) {
        if (count > remaining()) {
            throw std::runtime_error(std::string("truncated MX ") + name);
        }
        auto value = blob_.subspan(offset_, count);
        offset_ += count;
        return value;
    }

    std::size_t remaining() const noexcept { return blob_.size() - offset_; }

private:
    std::span<const std::uint8_t> blob_;
    std::size_t offset_ = 0;
};

std::size_t checked_product(
    std::uint64_t left,
    std::uint64_t right,
    const char* name) {
    if (left == 0 || right == 0 ||
        left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::runtime_error(std::string("invalid MX ") + name);
    }
    return static_cast<std::size_t>(left * right);
}

int checked_dimension(std::uint64_t value, const char* name) {
    if (value == 0 || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid MX ") + name);
    }
    return static_cast<int>(value);
}

template <typename T>
array make_array(std::span<const T> values, Shape shape) {
    return array(values.begin(), std::move(shape));
}

constexpr const char* kMxHeader = R"METAL(
inline float mfq_mx_e8m0(uchar raw) {
    if (raw == 255u) {
        return NAN;
    }
    // E8M0 uses the same exponent bias as IEEE-754 float.  Constructing the
    // power of two directly avoids an exp2 for every decoded block.
    uint bits = raw == 0u ? 0x00400000u : uint(raw) << 23u;
    return as_type<float>(bits);
}

inline float mfq_mx_fp4(uchar raw) {
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

constant ushort mfq_mx_fp8_half_lut[256] = {
    0x0000u, 0x1800u, 0x1c00u, 0x1e00u, 0x2000u, 0x2100u, 0x2200u, 0x2300u,
    0x2400u, 0x2480u, 0x2500u, 0x2580u, 0x2600u, 0x2680u, 0x2700u, 0x2780u,
    0x2800u, 0x2880u, 0x2900u, 0x2980u, 0x2a00u, 0x2a80u, 0x2b00u, 0x2b80u,
    0x2c00u, 0x2c80u, 0x2d00u, 0x2d80u, 0x2e00u, 0x2e80u, 0x2f00u, 0x2f80u,
    0x3000u, 0x3080u, 0x3100u, 0x3180u, 0x3200u, 0x3280u, 0x3300u, 0x3380u,
    0x3400u, 0x3480u, 0x3500u, 0x3580u, 0x3600u, 0x3680u, 0x3700u, 0x3780u,
    0x3800u, 0x3880u, 0x3900u, 0x3980u, 0x3a00u, 0x3a80u, 0x3b00u, 0x3b80u,
    0x3c00u, 0x3c80u, 0x3d00u, 0x3d80u, 0x3e00u, 0x3e80u, 0x3f00u, 0x3f80u,
    0x4000u, 0x4080u, 0x4100u, 0x4180u, 0x4200u, 0x4280u, 0x4300u, 0x4380u,
    0x4400u, 0x4480u, 0x4500u, 0x4580u, 0x4600u, 0x4680u, 0x4700u, 0x4780u,
    0x4800u, 0x4880u, 0x4900u, 0x4980u, 0x4a00u, 0x4a80u, 0x4b00u, 0x4b80u,
    0x4c00u, 0x4c80u, 0x4d00u, 0x4d80u, 0x4e00u, 0x4e80u, 0x4f00u, 0x4f80u,
    0x5000u, 0x5080u, 0x5100u, 0x5180u, 0x5200u, 0x5280u, 0x5300u, 0x5380u,
    0x5400u, 0x5480u, 0x5500u, 0x5580u, 0x5600u, 0x5680u, 0x5700u, 0x5780u,
    0x5800u, 0x5880u, 0x5900u, 0x5980u, 0x5a00u, 0x5a80u, 0x5b00u, 0x5b80u,
    0x5c00u, 0x5c80u, 0x5d00u, 0x5d80u, 0x5e00u, 0x5e80u, 0x5f00u, 0x7e00u,
    0x8000u, 0x9800u, 0x9c00u, 0x9e00u, 0xa000u, 0xa100u, 0xa200u, 0xa300u,
    0xa400u, 0xa480u, 0xa500u, 0xa580u, 0xa600u, 0xa680u, 0xa700u, 0xa780u,
    0xa800u, 0xa880u, 0xa900u, 0xa980u, 0xaa00u, 0xaa80u, 0xab00u, 0xab80u,
    0xac00u, 0xac80u, 0xad00u, 0xad80u, 0xae00u, 0xae80u, 0xaf00u, 0xaf80u,
    0xb000u, 0xb080u, 0xb100u, 0xb180u, 0xb200u, 0xb280u, 0xb300u, 0xb380u,
    0xb400u, 0xb480u, 0xb500u, 0xb580u, 0xb600u, 0xb680u, 0xb700u, 0xb780u,
    0xb800u, 0xb880u, 0xb900u, 0xb980u, 0xba00u, 0xba80u, 0xbb00u, 0xbb80u,
    0xbc00u, 0xbc80u, 0xbd00u, 0xbd80u, 0xbe00u, 0xbe80u, 0xbf00u, 0xbf80u,
    0xc000u, 0xc080u, 0xc100u, 0xc180u, 0xc200u, 0xc280u, 0xc300u, 0xc380u,
    0xc400u, 0xc480u, 0xc500u, 0xc580u, 0xc600u, 0xc680u, 0xc700u, 0xc780u,
    0xc800u, 0xc880u, 0xc900u, 0xc980u, 0xca00u, 0xca80u, 0xcb00u, 0xcb80u,
    0xcc00u, 0xcc80u, 0xcd00u, 0xcd80u, 0xce00u, 0xce80u, 0xcf00u, 0xcf80u,
    0xd000u, 0xd080u, 0xd100u, 0xd180u, 0xd200u, 0xd280u, 0xd300u, 0xd380u,
    0xd400u, 0xd480u, 0xd500u, 0xd580u, 0xd600u, 0xd680u, 0xd700u, 0xd780u,
    0xd800u, 0xd880u, 0xd900u, 0xd980u, 0xda00u, 0xda80u, 0xdb00u, 0xdb80u,
    0xdc00u, 0xdc80u, 0xdd00u, 0xdd80u, 0xde00u, 0xde80u, 0xdf00u, 0xfe00u,
};

inline float mfq_mx_fp8(uchar raw) {
    return float(as_type<half>(mfq_mx_fp8_half_lut[uint(raw)]));
}

template <typename ValueStream, typename ScaleStream>
inline float mfq_mx_weight(
    ValueStream values,
    ScaleStream scales,
    uint output,
    uint column,
    uint mx_bits,
    uint width
) {
    if (mx_bits == 4u) {
        uchar packed = values[output * (width / 2u) + (column >> 1u)];
        uchar code = (column & 1u) == 0u ? packed & 15u : packed >> 4u;
        uchar scale = scales[output * (width / 32u) + column / 32u];
        return mfq_mx_fp4(code) * mfq_mx_e8m0(scale);
    } else {
        uchar code = values[output * width + column];
        uint scale_row = output / 128u;
        uint scale_column = column / 128u;
        uchar scale = scales[scale_row * (width / 128u) + scale_column];
        return mfq_mx_fp8(code) * mfq_mx_e8m0(scale);
    }
}
)METAL";

constexpr const char* kMxMatmul = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint workgroup = thread_position_in_grid.x >> 5u;
    uint output = workgroup % uint(OUT);
    uint row_tile = workgroup / uint(OUT);
    uint first_row = row_tile * uint(TILE_M);
    if (output >= uint(OUT) || first_row >= uint(M)) {
        return;
    }
    float accum[TILE_M];
    for (uint row = 0u; row < uint(TILE_M); ++row) {
        accum[row] = 0.0f;
    }
    for (uint column = lane; column < uint(K); column += 32u) {
        float weight = mfq_mx_weight(
            values, scales, output, column, uint(MX_BITS), uint(K));
        for (uint local = 0u; local < uint(TILE_M); ++local) {
            uint row = first_row + local;
            if (row < uint(M)) {
                accum[local] += float(x[row * uint(K) + column]) * weight;
            }
        }
    }
    for (uint local = 0u; local < uint(TILE_M); ++local) {
        uint row = first_row + local;
        float total = simd_sum(accum[local]);
        if (lane == 0u && row < uint(M)) {
            y[row * uint(OUT) + output] = T(total);
        }
    }
)METAL";

constexpr const char* kMxGemv = R"METAL(
    constexpr uint OUTPUTS_PER_SIMD = 4u;
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint OUTPUTS_PER_TG = OUTPUTS_PER_SIMD * SIMD_GROUPS;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint tg_index = thread_position_in_grid.x / 64u;
    uint first_output = tg_index * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD;
    float accum[OUTPUTS_PER_SIMD] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint column = lane; column < uint(K); column += 32u) {
        float activation = float(x[column]);
        for (uint local = 0u; local < OUTPUTS_PER_SIMD; ++local) {
            uint output = first_output + local;
            if (output < uint(OUT)) {
                accum[local] += activation
                    * mfq_mx_weight(
                        values,
                        scales,
                        output,
                        column,
                        uint(MX_BITS),
                        uint(K));
            }
        }
    }
    for (uint local = 0u; local < OUTPUTS_PER_SIMD; ++local) {
        uint output = first_output + local;
        float total = simd_sum(accum[local]);
        if (lane == 0u && output < uint(OUT)) {
            y[output] = T(total);
        }
    }
)METAL";

// FP16 single-token MXFP8 decode.  Eight lanes cooperate on each output and
// each lane owns complete 128-column MX blocks.  The E8M0 scale is therefore
// decoded once per block instead of once per weight.  Four outputs per SIMD
// group and four SIMD groups per threadgroup produce 16 rows per dispatch.
constexpr const char* kMxfp8Gemv = R"METAL(
    constexpr uint SIMD_GROUPS = 4u;
    constexpr uint K_LANES = 8u;
    constexpr uint ROWS_PER_SIMD = 32u / K_LANES;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint BLOCK = 128u;
    constexpr uint BLOCKS = uint(K) / BLOCK;

    threadgroup half fp8_lut[256];
    uint local_thread = thread_index_in_threadgroup;
    fp8_lut[local_thread] =
        as_type<half>(mfq_mx_fp8_half_lut[local_thread]);
    fp8_lut[local_thread + 128u] =
        as_type<half>(mfq_mx_fp8_half_lut[local_thread + 128u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_row = lane / K_LANES;
    uint output_index =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD + simd_row;
    uint output = min(output_index, uint(OUT) - 1u);
    uint value_base = output * uint(K);
    uint scale_base = (output / BLOCK) * BLOCKS;
    float accumulator = 0.0f;

    for (uint block = k_lane; block < BLOCKS; block += K_LANES) {
        uint column_base = block * BLOCK;
        float block_dot = 0.0f;
        for (uint element = 0u; element < BLOCK; element += 4u) {
            uint column = column_base + element;
            half4 activation = *(device const half4*)(x + column);
            uchar4 code = *(device const uchar4*)(
                values + value_base + column);
            float4 weight = float4(
                float(fp8_lut[uint(code.x)]),
                float(fp8_lut[uint(code.y)]),
                float(fp8_lut[uint(code.z)]),
                float(fp8_lut[uint(code.w)]));
            block_dot += dot(float4(activation), weight);
        }
        accumulator = fma(
            mfq_mx_e8m0(scales[scale_base + block]),
            block_dot,
            accumulator);
    }

    accumulator += simd_shuffle_down(accumulator, 4);
    accumulator += simd_shuffle_down(accumulator, 2);
    accumulator += simd_shuffle_down(accumulator, 1);
    if (k_lane == 0u && output_index < uint(OUT)) {
        y[output_index] = half(accumulator);
    }
)METAL";

// Decode-only DeepSeek-V4 O-LoRA projection.  The output row selects its
// diagonal input group, and inverse RoPE is applied while the activation is
// in registers.  This avoids both the standalone de-rotation tensor and the
// off-diagonal work of a generic multi-row matmul.
constexpr const char* kMxfp8GroupedInverseRope = R"METAL(
    constexpr uint SIMD_GROUPS = 4u;
    constexpr uint K_LANES = 8u;
    constexpr uint ROWS_PER_SIMD = 32u / K_LANES;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint BLOCK = 128u;
    constexpr uint BLOCKS = uint(K) / BLOCK;
    constexpr uint PREFIX = uint(HEAD_DIM - ROTARY);
    constexpr uint PAIRS = uint(ROTARY / 2);

    threadgroup half fp8_lut[256];
    uint local_thread = thread_index_in_threadgroup;
    fp8_lut[local_thread] =
        as_type<half>(mfq_mx_fp8_half_lut[local_thread]);
    fp8_lut[local_thread + 128u] =
        as_type<half>(mfq_mx_fp8_half_lut[local_thread + 128u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_row = lane / K_LANES;
    uint output_index =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD + simd_row;
    uint output = min(output_index, uint(OUT) - 1u);
    uint input_group = output / uint(OUT_PER_GROUP);
    uint value_base = output * uint(K);
    uint scale_base = (output / BLOCK) * BLOCKS;
    uint input_base = input_group * uint(K);
    float accumulator = 0.0f;

    for (uint block = k_lane; block < BLOCKS; block += K_LANES) {
        uint column_base = block * BLOCK;
        float block_dot = 0.0f;
        for (uint element = 0u; element < BLOCK; element += 4u) {
            uint column = column_base + element;
            half4 source = *(device const half4*)(
                x + input_base + column);
            float4 activation = float4(source);
            uint head_column = column % uint(HEAD_DIM);
            if (head_column >= PREFIX) {
                uint pair = (head_column - PREFIX) >> 1u;
                float cosine0 = float(cos_values[pair]);
                float sine0 = float(sin_values[pair]);
                float cosine1 = float(cos_values[pair + 1u]);
                float sine1 = float(sin_values[pair + 1u]);
                // Preserve the original graph's FP16 inverse-RoPE boundary.
                half rotated0 = half(
                    activation.x * cosine0 + activation.y * sine0);
                half rotated1 = half(
                    activation.y * cosine0 - activation.x * sine0);
                half rotated2 = half(
                    activation.z * cosine1 + activation.w * sine1);
                half rotated3 = half(
                    activation.w * cosine1 - activation.z * sine1);
                activation = float4(
                    float(rotated0),
                    float(rotated1),
                    float(rotated2),
                    float(rotated3));
            }
            uchar4 code = *(device const uchar4*)(
                values + value_base + column);
            float4 weight = float4(
                float(fp8_lut[uint(code.x)]),
                float(fp8_lut[uint(code.y)]),
                float(fp8_lut[uint(code.z)]),
                float(fp8_lut[uint(code.w)]));
            block_dot += dot(activation, weight);
        }
        accumulator = fma(
            mfq_mx_e8m0(scales[scale_base + block]),
            block_dot,
            accumulator);
    }

    accumulator += simd_shuffle_down(accumulator, 4);
    accumulator += simd_shuffle_down(accumulator, 2);
    accumulator += simd_shuffle_down(accumulator, 1);
    if (k_lane == 0u && output_index < uint(OUT)) {
        y[output_index] = half(accumulator);
    }
)METAL";

constexpr const char* kMxDequantize = R"METAL(
    uint index = thread_position_in_grid.x;
    uint count = uint(OUT) * uint(K);
    if (index < count) {
        uint output = index / uint(K);
        uint column = index - output * uint(K);
        y[index] = T(mfq_mx_weight(
            values,
            scales,
            output,
            column,
            uint(MX_BITS),
            uint(K)));
    }
)METAL";

constexpr const char* kMxEmbedding = R"METAL(
    uint index = thread_position_in_grid.x;
    uint count = uint(M) * uint(K);
    if (index < count) {
        uint token = index / uint(K);
        uint column = index - token * uint(K);
        uint output = uint(x[token]);
        y[index] = output < uint(OUT)
            ? T(mfq_mx_weight(
                values, scales, output, column, uint(MX_BITS), uint(K)))
            : T(NAN);
    }
)METAL";

mlx::core::fast::CustomKernelFunction make_kernel(
    std::string name,
    const char* source) {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        std::move(name),
        {"values", "scales", "x"},
        {"y"},
        source,
        kMxHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& mx_matmul_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_mx_packed_matmul", kMxMatmul);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& mx_gemv_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_mx_packed_gemv", kMxGemv);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& mxfp8_gemv_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_mxfp8_block_gemv", kMxfp8Gemv);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
mxfp8_grouped_inverse_rope_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_mxfp8_grouped_inverse_rope",
            {"values", "scales", "x", "cos_values", "sin_values"},
            {"y"},
            kMxfp8GroupedInverseRope,
            kMxHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& mx_dequantize_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_mx_dequantize", kMxDequantize);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& mx_embedding_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_mx_embedding", kMxEmbedding);
    return kernel;
}

std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
templates(Dtype dtype, int bits, int input_size, int output_size, int rows = 1,
          int tile_rows = 1) {
    return {
        {"T", dtype},
        {"MX_BITS", bits},
        {"K", input_size},
        {"OUT", output_size},
        {"M", rows},
        {"TILE_M", tile_rows},
    };
}

} // namespace

bool is_mx_dtype(std::string_view dtype) noexcept {
    return dtype == "MXFP4" || dtype == "MXFP8";
}

MlxMxWeight::MlxMxWeight(
    array values,
    array scales,
    int bits,
    int input_size,
    int output_size)
    : values_(std::move(values)),
      scales_(std::move(scales)),
      bits_(bits),
      input_size_(input_size),
      output_size_(output_size) {
    const auto* native_env = std::getenv("MFQ_METAL_MXFP8_NATIVE_QMV");
    const bool native_enabled = native_env == nullptr ||
        std::string_view(native_env) != "0";
    if (bits_ == 8 && native_enabled) {
        // MLX's native MXFP8 kernels use one E8M0 scale per output row and
        // 32 input columns. Official block-FP8 checkpoints store the same
        // scale on a 128x128 grid, so expand only the tiny scale sidecar once
        // while leaving the FP8 payload zero-copy.
        auto expanded = mlx::core::repeat(scales_, 4, 1);
        expanded = mlx::core::repeat(expanded, 128, 0);
        if (expanded.shape(0) != output_size_) {
            expanded = mlx::core::slice(
                expanded,
                Shape{0, 0},
                Shape{output_size_, input_size_ / 32});
        }
        expanded_mxfp8_scales_ = mlx::core::contiguous(std::move(expanded));
    }
}

MlxMxWeight MlxMxWeight::from_blob(
    std::string_view dtype,
    const std::vector<std::uint8_t>& blob) {
    return from_blob(
        dtype,
        std::span<const std::uint8_t>(blob));
}

MlxMxWeight MlxMxWeight::from_blob(
    std::string_view dtype,
    std::span<const std::uint8_t> blob) {
    if (!is_mx_dtype(dtype)) {
        throw std::runtime_error("unsupported MX MFQ dtype: " + std::string(dtype));
    }
    Cursor cursor(blob);
    for (const auto expected : kMagic) {
        if (cursor.scalar<std::uint8_t>("magic") != expected) {
            throw std::runtime_error("invalid MX MFQ magic");
        }
    }
    const auto version = cursor.scalar<std::uint8_t>("version");
    const auto kind = cursor.scalar<std::uint8_t>("kind");
    const auto reserved = cursor.scalar<std::uint16_t>("reserved");
    const auto rows = cursor.scalar<std::uint64_t>("logical rows");
    const auto columns = cursor.scalar<std::uint64_t>("logical columns");
    const auto storage_rows = cursor.scalar<std::uint64_t>("storage rows");
    const auto storage_columns = cursor.scalar<std::uint64_t>("storage columns");
    const auto scale_rows = cursor.scalar<std::uint64_t>("scale rows");
    const auto scale_columns = cursor.scalar<std::uint64_t>("scale columns");
    const int bits = dtype == "MXFP4" ? 4 : 8;
    if (version != kVersion || kind != bits || reserved != 0) {
        throw std::runtime_error("invalid MX MFQ header version/kind");
    }
    const auto output_size = checked_dimension(rows, "output size");
    const auto input_size = checked_dimension(columns, "input size");
    const auto expected_storage_columns = bits == 4 ? columns / 2 : columns;
    const auto expected_scale_rows = bits == 4 ? rows : (rows + 127) / 128;
    const auto expected_scale_columns = bits == 4 ? columns / 32 : columns / 128;
    if ((bits == 4 && columns % 32 != 0) ||
        (bits == 8 && columns % 128 != 0) ||
        storage_rows != rows ||
        storage_columns != expected_storage_columns ||
        scale_rows != expected_scale_rows ||
        scale_columns != expected_scale_columns) {
        throw std::runtime_error("invalid MX MFQ block geometry");
    }
    const auto value_count = checked_product(
        storage_rows, storage_columns, "value byte count");
    const auto scale_count = checked_product(
        scale_rows, scale_columns, "scale byte count");
    auto values = cursor.bytes(value_count, "values");
    auto scales = cursor.bytes(scale_count, "scales");
    if (cursor.remaining() != 0) {
        throw std::runtime_error("trailing bytes in MX MFQ tensor");
    }
    return MlxMxWeight(
        make_array(
            values,
            Shape{
                checked_dimension(storage_rows, "storage rows"),
                checked_dimension(storage_columns, "storage columns"),
            }),
        make_array(
            scales,
            Shape{
                checked_dimension(scale_rows, "scale rows"),
                checked_dimension(scale_columns, "scale columns"),
            }),
        bits,
        input_size,
        output_size);
}

MlxMxWeight MlxMxWeight::from_arrays(
    std::string_view dtype,
    array values,
    array scales,
    int input_size,
    int output_size) {
    if (!is_mx_dtype(dtype) || input_size <= 0 || output_size <= 0) {
        throw std::invalid_argument("invalid MX array geometry");
    }
    const int bits = dtype == "MXFP4" ? 4 : 8;
    if ((bits == 4 && input_size % 32 != 0) ||
        (bits == 8 && input_size % 128 != 0) ||
        values.dtype() != mlx::core::uint8 ||
        scales.dtype() != mlx::core::uint8 ||
        !values.flags().row_contiguous ||
        !scales.flags().row_contiguous) {
        throw std::invalid_argument("invalid MX packed arrays");
    }
    const auto expected_values = checked_product(
        static_cast<std::uint64_t>(output_size),
        static_cast<std::uint64_t>(bits == 4 ? input_size / 2 : input_size),
        "array value byte count");
    const auto expected_scales = checked_product(
        static_cast<std::uint64_t>(
            bits == 4 ? output_size : (output_size + 127) / 128),
        static_cast<std::uint64_t>(
            bits == 4 ? input_size / 32 : input_size / 128),
        "array scale byte count");
    if (values.size() != expected_values || scales.size() != expected_scales) {
        throw std::invalid_argument("MX packed array size mismatch");
    }
    return MlxMxWeight(
        std::move(values),
        std::move(scales),
        bits,
        input_size,
        output_size);
}

array MlxMxWeight::dequantize(Dtype dtype) const {
    if (dtype != mlx::core::float16 && dtype != mlx::core::float32) {
        throw std::runtime_error("MX dequantization requires float16 or float32");
    }
    const auto elements = checked_product(
        static_cast<std::uint64_t>(output_size_),
        static_cast<std::uint64_t>(input_size_),
        "dequantization grid");
    if (elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MX dequantization grid exceeds MLX limits");
    }
    auto outputs = mx_dequantize_kernel()(
        {values_, scales_, values_},
        {Shape{output_size_, input_size_}},
        {dtype},
        {static_cast<int>(elements), 1, 1},
        {std::min(256, static_cast<int>(elements)), 1, 1},
        templates(dtype, bits_, input_size_, output_size_),
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array MlxMxWeight::matmul(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size_) {
        throw std::runtime_error("MX input width does not match packed weight");
    }
    const auto rows = input.size() / static_cast<std::size_t>(input_size_);
    if (rows == 0 || rows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("unsupported MX input row count");
    }
    Shape output_shape = input.shape();
    output_shape.back() = output_size_;
    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::reshape(
        source,
        Shape{static_cast<int>(rows), input_size_});
    if (rows == 1 && bits_ == 8 &&
        source.dtype() == mlx::core::float16 &&
        expanded_mxfp8_scales_.has_value()) {
        auto packed = mlx::core::reshape(
            mlx::core::view(values_, mlx::core::uint32),
            Shape{output_size_, input_size_ / 4});
        auto result = mlx::core::quantized_matmul(
            std::move(source),
            std::move(packed),
            *expanded_mxfp8_scales_,
            std::nullopt,
            true,
            32,
            8,
            "mxfp8");
        return mlx::core::reshape(std::move(result), std::move(output_shape));
    }
    if (rows >= 64) {
        auto dense = dequantize(source.dtype());
        auto result = mlx::core::matmul(source, mlx::core::transpose(dense));
        return mlx::core::reshape(std::move(result), std::move(output_shape));
    }

    const bool gemv = rows == 1;
    const bool mxfp8_gemv =
        gemv && bits_ == 8 && source.dtype() == mlx::core::float16;
    const int tile_rows = gemv ? 1 : (rows <= 16 ? static_cast<int>(rows) : 8);
    const auto row_tiles = (rows + static_cast<std::size_t>(tile_rows) - 1) /
        static_cast<std::size_t>(tile_rows);
    const auto grid = mxfp8_gemv
        ? static_cast<std::size_t>((output_size_ + 15) / 16) * 128
        : gemv
        ? static_cast<std::size_t>((output_size_ + 7) / 8) * 64
        : row_tiles * static_cast<std::size_t>(output_size_) * 32;
    if (grid > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MX Metal grid exceeds MLX limits");
    }
    const auto& kernel = mxfp8_gemv
        ? mxfp8_gemv_kernel()
        : gemv
        ? mx_gemv_kernel()
        : mx_matmul_kernel();
    auto outputs = kernel(
        {values_, scales_, source},
        {Shape{static_cast<int>(rows), output_size_}},
        {source.dtype()},
        {static_cast<int>(grid), 1, 1},
        {mxfp8_gemv ? 128 : gemv ? 64 : 32, 1, 1},
        templates(
            source.dtype(),
            bits_,
            input_size_,
            output_size_,
            static_cast<int>(rows),
            tile_rows),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(std::move(outputs.front()), std::move(output_shape));
}

array MlxMxWeight::grouped_row_matmul(
    const array& input,
    int group_count) const {
    if (bits_ != 8 ||
        group_count <= 0 ||
        input.ndim() < 3 ||
        input.shape(-2) != group_count ||
        input.shape(-1) != input_size_ ||
        output_size_ % group_count != 0 ||
        input_size_ % 128 != 0) {
        throw std::runtime_error(
            "MXFP8 grouped-row matmul shape is incompatible");
    }
    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(
            source,
            mlx::core::float16);
    }
    const int output_per_group = output_size_ / group_count;
    auto dense = dequantize(source.dtype());
    std::vector<array> pieces;
    pieces.reserve(static_cast<std::size_t>(group_count));
    for (int group = 0; group < group_count; ++group) {
        auto group_input = mlx::core::take(
            source,
            group,
            source.ndim() - 2);
        auto group_weight = mlx::core::slice(
            dense,
            Shape{group * output_per_group, 0},
            Shape{(group + 1) * output_per_group, input_size_});
        pieces.push_back(
            mlx::core::matmul(
                std::move(group_input),
                mlx::core::transpose(group_weight)));
    }
    return mlx::core::stack(
        pieces,
        input.ndim() - 2);
}

array MlxMxWeight::grouped_row_matmul_inverse_rope(
    const array& input,
    int group_count,
    const array& cosine,
    const array& sine,
    int head_dimension,
    int rotary_dimension) const {
    if (bits_ != 8 ||
        group_count <= 0 ||
        input.ndim() < 3 ||
        input.shape(-2) != group_count ||
        input.shape(-1) != input_size_ ||
        output_size_ % group_count != 0 ||
        head_dimension <= 0 ||
        rotary_dimension <= 0 ||
        rotary_dimension > head_dimension ||
        head_dimension % 4 != 0 ||
        rotary_dimension % 4 != 0 ||
        input_size_ % head_dimension != 0 ||
        cosine.shape() != sine.shape() ||
        cosine.ndim() != 2 ||
        cosine.shape(0) != 1 ||
        cosine.shape(1) != rotary_dimension / 2) {
        throw std::runtime_error(
            "MXFP8 inverse-RoPE grouped-row shape is incompatible");
    }
    std::size_t rows = 1;
    Shape output_shape(
        input.shape().begin(),
        input.shape().end() - 2);
    for (std::size_t index = 0; index + 2 < input.ndim(); ++index) {
        const int extent = input.shape(static_cast<int>(index));
        if (extent <= 0 ||
            rows > std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(extent)) {
            throw std::runtime_error(
                "unsupported MXFP8 inverse-RoPE grouped-row count");
        }
        rows *= static_cast<std::size_t>(extent);
    }
    if (rows != 1) {
        throw std::runtime_error(
            "MXFP8 fused inverse-RoPE grouped-row is decode-only");
    }

    const int out_per_group = output_size_ / group_count;
    output_shape.push_back(group_count);
    output_shape.push_back(out_per_group);
    auto source = input;
    if (source.dtype() != mlx::core::float16) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::contiguous(
        mlx::core::reshape(
            source,
            Shape{group_count, input_size_}));
    auto cos_values = cosine.dtype() == mlx::core::float32
        ? cosine
        : mlx::core::astype(cosine, mlx::core::float32);
    auto sin_values = sine.dtype() == mlx::core::float32
        ? sine
        : mlx::core::astype(sine, mlx::core::float32);
    cos_values = mlx::core::contiguous(
        mlx::core::reshape(cos_values, Shape{rotary_dimension / 2}));
    sin_values = mlx::core::contiguous(
        mlx::core::reshape(sin_values, Shape{rotary_dimension / 2}));

    const auto grid =
        static_cast<std::size_t>((output_size_ + 15) / 16) * 128;
    if (grid > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "MXFP8 inverse-RoPE grouped-row Metal grid exceeds MLX limits");
    }
    auto outputs = mxfp8_grouped_inverse_rope_kernel()(
        {values_, scales_, source, cos_values, sin_values},
        {Shape{output_size_}},
        {mlx::core::float16},
        {static_cast<int>(grid), 1, 1},
        {128, 1, 1},
        {
            {"GROUP_COUNT", group_count},
            {"OUT_PER_GROUP", out_per_group},
            {"OUT", output_size_},
            {"K", input_size_},
            {"HEAD_DIM", head_dimension},
            {"ROTARY", rotary_dimension},
        },
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        std::move(outputs.front()),
        std::move(output_shape));
}

array MlxMxWeight::embedding(const array& token_ids, Dtype dtype) const {
    if (dtype != mlx::core::float16 && dtype != mlx::core::float32) {
        throw std::runtime_error("MX embedding requires float16 or float32");
    }
    auto ids = token_ids;
    if (ids.dtype() != mlx::core::int32 && ids.dtype() != mlx::core::uint32) {
        ids = mlx::core::astype(ids, mlx::core::int32);
    }
    const auto tokens = ids.size();
    Shape output_shape = ids.shape();
    output_shape.push_back(input_size_);
    if (tokens == 0) {
        return mlx::core::zeros(output_shape, dtype);
    }
    const auto elements = checked_product(
        static_cast<std::uint64_t>(tokens),
        static_cast<std::uint64_t>(input_size_),
        "embedding grid");
    if (tokens > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MX embedding grid exceeds MLX limits");
    }
    auto outputs = mx_embedding_kernel()(
        {values_, scales_, mlx::core::reshape(
            ids, Shape{static_cast<int>(tokens)})},
        {output_shape},
        {dtype},
        {static_cast<int>(elements), 1, 1},
        {std::min(256, static_cast<int>(elements)), 1, 1},
        templates(
            dtype,
            bits_,
            input_size_,
            output_size_,
            static_cast<int>(tokens)),
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

std::size_t MlxMxWeight::packed_nbytes() const noexcept {
    return values_.nbytes() + scales_.nbytes();
}

} // namespace mfq::metal
