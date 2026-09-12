#include "mlx_deepseek_v4_attention.h"
#include "mlx_eval_timing.h"

#include "mlx_detached_copy.h"
#include "mlx_grouped_linear.h"
#include "mlx_transformer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

bool v41_fast_indexer_enabled() noexcept {
    // The tiled kernel keeps indexer scores in FP32 while avoiding the
    // [B, T, H, K, D] broadcast used by the reference expression.  Keep an
    // explicit reference-path escape hatch for numerical investigations.
    const char* value = std::getenv(
        "MFQ_METAL_DSV41_FAST_INDEXER");
    if (value == nullptr) {
        return true;
    }
    const auto setting = std::string_view(value);
    return setting != "0"
        && setting != "false"
        && setting != "off";
}

bool v41_circular_prefill_enabled() noexcept {
    // The direct long-prefill kernel consumes chronological local rows plus
    // the capacity-backed CSA pool. It avoids rebuilding a unified cache and
    // a dense index/mask plan on every V4.1 layer. Keep a parity escape hatch.
    const char* value = std::getenv(
        "MFQ_METAL_DSV41_CIRCULAR_PREFILL");
    if (value == nullptr) {
        return true;
    }
    const auto setting = std::string_view(value);
    return setting != "0"
        && setting != "false"
        && setting != "off";
}

bool block32_inverse_rope_qmv_enabled() noexcept {
    const char* value = std::getenv(
        "MFQ_METAL_MXFP8_BLOCK32_INVERSE_ROPE");
    if (value == nullptr) {
        return true;
    }
    const auto setting = std::string_view(value);
    return setting != "0"
        && setting != "false"
        && setting != "off";
}

bool apple_m3_ultra() noexcept {
#if defined(__APPLE__)
    static const bool is_m3_ultra = [] {
        char name[64]{};
        std::size_t size = sizeof(name);
        return ::sysctlbyname(
                "machdep.cpu.brand_string",
                name,
                &size,
                nullptr,
                0) == 0 &&
            std::string_view(name).rfind("Apple M3 Ultra", 0) == 0;
    }();
    return is_m3_ultra;
#else
    return false;
#endif
}

bool v41_fused_kv_prepare_enabled() noexcept {
    if (const char* value = std::getenv(
            "MFQ_METAL_DSV41_FUSED_KV_PREP")) {
        const auto setting = std::string_view(value);
        return setting != "0"
            && setting != "false"
            && setting != "off";
    }
    return apple_m3_ultra();
}

constexpr const char* kHadamardSource = R"METAL(
    uint row = thread_position_in_grid.x / 256u;
    uint lane = thread_index_in_threadgroup;
    if (row >= uint(M)) {
        return;
    }
    threadgroup float values[BLOCK];
    for (
        uint local_block = 0u;
        local_block < uint(K) / uint(BLOCK);
        ++local_block
    ) {
        uint column_base = local_block * uint(BLOCK);
        for (
            uint index = lane;
            index < uint(BLOCK);
            index += 256u
        ) {
            values[index] = float(
                x[row * uint(K) + column_base + index]
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (
            uint stride = 1u;
            stride < uint(BLOCK);
            stride <<= 1u
        ) {
            for (
                uint pair = lane;
                pair < uint(BLOCK) / 2u;
                pair += 256u
            ) {
                uint pair_block = pair / stride;
                uint within = pair - pair_block * stride;
                uint first =
                    pair_block * (stride << 1u) + within;
                uint second = first + stride;
                float first_value = values[first];
                float second_value = values[second];
                values[first] = first_value + second_value;
                values[second] = first_value - second_value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inverse = rsqrt(float(BLOCK));
        for (
            uint index = lane;
            index < uint(BLOCK);
            index += 256u
        ) {
            y[
                row * uint(K) + column_base + index
            ] = T(values[index] * inverse);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
)METAL";

constexpr const char* kPartialAdjacentRopeSource = R"METAL(
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }
    uint column = index % uint(DIM);
    uint row = index / uint(DIM);
    constexpr uint PREFIX = uint(DIM - ROTARY);
    if (column < PREFIX) {
        y[index] = x[index];
        return;
    }

    uint rotary_column = column - PREFIX;
    uint pair = rotary_column >> 1u;
    uint token = (row / uint(HEADS)) % uint(TOKENS);
    float cosine = float(cos_values[token * uint(PAIRS) + pair]);
    float sine = float(sin_values[token * uint(PAIRS) + pair]);
    if (INVERSE != 0) {
        sine = -sine;
    }
    uint pair_base =
        row * uint(DIM) + PREFIX + (pair << 1u);
    float first = float(x[pair_base]);
    float second = float(x[pair_base + 1u]);
    float result = (rotary_column & 1u) == 0u
        ? first * cosine - second * sine
        : first * sine + second * cosine;
    y[index] = T(result);
)METAL";

constexpr const char* kRmsPartialAdjacentRopeSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint local_thread = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }

    threadgroup float reductions[8];
    float sum_squares = 0.0f;
    uint row_base = row * uint(DIM);
    for (uint column = local_thread;
         column < uint(DIM);
         column += 256u) {
        float value = float(x[row_base + column]);
        sum_squares += value * value;
    }
    float subtotal = simd_sum(sum_squares);
    if (lane == 0u) {
        reductions[simd_group] = subtotal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_thread == 0u) {
        float total = 0.0f;
        for (uint group = 0u; group < 8u; ++group) {
            total += reductions[group];
        }
        reductions[0] = rsqrt(total / float(DIM) + params[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse_rms = reductions[0];

    constexpr uint PREFIX = uint(DIM - ROTARY);
    uint token = (row / uint(HEADS)) % uint(TOKENS);
    for (uint column = local_thread;
         column < uint(DIM);
         column += 256u) {
        // Preserve the original FP16/BF16 boundary between RMSNorm and RoPE.
        float weight = WEIGHTED != 0
            ? float(weights[column])
            : 1.0f;
        T normalized = T(
            float(x[row_base + column]) * inverse_rms * weight);
        if (column < PREFIX) {
            y[row_base + column] = normalized;
            continue;
        }
        uint rotary_column = column - PREFIX;
        uint pair = rotary_column >> 1u;
        uint pair_base = row_base + PREFIX + (pair << 1u);
        float first_weight = WEIGHTED != 0
            ? float(weights[PREFIX + (pair << 1u)])
            : 1.0f;
        float second_weight = WEIGHTED != 0
            ? float(weights[PREFIX + (pair << 1u) + 1u])
            : 1.0f;
        float first = float(T(
            float(x[pair_base]) * inverse_rms * first_weight));
        float second = float(T(
            float(x[pair_base + 1u]) * inverse_rms * second_weight));
        float cosine =
            float(cos_values[token * uint(PAIRS) + pair]);
        float sine =
            float(sin_values[token * uint(PAIRS) + pair]);
        float result = (rotary_column & 1u) == 0u
            ? first * cosine - second * sine
            : first * sine + second * cosine;
        y[row_base + column] = T(result);
    }
)METAL";

constexpr const char* kKvFp8SimSource = R"METAL(
    uint group = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_threadgroup;
    constexpr uint GROUP_SIZE = 64u;
    constexpr uint PREFIX_GROUPS = uint(PREFIX / GROUP_SIZE);
    uint row = group / PREFIX_GROUPS;
    uint block = group - row * PREFIX_GROUPS;
    uint column = block * GROUP_SIZE + lane;
    uint index = row * uint(DIM) + column;

    threadgroup float maxima[GROUP_SIZE];
    float value = float(x[index]);
    maxima[lane] = abs(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GROUP_SIZE / 2u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            maxima[lane] = max(maxima[lane], maxima[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float amax = max(maxima[0], 1.0e-4f);
    float raw_scale = amax / 448.0f;
    uint bits = as_type<uint>(raw_scale);
    uint exponent = (bits >> 23u) & 0xffu;
    bool has_mantissa = (bits & 0x7fffffu) != 0u;
    float scale = as_type<float>(
        (exponent + uint(has_mantissa)) << 23u);
    float normalized = clamp(value / scale, -448.0f, 448.0f);
    float magnitude = abs(normalized);
    float quantized;
    if (magnitude < 0x1p-6f) {
        quantized = rint(magnitude * 512.0f) / 512.0f;
    } else {
        float quant_exponent = floor(log2(magnitude));
        float step = exp2(quant_exponent - 3.0f);
        quantized = min(rint(magnitude / step) * step, 448.0f);
    }
    y[row * uint(PREFIX) + column] =
        T(copysign(quantized * scale, normalized));
)METAL";

// V4.1 stores dequantized cache values, but the values must cross the same
// fake-quantization boundary as the released checkpoint.  MODE selects:
//   0: E4M3 value / E8M0 scale (window KV, groups of 32)
//   1: E2M1 value / E8M0 scale (Indexer q/k, groups of 32)
//   2: E2M1 value / E4M3FN scale (compressed KV, groups of 16)
constexpr const char* kV41ActivationQuantHeader = R"METAL(
    METAL_FUNC float fp8_e4m3(float value) {
        float magnitude = abs(value);
        float quantized;
        if (magnitude < 0x1p-6f) {
            quantized = rint(magnitude * 512.0f) / 512.0f;
        } else {
            float exponent = floor(log2(magnitude));
            float step = exp2(exponent - 3.0f);
            quantized = min(rint(magnitude / step) * step, 448.0f);
        }
        return copysign(quantized, value);
    }

    METAL_FUNC float fp4_e2m1(float value) {
        float magnitude = min(abs(value), 6.0f);
        float quantized;
        if (magnitude <= 0.25f) {
            quantized = 0.0f;
        } else if (magnitude < 0.75f) {
            quantized = 0.5f;
        } else if (magnitude <= 1.25f) {
            quantized = 1.0f;
        } else if (magnitude < 1.75f) {
            quantized = 1.5f;
        } else if (magnitude <= 2.5f) {
            quantized = 2.0f;
        } else if (magnitude < 3.5f) {
            quantized = 3.0f;
        } else if (magnitude <= 5.0f) {
            quantized = 4.0f;
        } else {
            quantized = 6.0f;
        }
        return copysign(quantized, value);
    }

    METAL_FUNC float pow2_ceil(float value) {
        uint bits = as_type<uint>(value);
        uint exponent = (bits >> 23u) & 0xffu;
        bool has_mantissa = (bits & 0x7fffffu) != 0u;
        return as_type<float>((exponent + uint(has_mantissa)) << 23u);
    }
)METAL";

constexpr const char* kV41ActivationQuantSource = R"METAL(
    uint group = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_threadgroup;
    uint index = group * uint(GROUP) + lane;
    threadgroup float maxima[32];
    maxima[lane] = abs(float(x[index]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = uint(GROUP) / 2u;
         stride != 0u;
         stride >>= 1u) {
        if (lane < stride) {
            maxima[lane] = max(maxima[lane], maxima[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float maximum = maxima[0];
    float scale;
    if (MODE == 0) {
        scale = pow2_ceil(max(maximum, 1.0e-4f) / 448.0f);
    } else if (MODE == 1) {
        scale = pow2_ceil(max(maximum, 6.0f * 0x1p-126f) / 6.0f);
    } else {
        // Casting the scale to E4M3FN is itself part of the checkpoint graph.
        scale = fp8_e4m3(max(maximum, 6.0f * 0x1p-9f) / 6.0f);
    }
    float normalized = float(x[index]) / scale;
    float quantized = MODE == 0
        ? fp8_e4m3(clamp(normalized, -448.0f, 448.0f))
        : fp4_e2m1(normalized);
    y[index] = T(quantized * scale);
)METAL";

// Exact DSV4.1 window-KV geometry: one 256-thread group owns a D=512 row.
// Each SIMD group quantizes one 32-value block in each half of the row after
// sharing the row-wide RMS reduction. This collapses RMSNorm, RoPE, and the
// E4M3/E8M0 fake-quant boundary into a single read/write pass.
constexpr const char* kV41WeightedRmsRopeQuantSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint local_thread = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }

    constexpr uint HALF = 256u;
    constexpr uint PREFIX = uint(DIM - ROTARY);
    uint row_base = row * uint(DIM);
    uint first_column = local_thread;
    uint second_column = local_thread + HALF;
    float first_input = float(x[row_base + first_column]);
    float second_input = float(x[row_base + second_column]);

    threadgroup float reductions[8];
    float subtotal = simd_sum(
        first_input * first_input + second_input * second_input);
    if (lane == 0u) {
        reductions[simd_group] = subtotal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_thread == 0u) {
        float total = 0.0f;
        for (uint group = 0u; group < 8u; ++group) {
            total += reductions[group];
        }
        reductions[0] = rsqrt(total / float(DIM) + params[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse_rms = reductions[0];

    T first_value = T(
        first_input * inverse_rms * float(weights[first_column]));
    T second_value = T(
        second_input * inverse_rms * float(weights[second_column]));
    if (second_column >= PREFIX) {
        uint rotary_column = second_column - PREFIX;
        uint pair = rotary_column >> 1u;
        uint pair_column = PREFIX + (pair << 1u);
        float pair_first = float(T(
            float(x[row_base + pair_column]) *
            inverse_rms * float(weights[pair_column])));
        float pair_second = float(T(
            float(x[row_base + pair_column + 1u]) *
            inverse_rms * float(weights[pair_column + 1u])));
        uint token = row % uint(TOKENS);
        float cosine =
            float(cos_values[token * uint(PAIRS) + pair]);
        float sine =
            float(sin_values[token * uint(PAIRS) + pair]);
        float rotated = (rotary_column & 1u) == 0u
            ? pair_first * cosine - pair_second * sine
            : pair_first * sine + pair_second * cosine;
        second_value = T(rotated);
    }

    float first_float = float(first_value);
    float first_maximum = simd_max(abs(first_float));
    float first_scale = pow2_ceil(
        max(first_maximum, 1.0e-4f) / 448.0f);
    float first_normalized = first_float / first_scale;
    y[row_base + first_column] = T(
        fp8_e4m3(clamp(first_normalized, -448.0f, 448.0f)) *
        first_scale);

    float second_float = float(second_value);
    float second_maximum = simd_max(abs(second_float));
    float second_scale = pow2_ceil(
        max(second_maximum, 1.0e-4f) / 448.0f);
    float second_normalized = second_float / second_scale;
    y[row_base + second_column] = T(
        fp8_e4m3(clamp(second_normalized, -448.0f, 448.0f)) *
        second_scale);
)METAL";

const mlx::core::fast::CustomKernelFunction&
hadamard_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_dsv4_index_hadamard",
            {"x"},
            {"y"},
            kHadamardSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
partial_adjacent_rope_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_dsv4_partial_adjacent_rope",
            {"x", "cos_values", "sin_values"},
            {"y"},
            kPartialAdjacentRopeSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
rms_partial_adjacent_rope_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_dsv4_rms_partial_adjacent_rope",
            {"x", "cos_values", "sin_values", "weights", "params"},
            {"y"},
            kRmsPartialAdjacentRopeSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
kv_fp8_sim_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_dsv4_kv_fp8_sim",
            {"x"},
            {"y"},
            kKvFp8SimSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
v41_activation_quant_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_dsv41_activation_quant",
            {"x"},
            {"y"},
            kV41ActivationQuantSource,
            kV41ActivationQuantHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
v41_weighted_rms_rope_quant_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_dsv41_weighted_rms_rope_quant",
            {"x", "cos_values", "sin_values", "weights", "params"},
            {"y"},
            kV41WeightedRmsRopeQuantSource,
            kV41ActivationQuantHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

int checked_int(
    std::int64_t value,
    const char* label) {
    if (value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            std::string("invalid DeepSeek-V4 ") + label);
    }
    return static_cast<int>(value);
}

int checked_product(
    std::initializer_list<int> factors,
    const char* label) {
    std::int64_t result = 1;
    for (const int factor : factors) {
        if (factor < 0 ||
            (factor != 0 &&
             result >
                 std::numeric_limits<int>::max() /
                     factor)) {
            throw std::invalid_argument(
                std::string("DeepSeek-V4 ") + label +
                " exceeds MLX limits");
        }
        result *= factor;
    }
    return static_cast<int>(result);
}

array typed_contiguous(
    const array& input,
    Dtype dtype) {
    auto result = input;
    if (result.dtype() != dtype) {
        result = mlx::core::astype(result, dtype);
    }
    return mlx::core::contiguous(result);
}

array floating_contiguous(const array& input) {
    auto result = input;
    if (result.dtype() != mlx::core::float16 &&
        result.dtype() != mlx::core::bfloat16 &&
        result.dtype() != mlx::core::float32) {
        result = mlx::core::astype(
            result,
            mlx::core::float16);
    }
    return mlx::core::contiguous(result);
}

array slice_axis(
    const array& input,
    int axis,
    int begin,
    int end) {
    if (axis < 0) {
        axis += static_cast<int>(input.ndim());
    }
    if (axis < 0 ||
        axis >= static_cast<int>(input.ndim()) ||
        begin < 0 || end < begin ||
        end > input.shape(axis)) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 array slice");
    }
    Shape start(input.ndim(), 0);
    Shape stop = input.shape();
    start[axis] = begin;
    stop[axis] = end;
    return mlx::core::slice(
        input,
        std::move(start),
        std::move(stop));
}

array replace_last_rope(
    const array& input,
    int rotary,
    const array& cosine,
    const array& sine,
    bool inverse = false) {
    if (input.ndim() < 2 || input.ndim() > 4 ||
        input.shape(-1) <= 0 ||
        rotary <= 0 ||
        rotary > input.shape(-1) ||
        rotary % 2 != 0 ||
        cosine.shape() != sine.shape() ||
        cosine.shape(-1) != rotary / 2) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 partial adjacent RoPE input");
    }

    auto source = floating_contiguous(input);
    const int dimension = source.shape(-1);
    const int tokens = source.ndim() == 2
        ? source.shape(0)
        : source.shape(1);
    const int heads = source.ndim() == 4
        ? source.shape(2)
        : 1;
    const int pairs = rotary / 2;
    if (tokens <= 0 || heads <= 0 ||
        cosine.size() !=
            static_cast<std::size_t>(tokens) * pairs) {
        throw std::invalid_argument(
            "DeepSeek-V4 RoPE table is not token-broadcastable");
    }
    auto cos_values = typed_contiguous(
        cosine,
        mlx::core::float32);
    auto sin_values = typed_contiguous(
        sine,
        mlx::core::float32);
    const auto size = source.size();
    if (size >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "DeepSeek-V4 RoPE tensor exceeds Metal grid limits");
    }
    const int grid = static_cast<int>(size);
    auto outputs = partial_adjacent_rope_kernel()(
        {source, cos_values, sin_values},
        {source.shape()},
        {source.dtype()},
        {grid, 1, 1},
        {std::min(256, grid), 1, 1},
        {
            {"T", source.dtype()},
            {"SIZE", grid},
            {"DIM", dimension},
            {"ROTARY", rotary},
            {"PAIRS", pairs},
            {"TOKENS", tokens},
            {"HEADS", heads},
            {"INVERSE", inverse ? 1 : 0},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array rms_replace_last_rope(
    const array& input,
    int rotary,
    const array& cosine,
    const array& sine,
    const array& weights,
    bool weighted,
    const array& params) {
    if ((input.ndim() != 3 && input.ndim() != 4) ||
        input.shape(-1) <= 0 ||
        rotary <= 0 ||
        rotary > input.shape(-1) ||
        rotary % 2 != 0 ||
        cosine.shape() != sine.shape() ||
        cosine.shape(-1) != rotary / 2 ||
        params.dtype() != mlx::core::float32 ||
        params.size() != 1 ||
        (weighted &&
         (weights.dtype() != mlx::core::float32 ||
          weights.ndim() != 1 ||
          weights.size() !=
              static_cast<std::size_t>(input.shape(-1))))) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 fused RMS/RoPE input");
    }
    auto source = floating_contiguous(input);
    const int dimension = source.shape(-1);
    const int tokens = source.shape(1);
    const int heads = source.ndim() == 4
        ? source.shape(2)
        : 1;
    const int pairs = rotary / 2;
    if (tokens <= 0 || heads <= 0 ||
        cosine.size() !=
            static_cast<std::size_t>(tokens) * pairs) {
        throw std::invalid_argument(
            "DeepSeek-V4 fused RMS/RoPE table mismatch");
    }
    const auto rows = source.size() /
        static_cast<std::size_t>(dimension);
    if (rows == 0 ||
        rows > static_cast<std::size_t>(
            std::numeric_limits<int>::max() / 256)) {
        throw std::invalid_argument(
            "DeepSeek-V4 fused RMS/RoPE grid exceeds Metal limits");
    }
    auto outputs = rms_partial_adjacent_rope_kernel()(
        {
            source,
            typed_contiguous(cosine, mlx::core::float32),
            typed_contiguous(sine, mlx::core::float32),
            weights,
            params,
        },
        {source.shape()},
        {source.dtype()},
        {static_cast<int>(rows) * 256, 1, 1},
        {256, 1, 1},
        {
            {"T", source.dtype()},
            {"ROWS", static_cast<int>(rows)},
            {"DIM", dimension},
            {"ROTARY", rotary},
            {"PAIRS", pairs},
            {"TOKENS", tokens},
            {"HEADS", heads},
            {"WEIGHTED", weighted ? 1 : 0},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array kv_fp8_sim_prefix_impl(
    const array& input,
    int rotary) {
    auto source = floating_contiguous(input);
    // V4F's released QAT graph fixes this operation to D=512/RoPE=64.
    // Keep synthetic/smaller compatibility configurations unchanged.
    if (source.shape(-1) != 512 || rotary != 64) {
        return source;
    }
    if ((source.ndim() != 3 && source.ndim() != 4) ||
        rotary <= 0 || rotary >= source.shape(-1)) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 KV FP8 simulation input");
    }
    const int dimension = source.shape(-1);
    const int prefix = dimension - rotary;
    if (prefix % 64 != 0) {
        throw std::invalid_argument(
            "DeepSeek-V4 KV FP8 prefix must be 64-aligned");
    }
    const int rows = checked_int(
        source.size() / static_cast<std::size_t>(dimension),
        "DeepSeek-V4 KV FP8 row count");
    const int groups = checked_product(
        {rows, prefix / 64},
        "DeepSeek-V4 KV FP8 group count");
    Shape prefix_shape = source.shape();
    prefix_shape.back() = prefix;
    auto outputs = kv_fp8_sim_kernel()(
        {source},
        {prefix_shape},
        {source.dtype()},
        {groups * 64, 1, 1},
        {64, 1, 1},
        {
            {"T", source.dtype()},
            {"DIM", dimension},
            {"PREFIX", prefix},
        },
        std::nullopt,
        false,
        {});
    return mlx::core::concatenate(
        {
            std::move(outputs.front()),
            slice_axis(
                source,
                static_cast<int>(source.ndim()) - 1,
                prefix,
                dimension),
        },
        static_cast<int>(source.ndim()) - 1);
}

array v41_activation_quant(
    const array& input,
    int group_size,
    int mode) {
    auto source = floating_contiguous(input);
    if (source.ndim() == 0 || source.size() == 0 ||
        (group_size != 16 && group_size != 32) ||
        mode < 0 || mode > 2 ||
        (mode != 2 && group_size != 32) ||
        (mode == 2 && group_size != 16) ||
        source.shape(-1) % group_size != 0 ||
        source.size() % static_cast<std::size_t>(group_size) != 0) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 activation quantization input");
    }
    const auto groups = source.size() /
        static_cast<std::size_t>(group_size);
    if (groups > static_cast<std::size_t>(
            std::numeric_limits<int>::max() / group_size)) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 activation quantization grid is too large");
    }
    auto outputs = v41_activation_quant_kernel()(
        {source},
        {source.shape()},
        {source.dtype()},
        {static_cast<int>(groups) * group_size, 1, 1},
        {group_size, 1, 1},
        {
            {"T", source.dtype()},
            {"GROUP", group_size},
            {"MODE", mode},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array weighted_rms(
    const array& input,
    const array& weight,
    Dtype output_dtype,
    float eps) {
    if (input.ndim() == 0 || input.shape(-1) <= 0 ||
        weight.ndim() != 1 ||
        weight.size() != static_cast<std::size_t>(input.shape(-1)) ||
        !std::isfinite(eps) || eps <= 0.0f) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 RMSNorm input");
    }
    auto source = mlx::core::astype(input, mlx::core::float32);
    auto normalized = source * mlx::core::rsqrt(
        mlx::core::mean(source * source, -1, true) + eps);
    auto result = normalized * typed_contiguous(
        weight,
        mlx::core::float32);
    return output_dtype == mlx::core::float32
        ? result
        : mlx::core::astype(result, output_dtype);
}

array v41_weighted_rms_rope_activation_quant(
    const array& input,
    const array& weight,
    Dtype output_dtype,
    float eps,
    int rotary,
    const array& cosine,
    const array& sine,
    const array& params) {
    const auto reference = [&] {
        auto result = weighted_rms(
            input,
            weight,
            output_dtype,
            eps);
        result = replace_last_rope(
            result,
            rotary,
            cosine,
            sine);
        return v41_activation_quant(result, 32, 0);
    };
    // This schedule is intentionally specific to the released V4.1 window
    // KV shape. Synthetic configurations and other devices keep the portable
    // MLX composition unless explicitly forced through the feature switch.
    if (!v41_fused_kv_prepare_enabled() ||
        input.ndim() != 3 ||
        input.shape(-1) != 512 ||
        input.dtype() != output_dtype ||
        rotary != 64) {
        return reference();
    }
    auto source = floating_contiguous(input);
    if (weight.dtype() != mlx::core::float32 ||
        weight.ndim() != 1 || weight.size() != 512u ||
        params.dtype() != mlx::core::float32 ||
        params.size() != 1 ||
        !std::isfinite(eps) || eps <= 0.0f ||
        cosine.shape() != sine.shape() ||
        cosine.shape(-1) != 32) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 fused KV preparation input");
    }
    const int tokens = source.shape(1);
    const int rows = checked_int(
        source.size() / 512u,
        "DeepSeek-V4.1 fused KV row count");
    const int grid = checked_product(
        {rows, 256},
        "DeepSeek-V4.1 fused KV grid");
    if (tokens <= 0 ||
        cosine.size() != static_cast<std::size_t>(tokens) * 32u) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 fused KV RoPE table mismatch");
    }
    auto outputs = v41_weighted_rms_rope_quant_kernel()(
        {
            source,
            typed_contiguous(cosine, mlx::core::float32),
            typed_contiguous(sine, mlx::core::float32),
            weight,
            params,
        },
        {source.shape()},
        {source.dtype()},
        {grid, 1, 1},
        {256, 1, 1},
        {
            {"T", source.dtype()},
            {"ROWS", rows},
            {"DIM", 512},
            {"ROTARY", 64},
            {"PAIRS", 32},
            {"TOKENS", tokens},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array stable_softmax(
    const array& input,
    int axis) {
    auto source = mlx::core::astype(input, mlx::core::float32);
    auto maximum = mlx::core::max(source, axis, true);
    auto exponentials = mlx::core::exp(source - maximum);
    return exponentials /
        mlx::core::sum(exponentials, axis, true);
}

array load_float_array(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" &&
        record.dtype != "F16" &&
        record.dtype != "F32") {
        throw std::runtime_error(
            "DeepSeek-V4 attention tensor must be BF16/F16/F32: " +
            name);
    }
    const auto mapped = model.map_record(name);
    return typed_contiguous(
        load_dense_array(
            record.dtype,
            mapped.view()),
        mlx::core::float32);
}

array signed_hadamard(
    const array& input,
    int block) {
    auto source = floating_contiguous(input);
    if (source.ndim() == 0 ||
        block <= 0 ||
        (block & (block - 1)) != 0 ||
        block > 8192 ||
        source.shape(-1) % block != 0) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 Hadamard input");
    }
    Shape output_shape = source.shape();
    int rows = 1;
    for (std::size_t axis = 0;
         axis + 1 < source.ndim();
         ++axis) {
        rows = checked_product(
            {rows, source.shape(static_cast<int>(axis))},
            "Hadamard row count");
    }
    const int width = source.shape(-1);
    source = mlx::core::contiguous(
        mlx::core::reshape(
            source,
            Shape{rows, width}));
    const int grid = checked_product(
        {rows, 256},
        "Hadamard grid");
    auto outputs = hadamard_kernel()(
        {source},
        {Shape{rows, width}},
        {source.dtype()},
        {grid, 1, 1},
        {256, 1, 1},
        {
            {"T", source.dtype()},
            {"M", rows},
            {"K", width},
            {"BLOCK", block},
        },
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        std::move(outputs.front()),
        std::move(output_shape));
}

array pool_prefix(
    const MlxDeepseekV4PoolState& state) {
    return slice_axis(
        state.pool(),
        1,
        0,
        state.pool_len());
}

array gather_selected(
    const array& cache,
    const array& indices) {
    const int batch = cache.shape(0);
    const int tokens = indices.shape(1);
    const int selected = indices.shape(2);
    const int sequence = cache.shape(1);
    const int dimension = cache.shape(2);
    auto expanded_cache = mlx::core::broadcast_to(
        mlx::core::expand_dims(cache, 1),
        Shape{
            batch,
            tokens,
            sequence,
            dimension,
        });
    auto expanded_indices = mlx::core::broadcast_to(
        mlx::core::expand_dims(indices, -1),
        Shape{
            batch,
            tokens,
            selected,
            dimension,
        });
    return mlx::core::take_along_axis(
        expanded_cache,
        expanded_indices,
        2);
}

array generic_sparse_attention(
    const array& query,
    const array& cache,
    const array& indices,
    const array& mask,
    const array& sinks) {
    const int heads = query.shape(2);
    const int dimension = query.shape(3);
    auto selected = mlx::core::astype(
        gather_selected(cache, indices),
        mlx::core::float32);
    auto query_values = mlx::core::astype(
        query,
        mlx::core::float32);
    auto scores = mlx::core::sum(
        mlx::core::expand_dims(query_values, 3) *
            mlx::core::expand_dims(selected, 2),
        -1) /
        std::sqrt(static_cast<double>(dimension));
    scores = scores +
        mlx::core::expand_dims(
            mlx::core::astype(
                mask,
                mlx::core::float32),
            2);
    auto sink_values = mlx::core::reshape(
        typed_contiguous(
            sinks,
            mlx::core::float32),
        Shape{1, 1, heads});
    auto maximum = mlx::core::maximum(
        mlx::core::max(scores, -1),
        sink_values);
    auto exponentials = mlx::core::exp(
        scores -
        mlx::core::expand_dims(maximum, -1));
    auto denominator =
        mlx::core::sum(exponentials, -1) +
        mlx::core::exp(sink_values - maximum);
    auto probabilities =
        exponentials /
        mlx::core::expand_dims(denominator, -1);
    return mlx::core::sum(
        mlx::core::expand_dims(probabilities, -1) *
            mlx::core::expand_dims(selected, 2),
        3);
}

array generic_index_topk(
    const array& query,
    const array& keys,
    const array& weights,
    const array& positions,
    int ratio,
    int count) {
    const int batch = query.shape(0);
    const int tokens = query.shape(1);
    const int heads = query.shape(2);
    const int pool_len = keys.shape(1);
    auto query_values = mlx::core::astype(
        query,
        mlx::core::float32);
    auto key_values = mlx::core::astype(
        keys,
        mlx::core::float32);
    auto dots = mlx::core::sum(
        mlx::core::expand_dims(query_values, 3) *
            mlx::core::expand_dims(
                mlx::core::expand_dims(
                    key_values,
                    1),
                1),
        -1);
    auto positive = mlx::core::maximum(
        dots,
        array(0.0f, mlx::core::float32));
    auto score = mlx::core::sum(
        positive *
            mlx::core::expand_dims(
                mlx::core::astype(
                    weights,
                    mlx::core::float32),
                -1),
        2) /
        std::sqrt(
            static_cast<double>(
                query.shape(3) * heads));
    auto visible_count = mlx::core::floor_divide(
        positions + array(1, mlx::core::int32),
        array(ratio, mlx::core::int32));
    auto key_ids = mlx::core::reshape(
        mlx::core::arange(
            pool_len,
            mlx::core::int32),
        Shape{1, 1, pool_len});
    auto visible = mlx::core::less(
        key_ids,
        mlx::core::reshape(
            visible_count,
            Shape{1, tokens, 1}));
    score = mlx::core::where(
        mlx::core::broadcast_to(
            visible,
            Shape{batch, tokens, pool_len}),
        score,
        array(
            -std::numeric_limits<float>::infinity(),
            mlx::core::float32));
    auto partitioned = mlx::core::argpartition(
        score,
        pool_len - count,
        -1);
    return typed_contiguous(
        slice_axis(
            partitioned,
            -1,
            pool_len - count,
            pool_len),
        mlx::core::int32);
}

class ProjectionGroup {
public:
    explicit ProjectionGroup(
        std::vector<const MlxLinear*> projections)
        : projections_(std::move(projections)) {
        std::vector<MlxGroupedLinearWeightRef> refs;
        refs.reserve(projections_.size());
        for (const auto* projection : projections_) {
            const auto reference =
                projection->grouped_weight_ref();
            if (!reference.has_value()) {
                return;
            }
            refs.push_back(*reference);
        }
        if (refs.size() < 2) {
            return;
        }
        try {
            grouped_.emplace(std::move(refs));
        } catch (const MlxGroupedLinearUnsupported&) {
            grouped_.reset();
        }
    }

    std::vector<array> operator()(
        const array& input) const {
        const std::size_t rows =
            input.ndim() == 0 || input.shape(-1) <= 0
            ? 0
            : input.size() /
                static_cast<std::size_t>(input.shape(-1));
        if (grouped_.has_value() &&
            grouped_->supports(input) &&
            (rows > 1 ||
             grouped_->has_single_row_nint_fast_path() ||
             grouped_->has_single_row_mxfp8_fast_path())) {
            return (*grouped_)(input);
        }
        std::vector<array> result;
        result.reserve(projections_.size());
        for (const auto* projection : projections_) {
            result.push_back((*projection)(input));
        }
        return result;
    }

private:
    std::vector<const MlxLinear*> projections_;
    std::optional<MlxGroupedLinear> grouped_;
};

void require_linear(
    const MlxLinear& linear,
    int input,
    int output,
    const char* name) {
    if (linear.input_size() != input ||
        linear.output_size() != output) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4 ") + name +
            " projection shape mismatch");
    }
}

void require_vector(
    const array& value,
    int size,
    const char* name) {
    if (value.ndim() != 1 ||
        value.size() !=
            static_cast<std::size_t>(size)) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4 ") + name +
            " shape mismatch");
    }
}

void require_matrix(
    const array& value,
    int rows,
    int columns,
    const char* name) {
    if (value.shape() != Shape{rows, columns}) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4 ") + name +
            " shape mismatch");
    }
}

} // namespace

array deepseek_v4_kv_fp8_sim_prefix(
    const array& input,
    int rotary_dimension) {
    return kv_fp8_sim_prefix_impl(input, rotary_dimension);
}

array deepseek_v41_kv_fp8_sim(
    const array& input) {
    return v41_activation_quant(input, 32, 0);
}

array deepseek_v41_fused_kv_prepare(
    const array& input,
    const array& weight,
    float eps,
    int rotary_dimension,
    const array& cosine,
    const array& sine) {
    return v41_weighted_rms_rope_activation_quant(
        input,
        weight,
        input.dtype(),
        eps,
        rotary_dimension,
        cosine,
        sine,
        array({eps}, mlx::core::float32));
}

MlxDeepseekV4ImageVisibility deepseek_v4_image_visibility(
    const std::vector<std::int64_t>& token_ids,
    std::int64_t vocab_size,
    int max_image_tokens) {
    if (token_ids.empty() || vocab_size <= 0 || max_image_tokens <= 0) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 image visibility input");
    }
    std::vector<std::int32_t> left(token_ids.size(), 0);
    std::vector<std::int32_t> right(token_ids.size(), 0);
    std::optional<std::size_t> image_start;
    for (std::size_t index = 0; index < token_ids.size(); ++index) {
        const auto token = token_ids[index];
        if (token == vocab_size) {
            if (image_start.has_value()) {
                throw std::invalid_argument(
                    "DeepSeek-V4 image sentinel spans cannot nest");
            }
            image_start = index;
            continue;
        }
        if (token != vocab_size + 4) continue;
        if (!image_start.has_value()) {
            throw std::invalid_argument(
                "DeepSeek-V4 IMAGE_END has no IMAGE_START");
        }
        const auto begin = *image_start;
        for (std::size_t position = begin; position <= index; ++position) {
            left[position] = static_cast<std::int32_t>(std::min<std::size_t>(
                position - begin,
                static_cast<std::size_t>(max_image_tokens - 1)));
            right[position] = static_cast<std::int32_t>(std::min<std::size_t>(
                index - position,
                static_cast<std::size_t>(max_image_tokens)));
        }
        image_start.reset();
    }
    if (image_start.has_value()) {
        throw std::invalid_argument(
            "DeepSeek-V4 IMAGE_START has no IMAGE_END");
    }
    return {
        array(left.begin(), Shape{1, static_cast<int>(left.size())}),
        array(right.begin(), Shape{1, static_cast<int>(right.size())}),
        max_image_tokens,
    };
}

std::pair<array, array> deepseek_v4_yarn_tables(
    int dimension,
    int length,
    float theta,
    const DeepseekV4RopeScaling& scaling) {
    if (dimension <= 0 ||
        dimension % 2 != 0 ||
        length <= 0 ||
        !std::isfinite(theta) ||
        theta <= 0.0f) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 Yarn table parameters");
    }
    const int pairs = dimension / 2;
    std::vector<float> frequency(pairs);
    for (int pair = 0; pair < pairs; ++pair) {
        frequency[pair] =
            1.0f /
            std::pow(
                theta,
                static_cast<float>(2 * pair) /
                    static_cast<float>(dimension));
    }
    if (scaling.enabled &&
        scaling.original_max_position_embeddings > 0) {
        if (!std::isfinite(scaling.factor) ||
            scaling.factor <= 0.0 ||
            !std::isfinite(scaling.beta_fast) ||
            scaling.beta_fast <= 0.0 ||
            !std::isfinite(scaling.beta_slow) ||
            scaling.beta_slow <= 0.0) {
            throw std::invalid_argument(
                "invalid DeepSeek-V4 Yarn scaling");
        }
        const auto correction =
            [&](double rotations) {
                return static_cast<double>(dimension) *
                    std::log(
                        static_cast<double>(
                            scaling
                                .original_max_position_embeddings) /
                        (rotations * 2.0 * std::acos(-1.0))) /
                    (2.0 *
                     std::log(
                         static_cast<double>(theta)));
            };
        const int low = std::max(
            static_cast<int>(
                std::floor(
                    correction(scaling.beta_fast))),
            0);
        double high = std::min(
            std::ceil(
                correction(scaling.beta_slow)),
            static_cast<double>(dimension - 1));
        if (static_cast<double>(low) == high) {
            high += 0.001;
        }
        for (int pair = 0; pair < pairs; ++pair) {
            const float ramp = std::clamp(
                static_cast<float>(
                    (static_cast<double>(pair) - low) /
                    (high - low)),
                0.0f,
                1.0f);
            const float smooth = 1.0f - ramp;
            frequency[pair] =
                frequency[pair] /
                    static_cast<float>(scaling.factor) *
                    (1.0f - smooth) +
                frequency[pair] * smooth;
        }
    }
    std::vector<float> cosine(
        static_cast<std::size_t>(length) * pairs);
    std::vector<float> sine(cosine.size());
    for (int position = 0;
         position < length;
         ++position) {
        for (int pair = 0; pair < pairs; ++pair) {
            const float angle =
                static_cast<float>(position) *
                frequency[pair];
            cosine[position * pairs + pair] =
                std::cos(angle);
            sine[position * pairs + pair] =
                std::sin(angle);
        }
    }
    return {
        array(
            cosine.begin(),
            Shape{length, pairs}),
        array(
            sine.begin(),
            Shape{length, pairs}),
    };
}

array deepseek_v4_rope_adjacent(
    const array& value,
    const array& cosine,
    const array& sine,
    bool inverse) {
    if (value.ndim() == 0 ||
        value.shape(-1) <= 0 ||
        value.shape(-1) % 2 != 0 ||
        cosine.shape() != sine.shape() ||
        cosine.size() == 0 ||
        cosine.shape(-1) != value.shape(-1) / 2) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 adjacent RoPE input");
    }
    return replace_last_rope(
        value,
        value.shape(-1),
        cosine,
        sine,
        inverse);
}

array deepseek_v4_unweighted_rms(
    const array& value,
    float eps) {
    if (value.ndim() == 0 ||
        value.shape(-1) <= 0 ||
        !std::isfinite(eps) ||
        eps <= 0.0f) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 unweighted RMS input");
    }
    auto source = mlx::core::astype(
        value,
        mlx::core::float32);
    auto inverse = mlx::core::rsqrt(
        mlx::core::mean(
            source * source,
            -1,
            true) +
        eps);
    auto result = source * inverse;
    return result.dtype() == value.dtype()
        ? result
        : mlx::core::astype(result, value.dtype());
}

MlxDeepseekV4PoolState::MlxDeepseekV4PoolState(
    int ratio,
    int head_dim,
    bool overlap,
    int batch,
    int capacity,
    Dtype dtype,
    array pool,
    array state_kv,
    array state_gate,
    std::optional<array> prev_kv,
    std::optional<array> prev_gate)
    : ratio_(ratio),
      head_dim_(head_dim),
      overlap_(overlap),
      batch_(batch),
      capacity_(capacity),
      dtype_(dtype),
      pool_(std::move(pool)),
      state_kv_(std::move(state_kv)),
      state_gate_(std::move(state_gate)),
      prev_kv_(std::move(prev_kv)),
      prev_gate_(std::move(prev_gate)) {}

MlxDeepseekV4PoolState
MlxDeepseekV4PoolState::snapshot() const {
    const auto copy_optional = [](
        const std::optional<array>& value)
        -> std::optional<array> {
        return value
            ? std::optional<array>(detached_copy(*value))
            : std::nullopt;
    };
    MlxDeepseekV4PoolState result(
        ratio_,
        head_dim_,
        overlap_,
        batch_,
        capacity_,
        dtype_,
        // A rollback/session snapshot needs only the compact live prefix.
        // Retaining the fixed-capacity pool here would pin one context-sized
        // allocation per historical session snapshot after reset_cache().
        mlx::core::array(0.0f),
        detached_copy(state_kv_),
        detached_copy(state_gate_),
        copy_optional(prev_kv_),
        copy_optional(prev_gate_));
    result.pool_len_ = pool_len_;
    result.remainder_ = remainder_;
    if (pool_len_ > 0) {
        const auto& visible_pool = pool_prefix_backup_
            ? *pool_prefix_backup_
            : pool_prefix(*this);
        result.pool_prefix_backup_ = detached_copy(visible_pool);
    }
    return result;
}

void MlxDeepseekV4PoolState::restore_snapshot(
    MlxDeepseekV4PoolState snapshot) {
    if (ratio_ != snapshot.ratio_ ||
        head_dim_ != snapshot.head_dim_ ||
        overlap_ != snapshot.overlap_ ||
        batch_ != snapshot.batch_ ||
        capacity_ != snapshot.capacity_ ||
        dtype_ != snapshot.dtype_) {
        throw std::invalid_argument(
            "DeepSeek-V4 pool snapshot topology mismatch");
    }
    state_kv_ = std::move(snapshot.state_kv_);
    state_gate_ = std::move(snapshot.state_gate_);
    prev_kv_ = std::move(snapshot.prev_kv_);
    prev_gate_ = std::move(snapshot.prev_gate_);
    pool_len_ = snapshot.pool_len_;
    remainder_ = snapshot.remainder_;
    if (pool_len_ <= 0) {
        return;
    }
    if (!snapshot.pool_prefix_backup_ ||
        snapshot.pool_prefix_backup_->shape() != Shape{
            batch_, pool_len_, head_dim_}) {
        throw std::runtime_error(
            "DeepSeek-V4 pool snapshot prefix is unavailable");
    }
    auto rows = mlx::core::broadcast_to(
        mlx::core::reshape(
            mlx::core::arange(
                pool_len_,
                mlx::core::int32),
            Shape{1, pool_len_}),
        Shape{batch_, pool_len_});
    pool_ = dsv4_cache_write_inplace(
        pool_,
        *snapshot.pool_prefix_backup_,
        rows);
}

MlxDeepseekV4PoolState
MlxDeepseekV4PoolState::speculative_snapshot() const {
    MlxDeepseekV4PoolState result(
        ratio_,
        head_dim_,
        overlap_,
        batch_,
        capacity_,
        dtype_,
        pool_,
        state_kv_,
        state_gate_,
        prev_kv_,
        prev_gate_);
    result.pool_len_ = pool_len_;
    result.remainder_ = remainder_;
    return result;
}

void MlxDeepseekV4PoolState::restore_speculative_snapshot(
    MlxDeepseekV4PoolState snapshot) {
    if (ratio_ != snapshot.ratio_ ||
        head_dim_ != snapshot.head_dim_ ||
        overlap_ != snapshot.overlap_ ||
        batch_ != snapshot.batch_ ||
        capacity_ != snapshot.capacity_ ||
        dtype_ != snapshot.dtype_ ||
        snapshot.pool_prefix_backup_) {
        throw std::invalid_argument(
            "DeepSeek-V4 speculative pool snapshot mismatch");
    }
    // dsv4_decode_pool_step produces new state arrays. Pool writes are
    // append-only, so the old handle still contains every live prefix row;
    // restoring the old logical length hides any rejected appended rows.
    pool_ = std::move(snapshot.pool_);
    state_kv_ = std::move(snapshot.state_kv_);
    state_gate_ = std::move(snapshot.state_gate_);
    prev_kv_ = std::move(snapshot.prev_kv_);
    prev_gate_ = std::move(snapshot.prev_gate_);
    pool_prefix_backup_.reset();
    pool_len_ = snapshot.pool_len_;
    remainder_ = snapshot.remainder_;
}

void MlxDeepseekV4PoolState::reset_v41() {
    if (overlap_ || (ratio_ != 1 && ratio_ != 2)) {
        throw std::logic_error(
            "logical pool reset requires a DeepSeek-V4.1 cache");
    }
    // Old rows and partial state are unreachable once both logical lengths
    // are zero. The next ratio-1 append overwrites row zero; ratio-2
    // compress_v41 constructs fresh partial storage without reading the old
    // arrays when remainder is zero.
    pool_prefix_backup_.reset();
    pool_len_ = 0;
    remainder_ = 0;
}

MlxDeepseekV4PoolState
MlxDeepseekV4PoolState::allocate(
    int ratio,
    int head_dim,
    bool overlap,
    int batch,
    int max_context,
    Dtype dtype) {
    if (ratio <= 0 ||
        ratio > 128 ||
        (head_dim != 128 && head_dim != 512) ||
        batch <= 0 ||
        max_context <= 0 ||
        (dtype != mlx::core::float16 &&
         dtype != mlx::core::bfloat16 &&
         dtype != mlx::core::float32)) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 pool allocation");
    }
    const int capacity =
        std::max(1, (max_context + ratio - 1) / ratio);
    const int output_dim =
        head_dim * (overlap ? 2 : 1);
    const Shape previous_shape{
        batch,
        ratio,
        head_dim,
    };
    std::optional<array> previous_kv;
    std::optional<array> previous_gate;
    if (overlap) {
        previous_kv = mlx::core::zeros(
            previous_shape,
            dtype);
        previous_gate = mlx::core::full(
            previous_shape,
            -std::numeric_limits<float>::infinity(),
            dtype);
    }
    return MlxDeepseekV4PoolState(
        ratio,
        head_dim,
        overlap,
        batch,
        capacity,
        dtype,
        mlx::core::zeros(
            Shape{batch, capacity, head_dim},
            dtype),
        mlx::core::zeros(
            Shape{batch, ratio, output_dim},
            dtype),
        mlx::core::full(
            Shape{batch, ratio, output_dim},
            -std::numeric_limits<float>::infinity(),
            dtype),
        std::move(previous_kv),
        std::move(previous_gate));
}

MlxDeepseekV4PoolState
MlxDeepseekV4PoolState::allocate_v41(
    int ratio,
    int head_dim,
    int batch,
    int max_context,
    Dtype cache_dtype) {
    if ((ratio != 1 && ratio != 2) ||
        head_dim <= 0 || head_dim % 32 != 0 ||
        batch <= 0 || max_context <= 0 ||
        (cache_dtype != mlx::core::float16 &&
         cache_dtype != mlx::core::bfloat16 &&
         cache_dtype != mlx::core::float32)) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 pool allocation");
    }
    const int capacity = std::max(
        1,
        (max_context + ratio - 1) / ratio);
    const Shape state_shape{batch, ratio, head_dim};
    return MlxDeepseekV4PoolState(
        ratio,
        head_dim,
        false,
        batch,
        capacity,
        cache_dtype,
        mlx::core::zeros(
            Shape{batch, capacity, head_dim},
            cache_dtype),
        mlx::core::zeros(
            state_shape,
            mlx::core::float32),
        mlx::core::full(
            state_shape,
            -std::numeric_limits<float>::infinity(),
            mlx::core::float32),
        std::nullopt,
        std::nullopt);
}

array MlxDeepseekV4PoolState::compress_v41(
    const array& kv,
    const std::optional<array>& gate,
    const array& norm,
    int start_position,
    Dtype output_dtype,
    float eps) {
    const int tokens = kv.ndim() == 3 ? kv.shape(1) : 0;
    const int expected = pool_len_ * ratio_ + remainder_;
    if ((ratio_ != 1 && ratio_ != 2) || overlap_ ||
        kv.ndim() != 3 || kv.shape(0) != batch_ ||
        kv.shape(2) != head_dim_ || tokens <= 0 ||
        start_position != expected ||
        start_position > capacity_ * ratio_ - tokens ||
        norm.ndim() != 1 ||
        norm.size() != static_cast<std::size_t>(head_dim_) ||
        (ratio_ == 1 && gate.has_value()) ||
        (ratio_ == 2 &&
         (!gate.has_value() || gate->shape() != kv.shape()))) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 compressor input");
    }

    // Ratio one deliberately stays on the checkpoint's BF16 path: there is
    // no learned gate and no accumulation across tokens.
    if (ratio_ == 1) {
        remainder_ = 0;
        return weighted_rms(
            kv,
            norm,
            output_dtype,
            eps);
    }

    auto values = mlx::core::astype(kv, mlx::core::float32);
    auto scores = mlx::core::astype(*gate, mlx::core::float32);
    if (remainder_ > 0) {
        values = mlx::core::concatenate(
            {
                slice_axis(state_kv_, 1, 0, remainder_),
                values,
            },
            1);
        scores = mlx::core::concatenate(
            {
                slice_axis(state_gate_, 1, 0, remainder_),
                scores,
            },
            1);
    }
    const int available = values.shape(1);
    const int windows = available / ratio_;
    const int cutoff = windows * ratio_;
    const int tail = available - cutoff;
    if (pool_len_ > capacity_ - windows) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 compressor exceeds cache capacity");
    }

    array compressed = mlx::core::zeros(
        Shape{batch_, 0, head_dim_},
        output_dtype);
    if (windows > 0) {
        auto grouped_values = mlx::core::reshape(
            slice_axis(values, 1, 0, cutoff),
            Shape{batch_, windows, ratio_, head_dim_});
        auto grouped_scores = mlx::core::reshape(
            slice_axis(scores, 1, 0, cutoff),
            Shape{batch_, windows, ratio_, head_dim_});
        auto pooled = mlx::core::sum(
            grouped_values * stable_softmax(grouped_scores, 2),
            2);
        compressed = weighted_rms(
            mlx::core::astype(pooled, output_dtype),
            norm,
            output_dtype,
            eps);
    }

    auto zero_kv = mlx::core::zeros(
        Shape{batch_, ratio_ - tail, head_dim_},
        mlx::core::float32);
    auto empty_gate = mlx::core::full(
        Shape{batch_, ratio_ - tail, head_dim_},
        -std::numeric_limits<float>::infinity(),
        mlx::core::float32);
    state_kv_ = tail == 0
        ? std::move(zero_kv)
        : mlx::core::concatenate(
              {
                  slice_axis(values, 1, cutoff, available),
                  zero_kv,
              },
              1);
    state_gate_ = tail == 0
        ? std::move(empty_gate)
        : mlx::core::concatenate(
              {
                  slice_axis(scores, 1, cutoff, available),
                  empty_gate,
              },
              1);
    remainder_ = tail;
    return compressed;
}

void MlxDeepseekV4PoolState::append_v41(
    const array& values) {
    const int rows = values.ndim() == 3 ? values.shape(1) : -1;
    if ((ratio_ != 1 && ratio_ != 2) || overlap_ ||
        values.ndim() != 3 || values.shape(0) != batch_ ||
        values.shape(2) != head_dim_ || rows < 0 ||
        pool_len_ > capacity_ - rows) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 cache append");
    }
    if (rows == 0) {
        return;
    }
    auto positions = mlx::core::broadcast_to(
        mlx::core::reshape(
            mlx::core::arange(
                pool_len_,
                pool_len_ + rows,
                1,
                mlx::core::int32),
            Shape{1, rows}),
        Shape{batch_, rows});
    pool_ = dsv4_cache_write_inplace(
        pool_,
        typed_contiguous(values, dtype_),
        positions);
    pool_len_ += rows;
}

void MlxDeepseekV4PoolState::update(
    const array& kv_token,
    const array& gate_token,
    const array& ape,
    const array& norm,
    int length,
    const array& cosine,
    const array& sine,
    int quant_mode,
    float eps) {
    const int expected =
        pool_len_ * ratio_ + remainder_ + 1;
    if (length != expected ||
        length <= 0 ||
        length > capacity_ * ratio_) {
        throw std::invalid_argument(
            "DeepSeek-V4 compressor update is not contiguous "
            "or exceeds capacity");
    }
    const int required_rows =
        (length + ratio_ - 1) / ratio_;
    if (cosine.ndim() != 2 ||
        sine.shape() != cosine.shape() ||
        cosine.shape(0) < required_rows ||
        cosine.shape(1) < 32) {
        throw std::invalid_argument(
            "DeepSeek-V4 compressor rotary table is too short");
    }
    auto step = dsv4_decode_pool_step(
        kv_token,
        gate_token,
        ape,
        norm,
        state_kv_,
        state_gate_,
        prev_kv_,
        prev_gate_,
        mlx::core::full(
            Shape{batch_},
            length,
            mlx::core::int32),
        cosine,
        sine,
        ratio_,
        overlap_,
        quant_mode,
        eps);
    state_kv_ = std::move(step.state_kv);
    state_gate_ = std::move(step.state_gate);
    prev_kv_ = std::move(step.prev_kv);
    prev_gate_ = std::move(step.prev_gate);
    remainder_ = length % ratio_;
    if (remainder_ != 0) {
        return;
    }
    const int row = length / ratio_ - 1;
    auto row_indices = mlx::core::full(
        Shape{batch_, 1},
        row,
        mlx::core::int32);
    pool_ = dsv4_cache_write_inplace(
        pool_,
        step.emitted,
        row_indices);
    pool_len_ = std::max(pool_len_, row + 1);
}

void MlxDeepseekV4PoolState::prefill(
    const array& kv,
    const array& gate,
    const array& ape,
    const array& norm,
    int start_position,
    const array& cosine,
    const array& sine,
    int quant_mode,
    float eps) {
    auto values = typed_contiguous(kv, dtype_);
    auto gates = typed_contiguous(gate, dtype_);
    const int output_dim =
        head_dim_ * (overlap_ ? 2 : 1);
    const int tokens = values.ndim() == 3
        ? values.shape(1)
        : 0;
    const int expected =
        pool_len_ * ratio_ + remainder_;
    if (values.ndim() != 3 ||
        values.shape(0) != batch_ ||
        values.shape(2) != output_dim ||
        gates.shape() != values.shape() ||
        tokens <= 0 ||
        remainder_ != 0 ||
        start_position != expected ||
        start_position > capacity_ * ratio_ - tokens) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 compressor prefill input");
    }

    const int windows = tokens / ratio_;
    const int cutoff = windows * ratio_;
    array tail_kv = state_kv_;
    array tail_gate = state_gate_;
    if (windows > 0) {
        auto grouped_kv = mlx::core::contiguous(
            mlx::core::reshape(
                slice_axis(values, 1, 0, cutoff),
                Shape{
                    batch_,
                    windows,
                    ratio_,
                    output_dim,
                }));
        auto grouped_gate = mlx::core::contiguous(
            mlx::core::reshape(
                slice_axis(gates, 1, 0, cutoff),
                Shape{
                    batch_,
                    windows,
                    ratio_,
                    output_dim,
                }));
        auto positions = mlx::core::contiguous(
            mlx::core::broadcast_to(
                mlx::core::reshape(
                    mlx::core::arange(
                        pool_len_,
                        pool_len_ + windows,
                        1,
                        mlx::core::int32),
                    Shape{1, windows}),
                Shape{batch_, windows}));
        const bool has_previous =
            overlap_ && pool_len_ > 0;
        auto compressed = dsv4_compress(
            grouped_kv,
            grouped_gate,
            ape,
            norm,
            has_previous ? prev_kv_ : std::nullopt,
            has_previous ? prev_gate_ : std::nullopt,
            positions,
            cosine,
            sine,
            ratio_,
            overlap_,
            quant_mode,
            eps);
        pool_ = dsv4_cache_write_inplace(
            pool_,
            compressed,
            positions);

        tail_kv = mlx::core::contiguous(
            slice_axis(
                values,
                1,
                cutoff - ratio_,
                cutoff));
        tail_gate = mlx::core::contiguous(
            slice_axis(
                gates,
                1,
                cutoff - ratio_,
                cutoff));
        if (overlap_) {
            prev_kv_ = mlx::core::contiguous(
                slice_axis(
                    tail_kv,
                    2,
                    0,
                    head_dim_));
            prev_gate_ = mlx::core::contiguous(
                slice_axis(
                    tail_gate,
                    2,
                    0,
                    head_dim_));
        }
        pool_len_ += windows;
    }

    const int remainder = tokens - cutoff;
    if (remainder > 0) {
        tail_kv = mlx::core::concatenate(
            {
                slice_axis(values, 1, cutoff, tokens),
                slice_axis(tail_kv, 1, remainder, ratio_),
            },
            1);
        tail_gate = mlx::core::concatenate(
            {
                slice_axis(gates, 1, cutoff, tokens),
                slice_axis(tail_gate, 1, remainder, ratio_),
            },
            1);
    }
    state_kv_ = mlx::core::contiguous(tail_kv);
    state_gate_ = mlx::core::contiguous(tail_gate);
    remainder_ = remainder;
}

MlxDeepseekV4LayerState::MlxDeepseekV4LayerState(
    array local,
    std::optional<MlxDeepseekV4PoolState> main,
    std::optional<MlxDeepseekV4PoolState> indexer)
    : local_(std::move(local)),
      main_(std::move(main)),
      indexer_(std::move(indexer)) {}

struct MlxDeepseekV4LayerSpeculation {
    MlxDeepseekV4LayerState checkpoint;
    int confirmed_tokens;
    int total_tokens;
    int start_position;
    std::optional<array> local_kv;
    std::optional<array> main_kv;
    std::optional<array> main_gate;
    std::optional<array> index_kv;
    std::optional<array> index_gate;
};

MlxDeepseekV4LayerState
MlxDeepseekV4LayerState::snapshot() const {
    MlxDeepseekV4LayerState result(
        detached_copy(local_),
        main_
            ? std::optional<MlxDeepseekV4PoolState>(
                  main_->snapshot())
            : std::nullopt,
        indexer_
            ? std::optional<MlxDeepseekV4PoolState>(
                  indexer_->snapshot())
            : std::nullopt);
    result.position_ = position_;
    return result;
}

void MlxDeepseekV4LayerState::restore_snapshot(
    MlxDeepseekV4LayerState snapshot) {
    if (static_cast<bool>(main_) !=
            static_cast<bool>(snapshot.main_) ||
        static_cast<bool>(indexer_) !=
            static_cast<bool>(snapshot.indexer_)) {
        throw std::invalid_argument(
            "DeepSeek-V4 layer snapshot topology mismatch");
    }
    local_ = std::move(snapshot.local_);
    if (main_) {
        main_->restore_snapshot(
            std::move(*snapshot.main_));
    }
    if (indexer_) {
        indexer_->restore_snapshot(
            std::move(*snapshot.indexer_));
    }
    speculative_.reset();
    position_ = snapshot.position_;
}

void MlxDeepseekV4LayerState::reset_v41() {
    speculative_.reset();
    if (main_) main_->reset_v41();
    if (indexer_) indexer_->reset_v41();
    position_ = 0;
}

void MlxDeepseekV4LayerState::restore_speculative_snapshot(
    MlxDeepseekV4LayerState snapshot,
    int start_position,
    int total_tokens) {
    if (static_cast<bool>(main_) !=
            static_cast<bool>(snapshot.main_) ||
        static_cast<bool>(indexer_) !=
            static_cast<bool>(snapshot.indexer_) ||
        snapshot.position_ != start_position ||
        snapshot.local_.ndim() != 3 ||
        snapshot.local_.shape(0) != batch() ||
        snapshot.local_.shape(1) != total_tokens ||
        snapshot.local_.shape(2) != local_.shape(2)) {
        throw std::invalid_argument(
            "DeepSeek-V4 speculative layer snapshot mismatch");
    }
    const int window = local_.shape(1);
    auto slots = mlx::core::remainder(
        mlx::core::arange(
            start_position,
            start_position + total_tokens,
            1,
            mlx::core::int32),
        array(window, mlx::core::int32));
    auto rows = mlx::core::broadcast_to(
        mlx::core::reshape(slots, Shape{1, total_tokens}),
        Shape{batch(), total_tokens});
    local_ = dsv4_cache_write_inplace(
        local_,
        snapshot.local_,
        rows);
    if (main_) {
        main_->restore_speculative_snapshot(
            std::move(*snapshot.main_));
    }
    if (indexer_) {
        indexer_->restore_speculative_snapshot(
            std::move(*snapshot.indexer_));
    }
    position_ = start_position;
}

void MlxDeepseekV4LayerState::begin_speculative(
    int confirmed_tokens,
    int total_tokens) {
    const int window = local_.shape(1);
    if (speculative_ || confirmed_tokens <= 0 ||
        total_tokens <= confirmed_tokens ||
        total_tokens > window) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 speculative cache transaction");
    }
    auto slots = mlx::core::remainder(
        mlx::core::arange(
            position_,
            position_ + total_tokens,
            1,
            mlx::core::int32),
        array(window, mlx::core::int32));
    MlxDeepseekV4LayerState checkpoint(
        detached_copy(mlx::core::take(local_, slots, 1)),
        main_
            ? std::optional<MlxDeepseekV4PoolState>(
                  main_->speculative_snapshot())
            : std::nullopt,
        indexer_
            ? std::optional<MlxDeepseekV4PoolState>(
                  indexer_->speculative_snapshot())
            : std::nullopt);
    checkpoint.position_ = position_;
    speculative_ = std::make_shared<MlxDeepseekV4LayerSpeculation>(
        MlxDeepseekV4LayerSpeculation{
            std::move(checkpoint),
            confirmed_tokens,
            total_tokens,
            position_,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
        });
}

const MlxDeepseekV4LayerState&
MlxDeepseekV4LayerState::speculative_checkpoint() const {
    if (!speculative_) {
        throw std::runtime_error(
            "DeepSeek-V4 speculative checkpoint is unavailable");
    }
    return speculative_->checkpoint;
}

bool MlxDeepseekV4LayerState::has_speculative() const noexcept {
    return static_cast<bool>(speculative_);
}

array MlxDeepseekV4LayerState::local_positions() const {
    const int window = local_.shape(1);
    auto slots = mlx::core::arange(
        window,
        mlx::core::int32);
    array positions = mlx::core::full(
        Shape{window},
        -1,
        mlx::core::int32);
    if (position_ > 0 && position_ < window) {
        positions = mlx::core::where(
            mlx::core::less(
                slots,
                array(position_, mlx::core::int32)),
            slots,
            positions);
    } else if (position_ >= window) {
        const int last = position_ - 1;
        positions =
            array(last, mlx::core::int32) -
            mlx::core::remainder(
                array(last, mlx::core::int32) - slots,
                array(window, mlx::core::int32));
    }
    return mlx::core::broadcast_to(
        mlx::core::reshape(
            positions,
            Shape{1, window}),
        Shape{batch(), window});
}

MlxDeepseekV4LayerState
MlxDeepseekV4LayerState::allocate(
    const DeepseekV4Config& config,
    int ratio,
    int batch,
    int max_context,
    Dtype dtype,
    std::optional<std::size_t> layer) {
    if (config.is_v41()) {
        if ((ratio != 0 && ratio != 1 && ratio != 2) ||
            !layer.has_value() || *layer >= config.compress_ratios.size() ||
            config.compress_ratios[*layer] != ratio ||
            batch <= 0 || max_context <= 0) {
            throw std::invalid_argument(
                "invalid DeepSeek-V4.1 layer cache allocation");
        }
        const int window = checked_int(
            config.sliding_window,
            "sliding window");
        const int head_dim = checked_int(
            config.head_dim,
            "head dimension");
        std::optional<MlxDeepseekV4PoolState> main;
        std::optional<MlxDeepseekV4PoolState> indexer;
        if (config.is_kv_source(*layer)) {
            if (ratio == 0) {
                throw std::invalid_argument(
                    "DeepSeek-V4.1 KV source has ratio zero");
            }
            main = MlxDeepseekV4PoolState::allocate_v41(
                ratio,
                head_dim,
                batch,
                max_context,
                mlx::core::bfloat16);
            indexer = MlxDeepseekV4PoolState::allocate_v41(
                ratio,
                checked_int(
                    config.index_head_dim,
                    "indexer head dimension"),
                batch,
                max_context,
                mlx::core::bfloat16);
        }
        return MlxDeepseekV4LayerState(
            mlx::core::zeros(
                Shape{batch, window, head_dim},
                mlx::core::bfloat16),
            std::move(main),
            std::move(indexer));
    }
    if (ratio != 0 && ratio != 4 && ratio != 128) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 layer compression ratio");
    }
    const int window = checked_int(
        config.sliding_window,
        "sliding window");
    const int head_dim = checked_int(
        config.head_dim,
        "head dimension");
    if (batch <= 0 || max_context <= 0) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 layer cache allocation");
    }
    std::optional<MlxDeepseekV4PoolState> main;
    std::optional<MlxDeepseekV4PoolState> indexer;
    if (ratio != 0) {
        main = MlxDeepseekV4PoolState::allocate(
            ratio,
            head_dim,
            ratio == 4,
            batch,
            max_context,
            dtype);
    }
    if (ratio == 4) {
        indexer = MlxDeepseekV4PoolState::allocate(
            ratio,
            checked_int(
                config.index_head_dim,
                "indexer head dimension"),
            true,
            batch,
            max_context,
            dtype);
    }
    return MlxDeepseekV4LayerState(
        mlx::core::zeros(
            Shape{batch, window, head_dim},
            dtype),
        std::move(main),
        std::move(indexer));
}

struct MlxDeepseekV4Attention::Impl {
    DeepseekV4Config config;
    int layer;
    int ratio;
    int maximum_context;
    MlxDeepseekV4AttentionComponents components;
    std::pair<array, array> rope;
    std::optional<std::pair<array, array>> pool_rope;
    array rms_params;
    MlxRmsNorm q_norm;
    std::optional<ProjectionGroup> projections;

    Impl(
        DeepseekV4Config selected_config,
        int selected_layer,
        int selected_ratio,
        int max_context,
        MlxDeepseekV4AttentionComponents selected_components,
        std::pair<array, array> selected_rope)
        : config(std::move(selected_config)),
          layer(selected_layer),
          ratio(selected_ratio),
          maximum_context(max_context),
          components(std::move(selected_components)),
          rope{
              typed_contiguous(
                  selected_rope.first,
                  mlx::core::float32),
              typed_contiguous(
                  selected_rope.second,
                  mlx::core::float32),
          },
          rms_params(
              {static_cast<float>(config.rms_eps)},
              mlx::core::float32),
          q_norm(
              components.q_norm,
              static_cast<float>(config.rms_eps)) {
        validate();
        if (ratio != 0) {
            const Shape strides{ratio, 1};
            pool_rope.emplace(
                mlx::core::contiguous(
                    mlx::core::slice(
                        rope.first,
                        Shape{0, 0},
                        rope.first.shape(),
                        strides)),
                mlx::core::contiguous(
                    mlx::core::slice(
                        rope.second,
                        Shape{0, 0},
                        rope.second.shape(),
                        strides)));
        }
        projections.emplace(projection_list());
    }

    std::vector<const MlxLinear*> projection_list() {
        std::vector<const MlxLinear*> result{
            &components.q_a,
            &components.kv,
        };
        if (config.is_v41()) {
            if (config.is_kv_source(
                    static_cast<std::size_t>(layer))) {
                result.push_back(&*components.main_kv);
                if (ratio > 1) {
                    result.push_back(&*components.main_gate);
                }
            }
            if (config.is_index_source(
                    static_cast<std::size_t>(layer))) {
                result.push_back(&*components.index_weights);
            }
            return result;
        }
        if (ratio != 0) {
            result.push_back(&*components.main_kv);
            result.push_back(&*components.main_gate);
        }
        if (ratio == 4) {
            result.push_back(&*components.index_kv);
            result.push_back(&*components.index_gate);
            result.push_back(&*components.index_weights);
        }
        return result;
    }

    void validate() {
        config.validate();
        const int hidden = checked_int(
            config.hidden,
            "hidden size");
        const int heads = checked_int(
            config.n_heads,
            "attention heads");
        const int head_dim = checked_int(
            config.head_dim,
            "attention head dimension");
        const int q_rank = checked_int(
            config.q_lora_rank,
            "query rank");
        const int groups = checked_int(
            config.o_groups,
            "output groups");
        const int o_rank = checked_int(
            config.o_lora_rank,
            "output rank");
        const int attention = checked_product(
            {heads, head_dim},
            "attention width");
        if (layer < 0 ||
            layer >= config.n_layers ||
            config.compress_ratios[layer] != ratio ||
            maximum_context <= 0 ||
            maximum_context >
                config.max_position_embeddings ||
            (config.is_v41()
                 ? (ratio != 0 && ratio != 1 && ratio != 2)
                 : (ratio != 0 && ratio != 4 && ratio != 128))) {
            throw std::invalid_argument(
                "invalid DeepSeek-V4 attention layer schedule");
        }
        require_linear(
            components.q_a,
            hidden,
            q_rank,
            "q_a");
        require_linear(
            components.kv,
            hidden,
            head_dim,
            "kv");
        require_linear(
            components.q_b,
            q_rank,
            attention,
            "q_b");
        require_linear(
            components.wo_a,
            attention / groups,
            groups * o_rank,
            "wo_a");
        require_linear(
            components.wo_b,
            groups * o_rank,
            hidden,
            "wo_b");
        require_vector(
            components.q_norm,
            q_rank,
            "q_norm");
        require_vector(
            components.kv_norm,
            head_dim,
            "kv_norm");
        require_vector(
            components.sinks,
            heads,
            "attention sinks");
        const int rotary = checked_int(
            config.qk_rope_head_dim,
            "rotary dimension");
        if (rope.first.shape() != rope.second.shape() ||
            rope.first.ndim() != 2 ||
            rope.first.shape(0) < maximum_context ||
            rope.first.shape(1) != rotary / 2) {
            throw std::invalid_argument(
                "DeepSeek-V4 attention RoPE table mismatch");
        }
        if (config.is_v41()) {
            const bool owns_kv = config.is_kv_source(
                static_cast<std::size_t>(layer));
            const bool owns_index = config.is_index_source(
                static_cast<std::size_t>(layer));
            const bool common_legacy_index =
                components.index_kv || components.index_gate ||
                components.index_ape || components.index_norm;
            if (components.main_ape || common_legacy_index ||
                static_cast<bool>(components.main_kv) != owns_kv ||
                static_cast<bool>(components.main_norm) != owns_kv ||
                static_cast<bool>(components.main_gate) !=
                    (owns_kv && ratio > 1) ||
                static_cast<bool>(components.index_q_b) != owns_index ||
                static_cast<bool>(components.index_weights) != owns_index ||
                static_cast<bool>(components.index_key) != owns_kv ||
                static_cast<bool>(components.index_key_norm) != owns_kv) {
                throw std::invalid_argument(
                    "DeepSeek-V4.1 sparse attention components mismatch");
            }
            if (owns_kv) {
                require_linear(
                    *components.main_kv,
                    hidden,
                    head_dim,
                    "main compressor KV");
                if (ratio > 1) {
                    require_linear(
                        *components.main_gate,
                        hidden,
                        head_dim,
                        "main compressor gate");
                }
                require_vector(
                    *components.main_norm,
                    head_dim,
                    "main compressor norm");
                require_linear(
                    *components.index_key,
                    head_dim,
                    checked_int(
                        config.index_head_dim,
                        "index dimension"),
                    "index key");
                require_vector(
                    *components.index_key_norm,
                    checked_int(
                        config.index_head_dim,
                        "index dimension"),
                    "index key norm");
            }
            if (owns_index) {
                const int index_heads = checked_int(
                    config.index_n_heads,
                    "index heads");
                const int index_dim = checked_int(
                    config.index_head_dim,
                    "index dimension");
                require_linear(
                    *components.index_q_b,
                    q_rank,
                    checked_product(
                        {index_heads, index_dim},
                        "index query width"),
                    "index query");
                require_linear(
                    *components.index_weights,
                    hidden,
                    index_heads,
                    "index weights");
            }
            return;
        }
        if (ratio == 0) {
            if (components.main_kv ||
                components.main_gate ||
                components.main_ape ||
                components.main_norm ||
                components.index_q_b ||
                components.index_kv ||
                components.index_gate ||
                components.index_weights ||
                components.index_ape ||
                components.index_norm ||
                components.index_key ||
                components.index_key_norm) {
                throw std::invalid_argument(
                    "ratio-zero DeepSeek-V4 layer received "
                    "compressor components");
            }
            return;
        }
        const int main_width =
            head_dim * (ratio == 4 ? 2 : 1);
        if (!components.main_kv ||
            !components.main_gate ||
            !components.main_ape ||
            !components.main_norm) {
            throw std::invalid_argument(
                "DeepSeek-V4 main compressor components missing");
        }
        require_linear(
            *components.main_kv,
            hidden,
            main_width,
            "main compressor KV");
        require_linear(
            *components.main_gate,
            hidden,
            main_width,
            "main compressor gate");
        require_matrix(
            *components.main_ape,
            ratio,
            main_width,
            "main compressor APE");
        require_vector(
            *components.main_norm,
            head_dim,
            "main compressor norm");
        if (ratio != 4) {
            if (components.index_q_b ||
                components.index_kv ||
                components.index_gate ||
                components.index_weights ||
                components.index_ape ||
                components.index_norm ||
                components.index_key ||
                components.index_key_norm) {
                throw std::invalid_argument(
                    "ratio-128 DeepSeek-V4 layer received "
                    "Indexer components");
            }
            return;
        }
        const int index_heads = checked_int(
            config.index_n_heads,
            "index heads");
        const int index_dim = checked_int(
            config.index_head_dim,
            "index dimension");
        const int index_width = checked_product(
            {index_heads, index_dim},
            "index query width");
        if (!components.index_q_b ||
            !components.index_kv ||
            !components.index_gate ||
            !components.index_weights ||
            !components.index_ape ||
            !components.index_norm) {
            throw std::invalid_argument(
                "DeepSeek-V4 Indexer components missing");
        }
        require_linear(
            *components.index_q_b,
            q_rank,
            index_width,
            "index query");
        require_linear(
            *components.index_kv,
            hidden,
            2 * index_dim,
            "index compressor KV");
        require_linear(
            *components.index_gate,
            hidden,
            2 * index_dim,
            "index compressor gate");
        require_linear(
            *components.index_weights,
            hidden,
            index_heads,
            "index weights");
        require_matrix(
            *components.index_ape,
            ratio,
            2 * index_dim,
            "index compressor APE");
        require_vector(
            *components.index_norm,
            index_dim,
            "index compressor norm");
    }

    std::vector<array> project(
        const array& input) const {
        return (*projections)(input);
    }

    array index_query(
        const array& q_rank,
        const array& positions) const {
        const int batch = q_rank.shape(0);
        const int tokens = q_rank.shape(1);
        const int heads = checked_int(
            config.index_n_heads,
            "index heads");
        const int dimension = checked_int(
            config.index_head_dim,
            "index dimension");
        auto query = mlx::core::reshape(
            (*components.index_q_b)(q_rank),
            Shape{batch, tokens, heads, dimension});
        const int rotary = checked_int(
            config.qk_rope_head_dim,
            "rotary dimension");
        auto cosine = mlx::core::take(
            rope.first,
            positions,
            0);
        auto sine = mlx::core::take(
            rope.second,
            positions,
            0);
        cosine = mlx::core::expand_dims(
            mlx::core::expand_dims(cosine, 0),
            2);
        sine = mlx::core::expand_dims(
            mlx::core::expand_dims(sine, 0),
            2);
        query = replace_last_rope(
            query,
            rotary,
            cosine,
            sine);
        if (config.is_v41()) {
            // V4.1 removed V4F's Hadamard stage. Both q and k cross an
            // E2M1/E8M0 fake-quant boundary in groups of 32.
            return v41_activation_quant(query, 32, 1);
        }
        if (config.fast_indexer()) {
            query = signed_hadamard(
                query,
                dimension);
            query = dsv4_fp4_sim(
                mlx::core::astype(
                    query,
                    mlx::core::float16));
        }
        return query;
    }

    array v41_candidate_mask(
        const array& scores,
        const array& visible_counts) const {
        const int batch = scores.shape(0);
        const int tokens = scores.shape(1);
        const int width = scores.shape(2);
        const int block = checked_int(
            config.candidate_block_size,
            "candidate block size");
        const int budget = checked_int(
            config.candidate_topk_blocks,
            "candidate block budget");
        const int padding = (block - width % block) % block;
        auto padded = scores;
        if (padding != 0) {
            padded = mlx::core::concatenate(
                {
                    scores,
                    mlx::core::full(
                        Shape{batch, tokens, padding},
                        -std::numeric_limits<float>::infinity(),
                        mlx::core::float32),
                },
                -1);
        }
        const int blocks = padded.shape(2) / block;
        auto block_scores = mlx::core::max(
            mlx::core::reshape(
                padded,
                Shape{batch, tokens, blocks, block}),
            -1);
        auto block_ids = mlx::core::reshape(
            mlx::core::arange(blocks, mlx::core::int32),
            Shape{1, 1, blocks});
        auto newest = mlx::core::floor_divide(
            visible_counts - array(1, mlx::core::int32),
            array(block, mlx::core::int32));
        auto pinned = mlx::core::equal(
            block_ids,
            mlx::core::reshape(newest, Shape{1, tokens, 1}));
        block_scores = mlx::core::where(
            mlx::core::broadcast_to(
                pinned,
                Shape{batch, tokens, blocks}),
            array(
                std::numeric_limits<float>::infinity(),
                mlx::core::float32),
            block_scores);

        const int selected_count = std::min(budget, blocks);
        if (selected_count == blocks) {
            // Every block is a candidate below the configured budget.  The
            // generic expansion compares every position with every selected
            // block, producing [B,T,width,blocks].  At V4.1's 2K-block
            // budget this is both redundant and quadratic in the prompt
            // width.  Gather the per-block validity bit for each position
            // instead; this exactly preserves the pinned partial-block and
            // invisible-block behaviour while keeping only [B,T,width].
            auto position_blocks = mlx::core::broadcast_to(
                mlx::core::reshape(
                    mlx::core::floor_divide(
                        mlx::core::arange(width, mlx::core::int32),
                        array(block, mlx::core::int32)),
                    Shape{1, 1, width}),
                Shape{batch, tokens, width});
            return mlx::core::take_along_axis(
                mlx::core::greater(
                    block_scores,
                    array(
                        -std::numeric_limits<float>::infinity(),
                        mlx::core::float32)),
                position_blocks,
                -1);
        }
        auto selected = slice_axis(
            mlx::core::argpartition(
                block_scores,
                blocks - selected_count,
                -1),
            -1,
            blocks - selected_count,
            blocks);
        selected = mlx::core::astype(selected, mlx::core::int32);
        auto selected_scores = mlx::core::take_along_axis(
            block_scores,
            selected,
            -1);
        auto selected_valid = mlx::core::greater(
            selected_scores,
            array(
                -std::numeric_limits<float>::infinity(),
                mlx::core::float32));
        auto position_blocks = mlx::core::floor_divide(
            mlx::core::arange(width, mlx::core::int32),
            array(block, mlx::core::int32));
        auto matches = mlx::core::equal(
            mlx::core::reshape(
                position_blocks,
                Shape{1, 1, width, 1}),
            mlx::core::expand_dims(selected, 2));
        matches = mlx::core::logical_and(
            matches,
            mlx::core::expand_dims(selected_valid, 2));
        return mlx::core::greater(
            mlx::core::sum(
                mlx::core::astype(matches, mlx::core::int32),
                -1),
            array(0, mlx::core::int32));
    }

    array v41_index_topk(
        const array& source,
        const array& q_rank,
        const array& index_weights,
        const array& positions,
        int query_offset,
        MlxDeepseekV41HfSharedAttentionState& shared) const {
        if (shared.index_keys == nullptr ||
            shared.compressed_kv == nullptr ||
            shared.index_keys->pool_len() !=
                shared.compressed_kv->pool_len()) {
            throw std::runtime_error(
                "DeepSeek-V4.1 shared Indexer cache is unavailable");
        }
        const int batch = source.shape(0);
        const int tokens = source.shape(1);
        const int pool_len = shared.index_keys->pool_len();
        if (pool_len == 0) {
            auto empty = mlx::core::zeros(
                Shape{batch, tokens, 0},
                mlx::core::int32);
            shared.topk = empty;
            return empty;
        }
        auto visible_counts = mlx::core::floor_divide(
            positions + array(1, mlx::core::int32),
            array(ratio, mlx::core::int32));
        auto key_ids = mlx::core::reshape(
            mlx::core::arange(pool_len, mlx::core::int32),
            Shape{1, 1, pool_len});
        auto visible = mlx::core::less(
            key_ids,
            mlx::core::reshape(
                visible_counts,
                Shape{1, tokens, 1}));
        const int count = std::min(
            checked_int(config.index_topk, "index top-k"),
            pool_len);
        if (count == pool_len) {
            // Until the compressed pool exceeds index_topk, ranking cannot
            // affect the selected set.  Avoid constructing index queries and
            // scoring every query/key pair.  The candidate mask is likewise
            // observational only in this case; retain its visible-key shape
            // so later index-source layers see the expected shared state.
            auto reachable = mlx::core::broadcast_to(
                visible,
                Shape{batch, tokens, pool_len});
            if (layer == config.candidate_source_layer_id) {
                shared.candidates = reachable;
            }
            auto selected = mlx::core::where(
                reachable,
                mlx::core::broadcast_to(
                    key_ids,
                    Shape{batch, tokens, pool_len}),
                array(-1, mlx::core::int32));
            shared.topk = selected;
            return selected;
        }

        auto query = index_query(q_rank, positions);
        const bool fast_indexer = v41_fast_indexer_enabled();
        auto scores = fast_indexer
            ? (tokens == 1
                   ? dsv4_indexer_scores_decode(
                         query,
                         pool_prefix(*shared.index_keys),
                         index_weights,
                         query_offset,
                         ratio,
                         pool_len)
                   : dsv4_indexer_scores(
                         query,
                         pool_prefix(*shared.index_keys),
                         index_weights,
                         query_offset,
                         ratio))
            : [&] {
                  auto keys = mlx::core::astype(
                      pool_prefix(*shared.index_keys),
                      mlx::core::float32);
                  auto dots = mlx::core::sum(
                      mlx::core::expand_dims(
                          mlx::core::astype(
                              query,
                              mlx::core::float32),
                          3) *
                          mlx::core::expand_dims(
                              mlx::core::expand_dims(keys, 1),
                              1),
                      -1);
                  auto weights = mlx::core::astype(
                      index_weights,
                      mlx::core::float32) /
                      std::sqrt(static_cast<double>(
                          checked_int(
                              config.index_head_dim,
                              "index dimension") *
                          checked_int(
                              config.index_n_heads,
                              "index heads")));
                  return mlx::core::sum(
                      mlx::core::maximum(
                          dots,
                          array(0.0f, mlx::core::float32)) *
                          mlx::core::expand_dims(weights, -1),
                      2);
              }();
        if (!fast_indexer) {
            scores = mlx::core::where(
                mlx::core::broadcast_to(
                    visible,
                    Shape{batch, tokens, pool_len}),
                scores,
                array(
                    -std::numeric_limits<float>::infinity(),
                    mlx::core::float32));
        }

        if (layer == config.candidate_source_layer_id) {
            shared.candidates = v41_candidate_mask(
                scores,
                visible_counts);
            if (detail::component_profile_active()) {
                detail::profile_eval(
                    std::string("attention.r")
                        + std::to_string(ratio)
                        + ".candidate_mask",
                    *shared.candidates);
            }
        } else if (config.candidate_source_layer_id >= 0 &&
                   layer > config.candidate_source_layer_id) {
            if (!shared.candidates.has_value() ||
                shared.candidates->shape() != scores.shape()) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 candidate mask is unavailable");
            }
            scores = mlx::core::where(
                *shared.candidates,
                scores,
                array(
                    -std::numeric_limits<float>::infinity(),
                    mlx::core::float32));
        }

        array selected = [&]() -> array {
            const bool use_deepselect =
                config.index_topk == 512 &&
                count == 512 &&
                dsv41_deepselect_topk_preferred(
                    pool_len,
                    batch * tokens);
            if (use_deepselect) {
                auto per_row_visible = mlx::core::broadcast_to(
                    mlx::core::reshape(
                        visible_counts,
                        Shape{1, tokens}),
                    Shape{batch, tokens});
                return dsv41_deepselect_topk512(
                    scores,
                    per_row_visible);
            }
            return slice_axis(
                mlx::core::argpartition(
                    scores,
                    pool_len - count,
                    -1),
                -1,
                pool_len - count,
                pool_len);
        }();
        selected = mlx::core::sort(
            mlx::core::astype(selected, mlx::core::int32),
            -1);
        auto reachable = mlx::core::less(
            selected,
            mlx::core::reshape(
                visible_counts,
                Shape{1, tokens, 1}));
        auto result = mlx::core::where(
            mlx::core::broadcast_to(
                reachable,
                Shape{batch, tokens, count}),
            selected,
            array(-1, mlx::core::int32));
        shared.topk = result;
        return result;
    }

    array topk(
        const array& q_rank,
        const array& index_weights,
        const MlxDeepseekV4LayerState& state,
        const array& positions,
        int pos0) const {
        const int batch = q_rank.shape(0);
        const int tokens = q_rank.shape(1);
        const int pool_len =
            state.main_.has_value()
            ? state.main_->pool_len()
            : 0;
        if (pool_len <= 0) {
            return mlx::core::zeros(
                Shape{batch, tokens, 0},
                mlx::core::int32);
        }
        const int requested = checked_int(
            config.index_topk,
            "index top-k");
        if (ratio != 4 || pool_len <= requested) {
            return mlx::core::broadcast_to(
                mlx::core::reshape(
                    mlx::core::arange(
                        pool_len,
                        mlx::core::int32),
                    Shape{1, 1, pool_len}),
                Shape{batch, tokens, pool_len});
        }
        auto query = index_query(
            q_rank,
            positions);
        if (config.fast_indexer() &&
            requested == 512) {
            if (tokens == 1) {
                return dsv4_topk512(
                    dsv4_indexer_scores_decode(
                        query,
                        state.indexer_->pool(),
                        index_weights,
                        pos0,
                        ratio,
                        pool_len),
                    true,
                    pool_len);
            }
            return dsv4_topk512(
                dsv4_indexer_scores(
                    query,
                    pool_prefix(*state.indexer_),
                    index_weights,
                    pos0,
                    ratio));
        }
        return generic_index_topk(
            query,
            pool_prefix(*state.indexer_),
            index_weights,
            positions,
            ratio,
            requested);
    }

    array forward_v41(
        const array& source,
        MlxDeepseekV4LayerState& state,
        int pos0,
        const MlxDeepseekV4ImageVisibility* visibility,
        std::vector<array>* debug_stages,
        MlxDeepseekV41HfSharedAttentionState* shared) const {
        const int batch = source.shape(0);
        const int tokens = source.shape(1);
        const int heads = checked_int(
            config.n_heads,
            "attention heads");
        const int head_dim = checked_int(
            config.head_dim,
            "head dimension");
        const int window = checked_int(
            config.sliding_window,
            "sliding window");
        const bool owns_kv = config.is_kv_source(
            static_cast<std::size_t>(layer));
        const bool owns_index = config.is_index_source(
            static_cast<std::size_t>(layer));
        const auto profile_component =
            [ratio = this->ratio](const char* component) {
                return std::string("attention.r")
                    + std::to_string(ratio)
                    + ".v41." + component;
            };
        if ((ratio != 0 && shared == nullptr) ||
            state.main_.has_value() != owns_kv ||
            state.indexer_.has_value() != owns_kv ||
            (owns_kv && ratio == 0)) {
            throw std::invalid_argument(
                "invalid DeepSeek-V4.1 shared attention state");
        }

        MlxDeepseekV4LayerSpeculation* speculation = nullptr;
        if (state.speculative_) {
            auto& transaction = *state.speculative_;
            if (transaction.local_kv ||
                transaction.start_position != pos0 ||
                transaction.total_tokens != tokens) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 speculative attention transaction mismatch");
            }
            speculation = &transaction;
        }
        auto positions = mlx::core::arange(
            pos0,
            pos0 + tokens,
            1,
            mlx::core::int32);
        auto projected = project(source);
        if (detail::component_profile_active()) {
            detail::profile_eval(
                profile_component("input_projections"),
                projected);
        }
        std::size_t output = 0;
        auto q_rank = q_norm(projected.at(output++));
        auto kv = projected.at(output++);
        std::optional<array> main_kv;
        std::optional<array> main_gate;
        if (owns_kv) {
            main_kv = projected.at(output++);
            if (ratio > 1) {
                main_gate = projected.at(output++);
            }
        }
        array index_weights = mlx::core::zeros(
            Shape{batch, tokens, 0},
            source.dtype());
        if (owns_index) {
            index_weights = projected.at(output++);
        }
        if (output != projected.size()) {
            throw std::runtime_error(
                "DeepSeek-V4.1 projection group output mismatch");
        }

        const int rotary = checked_int(
            config.qk_rope_head_dim,
            "rotary dimension");
        auto cosine = mlx::core::take(rope.first, positions, 0);
        auto sine = mlx::core::take(rope.second, positions, 0);
        auto q = mlx::core::reshape(
            components.q_b(q_rank),
            Shape{batch, tokens, heads, head_dim});
        q = replace_last_rope(
            q,
            rotary,
            mlx::core::expand_dims(
                mlx::core::expand_dims(cosine, 0),
                2),
            mlx::core::expand_dims(
                mlx::core::expand_dims(sine, 0),
                2));
        kv = v41_weighted_rms_rope_activation_quant(
            kv,
            components.kv_norm,
            source.dtype(),
            static_cast<float>(config.rms_eps),
            rotary,
            mlx::core::expand_dims(cosine, 0),
            mlx::core::expand_dims(sine, 0),
            rms_params);
        if (detail::component_profile_active()) {
            detail::profile_eval(
                profile_component("q_kv_prepare"),
                std::vector<array>{q_rank, q, kv});
        }
        if (speculation != nullptr) {
            speculation->local_kv = kv;
            if (owns_kv) {
                speculation->main_kv = *main_kv;
                speculation->main_gate = main_gate;
            }
        }
        if (debug_stages != nullptr) {
            debug_stages->push_back(source);
            debug_stages->push_back(projected.at(0));
            debug_stages->push_back(q_rank);
            debug_stages->push_back(q);
            debug_stages->push_back(kv);
        }

        if (owns_kv) {
            shared->topk.reset();
            shared->candidates.reset();
            const int first_row = state.main_->pool_len();
            auto latent = state.main_->compress_v41(
                *main_kv,
                main_gate,
                *components.main_norm,
                pos0,
                source.dtype(),
                static_cast<float>(config.rms_eps));
            const int emitted = latent.shape(1);
            if (emitted > 0) {
                auto pool_rows = mlx::core::arange(
                    first_row,
                    first_row + emitted,
                    1,
                    mlx::core::int32);
                auto pool_cosine = mlx::core::take(
                    pool_rope->first,
                    pool_rows,
                    0);
                auto pool_sine = mlx::core::take(
                    pool_rope->second,
                    pool_rows,
                    0);

                auto index_key = weighted_rms(
                    (*components.index_key)(latent),
                    *components.index_key_norm,
                    source.dtype(),
                    static_cast<float>(config.rms_eps));
                index_key = replace_last_rope(
                    index_key,
                    rotary,
                    mlx::core::expand_dims(pool_cosine, 0),
                    mlx::core::expand_dims(pool_sine, 0));
                index_key = v41_activation_quant(
                    index_key,
                    32,
                    1);
                state.indexer_->append_v41(index_key);

                auto compressed = replace_last_rope(
                    latent,
                    rotary,
                    mlx::core::expand_dims(pool_cosine, 0),
                    mlx::core::expand_dims(pool_sine, 0));
                compressed = v41_activation_quant(
                    compressed,
                    16,
                    2);
                state.main_->append_v41(compressed);
            }
            if (state.main_->pool_len() !=
                state.indexer_->pool_len()) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 shared cache lengths diverged");
            }
            shared->compressed_kv = &*state.main_;
            shared->index_keys = &*state.indexer_;
            if (detail::component_profile_active()) {
                detail::profile_eval(
                    profile_component("compressor_cache"),
                    std::vector<array>{
                        state.main_->pool(),
                        state.indexer_->pool(),
                    });
            }
        }

        array selected = mlx::core::zeros(
            Shape{batch, tokens, 0},
            mlx::core::int32);
        int pool_len = 0;
        if (ratio != 0) {
            if (shared->compressed_kv == nullptr ||
                shared->index_keys == nullptr ||
                shared->compressed_kv->ratio() != ratio ||
                shared->index_keys->ratio() != ratio ||
                shared->compressed_kv->batch() != batch) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 compressed attention source is unavailable");
            }
            pool_len = shared->compressed_kv->pool_len();
            if (owns_index) {
                selected = v41_index_topk(
                    source,
                    q_rank,
                    index_weights,
                    positions,
                    pos0,
                    *shared);
                if (detail::component_profile_active()) {
                    detail::profile_eval(
                        profile_component("indexer_topk"),
                        selected);
                }
            } else {
                if (!shared->topk.has_value() ||
                    shared->topk->ndim() != 3 ||
                    shared->topk->shape(0) != batch ||
                    shared->topk->shape(1) != tokens) {
                    throw std::runtime_error(
                        "DeepSeek-V4.1 shared top-k is unavailable");
                }
                selected = *shared->topk;
            }
        }

        array unified = state.local_;
        std::optional<std::pair<array, array>> plan;
        std::optional<array> direct_decode;
        std::optional<array> direct_prefill;
        if (tokens == 1 && !config.has_dspark()) {
            const int slot = pos0 % window;
            state.local_ = dsv4_cache_write_inplace(
                state.local_,
                mlx::core::astype(kv, state.local_.dtype()),
                mlx::core::full(
                    Shape{batch, 1},
                    slot,
                    mlx::core::int32));
            unified = state.local_;
            if (config.fast_attention()) {
                std::optional<array> pooled;
                if (ratio != 0) {
                    pooled = shared->compressed_kv->pool();
                }
                direct_decode = attention_dsv4_sparse_decode(
                    mlx::core::transpose(q, {0, 2, 1, 3}),
                    state.local_,
                    pooled,
                    pool_len,
                    selected,
                    components.sinks,
                    pos0 + 1,
                    ratio == 0 ? 1 : ratio,
                    window);
            } else {
                if (ratio != 0) {
                    unified = mlx::core::concatenate(
                        {
                            state.local_,
                            pool_prefix(*shared->compressed_kv),
                        },
                        1);
                }
                plan = dsv4_build_decode_plan(
                    selected,
                    mlx::core::full(
                        Shape{batch},
                        pos0 + 1,
                        mlx::core::int32),
                    pool_len,
                    ratio == 0 ? 1 : ratio,
                    window);
            }
        } else {
            const int history = std::min(pos0, window);
            array history_values = slice_axis(
                state.local_,
                1,
                0,
                0);
            if (history > 0) {
                auto history_positions = mlx::core::arange(
                    pos0 - history,
                    pos0,
                    1,
                    mlx::core::int32);
                history_values = mlx::core::take(
                    state.local_,
                    mlx::core::remainder(
                        history_positions,
                        array(window, mlx::core::int32)),
                    1);
            }
            std::vector<array> local_parts{
                history_values,
                mlx::core::astype(kv, state.local_.dtype()),
            };
            auto chronological_local = mlx::core::concatenate(
                std::move(local_parts), 1);
            // V4.1 quantized verification is reproducible through its
            // acceptance-only depth policy rather than by forcing every
            // width onto the slower selected-attention reduction. Reuse the
            // direct circular kernel for both long prefill and MTP rows.
            const bool use_circular_prefill =
                tokens > 1 &&
                visibility == nullptr &&
                ratio != 0 &&
                pool_len > 0 &&
                selected.shape(2) > 0 &&
                config.fast_attention() &&
                v41_circular_prefill_enabled();
            if (use_circular_prefill) {
                direct_prefill = attention_dsv4_sparse_prefill(
                    mlx::core::transpose(q, {0, 2, 1, 3}),
                    chronological_local,
                    shared->compressed_kv->pool(),
                    pool_len,
                    selected,
                    components.sinks,
                    pos0,
                    ratio,
                    window);
            } else {
                std::vector<array> parts{chronological_local};
                if (ratio != 0) {
                    parts.push_back(pool_prefix(*shared->compressed_kv));
                }
                unified = mlx::core::concatenate(std::move(parts), 1);
                plan = visibility != nullptr
                    ? dsv4_build_prefill_plan_visible(
                          selected,
                          visibility->left,
                          visibility->right,
                          pos0,
                          history,
                          pool_len,
                          ratio == 0 ? 1 : ratio,
                          window,
                          visibility->max_image_tokens)
                    : dsv4_build_prefill_plan(
                          selected,
                          pos0,
                          history,
                          pool_len,
                          ratio == 0 ? 1 : ratio,
                          window);
            }
            const int recent = std::min(tokens, window);
            auto recent_positions = mlx::core::arange(
                pos0 + tokens - recent,
                pos0 + tokens,
                1,
                mlx::core::int32);
            auto local_indices = mlx::core::broadcast_to(
                mlx::core::reshape(
                    mlx::core::remainder(
                        recent_positions,
                        array(window, mlx::core::int32)),
                    Shape{1, recent}),
                Shape{batch, recent});
            state.local_ = dsv4_cache_write_inplace(
                state.local_,
                mlx::core::astype(
                    slice_axis(kv, 1, tokens - recent, tokens),
                    state.local_.dtype()),
                local_indices);
        }

        if (detail::component_profile_active()) {
            if (direct_decode || direct_prefill) {
                detail::profile_eval(
                    profile_component("cache_update"),
                    std::vector<array>{state.local_});
            } else {
                detail::profile_eval(
                    profile_component("cache_plan"),
                    std::vector<array>{
                        state.local_,
                        unified,
                        plan->first,
                        plan->second,
                    });
            }
        }

        auto attended = direct_decode
            ? *direct_decode
            : direct_prefill
                ? *direct_prefill
            : config.fast_attention()
                ? attention_dsv4_sparse(
                      mlx::core::transpose(q, {0, 2, 1, 3}),
                      unified,
                      plan->first,
                      plan->second,
                      components.sinks)
                : generic_sparse_attention(
                      q,
                      unified,
                      plan->first,
                      plan->second,
                      components.sinks);
        if (debug_stages != nullptr) {
            debug_stages->push_back(attended);
        }
        if (detail::component_profile_active()) {
            detail::profile_eval(
                profile_component("sparse_core"),
                attended);
        }
        auto result = output_projection(
            attended,
            cosine,
            sine,
            debug_stages);
        state.position_ += tokens;
        return result;
    }

    array output_projection(
        const array& value,
        const array& cosine,
        const array& sine,
        std::vector<array>* debug_stages = nullptr) const {
        const int batch = value.shape(0);
        const int tokens = value.shape(1);
        const int groups = checked_int(
            config.o_groups,
            "output groups");
        const int rank = checked_int(
            config.o_lora_rank,
            "output rank");
        const int input_width =
            checked_product(
                {
                    checked_int(
                        config.n_heads,
                        "attention heads"),
                    checked_int(
                        config.head_dim,
                        "head dimension"),
                },
                "attention width") /
            groups;
        const int head_dim = checked_int(
            config.head_dim,
            "head dimension");
        const int rotary = checked_int(
            config.qk_rope_head_dim,
            "rotary dimension");
        auto grouped = mlx::core::reshape(
            value,
            Shape{
                batch,
                tokens,
                groups,
                input_width,
            });
        array low_rank =
            components.wo_a.nint8_zero_weight_ref()
            ? components.wo_a.nint8_zero_weight_ref()
                  ->grouped_row_matmul_inverse_rope(
                      grouped,
                      groups,
                      cosine,
                      sine,
                      head_dim,
                      rotary)
            : components.wo_a.mx_weight_ref() &&
                    ((components.wo_a.mx_weight_ref()
                              ->scale_block_size() == 128 &&
                      tokens <= 6) ||
                     (components.wo_a.mx_weight_ref()
                              ->scale_block_size() == 32 &&
                      block32_inverse_rope_qmv_enabled() &&
                      value.dtype() == mlx::core::float32 &&
                      batch * tokens == 1))
            ? components.wo_a.mx_weight_ref()
                  ->grouped_row_matmul_inverse_rope(
                      grouped,
                      groups,
                      cosine,
                      sine,
                      head_dim,
                      rotary)
            : components.wo_a.grouped_row_matmul(
                  mlx::core::reshape(
                      replace_last_rope(
                          value,
                          rotary,
                          cosine,
                          sine,
                          true),
                      Shape{
                          batch,
                          tokens,
                          groups,
                          input_width,
                      }),
                  groups);
        low_rank = mlx::core::reshape(
            std::move(low_rank),
            Shape{
                batch,
                tokens,
                groups * rank,
            });
        if (debug_stages != nullptr) {
            debug_stages->push_back(low_rank);
        }
        if (detail::component_profile_active()) {
            detail::profile_eval(
                std::string("attention.r")
                    + std::to_string(ratio)
                    + ".output_projection_a",
                low_rank);
        }
        auto result = components.wo_b(low_rank);
        if (debug_stages != nullptr) {
            debug_stages->push_back(result);
        }
        if (detail::component_profile_active()) {
            detail::profile_eval(
                std::string("attention.r")
                    + std::to_string(ratio)
                    + ".output_projection_b",
                result);
        }
        return result;
    }
};

MlxDeepseekV4Attention MlxDeepseekV4Attention::load(
    const MfqContainer& model,
    const DeepseekV4Config& config,
    int layer,
    int ratio,
    int max_context) {
    auto base = deepseek_v4_yarn_tables(
        checked_int(
            config.qk_rope_head_dim,
            "rotary dimension"),
        max_context,
        static_cast<float>(config.rope_theta),
        config.rope_scaling);
    auto compressed = deepseek_v4_yarn_tables(
        checked_int(
            config.qk_rope_head_dim,
            "rotary dimension"),
        max_context,
        static_cast<float>(
            config.compress_rope_theta),
        config.rope_scaling);
    return load(
        model,
        config,
        layer,
        ratio,
        max_context,
        std::move(base),
        std::move(compressed));
}

MlxDeepseekV4Attention MlxDeepseekV4Attention::load(
    const MfqContainer& model,
    const DeepseekV4Config& config,
    int layer,
    int ratio,
    int max_context,
    std::pair<array, array> rope_base,
    std::pair<array, array> rope_compressed) {
    config.validate();
    if (layer < 0 ||
        layer >= config.n_layers ||
        static_cast<std::size_t>(layer) >=
            config.compress_ratios.size() ||
        config.compress_ratios[layer] != ratio ||
        max_context <= 0 ||
        max_context >
            config.max_position_embeddings) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 attention load schedule");
    }
    const auto name =
        [layer](std::string_view suffix) {
            return DeepseekV4TensorNames::layer(
                static_cast<std::size_t>(layer),
                suffix);
        };
    MlxDeepseekV4AttentionComponents components{
        MlxLinear::load(
            model,
            name("attention.query_a.weight")),
        MlxLinear::load(
            model,
            name("attention.key_value_a.weight")),
        MlxLinear::load(
            model,
            name("attention.query_b.weight")),
        MlxLinear::load(
            model,
            name("attention.output_a.weight")),
        MlxLinear::load(
            model,
            name("attention.output_b.weight")),
        load_float_array(
            model,
            name("attention.query_a_norm.weight")),
        load_float_array(
            model,
            name("attention.key_value_a_norm.weight")),
        load_float_array(
            model,
            name("attention.sink")),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
    };
    if (config.is_v41()) {
        const bool owns_kv = config.is_kv_source(
            static_cast<std::size_t>(layer));
        const bool owns_index = config.is_index_source(
            static_cast<std::size_t>(layer));
        if (owns_kv) {
            components.main_kv.emplace(
                MlxLinear::load(
                    model,
                    name("attention.compressor.key_value.weight")));
            if (ratio > 1) {
                components.main_gate.emplace(
                    MlxLinear::load(
                        model,
                        name("attention.compressor.gate.weight")));
            }
            components.main_norm = load_float_array(
                model,
                name("attention.compressor.norm.weight"));
            components.index_key.emplace(
                MlxLinear::load(
                    model,
                    name("attention.indexer.key.weight")));
            components.index_key_norm = load_float_array(
                model,
                name("attention.indexer.key_norm.weight"));
        }
        if (owns_index) {
            components.index_q_b.emplace(
                MlxLinear::load(
                    model,
                    name("attention.indexer.query.weight")));
            components.index_weights.emplace(
                MlxLinear::load(
                    model,
                    name("attention.indexer.score.weight")));
        }
    } else if (ratio != 0) {
        components.main_kv.emplace(
            MlxLinear::load(
                model,
                name("attention.compressor.key_value.weight")));
        components.main_gate.emplace(
            MlxLinear::load(
                model,
                name("attention.compressor.gate.weight")));
        components.main_ape = load_float_array(
            model,
            name("attention.compressor.position"));
        components.main_norm = load_float_array(
            model,
            name("attention.compressor.norm.weight"));
    }
    if (!config.is_v41() && ratio == 4) {
        components.index_q_b.emplace(
            MlxLinear::load(
                model,
                name("attention.indexer.query.weight")));
        components.index_kv.emplace(
            MlxLinear::load(
                model,
                name(
                    "attention.indexer.compressor.key_value.weight")));
        components.index_gate.emplace(
            MlxLinear::load(
                model,
                name(
                    "attention.indexer.compressor.gate.weight")));
        components.index_weights.emplace(
            MlxLinear::load(
                model,
                name("attention.indexer.score.weight")));
        components.index_ape = load_float_array(
            model,
            name("attention.indexer.compressor.position"));
        components.index_norm = load_float_array(
            model,
            name(
                "attention.indexer.compressor.norm.weight"));
    }
    return MlxDeepseekV4Attention(
        config,
        layer,
        ratio,
        max_context,
        std::move(components),
        std::move(rope_base),
        std::move(rope_compressed));
}

MlxDeepseekV4Attention MlxDeepseekV4Attention::load(
    const MlxHfTensorStore& model,
    const DeepseekV4Config& config,
    int layer,
    int ratio,
    int max_context,
    std::pair<array, array> rope_base,
    std::pair<array, array> rope_compressed) {
    config.validate();
    if (layer < 0 ||
        layer >= config.n_layers ||
        static_cast<std::size_t>(layer) >= config.compress_ratios.size() ||
        config.compress_ratios[layer] != ratio ||
        max_context <= 0 || max_context > config.max_position_embeddings) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 HF attention load schedule");
    }
    const auto name = [layer](std::string_view suffix) {
        return DeepseekV4TensorNames::layer(
            static_cast<std::size_t>(layer), suffix);
    };
    const auto load_float = [&model](const std::string& tensor) {
        return typed_contiguous(
            model.load_dense(tensor),
            mlx::core::float32);
    };
    MlxDeepseekV4AttentionComponents components{
        model.load_linear(name("attention.query_a.weight")),
        model.load_linear(name("attention.key_value_a.weight")),
        model.load_linear(name("attention.query_b.weight")),
        model.load_linear(name("attention.output_a.weight")),
        model.load_linear(name("attention.output_b.weight")),
        load_float(name("attention.query_a_norm.weight")),
        load_float(name("attention.key_value_a_norm.weight")),
        load_float(name("attention.sink")),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
    };
    if (config.is_v41()) {
        const bool owns_kv = config.is_kv_source(
            static_cast<std::size_t>(layer));
        const bool owns_index = config.is_index_source(
            static_cast<std::size_t>(layer));
        if (owns_kv) {
            components.main_kv.emplace(model.load_linear(
                name("attention.compressor.key_value.weight")));
            if (ratio > 1) {
                components.main_gate.emplace(model.load_linear(
                    name("attention.compressor.gate.weight")));
            }
            components.main_norm = load_float(
                name("attention.compressor.norm.weight"));
            components.index_key.emplace(model.load_linear(
                name("attention.indexer.key.weight")));
            components.index_key_norm = load_float(
                name("attention.indexer.key_norm.weight"));
        }
        if (owns_index) {
            components.index_q_b.emplace(model.load_linear(
                name("attention.indexer.query.weight")));
            components.index_weights.emplace(model.load_linear(
                name("attention.indexer.score.weight")));
        }
    } else if (ratio != 0) {
        components.main_kv.emplace(
            model.load_linear(name("attention.compressor.key_value.weight")));
        components.main_gate.emplace(
            model.load_linear(name("attention.compressor.gate.weight")));
        components.main_ape = load_float(name("attention.compressor.position"));
        components.main_norm = load_float(
            name("attention.compressor.norm.weight"));
    }
    if (!config.is_v41() && ratio == 4) {
        components.index_q_b.emplace(
            model.load_linear(name("attention.indexer.query.weight")));
        components.index_kv.emplace(model.load_linear(
            name("attention.indexer.compressor.key_value.weight")));
        components.index_gate.emplace(model.load_linear(
            name("attention.indexer.compressor.gate.weight")));
        components.index_weights.emplace(model.load_linear(
            name("attention.indexer.score.weight")));
        components.index_ape = load_float(
            name("attention.indexer.compressor.position"));
        components.index_norm = load_float(
            name("attention.indexer.compressor.norm.weight"));
    }
    return MlxDeepseekV4Attention(
        config,
        layer,
        ratio,
        max_context,
        std::move(components),
        std::move(rope_base),
        std::move(rope_compressed));
}

MlxDeepseekV4Attention::MlxDeepseekV4Attention(
    DeepseekV4Config config,
    int layer,
    int ratio,
    int max_context,
    MlxDeepseekV4AttentionComponents components,
    std::pair<array, array> rope_base,
    std::pair<array, array> rope_compressed)
    : impl_(std::make_shared<Impl>(
          std::move(config),
          layer,
          ratio,
          max_context,
          std::move(components),
          ratio == 0
              ? std::move(rope_base)
              : std::move(rope_compressed))) {}

array MlxDeepseekV4Attention::operator()(
    const array& input,
    MlxDeepseekV4LayerState& state,
    int pos0) const {
    return (*this)(input, state, pos0, nullptr);
}

array MlxDeepseekV4Attention::operator()(
    const array& input,
    MlxDeepseekV4LayerState& state,
    int pos0,
    const MlxDeepseekV4ImageVisibility* visibility) const {
    return (*this)(input, state, pos0, visibility, nullptr);
}

array MlxDeepseekV4Attention::operator()(
    const array& input,
    MlxDeepseekV4LayerState& state,
    int pos0,
    const MlxDeepseekV4ImageVisibility* visibility,
    std::vector<array>* debug_stages,
    MlxDeepseekV41HfSharedAttentionState* shared) const {
    auto source = floating_contiguous(input);
    const auto& config = impl_->config;
    const int hidden = checked_int(
        config.hidden,
        "hidden size");
    const int heads = checked_int(
        config.n_heads,
        "attention heads");
    const int head_dim = checked_int(
        config.head_dim,
        "head dimension");
    if (source.ndim() != 3 ||
        source.shape(0) <= 0 ||
        source.shape(1) <= 0 ||
        source.shape(2) != hidden ||
        state.batch() != source.shape(0) ||
        state.position_ != pos0 ||
        pos0 < 0 ||
        source.shape(1) >
            impl_->maximum_context - pos0 ||
        state.local_.shape() != Shape{
            source.shape(0),
            checked_int(
                config.sliding_window,
                "sliding window"),
            head_dim,
        } ||
        state.main_.has_value() !=
            (config.is_v41()
                 ? config.is_kv_source(
                       static_cast<std::size_t>(impl_->layer))
                 : impl_->ratio != 0) ||
        state.indexer_.has_value() !=
            (config.is_v41()
                 ? config.is_kv_source(
                       static_cast<std::size_t>(impl_->layer))
                 : impl_->ratio == 4)) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 attention input/cache state");
    }
    const int batch = source.shape(0);
    const int tokens = source.shape(1);
    MlxDeepseekV4LayerSpeculation* speculation = nullptr;
    if (state.speculative_) {
        auto& transaction = *state.speculative_;
        if (transaction.local_kv ||
            transaction.start_position != pos0 ||
            transaction.total_tokens != tokens) {
            throw std::runtime_error(
                "DeepSeek-V4 speculative attention transaction mismatch");
        }
        speculation = &transaction;
    }
    if (visibility != nullptr &&
        (tokens == 1 || pos0 != 0 || visibility->max_image_tokens <= 0 ||
         visibility->left.shape() != Shape{batch, tokens} ||
         visibility->right.shape() != Shape{batch, tokens})) {
        throw std::invalid_argument(
            "DeepSeek-V4 image visibility requires one full prefill chunk");
    }
    if (config.is_v41()) {
        return impl_->forward_v41(
            source,
            state,
            pos0,
            visibility,
            debug_stages,
            shared);
    }
    if (shared != nullptr) {
        throw std::invalid_argument(
            "legacy DeepSeek-V4 attention received V4.1 shared state");
    }
    auto positions = mlx::core::arange(
        pos0,
        pos0 + tokens,
        1,
        mlx::core::int32);
    auto projected = impl_->project(source);
    const auto profile_component =
        [ratio = impl_->ratio](const char* component) {
            return std::string("attention.r")
                + std::to_string(ratio)
                + "." + component;
        };
    if (detail::component_profile_active()) {
        detail::profile_eval(
            profile_component("input_projections"),
            projected);
    }
    std::size_t output = 0;
    auto q_rank = impl_->q_norm(
        projected.at(output++));
    auto kv = projected.at(output++);
    if (debug_stages != nullptr) {
        debug_stages->push_back(source);
        debug_stages->push_back(projected.at(0));
        debug_stages->push_back(kv);
        debug_stages->push_back(q_rank);
    }
    if (detail::component_profile_active()) {
        detail::profile_eval(
            profile_component("q_rank_norm"),
            q_rank);
    }
    auto q_projected = mlx::core::reshape(
        impl_->components.q_b(q_rank),
        Shape{batch, tokens, heads, head_dim});
    if (debug_stages != nullptr) {
        debug_stages->push_back(q_projected);
    }
    if (detail::component_profile_active()) {
        detail::profile_eval(
            profile_component("q_b_projection"),
            q_projected);
    }
    const int rotary = checked_int(
        config.qk_rope_head_dim,
        "rotary dimension");
    auto cosine = mlx::core::take(
        impl_->rope.first,
        positions,
        0);
    auto sine = mlx::core::take(
        impl_->rope.second,
        positions,
        0);
    auto q_cosine = mlx::core::expand_dims(
        mlx::core::expand_dims(cosine, 0),
        2);
    auto q_sine = mlx::core::expand_dims(
        mlx::core::expand_dims(sine, 0),
        2);
    auto q = rms_replace_last_rope(
        q_projected,
        rotary,
        q_cosine,
        q_sine,
        impl_->rms_params,
        false,
        impl_->rms_params);
    kv = rms_replace_last_rope(
        kv,
        rotary,
        mlx::core::expand_dims(cosine, 0),
        mlx::core::expand_dims(sine, 0),
        impl_->components.kv_norm,
        true,
        impl_->rms_params);
    // The released V4F graph was trained with dynamic MXFP8 simulation on
    // the non-RoPE KV channels (64 values per UE8M0-scaled group).  Preserve
    // the RoPE channels verbatim and use this same fake-quantized KV for both
    // prefill and incremental cache writes.
    kv = deepseek_v4_kv_fp8_sim_prefix(kv, rotary);
    if (speculation != nullptr) {
        speculation->local_kv = kv;
    }
    if (debug_stages != nullptr) {
        debug_stages->push_back(q);
        debug_stages->push_back(kv);
    }
    if (detail::component_profile_active()) {
        detail::profile_eval(
            profile_component("q_rms_rope"),
            q);
        detail::profile_eval(
            profile_component("kv_norm_rope"),
            kv);
    }

    array index_weights = mlx::core::zeros(
        Shape{batch, tokens, 0},
        source.dtype());
    if (impl_->ratio != 0) {
        auto main_kv = projected.at(output++);
        auto main_gate = projected.at(output++);
        if (debug_stages != nullptr) {
            debug_stages->push_back(main_kv);
            debug_stages->push_back(main_gate);
        }
        array index_kv = mlx::core::zeros(
            Shape{1},
            source.dtype());
        array index_gate = mlx::core::zeros(
            Shape{1},
            source.dtype());
        if (impl_->ratio == 4) {
            index_kv = projected.at(output++);
            index_gate = projected.at(output++);
            index_weights = projected.at(output++);
            if (debug_stages != nullptr) {
                debug_stages->push_back(index_kv);
                debug_stages->push_back(index_gate);
                debug_stages->push_back(index_weights);
            }
        }
        if (speculation != nullptr) {
            speculation->main_kv = main_kv;
            speculation->main_gate = main_gate;
            if (impl_->ratio == 4) {
                speculation->index_kv = index_kv;
                speculation->index_gate = index_gate;
            }
        }
        const bool batch_prefill =
            tokens > 1 &&
            state.main_->remainder() == 0 &&
            (!state.indexer_ ||
             state.indexer_->remainder() == 0);
        if (batch_prefill) {
            state.main_->prefill(
                main_kv,
                main_gate,
                *impl_->components.main_ape,
                *impl_->components.main_norm,
                pos0,
                impl_->pool_rope->first,
                impl_->pool_rope->second,
                0,
                static_cast<float>(config.rms_eps));
            if (impl_->ratio == 4) {
                state.indexer_->prefill(
                    index_kv,
                    index_gate,
                    *impl_->components.index_ape,
                    *impl_->components.index_norm,
                    pos0,
                    impl_->pool_rope->first,
                    impl_->pool_rope->second,
                    config.fast_indexer() ? 2 : 0,
                    static_cast<float>(config.rms_eps));
            }
        } else {
            for (int token = 0; token < tokens; ++token) {
                const int length = pos0 + token + 1;
                state.main_->update(
                    slice_axis(
                        main_kv,
                        1,
                        token,
                        token + 1),
                    slice_axis(
                        main_gate,
                        1,
                        token,
                        token + 1),
                    *impl_->components.main_ape,
                    *impl_->components.main_norm,
                    length,
                    impl_->pool_rope->first,
                    impl_->pool_rope->second,
                    0,
                    static_cast<float>(config.rms_eps));
                if (impl_->ratio == 4) {
                    state.indexer_->update(
                        slice_axis(
                            index_kv,
                            1,
                            token,
                            token + 1),
                        slice_axis(
                            index_gate,
                            1,
                            token,
                            token + 1),
                        *impl_->components.index_ape,
                        *impl_->components.index_norm,
                        length,
                        impl_->pool_rope->first,
                        impl_->pool_rope->second,
                        config.fast_indexer() ? 2 : 0,
                        static_cast<float>(config.rms_eps));
                }
            }
        }
        if (state.indexer_ &&
            state.indexer_->pool_len() !=
                state.main_->pool_len()) {
            throw std::runtime_error(
                "DeepSeek-V4 main and Indexer "
                "pool lengths diverged");
        }
    }
    if (output != projected.size()) {
        throw std::runtime_error(
            "DeepSeek-V4 projection group output mismatch");
    }
    if (
        detail::component_profile_active()
        && impl_->ratio != 0
    ) {
        std::vector<array> compressor_state;
        const auto append_pool =
            [&compressor_state](
                const MlxDeepseekV4PoolState& pool) {
                compressor_state.push_back(pool.pool());
                compressor_state.push_back(pool.state_kv());
                compressor_state.push_back(pool.state_gate());
                if (pool.prev_kv()) {
                    compressor_state.push_back(*pool.prev_kv());
                }
                if (pool.prev_gate()) {
                    compressor_state.push_back(*pool.prev_gate());
                }
            };
        append_pool(*state.main_);
        if (state.indexer_) {
            append_pool(*state.indexer_);
        }
        detail::profile_eval(
            profile_component("compressor_update"),
            std::move(compressor_state));
    }

    auto topk = impl_->topk(
        q_rank,
        index_weights,
        state,
        positions,
        pos0);
    if (detail::component_profile_active()) {
        detail::profile_eval(
            profile_component("indexer_topk"),
            topk);
    }
    const int window = checked_int(
        config.sliding_window,
        "sliding window");
    const int pool_len =
        state.main_.has_value()
        ? state.main_->pool_len()
        : 0;
    array unified = state.local_;
    std::optional<std::pair<array, array>> plan;
    std::optional<array> direct_decode;
    // DSpark verifier rows must use the same selected-attention arithmetic as
    // ordinary one-token decoding.  Mixing the circular decode kernel here
    // with the selected verifier kernel changes a few final FP16 bits; those
    // differences compound through the transformer and lower MTP acceptance.
    // The selected plan is compact for decode (actual history, rounded to 32),
    // so this also avoids scanning the full sliding-window cache.
    if (tokens == 1 && !config.has_dspark()) {
        const int slot = pos0 % window;
        auto local_index = mlx::core::full(
            Shape{batch, 1},
            slot,
            mlx::core::int32);
        state.local_ = dsv4_cache_write_inplace(
            state.local_,
            mlx::core::astype(
                kv,
                state.local_.dtype()),
            local_index);
        unified = state.local_;
        if (config.fast_attention()) {
            std::optional<array> pooled;
            if (state.main_.has_value()) {
                pooled = state.main_->pool();
            }
            direct_decode = attention_dsv4_sparse_decode(
                mlx::core::transpose(
                    q,
                    {0, 2, 1, 3}),
                state.local_,
                pooled,
                pool_len,
                topk,
                impl_->components.sinks,
                pos0 + 1,
                impl_->ratio == 0
                    ? 1
                    : impl_->ratio,
                window);
        } else {
            if (state.main_.has_value()) {
                unified = mlx::core::concatenate(
                    {
                        state.local_,
                        pool_prefix(*state.main_),
                    },
                    1);
            }
            plan = dsv4_build_decode_plan(
                topk,
                mlx::core::full(
                    Shape{batch},
                    pos0 + 1,
                    mlx::core::int32),
                pool_len,
                impl_->ratio == 0
                    ? 1
                    : impl_->ratio,
                window);
        }
    } else {
        const int history = std::min(pos0, window);
        array history_values = slice_axis(
            state.local_,
            1,
            0,
            0);
        if (history != 0) {
            auto history_positions = mlx::core::arange(
                pos0 - history,
                pos0,
                1,
                mlx::core::int32);
            auto history_slots =
                mlx::core::remainder(
                    history_positions,
                    array(window, mlx::core::int32));
            history_values = mlx::core::take(
                state.local_,
                history_slots,
                1);
        }
        std::vector<array> parts{
            history_values,
            mlx::core::astype(
                kv,
                state.local_.dtype()),
        };
        if (state.main_.has_value()) {
            parts.push_back(
                pool_prefix(*state.main_));
        }
        unified = mlx::core::concatenate(
            std::move(parts),
            1);
        plan = visibility != nullptr
            ? dsv4_build_prefill_plan_visible(
                  topk,
                  visibility->left,
                  visibility->right,
                  pos0,
                  history,
                  pool_len,
                  impl_->ratio == 0 ? 1 : impl_->ratio,
                  window,
                  visibility->max_image_tokens)
            : dsv4_build_prefill_plan(
                  topk,
                  pos0,
                  history,
                  pool_len,
                  impl_->ratio == 0 ? 1 : impl_->ratio,
                  window);
        const int recent = std::min(tokens, window);
        auto recent_values = slice_axis(
            kv,
            1,
            tokens - recent,
            tokens);
        auto recent_positions = mlx::core::arange(
            pos0 + tokens - recent,
            pos0 + tokens,
            1,
            mlx::core::int32);
        auto recent_slots = mlx::core::remainder(
            recent_positions,
            array(window, mlx::core::int32));
        auto local_indices = mlx::core::broadcast_to(
            mlx::core::reshape(
                recent_slots,
                Shape{1, recent}),
            Shape{batch, recent});
        state.local_ = dsv4_cache_write_inplace(
            state.local_,
            mlx::core::astype(
                recent_values,
                state.local_.dtype()),
            local_indices);
    }

    array attended = direct_decode
        ? *direct_decode
        : config.fast_attention()
            ? attention_dsv4_sparse(
                  mlx::core::transpose(
                      q,
                      {0, 2, 1, 3}),
                  unified,
                  plan->first,
                  plan->second,
                  impl_->components.sinks)
            : generic_sparse_attention(
                  q,
                  unified,
                  plan->first,
                  plan->second,
                  impl_->components.sinks);
    if (debug_stages != nullptr) {
        debug_stages->push_back(attended);
    }
    if (detail::component_profile_active()) {
        if (direct_decode) {
            detail::profile_eval(
                profile_component("cache_update"),
                std::vector<array>{state.local_});
        } else {
            detail::profile_eval(
                profile_component("cache_plan"),
                std::vector<array>{
                    state.local_,
                    unified,
                    plan->first,
                    plan->second,
                });
        }
        detail::profile_eval(
            profile_component("sparse_core"),
            attended);
    }
    auto result = impl_->output_projection(
        attended,
        cosine,
        sine,
        debug_stages);
    state.position_ += tokens;
    return result;
}

void MlxDeepseekV4Attention::commit_speculative(
    MlxDeepseekV4LayerState& state) const noexcept {
    state.speculative_.reset();
}

void MlxDeepseekV4Attention::rollback_speculative(
    MlxDeepseekV4LayerState& state,
    int accepted_tokens) const {
    auto transaction = std::move(state.speculative_);
    state.speculative_.reset();
    if (!transaction || !transaction->local_kv) {
        throw std::runtime_error(
            "DeepSeek-V4 speculative attention checkpoint is unavailable");
    }
    const int speculative_tokens =
        transaction->total_tokens - transaction->confirmed_tokens;
    if (accepted_tokens < 0 || accepted_tokens > speculative_tokens) {
        throw std::invalid_argument(
            "DeepSeek-V4 speculative acceptance count is invalid");
    }
    const int keep = transaction->confirmed_tokens + accepted_tokens;
    const int start = transaction->start_position;
    state.restore_speculative_snapshot(
        std::move(transaction->checkpoint),
        start,
        transaction->total_tokens);
    if (transaction->local_kv->ndim() != 3 ||
        transaction->local_kv->shape(0) != state.batch() ||
        transaction->local_kv->shape(1) < keep) {
        throw std::runtime_error(
            "DeepSeek-V4 speculative KV capture is malformed");
    }

    // The verifier has already computed every cache projection. Restore the
    // checkpoint and commit the accepted prefix from those values directly;
    // replaying attention across 43 layers makes partial acceptance dominate
    // otherwise profitable MTP cycles.
    const int batch = state.batch();
    const int window = state.local_.shape(1);
    auto local_values = slice_axis(
        *transaction->local_kv, 1, 0, keep);
    auto local_positions = mlx::core::remainder(
        mlx::core::arange(
            start,
            start + keep,
            1,
            mlx::core::int32),
        array(window, mlx::core::int32));
    auto local_indices = mlx::core::broadcast_to(
        mlx::core::reshape(local_positions, Shape{1, keep}),
        Shape{batch, keep});
    state.local_ = dsv4_cache_write_inplace(
        state.local_,
        mlx::core::astype(local_values, state.local_.dtype()),
        local_indices);

    if (impl_->config.is_v41()) {
        const bool owns_kv = impl_->config.is_kv_source(
            static_cast<std::size_t>(impl_->layer));
        if (owns_kv) {
            if (!state.main_ || !state.indexer_ ||
                !transaction->main_kv ||
                transaction->main_kv->shape(1) < keep ||
                (impl_->ratio > 1 &&
                 (!transaction->main_gate ||
                  transaction->main_gate->shape(1) < keep))) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 speculative compressor capture is malformed");
            }
            const int first_row = state.main_->pool_len();
            std::optional<array> gate;
            if (impl_->ratio > 1) {
                gate = slice_axis(
                    *transaction->main_gate,
                    1,
                    0,
                    keep);
            }
            auto latent = state.main_->compress_v41(
                slice_axis(*transaction->main_kv, 1, 0, keep),
                gate,
                *impl_->components.main_norm,
                start,
                local_values.dtype(),
                static_cast<float>(impl_->config.rms_eps));
            const int emitted = latent.shape(1);
            if (emitted > 0) {
                auto rows = mlx::core::arange(
                    first_row,
                    first_row + emitted,
                    1,
                    mlx::core::int32);
                auto cosine = mlx::core::take(
                    impl_->pool_rope->first,
                    rows,
                    0);
                auto sine = mlx::core::take(
                    impl_->pool_rope->second,
                    rows,
                    0);
                auto index_key = weighted_rms(
                    (*impl_->components.index_key)(latent),
                    *impl_->components.index_key_norm,
                    local_values.dtype(),
                    static_cast<float>(impl_->config.rms_eps));
                index_key = replace_last_rope(
                    index_key,
                    checked_int(
                        impl_->config.qk_rope_head_dim,
                        "rotary dimension"),
                    mlx::core::expand_dims(cosine, 0),
                    mlx::core::expand_dims(sine, 0));
                state.indexer_->append_v41(
                    v41_activation_quant(index_key, 32, 1));
                auto compressed = replace_last_rope(
                    latent,
                    checked_int(
                        impl_->config.qk_rope_head_dim,
                        "rotary dimension"),
                    mlx::core::expand_dims(cosine, 0),
                    mlx::core::expand_dims(sine, 0));
                state.main_->append_v41(
                    v41_activation_quant(compressed, 16, 2));
            }
        }
        state.position_ = start + keep;
        return;
    }

    if (impl_->ratio != 0) {
        if (!state.main_ || !transaction->main_kv ||
            !transaction->main_gate ||
            transaction->main_kv->shape(1) < keep ||
            transaction->main_gate->shape(1) < keep ||
            (impl_->ratio == 4 &&
             (!state.indexer_ || !transaction->index_kv ||
              !transaction->index_gate ||
              transaction->index_kv->shape(1) < keep ||
              transaction->index_gate->shape(1) < keep))) {
            throw std::runtime_error(
                "DeepSeek-V4 speculative compressor capture is malformed");
        }
        for (int token = 0; token < keep; ++token) {
            const int length = start + token + 1;
            state.main_->update(
                slice_axis(*transaction->main_kv, 1, token, token + 1),
                slice_axis(*transaction->main_gate, 1, token, token + 1),
                *impl_->components.main_ape,
                *impl_->components.main_norm,
                length,
                impl_->pool_rope->first,
                impl_->pool_rope->second,
                0,
                static_cast<float>(impl_->config.rms_eps));
            if (impl_->ratio == 4) {
                state.indexer_->update(
                    slice_axis(*transaction->index_kv, 1, token, token + 1),
                    slice_axis(*transaction->index_gate, 1, token, token + 1),
                    *impl_->components.index_ape,
                    *impl_->components.index_norm,
                    length,
                    impl_->pool_rope->first,
                    impl_->pool_rope->second,
                    impl_->config.fast_indexer() ? 2 : 0,
                    static_cast<float>(impl_->config.rms_eps));
            }
        }
        if (state.indexer_ &&
            state.indexer_->pool_len() != state.main_->pool_len()) {
            throw std::runtime_error(
                "DeepSeek-V4 speculative pool lengths diverged");
        }
    }
    state.position_ = start + keep;
}

int MlxDeepseekV4Attention::ratio() const noexcept {
    return impl_->ratio;
}

int MlxDeepseekV4Attention::layer() const noexcept {
    return impl_->layer;
}

int MlxDeepseekV4Attention::max_context() const noexcept {
    return impl_->maximum_context;
}

} // namespace mfq::metal
