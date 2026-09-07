#include "mfq_container.h"
#include "mlx_tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace {

using Clock = std::chrono::steady_clock;
using mlx::core::CompileOptions;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MfqContainer;
using mfq::metal::MlxLinear;

constexpr const char* kDenseSmallM = R"METAL(
    constexpr uint K_LANES_VALUE = uint(K_LANES);
    constexpr uint SIMD_GROUPS_VALUE = uint(SIMD_GROUPS);
    constexpr uint OUTPUTS_PER_SIMD = 32u / K_LANES_VALUE;
    constexpr uint OUTPUTS_PER_TG =
        SIMD_GROUPS_VALUE * OUTPUTS_PER_SIMD;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES_VALUE - 1u);
    uint simd_output = lane / K_LANES_VALUE;
    uint output_index =
        threadgroup_position_in_grid.y * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD + simd_output;
    uint output = min(output_index, uint(OUT) - 1u);

    float accumulators[M];
    for (uint row = 0u; row < uint(M); ++row) {
        accumulators[row] = 0.0f;
    }

    uint weight_base = output * uint(K);
    for (uint vector = k_lane;
         vector < uint(K) / 4u;
         vector += K_LANES_VALUE) {
        uint column = vector * 4u;
        vec<T, 4> packed_weight = *(device const vec<T, 4>*)(
            weight + weight_base + column);
        float4 weight_values = float4(packed_weight);
        for (uint row = 0u; row < uint(M); ++row) {
            vec<T, 4> packed_input = *(device const vec<T, 4>*)(
                x + row * uint(K) + column);
            accumulators[row] += dot(float4(packed_input), weight_values);
        }
    }

    for (uint row = 0u; row < uint(M); ++row) {
        if (K_LANES_VALUE >= 32u) {
            accumulators[row] += simd_shuffle_down(accumulators[row], 16u);
        }
        if (K_LANES_VALUE >= 16u) {
            accumulators[row] += simd_shuffle_down(accumulators[row], 8u);
        }
        if (K_LANES_VALUE >= 8u) {
            accumulators[row] += simd_shuffle_down(accumulators[row], 4u);
        }
        accumulators[row] += simd_shuffle_down(accumulators[row], 2u);
        accumulators[row] += simd_shuffle_down(accumulators[row], 1u);
        if (k_lane == 0u && output_index < uint(OUT)) {
            y[row * uint(OUT) + output_index] = T(accumulators[row]);
        }
    }
)METAL";

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

const mlx::core::fast::CustomKernelFunction& dense_small_m_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_dense_small_m_benchmark",
            {"weight", "x"},
            {"y"},
            kDenseSmallM,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

array make_input(int rows, int width, mlx::core::Dtype dtype) {
    std::vector<float> values(
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(width));
    for (std::size_t index = 0; index < values.size(); ++index) {
        const int residue = static_cast<int>((index * 17 + 11) % 127);
        values[index] = static_cast<float>(residue - 63) / 256.0f;
    }
    return mlx::core::astype(
        array(values.begin(), Shape{rows, width}), dtype);
}

array candidate(
    const array& weight,
    const array& input,
    int k_lanes,
    int simd_groups) {
    const int rows = input.shape(0);
    const int input_size = input.shape(1);
    const int output_size = weight.shape(0);
    const int outputs_per_threadgroup = simd_groups * 32 / k_lanes;
    auto outputs = dense_small_m_kernel()(
        {weight, input},
        {Shape{rows, output_size}},
        {weight.dtype()},
        {
            simd_groups * 32,
            (output_size + outputs_per_threadgroup - 1) /
                outputs_per_threadgroup,
            1,
        },
        {simd_groups * 32, 1, 1},
        {
            {"T", weight.dtype()},
            {"M", rows},
            {"K", input_size},
            {"OUT", output_size},
            {"K_LANES", k_lanes},
            {"SIMD_GROUPS", simd_groups},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

float max_difference(const array& left, const array& right) {
    auto left32 = mlx::core::astype(left, mlx::core::float32);
    auto right32 = mlx::core::astype(right, mlx::core::float32);
    mlx::core::eval(left32, right32);
    float maximum = 0.0f;
    for (std::size_t index = 0; index < left32.size(); ++index) {
        maximum = std::max(
            maximum,
            std::abs(left32.data<float>()[index] - right32.data<float>()[index]));
    }
    return maximum;
}

template <typename Projection>
double measure(const Projection& projection, int repetitions) {
    auto result = projection();
    mlx::core::eval(result);
    for (int index = 0; index < 4; ++index) {
        result = projection();
        mlx::core::eval(result);
    }
    mlx::core::synchronize();
    const auto started = Clock::now();
    for (int index = 0; index < repetitions; ++index) {
        result = projection();
        mlx::core::eval(result);
    }
    mlx::core::synchronize();
    return std::chrono::duration<double, std::milli>(Clock::now() - started)
        .count() / static_cast<double>(repetitions);
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(
            argc == 4 || argc == 5,
            "usage: mfq-metal-dense-smallm-benchmark MODEL.mfq TENSOR REPETITIONS [F16]");
        require(
            argc != 5 || std::string(argv[4]) == "F16",
            "optional dtype override must be F16");
        const int repetitions = std::stoi(argv[3]);
        require(repetitions > 0, "repetitions must be positive");
        const MfqContainer model(argv[1]);
        const auto loaded = MlxLinear::load(model, argv[2]);
        const auto* loaded_weight = loaded.dense_weight_ref();
        require(loaded_weight != nullptr, "selected tensor is not dense");
        auto dense = argc == 5 && std::string(argv[4]) == "F16"
            ? mlx::core::astype(*loaded_weight, mlx::core::float16)
            : *loaded_weight;
        dense.eval();
        const MlxLinear linear(dense);
        const auto* weight = linear.dense_weight_ref();
        require(weight->shape(1) % 4 == 0, "input width must be divisible by four");

        std::cout << "dtype\tM\tmethod\tms\tGB/s\tmax_abs\n";
        for (int rows = 2; rows <= 6; ++rows) {
            const auto input = make_input(rows, linear.input_size(), weight->dtype());
            const auto stock = [&] {
                return mlx::core::matmul(input, mlx::core::transpose(*weight));
            };
            auto reference = stock();
            auto runtime = linear(input);
            auto candidate_8x2 = candidate(*weight, input, 8, 2);
            auto candidate_16x4 = candidate(*weight, input, 16, 4);
            auto candidate_32x4 = candidate(*weight, input, 32, 4);
            auto candidate_32x8 = candidate(*weight, input, 32, 8);
            mlx::core::eval(
                reference,
                runtime,
                candidate_8x2,
                candidate_16x4,
                candidate_32x4,
                candidate_32x8);
            const float runtime_difference = max_difference(reference, runtime);
            const float difference_8x2 = max_difference(reference, candidate_8x2);
            const float difference_16x4 = max_difference(reference, candidate_16x4);
            const float difference_32x4 = max_difference(reference, candidate_32x4);
            const float difference_32x8 = max_difference(reference, candidate_32x8);
            for (int trial = 0; trial < 3; ++trial) {
                const double stock_ms = measure(stock, repetitions);
                const double runtime_ms = measure(
                    [&] { return linear(input); }, repetitions);
                const double custom_8x2_ms = measure(
                    [&] { return candidate(*weight, input, 8, 2); }, repetitions);
                const double custom_16x4_ms = measure(
                    [&] { return candidate(*weight, input, 16, 4); }, repetitions);
                const double custom_32x4_ms = measure(
                    [&] { return candidate(*weight, input, 32, 4); }, repetitions);
                const double custom_32x8_ms = measure(
                    [&] { return candidate(*weight, input, 32, 8); }, repetitions);
                const auto print = [&](const char* method, double ms, float difference) {
                    std::cout << weight->dtype() << '\t' << rows << '\t' << method << '\t'
                              << std::fixed << std::setprecision(4) << ms << '\t'
                              << std::setprecision(1)
                              << static_cast<double>(weight->nbytes()) / (ms * 1.0e6)
                              << '\t' << std::setprecision(6) << difference << '\n';
                };
                print("mlx", stock_ms, 0.0f);
                print("runtime", runtime_ms, runtime_difference);
                print("custom_8x2", custom_8x2_ms, difference_8x2);
                print("custom_16x4", custom_16x4_ms, difference_16x4);
                print("custom_32x4", custom_32x4_ms, difference_32x4);
                print("custom_32x8", custom_32x8_ms, difference_32x8);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MFQ dense small-M benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
