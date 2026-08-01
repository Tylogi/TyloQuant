#include "mlx_deepseek_v4_hc.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kConnections = 4;
constexpr int kMixWidth =
    2 * kConnections + kConnections * kConnections;
constexpr int kThreads = 256;
constexpr int kSinkhornIterations = 20;

constexpr const char* kHcPreSource = R"METAL(
    constexpr uint CONNECTIONS = 4u;
    constexpr uint MIX_WIDTH = 24u;
    uint row = threadgroup_position_in_grid.x;
    uint local_thread = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float pre[CONNECTIONS];

    if (simd_group == 0u) {
        uint mix_base = row * MIX_WIDTH;
        float active = lane < CONNECTIONS ? 1.0f : 0.0f;
        uint connection = min(lane, CONNECTIONS - 1u);
        float pre_affine =
            mixes[mix_base + connection] * scale[0]
            + base[connection];
        float post_affine =
            mixes[mix_base + CONNECTIONS + connection] * scale[1]
            + base[CONNECTIONS + connection];
        float pre_value =
            1.0f / (1.0f + metal::fast::exp(-pre_affine)) + params[0];
        float post_value =
            2.0f / (1.0f + metal::fast::exp(-post_affine));
        if (lane < CONNECTIONS) {
            pre[lane] = pre_value;
            post[row * CONNECTIONS + lane] = post_value;
        }

        float4 values =
            (*(const device float4*)(
                mixes + mix_base + 2u * CONNECTIONS
                    + connection * CONNECTIONS
            ) * scale[2]
            + *(const device float4*)(
                base + 2u * CONNECTIONS
                    + connection * CONNECTIONS
            )) * active;
        float maximum = max(
            max(values.x, values.y),
            max(values.z, values.w)
        );
        float4 probabilities =
            metal::fast::exp(values - maximum) * active;
        probabilities =
            probabilities / (
                probabilities.x + probabilities.y
                    + probabilities.z + probabilities.w
                    + params[0]
            )
            + params[0] * active;
        probabilities /= float4(
            simd_sum(probabilities.x),
            simd_sum(probabilities.y),
            simd_sum(probabilities.z),
            simd_sum(probabilities.w)
        ) + params[0];

        for (uint iteration = 1u; iteration < 20u; ++iteration) {
            probabilities *= (
                active / (
                    probabilities.x + probabilities.y
                        + probabilities.z + probabilities.w
                        + params[0]
                )
            );
            probabilities /= float4(
                simd_sum(probabilities.x),
                simd_sum(probabilities.y),
                simd_sum(probabilities.z),
                simd_sum(probabilities.w)
            ) + params[0];
        }
        if (lane < CONNECTIONS) {
            *(device float4*)(
                combination
                    + row * CONNECTIONS * CONNECTIONS
                    + lane * CONNECTIONS
            ) = probabilities;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint HIDDEN4 = uint(HIDDEN) / 4u;
    const device half4* x0 = (const device half4*)(
        residual + (row * CONNECTIONS + 0u) * uint(HIDDEN)
    );
    const device half4* x1 = (const device half4*)(
        residual + (row * CONNECTIONS + 1u) * uint(HIDDEN)
    );
    const device half4* x2 = (const device half4*)(
        residual + (row * CONNECTIONS + 2u) * uint(HIDDEN)
    );
    const device half4* x3 = (const device half4*)(
        residual + (row * CONNECTIONS + 3u) * uint(HIDDEN)
    );
    device half4* reduced4 =
        (device half4*)(reduced + row * uint(HIDDEN));
    for (
        uint feature4 = local_thread;
        feature4 < HIDDEN4;
        feature4 += 256u
    ) {
        float4 value = fma(
            float4(pre[0]), float4(x0[feature4]),
            fma(
                float4(pre[1]), float4(x1[feature4]),
                fma(
                    float4(pre[2]), float4(x2[feature4]),
                    float4(pre[3]) * float4(x3[feature4])
                )
            )
        );
        reduced4[feature4] = half4(value);
    }
)METAL";

constexpr const char* kHcPostSource = R"METAL(
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }
    uint feature = index % uint(HIDDEN);
    uint destination_row = index / uint(HIDDEN);
    uint destination = destination_row % 4u;
    uint row = destination_row / 4u;
    float residual_sum = 0.0f;
    for (uint source = 0u; source < 4u; ++source) {
        residual_sum += combination[
            (row * 4u + source) * 4u + destination
        ] * float(residual[
            (row * 4u + source) * uint(HIDDEN) + feature
        ]);
    }
    float direct = post[row * 4u + destination]
        * float(branch[row * uint(HIDDEN) + feature]);
    output[index] = half(direct + residual_sum);
)METAL";

mlx::core::fast::CustomKernelFunction make_kernel(
    const char* name,
    std::vector<std::string> inputs,
    std::vector<std::string> outputs,
    const char* source) {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        name,
        std::move(inputs),
        std::move(outputs),
        source,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& hc_pre_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_hc_pre",
        {"residual", "mixes", "scale", "base", "params"},
        {"reduced", "post", "combination"},
        kHcPreSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& hc_post_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_hc_post",
        {"branch", "residual", "post", "combination"},
        {"output"},
        kHcPostSource);
    return kernel;
}

int checked_int(std::size_t value, const char* name) {
    if (value >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4 HC ") + name
            + " exceeds MLX limits");
    }
    return static_cast<int>(value);
}

array float16_contiguous(const array& input) {
    auto result = input;
    if (result.dtype() != mlx::core::float16) {
        result = mlx::core::astype(result, mlx::core::float16);
    }
    return mlx::core::contiguous(result);
}

array float32_contiguous(const array& input) {
    auto result = input;
    if (result.dtype() != mlx::core::float32) {
        result = mlx::core::astype(result, mlx::core::float32);
    }
    return mlx::core::contiguous(result);
}

} // namespace

MlxDeepseekV4HcPreResult deepseek_v4_hc_pre(
    const array& residual,
    const array& mixes,
    const array& scale,
    const array& base,
    int sinkhorn_iterations,
    float eps) {
    auto residual_values = float16_contiguous(residual);
    auto mix_values = float32_contiguous(mixes);
    auto scale_values = float32_contiguous(scale);
    auto base_values = float32_contiguous(base);
    if (residual_values.ndim() != 4 ||
        residual_values.shape(2) != kConnections ||
        residual_values.shape(3) <= 0 ||
        residual_values.shape(3) % 4 != 0) {
        throw std::invalid_argument(
            "DeepSeek-V4 HC residual must have "
            "[batch,tokens,4,hidden] shape with hidden divisible by four");
    }
    const int batch = residual_values.shape(0);
    const int tokens = residual_values.shape(1);
    const int hidden = residual_values.shape(3);
    if (batch <= 0 || tokens <= 0 ||
        mix_values.shape() != Shape{batch, tokens, kMixWidth} ||
        scale_values.size() != 3 ||
        base_values.size() != kMixWidth ||
        sinkhorn_iterations != kSinkhornIterations ||
        !std::isfinite(eps) || eps <= 0.0f) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 HC pre input");
    }
    scale_values = mlx::core::reshape(scale_values, Shape{3});
    base_values = mlx::core::reshape(
        base_values,
        Shape{kMixWidth});
    const int rows = checked_int(
        static_cast<std::size_t>(batch) * tokens,
        "row count");
    const array params({eps}, mlx::core::float32);
    auto outputs = hc_pre_kernel()(
        {
            residual_values,
            mix_values,
            scale_values,
            base_values,
            params,
        },
        {
            Shape{batch, tokens, hidden},
            Shape{batch, tokens, kConnections},
            Shape{
                batch,
                tokens,
                kConnections,
                kConnections,
            },
        },
        {
            mlx::core::float16,
            mlx::core::float32,
            mlx::core::float32,
        },
        {rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        {
            {"ROWS", rows},
            {"HIDDEN", hidden},
        },
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs.at(0)),
        std::move(outputs.at(1)),
        std::move(outputs.at(2)),
    };
}

array deepseek_v4_hc_post(
    const array& branch,
    const array& residual,
    const array& post,
    const array& combination) {
    auto branch_values = float16_contiguous(branch);
    auto residual_values = float16_contiguous(residual);
    auto post_values = float32_contiguous(post);
    auto combination_values =
        float32_contiguous(combination);
    if (branch_values.ndim() != 3 ||
        branch_values.shape(2) <= 0) {
        throw std::invalid_argument(
            "DeepSeek-V4 HC branch must have "
            "[batch,tokens,hidden] shape");
    }
    const int batch = branch_values.shape(0);
    const int tokens = branch_values.shape(1);
    const int hidden = branch_values.shape(2);
    if (batch <= 0 || tokens <= 0 ||
        residual_values.shape() !=
            Shape{
                batch,
                tokens,
                kConnections,
                hidden,
            } ||
        post_values.shape() !=
            Shape{batch, tokens, kConnections} ||
        combination_values.shape() !=
            Shape{
                batch,
                tokens,
                kConnections,
                kConnections,
            }) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 HC post input");
    }
    const int size = checked_int(
        static_cast<std::size_t>(batch)
            * tokens * kConnections * hidden,
        "output size");
    auto outputs = hc_post_kernel()(
        {
            branch_values,
            residual_values,
            post_values,
            combination_values,
        },
        {
            Shape{
                batch,
                tokens,
                kConnections,
                hidden,
            },
        },
        {mlx::core::float16},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {
            {"SIZE", size},
            {"HIDDEN", hidden},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

} // namespace mfq::metal
