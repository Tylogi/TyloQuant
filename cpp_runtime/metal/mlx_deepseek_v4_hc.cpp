#include "mlx_deepseek_v4_hc.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
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
    device activation_t* reduced = (device activation_t*)packed;
    device float* metadata = (device float*)(packed + uint(REDUCED_BYTES));
    threadgroup float pre[CONNECTIONS];
    threadgroup float reductions[8];
    threadgroup float mix_inverse_shared;

    if (SCALE_MIXES != 0) {
        float square_sum = 0.0f;
        constexpr uint HC_WIDTH = CONNECTIONS * uint(HIDDEN);
        uint residual_base = row * HC_WIDTH;
        for (uint index = local_thread;
             index < HC_WIDTH;
             index += 256u) {
            float value = float(residual[residual_base + index]);
            square_sum += value * value;
        }
        float subtotal = simd_sum(square_sum);
        if (lane == 0u) {
            reductions[simd_group] = subtotal;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_thread == 0u) {
            float total = 0.0f;
            for (uint group = 0u; group < 8u; ++group) {
                total += reductions[group];
            }
            mix_inverse_shared = rsqrt(
                total / float(HC_WIDTH) + params[2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else if (local_thread == 0u) {
        mix_inverse_shared = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mix_inverse = mix_inverse_shared;

    if (simd_group == 0u) {
        uint mix_base = row * MIX_WIDTH;
        float active = lane < CONNECTIONS ? 1.0f : 0.0f;
        uint connection = min(lane, CONNECTIONS - 1u);
        float pre_mix =
            mixes[mix_base + connection] * mix_inverse;
        float post_mix =
            mixes[mix_base + CONNECTIONS + connection]
            * mix_inverse;
        float pre_affine =
            pre_mix * scale[0]
            + base[connection];
        float post_affine =
            post_mix * scale[1]
            + base[CONNECTIONS + connection];
        float pre_value =
            1.0f / (1.0f + metal::fast::exp(-pre_affine)) + params[0];
        float post_value =
            2.0f / (1.0f + metal::fast::exp(-post_affine));
        if (lane < CONNECTIONS) {
            pre[lane] = pre_value;
            metadata[row * 20u + lane] = post_value;
        }

        float4 normalized_values =
            *(const device float4*)(
                mixes + mix_base + 2u * CONNECTIONS
                    + connection * CONNECTIONS
            ) * mix_inverse;
        float4 values =
            (normalized_values * scale[2]
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
                metadata
                    + row * 20u + CONNECTIONS
                    + lane * CONNECTIONS
            ) = probabilities;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup collapsed_t collapsed[HIDDEN];
    constexpr uint HIDDEN4 = uint(HIDDEN) / 4u;
    const device activation4_t* x0 = (const device activation4_t*)(
        residual + (row * CONNECTIONS + 0u) * uint(HIDDEN)
    );
    const device activation4_t* x1 = (const device activation4_t*)(
        residual + (row * CONNECTIONS + 1u) * uint(HIDDEN)
    );
    const device activation4_t* x2 = (const device activation4_t*)(
        residual + (row * CONNECTIONS + 2u) * uint(HIDDEN)
    );
    const device activation4_t* x3 = (const device activation4_t*)(
        residual + (row * CONNECTIONS + 3u) * uint(HIDDEN)
    );
    device activation4_t* reduced4 = (device activation4_t*)(
        reduced + row * uint(HIDDEN));
    threadgroup collapsed4_t* collapsed4 =
        (threadgroup collapsed4_t*)collapsed;
    float square_sum = 0.0f;
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
        activation4_t rounded = activation4_t(value);
        if (NORMALIZE != 0) {
            float4 norm_value = float4(rounded);
            collapsed4[feature4] = collapsed4_t(norm_value);
            square_sum += dot(norm_value, norm_value);
        } else {
            reduced4[feature4] = rounded;
        }
    }

    if (NORMALIZE != 0) {
        float subtotal = simd_sum(square_sum);
        if (lane == 0u) {
            reductions[simd_group] = subtotal;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_thread == 0u) {
            float total = 0.0f;
            for (uint group = 0u; group < 8u; ++group) {
                total += reductions[group];
            }
            reductions[0] = rsqrt(
                total / float(HIDDEN) + params[1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inverse_rms = reductions[0];
        const device float4* norm4 =
            (const device float4*)norm;
        for (
            uint feature4 = local_thread;
            feature4 < HIDDEN4;
            feature4 += 256u
        ) {
            float4 value = float4(collapsed4[feature4]);
            reduced4[feature4] = activation4_t(
                value * inverse_rms * norm4[feature4]);
        }
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
        uint combination_index =
            (row * 4u + source) * 4u + destination;
        float mixing = PACKED_META != 0
            ? metadata[row * 20u + 4u + source * 4u + destination]
            : combination[combination_index];
        residual_sum += mixing * float(residual[
            (row * 4u + source) * uint(HIDDEN) + feature
        ]);
    }
    float branch_value = float(
        branch[row * uint(HIDDEN) + feature]
    );
    if (ADD_BRANCH != 0) {
        branch_value += float(
            branch2[row * uint(HIDDEN) + feature]
        );
    }
    float post_value = PACKED_META != 0
        ? metadata[row * 20u + destination]
        : post[row * 4u + destination];
    float direct = post_value
        * branch_value;
    output[index] = output_activation_t(direct + residual_sum);
)METAL";

mlx::core::fast::CustomKernelFunction make_kernel(
    const char* name,
    std::vector<std::string> inputs,
    std::vector<std::string> outputs,
    std::string source) {
    CompileOptions options;
    options.math_mode = mlx::core::MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        name,
        std::move(inputs),
        std::move(outputs),
        std::move(source),
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& hc_pre_kernel(
    mlx::core::Dtype dtype) {
    static const auto fp16_kernel = make_kernel(
        "mfq_cpp_dsv4_hc_pre_f16",
        {"residual", "mixes", "scale", "base", "norm", "params"},
        {"packed"},
        std::string(
            "using activation_t = half;\n"
            "using activation4_t = half4;\n"
            "using collapsed_t = half;\n"
            "using collapsed4_t = half4;\n") + kHcPreSource);
    static const auto bf16_kernel = make_kernel(
        "mfq_cpp_dsv4_hc_pre_bf16",
        {"residual", "mixes", "scale", "base", "norm", "params"},
        {"packed"},
        std::string(
            "using activation_t = bfloat;\n"
            "using activation4_t = bfloat4;\n"
            "using collapsed_t = bfloat;\n"
            "using collapsed4_t = bfloat4;\n") + kHcPreSource);
    return dtype == mlx::core::bfloat16
        ? bf16_kernel
        : fp16_kernel;
}

const mlx::core::fast::CustomKernelFunction& hc_post_kernel(
    mlx::core::Dtype dtype) {
    static const auto fp16_kernel = make_kernel(
        "mfq_cpp_dsv4_hc_post_f16",
        {
            "branch",
            "branch2",
            "residual",
            "post",
            "combination",
            "metadata",
        },
        {"output"},
        std::string("using output_activation_t = half;\n") +
            kHcPostSource);
    static const auto bf16_kernel = make_kernel(
        "mfq_cpp_dsv4_hc_post_bf16",
        {
            "branch",
            "branch2",
            "residual",
            "post",
            "combination",
            "metadata",
        },
        {"output"},
        std::string("using output_activation_t = bfloat;\n") +
            kHcPostSource);
    return dtype == mlx::core::bfloat16
        ? bf16_kernel
        : fp16_kernel;
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

array activation16_contiguous(const array& input) {
    auto result = input;
    if (result.dtype() != mlx::core::float16 &&
        result.dtype() != mlx::core::bfloat16) {
        result = mlx::core::astype(result, mlx::core::float16);
    }
    return mlx::core::contiguous(result);
}

array floating_contiguous(const array& input) {
    auto result = input;
    if (result.dtype() != mlx::core::float16 &&
        result.dtype() != mlx::core::bfloat16 &&
        result.dtype() != mlx::core::float32) {
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

array hc_post_impl(
    const array& branch,
    const array& branch2,
    const array& residual,
    const array& post,
    const array& combination,
    const std::optional<array>& packed_metadata,
    bool add_branch) {
    auto branch_values = floating_contiguous(branch);
    auto branch2_values = add_branch
        ? floating_contiguous(branch2)
        : branch_values;
    auto residual_values = activation16_contiguous(residual);
    auto metadata_values = packed_metadata.has_value()
        ? float32_contiguous(*packed_metadata)
        : float32_contiguous(post);
    auto post_values = packed_metadata.has_value()
        ? metadata_values
        : float32_contiguous(post);
    auto combination_values = packed_metadata.has_value()
        ? metadata_values
        : float32_contiguous(combination);
    if (branch_values.ndim() != 3 ||
        branch_values.shape(2) <= 0 ||
        branch2_values.shape() != branch_values.shape()) {
        throw std::invalid_argument(
            "DeepSeek-V4 HC branch must have matching "
            "[batch,tokens,hidden] shapes");
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
        (packed_metadata.has_value()
             ? metadata_values.shape() !=
                   Shape{batch, tokens, kMixWidth - kConnections}
             : post_values.shape() !=
                       Shape{batch, tokens, kConnections} ||
                   combination_values.shape() !=
                       Shape{
                           batch,
                           tokens,
                           kConnections,
                           kConnections,
                       })) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 HC post input");
    }
    const int size = checked_int(
        static_cast<std::size_t>(batch)
            * tokens * kConnections * hidden,
        "output size");
    auto outputs = hc_post_kernel(residual_values.dtype())(
        {
            branch_values,
            branch2_values,
            residual_values,
            post_values,
            combination_values,
            metadata_values,
        },
        {
            Shape{
                batch,
                tokens,
                kConnections,
                hidden,
            },
        },
        {residual_values.dtype()},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {
            {"SIZE", size},
            {"HIDDEN", hidden},
            {"ADD_BRANCH", static_cast<int>(add_branch)},
            {"PACKED_META", packed_metadata.has_value() ? 1 : 0},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

} // namespace

MlxDeepseekV4HcPreResult deepseek_v4_hc_pre(
    const array& residual,
    const array& mixes,
    const array& scale,
    const array& base,
    int sinkhorn_iterations,
    float eps) {
    auto residual_values = activation16_contiguous(residual);
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
    const auto reduced_dtype = residual_values.dtype();
    const std::size_t reduced_item_size = sizeof(std::uint16_t);
    const int reduced_bytes = checked_int(
        static_cast<std::size_t>(rows) * hidden * reduced_item_size,
        "packed reduced bytes");
    const int packed_bytes = checked_int(
        static_cast<std::size_t>(reduced_bytes) +
            static_cast<std::size_t>(rows) *
                (kMixWidth - kConnections) * sizeof(float),
        "packed output bytes");
    const array params({eps, eps, eps}, mlx::core::float32);
    auto outputs = hc_pre_kernel(residual_values.dtype())(
        {
            residual_values,
            mix_values,
            scale_values,
            base_values,
            base_values,
            params,
        },
        {
            Shape{packed_bytes},
        },
        {
            mlx::core::uint8,
        },
        {rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        {
            {"ROWS", rows},
            {"HIDDEN", hidden},
            {"REDUCED_BYTES", reduced_bytes},
            {"NORMALIZE", 0},
            {"SCALE_MIXES", 0},
        },
        std::nullopt,
        false,
        {});
    auto packed = std::move(outputs.at(0));
    auto reduced = mlx::core::reshape(
        mlx::core::view(
            mlx::core::slice(
                packed,
                Shape{0},
                Shape{reduced_bytes}),
            reduced_dtype),
        Shape{batch, tokens, hidden});
    auto metadata = mlx::core::reshape(
        mlx::core::view(
            mlx::core::slice(
                packed,
                Shape{reduced_bytes},
                Shape{packed_bytes}),
            mlx::core::float32),
        Shape{batch, tokens, kMixWidth - kConnections});
    auto post = mlx::core::contiguous(mlx::core::slice(
        metadata,
        Shape{0, 0, 0},
        Shape{batch, tokens, kConnections}));
    auto combination = mlx::core::reshape(
        mlx::core::contiguous(mlx::core::slice(
            metadata,
            Shape{0, 0, kConnections},
            Shape{batch, tokens, kMixWidth - kConnections})),
        Shape{batch, tokens, kConnections, kConnections});
    return {
        std::move(reduced),
        std::move(post),
        std::move(combination),
        std::move(metadata),
    };
}

MlxDeepseekV4HcPreResult deepseek_v4_hc_pre_norm(
    const array& residual,
    const array& mixes,
    const array& scale,
    const array& base,
    const array& norm,
    int sinkhorn_iterations,
    float hc_eps,
    float norm_eps,
    bool normalize_mixes_from_residual) {
    auto residual_values = activation16_contiguous(residual);
    auto mix_values = float32_contiguous(mixes);
    auto scale_values = float32_contiguous(scale);
    auto base_values = float32_contiguous(base);
    auto norm_values = float32_contiguous(norm);
    if (residual_values.ndim() != 4 ||
        residual_values.shape(2) != kConnections ||
        residual_values.shape(3) <= 0 ||
        residual_values.shape(3) % 4 != 0) {
        throw std::invalid_argument(
            "DeepSeek-V4 fused HC/Norm residual must have "
            "[batch,tokens,4,hidden] shape with hidden divisible by four");
    }
    const int batch = residual_values.shape(0);
    const int tokens = residual_values.shape(1);
    const int hidden = residual_values.shape(3);
    if (batch <= 0 || tokens <= 0 ||
        mix_values.shape() != Shape{batch, tokens, kMixWidth} ||
        scale_values.size() != 3 ||
        base_values.size() != kMixWidth ||
        norm_values.shape() != Shape{hidden} ||
        sinkhorn_iterations != kSinkhornIterations ||
        !std::isfinite(hc_eps) || hc_eps <= 0.0f ||
        !std::isfinite(norm_eps) || norm_eps <= 0.0f) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 fused HC/Norm input");
    }
    scale_values = mlx::core::reshape(scale_values, Shape{3});
    base_values = mlx::core::reshape(
        base_values,
        Shape{kMixWidth});
    norm_values = mlx::core::reshape(
        norm_values,
        Shape{hidden});
    const int rows = checked_int(
        static_cast<std::size_t>(batch) * tokens,
        "row count");
    const auto reduced_dtype = residual_values.dtype();
    const std::size_t reduced_item_size = sizeof(std::uint16_t);
    const int reduced_bytes = checked_int(
        static_cast<std::size_t>(rows) * hidden * reduced_item_size,
        "packed reduced bytes");
    const int packed_bytes = checked_int(
        static_cast<std::size_t>(reduced_bytes) +
            static_cast<std::size_t>(rows) *
                (kMixWidth - kConnections) * sizeof(float),
        "packed output bytes");
    const array params(
        {hc_eps, norm_eps, norm_eps},
        mlx::core::float32);
    auto outputs = hc_pre_kernel(residual_values.dtype())(
        {
            residual_values,
            mix_values,
            scale_values,
            base_values,
            norm_values,
            params,
        },
        {
            Shape{packed_bytes},
        },
        {
            mlx::core::uint8,
        },
        {rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        {
            {"ROWS", rows},
            {"HIDDEN", hidden},
            {"REDUCED_BYTES", reduced_bytes},
            {"NORMALIZE", 1},
            {
                "SCALE_MIXES",
                normalize_mixes_from_residual ? 1 : 0,
            },
        },
        std::nullopt,
        false,
        {});
    auto packed = std::move(outputs.at(0));
    auto reduced = mlx::core::reshape(
        mlx::core::view(
            mlx::core::slice(
                packed,
                Shape{0},
                Shape{reduced_bytes}),
            reduced_dtype),
        Shape{batch, tokens, hidden});
    auto metadata = mlx::core::reshape(
        mlx::core::view(
            mlx::core::slice(
                packed,
                Shape{reduced_bytes},
                Shape{packed_bytes}),
            mlx::core::float32),
        Shape{batch, tokens, kMixWidth - kConnections});
    auto post = mlx::core::contiguous(mlx::core::slice(
        metadata,
        Shape{0, 0, 0},
        Shape{batch, tokens, kConnections}));
    auto combination = mlx::core::reshape(
        mlx::core::contiguous(mlx::core::slice(
            metadata,
            Shape{0, 0, kConnections},
            Shape{batch, tokens, kMixWidth - kConnections})),
        Shape{batch, tokens, kConnections, kConnections});
    return {
        std::move(reduced),
        std::move(post),
        std::move(combination),
        std::move(metadata),
    };
}

array deepseek_v4_hc_post(
    const array& branch,
    const array& residual,
    const array& post,
    const array& combination) {
    return hc_post_impl(
        branch,
        branch,
        residual,
        post,
        combination,
        std::nullopt,
        false);
}

array deepseek_v4_hc_post_sum(
    const array& routed,
    const array& shared,
    const array& residual,
    const array& post,
    const array& combination) {
    return hc_post_impl(
        routed,
        shared,
        residual,
        post,
        combination,
        std::nullopt,
        true);
}

array deepseek_v4_hc_post_packed(
    const array& branch,
    const array& residual,
    const array& metadata) {
    return hc_post_impl(
        branch,
        branch,
        residual,
        metadata,
        metadata,
        metadata,
        false);
}

array deepseek_v4_hc_post_sum_packed(
    const array& routed,
    const array& shared,
    const array& residual,
    const array& metadata) {
    return hc_post_impl(
        routed,
        shared,
        residual,
        metadata,
        metadata,
        metadata,
        true);
}

} // namespace mfq::metal
