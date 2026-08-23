#include "mlx_moe_ops.h"

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

constexpr int kThreads = 256;
constexpr int kMaximumExperts = 4096;
constexpr int kMaximumRoutes = 16;

constexpr const char* kTopKSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float transformed[EXPERTS];
    threadgroup float partial[256];
    uint row_offset = row * uint(EXPERTS);

    float local_max = -INFINITY;
    if (MODE == 0) {
        for (uint expert = tid; expert < uint(EXPERTS); expert += 256u) {
            float raw = float(logits[row_offset + expert]);
            raw = isnan(raw) ? -FLT_MAX : raw;
            local_max = max(local_max, raw);
        }
        partial[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] = max(partial[tid], partial[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    float maximum = partial[0];

    float local_sum = 0.0f;
    for (uint expert = tid; expert < uint(EXPERTS); expert += 256u) {
        float raw = float(logits[row_offset + expert]);
        raw = isnan(raw) ? -FLT_MAX : raw;
        float value;
        if (MODE == 0) {
            value = exp(raw - maximum);
            local_sum += value;
        } else if (MODE == 1) {
            value = 1.0f / (1.0f + exp(-raw));
        } else if (MODE == 2) {
            float softplus = raw > 20.0f ? raw : log1p(exp(raw));
            value = sqrt(softplus);
        } else {
            value = raw;
        }
        transformed[expert] = value;
    }
    if (MODE == 0) {
        partial[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float denominator = partial[0];
        for (uint expert = tid; expert < uint(EXPERTS); expert += 256u) {
            transformed[expert] /= denominator;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float selected_weights[TOP_K];
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            float best_score = -INFINITY;
            uint best_expert = uint(EXPERTS);
            for (uint expert = 0u; expert < uint(EXPERTS); ++expert) {
                float weight = transformed[expert];
                float score = (
                    HAS_AVAILABLE == 0 || available[expert]
                )
                    ? weight + (HAS_BIAS != 0 ? bias[expert] : 0.0f)
                    : -INFINITY;
                if (
                    score > best_score
                    || (score == best_score && expert < best_expert)
                ) {
                    best_score = score;
                    best_expert = expert;
                }
            }
            uint output_index = row * uint(TOP_K) + rank;
            ids[output_index] = int(best_expert);
            selected_weights[rank] = transformed[best_expert];
            transformed[best_expert] = -INFINITY;
        }

        float denominator = 1.0f;
        if (MODE == 3) {
            float selected_max = -INFINITY;
            for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
                selected_max = max(selected_max, selected_weights[rank]);
            }
            denominator = 0.0f;
            for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
                selected_weights[rank] = exp(
                    selected_weights[rank] - selected_max
                );
                denominator += selected_weights[rank];
            }
        } else if (NORMALIZE != 0) {
            denominator = 0.0f;
            for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
                denominator += selected_weights[rank];
            }
            denominator = max(denominator, params[0]);
        }
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            float value = selected_weights[rank];
            if (MODE == 3 || NORMALIZE != 0) {
                value /= denominator;
            }
            weights[row * uint(TOP_K) + rank] = value * params[1];
        }
    }
)METAL";

constexpr const char* kDenseRouterTopKSource = R"METAL(
    constexpr uint EXPERTS = 256u;
    constexpr uint TOP_K = 6u;
    constexpr uint SIMD_GROUPS = 32u;

    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    threadgroup float route_weights[EXPERTS];
    threadgroup float scores[EXPERTS];

    // One SIMD group consumes one row at a time.  Eight rounds cover all 256
    // experts while every weight and activation load remains contiguous.
    for (uint round = 0u; round < 8u; ++round) {
        uint expert = round * SIMD_GROUPS + simd_group;
        uint weight_base = expert * uint(K);
        float accumulator = 0.0f;
        for (
            uint column = lane * 4u;
            column < uint(K);
            column += 128u
        ) {
            float4 activation = float4(
                *(device const activation4_t*)(input + column));
            float4 values = float4(
                *(device const activation4_t*)(
                    weight + weight_base + column));
            accumulator += dot(activation, values);
        }
        float raw = simd_sum(accumulator);
        if (lane == 0u) {
            // DeepSeek-V4 routes from x.float() @ weight.float().  Keep the
            // SIMD reduction in FP32: rounding router logits to FP16 here can
            // change the discrete top-k expert set.
            raw = isnan(raw) ? -FLT_MAX : raw;
            float softplus = raw > 20.0f ? raw : log1p(exp(raw));
            float route_weight = sqrt(softplus);
            route_weights[expert] = route_weight;
            scores[expert] = (
                HAS_AVAILABLE == 0 || available[expert]
            )
                ? route_weight
                    + (HAS_BIAS != 0 ? bias[expert] : 0.0f)
                : -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float selected_weights[TOP_K];
        for (uint rank = 0u; rank < TOP_K; ++rank) {
            float best_score = -INFINITY;
            uint best_expert = EXPERTS;
            for (uint expert = 0u; expert < EXPERTS; ++expert) {
                float score = scores[expert];
                if (
                    score > best_score
                    || (score == best_score && expert < best_expert)
                ) {
                    best_score = score;
                    best_expert = expert;
                }
            }
            ids[rank] = int(best_expert);
            selected_weights[rank] = route_weights[best_expert];
            scores[best_expert] = -INFINITY;
        }
        float denominator = 0.0f;
        for (uint rank = 0u; rank < TOP_K; ++rank) {
            denominator += selected_weights[rank];
        }
        denominator = max(denominator, params[0]);
        for (uint rank = 0u; rank < TOP_K; ++rank) {
            weights[rank] =
                selected_weights[rank] / denominator * params[1];
        }
    }
)METAL";

constexpr const char* kSqrtSoftplusSource = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x >> 5;
    if (row >= uint(ROWS)) {
        return;
    }
    float value = 0.0f;
    if (lane < uint(TOP_K)) {
        uint expert = uint(ids[row * uint(TOP_K) + lane]);
        float raw = float(logits[row * uint(EXPERTS) + expert]);
        float softplus = raw > 20.0f ? raw : log1p(exp(raw));
        value = sqrt(softplus);
    }
    float denominator = NORMALIZE != 0
        ? max(simd_sum(value), params[0])
        : 1.0f;
    if (lane < uint(TOP_K)) {
        weights[row * uint(TOP_K) + lane] =
            value / denominator * params[1];
    }
)METAL";

constexpr const char* kRepairHashIdsSource = R"METAL(
    uint row = thread_position_in_grid.x;
    if (row >= uint(ROWS)) {
        return;
    }
    uint output_base = row * uint(TOP_K);
    uint candidate_base = row * uint(CANDIDATES);
    for (uint route = 0u; route < uint(TOP_K); ++route) {
        result[output_base + route] =
            static_ids[output_base + route];
    }
    for (uint route = 0u; route < uint(TOP_K); ++route) {
        int current = result[output_base + route];
        bool bad = current < 0
            || current >= int(EXPERTS)
            || !available[uint(max(current, 0))];
        if (!bad) {
            continue;
        }
        int replacement = -1;
        for (
            uint candidate_slot = 0u;
            candidate_slot < uint(CANDIDATES);
            ++candidate_slot
        ) {
            int candidate =
                candidate_ids[candidate_base + candidate_slot];
            if (
                candidate < 0
                || candidate >= int(EXPERTS)
                || !available[uint(candidate)]
            ) {
                continue;
            }
            bool duplicate = false;
            for (
                uint existing = 0u;
                existing < uint(TOP_K);
                ++existing
            ) {
                duplicate =
                    duplicate
                    || result[output_base + existing] == candidate;
            }
            if (!duplicate) {
                replacement = candidate;
                break;
            }
        }
        result[output_base + route] = replacement;
    }
)METAL";

constexpr const char* kWeightedReduceSource = R"METAL(
    uint index = thread_position_in_grid.x;
    if (index >= uint(TOKENS * WIDTH)) {
        return;
    }
    uint token = index / uint(WIDTH);
    uint column = index - token * uint(WIDTH);
    float value = 0.0f;
    for (uint route = 0u; route < uint(ROUTES); ++route) {
        value += float(pair_output[
            (token * uint(ROUTES) + route) * uint(WIDTH) + column
        ]) * weights[token * uint(ROUTES) + route];
    }
    output[index] = T(value);
)METAL";

constexpr const char* kGluSplitSource = R"METAL(
    uint index = thread_position_in_grid.x;
    if (index >= uint(ROWS * WIDTH)) {
        return;
    }
    uint row = index / uint(WIDTH);
    uint column = index - row * uint(WIDTH);
    uint offset = row * uint(WIDTH * 2);
    float gate = float(gate_up[offset + column]);
    float up = float(gate_up[offset + uint(WIDTH) + column]);
    if (HAS_LIMIT != 0) {
        gate = min(gate, params[0]);
        up = clamp(up, -params[0], params[0]);
    }
    float activated;
    if (GEGLU != 0) {
        constexpr float gelu_scale = 0.7978845608028654f;
        float inner =
            gelu_scale * (gate + 0.044715f * gate * gate * gate);
        activated = 0.5f * gate * (1.0f + tanh(inner));
    } else {
        activated = gate / (1.0f + exp(-gate));
    }
    output[index] = T(activated * up);
)METAL";

constexpr const char* kSharedGateSource = R"METAL(
    uint index = thread_position_in_grid.x;
    if (index >= uint(TOKENS * WIDTH)) {
        return;
    }
    uint token = index / uint(WIDTH);
    float gate = 1.0f / (1.0f + exp(-gate_logits[token]));
    output[index] =
        T(float(routed[index]) + gate * float(shared[index]));
)METAL";

constexpr const char* kReduceSharedGateSource = R"METAL(
    uint index = thread_position_in_grid.x;
    if (index >= uint(TOKENS * WIDTH)) {
        return;
    }
    uint token = index / uint(WIDTH);
    uint column = index - token * uint(WIDTH);
    float value = 0.0f;
    for (uint route = 0u; route < uint(ROUTES); ++route) {
        value += float(pair_output[
            (token * uint(ROUTES) + route) * uint(WIDTH) + column
        ]) * weights[token * uint(ROUTES) + route];
    }
    T routed_value = T(value);
    float gate = 1.0f / (1.0f + exp(-gate_logits[token]));
    output[index] =
        T(float(routed_value) + gate * float(shared[index]));
)METAL";

constexpr const char* kExpertScaleSource = R"METAL(
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }
    output[index] = weights[index] * scales[uint(ids[index])];
)METAL";

mlx::core::fast::CustomKernelFunction make_kernel(
    const char* name,
    std::vector<std::string> inputs,
    std::vector<std::string> outputs,
    std::string source) {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
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

const mlx::core::fast::CustomKernelFunction& top_k_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_moe_topk",
        {"logits", "bias", "available", "params"},
        {"ids", "weights"},
        kTopKSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
dense_router_top_k_kernel(mlx::core::Dtype dtype) {
    static const auto fp16_kernel = make_kernel(
        "mfq_cpp_moe_dense_router_topk_f16",
        {"input", "weight", "bias", "available", "params"},
        {"ids", "weights"},
        std::string("using activation4_t = half4;\n") +
            kDenseRouterTopKSource);
    static const auto bf16_kernel = make_kernel(
        "mfq_cpp_moe_dense_router_topk_bf16",
        {"input", "weight", "bias", "available", "params"},
        {"ids", "weights"},
        std::string("using activation4_t = bfloat4;\n") +
            kDenseRouterTopKSource);
    return dtype == mlx::core::bfloat16
        ? bf16_kernel
        : fp16_kernel;
}

const mlx::core::fast::CustomKernelFunction&
sqrt_softplus_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_moe_sqrtsoftplus_weights",
        {"logits", "ids", "params"},
        {"weights"},
        kSqrtSoftplusSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
repair_hash_ids_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_moe_repair_hash_ids",
        {"static_ids", "candidate_ids", "available"},
        {"result"},
        kRepairHashIdsSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
weighted_reduce_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_moe_weighted_reduce",
        {"pair_output", "weights"},
        {"output"},
        kWeightedReduceSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& glu_split_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_moe_glu_split",
        {"gate_up", "params"},
        {"output"},
        kGluSplitSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& shared_gate_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_moe_shared_gate",
        {"routed", "shared", "gate_logits"},
        {"output"},
        kSharedGateSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
reduce_shared_gate_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_moe_reduce_shared_gate",
        {"pair_output", "weights", "shared", "gate_logits"},
        {"output"},
        kReduceSharedGateSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
expert_scale_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_moe_expert_scale",
        {"weights", "ids", "scales"},
        {"output"},
        kExpertScaleSource);
    return kernel;
}

int checked_int(std::size_t value, const char* name) {
    if (value >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            std::string("MoE ") + name + " exceeds MLX limits");
    }
    return static_cast<int>(value);
}

array floating_contiguous(const array& input) {
    auto result = input;
    if (result.dtype() != mlx::core::float16 &&
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

array int32_contiguous(const array& input) {
    auto result = input;
    if (result.dtype() != mlx::core::int32) {
        result = mlx::core::astype(result, mlx::core::int32);
    }
    return mlx::core::contiguous(result);
}

array reshape_gate_logits(const array& input, int tokens) {
    auto result = float32_contiguous(input);
    if (result.size() != static_cast<std::size_t>(tokens)) {
        throw std::invalid_argument(
            "shared gate logits must contain one value per token");
    }
    return mlx::core::reshape(result, Shape{tokens});
}

array glu_split(
    const array& gate_up,
    bool geglu,
    float limit = 0.0f) {
    auto values = floating_contiguous(gate_up);
    if (values.ndim() < 2 ||
        values.shape(-1) <= 0 ||
        values.shape(-1) % 2 != 0) {
        throw std::invalid_argument(
            "gate_up must end in an even 2*width dimension");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "SwiGLU limit must be finite and non-negative");
    }
    const int width = values.shape(-1) / 2;
    const int rows = checked_int(
        values.size() /
            static_cast<std::size_t>(2 * width),
        "GLU row count");
    const int size = checked_int(
        static_cast<std::size_t>(rows) * width,
        "GLU output size");
    Shape output_shape(
        values.shape().begin(),
        values.shape().end() - 1);
    output_shape.push_back(width);
    const array params({limit}, mlx::core::float32);
    auto outputs = glu_split_kernel()(
        {values, params},
        {std::move(output_shape)},
        {values.dtype()},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {
            {"T", values.dtype()},
            {"ROWS", rows},
            {"WIDTH", width},
            {"GEGLU", static_cast<int>(geglu)},
            {"HAS_LIMIT", static_cast<int>(limit > 0.0f)},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

} // namespace

MlxMoeTopKResult moe_topk(
    const array& logits,
    int top_k,
    bool use_sigmoid,
    bool use_sqrt_softplus,
    bool normalize,
    bool delayed_softmax,
    const std::optional<array>& bias,
    const std::optional<array>& available,
    float norm_floor,
    float scale) {
    auto values = floating_contiguous(logits);
    if (values.ndim() < 2 || values.shape(-1) <= 0) {
        throw std::invalid_argument(
            "MoE router logits must end in an expert dimension");
    }
    const int experts = values.shape(-1);
    const int rows = checked_int(
        values.size() / static_cast<std::size_t>(experts),
        "router row count");
    if (rows <= 0) {
        throw std::invalid_argument(
            "MoE router logits cannot be empty");
    }
    if (top_k < 1 ||
        top_k > std::min(kMaximumRoutes, experts)) {
        throw std::invalid_argument(
            "MoE top_k must be in [1,min(16,experts)]");
    }
    if (experts > kMaximumExperts) {
        throw std::invalid_argument(
            "MoE top-k supports at most 4096 experts");
    }
    const int mode_count =
        static_cast<int>(use_sigmoid) +
        static_cast<int>(use_sqrt_softplus) +
        static_cast<int>(delayed_softmax);
    if (mode_count > 1) {
        throw std::invalid_argument(
            "sigmoid, sqrt-softplus, and delayed softmax "
            "are mutually exclusive");
    }
    if (delayed_softmax && normalize) {
        throw std::invalid_argument(
            "normalize and delayed softmax are mutually exclusive");
    }
    if (!std::isfinite(norm_floor) || norm_floor < 0.0f) {
        throw std::invalid_argument(
            "MoE norm floor must be finite and non-negative");
    }
    if (!std::isfinite(scale)) {
        throw std::invalid_argument(
            "MoE router scale must be finite");
    }
    const int mode = use_sigmoid
        ? 1
        : (use_sqrt_softplus
               ? 2
               : (delayed_softmax ? 3 : 0));

    array bias_values =
        mlx::core::zeros(Shape{experts}, mlx::core::float32);
    if (bias.has_value()) {
        bias_values = float32_contiguous(*bias);
        if (bias_values.shape() != Shape{experts}) {
            throw std::invalid_argument(
                "MoE router bias shape mismatch");
        }
    }

    array available_values =
        mlx::core::ones(Shape{experts}, mlx::core::bool_);
    if (available.has_value()) {
        available_values = mlx::core::contiguous(
            mlx::core::reshape(
                mlx::core::astype(
                    *available,
                    mlx::core::bool_),
                Shape{checked_int(
                    available->size(),
                    "availability size")}));
        if (available_values.shape() != Shape{experts}) {
            throw std::invalid_argument(
                "MoE router availability shape mismatch");
        }
    }

    values = mlx::core::reshape(values, Shape{rows, experts});
    const array params(
        {norm_floor, scale},
        mlx::core::float32);
    auto outputs = top_k_kernel()(
        {values, bias_values, available_values, params},
        {
            Shape{rows, top_k},
            Shape{rows, top_k},
        },
        {
            mlx::core::int32,
            mlx::core::float32,
        },
        {rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        {
            {"T", values.dtype()},
            {"ROWS", rows},
            {"EXPERTS", experts},
            {"TOP_K", top_k},
            {"MODE", mode},
            {"NORMALIZE", static_cast<int>(normalize)},
            {"HAS_BIAS", static_cast<int>(bias.has_value())},
            {
                "HAS_AVAILABLE",
                static_cast<int>(available.has_value()),
            },
        },
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs.at(0)),
        std::move(outputs.at(1)),
    };
}

bool moe_dense_router_topk_supported(
    const array& input,
    const array& weight) noexcept {
    const bool activation16 =
        input.dtype() == mlx::core::float16 ||
        input.dtype() == mlx::core::bfloat16;
    return input.ndim() > 0
        && activation16
        && input.size() == static_cast<std::size_t>(
            input.shape(-1))
        && input.shape(-1) > 0
        && (input.shape(-1) % 4) == 0
        && weight.ndim() == 2
        && weight.dtype() == input.dtype()
        && weight.shape(0) == 256
        && weight.shape(1) == input.shape(-1);
}

MlxMoeTopKResult moe_dense_router_topk(
    const array& input,
    const array& weight,
    const std::optional<array>& bias,
    const std::optional<array>& available,
    float norm_floor,
    float scale) {
    if (!moe_dense_router_topk_supported(input, weight)) {
        throw std::invalid_argument(
            "fused dense router requires one FP16/BF16 row and a "
            "matching contiguous [256,K] weight with K divisible by four");
    }
    if (!std::isfinite(norm_floor) || norm_floor < 0.0f ||
        !std::isfinite(scale)) {
        throw std::invalid_argument(
            "fused dense router parameters are invalid");
    }
    auto source = mlx::core::contiguous(
        mlx::core::reshape(
            input,
            Shape{1, input.shape(-1)}));
    auto weights = mlx::core::contiguous(weight);
    array bias_values =
        mlx::core::zeros(Shape{256}, mlx::core::float32);
    if (bias.has_value()) {
        bias_values = float32_contiguous(*bias);
        if (bias_values.shape() != Shape{256}) {
            throw std::invalid_argument(
                "fused dense router bias shape mismatch");
        }
    }
    array available_values =
        mlx::core::ones(Shape{256}, mlx::core::bool_);
    if (available.has_value()) {
        available_values = mlx::core::contiguous(
            mlx::core::reshape(
                mlx::core::astype(
                    *available,
                    mlx::core::bool_),
                Shape{checked_int(
                    available->size(),
                    "fused router availability size")}));
        if (available_values.shape() != Shape{256}) {
            throw std::invalid_argument(
                "fused dense router availability shape mismatch");
        }
    }
    const array params(
        {norm_floor, scale},
        mlx::core::float32);
    auto outputs = dense_router_top_k_kernel(source.dtype())(
        {
            source,
            weights,
            bias_values,
            available_values,
            params,
        },
        {Shape{1, 6}, Shape{1, 6}},
        {mlx::core::int32, mlx::core::float32},
        {1024, 1, 1},
        {1024, 1, 1},
        {
            {"K", input.shape(-1)},
            {"HAS_BIAS", static_cast<int>(bias.has_value())},
            {
                "HAS_AVAILABLE",
                static_cast<int>(available.has_value()),
            },
        },
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs.at(0)),
        std::move(outputs.at(1)),
    };
}

array moe_selected_sqrtsoftplus_weights(
    const array& logits,
    const array& ids,
    bool normalize,
    float norm_floor,
    float scale) {
    auto values = floating_contiguous(logits);
    auto selected = int32_contiguous(ids);
    if (values.ndim() < 2 || selected.ndim() != 2 ||
        values.shape(-1) <= 0) {
        throw std::invalid_argument(
            "MoE sqrt-softplus expects logits and rank-two ids");
    }
    const int experts = values.shape(-1);
    const int rows = selected.shape(0);
    const int top_k = selected.shape(1);
    if (rows <= 0 || top_k < 1 || top_k > kMaximumRoutes ||
        values.size() !=
            static_cast<std::size_t>(rows) * experts) {
        throw std::invalid_argument(
            "MoE sqrt-softplus shapes are incompatible");
    }
    if (!std::isfinite(norm_floor) || norm_floor < 0.0f ||
        !std::isfinite(scale)) {
        throw std::invalid_argument(
            "MoE sqrt-softplus parameters are invalid");
    }
    values = mlx::core::reshape(values, Shape{rows, experts});
    const array params(
        {norm_floor, scale},
        mlx::core::float32);
    auto outputs = sqrt_softplus_kernel()(
        {values, selected, params},
        {selected.shape()},
        {mlx::core::float32},
        {rows * 32, 1, 1},
        {32, 1, 1},
        {
            {"T", values.dtype()},
            {"ROWS", rows},
            {"EXPERTS", experts},
            {"TOP_K", top_k},
            {"NORMALIZE", static_cast<int>(normalize)},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array moe_sqrtsoftplus_weights(
    const array& logits,
    const array& ids,
    float norm_floor,
    float scale) {
    return moe_selected_sqrtsoftplus_weights(
        logits,
        ids,
        true,
        norm_floor,
        scale);
}

array moe_repair_hash_ids(
    const array& static_ids,
    const array& candidate_ids,
    const array& available) {
    auto fixed = int32_contiguous(static_ids);
    auto candidates = int32_contiguous(candidate_ids);
    auto availability = mlx::core::contiguous(
        mlx::core::astype(
            available,
            mlx::core::bool_));
    if (fixed.ndim() != 2 ||
        candidates.ndim() != 2 ||
        fixed.shape(0) <= 0 ||
        fixed.shape(1) <= 0 ||
        fixed.shape(1) > kMaximumRoutes ||
        candidates.shape(0) != fixed.shape(0) ||
        candidates.shape(1) < fixed.shape(1) ||
        candidates.shape(1) > kMaximumRoutes ||
        availability.ndim() != 1 ||
        availability.shape(0) <= 0) {
        throw std::invalid_argument(
            "MoE hash routing shapes are incompatible");
    }
    const int rows = fixed.shape(0);
    const int top_k = fixed.shape(1);
    const int candidate_count = candidates.shape(1);
    const int experts = availability.shape(0);
    auto outputs = repair_hash_ids_kernel()(
        {fixed, candidates, availability},
        {fixed.shape()},
        {mlx::core::int32},
        {rows, 1, 1},
        {std::min(kThreads, rows), 1, 1},
        {
            {"ROWS", rows},
            {"TOP_K", top_k},
            {"CANDIDATES", candidate_count},
            {"EXPERTS", experts},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array moe_weighted_reduce(
    const array& pair_output,
    const array& weights) {
    auto pairs = floating_contiguous(pair_output);
    auto route_weights = float32_contiguous(weights);
    if (pairs.ndim() != 3) {
        throw std::invalid_argument(
            "MoE pair output must have [tokens,routes,width] shape");
    }
    const int tokens = pairs.shape(0);
    const int routes = pairs.shape(1);
    const int width = pairs.shape(2);
    if (tokens <= 0 || routes <= 0 || width <= 0 ||
        route_weights.shape() != Shape{tokens, routes}) {
        throw std::invalid_argument(
            "MoE route weights shape mismatch");
    }
    const int size = checked_int(
        static_cast<std::size_t>(tokens) * width,
        "weighted reduce size");
    auto outputs = weighted_reduce_kernel()(
        {pairs, route_weights},
        {Shape{tokens, width}},
        {pairs.dtype()},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {
            {"T", pairs.dtype()},
            {"TOKENS", tokens},
            {"ROUTES", routes},
            {"WIDTH", width},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array moe_swiglu_split(const array& gate_up) {
    return glu_split(gate_up, false);
}

array moe_limited_swiglu_split(
    const array& gate_up,
    float limit) {
    return glu_split(gate_up, false, limit);
}

array moe_geglu_split(const array& gate_up) {
    return glu_split(gate_up, true);
}

array moe_add_shared_gate(
    const array& routed,
    const array& shared,
    const array& gate_logits) {
    auto routed_values = floating_contiguous(routed);
    auto shared_values = floating_contiguous(shared);
    if (routed_values.ndim() != 2 ||
        shared_values.shape() != routed_values.shape()) {
        throw std::invalid_argument(
            "MoE routed/shared values must have matching "
            "[tokens,width] shapes");
    }
    if (shared_values.dtype() != routed_values.dtype()) {
        shared_values = mlx::core::contiguous(
            mlx::core::astype(
                shared_values,
                routed_values.dtype()));
    }
    const int tokens = routed_values.shape(0);
    const int width = routed_values.shape(1);
    if (tokens <= 0 || width <= 0) {
        throw std::invalid_argument(
            "MoE routed/shared values cannot be empty");
    }
    auto gates = reshape_gate_logits(gate_logits, tokens);
    const int size = checked_int(
        static_cast<std::size_t>(tokens) * width,
        "shared gate size");
    auto outputs = shared_gate_kernel()(
        {routed_values, shared_values, gates},
        {routed_values.shape()},
        {routed_values.dtype()},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {
            {"T", routed_values.dtype()},
            {"TOKENS", tokens},
            {"WIDTH", width},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array moe_weighted_reduce_shared_gate(
    const array& pair_output,
    const array& weights,
    const array& shared,
    const array& gate_logits) {
    auto pairs = floating_contiguous(pair_output);
    auto route_weights = float32_contiguous(weights);
    auto shared_values = floating_contiguous(shared);
    if (pairs.ndim() != 3) {
        throw std::invalid_argument(
            "MoE pair output must have [tokens,routes,width] shape");
    }
    const int tokens = pairs.shape(0);
    const int routes = pairs.shape(1);
    const int width = pairs.shape(2);
    if (tokens <= 0 || routes <= 0 || width <= 0 ||
        route_weights.shape() != Shape{tokens, routes} ||
        shared_values.shape() != Shape{tokens, width}) {
        throw std::invalid_argument(
            "MoE fused reduce/shared gate shape mismatch");
    }
    if (shared_values.dtype() != pairs.dtype()) {
        shared_values = mlx::core::contiguous(
            mlx::core::astype(
                shared_values,
                pairs.dtype()));
    }
    auto gates = reshape_gate_logits(gate_logits, tokens);
    const int size = checked_int(
        static_cast<std::size_t>(tokens) * width,
        "fused reduce/shared gate size");
    auto outputs = reduce_shared_gate_kernel()(
        {pairs, route_weights, shared_values, gates},
        {Shape{tokens, width}},
        {pairs.dtype()},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {
            {"T", pairs.dtype()},
            {"TOKENS", tokens},
            {"ROUTES", routes},
            {"WIDTH", width},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array moe_apply_expert_scale(
    const array& weights,
    const array& ids,
    const array& scales) {
    auto values = float32_contiguous(weights);
    auto selected = int32_contiguous(ids);
    auto expert_scales = float32_contiguous(scales);
    if (values.shape() != selected.shape() ||
        expert_scales.ndim() != 1 ||
        expert_scales.shape(0) <= 0 ||
        values.size() == 0) {
        throw std::invalid_argument(
            "MoE expert scale shapes are invalid");
    }
    const int size = checked_int(
        values.size(),
        "expert scale size");
    auto outputs = expert_scale_kernel()(
        {values, selected, expert_scales},
        {values.shape()},
        {mlx::core::float32},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {{"SIZE", size}},
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

} // namespace mfq::metal
