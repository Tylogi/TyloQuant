#include "mlx_deepseek_v41_mhc.h"
#include "mlx_kernel_prepare.h"

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
constexpr int kMixWidth = 24;
constexpr int kSinkhornIterations = 20;
constexpr int kPostThreads = 256;
constexpr int kOfficialHidden = 5120;
constexpr int kCollapseThreads = 1024;

const mlx::core::fast::CustomKernelFunction& projection_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = mlx::core::MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_dsv41_hc_projection", {"residual", "weight", "params"}, {"output"},
            R"METAL(
                uint row = threadgroup_position_in_grid.x / 24u;
                uint projection = threadgroup_position_in_grid.x % 24u;
                uint tid = thread_index_in_threadgroup;
                uint lane = thread_index_in_simdgroup;
                uint group = simdgroup_index_in_threadgroup;
                float product = 0.0f, square = 0.0f;
                for (uint column = tid * 4u; column < uint(WIDTH); column += 1024u) {
                    float4 x = float4(float(residual[row * WIDTH + column]),
                        float(residual[row * WIDTH + column + 1u]),
                        float(residual[row * WIDTH + column + 2u]),
                        float(residual[row * WIDTH + column + 3u]));
                    float4 w = reinterpret_cast<device const float4*>(weight + projection * WIDTH)[column / 4u];
                    product += dot(x, w);
                    square += dot(x, x);
                }
                product = simd_sum(product);
                square = simd_sum(square);
                threadgroup float products[8], squares[8];
                if (lane == 0u) { products[group] = product; squares[group] = square; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (group == 0u) {
                    product = simd_sum(lane < 8u ? products[lane] : 0.0f);
                    square = simd_sum(lane < 8u ? squares[lane] : 0.0f);
                    if (lane == 0u) output[row * 24u + projection] =
                        product * metal::precise::rsqrt(square / float(WIDTH) + params[0]);
                }
            )METAL", "", true, false, options);
    }();
    return kernel;
}

constexpr const char* kV41HcCollapseNormSource = R"METAL(
    const int ROWS = residual_shape[0] * residual_shape[1];
    constexpr uint CONNECTIONS = 4u;
    constexpr uint READS = 4u;
    constexpr uint THREADS = 1024u;
    constexpr uint SIMD_SIZE = 32u;
    uint row = threadgroup_position_in_grid.x;
    uint local_thread = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }

    threadgroup volatile hc_activation_t collapsed[HIDDEN];
    threadgroup float local_inverse[1];
    threadgroup float local_sums[SIMD_SIZE];
    float accumulator = 0.0f;
    uint residual_base = row * CONNECTIONS * uint(HIDDEN);
    uint pre_base = row * CONNECTIONS;
    uint first_feature = local_thread * READS;
    for (uint offset = 0u;
         offset < uint(HIDDEN);
         offset += THREADS * READS) {
        for (uint local = 0u; local < READS; ++local) {
            uint feature = first_feature + offset + local;
            if (feature < uint(HIDDEN)) {
                float product0 = pre[pre_base]
                    * float(residual[residual_base + feature]);
                float product1 = pre[pre_base + 1u]
                    * float(residual[
                        residual_base + uint(HIDDEN) + feature]);
                float product2 = pre[pre_base + 2u]
                    * float(residual[
                        residual_base + 2u * uint(HIDDEN) + feature]);
                float product3 = pre[pre_base + 3u]
                    * float(residual[
                        residual_base + 3u * uint(HIDDEN) + feature]);
                hc_activation_t value = hc_activation_t(
                    product3 + (product2 + (product1 + product0)));
                collapsed[feature] = value;
                float x = float(value);
                accumulator += x * x;
            }
        }
    }

    accumulator = simd_sum(accumulator);
    if (simd_group == 0u) {
        local_sums[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        local_sums[simd_group] = accumulator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        accumulator = simd_sum(local_sums[lane]);
        if (lane == 0u) {
            local_inverse[0] = metal::precise::rsqrt(
                accumulator / float(HIDDEN) + params[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inverse = local_inverse[0];
    for (uint offset = 0u;
         offset < uint(HIDDEN);
         offset += THREADS * READS) {
        for (uint local = 0u; local < READS; ++local) {
            uint feature = first_feature + offset + local;
            if (feature < uint(HIDDEN)) {
                float normalized = float(collapsed[feature]) * inverse;
                output[row * uint(HIDDEN) + feature] = hc_activation_t(
                    norm[feature] * normalized);
            }
        }
    }
)METAL";

constexpr const char* kV41HcMetadataExactSource = R"METAL(
    const int ROWS = logits_shape[0] * logits_shape[1];
    uint row = thread_position_in_grid.x;
    if (row >= uint(ROWS)) {
        return;
    }

    constexpr uint CONNECTIONS = 4u;
    uint value_base = row * 16u;
    float epsilon = params[0];
    float probabilities[16];
    for (uint source = 0u; source < CONNECTIONS; ++source) {
        uint source_base = source * CONNECTIONS;
        float values[4];
        for (uint destination = 0u;
             destination < CONNECTIONS;
             ++destination) {
            values[destination] =
                logits[value_base + source_base + destination];
        }
        float maximum = max(max(values[0], values[1]),
                            max(values[2], values[3]));
        float exponentials[4];
        for (uint destination = 0u;
             destination < CONNECTIONS;
             ++destination) {
            exponentials[destination] = metal::precise::exp(
                values[destination] - maximum);
        }
        float denominator =
            exponentials[3] +
            (exponentials[2] +
             (exponentials[1] + exponentials[0]));
        for (uint destination = 0u;
             destination < CONNECTIONS;
             ++destination) {
            probabilities[source_base + destination] =
                exponentials[destination] / denominator + epsilon;
        }
    }

    for (uint destination = 0u;
         destination < CONNECTIONS;
         ++destination) {
        float denominator =
            probabilities[12u + destination] +
            (probabilities[8u + destination] +
             (probabilities[4u + destination] +
              probabilities[destination]));
        denominator = denominator + epsilon;
        for (uint source = 0u; source < CONNECTIONS; ++source) {
            uint index = source * CONNECTIONS + destination;
            probabilities[index] = probabilities[index] / denominator;
        }
    }

    for (uint iteration = 1u;
         iteration < uint(SINKHORN_ITERATIONS);
         ++iteration) {
        for (uint source = 0u; source < CONNECTIONS; ++source) {
            uint source_base = source * CONNECTIONS;
            float denominator =
                probabilities[source_base + 3u] +
                (probabilities[source_base + 2u] +
                 (probabilities[source_base + 1u] +
                  probabilities[source_base]));
            denominator = denominator + epsilon;
            for (uint destination = 0u;
                 destination < CONNECTIONS;
                 ++destination) {
                uint index = source_base + destination;
                probabilities[index] = probabilities[index] / denominator;
            }
        }
        for (uint destination = 0u;
             destination < CONNECTIONS;
             ++destination) {
            float denominator =
                probabilities[12u + destination] +
                (probabilities[8u + destination] +
                 (probabilities[4u + destination] +
                  probabilities[destination]));
            denominator = denominator + epsilon;
            for (uint source = 0u; source < CONNECTIONS; ++source) {
                uint index = source * CONNECTIONS + destination;
                probabilities[index] = probabilities[index] / denominator;
            }
        }
    }

    for (uint index = 0u; index < 16u; ++index) {
        combination[value_base + index] = probabilities[index];
    }
)METAL";

constexpr const char* kV41HcPostSource = R"METAL(
    const int SIZE = residual_shape[0] * residual_shape[1] * 4 * HIDDEN;
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }
    uint feature = index % uint(HIDDEN);
    uint destination_row = index / uint(HIDDEN);
    uint destination = destination_row % 4u;
    uint row = destination_row / 4u;
    float residual_products[4];
    for (uint source = 0u; source < 4u; ++source) {
        uint combination_index =
            (row * 4u + source) * 4u + destination;
        residual_products[source] =
            combination[combination_index] * float(residual[
                (row * 4u + source) * uint(HIDDEN) + feature
            ]);
    }
    float residual_sum =
        ((residual_products[0] + residual_products[1])
            + residual_products[2])
        + residual_products[3];
    float branch_value = float(
        branch[row * uint(HIDDEN) + feature]
    );
    if (ADD_BRANCH != 0) {
        branch_value = float(output_activation_t(
            branch_value + float(
                branch2[row * uint(HIDDEN) + feature]
            )
        ));
    }
    float direct = post[row * 4u + destination] * branch_value;
    output[index] = output_activation_t(direct + residual_sum);
)METAL";

constexpr const char* kV41HcMetadataLanesSource = R"METAL(
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    const uint rows = uint(mixes_shape[0] * mixes_shape[1]);
    const uint row = thread_position_in_grid.x / 16u;
    const uint lane = thread_index_in_simdgroup;
    const uint first = lane & ~15u;
    const uint entry = lane & 15u;
    const uint source = entry / 4u, destination = entry & 3u;
    const uint offset = (row < rows ? row : 0u) * 24u;
    const float epsilon = params[0];
    float value = mixes[offset + 8u + entry] * scale[2] + base[8u + entry];
    const float v0 = simd_shuffle(value, first + source * 4u);
    const float v1 = simd_shuffle(value, first + source * 4u + 1u);
    const float v2 = simd_shuffle(value, first + source * 4u + 2u);
    const float v3 = simd_shuffle(value, first + source * 4u + 3u);
    const float maximum = max(max(v0, v1), max(v2, v3));
    value = metal::precise::exp(value - maximum);
    float total = simd_shuffle(value, first + source * 4u + 3u) +
        (simd_shuffle(value, first + source * 4u + 2u) +
         (simd_shuffle(value, first + source * 4u + 1u) +
          simd_shuffle(value, first + source * 4u)));
    value = value / total + epsilon;
    for (uint iteration = 0u; iteration < uint(SINKHORN_ITERATIONS); ++iteration) {
        if (iteration != 0u) {
            total = simd_shuffle(value, first + source * 4u + 3u) +
                (simd_shuffle(value, first + source * 4u + 2u) +
                 (simd_shuffle(value, first + source * 4u + 1u) +
                  simd_shuffle(value, first + source * 4u)));
            value = value / (total + epsilon);
        }
        total = simd_shuffle(value, first + 12u + destination) +
            (simd_shuffle(value, first + 8u + destination) +
             (simd_shuffle(value, first + 4u + destination) +
              simd_shuffle(value, first + destination)));
        value = value / (total + epsilon);
    }
    if (row < rows) {
        combination[row * 16u + entry] = value;
        if (entry < 4u) {
            const float a = mixes[offset + entry] * scale[0] + base[entry];
            const float b = mixes[offset + 4u + entry] * scale[1] + base[4u + entry];
            const float pa = 1.0f / (1.0f + metal::precise::exp(metal::abs(a)));
            const float pb = 1.0f / (1.0f + metal::precise::exp(metal::abs(b)));
            pre[row * 4u + entry] = (a < 0.0f ? pa : 1.0f - pa) + epsilon;
            post[row * 4u + entry] = 2.0f * (b < 0.0f ? pb : 1.0f - pb);
        }
    }
)METAL";

mlx::core::fast::CustomKernelFunction make_kernel(
    const char* name,
    std::vector<std::string> inputs,
    std::vector<std::string> outputs,
    std::string source) {
    CompileOptions options;
    options.math_mode = mlx::core::MathMode::Safe;
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

const mlx::core::fast::CustomKernelFunction& collapse_norm_kernel(
    mlx::core::Dtype dtype) {
    static const auto fp16_kernel = make_kernel(
        "mfq_cpp_dsv41_hc_collapse_norm_f16_h5120",
        {"residual", "pre", "norm", "params"},
        {"output"},
        std::string("using hc_activation_t = half;\n") +
            kV41HcCollapseNormSource);
    static const auto bf16_kernel = make_kernel(
        "mfq_cpp_dsv41_hc_collapse_norm_bf16_h5120",
        {"residual", "pre", "norm", "params"},
        {"output"},
        std::string("using hc_activation_t = bfloat;\n") +
            kV41HcCollapseNormSource);
    return dtype == mlx::core::bfloat16 ? bf16_kernel : fp16_kernel;
}

const mlx::core::fast::CustomKernelFunction& metadata_exact_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv41_hc_metadata_exact_f32",
        {"logits", "params"},
        {"combination"},
        kV41HcMetadataExactSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& post_collapse_kernel() {
    static const auto kernel = make_kernel("mfq_dsv41_hc_post_collapse_norm",
        {"residual", "branch", "post", "combination", "pre", "norm", "params"},
        {"expanded", "normalized"}, R"MSL(
        const uint row = threadgroup_position_in_grid.x;
        const uint tid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint simd = simdgroup_index_in_threadgroup;
        threadgroup volatile T collapsed[HIDDEN];
        threadgroup float partial[32];
        threadgroup float inverse;
        float square = 0.0f;
        for (uint offset = 0u; offset < HIDDEN; offset += 4096u) {
            for (uint local = 0u; local < 4u; ++local) {
                const uint d = tid * 4u + offset + local;
                if (d < HIDDEN) {
                    float x[4];
                    const float v = float(branch[row * HIDDEN + d]);
                    for (uint dest = 0u; dest < 4u; ++dest) {
                        float products[4];
                        for (uint source = 0u; source < 4u; ++source)
                            products[source] = combination[(row * 4u + source) * 4u + dest]
                                * float(residual[(row * 4u + source) * HIDDEN + d]);
                        const float sum = ((products[0] + products[1]) + products[2]) + products[3];
                        const float direct = post[row * 4u + dest] * v;
                        const T value = T(direct + sum);
                        expanded[(row * 4u + dest) * HIDDEN + d] = value;
                        x[dest] = float(value);
                    }
                    const float p0 = pre[row * 4u] * x[0];
                    const float p1 = pre[row * 4u + 1u] * x[1];
                    const float p2 = pre[row * 4u + 2u] * x[2];
                    const float p3 = pre[row * 4u + 3u] * x[3];
                    const T value = T(p3 + (p2 + (p1 + p0)));
                    collapsed[d] = value;
                    square += float(value) * float(value);
                }
            }
        }
        square = simd_sum(square);
        if (lane == 0u) partial[simd] = square;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            const float total = simd_sum(partial[lane]);
            if (lane == 0u) inverse = metal::precise::rsqrt(total / float(HIDDEN) + params[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = tid; d < HIDDEN; d += 1024u)
            normalized[row * HIDDEN + d] = T(norm[d] * (float(collapsed[d]) * inverse));
        )MSL");
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& metadata_lanes_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv41_hc_metadata_lanes_f32",
        {"mixes", "scale", "base", "params"},
        {"post", "combination", "pre"},
        kV41HcMetadataLanesSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& post_kernel(
    mlx::core::Dtype dtype) {
    static const auto fp16_kernel = make_kernel(
        "mfq_cpp_dsv41_hc_post_f16",
        {"branch", "branch2", "residual", "post", "combination"},
        {"output"},
        std::string("using output_activation_t = half;\n") +
            kV41HcPostSource);
    static const auto bf16_kernel = make_kernel(
        "mfq_cpp_dsv41_hc_post_bf16",
        {"branch", "branch2", "residual", "post", "combination"},
        {"output"},
        std::string("using output_activation_t = bfloat;\n") +
            kV41HcPostSource);
    return dtype == mlx::core::bfloat16 ? bf16_kernel : fp16_kernel;
}

int checked_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4.1 HC ") + name +
            " exceeds MLX limits");
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

array slice_last(const array& input, int begin, int end) {
    Shape start(input.ndim(), 0);
    Shape stop = input.shape();
    start.back() = begin;
    stop.back() = end;
    return mlx::core::slice(input, std::move(start), std::move(stop));
}

array softmax_last(const array& input) {
    auto source = mlx::core::contiguous(input);
    auto maximum = mlx::core::max(source, -1, true);
    auto exponential = mlx::core::exp(source - maximum);
    return exponential / mlx::core::sum(exponential, -1, true);
}

array generic_collapse_norm(
    const array& residual,
    const array& pre,
    const array& norm,
    float norm_eps) {
    auto reduced = mlx::core::sum(
        mlx::core::expand_dims(pre, -1) *
            mlx::core::astype(residual, mlx::core::float32),
        2);
    reduced = mlx::core::astype(reduced, residual.dtype());
    return MlxRmsNorm(norm, norm_eps)(reduced);
}

array post_impl(
    const array& branch,
    const array& branch2,
    const array& residual,
    const array& post,
    const array& combination,
    bool add_branch) {
    auto branch_values = floating_contiguous(branch);
    auto branch2_values = add_branch
        ? floating_contiguous(branch2)
        : branch_values;
    auto residual_values = activation16_contiguous(residual);
    auto post_values = float32_contiguous(post);
    auto combination_values = float32_contiguous(combination);
    if (branch_values.ndim() != 3 ||
        branch_values.shape(0) <= 0 ||
        branch_values.shape(1) <= 0 ||
        branch_values.shape(2) <= 0 ||
        branch2_values.shape() != branch_values.shape()) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 HC branch geometry disagrees");
    }
    const int batch = branch_values.shape(0);
    const int tokens = branch_values.shape(1);
    const int hidden = branch_values.shape(2);
    if (residual_values.shape() !=
            Shape{batch, tokens, kConnections, hidden} ||
        post_values.shape() != Shape{batch, tokens, kConnections} ||
        combination_values.shape() !=
            Shape{batch, tokens, kConnections, kConnections}) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 HC expansion geometry disagrees");
    }
    const int size = checked_int(
        static_cast<std::size_t>(batch) * tokens *
            kConnections * hidden,
        "output size");
    auto outputs = post_kernel(residual_values.dtype())(
        {
            branch_values,
            branch2_values,
            residual_values,
            post_values,
            combination_values,
        },
        {Shape{batch, tokens, kConnections, hidden}},
        {residual_values.dtype()},
        {size, 1, 1},
        {std::min(kPostThreads, size), 1, 1},
        {
            {"HIDDEN", hidden},
            {"ADD_BRANCH", static_cast<int>(add_branch)},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array dense_array(const MfqContainer& model, const std::string& name) {
    const auto mapped = model.map_record(name);
    return load_dense_array(model.record(name).dtype, mapped.view());
}

} // namespace

array deepseek_v41_hc_projection(
    const array& residual,
    const array& weight,
    float eps) {
    if (residual.ndim() != 4 || residual.shape(0) <= 0 || residual.shape(1) <= 0 ||
        residual.shape(2) != 4 || residual.shape(3) <= 0 ||
        residual.shape(3) > std::numeric_limits<int>::max() / 4 ||
        weight.shape() != Shape{24, 4 * residual.shape(3)} ||
        weight.dtype() != mlx::core::float32 || !std::isfinite(eps) || eps <= 0) {
        throw std::invalid_argument("invalid DeepSeek-V4.1 HC projection");
    }
    const int width = 4 * residual.shape(3);
    const int rows = checked_int(static_cast<std::size_t>(residual.shape(0)) *
        residual.shape(1), "projection row count");
    if (rows <= 6 && width % 1024 == 0 &&
        (residual.dtype() == mlx::core::float16 || residual.dtype() == mlx::core::bfloat16)) {
        auto outputs = projection_kernel()(
            {mlx::core::contiguous(residual), mlx::core::contiguous(weight), array({eps})},
            {Shape{residual.shape(0), residual.shape(1), 24}}, {mlx::core::float32},
            {rows * 24 * 256, 1, 1}, {256, 1, 1}, {{"WIDTH", width}},
            std::nullopt, false, {});
        return std::move(outputs.front());
    }
    auto flattened = mlx::core::reshape(mlx::core::astype(residual, mlx::core::float32),
        Shape{residual.shape(0), residual.shape(1), width});
    auto inverse = mlx::core::rsqrt(mlx::core::mean(flattened * flattened, -1, true) + eps);
    return mlx::core::matmul(flattened * inverse, mlx::core::transpose(weight));
}

array deepseek_v41_hc_collapse_norm(
    const array& residual,
    const array& pre,
    const array& norm,
    float norm_eps) {
    auto residual_values = activation16_contiguous(residual);
    auto pre_values = float32_contiguous(pre);
    auto norm_values = float32_contiguous(norm);
    if (residual_values.ndim() != 4 ||
        residual_values.shape(0) <= 0 ||
        residual_values.shape(1) <= 0 ||
        residual_values.shape(2) != kConnections ||
        residual_values.shape(3) <= 0 ||
        pre_values.shape() != Shape{
            residual_values.shape(0),
            residual_values.shape(1),
            kConnections,
        } ||
        norm_values.shape() != Shape{residual_values.shape(3)} ||
        !std::isfinite(norm_eps) || norm_eps <= 0.0f) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 HC collapse/RMSNorm input");
    }
    if ((residual_values.dtype() != mlx::core::float16 &&
         residual_values.dtype() != mlx::core::bfloat16) ||
        residual_values.shape(3) != kOfficialHidden) {
        return generic_collapse_norm(
            residual_values, pre_values, norm_values, norm_eps);
    }
    const int rows = checked_int(
        static_cast<std::size_t>(residual_values.shape(0)) *
            residual_values.shape(1),
        "collapse row count");
    if (rows > std::numeric_limits<int>::max() / kCollapseThreads) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 HC collapse grid exceeds Metal limits");
    }
    const array params({norm_eps}, mlx::core::float32);
    auto outputs = collapse_norm_kernel(residual_values.dtype())(
        {residual_values, pre_values, norm_values, params},
        {Shape{
            residual_values.shape(0),
            residual_values.shape(1),
            kOfficialHidden,
        }},
        {residual_values.dtype()},
        {rows * kCollapseThreads, 1, 1},
        {kCollapseThreads, 1, 1},
        {
            {"HIDDEN", kOfficialHidden},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

MlxDeepseekV41HcMetadataResult deepseek_v41_hc_metadata_exact(
    const array& normalized_mixes,
    const array& scale,
    const array& base,
    int sinkhorn_iterations,
    float eps) {
    auto mix_values = float32_contiguous(normalized_mixes);
    auto scale_values = float32_contiguous(scale);
    auto base_values = float32_contiguous(base);
    if (mix_values.ndim() != 3 ||
        mix_values.shape(0) <= 0 ||
        mix_values.shape(1) <= 0 ||
        mix_values.shape(2) != kMixWidth ||
        scale_values.size() != 3 ||
        base_values.size() != kMixWidth ||
        sinkhorn_iterations != kSinkhornIterations ||
        !std::isfinite(eps) || eps <= 0.0f) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 HC metadata input");
    }
    scale_values = mlx::core::reshape(scale_values, Shape{3});
    base_values = mlx::core::reshape(base_values, Shape{kMixWidth});
    const int batch = mix_values.shape(0);
    const int tokens = mix_values.shape(1);
    const int rows = checked_int(
        static_cast<std::size_t>(batch) * tokens,
        "metadata row count");
    if (rows <= 512) {
        auto outputs = metadata_lanes_kernel()(
            {mix_values, scale_values, base_values, array({eps}, mlx::core::float32)},
            {Shape{batch, tokens, 4}, Shape{batch, tokens, 4, 4}, Shape{batch, tokens, 4}},
            {mlx::core::float32, mlx::core::float32, mlx::core::float32},
            {((rows + 1) / 2) * 32, 1, 1}, {32, 1, 1},
            {{"SINKHORN_ITERATIONS", sinkhorn_iterations}}, std::nullopt, false, {});
        return {std::move(outputs[0]), std::move(outputs[1]), std::move(outputs[2])};
    }
    auto pre = mlx::core::sigmoid(
        slice_last(mix_values, 0, kConnections) *
            slice_last(scale_values, 0, 1) +
        slice_last(base_values, 0, kConnections)) + eps;
    auto post = 2.0f * mlx::core::sigmoid(
        slice_last(mix_values, kConnections, 2 * kConnections) *
            slice_last(scale_values, 1, 2) +
        slice_last(base_values, kConnections, 2 * kConnections));
    const Shape matrix_shape{
        batch, tokens, kConnections, kConnections};
    auto logits = float32_contiguous(
        mlx::core::reshape(
            slice_last(mix_values, 2 * kConnections, kMixWidth),
            matrix_shape) *
            slice_last(scale_values, 2, 3) +
        mlx::core::reshape(
            slice_last(base_values, 2 * kConnections, kMixWidth),
            Shape{kConnections, kConnections}));
    const array params({eps}, mlx::core::float32);
    auto outputs = metadata_exact_kernel()(
        {logits, params},
        {matrix_shape},
        {mlx::core::float32},
        {((rows + 31) / 32) * 32, 1, 1},
        {32, 1, 1},
        {
            {"SINKHORN_ITERATIONS", sinkhorn_iterations},
        },
        std::nullopt,
        false,
        {});
    array combination = std::move(outputs.front());
    return {
        std::move(post),
        std::move(combination),
        std::move(pre),
    };
}

array deepseek_v41_hc_post(
    const array& branch,
    const array& residual,
    const array& post,
    const array& combination) {
    return post_impl(
        branch, branch, residual, post, combination, false);
}

array deepseek_v41_hc_post_sum(
    const array& routed,
    const array& shared,
    const array& residual,
    const array& post,
    const array& combination) {
    return post_impl(
        routed, shared, residual, post, combination, true);
}

MlxDeepseekV41Mhc MlxDeepseekV41Mhc::load(
    const MfqContainer& model,
    const DeepseekV41Config& config,
    const std::string& prefix,
    const std::string& norm_name) {
    return MlxDeepseekV41Mhc(
        config,
        MlxLinear::load(model, prefix + ".function"),
        dense_array(model, prefix + ".base"),
        dense_array(model, prefix + ".scale"),
        dense_array(model, norm_name));
}

MlxDeepseekV41Mhc::MlxDeepseekV41Mhc(
    DeepseekV41Config config,
    MlxLinear function,
    array base,
    array scale,
    array norm)
    : config_(std::move(config)),
      function_(std::move(function)),
      base_(mlx::core::reshape(
          mlx::core::astype(base, mlx::core::float32), Shape{24})),
      scale_(mlx::core::reshape(
          mlx::core::astype(scale, mlx::core::float32), Shape{3})),
      norm_(std::move(norm), static_cast<float>(config_.rms_eps)) {
    if (config_.hc_mult != 4 ||
        function_.input_size() != 4 * config_.hidden ||
        function_.output_size() != 24) {
        throw std::runtime_error(
            "DeepSeek-V4.1 Mega-mHC geometry disagrees");
    }
    if (const auto* tq = function_.tq_weight_ref())
        if (auto* preparation = MlxKernelPreparation::current()) preparation->collect([&] {
            for (const auto rows : preparation->row_buckets()) if (rows <= 32)
                for (auto dtype : {mlx::core::float16, mlx::core::float32})
                    preparation->add(tq->rms_matmul(mlx::core::zeros(Shape{rows, function_.input_size()}, dtype),
                        static_cast<float>(config_.rms_eps)));
            for (const int requested : {32, 128, 256, 512, 1024, 2048, 4096}) {
                const int rows = std::min(preparation->context(), requested);
                preparation->add(tq->rms_matmul(mlx::core::zeros(Shape{rows, function_.input_size()}, mlx::core::float16),
                    static_cast<float>(config_.rms_eps)));
            }
        });
}

MlxDeepseekV41MhcResult MlxDeepseekV41Mhc::collapse(
    const array& residual,
    const array& previous_pre) const {
    if (residual.ndim() != 4 || residual.shape(0) <= 0 ||
        residual.shape(1) <= 0 || residual.shape(2) != 4 ||
        residual.shape(3) != config_.hidden ||
        previous_pre.shape() !=
            Shape{residual.shape(0), residual.shape(1), 4}) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 Mega-mHC input mismatch");
    }
    auto expansion = metadata(residual);
    auto branch = deepseek_v41_hc_collapse_norm(
        residual, previous_pre, norm_.weight(), norm_.eps());
    auto next_pre = expansion.pre;
    return {std::move(branch), std::move(next_pre), std::move(expansion)};
}

MlxDeepseekV41HcMetadataResult MlxDeepseekV41Mhc::metadata(const array& residual) const {
    const int batch = residual.shape(0);
    const int tokens = residual.shape(1);
    auto mixes = [&]() {
        if (const auto* tq = function_.tq_weight_ref())
            return tq->rms_matmul(mlx::core::reshape(residual,
                Shape{batch, tokens, static_cast<int>(4 * config_.hidden)}), static_cast<float>(config_.rms_eps));
        const auto* dense = function_.dense_weight_ref();
        if (dense && dense->dtype() == mlx::core::float32) {
            return deepseek_v41_hc_projection(residual, *dense,
                static_cast<float>(config_.rms_eps));
        }
        auto flattened = mlx::core::reshape(
            mlx::core::astype(residual, mlx::core::float32),
            Shape{batch, tokens, static_cast<int>(4 * config_.hidden)});
        auto inverse = mlx::core::rsqrt(
            mlx::core::mean(flattened * flattened, -1, true) +
            static_cast<float>(config_.rms_eps));
        return mlx::core::astype(function_(flattened * inverse), mlx::core::float32);
    }();
    return deepseek_v41_hc_metadata_exact(
        mixes,
        scale_,
        base_,
        static_cast<int>(config_.hc_sinkhorn_iters),
        static_cast<float>(config_.hc_eps));
}

array MlxDeepseekV41Mhc::expand(
    const array& branch,
    const array& residual,
    const MlxDeepseekV41HcMetadataResult& expansion) const {
    return deepseek_v41_hc_post(
        branch,
        residual,
        expansion.post,
        expansion.combination);
}

MlxDeepseekV41HcPostPreResult deepseek_v41_hc_post_collapse_norm(
    const array& branch, const array& residual,
    const MlxDeepseekV41HcMetadataResult& expansion,
    const array& previous_pre, const array& norm, float eps) {
    if (residual.ndim() != 4)
        throw std::invalid_argument("invalid DeepSeek-V4.1 fused Post/Pre rank");
    if (residual.shape(-1) != kOfficialHidden ||
        (residual.dtype() != mlx::core::float16 && residual.dtype() != mlx::core::bfloat16)) {
        auto expanded = deepseek_v41_hc_post(branch, residual, expansion.post, expansion.combination);
        auto normalized = deepseek_v41_hc_collapse_norm(expanded, previous_pre, norm, eps);
        return {std::move(expanded), std::move(normalized)};
    }
    if (residual.shape(2) != 4 || branch.shape() != Shape{residual.shape(0), residual.shape(1), kOfficialHidden} ||
        previous_pre.shape() != Shape{residual.shape(0), residual.shape(1), 4} ||
        expansion.post.shape() != previous_pre.shape() ||
        expansion.combination.shape() != Shape{residual.shape(0), residual.shape(1), 4, 4} ||
        norm.size() != kOfficialHidden)
        throw std::invalid_argument("invalid DeepSeek-V4.1 fused Post/Pre boundary");
    auto output = post_collapse_kernel()({mlx::core::contiguous(residual),
        floating_contiguous(branch),
        float32_contiguous(expansion.post), float32_contiguous(expansion.combination),
        float32_contiguous(previous_pre), float32_contiguous(norm), array({eps})},
        {residual.shape(), branch.shape()}, {residual.dtype(), residual.dtype()},
        {residual.shape(0) * residual.shape(1) * 1024, 1, 1}, {1024, 1, 1},
        {{"T", residual.dtype()}, {"HIDDEN", kOfficialHidden}}, std::nullopt, false, {});
    return {std::move(output[0]), std::move(output[1])};
}

MlxDeepseekV41MhcResult MlxDeepseekV41Mhc::collapse_after(
    const array& branch, const array& residual,
    const MlxDeepseekV41HcMetadataResult& expansion,
    const array& previous_pre, array& expanded) const {
    auto output = deepseek_v41_hc_post_collapse_norm(branch, residual, expansion,
        previous_pre, norm_.weight(), norm_.eps());
    expanded = std::move(output.residual);
    auto next = metadata(expanded);
    auto next_pre = next.pre;
    return {std::move(output.branch), std::move(next_pre), std::move(next)};
}

array MlxDeepseekV41Mhc::expand_sum(
    const array& routed,
    const array& shared,
    const array& residual,
    const MlxDeepseekV41HcMetadataResult& expansion) const {
    return deepseek_v41_hc_post_sum(
        routed,
        shared,
        residual,
        expansion.post,
        expansion.combination);
}

array MlxDeepseekV41Mhc::identity_pre(int batch, int tokens) {
    if (batch <= 0 || tokens <= 0) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 identity pre-mix is empty");
    }
    const array first(
        {1.0f, 0.0f, 0.0f, 0.0f}, Shape{1, 1, 4});
    return mlx::core::broadcast_to(first, Shape{batch, tokens, 4});
}

} // namespace mfq::metal
