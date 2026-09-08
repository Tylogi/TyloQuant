#include "qwen4_ops.h"

#include "mlx_transformer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;
using mlx::core::CompileOptions;
using mlx::core::MathMode;

constexpr int kGatedHcProjectionThreads = 256;

constexpr const char* kGroupedRmsAffineSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint local_thread = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    constexpr uint VALUES_PER_THREAD = 4u;
    constexpr uint SIMD_SIZE = 32u;
    threadgroup float inverse_rms[1];
    threadgroup float local_sums[SIMD_SIZE];

    uint input_offset = row * uint(GROUP_SIZE);
    uint weight_offset =
        (row % uint(GROUP_COUNT)) * uint(GROUP_SIZE);
    uint column = local_thread * VALUES_PER_THREAD;
    float values[VALUES_PER_THREAD] = {
        0.0f, 0.0f, 0.0f, 0.0f};
    float sum_squares = 0.0f;
    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint current = column + item;
        if (current < uint(GROUP_SIZE)) {
            float value = float(input[input_offset + current]);
            values[item] = value;
            sum_squares += value * value;
        }
    }

    // Mirror MLX's non-looped FP32 RMS kernel exactly: four adjacent reads
    // per thread, SIMD reduction, then a 32-slot cross-SIMD reduction.
    sum_squares = simd_sum(sum_squares);
    if (simd_group == 0u)
        local_sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u)
        local_sums[simd_group] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        float total = simd_sum(local_sums[lane]);
        if (lane == 0u) {
            inverse_rms[0] = metal::precise::rsqrt(
                total / float(GROUP_SIZE) + epsilon[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint item = 0u; item < VALUES_PER_THREAD; ++item) {
        uint current = column + item;
        if (current < uint(GROUP_SIZE)) {
            float scaled = (values[item] * inverse_rms[0]) *
                (1.0f + float(weight[weight_offset + current]));
            output[input_offset + current] = T(scaled);
        }
    }
)METAL";

constexpr const char* kGatedHcProjectionSource = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    constexpr uint ROWS_PER_WORKGROUP = 4u;
    constexpr uint DOWN_WORKGROUPS = uint(LOW_RANK) / ROWS_PER_WORKGROUP;
    constexpr uint SIMD_GROUPS = 8u;
    threadgroup float partials[SIMD_GROUPS * ROWS_PER_WORKGROUP];

    bool is_injection = workgroup >= DOWN_WORKGROUPS;
    uint matrix_workgroup = is_injection
        ? workgroup - DOWN_WORKGROUPS
        : workgroup;
    uint row_base = matrix_workgroup * ROWS_PER_WORKGROUP;
    const device T* weights = is_injection
        ? injection_weight
        : down_weight;

    // This is the same 8-way K split and four-adjacent-value traversal used
    // by MLX's BF16/F16 GEMV for the decode geometry. Both projections share
    // normalized, so one launch can produce the 320 low-rank values and the
    // four residual-injection values without a second tiny GEMV dispatch.
    float accumulators[ROWS_PER_WORKGROUP] = {
        0.0f, 0.0f, 0.0f, 0.0f};
    uint column = simd_group * 128u + lane * 4u;
    for (; column < uint(WIDTH); column += 1024u) {
        for (uint item = 0u; item < 4u; ++item) {
            float value = float(normalized[column + item]);
            for (uint row = 0u; row < ROWS_PER_WORKGROUP; ++row) {
                accumulators[row] += float(
                    weights[(row_base + row) * uint(WIDTH) + column + item])
                    * value;
            }
        }
    }

    for (uint row = 0u; row < ROWS_PER_WORKGROUP; ++row) {
        float value = accumulators[row];
        for (ushort delta = 16; delta >= 1; delta >>= 1)
            value += simd_shuffle_down(value, delta);
        if (lane == 0u)
            partials[simd_group * ROWS_PER_WORKGROUP + row] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0u && lane < ROWS_PER_WORKGROUP) {
        float value = partials[lane];
        for (uint group = 1u; group < SIMD_GROUPS; ++group)
            value += partials[group * ROWS_PER_WORKGROUP + lane];
        uint output_row = is_injection
            ? uint(LOW_RANK) + row_base + lane
            : row_base + lane;
        if (is_injection) {
            T projected = T(value);
            T scaled = T(projected / T(HC_COUNT));
            T tail = T(1) /
                (T(1) + metal::exp(metal::abs(scaled)));
            T sigmoid = scaled < T(0) ? tail : T(1) - tail;
            projections[output_row] = T(T(2) * sigmoid);
        } else {
            projections[output_row] = T(value);
        }
    }
)METAL";

constexpr const char* kGatedHcUpCollapseSource = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint feature = threadgroup_position_in_grid.x * 8u + simd_group;
    if (feature >= uint(HIDDEN)) return;

    float accumulators[HC_COUNT];
    for (uint stream = 0u; stream < uint(HC_COUNT); ++stream)
        accumulators[stream] = 0.0f;

    // Match MLX's BF16/F16 GEMV traversal for the Qwen projection: every
    // lane consumes four adjacent low-rank values, and a SIMD group reduces
    // one output feature from every hyper-connection stream.
    constexpr uint VALUES_PER_LANE = 4u;
    constexpr uint BLOCK = VALUES_PER_LANE * 32u;
    for (uint base = 0u; base < uint(LOW_RANK); base += BLOCK) {
        uint low_index = base + lane * VALUES_PER_LANE;
        T activated[VALUES_PER_LANE];
        for (uint item = 0u; item < VALUES_PER_LANE; ++item) {
            if (low_index + item < uint(LOW_RANK)) {
                T value = T(
                    down_projection[low_index + item] / T(HC_COUNT));
                T tail = T(1) /
                    (T(1) + metal::exp(metal::abs(value)));
                T sigmoid = value < T(0) ? tail : T(1) - tail;
                activated[item] = T(value * sigmoid);
            } else {
                activated[item] = T(0);
            }
        }

        for (uint stream = 0u;
             stream < uint(HC_COUNT);
             ++stream) {
            uint row = stream * uint(HIDDEN) + feature;
            uint weight_base =
                row * uint(LOW_RANK) + low_index;
            for (uint item = 0u;
                 item < VALUES_PER_LANE;
                 ++item) {
                if (low_index + item < uint(LOW_RANK)) {
                    accumulators[stream] +=
                        up_weight[weight_base + item] *
                        float(activated[item]);
                }
            }
        }
    }

    for (uint stream = 0u; stream < uint(HC_COUNT); ++stream) {
        for (ushort delta = 16; delta >= 1; delta >>= 1) {
            accumulators[stream] += simd_shuffle_down(
                accumulators[stream], delta);
        }
    }

    if (lane == 0u) {
        // The reference mx.mean on BF16/F16 performs the four additions in
        // activation precision. Preserve those boundaries so this fused
        // kernel is bit-exact with the canonical graph.
        T total = T(0);
        for (uint stream = 0u;
             stream < uint(HC_COUNT);
             ++stream) {
            T projected = T(accumulators[stream]);
            T tail = T(1) /
                (T(1) + metal::exp(metal::abs(projected)));
            T gate = projected < T(0) ? tail : T(1) - tail;
            T product = T(
                gate * normalized[
                    stream * uint(HIDDEN) + feature]);
            total = T(total + product);
        }
        branch[feature] = T(total / T(HC_COUNT));
    }
)METAL";

const mlx::core::fast::CustomKernelFunction& grouped_rms_affine_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_grouped_rms_affine",
            {"input", "weight", "epsilon"},
            {"output"},
            kGroupedRmsAffineSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& gated_hc_projection_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_projection",
            {"normalized", "down_weight", "injection_weight"},
            {"projections"},
            kGatedHcProjectionSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& gated_hc_up_collapse_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_qwen_gated_hc_up_collapse",
            {"down_projection", "up_weight", "normalized"},
            {"branch"},
            kGatedHcUpCollapseSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

bool gated_hc_fast_path_enabled() noexcept {
    static const bool enabled = [] {
        const char* setting =
            std::getenv("MFQ_METAL_QWEN_GATED_HC_FAST");
        return setting == nullptr || std::atoi(setting) != 0;
    }();
    return enabled;
}

bool can_fuse_grouped_rms_affine(
    const array& value,
    const array& weight,
    int group_size) {
    const auto dtype = value.dtype();
    return gated_hc_fast_path_enabled() &&
        (dtype == mlx::core::float16 ||
         dtype == mlx::core::bfloat16) &&
        weight.dtype() == dtype &&
        value.ndim() >= 1 &&
        group_size > 0 && group_size <= 4096 &&
        value.shape(-1) % group_size == 0 &&
        weight.ndim() == 1 &&
        weight.shape(0) == value.shape(-1);
}

array fused_grouped_rms_affine(
    const array& value,
    const array& weight,
    int group_size,
    float eps) {
    constexpr int values_per_thread = 4;
    constexpr int simd_size = 32;
    const int threads_needed =
        (group_size + values_per_thread - 1) / values_per_thread;
    const int simdgroups =
        (threads_needed + simd_size - 1) / simd_size;
    const int threads = simdgroups * simd_size;
    const int rows = static_cast<int>(value.size() / group_size);
    const int group_count = value.shape(-1) / group_size;
    const array epsilon({eps}, Shape{1});
    auto outputs = grouped_rms_affine_kernel()(
        {value, weight, epsilon},
        {value.shape()},
        {value.dtype()},
        {rows * threads, 1, 1},
        {threads, 1, 1},
        {
            {"T", value.dtype()},
            {"GROUP_SIZE", group_size},
            {"GROUP_COUNT", group_count},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

bool can_fuse_gated_hc_projections(
    const array& normalized,
    const array& down_weight,
    const array& injection_weight,
    int hc_count) {
    const auto dtype = normalized.dtype();
    if (!gated_hc_fast_path_enabled() ||
        (dtype != mlx::core::float16 &&
         dtype != mlx::core::bfloat16) ||
        normalized.ndim() != 3 ||
        normalized.shape(0) != 1 ||
        normalized.shape(1) != 1 ||
        down_weight.ndim() != 2 ||
        injection_weight.ndim() != 2 ||
        down_weight.dtype() != dtype ||
        injection_weight.dtype() != dtype ||
        hc_count != 4) {
        return false;
    }
    const int width = normalized.shape(2);
    const int low_rank = down_weight.shape(0);
    return width > 0 && width % 1024 == 0 &&
        low_rank > 0 && low_rank % 4 == 0 &&
        down_weight.shape(1) == width &&
        injection_weight.shape() == Shape{hc_count, width} &&
        width >= 16 * low_rank;
}

struct GatedHcProjections {
    array down;
    array injection;
};

GatedHcProjections fused_gated_hc_projections(
    const array& normalized,
    const array& down_weight,
    const array& injection_weight,
    int hc_count) {
    const int width = normalized.shape(2);
    const int low_rank = down_weight.shape(0);
    const int workgroups = low_rank / 4 + hc_count / 4;
    auto outputs = gated_hc_projection_kernel()(
        {normalized, down_weight, injection_weight},
        {Shape{1, 1, low_rank + hc_count}},
        {normalized.dtype()},
        {workgroups * kGatedHcProjectionThreads, 1, 1},
        {kGatedHcProjectionThreads, 1, 1},
        {
            {"T", normalized.dtype()},
            {"WIDTH", width},
            {"LOW_RANK", low_rank},
            {"HC_COUNT", hc_count},
        },
        std::nullopt,
        false,
        {});
    auto packed = std::move(outputs.front());
    return {
        mlx::core::slice(
            packed,
            Shape{0, 0, 0},
            Shape{1, 1, low_rank}),
        mlx::core::slice(
            packed,
            Shape{0, 0, low_rank},
            Shape{1, 1, low_rank + hc_count}),
    };
}

bool can_fuse_gated_hc_up_collapse(
    const array& down_projection,
    const array& up_weight,
    const array& normalized,
    int hidden_size,
    int hc_count) {
    const auto dtype = normalized.dtype();
    return gated_hc_fast_path_enabled() &&
        (dtype == mlx::core::float16 ||
         dtype == mlx::core::bfloat16) &&
        down_projection.dtype() == dtype &&
        up_weight.dtype() == dtype &&
        normalized.ndim() == 3 &&
        normalized.shape(0) == 1 &&
        normalized.shape(1) == 1 &&
        normalized.shape(2) == hidden_size * hc_count &&
        down_projection.ndim() == 3 &&
        down_projection.shape(0) == 1 &&
        down_projection.shape(1) == 1 &&
        down_projection.shape(2) > 0 &&
        up_weight.ndim() == 2 &&
        up_weight.shape(0) == hidden_size * hc_count &&
        up_weight.shape(1) == down_projection.shape(2) &&
        hc_count > 1 && hc_count <= 8;
}

array fused_gated_hc_up_collapse(
    const array& down_projection,
    const array& up_weight,
    const array& normalized,
    int hidden_size,
    int hc_count) {
    const int workgroups = (hidden_size + 7) / 8;
    auto outputs = gated_hc_up_collapse_kernel()(
        {down_projection, up_weight, normalized},
        {Shape{1, 1, hidden_size}},
        {normalized.dtype()},
        {workgroups * kGatedHcProjectionThreads, 1, 1},
        {kGatedHcProjectionThreads, 1, 1},
        {
            {"T", normalized.dtype()},
            {"HIDDEN", hidden_size},
            {"LOW_RANK", down_projection.shape(2)},
            {"HC_COUNT", hc_count},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array floating(const array& value, mlx::core::Dtype dtype) {
    auto result = value.dtype() == dtype ? value : mlx::core::astype(value, dtype);
    return mlx::core::contiguous(result);
}

void require_rank(const array& value, int rank, const char* name) {
    if (value.ndim() != rank) {
        throw std::invalid_argument(std::string(name) + " has an invalid rank");
    }
}

} // namespace

array qwen4_grouped_rms_norm(
    const array& value,
    const array& weight,
    int group_size,
    float eps) {
    if (value.ndim() == 0 || group_size <= 0 ||
        value.shape(-1) % group_size != 0 ||
        weight.ndim() != 1 || weight.shape(0) != value.shape(-1) ||
        !std::isfinite(eps) || eps <= 0.0f) {
        throw std::invalid_argument("Qwen4 grouped RMSNorm dimensions disagree");
    }
    if (can_fuse_grouped_rms_affine(value, weight, group_size)) {
        return fused_grouped_rms_affine(
            value, weight, group_size, eps);
    }
    const auto source_dtype = value.dtype();
    auto shape = value.shape();
    const int width = shape.back();
    shape.pop_back();
    shape.push_back(width / group_size);
    shape.push_back(group_size);
    auto source = mlx::core::reshape(
        mlx::core::astype(value, mlx::core::float32), shape);
    // Treat every hyper-connection stream as an independent RMSNorm row.
    // MLX's fast primitive keeps the reduction and normalization in one
    // dispatch; the per-stream centred weights remain a separate affine so
    // all groups can retain their own parameters.
    auto normalized = mlx::core::fast::rms_norm(
        source, std::nullopt, eps);
    normalized = mlx::core::reshape(normalized, value.shape()) *
        (array(1.0f) + mlx::core::astype(weight, mlx::core::float32));
    return normalized.dtype() == source_dtype
        ? normalized : mlx::core::astype(normalized, source_dtype);
}

MlxQwen4GatedResidualPre qwen4_gated_residual_pre(
    const array& hyper_input,
    const array& norm_weight,
    const array& down_weight,
    const array& up_weight,
    const std::optional<array>& inject_weight,
    int hidden_size,
    int hc_count,
    float eps) {
    const int width = hidden_size * hc_count;
    if (hidden_size <= 0 || hc_count <= 1 || hyper_input.ndim() < 2 ||
        hyper_input.shape(-1) != width ||
        down_weight.ndim() != 2 || down_weight.shape(1) != width ||
        up_weight.ndim() != 2 || up_weight.shape(0) != width ||
        up_weight.shape(1) != down_weight.shape(0)) {
        throw std::invalid_argument("Qwen4 gated-residual projection mismatch");
    }
    auto normalized = qwen4_grouped_rms_norm(
        hyper_input, norm_weight, hidden_size, eps);
    std::optional<array> fused_injection;
    array down_projection = [&] {
        if (inject_weight && can_fuse_gated_hc_projections(
                normalized,
                down_weight,
                *inject_weight,
                hc_count)) {
            auto projections = fused_gated_hc_projections(
                normalized,
                down_weight,
                *inject_weight,
                hc_count);
            fused_injection = std::move(projections.injection);
            return std::move(projections.down);
        }
        return mlx::core::matmul(
            normalized, mlx::core::transpose(down_weight));
    }();
    const array connection_count(
        static_cast<float>(hc_count), normalized.dtype());
    array mixed = [&] {
        if (can_fuse_gated_hc_up_collapse(
                down_projection,
                up_weight,
                normalized,
                hidden_size,
                hc_count)) {
            return fused_gated_hc_up_collapse(
                down_projection,
                up_weight,
                normalized,
                hidden_size,
                hc_count);
        }
        auto low = down_projection / connection_count;
        low = low * mlx::core::sigmoid(low);
        auto mixing = mlx::core::sigmoid(
            mlx::core::matmul(low, mlx::core::transpose(up_weight)));
        auto stream_shape = hyper_input.shape();
        stream_shape.back() = hc_count;
        stream_shape.push_back(hidden_size);
        return mlx::core::mean(
            mlx::core::reshape(mixing, stream_shape) *
                mlx::core::reshape(normalized, stream_shape),
            -2);
    }();
    std::optional<array> injection;
    if (inject_weight) {
        if (inject_weight->ndim() != 2 ||
            inject_weight->shape() != Shape{hc_count, width}) {
            throw std::invalid_argument("Qwen4 residual injection mismatch");
        }
        if (fused_injection) {
            injection = std::move(*fused_injection);
        } else {
            auto projected = mlx::core::matmul(
                normalized, mlx::core::transpose(*inject_weight));
            injection = array(2.0f, normalized.dtype()) *
                mlx::core::sigmoid(projected / connection_count);
        }
    }
    return {std::move(mixed), hyper_input, std::move(injection)};
}

array qwen4_gated_residual_post(
    const array& branch,
    const array& residual,
    const array& injection,
    int hc_count) {
    if (branch.ndim() < 2 || hc_count <= 1 ||
        residual.shape(-1) != hc_count * branch.shape(-1)) {
        throw std::invalid_argument("Qwen4 gated-residual post mismatch");
    }
    auto expected = branch.shape();
    expected.back() = hc_count;
    if (injection.shape() != expected) {
        throw std::invalid_argument("Qwen4 gated-residual gate mismatch");
    }
    auto update = mlx::core::expand_dims(branch, -2) *
        mlx::core::expand_dims(injection, -1);
    return residual + mlx::core::reshape(update, residual.shape());
}

array qwen4_qsa_block_scores(
    const array& query,
    const array& pooled_keys) {
    require_rank(query, 4, "Qwen4 index query");
    require_rank(pooled_keys, 3, "Qwen4 pooled key");
    if (query.shape(0) != pooled_keys.shape(0) ||
        query.shape(3) != pooled_keys.shape(2)) {
        throw std::invalid_argument("Qwen4 QSA score dimensions disagree");
    }
    auto keys = mlx::core::expand_dims(
        mlx::core::transpose(
            mlx::core::astype(pooled_keys, mlx::core::float32), {0, 2, 1}),
        1);
    auto scores = mlx::core::matmul(
        mlx::core::astype(query, mlx::core::float32), keys);
    scores = mlx::core::maximum(scores, array(0.0f));
    return mlx::core::sum(scores, -2) /
        std::sqrt(static_cast<float>(query.shape(3)));
}

array qwen4_dense_gqa_attention(
    const array& query,
    const array& key,
    const array& value,
    int query_offset) {
    require_rank(query, 4, "Qwen4 attention query");
    require_rank(key, 4, "Qwen4 attention key");
    if (value.shape() != key.shape() || query.shape(0) != key.shape(0) ||
        query.shape(1) % key.shape(1) != 0 ||
        query.shape(3) != key.shape(3) || query_offset < 0 ||
        query_offset + query.shape(2) > key.shape(2)) {
        throw std::invalid_argument("Qwen4 dense GQA dimensions disagree");
    }
    const int tokens = query.shape(2);
    const int keys = key.shape(2);
    if (tokens == 1 && query_offset + 1 == keys) {
        auto output = scaled_dot_product_attention(
            query,
            key,
            value,
            false,
            1.0f / std::sqrt(static_cast<float>(query.shape(3))));
        return mlx::core::transpose(output, {0, 2, 1, 3});
    }
    auto key_positions = mlx::core::reshape(
        mlx::core::arange(0, keys, 1, mlx::core::int32), Shape{1, keys});
    auto query_positions = mlx::core::reshape(
        mlx::core::arange(
            query_offset, query_offset + tokens, 1, mlx::core::int32),
        Shape{tokens, 1});
    auto mask = mlx::core::expand_dims(
        mlx::core::expand_dims(key_positions <= query_positions, 0), 0);
    auto output = scaled_dot_product_attention(
        query, key, value, false,
        1.0f / std::sqrt(static_cast<float>(query.shape(3))), mask);
    return mlx::core::transpose(output, {0, 2, 1, 3});
}

} // namespace mfq::metal
