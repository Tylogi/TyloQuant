#include "mlx_sampling.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kThreads = 256;
constexpr int kMaxTopK = 1024;
constexpr int kDirectTopK = 64;
constexpr int kHierarchicalTopK = 128;
constexpr int kTopKBlock = 1024;

constexpr const char* kGreedySource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float values[256];
    threadgroup int indices[256];
    float best = -FLT_MAX;
    int best_index = 0;
    uint offset = row * uint(VOCAB);
    for (uint token = tid; token < uint(VOCAB); token += 256u) {
        float value = float(logits[offset + token]);
        value = isnan(value) ? -FLT_MAX : value;
        if (value > best || (value == best && int(token) < best_index)) {
            best = value;
            best_index = int(token);
        }
    }
    values[tid] = best;
    indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float other = values[tid + stride];
            int other_index = indices[tid + stride];
            if (
                other > values[tid]
                || (other == values[tid] && other_index < indices[tid])
            ) {
                values[tid] = other;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        output[row] = indices[0];
    }
)METAL";

constexpr const char* kSoftmaxSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float reduction[256];
    threadgroup float maximum;
    uint offset = row * uint(VOCAB);
    float local_max = -FLT_MAX;
    for (uint token = tid; token < uint(VOCAB); token += 256u) {
        float value = float(logits[offset + token]) / params[0];
        value = isnan(value) ? -FLT_MAX : value;
        local_max = max(local_max, value);
    }
    reduction[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            reduction[tid] = max(reduction[tid], reduction[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        maximum = reduction[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint token = tid; token < uint(VOCAB); token += 256u) {
        float value = float(logits[offset + token]) / params[0];
        value = isnan(value) ? -FLT_MAX : value;
        local_sum += exp(value - maximum);
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float uniform = clamp(random[row], 0.0f, 0.99999994f);
        float target = uniform * reduction[0];
        float cumulative = 0.0f;
        int chosen = int(VOCAB) - 1;
        for (uint token = 0u; token < uint(VOCAB); ++token) {
            float value = float(logits[offset + token]) / params[0];
            value = isnan(value) ? -FLT_MAX : value;
            cumulative += exp(value - maximum);
            if (cumulative >= target) {
                chosen = int(token);
                break;
            }
        }
        output[row] = chosen;
    }
)METAL";

constexpr const char* kTopKSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;

    // Each lane scans its vocabulary stripe once and keeps a stable, sorted
    // local top-k.  The previous implementation scanned the entire vocabulary
    // once per output rank and also searched the already-selected ranks.
    float local_values[TOP_K];
    int local_indices[TOP_K];
    for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
        local_values[rank] = -INFINITY;
        local_indices[rank] = int(VOCAB);
    }

    uint offset = row * uint(VOCAB);
    for (uint token = tid; token < uint(VOCAB); token += 256u) {
        float value = float(logits[offset + token]) / params[0];
        value = isnan(value) ? -FLT_MAX : value;
        int token_index = int(token);
        uint last = uint(TOP_K) - 1u;
        bool enters =
            value > local_values[last]
            || (
                value == local_values[last]
                && token_index < local_indices[last]
            );
        if (!enters) {
            continue;
        }

        uint insert = last;
        while (insert > 0u) {
            float previous_value = local_values[insert - 1u];
            int previous_index = local_indices[insert - 1u];
            bool before =
                value > previous_value
                || (
                    value == previous_value
                    && token_index < previous_index
                );
            if (!before) {
                break;
            }
            local_values[insert] = previous_value;
            local_indices[insert] = previous_index;
            --insert;
        }
        local_values[insert] = value;
        local_indices[insert] = token_index;
    }

    // Merge the 32 sorted lane lists in each SIMD group.  Only TOP_K values
    // per SIMD group survive, so the threadgroup merge remains small even for
    // a 248k-token vocabulary.
    threadgroup float simd_values[8 * TOP_K];
    threadgroup int simd_indices[8 * TOP_K];
    uint local_rank = 0u;
    for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
        float best = local_values[local_rank];
        int best_index = local_indices[local_rank];
        for (uint stride = 16u; stride > 0u; stride >>= 1u) {
            float other = simd_shuffle_down(best, stride);
            int other_index = simd_shuffle_down(best_index, stride);
            if (
                lane < stride
                && (
                    other > best
                    || (other == best && other_index < best_index)
                )
            ) {
                best = other;
                best_index = other_index;
            }
        }
        best = simd_broadcast_first(best);
        best_index = simd_broadcast_first(best_index);
        if (local_indices[local_rank] == best_index) {
            ++local_rank;
        }
        if (lane == 0u) {
            uint destination =
                simd_group * uint(TOP_K) + rank;
            simd_values[destination] = best;
            simd_indices[destination] = best_index;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float top_values[TOP_K];
    threadgroup int top_indices[TOP_K];
    threadgroup float probabilities[TOP_K];

    if (tid == 0u) {
        // Merge the eight SIMD-group lists.  Ties remain deterministic:
        // greater logit first, then the lower token id.
        uint simd_ranks[8] = {
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        };
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            float best = -INFINITY;
            int best_index = int(VOCAB);
            uint best_group = 0u;
            for (uint group = 0u; group < 8u; ++group) {
                uint candidate =
                    group * uint(TOP_K) + simd_ranks[group];
                float value = simd_values[candidate];
                int index = simd_indices[candidate];
                if (
                    value > best
                    || (value == best && index < best_index)
                ) {
                    best = value;
                    best_index = index;
                    best_group = group;
                }
            }
            top_values[rank] = best;
            top_indices[rank] = best_index;
            ++simd_ranks[best_group];
        }

        float maximum = top_values[0];
        float total = 0.0f;
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            float probability = exp(top_values[rank] - maximum);
            probabilities[rank] = probability;
            total += probability;
        }
        uint keep = uint(TOP_K);
        float keep_sum = total;
        if (params[1] > 0.0f && params[1] < 1.0f) {
            float cutoff = params[1] * total;
            float cumulative = 0.0f;
            for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
                cumulative += probabilities[rank];
                if (cumulative >= cutoff) {
                    keep = rank + 1u;
                    keep_sum = cumulative;
                    break;
                }
            }
        }
        float uniform = clamp(random[row], 0.0f, 0.99999994f);
        float target = uniform * keep_sum;
        float cumulative = 0.0f;
        int chosen = top_indices[keep - 1u];
        for (uint rank = 0u; rank < keep; ++rank) {
            cumulative += probabilities[rank];
            if (cumulative >= target) {
                chosen = top_indices[rank];
                break;
            }
        }
        output[row] = chosen;
    }
)METAL";

// Exact single-row top-k selection for the common 65..128 range. Each pass
// sorts independent 1024-value tiles and retains 128 candidates. Repeating
// the pass cannot discard a member of the global top-k, while avoiding the
// full-vocabulary argpartition/argsort graph used by the generic fallback.
constexpr const char* kHierarchicalTopKFirstSource = R"METAL(
    constexpr uint BLOCK = 1024u;
    constexpr uint KEEP = 128u;
    uint group = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    threadgroup float tile_values[BLOCK];
    threadgroup int tile_indices[BLOCK];

    uint begin = group * BLOCK;
    for (uint offset = tid; offset < BLOCK; offset += 256u) {
        uint index = begin + offset;
        float value = index < uint(COUNT)
            ? float(logits[index])
            : -INFINITY;
        tile_values[offset] = isnan(value) ? -INFINITY : value;
        tile_indices[offset] = index < uint(COUNT)
            ? int(index)
            : INT_MAX;
    }

    for (uint width = 2u; width <= BLOCK; width <<= 1u) {
        for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint left = tid; left < BLOCK; left += 256u) {
                uint right = left ^ stride;
                if (right <= left) continue;
                float left_value = tile_values[left];
                float right_value = tile_values[right];
                int left_index = tile_indices[left];
                int right_index = tile_indices[right];
                bool right_before_left =
                    right_value > left_value ||
                    (right_value == left_value && right_index < left_index);
                bool left_before_right =
                    left_value > right_value ||
                    (left_value == right_value && left_index < right_index);
                bool descending = (left & width) == 0u;
                if ((descending && right_before_left) ||
                    (!descending && left_before_right)) {
                    tile_values[left] = right_value;
                    tile_values[right] = left_value;
                    tile_indices[left] = right_index;
                    tile_indices[right] = left_index;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEEP) {
        uint output = group * KEEP + tid;
        scores_out[output] = tile_values[tid];
        indices_out[output] = tile_indices[tid];
    }
)METAL";

constexpr const char* kHierarchicalTopKMergeSource = R"METAL(
    constexpr uint BLOCK = 1024u;
    constexpr uint KEEP = 128u;
    uint group = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    threadgroup float tile_values[BLOCK];
    threadgroup int tile_indices[BLOCK];

    uint begin = group * BLOCK;
    for (uint offset = tid; offset < BLOCK; offset += 256u) {
        uint index = begin + offset;
        tile_values[offset] = index < uint(COUNT)
            ? scores_in[index]
            : -INFINITY;
        tile_indices[offset] = index < uint(COUNT)
            ? indices_in[index]
            : INT_MAX;
    }

    for (uint width = 2u; width <= BLOCK; width <<= 1u) {
        for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint left = tid; left < BLOCK; left += 256u) {
                uint right = left ^ stride;
                if (right <= left) continue;
                float left_value = tile_values[left];
                float right_value = tile_values[right];
                int left_index = tile_indices[left];
                int right_index = tile_indices[right];
                bool right_before_left =
                    right_value > left_value ||
                    (right_value == left_value && right_index < left_index);
                bool left_before_right =
                    left_value > right_value ||
                    (left_value == right_value && left_index < right_index);
                bool descending = (left & width) == 0u;
                if ((descending && right_before_left) ||
                    (!descending && left_before_right)) {
                    tile_values[left] = right_value;
                    tile_values[right] = left_value;
                    tile_indices[left] = right_index;
                    tile_indices[right] = left_index;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEEP) {
        uint output = group * KEEP + tid;
        scores_out[output] = tile_values[tid];
        indices_out[output] = tile_indices[tid];
    }
)METAL";

constexpr const char* kHierarchicalTopKFinalSource = R"METAL(
    constexpr uint BLOCK = 1024u;
    uint tid = thread_index_in_threadgroup;
    threadgroup float tile_values[BLOCK];
    threadgroup int tile_indices[BLOCK];

    for (uint index = tid; index < BLOCK; index += 256u) {
        tile_values[index] = index < uint(COUNT)
            ? scores[index]
            : -INFINITY;
        tile_indices[index] = index < uint(COUNT)
            ? indices[index]
            : INT_MAX;
    }
    for (uint width = 2u; width <= BLOCK; width <<= 1u) {
        for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint left = tid; left < BLOCK; left += 256u) {
                uint right = left ^ stride;
                if (right <= left) continue;
                float left_value = tile_values[left];
                float right_value = tile_values[right];
                int left_index = tile_indices[left];
                int right_index = tile_indices[right];
                bool right_before_left =
                    right_value > left_value ||
                    (right_value == left_value && right_index < left_index);
                bool left_before_right =
                    left_value > right_value ||
                    (left_value == right_value && left_index < right_index);
                bool descending = (left & width) == 0u;
                if ((descending && right_before_left) ||
                    (!descending && left_before_right)) {
                    tile_values[left] = right_value;
                    tile_values[right] = left_value;
                    tile_indices[left] = right_index;
                    tile_indices[right] = left_index;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float inverse_temperature = 1.0f / params[0];
        float maximum = tile_values[0] * inverse_temperature;
        float probabilities[TOP_K];
        float total = 0.0f;
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            float probability = exp(
                tile_values[rank] * inverse_temperature - maximum);
            probabilities[rank] = probability;
            total += probability;
        }
        uint keep = uint(TOP_K);
        float keep_sum = total;
        if (params[1] > 0.0f && params[1] < 1.0f) {
            float cutoff = params[1] * total;
            float cumulative = 0.0f;
            for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
                cumulative += probabilities[rank];
                if (cumulative >= cutoff) {
                    keep = rank + 1u;
                    keep_sum = cumulative;
                    break;
                }
            }
        }
        float uniform = clamp(random[0], 0.0f, 0.99999994f);
        float target = uniform * keep_sum;
        float cumulative = 0.0f;
        int chosen = tile_indices[keep - 1u];
        for (uint rank = 0u; rank < keep; ++rank) {
            cumulative += probabilities[rank];
            if (cumulative >= target) {
                chosen = tile_indices[rank];
                break;
            }
        }
        output[0] = chosen;
    }
)METAL";

constexpr const char* kSortedSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float reduction[256];
    uint offset = row * uint(COUNT);
    float maximum = scores[offset];
    float local_sum = 0.0f;
    for (uint rank = tid; rank < uint(COUNT); rank += 256u) {
        local_sum += exp(scores[offset + rank] - maximum);
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float total = reduction[0];
        uint keep = uint(COUNT);
        float keep_sum = total;
        if (params[0] > 0.0f && params[0] < 1.0f) {
            float cutoff = params[0] * total;
            float cumulative = 0.0f;
            for (uint rank = 0u; rank < uint(COUNT); ++rank) {
                cumulative += exp(scores[offset + rank] - maximum);
                if (cumulative >= cutoff) {
                    keep = rank + 1u;
                    keep_sum = cumulative;
                    break;
                }
            }
        }
        float uniform = clamp(random[row], 0.0f, 0.99999994f);
        float target = uniform * keep_sum;
        float cumulative = 0.0f;
        int chosen = indices[offset + keep - 1u];
        for (uint rank = 0u; rank < keep; ++rank) {
            cumulative += exp(scores[offset + rank] - maximum);
            if (cumulative >= target) {
                chosen = indices[offset + rank];
                break;
            }
        }
        output[row] = chosen;
    }
)METAL";

constexpr const char* kTokenCountsSource = R"METAL(
    uint token = thread_position_in_grid.x;
    if (token >= uint(VOCAB)) {
        return;
    }
    int value = counts[token];
    for (uint index = 0u; index < uint(TOKENS); ++index) {
        value += int(token_ids[index] == int(token));
    }
    output[token] = value;
)METAL";

constexpr const char* kPenaltiesSource = R"METAL(
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }
    uint token = index % uint(VOCAB);
    int count = counts[token];
    float value = float(logits[index]);
    if (count > 0) {
        if (params[2] != 1.0f) {
            value = value < 0.0f
                ? value * params[2]
                : value / params[2];
        }
        value -= params[0] + params[1] * float(count);
    }
    output[index] = T(value);
)METAL";

mlx::core::fast::CustomKernelFunction make_kernel(
    const char* name,
    std::vector<std::string> inputs,
    const char* source,
    bool fast_math = true) {
    CompileOptions options;
    if (fast_math) {
        options.math_mode = MathMode::Fast;
    }
    return mlx::core::fast::metal_kernel(
        name,
        inputs,
        {"output"},
        source,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& greedy_kernel() {
    static const auto kernel =
        make_kernel("mfq_cpp_sample_greedy", {"logits"}, kGreedySource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& softmax_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_sample_softmax",
        {"logits", "random", "params"},
        kSoftmaxSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& top_k_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_sample_top_k_top_p",
        {"logits", "random", "params"},
        kTopKSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
hierarchical_top_k_first_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_sample_top_k_128_first",
            {"logits"},
            {"scores_out", "indices_out"},
            kHierarchicalTopKFirstSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
hierarchical_top_k_merge_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_sample_top_k_128_merge",
            {"scores_in", "indices_in"},
            {"scores_out", "indices_out"},
            kHierarchicalTopKMergeSource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
hierarchical_top_k_final_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_sample_top_k_128_final",
        {"scores", "indices", "random", "params"},
        kHierarchicalTopKFinalSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& sorted_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_sample_sorted_top_p",
        {"scores", "indices", "random", "params"},
        kSortedSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& token_counts_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_sample_token_counts_add",
        {"counts", "token_ids"},
        kTokenCountsSource,
        false);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& penalties_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_sample_apply_penalties",
        {"logits", "counts", "params"},
        kPenaltiesSource);
    return kernel;
}

struct LogitsView {
    array values;
    Shape prefix;
    int rows;
    int vocab;
};

int checked_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            std::string("sampling ") + name + " exceeds MLX limits");
    }
    return static_cast<int>(value);
}

LogitsView normalize_logits(const array& logits) {
    if (logits.ndim() < 1 || logits.shape(-1) <= 0) {
        throw std::invalid_argument(
            "sampling logits must end in a non-empty vocabulary");
    }
    auto values = logits;
    if (values.dtype() != mlx::core::float16 &&
        values.dtype() != mlx::core::float32) {
        values = mlx::core::astype(values, mlx::core::float32);
    }
    values = mlx::core::contiguous(values);
    const int vocab = values.shape(-1);
    if (values.size() % static_cast<std::size_t>(vocab) != 0) {
        throw std::invalid_argument("invalid sampling logits shape");
    }
    const int rows = checked_int(
        values.size() / static_cast<std::size_t>(vocab),
        "row count");
    if (rows <= 0) {
        throw std::invalid_argument("sampling logits cannot have zero rows");
    }
    Shape prefix(values.shape().begin(), values.shape().end() - 1);
    values = mlx::core::reshape(values, Shape{rows, vocab});
    return {
        std::move(values),
        std::move(prefix),
        rows,
        vocab,
    };
}

array normalize_random(const array& random, int rows) {
    auto values = mlx::core::astype(random, mlx::core::float32);
    values = mlx::core::contiguous(
        mlx::core::reshape(
            values,
            Shape{checked_int(values.size(), "random value count")}));
    if (values.size() != static_cast<std::size_t>(rows)) {
        throw std::invalid_argument(
            "sampling random values must contain one item per row");
    }
    return values;
}

array make_cpu_random(std::mt19937_64& rng, int rows) {
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::vector<float> values(static_cast<std::size_t>(rows));
    for (auto& value : values) {
        value = uniform(rng);
    }
    return array(values.begin(), Shape{rows}, mlx::core::float32);
}

array make_seeded_random(std::uint64_t seed, int rows) {
    std::mt19937_64 rng(seed);
    return make_cpu_random(rng, rows);
}

std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
base_templates(const LogitsView& view) {
    return {
        {"T", view.values.dtype()},
        {"ROWS", view.rows},
        {"VOCAB", view.vocab},
    };
}

array run_sorted(
    const LogitsView& view,
    const array& random,
    double temperature,
    int top_k,
    double top_p) {
    auto scores =
        mlx::core::astype(view.values, mlx::core::float32) /
        temperature;
    scores = mlx::core::where(
        mlx::core::isnan(scores),
        mlx::core::full_like(
            scores,
            -std::numeric_limits<float>::infinity()),
        scores);
    const int count =
        top_k <= 0 ? view.vocab : std::min(top_k, view.vocab);
    auto order = [&]() {
        if (count >= view.vocab) {
            return mlx::core::flip(
            mlx::core::argsort(scores, -1),
            -1);
        }
        const int start = view.vocab - count;
        auto candidates = mlx::core::slice(
            mlx::core::argpartition(scores, start, -1),
            Shape{0, start},
            Shape{view.rows, view.vocab});
        candidates = mlx::core::take_along_axis(
            candidates,
            mlx::core::argsort(candidates, -1),
            -1);
        auto candidate_scores = mlx::core::take_along_axis(
            scores,
            candidates,
            -1);
        auto candidate_order = mlx::core::argsort(
            -candidate_scores,
            -1);
        return mlx::core::take_along_axis(
            candidates,
            candidate_order,
            -1);
    }();
    order = mlx::core::contiguous(
        mlx::core::astype(order, mlx::core::int32));
    auto ordered = mlx::core::contiguous(
        mlx::core::take_along_axis(scores, order, -1));
    const array params(
        {static_cast<float>(top_p)},
        mlx::core::float32);
    auto outputs = sorted_kernel()(
        {ordered, order, random, params},
        {Shape{view.rows}},
        {mlx::core::int32},
        {view.rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        {
            {"ROWS", view.rows},
            {"COUNT", count},
        },
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(outputs.front(), view.prefix);
}

array run_hierarchical_top_k(
    const LogitsView& view,
    const array& random,
    double temperature,
    int top_k,
    double top_p) {
    if (view.rows != 1 || top_k < 1 ||
        top_k > kHierarchicalTopK) {
        throw std::invalid_argument(
            "hierarchical top-k requires one row and top_k <= 128");
    }
    int count = view.vocab;
    int groups = (count + kTopKBlock - 1) / kTopKBlock;
    auto selected = hierarchical_top_k_first_kernel()(
        {view.values},
        {
            Shape{groups * kHierarchicalTopK},
            Shape{groups * kHierarchicalTopK},
        },
        {mlx::core::float32, mlx::core::int32},
        {groups * kThreads, 1, 1},
        {kThreads, 1, 1},
        {{"COUNT", count}},
        std::nullopt,
        false,
        {});
    auto scores = std::move(selected.at(0));
    auto indices = std::move(selected.at(1));
    count = groups * kHierarchicalTopK;

    while (count > kTopKBlock) {
        groups = (count + kTopKBlock - 1) / kTopKBlock;
        selected = hierarchical_top_k_merge_kernel()(
            {scores, indices},
            {
                Shape{groups * kHierarchicalTopK},
                Shape{groups * kHierarchicalTopK},
            },
            {mlx::core::float32, mlx::core::int32},
            {groups * kThreads, 1, 1},
            {kThreads, 1, 1},
            {{"COUNT", count}},
            std::nullopt,
            false,
            {});
        scores = std::move(selected.at(0));
        indices = std::move(selected.at(1));
        count = groups * kHierarchicalTopK;
    }

    const array params(
        {
            static_cast<float>(temperature),
            static_cast<float>(top_p),
        },
        mlx::core::float32);
    auto output = hierarchical_top_k_final_kernel()(
        {scores, indices, random, params},
        {Shape{1}},
        {mlx::core::int32},
        {kThreads, 1, 1},
        {kThreads, 1, 1},
        {
            {"COUNT", count},
            {"TOP_K", top_k},
        },
        std::nullopt,
        false,
        {}).front();
    return mlx::core::reshape(std::move(output), view.prefix);
}

} // namespace

array sample_greedy(const array& logits) {
    auto view = normalize_logits(logits);
    auto outputs = greedy_kernel()(
        {view.values},
        {Shape{view.rows}},
        {mlx::core::int32},
        {view.rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        base_templates(view),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        outputs.front(),
        std::move(view.prefix));
}

array sample_softmax(
    const array& logits,
    const array& random,
    double temperature) {
    if (!std::isfinite(temperature) || temperature <= 0.0) {
        throw std::invalid_argument(
            "temperature must be finite and positive");
    }
    auto view = normalize_logits(logits);
    auto uniforms = normalize_random(random, view.rows);
    const array params(
        {static_cast<float>(temperature)},
        mlx::core::float32);
    auto outputs = softmax_kernel()(
        {view.values, uniforms, params},
        {Shape{view.rows}},
        {mlx::core::int32},
        {view.rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        base_templates(view),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        outputs.front(),
        std::move(view.prefix));
}

array sample_top_k_top_p(
    const array& logits,
    const array& random,
    double temperature,
    int top_k,
    double top_p) {
    if (!std::isfinite(temperature) || temperature <= 0.0) {
        throw std::invalid_argument(
            "temperature must be finite and positive");
    }
    if (!std::isfinite(top_p) || top_p <= 0.0 || top_p > 1.0) {
        throw std::invalid_argument("top_p must be in (0,1]");
    }
    auto view = normalize_logits(logits);
    if (top_k < 1 || top_k > std::min(view.vocab, kMaxTopK)) {
        throw std::invalid_argument(
            "top_k must be in [1,min(vocab,1024)]");
    }
    auto uniforms = normalize_random(random, view.rows);
    if (view.rows == 1 && view.vocab > kTopKBlock &&
        top_k <= kHierarchicalTopK) {
        return run_hierarchical_top_k(
            view,
            uniforms,
            temperature,
            top_k,
            top_p);
    }
    if (top_k > kDirectTopK) {
        return run_sorted(
            view,
            uniforms,
            temperature,
            top_k,
            top_p);
    }

    const array params(
        {
            static_cast<float>(temperature),
            static_cast<float>(top_p),
        },
        mlx::core::float32);
    auto templates = base_templates(view);
    templates.push_back({"TOP_K", top_k});
    auto outputs = top_k_kernel()(
        {view.values, uniforms, params},
        {Shape{view.rows}},
        {mlx::core::int32},
        {view.rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        outputs.front(),
        std::move(view.prefix));
}

array sample(
    const array& logits,
    const MlxSamplingParams& params,
    const std::optional<array>& random) {
    if (params.top_k < 0) {
        throw std::invalid_argument("top_k must be non-negative");
    }
    if (params.greedy()) {
        return sample_greedy(logits);
    }
    if (!std::isfinite(params.top_p) ||
        params.top_p <= 0.0 ||
        params.top_p > 1.0) {
        throw std::invalid_argument("top_p must be in (0,1]");
    }

    const auto view = normalize_logits(logits);
    const auto uniforms = random.has_value()
        ? normalize_random(*random, view.rows)
        : make_seeded_random(params.seed, view.rows);
    if (params.top_k > 0) {
        return sample_top_k_top_p(
            logits,
            uniforms,
            params.temperature,
            params.top_k,
            params.top_p);
    }
    if (params.top_p >= 1.0) {
        return sample_softmax(
            logits,
            uniforms,
            params.temperature);
    }
    if (!std::isfinite(params.temperature) ||
        params.temperature <= 0.0) {
        throw std::invalid_argument(
            "temperature must be finite and positive");
    }
    return run_sorted(
        view,
        uniforms,
        params.temperature,
        0,
        params.top_p);
}

array sample_token_counts_add(
    const array& counts,
    const array& tokens) {
    auto current = mlx::core::contiguous(
        mlx::core::reshape(
            mlx::core::astype(counts, mlx::core::int32),
            Shape{checked_int(counts.size(), "token count size")}));
    if (current.size() == 0) {
        throw std::invalid_argument("token counts cannot be empty");
    }
    auto token_ids = mlx::core::contiguous(
        mlx::core::reshape(
            mlx::core::astype(tokens, mlx::core::int32),
            Shape{checked_int(tokens.size(), "token id count")}));
    const int vocab = checked_int(current.size(), "vocabulary size");
    const int token_count = checked_int(
        token_ids.size(),
        "token id count");
    auto outputs = token_counts_kernel()(
        {current, token_ids},
        {Shape{vocab}},
        {mlx::core::int32},
        {vocab, 1, 1},
        {std::min(kThreads, vocab), 1, 1},
        {
            {"VOCAB", vocab},
            {"TOKENS", token_count},
        },
        std::nullopt,
        false,
        {});
    return outputs.front();
}

array sample_apply_penalties(
    const array& logits,
    const array& counts,
    double presence_penalty,
    double frequency_penalty,
    double repetition_penalty) {
    const double parameters[] = {
        presence_penalty,
        frequency_penalty,
        repetition_penalty,
    };
    if (!std::all_of(
            std::begin(parameters),
            std::end(parameters),
            [](double value) { return std::isfinite(value); })) {
        throw std::invalid_argument(
            "sampling penalties must be finite");
    }
    if (repetition_penalty <= 0.0) {
        throw std::invalid_argument(
            "repetition_penalty must be positive");
    }
    auto view = normalize_logits(logits);
    auto token_counts = mlx::core::contiguous(
        mlx::core::reshape(
            mlx::core::astype(counts, mlx::core::int32),
            Shape{checked_int(counts.size(), "penalty count size")}));
    if (token_counts.size() !=
        static_cast<std::size_t>(view.vocab)) {
        throw std::invalid_argument(
            "penalty counts must match the vocabulary size");
    }
    const array params(
        {
            static_cast<float>(presence_penalty),
            static_cast<float>(frequency_penalty),
            static_cast<float>(repetition_penalty),
        },
        mlx::core::float32);
    const int size = checked_int(
        view.values.size(),
        "penalty logits size");
    auto outputs = penalties_kernel()(
        {view.values, token_counts, params},
        {Shape{view.rows, view.vocab}},
        {view.values.dtype()},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {
            {"T", view.values.dtype()},
            {"SIZE", size},
            {"VOCAB", view.vocab},
        },
        std::nullopt,
        false,
        {});
    Shape output_shape = std::move(view.prefix);
    output_shape.push_back(view.vocab);
    return mlx::core::reshape(
        outputs.front(),
        std::move(output_shape));
}

MlxSampler::MlxSampler(MlxSamplingParams params)
    : params_(std::move(params)),
      rng_(params_.seed) {
    if (params_.top_k < 0) {
        throw std::invalid_argument("top_k must be non-negative");
    }
}

void MlxSampler::reset_seed(std::uint64_t seed) {
    params_.seed = seed;
    rng_.seed(seed);
}

array MlxSampler::next_random(const array& logits) {
    const auto view = normalize_logits(logits);
    return make_cpu_random(rng_, view.rows);
}

array MlxSampler::sample(const array& logits) {
    if (params_.greedy()) {
        return mfq::metal::sample(logits, params_);
    }
    return mfq::metal::sample(
        logits,
        params_,
        next_random(logits));
}

array MlxSampler::sample(
    const array& logits,
    const array& counts) {
    if (!params_.has_penalties()) {
        return sample(logits);
    }
    return sample(apply_penalties(logits, counts));
}

array MlxSampler::apply_penalties(
    const array& logits,
    const array& counts) const {
    return sample_apply_penalties(
        logits,
        counts,
        params_.presence_penalty,
        params_.frequency_penalty,
        params_.repetition_penalty);
}

} // namespace mfq::metal
