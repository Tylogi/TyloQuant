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
    const uint ROWS = uint(logits_shape[0]);
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    if (row >= uint(ROWS)) {
        return;
    }
    threadgroup float transformed[EXPERTS];
    threadgroup float partial[256];
    threadgroup uint partial_ids[8];
    threadgroup float selected_weights[TOP_K];
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
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

    // Every thread scans a strided expert subset, then the eight SIMD groups
    // reduce their candidates in parallel. This preserves the exact stable
    // lower-expert-ID tie break without making thread 0 rescan every expert
    // for every selected route.
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
        float best_score = -INFINITY;
        uint best_expert = uint(EXPERTS);
        for (uint expert = tid; expert < uint(EXPERTS); expert += 256u) {
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
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            float other_score = simd_shuffle_down(best_score, offset);
            uint other_expert = simd_shuffle_down(best_expert, offset);
            if (
                lane + offset < 32u &&
                (other_score > best_score ||
                 (other_score == best_score &&
                  other_expert < best_expert))
            ) {
                best_score = other_score;
                best_expert = other_expert;
            }
        }
        if (lane == 0u) {
            partial[simd_group] = best_score;
            partial_ids[simd_group] = best_expert;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            best_score = partial[0];
            best_expert = partial_ids[0];
            for (uint group = 1u; group < 8u; ++group) {
                float score = partial[group];
                uint expert = partial_ids[group];
                if (
                    score > best_score ||
                    (score == best_score && expert < best_expert)
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
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

constexpr const char* kSoftmaxTopKRowSource = R"METAL(
    constexpr uint PER_LANE =
        (uint(EXPERTS) + 31u) / 32u;
    uint lane = thread_index_in_simdgroup;
#if MFQ_TOPK_ROUTE_GROUPS
    uint row = simdgroup_index_in_threadgroup;
    bool active = row < uint(logits_shape[0]);
#ifndef MFQ_TOPK_MAX_ROWS
#define MFQ_TOPK_MAX_ROWS 6
#endif
    threadgroup int selected_routes[uint(MFQ_TOPK_MAX_ROWS) * uint(TOP_K)];
#else
    uint row = threadgroup_position_in_grid.y;
    constexpr bool active = true;
#endif
    float values[PER_LANE];
    bool taken[PER_LANE];
    for (uint local = 0u; local < PER_LANE; ++local) {
        uint expert = lane + local * 32u;
        float value = active && expert < uint(EXPERTS)
            ? float(logits[row * uint(EXPERTS) + expert])
            : -INFINITY;
        values[local] = isnan(value) ? -FLT_MAX : value;
        taken[local] = false;
    }

    float selected[TOP_K];
    uint selected_ids[TOP_K];
    for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
        float best = -INFINITY;
        uint best_id = uint(EXPERTS);
        uint best_local = 0u;
        for (uint local = 0u; local < PER_LANE; ++local) {
            uint expert = lane + local * 32u;
            float value = values[local];
            if (!taken[local] && expert < uint(EXPERTS) &&
                (value > best ||
                 (value == best && expert < best_id))) {
                best = value;
                best_id = expert;
                best_local = local;
            }
        }
        float global_best = simd_max(best);
        uint candidate = best == global_best
            ? best_id
            : uint(EXPERTS);
        uint global_id = simd_min(candidate);
        if (best == global_best && best_id == global_id) {
            taken[best_local] = true;
        }
        selected[rank] = global_best;
        selected_ids[rank] = global_id;
    }

    if (lane == 0u && active) {
        float maximum = selected[0];
        float denominator = 0.0f;
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            selected[rank] = exp(selected[rank] - maximum);
            denominator += selected[rank];
        }
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            ids[row * uint(TOP_K) + rank] = int(selected_ids[rank]);
            weights[row * uint(TOP_K) + rank] = selected[rank] / denominator;
#if MFQ_TOPK_ROUTE_GROUPS
            selected_routes[row * uint(TOP_K) + rank] = int(selected_ids[rank]);
#endif
        }
    }
#if MFQ_TOPK_ROUTE_GROUPS
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint index = row * 32u + lane;
    uint route_count = uint(logits_shape[0]) * uint(TOP_K);
    if (index < route_count) {
        int expert = selected_routes[index];
        device int* group = groups + index * 8u;
        for (uint slot = 0u; slot < 6u; ++slot) group[slot + 2u] = -1;
#if MFQ_TOPK_MAX_ROWS > 6
        uint preceding = 0u;
        for (uint route = 0u; route < index; ++route)
            preceding += uint(selected_routes[route] == expert);
        uint first = (preceding / 6u) * 6u;
#endif
        uint matches = 0u;
        for (uint route = 0u; route < route_count; ++route) {
            if (selected_routes[route] != expert) continue;
#if MFQ_TOPK_MAX_ROWS > 6
            if (matches >= first && matches < first + 6u)
                group[2u + matches - first] = int(route);
#else
            group[2u + matches] = int(route);
#endif
            ++matches;
        }
        group[0] = expert;
#if MFQ_TOPK_MAX_ROWS > 6
        group[1] = int(min(matches - first, 6u));
#else
        group[1] = int(matches);
#endif
    }
#endif
)METAL";

// Adapted from oMLX's Qwen router GEMV topology (Apache-2.0): spread one
// router row over each SIMD group instead of packing a narrow 512-row GEMV
// into the stock matmul schedule. MFQ supplies ordinary dense FP16/BF16
// weights here; no external quantized layout is part of this ABI.
constexpr const char* kDenseRouterLogitsSource = R"METAL(
    constexpr uint SIMD_GROUPS = 4u;
    uint lane = thread_index_in_simdgroup;
    uint row = threadgroup_position_in_grid.y * SIMD_GROUPS
        + simdgroup_index_in_threadgroup;
    if (row >= uint(EXPERTS)) {
        return;
    }

    float accumulator = 0.0f;
    device const activation_t* row_weight =
        weight + row * uint(K);
    for (uint column = lane * 4u;
         column < uint(K);
         column += 128u) {
        activation4_t activation =
            *(device const activation4_t*)(input + column);
        activation4_t values =
            *(device const activation4_t*)(row_weight + column);
        accumulator = fma(float(values[0]), float(activation[0]), accumulator);
        accumulator = fma(float(values[1]), float(activation[1]), accumulator);
        accumulator = fma(float(values[2]), float(activation[2]), accumulator);
        accumulator = fma(float(values[3]), float(activation[3]), accumulator);
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) {
        output[row] = activation_t(accumulator);
    }
)METAL";

constexpr const char* kDenseRouterTopKSource = R"METAL(
    constexpr uint EXPERTS = uint(N_EXPERTS);
    constexpr uint TOP_K = 6u;
    constexpr uint SIMD_GROUPS = 32u;
    constexpr uint EXPERTS_PER_SIMD = 4u;
    constexpr uint EXPERTS_PER_ROUND = SIMD_GROUPS * EXPERTS_PER_SIMD;
    constexpr uint REDUCTION_SIMD_GROUPS =
        (EXPERTS + 31u) / 32u;

    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.x;
    threadgroup activation_t cached_input[K];
    threadgroup float route_weights[EXPERTS];
    threadgroup float scores[EXPERTS];
    threadgroup float topk_weights[TOP_K];
    threadgroup float partial_scores[REDUCTION_SIMD_GROUPS];
    threadgroup uint partial_ids[REDUCTION_SIMD_GROUPS];

    // All router rows consume the same activation. Load it once instead of
    // issuing identical device reads from every SIMD group.
    device const activation_t* row_input = input + row * uint(K);
    for (uint column = tid * 4u; column < uint(K); column += 4096u) {
        *(threadgroup activation4_t*)(cached_input + column) =
            *(device const activation4_t*)(row_input + column);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each SIMD group consumes multiple rows at a time, and one cached
    // activation load feeds every row accumulator in the tile.
    for (uint round = 0u;
         round < EXPERTS / EXPERTS_PER_ROUND;
         ++round) {
        uint expert_base = round * EXPERTS_PER_ROUND
            + simd_group * EXPERTS_PER_SIMD;
        float accumulators[EXPERTS_PER_SIMD] = {0.0f};
        for (
            uint column = lane * 4u;
            column < uint(K);
            column += 128u
        ) {
            float4 activation = float4(
                *(threadgroup activation4_t*)(cached_input + column));
            for (uint local = 0u; local < EXPERTS_PER_SIMD; ++local) {
                uint weight_base = (expert_base + local) * uint(K);
                float4 values = float4(
                    *(device const activation4_t*)(
                        weight + weight_base + column));
                accumulators[local] += dot(activation, values);
            }
        }
        for (uint local = 0u; local < EXPERTS_PER_SIMD; ++local) {
            float raw = simd_sum(accumulators[local]);
            if (lane == 0u) {
                uint expert = expert_base + local;
                // Keep each expert's SIMD reduction in FP32 order so the
                // fused router matches its unfused dense reference.
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
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The first ceil(EXPERTS/32) SIMD groups reduce the scores in parallel.
    // The score/ID comparator exactly preserves the stable tie-break.
    for (uint rank = 0u; rank < TOP_K; ++rank) {
        if (simd_group < REDUCTION_SIMD_GROUPS) {
            uint candidate_id = tid;
            float candidate_score = candidate_id < EXPERTS
                ? scores[candidate_id]
                : -INFINITY;
            for (uint offset = 16u; offset > 0u; offset >>= 1u) {
                float other_score = simd_shuffle_down(
                    candidate_score, offset);
                uint other_id = simd_shuffle_down(candidate_id, offset);
                if (lane + offset < 32u &&
                    (other_score > candidate_score ||
                     (other_score == candidate_score &&
                      other_id < candidate_id))) {
                    candidate_score = other_score;
                    candidate_id = other_id;
                }
            }
            if (lane == 0u) {
                partial_scores[simd_group] = candidate_score;
                partial_ids[simd_group] = candidate_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float best_score = partial_scores[0];
            uint best_expert = partial_ids[0];
            for (uint group = 1u;
                 group < REDUCTION_SIMD_GROUPS;
                 ++group) {
                float score = partial_scores[group];
                uint expert = partial_ids[group];
                if (score > best_score ||
                    (score == best_score && expert < best_expert)) {
                    best_score = score;
                    best_expert = expert;
                }
            }
#if MFQ_HAS_EXPERT_MAP
            int slot = expert_map[best_expert];
            ids[row * TOP_K + rank] =
                int(best_expert) | ((slot + 1) << 8);
#else
            ids[row * TOP_K + rank] = int(best_expert);
#endif
            topk_weights[rank] = route_weights[best_expert];
            scores[best_expert] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float denominator = 0.0f;
        for (uint rank = 0u; rank < TOP_K; ++rank) {
            denominator += topk_weights[rank];
        }
        denominator = max(denominator, params[0]);
        for (uint rank = 0u; rank < TOP_K; ++rank) {
            weights[row * TOP_K + rank] =
                topk_weights[rank] / denominator * params[1];
        }
    }
)METAL";


constexpr const char* kDenseHashRouterSource = R"METAL(
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint route = simdgroup_index_in_threadgroup;
    threadgroup activation_t cached_input[K];
    threadgroup int selected_ids[TOP_K];
    threadgroup float selected_weights[TOP_K];

    device const activation_t* row_input =
        input + row * uint(K);
    for (uint column = tid * 4u; column < uint(K); column += 1024u) {
        *(threadgroup activation4_t*)(cached_input + column) =
            *(device const activation4_t*)(row_input + column);
    }
    if (tid < uint(TOP_K)) {
        int token = token_ids[row];
        selected_ids[tid] = token >= 0 && token < int(VOCAB)
            ? token_experts[uint(token) * uint(TOP_K) + tid]
            : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (
        uint selected_route = route;
        selected_route < uint(TOP_K);
        selected_route += 8u
    ) {
        int expert = selected_ids[selected_route];
        bool valid = expert >= 0 && expert < int(N_EXPERTS);
        float accumulator = 0.0f;
        if (valid) {
            uint weight_base = uint(expert) * uint(K);
            for (
                uint column = lane * 4u;
                column < uint(K);
                column += 128u
            ) {
                float4 activation = float4(
                    *(threadgroup activation4_t*)(
                        cached_input + column));
                float4 values = float4(
                    *(device const activation4_t*)(
                        weight + weight_base + column));
                accumulator += dot(activation, values);
            }
        }
        float raw = simd_sum(accumulator);
        if (lane == 0u) {
            raw = valid && !isnan(raw) ? raw : -FLT_MAX;
            float softplus = raw > 20.0f ? raw : log1p(exp(raw));
            selected_weights[selected_route] = sqrt(softplus);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float denominator = 1.0f;
        if (NORMALIZE != 0) {
            denominator = 0.0f;
            for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
                denominator += selected_weights[rank];
            }
            denominator = max(denominator, params[0]);
        }
        uint output_base = row * uint(TOP_K);
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            int expert = selected_ids[rank];
#if MFQ_HAS_EXPERT_MAP
            int slot = expert >= 0 ? expert_map[uint(expert)] : -1;
            ids[output_base + rank] =
                expert | ((slot + 1) << 8);
#else
            ids[output_base + rank] = expert;
#endif
            weights[output_base + rank] =
                selected_weights[rank] / denominator * params[1];
        }
    }
)METAL";

constexpr const char* kSqrtSoftplusSource = R"METAL(
    const uint ROWS = uint(ids_shape[0]);
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
    const uint ROWS = uint(static_ids_shape[0]);
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
    const uint TOKENS = uint(weights_shape[0]);
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

constexpr const char* kWeightedReduceSortedSource = R"METAL(
    const uint TOKENS = uint(weights_shape[0]);
    uint index = thread_position_in_grid.x;
    if (index >= uint(TOKENS * WIDTH)) {
        return;
    }
    uint token = index / uint(WIDTH);
    uint column = index - token * uint(WIDTH);
    float value = 0.0f;
    for (uint route = 0u; route < uint(ROUTES); ++route) {
        uint original_row = token * uint(ROUTES) + route;
        uint sorted_row = uint(inverse_route_order[original_row]);
        value += float(sorted_pair_output[
            sorted_row * uint(WIDTH) + column
        ]) * weights[original_row];
    }
    output[index] = T(value);
)METAL";

constexpr const char* kInversePermutationSource = R"METAL(
    const int SIZE = order_shape[0];
    uint sorted_row = thread_position_in_grid.x;
    if (sorted_row >= uint(SIZE)) {
        return;
    }
    int original_row = order[sorted_row];
    if (original_row >= 0 && original_row < SIZE) {
        inverse[uint(original_row)] = int(sorted_row);
    }
)METAL";

constexpr const char* kRestoreRoutedRowsHeader = R"METAL(
inline uint4 mfq_moe_load16(device const uchar* value) {
    return *reinterpret_cast<device const uint4*>(value);
}
inline uint4 mfq_moe_load16(constant const uchar* value) {
    return *reinterpret_cast<constant const uint4*>(value);
}
)METAL";

constexpr const char* kRestoreRoutedRowsSource = R"METAL(
    const uint width = uint(values_shape[1]);
    const uint chunks = (width + 15u) / 16u;
    const uint index = thread_position_in_grid.x;
    const uint row = index / chunks;
    if (row >= uint(values_shape[0])) return;
    const uint column = (index - row * chunks) * 16u;
    const int original = order[row];
    if (original < 0 || original >= values_shape[0]) return;
    const bool valid = ids[original] >= 0;
    const ulong source = ulong(row) * width + column;
    const ulong destination = ulong(original) * width + column;
    if (width % 16u == 0u) {
        const uint4 value = valid ? mfq_moe_load16(values + source) : uint4(0);
        *reinterpret_cast<device uint4*>(output + destination) = value;
    } else {
        for (uint item = 0u; item < 16u && column + item < width; ++item)
            output[destination + item] = valid ? values[source + item] : uchar(0);
    }
)METAL";

constexpr const char* kGluSplitSource = R"METAL(
    const uint ROWS = uint(gate_up_shape[0]);
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

constexpr const char* kGluPairSource = R"METAL(
    const uint SIZE = uint(gate_shape[0]);
    uint index = thread_position_in_grid.x;
    if (index >= uint(SIZE)) {
        return;
    }
    float gate_value = float(gate[index]);
    float up_value = float(up[index]);
    if (HAS_LIMIT != 0) {
        gate_value = min(gate_value, params[0]);
        up_value = clamp(up_value, -params[0], params[0]);
    }
    float activated = gate_value / (1.0f + exp(-gate_value));
    output[index] = T(activated * up_value);
)METAL";

constexpr const char* kSharedGateSource = R"METAL(
    const uint TOKENS = uint(routed_shape[0]);
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
    const uint TOKENS = uint(weights_shape[0]);
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

constexpr const char* kReduceSharedGateSortedSource = R"METAL(
    const uint TOKENS = uint(weights_shape[0]);
    uint index = thread_position_in_grid.x;
    if (index >= uint(TOKENS * WIDTH)) {
        return;
    }
    uint token = index / uint(WIDTH);
    uint column = index - token * uint(WIDTH);
    float value = 0.0f;
    for (uint route = 0u; route < uint(ROUTES); ++route) {
        uint original_row = token * uint(ROUTES) + route;
        uint sorted_row = uint(inverse_route_order[original_row]);
        value += float(sorted_pair_output[
            sorted_row * uint(WIDTH) + column
        ]) * weights[original_row];
    }
    T routed_value = T(value);
    float gate = 1.0f / (1.0f + exp(-gate_logits[token]));
    output[index] =
        T(float(routed_value) + gate * float(shared[index]));
)METAL";

constexpr const char* kExpertScaleSource = R"METAL(
    const uint SIZE = uint(weights_shape[0]);
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
    std::string source,
    std::string header = {}) {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        name,
        std::move(inputs),
        std::move(outputs),
        std::move(source),
        std::move(header),
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
softmax_top_k_row_kernel(mlx::core::Dtype dtype) {
    static const auto fp16_kernel = make_kernel(
        "mfq_cpp_moe_softmax_topk_row_f16",
        {"logits"},
        {"ids", "weights"},
        std::string("using T = half;\n") + kSoftmaxTopKRowSource);
    static const auto bf16_kernel = make_kernel(
        "mfq_cpp_moe_softmax_topk_row_bf16",
        {"logits"},
        {"ids", "weights"},
        std::string("using T = bfloat;\n") + kSoftmaxTopKRowSource);
    static const auto fp32_kernel = make_kernel(
        "mfq_cpp_moe_softmax_topk_row_f32",
        {"logits"},
        {"ids", "weights"},
        std::string("using T = float;\n") + kSoftmaxTopKRowSource);
    if (dtype == mlx::core::float16) return fp16_kernel;
    return dtype == mlx::core::bfloat16 ? bf16_kernel : fp32_kernel;
}

const mlx::core::fast::CustomKernelFunction&
softmax_top_k_group_kernel(mlx::core::Dtype dtype) {
    static const auto fp16_kernel = make_kernel(
        "mfq_cpp_moe_softmax_topk_groups_f16", {"logits"}, {"ids", "weights", "groups"},
        std::string("#define MFQ_TOPK_ROUTE_GROUPS 1\nusing T = half;\n") + kSoftmaxTopKRowSource);
    static const auto bf16_kernel = make_kernel(
        "mfq_cpp_moe_softmax_topk_groups_bf16", {"logits"}, {"ids", "weights", "groups"},
        std::string("#define MFQ_TOPK_ROUTE_GROUPS 1\nusing T = bfloat;\n") + kSoftmaxTopKRowSource);
    static const auto fp32_kernel = make_kernel(
        "mfq_cpp_moe_softmax_topk_groups_f32", {"logits"}, {"ids", "weights", "groups"},
        std::string("#define MFQ_TOPK_ROUTE_GROUPS 1\nusing T = float;\n") + kSoftmaxTopKRowSource);
    if (dtype == mlx::core::float16) return fp16_kernel;
    return dtype == mlx::core::bfloat16 ? bf16_kernel : fp32_kernel;
}

const mlx::core::fast::CustomKernelFunction&
softmax_top_k_large_group_kernel(mlx::core::Dtype dtype) {
    static const auto fp16_kernel = make_kernel(
        "mfq_cpp_moe_softmax_topk_groups16_f16", {"logits"}, {"ids", "weights", "groups"},
        std::string("#define MFQ_TOPK_ROUTE_GROUPS 1\n#define MFQ_TOPK_MAX_ROWS 16\nusing T = half;\n") + kSoftmaxTopKRowSource);
    static const auto bf16_kernel = make_kernel(
        "mfq_cpp_moe_softmax_topk_groups16_bf16", {"logits"}, {"ids", "weights", "groups"},
        std::string("#define MFQ_TOPK_ROUTE_GROUPS 1\n#define MFQ_TOPK_MAX_ROWS 16\nusing T = bfloat;\n") + kSoftmaxTopKRowSource);
    static const auto fp32_kernel = make_kernel(
        "mfq_cpp_moe_softmax_topk_groups16_f32", {"logits"}, {"ids", "weights", "groups"},
        std::string("#define MFQ_TOPK_ROUTE_GROUPS 1\n#define MFQ_TOPK_MAX_ROWS 16\nusing T = float;\n") + kSoftmaxTopKRowSource);
    if (dtype == mlx::core::float16) return fp16_kernel;
    return dtype == mlx::core::bfloat16 ? bf16_kernel : fp32_kernel;
}

const mlx::core::fast::CustomKernelFunction&
dense_router_logits_kernel(mlx::core::Dtype dtype) {
    static const auto fp16_kernel = make_kernel(
        "mfq_cpp_moe_dense_router_logits_f16",
        {"input", "weight"},
        {"output"},
        std::string(
            "using activation_t = half;\n"
            "using activation4_t = half4;\n") +
            kDenseRouterLogitsSource);
    static const auto bf16_kernel = make_kernel(
        "mfq_cpp_moe_dense_router_logits_bf16",
        {"input", "weight"},
        {"output"},
        std::string(
            "using activation_t = bfloat;\n"
            "using activation4_t = bfloat4;\n") +
            kDenseRouterLogitsSource);
    return dtype == mlx::core::bfloat16 ? bf16_kernel : fp16_kernel;
}

const mlx::core::fast::CustomKernelFunction&
dense_router_top_k_kernel(mlx::core::Dtype dtype, bool mapped) {
    static const auto fp16_kernel = make_kernel(
        "mfq_cpp_moe_dense_router_topk_f16",
        {"input", "weight", "bias", "available", "params"},
        {"ids", "weights"},
        std::string(
            "#define MFQ_HAS_EXPERT_MAP 0\n"
            "using activation_t = half;\n"
            "using activation4_t = half4;\n") +
            kDenseRouterTopKSource);
    static const auto bf16_kernel = make_kernel(
        "mfq_cpp_moe_dense_router_topk_bf16",
        {"input", "weight", "bias", "available", "params"},
        {"ids", "weights"},
        std::string(
            "#define MFQ_HAS_EXPERT_MAP 0\n"
            "using activation_t = bfloat;\n"
            "using activation4_t = bfloat4;\n") +
            kDenseRouterTopKSource);
    static const auto mapped_fp16_kernel = make_kernel(
        "mfq_cpp_moe_dense_router_topk_packed_f16",
        {"input", "weight", "bias", "available", "params", "expert_map"},
        {"ids", "weights"},
        std::string(
            "#define MFQ_HAS_EXPERT_MAP 1\n"
            "using activation_t = half;\n"
            "using activation4_t = half4;\n") +
            kDenseRouterTopKSource);
    static const auto mapped_bf16_kernel = make_kernel(
        "mfq_cpp_moe_dense_router_topk_packed_bf16",
        {"input", "weight", "bias", "available", "params", "expert_map"},
        {"ids", "weights"},
        std::string(
            "#define MFQ_HAS_EXPERT_MAP 1\n"
            "using activation_t = bfloat;\n"
            "using activation4_t = bfloat4;\n") +
            kDenseRouterTopKSource);
    if (mapped) {
        return dtype == mlx::core::bfloat16
            ? mapped_bf16_kernel
            : mapped_fp16_kernel;
    }
    return dtype == mlx::core::bfloat16 ? bf16_kernel : fp16_kernel;
}

const mlx::core::fast::CustomKernelFunction&
dense_hash_router_kernel(mlx::core::Dtype dtype, bool mapped) {
    static const auto fp16_kernel = make_kernel(
        "mfq_cpp_moe_dense_hash_router_f16",
        {"input", "weight", "token_ids", "token_experts", "params"},
        {"ids", "weights"},
        std::string(
            "#define MFQ_HAS_EXPERT_MAP 0\n"
            "using activation_t = half;\n"
            "using activation4_t = half4;\n") +
            kDenseHashRouterSource);
    static const auto bf16_kernel = make_kernel(
        "mfq_cpp_moe_dense_hash_router_bf16",
        {"input", "weight", "token_ids", "token_experts", "params"},
        {"ids", "weights"},
        std::string(
            "#define MFQ_HAS_EXPERT_MAP 0\n"
            "using activation_t = bfloat;\n"
            "using activation4_t = bfloat4;\n") +
            kDenseHashRouterSource);
    static const auto mapped_fp16_kernel = make_kernel(
        "mfq_cpp_moe_dense_hash_router_packed_f16",
        {"input", "weight", "token_ids", "token_experts", "params", "expert_map"},
        {"ids", "weights"},
        std::string(
            "#define MFQ_HAS_EXPERT_MAP 1\n"
            "using activation_t = half;\n"
            "using activation4_t = half4;\n") +
            kDenseHashRouterSource);
    static const auto mapped_bf16_kernel = make_kernel(
        "mfq_cpp_moe_dense_hash_router_packed_bf16",
        {"input", "weight", "token_ids", "token_experts", "params", "expert_map"},
        {"ids", "weights"},
        std::string(
            "#define MFQ_HAS_EXPERT_MAP 1\n"
            "using activation_t = bfloat;\n"
            "using activation4_t = bfloat4;\n") +
            kDenseHashRouterSource);
    if (mapped) {
        return dtype == mlx::core::bfloat16
            ? mapped_bf16_kernel
            : mapped_fp16_kernel;
    }
    return dtype == mlx::core::bfloat16 ? bf16_kernel : fp16_kernel;
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

const mlx::core::fast::CustomKernelFunction&
weighted_reduce_sorted_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_moe_weighted_reduce_sorted",
        {"sorted_pair_output", "inverse_route_order", "weights"},
        {"output"},
        kWeightedReduceSortedSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
inverse_permutation_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_moe_inverse_permutation",
        {"order"},
        {"inverse"},
        kInversePermutationSource);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& restore_routed_rows_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_moe_restore_routed_rows",
        {"values", "order", "ids"}, {"output"}, kRestoreRoutedRowsSource, kRestoreRoutedRowsHeader);
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

const mlx::core::fast::CustomKernelFunction& glu_pair_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_moe_glu_pair",
        {"gate", "up", "params"},
        {"output"},
        kGluPairSource);
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
reduce_shared_gate_sorted_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_moe_reduce_shared_gate_sorted",
        {
            "sorted_pair_output",
            "inverse_route_order",
            "weights",
            "shared",
            "gate_logits",
        },
        {"output"},
        kReduceSharedGateSortedSource);
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
        {mlx::core::reshape(values, Shape{rows, 2 * width}), params},
        {std::move(output_shape)},
        {values.dtype()},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {
            {"T", values.dtype()},
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

std::optional<array> moe_route_groups_host(
    std::span<const std::int32_t> ids, int tokens, int routes) {
    if (tokens < 1 || routes < 1 || routes > kMaximumRoutes
        || ids.size() != static_cast<std::size_t>(tokens) * routes)
        throw std::invalid_argument("host MoE routes have invalid geometry");
    if (tokens == 1 || tokens > 24) return std::nullopt;
    std::vector<std::int32_t> groups(ids.size() * 8, -1);
    std::vector<std::int32_t> matches;
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (groups[index * 8 + 1] != -1) continue;
        matches.clear();
        for (std::size_t other = index; other < ids.size(); ++other)
            if (ids[other] == ids[index]) matches.push_back(static_cast<std::int32_t>(other));
        for (std::size_t begin = 0; begin < matches.size(); begin += 6) {
            const auto count = std::min<std::size_t>(6, matches.size() - begin);
            for (std::size_t member = 0; member < count; ++member) {
                auto* group = groups.data() + matches[begin + member] * 8;
                group[0] = ids[index];
                group[1] = static_cast<std::int32_t>(count);
                std::copy_n(matches.begin() + begin, count, group + 2);
            }
        }
    }
    return array(groups.begin(), Shape{tokens, routes, 8});
}

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
    float scale,
    bool route_groups) {
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

    if (rows <= 16 && mode == 0 && normalize &&
        norm_floor == 1e-20f && scale == 1.0f &&
        !bias.has_value() && !available.has_value() &&
        experts >= 32 && experts <= 4096 && experts % 32 == 0) {
        values = mlx::core::reshape(values, Shape{rows, experts});
        const bool grouped = route_groups && rows > 1;
        std::vector<Shape> shapes{Shape{rows, top_k}, Shape{rows, top_k}};
        std::vector<mlx::core::Dtype> dtypes{mlx::core::int32, mlx::core::float32};
        if (grouped) {
            shapes.emplace_back(Shape{rows, top_k, 8});
            dtypes.push_back(mlx::core::int32);
        }
        const auto& kernel = grouped && rows > 6 ? softmax_top_k_large_group_kernel(values.dtype())
            : grouped ? softmax_top_k_group_kernel(values.dtype())
            : softmax_top_k_row_kernel(values.dtype());
        const int grouped_threads = rows > 6 ? 512 : 192;
        auto outputs = kernel(
            {std::move(values)},
            shapes,
            dtypes,
            grouped ? std::tuple{grouped_threads, 1, 1} : std::tuple{32, rows, 1},
            grouped ? std::tuple{grouped_threads, 1, 1} : std::tuple{32, 1, 1},
            {
                {"EXPERTS", experts},
                {"TOP_K", top_k},
            },
            std::nullopt,
            false,
            {});
        return {
            std::move(outputs.at(0)),
            std::move(outputs.at(1)),
            grouped ? std::optional<array>(std::move(outputs.at(2))) : std::nullopt,
        };
    }

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

bool moe_dense_router_logits_supported(
    const array& input,
    const array& weight) noexcept {
    const bool dense16 = weight.dtype() == mlx::core::float16 ||
        weight.dtype() == mlx::core::bfloat16;
    if (!dense16 || weight.ndim() != 2 ||
        !weight.flags().row_contiguous || input.ndim() == 0) {
        return false;
    }
    const int experts = weight.shape(0);
    const int width = weight.shape(1);
    return (experts == 256 || experts == 384 || experts == 512) &&
        width > 64 && width < 16 * experts && width % 128 == 0 &&
        input.shape(-1) == width &&
        input.size() == static_cast<std::size_t>(width);
}

array moe_dense_router_logits(
    const array& input,
    const array& weight) {
    if (!moe_dense_router_logits_supported(input, weight)) {
        throw std::invalid_argument(
            "dense router GEMV requires one row, 256/384/512 experts, "
            "and contiguous FP16/BF16 weights");
    }
    auto source = mlx::core::contiguous(mlx::core::reshape(
        input.dtype() == weight.dtype()
            ? input
            : mlx::core::astype(input, weight.dtype()),
        Shape{weight.shape(1)}));
    const int experts = weight.shape(0);
    auto outputs = dense_router_logits_kernel(weight.dtype())(
        {std::move(source), weight},
        {Shape{1, experts}},
        {weight.dtype()},
        {32, experts, 1},
        {32, 4, 1},
        {
            {"K", weight.shape(1)},
            {"EXPERTS", experts},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

bool moe_dense_router_topk_supported(
    const array& input,
    const array& weight) noexcept {
    const bool activation16 =
        input.dtype() == mlx::core::float16 ||
        input.dtype() == mlx::core::bfloat16;
    if (input.ndim() == 0) {
        return false;
    }
    const int width = input.ndim() > 0 ? input.shape(-1) : 0;
    if (!activation16 || width <= 0 || (width % 4) != 0 ||
        input.size() % static_cast<std::size_t>(width) != 0) {
        return false;
    }
    const auto rows = input.size() / static_cast<std::size_t>(width);
    return rows >= 1 && rows <= 16 && weight.ndim() == 2
        && weight.dtype() == input.dtype()
        && (weight.shape(0) == 256 || weight.shape(0) == 384)
        && weight.shape(1) == width;
}

MlxMoeTopKResult dense_router_topk_impl(
    const array& input,
    const array& weight,
    const array* expert_map,
    const std::optional<array>& bias,
    const std::optional<array>& available,
    float norm_floor,
    float scale) {
    if (!moe_dense_router_topk_supported(input, weight)) {
        throw std::invalid_argument(
            "fused dense router requires 1..16 FP16/BF16 rows and a "
            "matching contiguous [256|384,K] weight with K divisible by four");
    }
    if (!std::isfinite(norm_floor) || norm_floor < 0.0f ||
        !std::isfinite(scale)) {
        throw std::invalid_argument(
            "fused dense router parameters are invalid");
    }
    const int width = input.shape(-1);
    const int rows = checked_int(
        input.size() / static_cast<std::size_t>(width),
        "fused router row count");
    auto source = mlx::core::contiguous(
        mlx::core::reshape(input, Shape{rows, width}));
    auto weights = mlx::core::contiguous(weight);
    const int experts = weight.shape(0);
    if (expert_map != nullptr && experts > 256) {
        throw std::invalid_argument(
            "packed fused router supports at most 256 global experts");
    }
    array bias_values =
        mlx::core::zeros(Shape{experts}, mlx::core::float32);
    if (bias.has_value()) {
        bias_values = float32_contiguous(*bias);
        if (bias_values.shape() != Shape{experts}) {
            throw std::invalid_argument(
                "fused dense router bias shape mismatch");
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
                    "fused router availability size")}));
        if (available_values.shape() != Shape{experts}) {
            throw std::invalid_argument(
                "fused dense router availability shape mismatch");
        }
    }
    const array params(
        {norm_floor, scale},
        mlx::core::float32);
    array expert_map_values = mlx::core::zeros(
        Shape{1},
        mlx::core::int32);
    if (expert_map != nullptr) {
        if (expert_map->dtype() != mlx::core::int32 ||
            expert_map->shape() != Shape{experts} ||
            !expert_map->flags().row_contiguous) {
            throw std::invalid_argument(
                "fused dense router expert map shape/dtype mismatch");
        }
        expert_map_values = *expert_map;
    }
    std::vector<array> kernel_inputs{
        source,
        weights,
        bias_values,
        available_values,
        params,
    };
    std::vector<Shape> output_shapes{
        Shape{rows, 6},
        Shape{rows, 6},
    };
    std::vector<mlx::core::Dtype> output_dtypes{
        mlx::core::int32,
        mlx::core::float32,
    };
    if (expert_map != nullptr) {
        kernel_inputs.push_back(expert_map_values);
    }
    auto outputs = dense_router_top_k_kernel(
        source.dtype(),
        expert_map != nullptr)(
        std::move(kernel_inputs),
        std::move(output_shapes),
        std::move(output_dtypes),
        {rows * 1024, 1, 1},
        {1024, 1, 1},
        {
            {"K", width},
            {"N_EXPERTS", experts},
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

MlxMoeTopKResult moe_dense_router_topk(
    const array& input,
    const array& weight,
    const std::optional<array>& bias,
    const std::optional<array>& available,
    float norm_floor,
    float scale) {
    return dense_router_topk_impl(
        input,
        weight,
        nullptr,
        bias,
        available,
        norm_floor,
        scale);
}

MlxMoeTopKResult moe_dense_router_topk_packed(
    const array& input,
    const array& weight,
    const array& expert_map,
    const std::optional<array>& bias,
    const std::optional<array>& available,
    float norm_floor,
    float scale) {
    return dense_router_topk_impl(
        input,
        weight,
        &expert_map,
        bias,
        available,
        norm_floor,
        scale);
}

bool moe_dense_hash_router_supported(
    const array& input,
    const array& weight,
    const array& token_ids,
    const array& token_experts) noexcept {
    const bool activation16 =
        input.dtype() == mlx::core::float16 ||
        input.dtype() == mlx::core::bfloat16;
    if (input.ndim() < 2 || !activation16 || input.shape(-1) <= 0 ||
        (input.shape(-1) % 4) != 0 || weight.ndim() != 2 ||
        weight.dtype() != input.dtype() ||
        weight.shape(1) != input.shape(-1) ||
        token_experts.ndim() != 2 || token_experts.shape(0) <= 0 ||
        token_experts.shape(1) <= 0 ||
        token_experts.shape(1) > kMaximumRoutes ||
        token_experts.dtype() != mlx::core::int32 ||
        token_ids.size() == 0 ||
        (token_ids.dtype() != mlx::core::int32 &&
         token_ids.dtype() != mlx::core::uint32)) {
        return false;
    }
    const auto width = static_cast<std::size_t>(input.shape(-1));
    const auto rows = input.size() / width;
    return rows * width == input.size() && token_ids.size() == rows &&
        weight.shape(0) > 0 && weight.shape(0) <= kMaximumExperts;
}

MlxMoeTopKResult dense_hash_router_impl(
    const array& input,
    const array& weight,
    const array& token_ids,
    const array& token_experts,
    const array* expert_map,
    float norm_floor,
    float scale) {
    if (!moe_dense_hash_router_supported(
            input, weight, token_ids, token_experts)) {
        throw std::invalid_argument(
            "fused dense hash router shapes or dtypes are incompatible");
    }
    if (!std::isfinite(norm_floor) || norm_floor < 0.0f ||
        !std::isfinite(scale)) {
        throw std::invalid_argument(
            "fused dense hash router parameters are invalid");
    }
    const int width = input.shape(-1);
    const int rows = checked_int(
        input.size() / static_cast<std::size_t>(width),
        "fused hash router row count");
    const int experts = weight.shape(0);
    const int routes = token_experts.shape(1);
    const int vocab = token_experts.shape(0);
    if (expert_map != nullptr && experts > 256) {
        throw std::invalid_argument(
            "packed fused hash router supports at most 256 global experts");
    }
    auto source = mlx::core::contiguous(
        mlx::core::reshape(input, Shape{rows, width}));
    auto weights = mlx::core::contiguous(weight);
    auto ids = int32_contiguous(token_ids);
    ids = mlx::core::reshape(ids, Shape{rows});
    auto table = int32_contiguous(token_experts);
    const array params({norm_floor, scale}, mlx::core::float32);
    std::vector<array> kernel_inputs{
        source,
        weights,
        ids,
        table,
        params,
    };
    if (expert_map != nullptr) {
        if (expert_map->dtype() != mlx::core::int32 ||
            expert_map->shape() != Shape{experts} ||
            !expert_map->flags().row_contiguous) {
            throw std::invalid_argument(
                "fused dense hash router expert map shape/dtype mismatch");
        }
        kernel_inputs.push_back(*expert_map);
    }
    auto outputs = dense_hash_router_kernel(
        source.dtype(), expert_map != nullptr)(
        std::move(kernel_inputs),
        {Shape{rows, routes}, Shape{rows, routes}},
        {mlx::core::int32, mlx::core::float32},
        {rows * kThreads, 1, 1},
        {kThreads, 1, 1},
        {
            {"K", width},
            {"N_EXPERTS", experts},
            {"VOCAB", vocab},
            {"TOP_K", routes},
            {"NORMALIZE", 1},
        },
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs.at(0)),
        std::move(outputs.at(1)),
    };
}

MlxMoeTopKResult moe_dense_hash_router(
    const array& input,
    const array& weight,
    const array& token_ids,
    const array& token_experts,
    float norm_floor,
    float scale) {
    return dense_hash_router_impl(
        input,
        weight,
        token_ids,
        token_experts,
        nullptr,
        norm_floor,
        scale);
}

MlxMoeTopKResult moe_dense_hash_router_packed(
    const array& input,
    const array& weight,
    const array& token_ids,
    const array& token_experts,
    const array& expert_map,
    float norm_floor,
    float scale) {
    return dense_hash_router_impl(
        input,
        weight,
        token_ids,
        token_experts,
        &expert_map,
        norm_floor,
        scale);
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
            {"ROUTES", routes},
            {"WIDTH", width},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array moe_weighted_reduce_sorted(
    const array& sorted_pair_output,
    const array& inverse_route_order,
    const array& weights) {
    auto pairs = floating_contiguous(sorted_pair_output);
    auto inverse = int32_contiguous(inverse_route_order);
    auto route_weights = float32_contiguous(weights);
    if (pairs.ndim() != 2 || inverse.ndim() != 1
        || route_weights.ndim() != 2) {
        throw std::invalid_argument(
            "MoE sorted weighted reduction rank mismatch");
    }
    const int tokens = route_weights.shape(0);
    const int routes = route_weights.shape(1);
    const int width = pairs.shape(1);
    const auto route_count = static_cast<std::size_t>(tokens)
        * static_cast<std::size_t>(routes);
    if (tokens <= 0 || routes <= 0 || width <= 0
        || pairs.shape(0) != checked_int(route_count, "sorted route count")
        || inverse.size() != route_count) {
        throw std::invalid_argument(
            "MoE sorted weighted reduction shape mismatch");
    }
    const int size = checked_int(
        static_cast<std::size_t>(tokens) * width,
        "sorted weighted reduce size");
    auto outputs = weighted_reduce_sorted_kernel()(
        {pairs, inverse, route_weights},
        {Shape{tokens, width}},
        {pairs.dtype()},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {
            {"T", pairs.dtype()},
            {"ROUTES", routes},
            {"WIDTH", width},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array moe_inverse_permutation(const array& order) {
    auto permutation = int32_contiguous(order);
    if (permutation.ndim() != 1 || permutation.size() == 0) {
        throw std::invalid_argument(
            "MoE route order must be a non-empty one-dimensional permutation");
    }
    const int size = checked_int(
        permutation.size(),
        "route permutation size");
    auto outputs = inverse_permutation_kernel()(
        {std::move(permutation)},
        {Shape{size}},
        {mlx::core::int32},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {},
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array moe_restore_routed_rows(
    const array& sorted_output, const array& route_order, const array& expert_ids) {
    if (sorted_output.ndim() != 2 || route_order.ndim() != 1 ||
        route_order.size() != static_cast<std::size_t>(sorted_output.shape(0)) ||
        expert_ids.size() != route_order.size())
        throw std::invalid_argument("MoE sorted output and route metadata do not match");
    if (sorted_output.size() == 0) return sorted_output;
    auto values = mlx::core::view(mlx::core::contiguous(sorted_output), mlx::core::uint8);
    const int chunks = (values.shape(1) + 15) / 16;
    const int size = checked_int(std::size_t(values.shape(0)) * chunks, "restore grid");
    auto result = restore_routed_rows_kernel()(
        {values, int32_contiguous(route_order), int32_contiguous(expert_ids)},
        {values.shape()}, {mlx::core::uint8}, {size,1,1}, {std::min(kThreads,size),1,1},
        {}, std::nullopt, false, {}).front();
    return mlx::core::view(std::move(result),sorted_output.dtype());
}

array moe_swiglu_split(const array& gate_up) {
    return glu_split(gate_up, false);
}

array moe_limited_swiglu_split(
    const array& gate_up,
    float limit) {
    return glu_split(gate_up, false, limit);
}

static array swiglu_pair(
    const array& gate,
    const array& up,
    float limit) {
    auto gate_values = floating_contiguous(gate);
    auto up_values = floating_contiguous(up);
    if (gate_values.ndim() < 2 || gate_values.size() == 0
        || up_values.shape() != gate_values.shape()) {
        throw std::invalid_argument(
            "Gate and Up must have matching non-empty shapes");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "SwiGLU pair limit must be finite and non-negative");
    }
    if (up_values.dtype() != gate_values.dtype()) {
        up_values = mlx::core::contiguous(
            mlx::core::astype(up_values, gate_values.dtype()));
    }
    const int size = checked_int(
        gate_values.size(),
        "paired GLU element count");
    const array params({limit}, mlx::core::float32);
    auto outputs = glu_pair_kernel()(
        {
            mlx::core::reshape(gate_values, Shape{size}),
            mlx::core::reshape(up_values, Shape{size}),
            params,
        },
        {gate_values.shape()},
        {gate_values.dtype()},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {{"T", gate_values.dtype()}, {"HAS_LIMIT", int(limit > 0.0f)}},
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array moe_swiglu_pair(const array& gate, const array& up) {
    return swiglu_pair(gate, up, 0.0f);
}

array moe_limited_swiglu_pair(
    const array& gate,
    const array& up,
    float limit) {
    if (!std::isfinite(limit) || limit <= 0.0f) {
        throw std::invalid_argument(
            "limited SwiGLU pair limit must be finite and positive");
    }
    return swiglu_pair(gate, up, limit);
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
            {"ROUTES", routes},
            {"WIDTH", width},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array moe_weighted_reduce_shared_gate_sorted(
    const array& sorted_pair_output,
    const array& inverse_route_order,
    const array& weights,
    const array& shared,
    const array& gate_logits) {
    auto pairs = floating_contiguous(sorted_pair_output);
    auto inverse = int32_contiguous(inverse_route_order);
    auto route_weights = float32_contiguous(weights);
    auto shared_values = floating_contiguous(shared);
    if (pairs.ndim() != 2 || inverse.ndim() != 1
        || route_weights.ndim() != 2) {
        throw std::invalid_argument(
            "MoE sorted fused reduction rank mismatch");
    }
    const int tokens = route_weights.shape(0);
    const int routes = route_weights.shape(1);
    const int width = pairs.shape(1);
    const auto route_count = static_cast<std::size_t>(tokens)
        * static_cast<std::size_t>(routes);
    if (tokens <= 0 || routes <= 0 || width <= 0
        || pairs.shape(0) != checked_int(route_count, "sorted route count")
        || inverse.size() != route_count
        || shared_values.shape() != Shape{tokens, width}) {
        throw std::invalid_argument(
            "MoE sorted fused reduce/shared gate shape mismatch");
    }
    if (shared_values.dtype() != pairs.dtype()) {
        shared_values = mlx::core::contiguous(
            mlx::core::astype(shared_values, pairs.dtype()));
    }
    auto gates = reshape_gate_logits(gate_logits, tokens);
    const int size = checked_int(
        static_cast<std::size_t>(tokens) * width,
        "sorted fused reduce/shared gate size");
    auto outputs = reduce_shared_gate_sorted_kernel()(
        {pairs, inverse, route_weights, shared_values, gates},
        {Shape{tokens, width}},
        {pairs.dtype()},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {
            {"T", pairs.dtype()},
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
        {
            mlx::core::reshape(values, Shape{size}),
            mlx::core::reshape(selected, Shape{size}),
            expert_scales,
        },
        {values.shape()},
        {mlx::core::float32},
        {size, 1, 1},
        {std::min(kThreads, size), 1, 1},
        {},
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

} // namespace mfq::metal
