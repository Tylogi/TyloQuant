#include "mlx_tq.h"
#include "mfq_container.h"
#include "mfq_tq_embedded.h"
#include "mlx_weight_residency.h"
#include "mlx_moe_ops.h"
#include "mlx_platform.h"
#include "mlx_eval_timing.h"
#include "mlx_tq_rows.h"

#include <mlx/allocator.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <climits>
#include <bit>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace mfq::metal {
namespace {
using namespace mlx::core;

void record_dispatch(const array& input, int outputs, int experts, const char* mode, int rows = 0) {
    if (!detail::component_profile_active()) return;
    detail::profile_marker(std::string("tq.dispatch.") + mode + ".dtype"
        + (input.dtype() == float16 ? "f16" : input.dtype() == bfloat16 ? "bf16" : "f32")
        + ".M" + std::to_string(rows ? rows : input.size() / input.shape(-1))
        + ".K" + std::to_string(input.shape(-1)) + ".N" + std::to_string(outputs)
        + ".E" + std::to_string(experts));
}

constexpr const char* kProject = R"METAL(
uint row = thread_position_in_grid.x / LANES;
uint lane = thread_position_in_grid.x % LANES;
uint route = thread_position_in_grid.y;
if (row >= OUT) return;
int expert = ids[route];
uint input_row = SHARED_INPUT ? route / ROUTES : route;
float gate = 0.f, up = 0.f;
if (expert >= 0 && expert < EXPERTS) {
    uint weight_row = uint(expert) * STRIDE_G + row;
    auto packed_g = tq_words<POOLED_G>(words_g, descriptors_g, EXPERTS * STRIDE_G, uint(expert));
    if (PAIR) {
        uint up_row = uint(expert) * STRIDE_U + BASE_U + row;
        auto packed_u = tq_words<POOLED_U>(words_u, descriptors_u, EXPERTS * STRIDE_U, uint(expert));
        float2 result = tq_dot_pair<T, LANES, COLUMN_G, COLUMN_U, WIDTH_G, WIDTH_U, K, GS_G, GS_U>(
            packed_g, descriptors_g[weight_row], scales_g, packed_u, descriptors_u[up_row], scales_u,
            relative, weight_row, up_row, x + input_row * K, lane);
        gate = result.x; up = result.y;
    } else gate = tq_dot<T, LANES, COLUMN_G, WIDTH_G, K, GS_G>(
        packed_g, descriptors_g[weight_row], scales_g, relative, weight_row, x + input_row * K, lane);
}
for (uint step = LANES / 2; step; step >>= 1) {
    gate += simd_shuffle_down(gate, step);
    if (PAIR) up += simd_shuffle_down(up, step);
}
if (lane == 0) {
    if (PAIR) {
        gate = float(T(gate)); up = float(T(up));
        float limit = params[0];
        if (limit > 0.f) { gate = min(gate, limit); up = clamp(up, -limit, limit); }
        gate = GELU ? .5f * gate * (1.f + tanh(.7978845608028654f * gate * (1.f + .044715f * gate * gate)))
                    : gate / (1.f + exp(-gate));
        gate *= up;
    }
    y[route * OUT + row] = T(gate);
}
)METAL";

constexpr const char* kEmbedding = R"METAL(
uint col = thread_position_in_grid.x * 4u;
uint row = thread_position_in_grid.y;
if (col >= K) return;
int source = ids[row];
half4 value = 0.f;
if (source >= 0 && source < ROWS) {
    ulong descriptor = descriptors[source];
    value = tq_decode4<COLUMNS, WIDTH, GS>(tq_words<POOLED>(words, descriptors, ROWS, uint(source) / OUT),
        descriptor, scales, relative, uint(source), col);
}
tq_store4<K>(y, row, col, value);
)METAL";

constexpr const char* kRoutedReduce = R"METAL(
threadgroup T partial[ROUTES];
uint row = threadgroup_position_in_grid.x;
uint local = thread_position_in_threadgroup.x;
uint lane = local % LANES, route = local / LANES;
uint token = threadgroup_position_in_grid.y;
int expert = ids[token * ROUTES + route];
float value = 0.f;
if (expert >= 0 && expert < EXPERTS) {
    uint weight_row = uint(expert) * OUT + row;
    value = tq_dot<T, LANES, COLUMNS, WIDTH, K, GS>(
        tq_words<POOLED>(words, descriptors, EXPERTS * OUT, uint(expert)), descriptors[weight_row],
        scales, relative, weight_row, x + (SHARED_INPUT ? token : token * ROUTES + route) * K, lane);
}
for (uint step = LANES / 2; step; step >>= 1) value += simd_shuffle_down(value, step);
if (lane == 0) partial[route] = T(value);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local == 0) {
    float total = 0.f;
    for (uint r = 0; r < ROUTES; ++r) total = fma(float(partial[r]), weights[token * ROUTES + r], total);
    y[token * OUT + row] = T(total);
}
)METAL";

const fast::CustomKernelFunction& projection_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_project",
        {"words_g", "descriptors_g", "scales_g", "relative", "words_u",
         "descriptors_u", "scales_u", "x", "ids", "params"}, {"y"},
        kProject, kTqDecode, true, false);
    return kernel;
}

const fast::CustomKernelFunction& column_input_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_tq_column_inputs", {"x", "scales", "ids"}, {"prepared"}, R"METAL(
uint col = thread_position_in_grid.x * 4u, route = thread_position_in_grid.y;
if (col >= K) return;
int expert = ids[route];
float4 value = 0.f;
if (expert >= 0 && expert < EXPERTS) {
    uint input_row = SHARED_INPUT ? route / ROUTES : route;
    uint offset = COLUMN == 2u ? uint(expert) * WIDTH : 0u;
    value = tq_input4<K>(x + input_row * K, col, 0u) * tq_column_scale4(scales, offset + col);
}
for (uint j = 0u; j < 4u; ++j) if (col + j < K) prepared[route * K + col + j] = value[j];
)METAL", kTqDecode, true, false);
    return kernel;
}

const fast::CustomKernelFunction& column_projection_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_tq_column_project", {"words", "descriptors", "relative", "prepared", "ids"}, {"y"}, R"METAL(
uint row = thread_position_in_grid.x / 16u, lane = thread_position_in_grid.x & 15u;
uint route = thread_position_in_grid.y;
if (row >= OUT) return;
int expert = ids[route];
float value = 0.f, anchor = 0.f;
if (expert >= 0 && expert < EXPERTS) {
    uint wr = uint(expert) * OUT + row;
    ulong descriptor = descriptors[wr];
    auto packed = tq_words<POOLED>(words, descriptors, EXPERTS * OUT, uint(expert));
    value = tq_dot<float, 16, 0, WIDTH, K, 512>(packed, descriptor, TqUnitScale{}, relative, wr, prepared + route * K, lane);
    anchor = float(as_type<half>(ushort(descriptor >> 40u)));
}
for (uint step = 8u; step; step >>= 1u) value += simd_shuffle_down(value, step);
if (lane == 0u) y[route * OUT + row] = half(value * anchor);
)METAL", std::string(R"METAL(
struct TqUnitScale {};
inline float tq_group_scale(TqUnitScale, device const half*, uint, float) { return 1.f; }
)METAL") + kTqDecode, true, false);
    return kernel;
}

const fast::CustomKernelFunction& column_pair_input_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_tq_column_pair_inputs",
        {"x", "scales_g", "scales_u", "ids"}, {"prepared_g", "prepared_u"}, R"METAL(
uint col = thread_position_in_grid.x * 4u, route = thread_position_in_grid.y;
if (col >= K) return;
int expert = ids[route];
float4 gate = 0.f, up = 0.f;
if (expert >= 0 && expert < EXPERTS) {
    uint input_row = SHARED_INPUT ? route / ROUTES : route;
    float4 value = tq_input4<K>(x + input_row * K, col, 0u);
    gate = value * tq_column_scale4(scales_g, col + (COLUMN_G == 2u ? uint(expert) * WIDTH_G : 0u));
    up = value * tq_column_scale4(scales_u, col + (COLUMN_U == 2u ? uint(expert) * WIDTH_U : 0u));
}
for (uint j = 0u; j < 4u; ++j) if (col + j < K) {
    prepared_g[route * K + col + j] = gate[j];
    prepared_u[route * K + col + j] = up[j];
}
)METAL", kTqDecode, true, false);
    return kernel;
}

const fast::CustomKernelFunction& column_pair_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_tq_column_pair",
        {"words_g", "desc_g", "words_u", "desc_u", "relative", "prepared_g", "prepared_u", "ids", "params"}, {"y"}, R"METAL(
uint row = thread_position_in_grid.x / 32u, lane = thread_position_in_grid.x & 31u;
uint route = thread_position_in_grid.y;
if (row >= OUT) return;
int expert = ids[route];
bool upper = lane >= 16u;
float value = 0.f, anchor = 0.f;
if (expert >= 0 && expert < EXPERTS) {
    uint wr = uint(expert) * OUT + row;
    auto descriptors = upper ? desc_u : desc_g;
    auto packed_g = tq_words<POOLED_G>(words_g, desc_g, EXPERTS * OUT, uint(expert));
    auto packed_u = tq_words<POOLED_U>(words_u, desc_u, EXPERTS * OUT, uint(expert));
    ulong descriptor = descriptors[wr];
    auto input = (upper ? prepared_u : prepared_g) + route * K;
    value = tq_dot<float, 16, 0, K, K, 512>(upper ? packed_u : packed_g, descriptor, TqUnitScale{}, relative, wr, input, lane & 15u);
    anchor = float(as_type<half>(ushort(descriptor >> 40u)));
}
for (uint step = 8u; step; step >>= 1u) value += simd_shuffle_down(value, step);
value *= anchor;
float other = simd_shuffle(value, 16u);
if (lane == 0u) {
    float gate = float(half(value)), up = float(half(other)), limit = params[0];
    if (limit > 0.f) { gate = min(gate, limit); up = clamp(up, -limit, limit); }
    gate = GELU ? .5f * gate * (1.f + tanh(.7978845608028654f * gate * (1.f + .044715f * gate * gate)))
                : gate / (1.f + exp(-gate));
    y[route * OUT + row] = half(gate * up);
}
)METAL", std::string(R"METAL(
struct TqUnitScale {};
inline float tq_group_scale(TqUnitScale, device const half*, uint, float) { return 1.f; }
)METAL") + kTqDecode, true, false);
    return kernel;
}

const fast::CustomKernelFunction& paired_route_kernel() {
    static const auto kernel = [] {
        const std::string source = std::string(R"METAL(
uint row = thread_position_in_grid.x / 64u;
uint slot = (thread_position_in_grid.x / 32u) & 1u;
uint lane = thread_position_in_grid.x % 32u;
uint pair = thread_position_in_grid.y / ROUTES, selected = thread_position_in_grid.y % ROUTES;
uint first_token = pair * 2u, first = first_token * ROUTES + selected;
uint route = first + slot * ROUTES;
if (row >= OUT) return;
int expert0 = ids[first], expert1 = ids[first + ROUTES];
if (expert0 == expert1) {
    if (slot) return;
    int4 sources = SHARED_INPUT ? int4(first_token, first_token + 1u, -1, -1)
        : int4(first, first + ROUTES, -1, -1);
    float4 value = 0.f;
    if (expert0 >= 0 && expert0 < EXPERTS) {
        bool upper = lane >= 16u;
        auto packed_g = tq_words<POOLED_G>(words_g, descriptors_g, EXPERTS * STRIDE_G, uint(expert0));
        auto packed_u = tq_words<POOLED_U>(words_u, descriptors_u, EXPERTS * STRIDE_U, uint(expert0));
        auto words = upper ? packed_u : packed_g;
        auto descriptors = upper ? descriptors_u : descriptors_g;
        auto scales = upper ? scales_u : scales_g;
        uint wr = uint(expert0) * (upper ? STRIDE_U : STRIDE_G) + (upper ? BASE_U : 0u) + row;
        value = tq_dot_batch<T, COLUMN_G, WIDTH_G, K, GS_G, 2>(words, descriptors[wr], scales,
            relative, wr, x, sources, lane & 15u);
    }
    for (uint step = 8u; step; step >>= 1u) value += simd_shuffle_down(value, step);
    float4 up = simd_shuffle(value, 16u);
    if (lane == 0u) for (uint m = 0u; m < 2u; ++m) {
        float result = float(T(value[m])), other = float(T(up[m])), limit = params[0];
        if (limit > 0.f) { result = min(result, limit); other = clamp(other, -limit, limit); }
        result = GELU ? .5f * result * (1.f + tanh(.7978845608028654f * result * (1.f + .044715f * result * result)))
                      : result / (1.f + exp(-result));
        y[(first + m * ROUTES) * OUT + row] = T(result * other);
    }
    return;
}
)METAL") + std::strstr(kProject, "int expert = ids[route];");
        return fast::metal_kernel("mfq_cpp_tq_paired_routes",
            {"words_g", "descriptors_g", "scales_g", "relative", "words_u",
             "descriptors_u", "scales_u", "x", "ids", "params"}, {"y"}, source, kTqDecode, true, false);
    }();
    return kernel;
}

const fast::CustomKernelFunction& dense_pair_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_dense_pair",
        {"words_g", "descriptors_g", "scales_g", "relative", "words_u",
         "descriptors_u", "scales_u", "x", "ids", "params"}, {"y"}, R"METAL(
threadgroup T partial[2u * 2u * BM];
uint row = thread_position_in_grid.x / 64u, lane = thread_position_in_grid.x % 32u;
bool upper = (thread_position_in_threadgroup.x & 32u) != 0u;
int sources[BM];
#pragma unroll
for (uint m = 0u; m < BM; ++m) sources[m] = int(m);
float4 acc[BM];
#pragma unroll
for (uint m = 0u; m < BM; ++m) acc[m] = 0.f;
if (row < OUT) {
    auto packed_g = tq_words<POOLED_G>(words_g, descriptors_g, STRIDE_G, 0u);
    auto packed_u = tq_words<POOLED_U>(words_u, descriptors_u, STRIDE_U, 0u);
    tq_dot_pair_batch_accumulate<T, COLUMN_G, COLUMN_U, WIDTH_G, WIDTH_U, K, GS_G, GS_U, BM>(
        packed_g, descriptors_g[row], scales_g, packed_u, descriptors_u[row + BASE_U], scales_u,
        relative, row, row + BASE_U, x, sources, lane, upper, acc);
}
#pragma unroll
for (uint m = 0u; m < BM; ++m) {
    float value = acc[m].x + acc[m].y + acc[m].z + acc[m].w;
    for (uint step = 16u; step; step >>= 1u) value += simd_shuffle_down(value, step);
    if (lane == 0u) partial[(row & 1u) * 2u * BM + uint(upper) * BM + m] = T(value);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (!upper && lane == 0u && row < OUT) {
    for (uint m = 0u; m < BM; ++m) {
        float gate = float(partial[(row & 1u) * 2u * BM + m]),
            up = float(partial[(row & 1u) * 2u * BM + BM + m]), limit = params[0];
        if (limit > 0.f) { gate = min(gate, limit); up = clamp(up, -limit, limit); }
        gate = GELU ? .5f * gate * (1.f + tanh(.7978845608028654f * gate * (1.f + .044715f * gate * gate)))
                    : gate / (1.f + exp(-gate));
        y[m * OUT + row] = ids[m] == 0 ? T(gate * up) : T(0);
    }
}
)METAL", kTqDecode, true, false);
    return kernel;
}

const fast::CustomKernelFunction& reuse_kernel(bool nax, bool direct) {
    static std::map<std::pair<bool, bool>, fast::CustomKernelFunction> kernels;
    static std::mutex mutex;
    std::lock_guard lock(mutex);
    const auto key = std::pair{nax, direct};
    if (const auto found = kernels.find(key); found != kernels.end()) return found->second;
    std::string source;
    if (nax) source = R"METAL(
uint reuse_group = threadgroup_position_in_grid.y;
int reuse_expert = plan[reuse_group * (DIRECT_PLAN ? 18u : REUSE_ROWS + 1u)];
if (reuse_expert >= 0 && reuse_expert < EXPERTS) {
    uint first = reuse_group, last = reuse_group + 1u;
    uint count = 0u;
    uint4 group_rows = 0u;
    if constexpr (DIRECT_PLAN) count = uint(plan[reuse_group * 18u + 1u]);
    else {
        while (first > 0u && plan[(first - 1u) * (REUSE_ROWS + 1u)] == reuse_expert) --first;
        while (last < uint(plan_shape[0]) && plan[last * (REUSE_ROWS + 1u)] == reuse_expert) ++last;
        for (uint block = first; block < last; ++block) {
            uint rows = 0u;
            #pragma unroll
            for (uint m = 0u; m < REUSE_ROWS; ++m)
                rows += plan[block * (REUSE_ROWS + 1u) + m + 1u] >= 0;
            count += rows;
            if (block - first < 4u) group_rows[block - first] = rows;
        }
    }
    if (count >= MIN_NAX_ROWS && count <= 16u && (DIRECT_PLAN || last - first <= 4u)) {
        if (reuse_group != first) return;
        uint local = thread_position_in_threadgroup.x;
        uint n0 = threadgroup_position_in_grid.x * 128u + (local / 32u) * 32u;
        if (n0 >= OUT) return;
        auto output_row = [&](uint row) {
            if constexpr (DIRECT_PLAN) return uint(plan[reuse_group * 18u + row + 2u]);
            uint block = first;
            if (row >= group_rows.x) {
                row -= group_rows.x; ++block;
                if (row >= group_rows.y) {
                    row -= group_rows.y; ++block;
                    if (row >= group_rows.z) { row -= group_rows.z; ++block; }
                }
            }
            return uint(plan[block * (REUSE_ROWS + 1u) + row + 1u]);
        };
        auto input_row = [&](uint row) { return output_row(row) / (SHARED_INPUT ? ROUTES : 1u); };
        auto packed_g = tq_words<POOLED_G>(words_g, descriptors_g, EXPERTS * OUT, uint(reuse_expert));
        uint cached_row = n0 + uint(mlx::steel::BaseNAXFrag::get_coord().y)
            + ((__metal_get_thread_index_in_simdgroup(ushort()) & 1u) ? 8u : 0u);
        ulong descriptor0 = 0ul, descriptor1 = 0ul;
        if constexpr (!PAIR) {
            if (cached_row < OUT) descriptor0 = descriptors_g[uint(reuse_expert) * OUT + cached_row];
            if (cached_row + 16u < OUT) descriptor1 = descriptors_g[uint(reuse_expert) * OUT + cached_row + 16u];
        }
        auto decode_g = [&](uint row, uint col) {
            uint wr = uint(reuse_expert) * OUT + row;
            return tq_decode8<COLUMN_G, WIDTH_G, GS_G>(packed_g,
                PAIR ? descriptors_g[wr] : row == cached_row ? descriptor0 : descriptor1,
                scales_g, relative, wr, col);
        };
        float4 values[4];
        auto store_g = [&](uint row, uint col, float4 value) {
            if constexpr (PAIR) values[(row / 8u) * 2u + (col - n0) / 16u] = float4(vec<T, 4>(value));
            else tq_store_projection4<OUT>(y, output_row(row), col, value);
        };
        tq_nax_matmul_apply<T, K, OUT, 16>(x, count, n0, 0u, decode_g, input_row, store_g);
        if constexpr (PAIR) {
            auto packed_u = tq_words<POOLED_U>(words_u, descriptors_u, EXPERTS * OUT, uint(reuse_expert));
            auto decode_u = [&](uint row, uint col) {
                uint wr = uint(reuse_expert) * OUT + row;
                return tq_decode8<COLUMN_G, WIDTH_G, GS_G>(packed_u, descriptors_u[wr], scales_u, relative, wr, col);
            };
            auto store_u = [&](uint row, uint col, float4 value) {
                float4 gate = values[(row / 8u) * 2u + (col - n0) / 16u];
                float4 up = float4(vec<T, 4>(value));
                float limit = params[0];
                if (limit > 0.f) { gate = min(gate, limit); up = clamp(up, -limit, limit); }
                gate = GELU ? .5f * gate * (1.f + tanh(.7978845608028654f * gate * (1.f + .044715f * gate * gate)))
                            : gate / (1.f + exp(-gate));
                tq_store_projection4<OUT>(y, output_row(row), col, gate * up);
            };
            tq_nax_matmul_apply<T, K, OUT, 16>(x, count, n0, 0u, decode_u, input_row, store_u);
        }
        return;
    }
}
)METAL";
    if (!direct) source += "constexpr bool COLUMN_SINGLE = false;\ndevice const float* prepared_g = nullptr;\ndevice const float* prepared_u = nullptr;\n";
    source += R"METAL(
constexpr uint LANES = PAIR ? 32u : 16u;
constexpr bool COMPACT = REUSE_ROWS >= 4u && K >= 1024u && metal::is_same_v<T, half>;
uint row = thread_position_in_grid.x / LANES, lane = thread_position_in_grid.x % LANES;
uint group = threadgroup_position_in_grid.y;
auto selected = plan + group * (DIRECT_PLAN ? 18u : REUSE_ROWS + 1u);
const uint count = DIRECT_PLAN ? uint(selected[1]) : REUSE_ROWS;
if (row >= OUT || (DIRECT_PLAN ? count == 0u : selected[1] < 0)) return;
int expert = selected[0];
if constexpr (COLUMN_SINGLE) {
    if (count == 1u) {
        uint destination = uint(selected[2]);
        float result = 0.f, anchor = 0.f;
        if (expert >= 0 && expert < EXPERTS) {
            bool upper = PAIR && lane >= 16u;
            uint wr = uint(expert) * OUT + row;
            auto packed_g = tq_words<POOLED_G>(words_g, descriptors_g, EXPERTS * OUT, uint(expert));
            auto packed_u = tq_words<POOLED_U>(words_u, descriptors_u, EXPERTS * OUT, uint(expert));
            ulong descriptor = upper ? descriptors_u[wr] : descriptors_g[wr];
            auto prepared = (upper ? prepared_u : prepared_g) + destination * K;
            result = tq_dot<float, 16, 0, K, K, 512>(upper ? packed_u : packed_g,
                descriptor, TqUnitScale{}, relative, wr, prepared, lane & 15u);
            anchor = float(as_type<half>(ushort(descriptor >> 40u)));
        }
        for (uint step = 8u; step; step >>= 1u) result += simd_shuffle_down(result, step);
        result *= anchor;
        float up = PAIR ? simd_shuffle(result, 16u) : 0.f;
        if (lane == 0u) {
            result = float(T(result));
            if (PAIR) {
                float other = float(T(up)), limit = params[0];
                if (limit > 0.f) { result = min(result, limit); other = clamp(other, -limit, limit); }
                result = GELU ? .5f * result * (1.f + tanh(.7978845608028654f * result * (1.f + .044715f * result * result)))
                              : result / (1.f + exp(-result));
                result *= other;
            }
            y[destination * OUT + row] = T(result);
        }
        return;
    }
}
auto output_at = [&](uint offset, uint member) {
    if constexpr (DIRECT_PLAN) return offset + member < count ? selected[offset + member + 2u] : -1;
    else return selected[member + 1u];
};
for (uint offset = 0u; offset < count; offset += REUSE_ROWS) {
if constexpr (REUSE_ROWS == 6) {
    int outputs[REUSE_ROWS], sources[REUSE_ROWS];
    #pragma unroll
    for (uint m = 0u; m < REUSE_ROWS; ++m) {
        outputs[m] = output_at(offset, m);
        sources[m] = outputs[m] < 0 ? -1 : outputs[m] / (SHARED_INPUT ? int(ROUTES) : 1);
    }
    if (outputs[1] < 0) {
        float result = 0.f;
        if (expert >= 0 && expert < EXPERTS) {
            bool upper = PAIR && lane >= 16u;
            uint wr = uint(expert) * OUT + row;
            auto packed_g = tq_words<POOLED_G>(words_g, descriptors_g, EXPERTS * OUT, uint(expert));
            auto packed_u = tq_words<POOLED_U>(words_u, descriptors_u, EXPERTS * OUT, uint(expert));
            auto words = upper ? packed_u : packed_g;
            auto descriptors = upper ? descriptors_u : descriptors_g;
            auto scales = upper ? scales_u : scales_g;
            result = tq_dot<T, 16, COLUMN_G, WIDTH_G, K, GS_G>(words, descriptors[wr], scales,
                relative, wr, x + uint(sources[0]) * K, lane & 15u);
        }
        for (uint step = 8u; step; step >>= 1u) result += simd_shuffle_down(result, step);
        float up = PAIR ? simd_shuffle(result, 16u) : 0.f;
        if (lane == 0u) {
            result = float(T(result));
            if (PAIR) {
                float other = float(T(up)), limit = params[0];
                if (limit > 0.f) { result = min(result, limit); other = clamp(other, -limit, limit); }
                result = GELU ? .5f * result * (1.f + tanh(.7978845608028654f * result * (1.f + .044715f * result * result)))
                              : result / (1.f + exp(-result));
                result *= other;
            }
            y[uint(outputs[0]) * OUT + row] = T(result);
        }
        return;
    }
    float value[REUSE_ROWS];
    #pragma unroll
    for (uint m = 0u; m < REUSE_ROWS; ++m) value[m] = 0.f;
    if (expert >= 0 && expert < EXPERTS) {
        bool upper = PAIR && lane >= 16u;
        uint wr = uint(expert) * OUT + row;
        auto packed_g = tq_words<POOLED_G>(words_g, descriptors_g, EXPERTS * OUT, uint(expert));
        auto packed_u = tq_words<POOLED_U>(words_u, descriptors_u, EXPERTS * OUT, uint(expert));
        auto words = upper ? packed_u : packed_g;
        auto descriptors = upper ? descriptors_u : descriptors_g;
        auto scales = upper ? scales_u : scales_g;
        {
            float4 acc[REUSE_ROWS];
            tq_dot_batch_accumulate<T, COLUMN_G, WIDTH_G, K, GS_G, REUSE_ROWS, 16, COMPACT>(
                words, descriptors[wr], scales, relative, wr, x, sources, lane & 15u, acc);
            #pragma unroll
            for (uint m = 0u; m < REUSE_ROWS; ++m) value[m] = acc[m].x + acc[m].y + acc[m].z + acc[m].w;
        }
    }
    #pragma unroll
    for (uint m = 0u; m < REUSE_ROWS; ++m) {
        float result = value[m];
        for (uint step = 8u; step; step >>= 1u) result += simd_shuffle_down(result, step);
        float up = PAIR ? simd_shuffle(result, 16u) : 0.f;
        if (lane == 0u && outputs[m] >= 0) {
            result = float(T(result));
            if (PAIR) {
                float other = float(T(up)), limit = params[0];
                if (limit > 0.f) { result = min(result, limit); other = clamp(other, -limit, limit); }
                result = GELU ? .5f * result * (1.f + tanh(.7978845608028654f * result * (1.f + .044715f * result * result)))
                              : result / (1.f + exp(-result));
                result *= other;
            }
            y[uint(outputs[m]) * OUT + row] = T(result);
        }
    }
} else {
int4 outputs = int4(-1);
#pragma unroll
for (uint m = 0u; m < REUSE_ROWS; ++m) outputs[m] = output_at(offset, m);
int4 sources = select(int4(-1), outputs / (SHARED_INPUT ? int(ROUTES) : 1), outputs >= 0);
float4 value = 0.f;
if (expert >= 0 && expert < EXPERTS) {
    bool upper = PAIR && lane >= 16u;
    uint wr = uint(expert) * OUT + row;
    auto packed_g = tq_words<POOLED_G>(words_g, descriptors_g, EXPERTS * OUT, uint(expert));
    auto packed_u = tq_words<POOLED_U>(words_u, descriptors_u, EXPERTS * OUT, uint(expert));
    auto words = upper ? packed_u : packed_g;
    auto descriptors = upper ? descriptors_u : descriptors_g;
    auto scales = upper ? scales_u : scales_g;
    if (outputs.y < 0) value.x = tq_dot<T, 16, COLUMN_G, WIDTH_G, K, GS_G>(
        words, descriptors[wr], scales, relative, wr, x + uint(sources.x) * K, lane & 15u);
    else value = tq_dot_batch<T, COLUMN_G, WIDTH_G, K, GS_G, REUSE_ROWS, 16, COMPACT>(
        words, descriptors[wr], scales, relative, wr, x, sources, lane & 15u);
}
for (uint step = 8u; step; step >>= 1u) value += simd_shuffle_down(value, step);
float4 up = PAIR ? simd_shuffle(value, 16u) : float4(0.f);
if (lane == 0u) for (uint m = 0u; m < REUSE_ROWS; ++m) if (outputs[m] >= 0) {
    float result = float(T(value[m]));
    if (PAIR) {
        float other = float(T(up[m])), limit = params[0];
        if (limit > 0.f) { result = min(result, limit); other = clamp(other, -limit, limit); }
        result = GELU ? .5f * result * (1.f + tanh(.7978845608028654f * result * (1.f + .044715f * result * result)))
                      : result / (1.f + exp(-result));
        result *= other;
    }
    y[uint(outputs[m]) * OUT + row] = T(result);
}
}
}
)METAL";
    std::vector<std::string> inputs{"words_g", "descriptors_g", "scales_g", "relative", "words_u",
        "descriptors_u", "scales_u", "x", "plan", "params"};
    if (direct) inputs.insert(inputs.end(), {"prepared_g", "prepared_u"});
    return kernels.emplace(key, fast::metal_kernel(nax ? "mfq_cpp_tq_routed_reuse_nax" : "mfq_cpp_tq_routed_reuse",
        std::move(inputs), {"y"}, source,
        std::string("struct TqUnitScale {};\ninline float tq_group_scale(TqUnitScale, device const half*, uint, float) { return 1.f; }\n")
            + kTqDecode + (nax ? kTqNax : ""), true, false)).first->second;
}

const fast::CustomKernelFunction& embedding_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_embedding",
        {"words", "descriptors", "scales", "relative", "ids"}, {"y"},
        kEmbedding, kTqDecode, true, false);
    return kernel;
}

const fast::CustomKernelFunction& gated_mix_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_gated_mix",
        {"words", "descriptors", "scales", "relative", "x", "streams"}, {"y"}, R"METAL(
uint hidden = thread_position_in_grid.x / 32u, lane = thread_position_in_grid.x % 32u;
uint token = thread_position_in_grid.y;
if (hidden >= HIDDEN) return;
float total[BM];
int sources[BM];
#pragma unroll
for (uint m = 0u; m < BM; ++m) { total[m] = 0.f; sources[m] = int(m); }
auto packed = tq_words<POOLED>(words, descriptors, HIDDEN * GROUPS, 0u);
for (uint group = lane / 16u; group < GROUPS; group += 2u) {
    uint row = group * HIDDEN + hidden;
    float4 acc[BM];
    if constexpr (BM == 1) {
        acc[0] = float4(tq_dot<T, 16, COLUMNS, WIDTH, K, GS>(packed, descriptors[row], scales,
            relative, row, x + token * K, lane % 16u), 0.f, 0.f, 0.f);
    } else {
        tq_dot_batch_accumulate<T, COLUMNS, WIDTH, K, GS, BM, 16, COMPACT>(packed,
            descriptors[row], scales, relative, row, x, sources, lane % 16u, acc);
    }
    #pragma unroll
    for (uint m = 0u; m < BM; ++m) {
        float value = acc[m].x + acc[m].y + acc[m].z + acc[m].w;
        for (uint step = 8u; step; step >>= 1u) value += simd_shuffle_down(value, step);
        if (lane % 16u == 0u) {
            T gate = T(1.f / (1.f + exp(-float(T(value)))));
            total[m] += float(T(gate * streams[(token + m) * (HIDDEN * GROUPS) + row]));
        }
    }
}
#pragma unroll
for (uint m = 0u; m < BM; ++m) {
    total[m] += simd_shuffle(total[m], 16u);
    if (lane == 0u) y[(token + m) * HIDDEN + hidden] = T(total[m] / float(GROUPS));
}
)METAL", kTqDecode, true, false);
    return kernel;
}

const fast::CustomKernelFunction& routed_reduce_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_routed_reduce",
        {"words", "descriptors", "scales", "relative", "x", "ids", "weights"}, {"y"},
        kRoutedReduce, kTqDecode, true, false);
    return kernel;
}

const fast::CustomKernelFunction& gated_mix_nax_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_gated_mix_nax",
        {"words", "descriptors", "scales", "relative", "x", "streams"}, {"y"}, R"METAL(
uint n0 = threadgroup_position_in_grid.x * 32u, m0 = threadgroup_position_in_grid.y * BM;
uint local = thread_position_in_threadgroup.x, group = local / 32u;
uint rows = uint(x_shape[0]);
auto packed = tq_words<POOLED>(words, descriptors, HIDDEN * GROUPS, 0u);
auto decode = [&](uint row, uint col) {
    uint wr = group * HIDDEN + row;
    return tq_decode8<COLUMNS, WIDTH, GS>(packed, descriptors[wr], scales, relative, wr, col);
};
threadgroup T partial[GROUPS * BM * 32u];
auto store = [&](uint row, uint col, float4 values) {
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) if (col + j < HIDDEN) {
        T value = T(values[j]), gate = T(1.f / (1.f + exp(-float(value))));
        partial[(group * BM + row - m0) * 32u + col - n0 + j] =
            T(gate * streams[row * (HIDDEN * GROUPS) + group * HIDDEN + col + j]);
    }
};
auto identity = [](uint row) { return row; };
tq_nax_matmul_apply<T, K, HIDDEN, BM>(x, rows, n0, m0, decode, identity, store);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint index = local; index < BM * 32u; index += GROUPS * 32u) {
    uint m = index / 32u, n = index % 32u;
    if (m0 + m < rows && n0 + n < HIDDEN) {
        float even = 0.f, odd = 0.f;
        for (uint g = 0u; g < GROUPS; g += 2u) even += float(partial[(g * BM + m) * 32u + n]);
        for (uint g = 1u; g < GROUPS; g += 2u) odd += float(partial[(g * BM + m) * 32u + n]);
        y[(m0 + m) * HIDDEN + n0 + n] = T((even + odd) / float(GROUPS));
    }
}
)METAL", std::string(kTqDecode) + kTqNax, true, false);
    return kernel;
}

const fast::CustomKernelFunction& routed_nax_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_routed_nax",
        {"words", "descriptors", "scales", "relative", "x", "order", "blocks", "block_count"},
        {"y"}, R"METAL(
uint n0 = !NAX ? threadgroup_position_in_grid.x * 32u
    : SHARED_TILE ? threadgroup_position_in_grid.x * 64u : (thread_position_in_grid.x / 32u) * 32u;
uint block = thread_position_in_grid.y;
if (n0 >= OUT || block >= uint(block_count[0])) return;
uint begin = uint(blocks[block * 3u]);
int expert = blocks[block * 3u + 1u];
uint rows = begin + uint(blocks[block * 3u + 2u]);
if (rows <= begin) return;
if (expert < 0 || expert >= EXPERTS) {
    if constexpr (!NAX) tq_zero_rows<T, OUT, 32>(y, order, begin, rows - begin,
        threadgroup_position_in_grid.x * 32u, thread_position_in_threadgroup.x, BM * 4u);
    else if constexpr (SHARED_TILE) tq_zero_rows<T, OUT, 64>(y, order, begin, rows - begin,
        threadgroup_position_in_grid.x * 64u, thread_position_in_threadgroup.x, BM * 2u);
    else tq_zero_rows<T, OUT, 128>(y, order, begin, rows - begin,
        threadgroup_position_in_grid.x * 128u, thread_position_in_threadgroup.x, 128u);
    return;
}
auto packed = tq_words<POOLED>(words, descriptors, EXPERTS * OUT, uint(expert));
ulong row_descriptor = 0ul;
if constexpr (NAX && SHARED_TILE) {
    uint row = n0 + thread_position_in_threadgroup.x / (BM / 32u);
    if (row < OUT) row_descriptor = descriptors[uint(expert) * OUT + row];
}
auto decode = [&](uint row, uint col) {
    uint wr = uint(expert) * OUT + row;
    if constexpr (NAX)
        return tq_decode8<COLUMNS, WIDTH, GS>(packed, SHARED_TILE ? row_descriptor : descriptors[wr], scales, relative, wr, col);
    else return tq_decode4<COLUMNS, WIDTH, GS>(packed, descriptors[wr], scales, relative, wr, col);
};
auto input_row = [&](uint row) { return uint(order[row]) / (SHARED_INPUT ? ROUTES : 1u); };
auto output_row = [&](uint row) { return uint(order[row]); };
if constexpr (!NAX) {
    threadgroup half shared_a[BM * 32u], shared_b[32u * 32u];
    tq_simd_matmul<T, K, OUT>(x, y, rows, begin, n0, thread_position_in_threadgroup.x,
        decode, input_row, output_row, shared_a, shared_b);
} else if constexpr (SHARED_TILE) {
    threadgroup half scratch[BM * 32u + 64u * 32u];
    if (BM == 128 && rows - begin <= 64u)
        tq_nax_shared_matmul<T, K, OUT, BM, 16>(x, y, rows, begin, n0,
            thread_position_in_threadgroup.x, decode, input_row, output_row, scratch, scratch + BM * 32u);
    else if (BM == 64 && rows - begin <= 32u)
        tq_nax_shared_matmul<T, K, OUT, BM, 16, (OUT >= K * 2u)>(x, y, rows, begin, n0,
            thread_position_in_threadgroup.x, decode, input_row, output_row, scratch,
            OUT >= K * 2u ? scratch : scratch + BM * 32u);
    else if constexpr (BM == 64 && OUT >= K * 2u)
        tq_nax_shared_matmul<T, K, OUT, BM, 32, true>(x, y, rows, begin, n0,
            thread_position_in_threadgroup.x, decode, input_row, output_row, scratch, scratch);
    else tq_nax_shared_matmul<T, K, OUT, BM, 32>(x, y, rows, begin, n0,
        thread_position_in_threadgroup.x, decode, input_row, output_row, scratch, scratch + BM * 32u);
} else tq_nax_matmul<T, K, OUT, BM>(x, y, rows, n0, begin, decode, input_row, output_row);
)METAL", std::string(kTqDecode) + kTqNax, true, false);
    return kernel;
}

MlxTqRoutePlan make_tq_route_plan(const array& ids, const array& order, int experts, int block_rows) {
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_route_blocks", {"ids", "order"}, {"blocks", "count"}, R"METAL(
uint expert = thread_position_in_threadgroup.x;
uint routes = order_shape[0];
threadgroup atomic_uint total;
if (expert == 0u) atomic_store_explicit(&total, 0u, memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
uint lo = 0u, hi = routes;
while (lo < hi) {
    uint mid = (lo + hi) / 2u;
    if (ids[order[mid]] < int(expert)) lo = mid + 1u;
    else hi = mid;
}
uint begin = lo;
hi = routes;
while (lo < hi) {
    uint mid = (lo + hi) / 2u;
    if (ids[order[mid]] <= int(expert)) lo = mid + 1u;
    else hi = mid;
}
for (uint row = begin; row < lo; row += BM) {
    uint block = atomic_fetch_add_explicit(&total, 1u, memory_order_relaxed);
    blocks[block * 3u] = int(row);
    blocks[block * 3u + 1u] = int(expert);
    blocks[block * 3u + 2u] = int(min(uint(BM), lo - row));
}
if (expert == 0u) for (uint row = 0u; row < begin; row += BM) {
    uint block = atomic_fetch_add_explicit(&total, 1u, memory_order_relaxed);
    blocks[block * 3u] = int(row);
    blocks[block * 3u + 1u] = -1;
    blocks[block * 3u + 2u] = int(min(uint(BM), begin - row));
}
if (expert + 1u == EXPERTS) for (uint row = lo; row < routes; row += BM) {
    uint block = atomic_fetch_add_explicit(&total, 1u, memory_order_relaxed);
    blocks[block * 3u] = int(row);
    blocks[block * 3u + 1u] = -1;
    blocks[block * 3u + 2u] = int(min(uint(BM), routes - row));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (expert == 0u) count[0] = int(atomic_load_explicit(&total, memory_order_relaxed));
)METAL", "", true, false);
    if (ids.ndim() != 2 || ids.size() == 0 || ids.size() > std::size_t(INT_MAX)
        || order.ndim() != 1 || order.size() != ids.size() || experts <= 0 || experts > 1024
        || (block_rows != 16 && block_rows != 32 && block_rows != 64 && block_rows != 128))
        throw std::invalid_argument("TQ route plan geometry differs");
    const int routes = ids.size();
    const int max_blocks = std::min<std::int64_t>(routes,
        (std::int64_t(routes) + block_rows - 1) / block_rows + experts + 2);
    auto result = kernel({contiguous(reshape(astype(ids, int32), {-1})), contiguous(astype(order, int32))},
        {Shape{max_blocks, 3}, Shape{1}}, {int32, int32}, {experts, 1, 1}, {experts, 1, 1},
        {{"BM", block_rows}, {"EXPERTS", experts}}, std::nullopt, false, {});
    return {result[0], result[1], max_blocks, block_rows, routes, experts};
}

const fast::CustomKernelFunction& pair_activation_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_pair_activation", {"g", "u", "params", "ids"}, {"y"}, R"METAL(
uint index = thread_position_in_grid.x;
uint row = index / OUT, col = index % OUT;
if constexpr (MASK_SINGLE_EXPERT) if (ids[row] != 0) { y[index] = T(0); return; }
float gate = float(g[SPLIT ? row * (2u * OUT) + col : index]);
float up = float(u[SPLIT ? row * (2u * OUT) + OUT + col : index]);
float limit = params[0];
if (limit > 0.f) { gate = min(gate, limit); up = clamp(up, -limit, limit); }
gate = GELU ? .5f * gate * (1.f + tanh(.7978845608028654f * gate * (1.f + .044715f * gate * gate)))
            : gate / (1.f + exp(-gate));
y[index] = T(gate * up);
)METAL", "", true, false);
    return kernel;
}

bool has_nax() {
    return !mlx_metal_nax_disabled() && mlx_apple_chip_starts_with("Apple M5");
}

bool use_nax(const array& input, int rows = 0) {
    return input.dtype() == float16 && (rows ? rows : input.size() / input.shape(-1)) >= 7
        && has_nax();
}

bool use_routed_simd(const array& input, std::size_t routes, int experts) {
    return input.dtype() == float16 && input.size() / input.shape(-1) >= 256
        && routes / experts >= 32;
}

bool staged_weights_fit(int width, std::int64_t outputs, Dtype dtype) {
    const auto bytes = dtype == float16 ? sizeof(float16_t) : sizeof(float);
    return std::uint64_t(outputs) * width * bytes <= (std::uint64_t{256} << 20);
}

bool stage_weights(std::size_t tokens, int width, std::int64_t outputs, Dtype dtype) {
    const bool contraction = dtype == float16 && tokens >= 17
        && width >= 8192 && outputs <= width / 8
        && (tokens > 32 || std::uint64_t(outputs) * width >= (std::uint64_t{4} << 20));
    const bool short_expansion = dtype == float16 && width >= 256 && width < 512
        && outputs >= 1024 && tokens >= 32 && (tokens >= 64 || !has_nax());
    if ((!contraction && !short_expansion && tokens < (dtype == float16 ? 128 : 256))
        || (dtype != float16 && outputs < 64 && tokens < 1024)) return false;
    return staged_weights_fit(width, outputs, dtype);
}

bool stage_weights(const array& input, std::int64_t outputs) {
    return stage_weights(input.size() / input.shape(-1), input.shape(-1), outputs, input.dtype());
}

int small_batch_rows(int tokens, int outputs, int width) {
    if (tokens < 2 || width < 256 || outputs < 256 || outputs < width / 4
        || (width < 512 && outputs < 1024)) return 1;
    if (outputs >= 1024) return tokens == 2 ? 2 : tokens == 5 || tokens == 6 ? 6 : 4;
    return tokens >= 4 ? 2 : 1;
}

int split_count(int tokens, int outputs, int width, int batch_rows = 1, bool fused = false) {
    if (tokens > 32 || width < (fused ? 2048 : 8192) || outputs > width / (fused ? 4 : 8)) return 1;
    int splits = 2;
    const auto groups = ((std::int64_t(outputs) + 7) / 8) * ((tokens + batch_rows - 1) / batch_rows);
    while (splits < 32 && groups * splits < 160) splits *= 2;
    return splits;
}

int split_batch_rows(int tokens, int outputs, int width) {
    if (tokens < 2 || split_count(tokens, outputs, width) == 1) return 1;
    if (tokens == 2 && outputs < width / 16) return 1;
    for (int rows : {4, 2}) {
        const auto groups = ((std::int64_t(outputs) + 7) / 8) * ((tokens + rows - 1) / rows)
            * split_count(tokens, outputs, width, rows);
        if (tokens >= rows && groups >= (rows == 4 ? 256 : 160)) return rows;
    }
    return 1;
}

int nax_split_count(int tokens, int outputs, int width, int block_rows) {
    const auto groups = ((std::int64_t(outputs) + 127) / 128) * ((tokens + block_rows - 1) / block_rows);
    if (width < 1024 || groups >= 160) return 1;
    const auto rings = (std::int64_t(width) + 255) / 256;
    const auto desired = std::min<std::int64_t>(32, (160 + groups - 1) / groups);
    const auto chunk = (rings + desired - 1) / desired;
    return int((rings + chunk - 1) / chunk);
}

const fast::CustomKernelFunction& split_reduce_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_split_reduce", {"parts"}, {"y"}, R"METAL(
uint index = thread_position_in_grid.x;
uint count = uint(parts_shape[1]) * TOTAL;
if (index >= count) return;
float value = 0.f;
for (uint split = 0u; split < SPLITS; ++split) value += parts[split * count + index];
y[index] = T(value);
)METAL", "", true, false);
    return kernel;
}

enum class TqBatchMode { Vector, Reuse, Nax, NaxSplit, SharedNax, SplitK, SplitReuse, FusedSplitReuse, StageWeights };

const fast::CustomKernelFunction& batch_kernel(std::size_t count, TqBatchMode mode,
    bool select_decode = false, bool fixed_sources = false) {
    const bool shared = mode == TqBatchMode::SharedNax;
    const bool fused_split = mode == TqBatchMode::FusedSplitReuse;
    const bool nax = shared || mode == TqBatchMode::Nax || mode == TqBatchMode::NaxSplit;
    const bool small = mode == TqBatchMode::Reuse || mode == TqBatchMode::SplitReuse || fused_split;
    const bool split = mode == TqBatchMode::SplitK || mode == TqBatchMode::SplitReuse || mode == TqBatchMode::NaxSplit || fused_split;
    const bool staging = mode == TqBatchMode::StageWeights;
    const bool cache_rows = nax && !shared;
    static std::mutex mutex;
    static std::map<std::tuple<std::size_t, TqBatchMode, bool, bool>, fast::CustomKernelFunction> kernels;
    std::lock_guard lock(mutex);
    const auto key = std::tuple{count, mode, select_decode, fixed_sources};
    const auto found = kernels.find(key);
    if (found != kernels.end()) return found->second;
    std::vector<std::string> names{"x", "relative"};
    std::string source = fused_split ? R"METAL(
uint row = threadgroup_position_in_grid.x;
uint local_thread = thread_position_in_threadgroup.x;
uint part = local_thread / 32u, lane = local_thread % 32u;
int sources[BM];
)METAL" : staging ? R"METAL(
uint col = thread_position_in_grid.x * 4u;
uint row = thread_position_in_grid.y;
if (col >= K || row >= TOTAL) return;
half4 value = 0.h;
)METAL" : shared ? R"METAL(
uint n0 = threadgroup_position_in_grid.x * 64u;
uint m0 = threadgroup_position_in_grid.y * 128u;
uint rows = uint(x_shape[0]);
if (n0 >= TOTAL || m0 >= rows) return;
uint cached_row = n0 + thread_position_in_threadgroup.x / 4u;
ulong row_descriptor = 0ul;
)METAL" : nax ? R"METAL(
uint n0 = (thread_position_in_grid.x / 32u) * 32u;
uint m0 = thread_position_in_grid.y * BM;
uint rows = uint(x_shape[0]);
if (n0 >= TOTAL || m0 >= rows) return;
)METAL" : small ? R"METAL(
uint row = thread_position_in_grid.x / LANES;
uint lane = thread_position_in_grid.x % LANES;
uint token = thread_position_in_grid.y * BM;
if (row >= TOTAL) return;
int sources[BM];
)METAL" : R"METAL(
uint row = thread_position_in_grid.x / LANES;
uint lane = thread_position_in_grid.x % LANES;
uint token = thread_position_in_grid.y;
if (row >= TOTAL) return;
float value = 0.f;
)METAL";
    if (small) source += std::string("#pragma unroll\nfor (uint i = 0u; i < BM; ++i) sources[i] = ")
        + (fixed_sources ? "int(i)" : "token + i < uint(x_shape[0]) ? int(token + i) : -1")
        + ";\nfloat4 acc[BM];\n";
    if (split && !nax) source += std::string("uint begin = ")
        + (fused_split ? "part" : "thread_position_in_grid.z")
        + " * CHUNK;\nuint end = min(uint(K), begin + CHUNK);\n";
    if (nax && !shared) {
        if (cache_rows) source += R"METAL(
uint cached_row0 = n0 + uint(mlx::steel::BaseNAXFrag::get_coord().y)
    + ((__metal_get_thread_index_in_simdgroup(ushort()) & 1u) ? 8u : 0u);
uint cached_row1 = cached_row0 + 16u;
ulong descriptor0 = 0ul, descriptor1 = 0ul;
)METAL";
        if (cache_rows) source += "constexpr bool CACHE_ROWS = "
            + std::string(count == 1 ? "K >= 4096" : "BM <= 16") + ";\nif constexpr (CACHE_ROWS) {\n";
        if (cache_rows) for (std::size_t p = 0; p < count; ++p) {
            const auto suffix = std::to_string(p);
            source += "if (cached_row0 >= BEGIN" + suffix + " && cached_row0 < END" + suffix
                + ") descriptor0 = descriptors" + suffix + "[cached_row0 - BEGIN" + suffix + "];\n";
            source += "if (cached_row1 >= BEGIN" + suffix + " && cached_row1 < END" + suffix
                + ") descriptor1 = descriptors" + suffix + "[cached_row1 - BEGIN" + suffix + "];\n";
        }
        if (cache_rows) source += "}\n";
        source += "auto decode = [&](uint row, uint col) -> TqDecoded8 {\n";
    }
    if (shared) {
        for (std::size_t p = 0; p < count; ++p) {
            const auto suffix = std::to_string(p);
            source += "if (cached_row >= BEGIN" + suffix + " && cached_row < END" + suffix
                + ") row_descriptor = descriptors" + suffix + "[cached_row - BEGIN" + suffix + "];\n";
        }
        source += "auto decode = [&](uint row, uint col) -> TqDecoded8 {\n";
    }
    if (select_decode) source += "auto packed = words0;\nauto selected_scales = scales0;\nulong descriptor = 0ul;\nuint local = 0u;\n";
    for (std::size_t p = 0; p < count; ++p) {
        auto suffix = std::to_string(p);
        for (const auto* label : {"words", "descriptors", "scales"}) names.push_back(label + suffix);
        if (select_decode) {
            source += "if (row >= BEGIN" + suffix + " && row < END" + suffix + ") {\n"
                "local = row - BEGIN" + suffix + ";\n"
                "packed = tq_words<POOLED" + suffix + ">(words" + suffix + ", descriptors" + suffix
                + ", END" + suffix + " - BEGIN" + suffix + ", 0u);\n"
                "descriptor = descriptors" + suffix + "[local];\nselected_scales = scales" + suffix + ";\n}\n";
            continue;
        }
        source += "if (row >= BEGIN" + suffix + " && row < END" + suffix + ") {\n";
        source += "uint local = row - BEGIN" + suffix + ";\n";
        source += "auto packed = tq_words<POOLED" + suffix + ">(words" + suffix + ", descriptors" + suffix
            + ", END" + suffix + " - BEGIN" + suffix + ", 0u);\n";
        if (staging || nax) source += std::string(staging ? "value = " : "return ")
            + (staging ? "tq_decode4<COLUMN" : "tq_decode8<COLUMN") + suffix + ", WIDTH" + suffix + ", GS" + suffix
            + ">(packed, " + (shared ? std::string("row_descriptor") : cache_rows
                ? "(CACHE_ROWS ? (row == cached_row0 ? descriptor0 : descriptor1) : descriptors" + suffix + "[local])"
                : "descriptors" + suffix + "[local]") + ", scales" + suffix
            + ", relative, local, col);\n}\n";
        else if (small) source += "tq_dot_batch_accumulate<T, COLUMN" + suffix + ", WIDTH" + suffix + ", K, GS" + suffix
                + ", BM, LANES, " + (count == 1 ? "(BM >= 3 && ((K >= 4096 && TOTAL <= K / 2) || (K >= 256 && K < 512 && TOTAL >= 1024)) && metal::is_same_v<T, half>)"
                    : split ? "(BM >= 3 && K >= 8192 && TOTAL >= 256 && TOTAL <= K / 8 && metal::is_same_v<T, half>)" : "false")
                + ">(packed, descriptors" + suffix + "[local], scales" + suffix
                + ", relative, local, x, sources, lane, acc" + (split ? ", begin, end" : "") + ");\n}\n";
        else source += "value = tq_dot<T, LANES, COLUMN" + suffix + ", WIDTH" + suffix + ", K, GS" + suffix
                + ">(packed, descriptors" + suffix + "[local], scales" + suffix
                + ", relative, local, x + token * K, lane" + (split ? ", begin, end" : "") + ");\n}\n";
    }
    if (select_decode) source += "value = tq_dot<T, LANES, COLUMN0, WIDTH0, K, GS0>(packed, descriptor, selected_scales, relative, local, x + token * K, lane);\n";
    source += staging ? R"METAL(
tq_store4<K>(y, row, col, value);
)METAL" : shared ? R"METAL(
return TqDecoded8{half4(0.h), half4(0.h)};
};
threadgroup half shared_a[128u * 32u], shared_b[64u * 32u];
auto identity = [](uint row) { return row; };
tq_nax_shared_matmul<T, K, TOTAL, 128, 32>(x, y, rows, m0, n0,
    thread_position_in_threadgroup.x, decode, identity, identity, shared_a, shared_b);
)METAL" : nax ? std::string(R"METAL(
return TqDecoded8{half4(0.h), half4(0.h)};
};
auto identity = [](uint row) { return row; };
)METAL") + (split ? R"METAL(
uint begin = thread_position_in_grid.z * CHUNK;
auto destination = [&](uint row) { return thread_position_in_grid.z * rows + row; };
tq_nax_matmul<T, K, TOTAL, BM>(x, y, rows, n0, m0, decode, identity, destination,
    begin, min(uint(K), begin + CHUNK));
)METAL" : "tq_nax_matmul<T, K, TOTAL, BM>(x, y, rows, n0, m0, decode, identity, identity);\n")
    : fused_split ? R"METAL(
threadgroup float partial[SPLITS * BM];
#pragma unroll
for (uint i = 0u; i < BM; ++i) {
    float value = acc[i].x + acc[i].y + acc[i].z + acc[i].w;
    for (uint step = 16u; step; step >>= 1u) value += simd_shuffle_down(value, step);
    if (lane == 0u) partial[part * BM + i] = value;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (local_thread < BM) {
    float value = 0.f;
    for (uint p = 0u; p < SPLITS; ++p) value += partial[p * BM + local_thread];
    y[local_thread * TOTAL + row] = T(value);
}
)METAL" : small ? std::string(R"METAL(
float value[BM];
#pragma unroll
for (uint i = 0u; i < BM; ++i) value[i] = acc[i].x + acc[i].y + acc[i].z + acc[i].w;
for (uint step = LANES / 2u; step; step >>= 1)
    for (uint i = 0u; i < BM; ++i) value[i] += simd_shuffle_down(value[i], step);
if (lane == 0u) for (uint i = 0u; i < BM; ++i)
    if (sources[i] >= 0) y[
)METAL") + (split ? "thread_position_in_grid.z * uint(x_shape[0]) * TOTAL + " : "")
    + "uint(sources[i]) * TOTAL + row] = " + (split ? "value[i];\n" : "T(value[i]);\n") : split ? R"METAL(
for (uint step = LANES / 2; step; step >>= 1) value += simd_shuffle_down(value, step);
if (lane == 0u) y[(thread_position_in_grid.z * uint(x_shape[0]) + token) * TOTAL + row] = value;
)METAL" : R"METAL(
for (uint step = LANES / 2; step; step >>= 1) value += simd_shuffle_down(value, step);
if (lane == 0) y[token * TOTAL + row] = T(value);
)METAL";
    auto header = std::string(kTqDecode);
    if (nax) header += kTqNax;
    return kernels.emplace(key, fast::metal_kernel("mfq_cpp_tq_batch_" + std::to_string(count)
        + (select_decode ? "_select" : staging ? "_stage" : shared ? "_nax_shared" : nax ? split ? "_nax_split" : "_nax" : fused_split ? "_split_fused" : small && split ? "_split_reuse"
            : small ? "_small" : split ? "_split" : "") + (fixed_sources ? "_fixed" : ""),
        names, {"y"}, source, header, true, false)).first->second;
}

const fast::CustomKernelFunction& grouped_kernel(bool nax) {
    static std::mutex mutex;
    static std::map<bool, fast::CustomKernelFunction> kernels;
    std::lock_guard lock(mutex);
    if (const auto found = kernels.find(nax); found != kernels.end()) return found->second;
    const std::string source = std::string(R"METAL(
uint rows = FIXED ? BM : uint(sizes[0]);
auto packed = tq_words<POOLED>(words, descriptors, TOTAL, 0u);
uint begin = thread_position_in_grid.z * CHUNK, end = min(uint(K), begin + CHUNK);
)METAL") + (nax ? R"METAL(
constexpr uint TILES = (OUT + 127u) / 128u;
uint block = threadgroup_position_in_grid.x;
uint group = block / TILES;
uint n0 = group * OUT + (block % TILES) * 128u + (thread_position_in_threadgroup.x / 32u) * 32u;
uint n_end = (group + 1u) * OUT, m0 = thread_position_in_grid.y * BM;
if (n0 >= n_end) return;
auto decode = [&](uint row, uint col) { return tq_decode8<COLUMNS, WIDTH, GS>(packed,
    descriptors[row], scales, relative, row, col); };
auto input_row = [&](uint row) { return row * GROUPS + group; };
auto output_row = [&](uint row) { return thread_position_in_grid.z * rows + row; };
tq_nax_matmul<T, K, TOTAL, BM>(x, y, rows, n0, m0, decode, input_row, output_row, begin, end, n_end);
)METAL" : R"METAL(
uint row = thread_position_in_grid.x / LANES, lane = thread_position_in_grid.x % LANES;
uint token = FIXED ? 0u : thread_position_in_grid.y * BM;
if (row >= TOTAL) return;
uint group = row / OUT;
int sources[BM];
#pragma unroll
for (uint m = 0u; m < BM; ++m)
    sources[m] = FIXED ? int(m * GROUPS + group)
        : token + m < rows ? int((token + m) * GROUPS + group) : -1;
float value[BM];
if constexpr (BM == 1) value[0] = tq_dot<T, LANES, COLUMNS, WIDTH, K, GS>(packed, descriptors[row],
    scales, relative, row, x + uint(sources[0]) * K, lane, begin, end);
else {
    float4 acc[BM];
    tq_dot_batch_accumulate<T, COLUMNS, WIDTH, K, GS, BM, LANES, COMPACT>(packed, descriptors[row], scales,
        relative, row, x, sources, lane, acc, begin, end);
    #pragma unroll
    for (uint m = 0u; m < BM; ++m) value[m] = acc[m].x + acc[m].y + acc[m].z + acc[m].w;
}
for (uint step = LANES / 2u; step; step >>= 1u)
    for (uint m = 0u; m < BM; ++m) value[m] += simd_shuffle_down(value[m], step);
if (lane == 0u) for (uint m = 0u; m < BM; ++m) if (token + m < rows) {
    uint index = (thread_position_in_grid.z * rows + token + m) * TOTAL + row;
    y[index] = O(value[m]);
}
)METAL");
    auto header = std::string(kTqDecode);
    if (nax) header += kTqNax;
    return kernels.emplace(nax, fast::metal_kernel(nax ? "mfq_cpp_tq_grouped_nax" : "mfq_cpp_tq_grouped_vector",
        {"words", "descriptors", "scales", "relative", "x", "sizes"}, {"y"}, source, header, true, false)).first->second;
}

const fast::CustomKernelFunction& rms_projection_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_rms_projection",
        {"words", "descriptors", "scales", "relative", "x", "params"}, {"y"}, R"METAL(
constexpr uint THREADS = 512u, GROUPS = THREADS / 32u;
constexpr uint CHUNK = ((K + GROUPS * 256u - 1u) / (GROUPS * 256u)) * 256u;
uint row = threadgroup_position_in_grid.x, token = threadgroup_position_in_grid.y;
uint local = thread_position_in_threadgroup.x, lane = local & 31u, group = local / 32u;
threadgroup float partial[GROUPS], inverse[1];
float4 square = 0.f;
for (uint col = local * 4u; col < K; col += THREADS * 4u) {
    float4 value = tq_input4<K>(x + token * K, col, 0u);
    square += value * value;
}
float value = simd_sum(square.x + square.y + square.z + square.w);
if (lane == 0u) partial[group] = value;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group == 0u) {
    value = simd_sum(lane < GROUPS ? partial[lane] : 0.f);
    if (lane == 0u) inverse[0] = metal::precise::rsqrt(value / float(K) + params[0]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint begin = group * CHUNK;
value = tq_dot<T, 32, COLUMNS, WIDTH, K, GS>(tq_words<POOLED>(words, descriptors, OUT, 0u),
    descriptors[row], scales, relative, row, TqRmsInput<T>{x + token * K, inverse[0]},
    lane, begin, min(uint(K), begin + CHUNK));
value = simd_sum(value);
if (lane == 0u) partial[group] = value;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group == 0u) {
    value = simd_sum(lane < GROUPS ? partial[lane] : 0.f);
    if (lane == 0u) y[token * OUT + row] = value;
}
)METAL", kTqDecode, true, false);
    return kernel;
}

const fast::CustomKernelFunction& rms_nax_projection_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_rms_nax_projection",
        {"x", "words", "descriptors", "scales", "relative"}, {"parts"}, R"METAL(
uint n0 = threadgroup_position_in_grid.x * 32u, m0 = threadgroup_position_in_grid.y * 16u;
uint rows = uint(x_shape[0]), begin = threadgroup_position_in_grid.z * CHUNK;
auto packed = tq_words<POOLED>(words, descriptors, OUT, 0u);
auto decode = [&](uint row, uint col) {
    return tq_decode8<COLUMNS, WIDTH, GS>(packed, descriptors[row], scales, relative, row, col);
};
auto identity = [](uint row) { return row; };
auto destination = [&](uint row) { return threadgroup_position_in_grid.z * rows + row; };
tq_nax_matmul<T, K, OUT, 16>(x, parts, rows, n0, m0, decode, identity, destination,
    begin, min(uint(K), begin + CHUNK));
)METAL", std::string(kTqDecode) + kTqNax, true, false);
    return kernel;
}

const fast::CustomKernelFunction& rms_nax_reduce_kernel() {
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_rms_nax_reduce",
        {"x", "parts", "params"}, {"y"}, R"METAL(
uint row = thread_position_in_grid.x / 32u, lane = thread_position_in_grid.x % 32u;
if (row >= uint(x_shape[0])) return;
float4 square = 0.f;
for (uint col = lane * 4u; col < K; col += 128u) {
    float4 value = tq_input4<K>(x + row * K, col, 0u);
    square += value * value;
}
float inverse = metal::precise::rsqrt(simd_sum(square.x + square.y + square.z + square.w) / float(K) + params[0]);
for (uint col = lane; col < OUT; col += 32u) {
    float value = 0.f;
    for (uint part = 0u; part < SPLITS; ++part) value += parts[(part * uint(x_shape[0]) + row) * OUT + col];
    y[row * OUT + col] = value * inverse;
}
)METAL", kTqDecode, true, false);
    return kernel;
}

const array& relative_table() {
    static const auto table = [] {
        std::vector<float> values;
        for (int i = 0; i < 16; ++i) values.push_back(std::exp2(-2.f + 4.f * i / 15.f));
        auto result = astype(array(values.begin(), Shape{16}), float16);
        result.eval();
        return result;
    }();
    return table;
}

template <typename T> T load_le(std::span<const std::uint8_t> data, std::size_t offset) {
    if (offset > data.size() || sizeof(T) > data.size() - offset)
        throw std::runtime_error("truncated TQ payload");
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}

array copy_array(const std::uint8_t* data, std::size_t count, Dtype dtype) {
    if (count > std::size_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("TQ array exceeds backend dimension limit");
    auto result = array(allocator::malloc(count * size_of(dtype)), Shape{int(count)}, dtype);
    std::memcpy(result.data<std::uint8_t>(), data, count * size_of(dtype));
    MlxWeightResidency::track(result);
    return result;
}

array prepare_scales(std::span<const std::uint8_t> blob, std::size_t offset,
    std::size_t rows, std::size_t groups) {
    const auto count = rows * groups;
    if (count > std::size_t(INT_MAX)) throw std::runtime_error("TQ scale metadata exceeds backend dimension limit");
    auto result = array(allocator::malloc(count * sizeof(float)), Shape{int(count)}, float32);
    auto* target = result.data<float>();
    const auto* relative = relative_table().data<float16_t>();
    for (std::size_t row = 0; row < rows; ++row) {
        const auto descriptor = load_le<std::uint64_t>(blob, 64 + row * 8);
        const float anchor = float(std::bit_cast<float16_t>(std::uint16_t(descriptor >> 40)));
        for (std::size_t group = row * groups; group < (row + 1) * groups; ++group) {
            const auto code = (blob[offset + group / 2] >> ((group & 1) * 4)) & 15;
            target[group] = anchor * float(relative[code]);
        }
    }
    MlxWeightResidency::track(result);
    return result;
}
} // namespace

MlxTqWeight::MlxTqWeight(array words, array descriptors, array scales,
    int rows, int width, int input_columns, int experts, int group_size,
    int columns, std::size_t bytes, bool pooled)
    : words_(std::move(words)), descriptors_(std::move(descriptors)), scales_(std::move(scales)),
      rows_(rows), width_(width), input_columns_(input_columns), experts_(experts),
      group_size_(group_size), columns_(columns), pooled_(pooled), nbytes_(bytes) {}

MlxTqWeight MlxTqWeight::from_blob(std::span<const std::uint8_t> blob,
    int input_columns, int experts) {
    return from_blob_impl(blob, input_columns, experts, nullptr, {});
}

MlxTqWeight MlxTqWeight::from_pool(std::span<const std::uint8_t> blob, const array& words,
        std::span<const std::uint64_t> expert_offsets, int input_columns) {
    if (expert_offsets.empty() || expert_offsets.size() > std::size_t(INT_MAX)
        || words.dtype() != uint32 || !words.flags().row_contiguous || words.size() == 0)
        throw std::invalid_argument("invalid TQ word pool");
    return from_blob_impl(blob, input_columns, int(expert_offsets.size()), &words, expert_offsets);
}

MlxTqWeight MlxTqWeight::from_expert_blob(std::span<const std::uint8_t> blob,
        int input_columns, int experts, const array* word_pool, std::span<const std::uint64_t> expert_offsets) {
    const MlxMappedTqRows table(blob, input_columns);
    return from_expert_rows(table, experts, word_pool, expert_offsets);
}

MlxTqWeight MlxTqWeight::from_expert_rows(const MlxMappedTqRows& table,
        int experts, const array* word_pool, std::span<const std::uint64_t> expert_offsets) {
    if (!table.expert_container()) throw std::invalid_argument("TQ expert rows require an expert container");
    const int count = int(table.expert_count()), rows = int(table.rows());
    const int out = rows / count, width = table.stored_width();
    if (experts < 1 || (experts != 1 && experts != count)
        || (word_pool && expert_offsets.size() != std::size_t(count))
        || rows > INT_MAX - (word_pool ? count : 0))
        throw std::runtime_error("TQ expert container shape differs");
    const bool columns = table.columns();
    int gs = table.group_size();
    std::size_t words = 0;
    for (int expert = 0; expert < count; ++expert) {
        const auto& child = table.expert(expert);
        if (child.columns() != columns) throw std::runtime_error("TQ expert scale layouts differ");
        gs = std::min(gs, child.group_size());
        const auto range = child.packed_rows(0, out);
        const auto bytes = child.payload();
        if (load_le<std::uint32_t>(bytes, 28) != range.bytes || range.offset + range.bytes != bytes.size())
            throw std::runtime_error("TQ expert packed offsets differ");
        words += range.bytes / 4;
    }
    const std::size_t groups = columns ? 0 : (std::size_t(width) + gs - 1) / gs;
    const std::size_t factors = columns ? std::size_t(count) * width : std::size_t(rows) * groups;
    if (factors > std::size_t(INT_MAX) || (!word_pool && words > std::size_t(INT_MAX)))
        throw std::overflow_error("TQ expert container exceeds backend dimension limit");
    auto packed = word_pool ? *word_pool : array(allocator::malloc(words * 4), Shape{int(words)}, uint32);
    auto descriptors = array(allocator::malloc(std::size_t(rows + (word_pool ? count : 0)) * 8),
        Shape{rows + (word_pool ? count : 0)}, uint64);
    auto scales = array(allocator::malloc(factors * (columns ? 2 : 4)), Shape{int(factors)}, columns ? float16 : float32);
    std::size_t base = 0;
    const auto* relative = relative_table().data<float16_t>();
    for (int expert = 0; expert < count; ++expert) {
        const auto& child = table.expert(expert);
        const auto bytes = child.payload();
        const auto range = child.packed_rows(0, out);
        if (word_pool) {
            const auto target = expert_offsets[expert];
            if (target > word_pool->size() || range.bytes / 4 > word_pool->size() - target)
                throw std::out_of_range("TQ expert exceeds the word pool");
            descriptors.data<std::uint64_t>()[rows + expert] = target;
        } else std::memcpy(packed.data<std::uint32_t>() + base, bytes.data() + range.offset, range.bytes);
        auto target = std::span(descriptors.data<std::uint64_t>() + std::size_t(expert) * out, out);
        child.write_descriptors(0, word_pool ? 0 : std::uint32_t(base), target);
        const auto scale_start = 64 + std::size_t(out) * 8;
        if (columns) std::memcpy(scales.data<float16_t>() + std::size_t(expert) * width,
            bytes.data() + scale_start, std::size_t(width) * 2);
        else {
            const auto child_groups = (std::size_t(width) + child.group_size() - 1) / child.group_size();
            for (int row = 0; row < out; ++row) {
                const auto descriptor = target[row];
                const float anchor = float(std::bit_cast<float16_t>(std::uint16_t(descriptor >> 40)));
                for (std::size_t group = 0; group < groups; ++group) {
                    const auto source = std::size_t(row) * child_groups + group * gs / child.group_size();
                    const auto code = (bytes[scale_start + source / 2] >> ((source & 1) * 4)) & 15;
                    scales.data<float>()[(std::size_t(expert) * out + row) * groups + group] = anchor * float(relative[code]);
                }
            }
        }
        base += range.bytes / 4;
    }
    if (!word_pool) MlxWeightResidency::track(packed);
    MlxWeightResidency::track(descriptors); MlxWeightResidency::track(scales);
    const auto resident_bytes = 64 + words * 4 + descriptors.nbytes() + scales.nbytes();
    return MlxTqWeight(std::move(packed), std::move(descriptors), std::move(scales), rows,
        width, table.width(), count, columns ? out : gs, columns ? 2 : 0, resident_bytes, word_pool != nullptr);
}

MlxTqWeight MlxTqWeight::from_blob_impl(std::span<const std::uint8_t> blob,
        int input_columns, int experts, const array* word_pool, std::span<const std::uint64_t> expert_offsets) {
    if (blob.size() >= 8 && !std::memcmp(blob.data(), "RTQEXP01", 8))
        return from_expert_blob(blob, input_columns, experts, word_pool, expert_offsets);
    if (blob.size() < 64 || std::memcmp(blob.data(), "RTRELL05", 8) || load_le<std::uint32_t>(blob, 8) != 5)
        throw std::runtime_error("invalid TQ RTRELL05 header");
    auto rows = load_le<std::uint32_t>(blob, 12), width = load_le<std::uint32_t>(blob, 16);
    auto columns = load_le<std::uint32_t>(blob, 20), gs = load_le<std::uint32_t>(blob, 24);
    auto packed = load_le<std::uint32_t>(blob, 28);
    if (!rows || rows > INT_MAX || !width || width > INT_MAX || width % 256 || columns > 1
        || (!columns && (gs < 16 || gs > (1u << 20) || (gs & (gs - 1))))
        || experts <= 0 || rows % experts || input_columns < 0 || input_columns > int(width))
        throw std::runtime_error("invalid TQ geometry");
    if (std::any_of(blob.begin() + 32, blob.begin() + 64, [](auto b) { return b != 0; }))
        throw std::runtime_error("TQ reserved coefficients are nonzero");
    const std::size_t groups = columns ? 0 : (std::size_t(width) + gs - 1) / gs;
    const std::size_t scale_bytes = columns ? std::size_t(width) * 2 : (std::size_t(rows) * groups + 1) / 2;
    const std::size_t descriptors_end = 64 + std::size_t(rows) * 8;
    if (descriptors_end + scale_bytes + packed != blob.size() || packed % 4)
        throw std::runtime_error("TQ payload length differs");
    std::uint64_t offset = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        auto d = load_le<std::uint64_t>(blob, 64 + row * 8);
        auto anchor = std::uint16_t(d >> 40);
        if ((d & ~std::uint64_t(0x00FFFF07FFFFFFFF)) || std::uint32_t(d) != offset
            || (anchor & 0x8000) || (anchor & 0x7c00) == 0x7c00)
            throw std::runtime_error("invalid TQ row descriptor");
        offset += std::uint64_t(width) * (((d >> 32) & 7) + 1) / 32;
    }
    if (offset * 4 != packed) throw std::runtime_error("TQ row offsets differ");
    if (columns) for (std::size_t col = 0; col < width; ++col) {
        auto h = load_le<std::uint16_t>(blob, descriptors_end + col * 2);
        if (!h || (h & 0x8000) || (h & 0x7c00) == 0x7c00)
            throw std::runtime_error("invalid TQ column scale");
    }
    auto scales = !columns && gs >= 256 ? prepare_scales(blob, descriptors_end, rows, groups)
        : copy_array(blob.data() + descriptors_end, columns ? width : scale_bytes, columns ? float16 : uint8);
    const auto resident_bytes = blob.size() - scale_bytes + scales.nbytes();
    if (word_pool) {
        if (rows > std::size_t(INT_MAX) - experts) throw std::overflow_error("TQ pool descriptors are too large");
        const auto expert_rows = rows / experts;
        std::vector<std::uint64_t> descriptors(rows + experts);
        for (int expert = 0; expert < experts; ++expert) {
            const auto first_row = std::size_t(expert) * expert_rows;
            const auto source_base = std::uint32_t(load_le<std::uint64_t>(blob, 64 + first_row * 8));
            const auto source_end = expert + 1 == experts ? packed / 4
                : std::uint32_t(load_le<std::uint64_t>(blob, 64 + (first_row + expert_rows) * 8));
            const auto target = expert_offsets[expert];
            if (target > word_pool->size() || source_end - source_base > word_pool->size() - target)
                throw std::out_of_range("TQ expert exceeds the word pool");
            for (std::size_t row = first_row; row < first_row + expert_rows; ++row) {
                const auto descriptor = load_le<std::uint64_t>(blob, 64 + row * 8);
                descriptors[row] = (descriptor & 0xffffffff00000000ull)
                    | (std::uint32_t(descriptor) - source_base);
            }
            descriptors[rows + expert] = target;
        }
        return MlxTqWeight(*word_pool,
            copy_array(reinterpret_cast<const std::uint8_t*>(descriptors.data()), descriptors.size(), uint64),
            std::move(scales), rows, width, input_columns ? input_columns : width, experts,
            columns ? 512 : gs, columns, resident_bytes + experts * 8, true);
    }
    return MlxTqWeight(copy_array(blob.data() + descriptors_end + scale_bytes, packed / 4, uint32),
        copy_array(blob.data() + 64, rows, uint64), std::move(scales),
        rows, width, input_columns ? input_columns : width, experts, columns ? 512 : gs, columns, resident_bytes);
}

MlxTqWeight MlxTqWeight::load(const MfqContainer& model, const std::string& name) {
    if (model.record(name).dtype != "TQ") throw std::runtime_error("tensor is not TQ: " + name);
    int experts = 1, width = 0;
    if (const auto meta = model.header().extra_json.find("tq_shapes"); meta != model.header().extra_json.end()) {
        auto shapes = nlohmann::json::parse(meta->second);
        if (shapes.contains(name)) {
            auto shape = shapes.at(name).get<std::vector<int>>();
            if (shape.size() != 2 && shape.size() != 3) throw std::runtime_error("invalid TQ logical shape");
            if (std::any_of(shape.begin(), shape.end(), [](int v) { return v <= 0; }))
                throw std::runtime_error("invalid TQ logical dimension");
            width = shape.back();
            if (shape.size() == 3) experts = shape.front();
            const auto mapped = model.map_record(name);
            auto result = from_blob(mapped.view(), width, experts);
            if (result.experts() != experts || result.output_size() != shape[shape.size() - 2])
                throw std::runtime_error("TQ logical rows differ");
            return result;
        }
    }
    const auto mapped = model.map_record(name);
    return from_blob(mapped.view(), width, experts);
}

bool MlxTqWeight::can_concatenate_rows(const MlxTqWeight& other) const {
    return !pooled_ && !other.pooled_ && columns_ != 2 && other.columns_ != 2
        && input_columns_ == other.input_columns_ && width_ == other.width_
        && group_size_ == other.group_size_ && columns_ == other.columns_
        && scales_.dtype() == other.scales_.dtype()
        && (!columns_ || std::memcmp(scales_.data<std::uint8_t>(),
            other.scales_.data<std::uint8_t>(), scales_.nbytes()) == 0);
}

MlxTqWeight MlxTqWeight::concatenate_projections(const std::vector<MlxTqWeight>& weights) {
    return concatenate(weights, true);
}

MlxTqWeight MlxTqWeight::concatenate_experts(const std::vector<MlxTqWeight>& weights) {
    return concatenate(weights, false);
}

MlxTqWeight MlxTqWeight::concatenate(const std::vector<MlxTqWeight>& weights, bool projections) {
    if (weights.empty()) throw std::invalid_argument("TQ concatenation requires weights");
    if (weights.size() == 1) return weights.front();
    const auto& first = weights.front();
    std::size_t words = 0, rows = 0, experts = 0, outputs = 0;
    for (const auto& weight : weights) {
        if (!first.can_concatenate_rows(weight)
            || (!projections && first.output_size() != weight.output_size())
            || (projections && first.experts_ != weight.experts_))
            throw std::invalid_argument("TQ concatenation geometry differs");
        words += weight.words_.size(); rows += weight.rows_; experts += weight.experts_;
        outputs += weight.output_size();
    }
    if (words > std::size_t(INT_MAX) || rows > std::size_t(INT_MAX) || experts > std::size_t(INT_MAX))
        throw std::invalid_argument("TQ concatenation exceeds backend dimension limit");
    const int groups = (first.width_ + first.group_size_ - 1) / first.group_size_;
    const std::size_t scale_stride = first.scales_.dtype() == float32 ? groups * sizeof(float) : groups / 2;
    const std::size_t scale_bytes = first.columns_ ? first.scales_.nbytes() : rows * scale_stride;
    const std::size_t scale_count = scale_bytes / size_of(first.scales_.dtype());
    if (scale_count > std::size_t(INT_MAX))
        throw std::invalid_argument("TQ concatenation scale metadata exceeds backend dimension limit");
    auto packed = array(allocator::malloc(words * 4), Shape{int(words)}, uint32);
    auto descriptors = array(allocator::malloc(rows * 8), Shape{int(rows)}, uint64);
    auto scales = first.columns_ ? first.scales_
        : array(allocator::malloc(scale_bytes), Shape{int(scale_count)}, first.scales_.dtype());
    auto* target_desc = descriptors.data<std::uint64_t>();
    std::size_t word_base = 0, row_base = 0, projection_base = 0;
    for (std::size_t p = 0; p < weights.size(); ++p) {
        const auto& weight = weights[p];
        std::memcpy(packed.data<std::uint32_t>() + word_base,
            weight.words_.data<std::uint32_t>(), weight.words_.nbytes());
        const auto* source = weight.descriptors_.data<std::uint64_t>();
        const int block_rows = projections ? weight.output_size() : weight.rows_;
        const int blocks = projections ? first.experts_ : 1;
        for (int e = 0; e < blocks; ++e) {
            const std::size_t from = std::size_t(e) * block_rows;
            const std::size_t to = projections
                ? std::size_t(e) * outputs + projection_base : row_base;
            for (int r = 0; r < block_rows; ++r)
                target_desc[to + r] = (source[from + r] & 0xffffffff00000000ull)
                    | (std::uint32_t(source[from + r]) + std::uint32_t(word_base));
            if (!first.columns_) std::memcpy(scales.data<std::uint8_t>() + to * scale_stride,
                weight.scales_.data<std::uint8_t>() + from * scale_stride, block_rows * scale_stride);
        }
        word_base += weight.words_.size(); row_base += weight.rows_;
        projection_base += weight.output_size();
    }
    MlxWeightResidency::track(packed); MlxWeightResidency::track(descriptors);
    if (!first.columns_) MlxWeightResidency::track(scales);
    return MlxTqWeight(std::move(packed), std::move(descriptors), std::move(scales), rows,
        first.width_, first.input_columns_, projections ? first.experts_ : int(experts),
        first.group_size_, first.columns_, 64 + words * 4 + rows * 8 + scale_bytes);
}

std::vector<array> MlxTqWeight::matmul_many(const std::vector<MlxTqWeight>& weights, const array& input) {
    return matmul_many_impl(weights, input);
}

bool MlxTqWeight::use_column_inputs(const array& input, int tokens, int routes) const {
    return columns_ && input.dtype() == float16 && tokens >= 1
        && output_size() >= 1024
        && std::uint64_t(input_columns_) * tokens * routes * sizeof(float) <= (std::uint64_t{32} << 20)
        && std::uint64_t(output_size()) * input_columns_ * tokens * routes >= (std::uint64_t{32} << 20);
}

array MlxTqWeight::project_column_inputs(const array& input, const array& ids,
    const MlxTqWeight* up, float limit, bool gelu) const {
    const int tokens = ids.shape(0), routes = ids.shape(1), total = tokens * routes;
    auto canonical = contiguous(astype(ids, int32));
    record_dispatch(input, output_size(), experts_, up ? "column_pair_inputs" : "column_inputs");
    if (up) {
        auto prepared = prepare_column_pair_inputs(*up, input, canonical);
        return column_pair_kernel()({words_, descriptors_, up->words_, up->descriptors_, relative_table(),
                prepared[0], prepared[1], canonical, array({limit})}, {Shape{tokens, routes, output_size()}}, {float16},
            {((output_size() + 3) / 4) * 128, total, 1}, {128, 1, 1},
            {{"K", input_columns_}, {"OUT", output_size()}, {"EXPERTS", experts_}, {"GELU", gelu},
             {"POOLED_G", pooled_}, {"POOLED_U", up->pooled_}}, std::nullopt, false, {}).front();
    }
    return column_projection_kernel()({words_, descriptors_, relative_table(), prepare_column_inputs(input, canonical), canonical},
        {Shape{tokens, routes, output_size()}}, {float16}, {((output_size() + 7) / 8) * 128, total, 1}, {128, 1, 1},
        {{"K", input_columns_}, {"WIDTH", width_}, {"OUT", output_size()}, {"EXPERTS", experts_}, {"POOLED", pooled_}},
        std::nullopt, false, {}).front();
}

array MlxTqWeight::prepare_column_inputs(const array& input, const array& ids) const {
    const int tokens = ids.shape(0), routes = ids.shape(1), total = tokens * routes;
    const auto source = contiguous(reshape(input, {-1, input_columns_}));
    return column_input_kernel()({source, scales_, ids}, {Shape{total, input_columns_}}, {float32},
        {(input_columns_ + 3) / 4, total, 1}, {128, 1, 1},
        {{"K", input_columns_}, {"WIDTH", width_}, {"EXPERTS", experts_}, {"ROUTES", routes},
         {"COLUMN", columns_}, {"SHARED_INPUT", source.shape(0) == tokens}}, std::nullopt, false, {}).front();
}

std::vector<array> MlxTqWeight::prepare_column_pair_inputs(const MlxTqWeight& up,
        const array& input, const array& ids) const {
    const int tokens = ids.shape(0), routes = ids.shape(1), total = tokens * routes;
    const auto source = contiguous(reshape(input, {-1, input_columns_}));
    return column_pair_input_kernel()({source, scales_, up.scales_, ids},
        {Shape{total, input_columns_}, Shape{total, input_columns_}}, {float32, float32},
        {(input_columns_ + 3) / 4, total, 1}, {128, 1, 1},
        {{"K", input_columns_}, {"WIDTH_G", width_}, {"WIDTH_U", up.width_},
         {"COLUMN_G", columns_}, {"COLUMN_U", up.columns_}, {"EXPERTS", experts_},
         {"ROUTES", routes}, {"SHARED_INPUT", source.shape(0) == tokens}}, std::nullopt, false, {});
}

std::vector<array> MlxTqWeight::matmul_many_impl(std::span<const MlxTqWeight> weights, const array& input) {
    if (weights.empty() || weights.size() > 8 || input.ndim() == 0 || input.size() == 0
        || input.shape(-1) != weights.front().input_size())
        throw std::invalid_argument("invalid TQ projection batch");
    const int width = weights.front().input_size();
    const int tokens = input.size() / width;
    if (weights.size() == 1 && weights.front().experts_ == 1 && tokens == 1
        && weights.front().use_column_inputs(input, tokens, 1)) {
        auto shape = input.shape(); shape.back() = weights.front().output_size();
        return {reshape(weights.front().project_column_inputs(input, zeros({tokens, 1}, int32)), shape)};
    }
    std::int64_t output_sum = 0;
    for (const auto& weight : weights) output_sum += weight.output_size();
    if (output_sum > INT_MAX) throw std::invalid_argument("TQ projection batch is too large");
    const bool staging = stage_weights(input, output_sum);
    const auto dtype = input.dtype();
    const bool fused_split = dtype == float16 && tokens <= 6 && width >= 2048
        && output_sum >= 256 && output_sum * 4 < width && (tokens > 1 || output_sum <= 1024);
    const int split_rows = staging ? 1 : fused_split ? tokens : split_batch_rows(tokens, output_sum, width);
    int splits = staging ? 1 : split_count(tokens, output_sum, width, split_rows, fused_split);
    const bool nax = use_nax(input) && splits == 1;
    const bool shared = nax && !staging && tokens >= 256 && output_sum >= 128;
    const int small_rows = staging || nax ? 1 : splits > 1 ? split_rows : small_batch_rows(tokens, output_sum, width);
    const bool fixed_sources = fused_split || (dtype == float16 && tokens <= 6 && small_rows > 1);
    const int lanes = fused_split ? 32 : small_rows > 1 && splits == 1 && tokens <= small_rows
        && width >= 2048 && output_sum * 2 <= width ? 32 : 16;
    if (dtype != float16 && dtype != bfloat16 && dtype != float32)
        throw std::invalid_argument("TQ requires floating activations");
    std::vector<array> inputs;
    inputs.reserve(2 + weights.size() * 3);
    inputs.push_back(contiguous(reshape(input, {-1, width})));
    inputs.push_back(relative_table());
    const int reuse_rows = fixed_sources ? tokens : weights.size() == 1 && dtype == float16
        && ((width >= 4096 && output_sum * 2 <= width) || (width >= 256 && width < 512 && output_sum >= 1024))
        ? std::min(small_rows, tokens) : small_rows;
    const int block_rows = shared ? 128 : nax ? (tokens <= 16 ? 16 : tokens <= 32 ? 32 : 64) : reuse_rows;
    if (nax && !shared && !staging) splits = nax_split_count(tokens, output_sum, width, block_rows);
    std::vector<std::pair<std::string, fast::TemplateArg>> templates;
    templates.reserve(8 + weights.size() * 7);
    templates.emplace_back("T", dtype);
    templates.emplace_back("LANES", lanes);
    templates.emplace_back("K", width);
    int total = 0;
    for (std::size_t p = 0; p < weights.size(); ++p) {
        const auto& weight = weights[p];
        if (weight.experts_ != 1 || weight.input_size() != width || total > INT_MAX - weight.output_size())
            throw std::invalid_argument("TQ projection batch geometry differs");
        const auto suffix = std::to_string(p);
        inputs.insert(inputs.end(), {weight.words_, weight.descriptors_, weight.scales_});
        templates.emplace_back("BEGIN" + suffix, total);
        total += weight.output_size();
        templates.emplace_back("END" + suffix, total);
        templates.emplace_back("COLUMN" + suffix, weight.columns_);
        templates.emplace_back("WIDTH" + suffix, weight.width_);
        templates.emplace_back("GS" + suffix, weight.group_size_);
        templates.emplace_back("POOLED" + suffix, weight.pooled_);
    }
    templates.emplace_back("TOTAL", total);
    record_dispatch(input, total, 1, staging ? "dense_staged_gemm" : shared ? "dense_shared_nax" : nax ? splits > 1 ? "dense_nax_split" : "dense_nax"
        : fused_split ? "dense_split_fused" : splits > 1 ? small_rows > 1 ? "dense_split_reuse" : "dense_split_k"
        : small_rows > 1 ? fixed_sources ? "dense_reuse_fixed" : "dense_reuse" : "dense_matvec");
    if (fused_split || (nax && !staging) || small_rows > 1) templates.emplace_back("BM", block_rows);
    if (fused_split) templates.emplace_back("SPLITS", splits);
    if (splits > 1) templates.emplace_back("CHUNK", int(
        (std::int64_t(width) + splits * 256 - 1) / (splits * 256) * 256));
    auto shape = input.shape(); shape.back() = total;
    const int threads = fused_split ? splits * 32 : shared ? 256 : 128;
    const int rows_per_group = fused_split ? 1 : shared ? 64 : nax ? 128 : 128 / lanes;
    const auto mode = staging ? TqBatchMode::StageWeights : shared ? TqBatchMode::SharedNax : nax ? splits > 1 ? TqBatchMode::NaxSplit : TqBatchMode::Nax
        : fused_split ? TqBatchMode::FusedSplitReuse : splits > 1 ? small_rows > 1 ? TqBatchMode::SplitReuse : TqBatchMode::SplitK
        : small_rows > 1 ? TqBatchMode::Reuse : TqBatchMode::Vector;
    const auto& first = weights.front();
    const bool select_decode = tokens == 1 && mode == TqBatchMode::Vector && weights.size() > 1
        && std::all_of(weights.begin(), weights.end(), [&](const auto& weight) {
            return weight.columns_ == first.columns_ && weight.width_ == first.width_
                && weight.group_size_ == first.group_size_ && weight.scales_.dtype() == first.scales_.dtype()
                && weight.words_.size() >= 8 && weight.descriptors_.size() >= 8 && weight.scales_.size() >= 8;
        });
    auto values = batch_kernel(weights.size(), mode, select_decode, fixed_sources)(inputs,
        {staging ? Shape{total, width} : splits > 1 && !fused_split ? Shape{splits, tokens, total} : Shape{tokens, total}},
        {staging ? (dtype == float16 ? float16 : float32) : splits > 1 && !fused_split ? float32 : dtype},
        {staging ? (width + 3) / 4 : ((total + rows_per_group - 1) / rows_per_group) * threads,
         staging ? total : fused_split ? 1 : (nax || small_rows > 1) ? (tokens + block_rows - 1) / block_rows : tokens,
         fused_split ? 1 : splits},
        {threads, 1, 1},
        std::move(templates), std::nullopt, false, {}).front();
    if (staging) values = astype(mlx::core::matmul(astype(inputs[0], values.dtype()), transpose(values)), dtype);
    if (splits > 1 && !fused_split) values = split_reduce_kernel()({values}, {Shape{tokens, total}}, {dtype},
        {tokens * total, 1, 1}, {128, 1, 1}, {{"T", dtype}, {"TOTAL", total}, {"SPLITS", splits}},
        std::nullopt, false, {}).front();
    std::vector<array> results;
    results.reserve(weights.size());
    int start = 0;
    for (const auto& weight : weights) {
        shape.back() = weight.output_size();
        results.push_back(reshape(slice(values, {0, start}, {tokens, start + weight.output_size()}), shape));
        start += weight.output_size();
    }
    return results;
}

MlxTqWeight MlxTqWeight::expert_view(int expert) const {
    if (expert < 0 || expert >= experts_) throw std::invalid_argument("TQ expert index is out of range");
    auto result = *this;
    result.rows_ = output_size(); result.experts_ = 1;
    const int begin = expert * result.rows_;
    result.descriptors_ = slice(descriptors_, {begin}, {begin + result.rows_});
    if (pooled_) result.descriptors_ = mlx::core::concatenate({result.descriptors_,
        slice(descriptors_, {rows_ + expert}, {rows_ + expert + 1})});
    if (!columns_) {
        const int groups = (width_ + group_size_ - 1) / group_size_;
        const int stride = scales_.dtype() == float32 ? groups : groups / 2;
        result.scales_ = slice(scales_, {begin * stride}, {(begin + result.rows_) * stride});
    } else if (columns_ == 2) {
        result.scales_ = slice(scales_, {expert * width_}, {(expert + 1) * width_});
        result.columns_ = 1;
        result.group_size_ = 512;
    }
    return result;
}

array MlxTqWeight::project(const array& input, const array& ids,
    const MlxTqWeight* up, float limit, bool gelu, int split, const array* reuse) const {
    if (input.ndim() == 0 || input.shape(-1) != input_columns_ || ids.ndim() != 2
        || ids.shape(0) <= 0 || ids.shape(1) <= 0 || !std::isfinite(limit) || limit < 0)
        throw std::invalid_argument("TQ projection geometry differs");
    if (up && (up->experts_ != experts_ || up->rows_ != rows_ || up->input_columns_ != input_columns_))
        throw std::invalid_argument("TQ Gate/Up geometry differs");
    const int tokens = ids.shape(0), routes = ids.shape(1), total = tokens * routes;
    const auto source_rows = input.size() / input_columns_;
    if (source_rows != std::size_t(tokens) && source_rows != std::size_t(total))
        throw std::invalid_argument("TQ activation rows differ from route IDs");
    const auto& u = up ? *up : *this;
    const int out = split ? split : output_size();
    const int lanes = (!up && !split) || input_columns_ < 1024 ? 16 : 32;
    const int rows_per_group = 128 / lanes;
    auto dtype = input.dtype();
    if (dtype != float16 && dtype != bfloat16 && dtype != float32)
        throw std::invalid_argument("TQ requires floating activations");
    const bool nax = use_nax(input);
    const bool grouped = experts_ <= 1024 && ids.size() >= std::size_t(experts_) * 3
        && (nax || use_routed_simd(input, ids.size(), experts_));
    if (!split && !reuse && !grouped && !(experts_ == 1 && routes == 1 && tokens > 1)
        && use_column_inputs(input, tokens, routes) && (!up || up->use_column_inputs(input, tokens, routes)))
        return project_column_inputs(input, ids, up, limit, gelu);
    if (reuse && !split && (!up || (input_columns_ >= 1024
        && columns_ == u.columns_ && width_ == u.width_ && group_size_ == u.group_size_
        && scales_.dtype() == u.scales_.dtype()))) {
        if (reuse->ndim() != 2 || (reuse->shape(1) != 3 && reuse->shape(1) != 5 && reuse->shape(1) != 7 && reuse->shape(1) != 18) || reuse->shape(0) <= 0
            || reuse->dtype() != int32 || !reuse->flags().row_contiguous || tokens > 16)
            throw std::invalid_argument("invalid TQ repeated route plan");
        const bool direct = reuse->shape(1) == 18;
        const int reuse_lanes = up ? 32 : 16;
        const auto nax_groups = ((std::int64_t(out) + 127) / 128) * routes;
        const int nax_rows = std::int64_t(out) >= std::int64_t(input_columns_) * 2 || nax_groups >= 64 ? 8 : 16;
        const bool reuse_nax = dtype == float16 && tokens >= nax_rows && input_columns_ >= 512 && out >= 512 && has_nax();
        record_dispatch(input, out, experts_, reuse_nax ? "routed_reuse_nax" : up ? "routed_pair_reuse" : "routed_reuse");
        std::vector<array> arguments{words_, descriptors_, scales_, relative_table(), u.words_, u.descriptors_, u.scales_,
            contiguous(reshape(input, {-1, input_columns_})), *reuse, array({limit})};
        std::vector<std::pair<std::string, fast::TemplateArg>> templates{
             {"T", dtype}, {"OUT", out}, {"K", input_columns_}, {"EXPERTS", experts_}, {"ROUTES", routes},
             {"WIDTH_G", width_}, {"GS_G", group_size_}, {"COLUMN_G", columns_},
             {"POOLED_G", pooled_}, {"POOLED_U", u.pooled_},
             {"REUSE_ROWS", direct ? tokens <= 2 ? 2 : tokens <= 4 ? 4 : tokens <= 6 ? 6 : 4 : reuse->shape(1) - 1},
             {"DIRECT_PLAN", direct},
             {"MIN_NAX_ROWS", nax_rows},
             {"SHARED_INPUT", source_rows == std::size_t(tokens)}, {"PAIR", up != nullptr}, {"GELU", gelu}};
        if (direct) {
            const bool single_columns = use_column_inputs(input, tokens, routes)
                && (!up || up->use_column_inputs(input, tokens, routes));
            const auto canonical = contiguous(astype(ids, int32));
            auto prepared = single_columns ? prepare_column_inputs(input, canonical) : array(0.f);
            arguments.push_back(prepared);
            arguments.push_back(single_columns && up ? up->prepare_column_inputs(input, canonical) : prepared);
            templates.emplace_back("COLUMN_SINGLE", single_columns);
        }
        return reuse_kernel(reuse_nax, direct)(arguments, {Shape{tokens, routes, out}}, {dtype},
            {((out + 128 / reuse_lanes - 1) / (128 / reuse_lanes)) * 128, reuse->shape(0), 1}, {128, 1, 1},
            std::move(templates), std::nullopt, false, {}).front();
    }
    if (experts_ == 1 && routes == 1 && (up || split) && tokens >= 2 && tokens <= 6
        && input_columns_ >= 2048 && std::int64_t(out) * 2 <= input_columns_) {
        record_dispatch(input, out, 1, "dense_pair_reuse");
        return dense_pair_kernel()(
            {words_, descriptors_, scales_, relative_table(), u.words_, u.descriptors_, u.scales_,
             contiguous(reshape(input, {tokens, input_columns_})), contiguous(astype(ids, int32)), array({limit})},
            {Shape{tokens, 1, out}}, {dtype}, {((out + 1) / 2) * 128, 1, 1}, {128, 1, 1},
            {{"T", dtype}, {"OUT", out}, {"K", input_columns_}, {"BM", tokens},
             {"STRIDE_G", output_size()}, {"STRIDE_U", u.output_size()}, {"BASE_U", split},
             {"WIDTH_G", width_}, {"WIDTH_U", u.width_}, {"GS_G", group_size_}, {"GS_U", u.group_size_},
             {"COLUMN_G", columns_}, {"COLUMN_U", u.columns_}, {"POOLED_G", pooled_}, {"POOLED_U", u.pooled_},
             {"GELU", gelu}}, std::nullopt, false, {}).front();
    }
    const bool paired_routes = (up || split) && (experts_ > 1 || (tokens <= 6 && out < input_columns_))
        && tokens <= 16 && tokens % 2 == 0
        && lanes == 32 && columns_ == u.columns_ && width_ == u.width_ && group_size_ == u.group_size_
        && scales_.dtype() == u.scales_.dtype();
    if (experts_ == 1 && routes == 1 && !split && up && !paired_routes
        && (use_nax(input) || stage_weights(input, std::int64_t(output_size()) * 2)
            || (output_size() >= input_columns_
                && small_batch_rows(tokens, output_size(), input_columns_) > 1))) {
        auto projected = matmul_many({*this, *up}, input);
        return pair_activation_kernel()({projected[0], projected[1], array({limit}), contiguous(astype(ids, int32))},
            {Shape{tokens, 1, out}}, {dtype}, {tokens * out, 1, 1}, {128, 1, 1},
            {{"T", dtype}, {"OUT", out}, {"SPLIT", false}, {"GELU", gelu}, {"MASK_SINGLE_EXPERT", true}},
            std::nullopt, false, {}).front();
    }
    if (grouped) {
        auto canonical = astype(ids, int32);
        auto order = astype(argsort(reshape(canonical, {-1})), int32);
        const auto mean_rows = ids.size() / experts_;
        const bool expanding = std::int64_t(output_size()) >= std::int64_t(input_columns_) * 2;
        const bool shared = experts_ > 1 && tokens >= 256 && mean_rows >= 16
            && input_columns_ >= 512 && (input_columns_ >= 1024 || expanding);
        const int block_rows = !nax ? 32 : shared
            ? (!expanding && input_columns_ >= 1024 && mean_rows >= 64 ? 128 : 64)
            : mean_rows <= 8 ? 16 : 32;
        auto plan = make_tq_route_plan(canonical, order, experts_, block_rows);
        auto gate = routed_matmul_sorted(input, canonical, order, plan);
        if (!up && !split) return gate;
        auto other = up ? up->routed_matmul_sorted(input, canonical, order, plan) : gate;
        return pair_activation_kernel()({gate, other, array({limit}), canonical}, {Shape{tokens, routes, out}}, {dtype},
            {total * out, 1, 1}, {128, 1, 1},
            {{"T", dtype}, {"OUT", out}, {"SPLIT", split != 0}, {"GELU", gelu}, {"MASK_SINGLE_EXPERT", false}},
            std::nullopt, false, {}).front();
    }
    const int outputs_per_group = paired_routes ? 2 : rows_per_group;
    record_dispatch(input, out, experts_, paired_routes ? "routed_pair_tokens" : up || split ? "routed_pair" : "routed_matvec");
    return (paired_routes ? paired_route_kernel() : projection_kernel())(
        {words_, descriptors_, scales_, relative_table(), u.words_, u.descriptors_, u.scales_,
         contiguous(reshape(input, {-1, input_columns_})), contiguous(astype(ids, int32)), array({limit})},
        {Shape{tokens, routes, out}}, {dtype},
        {((out + outputs_per_group - 1) / outputs_per_group) * 128, paired_routes ? total / 2 : total, 1}, {128, 1, 1},
        {{"T", dtype}, {"LANES", lanes}, {"OUT", out}, {"K", input_columns_},
         {"STRIDE_G", output_size()}, {"STRIDE_U", u.output_size()}, {"BASE_U", split},
         {"WIDTH_G", width_}, {"WIDTH_U", u.width_}, {"GS_G", group_size_}, {"GS_U", u.group_size_},
         {"COLUMN_G", columns_}, {"COLUMN_U", u.columns_}, {"EXPERTS", experts_}, {"ROUTES", routes},
         {"POOLED_G", pooled_}, {"POOLED_U", u.pooled_},
         {"SHARED_INPUT", source_rows == std::size_t(tokens)}, {"PAIR", up != nullptr || split != 0}, {"GELU", gelu}},
        std::nullopt, false, {}).front();
}

array MlxTqWeight::matmul(const array& input) const {
    if (experts_ != 1) throw std::invalid_argument("TQ dense projection has routed experts");
    if (input.ndim() == 0 || input.shape(-1) != input_columns_)
        throw std::invalid_argument("TQ activation width differs");
    return matmul_many_impl(std::span<const MlxTqWeight>(this, 1), input).front();
}

array MlxTqWeight::rms_matmul(const array& input, float eps) const {
    if (experts_ != 1 || input.ndim() == 0 || input.shape(-1) != input_columns_ || input.size() == 0
        || !std::isfinite(eps) || eps <= 0.f
        || (input.dtype() != float16 && input.dtype() != bfloat16 && input.dtype() != float32))
        throw std::invalid_argument("invalid TQ normalized projection");
    const auto tokens = input.size() / input_columns_;
    if (tokens >= 32 && input.dtype() == float16 && rows_ >= 16 && rows_ <= 64
        && input_columns_ >= 8192 && has_nax()) {
        const auto blocks = ((rows_ + 31) / 32) * ((tokens + 15) / 16);
        int splits = 1;
        while (splits < 32 && blocks * splits < 160) splits *= 2;
        const int chunk = (input_columns_ + splits * 256 - 1) / (splits * 256) * 256;
        const auto source = contiguous(reshape(input, {int(tokens), input_columns_}));
        record_dispatch(input, rows_, 1, "rms_projection_nax");
        const auto parts = rms_nax_projection_kernel()(
            {source, words_, descriptors_, scales_, relative_table()}, {Shape{splits, int(tokens), rows_}}, {float32},
            {((rows_ + 31) / 32) * 32, int((tokens + 15) / 16), splits}, {32, 1, 1},
            {{"T", input.dtype()}, {"OUT", rows_}, {"K", input_columns_}, {"WIDTH", width_}, {"GS", group_size_},
             {"COLUMNS", columns_}, {"POOLED", pooled_}, {"CHUNK", chunk}}, std::nullopt, false, {}).front();
        auto shape = input.shape(); shape.back() = rows_;
        return reshape(rms_nax_reduce_kernel()({source, parts, array({eps})}, {Shape{int(tokens), rows_}}, {float32},
            {int(tokens) * 32, 1, 1}, {128, 1, 1},
            {{"T", input.dtype()}, {"K", input_columns_}, {"OUT", rows_}, {"SPLITS", splits}},
            std::nullopt, false, {}).front(), shape);
    }
    if (tokens > 6 || rows_ < 16 || rows_ > 64 || input_columns_ < 8192) {
        const auto source = astype(input, float32);
        return matmul(source * rsqrt(mean(source * source, -1, true) + array(eps)));
    }
    record_dispatch(input, rows_, 1, "rms_projection");
    auto shape = input.shape(); shape.back() = rows_;
    return rms_projection_kernel()(
        {words_, descriptors_, scales_, relative_table(), contiguous(reshape(input, {int(tokens), input_columns_})), array({eps})},
        {shape}, {float32}, {rows_ * 512, int(tokens), 1}, {512, 1, 1},
        {{"T", input.dtype()}, {"OUT", rows_}, {"K", input_columns_}, {"WIDTH", width_}, {"GS", group_size_},
         {"COLUMNS", columns_}, {"POOLED", pooled_}}, std::nullopt, false, {}).front();
}

bool MlxTqWeight::can_fuse_gated_mix(const array& input) const {
    if (input.ndim() == 0 || input.size() == 0 || input.shape(-1) != input_columns_) return false;
    const auto tokens = input.size() / input_columns_;
    return tokens <= 6 || (tokens <= 64 && input.dtype() == float16 && input_columns_ >= 256
        && input_columns_ < 512 && output_size() >= 1024 && has_nax());
}

array MlxTqWeight::gated_mix(const array& input, const array& streams, int groups) const {
    if (experts_ != 1 || input.ndim() == 0 || streams.ndim() == 0 || input.size() == 0
        || input.shape(-1) != input_columns_ || groups <= 0 || groups > 16 || output_size() % groups
        || streams.shape(-1) != output_size() || input.size() / input_columns_ != streams.size() / output_size()
        || input.dtype() != streams.dtype()
        || (input.dtype() != float16 && input.dtype() != bfloat16 && input.dtype() != float32))
        throw std::invalid_argument("TQ gated projection geometry differs");
    const int tokens = input.size() / input_columns_, hidden = output_size() / groups;
    const bool nax = tokens > 6 && can_fuse_gated_mix(input);
    const int batch = nax ? 16 : tokens <= 6 ? tokens : 1;
    auto shape = input.shape(); shape.back() = hidden;
    record_dispatch(input, output_size(), 1, nax ? "gated_mix_nax" : "gated_mix");
    const int threads = nax ? groups * 32 : 128;
    return reshape((nax ? gated_mix_nax_kernel() : gated_mix_kernel())({words_, descriptors_, scales_, relative_table(),
        contiguous(reshape(input, {tokens, input_columns_})), contiguous(reshape(streams, {tokens, output_size()}))},
        {Shape{tokens, hidden}}, {input.dtype()},
        {nax ? ((hidden + 31) / 32) * threads : ((hidden + 3) / 4) * 128, (tokens + batch - 1) / batch, 1}, {threads, 1, 1},
        {{"T", input.dtype()}, {"K", input_columns_}, {"WIDTH", width_}, {"GS", group_size_},
         {"COLUMNS", columns_}, {"HIDDEN", hidden}, {"GROUPS", groups}, {"POOLED", pooled_}, {"BM", batch},
         {"COMPACT", input.dtype() == float16 && input_columns_ >= 256 && input_columns_ < 512 && batch >= 3}},
        std::nullopt, false, {}).front(), shape);
}

std::optional<array> make_tq_reuse_plan(std::span<const int> ids, int tokens, int routes, int experts) {
    if (tokens <= 0 || routes <= 0 || experts <= 0 || ids.size() != std::size_t(tokens) * routes)
        throw std::invalid_argument("invalid TQ route reuse geometry");
    if (tokens < 2 || tokens > 16 || ids.size() > 512) return std::nullopt;
    std::vector<std::pair<int, int>> ordered;
    ordered.reserve(ids.size());
    for (std::size_t row = 0; row < ids.size(); ++row)
        ordered.emplace_back(ids[row] >= 0 && ids[row] < experts ? ids[row] : -1, int(row));
    std::sort(ordered.begin(), ordered.end());
    int blocks_four = 0, blocks_six = 0;
    for (std::size_t begin = 0; begin < ordered.size();) {
        auto end = begin + 1;
        while (end < ordered.size() && ordered[end].first == ordered[begin].first) ++end;
        blocks_four += (end - begin + 3) / 4;
        blocks_six += (end - begin + 5) / 6;
        begin = end;
    }
    if (std::size_t(blocks_four) * 4 > ids.size() * 3) return std::nullopt;
    if (blocks_six < blocks_four) {
        std::vector<int> blocks;
        blocks.reserve(blocks_six * 7);
        for (std::size_t begin = 0; begin < ordered.size();) {
            auto end = begin + 1;
            while (end < ordered.size() && ordered[end].first == ordered[begin].first) ++end;
            const auto count = end - begin, groups = (count + 5) / 6;
            for (std::size_t group = 0; group < groups; ++group) {
                const auto offset = blocks.size(), rows = count / groups + (group < count % groups ? 1 : 0);
                blocks.resize(offset + 7, -1);
                blocks[offset] = ordered[begin].first;
                for (std::size_t row = 0; row < rows; ++row, ++begin) blocks[offset + row + 1] = ordered[begin].second;
            }
        }
        return array(blocks.begin(), {blocks_six, 7}, int32);
    }
    std::vector<int> blocks;
    blocks.reserve(ids.size() * 5);
    for (std::size_t begin = 0; begin < ordered.size();) {
        const auto offset = blocks.size();
        blocks.insert(blocks.end(), {ordered[begin].first, -1, -1, -1, -1});
        for (int row = 0; row < 4 && begin < ordered.size()
            && ordered[begin].first == blocks[offset]; ++row, ++begin)
            blocks[offset + row + 1] = ordered[begin].second;
    }
    const int count = blocks.size() / 5;
    if (std::size_t(count) * 4 > ids.size() * 3) return std::nullopt;
    bool pairs_only = true;
    for (int group = 0; group < count; ++group) if (blocks[group * 5 + 3] >= 0) pairs_only = false;
    if (pairs_only) {
        std::vector<int> compact;
        compact.reserve(count * 3);
        for (int group = 0; group < count; ++group)
            compact.insert(compact.end(), blocks.begin() + group * 5, blocks.begin() + group * 5 + 3);
        return array(compact.begin(), {count, 3}, int32);
    }
    return array(blocks.begin(), {count, 5}, int32);
}

std::optional<array> make_tq_reuse_plan(const array& ids, int experts) {
    if (ids.ndim() != 2 || ids.shape(0) < 1 || ids.shape(1) < 1 || experts < 1
        || (ids.dtype() != int32 && ids.dtype() != uint32))
        throw std::invalid_argument("invalid TQ route reuse geometry");
    if (ids.shape(0) < 3 || ids.shape(0) > 16 || ids.size() > 512) return std::nullopt;
    static const auto kernel = fast::metal_kernel("mfq_tq_device_reuse_plan", {"ids"}, {"plan"}, R"METAL(
uint index = thread_position_in_grid.x, total = uint(ids_shape[0]);
if (index >= total) return;
for (uint j = 0u; j < 18u; ++j) plan[index * 18u + j] = -1;
plan[index * 18u + 1u] = 0;
int expert = ids[index];
uint rank = 0u;
for (uint j = 0u; j < index; ++j) rank += ids[j] == expert;
if (rank % 16u) return;
uint count = 0u;
for (uint j = index; j < total && count < 16u; ++j)
    if (ids[j] == expert) plan[index * 18u + 2u + count++] = int(j);
plan[index * 18u] = expert;
plan[index * 18u + 1u] = int(count);
)METAL", "", true, false);
    const int total = int(ids.size());
    return kernel({contiguous(reshape(astype(ids, int32), {-1}))}, {Shape{total, 18}}, {int32},
        {total, 1, 1}, {32, 1, 1}, {}, std::nullopt, false, {}).front();
}

array MlxTqWeight::routed_matmul(const array& input, const array& ids, const array* reuse) const {
    return project(input, ids, nullptr, 0.f, false, 0, reuse);
}

array MlxTqWeight::routed_matmul_sorted(const array& input, const array& ids,
    const array& order, const MlxTqRoutePlan& plan) const {
    if (input.ndim() == 0 || input.shape(-1) != input_columns_ || ids.ndim() != 2
        || ids.size() == 0 || order.ndim() != 1 || order.size() != ids.size()
        || plan.route_count != int(ids.size()) || plan.experts != experts_
        || (plan.block_rows != 16 && plan.block_rows != 32 && plan.block_rows != 64 && plan.block_rows != 128))
        throw std::invalid_argument("TQ grouped projection geometry differs");
    const int tokens = ids.shape(0), routes = ids.shape(1);
    const auto source_rows = input.size() / input_columns_;
    if ((source_rows != std::size_t(tokens) && source_rows != ids.size()) || input.dtype() != float16)
        throw std::invalid_argument("TQ grouped projection activations differ");
    const bool nax = use_nax(input);
    if (!nax && (!use_routed_simd(input, ids.size(), experts_) || plan.block_rows != 32))
        return project(input, ids, nullptr, 0.f, false);
    const bool shared = nax && plan.block_rows >= 64;
    const int threads = !nax ? plan.block_rows * 4 : shared ? plan.block_rows * 2 : 128;
    const int columns = !nax ? 32 : shared ? 64 : 128;
    record_dispatch(input, output_size(), experts_, !nax ? "routed_simd" : shared ? "routed_shared_nax" : "routed_nax");
    return routed_nax_kernel()({words_, descriptors_, scales_, relative_table(),
        contiguous(reshape(input, {-1, input_columns_})), contiguous(astype(order, int32)),
        plan.block_meta, plan.block_count},
        {Shape{tokens, routes, output_size()}}, {input.dtype()},
        {((output_size() + columns - 1) / columns) * threads, plan.max_blocks, 1}, {threads, 1, 1},
        {{"T", input.dtype()}, {"K", input_columns_}, {"OUT", output_size()}, {"WIDTH", width_},
         {"GS", group_size_}, {"COLUMNS", columns_}, {"EXPERTS", experts_}, {"BM", plan.block_rows},
         {"POOLED", pooled_},
         {"ROUTES", routes}, {"SHARED_INPUT", source_rows == std::size_t(tokens)}, {"SHARED_TILE", shared}, {"NAX", nax}},
        std::nullopt, false, {}).front();
}

array MlxTqWeight::routed_matmul_reduce(const array& input, const array& ids, const array& weights,
    const array* reuse) const {
    if (input.ndim() == 0 || input.shape(-1) != input_columns_ || ids.ndim() != 2
        || ids.shape(0) <= 0 || ids.shape(1) <= 0 || weights.shape() != ids.shape())
        throw std::invalid_argument("TQ routed reduction geometry differs");
    const int tokens = ids.shape(0), routes = ids.shape(1), lanes = 16;
    const auto source_rows = input.size() / input_columns_;
    auto dtype = input.dtype();
    if ((source_rows != std::size_t(tokens) && source_rows != std::size_t(tokens) * routes)
        || (dtype != float16 && dtype != bfloat16 && dtype != float32))
        throw std::invalid_argument("TQ routed reduction activations differ");
    if (reuse && tokens == 2 && input_columns_ < 1024 && reuse->shape(0) > routes) reuse = nullptr;
    const bool grouped = experts_ <= 1024 && ids.size() >= std::size_t(experts_) * 3
        && (use_nax(input) || use_routed_simd(input, ids.size(), experts_));
    if (reuse || grouped || input_columns_ >= 1024 || routes < 2 || routes > 32 || output_size() > INT_MAX / (routes * lanes)) {
        auto pairs = routed_matmul(input, ids, reuse);
        if (dtype == bfloat16) pairs = astype(pairs, float32);
        return astype(moe_weighted_reduce(pairs, weights), dtype);
    }
    record_dispatch(input, output_size(), experts_, "routed_reduce");
    return routed_reduce_kernel()(
        {words_, descriptors_, scales_, relative_table(), contiguous(reshape(input, {-1, input_columns_})),
         contiguous(astype(ids, int32)), contiguous(astype(weights, float32))},
        {Shape{tokens, output_size()}}, {dtype}, {output_size() * routes * lanes, tokens, 1},
        {routes * lanes, 1, 1},
        {{"T", dtype}, {"LANES", lanes}, {"OUT", output_size()}, {"K", input_columns_},
         {"WIDTH", width_}, {"GS", group_size_}, {"COLUMNS", columns_}, {"EXPERTS", experts_},
         {"POOLED", pooled_},
         {"ROUTES", routes}, {"SHARED_INPUT", source_rows == std::size_t(tokens)}},
        std::nullopt, false, {}).front();
}

array MlxTqWeight::grouped_row_matmul(const array& input, int groups) const {
    if (experts_ != 1 || groups <= 0 || rows_ % groups || input.ndim() < 2
        || input.shape(-2) != groups || input.shape(-1) != input_columns_ || input.size() == 0
        || (input.dtype() != float16 && input.dtype() != bfloat16 && input.dtype() != float32))
        throw std::invalid_argument("TQ grouped projection geometry differs");
    const auto token_count = input.size() / (std::size_t(groups) * input_columns_);
    if (token_count > INT_MAX) throw std::invalid_argument("TQ grouped projection batch is too large");
    const int tokens = int(token_count), out = rows_ / groups;
    auto shape = input.shape(); shape.back() = out;
    if (groups == 1) return reshape(matmul(reshape(input, {tokens, input_columns_})), shape);
    const auto dtype = input.dtype();
    const int scalar_splits = split_count(tokens, rows_, input_columns_);
    const bool hardware_nax = use_nax(input, tokens);
    if (stage_weights(tokens, input_columns_, rows_, dtype)
        || (dtype == float16 && tokens >= 32 && !hardware_nax && scalar_splits == 1
            && staged_weights_fit(input_columns_, rows_, dtype))) {
        record_dispatch(input, out, groups, "grouped_staged_gemm", tokens);
        auto values = batch_kernel(1, TqBatchMode::StageWeights)(
            {contiguous(reshape(input, {-1, input_columns_})), relative_table(), words_, descriptors_, scales_},
            {Shape{rows_, input_columns_}}, {dtype == float16 ? float16 : float32},
            {(input_columns_ + 3) / 4, rows_, 1}, {128, 1, 1},
            {{"T", dtype}, {"LANES", 16}, {"K", input_columns_}, {"BEGIN0", 0}, {"END0", rows_},
             {"COLUMN0", columns_}, {"WIDTH0", width_}, {"GS0", group_size_}, {"POOLED0", pooled_}, {"TOTAL", rows_}},
            std::nullopt, false, {}).front();
        auto source = astype(transpose(reshape(input, {tokens, groups, input_columns_}), {1, 0, 2}), values.dtype());
        values = transpose(reshape(values, {groups, out, input_columns_}), {0, 2, 1});
        return reshape(contiguous(astype(transpose(mlx::core::matmul(source, values), {1, 0, 2}), dtype)), shape);
    }
    const auto padded = std::int64_t(groups) * ((std::int64_t(out) + 127) / 128) * 128;
    const bool nax = hardware_nax && scalar_splits == 1 && padded <= INT_MAX;
    if (!(nax && tokens >= 128) && !(!hardware_nax && scalar_splits == 1
        && use_routed_simd(input, std::size_t(tokens) * groups, groups))) {
        const bool fixed = dtype == float16 && tokens >= 2 && tokens <= 6 && scalar_splits == 1
            && small_batch_rows(tokens, out, input_columns_) > 1;
        const int lanes = fixed && input_columns_ >= 2048 && out <= input_columns_ ? 32 : 16;
        int splits = scalar_splits;
        const int bm = nax ? (tokens <= 16 ? 16 : tokens <= 32 ? 32 : 64)
            : fixed ? tokens
            : splits > 1 ? split_batch_rows(tokens, rows_, input_columns_)
            : small_batch_rows(tokens, out, input_columns_);
        if (nax) {
            splits = nax_split_count(tokens, int(padded), input_columns_, bm);
        } else splits = split_count(tokens, rows_, input_columns_, bm);
        record_dispatch(input, out, groups, nax ? splits > 1 ? "grouped_nax_split" : "grouped_nax"
            : bm > 1 ? "grouped_reuse" : "grouped_matvec", tokens);
        const auto threads = nax ? padded : ((std::int64_t(rows_) + 128 / lanes - 1) / (128 / lanes)) * 128;
        if (threads > INT_MAX) throw std::invalid_argument("TQ grouped projection grid is too large");
        const array sizes({tokens}, int32);
        auto values = grouped_kernel(nax)(
            {words_, descriptors_, scales_, relative_table(), contiguous(reshape(input, {-1, input_columns_})), sizes},
            {splits > 1 ? Shape{splits, tokens, rows_} : Shape{tokens, rows_}}, {splits > 1 ? float32 : dtype},
            {int(threads), (tokens + bm - 1) / bm, splits}, {128, 1, 1},
            {{"T", dtype}, {"O", splits > 1 ? float32 : dtype}, {"TOTAL", rows_}, {"OUT", out}, {"GROUPS", groups}, {"K", input_columns_},
             {"WIDTH", width_}, {"GS", group_size_}, {"COLUMNS", columns_}, {"POOLED", pooled_},
             {"LANES", lanes}, {"FIXED", fixed},
             {"COMPACT", fixed && lanes == 32 && tokens >= 3 && input_columns_ >= 4096 && out <= input_columns_ / 2},
             {"BM", bm}, {"SPLITS", splits}, {"CHUNK", int((std::int64_t(input_columns_) + splits * 256 - 1) / (splits * 256) * 256)}},
            std::nullopt, false, {}).front();
        if (splits > 1) values = split_reduce_kernel()({values}, {Shape{tokens, rows_}}, {dtype},
            {tokens * rows_, 1, 1}, {128, 1, 1}, {{"T", dtype}, {"TOTAL", rows_}, {"SPLITS", splits}},
            std::nullopt, false, {}).front();
        return reshape(values, shape);
    }
    auto grouped = *this;
    grouped.experts_ = groups;
    if (pooled_) grouped.descriptors_ = mlx::core::concatenate({slice(descriptors_, {0}, {rows_}),
        broadcast_to(slice(descriptors_, {rows_}, {rows_ + 1}), {groups})});
    auto ids = broadcast_to(arange(groups, int32), {tokens, groups});
    return reshape(grouped.routed_matmul(input, ids), shape);
}

array MlxTqWeight::swiglu(const MlxTqWeight& up, const array& input,
    const array& ids, float limit, bool gelu, const array* reuse) const {
    return project(input, ids, &up, limit, gelu, 0, reuse);
}

array MlxTqWeight::swiglu_split(const array& input, const array& ids, float limit, bool gelu) const {
    if (output_size() % 2) throw std::invalid_argument("TQ Gate/Up bank has an odd row count");
    return project(input, ids, nullptr, limit, gelu, output_size() / 2);
}

array MlxTqWeight::embedding(const array& ids, Dtype dtype) const {
    if (!ids.size() || (dtype != float16 && dtype != bfloat16 && dtype != float32))
        throw std::invalid_argument("invalid TQ embedding request");
    auto shape = ids.shape(); shape.push_back(input_columns_);
    return embedding_kernel()({words_, descriptors_, scales_, relative_table(), contiguous(astype(ids, int32))},
        {shape}, {dtype}, {(input_columns_ + 3) / 4, int(ids.size()), 1}, {128, 1, 1},
        {{"T", dtype}, {"ROWS", rows_}, {"K", input_columns_}, {"WIDTH", width_}, {"OUT", output_size()},
         {"GS", group_size_}, {"COLUMNS", columns_}, {"POOLED", pooled_}}, std::nullopt, false, {}).front();
}

void MlxTqWeight::set_expert_word_offset(int expert, std::uint64_t word_offset) {
    if (!pooled_ || expert < 0 || expert >= experts_)
        throw std::out_of_range("invalid TQ pooled expert");
    auto* descriptors = descriptors_.data<std::uint64_t>();
    const auto last = descriptors[(expert + 1) * output_size() - 1];
    const auto words = std::uint32_t(last) + std::uint64_t(width_) * (((last >> 32) & 7) + 1) / 32;
    if (word_offset > words_.size() || words > words_.size() - word_offset)
        throw std::out_of_range("TQ expert exceeds the word pool");
    descriptors[rows_ + expert] = word_offset;
}
} // namespace mfq::metal
