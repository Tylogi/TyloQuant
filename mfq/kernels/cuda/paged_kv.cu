// Physical-page KV writer and page-aware decode attention.
// Page table: [batch, logical_pages] int32 physical page ids.
// Chunk pointer tables: [chunks] int64 device addresses. Each chunk stores
// [pages_per_chunk, kv_heads, page_size, head_dim].

#include <cuda_runtime.h>

#include "mfq_cuda_paged_kv.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace {

template <typename scalar_t>
__device__ __forceinline__ scalar_t * page_address(
        const int64_t * chunk_ptrs, int physical_page,
        int pages_per_chunk, size_t page_elements) {
    const int chunk = physical_page / pages_per_chunk;
    const int local_page = physical_page - chunk * pages_per_chunk;
    return reinterpret_cast<scalar_t *>(
        static_cast<uintptr_t>(chunk_ptrs[chunk])) +
        static_cast<size_t>(local_page) * page_elements;
}

template <int Count, typename scalar_t>
__device__ __forceinline__ void paged_load_values(
        const scalar_t * source, float (&values)[Count]) {
    if constexpr (Count == 8 && std::is_same_v<scalar_t, mfq_half>) {
        const uint4 packed = *reinterpret_cast<const uint4 *>(source);
        values[0] = __half2float(__ushort_as_half(
            static_cast<unsigned short>(packed.x)));
        values[1] = __half2float(__ushort_as_half(
            static_cast<unsigned short>(packed.x >> 16)));
        values[2] = __half2float(__ushort_as_half(
            static_cast<unsigned short>(packed.y)));
        values[3] = __half2float(__ushort_as_half(
            static_cast<unsigned short>(packed.y >> 16)));
        values[4] = __half2float(__ushort_as_half(
            static_cast<unsigned short>(packed.z)));
        values[5] = __half2float(__ushort_as_half(
            static_cast<unsigned short>(packed.z >> 16)));
        values[6] = __half2float(__ushort_as_half(
            static_cast<unsigned short>(packed.w)));
        values[7] = __half2float(__ushort_as_half(
            static_cast<unsigned short>(packed.w >> 16)));
    } else {
        #pragma unroll
        for (int index = 0; index < Count; ++index) {
            values[index] = static_cast<float>(source[index]);
        }
    }
}

template <typename scalar_t>
__global__ void paged_kv_cache_write_kernel(
    const int64_t * __restrict__ k_chunk_ptrs,
    const int64_t * __restrict__ v_chunk_ptrs,
    const int32_t * __restrict__ page_table,
    const scalar_t * __restrict__ k,
    const scalar_t * __restrict__ v,
    const int64_t * __restrict__ positions,
    int B, int H, int T, int D, int logical_pages,
    int page_size, int pages_per_chunk, int chunks, int position_dims) {
    const size_t elements = static_cast<size_t>(B) * H * T * D;
    const size_t page_elements = static_cast<size_t>(H) * page_size * D;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
            threadIdx.x;
         index < elements;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int d = static_cast<int>(index % D);
        const size_t token_index = index / D;
        const int t = static_cast<int>(token_index % T);
        const size_t head_index = token_index / T;
        const int h = static_cast<int>(head_index % H);
        const int b = static_cast<int>(head_index / H);
        const int64_t position = position_dims == 1
            ? positions[t] : positions[static_cast<size_t>(b) * T + t];
        if (position < 0) continue;
        const int logical_page = static_cast<int>(position / page_size);
        if (logical_page >= logical_pages) continue;
        const int physical_page =
            page_table[static_cast<size_t>(b) * logical_pages + logical_page];
        const int chunk = physical_page >= 0
            ? physical_page / pages_per_chunk : -1;
        if (chunk < 0 || chunk >= chunks ||
                k_chunk_ptrs[chunk] == 0 || v_chunk_ptrs[chunk] == 0) {
            continue;
        }
        const int page_offset = static_cast<int>(position % page_size);
        const size_t page_index =
            (static_cast<size_t>(h) * page_size + page_offset) * D + d;
        const size_t source =
            ((static_cast<size_t>(b) * H + h) * T + t) * D + d;
        page_address<scalar_t>(
            k_chunk_ptrs, physical_page, pages_per_chunk,
            page_elements)[page_index] = k[source];
        page_address<scalar_t>(
            v_chunk_ptrs, physical_page, pages_per_chunk,
            page_elements)[page_index] = v[source];
    }
}

template <int Warps>
__device__ __forceinline__ float paged_block_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    __shared__ float warp_values[Warps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    value = lane < Warps ? warp_values[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
    }
    __shared__ float total;
    if (threadIdx.x == 0) total = value;
    __syncthreads();
    return total;
}

__device__ __forceinline__ float paged_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ int paged_active_parts(
        int token_count, int launch_parts, int dynamic_parts) {
    if (!dynamic_parts) return launch_parts;
    const int cache_position = token_count > 0 ? token_count - 1 : 0;
    if (cache_position < 192) return 1;
    const int active = (cache_position + 127) / 128;
    return active < launch_parts ? active : launch_parts;
}

template <int Block, typename scalar_t>
__global__ void paged_attention_decode_kernel(
    const scalar_t * __restrict__ q,
    const int64_t * __restrict__ k_chunk_ptrs,
    const int64_t * __restrict__ v_chunk_ptrs,
    const int32_t * __restrict__ page_table,
    const int64_t * __restrict__ seq_len,
    scalar_t * __restrict__ output,
    int B, int Hq, int Hk, int D, int logical_pages,
    int page_size, int pages_per_chunk, int chunks, int rep, float scale) {
    constexpr int Warps = Block / 32;
    const int query_head = blockIdx.x % Hq;
    const int batch = blockIdx.x / Hq;
    const int kv_head = query_head / rep;
    const int tid = threadIdx.x;
    const int maximum_tokens = logical_pages * page_size;
    int tokens = static_cast<int>(seq_len[batch]);
    tokens = tokens < 0 ? 0 : (tokens > maximum_tokens ? maximum_tokens : tokens);
    const float query = tid < D
        ? static_cast<float>(q[(static_cast<size_t>(batch) * Hq +
            query_head) * D + tid]) : 0.0f;
    float maximum = -1e30f;
    float denominator = 0.0f;
    float value_sum = 0.0f;
    const size_t page_elements =
        static_cast<size_t>(Hk) * page_size * D;
    __shared__ uintptr_t key_page_address;
    __shared__ uintptr_t value_page_address;
    for (int token = 0; token < tokens; ++token) {
        const int offset = token % page_size;
        if (offset == 0) {
            if (tid == 0) {
                const int physical = page_table[
                    static_cast<size_t>(batch) * logical_pages +
                    token / page_size];
                const int chunk = physical >= 0
                    ? physical / pages_per_chunk : -1;
                key_page_address = chunk >= 0 && chunk < chunks &&
                        k_chunk_ptrs[chunk] != 0
                    ? reinterpret_cast<uintptr_t>(page_address<scalar_t>(
                        k_chunk_ptrs, physical, pages_per_chunk,
                        page_elements)) : 0;
                value_page_address = chunk >= 0 && chunk < chunks &&
                        v_chunk_ptrs[chunk] != 0
                    ? reinterpret_cast<uintptr_t>(page_address<scalar_t>(
                        v_chunk_ptrs, physical, pages_per_chunk,
                        page_elements)) : 0;
            }
            __syncthreads();
        }
        const auto * key_page = reinterpret_cast<const scalar_t *>(
            key_page_address);
        const auto * value_page = reinterpret_cast<const scalar_t *>(
            value_page_address);
        const size_t element =
            (static_cast<size_t>(kv_head) * page_size + offset) * D + tid;
        const float key = tid < D && key_page != nullptr
            ? static_cast<float>(key_page[element]) : 0.0f;
        const float dot = paged_block_sum<Warps>(query * key);
        const float score = dot * scale;
        const float new_maximum = fmaxf(maximum, score);
        const float previous_factor = expf(maximum - new_maximum);
        const float probability = expf(score - new_maximum);
        const float value = tid < D && value_page != nullptr
            ? static_cast<float>(value_page[element]) : 0.0f;
        value_sum = value_sum * previous_factor + probability * value;
        denominator = denominator * previous_factor + probability;
        maximum = new_maximum;
    }
    if (tid < D) {
        output[(static_cast<size_t>(batch) * Hq + query_head) * D + tid] =
            static_cast<scalar_t>(value_sum /
                (denominator > 0.0f ? denominator : 1.0f));
    }
}

template <typename scalar_t>
__global__ void paged_attention_decode_gqa4_d256_kernel(
    const scalar_t * __restrict__ q,
    const int64_t * __restrict__ k_chunk_ptrs,
    const int64_t * __restrict__ v_chunk_ptrs,
    const int32_t * __restrict__ page_table,
    const int64_t * __restrict__ seq_len,
    scalar_t * __restrict__ output,
    int Hq, int Hk, int logical_pages, int page_size,
    int pages_per_chunk, int chunks, float scale) {
    constexpr int D = 256;
    constexpr int Rep = 4;
    constexpr int Warps = D / 32;
    const int kv_head = blockIdx.x % Hk;
    const int batch = blockIdx.x / Hk;
    const int first_query_head = kv_head * Rep;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int maximum_tokens = logical_pages * page_size;
    int tokens = static_cast<int>(seq_len[batch]);
    tokens = tokens < 0 ? 0 : (tokens > maximum_tokens ? maximum_tokens : tokens);
    float query[Rep];
    float value_sum[Rep];
    #pragma unroll
    for (int index = 0; index < Rep; ++index) {
        query[index] = static_cast<float>(q[
            (static_cast<size_t>(batch) * Hq + first_query_head + index) *
            D + tid]);
        value_sum[index] = 0.0f;
    }
    __shared__ float warp_dot[Rep][Warps];
    __shared__ float maximum[Rep];
    __shared__ float denominator[Rep];
    __shared__ float previous_factor[Rep];
    __shared__ float probability[Rep];
    __shared__ uintptr_t key_page_address;
    __shared__ uintptr_t value_page_address;
    if (tid < Rep) {
        maximum[tid] = -1e30f;
        denominator[tid] = 0.0f;
    }
    __syncthreads();
    const size_t page_elements =
        static_cast<size_t>(Hk) * page_size * D;
    for (int token = 0; token < tokens; ++token) {
        const int offset = token % page_size;
        if (offset == 0) {
            if (tid == 0) {
                const int physical = page_table[
                    static_cast<size_t>(batch) * logical_pages +
                    token / page_size];
                const int chunk = physical >= 0
                    ? physical / pages_per_chunk : -1;
                key_page_address = chunk >= 0 && chunk < chunks &&
                        k_chunk_ptrs[chunk] != 0
                    ? reinterpret_cast<uintptr_t>(page_address<scalar_t>(
                        k_chunk_ptrs, physical, pages_per_chunk,
                        page_elements)) : 0;
                value_page_address = chunk >= 0 && chunk < chunks &&
                        v_chunk_ptrs[chunk] != 0
                    ? reinterpret_cast<uintptr_t>(page_address<scalar_t>(
                        v_chunk_ptrs, physical, pages_per_chunk,
                        page_elements)) : 0;
            }
            __syncthreads();
        }
        const auto * key_page = reinterpret_cast<const scalar_t *>(
            key_page_address);
        const auto * value_page = reinterpret_cast<const scalar_t *>(
            value_page_address);
        const size_t element =
            (static_cast<size_t>(kv_head) * page_size + offset) * D + tid;
        const float key = key_page != nullptr
            ? static_cast<float>(key_page[element]) : 0.0f;
        const float value = value_page != nullptr
            ? static_cast<float>(value_page[element]) : 0.0f;
        #pragma unroll
        for (int index = 0; index < Rep; ++index) {
            const float dot = paged_warp_sum(query[index] * key);
            if (lane == 0) warp_dot[index][warp] = dot;
        }
        __syncthreads();
        if (warp == 0) {
            #pragma unroll
            for (int index = 0; index < Rep; ++index) {
                float dot = lane < Warps ? warp_dot[index][lane] : 0.0f;
                dot = paged_warp_sum(dot);
                if (lane == 0) {
                    const float score = dot * scale;
                    const float new_maximum = fmaxf(maximum[index], score);
                    previous_factor[index] = expf(
                        maximum[index] - new_maximum);
                    probability[index] = expf(score - new_maximum);
                    denominator[index] = denominator[index] *
                        previous_factor[index] + probability[index];
                    maximum[index] = new_maximum;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int index = 0; index < Rep; ++index) {
            value_sum[index] = value_sum[index] * previous_factor[index] +
                probability[index] * value;
        }
    }
    #pragma unroll
    for (int index = 0; index < Rep; ++index) {
        output[(static_cast<size_t>(batch) * Hq + first_query_head + index) *
            D + tid] = static_cast<scalar_t>(value_sum[index] /
                (denominator[index] > 0.0f ? denominator[index] : 1.0f));
    }
}

template <int Block, typename scalar_t>
__global__ void paged_attention_decode_split_kernel(
    const scalar_t * __restrict__ q,
    const int64_t * __restrict__ k_chunk_ptrs,
    const int64_t * __restrict__ v_chunk_ptrs,
    const int32_t * __restrict__ page_table,
    const int64_t * __restrict__ seq_len,
    float * __restrict__ partial_o,
    float * __restrict__ partial_m,
    float * __restrict__ partial_l,
    int Hq, int Hk, int D, int logical_pages,
    int page_size, int pages_per_chunk, int chunks, int rep,
    int parts, int workspace_parts, float scale, int dynamic_parts) {
    constexpr int Warps = Block / 32;
    const int part = blockIdx.x % parts;
    const int query = blockIdx.x / parts;
    const int query_head = query % Hq;
    const int batch = query / Hq;
    const int kv_head = query_head / rep;
    const int tid = threadIdx.x;
    const int maximum_tokens = logical_pages * page_size;
    int tokens = static_cast<int>(seq_len[batch]);
    tokens = tokens < 0 ? 0 : (tokens > maximum_tokens ? maximum_tokens : tokens);
    const int active_parts = paged_active_parts(
        tokens, parts, dynamic_parts);
    if (part >= active_parts) return;
    const size_t statistic =
        static_cast<size_t>(query) * workspace_parts + part;
    const int start = static_cast<int>(
        static_cast<int64_t>(tokens) * part / active_parts);
    const int end = static_cast<int>(
        static_cast<int64_t>(tokens) * (part + 1) / active_parts);
    const float query_value = tid < D
        ? static_cast<float>(q[static_cast<size_t>(query) * D + tid]) : 0.0f;
    float maximum = -1e30f;
    float denominator = 0.0f;
    float value_sum = 0.0f;
    const size_t page_elements =
        static_cast<size_t>(Hk) * page_size * D;
    __shared__ uintptr_t key_page_address;
    __shared__ uintptr_t value_page_address;
    for (int token = start; token < end; ++token) {
        const int offset = token % page_size;
        if (token == start || offset == 0) {
            if (tid == 0) {
                const int physical = page_table[
                    static_cast<size_t>(batch) * logical_pages +
                    token / page_size];
                const int chunk = physical >= 0
                    ? physical / pages_per_chunk : -1;
                key_page_address = chunk >= 0 && chunk < chunks &&
                        k_chunk_ptrs[chunk] != 0
                    ? reinterpret_cast<uintptr_t>(page_address<scalar_t>(
                        k_chunk_ptrs, physical, pages_per_chunk,
                        page_elements)) : 0;
                value_page_address = chunk >= 0 && chunk < chunks &&
                        v_chunk_ptrs[chunk] != 0
                    ? reinterpret_cast<uintptr_t>(page_address<scalar_t>(
                        v_chunk_ptrs, physical, pages_per_chunk,
                        page_elements)) : 0;
            }
            __syncthreads();
        }
        const auto * key_page = reinterpret_cast<const scalar_t *>(
            key_page_address);
        const auto * value_page = reinterpret_cast<const scalar_t *>(
            value_page_address);
        const size_t element =
            (static_cast<size_t>(kv_head) * page_size + offset) * D + tid;
        const float key = tid < D && key_page != nullptr
            ? static_cast<float>(key_page[element]) : 0.0f;
        const float dot = paged_block_sum<Warps>(query_value * key);
        const float score = dot * scale;
        const float new_maximum = fmaxf(maximum, score);
        const float previous_factor = expf(maximum - new_maximum);
        const float probability = expf(score - new_maximum);
        const float value = tid < D && value_page != nullptr
            ? static_cast<float>(value_page[element]) : 0.0f;
        value_sum = value_sum * previous_factor + probability * value;
        denominator = denominator * previous_factor + probability;
        maximum = new_maximum;
    }
    if (tid < D) partial_o[statistic * D + tid] = value_sum;
    if (tid == 0) {
        partial_m[statistic] = start < end ? maximum : -1e30f;
        partial_l[statistic] = start < end ? denominator : 0.0f;
    }
}

template <int FixedPageSize, int FixedPagesPerChunk,
          int ValuesPerThread, typename scalar_t>
__global__ void paged_attention_decode_split_gqa4_d256_kernel(
    const scalar_t * __restrict__ q,
    const int64_t * __restrict__ k_chunk_ptrs,
    const int64_t * __restrict__ v_chunk_ptrs,
    const int32_t * __restrict__ page_table,
    const int64_t * __restrict__ seq_len,
    float * __restrict__ partial_o,
    float * __restrict__ partial_m,
    float * __restrict__ partial_l,
    int Hq, int Hk, int logical_pages, int page_size,
    int pages_per_chunk, int chunks, int parts, int workspace_parts,
    float scale, int dynamic_parts) {
    constexpr int D = 256;
    constexpr int Rep = 4;
    constexpr int Threads = D / ValuesPerThread;
    constexpr int Warps = Threads / 32;
    static_assert(ValuesPerThread == 1 || ValuesPerThread == 2 ||
        ValuesPerThread == 4 || ValuesPerThread == 8);
    const int part = blockIdx.x % parts;
    const int kv = blockIdx.x / parts;
    const int kv_head = kv % Hk;
    const int batch = kv / Hk;
    const int first_query_head = kv_head * Rep;
    const int tid = threadIdx.x;
    const int first_dimension = tid * ValuesPerThread;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr bool FixedGeometry = FixedPageSize > 0;
    const int active_page_size = FixedGeometry ? FixedPageSize : page_size;
    const int maximum_tokens = logical_pages * active_page_size;
    int tokens = static_cast<int>(seq_len[batch]);
    tokens = tokens < 0 ? 0 : (tokens > maximum_tokens ? maximum_tokens : tokens);
    const int active_parts = paged_active_parts(tokens, parts, dynamic_parts);
    if (part >= active_parts) return;
    const int start = static_cast<int>(
        static_cast<int64_t>(tokens) * part / active_parts);
    const int end = static_cast<int>(
        static_cast<int64_t>(tokens) * (part + 1) / active_parts);
    float query[Rep][ValuesPerThread];
    float value_sum[Rep][ValuesPerThread];
    #pragma unroll
    for (int index = 0; index < Rep; ++index) {
        const auto * query_source = q +
            (static_cast<size_t>(batch) * Hq + first_query_head + index) * D +
            first_dimension;
        paged_load_values(query_source, query[index]);
        #pragma unroll
        for (int value_index = 0;
                value_index < ValuesPerThread; ++value_index) {
            value_sum[index][value_index] = 0.0f;
        }
    }
    __shared__ float warp_dot[Rep][Warps];
    __shared__ float previous_factor[Rep];
    __shared__ float probability[Rep];
    __shared__ uintptr_t key_page_address;
    __shared__ uintptr_t value_page_address;
    float maximum[Rep];
    float denominator[Rep];
    if (tid == 0) {
        #pragma unroll
        for (int index = 0; index < Rep; ++index) {
            maximum[index] = -1e30f;
            denominator[index] = 0.0f;
        }
    }
    const size_t page_elements =
        static_cast<size_t>(Hk) * active_page_size * D;
    int token = start;
    while (token < end) {
        int offset = 0;
        int logical_page = 0;
        if constexpr (FixedGeometry) {
            static_assert(FixedPageSize == 16 && FixedPagesPerChunk == 64);
            offset = token & (FixedPageSize - 1);
            logical_page = token >> 4;
        } else {
            offset = token % page_size;
            logical_page = token / page_size;
        }
        if (tid == 0) {
            const int physical = page_table[
                static_cast<size_t>(batch) * logical_pages + logical_page];
            int chunk = -1;
            int local_page = 0;
            if constexpr (FixedGeometry) {
                chunk = physical >= 0 ? physical >> 6 : -1;
                local_page = physical & (FixedPagesPerChunk - 1);
            } else {
                chunk = physical >= 0
                    ? physical / pages_per_chunk : -1;
                local_page = physical - chunk * pages_per_chunk;
            }
            key_page_address = chunk >= 0 && chunk < chunks &&
                    k_chunk_ptrs[chunk] != 0
                ? reinterpret_cast<uintptr_t>(
                    reinterpret_cast<const scalar_t *>(
                        static_cast<uintptr_t>(k_chunk_ptrs[chunk])) +
                    static_cast<size_t>(local_page) * page_elements) : 0;
            value_page_address = chunk >= 0 && chunk < chunks &&
                    v_chunk_ptrs[chunk] != 0
                ? reinterpret_cast<uintptr_t>(
                    reinterpret_cast<const scalar_t *>(
                        static_cast<uintptr_t>(v_chunk_ptrs[chunk])) +
                    static_cast<size_t>(local_page) * page_elements) : 0;
        }
        if constexpr (Warps == 1) {
            __syncwarp();
        } else {
            __syncthreads();
        }
        const auto * key_page = reinterpret_cast<const scalar_t *>(
            key_page_address);
        const auto * value_page = reinterpret_cast<const scalar_t *>(
            value_page_address);
        const int page_end = min(
            end, token + active_page_size - offset);
        #pragma unroll 4
        for (; token < page_end; ++token, ++offset) {
            const size_t first_element =
                (static_cast<size_t>(kv_head) * active_page_size + offset) * D +
                first_dimension;
            float key[ValuesPerThread];
            float value[ValuesPerThread];
            if (key_page != nullptr) {
                paged_load_values(key_page + first_element, key);
            } else {
                #pragma unroll
                for (int value_index = 0;
                        value_index < ValuesPerThread; ++value_index) {
                    key[value_index] = 0.0f;
                }
            }
            if (value_page != nullptr) {
                paged_load_values(value_page + first_element, value);
            } else {
                #pragma unroll
                for (int value_index = 0;
                        value_index < ValuesPerThread; ++value_index) {
                    value[value_index] = 0.0f;
                }
            }
            #pragma unroll
            for (int index = 0; index < Rep; ++index) {
                float dot = 0.0f;
                #pragma unroll
                for (int value_index = 0;
                        value_index < ValuesPerThread; ++value_index) {
                    dot += query[index][value_index] * key[value_index];
                }
                dot = paged_warp_sum(dot);
                if constexpr (Warps == 1) {
                    if (lane == 0) {
                        const float score = dot * scale;
                        const float new_maximum = fmaxf(
                            maximum[index], score);
                        previous_factor[index] = expf(
                            maximum[index] - new_maximum);
                        probability[index] = expf(score - new_maximum);
                        denominator[index] = denominator[index] *
                            previous_factor[index] + probability[index];
                        maximum[index] = new_maximum;
                    }
                } else if (lane == 0) {
                    warp_dot[index][warp] = dot;
                }
            }
            if constexpr (Warps == 1) {
                __syncwarp();
            } else {
                __syncthreads();
                if (warp == 0) {
                    #pragma unroll
                    for (int index = 0; index < Rep; ++index) {
                        float dot = lane < Warps
                            ? warp_dot[index][lane] : 0.0f;
                        dot = paged_warp_sum(dot);
                        if (lane == 0) {
                            const float score = dot * scale;
                            const float new_maximum = fmaxf(
                                maximum[index], score);
                            previous_factor[index] = expf(
                                maximum[index] - new_maximum);
                            probability[index] = expf(score - new_maximum);
                            denominator[index] = denominator[index] *
                                previous_factor[index] + probability[index];
                            maximum[index] = new_maximum;
                        }
                    }
                }
                __syncthreads();
            }
            #pragma unroll
            for (int index = 0; index < Rep; ++index) {
                #pragma unroll
                for (int value_index = 0;
                        value_index < ValuesPerThread; ++value_index) {
                    value_sum[index][value_index] =
                        value_sum[index][value_index] * previous_factor[index] +
                        probability[index] * value[value_index];
                }
            }
        }
    }
    #pragma unroll
    for (int index = 0; index < Rep; ++index) {
        const int query_index = batch * Hq + first_query_head + index;
        const size_t statistic =
            static_cast<size_t>(query_index) * workspace_parts + part;
        #pragma unroll
        for (int value_index = 0;
                value_index < ValuesPerThread; ++value_index) {
            partial_o[statistic * D + first_dimension + value_index] =
                value_sum[index][value_index];
        }
        if (tid == 0) {
            partial_m[statistic] = start < end ? maximum[index] : -1e30f;
            partial_l[statistic] = start < end ? denominator[index] : 0.0f;
        }
    }
}

template <int Block, typename scalar_t>
__global__ void paged_attention_decode_reduce_kernel(
    const float * __restrict__ partial_o,
    const float * __restrict__ partial_m,
    const float * __restrict__ partial_l,
    const int64_t * __restrict__ seq_len,
    scalar_t * __restrict__ output,
    int total, int Hq, int D, int parts, int workspace_parts,
    int dynamic_parts) {
    const int query = blockIdx.x;
    const int tid = threadIdx.x;
    if (query >= total) return;
    const int active_parts = paged_active_parts(
        static_cast<int>(seq_len[query / Hq]), parts, dynamic_parts);
    __shared__ float part_weight[64];
    __shared__ float denominator;
    if (tid < 32) {
        const int first_part = tid;
        const int second_part = tid + 32;
        float maximum = first_part < active_parts
            ? partial_m[static_cast<size_t>(query) * workspace_parts +
                first_part] : -1e30f;
        if (second_part < active_parts) {
            maximum = fmaxf(maximum, partial_m[
                static_cast<size_t>(query) * workspace_parts + second_part]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_down_sync(
                0xffffffffu, maximum, offset));
        }
        maximum = __shfl_sync(0xffffffffu, maximum, 0);
        float sum = 0.0f;
        if (first_part < active_parts) {
            const size_t statistic =
                static_cast<size_t>(query) * workspace_parts + first_part;
            const float local_denominator = partial_l[statistic];
            const float weight = local_denominator > 0.0f
                ? expf(partial_m[statistic] - maximum) : 0.0f;
            part_weight[first_part] = weight;
            sum += weight * local_denominator;
        }
        if (second_part < active_parts) {
            const size_t statistic =
                static_cast<size_t>(query) * workspace_parts + second_part;
            const float local_denominator = partial_l[statistic];
            const float weight = local_denominator > 0.0f
                ? expf(partial_m[statistic] - maximum) : 0.0f;
            part_weight[second_part] = weight;
            sum += weight * local_denominator;
        }
        sum = paged_warp_sum(sum);
        if (tid == 0) denominator = sum;
    }
    __syncthreads();
    float value_sum = 0.0f;
    if (tid < D) {
        for (int part = 0; part < active_parts; ++part) {
            const size_t statistic =
                static_cast<size_t>(query) * workspace_parts + part;
            value_sum += part_weight[part] *
                partial_o[statistic * D + tid];
        }
        output[static_cast<size_t>(query) * D + tid] =
            static_cast<scalar_t>(value_sum /
                (denominator > 0.0f ? denominator : 1.0f));
    }
}

void validate_pointer_table(
        const mfq_tensor_backend::Tensor & pointers, const char * name) {
    MFQ_RUNTIME_CHECK(pointers.is_cuda() && pointers.is_contiguous() &&
        pointers.scalar_type() == mfq_tensor_backend::kInt64 &&
        pointers.dim() == 1 && pointers.numel() > 0,
        name, " must be a non-empty contiguous CUDA int64 vector");
}

} // namespace

void paged_kv_cache_write_cuda(
    mfq_tensor_backend::Tensor k_chunk_ptrs,
    mfq_tensor_backend::Tensor v_chunk_ptrs,
    mfq_tensor_backend::Tensor page_table,
    mfq_tensor_backend::Tensor k,
    mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor positions,
    int64_t page_size,
    int64_t pages_per_chunk) {
    validate_pointer_table(k_chunk_ptrs, "paged KV key chunk table");
    validate_pointer_table(v_chunk_ptrs, "paged KV value chunk table");
    MFQ_RUNTIME_CHECK(k_chunk_ptrs.numel() == v_chunk_ptrs.numel(),
        "paged KV chunk table sizes differ");
    MFQ_RUNTIME_CHECK(page_table.is_cuda() && page_table.is_contiguous() &&
        page_table.scalar_type() == mfq_tensor_backend::kInt32 &&
        page_table.dim() == 2,
        "paged KV page table must be contiguous CUDA int32[batch,pages]");
    MFQ_RUNTIME_CHECK(k.is_cuda() && k.is_contiguous() &&
        v.is_cuda() && v.is_contiguous() && k.dim() == 4 &&
        k.sizes() == v.sizes(),
        "paged KV inputs must be matching contiguous CUDA rank-4 tensors");
    MFQ_RUNTIME_CHECK(k.scalar_type() == v.scalar_type() &&
        (k.scalar_type() == mfq_tensor_backend::kFloat16 ||
         k.scalar_type() == mfq_tensor_backend::kBFloat16 ||
         k.scalar_type() == mfq_tensor_backend::kFloat32),
        "paged KV inputs require a supported matching floating dtype");
    MFQ_RUNTIME_CHECK(positions.is_cuda() && positions.is_contiguous() &&
        positions.scalar_type() == mfq_tensor_backend::kInt64 &&
        (positions.dim() == 1 || positions.dim() == 2),
        "paged KV positions must be contiguous CUDA int64 [T] or [B,T]");
    MFQ_RUNTIME_CHECK(page_size > 0 && page_size <= INT_MAX &&
        pages_per_chunk > 0 && pages_per_chunk <= INT_MAX,
        "paged KV page geometry is invalid");
    const int B = static_cast<int>(k.size(0));
    const int H = static_cast<int>(k.size(1));
    const int T = static_cast<int>(k.size(2));
    const int D = static_cast<int>(k.size(3));
    MFQ_RUNTIME_CHECK(page_table.size(0) >= B && page_table.size(1) > 0,
        "paged KV page table does not cover the input batch");
    if (positions.dim() == 1) {
        MFQ_RUNTIME_CHECK(positions.numel() == T,
            "paged KV position vector length mismatch");
    } else {
        MFQ_RUNTIME_CHECK(positions.size(0) == B && positions.size(1) == T,
            "paged KV position matrix shape mismatch");
    }
    if (T == 0) return;
    constexpr int Threads = 256;
    const size_t elements = static_cast<size_t>(B) * H * T * D;
    const int blocks = static_cast<int>(std::min<size_t>(
        4096, (elements + Threads - 1) / Threads));
    MFQ_DISPATCH_FLOATING_TYPES_AND2(
        mfq_dispatch_half, mfq_dispatch_bfloat16,
        k.scalar_type(), "paged_kv_cache_write_cuda", [&] {
        paged_kv_cache_write_kernel<scalar_t><<<
            blocks, Threads, 0, mfq_current_cuda_stream()>>>(
            k_chunk_ptrs.data_ptr<int64_t>(),
            v_chunk_ptrs.data_ptr<int64_t>(),
            page_table.data_ptr<int32_t>(), k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(), positions.data_ptr<int64_t>(),
            B, H, T, D, static_cast<int>(page_table.size(1)),
            static_cast<int>(page_size), static_cast<int>(pages_per_chunk),
            static_cast<int>(k_chunk_ptrs.numel()), positions.dim());
    });
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}

mfq_tensor_backend::Tensor attention_paged_cache_decode_cuda(
    mfq_tensor_backend::Tensor q,
    mfq_tensor_backend::Tensor k_chunk_ptrs,
    mfq_tensor_backend::Tensor v_chunk_ptrs,
    mfq_tensor_backend::Tensor page_table,
    mfq_tensor_backend::Tensor seq_len,
    double scale,
    int64_t page_size,
    int64_t pages_per_chunk,
    int64_t kv_heads,
    mfq_tensor_backend::Tensor partial_o,
    mfq_tensor_backend::Tensor partial_m,
    mfq_tensor_backend::Tensor partial_l,
    int64_t parts,
    bool dynamic_parts) {
    validate_pointer_table(k_chunk_ptrs, "paged attention key chunk table");
    validate_pointer_table(v_chunk_ptrs, "paged attention value chunk table");
    MFQ_RUNTIME_CHECK(k_chunk_ptrs.numel() == v_chunk_ptrs.numel(),
        "paged attention chunk table sizes differ");
    MFQ_RUNTIME_CHECK(page_table.is_cuda() && page_table.is_contiguous() &&
        page_table.scalar_type() == mfq_tensor_backend::kInt32 &&
        page_table.dim() == 2,
        "paged attention page table must be contiguous CUDA int32[batch,pages]");
    MFQ_RUNTIME_CHECK(seq_len.is_cuda() && seq_len.is_contiguous() &&
        seq_len.scalar_type() == mfq_tensor_backend::kInt64 &&
        seq_len.dim() == 1,
        "paged attention lengths must be contiguous CUDA int64[batch]");
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous() && q.dim() == 4 &&
        q.size(2) == 1 &&
        (q.scalar_type() == mfq_tensor_backend::kFloat16 ||
         q.scalar_type() == mfq_tensor_backend::kBFloat16 ||
         q.scalar_type() == mfq_tensor_backend::kFloat32),
        "paged attention query must be CUDA [batch,heads,1,dim]");
    MFQ_RUNTIME_CHECK(page_size > 0 && page_size <= INT_MAX &&
        pages_per_chunk > 0 && pages_per_chunk <= INT_MAX &&
        parts >= 1 && parts <= 64,
        "paged attention geometry is invalid");
    const int B = static_cast<int>(q.size(0));
    const int Hq = static_cast<int>(q.size(1));
    const int D = static_cast<int>(q.size(3));
    MFQ_RUNTIME_CHECK(page_table.size(0) >= B && page_table.size(1) > 0 &&
        seq_len.numel() == B,
        "paged attention metadata does not cover the query batch");
    MFQ_RUNTIME_CHECK(partial_o.defined() && partial_o.dim() == 3 &&
        partial_o.size(0) == B * Hq && partial_o.size(2) == D,
        "paged attention partial output shape mismatch");
    const int workspace_parts = static_cast<int>(partial_o.size(1));
    MFQ_RUNTIME_CHECK(parts <= workspace_parts && partial_m.defined() &&
        partial_l.defined() && partial_m.is_cuda() && partial_l.is_cuda() &&
        partial_m.is_contiguous() && partial_l.is_contiguous() &&
        partial_m.scalar_type() == mfq_tensor_backend::kFloat32 &&
        partial_l.scalar_type() == mfq_tensor_backend::kFloat32 &&
        partial_m.sizes() == mfq_tensor_backend::IntArrayRef(
            {B * Hq, workspace_parts}) &&
        partial_l.sizes() == partial_m.sizes(),
        "paged attention partial statistics shape mismatch");
    MFQ_RUNTIME_CHECK(kv_heads > 0 && kv_heads <= Hq &&
        Hq % kv_heads == 0,
        "paged attention received an invalid KV-head count");
    const int Hk = static_cast<int>(kv_heads);
    const int gqa_ratio = Hq / Hk;
    auto output = mfq_tensor_backend::empty_like(q);
    const int total = B * Hq;
    const int chunks = static_cast<int>(k_chunk_ptrs.numel());
    const int logical_pages = static_cast<int>(page_table.size(1));
    const int page = static_cast<int>(page_size);
    const int chunk_pages = static_cast<int>(pages_per_chunk);
    const int split_parts = static_cast<int>(parts);
    auto stream = mfq_current_cuda_stream();
    MFQ_DISPATCH_FLOATING_TYPES_AND2(
        mfq_dispatch_half, mfq_dispatch_bfloat16,
        q.scalar_type(), "attention_paged_cache_decode_cuda", [&] {
        if (split_parts == 1) {
            if (D == 256 && gqa_ratio == 4) {
                paged_attention_decode_gqa4_d256_kernel<scalar_t><<<
                    B * Hk, 256, 0, stream>>>(
                    q.data_ptr<scalar_t>(),
                    k_chunk_ptrs.data_ptr<int64_t>(),
                    v_chunk_ptrs.data_ptr<int64_t>(),
                    page_table.data_ptr<int32_t>(),
                    seq_len.data_ptr<int64_t>(), output.data_ptr<scalar_t>(),
                    Hq, Hk, logical_pages, page, chunk_pages, chunks,
                    static_cast<float>(scale));
            } else if (D <= 64) {
                paged_attention_decode_kernel<64, scalar_t><<<
                    total, 64, 0, stream>>>(q.data_ptr<scalar_t>(),
                    k_chunk_ptrs.data_ptr<int64_t>(), v_chunk_ptrs.data_ptr<int64_t>(),
                    page_table.data_ptr<int32_t>(), seq_len.data_ptr<int64_t>(),
                    output.data_ptr<scalar_t>(), B, Hq, Hk, D, logical_pages,
                    page, chunk_pages, chunks, gqa_ratio, static_cast<float>(scale));
            } else if (D <= 128) {
                paged_attention_decode_kernel<128, scalar_t><<<
                    total, 128, 0, stream>>>(q.data_ptr<scalar_t>(),
                    k_chunk_ptrs.data_ptr<int64_t>(), v_chunk_ptrs.data_ptr<int64_t>(),
                    page_table.data_ptr<int32_t>(), seq_len.data_ptr<int64_t>(),
                    output.data_ptr<scalar_t>(), B, Hq, Hk, D, logical_pages,
                    page, chunk_pages, chunks, gqa_ratio, static_cast<float>(scale));
            } else if (D <= 256) {
                paged_attention_decode_kernel<256, scalar_t><<<
                    total, 256, 0, stream>>>(q.data_ptr<scalar_t>(),
                    k_chunk_ptrs.data_ptr<int64_t>(), v_chunk_ptrs.data_ptr<int64_t>(),
                    page_table.data_ptr<int32_t>(), seq_len.data_ptr<int64_t>(),
                    output.data_ptr<scalar_t>(), B, Hq, Hk, D, logical_pages,
                    page, chunk_pages, chunks, gqa_ratio, static_cast<float>(scale));
            } else if (D <= 512) {
                paged_attention_decode_kernel<512, scalar_t><<<
                    total, 512, 0, stream>>>(q.data_ptr<scalar_t>(),
                    k_chunk_ptrs.data_ptr<int64_t>(), v_chunk_ptrs.data_ptr<int64_t>(),
                    page_table.data_ptr<int32_t>(), seq_len.data_ptr<int64_t>(),
                    output.data_ptr<scalar_t>(), B, Hq, Hk, D, logical_pages,
                    page, chunk_pages, chunks, gqa_ratio, static_cast<float>(scale));
            } else {
                MFQ_RUNTIME_CHECK(false, "paged attention head_dim exceeds 512");
            }
        } else {
            const int shmem = D * static_cast<int>(sizeof(float));
            if (D == 256 && gqa_ratio == 4) {
                if (page == 16 && chunk_pages == 64) {
                    paged_attention_decode_split_gqa4_d256_kernel<
                        16, 64, 8, scalar_t><<<
                        B * Hk * split_parts, 32, 0, stream>>>(
                        q.data_ptr<scalar_t>(), k_chunk_ptrs.data_ptr<int64_t>(),
                        v_chunk_ptrs.data_ptr<int64_t>(),
                        page_table.data_ptr<int32_t>(), seq_len.data_ptr<int64_t>(),
                        partial_o.data_ptr<float>(), partial_m.data_ptr<float>(),
                        partial_l.data_ptr<float>(), Hq, Hk, logical_pages, page,
                        chunk_pages, chunks, split_parts, workspace_parts,
                        static_cast<float>(scale), dynamic_parts ? 1 : 0);
                } else {
                    paged_attention_decode_split_gqa4_d256_kernel<
                        0, 0, 1, scalar_t><<<
                        B * Hk * split_parts, 256, 0, stream>>>(
                        q.data_ptr<scalar_t>(), k_chunk_ptrs.data_ptr<int64_t>(),
                        v_chunk_ptrs.data_ptr<int64_t>(),
                        page_table.data_ptr<int32_t>(), seq_len.data_ptr<int64_t>(),
                        partial_o.data_ptr<float>(), partial_m.data_ptr<float>(),
                        partial_l.data_ptr<float>(), Hq, Hk, logical_pages, page,
                        chunk_pages, chunks, split_parts, workspace_parts,
                        static_cast<float>(scale), dynamic_parts ? 1 : 0);
                }
            } else if (D <= 64) {
                paged_attention_decode_split_kernel<64, scalar_t><<<
                    total * split_parts, 64, shmem, stream>>>(q.data_ptr<scalar_t>(),
                    k_chunk_ptrs.data_ptr<int64_t>(), v_chunk_ptrs.data_ptr<int64_t>(),
                    page_table.data_ptr<int32_t>(), seq_len.data_ptr<int64_t>(),
                    partial_o.data_ptr<float>(), partial_m.data_ptr<float>(),
                    partial_l.data_ptr<float>(), Hq, Hk, D, logical_pages, page,
                    chunk_pages, chunks, gqa_ratio, split_parts, workspace_parts,
                    static_cast<float>(scale), dynamic_parts ? 1 : 0);
            } else if (D <= 128) {
                paged_attention_decode_split_kernel<128, scalar_t><<<
                    total * split_parts, 128, shmem, stream>>>(q.data_ptr<scalar_t>(),
                    k_chunk_ptrs.data_ptr<int64_t>(), v_chunk_ptrs.data_ptr<int64_t>(),
                    page_table.data_ptr<int32_t>(), seq_len.data_ptr<int64_t>(),
                    partial_o.data_ptr<float>(), partial_m.data_ptr<float>(),
                    partial_l.data_ptr<float>(), Hq, Hk, D, logical_pages, page,
                    chunk_pages, chunks, gqa_ratio, split_parts, workspace_parts,
                    static_cast<float>(scale), dynamic_parts ? 1 : 0);
            } else if (D <= 256) {
                paged_attention_decode_split_kernel<256, scalar_t><<<
                    total * split_parts, 256, shmem, stream>>>(q.data_ptr<scalar_t>(),
                    k_chunk_ptrs.data_ptr<int64_t>(), v_chunk_ptrs.data_ptr<int64_t>(),
                    page_table.data_ptr<int32_t>(), seq_len.data_ptr<int64_t>(),
                    partial_o.data_ptr<float>(), partial_m.data_ptr<float>(),
                    partial_l.data_ptr<float>(), Hq, Hk, D, logical_pages, page,
                    chunk_pages, chunks, gqa_ratio, split_parts, workspace_parts,
                    static_cast<float>(scale), dynamic_parts ? 1 : 0);
            } else if (D <= 512) {
                paged_attention_decode_split_kernel<512, scalar_t><<<
                    total * split_parts, 512, shmem, stream>>>(q.data_ptr<scalar_t>(),
                    k_chunk_ptrs.data_ptr<int64_t>(), v_chunk_ptrs.data_ptr<int64_t>(),
                    page_table.data_ptr<int32_t>(), seq_len.data_ptr<int64_t>(),
                    partial_o.data_ptr<float>(), partial_m.data_ptr<float>(),
                    partial_l.data_ptr<float>(), Hq, Hk, D, logical_pages, page,
                    chunk_pages, chunks, gqa_ratio, split_parts, workspace_parts,
                    static_cast<float>(scale), dynamic_parts ? 1 : 0);
            } else {
                MFQ_RUNTIME_CHECK(false, "paged attention head_dim exceeds 512");
            }
            if (D <= 64) {
                paged_attention_decode_reduce_kernel<64, scalar_t><<<
                    total, 64, 0, stream>>>(partial_o.data_ptr<float>(),
                    partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                    seq_len.data_ptr<int64_t>(), output.data_ptr<scalar_t>(), total,
                    Hq, D, split_parts, workspace_parts, dynamic_parts ? 1 : 0);
            } else if (D <= 128) {
                paged_attention_decode_reduce_kernel<128, scalar_t><<<
                    total, 128, 0, stream>>>(partial_o.data_ptr<float>(),
                    partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                    seq_len.data_ptr<int64_t>(), output.data_ptr<scalar_t>(), total,
                    Hq, D, split_parts, workspace_parts, dynamic_parts ? 1 : 0);
            } else if (D <= 256) {
                paged_attention_decode_reduce_kernel<256, scalar_t><<<
                    total, 256, 0, stream>>>(partial_o.data_ptr<float>(),
                    partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                    seq_len.data_ptr<int64_t>(), output.data_ptr<scalar_t>(), total,
                    Hq, D, split_parts, workspace_parts, dynamic_parts ? 1 : 0);
            } else {
                paged_attention_decode_reduce_kernel<512, scalar_t><<<
                    total, 512, 0, stream>>>(partial_o.data_ptr<float>(),
                    partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                    seq_len.data_ptr<int64_t>(), output.data_ptr<scalar_t>(), total,
                    Hq, D, split_parts, workspace_parts, dynamic_parts ? 1 : 0);
            }
        }
    });
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
