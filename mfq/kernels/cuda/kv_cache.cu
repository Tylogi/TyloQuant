// KV cache writer for decode/prefill plumbing.
// Layout: k/v input [B, H, T, D], cache [B, H, max_seq, D].

#include <cuda_runtime.h>
#include "../../../cpp_runtime/cuda/mfq_tensor_backend.h"
#include <algorithm>
#include <cstdint>
#include <vector>

template <typename scalar_t>
__global__ void kv_cache_write_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    const int64_t* __restrict__ positions,
    int B,
    int H,
    int T,
    int D,
    int max_seq,
    int pos_dim)
{
    size_t n = (size_t)B * H * T * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        int d = (int)(idx % D);
        size_t t0 = idx / D;
        int t = (int)(t0 % T);
        size_t h0 = t0 / T;
        int h = (int)(h0 % H);
        int b = (int)(h0 / H);
        int64_t p = pos_dim == 1 ? positions[t] : positions[(size_t)b * T + t];
        if (p < 0 || p >= max_seq) {
            continue;
        }
        size_t src = (((size_t)b * H + h) * T + t) * D + d;
        size_t dst = (((size_t)b * H + h) * max_seq + (size_t)p) * D + d;
        k_cache[dst] = k[src];
        v_cache[dst] = v[src];
    }
}

std::vector<mfq_tensor_backend::Tensor> kv_cache_write_cuda(
    mfq_tensor_backend::Tensor k_cache,
    mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor k,
    mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor positions)
{
    MFQ_RUNTIME_CHECK(k_cache.is_cuda() && k_cache.is_contiguous(), "kv_cache_write: k_cache must be cuda contiguous");
    MFQ_RUNTIME_CHECK(v_cache.is_cuda() && v_cache.is_contiguous(), "kv_cache_write: v_cache must be cuda contiguous");
    MFQ_RUNTIME_CHECK(k.is_cuda() && k.is_contiguous(), "kv_cache_write: k must be cuda contiguous");
    MFQ_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous(), "kv_cache_write: v must be cuda contiguous");
    MFQ_RUNTIME_CHECK(k_cache.scalar_type() == v_cache.scalar_type() && k_cache.scalar_type() == k.scalar_type()
                && k_cache.scalar_type() == v.scalar_type(),
                "kv_cache_write: k/v/cache dtype mismatch");
    MFQ_RUNTIME_CHECK(k_cache.scalar_type() == mfq_tensor_backend::kFloat16 || k_cache.scalar_type() == mfq_tensor_backend::kFloat32,
                "kv_cache_write: dtype must be f16 or f32");
    MFQ_RUNTIME_CHECK(positions.is_cuda() && positions.is_contiguous() && positions.scalar_type() == mfq_tensor_backend::kInt64,
                "kv_cache_write: positions must be cuda contiguous int64");
    MFQ_RUNTIME_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "kv_cache_write: all k/v tensors must be 4D");
    MFQ_RUNTIME_CHECK(k.sizes() == v.sizes(), "kv_cache_write: k/v input shapes must match");
    MFQ_RUNTIME_CHECK(k_cache.sizes() == v_cache.sizes(), "kv_cache_write: k/v cache shapes must match");

    int B = (int)k.size(0);
    int H = (int)k.size(1);
    int T = (int)k.size(2);
    int D = (int)k.size(3);
    int max_seq = (int)k_cache.size(2);
    MFQ_RUNTIME_CHECK(k_cache.size(0) == B && k_cache.size(1) == H && k_cache.size(3) == D,
                "kv_cache_write: cache shape must be [B,H,max_seq,D]");

    int pos_dim = positions.dim();
    MFQ_RUNTIME_CHECK(pos_dim == 1 || pos_dim == 2, "kv_cache_write: positions must be [T] or [B,T]");
    if (pos_dim == 1) {
        MFQ_RUNTIME_CHECK(positions.size(0) == T, "kv_cache_write: positions [T] length mismatch");
    } else {
        MFQ_RUNTIME_CHECK(positions.size(0) == B && positions.size(1) == T,
                    "kv_cache_write: positions [B,T] shape mismatch");
    }

    constexpr int BD = 256;
    size_t n = (size_t)B * H * T * D;
    int grid = (int)((n + BD - 1) / BD);
    grid = grid > 4096 ? 4096 : grid;
    MFQ_DISPATCH_FLOATING_TYPES_AND_HALF(k_cache.scalar_type(), "kv_cache_write_cuda", [&] {
        kv_cache_write_kernel<scalar_t><<<grid, BD, 0, mfq_current_cuda_stream()>>>(
            k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), k_cache.data_ptr<scalar_t>(), v_cache.data_ptr<scalar_t>(),
            positions.data_ptr<int64_t>(), B, H, T, D, max_seq, pos_dim);
    });
    return {k_cache, v_cache};
}

template <typename scalar_t>
__global__ void kv_cache_write_ring_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    int64_t position_start,
    int B, int H, int T, int D, int capacity, int source_start)
{
    const int kept = T - source_start;
    const size_t n = (size_t)B * H * kept * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int d = (int)(idx % D);
        const size_t t0 = idx / D;
        const int local_t = (int)(t0 % kept);
        const size_t h0 = t0 / kept;
        const int h = (int)(h0 % H);
        const int b = (int)(h0 / H);
        const int t = source_start + local_t;
        const int slot = (int)((position_start + t) % capacity);
        const size_t src = (((size_t)b * H + h) * T + t) * D + d;
        const size_t dst = (((size_t)b * H + h) * capacity + slot) * D + d;
        k_cache[dst] = k[src];
        v_cache[dst] = v[src];
    }
}

std::vector<mfq_tensor_backend::Tensor> kv_cache_write_ring_cuda(
    mfq_tensor_backend::Tensor k_cache,
    mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor k,
    mfq_tensor_backend::Tensor v,
    int64_t position_start)
{
    MFQ_RUNTIME_CHECK(position_start >= 0, "kv_cache_write_ring: position_start must be nonnegative");
    MFQ_RUNTIME_CHECK(k_cache.is_cuda() && k_cache.is_contiguous() &&
                v_cache.is_cuda() && v_cache.is_contiguous(),
                "kv_cache_write_ring: caches must be cuda contiguous");
    MFQ_RUNTIME_CHECK(k.is_cuda() && k.is_contiguous() && v.is_cuda() && v.is_contiguous(),
                "kv_cache_write_ring: inputs must be cuda contiguous");
    MFQ_RUNTIME_CHECK(k_cache.scalar_type() == v_cache.scalar_type() &&
                k_cache.scalar_type() == k.scalar_type() && k_cache.scalar_type() == v.scalar_type(),
                "kv_cache_write_ring: k/v/cache dtype mismatch");
    MFQ_RUNTIME_CHECK(k_cache.scalar_type() == mfq_tensor_backend::kFloat16 || k_cache.scalar_type() == mfq_tensor_backend::kFloat32,
                "kv_cache_write_ring: dtype must be f16 or f32");
    MFQ_RUNTIME_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "kv_cache_write_ring: all tensors must be rank 4");
    MFQ_RUNTIME_CHECK(k_cache.sizes() == v_cache.sizes() && k.sizes() == v.sizes(),
                "kv_cache_write_ring: k/v shapes must match");

    const int B = (int)k.size(0);
    const int H = (int)k.size(1);
    const int T = (int)k.size(2);
    const int D = (int)k.size(3);
    const int capacity = (int)k_cache.size(2);
    MFQ_RUNTIME_CHECK(capacity > 0 && k_cache.size(0) == B && k_cache.size(1) == H && k_cache.size(3) == D,
                "kv_cache_write_ring: cache shape mismatch");
    if (T == 0) return {k_cache, v_cache};

    // Keeping only the latest capacity entries makes every destination unique,
    // even when one prefill chunk is larger than the physical ring.
    const int source_start = std::max(0, T - capacity);
    const size_t n = (size_t)B * H * (T - source_start) * D;
    constexpr int block = 256;
    const int grid = (int)std::min<size_t>(4096, (n + block - 1) / block);
    MFQ_DISPATCH_FLOATING_TYPES_AND_HALF(k_cache.scalar_type(), "kv_cache_write_ring_cuda", [&] {
        kv_cache_write_ring_kernel<scalar_t><<<grid, block, 0, mfq_current_cuda_stream()>>>(
            k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(), v_cache.data_ptr<scalar_t>(),
            position_start, B, H, T, D, capacity, source_start);
    });
    return {k_cache, v_cache};
}

template <typename scalar_t>
__global__ void kv_cache_write_ring_positions_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    const int64_t* __restrict__ positions,
    int B, int H, int T, int D, int capacity, int source_start)
{
    const int kept = T - source_start;
    const size_t n = (size_t)B * H * kept * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int d = (int)(idx % D);
        const size_t t0 = idx / D;
        const int local_t = (int)(t0 % kept);
        const int t = source_start + local_t;
        const size_t h0 = t0 / kept;
        const int h = (int)(h0 % H);
        const int b = (int)(h0 / H);
        const int slot = (int)(positions[t] % capacity);
        const size_t src = (((size_t)b * H + h) * T + t) * D + d;
        const size_t dst = (((size_t)b * H + h) * capacity + slot) * D + d;
        k_cache[dst] = k[src];
        v_cache[dst] = v[src];
    }
}

std::vector<mfq_tensor_backend::Tensor> kv_cache_write_ring_positions_cuda(
    mfq_tensor_backend::Tensor k_cache,
    mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor k,
    mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor positions)
{
    MFQ_RUNTIME_CHECK(k_cache.is_cuda() && k_cache.is_contiguous() &&
                v_cache.is_cuda() && v_cache.is_contiguous(),
                "kv_cache_write_ring_positions: caches must be cuda contiguous");
    MFQ_RUNTIME_CHECK(k.is_cuda() && k.is_contiguous() && v.is_cuda() && v.is_contiguous(),
                "kv_cache_write_ring_positions: inputs must be cuda contiguous");
    MFQ_RUNTIME_CHECK(positions.is_cuda() && positions.is_contiguous() &&
                positions.scalar_type() == mfq_tensor_backend::kInt64 && positions.dim() == 1,
                "kv_cache_write_ring_positions: positions must be cuda contiguous int64[T]");
    MFQ_RUNTIME_CHECK(k_cache.scalar_type() == v_cache.scalar_type() &&
                k_cache.scalar_type() == k.scalar_type() && k_cache.scalar_type() == v.scalar_type(),
                "kv_cache_write_ring_positions: k/v/cache dtype mismatch");
    MFQ_RUNTIME_CHECK(k_cache.scalar_type() == mfq_tensor_backend::kFloat16 || k_cache.scalar_type() == mfq_tensor_backend::kFloat32,
                "kv_cache_write_ring_positions: dtype must be f16 or f32");
    MFQ_RUNTIME_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "kv_cache_write_ring_positions: all tensors must be rank 4");
    MFQ_RUNTIME_CHECK(k_cache.sizes() == v_cache.sizes() && k.sizes() == v.sizes(),
                "kv_cache_write_ring_positions: k/v shapes must match");

    const int B = (int)k.size(0);
    const int H = (int)k.size(1);
    const int T = (int)k.size(2);
    const int D = (int)k.size(3);
    const int capacity = (int)k_cache.size(2);
    MFQ_RUNTIME_CHECK(capacity > 0 && k_cache.size(0) == B && k_cache.size(1) == H && k_cache.size(3) == D,
                "kv_cache_write_ring_positions: cache shape mismatch");
    MFQ_RUNTIME_CHECK(positions.size(0) == T,
                "kv_cache_write_ring_positions: positions length mismatch");
    if (T == 0) return {k_cache, v_cache};

    const int source_start = std::max(0, T - capacity);
    const size_t n = (size_t)B * H * (T - source_start) * D;
    constexpr int block = 256;
    const int grid = (int)std::min<size_t>(4096, (n + block - 1) / block);
    MFQ_DISPATCH_FLOATING_TYPES_AND_HALF(k_cache.scalar_type(), "kv_cache_write_ring_positions_cuda", [&] {
        kv_cache_write_ring_positions_kernel<scalar_t><<<
            grid, block, 0, mfq_current_cuda_stream()>>>(
            k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(), v_cache.data_ptr<scalar_t>(),
            positions.data_ptr<int64_t>(), B, H, T, D, capacity, source_start);
    });
    return {k_cache, v_cache};
}
