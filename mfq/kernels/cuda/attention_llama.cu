#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>

#define MFQ_LLAMA_FATTN_KERNEL_ONLY
#include "llama_fattn_mma_f16.cuh"

__global__ void mfq_causal_mask_kernel(
    half* mask, int* kv_max, int B, int T, int mask_stride, int tiles,
    int query_tile, int kv_tile)
{
    size_t total = (size_t)T * mask_stride;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (size_t)gridDim.x * blockDim.x) {
        int q = (int)(i / mask_stride);
        int k = (int)(i - (size_t)q * mask_stride);
        mask[i] = k < T && k <= q ? __float2half(0.0f) : __float2half(-INFINITY);
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < B * tiles; i += gridDim.x * blockDim.x) {
        int tile = i % tiles;
        int visible = min(T, (tile + 1) * query_tile);
        kv_max[i] = ((visible + kv_tile - 1) / kv_tile) * kv_tile;
    }
}

__global__ void mfq_swa_causal_mask_kernel(
    half* mask, int* kv_max, int B, int T, int mask_stride, int tiles,
    int window, int query_tile, int kv_tile)
{
    const size_t total = (size_t)T * mask_stride;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int query = (int)(i / mask_stride);
        const int key = (int)(i - (size_t)query * mask_stride);
        const bool visible = key < T && key <= query && query - key < window;
        mask[i] = visible ? __float2half(0.0f) : __float2half(-INFINITY);
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < B * tiles; i += gridDim.x * blockDim.x) {
        const int tile = i % tiles;
        const int visible = min(T, (tile + 1) * query_tile);
        kv_max[i] = ((visible + kv_tile - 1) / kv_tile) * kv_tile;
    }
}

__global__ void mfq_decode_mask_kernel(
    half* mask, int* kv_max, const int64_t* seq_len,
    int B, int mask_stride, int kv_tile)
{
    const int length = max(0, min(mask_stride, (int)seq_len[0]));
    const size_t total = (size_t)B * mask_stride;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int key = (int)(i % mask_stride);
        mask[i] = key < length ? __float2half(0.0f) : __float2half(-INFINITY);
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < B; i += gridDim.x * blockDim.x) {
        kv_max[i] = ((length + kv_tile - 1) / kv_tile) * kv_tile;
    }
}

__global__ void mfq_glm_mla_mask_kernel(
    half * mask, int * kv_max, int B, int M, int logical_k,
    int query_offset, int mask_stride, int query_tiles,
    int query_tile, int kv_tile)
{
    const size_t total = static_cast<size_t>(M) * mask_stride;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int query = static_cast<int>(i / mask_stride);
        const int key = static_cast<int>(i - static_cast<size_t>(query) * mask_stride);
        const bool visible = key < logical_k && key <= query_offset + query;
        mask[i] = visible ? __float2half(0.0f) : __float2half(-INFINITY);
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < B * query_tiles; i += gridDim.x * blockDim.x) {
        const int tile = i % query_tiles;
        const int visible = min(logical_k, query_offset + (tile + 1) * query_tile);
        kv_max[i] = ((visible + kv_tile - 1) / kv_tile) * kv_tile;
    }
}

struct MfqLlamaFattnCache {
    int B = 0;
    int T = 0;
    int mask_stride = 0;
    int query_tile = 0;
    int kv_tile = 0;
    torch::Tensor mask;
    torch::Tensor kv_max;
};

static MfqLlamaFattnCache & mfq_llama_fattn_cache(
    int B, int T, int query_tile, int kv_tile)
{
    static MfqLlamaFattnCache cache;
    if (cache.B != B || cache.T != T || cache.query_tile != query_tile ||
        cache.kv_tile != kv_tile) {
        auto cuda = torch::TensorOptions().device(torch::kCUDA);
        cache.mask_stride = ((T + kv_tile - 1) / kv_tile) * kv_tile;
        cache.mask = torch::empty({T, cache.mask_stride}, cuda.dtype(torch::kFloat16));
        const int tiles = (T + query_tile - 1) / query_tile;
        cache.kv_max = torch::empty({B, tiles}, cuda.dtype(torch::kInt32));
        int blocks = min(65535, (T * cache.mask_stride + 255) / 256);
        mfq_causal_mask_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<half*>(cache.mask.data_ptr<at::Half>()),
            cache.kv_max.data_ptr<int>(), B, T, cache.mask_stride, tiles,
            query_tile, kv_tile);
        cache.B = B;
        cache.T = T;
        cache.query_tile = query_tile;
        cache.kv_tile = kv_tile;
    }
    return cache;
}

static MfqLlamaFattnCache & mfq_llama_swa_fattn_cache(
    int B, int T, int window, int query_tile, int kv_tile)
{
    static MfqLlamaFattnCache cache;
    static int cache_window = 0;
    if (cache.B != B || cache.T != T || cache_window != window ||
        cache.query_tile != query_tile || cache.kv_tile != kv_tile) {
        auto cuda = torch::TensorOptions().device(torch::kCUDA);
        cache.mask_stride = ((T + kv_tile - 1) / kv_tile) * kv_tile;
        cache.mask = torch::empty({T, cache.mask_stride}, cuda.dtype(torch::kFloat16));
        const int tiles = (T + query_tile - 1) / query_tile;
        cache.kv_max = torch::empty({B, tiles}, cuda.dtype(torch::kInt32));
        const int blocks = min(65535, (T * cache.mask_stride + 255) / 256);
        mfq_swa_causal_mask_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<half*>(cache.mask.data_ptr<at::Half>()),
            cache.kv_max.data_ptr<int>(), B, T, cache.mask_stride, tiles,
            window, query_tile, kv_tile);
        cache.B = B;
        cache.T = T;
        cache.query_tile = query_tile;
        cache.kv_tile = kv_tile;
        cache_window = window;
    }
    return cache;
}

template<int DKQ, int DV, int ncols1, int ncols2>
static torch::Tensor mfq_llama_flash_launch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale,
    bool sliding, int window)
{
    constexpr int ncols = ncols1 * ncols2;
    const int B = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int T = (int)q.size(2);
    const int D = (int)q.size(3);
    const int Hk = (int)k.size(1);

    int device = 0;
    cudaDeviceProp properties{};
    TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "llama_flash256: cudaGetDevice failed");
    TORCH_CHECK(cudaGetDeviceProperties(&properties, device) == cudaSuccess,
                "llama_flash256: cudaGetDeviceProperties failed");
    const int cc = properties.major * 100 + properties.minor * 10;
    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols, cc);
    const int nwarps = nthreads / 32;
    const int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, cc);
    const int nbatch_K2 = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols, cc);
    const int nbatch_V2 = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols, cc);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const int nstages = ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, cc);
    const bool q_in_reg = ggml_cuda_fattn_mma_get_Q_in_reg(DKQ, DV, ncols, cc);
    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const size_t shared_kv_1 = nbatch_fa * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t shared_kv_2 = nbatch_fa * (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
    const size_t shared_q = ncols * (DKQ / 2 + 4) * sizeof(half2);
    const size_t shared_mask = ncols1 * (nbatch_fa / 2 + 4) * sizeof(half2);
    const size_t shared_combine = nwarps * cols_per_warp *
        (nbatch_combine + 4) * sizeof(half2);
    const size_t shared_kv = nstages <= 1 ? shared_kv_1 : shared_kv_2;
    const size_t shared_total = std::max(
        shared_combine,
        q_in_reg ? std::max(shared_q, shared_kv + shared_mask)
                 : shared_q + shared_kv + shared_mask);

    auto & cache = sliding
        ? mfq_llama_swa_fattn_cache(B, T, window, ncols1, nbatch_fa)
        : mfq_llama_fattn_cache(B, T, ncols1, nbatch_fa);
    auto out = torch::empty({B, T, Hq, DV}, q.options());
    using Kernel = decltype(&flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, false>);
    Kernel kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, false>;
    static bool shared_limit_set[32] = {};
    TORCH_CHECK(device >= 0 && device < 32, "llama_flash256: unsupported CUDA device index");
    if (!shared_limit_set[device]) {
        const auto status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_total);
        TORCH_CHECK(status == cudaSuccess,
                    "llama_flash256: cudaFuncSetAttribute failed: ", cudaGetErrorString(status));
        shared_limit_set[device] = true;
    }

    const int iter_j = (T + ncols1 - 1) / ncols1;
    const int iter_z_gqa = (Hq / Hk + ncols2 - 1) / ncols2;
    const int ntiles_dst = iter_j * iter_z_gqa * Hk * B;
    const uint3 ne01 = init_fastdiv_values(T);
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)Hq));
    const dim3 block(32, nwarps, 1);
    kernel<<<ntiles_dst, block, shared_total, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const char*>(q.data_ptr<float>()),
        reinterpret_cast<const char*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const char*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const char*>(cache.mask.data_ptr()),
        nullptr, static_cast<int*>(cache.kv_max.data_ptr()), out.data_ptr<float>(), nullptr,
        (float)scale, 0.0f, 1.0f, 1.0f, n_head_log2, 0.0f,
        D, ne01, Hq, B,
        D * (int)sizeof(float), T * D * (int)sizeof(float), Hq * T * D * (int)sizeof(float),
        D, T, Hk, B,
        D * (int)sizeof(half), T * D * (int)sizeof(half), (int64_t)Hk * T * D * (int)sizeof(half),
        D * (int)sizeof(half), T * D * (int)sizeof(half), (int64_t)Hk * T * D * (int)sizeof(half),
        T, 1, 1,
        cache.mask_stride * (int)sizeof(half),
        T * cache.mask_stride * (int)sizeof(half),
        (int64_t)T * cache.mask_stride * (int)sizeof(half));
    const auto status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess,
                "llama_flash256 launch failed: ", cudaGetErrorString(status));
    return out;
}

torch::Tensor attention_llama_flash256_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kFloat32,
                "llama_flash256: q must be contiguous CUDA f32");
    TORCH_CHECK(k.is_cuda() && k.is_contiguous() && k.scalar_type() == torch::kFloat16,
                "llama_flash256: k must be contiguous CUDA f16");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == torch::kFloat16,
                "llama_flash256: v must be contiguous CUDA f16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "llama_flash256: rank must be 4");
    const int B = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int T = (int)q.size(2);
    const int D = (int)q.size(3);
    const int Hk = (int)k.size(1);
    TORCH_CHECK(D == 256 && k.size(3) == D && v.size(3) == D, "llama_flash256: head_dim must be 256");
    TORCH_CHECK(Hq == 4 * Hk && k.size(0) == B && v.size(0) == B &&
                k.size(2) == T && v.size(2) == T && v.size(1) == Hk,
                "llama_flash256: requires self-attention with GQA ratio 4");
    if (T % FATTN_KQ_STRIDE == 0) {
        return mfq_llama_flash_launch<256, 256, 16, 4>(q, k, v, scale, false, 0);
    }
    if (T <= 8) return mfq_llama_flash_launch<256, 256, 8, 1>(q, k, v, scale, false, 0);
    if (T <= 16) return mfq_llama_flash_launch<256, 256, 16, 1>(q, k, v, scale, false, 0);
    if (T <= 32) return mfq_llama_flash_launch<256, 256, 32, 1>(q, k, v, scale, false, 0);
    return mfq_llama_flash_launch<256, 256, 64, 1>(q, k, v, scale, false, 0);
}

torch::Tensor attention_llama_flash512_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kFloat32,
                "llama_flash512: q must be contiguous CUDA f32");
    TORCH_CHECK(k.is_cuda() && k.is_contiguous() && k.scalar_type() == torch::kFloat16,
                "llama_flash512: k must be contiguous CUDA f16");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == torch::kFloat16,
                "llama_flash512: v must be contiguous CUDA f16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "llama_flash512: rank must be 4");
    const int B = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int T = (int)q.size(2);
    const int D = (int)q.size(3);
    const int Hk = (int)k.size(1);
    TORCH_CHECK(D == 512 && k.size(3) == D && v.size(3) == D,
                "llama_flash512: head_dim must be 512");
    TORCH_CHECK(Hq == 8 * Hk && k.size(0) == B && v.size(0) == B &&
                k.size(2) == T && v.size(2) == T && v.size(1) == Hk,
                "llama_flash512: requires self-attention with GQA ratio 8");
    TORCH_CHECK(T % FATTN_KQ_STRIDE == 0,
                "llama_flash512: GQA path requires K length aligned to 64");
    return mfq_llama_flash_launch<512, 512, 8, 8>(q, k, v, scale, false, 0);
}

torch::Tensor attention_glm_mla576_cuda(
    torch::Tensor q, torch::Tensor kv, double scale)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kFloat32,
                "glm_mla576: q must be contiguous CUDA f32");
    TORCH_CHECK(kv.is_cuda() && kv.is_contiguous() && kv.scalar_type() == torch::kFloat16,
                "glm_mla576: kv must be contiguous CUDA f16");
    TORCH_CHECK(q.dim() == 4 && kv.dim() == 4,
                "glm_mla576: expected q[B,64,T,576], kv[B,1,T,576]");
    const int B = (int)q.size(0);
    const int T = (int)q.size(2);
    TORCH_CHECK(q.size(1) == 64 && q.size(3) == 576 &&
                kv.size(0) == B && kv.size(1) == 1 && kv.size(2) == T &&
                kv.size(3) == 576,
                "glm_mla576: unsupported shape");
    if (T == 1) {
        return mfq_llama_flash_launch<576, 512, 1, 16>(q, kv, kv, scale, false, 0);
    }
    if (T == 2) {
        return mfq_llama_flash_launch<576, 512, 2, 16>(q, kv, kv, scale, false, 0);
    }
    return mfq_llama_flash_launch<576, 512, 4, 16>(q, kv, kv, scale, false, 0);
}

template<int ncols1>
static torch::Tensor attention_glm_mla576_cached_impl(
    torch::Tensor q, torch::Tensor kv_cache, int64_t logical_len,
    torch::Tensor mask, torch::Tensor kv_max, torch::Tensor meta,
    double scale)
{
    constexpr int DKQ = 576;
    constexpr int DV = 512;
    constexpr int ncols2 = 16;
    constexpr int ncols = ncols1 * ncols2;
    const int B = static_cast<int>(q.size(0));
    const int M = static_cast<int>(q.size(2));
    const int max_seq = static_cast<int>(kv_cache.size(2));
    const int query_offset = static_cast<int>(logical_len) - M;

    int device = 0;
    cudaDeviceProp properties{};
    TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess,
                "glm_mla576_cached: cudaGetDevice failed");
    TORCH_CHECK(cudaGetDeviceProperties(&properties, device) == cudaSuccess,
                "glm_mla576_cached: cudaGetDeviceProperties failed");
    const int cc = properties.major * 100 + properties.minor * 10;
    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols, cc);
    const int nwarps = nthreads / 32;
    const int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, cc);
    const int nbatch_K2 = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols, cc);
    const int nbatch_V2 = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols, cc);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const int nstages = ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, cc);
    const bool q_in_reg = ggml_cuda_fattn_mma_get_Q_in_reg(DKQ, DV, ncols, cc);
    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const size_t shared_kv_1 = static_cast<size_t>(nbatch_fa) *
        std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t shared_kv_2 = static_cast<size_t>(nbatch_fa) *
        (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
    const size_t shared_q = static_cast<size_t>(ncols) * (DKQ / 2 + 4) * sizeof(half2);
    const size_t shared_mask = static_cast<size_t>(ncols1) * (nbatch_fa / 2 + 4) * sizeof(half2);
    const size_t shared_combine = static_cast<size_t>(nwarps) * cols_per_warp *
        (nbatch_combine + 4) * sizeof(half2);
    const size_t shared_kv = nstages <= 1 ? shared_kv_1 : shared_kv_2;
    const size_t shmem = std::max(
        shared_combine,
        q_in_reg ? std::max(shared_q, shared_kv + shared_mask)
                 : shared_q + shared_kv + shared_mask);

    using Kernel = decltype(&flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, true>);
    Kernel kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, true>;
    static bool shmem_set[32] = {};
    TORCH_CHECK(device >= 0 && device < 32,
                "glm_mla576_cached: unsupported CUDA device index");
    if (!shmem_set[device]) {
        const cudaError_t status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem));
        TORCH_CHECK(status == cudaSuccess,
                    "glm_mla576_cached: shared-memory attribute failed: ",
                    cudaGetErrorString(status));
        shmem_set[device] = true;
    }

    const int iter_j = (M + ncols1 - 1) / ncols1;
    constexpr int iter_z_gqa = 64 / ncols2;
    const int ntiles_dst = B * iter_j * iter_z_gqa;
    const int ntiles_kv = (static_cast<int>(logical_len) + nbatch_fa - 1) / nbatch_fa;
    int max_blocks_per_sm = 0;
    cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, kernel, nthreads, shmem);
    TORCH_CHECK(status == cudaSuccess && max_blocks_per_sm > 0,
                "glm_mla576_cached: occupancy query failed: ",
                cudaGetErrorString(status));
    const int resident_blocks = max_blocks_per_sm * properties.multiProcessorCount;
    const int raw_blocks = std::min(resident_blocks, ntiles_dst * ntiles_kv);
    const int rounded_blocks = std::max(
        ntiles_dst, (raw_blocks / ntiles_dst) * ntiles_dst);
    const int blocks_per_tile = rounded_blocks / ntiles_dst;
    const size_t meta_float2 = static_cast<size_t>(rounded_blocks) *
        ncols * (2 + DV / 2);
    const int mask_stride = ((static_cast<int>(logical_len) + nbatch_fa - 1) /
                             nbatch_fa) * nbatch_fa;
    TORCH_CHECK(mask.dim() == 2 && mask.size(0) >= M && mask.size(1) >= mask_stride &&
                kv_max.numel() >= B * iter_j &&
                (blocks_per_tile == 1 ||
                 static_cast<size_t>(meta.numel()) >= 2 * meta_float2),
                "glm_mla576_cached: workspace too small");

    auto stream = at::cuda::getCurrentCUDAStream();
    const int prep_blocks = std::min(
        65535, std::max(1, (M * mask_stride + 255) / 256));
    mfq_glm_mla_mask_kernel<<<prep_blocks, 256, 0, stream>>>(
        reinterpret_cast<half *>(mask.data_ptr<at::Half>()),
        kv_max.data_ptr<int>(), B, M, static_cast<int>(logical_len),
        query_offset, static_cast<int>(mask.size(1)), iter_j,
        ncols1, nbatch_fa);

    auto out = torch::empty({B, M, 64, DV}, q.options());
    const uint3 ne01 = init_fastdiv_values(M);
    kernel<<<rounded_blocks, dim3(32, nwarps, 1), shmem, stream>>>(
        reinterpret_cast<const char *>(q.data_ptr<float>()),
        reinterpret_cast<const char *>(kv_cache.data_ptr<at::Half>()),
        reinterpret_cast<const char *>(kv_cache.data_ptr<at::Half>()),
        reinterpret_cast<const char *>(mask.data_ptr<at::Half>()),
        nullptr, kv_max.data_ptr<int>(), out.data_ptr<float>(),
        reinterpret_cast<float2 *>(meta.data_ptr<float>()),
        static_cast<float>(scale), 0.0f, 1.0f, 1.0f, 64, 0.0f,
        DKQ, ne01, 64, B,
        DKQ * static_cast<int>(sizeof(float)),
        M * DKQ * static_cast<int>(sizeof(float)),
        static_cast<int64_t>(64) * M * DKQ * sizeof(float),
        DKQ, static_cast<int>(logical_len), 1, B,
        DKQ * static_cast<int>(sizeof(half)),
        max_seq * DKQ * static_cast<int>(sizeof(half)),
        static_cast<int64_t>(max_seq) * DKQ * sizeof(half),
        DKQ * static_cast<int>(sizeof(half)),
        max_seq * DKQ * static_cast<int>(sizeof(half)),
        static_cast<int64_t>(max_seq) * DKQ * sizeof(half),
        M, 1, 1,
        static_cast<int>(mask.size(1)) * static_cast<int>(sizeof(half)),
        M * static_cast<int>(mask.size(1)) * static_cast<int>(sizeof(half)),
        static_cast<int64_t>(M) * mask.size(1) * sizeof(half));
    status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, "glm_mla576_cached launch failed: ",
                cudaGetErrorString(status));

    if (blocks_per_tile > 1) {
        const uint3 fd0 = init_fastdiv_values(iter_j * iter_z_gqa);
        const uint3 fd1 = init_fastdiv_values(iter_j * iter_z_gqa);
        const uint3 fd2 = init_fastdiv_values(iter_j);
        flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>
            <<<dim3(ntiles_dst, ncols1, ncols2), DV, 0, stream>>>(
                out.data_ptr<float>(),
                reinterpret_cast<const float2 *>(meta.data_ptr<float>()),
                M, 64, 1, rounded_blocks, 64, blocks_per_tile,
                fd0, fd1, fd2);
        status = cudaGetLastError();
        TORCH_CHECK(status == cudaSuccess, "glm_mla576_cached fixup failed: ",
                    cudaGetErrorString(status));
    }
    return out;
}

torch::Tensor attention_glm_mla576_cached_cuda(
    torch::Tensor q, torch::Tensor kv_cache, int64_t logical_len,
    torch::Tensor mask, torch::Tensor kv_max, torch::Tensor meta,
    double scale)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kFloat32 &&
                kv_cache.is_cuda() && kv_cache.is_contiguous() &&
                kv_cache.scalar_type() == torch::kFloat16 &&
                mask.is_cuda() && mask.is_contiguous() && mask.scalar_type() == torch::kFloat16 &&
                kv_max.is_cuda() && kv_max.is_contiguous() && kv_max.scalar_type() == torch::kInt32 &&
                meta.is_cuda() && meta.is_contiguous() && meta.scalar_type() == torch::kFloat32,
                "glm_mla576_cached: unsupported dtype or placement");
    TORCH_CHECK(q.dim() == 4 && q.size(1) == 64 && q.size(3) == 576 &&
                kv_cache.dim() == 4 && kv_cache.size(0) == q.size(0) &&
                kv_cache.size(1) == 1 && kv_cache.size(3) == 576 &&
                logical_len >= q.size(2) && logical_len <= kv_cache.size(2),
                "glm_mla576_cached: shape mismatch");
    const int M = static_cast<int>(q.size(2));
    if (M == 1) {
        return attention_glm_mla576_cached_impl<1>(
            q, kv_cache, logical_len, mask, kv_max, meta, scale);
    }
    if (M == 2) {
        return attention_glm_mla576_cached_impl<2>(
            q, kv_cache, logical_len, mask, kv_max, meta, scale);
    }
    return attention_glm_mla576_cached_impl<4>(
        q, kv_cache, logical_len, mask, kv_max, meta, scale);
}

torch::Tensor attention_llama_flash256_swa_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, int64_t window)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kFloat32,
                "llama_flash256_swa: q must be contiguous CUDA f32");
    TORCH_CHECK(k.is_cuda() && k.is_contiguous() && k.scalar_type() == torch::kFloat16,
                "llama_flash256_swa: k must be contiguous CUDA f16");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == torch::kFloat16,
                "llama_flash256_swa: v must be contiguous CUDA f16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "llama_flash256_swa: rank must be 4");
    const int B = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int T = (int)q.size(2);
    const int D = (int)q.size(3);
    const int Hk = (int)k.size(1);
    TORCH_CHECK(D == 256 && k.size(3) == D && v.size(3) == D,
                "llama_flash256_swa: head_dim must be 256");
    TORCH_CHECK(Hq == 2 * Hk && k.size(0) == B && v.size(0) == B &&
                k.size(2) == T && v.size(2) == T && v.size(1) == Hk,
                "llama_flash256_swa: requires self-attention with GQA ratio 2");
    TORCH_CHECK(window > 0 && window <= INT_MAX, "llama_flash256_swa: invalid window");

    if (T % FATTN_KQ_STRIDE == 0) {
        return mfq_llama_flash_launch<256, 256, 32, 2>(q, k, v, scale, true, (int)window);
    }
    if (T <= 8) return mfq_llama_flash_launch<256, 256, 8, 1>(q, k, v, scale, true, (int)window);
    if (T <= 16) return mfq_llama_flash_launch<256, 256, 16, 1>(q, k, v, scale, true, (int)window);
    if (T <= 32) return mfq_llama_flash_launch<256, 256, 32, 1>(q, k, v, scale, true, (int)window);
    return mfq_llama_flash_launch<256, 256, 64, 1>(q, k, v, scale, true, (int)window);
}

template<int DKQ_EXPECTED, int DV_EXPECTED, int NCOLS1, int NCOLS2,
         bool V_IS_K_VIEW = false>
static torch::Tensor attention_llama_flash_decode_impl(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor seq_len, double scale, int64_t planned_len,
    torch::Tensor mask, torch::Tensor kv_max, torch::Tensor meta)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kFloat32,
                "llama_flash_decode: q must be contiguous CUDA f32");
    TORCH_CHECK(k_cache.is_cuda() && k_cache.is_contiguous() &&
                k_cache.scalar_type() == torch::kFloat16,
                "llama_flash_decode: k cache must be contiguous CUDA f16");
    TORCH_CHECK(v_cache.is_cuda() && v_cache.is_contiguous() &&
                v_cache.scalar_type() == torch::kFloat16,
                "llama_flash_decode: v cache must be contiguous CUDA f16");
    TORCH_CHECK(seq_len.is_cuda() && seq_len.is_contiguous() &&
                seq_len.scalar_type() == torch::kInt64 && seq_len.numel() == 1,
                "llama_flash_decode: seq_len must be contiguous CUDA int64[1]");
    TORCH_CHECK(mask.is_cuda() && mask.is_contiguous() && mask.scalar_type() == torch::kFloat16 &&
                mask.dim() == 2,
                "llama_flash_decode: mask must be contiguous CUDA f16[B, stride]");
    TORCH_CHECK(kv_max.is_cuda() && kv_max.is_contiguous() &&
                kv_max.scalar_type() == torch::kInt32,
                "llama_flash_decode: kv_max must be contiguous CUDA int32[B]");
    TORCH_CHECK(meta.is_cuda() && meta.is_contiguous() && meta.scalar_type() == torch::kFloat32,
                "llama_flash_decode: meta must be contiguous CUDA f32");
    TORCH_CHECK(q.dim() == 4 && q.size(2) == 1 && k_cache.dim() == 4 && v_cache.dim() == 4,
                "llama_flash_decode: expected q[B,Hq,1,D], cache[B,Hk,max_seq,D]");

    const int B = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int D = (int)q.size(3);
    const int Hk = (int)k_cache.size(1);
    const int max_seq = (int)k_cache.size(2);
    TORCH_CHECK(D == DKQ_EXPECTED && k_cache.size(3) == DKQ_EXPECTED &&
                v_cache.size(3) ==
                    (V_IS_K_VIEW ? DKQ_EXPECTED : DV_EXPECTED),
                 "llama_flash_decode: unexpected head_dim");
    TORCH_CHECK(Hq % Hk == 0 && Hq / Hk >= NCOLS2 &&
                (Hq / Hk) % NCOLS2 == 0 &&
                k_cache.size(0) == B && v_cache.size(0) == B &&
                 v_cache.size(1) == Hk && v_cache.size(2) == max_seq,
                 "llama_flash_decode: unsupported GQA ratio");
    TORCH_CHECK(planned_len >= 1 && planned_len <= max_seq,
                "llama_flash_decode: planned_len is outside cache capacity");
    const int mask_stride = (int)mask.size(1);
    TORCH_CHECK(mask.size(0) == B && mask_stride >= planned_len && kv_max.numel() >= B,
                "llama_flash_decode: workspace shape mismatch");

    constexpr int ncols1 = NCOLS1;
    constexpr int ncols2 = NCOLS2;
    constexpr int ncols = ncols1 * ncols2;
    int device = 0;
    cudaDeviceProp properties{};
    auto status = cudaGetDevice(&device);
    TORCH_CHECK(status == cudaSuccess, "llama_flash_decode: cudaGetDevice failed: ",
                cudaGetErrorString(status));
    status = cudaGetDeviceProperties(&properties, device);
    TORCH_CHECK(status == cudaSuccess, "llama_flash_decode: cudaGetDeviceProperties failed: ",
                cudaGetErrorString(status));
    const int cc = properties.major * 100 + properties.minor * 10;
    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ_EXPECTED, DV_EXPECTED, ncols, cc);
    const int nwarps = nthreads / 32;
    const int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ_EXPECTED, DV_EXPECTED, ncols, cc);
    const int nbatch_K2 = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ_EXPECTED, DV_EXPECTED, ncols, cc);
    const int nbatch_V2 = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ_EXPECTED, DV_EXPECTED, ncols, cc);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ_EXPECTED, DV_EXPECTED, ncols, cc);
    const int nstages = ggml_cuda_fattn_mma_get_nstages(DKQ_EXPECTED, DV_EXPECTED, ncols1, ncols2, cc);
    const bool q_in_reg = ggml_cuda_fattn_mma_get_Q_in_reg(DKQ_EXPECTED, DV_EXPECTED, ncols, cc);
    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const size_t shared_kv_1 = nbatch_fa * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t shared_kv_2 = nbatch_fa * (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
    const size_t shared_q = ncols * (DKQ_EXPECTED / 2 + 4) * sizeof(half2);
    const size_t shared_mask = ncols1 * (nbatch_fa / 2 + 4) * sizeof(half2);
    const size_t shared_combine = nwarps * cols_per_warp *
        (nbatch_combine + 4) * sizeof(half2);
    const size_t shared_kv = nstages <= 1 ? shared_kv_1 : shared_kv_2;
    const size_t shmem = std::max(
        shared_combine,
        q_in_reg ? std::max(shared_q, shared_kv + shared_mask)
                 : shared_q + shared_kv + shared_mask);

    using Kernel = decltype(&flash_attn_ext_f16<
        DKQ_EXPECTED, DV_EXPECTED, ncols1, ncols2, false, V_IS_K_VIEW>);
    Kernel kernel = flash_attn_ext_f16<
        DKQ_EXPECTED, DV_EXPECTED, ncols1, ncols2, false, V_IS_K_VIEW>;
    static bool shmem_set[32] = {};
    TORCH_CHECK(device >= 0 && device < 32, "llama_flash_decode: unsupported CUDA device index");
    if (!shmem_set[device]) {
        status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        TORCH_CHECK(status == cudaSuccess,
                    "llama_flash_decode: cudaFuncSetAttribute failed: ",
                    cudaGetErrorString(status));
        shmem_set[device] = true;
    }

    const int gqa_ratio = Hq / Hk;
    const int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
    const int ntiles_dst = B * Hk * iter_z_gqa;
    const int ntiles_kv = ((int)planned_len + nbatch_fa - 1) / nbatch_fa;
    int max_blocks_per_sm = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, kernel, 32 * nwarps, shmem);
    TORCH_CHECK(status == cudaSuccess && max_blocks_per_sm > 0,
                "llama_flash_decode: occupancy query failed: ", cudaGetErrorString(status));
    const int resident_blocks = max_blocks_per_sm * properties.multiProcessorCount;
    const int tile_waves = (ntiles_dst + resident_blocks - 1) / resident_blocks;
    const int tile_efficiency = 100 * ntiles_dst / (resident_blocks * tile_waves);
    bool stream_k = cc >= 890 || tile_efficiency < 75;
    const char * stream_k_env = std::getenv("MFQ_LLAMA_FLASH_DECODE_STREAMK");
    if (stream_k_env != nullptr) stream_k = stream_k_env[0] != '0';
    const int raw_blocks = stream_k
        ? min(resident_blocks, ntiles_dst * ntiles_kv)
        : ntiles_dst;
    const int rounded_blocks = max(ntiles_dst, (raw_blocks / ntiles_dst) * ntiles_dst);
    const int blocks_per_tile = rounded_blocks / ntiles_dst;
    const size_t meta_float2 = (size_t)rounded_blocks * ncols * (2 + DV_EXPECTED / 2);
    TORCH_CHECK((size_t)meta.numel() >= 2 * meta_float2,
                "llama_flash_decode: meta workspace too small");

    auto stream = at::cuda::getCurrentCUDAStream();
    const int prep_blocks = min(65535, max(1, (B * mask_stride + 255) / 256));
    mfq_decode_mask_kernel<<<prep_blocks, 256, 0, stream>>>(
        reinterpret_cast<half*>(mask.data_ptr<at::Half>()), kv_max.data_ptr<int>(),
        seq_len.data_ptr<int64_t>(), B, mask_stride, nbatch_fa);

    auto out = torch::empty({B, 1, Hq, DV_EXPECTED}, q.options());
    const uint3 ne01 = init_fastdiv_values(1);
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)Hq));
    kernel<<<rounded_blocks, dim3(32, nwarps, 1), shmem, stream>>>(
        reinterpret_cast<const char*>(q.data_ptr<float>()),
        reinterpret_cast<const char*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<const char*>(v_cache.data_ptr<at::Half>()),
        reinterpret_cast<const char*>(mask.data_ptr<at::Half>()),
        nullptr, kv_max.data_ptr<int>(), out.data_ptr<float>(),
        reinterpret_cast<float2*>(meta.data_ptr<float>()),
        (float)scale, 0.0f, 1.0f, 1.0f, n_head_log2, 0.0f,
        DKQ_EXPECTED, ne01, Hq, B,
        DKQ_EXPECTED * (int)sizeof(float), DKQ_EXPECTED * (int)sizeof(float), Hq * DKQ_EXPECTED * (int)sizeof(float),
        DKQ_EXPECTED, (int)planned_len, Hk, B,
        DKQ_EXPECTED * (int)sizeof(half), max_seq * DKQ_EXPECTED * (int)sizeof(half),
        (int64_t)Hk * max_seq * DKQ_EXPECTED * (int)sizeof(half),
        DKQ_EXPECTED * (int)sizeof(half), max_seq * DKQ_EXPECTED * (int)sizeof(half),
        (int64_t)Hk * max_seq * DKQ_EXPECTED * (int)sizeof(half),
        1, 1, B,
        mask_stride * (int)sizeof(half), mask_stride * (int)sizeof(half),
        (int64_t)mask_stride * (int)sizeof(half));
    status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, "llama_flash_decode launch failed: ",
                cudaGetErrorString(status));

    if (blocks_per_tile > 1) {
        const uint3 fd0 = init_fastdiv_values(Hk * iter_z_gqa);
        const uint3 fd1 = init_fastdiv_values(iter_z_gqa);
        const uint3 fd2 = init_fastdiv_values(1);
        flash_attn_stream_k_fixup_uniform<DV_EXPECTED, ncols1, ncols2>
            <<<dim3(ntiles_dst, ncols1, ncols2), DV_EXPECTED, 0, stream>>>(
                out.data_ptr<float>(), reinterpret_cast<const float2*>(meta.data_ptr<float>()),
                1, Hq, Hk, rounded_blocks, gqa_ratio, blocks_per_tile, fd0, fd1, fd2);
        status = cudaGetLastError();
        TORCH_CHECK(status == cudaSuccess, "llama_flash_decode fixup launch failed: ",
                    cudaGetErrorString(status));
    }
    return out;
}

torch::Tensor attention_llama_flash256_decode_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor seq_len, double scale, int64_t planned_len,
    torch::Tensor mask, torch::Tensor kv_max, torch::Tensor meta)
{
    return attention_llama_flash_decode_impl<256, 256, 1, 8>(
        q, k_cache, v_cache, seq_len, scale, planned_len, mask, kv_max, meta);
}

torch::Tensor attention_llama_flash512_decode_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor seq_len, double scale, int64_t planned_len,
    torch::Tensor mask, torch::Tensor kv_max, torch::Tensor meta)
{
    return attention_llama_flash_decode_impl<512, 512, 1, 8>(
        q, k_cache, v_cache, seq_len, scale, planned_len, mask, kv_max, meta);
}

torch::Tensor attention_glm_mla576_decode_cuda(
    torch::Tensor q, torch::Tensor kv_cache, torch::Tensor seq_len,
    double scale, int64_t planned_len, torch::Tensor mask,
    torch::Tensor kv_max, torch::Tensor meta)
{
    return attention_llama_flash_decode_impl<576, 512, 1, 16, true>(
        q, kv_cache, kv_cache, seq_len, scale, planned_len, mask, kv_max, meta);
}

torch::Tensor attention_llama_flash256_swa_decode_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor seq_len, double scale, int64_t planned_len,
    torch::Tensor mask, torch::Tensor kv_max, torch::Tensor meta)
{
    return attention_llama_flash_decode_impl<256, 256, 4, 2>(
        q, k_cache, v_cache, seq_len, scale, planned_len, mask, kv_max, meta);
}
