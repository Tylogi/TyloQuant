#include "hip/hip_runtime.h"
#include "mfq_hip_compat.h"

#include <hip/hip_fp16.h>
#include "mfq_tensor_backend.h"
#include <hip/hip_runtime.h>


#include <algorithm>
#include <cmath>

#include "mfq_sparse_attn_mma_f16.cuh"

namespace {

constexpr int kMlaDq = 576;
constexpr int kMlaDv = 512;
constexpr int kMlaHeads = 64;
constexpr int kMlaHeadsPerTile = 16;
constexpr int kIndexerHeads = 32;
constexpr int kIndexerDim = 128;
constexpr int kIndexerKeysPerTile = 64;

template<typename T>
__device__ __forceinline__ float glm_load(T value) {
    return static_cast<float>(value);
}

template<>
__device__ __forceinline__ float glm_load<half>(half value) {
    return __half2float(value);
}

template<typename T>
__device__ __forceinline__ T glm_store(float value) {
    return static_cast<T>(value);
}

template<>
__device__ __forceinline__ half glm_store<half>(float value) {
    return __float2half_rn(value);
}

template<typename T>
__global__ void glm_interleaved_rope_kernel(
    const T * __restrict__ x,
    const int64_t * __restrict__ positions,
    const float * __restrict__ cos,
    const float * __restrict__ sin,
    T * __restrict__ out,
    int B, int H, int Tn, int D, int rotary_dim, int table_stride)
{
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = B * H * Tn * (D / 2);
    if (pair >= pairs) return;

    int tmp = pair;
    const int p = tmp % (D / 2);
    tmp /= D / 2;
    const int token = tmp % Tn;
    const int64_t position = positions[token];
    const int base = pair * 2;
    const float x0 = glm_load(x[base]);
    const float x1 = glm_load(x[base + 1]);
    if (2 * p < rotary_dim) {
        const float c = cos[position * table_stride + p];
        const float s = sin[position * table_stride + p];
        out[base] = glm_store<T>(x0 * c - x1 * s);
        out[base + 1] = glm_store<T>(x1 * c + x0 * s);
    } else {
        out[base] = glm_store<T>(x0);
        out[base + 1] = glm_store<T>(x1);
    }
}

__device__ __forceinline__ float glm_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template<typename T,typename Output,bool TokenMajor>
__global__ void dsv4_tail_rope_kernel(const T* x,const int64_t* positions,
        const float* cosine,const float* sine,Output* output,int64_t pairs,int heads,int tokens,int width,
        int rotary_dim,int table_stride,int64_t batch_stride,int64_t head_stride,int64_t token_stride) {
    for(int64_t pair=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;pair<pairs;
            pair+=int64_t(gridDim.x)*blockDim.x) {
        const int column=int(pair%(width/2))*2;
        auto row=pair/(width/2);
        const int token=int(row%tokens);row/=tokens;
        const int head=int(row%heads),batch=int(row/heads);
        const int64_t destination=TokenMajor
            ? ((int64_t(batch)*tokens+token)*heads+head)*width+column : pair*2;
        const auto source=int64_t(batch)*batch_stride+int64_t(head)*head_stride+
            int64_t(token)*token_stride+column;
        const float a=glm_load(x[source]),b=glm_load(x[source+1]);
        if(column>=width-rotary_dim) {
            const auto index=positions[token]*table_stride+(column-width+rotary_dim)/2;
            const float c=cosine[index],s=sine[index];
            output[destination]=glm_store<Output>(a*c-b*s);
            output[destination+1]=glm_store<Output>(b*c+a*s);
        } else {
            output[destination]=glm_store<Output>(a);output[destination+1]=glm_store<Output>(b);
        }
    }
}

__global__ void glm_indexer_layer_norm_kernel(
    const half * __restrict__ x,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    half * __restrict__ out,
    int rows,
    float eps)
{
    constexpr int D = kIndexerDim;
    __shared__ float warp_sum[4];
    __shared__ float warp_sq_sum[4];
    __shared__ float mean;
    __shared__ float inv_std;

    const int row = blockIdx.x;
    const int d = threadIdx.x;
    if (row >= rows || d >= D) return;
    const float value = __half2float(x[static_cast<int64_t>(row) * D + d]);
    float sum = glm_warp_sum(value);
    float sq_sum = glm_warp_sum(value * value);
    const int lane = d & 31;
    const int warp = d >> 5;
    if (lane == 0) {
        warp_sum[warp] = sum;
        warp_sq_sum[warp] = sq_sum;
    }
    __syncthreads();
    if (warp == 0) {
        sum = lane < 4 ? warp_sum[lane] : 0.0f;
        sq_sum = lane < 4 ? warp_sq_sum[lane] : 0.0f;
        sum = glm_warp_sum(sum);
        sq_sum = glm_warp_sum(sq_sum);
        if (lane == 0) {
            mean = sum / static_cast<float>(D);
            const float variance = fmaxf(
                sq_sum / static_cast<float>(D) - mean * mean, 0.0f);
            inv_std = rsqrtf(variance + eps);
        }
    }
    __syncthreads();
    out[static_cast<int64_t>(row) * D + d] = __float2half_rn(
        (value - mean) * inv_std * weight[d] + bias[d]);
}

__global__ void glm_cache_write_kernel(
    half * __restrict__ cache,
    const half * __restrict__ values,
    const int64_t * __restrict__ positions,
    int B, int Tn, int max_seq, int D)
{
    const int64_t total = static_cast<int64_t>(B) * Tn * D;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int d = linear % D;
        const int row = linear / D;
        const int token = row % Tn;
        const int batch = row / Tn;
        const int64_t position = positions[token];
        if (position >= 0 && position < max_seq) {
            cache[(static_cast<int64_t>(batch) * max_seq + position) * D + d] = values[linear];
        }
    }
}

// One block computes 32 index heads by 64 keys. Eight warps map exactly to
// the 2x4 MFMA output tiles; the head-weighted ReLU reduction stays on chip.
__global__ void glm_indexer_scores_kernel(
    const half * __restrict__ q,
    const half * __restrict__ k,
    const float * __restrict__ weights,
    float * __restrict__ out,
    int B, int M, int K, int k_stride, int query_offset,
    const int64_t * __restrict__ seq_lens,
    float score_scale, float weight_scale)
{
#if defined(__HIP_DEVICE_COMPILE__)

    __shared__ half q_tile[kIndexerHeads][kIndexerDim];
    __shared__ half k_tile[kIndexerKeysPerTile][kIndexerDim];
    __shared__ float score_tile[kIndexerHeads][kIndexerKeysPerTile];

    const int query = blockIdx.y;
    const int batch = blockIdx.z;
    const int key_base = blockIdx.x * kIndexerKeysPerTile;
    const int tid = threadIdx.y * 32 + threadIdx.x;

    const half * q_row = q +
        (static_cast<int64_t>(batch) * M + query) * kIndexerHeads * kIndexerDim;
    for (int i = tid; i < kIndexerHeads * kIndexerDim; i += 256) {
        reinterpret_cast<half *>(q_tile)[i] = q_row[i];
    }
    for (int i = tid; i < kIndexerKeysPerTile * kIndexerDim; i += 256) {
        const int key_local = i / kIndexerDim;
        const int dim = i - key_local * kIndexerDim;
        const int key = key_base + key_local;
        reinterpret_cast<half *>(k_tile)[i] = key < K
            ? k[(static_cast<int64_t>(batch) * k_stride + key) * kIndexerDim + dim]
            : __float2half(0.0f);
    }
    __syncthreads();

    const int lane=tid&63, wave=tid>>6;
    const int head_base=(wave/2)*16;
    using F4=float __attribute__((ext_vector_type(4)));
    using H4=_Float16 __attribute__((ext_vector_type(4)));
    // Each wave computes two 16x16 head/key tiles with FP32 accumulation.
    #pragma unroll
    for(int tile=0;tile<2;++tile) {
        const int key_base=(wave%2)*32+tile*16;
        F4 acc{};
        #pragma unroll
        for(int d=0;d<kIndexerDim;d+=16) {
            const auto a=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(
                q_tile[head_base+(lane&15)]+d+(lane>>4)*4));
            const auto b=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(
                k_tile[key_base+(lane&15)]+d+(lane>>4)*4));
            acc=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc,0,0,0);
        }
        #pragma unroll
        for(int j=0;j<4;++j)
            score_tile[head_base+(lane>>4)*4+j][key_base+(lane&15)]=acc[j];
    }
    __syncthreads();

    if (tid < kIndexerKeysPerTile) {
        const int key = key_base + tid;
        if (key < K) {
            float sum = 0.0f;
#pragma unroll
            for (int head = 0; head < kIndexerHeads; ++head) {
                const float dot = score_tile[head][tid] * score_scale;
                sum += fmaxf(dot, 0.0f) *
                    weights[(static_cast<int64_t>(batch) * M + query) * kIndexerHeads + head];
            }
            const int visible_k = seq_lens == nullptr
                ? K : min(K, static_cast<int>(seq_lens[batch]));
            const int absolute_query = seq_lens == nullptr
                ? query_offset + query : visible_k - M + query;
            out[(static_cast<int64_t>(batch) * M + query) * K + key] =
                key < visible_k && key <= absolute_query
                    ? sum * weight_scale : -INFINITY;
        }
    }
#else
    (void)q;
    (void)k;
    (void)weights;
    (void)out;
    (void)B;
    (void)M;
    (void)K;
    (void)k_stride;
    (void)query_offset;
    (void)seq_lens;
    (void)score_scale;
    (void)weight_scale;
#endif
}


} // namespace

mfq_tensor_backend::Tensor glm_interleaved_rope_cuda(
    mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor positions,
    mfq_tensor_backend::Tensor cos, mfq_tensor_backend::Tensor sin, int64_t rotary_dim)
{
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 4,
                "glm_interleaved_rope: x must be contiguous CUDA [B,H,T,D]");
    MFQ_RUNTIME_CHECK((x.scalar_type() == mfq_tensor_backend::kFloat16 || x.scalar_type() == mfq_tensor_backend::kFloat32) &&
                positions.is_cuda() && positions.is_contiguous() &&
                positions.scalar_type() == mfq_tensor_backend::kInt64 &&
                cos.is_cuda() && sin.is_cuda() && cos.is_contiguous() && sin.is_contiguous() &&
                cos.scalar_type() == mfq_tensor_backend::kFloat32 && sin.scalar_type() == mfq_tensor_backend::kFloat32,
                "glm_interleaved_rope: unsupported dtype or placement");
    MFQ_RUNTIME_CHECK(x.size(3) % 2 == 0 && rotary_dim > 0 && rotary_dim <= x.size(3) &&
                rotary_dim % 2 == 0 && positions.numel() == x.size(2) &&
                cos.dim() == 2 && sin.sizes() == cos.sizes() && cos.size(1) >= rotary_dim / 2,
                "glm_interleaved_rope: shape mismatch");
    auto out = mfq_tensor_backend::empty_like(x);
    const int pairs = static_cast<int>(x.numel() / 2);
    const int blocks = std::min(65535, (pairs + 255) / 256);
    auto stream = mfq_current_cuda_stream();
    if (x.scalar_type() == mfq_tensor_backend::kFloat16) {
        glm_interleaved_rope_kernel<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const half *>(x.data_ptr<mfq_half>()),
            positions.data_ptr<int64_t>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
            reinterpret_cast<half *>(out.data_ptr<mfq_half>()),
            static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
            static_cast<int>(x.size(2)), static_cast<int>(x.size(3)),
            static_cast<int>(rotary_dim),
            static_cast<int>(cos.size(1)));
    } else {
        glm_interleaved_rope_kernel<<<blocks, 256, 0, stream>>>(
            x.data_ptr<float>(), positions.data_ptr<int64_t>(),
            cos.data_ptr<float>(), sin.data_ptr<float>(), out.data_ptr<float>(),
            static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
            static_cast<int>(x.size(2)), static_cast<int>(x.size(3)),
            static_cast<int>(rotary_dim),
            static_cast<int>(cos.size(1)));
    }
    const hipError_t status = hipGetLastError();
    MFQ_RUNTIME_CHECK(status == hipSuccess, "glm_interleaved_rope launch failed: ",
                hipGetErrorString(status));
    return out;
}

mfq_tensor_backend::Tensor dsv4_tail_rope_cuda(
    mfq_tensor_backend::Tensor x,mfq_tensor_backend::Tensor positions,
    mfq_tensor_backend::Tensor cosine,mfq_tensor_backend::Tensor sine,int64_t rotary_dim,
    bool float_output,bool token_major) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.dim()==4 && x.stride(3)==1 && x.numel()>0 &&
        (x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32) &&
        positions.device()==x.device() && positions.scalar_type()==tb::kInt64 && positions.is_contiguous() &&
        positions.numel()==x.size(2) && cosine.device()==x.device() && sine.device()==x.device() &&
        cosine.scalar_type()==tb::kFloat32 && sine.scalar_type()==tb::kFloat32 &&
        cosine.is_contiguous() && sine.is_contiguous() && cosine.dim()==2 && sine.sizes()==cosine.sizes() &&
        x.size(3)%2==0 && rotary_dim>0 && rotary_dim<=x.size(3) && rotary_dim%2==0 &&
        cosine.size(1)>=rotary_dim/2,"DeepSeek tail RoPE geometry or dtype mismatch");
    auto shape=x.sizes().vec();
    if(token_major)std::swap(shape[1],shape[2]);
    auto output=tb::empty(shape,x.options().dtype(float_output?tb::kFloat32:x.scalar_type()));
    const auto pairs=x.numel()/2;
    const auto blocks=unsigned(std::min<int64_t>(65535,(pairs+255)/256));
#define MFQ_TAIL_ROPE(TYPE, PTR, OUT, OUTPTR, TOKENMAJOR) \
    dsv4_tail_rope_kernel<TYPE,OUT,TOKENMAJOR><<<blocks,256,0,mfq_current_cuda_stream()>>>( \
        reinterpret_cast<const TYPE*>(x.data_ptr<PTR>()),positions.data_ptr<int64_t>(), \
        cosine.data_ptr<float>(),sine.data_ptr<float>(),reinterpret_cast<OUT*>(output.data_ptr<OUTPTR>()), \
        pairs,int(x.size(1)),int(x.size(2)),int(x.size(3)),int(rotary_dim),int(cosine.size(1)), \
        x.stride(0),x.stride(1),x.stride(2))
#define MFQ_TAIL_LAYOUT(TYPE, PTR, OUT, OUTPTR) \
    if(token_major) {MFQ_TAIL_ROPE(TYPE,PTR,OUT,OUTPTR,true);} \
    else {MFQ_TAIL_ROPE(TYPE,PTR,OUT,OUTPTR,false);}
    if(x.scalar_type()==tb::kFloat16) {
        if(float_output) {MFQ_TAIL_LAYOUT(half,mfq_half,float,float);}
        else {MFQ_TAIL_LAYOUT(half,mfq_half,half,mfq_half);}
    } else {MFQ_TAIL_LAYOUT(float,float,float,float);}
#undef MFQ_TAIL_LAYOUT
#undef MFQ_TAIL_ROPE
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

mfq_tensor_backend::Tensor glm_dsa_indexer_layer_norm_cuda(
    mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias, double eps)
{
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == mfq_tensor_backend::kFloat32 &&
                bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == mfq_tensor_backend::kFloat32,
                "glm_indexer_layer_norm: tensors must be contiguous CUDA f16/f32/f32");
    MFQ_RUNTIME_CHECK(x.dim() >= 1 && x.size(-1) == kIndexerDim &&
                weight.numel() == kIndexerDim && bias.numel() == kIndexerDim,
                "glm_indexer_layer_norm: shape mismatch");
    const int64_t rows64 = x.numel() / kIndexerDim;
    MFQ_RUNTIME_CHECK(rows64 > 0 && rows64 <= INT_MAX,
                "glm_indexer_layer_norm: invalid row count");
    auto out = mfq_tensor_backend::empty_like(x);
    glm_indexer_layer_norm_kernel<<<static_cast<int>(rows64), kIndexerDim, 0,
        mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half *>(x.data_ptr<mfq_half>()),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        reinterpret_cast<half *>(out.data_ptr<mfq_half>()),
        static_cast<int>(rows64), static_cast<float>(eps));
    const hipError_t status = hipGetLastError();
    MFQ_RUNTIME_CHECK(status == hipSuccess, "glm_indexer_layer_norm launch failed: ",
                hipGetErrorString(status));
    return out;
}

mfq_tensor_backend::Tensor glm_dsa_cache_write_cuda(
    mfq_tensor_backend::Tensor cache, mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor positions)
{
    MFQ_RUNTIME_CHECK(cache.is_cuda() && cache.is_contiguous() && cache.scalar_type() == mfq_tensor_backend::kFloat16 &&
                values.is_cuda() && values.is_contiguous() && values.scalar_type() == mfq_tensor_backend::kFloat16 &&
                positions.is_cuda() && positions.is_contiguous() && positions.scalar_type() == mfq_tensor_backend::kInt64,
                "glm_cache_write: tensors must be contiguous CUDA f16/f16/i64");
    MFQ_RUNTIME_CHECK(cache.dim() == 3 && values.dim() == 3 &&
                cache.size(0) == values.size(0) && cache.size(2) == values.size(2) &&
                positions.numel() == values.size(1),
                "glm_cache_write: shape mismatch");
    const int64_t total = values.numel();
    const int blocks = std::min<int64_t>(65535, (total + 255) / 256);
    glm_cache_write_kernel<<<blocks, 256, 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<half *>(cache.data_ptr<mfq_half>()),
        reinterpret_cast<const half *>(values.data_ptr<mfq_half>()),
        positions.data_ptr<int64_t>(), static_cast<int>(values.size(0)),
        static_cast<int>(values.size(1)), static_cast<int>(cache.size(1)),
        static_cast<int>(values.size(2)));
    const hipError_t status = hipGetLastError();
    MFQ_RUNTIME_CHECK(status == hipSuccess, "glm_cache_write launch failed: ",
                hipGetErrorString(status));
    return cache;
}

mfq_tensor_backend::Tensor glm_dsa_indexer_scores_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor weights,
    int64_t query_offset, int64_t logical_k)
{
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == mfq_tensor_backend::kFloat16 &&
                k.is_cuda() && k.is_contiguous() && k.scalar_type() == mfq_tensor_backend::kFloat16 &&
                weights.is_cuda() && weights.is_contiguous() && weights.scalar_type() == mfq_tensor_backend::kFloat32,
                "glm_indexer_scores: tensors must be contiguous CUDA f16/f16/f32");
    MFQ_RUNTIME_CHECK(q.dim() == 4 && k.dim() == 3 && weights.dim() == 3 &&
                q.size(2) == kIndexerHeads && q.size(3) == kIndexerDim &&
                k.size(0) == q.size(0) && k.size(2) == kIndexerDim &&
                weights.size(0) == q.size(0) && weights.size(1) == q.size(1) &&
                weights.size(2) == kIndexerHeads,
                "glm_indexer_scores: shape mismatch");
    const int B = static_cast<int>(q.size(0));
    const int M = static_cast<int>(q.size(1));
    const int k_stride = static_cast<int>(k.size(1));
    const int K = logical_k < 0 ? k_stride : static_cast<int>(logical_k);
    MFQ_RUNTIME_CHECK(K > 0 && K <= k_stride, "glm_indexer_scores: invalid logical K");
    MFQ_RUNTIME_CHECK(query_offset >= 0 && query_offset + M <= K,
                "glm_indexer_scores: invalid causal query offset");
    auto out = mfq_tensor_backend::empty({B, M, K}, weights.options());
    const dim3 grid((K + kIndexerKeysPerTile - 1) / kIndexerKeysPerTile, M, B);
    glm_indexer_scores_kernel<<<grid, dim3(32, 8, 1), 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half *>(q.data_ptr<mfq_half>()),
        reinterpret_cast<const half *>(k.data_ptr<mfq_half>()),
        weights.data_ptr<float>(), out.data_ptr<float>(), B, M, K, k_stride,
        static_cast<int>(query_offset), nullptr, 1.0f / std::sqrt(128.0f),
        1.0f / std::sqrt(32.0f));
    const hipError_t status = hipGetLastError();
    MFQ_RUNTIME_CHECK(status == hipSuccess, "glm_indexer_scores launch failed: ",
                hipGetErrorString(status));
    return out;
}

mfq_tensor_backend::Tensor glm_dsa_indexer_scores_decode_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor k, mfq_tensor_backend::Tensor weights,
    mfq_tensor_backend::Tensor seq_len, int64_t planned_k)
{
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == mfq_tensor_backend::kFloat16 &&
                k.is_cuda() && k.is_contiguous() && k.scalar_type() == mfq_tensor_backend::kFloat16 &&
                weights.is_cuda() && weights.is_contiguous() && weights.scalar_type() == mfq_tensor_backend::kFloat32 &&
                seq_len.is_cuda() && seq_len.is_contiguous() && seq_len.scalar_type() == mfq_tensor_backend::kInt64,
                "glm_indexer_scores_decode: unsupported dtype or placement");
    MFQ_RUNTIME_CHECK(q.dim() == 4 && q.size(1) == 1 &&
                q.size(2) == kIndexerHeads && q.size(3) == kIndexerDim &&
                k.dim() == 3 && k.size(0) == q.size(0) && k.size(2) == kIndexerDim &&
                weights.dim() == 3 && weights.size(0) == q.size(0) &&
                weights.size(1) == 1 && weights.size(2) == kIndexerHeads &&
                seq_len.numel() == q.size(0) && planned_k > 0 && planned_k <= k.size(1),
                "glm_indexer_scores_decode: shape mismatch");
    const int B = static_cast<int>(q.size(0));
    const int K = static_cast<int>(planned_k);
    const int k_stride = static_cast<int>(k.size(1));
    auto out = mfq_tensor_backend::empty({B, 1, K}, weights.options());
    const dim3 grid((K + kIndexerKeysPerTile - 1) / kIndexerKeysPerTile, 1, B);
    glm_indexer_scores_kernel<<<grid, dim3(32, 8, 1), 0,
        mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half *>(q.data_ptr<mfq_half>()),
        reinterpret_cast<const half *>(k.data_ptr<mfq_half>()),
        weights.data_ptr<float>(), out.data_ptr<float>(), B, 1, K, k_stride,
        0, seq_len.data_ptr<int64_t>(), 1.0f / std::sqrt(128.0f),
        1.0f / std::sqrt(32.0f));
    const hipError_t status = hipGetLastError();
    MFQ_RUNTIME_CHECK(status == hipSuccess, "glm_indexer_scores_decode launch failed: ",
                hipGetErrorString(status));
    return out;
}

mfq_tensor_backend::Tensor attention_glm_mla_sparse_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv, mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor meta, double scale)
{
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == mfq_tensor_backend::kFloat32 &&
                kv.is_cuda() && kv.is_contiguous() && kv.scalar_type() == mfq_tensor_backend::kFloat16 &&
                indices.is_cuda() && indices.is_contiguous() && indices.scalar_type() == mfq_tensor_backend::kInt32 &&
                meta.is_cuda() && meta.is_contiguous() && meta.scalar_type() == mfq_tensor_backend::kFloat32,
                "glm_sparse_mla: tensors must be contiguous CUDA f32/f16/i32/f32");
    MFQ_RUNTIME_CHECK(q.dim() == 4 && q.size(1) == kMlaHeads && q.size(3) == kMlaDq &&
                kv.dim() == 3 && kv.size(0) == q.size(0) && kv.size(2) == kMlaDq &&
                indices.dim() == 3 && indices.size(0) == q.size(0) &&
                indices.size(1) == q.size(2) && indices.size(2) > 0 &&
                indices.size(2) % 32 == 0,
                "glm_sparse_mla: shape mismatch");
    auto cache = kv.unsqueeze(1);
    return mfq_sparse_attn_mma::launch<
        kMlaDq, kMlaDv, 1, kMlaHeadsPerTile, true>(
        q, cache, cache, indices,
        mfq_tensor_backend::Tensor{}, mfq_tensor_backend::Tensor{},
        meta, scale, "glm_sparse_mla");
}
