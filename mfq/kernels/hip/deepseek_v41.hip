#include "hip/hip_runtime.h"
#include "mfq_hip_compat.h"
#include "deepseek_v41.h"

#include <hip/hip_fp16.h>
#include "mfq_fp_sim.h"
#include "reduce.cuh"
#include <hip/hip_runtime.h>

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <climits>
#include <limits>

namespace {

template<class Value>
__global__ void normalize_topk_kernel(const int64_t* selected, const Value* values,
        int32_t* output, int columns, int pool_length) {
    __shared__ int keys[512];
    const int row = blockIdx.x, tid = threadIdx.x;
    for (int i = tid; i < 512; i += 256) {
        int key = INT_MAX;
        if (i < columns) {
            const auto index = selected[int64_t(row) * columns + i];
            const float value = static_cast<float>(values[int64_t(row) * columns + i]);
            key = index >= 0 && index < pool_length && value != -INFINITY ? int(index) : pool_length;
        }
        keys[i] = key;
    }
    __syncthreads();
    for (int size = 2; size <= 512; size <<= 1) {
        for (int stride = size >> 1; stride; stride >>= 1) {
            for (int i = tid; i < 512; i += 256) {
                const int other = i ^ stride;
                if (i < other) {
                    const int a = keys[i], b = keys[other];
                    keys[i] = (i & size) ? max(a, b) : min(a, b);
                    keys[other] = (i & size) ? min(a, b) : max(a, b);
                }
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < columns; i += 256)
        output[int64_t(row) * columns + i] = keys[i] < pool_length ? keys[i] : -1;
}

template<bool Projection>
__global__ void hc_prepare_kernel(const half* residual, const float* previous_pre,
                                  const float* norm, float* normalized, half* branch,
                                  int hidden, float epsilon) {
    const auto row = static_cast<int64_t>(blockIdx.x);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const float* pre = previous_pre + row * 4;
    __shared__ float sums[2][8];
    __shared__ float inverse[2];
    float flat_square = 0, branch_square = 0;
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        float mixed = 0;
#pragma unroll
        for (int source = 0; source < 4; ++source) {
            const float value = __half2float(residual[(row * 4 + source) * hidden + col]);
            if constexpr (Projection) flat_square = __fadd_rn(flat_square, __fmul_rn(value, value));
            mixed = __fadd_rn(mixed, __fmul_rn(value, pre[source]));
        }
        branch_square = __fadd_rn(branch_square, __fmul_rn(mixed, mixed));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        flat_square += __shfl_down_sync(0xffffffff, flat_square, offset);
        branch_square += __shfl_down_sync(0xffffffff, branch_square, offset);
    }
    if (!lane) { sums[0][warp] = flat_square; sums[1][warp] = branch_square; }
    __syncthreads();
    if (!warp) {
        flat_square = lane < 8 ? sums[0][lane] : 0;
        branch_square = lane < 8 ? sums[1][lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            flat_square += __shfl_down_sync(0xffffffff, flat_square, offset);
            branch_square += __shfl_down_sync(0xffffffff, branch_square, offset);
        }
        if (!lane) {
            inverse[0] = rsqrtf(flat_square / (4.0f * hidden) + epsilon);
            inverse[1] = rsqrtf(branch_square / hidden + epsilon);
        }
    }
    __syncthreads();
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        float mixed = 0;
#pragma unroll
        for (int source = 0; source < 4; ++source) {
            const auto index = (row * 4 + source) * hidden + col;
            const float value = __half2float(residual[index]);
            if constexpr (Projection) normalized[index] = __fmul_rn(value, inverse[0]);
            mixed = __fadd_rn(mixed, __fmul_rn(value, pre[source]));
        }
        branch[row * hidden + col] = __float2half_rn(
            __fmul_rn(__fmul_rn(mixed, inverse[1]), norm[col]));
    }
}

__device__ __forceinline__ float e4m3(float value) {
    const float bounded = fminf(448.0f, fmaxf(-448.0f, value));
    return mfq_round_e4m3fn(bounded);
}

__device__ __forceinline__ float pow2_ceil(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t exponent = (bits >> 23u) & 0xffu;
    const bool has_mantissa = (bits & 0x7fffffu) != 0u;
    return __uint_as_float(
        (exponent + static_cast<std::uint32_t>(has_mantissa)) << 23u);
}

__device__ __forceinline__ float e2m1(float value, float scale) {
    const float normalized = fminf(6.0f, fmaxf(-6.0f, value / scale));
    const float magnitude = fabsf(normalized);
    float quantized = 0.0f;
    if (magnitude <= 0.25f) quantized = 0.0f;
    else if (magnitude < 0.75f) quantized = 0.5f;
    else if (magnitude <= 1.25f) quantized = 1.0f;
    else if (magnitude < 1.75f) quantized = 1.5f;
    else if (magnitude <= 2.5f) quantized = 2.0f;
    else if (magnitude < 3.5f) quantized = 3.0f;
    else if (magnitude <= 5.0f) quantized = 4.0f;
    else quantized = 6.0f;
    return copysignf(quantized * scale, normalized);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__global__ void mxfp8_sim_kernel(
    const half* __restrict__ input,
    half* __restrict__ output) {
    const int lane = threadIdx.x;
    const std::int64_t offset =
        static_cast<std::int64_t>(blockIdx.x) * 32 + lane;
    const float value = __half2float(input[offset]);
    const float maximum = warp_max(fabsf(value));
    const float scale = pow2_ceil(
        fmaxf(maximum, 448.0f * 0x1p-126f) / 448.0f);
    output[offset] = __float2half_rn(e4m3(value / scale) * scale);
}


// Single-token local KV path. Keep the norm and RoPE half-rounding boundaries;
// the model-defined FP8 simulation writes directly to its original cache slot.
__global__ void kv_norm_rope_store_kernel(const half* input,const float* weight,
        const int64_t* positions,const float* cosine,const float* sine,half* output,
        int table_stride,float epsilon) {
    const int tid=threadIdx.x;
    float square=0.f;
    #pragma unroll
    for(int col=tid;col<512;col+=256) {
        const float value=__half2float(input[col]);square+=value*value;
    }
    square=block_sum<8>(square);
    const float inverse=rsqrtf(square/512.f+epsilon);
    const int col=tid*2;
    float a=__half2float(input[col])*inverse*(weight[col]+0.f);
    float b=__half2float(input[col+1])*inverse*(weight[col+1]+0.f);
    asm volatile("" : "+v"(a), "+v"(b));
    a=__half2float(__float2half_rn(a));b=__half2float(__float2half_rn(b));
    if(col>=448) {
        const int64_t index=positions[0]*table_stride+(col-448)/2;
        const float c=cosine[index],s=sine[index];
        const float ra=a*c-b*s,rb=b*c+a*s;
        a=__half2float(__float2half_rn(ra));b=__half2float(__float2half_rn(rb));
    }
    float maximum=fmaxf(fabsf(a),fabsf(b));
    #pragma unroll
    for(int d=8;d;d>>=1)maximum=fmaxf(maximum,__shfl_down(maximum,d,16));
    maximum=__shfl(maximum,0,16);
    const float scale=pow2_ceil(fmaxf(maximum,448.f*0x1p-126f)/448.f);
    auto qa=__half_as_ushort(__float2half_rn(e4m3(a/scale)*scale));
    auto qb=__half_as_ushort(__float2half_rn(e4m3(b/scale)*scale));
    // The existing simulated-FP8 cache stores quantized zero as positive zero.
    if((qa&0x7fffu)==0)qa=0;
    if((qb&0x7fffu)==0)qb=0;
    output[col]=__ushort_as_half(qa);output[col+1]=__ushort_as_half(qb);
}

template<bool SingleBatch>
__global__ void join_kv_kernel(const uint4* first,const uint4* second,uint4* output,
        int64_t first_vectors,int64_t row_vectors,int64_t first_stride,int64_t second_stride,int64_t total) {
    for(int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;index<total;index+=int64_t(gridDim.x)*blockDim.x) {
        int64_t batch=0,local=index;
        if constexpr(!SingleBatch) {batch=index/row_vectors;local=index-batch*row_vectors;}
        output[index]=local<first_vectors?first[batch*first_stride+local]:
            second[batch*second_stride+local-first_vectors];
    }
}

__global__ void mxfp4_e4m3_scale_sim_kernel(
    const half* __restrict__ input,
    half* __restrict__ output) {
    const int lane = threadIdx.x;
    const std::int64_t offset =
        static_cast<std::int64_t>(blockIdx.x) * 16 +
        static_cast<std::int64_t>(lane < 16 ? lane : 0);
    const float value = lane < 16 ? __half2float(input[offset]) : 0.0f;
    const float maximum = warp_max(fabsf(value));
    const float raw_scale = fmaxf(maximum, 6.0f * 0x1p-9f) / 6.0f;
    const float scale = fmaxf(e4m3(raw_scale), 0x1p-9f);
    if (lane < 16) {
        output[offset] = __float2half_rn(e2m1(value, scale));
    }
}

void require_input(
    const mfq_tensor_backend::Tensor& input,
    std::int64_t group,
    const char* operation) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
            input.scalar_type() == mfq_tensor_backend::kFloat16 &&
            input.dim() >= 1 && input.numel() > 0 &&
            input.size(-1) % group == 0,
        operation,
        ": expected contiguous CUDA f16 with a grouped last dimension");
    MFQ_RUNTIME_CHECK(
        input.numel() / group <= std::numeric_limits<int>::max(),
        operation, ": input is too large");
}

} // namespace

mfq_tensor_backend::Tensor deepseek_v41_normalize_topk_cuda(
        mfq_tensor_backend::Tensor selected, mfq_tensor_backend::Tensor values, int64_t pool_length) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(selected.is_cuda() && selected.is_contiguous() && selected.scalar_type()==tb::kInt64 &&
        values.device()==selected.device() && values.is_contiguous() && values.sizes()==selected.sizes() &&
        (values.scalar_type()==tb::kFloat16 || values.scalar_type()==tb::kFloat32) && selected.dim()>0 &&
        selected.size(-1)>0 && selected.size(-1)<=512 && pool_length>=0 && pool_length<=INT_MAX &&
        selected.numel()/selected.size(-1)<=INT_MAX,"invalid normalized top-k inputs");
    auto result=tb::empty(selected.sizes(),selected.options().dtype(tb::kInt32));
    const auto rows=selected.numel()/selected.size(-1);
    if(rows) {
        if(values.scalar_type()==tb::kFloat16)
            normalize_topk_kernel<<<int(rows),256,0,mfq_current_cuda_stream()>>>(selected.data_ptr<int64_t>(),
                reinterpret_cast<const half*>(values.data_ptr<mfq_half>()),result.data_ptr<int32_t>(),int(selected.size(-1)),int(pool_length));
        else normalize_topk_kernel<<<int(rows),256,0,mfq_current_cuda_stream()>>>(selected.data_ptr<int64_t>(),
                values.data_ptr<float>(),result.data_ptr<int32_t>(),int(selected.size(-1)),int(pool_length));
        MFQ_CUDA_CHECK(hipGetLastError());
    }
    return result;
}

static std::vector<mfq_tensor_backend::Tensor> hc_prepare_impl(
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor previous_pre,
    mfq_tensor_backend::Tensor norm, double epsilon, bool projection) {
    MFQ_RUNTIME_CHECK(residual.is_cuda() && residual.is_contiguous() &&
                          residual.scalar_type() == mfq_tensor_backend::kFloat16 &&
                          residual.dim() == 4 && residual.size(2) == 4 &&
                          residual.size(3) > 0 && residual.size(3) <= INT_MAX,
                      "deepseek_v41_hc_prepare: expected CUDA f16 [B,T,4,H]");
    MFQ_RUNTIME_CHECK(previous_pre.is_cuda() && previous_pre.is_contiguous() &&
                          previous_pre.scalar_type() == mfq_tensor_backend::kFloat32 &&
                          previous_pre.dim() == 3 && previous_pre.size(0) == residual.size(0) &&
                          previous_pre.size(1) == residual.size(1) && previous_pre.size(2) == 4 &&
                          norm.is_cuda() && norm.is_contiguous() &&
                          norm.scalar_type() == mfq_tensor_backend::kFloat32 &&
                          norm.numel() == residual.size(3) && std::isfinite(epsilon) && epsilon > 0,
                      "deepseek_v41_hc_prepare: invalid previous weights, norm, or epsilon");
    const auto rows = residual.size(0) * residual.size(1), hidden = residual.size(3);
    MFQ_RUNTIME_CHECK(rows > 0 && rows <= INT_MAX,
                      "deepseek_v41_hc_prepare: row count is out of range");
    mfq_tensor_backend::Tensor normalized;
    if (projection) normalized = mfq_tensor_backend::empty(
        {residual.size(0), residual.size(1), 4 * hidden}, previous_pre.options());
    auto branch = mfq_tensor_backend::empty(
        {residual.size(0), residual.size(1), hidden}, residual.options());
    if (projection) {
        hc_prepare_kernel<true><<<static_cast<int>(rows), 256, 0, mfq_current_cuda_stream()>>>(
            reinterpret_cast<const half*>(residual.data_ptr<mfq_half>()), previous_pre.data_ptr<float>(),
            norm.data_ptr<float>(), normalized.data_ptr<float>(),
            reinterpret_cast<half*>(branch.data_ptr<mfq_half>()), static_cast<int>(hidden),
            static_cast<float>(epsilon));
    } else {
        hc_prepare_kernel<false><<<static_cast<int>(rows), 256, 0, mfq_current_cuda_stream()>>>(
            reinterpret_cast<const half*>(residual.data_ptr<mfq_half>()), previous_pre.data_ptr<float>(),
            norm.data_ptr<float>(), nullptr, reinterpret_cast<half*>(branch.data_ptr<mfq_half>()),
            static_cast<int>(hidden), static_cast<float>(epsilon));
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return {normalized, branch};
}

std::vector<mfq_tensor_backend::Tensor> deepseek_v41_hc_prepare_cuda(
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor previous_pre,
    mfq_tensor_backend::Tensor norm, double epsilon) {
    return hc_prepare_impl(residual, previous_pre, norm, epsilon, true);
}

mfq_tensor_backend::Tensor deepseek_v41_hc_input_cuda(
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor previous_pre,
    mfq_tensor_backend::Tensor norm, double epsilon) {
    return hc_prepare_impl(residual, previous_pre, norm, epsilon, false).at(1);
}

void deepseek_v41_mxfp8_e4m3_sim_into_cuda(
    mfq_tensor_backend::Tensor input,mfq_tensor_backend::Tensor output) {
    require_input(input, 32, "deepseek_v41_mxfp8_e4m3_sim");
    require_input(output,32,"deepseek_v41_mxfp8_e4m3_sim output");
    MFQ_RUNTIME_CHECK(input.sizes()==output.sizes() && input.device()==output.device(),
        "deepseek_v41_mxfp8_e4m3_sim: output must match input");
    mxfp8_sim_kernel<<<
        static_cast<int>(input.numel() / 32), 32, 0,
        mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<mfq_half>()),
        reinterpret_cast<half*>(output.data_ptr<mfq_half>()));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}


void deepseek_v41_kv_norm_rope_store_cuda(
        mfq_tensor_backend::Tensor input,mfq_tensor_backend::Tensor weight,
        mfq_tensor_backend::Tensor positions,mfq_tensor_backend::Tensor cosine,
        mfq_tensor_backend::Tensor sine,mfq_tensor_backend::Tensor output,double epsilon) {
    namespace tb=mfq_tensor_backend;
    require_input(input,512,"DeepSeek-V4.1 fused KV input");
    require_input(output,512,"DeepSeek-V4.1 fused KV output");
    MFQ_RUNTIME_CHECK(input.numel()==512 && output.numel()==512 &&
        weight.is_cuda() && weight.is_contiguous() && weight.scalar_type()==tb::kFloat32 && weight.numel()==512 &&
        positions.is_cuda() && positions.is_contiguous() && positions.scalar_type()==tb::kInt64 && positions.numel()==1 &&
        cosine.is_cuda() && sine.is_cuda() && cosine.is_contiguous() && sine.is_contiguous() &&
        cosine.scalar_type()==tb::kFloat32 && sine.scalar_type()==tb::kFloat32 &&
        cosine.dim()==2 && cosine.size(1)>=32 && sine.sizes()==cosine.sizes() &&
        input.device()==weight.device() && input.device()==positions.device() &&
        input.device()==cosine.device() && input.device()==sine.device() && input.device()==output.device(),
        "DeepSeek-V4.1 fused single-token KV geometry mismatch");
    kv_norm_rope_store_kernel<<<1,256,0,mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<mfq_half>()),weight.data_ptr<float>(),positions.data_ptr<int64_t>(),
        cosine.data_ptr<float>(),sine.data_ptr<float>(),reinterpret_cast<half*>(output.data_ptr<mfq_half>()),int(cosine.size(1)),float(epsilon));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}

mfq_tensor_backend::Tensor deepseek_v41_mxfp8_e4m3_sim_cuda(
    mfq_tensor_backend::Tensor input) {
    auto output=mfq_tensor_backend::empty_like(input);
    deepseek_v41_mxfp8_e4m3_sim_into_cuda(input,output);
    return output;
}

mfq_tensor_backend::Tensor deepseek_v41_join_kv_cuda(
        mfq_tensor_backend::Tensor first,mfq_tensor_backend::Tensor second) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(first.is_cuda() && second.is_cuda() && first.device()==second.device() &&
        first.scalar_type()==tb::kFloat16 && second.scalar_type()==tb::kFloat16 &&
        first.dim()==3 && second.dim()==3 && first.size(0)==second.size(0) && first.size(2)==second.size(2) &&
        first.size(0)>0 && first.size(1)>0 && second.size(1)>0 && first.size(2)>0 && first.size(2)%8==0 &&
        first.stride(2)==1 && second.stride(2)==1 && first.stride(1)==first.size(2) && second.stride(1)==second.size(2) &&
        first.stride(0)%8==0 && second.stride(0)%8==0 &&
        (reinterpret_cast<uintptr_t>(first.data_ptr())&15)==0 && (reinterpret_cast<uintptr_t>(second.data_ptr())&15)==0 &&
        first.size(1)<=INT64_MAX-second.size(1) && first.numel()<=INT64_MAX-second.numel(),
        "invalid native local/compressed KV join");
    MfqCudaGuard guard(first.device());
    auto output=tb::empty({first.size(0),first.size(1)+second.size(1),first.size(2)},first.options());
    const int64_t first_vectors=first.size(1)*first.size(2)/8;
    const int64_t row_vectors=(first.size(1)+second.size(1))*first.size(2)/8,total=output.numel()/8;
    const int blocks=int(std::min<int64_t>((total+255)/256,4096));
    const auto launch=[&]<bool Single>() {
        join_kv_kernel<Single><<<blocks,256,0,mfq_current_cuda_stream()>>>(
            reinterpret_cast<const uint4*>(first.data_ptr()),reinterpret_cast<const uint4*>(second.data_ptr()),
            reinterpret_cast<uint4*>(output.data_ptr()),first_vectors,row_vectors,first.stride(0)/8,second.stride(0)/8,total);
    };
    if(first.size(0)==1)launch.template operator()<true>();else launch.template operator()<false>();
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

mfq_tensor_backend::Tensor deepseek_v41_mxfp4_e4m3_scale_sim_cuda(
    mfq_tensor_backend::Tensor input) {
    require_input(input, 16, "deepseek_v41_mxfp4_e4m3_scale_sim");
    auto output = mfq_tensor_backend::empty_like(input);
    mxfp4_e4m3_scale_sim_kernel<<<
        static_cast<int>(input.numel() / 16), 32, 0,
        mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<mfq_half>()),
        reinterpret_cast<half*>(output.data_ptr<mfq_half>()));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

namespace {
__global__ void engram_rows_kernel(const uint8_t* packed, const int64_t* inverse,
                                   half* output, int width, int64_t elements) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const int col = int(index % width);
    const auto* row = packed + inverse[index / width] * (width + width / 32);
    const float value=mfq_decode_e4m3fn(row[col]);
    const float scale = ldexpf(1.f, int(row[width + col / 32]) - 127);
    output[index] = __float2half_rn(static_cast<float>(value) * scale);
}

template<class Projected>
__global__ void engram_gate_add_kernel(const half* source, const Projected* projected,
                                       const float* weight, half* output, int hidden,
                                       int streams, float epsilon) {
    const int stream = blockIdx.x % streams;
    const int64_t row = blockIdx.x / streams;
    const int64_t source_offset = (row * streams + stream) * hidden;
    const int64_t key_offset = (row * (streams + 1) + stream) * hidden;
    float xs = 0, ks = 0, dot = 0;
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float x = __half2float(source[source_offset + col]);
        const float k = static_cast<float>(projected[key_offset + col]);
        xs = __fadd_rn(xs, __fmul_rn(x, x));
        ks = __fadd_rn(ks, __fmul_rn(k, k));
        dot = __fadd_rn(dot, __fmul_rn(__fmul_rn(x, k), weight[stream * hidden + col]));
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int shift = 16; shift; shift >>= 1) {
        xs += __shfl_down_sync(0xffffffff, xs, shift);
        ks += __shfl_down_sync(0xffffffff, ks, shift);
        dot += __shfl_down_sync(0xffffffff, dot, shift);
    }
    __shared__ float sums[3][8];
    __shared__ float gate;
    if (!lane) { sums[0][warp] = xs; sums[1][warp] = ks; sums[2][warp] = dot; }
    __syncthreads();
    if (!warp) {
        xs = lane < 8 ? sums[0][lane] : 0;
        ks = lane < 8 ? sums[1][lane] : 0;
        dot = lane < 8 ? sums[2][lane] : 0;
        for (int shift = 16; shift; shift >>= 1) {
            xs += __shfl_down_sync(0xffffffff, xs, shift);
            ks += __shfl_down_sync(0xffffffff, ks, shift);
            dot += __shfl_down_sync(0xffffffff, dot, shift);
        }
        if (!lane) {
            const float inverse = __fmul_rn(rsqrtf(xs / hidden + epsilon), rsqrtf(ks / hidden + epsilon));
            const float score = __fmul_rn(dot, inverse) / sqrtf(float(hidden));
            const float sign = float(score > 0) - float(score < 0);
            const float transformed = sign * sqrtf(fmaxf(fabsf(score), 1.e-6f));
            gate = 1.f / (1.f + expf(-transformed));
        }
    }
    __syncthreads();
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float value = static_cast<float>(projected[(row * (streams + 1) + streams) * hidden + col]);
        output[source_offset + col] = __float2half_rn(__fadd_rn(
            __half2float(source[source_offset + col]), __fmul_rn(gate, value)));
    }
}
}

mfq_tensor_backend::Tensor deepseek_v41_engram_rows_cuda(
    mfq_tensor_backend::Tensor packed, mfq_tensor_backend::Tensor inverse, int64_t width) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(packed.is_cuda() && packed.is_contiguous() && packed.scalar_type() == tb::kUInt8 &&
        packed.dim() == 2 && packed.size(0) > 0 && width > 0 && width <= INT_MAX && width % 32 == 0 &&
        packed.size(1) == width + width / 32 && inverse.is_cuda() && inverse.is_contiguous() &&
        inverse.scalar_type() == tb::kInt64 && inverse.dim() == 1 && inverse.numel() > 0 &&
        packed.device() == inverse.device(), "invalid Engram MXFP8 row packet");
    MFQ_RUNTIME_CHECK(inverse.numel() <= int64_t(INT_MAX) * 256 / width,
                      "Engram row output exceeds launch bounds");
    auto output = tb::empty({inverse.numel(), width}, packed.options().dtype(tb::kFloat16));
    engram_rows_kernel<<<int((output.numel() + 255) / 256), 256, 0, mfq_current_cuda_stream()>>>(
        packed.data_ptr<uint8_t>(), inverse.data_ptr<int64_t>(),
        reinterpret_cast<half*>(output.data_ptr<mfq_half>()), int(width), output.numel());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

mfq_tensor_backend::Tensor deepseek_v41_engram_gate_add_cuda(
    mfq_tensor_backend::Tensor hidden, mfq_tensor_backend::Tensor projected,
    mfq_tensor_backend::Tensor query_key_weight, double epsilon) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(hidden.is_cuda() && hidden.is_contiguous() && hidden.scalar_type() == tb::kFloat16 &&
        hidden.dim() == 4 && hidden.size(2) > 0 && hidden.size(3) > 0 && hidden.size(3) <= INT_MAX &&
        projected.is_cuda() && projected.is_contiguous() && projected.dim() == 3 &&
        (projected.scalar_type() == tb::kFloat16 || projected.scalar_type() == tb::kFloat32) &&
        projected.size(0) == hidden.size(0) && projected.size(1) == hidden.size(1) &&
        projected.size(2) == (hidden.size(2) + 1) * hidden.size(3) &&
        query_key_weight.is_cuda() && query_key_weight.is_contiguous() &&
        query_key_weight.scalar_type() == tb::kFloat32 &&
        query_key_weight.numel() == hidden.size(2) * hidden.size(3) &&
        hidden.device() == projected.device() && hidden.device() == query_key_weight.device() &&
        std::isfinite(epsilon) && epsilon > 0, "invalid Engram gate/add tensors");
    const auto rows = hidden.size(0) * hidden.size(1) * hidden.size(2);
    MFQ_RUNTIME_CHECK(rows > 0 && rows <= INT_MAX, "Engram gate/add launch exceeds bounds");
    auto output = tb::empty_like(hidden);
    if (projected.scalar_type() == tb::kFloat16)
        engram_gate_add_kernel<<<int(rows), 256, 0, mfq_current_cuda_stream()>>>(
            reinterpret_cast<const half*>(hidden.data_ptr<mfq_half>()),
            reinterpret_cast<const half*>(projected.data_ptr<mfq_half>()), query_key_weight.data_ptr<float>(),
            reinterpret_cast<half*>(output.data_ptr<mfq_half>()), int(hidden.size(3)),
            int(hidden.size(2)), float(epsilon));
    else
        engram_gate_add_kernel<<<int(rows), 256, 0, mfq_current_cuda_stream()>>>(
            reinterpret_cast<const half*>(hidden.data_ptr<mfq_half>()), projected.data_ptr<float>(),
            query_key_weight.data_ptr<float>(), reinterpret_cast<half*>(output.data_ptr<mfq_half>()),
            int(hidden.size(3)), int(hidden.size(2)), float(epsilon));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

namespace {
// Split the small FP32 projection across the reduction dimension, as the GR
// projection kernels do. Normalization is a row scalar and is applied by the
// coefficient stage, so no expanded FP32 activation buffer is needed.
__global__ void hc_project_partial_kernel(const half* residual, const float* previous,
    const float* function, float* partials, int hidden, int parts) {
    const int part=blockIdx.x, row=blockIdx.y, tid=threadIdx.x;
    const int column=part*64+tid, width=4*hidden;
    float terms[26]{};
    if(column<width) {
        const float x=__half2float(residual[int64_t(row)*width+column]);
        #pragma unroll
        for(int f=0;f<24;++f)terms[f]=__fmul_rn(x,function[int64_t(f)*width+column]);
        terms[24]=__fmul_rn(x,x);
        if(column<hidden) {
            float mixed=0;
            #pragma unroll
            for(int stream=0;stream<4;++stream)
                mixed=__fadd_rn(mixed,__fmul_rn(previous[row*4+stream],
                    __half2float(residual[(int64_t(row)*4+stream)*hidden+column])));
            terms[25]=__fmul_rn(mixed,mixed);
        }
    }
    #pragma unroll
    for(int f=0;f<26;++f) {
        #pragma unroll
        for(int offset=16;offset;offset>>=1)terms[f]+=__shfl_down_sync(0xffffffff,terms[f],offset);
    }
    __shared__ float reduced[26][2];
    if((tid&31)==0) {
        #pragma unroll
        for(int f=0;f<26;++f)reduced[f][tid>>5]=terms[f];
    }
    __syncthreads();
    if(tid<26)partials[(int64_t(row)*parts+part)*26+tid]=reduced[tid][0]+reduced[tid][1];
}
}

mfq_tensor_backend::Tensor deepseek_v41_hc_project_cuda(mfq_tensor_backend::Tensor residual,
    mfq_tensor_backend::Tensor previous, mfq_tensor_backend::Tensor function) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(residual.is_cuda() && residual.is_contiguous() && residual.scalar_type()==tb::kFloat16 &&
        residual.dim()==4 && residual.size(2)==4 && residual.size(3)>0 && residual.size(3)<=INT_MAX/4 &&
        previous.is_cuda() && previous.is_contiguous() && previous.scalar_type()==tb::kFloat32 &&
        previous.dim()==3 && previous.size(0)==residual.size(0) && previous.size(1)==residual.size(1) &&
        previous.size(2)==4 && function.is_cuda() && function.is_contiguous() &&
        function.scalar_type()==tb::kFloat32 && function.dim()==2 && function.size(0)==24 &&
        function.size(1)==4*residual.size(3) && residual.device()==function.device() &&
        residual.device()==previous.device(), "invalid mHC split projection tensors");
    const auto rows=residual.size(0)*residual.size(1), hidden=residual.size(3), parts=(4*hidden+63)/64;
    MFQ_RUNTIME_CHECK(rows>0 && rows<=65535,"mHC split projection row count exceeds launch bounds");
    auto partials=tb::empty({rows,parts,26},previous.options());
    hc_project_partial_kernel<<<dim3(int(parts),int(rows)),64,0,mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(residual.data_ptr<mfq_half>()),previous.data_ptr<float>(),
        function.data_ptr<float>(),partials.data_ptr<float>(),int(hidden),int(parts));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return partials;
}
