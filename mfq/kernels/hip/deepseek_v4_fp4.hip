#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "deepseek_v4_fp4.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

__device__ __forceinline__ float dsv4_pow2_ceil(float value) {
    const uint32_t bits = __float_as_uint(value);
    const int exponent = static_cast<int>((bits >> 23) & 0xffu);
    const bool has_mantissa = (bits & 0x7fffffu) != 0;
    return __uint_as_float(
        static_cast<uint32_t>(
            exponent + static_cast<int>(has_mantissa)) << 23);
}

__device__ __forceinline__ float dsv4_fp4_e2m1_dequant(
    float value,
    float scale)
{
    // scale is a positive power of two. For the rounded F16 input domain,
    // its finite reciprocal and the multiplication are exactly representable.
    const uint32_t exponent = __float_as_uint(scale) >> 23;
    const float inverse = exponent == 255 ? 0.0f :
        __uint_as_float((254u - exponent) << 23);
    const float normalized =
        fminf(6.0f, fmaxf(-6.0f, value * inverse));

    const float magnitude = fabsf(normalized);
    float quantized;
    if (magnitude <= 0.25f) {
        quantized = 0.0f;
    } else if (magnitude < 0.75f) {
        quantized = 0.5f;
    } else if (magnitude <= 1.25f) {
        quantized = 1.0f;
    } else if (magnitude < 1.75f) {
        quantized = 1.5f;
    } else if (magnitude <= 2.5f) {
        quantized = 2.0f;
    } else if (magnitude < 3.5f) {
        quantized = 3.0f;
    } else if (magnitude <= 5.0f) {
        quantized = 4.0f;
    } else {
        quantized = 6.0f;
    }
    return copysignf(quantized * scale, normalized);

}

__global__ void dsv4_fp4_sim_half2_kernel(
    const half* __restrict__ input,half* __restrict__ output,int64_t count) {
    const int64_t at=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)*2;
    if(at>=count)return;
    const float2 values=__half22float2(*reinterpret_cast<const half2*>(input+at));
    float maximum=fmaxf(fabsf(values.x),fabsf(values.y));
    #pragma unroll
    for(int offset=8;offset>0;offset>>=1)
        maximum=fmaxf(maximum,__shfl_xor(maximum,offset,16));
    maximum=fmaxf(maximum,6.0f*0x1p-126f);
    const float scale=dsv4_pow2_ceil(maximum/6.0f);
    *reinterpret_cast<half2*>(output+at)=__halves2half2(
        __float2half_rn(dsv4_fp4_e2m1_dequant(values.x,scale)),
        __float2half_rn(dsv4_fp4_e2m1_dequant(values.y,scale)));
}

template<typename Input, bool SingleBatch>
__global__ void dsv4_index_query_rope_fp4_kernel(
    const Input *__restrict__ input, const int64_t *__restrict__ positions,
    const float *__restrict__ cosine, const float *__restrict__ sine,
    half *__restrict__ output, int64_t pairs, int tokens, int table_stride,
    int64_t batch_stride, int64_t head_stride, int64_t token_stride) {
    const int64_t pair = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;
    const int column = int(pair & 63) * 2;
    const auto row = pair >> 6;
    const int head = int(row & 31);
    const auto token_row = row >> 5;
    const int token = SingleBatch ? int(token_row) : int(token_row % tokens);
    const auto batch = SingleBatch ? int64_t(0) : token_row / tokens;
    const auto source = batch * batch_stride + head * head_stride +
                        token * token_stride + column;
    float2 value;
    if constexpr (std::is_same_v<Input, half>) {
        value = __half22float2(*reinterpret_cast<const half2 *>(input + source));
    } else {
        value = *reinterpret_cast<const float2 *>(input + source);
    }
    if (column >= 64) {
        const auto index = positions[token] * table_stride + (column - 64) / 2;
        const float c = cosine[index], s = sine[index];
        value = make_float2(value.x * c - value.y * s, value.y * c + value.x * s);
    }
    // The separate RoPE output / cast rounds to F16 before the group maximum.
    const half2 rounded = __floats2half2_rn(value.x, value.y);
    value = __half22float2(rounded);
    float maximum = fmaxf(fabsf(value.x), fabsf(value.y));
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 16));
    }
    const float scale = dsv4_pow2_ceil(fmaxf(maximum, 6.0f * 0x1p-126f) / 6.0f);
    *reinterpret_cast<half2 *>(output + pair * 2) = __halves2half2(
        __float2half_rn(dsv4_fp4_e2m1_dequant(value.x, scale)),
        __float2half_rn(dsv4_fp4_e2m1_dequant(value.y, scale)));
}

} // namespace

void mfq_dsv4_fp4_half2_launch(
    const half *input, half *output, int64_t count, hipStream_t stream) {
    dsv4_fp4_sim_half2_kernel<<<
        static_cast<unsigned>((count + 511) / 512), 256, 0, stream>>>(
        input, output, count);
}

mfq_tensor_backend::Tensor dsv4_index_query_rope_fp4_cuda(
    mfq_tensor_backend::Tensor input, mfq_tensor_backend::Tensor positions,
    mfq_tensor_backend::Tensor cosine, mfq_tensor_backend::Tensor sine) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.dim() == 4 && input.size(1) == 32 &&
        input.size(3) == 128 && input.stride(3) == 1 && input.numel() > 0 &&
        input.size(2) <= std::numeric_limits<int>::max() &&
        input.numel() / 32 <= std::numeric_limits<int>::max() &&
        input.stride(0) % 2 == 0 && input.stride(1) % 2 == 0 &&
        input.stride(2) % 2 == 0 &&
        (input.scalar_type() == tb::kFloat16 || input.scalar_type() == tb::kFloat32) &&
        reinterpret_cast<std::uintptr_t>(input.data_ptr()) %
            (input.scalar_type() == tb::kFloat16 ? 4 : 8) == 0 &&
        positions.device() == input.device() && positions.is_contiguous() &&
        positions.scalar_type() == tb::kInt64 && positions.numel() == input.size(2) &&
        cosine.device() == input.device() && sine.device() == input.device() &&
        cosine.scalar_type() == tb::kFloat32 && sine.scalar_type() == tb::kFloat32 &&
        cosine.is_contiguous() && sine.is_contiguous() && cosine.dim() == 2 &&
        sine.sizes() == cosine.sizes() && cosine.size(1) >= 32,
        "DeepSeek index-query RoPE/FP4 expects [B,32,T,128] f16/f32 and f32 tables");
    auto output = tb::empty({input.size(0), input.size(2), 32, 128},
                            input.options().dtype(tb::kFloat16));
    const auto pairs = input.numel() / 2;
    const auto blocks = static_cast<unsigned>((pairs + 255) / 256);
#define MFQ_INDEX_QUERY_FP4(TYPE, STORAGE, SINGLE_BATCH) \
    dsv4_index_query_rope_fp4_kernel<TYPE, SINGLE_BATCH><<<blocks, 256, 0, mfq_current_cuda_stream()>>>( \
        reinterpret_cast<const TYPE *>(input.data_ptr<STORAGE>()), \
        positions.data_ptr<int64_t>(), cosine.data_ptr<float>(), sine.data_ptr<float>(), \
        reinterpret_cast<half *>(output.data_ptr<mfq_half>()), pairs, \
        int(input.size(2)), int(cosine.size(1)), input.stride(0), input.stride(1), input.stride(2))
    if (input.scalar_type() == tb::kFloat16) {
        if (input.size(0) == 1) { MFQ_INDEX_QUERY_FP4(half, mfq_half, true); }
        else { MFQ_INDEX_QUERY_FP4(half, mfq_half, false); }
    } else {
        if (input.size(0) == 1) { MFQ_INDEX_QUERY_FP4(float, float, true); }
        else { MFQ_INDEX_QUERY_FP4(float, float, false); }
    }
#undef MFQ_INDEX_QUERY_FP4
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
