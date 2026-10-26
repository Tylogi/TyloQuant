#include "mfq_tensor_backend.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <climits>
#include <type_traits>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include "tq_projection.h"
#include "tq_vector.h"
#include "tq_float_tokens.h"

namespace {
__global__ __launch_bounds__(256) void tq_gr_down_injection_kernel(ProjectionView weight,
        const uint16_t* injection,const float* input,float* low,float* gates,int streams) {
    const int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
    if(int(blockIdx.x)<streams) {
        float sum=0.f;
        for(int column=tid;column<weight.input_columns;column+=256)
            sum=__fmaf_rn(__uint_as_float(uint32_t(injection[size_t(blockIdx.x)*weight.input_columns+column])<<16),input[column],sum);
        for(int shift=32;shift;shift>>=1)sum+=__shfl_down(sum,shift,64);
        __shared__ float partial[4];
        if(!lane)partial[wave]=sum;
        __syncthreads();
        if(!tid) {
            float total=0.f;for(int i=0;i<4;++i)total=__fadd_rn(total,partial[i]);
            gates[blockIdx.x]=__fmul_rn(2.f,1.f/(1.f+expf(-__fdiv_rn(total,float(streams)))));
        }
        return;
    }
    const int row=int(blockIdx.x)-streams;
    float value[1]{};
    if(weight.columns)mfq_tq_cuda_exact::generic_dispatch<2,true,4,1,16>(weight,row,input,0,1,weight.input_columns,value);
    else mfq_tq_cuda_exact::generic_dispatch<2,false,4,1,16>(weight,row,input,0,1,weight.input_columns,value);
    for(int shift=32;shift;shift>>=1)value[0]+=__shfl_down(value[0],shift,64);
    __shared__ float partial[4];
    if(!lane)partial[wave]=value[0];
    __syncthreads();
    float sum=lane<4?partial[lane]:0.f;
    for(int shift=32;shift;shift>>=1)sum+=__shfl_down(sum,shift,64);
    if(!tid) {
        const float projected=sum+(weight.bias?weight.bias[row]:0.f);
        const float scaled=__fdiv_rn(projected,float(streams));
        low[row]=__fmul_rn(scaled,1.f/(1.f+expf(-scaled)));
    }
}

__global__ void dense_gr_injection_kernel(const uint16_t* weight,const float* input,
        float* output,int width,int rows,float streams) {
    const int row=blockIdx.x,token=blockIdx.y,tid=threadIdx.x,lane=tid&63,wave=tid>>6;
    float sum=0.f;
    for(int column=tid;column<width;column+=256) {
        const float value=__uint_as_float(uint32_t(weight[size_t(row)*width+column])<<16);
        sum=__fmaf_rn(value,input[size_t(token)*width+column],sum);
    }
    for(int shift=32;shift;shift>>=1)sum+=__shfl_down(sum,shift,64);
    __shared__ float partial[4];
    if(!lane)partial[wave]=sum;
    __syncthreads();
    if(!tid) {
        float total=0.f;for(int i=0;i<4;++i)total=__fadd_rn(total,partial[i]);
        const float low=__fdiv_rn(total,streams);
        output[size_t(token)*rows+row]=__fmul_rn(2.f,1.f/(1.f+expf(-low)));
    }
}
}

std::vector<mfq_tensor_backend::Tensor> tq_gr_down_injection_cuda(ProjectionView weight,
        const mfq_tensor_backend::Tensor& injection,const mfq_tensor_backend::Tensor& input,int64_t streams) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type()==tb::kFloat32 &&
        input.numel()==weight.input_columns && input.size(-1)==weight.input_columns &&
        injection.device()==input.device() && injection.is_contiguous() && injection.scalar_type()==tb::kBFloat16 &&
        injection.dim()==2 && injection.size(0)==streams && injection.size(1)==weight.input_columns &&
        streams>0 && streams<=4 && weight.rows>0,"invalid paired TQ GR geometry");
    MfqCudaGuard guard(input.device());auto shape=input.sizes().vec();shape.back()=weight.rows;
    auto low=tb::empty(shape,input.options());shape.back()=streams;auto gates=tb::empty(shape,input.options());
    tq_gr_down_injection_kernel<<<weight.rows+streams,256,0,mfq_current_cuda_stream()>>>(weight,
        reinterpret_cast<const uint16_t*>(injection.data_ptr()),input.data_ptr<float>(),low.data_ptr<float>(),gates.data_ptr<float>(),int(streams));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return {low,gates};
}

mfq_tensor_backend::Tensor dense_gr_injection_cuda(const mfq_tensor_backend::Tensor& weight,
        const mfq_tensor_backend::Tensor& input,int64_t streams) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(weight.is_cuda() && input.device()==weight.device() &&
        weight.is_contiguous() && input.is_contiguous() && weight.scalar_type()==tb::kBFloat16 &&
        input.scalar_type()==tb::kFloat32 && weight.dim()==2 && input.dim()>0 &&
        weight.size(0)>0 && weight.size(0)<=4 && input.size(-1)==weight.size(1) &&
        weight.size(1)>0 && weight.size(1)<=INT_MAX && streams>0 &&
        input.numel()/weight.size(1)>0 && input.numel()/weight.size(1)<=65535,
        "invalid GR dense injection geometry");
    MfqCudaGuard guard(input.device());auto shape=input.sizes().vec();shape.back()=weight.size(0);
    auto output=tb::empty(shape,input.options());
    dense_gr_injection_kernel<<<dim3(weight.size(0),input.numel()/weight.size(1)),256,0,mfq_current_cuda_stream()>>>(
        reinterpret_cast<const uint16_t*>(weight.data_ptr()),input.data_ptr<float>(),output.data_ptr<float>(),
        int(weight.size(1)),int(weight.size(0)),float(streams));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
