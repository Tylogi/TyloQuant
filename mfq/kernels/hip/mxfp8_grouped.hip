#include "mfq_tensor_backend.h"
#include "mfq_fp_sim.h"
#include "mfq_hip_quant_ops.h"
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <climits>
#include <cstring>
#include <type_traits>

namespace {
using F4=float __attribute__((ext_vector_type(4)));
using H4=_Float16 __attribute__((ext_vector_type(4)));
constexpr int BM=32,BN=64,BK=32,Stride=40;

__device__ float scale_value(uint8_t raw) {
    return __uint_as_float(raw?uint32_t(raw)<<23:0x00400000u);
}

__device__ __forceinline__ H4 decode_weight_four(uint32_t packed, float scale) {
    H4 result;
#if defined(__gfx942__)
    const auto lo=__builtin_amdgcn_cvt_pk_f32_fp8(packed,false);
    const auto hi=__builtin_amdgcn_cvt_pk_f32_fp8(packed,true);
#endif
#pragma unroll
    for(int j=0;j<4;++j) {
        const uint8_t code=uint8_t(packed>>(j*8));
#if defined(__gfx942__)
        float value=(j<2?lo[j&1]:hi[j&1])*2.f;
        if(code==128u)value=0.f;
        if((code&127u)==127u)value=__int_as_float(0x7fffffff);
#else
        float value=mfq_decode_e4m3fn(code);
#endif
        result[j]=__builtin_bit_cast(_Float16,__float2half_rn(__fmul_rn(value,scale)));
    }
    return result;
}

// A complete 64x64 output tile. K partitions expose more work for short PP
// without constructing a full FP16 weight matrix.
__global__ __launch_bounds__(256) void packed_mxfp8_f32_kernel(
        const uint8_t* values,const uint8_t* scales,const half* input,float* output,
        int rows,int outputs,int width,int row_shift,int column_shift,int splits) {
    __shared__ __align__(16) half weights[64*64],activation[64*64];
    const int tid=threadIdx.x,lane=tid&63,wave=tid>>6,part=blockIdx.z;
    const int m0=blockIdx.y*64,n0=blockIdx.x*64;
    output+=size_t(part)*rows*outputs;
    F4 accum[4]{};
    const int span=((width+splits*64-1)/(splits*64))*64;
    const int begin=part*span,end=min(width,begin+span);
    for(int kb=begin;kb<end;kb+=64) {
#pragma unroll
        for(int i=tid;i<64*8;i+=256) {
            const int row=i/8,col=(i%8)*8,mask=(row&15)*4;
            const uint4 v=*reinterpret_cast<const uint4*>(input+size_t(m0+row)*width+kb+col);
            *reinterpret_cast<uint2*>(activation+row*64+(col^mask))=make_uint2(v.x,v.y);
            *reinterpret_cast<uint2*>(activation+row*64+((col+4)^mask))=make_uint2(v.z,v.w);
        }
#pragma unroll
        for(int i=tid;i<64*8;i+=256) {
            const int row=i/8,col=(i%8)*8,n=n0+row;
            const uint8_t code=scales[size_t(n>>row_shift)*(width>>column_shift)+((kb+col)>>column_shift)];
            const float scale=code==255u?__int_as_float(0x7fffffff):
                __uint_as_float(code?uint32_t(code)<<23:0x00400000u);
            const uint2 v=*reinterpret_cast<const uint2*>(values+size_t(n)*width+kb+col);
            *reinterpret_cast<uint2*>(weights+row*64+(col^((row&15)*4)))=
                __builtin_bit_cast(uint2,decode_weight_four(v.x,scale));
            *reinterpret_cast<uint2*>(weights+row*64+((col+4)^((row&15)*4)))=
                __builtin_bit_cast(uint2,decode_weight_four(v.y,scale));
        }
        __syncthreads();
#pragma unroll
        for(int d=0;d<64;d+=16) {
            const auto a=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(
                weights+(wave*16+(lane&15))*64+((d+(lane>>4)*4)^((lane&15)*4))));
            H4 x[4];
#pragma unroll
            for(int m=0;m<4;++m)x[m]=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(
                activation+(m*16+(lane&15))*64+((d+(lane>>4)*4)^((lane&15)*4))));
            asm volatile(
                "v_mfma_f32_16x16x16_f16 %0, %4, %5, %0\n\t"
                "v_mfma_f32_16x16x16_f16 %1, %4, %6, %1\n\t"
                "v_mfma_f32_16x16x16_f16 %2, %4, %7, %2\n\t"
                "v_mfma_f32_16x16x16_f16 %3, %4, %8, %3"
                : "+v"(accum[0]),"+v"(accum[1]),"+v"(accum[2]),"+v"(accum[3])
                : "v"(a),"v"(x[0]),"v"(x[1]),"v"(x[2]),"v"(x[3]));
        }
        __syncthreads();
    }
#pragma unroll
    for(int m=0;m<4;++m) {
        const int row=m0+m*16+(lane&15),col=n0+wave*16+(lane>>4)*4;
        *reinterpret_cast<float4*>(output+size_t(row)*outputs+col)=__builtin_bit_cast(float4,accum[m]);
    }
}

template<int Splits,class Output>
__global__ void reduce_mxfp8_vector(const float* partial,Output* output,size_t count) {
    const size_t i=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*4;
    if(i>=count)return;
    float4 sum=make_float4(0.f,0.f,0.f,0.f);
#pragma unroll
    for(int part=0;part<Splits;++part) {
        const float4 v=*reinterpret_cast<const float4*>(partial+size_t(part)*count+i);
        sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;
    }
    if constexpr(std::is_same_v<Output,float>) {
        *reinterpret_cast<float4*>(output+i)=sum;
    } else {
        H4 h;
        h[0]=__builtin_bit_cast(_Float16,__float2half_rn(sum.x));
        h[1]=__builtin_bit_cast(_Float16,__float2half_rn(sum.y));
        h[2]=__builtin_bit_cast(_Float16,__float2half_rn(sum.z));
        h[3]=__builtin_bit_cast(_Float16,__float2half_rn(sum.w));
        *reinterpret_cast<uint2*>(output+i)=__builtin_bit_cast(uint2,h);
    }
}

template<class Output>
void sum_parts(const mfq_tensor_backend::Tensor& partial,
        mfq_tensor_backend::Tensor& output,int splits) {
    const size_t count=size_t(output.numel());
    const auto launch=[&]<int Parts>() {
        reduce_mxfp8_vector<Parts,Output><<<(count+1023)/1024,256,0,mfq_current_cuda_stream()>>>(
            partial.data_ptr<float>(),reinterpret_cast<Output*>(output.data_ptr()),count);
    };
    switch(splits) {
        case 2:launch.template operator()<2>();break;
        case 4:launch.template operator()<4>();break;
        case 8:launch.template operator()<8>();break;
        case 16:launch.template operator()<16>();break;
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}

template<class Output>
__global__ __launch_bounds__(256) void grouped_mxfp8_kernel(const uint8_t* values,const uint8_t* scales,
        const half* input,Output* output,int rows,int outputs,int width,int groups,int per_group,
        int scale_row_shift,int scale_column_shift,int splits) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ half weights[BN*Stride],activation[BM*Stride];
    const int tid=threadIdx.x,lane=tid&63,wave=tid>>6,group=blockIdx.z/splits,part=blockIdx.z%splits;
    output+=size_t(part)*rows*outputs;
    const int m0=blockIdx.y*BM,n0=group*per_group+blockIdx.x*BN;
    const int nend=(group+1)*per_group,scale_columns=width>>scale_column_shift;
    F4 accum[BM/16]{};
    const int span=((width+splits*BK-1)/(splits*BK))*BK;
    const int begin=part*span,end=min(width,begin+span);
    for(int kb=begin;kb<end;kb+=BK) {
        for(int i=tid;i<BM*BK;i+=256) {
            const int m=i/BK,k=i%BK;
            activation[m*Stride+k]=m0+m<rows && kb+k<width?
                input[(size_t(m0+m)*groups+group)*width+kb+k]:__float2half_rn(0.f);
        }
        for(int i=tid;i<BN*BK;i+=256) {
            const int n=n0+i/BK,k=kb+i%BK;float value=0.f;
            if(n<nend && k<width) {
                const float scale=scale_value(scales[size_t(n>>scale_row_shift)*scale_columns+(k>>scale_column_shift)]);
                value=mfq_decode_e4m3fn(values[size_t(n)*width+k])*scale;
            }
            weights[(i/BK)*Stride+i%BK]=__float2half_rn(value);
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<BK;k+=16) {
            const auto a=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(weights+(wave*16+(lane&15))*Stride+k+(lane>>4)*4));
            #pragma unroll
            for(int mt=0;mt<BM/16;++mt) {
                const auto b=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(activation+(mt*16+(lane&15))*Stride+k+(lane>>4)*4));
                accum[mt]=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b,accum[mt],0,0,0);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int mt=0;mt<BM/16;++mt) {
        const int m=m0+mt*16+(lane&15);
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int n=n0+wave*16+(lane>>4)*4+j;
            if(m<rows && n<nend)output[size_t(m)*outputs+n]=static_cast<Output>(accum[mt][j]);
        }
    }
#endif
}

template<class Output>
__global__ void reduce_mxfp8_parts(const float* input,Output* output,size_t count,int splits) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=count)return;
    float sum=0.f;
    for(int part=0;part<splits;++part)sum+=input[size_t(part)*count+index];
    output[index]=static_cast<Output>(sum);
}

template<class Output>
mfq_tensor_backend::Tensor grouped(const mfq_tensor_backend::Tensor& values,
        const mfq_tensor_backend::Tensor& scales,const mfq_tensor_backend::Tensor& input,int64_t groups) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(values.is_cuda() && scales.device()==values.device() && input.device()==values.device() &&
        values.is_contiguous() && scales.is_contiguous() && input.is_contiguous() &&
        values.scalar_type()==tb::kUInt8 && scales.scalar_type()==tb::kUInt8 && input.scalar_type()==tb::kFloat16 &&
        values.dim()==2 && scales.dim()==2 && input.dim()==3 && values.size(0)>0 && values.size(0)<=INT_MAX &&
        values.size(1)>0 && values.size(1)<=INT_MAX && values.size(1)%32==0 && groups>0 && groups<=65535 &&
        input.size(1)==groups && input.size(0)>0 && input.size(0)<=BM*65535 && values.size(0)%groups==0 &&
        input.size(2)==values.size(1),"invalid grouped MXFP8 GEMM tensors");
    const auto outputs=values.size(0),width=values.size(1),per_group=outputs/groups;
    const bool block128=width%128==0 && scales.size(0)==(outputs+127)/128 && scales.size(1)==width/128;
    const bool block32=scales.size(0)==(outputs+31)/32 && scales.size(1)==width/32;
    const bool row32=scales.size(0)==outputs && scales.size(1)==width/32;
    MFQ_RUNTIME_CHECK(block128 || block32 || row32,"invalid grouped MXFP8 scale layout");
    MfqCudaGuard guard(input.device());hipDeviceProp_t properties{};
    MFQ_CUDA_CHECK(hipGetDeviceProperties(&properties,input.get_device()));
    MFQ_RUNTIME_CHECK(properties.warpSize==64,"grouped MXFP8 MFMA requires wave64");
    auto result=tb::empty({input.size(0),outputs},input.options().dtype(std::is_same_v<Output,float>?tb::kFloat32:tb::kFloat16));
    const bool packed_eligible=groups==1 &&
        std::strncmp(properties.gcnArchName,"gfx942",6)==0 &&
        outputs%64==0 && width%64==0 &&
        (reinterpret_cast<uintptr_t>(values.data_ptr<uint8_t>())&7u)==0 &&
        (reinterpret_cast<uintptr_t>(input.data_ptr<mfq_half>())&15u)==0;
    if(packed_eligible && input.size(0)>=64 && input.size(0)<=256 &&
            input.size(0)%64==0 && outputs<=(std::is_same_v<Output,float>?8192:2048) &&
            (std::is_same_v<Output,float> || input.size(0)<=128 || outputs<=512)) {
        const int tiles=int(outputs/64)*(int(input.size(0))/64);
        int splits=1;
        while(tiles*splits<properties.multiProcessorCount*2 && splits<16 &&
                width/(splits*2)>=64)splits*=2;
        if constexpr(std::is_same_v<Output,half>)splits=std::max(splits,2);
        auto target=splits==1?result:
            tb::empty({splits,input.size(0),outputs},input.options().dtype(tb::kFloat32));
        packed_mxfp8_f32_kernel<<<dim3(outputs/64,input.size(0)/64,splits),256,0,mfq_current_cuda_stream()>>>(
            values.data_ptr<uint8_t>(),scales.data_ptr<uint8_t>(),
            reinterpret_cast<const half*>(input.data_ptr<mfq_half>()),
            reinterpret_cast<float*>(target.data_ptr()),int(input.size(0)),int(outputs),int(width),
            row32?0:block128?7:5,block128?7:5,splits);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        if(splits!=1)sum_parts<Output>(target,result,splits);
        return result;
    }
    if((input.size(0)>=128 || (input.size(0)>=32 && outputs>=8192)) &&
       width%128==0 && groups*width<=INT_MAX) {
        // Expand once per invocation, then use the full MFMA GEMM schedule.
        // Batch strides address the existing grouped inputs and outputs directly.
        const auto dense=mxfp8_dequant_cuda(values,scales);
        const auto handle=mfq_current_cublas_handle();
        MFQ_RUNTIME_CHECK(hipblasSetStream(handle,mfq_current_cuda_stream())==HIPBLAS_STATUS_SUCCESS,
            "MXFP8 GEMM stream setup failed");
        const float alpha=1.f,beta=0.f;
        if(packed_eligible && outputs<=2048 && width>=2048 &&
                (std::is_same_v<Output,float>?
                    input.size(0)>=512 && input.size(0)<=4096:
                    input.size(0)==2048)) {
            // Narrow FP32-output projections otherwise use very few 256x256
            // library tiles. Bound K partitions to keep their temporary small.
            const int tiles=int((outputs+255)/256)*int((input.size(0)+255)/256);
            int splits=1;
            while(tiles*splits<128 && splits<8 && width%((splits*2)*32)==0)splits*=2;
            if(splits>1) {
                auto partial=tb::empty({splits,input.size(0),outputs},input.options().dtype(tb::kFloat32));
                const int span=int(width)/splits;
                const auto code=hipblasGemmStridedBatchedEx(handle,HIPBLAS_OP_T,HIPBLAS_OP_N,
                    int(outputs),int(input.size(0)),span,&alpha,
                    dense.data_ptr<mfq_half>(),HIP_R_16F,int(width),span,
                    input.data_ptr<mfq_half>(),HIP_R_16F,int(width),span,&beta,
                    partial.data_ptr(),HIP_R_32F,int(outputs),outputs*input.size(0),splits,
                    HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT);
                MFQ_RUNTIME_CHECK(code==HIPBLAS_STATUS_SUCCESS,"MXFP8 partitioned GEMM failed: ",int(code));
                sum_parts<Output>(partial,result,splits);
                return result;
            }
        }
        const auto code=hipblasGemmStridedBatchedEx(handle,HIPBLAS_OP_T,HIPBLAS_OP_N,
            int(per_group),int(input.size(0)),int(width),&alpha,
            dense.data_ptr<mfq_half>(),HIP_R_16F,int(width),per_group*width,
            input.data_ptr<mfq_half>(),HIP_R_16F,int(groups*width),width,&beta,
            result.data_ptr(),std::is_same_v<Output,float>?HIP_R_32F:HIP_R_16F,
            int(outputs),per_group,int(groups),HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT);
        MFQ_RUNTIME_CHECK(code==HIPBLAS_STATUS_SUCCESS,"MXFP8 grouped GEMM failed: ",int(code));
        return result;
    }
    const int tiles=((per_group+BN-1)/BN)*((input.size(0)+BM-1)/BM)*groups;
    int splits=1;
    while(tiles*splits<properties.multiProcessorCount && splits<16 && width/(splits*2)>=BK && groups*splits*2<=65535)splits*=2;
    const auto launch=[&]<class Target>(Target* target) {
    grouped_mxfp8_kernel<Target><<<dim3((per_group+BN-1)/BN,(input.size(0)+BM-1)/BM,groups*splits),256,0,mfq_current_cuda_stream()>>>(
        values.data_ptr<uint8_t>(),scales.data_ptr<uint8_t>(),reinterpret_cast<const half*>(input.data_ptr<mfq_half>()),
        target,int(input.size(0)),int(outputs),int(width),int(groups),int(per_group),
        row32?0:block128?7:5,block128?7:5,splits);
    };
    if(splits==1)launch(reinterpret_cast<Output*>(result.data_ptr()));
    else {
        auto partial=tb::empty({splits,input.size(0),outputs},input.options().dtype(tb::kFloat32));
        launch(partial.data_ptr<float>());
        const size_t count=size_t(input.size(0))*outputs;
        reduce_mxfp8_parts<<<(count+255)/256,256,0,mfq_current_cuda_stream()>>>(
            partial.data_ptr<float>(),reinterpret_cast<Output*>(result.data_ptr()),count,splits);
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
}

mfq_tensor_backend::Tensor mxfp8_groupwise_gemm_f16_cuda(mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,mfq_tensor_backend::Tensor input,int64_t groups) {
    return grouped<half>(values,scales,input,groups);
}
mfq_tensor_backend::Tensor mxfp8_groupwise_gemm_f32_cuda(mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,mfq_tensor_backend::Tensor input,int64_t groups) {
    return grouped<float>(values,scales,input,groups);
}
