// Copyright 2026 MFQ contributors. SPDX-License-Identifier: Apache-2.0
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include "tq_projection.h"
#include "tq_vector.h"

namespace mfq_hip_tq_prefill {
template<int DT> using Scalar=std::conditional_t<DT==2,float,uint16_t>;
template<int DT> __device__ float number(Scalar<DT> x) {
 if constexpr(DT==2)return x;
 else if constexpr(DT==1)return __bfloat162float(__ushort_as_bfloat16(x));
 else return __half2float(__ushort_as_half(x));
}
template<int DT> __device__ Scalar<DT> result(float x) {
 if constexpr(DT==2)return x;
 else if constexpr(DT==1)return __bfloat16_as_ushort(__float2bfloat16(x));
 else return __half_as_ushort(__float2half_rn(x));
}

// Two half components share a power-of-two scale for each original FP32 row.
// The low component retains the residual discarded by the high component.
template<int DT> __global__ void pack(const Scalar<DT>* input,half* high,half* low,
 float* scales,int width) {
 __shared__ float maxima[4];
 __shared__ int invalid[4];
 const int row=blockIdx.x,lane=threadIdx.x&63,wave=threadIdx.x/64;
 float maximum=0;int bad=0;
 for(int k=threadIdx.x;k<width;k+=256) {
  const float x=number<DT>(input[size_t(row)*width+k]);
  maximum=fmaxf(maximum,fabsf(x));bad|=!isfinite(x);
 }
 for(int shift=32;shift;shift/=2){maximum=fmaxf(maximum,__shfl_xor(maximum,shift,64));bad|=__shfl_xor(bad,shift,64);}
 if(!lane){maxima[wave]=maximum;invalid[wave]=bad;}
 __syncthreads();
 if(!threadIdx.x) {
  maximum=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
  int exponent=0;if(maximum>0)frexpf(maximum,&exponent);
  bad=invalid[0]|invalid[1]|invalid[2]|invalid[3];
  scales[row]=(bad || exponent>100 || exponent< -100)?0.f:ldexpf(1.f,maximum>0?14-exponent:0);
 }
 __syncthreads();const float scale=scales[row];
 for(int k=threadIdx.x;k<width;k+=256) {
  const size_t at=size_t(row)*width+k;
  const float x=scale?number<DT>(input[at])*scale:0.f;
  const half h=__float2half_rn(x);high[at]=h;low[at]=__float2half_rn(x-__half2float(h));
 }
}

}

namespace mfq_hip_tq_prefill {
struct PreparedWeight {mfq_tq_vector::U4 raw{},columns{};};
__device__ __forceinline__ PreparedWeight prefetch_weight(ProjectionView m,uint64_t descriptor,int row,int col) {
 PreparedWeight data;
 if(row>=m.rows || col>=m.input_columns)return data;
 const int bits=int((descriptor>>32)&7)+1,ringwords=8*bits;
 const int first=((31-(col/8)%32)*(8*bits))>>5;
 const auto* ring=m.words+uint32_t(descriptor)+(col>>8)*ringwords;
 #pragma unroll
 for(int j=0;j<3;++j){int at=first+j;if(at>=ringwords)at-=ringwords;data.raw[j]=__ldg(ring+at);}
 if(m.columns) {
  const auto* source=reinterpret_cast<const uint32_t*>(static_cast<const half*>(m.scales)+col);
  #pragma unroll
  for(int j=0;j<4;++j)data.columns[j]=__ldg(source+j);
 }
 return data;
}
struct PackedStore {
 half* output;int column,mask;uint32_t pending=0;
 __device__ __forceinline__ void operator()(int j,half a,half b) {
  const auto packed=__builtin_bit_cast(uint32_t,__halves2half2(a,b));
  if((j&2)==0)pending=packed;
  else *reinterpret_cast<uint2*>(output+((column+j-2)^mask))=make_uint2(pending,packed);
 }
};
struct OriginalDot {
 const float* input;float value=0.f;
 __device__ __forceinline__ void operator()(int j,float a,float b) {
  value=fmaf(input[j],a,value);value=fmaf(input[j+1],b,value);
 }
};
__global__ __launch_bounds__(64) void exceptional_rows(ProjectionView m,const float* input,
 float* output,const float* scales) {
 const int token=blockIdx.x;if(scales[token]!=0.f)return;
 for(int row=threadIdx.x;row<m.rows;row+=64) {
  OriginalDot dot;const auto descriptor=m.descriptors[row];
  for(int col=0;col<m.input_columns;col+=8) {
   dot.input=input+size_t(token)*m.input_columns+col;
   if(m.columns)mfq_tq_vector::visit_values<8,true,true,0,OriginalDot>(m.words,descriptor,m.scales,
    static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,dot);
   else mfq_tq_vector::visit_values<8,true,false,0,OriginalDot>(m.words,descriptor,m.scales,
    static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,dot);
  }
  output[size_t(token)*m.rows+row]=dot.value+(m.bias?m.bias[row]:0.f);
 }
}
template<int BM,int BK>__global__ __launch_bounds__(256) void two_phase(
 ProjectionView m,float* output,int tokens,const half* high,const half* low,const float* scales) {
 using F4=float __attribute__((ext_vector_type(4)));
 using H4=_Float16 __attribute__((ext_vector_type(4)));
 __shared__ half w[64*BK],x[BM*BK];
#if defined(__gfx942__)
 const int tid=threadIdx.x,lane=tid&63,wave=tid/64,n0=blockIdx.x*64,t0=blockIdx.y*BM;
 constexpr int Vectors=64*(BK/8),Count=(Vectors+255)/256;
 uint64_t descriptors[Count];
 #pragma unroll
 for(int i=0;i<Count;++i) {
  const int row=n0+(tid+i*256)/(BK/8);
  descriptors[i]=row<m.rows?m.descriptors[row]:0;
 }
 F4 accum[BM/16]{};
 PreparedWeight prepared[Count];
 #pragma unroll
 for(int i=0;i<Count;++i)prepared[i]=prefetch_weight(m,descriptors[i],n0+(tid+i*256)/(BK/8),(tid%(BK/8))*8);
 for(int kb=0;kb<m.input_columns;kb+=BK) {
  #pragma unroll
  for(int i=tid;i<Vectors;i+=256) {
   const int local_row=i/(BK/8),column=kb+(i%(BK/8))*8,row=n0+local_row;
   PackedStore store{w+local_row*BK,(i%(BK/8))*8,(local_row&(BK/4-1))*4};
   if(row<m.rows && column<m.stored_columns) {
    const auto& data=prepared[i/256];
    if(m.columns)mfq_tq_vector::visit_values<8,true,true,0,PackedStore,false,2,true>(m.words,descriptors[i/256],m.scales,static_cast<const half*>(m.relative),nullptr,nullptr,row,column,m.stored_columns,m.gslog,store,data.raw,data.columns);
    else mfq_tq_vector::visit_values<8,true,false,0,PackedStore,false,2,true>(m.words,descriptors[i/256],m.scales,static_cast<const half*>(m.relative),nullptr,nullptr,row,column,m.stored_columns,m.gslog,store,data.raw,data.columns);
   }else {
    #pragma unroll
    for(int j=0;j<8;j+=2)store(j,half{},half{});
   }
  }
  #pragma unroll
  for(int phase=0;phase<2;++phase) {
   const half* input=phase?low:high;
   #pragma unroll
   for(int i=tid;i<BM*(BK/8);i+=256) {
    const int row=i/(BK/8),column=(i%(BK/8))*8,mask=(row&(BK/4-1))*4;
    const uint4 packed=t0+row<tokens && kb+column<m.input_columns?
     *reinterpret_cast<const uint4*>(input+size_t(t0+row)*m.input_columns+kb+column):uint4{};
    *reinterpret_cast<uint2*>(x+row*BK+(column^mask))=make_uint2(packed.x,packed.y);
    *reinterpret_cast<uint2*>(x+row*BK+((column+4)^mask))=make_uint2(packed.z,packed.w);
   }
   __syncthreads();
   if(phase==0 && kb+BK<m.input_columns) {
    #pragma unroll
    for(int i=0;i<Count;++i)prepared[i]=prefetch_weight(m,descriptors[i],n0+(tid+i*256)/(BK/8),kb+BK+(tid%(BK/8))*8);
   }
   #pragma unroll
   for(int k=0;k<BK;k+=16) {
    const int wr=wave*16+(lane&15),wc=(k+(lane>>4)*4)^((lane&(BK/4-1))*4);
    const auto a=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(w+wr*BK+wc));
    #pragma unroll
    for(int mt=0;mt<BM/16;mt+=4) {
     H4 b[4];
     #pragma unroll
     for(int j=0;j<4;++j)b[j]=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(x+((mt+j)*16+(lane&15))*BK+wc));
     asm volatile(
      "v_mfma_f32_16x16x16_f16 %0, %4, %5, %0\n\t"
      "v_mfma_f32_16x16x16_f16 %1, %4, %6, %1\n\t"
      "v_mfma_f32_16x16x16_f16 %2, %4, %7, %2\n\t"
      "v_mfma_f32_16x16x16_f16 %3, %4, %8, %3"
      : "+a"(accum[mt]),"+a"(accum[mt+1]),"+a"(accum[mt+2]),"+a"(accum[mt+3])
      : "v"(a),"v"(b[0]),"v"(b[1]),"v"(b[2]),"v"(b[3]));
    }
   }
   __syncthreads();
  }
 }
 #pragma unroll
 for(int mt=0;mt<BM/16;++mt) {
  const int token=t0+mt*16+(lane&15),row=n0+wave*16+(lane>>4)*4;
  if(token<tokens) {
   #pragma unroll
   for(int j=0;j<4;++j)if(row+j<m.rows) {
    float value=accum[mt][j]/scales[token];if(m.bias)value+=m.bias[row+j];output[size_t(token)*m.rows+row+j]=value;
   }
  }
 }
#endif
}
}

#include <cstring>
extern "C" int mfq_tq_prefill_f32_supported(int device) {
 static thread_local int cached_device=-1;
 static thread_local bool supported=false;
 if(cached_device!=device) {
  hipDeviceProp_t properties{};
  if(hipGetDeviceProperties(&properties,device)!=hipSuccess)return 0;
  supported=std::strncmp(properties.gcnArchName,"gfx942",6)==0;
  cached_device=device;
 }
 return supported;
}
extern "C" int mfq_tq_prefill_f32(ProjectionView view,const float* input,float* output,int tokens,
 void* workspace,uint64_t workspace_bytes,hipStream_t stream) {
 if(!input || !output || !workspace || tokens<1 || view.rows<1 || view.input_columns<1 ||
    (view.input_columns&7) || view.stored_columns<view.input_columns)
  return int(hipErrorInvalidValue);
 const uint64_t values=uint64_t(tokens)*view.input_columns;
 const uint64_t required=values*4+uint64_t(tokens)*4;
 if(workspace_bytes<required)return int(hipErrorInvalidValue);
 int device=0;auto error=hipGetDevice(&device);if(error!=hipSuccess)return int(error);
 if(!mfq_tq_prefill_f32_supported(device))return int(hipErrorNotSupported);
 auto* high=static_cast<half*>(workspace);auto* low=high+values;
 auto* scales=reinterpret_cast<float*>(low+values);
 mfq_hip_tq_prefill::pack<2><<<tokens,256,0,stream>>>(input,high,low,scales,view.input_columns);
 mfq_hip_tq_prefill::two_phase<256,32><<<dim3((view.rows+63)/64,(tokens+255)/256),256,0,stream>>>(view,output,tokens,high,low,scales);
 mfq_hip_tq_prefill::exceptional_rows<<<tokens,64,0,stream>>>(view,input,output,scales);
 return int(hipGetLastError());
}

extern "C" uint64_t mfq_tq_prefill_f32_workspace(ProjectionView view,int tokens) {
 if(tokens<1 || view.rows<1 || view.input_columns<1)return 0;
 return uint64_t(tokens)*(uint64_t(view.input_columns)*4+4);
}
