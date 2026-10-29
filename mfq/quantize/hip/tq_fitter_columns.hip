// Copyright 2026 MFQ contributors. SPDX-License-Identifier: Apache-2.0
// Independent fixed-book TQ encoder: paired edges, register-cached points,
// FP16 path storage, and periodic cost normalization. No state pruning.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_math_constants.h>
#include "../../kernels/hip/mfq_hip_compat.h"
#include <cstdint>
#include <algorithm>

#ifdef _WIN32
#define TQ_FIT_EXPORT __declspec(dllexport)
#else
#define TQ_FIT_EXPORT
#endif

namespace {
#ifndef MFQ_TQ_STATE_BITS
#define MFQ_TQ_STATE_BITS 16
#endif
constexpr int STATE_BITS=MFQ_TQ_STATE_BITS;
#ifndef MFQ_TQ_WARMUP
#define MFQ_TQ_WARMUP 128
#endif
constexpr int WARMUP=MFQ_TQ_WARMUP;
template<int K> constexpr int E=1<<(STATE_BITS-K);
template<int K> constexpr int LANES=K>=7 ? 1<<(K-6) : 1;
template<int K> constexpr int THREADS=(K<=3||K==5?1024:((K==4||K==8)?256:512));
// Internal traceback compression; serialized row streams stay unchanged.
template<int K> constexpr bool PAIRED=WARMUP<128&&(K==2||K==3);
template<int K> constexpr int TRACE_DIV=K==1?8:((PAIRED<K>)?2:1);
template<int K=0> __device__ __forceinline__ half2 book(unsigned s){
 unsigned p=s*0x83dcd12du,q=(s+1)*0x83dcd12du;
 unsigned a,b;
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_udot4)
 if constexpr(K>=6){
  a=__builtin_amdgcn_udot4(p,0x01010101u,0u,false);
  b=__builtin_amdgcn_udot4(q,0x01010101u,0u,false);
 }else
#endif
 {
  a=(p&255)+((p>>8)&255)+((p>>16)&255)+(p>>24);
  b=(q&255)+((q>>8)&255)+((q>>16)&255)+(q>>24);
 }
 return __hfma2(__halves2half2(__ushort_as_half(0x6400+a),__ushort_as_half(0x6400+b)),
  __half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931)));
}
// Keep both FP16 lanes packed throughout candidate arithmetic.
using tq_half2 = _Float16 __attribute__((ext_vector_type(2)));
template<int K> __device__ __forceinline__ int choice(const uint8_t*trace,int t,int e){
 if constexpr(K==1){int off=t*(E<K>/32)+(e/64)*2+(e&1);return (reinterpret_cast<const uint32_t*>(trace)[off]>>((e/2)&31))&1;}
 else return trace[int64_t(t)*E<K>+e];
}
// One-bit costs use all 64 KiB LDS; its block metadata uses global scratch.
// Two-bit costs are replaced in place so paired traceback also fits in LDS.
template<int K> __device__ __forceinline__ half cost_read(const half* lds,const half* spill,int i){
 return lds[i];
}
template<int K> __device__ __forceinline__ void cost_write(half* lds,half* spill,int i,half v){
 lds[i]=v;
}
template<int K> __global__ __launch_bounds__(THREADS<K>,1) void solve(
 const float*x,const float*m,float*y,uint16_t*states,uint8_t*trace,int*bound,half*scratch,int norm,int total){
 if constexpr(WARMUP<128)norm=16;
 constexpr int F=LANES<K>, OWNERS=THREADS<K>/F;
 constexpr int N=E<K>/(2*OWNERS),C=1<<K;
 int tid=threadIdx.x, choice_lane=tid%F, owner=tid/F;
 const float*xbase=x,*mbase=m;float*ybase=y;uint16_t*statebase=states;
 trace+=int64_t(blockIdx.x)*256*E<K>/TRACE_DIV<K>;
 extern __shared__ half costs[];
 uint8_t*previous_choice=reinterpret_cast<uint8_t*>(costs)+(K==2?2:4)*E<K>;
 struct Metadata {uint32_t minima[THREADS<K>/32];float offset;int edge;uint16_t symbols[256];};
 __shared__ Metadata shared_metadata;
 Metadata* meta;
 if constexpr(K==1)meta=reinterpret_cast<Metadata*>(scratch+int64_t(blockIdx.x)*512);
 else meta=&shared_metadata;
 uint32_t* minima=meta->minima;float& offset=meta->offset;int& edge=meta->edge;uint16_t* symbols=meta->symbols;
 // Each lane caches its codebook points across the persistent ring loop.
 uint32_t points[N][C/F];
 {
  #pragma unroll
  for(int i=0;i<N;++i){int e=2*(owner+i*OWNERS);
   #pragma unroll
   for(int j=0;j<(C/F);++j)points[i][j]=__builtin_bit_cast(uint32_t,book<K>(e+(j*F+choice_lane)*E<K>));
  }
 }
 int count=1;
 if constexpr(WARMUP<128)count=(total-int(blockIdx.x)+int(gridDim.x)-1)/int(gridDim.x);
 for(int item=0;item<count;++item){
 int64_t r=int64_t(blockIdx.x)+int64_t(item)*gridDim.x;
 x=xbase+r*256;m=mbase+r*256;y=ybase+r*256;states=statebase+r*256;
 for(int pass=0;pass<2;++pass){
  half*spill=K==1?scratch+int64_t(blockIdx.x)*512:scratch;
  half*old=costs;half*next=costs+(K<=2?0:E<K>);
  for(int e=tid;e<E<K>;e+=THREADS<K>)cost_write<K>(old,spill,e,__float2half(pass&&e!=edge?INFINITY:0.f));
  __syncthreads();
  int steps=pass?256:2*WARMUP;
  #pragma unroll 2
  for(int t=0;t<steps;++t){int at=pass?t:(256-WARMUP+t)&255;
   half2 target=__half2half2(__float2half_rn(x[at])),metric=__half2half2(__float2half_rn(m[at]));
   half2 staged[K<=2?N:1];
   #pragma unroll
   for(int i=0;i<N;++i){int e=2*(owner+i*OWNERS);uint32_t lbest=0x7c000000u,hbest=0x7c000000u;
    #pragma unroll (C/F)
    for(int candidate=0;candidate<C/F;++candidate){
     int j=candidate*F+choice_lane;int s=e+j*E<K>;half2 q;
     uint32_t point;
     point=points[i][candidate];
     q=__builtin_bit_cast(half2,point);
     half2 v;
     tq_half2 d=(__builtin_bit_cast(tq_half2,q)-__builtin_bit_cast(tq_half2,target))*__builtin_bit_cast(tq_half2,metric);
     _Float16 prior=__builtin_bit_cast(_Float16,cost_read<K>(old,spill,s>>K));
     v=__builtin_bit_cast(half2,__builtin_elementwise_fma(d,d,(tq_half2){prior,prior}));
     uint32_t word=__builtin_bit_cast(uint32_t,v);
     lbest=min(lbest,(word<<16)|unsigned(j));hbest=min(hbest,(word&0xffff0000u)|unsigned(j));
    }
    half2 best=__halves2half2(__ushort_as_half(lbest>>16),__ushort_as_half(hbest>>16));
    uint32_t picks=(lbest&65535)|(hbest<<16);
    if constexpr(F>1){
     uint32_t low=(uint32_t(__half_as_ushort(__low2half(best)))<<16)|(picks&65535);
     uint32_t high=(uint32_t(__half_as_ushort(__high2half(best)))<<16)|(picks>>16);
     #pragma unroll
     for(int delta=F/2;delta;delta/=2){
      low=min(low,__shfl_xor(low,delta,64));high=min(high,__shfl_xor(high,delta,64));
     }
     best=__halves2half2(__ushort_as_half(low>>16),__ushort_as_half(high>>16));
     picks=(low&65535)|(high<<16);
    }
    uint32_t lo=picks&65535,hi=picks>>16;
    if(choice_lane==0){
    if constexpr(K<=2)staged[i]=best;else reinterpret_cast<half2*>(next)[e/2]=best;
    if constexpr(PAIRED<K>){
     if(!(t&1))reinterpret_cast<uint16_t*>(previous_choice)[e/2]=uint16_t(lo|(hi<<8));
    }
    if(pass||t>=WARMUP){
     if constexpr(K==1){unsigned a=__ballot_sync(0xffffffff,lo&1),b=__ballot_sync(0xffffffff,hi&1);
      if(!(tid&31)){int o=t*(E<K>/32)+(e/64)*2;reinterpret_cast<uint32_t*>(trace)[o]=a;reinterpret_cast<uint32_t*>(trace)[o+1]=b;}
     }else if constexpr(PAIRED<K>){
      if(t&1){
       uint32_t a=lo|(uint32_t(previous_choice[(e+lo*E<K>)>>K])<<K);
       uint32_t b=hi|(uint32_t(previous_choice[(e+1+hi*E<K>)>>K])<<K);
       reinterpret_cast<uint16_t*>(trace)[(int64_t(t/2)*E<K>+e)/2]=uint16_t(a|(b<<8));
      }
     }else reinterpret_cast<uint16_t*>(trace)[(int64_t(t)*E<K>+e)/2]=uint16_t((lo&255)|((hi&255)<<8));
    }
    }
   }
   if constexpr(K<=2){__syncthreads();
    #pragma unroll
    for(int i=0;i<N;++i){int e=2*(owner+i*OWNERS);
     cost_write<K>(next,spill,e,__low2half(staged[i]));
     cost_write<K>(next,spill,e+1,__high2half(staged[i]));
    }
   }
   __syncthreads();half*tmp=old;old=next;next=tmp;
   if(norm&&((t+1)%norm==0)&&t!=steps-1){
    if constexpr(K<=4){
    uint32_t pair=0x7c007c00u;
    for(int e=2*tid;e<E<K>;e+=2*THREADS<K>){
     uint32_t value=reinterpret_cast<const uint32_t*>(old)[e/2];
     asm("v_pk_min_f16 %0, %1, %2" : "=v"(pair) : "v"(pair), "v"(value));
    }
    uint32_t v=min(pair&65535u,pair>>16);
    for(int d=16;d;d/=2)v=min(v,__shfl_down_sync(0xffffffff,v,d));
    if(!(tid&31))minima[tid/32]=v;__syncthreads();
    if(!tid){uint32_t b=minima[0];for(int i=1;i<THREADS<K>/32;++i)b=min(b,minima[i]);offset=__half2float(__ushort_as_half(b));}
    }else{
    float v=INFINITY;for(int e=tid;e<E<K>;e+=THREADS<K>)v=fminf(v,__half2float(cost_read<K>(old,spill,e)));
    for(int d=16;d;d/=2)v=fminf(v,__shfl_down_sync(0xffffffff,v,d));
    if(!(tid&31))minima[tid/32]=__float_as_uint(v);__syncthreads();
    if(!tid){uint32_t b=minima[0];for(int i=1;i<THREADS<K>/32;++i)b=min(b,minima[i]);offset=__uint_as_float(b);}
    }
    __syncthreads();
    if constexpr(K<=4){
    half2 delta=__half2half2(__float2half_rn(offset));
    for(int e=2*tid;e<E<K>;e+=2*THREADS<K>){
     half2 value=reinterpret_cast<const half2*>(old)[e/2];
     reinterpret_cast<half2*>(old)[e/2]=__hsub2(value,delta);
    }
    }else{
    for(int e=tid;e<E<K>;e+=THREADS<K>)cost_write<K>(old,spill,e,__float2half_rn(__half2float(cost_read<K>(old,spill,e))-offset));
    }
    __syncthreads();
   }
  }
  if(!pass){uint32_t b=0x7c00ffffu;
   for(int e=tid;e<E<K>;e+=THREADS<K>)b=min(b,(uint32_t(__half_as_ushort(cost_read<K>(old,spill,e)))<<16)|e);
   for(int d=16;d;d/=2)b=min(b,__shfl_down_sync(0xffffffff,b,d));
   if(!(tid&31))minima[tid/32]=b;__syncthreads();
   if(!tid){for(int i=1;i<THREADS<K>/32;++i)minima[0]=min(minima[0],minima[i]);int e=minima[0]&65535;
    if constexpr(PAIRED<K>){
     for(int t=2*WARMUP-1;t>=WARMUP;t-=2){
      int p=trace[int64_t(t/2)*E<K>+e];
      e=(e+(p&((1<<K)-1))*E<K>)>>K;e=(e+(p>>K)*E<K>)>>K;
     }
    }else for(int t=2*WARMUP-1;t>=WARMUP;--t)e=(e+choice<K>(trace,t,e)*E<K>)>>K;
    edge=e;bound[r]=e;
   }__syncthreads();
  }else{
   if(!tid){int e=edge;
    if constexpr(PAIRED<K>){
     for(int t=255;t>=0;t-=2){
      int p=trace[int64_t(t/2)*E<K>+e],s=e+(p&((1<<K)-1))*E<K>;
      symbols[t]=s&((1<<K)-1);e=s>>K;
      s=e+(p>>K)*E<K>;symbols[t-1]=s&((1<<K)-1);e=s>>K;
     }
    }else for(int t=255;t>=0;--t){int s=e+choice<K>(trace,t,e)*E<K>;symbols[t]=s&((1<<K)-1);e=s>>K;}
   }
   __syncthreads();for(int out=tid;out<256;out+=THREADS<K>){unsigned s=0;
    #pragma unroll
    for(int j=(16+K-1)/K-1;j>=0;--j)s=(s<<K)|symbols[(out-j+256)&255];
    states[out]=uint16_t(s);y[out]=__half2float(__low2half(book(s&((1<<STATE_BITS)-1))));
   }
  }
 }
 if constexpr(WARMUP<128)__syncthreads();
 }
}
template<int K> constexpr int SH=(K==1?E<K>*2:E<K>*(K==2?(PAIRED<K>?3:2):(PAIRED<K>?5:4)));
template<int K> int capacity(int n){auto f=solve<K>;int a=0,d=0;auto e=hipFuncSetAttribute(reinterpret_cast<const void*>(f),hipFuncAttributeMaxDynamicSharedMemorySize,SH<K>);if(e)return -int(e);
 e=hipOccupancyMaxActiveBlocksPerMultiprocessor(&a,f,THREADS<K>,SH<K>);if(e)return -int(e);hipGetDevice(&d);hipDeviceProp_t p;hipGetDeviceProperties(&p,d);return a?std::min(n,a*p.multiProcessorCount):-1;
}
template<int K> int launch(const float*x,const float*m,float*y,uint16_t*states,uint8_t*trace,int*bound,int n,int wave,int norm,half*scratch,hipStream_t stream){
 if constexpr(WARMUP<128){
  hipLaunchKernelGGL((solve<K>),dim3(std::min(wave,n)),dim3(THREADS<K>),SH<K>,stream,x,m,y,states,trace,bound,scratch,norm,n);
  return int(hipGetLastError());
 }else{
  for(int i=0;i<n;i+=wave){int64_t a=int64_t(i)*256;int count=std::min(wave,n-i);
   hipLaunchKernelGGL((solve<K>),dim3(count),dim3(THREADS<K>),SH<K>,stream,x+a,m+a,y+a,states+a,trace,bound+i,scratch,norm,count);
   auto e=hipGetLastError();if(e)return int(e);
  }return 0;
 }
}

}
extern "C" TQ_FIT_EXPORT int half_wave(int n,int k){
 #define C(K) case K:return capacity<K>(n)
 switch(k){C(1);C(2);C(3);C(4);C(5);C(6);C(7);C(8);}return -1;
 #undef C
}
extern "C" TQ_FIT_EXPORT int half_encode(const float*x,const float*m,float*y,uint16_t*states,uint8_t*trace,int*bound,int n,int k,int wave,int norm,half*scratch,hipStream_t stream){
 #define C(K) case K:return launch<K>(x,m,y,states,trace,bound,n,wave,norm,scratch,stream)
 switch(k){C(1);C(2);C(3);C(4);C(5);C(6);C(7);C(8);}return -1;
 #undef C
}

namespace {
__device__ __forceinline__ float sumwarp(float x){for(int d=16;d;d/=2)x+=__shfl_down_sync(0xffffffff,x,d);return x;}
__global__ void refit(const float*w,const float*h,const float*y,const uint16_t*state,half*a,uint8_t*c,float*best,uint16_t*beststate,const half*relative,int width,int gs){
 int row=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x/32,G=width/gs;extern __shared__ float data[];
 float*num=data,*den=num+G,*old=den+G,*codes=old+G;__shared__ float olda,newa,warploss[16];__shared__ int use_new,improve;
 for(int g=threadIdx.x;g<G;g+=256)old[g]=c[row*G+g];
 for(int g=warp;g<G;g+=8){float n=0,d=0;for(int j=lane;j<gs;j+=32){int col=g*gs+j;int64_t i=int64_t(row)*width+col;float q=y[i],hw=h[col];n+=w[i]*q*hw;d+=q*q*hw;}
  n=sumwarp(n);d=sumwarp(d);if(!lane){num[g]=n;den[g]=d;}
 }
 __syncthreads();if(!threadIdx.x){olda=__half2float(a[row]);newa=olda;
  for(int it=0;it<2;++it){float n=0,d=0;for(int g=0;g<G;++g){float wanted=num[g]/fmaxf(newa*den[g],1e-30f);float bd=INFINITY;int code=int(old[g]);
    for(int j=0;j<16;++j){float delta=fabsf(wanted-__half2float(relative[j]));if(delta<bd){bd=delta;code=j;}}
    if(den[g]==0)code=int(old[g]);codes[g]=code;float gain=__half2float(relative[code]);n+=gain*num[g];d+=gain*gain*den[g];
   }newa=__half2float(__float2half_rn(fmaxf(n/fmaxf(d,1e-30f),0x1p-24f)));
  }
 }
 __syncthreads();float loss0=0,loss1=0;
 for(int col=threadIdx.x;col<width;col+=256){int64_t i=int64_t(row)*width+col;int g=col/gs;float q=y[i],v=w[i],hw=h[col];
  float before=__half2float(__float2half_rn(q*(olda*__half2float(relative[int(old[g])]))));
  float after=__half2float(__float2half_rn(q*(newa*__half2float(relative[int(codes[g])]))));
  float d0=before-v,d1=after-v;loss0+=d0*d0*hw;loss1+=d1*d1*hw;
 }
 loss0=sumwarp(loss0);loss1=sumwarp(loss1);if(!lane){warploss[warp]=loss0;warploss[8+warp]=loss1;}
 __syncthreads();if(!threadIdx.x){float b=0,n=0;for(int j=0;j<8;++j){b+=warploss[j];n+=warploss[8+j];}
  use_new=n<b;float candidate=use_new?n:b;improve=candidate<best[row];if(improve){best[row]=candidate;a[row]=__float2half_rn(use_new?newa:olda);}
 }
 __syncthreads();if(improve){for(int g=threadIdx.x;g<G;g+=256)c[row*G+g]=uint8_t(use_new?codes[g]:old[g]);
  for(int col=threadIdx.x;col<width;col+=256){int64_t i=int64_t(row)*width+col;beststate[i]=state[i];}
 }
}

}
extern "C" TQ_FIT_EXPORT int fit_refine(const float*w,const float*h,const float*y,const uint16_t*state,half*a,uint8_t*c,float*best,uint16_t*beststate,const half*relative,int rows,int width,int gs,hipStream_t s){
 hipLaunchKernelGGL(refit,dim3(rows),dim3(256),(width/gs)*16,s,w,h,y,state,a,c,best,beststate,relative,width,gs);return int(hipGetLastError());
}

namespace {
template<int K> __global__ void pack_words(const uint16_t*state,uint32_t*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*8*K)return;
 int ring=i/(8*K),word=i%(8*K),start=word*32,first=start/K,offset=start%K;
 uint32_t value=(state[int64_t(ring)*256+255-first]&((1<<K)-1))>>offset;
 #pragma unroll
 for(int j=1;j<(32+K-1)/K+1;++j){int pos=j*K-offset;
  if(pos<32 && first+j<256)value|=uint32_t(state[int64_t(ring)*256+255-first-j]&((1<<K)-1))<<pos;
 }
 out[i]=value;
}

}
extern "C" TQ_FIT_EXPORT int pack(const uint16_t*state,uint32_t*out,int n,int k,hipStream_t s){
 #define C(K) case K:hipLaunchKernelGGL((pack_words<K>),dim3((n*8*K+255)/256),dim3(256),0,s,state,out,n);break
 switch(k){C(1);C(2);C(3);C(4);C(5);C(6);C(7);C(8);default:return -1;}
 #undef C
 return int(hipGetLastError());
}
