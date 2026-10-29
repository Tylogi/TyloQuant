// Copyright 2026 MFQ contributors. SPDX-License-Identifier: Apache-2.0
// Independent fixed-book TQ encoder: paired edges, register-cached points,
// FP16 path storage, and periodic cost normalization. No state pruning.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_math_constants.h>
#include <cstdint>
#include <algorithm>
#include "../../kernels/hip/mfq_hip_compat.h"

#ifdef _WIN32
#define TQ_FIT_EXPORT __declspec(dllexport)
#else
#define TQ_FIT_EXPORT
#endif

namespace {
template<int K> constexpr int E=1<<(16-K);
constexpr int NT=1024;
__device__ __forceinline__ half2 book(unsigned s){
 unsigned p=s*0x83dcd12du,q=(s+1)*0x83dcd12du;
 unsigned a=(p&255)+((p>>8)&255)+((p>>16)&255)+(p>>24);
 unsigned b=(q&255)+((q>>8)&255)+((q>>16)&255)+(q>>24);
 return __hfma2(__halves2half2(__ushort_as_half(0x6400+a),__ushort_as_half(0x6400+b)),
  __half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931)));
}
template<int K> __device__ __forceinline__ int choice(const uint8_t*trace,int t,int e){
 if constexpr(K==1){int off=t*(E<K>/32)+(e/64)*2+(e&1);return (reinterpret_cast<const uint32_t*>(trace)[off]>>((e/2)&31))&1;}
 else return (trace[(int64_t(t)*E<K>+e)/2]>>(K*(e&1)))&((1<<K)-1);
}
// Leave room for block metadata while retaining almost all 1-bit costs in LDS.
template<int K> __device__ __forceinline__ half cost_read(const half* lds, const half* spill, int i) {
 if constexpr(K==1) return i<E<K>-512?lds[i]:spill[i-(E<K>-512)];
 else return lds[i];
}
template<int K> __device__ __forceinline__ void cost_write(half* lds, half* spill, int i, half v) {
 if constexpr(K==1) {if(i<E<K>-512) lds[i]=v;else spill[i-(E<K>-512)]=v;}
 else lds[i]=v;
}
template<int K> __global__ __launch_bounds__(NT,1) void solve(
 const float*x,const float*m,float*y,uint16_t*states,uint8_t*trace,int*bound,half*scratch,int norm){
 constexpr int N=E<K>/(2*NT),C=1<<K;
 int tid=threadIdx.x;int64_t r=blockIdx.x;x+=r*256;m+=r*256;y+=r*256;states+=r*256;
 trace+=r*256*E<K>/(K==1?8:2);
 extern __shared__ half costs[];
 __shared__ uint32_t minima[32];__shared__ float offset;__shared__ int edge;__shared__ uint16_t symbols[256];
 half2 points[N][C];
 {
  #pragma unroll
  for(int i=0;i<N;++i){int e=2*(tid+i*NT);
   #pragma unroll
   for(int j=0;j<C;++j)points[i][j]=book(e+j*E<K>);
  }
 }
 for(int pass=0;pass<2;++pass){
  half*spill=K==1?scratch+r*512:scratch;
  half*old=costs;half*next=costs+(K<=2?0:E<K>);
  for(int e=tid;e<E<K>;e+=NT)cost_write<K>(old,spill,e,__float2half(pass&&e!=edge?INFINITY:0.f));
  __syncthreads();
  for(int t=0;t<256;++t){int at=pass?t:(t+128)&255;
   half2 target=__half2half2(__float2half_rn(x[at])),metric=__half2half2(__float2half_rn(m[at]));
   half2 staged[K<=2?N:1];
   #pragma unroll
   for(int i=0;i<N;++i){int e=2*(tid+i*NT);uint32_t lo=0x7c00ffffu,hi=0x7c00ffffu;
    #pragma unroll
    for(int j=0;j<C;++j){int s=e+j*E<K>;half2 q=points[i][j];
     half2 d=__hmul2(__hsub2(q,target),metric);
     half2 v=__hfma2(d,d,__half2half2(cost_read<K>(old,spill,s>>K)));
     lo=min(lo,(uint32_t(__half_as_ushort(__low2half(v)))<<16)|j);
     hi=min(hi,(uint32_t(__half_as_ushort(__high2half(v)))<<16)|j);
    }
    half2 best=__halves2half2(__ushort_as_half(lo>>16),__ushort_as_half(hi>>16));
    if constexpr(K<=2)staged[i]=best;else reinterpret_cast<half2*>(next)[e/2]=best;
    if(pass||t>=128){
     if constexpr(K==1){unsigned a=__ballot_sync(0xffffffff,lo&1),b=__ballot_sync(0xffffffff,hi&1);
      if(!(tid&31)){int o=t*(E<K>/32)+(e/64)*2;reinterpret_cast<uint32_t*>(trace)[o]=a;reinterpret_cast<uint32_t*>(trace)[o+1]=b;}
     }else trace[(int64_t(t)*E<K>+e)/2]=uint8_t((lo&((1<<K)-1))|((hi&((1<<K)-1))<<K));
    }
   }
   if constexpr(K<=2){__syncthreads();
    #pragma unroll
    for(int i=0;i<N;++i){int e=2*(tid+i*NT);
     cost_write<K>(next,spill,e,__low2half(staged[i]));
     cost_write<K>(next,spill,e+1,__high2half(staged[i]));
    }
   }
   __syncthreads();half*tmp=old;old=next;next=tmp;
   if(norm&&((t+1)%norm==0)&&t!=255){
    float v=INFINITY;for(int e=tid;e<E<K>;e+=NT)v=fminf(v,__half2float(cost_read<K>(old,spill,e)));
    for(int d=16;d;d/=2)v=fminf(v,__shfl_down_sync(0xffffffff,v,d));
    if(!(tid&31))minima[tid/32]=__float_as_uint(v);__syncthreads();
    if(!tid){uint32_t b=minima[0];for(int i=1;i<32;++i)b=min(b,minima[i]);offset=__uint_as_float(b);}
    __syncthreads();for(int e=tid;e<E<K>;e+=NT)cost_write<K>(old,spill,e,__float2half_rn(__half2float(cost_read<K>(old,spill,e))-offset));__syncthreads();
   }
  }
  if(!pass){uint32_t b=0x7c00ffffu;
   for(int e=tid;e<E<K>;e+=NT)b=min(b,(uint32_t(__half_as_ushort(cost_read<K>(old,spill,e)))<<16)|e);
   for(int d=16;d;d/=2)b=min(b,__shfl_down_sync(0xffffffff,b,d));
   if(!(tid&31))minima[tid/32]=b;__syncthreads();
   if(!tid){for(int i=1;i<32;++i)minima[0]=min(minima[0],minima[i]);int e=minima[0]&65535;
    for(int t=255;t>=128;--t)e=(e+choice<K>(trace,t,e)*E<K>)>>K;edge=e;bound[r]=e;
   }__syncthreads();
  }else{
   if(!tid){int e=edge;for(int t=255;t>=0;--t){int s=e+choice<K>(trace,t,e)*E<K>;symbols[t]=s&((1<<K)-1);e=s>>K;}}
   __syncthreads();if(tid<256){unsigned s=0;
    #pragma unroll
    for(int j=(16+K-1)/K-1;j>=0;--j)s=(s<<K)|symbols[(tid-j+256)&255];
    states[tid]=uint16_t(s);y[tid]=__half2float(__low2half(book(s&65535)));
   }
  }
 }
}
template<int K> constexpr int SH=K==1?(E<K>-512)*2:E<K>*(K==2?2:4);
template<int K> int capacity(int n){auto f=solve<K>;int a=0,d=0;auto e=hipFuncSetAttribute(reinterpret_cast<const void*>(f),hipFuncAttributeMaxDynamicSharedMemorySize,SH<K>);if(e)return -int(e);
 e=hipOccupancyMaxActiveBlocksPerMultiprocessor(&a,f,NT,SH<K>);if(e)return -int(e);hipGetDevice(&d);hipDeviceProp_t p;hipGetDeviceProperties(&p,d);return a?std::min(n,a*p.multiProcessorCount):-1;
}
template<int K> int launch(const float*x,const float*m,float*y,uint16_t*states,uint8_t*trace,int*bound,int n,int wave,int norm,half*scratch,hipStream_t stream){
 for(int i=0;i<n;i+=wave){int64_t a=int64_t(i)*256;hipLaunchKernelGGL((solve<K>),dim3(std::min(wave,n-i)),dim3(NT),SH<K>,stream,x+a,m+a,y+a,states+a,trace,bound+i,scratch,norm);auto e=hipGetLastError();if(e)return int(e);}return 0;
}

}
extern "C" TQ_FIT_EXPORT int half_wave(int n,int k){
 #define C(K) case K:return capacity<K>(n)
 switch(k){C(1);C(2);C(3);C(4);}return -1;
 #undef C
}
extern "C" TQ_FIT_EXPORT int half_encode(const float*x,const float*m,float*y,uint16_t*states,uint8_t*trace,int*bound,int n,int k,int wave,int norm,half*scratch,hipStream_t stream){
 #define C(K) case K:return launch<K>(x,m,y,states,trace,bound,n,wave,norm,scratch,stream)
 switch(k){C(1);C(2);C(3);C(4);}return -1;
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
