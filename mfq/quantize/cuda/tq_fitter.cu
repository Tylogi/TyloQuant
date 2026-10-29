// Copyright 2026 MFQ contributors. SPDX-License-Identifier: Apache-2.0
// Independent fixed-book TQ encoder: paired edges, register-cached points,
// FP16 path storage, and periodic cost normalization. No state pruning.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
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
template<int K> constexpr int THREADS=std::min(512,E<K>/2);
// Internal traceback compression; serialized row streams stay unchanged.
template<int K> constexpr bool PAIRED=WARMUP<128&&(K==2||K==3);
template<int K> constexpr int TRACE_DIV=K==1?8:((PAIRED<K>)?2:1);
__device__ __forceinline__ half2 book(unsigned s){
 unsigned a=__dp4a(s*0x83dcd12du,0x01010101u,0u),b=__dp4a((s+1)*0x83dcd12du,0x01010101u,0u);
 return __hfma2(__halves2half2(__ushort_as_half(0x6400+a),__ushort_as_half(0x6400+b)),
  __half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931)));
}
template<int K> __device__ __forceinline__ int choice(const uint8_t*trace,int t,int e){
 if constexpr(K==1){int off=t*(E<K>/32)+(e/64)*2+(e&1);return (reinterpret_cast<const uint32_t*>(trace)[off]>>((e/2)&31))&1;}
 else return trace[int64_t(t)*E<K>+e];
}
template<int K> __global__ __launch_bounds__(THREADS<K>,1) void solve(
 const float*x,const float*m,float*y,uint16_t*states,uint8_t*trace,int*bound,int norm,int total){
 if constexpr(WARMUP<128)norm=16;
 constexpr int N=E<K>/(2*THREADS<K>),C=1<<K;
 int tid=threadIdx.x;
 const float*xbase=x,*mbase=m;float*ybase=y;uint16_t*statebase=states;
 trace+=int64_t(blockIdx.x)*256*E<K>/TRACE_DIV<K>;
 extern __shared__ half costs[];
 uint8_t*previous_choice=reinterpret_cast<uint8_t*>(costs)+4*E<K>;
 __shared__ uint32_t minima[THREADS<K>/32];__shared__ float offset;__shared__ int edge;__shared__ uint16_t symbols[256];
 // High bit rates have up to 256 choices per edge. Recompute their cheap
 // fixed-book values instead of spilling a 256-register point cache.
 half2 points[N][K<=4?C:1];
 {
  #pragma unroll
  for(int i=0;i<N;++i){int e=2*(tid+i*THREADS<K>);
   #pragma unroll
   for(int j=0;j<(K<=4?C:1);++j)points[i][j]=book(e+j*E<K>);
  }
 }
 int count=1;
 if constexpr(WARMUP<128)count=(total-int(blockIdx.x)+int(gridDim.x)-1)/int(gridDim.x);
 for(int item=0;item<count;++item){
 int64_t r=int64_t(blockIdx.x)+int64_t(item)*gridDim.x;
 x=xbase+r*256;m=mbase+r*256;y=ybase+r*256;states=statebase+r*256;
 for(int pass=0;pass<2;++pass){
  half*old=costs;half*next=costs+(K==1?0:E<K>);
  for(int e=tid;e<E<K>;e+=THREADS<K>)old[e]=__float2half(pass&&e!=edge?CUDART_INF_F:0.f);
  __syncthreads();
  int steps=pass?256:2*WARMUP;
  for(int t=0;t<steps;++t){int at=pass?t:(256-WARMUP+t)&255;
   half2 target=__half2half2(__float2half_rn(x[at])),metric=__half2half2(__float2half_rn(m[at]));
   half2 staged[K==1?N:1];
   #pragma unroll
   for(int i=0;i<N;++i){int e=2*(tid+i*THREADS<K>);half2 best=__half2half2(__ushort_as_half(0x7c00));uint32_t picks=0;
    #pragma unroll
    for(int j=0;j<C;++j){int s=e+j*E<K>;half2 q;
     if constexpr(K<=4)q=points[i][j];else q=book(s);
     half2 d=__hmul2(__hsub2(q,target),metric);
     half2 v=__hfma2(d,d,__half2half2(old[s>>K]));
     uint32_t take=__hlt2_mask(v,best);picks=(picks&~take)|(uint32_t(j|(j<<16))&take);best=__hmin2(v,best);
    }
    uint32_t lo=picks&65535,hi=picks>>16;
    if constexpr(K==1)staged[i]=best;else reinterpret_cast<half2*>(next)[e/2]=best;
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
   if constexpr(K==1){__syncthreads();
    #pragma unroll
    for(int i=0;i<N;++i)reinterpret_cast<half2*>(next)[tid+i*THREADS<K>]=staged[i];
   }
   __syncthreads();half*tmp=old;old=next;next=tmp;
   if(norm&&((t+1)%norm==0)&&t!=steps-1){
    float v=CUDART_INF_F;for(int e=tid;e<E<K>;e+=THREADS<K>)v=fminf(v,__half2float(old[e]));
    if constexpr(WARMUP<128)v=__uint_as_float(__reduce_min_sync(0xffffffff,__float_as_uint(v)));
    else for(int d=16;d;d/=2)v=fminf(v,__shfl_down_sync(0xffffffff,v,d));
    if(!(tid&31))minima[tid/32]=__float_as_uint(v);__syncthreads();
    if(!tid){uint32_t b=minima[0];for(int i=1;i<THREADS<K>/32;++i)b=min(b,minima[i]);offset=__uint_as_float(b);}
    __syncthreads();for(int e=tid;e<E<K>;e+=THREADS<K>)old[e]=__float2half_rn(__half2float(old[e])-offset);__syncthreads();
   }
  }
  if(!pass){uint32_t b=0x7c00ffffu;
   for(int e=tid;e<E<K>;e+=THREADS<K>)b=min(b,(uint32_t(__half_as_ushort(old[e]))<<16)|e);
   if constexpr(WARMUP<128)b=__reduce_min_sync(0xffffffff,b);
   else for(int d=16;d;d/=2)b=min(b,__shfl_down_sync(0xffffffff,b,d));
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
template<int K> constexpr int SH=E<K>*(K==1?2:((PAIRED<K>)?5:4));
template<int K> int capacity(int n){auto f=solve<K>;int a=0,d=0;auto e=cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,SH<K>);if(e)return -int(e);
 e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a,f,THREADS<K>,SH<K>);if(e)return -int(e);cudaGetDevice(&d);cudaDeviceProp p;cudaGetDeviceProperties(&p,d);return a?std::min(n,a*p.multiProcessorCount):-1;
}
template<int K> int launch(const float*x,const float*m,float*y,uint16_t*states,uint8_t*trace,int*bound,int n,int wave,int norm,cudaStream_t stream){
 if constexpr(WARMUP<128){
  solve<K><<<std::min(wave,n),THREADS<K>,SH<K>,stream>>>(x,m,y,states,trace,bound,norm,n);
  return int(cudaGetLastError());
 }else{
  for(int i=0;i<n;i+=wave){int64_t a=int64_t(i)*256;int count=std::min(wave,n-i);
   solve<K><<<count,THREADS<K>,SH<K>,stream>>>(x+a,m+a,y+a,states+a,trace,bound+i,norm,count);
   auto e=cudaGetLastError();if(e)return int(e);
  }return 0;
 }
}

}
extern "C" TQ_FIT_EXPORT int half_wave(int n,int k){
 #define C(K) case K:return capacity<K>(n)
 switch(k){C(1);C(2);C(3);C(4);C(5);C(6);C(7);C(8);}return -1;
 #undef C
}
extern "C" TQ_FIT_EXPORT int half_encode(const float*x,const float*m,float*y,uint16_t*states,uint8_t*trace,int*bound,int n,int k,int wave,int norm,cudaStream_t stream){
 #define C(K) case K:return launch<K>(x,m,y,states,trace,bound,n,wave,norm,stream)
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
  for(int it=0;it<2;++it){float n=0,d=0;for(int g=0;g<G;++g){float wanted=num[g]/fmaxf(newa*den[g],1e-30f);float bd=CUDART_INF_F;int code=int(old[g]);
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
extern "C" TQ_FIT_EXPORT int fit_refine(const float*w,const float*h,const float*y,const uint16_t*state,half*a,uint8_t*c,float*best,uint16_t*beststate,const half*relative,int rows,int width,int gs,cudaStream_t s){
 refit<<<rows,256,(width/gs)*16,s>>>(w,h,y,state,a,c,best,beststate,relative,width,gs);return int(cudaGetLastError());
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
extern "C" TQ_FIT_EXPORT int pack(const uint16_t*state,uint32_t*out,int n,int k,cudaStream_t s){
 #define C(K) case K:pack_words<K><<<(n*8*K+255)/256,256,0,s>>>(state,out,n);break
 switch(k){C(1);C(2);C(3);C(4);C(5);C(6);C(7);C(8);default:return -1;}
 #undef C
 return int(cudaGetLastError());
}
