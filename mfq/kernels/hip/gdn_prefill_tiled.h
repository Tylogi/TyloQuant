// Copyright 2026 MFQ contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
namespace mfq_hip_gdn_prefill {
template<bool Transposed,int C>__global__ __launch_bounds__(256) void recurrent(
 const float* q,const float* k,const float* v,const float* g,
 const float* beta,const float* initial,float* output,float* final_state,
 int query_heads,int value_heads,int tokens,bool tiled_heads) {
 constexpr int D=128,Lanes=16,Rows=8,Columns=16;
 __shared__ float keys[C*D],queries[C*D],values[C*Columns],decays[C],betas[C];
 const int tid=threadIdx.x,lane=tid%Lanes,column=blockIdx.y*Columns+tid/Lanes;
 const int group=blockIdx.x,batch=group/value_heads,head=group%value_heads;
 const int query_head=tiled_heads?head%query_heads:head/(value_heads/query_heads);
 const size_t qb=size_t(batch*query_heads+query_head)*tokens*D;
 const size_t vb=size_t(group)*tokens*D,gb=size_t(group)*tokens,sb=size_t(group)*D*D;
 float state[Rows];
 #pragma unroll
 for(int r=0;r<Rows;++r) {
  const int row=r*Lanes+lane;
  state[r]=initial?initial[sb+(Transposed?size_t(column)*D+row:size_t(row)*D+column)]:0.f;
 }
 for(int start=0;start<tokens;start+=C) {
  for(int i=tid;i<C*D;i+=256) {
   const int token=start+i/D,row=i%D;const size_t at=qb+size_t(token)*D+row;
   keys[i]=token<tokens?k[at]:0.f;queries[i]=token<tokens?q[at]:0.f;
  }
  for(int i=tid;i<C*Columns;i+=256) {
   const int token=start+i/Columns,col=blockIdx.y*Columns+i%Columns;
   values[i]=token<tokens?v[vb+size_t(token)*D+col]:0.f;
  }
  if(tid<C){decays[tid]=start+tid<tokens?expf(g[gb+start+tid]):1.f;betas[tid]=start+tid<tokens?beta[gb+start+tid]:0.f;}
  __syncthreads();
  #pragma unroll
  for(int t=0;t<C;++t)if(start+t<tokens) {
   float key[Rows],query[Rows],projection=0.f;
   #pragma unroll
   for(int r=0;r<Rows;++r){key[r]=keys[t*D+r*Lanes+lane];query[r]=queries[t*D+r*Lanes+lane];projection+=state[r]*key[r];}
   #pragma unroll
   for(int shift=8;shift;shift/=2)projection+=__shfl_xor(projection,shift,Lanes);
   const float decay=decays[t],delta=(values[t*Columns+tid/Lanes]-decay*projection)*betas[t];
   float result=0.f;
   #pragma unroll
   for(int r=0;r<Rows;++r){state[r]=decay*state[r]+key[r]*delta;result+=state[r]*query[r];}
   #pragma unroll
   for(int shift=8;shift;shift/=2)result+=__shfl_xor(result,shift,Lanes);
   if(!lane)output[vb+size_t(start+t)*D+column]=result*0.08838834764831845f;
  }
  __syncthreads();
 }
 #pragma unroll
 for(int r=0;r<Rows;++r) {
  const int row=r*Lanes+lane;
  final_state[sb+(Transposed?size_t(column)*D+row:size_t(row)*D+column)]=state[r];
 }
}
}
