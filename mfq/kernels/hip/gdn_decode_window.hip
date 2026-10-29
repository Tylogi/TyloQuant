#include "mfq_hip_linear_attention_ops.h"
#include "mfq_hip_compat.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <array>
#include "reduce.cuh"
#include <atomic>
#include <limits>
namespace tb=mfq_tensor_backend;
// A warp retains four state rows for each of its sixteen value columns.
// Time steps reuse those registers; only rollback prefixes and the final
// state are written. Convolution history remains read-only until completion.
struct GdnWindowSnapshots {float* conv[8]{};float* state[8]{};int confirmed=1;};

template<class Qkv,class Gate,class Output>
__global__ __launch_bounds__(256,1) void gdn_decode_window_kernel(
 const Qkv* __restrict__ qkv,const Gate* __restrict__ output_gate,
 const Gate* __restrict__ alpha,const Gate* __restrict__ beta,
 const float* __restrict__ conv,float* __restrict__ state,const float* __restrict__ weights,
 const float* __restrict__ bias,const float* __restrict__ a_log,const float* __restrict__ norm,
 Output* __restrict__ output,float* __restrict__ next_conv,
 int nk,int nv,int tokens,int layout,bool transposed,bool silu,float conv_eps,double norm_eps,
 int64_t qs,int64_t gs,int64_t as,int64_t ah,int64_t bs,int64_t bh,GdnWindowSnapshots snapshots) {
 constexpr int D=128,Warps=8,Columns=16;
 const int head=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 const int key_head=head/(nv/nk),channels=(2*nk+nv)*D,qk=2*nk*D;
 const int qc=key_head*D+tid,kc=nk*D+qc,vc=qk+head*D+tid;
 __shared__ float query[D],key[D],value[D],attended[D],qsum[256],ksum[256],norm_sum[D];
 __shared__ float decay_value,beta_value,inverse;
 float states[Columns][4];
 #pragma unroll
 for(int c=0;c<Columns;++c) {
  const int column=warp+c*Warps;
  #pragma unroll
  for(int r=0;r<4;++r) {
   const int i=r*32+lane;
   states[c][r]=state[size_t(head)*D*D+(transposed?size_t(column)*D+i:size_t(i)*D+column)];
  }
 }
 for(int t=0;t<tokens;++t) {
  // Every token reads the original history or an already projected QKV row.
  // No head may overwrite shared Q/K history while another still needs it.
  float q=0,k=0,v=0;
  if(tid<D) {
   #pragma unroll
   for(int tap=0;tap<4;++tap) {
    const int from=t+tap-3;
    const float qv=from<0?conv[size_t(from+3)*channels+qc]:float(qkv[int64_t(from)*qs+qc]);
    const float kv=from<0?conv[size_t(from+3)*channels+kc]:float(qkv[int64_t(from)*qs+kc]);
    const float vv=from<0?conv[size_t(from+3)*channels+vc]:float(qkv[int64_t(from)*qs+vc]);
    q+=qv*weights[layout==0?size_t(qc)*4+tap:size_t(tap)*channels+qc];
    k+=kv*weights[layout==0?size_t(kc)*4+tap:size_t(tap)*channels+kc];
    v+=vv*weights[layout==0?size_t(vc)*4+tap:size_t(tap)*channels+vc];
   }
   q=q/(1.f+expf(-q));k=k/(1.f+expf(-k));v=v/(1.f+expf(-v));
   query[tid]=q;key[tid]=k;value[tid]=v;
  }
  qsum[tid]=tid<D?q*q:0;ksum[tid]=tid<D?k*k:0;
  if(tid==0) {
   const float a=float(alpha[int64_t(t)*as+int64_t(head)*ah])+bias[head];
   const float sp=fmaxf(a,0.f)+log1pf(expf(-fabsf(a)));
   decay_value=expf(sp*-expf(a_log[head]));
   beta_value=1.f/(1.f+expf(-float(beta[int64_t(t)*bs+int64_t(head)*bh])));
  }
  __syncthreads();
  if(tid<32) {
   float qr=(qsum[tid]+qsum[tid+128])+(qsum[tid+64]+qsum[tid+192]);
   float kr=(ksum[tid]+ksum[tid+128])+(ksum[tid+64]+ksum[tid+192]);
   qr+=(qsum[tid+32]+qsum[tid+160])+(qsum[tid+96]+qsum[tid+224]);
   kr+=(ksum[tid+32]+ksum[tid+160])+(ksum[tid+96]+ksum[tid+224]);
   #pragma unroll
   for(int stride=16;stride>0;stride>>=1) {
    const float qother=__shfl_down_sync(0xffffffffu,qr,stride),kother=__shfl_down_sync(0xffffffffu,kr,stride);
    if(tid<stride){qr+=qother;kr+=kother;}
   }
   if(tid==0){qsum[0]=qr;ksum[0]=kr;}
  }
  __syncthreads();
  if(tid<D) {
   query[tid]*=1.f/fmaxf(sqrtf(qsum[0]),conv_eps);
   key[tid]*=1.f/fmaxf(sqrtf(ksum[0]),conv_eps);
  }
  __syncthreads();
  const int checkpoint=t+1-snapshots.confirmed;
  float* snapshot_state=checkpoint>=0 && t+1<tokens?snapshots.state[checkpoint]:nullptr;
  const float decay=decay_value,bt=beta_value;
  float keys[4],queries[4];
  #pragma unroll
  for(int r=0;r<4;++r){keys[r]=key[r*32+lane];queries[r]=query[r*32+lane];}
  #pragma unroll
  for(int c=0;c<Columns;++c) {
   const int column=warp+c*Warps;
   float projected=0;
   #pragma unroll
   for(int r=0;r<4;++r)projected+=states[c][r]*keys[r];
   projected=warp_sum(projected);
   const float delta=(value[column]-decay*projected)*bt;
   float result=0;
   #pragma unroll
   for(int r=0;r<4;++r) {
    states[c][r]=__fmaf_rn(decay,states[c][r],__fmul_rn(keys[r],delta));
    result+=states[c][r]*queries[r];
    if(snapshot_state) {
     const int i=r*32+lane;
     snapshot_state[size_t(head)*D*D+(transposed?size_t(column)*D+i:size_t(i)*D+column)]=states[c][r];
    }
   }
   result=warp_sum(result);
   if(lane==0)attended[column]=result*(1.f/sqrtf(float(D)));
  }
  __syncthreads();
  if(tid<D)norm_sum[tid]=__fadd_rn(0.f,__fmul_rn(attended[tid],attended[tid]));
  __syncthreads();
  if(tid<32) {
   float sum=__fadd_rn(__fadd_rn(norm_sum[tid],norm_sum[tid+64]),__fadd_rn(norm_sum[tid+32],norm_sum[tid+96]));
   #pragma unroll
   for(int stride=16;stride>0;stride>>=1) {
    const float other=__shfl_down_sync(0xffffffffu,sum,stride);
    if(tid<stride)sum=__fadd_rn(sum,other);
   }
   if(tid==0)inverse=__fdiv_rn(1.f,__fsqrt_rn(float(double(__fdiv_rn(sum,float(D)))+norm_eps)));
  }
  __syncthreads();
  if(tid<D) {
   const float z=float(output_gate[int64_t(t)*gs+head*D+tid]);
   const float sigmoid=__fdiv_rn(1.f,__fadd_rn(1.f,expf(-z)));
   const float gate=silu?__fmul_rn(z,sigmoid):sigmoid;
   output[(size_t(t)*nv+head)*D+tid]=static_cast<Output>(__fmul_rn(__fmul_rn(__fmul_rn(attended[tid],inverse),norm[tid]),gate));
   float* destination=t+1==tokens?next_conv:checkpoint>=0?snapshots.conv[checkpoint]:nullptr;
   if(destination) {
    #pragma unroll
    for(int tap=0;tap<3;++tap) {
     const int from=t+tap-2;
     destination[size_t(tap)*channels+vc]=from<0?conv[size_t(from+3)*channels+vc]:float(qkv[int64_t(from)*qs+vc]);
     if(head%(nv/nk)==0) {
      destination[size_t(tap)*channels+qc]=from<0?conv[size_t(from+3)*channels+qc]:float(qkv[int64_t(from)*qs+qc]);
      destination[size_t(tap)*channels+kc]=from<0?conv[size_t(from+3)*channels+kc]:float(qkv[int64_t(from)*qs+kc]);
     }
    }
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int c=0;c<Columns;++c) {
  const int column=warp+c*Warps;
  #pragma unroll
  for(int r=0;r<4;++r) {
   const int i=r*32+lane;
   state[size_t(head)*D*D+(transposed?size_t(column)*D+i:size_t(i)*D+column)]=states[c][r];
  }
 }
}

std::vector<tb::Tensor> gdn_decode_window_cuda(const tb::Tensor& qkv,const tb::Tensor& gate,
 const tb::Tensor& alpha,const tb::Tensor& beta,const tb::Tensor& conv,const tb::Tensor& state,
 const tb::Tensor& weights,const tb::Tensor& bias,const tb::Tensor& a_log,const tb::Tensor& norm,
 int nk,int nv,int confirmed,double conv_eps,double norm_eps,bool silu,bool half,bool transposed,
 const std::vector<tb::Tensor>& checkpoint_conv,const std::vector<tb::Tensor>& checkpoint_state) {
 MFQ_RUNTIME_CHECK(qkv.dim()==3 && qkv.size(0)==1 && qkv.size(1)>=2 && qkv.size(1)<=8 &&
  nk>0 && nv>0 && nv<=65535 && nv%nk==0 && int64_t(2)*nk+nv<=std::numeric_limits<int>::max()/128,
  "GDN window heads/sequence invalid");
 const int tokens=int(qkv.size(1)),channels=(2*nk+nv)*128;
 MFQ_RUNTIME_CHECK(confirmed>=1 && confirmed<=tokens &&
  checkpoint_conv.size()==size_t(tokens-confirmed) && checkpoint_state.size()==checkpoint_conv.size(),"GDN window geometry invalid");
 const auto device=qkv.device();
 const auto real=[&](const tb::Tensor& x){return x.is_cuda() && x.device()==device && x.is_contiguous() && x.scalar_type()==tb::kFloat32;};
 MFQ_RUNTIME_CHECK(qkv.is_cuda() && (qkv.scalar_type()==tb::kFloat16 || qkv.scalar_type()==tb::kFloat32) && qkv.size(2)==channels &&
  qkv.stride(2)==1 && qkv.stride(1)>=channels && std::isfinite(conv_eps) && conv_eps>0 && std::isfinite(norm_eps) && norm_eps>0,
  "GDN window QKV/epsilon invalid");
 MFQ_RUNTIME_CHECK(real(conv) && real(state) && real(weights) && real(bias) && real(a_log) && real(norm) &&
  conv.sizes().vec()==std::vector<int64_t>({1,3,channels}) && state.sizes().vec()==std::vector<int64_t>({1,nv,128,128}) &&
  (weights.sizes().vec()==std::vector<int64_t>({channels,1,4}) || weights.sizes().vec()==std::vector<int64_t>({4,channels})) &&
  bias.numel()==nv && a_log.numel()==nv && norm.numel()==128,"GDN window state/weights invalid");
 const auto gate_type=gate.scalar_type();
 const auto projection=[&](const tb::Tensor& x,const std::vector<int64_t>& shape) {
  return x.is_cuda() && x.device()==device && x.scalar_type()==gate_type && x.sizes().vec()==shape &&
   x.stride(1)>0 && x.stride(2)>0;
 };
 MFQ_RUNTIME_CHECK((gate_type==tb::kFloat16 || gate_type==tb::kFloat32) && gate.stride(2)==1 &&
  projection(gate,{1,tokens,nv*128}) && projection(alpha,{1,tokens,nv}) && projection(beta,{1,tokens,nv}),
  "GDN window gate projections invalid");
 std::vector<std::pair<uintptr_t,uintptr_t>> destinations;
 const auto add_destination=[&](const tb::Tensor& x) {
  const auto begin=reinterpret_cast<uintptr_t>(x.data_ptr<float>());
  const auto end=begin+size_t(x.numel())*sizeof(float);
  for(const auto& previous:destinations)MFQ_RUNTIME_CHECK(end<=previous.first || previous.second<=begin,
   "GDN window checkpoint storage overlaps");
  destinations.emplace_back(begin,end);
 };
 add_destination(conv);add_destination(state);
 for(int i=0;i<tokens-confirmed;++i) {
  MFQ_RUNTIME_CHECK(checkpoint_conv[i].defined()==checkpoint_state[i].defined(),
   "GDN window checkpoint buffers must be paired");
  if(!checkpoint_conv[i].defined())continue;
  MFQ_RUNTIME_CHECK(real(checkpoint_conv[i]) && real(checkpoint_state[i]) &&
   checkpoint_conv[i].sizes()==conv.sizes() && checkpoint_state[i].sizes()==state.sizes(),
   "GDN window checkpoint storage invalid");
  add_destination(checkpoint_conv[i]);add_destination(checkpoint_state[i]);
 }
 const MfqCudaGuard guard(device);
 auto output=tb::empty({1,tokens,nv*128},qkv.options().dtype(half?tb::kFloat16:tb::kFloat32));
 auto next_conv=tb::empty_like(conv);GdnWindowSnapshots snapshots;snapshots.confirmed=confirmed;
 for(int i=0;i<tokens-confirmed;++i) {
  if(checkpoint_conv[i].defined()) {snapshots.conv[i]=checkpoint_conv[i].data_ptr<float>();snapshots.state[i]=checkpoint_state[i].data_ptr<float>();}
 }
 const auto launch=[&](auto q,auto g,auto o) {
  using Qkv=decltype(q);using Gate=decltype(g);using Output=decltype(o);
  const auto kernel=gdn_decode_window_kernel<Qkv,Gate,Output>;
  static bool reported=false;
  if(!reported && std::getenv("MFQ_GDN_PROBE_RESOURCES")) {
   hipFuncAttributes a;MFQ_CUDA_CHECK(hipFuncGetAttributes(&a,reinterpret_cast<const void*>(kernel)));reported=true;
   std::fprintf(stderr,"gdn_register_resources registers=%d local=%zu shared=%zu\n",a.numRegs,a.localSizeBytes,a.sharedSizeBytes);
  }
  hipLaunchKernelGGL((gdn_decode_window_kernel<Qkv,Gate,Output>),dim3(nv),dim3(256),0,mfq_current_cuda_stream(),qkv.data_ptr<Qkv>(),gate.data_ptr<Gate>(),alpha.data_ptr<Gate>(),beta.data_ptr<Gate>(),
   conv.data_ptr<float>(),state.data_ptr<float>(),weights.data_ptr<float>(),bias.data_ptr<float>(),a_log.data_ptr<float>(),norm.data_ptr<float>(),
   output.data_ptr<Output>(),next_conv.data_ptr<float>(),nk,nv,tokens,weights.dim()==3?0:1,transposed,silu,float(conv_eps),norm_eps,
   qkv.stride(1),gate.stride(1),alpha.stride(1),alpha.stride(2),beta.stride(1),beta.stride(2),snapshots);
 };
 const auto typed=[&](auto q) {
  if(gate.scalar_type()==tb::kFloat16){if(half)launch(q,__half{},__half{});else launch(q,__half{},float{});}
  else {if(half)launch(q,float{},__half{});else launch(q,float{},float{});}
 };
 if(qkv.scalar_type()==tb::kFloat16)typed(__half{});else typed(float{});
 MFQ_CUDA_CHECK(hipGetLastError());
 static std::atomic<unsigned> seen{0};const unsigned bit=1u<<tokens;
 if(!(seen.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
  "runtime_mtp_gdn_window enabled=1 tokens=%d key_heads=%d value_heads=%d checkpoints=%d\n",tokens,nk,nv,tokens-confirmed);
 return {output,next_conv,state};
}
