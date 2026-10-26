#include "mfq_tensor_backend.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <type_traits>
#include <climits>
#include "tq_projection.h"
#include "tq_vector.h"
#define MFQ_TQ_ROW_HELPERS_ONLY
#include "tq_float_tokens.h"
#undef MFQ_TQ_ROW_HELPERS_ONLY

namespace {
__global__ __launch_bounds__(256) void tq_router_gate_kernel(ProjectionView router,const uint16_t* gate,
        const uint16_t* input,uint16_t* logits,float* coefficient) {
    using namespace mfq_tq_cuda_exact;
    const int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
    if(!blockIdx.x) {
        float sum=0.f;
        for(int column=tid;column<router.input_columns;column+=256)
            sum=__fmaf_rn(__uint_as_float(uint32_t(gate[column])<<16),number<0>(input[column]),sum);
        for(int shift=32;shift;shift>>=1)sum+=__shfl_down(sum,shift,64);
        __shared__ float partial[4];if(!lane)partial[wave]=sum;
        __syncthreads();
        if(!tid) {
            float total=0.f;for(int i=0;i<4;++i)total=__fadd_rn(total,partial[i]);
            coefficient[0]=1.f/(1.f+expf(-total));
        }
        return;
    }
    const int row=(int(blockIdx.x)-1)*4+wave;
    float value[1]{};
    if(row<router.rows) {
        if(router.columns)generic_dispatch<0,true,1,1,16>(router,row,input,0,1,router.input_columns,value);
        else generic_dispatch<0,false,1,1,16>(router,row,input,0,1,router.input_columns,value);
    }
    for(int shift=32;shift;shift>>=1)value[0]+=__shfl_down(value[0],shift,64);
    if(!lane && row<router.rows)logits[row]=rounded<0>(value[0]+(router.bias?router.bias[row]:0.f));
}

__global__ void tq_shared_add_kernel(const uint16_t* routed,const uint16_t* shared,
        const float* coefficient,float* output,int width) {
    const int column=int(blockIdx.x)*256+threadIdx.x;
    if(column<width)output[column]=__fadd_rn(mfq_tq_cuda_exact::number<0>(routed[column]),
        __fmul_rn(mfq_tq_cuda_exact::number<0>(shared[column]),coefficient[0]));
}

__global__ __launch_bounds__(256) void tq_moe_inputs_kernel(ProjectionView router,ProjectionView gate,ProjectionView up,
 const uint16_t* shared_gate,const uint16_t* input,uint16_t* logits,float* coefficient,uint16_t* output){
 const int group=int(blockIdx.x),router_groups=1+(router.rows+3)/4;
 if(group<router_groups){

    using namespace mfq_tq_cuda_exact;
    const int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
    if(!group) {
        float sum=0.f;
        for(int column=tid;column<router.input_columns;column+=256)
            sum=__fmaf_rn(__uint_as_float(uint32_t(shared_gate[column])<<16),number<0>(input[column]),sum);
        for(int shift=32;shift;shift>>=1)sum+=__shfl_down(sum,shift,64);
        __shared__ float partial[4];if(!lane)partial[wave]=sum;
        __syncthreads();
        if(!tid) {
            float total=0.f;for(int i=0;i<4;++i)total=__fadd_rn(total,partial[i]);
            coefficient[0]=1.f/(1.f+expf(-total));
        }
        return;
    }
    const int row=(int(group)-1)*4+wave;
    float value[1]{};
    if(row<router.rows) {
        if(router.columns)generic_dispatch<0,true,1,1,16>(router,row,input,0,1,router.input_columns,value);
        else generic_dispatch<0,false,1,1,16>(router,row,input,0,1,router.input_columns,value);
    }
    for(int shift=32;shift;shift>>=1)value[0]+=__shfl_down(value[0],shift,64);
    if(!lane && row<router.rows)logits[row]=rounded<0>(value[0]+(router.bias?router.bias[row]:0.f));

 }else{const int local_group=group-router_groups;

    using namespace mfq_tq_cuda_exact;
    const int wave=threadIdx.x/64,lane=threadIdx.x&63,role=wave&1,slot=wave/2,row=int(local_group)*2+slot;
    const auto weight=role?up:gate;
    float value[1]{};
    if(row<weight.rows) {
        if(weight.columns)generic_dispatch<0,true,1,1,16>(weight,row,input,0,1,weight.input_columns,value);
        else generic_dispatch<0,false,1,1,16>(weight,row,input,0,1,weight.input_columns,value);
    }
    for(int shift=32;shift;shift>>=1)value[0]+=__shfl_down(value[0],shift,64);
    __shared__ float joined[2][2];
    if(!lane && row<weight.rows)joined[slot][role]=number<0>(rounded<0>(value[0]+(weight.bias?weight.bias[row]:0.f)));
    __syncthreads();
    if(!role && !lane && row<weight.rows) {
        const float g=joined[slot][0],u=joined[slot][1];
        const float sigmoid=number<0>(rounded<0>(1.f/(1.f+expf(-g))));
        const float silu=number<0>(rounded<0>(__fmul_rn(g,sigmoid)));
        output[row]=rounded<0>(__fmul_rn(silu,u));
    }

 }
}
__global__ __launch_bounds__(256) void tq_shared_finish_kernel(ProjectionView down,
 const uint16_t* hidden,const uint16_t* pairs,const float* weights,const float* coefficient,float* output,int routes){
 using namespace mfq_tq_cuda_exact;
 const int wave=threadIdx.x/64,lane=threadIdx.x&63,row=int(blockIdx.x)*4+wave;
 float value[1]{};
 if(row<down.rows){
  if(down.columns)generic_dispatch<0,true,1,1,8>(down,row,hidden,0,1,down.input_columns,value);
  else generic_dispatch<0,false,1,1,8>(down,row,hidden,0,1,down.input_columns,value);
 }
 for(int shift=32;shift;shift>>=1)value[0]+=__shfl_down(value[0],shift,64);
 if(!lane&&row<down.rows){
  const float shared=number<0>(rounded<0>(value[0]+(down.bias?down.bias[row]:0.f)));
  float sum=0.f;
  for(int r=0;r<routes;++r)sum=__fadd_rn(sum,__fmul_rn(number<0>(pairs[size_t(r)*down.rows+row]),weights[r]));
  const float routed=number<0>(rounded<0>(sum));
  output[row]=__fadd_rn(routed,__fmul_rn(shared,coefficient[0]));
 }
}

template<int C> __global__ __launch_bounds__(256) void tq_shared_finish_batch_kernel(ProjectionView down,
 const uint16_t* hidden,const uint16_t* pairs,const float* weights,const mfq_tq_cuda_exact::S<C>* coefficient,float* output,int routes,int tokens){
 using namespace mfq_tq_cuda_exact;
 const int wave=threadIdx.x/64,lane=threadIdx.x&63,row=int(blockIdx.x)*4+wave,first=int(blockIdx.y)*2;
 float value[2]{};
 if(row<down.rows){
  if(down.columns)generic_dispatch<0,true,1,2,16>(down,row,hidden,first,tokens,down.input_columns,value);
  else generic_dispatch<0,false,1,2,16>(down,row,hidden,first,tokens,down.input_columns,value);
 }
 #pragma unroll
 for(int t=0;t<2;++t){
  #pragma unroll
  for(int shift=32;shift;shift>>=1)value[t]+=__shfl_down(value[t],shift,64);
 }
 if(!lane&&row<down.rows){
  #pragma unroll
  for(int t=0;t<2;++t)if(first+t<tokens){
   const int token=first+t;
   const float shared=number<0>(rounded<0>(value[t]+(down.bias?down.bias[row]:0.f)));
   float sum=0.f;
   for(int r=0;r<routes;++r)sum=__fadd_rn(sum,__fmul_rn(number<0>(pairs[(size_t(token)*routes+r)*down.rows+row]),weights[token*routes+r]));
   const float routed=number<0>(rounded<0>(sum));
   output[size_t(token)*down.rows+row]=__fadd_rn(routed,__fmul_rn(shared,number<C>(coefficient[token])));
  }
 }
}

__global__ __launch_bounds__(256) void tq_shared_swiglu_batch_kernel(ProjectionView gate,ProjectionView up,
 const uint16_t* input,uint16_t* output,int tokens){
 using namespace mfq_tq_cuda_exact;
 const int wave=threadIdx.x/64,lane=threadIdx.x&63,role=wave&1,slot=wave/2,row=int(blockIdx.x)*2+slot,first=blockIdx.y*2;
 const auto weight=role?up:gate;
 float value[2]{};
 if(row<weight.rows){
  if(weight.columns)generic_dispatch<0,true,1,2,16>(weight,row,input,first,tokens,weight.input_columns,value);
  else generic_dispatch<0,false,1,2,16>(weight,row,input,first,tokens,weight.input_columns,value);
 }
 #pragma unroll
 for(int t=0;t<2;++t){
  #pragma unroll
  for(int shift=32;shift;shift>>=1)value[t]+=__shfl_down(value[t],shift,64);
 }
 __shared__ float joined[2][2][2];
 if(!lane && row<weight.rows){
  #pragma unroll
  for(int t=0;t<2;++t)joined[slot][role][t]=number<0>(rounded<0>(value[t]+(weight.bias?weight.bias[row]:0.f)));
 }
 __syncthreads();
 if(!role && !lane && row<weight.rows){
  #pragma unroll
  for(int t=0;t<2;++t)if(first+t<tokens){
   const float g=joined[slot][0][t],u=joined[slot][1][t];
   const float sigmoid=number<0>(rounded<0>(1.f/(1.f+expf(-g))));
   const float silu=number<0>(rounded<0>(__fmul_rn(g,sigmoid)));
   output[size_t(first+t)*gate.rows+row]=rounded<0>(__fmul_rn(silu,u));
  }
 }
}

__global__ __launch_bounds__(256) void tq_shared_swiglu_kernel(ProjectionView gate,ProjectionView up,
        const uint16_t* input,uint16_t* output) {
    using namespace mfq_tq_cuda_exact;
    const int wave=threadIdx.x/64,lane=threadIdx.x&63,role=wave&1,slot=wave/2,row=int(blockIdx.x)*2+slot;
    const auto weight=role?up:gate;
    float value[1]{};
    if(row<weight.rows) {
        if(weight.columns)generic_dispatch<0,true,1,1,16>(weight,row,input,0,1,weight.input_columns,value);
        else generic_dispatch<0,false,1,1,16>(weight,row,input,0,1,weight.input_columns,value);
    }
    for(int shift=32;shift;shift>>=1)value[0]+=__shfl_down(value[0],shift,64);
    __shared__ float joined[2][2];
    if(!lane && row<weight.rows)joined[slot][role]=number<0>(rounded<0>(value[0]+(weight.bias?weight.bias[row]:0.f)));
    __syncthreads();
    if(!role && !lane && row<weight.rows) {
        const float g=joined[slot][0],u=joined[slot][1];
        const float sigmoid=number<0>(rounded<0>(1.f/(1.f+expf(-g))));
        const float silu=number<0>(rounded<0>(__fmul_rn(g,sigmoid)));
        output[row]=rounded<0>(__fmul_rn(silu,u));
    }
}
}

std::vector<mfq_tensor_backend::Tensor> tq_router_gate_cuda(ProjectionView router,
        const mfq_tensor_backend::Tensor& gate,const mfq_tensor_backend::Tensor& input) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type()==tb::kFloat16 &&
        input.numel()==router.input_columns && input.size(-1)==router.input_columns && router.rows>0 &&
        gate.device()==input.device() && gate.is_contiguous() && gate.scalar_type()==tb::kBFloat16 &&
        gate.dim()==2 && gate.size(0)==1 && gate.size(1)==router.input_columns,"invalid TQ router/shared gate geometry");
    MfqCudaGuard guard(input.device());auto shape=input.sizes().vec();shape.back()=router.rows;
    auto logits=tb::empty(shape,input.options());shape.back()=1;
    auto coefficient=tb::empty(shape,input.options().dtype(tb::kFloat32));
    tq_router_gate_kernel<<<1+(router.rows+3)/4,256,0,mfq_current_cuda_stream()>>>(router,
        reinterpret_cast<const uint16_t*>(gate.data_ptr()),reinterpret_cast<const uint16_t*>(input.data_ptr()),
        reinterpret_cast<uint16_t*>(logits.data_ptr()),coefficient.data_ptr<float>());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return {logits,coefficient};
}

std::vector<mfq_tensor_backend::Tensor> tq_moe_inputs_cuda(ProjectionView router,
 const mfq_tensor_backend::Tensor& shared_gate,ProjectionView gate,ProjectionView up,
 const mfq_tensor_backend::Tensor& input){
 namespace tb=mfq_tensor_backend;
 MFQ_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type()==tb::kFloat16 &&
  input.numel()==router.input_columns && input.size(-1)==router.input_columns && router.rows>0 &&
  shared_gate.device()==input.device() && shared_gate.is_contiguous() && shared_gate.scalar_type()==tb::kBFloat16 &&
  shared_gate.dim()==2 && shared_gate.size(0)==1 && shared_gate.size(1)==router.input_columns &&
  gate.rows>0 && gate.rows==up.rows && gate.input_columns==router.input_columns &&
  up.input_columns==router.input_columns,"invalid joint TQ MoE input projections");
 MfqCudaGuard guard(input.device());auto shape=input.sizes().vec();shape.back()=router.rows;
 auto logits=tb::empty(shape,input.options());shape.back()=1;
 auto coefficient=tb::empty(shape,input.options().dtype(tb::kFloat32));shape.back()=gate.rows;
 auto hidden=tb::empty(shape,input.options());
 tq_moe_inputs_kernel<<<1+(router.rows+3)/4+(gate.rows+1)/2,256,0,mfq_current_cuda_stream()>>>(
  router,gate,up,reinterpret_cast<const uint16_t*>(shared_gate.data_ptr()),
  reinterpret_cast<const uint16_t*>(input.data_ptr()),reinterpret_cast<uint16_t*>(logits.data_ptr()),
  coefficient.data_ptr<float>(),reinterpret_cast<uint16_t*>(hidden.data_ptr()));
 MFQ_CUDA_KERNEL_LAUNCH_CHECK();return {logits,coefficient,hidden};
}

mfq_tensor_backend::Tensor tq_shared_add_cuda(const mfq_tensor_backend::Tensor& routed,
        const mfq_tensor_backend::Tensor& shared,const mfq_tensor_backend::Tensor& coefficient) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(routed.is_cuda() && routed.is_contiguous() && shared.is_contiguous() &&
        coefficient.is_contiguous() && shared.device()==routed.device() && coefficient.device()==routed.device() &&
        routed.scalar_type()==tb::kFloat16 && shared.scalar_type()==tb::kFloat16 &&
        coefficient.scalar_type()==tb::kFloat32 && shared.sizes()==routed.sizes() && coefficient.numel()==1 &&
        routed.numel()>0 && routed.numel()<=INT_MAX,"invalid TQ shared output geometry");
    MfqCudaGuard guard(routed.device());auto output=tb::empty(routed.sizes(),routed.options().dtype(tb::kFloat32));
    tq_shared_add_kernel<<<(routed.numel()+255)/256,256,0,mfq_current_cuda_stream()>>>(
        reinterpret_cast<const uint16_t*>(routed.data_ptr()),reinterpret_cast<const uint16_t*>(shared.data_ptr()),
        coefficient.data_ptr<float>(),output.data_ptr<float>(),int(routed.numel()));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

mfq_tensor_backend::Tensor tq_shared_finish_cuda(ProjectionView down,
 const mfq_tensor_backend::Tensor& hidden,const mfq_tensor_backend::Tensor& pairs,
 const mfq_tensor_backend::Tensor& weights,const mfq_tensor_backend::Tensor& coefficient){
 namespace tb=mfq_tensor_backend;
 MFQ_RUNTIME_CHECK(hidden.is_cuda() && hidden.is_contiguous() && hidden.scalar_type()==tb::kFloat16 &&
  hidden.dim()==2 && hidden.size(0)>=1 && hidden.size(0)<=6 && hidden.size(1)==down.input_columns && down.input_columns>0 &&
  down.input_columns%8==0 && down.rows>=down.input_columns && pairs.device()==hidden.device() &&
  pairs.is_contiguous() && pairs.scalar_type()==tb::kFloat16 && pairs.dim()==3 && pairs.size(0)==hidden.size(0) &&
  pairs.size(1)>0 && pairs.size(1)<=INT_MAX && pairs.size(2)==down.rows && weights.device()==hidden.device() &&
  weights.is_contiguous() && weights.scalar_type()==tb::kFloat32 && weights.numel()==pairs.size(0)*pairs.size(1) &&
  coefficient.device()==hidden.device() && coefficient.is_contiguous() &&
  (coefficient.scalar_type()==tb::kFloat32 || (hidden.size(0)>1 && coefficient.scalar_type()==tb::kBFloat16)) &&
  coefficient.numel()==hidden.size(0),"invalid TQ shared Down/reduction geometry");
 MfqCudaGuard guard(hidden.device());const int tokens=int(hidden.size(0));
 auto output=tb::empty({tokens,down.rows},hidden.options().dtype(tb::kFloat32));
 if(tokens==1)tq_shared_finish_kernel<<<(down.rows+3)/4,256,0,mfq_current_cuda_stream()>>>(down,
  reinterpret_cast<const uint16_t*>(hidden.data_ptr()),reinterpret_cast<const uint16_t*>(pairs.data_ptr()),
  weights.data_ptr<float>(),coefficient.data_ptr<float>(),output.data_ptr<float>(),int(pairs.size(1)));
 else {
  const dim3 grid((down.rows+3)/4,(tokens+1)/2);
  #define FINISH(C,P) tq_shared_finish_batch_kernel<C><<<grid,256,0,mfq_current_cuda_stream()>>>(down,\
   reinterpret_cast<const uint16_t*>(hidden.data_ptr()),reinterpret_cast<const uint16_t*>(pairs.data_ptr()),\
   weights.data_ptr<float>(),P,output.data_ptr<float>(),int(pairs.size(1)),tokens)
  if(coefficient.scalar_type()==tb::kBFloat16){FINISH(1,reinterpret_cast<const uint16_t*>(coefficient.data_ptr()));}
  else {FINISH(2,coefficient.data_ptr<float>());}
  #undef FINISH
 }
 MFQ_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

mfq_tensor_backend::Tensor tq_shared_swiglu_cuda(ProjectionView gate,ProjectionView up,
        const mfq_tensor_backend::Tensor& input) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type()==tb::kFloat16 &&
        gate.input_columns>0 && input.size(-1)==gate.input_columns && input.numel()/gate.input_columns>=1 &&
        input.numel()/gate.input_columns<=6 && gate.rows==up.rows && gate.input_columns==up.input_columns && gate.rows>0,
        "invalid TQ shared SwiGLU geometry");
    MfqCudaGuard guard(input.device());auto shape=input.sizes().vec();shape.back()=gate.rows;
    auto output=tb::empty(shape,input.options());
    const int tokens=int(input.numel()/gate.input_columns);
    if(tokens==1)tq_shared_swiglu_kernel<<<(gate.rows+1)/2,256,0,mfq_current_cuda_stream()>>>(gate,up,
        reinterpret_cast<const uint16_t*>(input.data_ptr()),reinterpret_cast<uint16_t*>(output.data_ptr()));
    else tq_shared_swiglu_batch_kernel<<<dim3((gate.rows+1)/2,(tokens+1)/2),256,0,mfq_current_cuda_stream()>>>(gate,up,
        reinterpret_cast<const uint16_t*>(input.data_ptr()),reinterpret_cast<uint16_t*>(output.data_ptr()),tokens);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
