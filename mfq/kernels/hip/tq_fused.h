#pragma once
// CUDA TQ row primitives combined for ordinary projection groups and SwiGLU.
namespace mfq_tq_fused {
using namespace mfq_tq_cuda_exact;
template<int DT,int W,int V>__device__ __forceinline__ void group_row(
        ProjectionView m,const S<DT>* x,S<DT>* output,int block) {
    constexpr int R=4/W;
    const int wave=threadIdx.x/64,slot=wave/W,row=block*R+slot;
    float value[1]{};
    if(row<m.rows) {
        if(m.columns)generic_dispatch<DT,true,W,1,V>(m,row,x,0,1,m.input_columns,value);
        else generic_dispatch<DT,false,W,1,V>(m,row,x,0,1,m.input_columns,value);
    }
    for(int d=32;d;d/=2)value[0]+=__shfl_down(value[0],d,64);
    __shared__ float partial[R][W];
    if constexpr(W>1) {
        if(!(threadIdx.x&63))partial[slot][wave%W]=value[0];
        __syncthreads();
        value[0]=(threadIdx.x&63)<W?partial[slot][threadIdx.x&63]:0.f;
        for(int d=32;d;d/=2)value[0]+=__shfl_down(value[0],d,64);
    }
    if(threadIdx.x%(W*64)==0 && row<m.rows)
        output[row]=rounded<DT>(value[0]+(m.bias?m.bias[row]:0.f));
}
template<int DT>__global__ __launch_bounds__(256) void group(
        TqProjectionPlan plan,const S<DT>* x) {
    // Static indexes keep the argument image in registers without a private
    // per-thread copy of the entire plan.
    TqProjectionJob job;int begin;
    if(blockIdx.x<unsigned(plan.jobs[0].block_end)) {job=plan.jobs[0];begin=0;}
    else if(blockIdx.x<unsigned(plan.jobs[1].block_end)) {job=plan.jobs[1];begin=plan.jobs[0].block_end;}
    else if(blockIdx.x<unsigned(plan.jobs[2].block_end)) {job=plan.jobs[2];begin=plan.jobs[1].block_end;}
    else {job=plan.jobs[3];begin=plan.jobs[2].block_end;}
    const auto m=job.weight;auto* out=static_cast<S<DT>*>(job.output);
    const int block=blockIdx.x-begin;
    if(m.rows<=64)group_row<DT,4,16>(m,x,out,block);
    else if(m.rows>=m.input_columns) {
        group_row<DT,1,8>(m,x,out,block);
    } else if(m.input_columns>=8192)group_row<DT,2,16>(m,x,out,block);
    else group_row<DT,1,16>(m,x,out,block);
}
// Concurrent QSA retains the independent native token tile and dot-product order.
template<int T,int V>__global__ __launch_bounds__(256) void group_batch(TqProjectionPlan plan,const float* input,int tokens){
 TqProjectionJob job;int begin;
 if(blockIdx.x<unsigned(plan.jobs[0].block_end)){job=plan.jobs[0];begin=0;}
 else if(blockIdx.x<unsigned(plan.jobs[1].block_end)){job=plan.jobs[1];begin=plan.jobs[0].block_end;}
 else if(blockIdx.x<unsigned(plan.jobs[2].block_end)){job=plan.jobs[2];begin=plan.jobs[1].block_end;}
 else{job=plan.jobs[3];begin=plan.jobs[2].block_end;}
 const auto m=job.weight;const int row=(int(blockIdx.x)-begin)*4+threadIdx.x/64,first=blockIdx.y*T;
 float values[T]{};
 if(row<m.rows){
  if(m.columns)generic_dispatch<2,true,1,T,V>(m,row,input,first,tokens,m.input_columns,values);
  else generic_dispatch<2,false,1,T,V>(m,row,input,first,tokens,m.input_columns,values);
 }
 #pragma unroll
 for(int t=0;t<T;++t){
  #pragma unroll
  for(int shift=32;shift;shift>>=1)values[t]+=__shfl_down(values[t],shift,64);
  if(!(threadIdx.x&63)&&row<m.rows&&first+t<tokens){
   if(m.bias)values[t]+=__ldg(m.bias+row);
   static_cast<float*>(job.output)[size_t(first+t)*m.rows+row]=values[t];
  }
 }
}

template<int DT,int V>__global__ __launch_bounds__(256) void swiglu(
        ProjectionView gate,ProjectionView up,const S<DT>* x,S<DT>* output) {
    const int wave=threadIdx.x/64,lane=threadIdx.x&63,slot=wave/2;
    const int branch=__builtin_amdgcn_readfirstlane(wave&1);
    const int row=blockIdx.x*2+slot;
    const auto m=branch?up:gate;
    float value[1]{};
    if(row<m.rows) {
        if(m.columns)generic_dispatch<DT,true,1,1,V>(m,row,x,0,1,m.input_columns,value);
        else generic_dispatch<DT,false,1,1,V>(m,row,x,0,1,m.input_columns,value);
    }
    for(int d=32;d;d/=2)value[0]+=__shfl_down(value[0],d,64);
    __shared__ float sums[2][2];
    if(!lane)sums[slot][branch]=value[0]+(m.bias && row<m.rows?m.bias[row]:0.f);
    __syncthreads();
    if(threadIdx.x<2 && blockIdx.x*2+threadIdx.x<gate.rows) {
        const float g=number<DT>(rounded<DT>(sums[threadIdx.x][0]));
        const float u=number<DT>(rounded<DT>(sums[threadIdx.x][1]));
        const float activated=number<DT>(rounded<DT>(g/(1.f+expf(-g))));
        output[blockIdx.x*2+threadIdx.x]=rounded<DT>(activated*u);
    }
}
template<int DT,int W,int V>__global__ __launch_bounds__(256) void residual(
        ProjectionView m,const S<DT>* x,const S<DT>* skip,S<DT>* output) {
    constexpr int R=4/W;
    const int wave=threadIdx.x/64,slot=wave/W,row=blockIdx.x*R+slot;
    float value[1]{};
    if(row<m.rows) {
        if(m.columns)generic_dispatch<DT,true,W,1,V>(m,row,x,0,1,m.input_columns,value);
        else generic_dispatch<DT,false,W,1,V>(m,row,x,0,1,m.input_columns,value);
    }
    for(int d=32;d;d/=2)value[0]+=__shfl_down(value[0],d,64);
    __shared__ float partial[R][W];
    if constexpr(W>1) {
        if(!(threadIdx.x&63))partial[slot][wave%W]=value[0];
        __syncthreads();
        value[0]=(threadIdx.x&63)<W?partial[slot][threadIdx.x&63]:0.f;
        for(int d=32;d;d/=2)value[0]+=__shfl_down(value[0],d,64);
    }
    if(threadIdx.x%(W*64)==0 && row<m.rows) {
        const float projected=number<DT>(rounded<DT>(value[0]+(m.bias?m.bias[row]:0.f)));
        output[row]=rounded<DT>(projected+number<DT>(skip[row]));
    }
}
template<int DT>int launch_residual(ProjectionView m,const void* x,const void* skip,void* out,hipStream_t stream) {
    #define RUN(W,V) residual<DT,W,V><<<(m.rows+4/W-1)/(4/W),256,0,stream>>>(m,static_cast<const S<DT>*>(x),static_cast<const S<DT>*>(skip),static_cast<S<DT>*>(out))
    if(m.rows<=64){RUN(4,16);}
    else if(m.rows>=m.input_columns) {
        RUN(1,8);
    } else if(m.input_columns>=8192){RUN(2,16);}
    else{RUN(1,16);}
    #undef RUN
    return int(hipGetLastError());
}
}
extern "C" int mfq_tq_residual(ProjectionView m,const void* x,const void* skip,void* out,
        int dtype,hipStream_t stream) {
    if(!x || !skip || !out || m.rows<1 || m.input_columns<1)return int(hipErrorInvalidValue);
    if(dtype==0)return mfq_tq_fused::launch_residual<0>(m,x,skip,out,stream);
    if(dtype==1)return mfq_tq_fused::launch_residual<1>(m,x,skip,out,stream);
    if(dtype==2)return mfq_tq_fused::launch_residual<2>(m,x,skip,out,stream);
    return int(hipErrorInvalidValue);
}
extern "C" int mfq_tq_group(TqProjectionPlan plan,const void* x,int dtype,hipStream_t stream) {
    if(!x || plan.count<1 || plan.count>4 || dtype<0 || dtype>2)return int(hipErrorInvalidValue);
    int blocks=0;
    for(int j=0;j<plan.count;++j) {
        auto& job=plan.jobs[j];const auto& m=job.weight;
        if(!job.output || m.rows<1 || m.input_columns<1)return int(hipErrorInvalidValue);
        const int rows=m.rows<=64?1:(m.rows<m.input_columns && m.input_columns>=8192?2:4);
        blocks+=(m.rows+rows-1)/rows;job.block_end=blocks;
    }
    for(int j=plan.count;j<4;++j)plan.jobs[j].block_end=blocks;
    #define RUN(D) mfq_tq_fused::group<D><<<blocks,256,0,stream>>>(plan,static_cast<const mfq_tq_cuda_exact::S<D>*>(x))
    if(dtype==0){RUN(0);}else if(dtype==1){RUN(1);}else{RUN(2);}
    #undef RUN
    return int(hipGetLastError());
}
extern "C" int mfq_tq_swiglu(ProjectionView gate,ProjectionView up,const void* x,void* y,
        int dtype,hipStream_t stream) {
    if(!x || !y || gate.rows!=up.rows || gate.input_columns!=up.input_columns || gate.rows<1 ||
       gate.input_columns<1 || dtype<0 || dtype>2)return int(hipErrorInvalidValue);
    #define RUN(D,V) mfq_tq_fused::swiglu<D,V><<<(gate.rows+1)/2,256,0,stream>>>(gate,up,static_cast<const mfq_tq_cuda_exact::S<D>*>(x),static_cast<mfq_tq_cuda_exact::S<D>*>(y))
    if(dtype==0){RUN(0,8);}else if(dtype==1){RUN(1,8);}else{RUN(2,8);}
    #undef RUN
    return int(hipGetLastError());
}

extern "C" int mfq_tq_group_f32_batch(TqProjectionPlan plan,const float* input,int tokens,hipStream_t stream){
 if(!input||plan.count!=4||tokens<2||tokens>6)return int(hipErrorInvalidValue);
 const int expected_rows[4]={12288,512,512,640};int blocks=0;
 for(int i=0;i<4;++i){const auto&m=plan.jobs[i].weight;if(m.input_columns!=2560||m.rows!=expected_rows[i])return int(hipErrorInvalidValue);blocks+=(m.rows+3)/4;plan.jobs[i].block_end=blocks;}
 #define RUN(T,V) mfq_tq_fused::group_batch<T,V><<<dim3(blocks,(tokens+T-1)/T),256,0,stream>>>(plan,input,tokens)
 if(tokens==2){RUN(2,16);}else if(tokens==3||tokens==5||tokens==6){RUN(3,8);}else{RUN(4,8);}
 #undef RUN
 return int(hipGetLastError());
}
