// Port of CUDA tq_q8_tokens.cuh; Q8 activations and its affine book approximation.
#include <type_traits>
#include <climits>
#include <hip/hip_bf16.h>
#include "tq_projection.h"
#include "tq_vector.h"
namespace mfq_tq_q8 {
template<int DT>using S=std::conditional_t<DT==2,float,uint16_t>;
template<int DT>__device__ __forceinline__ float number(S<DT> x) {
    if constexpr(DT==2)return x;
    else return __half2float(__ushort_as_half(x));
}
template<int DT>__device__ __forceinline__ S<DT> rounded(float x) {
    if constexpr(DT==2)return x;
    else return __half_as_ushort(__float2half_rn(x));
}
// The fixed book before its final FP16 rounding is affine in the byte sum
// of state*0x83dcd12d. Center at 510: integers fit signed 10 bits. This avoids
// converting weights to FP16 and back to integers for each token tile.
// Q8 activation rounding dominates the omitted final book rounding.
inline constexpr float TQ_Q8_BOOK_INVERSE=0.00676727294921875f;
inline constexpr float TQ_Q8_BOOK_CENTER=-0.0018157958984375f;

template<int Bits,class Consumer>
__device__ __forceinline__ void q8_integer_book(ProjectionView m,uint64_t d,int col,Consumer& consumer) {
    constexpr int V=16,Step=V*Bits,Align=Step&-Step;
    constexpr int MaxShift=Align<32?32-Align:0;
    constexpr int Words=(MaxShift+(V-1)*Bits+16+31)/32;
    constexpr int Normalized=((V-1)*Bits+16+31)/32;
    const int minimum=(15-(col/16)%16)*Step,first=minimum/32;
    const auto* ring=m.words+uint32_t(d)+(col/256)*(8*Bits);
    uint32_t chunks[Words],normalized[Normalized];
    #pragma unroll
    for(int j=0;j<Words;++j) {
        int at=first+j;if(at>=8*Bits)at-=8*Bits;
        chunks[j]=__ldg(ring+at);
    }
    #pragma unroll
    for(int j=0;j<Normalized;++j)normalized[j]=__funnelshift_r(chunks[j],j+1<Words?chunks[j+1]:0,minimum&31);
    #pragma unroll
    for(int j=0;j<V;j+=2) {
        int centered[2];
        #pragma unroll
        for(int pair=0;pair<2;++pair) {
            constexpr uint32_t Multiplier=0x83dcd12du;
            const int pos=(15-j-pair)*Bits,index=pos/32;
            const uint32_t state=__funnelshift_r(normalized[index],index+1<Normalized?normalized[index+1]:0,pos&31)&65535;
            centered[pair]=int(mfq_tq_vector::dot_bytes(state*Multiplier,0x01010101u,0u))-510;
        }
        consumer(j,centered[0],centered[1]);
    }
}
template<int T>struct Q8DotConsumer {
    uint32_t input[T][4];
    int dot[T]{};
    __device__ __forceinline__ void operator()(int j,int qa,int qb) {
        const uint32_t pair=uint32_t(uint16_t(qa))|(uint32_t(uint16_t(qb))<<16);
        #pragma unroll
        for(int t=0;t<T;++t) {
            const auto bytes=input[t][j/4] >> ((j&2)*8);
            using I16x2=short __attribute__((ext_vector_type(2)));
            const I16x2 weights{short(qa),short(qb)};
            const I16x2 activation{short(int8_t(bytes)),short(int8_t(bytes>>8))};
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_sdot2)
            dot[t]=__builtin_amdgcn_sdot2(weights,activation,dot[t],false);
#else
            dot[t]+=qa*int8_t(bytes)+qb*int8_t(bytes>>8);
#endif
        }
    }
};
template<int T,int Bits>
__device__ __forceinline__ void q8_row(ProjectionView m,int row,const uint32_t* input,
        int qcolumns,float (&result)[T]) {
    const auto d=__ldg(m.descriptors+row);const int lane=threadIdx.x&63;
    const size_t ng=m.columns?0:(size_t(m.stored_columns)+(1<<m.gslog)-1)>>m.gslog;
    // One 512-column tile stores coalesced Q8 planes followed by paired
    // gain/offset planes. Constant tile strides avoid retaining three
    // independent input pointers and their token strides in registers.
    for(int col=lane*16;col<m.input_columns;col+=1024) {
        const int tile=col/512,group=(col%512)/16;
        constexpr int tile_groups=32;
        const auto* block=input+size_t(tile)*32*T*6;
        Q8DotConsumer<T> consumer;
        #pragma unroll
        for(int t=0;t<T;++t) {
            const auto v=__ldg(reinterpret_cast<const uint4*>(block+(size_t(t)*tile_groups+group)*4));
            consumer.input[t][0]=v.x;consumer.input[t][1]=v.y;consumer.input[t][2]=v.z;consumer.input[t][3]=v.w;
        }
        q8_integer_book<Bits>(m,d,col,consumer);
        float scale=__half2float(__ushort_as_half(uint16_t(d>>40)));
        if(!m.columns) {
            const size_t index=size_t(row)*ng+(col>>m.gslog);
            const uint8_t code=__ldg(static_cast<const uint8_t*>(m.scales)+(index>>1));
            scale*=__half2float(__ldg(static_cast<const half*>(m.relative)+((code>>((index&1)*4))&15)));
        }
        #pragma unroll
        for(int t=0;t<T;++t) {
            const auto coefficients=__ldg(reinterpret_cast<const float2*>(block+4*T*tile_groups)+size_t(t)*tile_groups+group);
            result[t]=__fmaf_rn(scale,__fmaf_rn(coefficients.x,float(consumer.dot[t]),coefficients.y),result[t]);
        }
    }
}
template<int T>
__device__ __forceinline__ void q8_dispatch(ProjectionView m,int row,const uint32_t* input,
        int qcolumns,float (&result)[T]) {
    const auto d=__ldg(m.descriptors+row);
    #define CASE(B) case B-1:q8_row<T,B>(m,row,input,qcolumns,result);return
    switch((d>>32)&7){CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);}
    #undef CASE
}

template<int DT> __device__ __forceinline__ void prepare_one(ProjectionView m,const mfq_tq_q8::S<DT>* x,
        uint32_t* output,int tokens,int qc) {
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int element=index%16;
    const uint64_t group=index/16;
    const int token=group/(qc/16),col=(group%(qc/16))*16+element;
    float value=0;
    if(token<tokens && col<m.input_columns) {
        value=mfq_tq_q8::number<DT>(x[uint64_t(token)*m.input_columns+col]);
        if(m.columns)value*=__half2float(static_cast<const half*>(m.scales)[col]);
    }
    float peak=fabsf(value);
    #pragma unroll
    for(int d=8;d;d/=2)peak=fmaxf(peak,__shfl_xor(peak,d,16));
    const float scale=peak>0?peak/127.f:1.f;
    int q=__float2int_rn(value/scale);q=max(-127,min(127,q));
    int sum=q;
    #pragma unroll
    for(int d=8;d;d/=2)sum+=__shfl_xor(sum,d,16);
    const uint32_t byte=uint8_t(q);
    const uint32_t packed=byte|(__shfl_down(byte,1,64)<<8)|
        (__shfl_down(byte,2,64)<<16)|(__shfl_down(byte,3,64)<<24);
    if(token<tokens) {
        const int tile=col/512,local=(col%512)/16,at=token*32+local;
        const uint64_t base=uint64_t(tile)*32*tokens*6;
        if(!(element&3))output[base+uint64_t(at)*4+element/4]=packed;
        if(!element) {
            float* c=reinterpret_cast<float*>(output+base+4*tokens*32+uint64_t(at)*2);
            c[0]=scale*TQ_Q8_BOOK_INVERSE;
            c[1]=float(sum)*TQ_Q8_BOOK_CENTER*scale;
        }
    }
}
template<int DT> __global__ void prepare(ProjectionView m,const S<DT>* x,
        uint32_t* output,int tokens,int qc) {
    prepare_one<DT>(m,x,output,tokens,qc);
}
template<int DT,int T> __global__ __launch_bounds__(256) void project(
        ProjectionView m,const uint32_t* input,mfq_tq_q8::S<DT>* output,int qc) {
    const int row=blockIdx.x*4+threadIdx.x/64;
    if(row>=m.rows)return;
    float value[T]{};
    q8_dispatch<T>(m,row,input,qc,value);
    #pragma unroll
    for(int t=0;t<T;++t) {
        #pragma unroll
        for(int d=32;d;d/=2)value[t]+=__shfl_down(value[t],d,64);
        if(!(threadIdx.x&63))output[uint64_t(t)*m.rows+row]=
            mfq_tq_q8::rounded<DT>(value[t]+(m.bias?m.bias[row]:0.f));
    }
}
template<int DT> int launch(ProjectionView m,const void* x,void* y,int tokens,
        uint32_t* scratch,int qc,hipStream_t stream) {
    prepare<DT><<<(uint64_t(tokens)*qc+255)/256,256,0,stream>>>(m,
        static_cast<const mfq_tq_q8::S<DT>*>(x),scratch,tokens,qc);
    const auto code=hipGetLastError();if(code!=hipSuccess)return int(code);
    #define RUN(T) project<DT,T><<<(m.rows+3)/4,256,0,stream>>>(m,scratch,static_cast<mfq_tq_q8::S<DT>*>(y),qc)
    if(tokens==1){RUN(1);}else if(tokens==2){RUN(2);}else if(tokens==3){RUN(3);}else{RUN(4);}
    #undef RUN
    return int(hipGetLastError());
}
} // namespace mfq_tq_q8
extern "C" int mfq_tq_q8_tokens(ProjectionView m,const void* x,void* y,int tokens,
        int dtype,void* scratch,uint64_t bytes,hipStream_t stream) {
    if(!x || !y || x==y || tokens<1 || tokens>4 || (dtype!=0 && dtype!=2) ||
       m.rows<1 || m.input_columns<1 || m.input_columns>INT_MAX-511 ||
       m.input_columns>m.stored_columns || m.stored_columns%256 ||
       !m.words || !m.descriptors || !m.scales || !m.relative ||
       (!m.columns && (m.gslog<4 || m.gslog>20)))return -1;
    const int qc=int((uint64_t(m.input_columns)+511)/512*512);
    const uint64_t required=uint64_t(tokens)*(qc/16)*24;
    if(!scratch || scratch==x || scratch==y || (uintptr_t(scratch)&15) || bytes<required)return -1;
    if(dtype==0)return mfq_tq_q8::launch<0>(m,x,y,tokens,static_cast<uint32_t*>(scratch),qc,stream);
    return mfq_tq_q8::launch<2>(m,x,y,tokens,static_cast<uint32_t*>(scratch),qc,stream);
}

namespace mfq_tq_q8 {
template<int DT>__global__ void prepare_group(TqProjectionPlan plan,const S<DT>* x,uint32_t* output,int qc) {
    ProjectionView m;
    if(blockIdx.y==0)m=plan.jobs[0].weight;
    else if(blockIdx.y==1)m=plan.jobs[1].weight;
    else if(blockIdx.y==2)m=plan.jobs[2].weight;
    else m=plan.jobs[3].weight;
    prepare_one<DT>(m,x,output+uint64_t(blockIdx.y)*(qc/16)*6,1,qc);
}
template<int DT>__global__ __launch_bounds__(256) void group_product(
        TqProjectionPlan plan,const uint32_t* input,int qc) {
    TqProjectionJob job;int begin,index;
    if(blockIdx.x<unsigned(plan.jobs[0].block_end)){job=plan.jobs[0];begin=0;index=0;}
    else if(blockIdx.x<unsigned(plan.jobs[1].block_end)){job=plan.jobs[1];begin=plan.jobs[0].block_end;index=1;}
    else if(blockIdx.x<unsigned(plan.jobs[2].block_end)){job=plan.jobs[2];begin=plan.jobs[1].block_end;index=2;}
    else{job=plan.jobs[3];begin=plan.jobs[2].block_end;index=3;}
    const auto m=job.weight;const int row=(blockIdx.x-begin)*4+threadIdx.x/64;
    if(row>=m.rows)return;
    float value[1]{};
    q8_dispatch<1>(m,row,input+uint64_t(index)*(qc/16)*6,qc,value);
    for(int d=32;d;d/=2)value[0]+=__shfl_down(value[0],d,64);
    if(!(threadIdx.x&63))static_cast<S<DT>*>(job.output)[row]=
        rounded<DT>(value[0]+(m.bias?m.bias[row]:0.f));
}
template<int DT>__global__ __launch_bounds__(256) void swiglu_product(
        ProjectionView gate,ProjectionView up,const uint32_t* input,S<DT>* output,int qc) {
    const int wave=threadIdx.x/64,lane=threadIdx.x&63,slot=wave/2;
    const int branch=__builtin_amdgcn_readfirstlane(wave&1),row=blockIdx.x*2+slot;
    const auto m=branch?up:gate;
    float value[1]{};
    if(row<m.rows)q8_dispatch<1>(m,row,input+uint64_t(branch)*(qc/16)*6,qc,value);
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
static bool valid(ProjectionView m) {
    return m.rows>0 && m.input_columns>0 && m.input_columns<=INT_MAX-511 &&
        m.input_columns<=m.stored_columns && m.stored_columns%256==0 &&
        m.words && m.descriptors && m.scales && m.relative && (m.columns || (m.gslog>=4 && m.gslog<=20));
}
template<int DT>int launch_group(TqProjectionPlan plan,const void* x,int qc,int blocks,
        uint32_t* scratch,hipStream_t stream) {
    prepare_group<DT><<<dim3((qc+255)/256,plan.count),256,0,stream>>>(plan,static_cast<const S<DT>*>(x),scratch,qc);
    auto code=hipGetLastError();if(code!=hipSuccess)return int(code);
    group_product<DT><<<blocks,256,0,stream>>>(plan,scratch,qc);
    return int(hipGetLastError());
}
template<int DT>int launch_swiglu(TqProjectionPlan plan,const void* x,void* y,int qc,
        uint32_t* scratch,hipStream_t stream) {
    prepare_group<DT><<<dim3((qc+255)/256,2),256,0,stream>>>(plan,static_cast<const S<DT>*>(x),scratch,qc);
    auto code=hipGetLastError();if(code!=hipSuccess)return int(code);
    swiglu_product<DT><<<(plan.jobs[0].weight.rows+1)/2,256,0,stream>>>(plan.jobs[0].weight,
        plan.jobs[1].weight,scratch,static_cast<S<DT>*>(y),qc);
    return int(hipGetLastError());
}
}
extern "C" int mfq_tq_q8_group(TqProjectionPlan plan,const void* x,int dtype,void* scratch,
        uint64_t bytes,hipStream_t stream) {
    if(plan.count<1 || plan.count>4 || !x || !scratch || (uintptr_t(scratch)&15) ||
       scratch==x || (dtype!=0 && dtype!=2))return -1;
    const int width=plan.jobs[0].weight.input_columns;
    int blocks=0;
    for(int j=0;j<plan.count;++j) {
        auto& job=plan.jobs[j];
        if(!mfq_tq_q8::valid(job.weight) || job.weight.input_columns!=width ||
           !job.output || job.output==x || job.output==scratch)return -1;
        blocks+=(job.weight.rows+3)/4;job.block_end=blocks;
    }
    for(int j=plan.count;j<4;++j)plan.jobs[j].block_end=blocks;
    const int qc=(width+511)/512*512;
    if(bytes<uint64_t(plan.count)*(qc/16)*24)return -1;
    if(dtype==0)return mfq_tq_q8::launch_group<0>(plan,x,qc,blocks,static_cast<uint32_t*>(scratch),stream);
    return mfq_tq_q8::launch_group<2>(plan,x,qc,blocks,static_cast<uint32_t*>(scratch),stream);
}
extern "C" int mfq_tq_q8_swiglu(ProjectionView gate,ProjectionView up,const void* x,void* y,
        int dtype,void* scratch,uint64_t bytes,hipStream_t stream) {
    if(!mfq_tq_q8::valid(gate) || !mfq_tq_q8::valid(up) || gate.rows!=up.rows ||
       gate.input_columns!=up.input_columns || !x || !y || !scratch || x==y ||
       scratch==x || scratch==y || (uintptr_t(scratch)&15) || (dtype!=0 && dtype!=2))return -1;
    const int qc=(gate.input_columns+511)/512*512;
    if(bytes<2*uint64_t(qc/16)*24)return -1;
    TqProjectionPlan plan{};plan.count=2;plan.jobs[0].weight=gate;plan.jobs[1].weight=up;
    if(dtype==0)return mfq_tq_q8::launch_swiglu<0>(plan,x,y,qc,static_cast<uint32_t*>(scratch),stream);
    return mfq_tq_q8::launch_swiglu<2>(plan,x,y,qc,static_cast<uint32_t*>(scratch),stream);
}
