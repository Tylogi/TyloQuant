
#pragma once
#include <climits>

// The fixed book before its final FP16 rounding is affine in the byte sum
// of state*0x83dcd12d. Center at 510: integers fit signed 10 bits. This avoids
// converting weights to FP16 and back to integers for each token tile.
// Q8 activation rounding dominates the omitted final book rounding.
inline constexpr float TQ_Q8_BOOK_INVERSE=0.00676727294921875f;
inline constexpr float TQ_Q8_BOOK_CENTER=-0.0018157958984375f;

template<int Bits,class Consumer>
__device__ __forceinline__ void q8_integer_book(TrellisProjectionView m,uint64_t d,int col,Consumer& consumer) {
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
            centered[pair]=int(__dp4a(state*Multiplier,0x01010101u,0u))-510;
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
            if(!(j&2))asm("dp2a.lo.s32.s32 %0, %1, %2, %0;" : "+r"(dot[t]) : "r"(pair),"r"(input[t][j/4]));
            else asm("dp2a.hi.s32.s32 %0, %1, %2, %0;" : "+r"(dot[t]) : "r"(pair),"r"(input[t][j/4]));
        }
    }
};
template<int T,int Bits>
__device__ __forceinline__ void q8_row(TrellisProjectionView m,int row,const uint32_t* input,
        int qcolumns,float (&result)[T]) {
    const auto d=__ldg(m.descriptors+row);const int lane=threadIdx.x&31;
    const size_t ng=m.columns?0:(size_t(m.stored_columns)+(1<<m.gslog)-1)>>m.gslog;
    // One 512-column tile stores coalesced Q8 planes followed by paired
    // gain/offset planes. Constant tile strides avoid retaining three
    // independent input pointers and their token strides in registers.
    for(int col=lane*16;col<m.input_columns;col+=512) {
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
__device__ __forceinline__ void q8_dispatch(TrellisProjectionView m,int row,const uint32_t* input,
        int qcolumns,float (&result)[T]) {
    const auto d=__ldg(m.descriptors+row);
    #define CASE(B) case B-1:q8_row<T,B>(m,row,input,qcolumns,result);return
    switch((d>>32)&7){CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);}
    #undef CASE
}
template<int DT,bool Parallel>
__global__ void q8_prepare(TrellisProjectionView plain,const TrellisProjectionJob* jobs,
        const Scalar<DT>* x,uint32_t* output,int tokens,int stride,int qcolumns,int groups) {
    auto m=plain;
    if constexpr(Parallel) {
        if(blockIdx.y<unsigned(groups)) {
            m=jobs[blockIdx.y].weight;
            if(!m.columns)return;
        }
    }
    const int lane=threadIdx.x&31;
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const uint64_t group=index/16;const int element=int(index%16);
    const int token=int(group/(qcolumns/16)),col=(group%(qcolumns/16))*16+element;
    float value=0.f;
    if(token<tokens && col<m.input_columns) {
        const auto input=x+size_t(token)*stride+(Parallel?0:size_t(blockIdx.y)*m.input_columns)+col;
        value=scalar_float<DT>(__ldg(input));
        if(m.columns)value*=__half2float(__ldg(static_cast<const half*>(m.scales)+col));
    }
    float peak=fabsf(value);
    #pragma unroll
    for(int delta=8;delta;delta/=2)peak=fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,delta,16));
    const float scale=peak>0.f?peak/127.f:1.f;
    int q=__float2int_rn(value/scale);q=max(-127,min(127,q));
    int sum=q;
    #pragma unroll
    for(int delta=8;delta;delta/=2)sum+=__shfl_xor_sync(0xffffffff,sum,delta,16);
    const uint32_t byte=uint32_t(uint8_t(q));
    const uint32_t packed=byte|(__shfl_down_sync(0xffffffff,byte,1)<<8)|
        (__shfl_down_sync(0xffffffff,byte,2)<<16)|(__shfl_down_sync(0xffffffff,byte,3)<<24);
    if(token<tokens) {
        const uint64_t slot_words=uint64_t(tokens)*(qcolumns/16)*6;
        const int tile=col/512,local=(col%512)/16;
        constexpr int tile_groups=32;
        const uint64_t base=uint64_t(blockIdx.y)*slot_words+uint64_t(tile)*32*tokens*6;
        const int at=token*tile_groups+local;
        if(!(lane&3))output[base+uint64_t(at)*4+element/4]=packed;
        if(!element) {
            auto* coefficients=reinterpret_cast<float*>(output+base+4*tokens*tile_groups+uint64_t(at)*2);
            coefficients[0]=scale*TQ_Q8_BOOK_INVERSE;
            coefficients[1]=float(sum)*TQ_Q8_BOOK_CENTER*scale;
        }
    }
}
template<int DT,int T,bool Parallel>
__global__ __launch_bounds__(128)
void q8_projection(TrellisProjectionView plain,const TrellisProjectionJob* jobs,int count,
        const uint32_t* input,Scalar<DT>* output,int qcolumns,int groups,
        int joined_count=0,int joined_rows=0) {
    TrellisProjectionJob job{};job.weight=plain;int begin=0,index=0;
    if constexpr(Parallel) {
        int hi=count;
        while(index<hi){const int mid=(index+hi)/2;if(blockIdx.x<unsigned(jobs[mid].block_end))hi=mid;else index=mid+1;}
        if(index==count)return;
        job=jobs[index];begin=index?jobs[index-1].block_end:0;
    }
    const auto m=job.weight;const int row=(blockIdx.x-begin)*4+threadIdx.x/32;
    if(row>=m.rows)return;
    const int projection=index;
    if constexpr(Parallel)index=m.columns?index:count;
    else index=row/(m.rows/groups);
    float value[T]{};
    const uint64_t slot_words=uint64_t(T)*(qcolumns/16)*6;
    q8_dispatch<T>(m,row,input+size_t(index)*slot_words,qcolumns,value);
    #pragma unroll
    for(int t=0;t<T;++t) {
        #pragma unroll
        for(int step=16;step;step/=2)value[t]+=__shfl_down_sync(0xffffffff,value[t],step);
        if((threadIdx.x&31)==0) {
            if(m.bias)value[t]+=__ldg(m.bias+row);
            const bool joined=Parallel && projection<joined_count;
            const size_t offset=Parallel?size_t(job.output_offset)*(joined?1:T):0;
            output[offset+size_t(t)*(joined?joined_rows:m.rows)+row]=scalar_round<DT>(value[t]);
        }
    }
}
// __grid_constant__ (SM70+) lets every thread address the same parameter
// image without making a local copy. Older targets read the existing device
// plan. 32 descriptors plus other arguments fit the 4-KiB parameter limit.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define TQ_GRID_CONSTANT __grid_constant__
#else
#define TQ_GRID_CONSTANT
#endif
struct TqInlineProjectionPlan {TrellisProjectionJob jobs[32];};
template<int DT,int T>
__global__ __launch_bounds__(128)
void q8_inline_projection(TQ_GRID_CONSTANT const TqInlineProjectionPlan plan,const TrellisProjectionJob* device_jobs,int count,int first,int begin_block,int common_slot,
        const uint32_t* input,Scalar<DT>* output,int qc,int joined,int joined_rows) {
#if __CUDA_ARCH__ >= 700
    const auto& jobs=plan.jobs;
#else
    const auto* jobs=device_jobs+first;
#endif
    const int block=blockIdx.x+begin_block;
    int lo=0,hi=count;
    while(lo<hi){const int mid=(lo+hi)/2;if(block<unsigned(jobs[mid].block_end))hi=mid;else lo=mid+1;}
    if(lo==count)return;
    const auto job=jobs[lo];const auto m=job.weight;
    const int row=(block-(lo?jobs[lo-1].block_end:begin_block))*4+threadIdx.x/32;
    if(row>=m.rows)return;
    const int projection=first+lo;
    const int slot=m.columns?projection:common_slot;
    float value[T]{};
    const uint64_t slot_words=uint64_t(T)*(qc/16)*6;
    q8_dispatch<T>(m,row,input+size_t(slot)*slot_words,qc,value);
    #pragma unroll
    for(int t=0;t<T;++t) {
        #pragma unroll
        for(int step=16;step;step/=2)value[t]+=__shfl_down_sync(0xffffffff,value[t],step);
        if((threadIdx.x&31)==0){if(m.bias)value[t]+=__ldg(m.bias+row);
            const size_t base=size_t(job.output_offset)*(projection<joined?1:T);
            output[base+size_t(t)*(projection<joined?joined_rows:m.rows)+row]=scalar_round<DT>(value[t]);}
    }
}

static_assert(sizeof(TqInlineProjectionPlan)+256<=4096,"projection parameters exceed 4 KiB");
#undef TQ_GRID_CONSTANT

extern "C" int mfq_tq_q8_tokens(TrellisProjectionView m,const void* x,void* y,
        int tokens,int groups,int dtype,void* scratch,uint64_t bytes,cudaStream_t stream) {
    if(!x || !y || x==y || tokens<1 || tokens>4 || groups<1 || groups>65535 || (dtype!=0 && dtype!=2) ||
       m.rows<1 || m.rows%groups || m.input_columns<1 || m.input_columns>INT_MAX-511 ||
       m.input_columns>INT_MAX/groups || m.input_columns>m.stored_columns || m.stored_columns%256 ||
       !m.words || !m.descriptors || !m.scales || !m.relative ||
       (!m.columns && (m.gslog<4 || m.gslog>20)))return -1;
    const int qc=int((uint64_t(m.input_columns)+511)/512*512);
    const uint64_t slot_bytes=uint64_t(tokens)*(qc/16)*24;
    const uint64_t required=uint64_t(groups)*slot_bytes;
    if(!scratch || scratch==x || scratch==y || (reinterpret_cast<uintptr_t>(scratch)&15) || bytes<required)return -1;
    auto* quant=static_cast<uint32_t*>(scratch);
    #define PREPARE(D) q8_prepare<D,false><<<dim3((uint64_t(tokens)*qc+127)/128,groups),128,0,stream>>>(m,nullptr,static_cast<const Scalar<D>*>(x),quant,tokens,m.input_columns*groups,qc,groups)
    if(dtype==0){PREPARE(0);}else{PREPARE(2);}
    #undef PREPARE
    const auto prepared=cudaGetLastError();if(prepared!=cudaSuccess)return int(prepared);
    #define RUN(D,T) q8_projection<D,T,false><<<(m.rows+3)/4,128,0,stream>>>(m,nullptr,0,quant,static_cast<Scalar<D>*>(y),qc,groups)
    #define TOKENS(D) if(tokens==1){RUN(D,1);}else if(tokens==2){RUN(D,2);}else if(tokens==3){RUN(D,3);}else {RUN(D,4);}
    if(dtype==0){TOKENS(0);}else{TOKENS(2);}
    #undef TOKENS
    #undef RUN
    return int(cudaGetLastError());
}
extern "C" int mfq_tq_q8_parallel_tokens(const TrellisProjectionJob* host_jobs,const TrellisProjectionJob* jobs,int count,int blocks,
        const void* x,void* y,int tokens,int stride,int dtype,void* scratch,uint64_t bytes,
        int joined_count,int joined_rows,cudaStream_t stream) {
    if(!host_jobs || !jobs || !x || !y || x==y || count<1 || count>=65535 || blocks<1 || tokens<1 || tokens>4 || (dtype!=0 && dtype!=2) ||
       stride<1 || stride>INT_MAX-511 || joined_count<0 || joined_count>count ||
       (joined_count && joined_rows<1))return -1;
    bool column_scaled=false;
    for(int i=0;i<count;++i)column_scaled|=host_jobs[i].weight.columns!=0;
    const int qc=int((uint64_t(stride)+511)/512*512);
    const uint64_t slot_bytes=uint64_t(tokens)*(qc/16)*24;
    const uint64_t required=(column_scaled?uint64_t(count+1):1)*slot_bytes;
    if(!scratch || scratch==x || scratch==y || (reinterpret_cast<uintptr_t>(scratch)&15) || bytes<required)return -1;
    auto* quant=static_cast<uint32_t*>(scratch);
    TrellisProjectionView common{};common.input_columns=stride;
    #define PREPARE(D) do { \
        if(column_scaled) { \
            q8_prepare<D,true><<<dim3((uint64_t(tokens)*qc+127)/128,count+1),128,0,stream>>>(common,jobs,static_cast<const Scalar<D>*>(x),quant,tokens,stride,qc,count); \
        }else { \
            q8_prepare<D,false><<<(uint64_t(tokens)*qc+127)/128,128,0,stream>>>(common,nullptr,static_cast<const Scalar<D>*>(x),quant,tokens,stride,qc,1); \
        } \
    }while(false)
    if(dtype==0){PREPARE(0);}else{PREPARE(2);}
    #undef PREPARE
    const auto prepared=cudaGetLastError();if(prepared!=cudaSuccess)return int(prepared);
    for(int first=0;first<count;first+=32) {
        const int batch=min(32,count-first);
        TqInlineProjectionPlan plan{};
        for(int i=0;i<batch;++i)plan.jobs[i]=host_jobs[first+i];
        const int begin=first?host_jobs[first-1].block_end:0;
        const int end=host_jobs[first+batch-1].block_end;
        if(end<=begin || end>blocks)return -1;
        #define RUN(D,T) q8_inline_projection<D,T><<<end-begin,128,0,stream>>>(plan,jobs,batch,first,begin,column_scaled?count:0,quant,static_cast<Scalar<D>*>(y),qc,joined_count,joined_rows)
        #define TOKENS(D) if(tokens==1){RUN(D,1);}else if(tokens==2){RUN(D,2);}else if(tokens==3){RUN(D,3);}else {RUN(D,4);}
        if(dtype==0){TOKENS(0);}else{TOKENS(2);}
        #undef TOKENS
        #undef RUN
        const auto error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    }
    return 0;
}
