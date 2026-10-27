// CUDA tq_float_tokens.cuh port: physical 64-lane waves, unchanged FP16 reconstruction.
#pragma once
#include <climits>

// Exact FP16 weight reconstruction for BF16/FP32 input, shared by a token tile.
namespace mfq_tq_cuda_exact {
template<int DT>using S=std::conditional_t<DT==2,float,uint16_t>;
template<int DT>__device__ __forceinline__ float number(S<DT> x) {
    if constexpr(DT==2)return x;
    else if constexpr(DT==0)return __half2float(__ushort_as_half(x));
    else return __bfloat162float(__ushort_as_bfloat16(x));
}
template<int DT>__device__ __forceinline__ S<DT> rounded(float x) {
    if constexpr(DT==2)return x;
    else if constexpr(DT==0)return __half_as_ushort(__float2half_rn(x));
    else return __bfloat16_as_ushort(__float2bfloat16(x));
}
template<int DT,int T=2,int V=16>struct GenericConsumer {
    uint32_t packed[T][V/2];
    float values[T][V],sum[T][2];
    __device__ __forceinline__ void operator()(int j,float a,float b) {
        #pragma unroll
        for(int t=0;t<T;++t) {
            float x,y;
            if constexpr(DT==2){x=values[t][j];y=values[t][j+1];}
            else {x=number<DT>(uint16_t(packed[t][j/2]));y=number<DT>(uint16_t(packed[t][j/2]>>16));}
            sum[t][0]=__fmaf_rn(a,x,sum[t][0]);sum[t][1]=__fmaf_rn(b,y,sum[t][1]);
        }
    }
    __device__ __forceinline__ void operator()(int j,half2 weights) {
        using H2=_Float16 __attribute__((ext_vector_type(2)));
        const H2 a=__builtin_bit_cast(H2,weights);
        #pragma unroll
        for(int t=0;t<T;++t) {
            const H2 b=__builtin_bit_cast(H2,packed[t][j/2]);
            const int slot=(j/2)&1;
            sum[t][slot]=__builtin_amdgcn_fdot2(a,b,sum[t][slot],false);
        }
    }
    __device__ __forceinline__ void operator()(int j,half a,half b) {
        const _Float16 wa=__builtin_bit_cast(_Float16,a),wb=__builtin_bit_cast(_Float16,b);
        #pragma unroll
        for(int t=0;t<T;++t) {
            asm("v_fma_mix_f32 %0, %1, %2, %0 op_sel:[0,0,0] op_sel_hi:[1,0,0]"
                : "+v"(sum[t][0]) : "v"(wa),"v"(values[t][j]));
            asm("v_fma_mix_f32 %0, %1, %2, %0 op_sel:[0,0,0] op_sel_hi:[1,0,0]"
                : "+v"(sum[t][1]) : "v"(wb),"v"(values[t][j+1]));
        }
    }
};
template<int DT,int Bits,bool Columns,int W,int T=2,int V=16,bool Indexed=false>
__device__ __forceinline__ void generic_row(ProjectionView m,int row,const S<DT>* x,
        int first,int tokens,int stride,float (&value)[T],const int64_t* offsets=nullptr) {
    row=__builtin_amdgcn_readfirstlane(row);
    #pragma unroll
    for(int t=0;t<T;++t)value[t]=0;
    const auto d=__ldg(m.descriptors+row);
    if constexpr(T==1 && !Indexed) {
        if(first==0 && tokens==1 && m.input_columns%V==0 && (reinterpret_cast<uintptr_t>(x)&15)==0) {
            GenericConsumer<DT,T,V> consumer{};
            for(int col=(threadIdx.x%(W*64))*V;col<m.input_columns;col+=W*64*V) {
                if constexpr(DT==2) {
                    #pragma unroll
                    for(int i=0;i<V/4;++i) {
                        const auto v=__ldg(reinterpret_cast<const float4*>(x+col+i*4));
                        consumer.values[0][i*4]=v.x;consumer.values[0][i*4+1]=v.y;
                        consumer.values[0][i*4+2]=v.z;consumer.values[0][i*4+3]=v.w;
                    }
                } else {
                    #pragma unroll
                    for(int i=0;i<V/8;++i) {
                        const auto v=__ldg(reinterpret_cast<const uint4*>(x+col+i*8));
                        consumer.packed[0][i*4]=v.x;consumer.packed[0][i*4+1]=v.y;
                        consumer.packed[0][i*4+2]=v.z;consumer.packed[0][i*4+3]=v.w;
                    }
                }
                mfq_tq_vector::visit_values<V,true,Columns,Bits,GenericConsumer<DT,T,V>,false,(DT==0?1:(DT==2?2:0))>(m.words,d,m.scales,
                    static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,consumer);
            }
            value[0]=consumer.sum[0][0]+consumer.sum[0][1];
            return;
        }
    }
    for(int col=(threadIdx.x%(W*64))*V;col<m.input_columns;col+=W*64*V) {
        GenericConsumer<DT,T,V> consumer{};
        #pragma unroll
        for(int t=0;t<T;++t) {
            const auto* source=x+(Indexed?offsets[t]:std::size_t(first+t)*stride)+col;
            if(first+t<tokens && col+V<=m.input_columns && (reinterpret_cast<uintptr_t>(source)&15)==0 && (DT==2 || V>=8)) {
                if constexpr(DT==2) {
                    #pragma unroll
                    for(int i=0;i<V/4;++i) {
                        const auto v=__ldg(reinterpret_cast<const float4*>(source+i*4));
                        consumer.values[t][i*4]=v.x;consumer.values[t][i*4+1]=v.y;
                        consumer.values[t][i*4+2]=v.z;consumer.values[t][i*4+3]=v.w;
                    }
                }else {
                    #pragma unroll
                    for(int j=0;j<V/8;++j) {
                        const uint4 a=__ldg(reinterpret_cast<const uint4*>(source+j*8));
                        consumer.packed[t][j*4]=a.x;consumer.packed[t][j*4+1]=a.y;
                        consumer.packed[t][j*4+2]=a.z;consumer.packed[t][j*4+3]=a.w;
                    }
                }
            }else {
                #pragma unroll
                for(int j=0;j<V;++j)if(first+t<tokens && col+j<m.input_columns) {
                    if constexpr(DT==2)consumer.values[t][j]=__ldg(source+j);
                    else consumer.packed[t][j/2]|=uint32_t(__ldg(source+j))<<((j&1)*16);
                }
            }
        }
        mfq_tq_vector::visit_values<V,true,Columns,Bits,GenericConsumer<DT,T,V>,false,(DT==0?1:(DT==2?2:0))>(m.words,d,m.scales,
            static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,consumer);
        #pragma unroll
        for(int t=0;t<T;++t)value[t]+=consumer.sum[t][0]+consumer.sum[t][1];
    }
}
template<int DT,bool Columns,int W,int T=2,int V=16,bool Indexed=false>
__device__ __forceinline__ void generic_dispatch(ProjectionView m,int row,const S<DT>* x,
        int first,int tokens,int stride,float (&value)[T],const int64_t* offsets=nullptr) {
    row=__builtin_amdgcn_readfirstlane(row);
    const auto d=__ldg(m.descriptors+row);
    #define C(B) case B-1:generic_row<DT,B,Columns,W,T,V,Indexed>(m,row,x,first,tokens,stride,value,offsets);return
    switch((d>>32)&7){C(1);C(2);C(3);C(4);C(5);C(6);C(7);C(8);}
    #undef C
}

#ifndef MFQ_TQ_ROW_HELPERS_ONLY
template<int DT,bool Grouped,int W,int T=2,int V=16>
__global__ __launch_bounds__(256)
void tq_generic_kernel(ProjectionView m,const S<DT>* x,S<DT>* output,int tokens,int groups) {
    constexpr int R=4/W;
    const int warp=threadIdx.x/64,slot=warp/W,row=blockIdx.x*R+slot,first=blockIdx.y*T;
    float value[T]{};
    if(row<m.rows) {
        if constexpr(Grouped)x+=std::size_t(row/(m.rows/groups))*m.input_columns;
        if(m.columns)generic_dispatch<DT,true,W,T,V>(m,row,x,first,tokens,m.input_columns*groups,value);
        else generic_dispatch<DT,false,W,T,V>(m,row,x,first,tokens,m.input_columns*groups,value);
    }
    __shared__ float partial[R][T][W];
    #pragma unroll
    for(int t=0;t<T;++t) {
        #pragma unroll
        for(int step=32;step;step/=2)value[t]+=__shfl_down(value[t],step,64);
        if constexpr(W>1)if((threadIdx.x&63)==0)partial[slot][t][warp%W]=value[t];
    }
    if constexpr(W>1)__syncthreads();
    #pragma unroll
    for(int t=0;t<T;++t) {
        if constexpr(W>1) {
            value[t]=(threadIdx.x&63)<W?partial[slot][t][threadIdx.x&63]:0.f;
            #pragma unroll
            for(int step=32;step;step/=2)value[t]+=__shfl_down(value[t],step,64);
        }
        if(threadIdx.x%(W*64)==0 && row<m.rows && first+t<tokens) {
            if(m.bias)value[t]+=__ldg(m.bias+row);
            output[std::size_t(first+t)*m.rows+row]=rounded<DT>(value[t]);
        }
    }
}
template<int T=2,int V=16>
int launch_tile(ProjectionView m,const void* x,void* y,
        int tokens,int groups,int dtype,hipStream_t stream) {
    if(!x || !y || tokens<1 || tokens>65535 || groups<1 || m.rows<1 || m.rows%groups ||
       m.input_columns<1 || m.input_columns>INT_MAX/groups || (dtype<0 || dtype>2))return -1;
    // Balance the independent rows against the 16-value input vectors.
    // All shapes use token reuse; only the number of warps per row changes.
    const auto vectors=(uint64_t(m.input_columns)+15)/16;
    int warps=1;
    while(warps<4 && uint64_t(m.rows)*2*warps<=vectors)warps*=2;
    if constexpr(T==1) {
        if(m.rows<=64)warps=4;
        else if(m.input_columns>=8192 && m.rows<m.input_columns)warps=2;
    }
    #define TQ_FLOAT_LAUNCH(D,G,W) tq_generic_kernel<D,G,W,T,V><<<dim3((m.rows+4/W-1)/(4/W),(tokens+T-1)/T),256,0,stream>>>(m,static_cast<const S<D>*>(x),static_cast<S<D>*>(y),tokens,groups)
    #define TQ_FLOAT_GROUP(D,W) if(groups==1){TQ_FLOAT_LAUNCH(D,false,W);}else{TQ_FLOAT_LAUNCH(D,true,W);}
    #define TQ_FLOAT_WARP(WIDTH) case WIDTH:if(dtype==0){TQ_FLOAT_GROUP(0,WIDTH);}else if(dtype==1){TQ_FLOAT_GROUP(1,WIDTH);}else{TQ_FLOAT_GROUP(2,WIDTH);}break
    switch(warps){TQ_FLOAT_WARP(1);TQ_FLOAT_WARP(2);TQ_FLOAT_WARP(4);default:return -1;}
    #undef TQ_FLOAT_WARP
    #undef TQ_FLOAT_GROUP
    #undef TQ_FLOAT_LAUNCH
    return int(hipGetLastError());
}

inline int launch(ProjectionView m,const void* x,void* y,
        int tokens,int groups,int dtype,hipStream_t stream) {
    if(tokens==1) {
        if(m.rows<=64)return launch_tile<1,16>(m,x,y,tokens,groups,dtype,stream);
        if(m.rows>=m.input_columns) {
            return launch_tile<1,8>(m,x,y,tokens,groups,dtype,stream);
        }
        return launch_tile<1,16>(m,x,y,tokens,groups,dtype,stream);
    }
    // FP32 activations use twice the registers of packed BF16 pairs. Shorter
    // vectors keep their register footprint bounded while sharing each decoded
    // row across three or four tokens. BF16 retains its two-token geometry.
    if(dtype==2 && tokens>2) {
        if(tokens==3 || tokens==5 || tokens==6)
            return launch_tile<3,8>(m,x,y,tokens,groups,dtype,stream);
        return launch_tile<4,8>(m,x,y,tokens,groups,dtype,stream);
    }
    return launch_tile(m,x,y,tokens,groups,dtype,stream);
}


#endif // MFQ_TQ_ROW_HELPERS_ONLY
} // namespace mfq_tq_cuda_exact
