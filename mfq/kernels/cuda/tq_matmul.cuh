#pragma once
// Decode one compressed row and reuse each Gaussian pair across a token tile.
template<int T,int V,bool Columns> struct MultiDotConsumer {
    uint32_t packed[T][V/2];
    float sum[T][2];
    const half* scales;
    __device__ __forceinline__ void operator()(int j,float a,float b) {
        if constexpr(Columns) {
            const uint32_t value=__ldg(reinterpret_cast<const uint32_t*>(scales+j));
            a=__fmul_rn(a,__half2float(__ushort_as_half(value)));
            b=__fmul_rn(b,__half2float(__ushort_as_half(value>>16)));
        }
        #pragma unroll
        for(int t=0;t<T;++t) {
            const uint32_t value=packed[t][j/2];
            sum[t][0]=__fmaf_rn(a,__half2float(__ushort_as_half(value)),sum[t][0]);
            sum[t][1]=__fmaf_rn(b,__half2float(__ushort_as_half(value>>16)),sum[t][1]);
        }
    }
};

template<int T,int Bits,bool Columns,bool Indexed=false>
__device__ __forceinline__ void multi_row(TrellisProjectionView m,int row,
        const uint16_t* x,int first,int tokens,int xs,float (&value)[T],const int64_t* offsets=nullptr) {
    const uint64_t d=__ldg(m.descriptors+row);
    const int lane=threadIdx.x&31;
    #pragma unroll
    for(int t=0;t<T;++t)value[t]=0;
    for(int col=lane*16;col<m.input_columns;col+=512) {
        MultiDotConsumer<T,16,Columns> consumer{};
        if constexpr(Columns)consumer.scales=static_cast<const half*>(m.scales)+col;
        #pragma unroll
        for(int t=0;t<T;++t) {
            const auto* source=x+(Indexed?offsets[t]:std::size_t(first+t)*xs)+col;
            if(first+t<tokens && col+16<=m.input_columns && (reinterpret_cast<uintptr_t>(source)&15)==0) {
                const uint4 a=__ldg(reinterpret_cast<const uint4*>(source));
                const uint4 b=__ldg(reinterpret_cast<const uint4*>(source+8));
                consumer.packed[t][0]=a.x;consumer.packed[t][1]=a.y;
                consumer.packed[t][2]=a.z;consumer.packed[t][3]=a.w;
                consumer.packed[t][4]=b.x;consumer.packed[t][5]=b.y;
                consumer.packed[t][6]=b.z;consumer.packed[t][7]=b.w;
            }else {
                #pragma unroll
                for(int j=0;j<16;++j)if(first+t<tokens && col+j<m.input_columns)
                    consumer.packed[t][j/2]|=uint32_t(__ldg(source+j))<<((j&1)*16);
            }
        }
        visit_values<16,true,Columns,Bits,MultiDotConsumer<T,16,Columns>,true>(
            m.words,d,m.scales,static_cast<const half*>(m.relative),nullptr,nullptr,
            row,col,m.stored_columns,m.gslog,consumer);
        float scale=1;
        if constexpr(!Columns) {
            const int group=row*((m.stored_columns+(1<<m.gslog)-1)>>m.gslog)+(col>>m.gslog);
            const uint8_t selector=__ldg(static_cast<const uint8_t*>(m.scales)+(group>>1));
            scale=__half2float(__ldg(static_cast<const half*>(m.relative)+((selector>>((group&1)*4))&15)));
        }
        #pragma unroll
        for(int t=0;t<T;++t)value[t]=__fmaf_rn(scale,consumer.sum[t][0]+consumer.sum[t][1],value[t]);
    }
    const float anchor=__half2float(__ushort_as_half(uint16_t(d>>40)));
    #pragma unroll
    for(int t=0;t<T;++t)value[t]=__fmul_rn(anchor,value[t]);
}

template<int T,bool Columns,bool Indexed=false>
__device__ __forceinline__ void multi_dispatch(TrellisProjectionView m,int row,
        const uint16_t* x,int first,int tokens,int xs,float (&value)[T],const int64_t* offsets=nullptr) {
    const uint64_t d=__ldg(m.descriptors+row);
    #define MC(K) case K-1:multi_row<T,K,Columns,Indexed>(m,row,x,first,tokens,xs,value,offsets);return
    switch((d>>32)&7){MC(1);MC(2);MC(3);MC(4);MC(5);MC(6);MC(7);MC(8);}
    #undef MC
}

template<int T,bool Grouped=false>
__global__ __launch_bounds__(128)
void tq_multi_projection_kernel(TrellisProjectionView m,const uint16_t* x,uint16_t* output,
        int tokens,int xs,int ys,int groups=1) {
    const int row=blockIdx.x*4+threadIdx.x/32,first=blockIdx.y*T;
    if(row>=m.rows)return;
    if constexpr(Grouped)x+=std::size_t(row/(m.rows/groups))*m.input_columns;
    float value[T];
    if(m.columns)multi_dispatch<T,true>(m,row,x,first,tokens,xs,value);
    else multi_dispatch<T,false>(m,row,x,first,tokens,xs,value);
    #pragma unroll
    for(int t=0;t<T;++t) {
        #pragma unroll
        for(int step=16;step;step/=2)value[t]+=__shfl_down_sync(0xffffffff,value[t],step);
        if((threadIdx.x&31)==0 && first+t<tokens) {
            if(m.bias)value[t]+=__ldg(m.bias+row);
            output[std::size_t(first+t)*ys+row]=scalar_round<0>(value[t]);
        }
    }
}


template<int T>
__global__ __launch_bounds__(256)
void tq_multi_swiglu_kernel(TrellisProjectionView gate,TrellisProjectionView up,
        const uint16_t* x,uint16_t* output,int tokens) {
    const int warp=threadIdx.x/32,role=warp&1,slot=warp/2;
    const int row=blockIdx.x*4+slot,first=blockIdx.y*T;
    const auto m=role?up:gate;
    float value[T]{};
    if(row<m.rows) {
        if(m.columns)multi_dispatch<T,true>(m,row,x,first,tokens,m.input_columns,value);
        else multi_dispatch<T,false>(m,row,x,first,tokens,m.input_columns,value);
    }
    __shared__ float joined[4][T][2];
    #pragma unroll
    for(int t=0;t<T;++t) {
        #pragma unroll
        for(int step=16;step;step/=2)value[t]+=__shfl_down_sync(0xffffffff,value[t],step);
        if((threadIdx.x&31)==0) {
            if(row<m.rows && m.bias)value[t]+=__ldg(m.bias+row);
            joined[slot][t][role]=__half2float(__float2half_rn(value[t]));
        }
    }
    __syncthreads();
    if(!role && (threadIdx.x&31)==0 && row<m.rows) {
        #pragma unroll
        for(int t=0;t<T;++t)if(first+t<tokens) {
            const float g=joined[slot][t][0],u=joined[slot][t][1];
            output[std::size_t(first+t)*m.rows+row]=__half_as_ushort(__float2half_rn(u*(g/(1.f+expf(-g)))));
        }
    }
}

template<int Bits,bool Columns>
__device__ __forceinline__ void tq_decode_tile(TrellisProjectionView m,uint64_t d,int row,int col,half* out) {
    half value[16];
    decode_vector<16,true,Columns,Bits>(m.words,d,m.scales,static_cast<const half*>(m.relative),
        nullptr,nullptr,row,col,m.stored_columns,m.gslog,value);
    if(col+16<=m.input_columns && (reinterpret_cast<uintptr_t>(out)&15)==0) {
        *reinterpret_cast<uint4*>(out)=*reinterpret_cast<const uint4*>(value);
        *reinterpret_cast<uint4*>(out+8)=*reinterpret_cast<const uint4*>(value+8);
    }else {
        #pragma unroll
        for(int j=0;j<16;++j)if(col+j<m.input_columns)out[j]=value[j];
    }
}
template<bool Columns>
__device__ __forceinline__ void tq_decode_tile_dispatch(TrellisProjectionView m,uint64_t d,int row,int col,half* out) {
    #define C(K) case K-1:tq_decode_tile<K,Columns>(m,d,row,col,out);return
    switch((d>>32)&7){C(1);C(2);C(3);C(4);C(5);C(6);C(7);C(8);}
    #undef C
}
__global__ void tq_decode_tile_kernel(TrellisProjectionView m,int first,half* output) {
    const int col=(blockIdx.x*blockDim.x+threadIdx.x)*16,row=first+blockIdx.y;
    if(col>=m.input_columns)return;
    const uint64_t d=__ldg(m.descriptors+row);
    auto* target=output+std::size_t(blockIdx.y)*m.input_columns+col;
    if(m.columns)tq_decode_tile_dispatch<true>(m,d,row,col,target);
    else tq_decode_tile_dispatch<false>(m,d,row,col,target);
}

__global__ void tq_prefill_bias_epilogue(const float* values,const float* bias,
        half* output,int first,int count,int rows,int tokens) {
    const std::size_t index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=std::size_t(count)*tokens)return;
    const int row=index%count;
    output[(index/count)*rows+first+row]=__float2half_rn(__fadd_rn(values[index],__ldg(bias+first+row)));
}

int tq_prefill_gemm(TrellisProjectionView m,const void* x,void* y,int tokens,
        void* workspace,uint64_t workspace_bytes,void* blas,cudaStream_t stream,int groups) {
    if(!x || !y || !workspace || !blas || x==y || workspace==x || workspace==y || tokens<1 ||
       m.rows<1 || groups<1 || m.rows%groups || m.input_columns<1 || m.input_columns>2147483647/groups ||
       m.input_columns>m.stored_columns || m.stored_columns%256 ||
       !m.words || !m.descriptors || !m.scales || !m.relative ||
       (!m.columns && (m.gslog<4 || m.gslog>20)))return -1;
    const uint64_t per_row=uint64_t(m.input_columns)*sizeof(half)+(m.bias?uint64_t(tokens)*sizeof(float):0);
    if(workspace_bytes<per_row || (reinterpret_cast<uintptr_t>(workspace)&15))return -1;
    if(m.bias && workspace_bytes<per_row+15)return -1;
    const uint64_t usable=workspace_bytes-(m.bias?15:0);
    // Fill the supplied workspace; grid.y bounds decoded rows per launch.
    // Aligned row tiles keep the tensor GEMM's leading dimensions efficient.
    int row_tile=int(min(uint64_t(min(m.rows/groups,65535)),usable/per_row));
    if(row_tile>=16)row_tile=row_tile/16*16;
    if(row_tile<1)return -1;
    auto handle=static_cast<cublasHandle_t>(blas);
    cudaStream_t previous=nullptr;
    auto status=cublasGetStream(handle,&previous);if(status)return 1000+status;
    if(previous!=stream){status=cublasSetStream(handle,stream);if(status)return 1000+status;}
    cublasPointerMode_t pointer_mode;
    status=cublasGetPointerMode(handle,&pointer_mode);if(status)return 1000+status;
    if(pointer_mode!=CUBLAS_POINTER_MODE_HOST)return -1;
    const float alpha=1,beta=0;
    // Each row is padded to maintain alignment of the optional FP32 epilogue.
    const uint64_t decoded_bytes=(uint64_t(row_tile)*m.input_columns*sizeof(half)+15)&~uint64_t(15);
    if(m.bias && decoded_bytes+uint64_t(row_tile)*tokens*sizeof(float)>workspace_bytes)return -1;
    auto* intermediate=reinterpret_cast<float*>(static_cast<uint8_t*>(workspace)+decoded_bytes);
    const int group_rows=m.rows/groups;
    for(int group=0;group<groups;++group)for(int first=group*group_rows;first<(group+1)*group_rows;first+=row_tile) {
        const int count=min(row_tile,(group+1)*group_rows-first);
        tq_decode_tile_kernel<<<dim3((m.input_columns+2047)/2048,count),128,0,stream>>>(m,first,static_cast<half*>(workspace));
        int error=cudaGetLastError();if(error)return error;
        status=cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,count,tokens,m.input_columns,
            &alpha,workspace,CUDA_R_16F,m.input_columns,static_cast<const half*>(x)+std::size_t(group)*m.input_columns,
            CUDA_R_16F,m.input_columns*groups,&beta,
            m.bias?static_cast<void*>(intermediate):static_cast<void*>(static_cast<half*>(y)+first),
            m.bias?CUDA_R_32F:CUDA_R_16F,m.bias?count:m.rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if(status)return 1000+status;
        if(m.bias)tq_prefill_bias_epilogue<<<(uint64_t(count)*tokens+255)/256,256,0,stream>>>(
            intermediate,m.bias,static_cast<half*>(y),first,count,m.rows,tokens);
    }
    return int(cudaGetLastError());
}

extern "C" int mfq_tq_gemm(TrellisProjectionView m,const void* x,void* y,int tokens,
        void* workspace,uint64_t bytes,void* blas,cudaStream_t stream) {
    return tq_prefill_gemm(m,x,y,tokens,workspace,bytes,blas,stream,1);
}

extern "C" int mfq_tq_grouped_gemm(TrellisProjectionView m,const void* x,void* y,int tokens,int groups,
        void* workspace,uint64_t bytes,void* blas,cudaStream_t stream) {
    return tq_prefill_gemm(m,x,y,tokens,workspace,bytes,blas,stream,groups);
}

template<int DT>
__global__ __launch_bounds__(128)
void tq_grouped_projection_kernel(TrellisProjectionView m,const Scalar<DT>* input,
        Scalar<DT>* output,int groups) {
    const int row=blockIdx.x*4+threadIdx.x/32;
    if(row>=m.rows)return;
    const int group=row/(m.rows/groups);
    const auto* x=input+(std::size_t(blockIdx.y)*groups+group)*m.input_columns;
    float value=matrix_dot<DT,1,true>(m,row,x);
    #pragma unroll
    for(int step=16;step;step/=2)value+=__shfl_down_sync(0xffffffff,value,step);
    if((threadIdx.x&31)==0) {
        if(m.bias)value+=__ldg(m.bias+row);
        output[std::size_t(blockIdx.y)*m.rows+row]=scalar_round<DT>(value);
    }
}

extern "C" int mfq_tq_grouped_tokens(TrellisProjectionView m,const void* x,void* y,
        int tokens,int groups,int dtype,cudaStream_t stream) {
    if(!x || !y || x==y || tokens<1 || tokens>65535 || m.rows<1 || groups<1 || m.rows%groups ||
       m.input_columns<1 || m.input_columns>2147483647/groups || m.input_columns>m.stored_columns || m.stored_columns%256 ||
       !m.words || !m.descriptors || !m.scales || !m.relative ||
       (!m.columns && (m.gslog<4 || m.gslog>20)))return -1;
    if(tokens>1 && (dtype==1 || dtype==2))
        return tq_float_tokens::launch(m,x,y,tokens,groups,dtype,stream);
    if(dtype==0) {
        const int xs=groups*m.input_columns;
        if(tokens==1)tq_multi_projection_kernel<1,true><<<dim3((m.rows+3)/4,1),128,0,stream>>>(m,
            static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),tokens,xs,m.rows,groups);
        else if(tokens==3)tq_multi_projection_kernel<3,true><<<dim3((m.rows+3)/4,1),128,0,stream>>>(m,
            static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),tokens,xs,m.rows,groups);
        else if(tokens<8 && tokens!=4)tq_multi_projection_kernel<2,true><<<dim3((m.rows+3)/4,(tokens+1)/2),128,0,stream>>>(m,
            static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),tokens,xs,m.rows,groups);
        else tq_multi_projection_kernel<4,true><<<dim3((m.rows+3)/4,(tokens+3)/4),128,0,stream>>>(m,
            static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),tokens,xs,m.rows,groups);
    }else if(dtype==1)tq_grouped_projection_kernel<1><<<dim3((m.rows+3)/4,tokens),128,0,stream>>>(m,
        static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),groups);
    else if(dtype==2)tq_grouped_projection_kernel<2><<<dim3((m.rows+3)/4,tokens),128,0,stream>>>(m,
        static_cast<const float*>(x),static_cast<float*>(y),groups);
    else return -1;
    return int(cudaGetLastError());
}
