#pragma once

// GR retains FP32 activations. Convert decoded FP16 weights exactly to FP32,
// rather than narrowing the residual input to make an FP16 GEMM possible.
__global__ void tq_decode_float_tile(TrellisProjectionView m,int first,float* output) {
    const int col=(blockIdx.x*blockDim.x+threadIdx.x)*16,row=first+blockIdx.y;
    if(col>=m.input_columns)return;
    half values[16];const auto d=__ldg(m.descriptors+row);
    if(m.columns)tq_decode_tile_dispatch<true>(m,d,row,col,values);
    else tq_decode_tile_dispatch<false>(m,d,row,col,values);
    auto* target=output+std::size_t(blockIdx.y)*m.input_columns+col;
    if(col+16<=m.input_columns && (reinterpret_cast<uintptr_t>(target)&15)==0) {
        #pragma unroll
        for(int j=0;j<16;j+=4)*reinterpret_cast<float4*>(target+j)=make_float4(
            __half2float(values[j]),__half2float(values[j+1]),__half2float(values[j+2]),__half2float(values[j+3]));
    }else {
        #pragma unroll
        for(int j=0;j<16;++j)if(col+j<m.input_columns)target[j]=__half2float(values[j]);
    }
}

__global__ void tq_prefill_bfloat_input(const uint16_t* input,float* output,
        int tokens,int columns,int stride) {
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<std::size_t(tokens)*columns)
        output[i]=__bfloat162float(__ushort_as_bfloat16(input[(i/columns)*stride+i%columns]));
}

template<int DT>
__global__ void tq_prefill_float_epilogue(const float* values,const float* bias,Scalar<DT>* output,
        int first,int count,int rows,int tokens,int source_stride) {
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=std::size_t(count)*tokens)return;
    const int row=i%count;
    float value=values[(i/count)*source_stride+row];
    if(bias)value=__fadd_rn(value,__ldg(bias+first+row));
    output[(i/count)*rows+first+row]=scalar_round<DT>(value);
}

__global__ void tq_decode_bfloat_pair(TrellisProjectionView m,int first,
        __nv_bfloat16* high,__nv_bfloat16* low) {
    const int col=(blockIdx.x*blockDim.x+threadIdx.x)*16,row=first+blockIdx.y;
    if(col>=m.input_columns)return;
    half values[16];const auto d=__ldg(m.descriptors+row);
    if(m.columns)tq_decode_tile_dispatch<true>(m,d,row,col,values);
    else tq_decode_tile_dispatch<false>(m,d,row,col,values);
    const auto at=std::size_t(blockIdx.y)*m.input_columns+col;
    if(col+16<=m.input_columns && (reinterpret_cast<uintptr_t>(high+at)&15)==0 && (reinterpret_cast<uintptr_t>(low+at)&15)==0) {
        #pragma unroll
        for(int begin=0;begin<16;begin+=8) {
            uint32_t h[4],l[4];
            #pragma unroll
            for(int pair=0;pair<4;++pair) {
                h[pair]=l[pair]=0;
                #pragma unroll
                for(int i=0;i<2;++i) {
                    const float value=__half2float(values[begin+pair*2+i]);
                    const auto hi=__float2bfloat16_rn(value);
                    const auto lo=__float2bfloat16_rn(isfinite(value)?__fsub_rn(value,__bfloat162float(hi)):0.f);
                    h[pair]|=uint32_t(__bfloat16_as_ushort(hi))<<(i*16);
                    l[pair]|=uint32_t(__bfloat16_as_ushort(lo))<<(i*16);
                }
            }
            *reinterpret_cast<uint4*>(high+at+begin)=make_uint4(h[0],h[1],h[2],h[3]);
            *reinterpret_cast<uint4*>(low+at+begin)=make_uint4(l[0],l[1],l[2],l[3]);
        }
    }else {
        #pragma unroll
        for(int j=0;j<16;++j)if(col+j<m.input_columns) {
            const float value=__half2float(values[j]);const auto h=__float2bfloat16_rn(value);
            high[at+j]=h;low[at+j]=__float2bfloat16_rn(isfinite(value)?__fsub_rn(value,__bfloat162float(h)):0.f);
        }
    }
}

int tq_prefill_bfloat_pair(TrellisProjectionView m,const uint16_t* x,uint16_t* y,int tokens,
        int groups,void* workspace,uint64_t bytes,cublasHandle_t handle,cudaStream_t stream) {
    // Every finite FP16 weight is exactly a sum of two BF16 numbers. Keeping
    // the BF16 input native avoids repeatedly converting it for each decoded
    // weight tile, and two tensor GEMMs retain the original weight precision.
    const auto per_row=uint64_t(m.input_columns)*4+uint64_t(tokens)*4;
    if(bytes<per_row+32)return -2;
    // Respect the decode kernel's grid.y limit and the actual workspace.
    int tile=int(min(uint64_t(min(m.rows/groups,65535)),(bytes-32)/per_row));
    // Aligned temporary leading dimensions keep the tensor GEMM efficient.
    // The final partial tile remains valid for arbitrary output row counts.
    if(tile>=16)tile=tile/16*16;
    const auto plane=(uint64_t(tile)*m.input_columns*2+15)&~uint64_t(15);
    if(tile<1 || plane*2+uint64_t(tile)*tokens*4>bytes)return -2;
    auto* high=static_cast<__nv_bfloat16*>(workspace);
    auto* low=reinterpret_cast<__nv_bfloat16*>(static_cast<uint8_t*>(workspace)+plane);
    auto* result=reinterpret_cast<float*>(static_cast<uint8_t*>(workspace)+plane*2);
    const float alpha=1,zero=0,one=1;
    const int group_rows=m.rows/groups;
    for(int group=0;group<groups;++group)for(int first=group*group_rows;first<(group+1)*group_rows;first+=tile) {
        const int count=min(tile,(group+1)*group_rows-first);
        tq_decode_bfloat_pair<<<dim3((uint64_t(m.input_columns)+2047)/2048,count),128,0,stream>>>(m,first,high,low);
        int error=cudaGetLastError();if(error)return error;
        const auto* input=x+std::size_t(group)*m.input_columns;
        for(int part=0;part<2;++part) {
            const auto status=cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,count,tokens,m.input_columns,
                &alpha,part?low:high,CUDA_R_16BF,m.input_columns,input,CUDA_R_16BF,m.input_columns*groups,
                part?&one:&zero,result,CUDA_R_32F,count,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if(status)return 1000+status;
        }
        tq_prefill_float_epilogue<1><<<(uint64_t(count)*tokens+255)/256,256,0,stream>>>(
            result,m.bias,y,first,count,m.rows,tokens,count);
    }
    return int(cudaGetLastError());
}

extern "C" int mfq_tq_float_gemm(TrellisProjectionView m,const void* x,void* y,int tokens,
        int groups,int dtype,void* workspace,uint64_t bytes,void* blas,cudaStream_t stream) {
    if(!x || !y || !workspace || !blas || x==y || workspace==x || workspace==y || tokens<1 ||
       m.rows<1 || groups<1 || m.rows%groups || m.input_columns<1 || m.input_columns>INT_MAX/groups ||
       m.input_columns>m.stored_columns || m.stored_columns%256 ||
       !m.words || !m.descriptors || !m.scales || !m.relative ||
       (!m.columns && (m.gslog<4 || m.gslog>20)) || (dtype!=1 && dtype!=2) ||
       (reinterpret_cast<uintptr_t>(workspace)&15))return -1;
    const uint64_t width_bytes=uint64_t(m.input_columns)*sizeof(float);
    int token_tile=tokens,row_tile=0;
    uint64_t decoded_bytes=0,input_bytes=0;
    if(dtype==2) {
        row_tile=int(min(uint64_t(min(m.rows/groups,65535)),bytes/width_bytes));
        if(row_tile<1)return -1;
    }else {
        // Leave room for at least one decoded row, one exact input row and
        // the FP32 result. Alignment slack is included before any launch.
        if(bytes<width_bytes*2+sizeof(float)+32)return -1;
        token_tile=min(tokens,256);
        while(token_tile>1 && bytes<width_bytes+uint64_t(token_tile)*(width_bytes+4)+32)
            token_tile=(token_tile+1)/2;
        row_tile=int(min(uint64_t(min(m.rows/groups,65535)),
            (bytes-uint64_t(token_tile)*width_bytes-32)/(width_bytes+uint64_t(token_tile)*4)));
        if(row_tile<1)return -1;
        decoded_bytes=(uint64_t(row_tile)*width_bytes+15)&~uint64_t(15);
        input_bytes=(uint64_t(token_tile)*width_bytes+15)&~uint64_t(15);
        if(decoded_bytes+input_bytes+uint64_t(token_tile)*row_tile*4>bytes)return -1;
    }
    auto handle=static_cast<cublasHandle_t>(blas);
    cudaStream_t previous=nullptr;
    auto status=cublasGetStream(handle,&previous);if(status)return 1000+status;
    if(previous!=stream){status=cublasSetStream(handle,stream);if(status)return 1000+status;}
    cublasPointerMode_t mode;status=cublasGetPointerMode(handle,&mode);if(status)return 1000+status;
    if(mode!=CUBLAS_POINTER_MODE_HOST)return -1;
    if(dtype==1) {
        const int result=tq_prefill_bfloat_pair(m,static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),
            tokens,groups,workspace,bytes,handle,stream);
        if(result!=-2 && result!=1000+CUBLAS_STATUS_NOT_SUPPORTED && result!=1000+CUBLAS_STATUS_ARCH_MISMATCH)
            return result;
    }
    const float alpha=1,beta=0;
    auto* decoded=static_cast<float*>(workspace);
    auto* converted=reinterpret_cast<float*>(static_cast<uint8_t*>(workspace)+decoded_bytes);
    auto* intermediate=reinterpret_cast<float*>(static_cast<uint8_t*>(workspace)+decoded_bytes+input_bytes);
    const int group_rows=m.rows/groups;
    for(int group=0;group<groups;++group)for(int first=group*group_rows;first<(group+1)*group_rows;first+=row_tile) {
        const int count=min(row_tile,(group+1)*group_rows-first);
        tq_decode_float_tile<<<dim3((uint64_t(m.input_columns)+2047)/2048,count),128,0,stream>>>(m,first,decoded);
        int error=cudaGetLastError();if(error)return error;
        for(int64_t start=0;start<tokens;start+=token_tile) {
            const int batch=min(token_tile,int(tokens-start));
            const auto offset=(std::size_t(start)*groups+group)*m.input_columns;
            const float* input;
            if(dtype==2)input=static_cast<const float*>(x)+offset;
            else {
                tq_prefill_bfloat_input<<<(uint64_t(batch)*m.input_columns+255)/256,256,0,stream>>>(
                    static_cast<const uint16_t*>(x)+offset,converted,batch,m.input_columns,m.input_columns*groups);
                input=converted;
            }
            auto* output=dtype==2?static_cast<float*>(y)+std::size_t(start)*m.rows+first:intermediate;
            const auto multiply=[&](cublasComputeType_t compute) {
                return cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,count,batch,m.input_columns,
                    &alpha,decoded,CUDA_R_32F,m.input_columns,input,CUDA_R_32F,
                    dtype==2?m.input_columns*groups:m.input_columns,&beta,output,CUDA_R_32F,
                    dtype==2?m.rows:count,compute,CUBLAS_GEMM_DEFAULT);
            };
            // FP16 reconstructed weights and BF16 input both fit TF32's
            // significand exactly. FP32 residual inputs keep pedantic SGEMM.
            // cuBLAS selects its supported hardware algorithm; older targets
            // that reject FAST_TF32 retain the full FP32 implementation.
            status=multiply(dtype==1?CUBLAS_COMPUTE_32F_FAST_TF32:CUBLAS_COMPUTE_32F_PEDANTIC);
            if(dtype==1 && (status==CUBLAS_STATUS_NOT_SUPPORTED || status==CUBLAS_STATUS_ARCH_MISMATCH))
                status=multiply(CUBLAS_COMPUTE_32F_PEDANTIC);
            if(status)return 1000+status;
            if(dtype==1)tq_prefill_float_epilogue<1><<<(uint64_t(count)*batch+255)/256,256,0,stream>>>(
                intermediate,m.bias,static_cast<uint16_t*>(y)+std::size_t(start)*m.rows,first,count,m.rows,batch,count);
            else if(m.bias)tq_prefill_float_epilogue<2><<<(uint64_t(count)*batch+255)/256,256,0,stream>>>(
                output,m.bias,static_cast<float*>(y)+std::size_t(start)*m.rows,first,count,m.rows,batch,m.rows);
        }
    }
    return int(cudaGetLastError());
}
