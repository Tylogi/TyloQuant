#pragma once
#include <mma.h>
#include <climits>

// A transient execution plan references the three ordinary TQ banks. Packed
// weights are reconstructed only in shared memory; no expanded expert cache.
namespace tq_prefill {
__global__ void pp_batch_gather(const TrellisPrefillMoeJob* jobs,const int64_t* slots,const half* input,
        half* gathered,int B,int T,int K,int routes) {
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int vectors=(K+7)/8;
    if(index>=uint64_t(B)*T*vectors)return;
    const int col=(index%vectors)*8,t=(index/vectors)%T,b=index/(uint64_t(T)*vectors);
    const auto job=jobs[b];const bool active=t<job.count;
    const uint64_t token=active?uint64_t(__ldg(slots+job.first+t)/routes):0;
    const auto* source=input+token*K+col;auto* target=gathered+(uint64_t(b)*T+t)*K+col;
    if(col+8<=K && ((reinterpret_cast<uintptr_t>(source)|reinterpret_cast<uintptr_t>(target))&15)==0)
        *reinterpret_cast<uint4*>(target)=active?__ldg(reinterpret_cast<const uint4*>(source)):uint4{};
    else {
        #pragma unroll
        for(int j=0;j<8;++j)if(col+j<K)target[j]=active?__ldg(source+j):__float2half_rn(0);
    }
}
__global__ void pp_batch_activate(const TrellisPrefillMoeJob* jobs,const half* gate,
        half* output,int B,int T,int H,float limit) {
    const uint64_t at=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at>=uint64_t(B)*T*H)return;
    const int row=at%H;
    const uint64_t pair=(at/H)*(2ull*H)+row;
    float g=__half2float(gate[pair]);
    float u=__half2float(gate[pair+H]);
    if(limit>0){g=fminf(g,limit);u=fminf(fmaxf(u,-limit),limit);}
    output[at]=__float2half_rn(u*(g/(1.f+expf(-g))));
}
__global__ void pp_batch_scatter(const TrellisPrefillMoeJob* jobs,const int64_t* slots,const half* input,
        half* pairs,int B,int T,int K) {
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int vectors=(K+7)/8;
    if(index>=uint64_t(B)*T*vectors)return;
    const int col=(index%vectors)*8,t=(index/vectors)%T,b=index/(uint64_t(T)*vectors);
    const auto job=jobs[b];if(t>=job.count)return;
    const auto* source=input+(uint64_t(b)*T+t)*K+col;
    auto* target=pairs+uint64_t(__ldg(slots+job.first+t))*K+col;
    if(col+8<=K && ((reinterpret_cast<uintptr_t>(source)|reinterpret_cast<uintptr_t>(target))&15)==0)
        *reinterpret_cast<uint4*>(target)=__ldg(reinterpret_cast<const uint4*>(source));
    else {
        #pragma unroll
        for(int j=0;j<8;++j)if(col+j<K)target[j]=__ldg(source+j);
    }
}
template<int V,bool Columns>
__device__ __forceinline__ void pp_runtime_values(TrellisProjectionView m,
        uint64_t d,int row,int col,half* out) {
    const int bits=int((d>>32)&7)+1,ringwords=8*bits;
    const int bit=(256-V-(col&255))*bits,at=bit>>5,shift=bit&31;
    const auto* ring=m.words+uint32_t(d)+(col>>8)*ringwords;
    const uint32_t w0=__ldg(ring+at);
    const uint32_t w1=__ldg(ring+(at+1<ringwords?at+1:at+1-ringwords));
    const uint32_t w2=__ldg(ring+(at+2<ringwords?at+2:at+2-ringwords));
    const uint32_t n0=__funnelshift_r(w0,w1,shift);
    const uint32_t n1=__funnelshift_r(w1,w2,shift);
    const uint32_t n2=w2>>shift;
    const float anchor=__half2float(__ushort_as_half(uint16_t(d>>40)));
    float scale=anchor;
    if constexpr(!Columns) {
        const uint64_t group=uint64_t(row)*((m.stored_columns+(1<<m.gslog)-1)>>m.gslog)+(col>>m.gslog);
        const uint8_t code=__ldg(static_cast<const uint8_t*>(m.scales)+(group>>1));
        scale=__fmul_rn(anchor,__half2float(__ldg(static_cast<const half*>(m.relative)+((code>>((group&1)*4))&15))));
    }
    uint32_t result[V/2];
    #pragma unroll
    for(int j=0;j<V;j+=2) {
        uint32_t sum[2];
        #pragma unroll
        for(int k=0;k<2;++k) {
            const int pos=(V-1-j-k)*bits;
            const uint32_t lo=pos<32?n0:n1,hi=pos<32?n1:n2;
            const uint32_t state=__funnelshift_r(lo,hi,pos&31)&65535;
            sum[k]=__dp4a(state*0x83dcd12du,0x01010101u,0x6400u);
        }
        const half2 base=__halves2half2(__ushort_as_half(sum[0]),__ushort_as_half(sum[1]));
        const float2 q=__half22float2(__hfma2(base,__half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931))));
        float s0=scale,s1=scale;
        if constexpr(Columns) {
            const uint32_t sc=__ldg(reinterpret_cast<const uint32_t*>(static_cast<const half*>(m.scales)+col+j));
            s0=__fmul_rn(anchor,__half2float(__ushort_as_half(sc)));
            s1=__fmul_rn(anchor,__half2float(__ushort_as_half(sc>>16)));
        }
        const half a=__float2half_rn(__fmul_rn(q.x,s0)),b=__float2half_rn(__fmul_rn(q.y,s1));
        result[j/2]=uint32_t(__half_as_ushort(a))|(uint32_t(__half_as_ushort(b))<<16);
    }
    if(col+V<=m.input_columns && (reinterpret_cast<uintptr_t>(out)&(V*2-1))==0) {
        if constexpr(V==4)*reinterpret_cast<uint2*>(out)=*reinterpret_cast<const uint2*>(result);
        else *reinterpret_cast<uint4*>(out)=*reinterpret_cast<const uint4*>(result);
    }else {
        #pragma unroll
        for(int j=0;j<V;++j)if(col+j<m.input_columns)out[j]=__ushort_as_half(uint16_t(result[j/2]>>((j&1)*16)));
    }
}

template<int Role>
__device__ __forceinline__ TrellisProjectionView fused_weight(const TrellisPrefillMoeJob& job,int row,int& local) {
    if constexpr(Role==0){const bool gate=row<job.gate.rows;local=gate?row:row-job.gate.rows;const auto* view=gate?&job.gate:&job.up;return *view;}
    else {local=row;return job.down;}
}
template<int Role,int BM,int BN>
__global__ __launch_bounds__(128)
void pp_fused_projection(const TrellisPrefillMoeJob* jobs,const half* x,half* y,int T,int K,int N) {
#if __CUDA_ARCH__ >= 700
    namespace wmma=nvcuda::wmma;
    constexpr int Stride=40,WM=BM/16,WN=4/WM,PerWarpN=BN/WN,Fragments=PerWarpN/16;
    constexpr int InputBytes=(BM+BN)*Stride*2;
    __shared__ __align__(32) uint8_t storage[InputBytes];
    auto* a=reinterpret_cast<half*>(storage);auto* b=a+BM*Stride;auto* c=reinterpret_cast<float*>(storage);
    const int warp=threadIdx.x/32,mwarp=warp/WN,nwarp=warp%WN;
    const int first_row=blockIdx.x*BM,first_token=blockIdx.y*BN,batch=blockIdx.z;
    const auto* job=jobs+batch;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc[Fragments];
    #pragma unroll
    for(int q=0;q<Fragments;++q)wmma::fill_fragment(acc[q],0.f);
    for(int k=0;k<K;k+=32) {
        // Launch activation reads before decoding packed weights. SM80's
        // cp.async overlaps the copy with Gaussian decode and avoids a live
        // uint4 load temporary. Older WMMA targets retain synchronous copies.
        for(int vec=threadIdx.x;vec<BN*4;vec+=blockDim.x) {
            const int token=first_token+vec/4,col=k+(vec%4)*8;
            auto* target=b+(vec/4)*Stride+(vec%4)*8;
            const bool valid=token<T && col<K;
            const auto* source=valid?x+(uint64_t(batch)*T+token)*K+col:x;
            const int bytes=valid?min(8,K-col)*2:0;
#if __CUDA_ARCH__ >= 800
            if((reinterpret_cast<uintptr_t>(source)&15)==0) {
                const uint32_t address=uint32_t(__cvta_generic_to_shared(target));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(address),"l"(source),"r"(bytes));
            }else
#endif
            if(bytes==16 && (reinterpret_cast<uintptr_t>(source)&15)==0)
                *reinterpret_cast<uint4*>(target)=__ldg(reinterpret_cast<const uint4*>(source));
            else {
                #pragma unroll
                for(int j=0;j<8;++j)target[j]=valid && col+j<K?__ldg(source+j):__float2half_rn(0);
            }
        }
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;");
#endif
        #pragma unroll 1
        for(int vec=threadIdx.x;vec<BM*4;vec+=blockDim.x) {
            const int row=first_row+vec/4,col=k+(vec%4)*8;
            auto* target=a+(vec/4)*Stride+(vec%4)*8;
            if(row<N) {
                int local;auto m=fused_weight<Role>(*job,row,local);
                const uint64_t d=__ldg(m.descriptors+local);
                m.input_columns=m.stored_columns;
                if(m.columns)pp_runtime_values<8,true>(m,d,local,col,target);
                else pp_runtime_values<8,false>(m,d,local,col,target);
            }else *reinterpret_cast<uint4*>(target)=uint4{};
        }
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group 0;");
#endif
        __syncthreads();
        #pragma unroll
        for(int offset=0;offset<32;offset+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
            wmma::load_matrix_sync(af,a+mwarp*16*Stride+offset,Stride);
            #pragma unroll
            for(int q=0;q<Fragments;++q) {
                wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
                wmma::load_matrix_sync(bf,b+(nwarp*PerWarpN+q*16)*Stride+offset,Stride);
                wmma::mma_sync(acc[q],af,bf,acc[q]);
            }
        }
        __syncthreads();
    }
    // Reuse the input tile as one 16x16 FP32 epilogue tile per warp.
    // This keeps a64-row/128-token tile below16KiB shared memory rather
    // than materializing its entire32KiB FP32 output before conversion.
    auto* warp_output=c+warp*256;
    #pragma unroll
    for(int q=0;q<Fragments;++q) {
        wmma::store_matrix_sync(warp_output,acc[q],16,wmma::mem_col_major);
        __syncwarp();
        #pragma unroll
        for(int v=threadIdx.x&31;v<256;v+=32) {
            const int row=first_row+mwarp*16+v%16;
            const int token=first_token+nwarp*PerWarpN+q*16+v/16;
            if(row<N && token<T) {
                int local;const auto m=fused_weight<Role>(*job,row,local);
                y[(uint64_t(batch)*T+token)*N+row]=__float2half_rn(warp_output[v]+(m.bias?__ldg(m.bias+local):0));
            }
        }
        __syncwarp();
    }
#endif
}

} // namespace tq_prefill

extern "C" int mfq_tq_prefill_moe(const TrellisPrefillMoeJob* jobs,const int64_t* slots,
        const void* input,void* pairs,int batch,int capacity,int input_width,int hidden_width,
        int routes,float limit,void* activations,uint64_t activation_bytes,cudaStream_t stream) {
    if(!jobs || !slots || !input || !pairs || !activations || batch<1 || batch>65535 ||
       capacity<1 || capacity>65535*128 || input_width<1 || hidden_width<1 ||
       hidden_width>INT_MAX/2 || input_width>INT_MAX-255 || routes<1 ||
       !std::isfinite(limit) || limit<0 || (reinterpret_cast<uintptr_t>(activations)&15))return -1;
    const uint64_t rows=uint64_t(batch)*capacity;
    const uint64_t width=uint64_t(input_width>hidden_width?input_width:hidden_width);
    // Bound multiplication before forming byte offsets or launch dimensions.
    if(rows>(UINT64_MAX-31)/(4*width+2*uint64_t(hidden_width)))return -1;
    const uint64_t input_bytes=(rows*width*2+15)&~uint64_t(15);
    const uint64_t gate_bytes=(rows*hidden_width*2+15)&~uint64_t(15);
    const uint64_t up_bytes=rows*width*2;
    if(input_bytes+gate_bytes+up_bytes>activation_bytes ||
       (rows*((uint64_t(input_width)+7)/8)+127)/128>INT_MAX ||
       (rows*hidden_width+255)/256>INT_MAX)return -1;
    // FP16 WMMA m16n16k16 is available from SM70. The native planner checks
    // this once per device; direct C API callers receive an explicit error.
    int device=0,major=0;
    auto error=cudaGetDevice(&device);if(error)return int(error);
    error=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);if(error)return int(error);
    if(major<7)return int(cudaErrorNotSupported);
    auto* x=static_cast<half*>(activations);
    auto* g=reinterpret_cast<half*>(static_cast<uint8_t*>(activations)+input_bytes);
    auto* u=reinterpret_cast<half*>(static_cast<uint8_t*>(activations)+input_bytes+gate_bytes);
    const auto vectors=rows*((uint64_t(input_width)+7)/8);
    tq_prefill::pp_batch_gather<<<(vectors+127)/128,128,0,stream>>>(jobs,slots,
        static_cast<const half*>(input),x,batch,capacity,input_width,routes);
    error=cudaGetLastError();if(error)return int(error);
    tq_prefill::pp_fused_projection<0,64,128><<<dim3((2*uint64_t(hidden_width)+63)/64,
        (uint64_t(capacity)+127)/128,batch),128,0,stream>>>(jobs,x,g,capacity,input_width,2*hidden_width);
    error=cudaGetLastError();if(error)return int(error);
    // All joined Gate/Up values are consumed before reusing gathered inputs.
    tq_prefill::pp_batch_activate<<<(rows*hidden_width+255)/256,256,0,stream>>>(
        jobs,g,x,batch,capacity,hidden_width,limit);
    error=cudaGetLastError();if(error)return int(error);
    tq_prefill::pp_fused_projection<2,64,128><<<dim3((uint64_t(input_width)+63)/64,
        (uint64_t(capacity)+127)/128,batch),128,0,stream>>>(jobs,x,u,capacity,hidden_width,input_width);
    error=cudaGetLastError();if(error)return int(error);
    tq_prefill::pp_batch_scatter<<<(vectors+127)/128,128,0,stream>>>(jobs,slots,u,
        static_cast<half*>(pairs),batch,capacity,input_width);
    return int(cudaGetLastError());
}
