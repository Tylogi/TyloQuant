#pragma once
#include <hip/hip_bf16.h>
#include <type_traits>

namespace mfq_hip_dense_moe {
template<class Scalar>
__global__ __launch_bounds__(256) void decode(const Scalar* weight,const Scalar* x,
    const int32_t* ids,Scalar* output,int experts,int rows,int columns,int routes,bool routed) {
    const int lane=threadIdx.x%warpSize,wave=threadIdx.x/warpSize;
    const int pair=blockIdx.y,row=blockIdx.x*(256/warpSize)+wave,expert=ids[pair];
    if(row>=rows)return;
    float sum=0.f;
    if(expert>=0 && expert<experts) {
        const size_t wbase=(size_t(expert)*rows+row)*columns;
        const size_t xbase=size_t(routed?pair:pair/routes)*columns;
        for(int k=lane;k<columns;k+=warpSize)sum=fmaf(float(weight[wbase+k]),float(x[xbase+k]),sum);
    }
    for(int d=warpSize/2;d;d>>=1)sum+=__shfl_down(sum,d,warpSize);
    if(lane==0)output[size_t(pair)*rows+row]=static_cast<Scalar>(sum);
}

// The expert route map is shared with the quantized path. A tile reads its
// expert directly from persistent storage; it never materializes [pairs,N,K].
template<class Scalar>
__global__ __launch_bounds__(256) void grouped(const Scalar* weight,const Scalar* x,
    const int32_t* ids,const int32_t* bounds,const int32_t* tiles,const int32_t* tile_experts,
    Scalar* output,int experts,int rows,int columns,int routes,bool routed) {
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    constexpr int BM=16,BN=64,BK=64,Stride=72;
    using F4=float __attribute__((ext_vector_type(4)));
    using H4=_Float16 __attribute__((ext_vector_type(4)));
    using B4=short __attribute__((ext_vector_type(4)));
    __shared__ Scalar w[BN*Stride],a[BM*Stride];
    const int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
    const int ntiles=(rows+BN-1)/BN,total=tiles[experts]*ntiles;
    for(int task=blockIdx.x;task<total;task+=gridDim.x) {
        const int tile=task/ntiles,ntile=task%ntiles,expert=tile_experts[tile];
        const int first=bounds[expert]+(tile-tiles[expert])*BM,last=min(first+BM,bounds[expert+1]);
        F4 acc{};
        for(int kb=0;kb<columns;kb+=BK) {
            for(int i=tid;i<BN*BK;i+=256) {
                const int n=i/BK,k=i%BK;
                w[n*Stride+k]=(ntile*BN+n<rows && kb+k<columns)?weight[(size_t(expert)*rows+ntile*BN+n)*columns+kb+k]:Scalar{};
            }
            for(int i=tid;i<BM*BK;i+=256) {
                const int m=i/BK,k=i%BK,compact=first+m;
                const int pair=compact<last?ids[compact]:-1;
                a[m*Stride+k]=(pair>=0 && kb+k<columns)?x[size_t(routed?pair:pair/routes)*columns+kb+k]:Scalar{};
            }
            __syncthreads();
            if constexpr(std::is_same_v<Scalar,float>) {
                #pragma unroll
                for(int k=0;k<BK;k+=4)acc=__builtin_amdgcn_mfma_f32_16x16x4f32(
                    w[(wave*16+(lane&15))*Stride+k+(lane>>4)],
                    a[(lane&15)*Stride+k+(lane>>4)],acc,0,0,0);
            } else {
                #pragma unroll
                for(int k=0;k<BK;k+=16) {
                    const auto wa=*reinterpret_cast<const uint2*>(w+(wave*16+(lane&15))*Stride+k+(lane>>4)*4);
                    const auto xa=*reinterpret_cast<const uint2*>(a+(lane&15)*Stride+k+(lane>>4)*4);
                    if constexpr(std::is_same_v<Scalar,__half>)acc=__builtin_amdgcn_mfma_f32_16x16x16f16(
                        __builtin_bit_cast(H4,wa),__builtin_bit_cast(H4,xa),acc,0,0,0);
                    else acc=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                        __builtin_bit_cast(B4,wa),__builtin_bit_cast(B4,xa),acc,0,0,0);
                }
            }
            __syncthreads();
        }
        const int compact=first+(lane&15);
        if(compact<last) {
            const int pair=ids[compact];
            #pragma unroll
            for(int j=0;j<4;++j) {
                const int row=ntile*BN+wave*16+(lane>>4)*4+j;
                if(row<rows)output[size_t(pair)*rows+row]=static_cast<Scalar>(acc[j]);
            }
        }
        __syncthreads();
    }
#endif
}
}
