#pragma once
#include <type_traits>

namespace mfq_hip_attention {
using PrefillHalf4=_Float16 __attribute__((ext_vector_type(4)));
using PrefillHalf8=_Float16 __attribute__((ext_vector_type(8)));
using PrefillFloat4=float __attribute__((ext_vector_type(4)));

// Four waves share the same KV tile and process sixteen queries each.
// The cache prefix changes the causal boundary, independently of query storage.
template<class Scalar,int KeyTile=16> __global__ __launch_bounds__(256,2) void prefill_group4(
        const Scalar* query,const __half* keys,const __half* values,Scalar* output,
        int heads,int kv_heads,int tokens,int capacity,int prefix,float scale) {
    constexpr int D=256,QueryTile=16,KeyStride=D+8,ValueStride=KeyTile+4;
    const int tid=threadIdx.x,lane=tid&63,head=tid/64;
    const int group=blockIdx.y,row=group*4+head;
    const int query_tile=blockIdx.x;
    const int position=query_tile*QueryTile+lane%16,component=(lane/16)*4;
    const int length=prefix+tokens;
    __shared__ __align__(16) _Float16 memory[D*ValueStride];
    __shared__ __align__(16) _Float16 probabilities[4][QueryTile][KeyTile];
    PrefillHalf4 q[16];PrefillFloat4 accum[16];
    #pragma unroll
    for(int tile=0;tile<16;++tile) {
        PrefillHalf4 x{};
        if(position<tokens) {
            const size_t at=(size_t(row)*tokens+position)*D+tile*16+component;
            if constexpr(std::is_same_v<Scalar,__half>) {
                x=__builtin_bit_cast(PrefillHalf4,*reinterpret_cast<const uint2*>(query+at));
                #pragma unroll
                for(int j=0;j<4;++j)x[j]=_Float16(float(x[j])*scale);
            } else {
                const auto source=*reinterpret_cast<const float4*>(query+at);
                x={_Float16(source.x*scale),_Float16(source.y*scale),
                   _Float16(source.z*scale),_Float16(source.w*scale)};
            }
        }
        q[tile]=x;accum[tile]={};
    }
    const size_t base=size_t(group)*capacity*D;
    float maximum=-1e30f,denominator=0;
    const int end=min(length,prefix+(query_tile+1)*QueryTile);
    for(int start=0;start<end;start+=KeyTile) {
        for(int i=tid;i<KeyTile*(D/8);i+=256) {
            const int r=i/(D/8),c=(i%(D/8))*8;
            const auto data=start+r<length
                ?*reinterpret_cast<const uint4*>(keys+base+size_t(start+r)*D+c):uint4{};
            *reinterpret_cast<uint4*>(memory+r*KeyStride+c)=data;
        }
        __syncthreads();
        PrefillFloat4 scores[KeyTile/16];float tile_max=-1e30f;
        #pragma unroll
        for(int kt=0;kt<KeyTile/16;++kt) {
            PrefillFloat4 score{};
            #pragma unroll
            for(int tile=0;tile<16;++tile) {
                const auto a=__builtin_bit_cast(PrefillHalf4,
                    *reinterpret_cast<const uint2*>(memory+(kt*16+lane%16)*KeyStride+tile*16+component));
                score=__builtin_amdgcn_mfma_f32_16x16x16f16(a,q[tile],score,0,0,0);
            }
            #pragma unroll
            for(int j=0;j<4;++j) {
                const int key=start+kt*16+component+j;
                if(key>=length || key>prefix+position)score[j]=-1e30f;
                tile_max=fmaxf(tile_max,score[j]);
            }
            scores[kt]=score;
        }
        tile_max=fmaxf(tile_max,__shfl_xor(tile_max,16,64));
        tile_max=fmaxf(tile_max,__shfl_xor(tile_max,32,64));
        const float next=fmaxf(maximum,tile_max),correction=expf(maximum-next);
        float sum=0;
        #pragma unroll
        for(int kt=0;kt<KeyTile/16;++kt) {
            PrefillHalf4 p{};
            #pragma unroll
            for(int j=0;j<4;++j) {
                const int key=start+kt*16+component+j;
                const float weight=key<length && key<=prefix+position?expf(scores[kt][j]-next):0.f;
                sum+=weight;p[j]=_Float16(weight);
            }
            *reinterpret_cast<uint2*>(probabilities[head][lane%16]+kt*16+component)=__builtin_bit_cast(uint2,p);
        }
        sum+=__shfl_xor(sum,16,64);sum+=__shfl_xor(sum,32,64);
        #pragma unroll
        for(int tile=0;tile<16;++tile)accum[tile]*=correction;
        denominator=denominator*correction+sum;maximum=next;
        __syncthreads();
        for(int i=tid;i<KeyTile*(D/8);i+=256) {
            const int r=i/(D/8),c=(i%(D/8))*8;
            const auto data=start+r<length
                ?*reinterpret_cast<const uint4*>(values+base+size_t(start+r)*D+c):uint4{};
            const auto packed=__builtin_bit_cast(PrefillHalf8,data);
            #pragma unroll
            for(int j=0;j<8;++j)memory[(c+j)*ValueStride+r]=packed[j];
        }
        __syncthreads();
        #pragma unroll
        for(int kt=0;kt<KeyTile/16;++kt) {
            const auto b=__builtin_bit_cast(PrefillHalf4,
                *reinterpret_cast<const uint2*>(probabilities[head][lane%16]+kt*16+component));
            #pragma unroll
            for(int tile=0;tile<16;++tile) {
                const auto a=__builtin_bit_cast(PrefillHalf4,
                    *reinterpret_cast<const uint2*>(memory+(tile*16+lane%16)*ValueStride+kt*16+component));
                accum[tile]=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b,accum[tile],0,0,0);
            }
        }
        __syncthreads();
    }
    if(position<tokens) {
        const float inverse=1.f/(denominator>0?denominator:1.f);
        #pragma unroll
        for(int tile=0;tile<16;++tile) {
            #pragma unroll
            for(int j=0;j<4;++j) {
                const size_t at=(size_t(row)*tokens+position)*D+tile*16+component+j;
                const float value=accum[tile][j]*inverse;
                if constexpr(std::is_same_v<Scalar,__half>)output[at]=__float2half_rn(value);
                else output[at]=value;
            }
        }
    }
}

}
