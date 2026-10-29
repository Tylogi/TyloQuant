#pragma once

namespace mfq_hip_attention {

template<bool Tail,int D> __device__ __forceinline__ void grouped_tile(
        const __half* k,const __half* v,uint32_t (*queries)[D/2],uint32_t* tile,
        __half (*probability)[64],float* maxima,__half2* accum,
        int begin,int length,size_t base,float& maximum,float& denominator) {
    constexpr int Rows=64,HalfD=D/2,KeyPairs=64,KeyStride=KeyPairs+4,ValueStride=HalfD;
    constexpr int ValueRows=32,OutputPairs=HalfD/32;
    using Pair=_Float16 __attribute__((ext_vector_type(2)));
    const int tid=threadIdx.x,lane=tid&31,wave=tid/32,query=wave/2,team=wave&1;
    const int token=team*32+lane;
    float score=0;
    #pragma unroll
    for(int dimension=0;dimension<HalfD;dimension+=KeyPairs) {
        for(int i=tid;i<Rows*(KeyPairs/4);i+=256) {
            const int r=i/(KeyPairs/4),c=(i%(KeyPairs/4))*4;
            const auto values=!Tail || begin+r<length
                ?reinterpret_cast<const uint4*>(k)[(base+size_t(begin+r)*HalfD+dimension+c)/4]:uint4{};
            *reinterpret_cast<uint4*>(tile+r*KeyStride+c)=values;
        }
        __syncthreads();
        #pragma unroll
        for(int i=0;i<KeyPairs;i+=4) {
            const auto a=*reinterpret_cast<const uint4*>(queries[query]+dimension+i);
            const auto b=*reinterpret_cast<const uint4*>(tile+token*KeyStride+i);
            score=__builtin_amdgcn_fdot2(__builtin_bit_cast(Pair,a.x),__builtin_bit_cast(Pair,b.x),score,false);
            score=__builtin_amdgcn_fdot2(__builtin_bit_cast(Pair,a.y),__builtin_bit_cast(Pair,b.y),score,false);
            score=__builtin_amdgcn_fdot2(__builtin_bit_cast(Pair,a.z),__builtin_bit_cast(Pair,b.z),score,false);
            score=__builtin_amdgcn_fdot2(__builtin_bit_cast(Pair,a.w),__builtin_bit_cast(Pair,b.w),score,false);
        }
        if(dimension+KeyPairs<HalfD)__syncthreads();
    }
    if(Tail && begin+token>=length)score=-1e30f;
    float local_max=score;
    for(int delta=16;delta;delta/=2)local_max=fmaxf(local_max,__shfl_xor(local_max,delta,32));
    if(!lane)maxima[wave]=local_max;
    __syncthreads();
    const float next=fmaxf(maximum,fmaxf(maxima[query*2],maxima[query*2+1])+2.0793f);
    const float correction=expf(maximum-next);
    const float weight=!Tail || begin+token<length?expf(score-next):0.f;
    probability[query][token]=__float2half_rn(weight);
    const auto multiplier=__float2half2_rn(correction);
    #pragma unroll
    for(int i=0;i<OutputPairs;++i)accum[i]=__hmul2(accum[i],multiplier);
    __syncthreads();
    denominator=denominator*correction+weight;
    maximum=next;
    #pragma unroll
    for(int offset=0;offset<Rows;offset+=ValueRows) {
        for(int i=tid;i<ValueRows*(HalfD/4);i+=256) {
            const int r=i/(HalfD/4),c=(i%(HalfD/4))*4;
            const auto values=!Tail || begin+offset+r<length
                ?reinterpret_cast<const uint4*>(v)[(base+size_t(begin+offset+r)*HalfD+c)/4]:uint4{};
            *reinterpret_cast<uint4*>(tile+r*ValueStride+c)=values;
        }
        __syncthreads();
        #pragma unroll
        for(int r=team;r<ValueRows;r+=2) {
            const auto p=probability[query][offset+r];
            const auto pair=__halves2half2(p,p);
            #pragma unroll
            for(int i=0;i<OutputPairs;i+=2) {
                const int column=(i/2)*64+lane*2;
                const auto values=*reinterpret_cast<const uint2*>(tile+r*ValueStride+column);
                accum[i]=__hfma2(pair,__builtin_bit_cast(__half2,values.x),accum[i]);
                accum[i+1]=__hfma2(pair,__builtin_bit_cast(__half2,values.y),accum[i+1]);
            }
        }
        __syncthreads();
    }

}

// One block owns four query heads sharing a KV head. Two 32-lane groups
// cooperate on each query. KV tiles stay in LDS for all four queries.
template<int D=256> __global__ __launch_bounds__(256,2) void decode_group4(
        const __half* q,const __half* k,const __half* v,const int64_t* lengths,
        float* partial_o,float* partial_m,float* partial_l,
        int heads,int kv_heads,int capacity,int parts,int workspace_parts,float scale) {
    using Pair=_Float16 __attribute__((ext_vector_type(2)));
    constexpr int Rows=64,HalfD=D/2,KeyPairs=64,KeyStride=KeyPairs+4,ValueStride=HalfD;
    constexpr int ValueRows=32,OutputPairs=HalfD/32;
    static_assert(D==256);
    const int tid=threadIdx.x,lane=tid&31,wave=tid/32,query=wave/2,team=wave&1;
    const int part=blockIdx.x%parts,group=blockIdx.x/parts,batch=group/kv_heads;
    const int row=group*4+query;
    const int length=max(0,min(int(lengths[batch]),capacity));
    const int active=min(parts,max(1,(length+Rows-1)/Rows));
    const size_t stat=size_t(row)*workspace_parts+part;
    __half2 accum[OutputPairs];
    #pragma unroll
    for(int i=0;i<OutputPairs;++i)accum[i]=__float2half2_rn(0.f);
    if(part>=active) {
        #pragma unroll
        for(int i=0;i<OutputPairs;++i) {
            const int c=2*((i/2)*64+lane*2+i%2);
            if(!team){partial_o[stat*D+c]=0;partial_o[stat*D+c+1]=0;}
        }
        if(!team && !lane){partial_m[stat]=-1e30f;partial_l[stat]=0;}
        return;
    }
    __shared__ __align__(16) uint32_t queries[4][HalfD];
    __shared__ __align__(16) uint32_t tile[Rows*KeyStride];
    __shared__ __half probability[4][Rows];
    __shared__ float maxima[8];
    for(int i=tid;i<4*HalfD;i+=256) {
        const auto packed=reinterpret_cast<const uint32_t*>(q)[size_t(group*4)*HalfD+i];
        auto values=__half22float2(__builtin_bit_cast(__half2,packed));
        values.x*=scale;values.y*=scale;
        queries[i/HalfD][i%HalfD]=__builtin_bit_cast(uint32_t,__float22half2_rn(values));
    }
    __syncthreads();
    const size_t base=size_t(group)*capacity*HalfD;
    float maximum=-1e30f,denominator=0;
    int begin=part*Rows;
    for(;begin+Rows<=length;begin+=active*Rows)
        grouped_tile<false,D>(k,v,queries,tile,probability,maxima,accum,begin,length,base,maximum,denominator);
    if(begin<length)
        grouped_tile<true,D>(k,v,queries,tile,probability,maxima,accum,begin,length,base,maximum,denominator);
    #pragma unroll
    for(int i=0;i<OutputPairs;++i) {
        const int c=2*((i/2)*64+lane*2+i%2);
        auto value=__half22float2(accum[i]);
        value.x+=__shfl_xor(value.x,32,64);value.y+=__shfl_xor(value.y,32,64);
        if(!team){partial_o[stat*D+c]=value.x;partial_o[stat*D+c+1]=value.y;}
    }
    for(int delta=16;delta;delta/=2)denominator+=__shfl_xor(denominator,delta,32);
    denominator+=__shfl_xor(denominator,32,64);
    if(!team && !lane){partial_m[stat]=maximum;partial_l[stat]=denominator;}
}

}
