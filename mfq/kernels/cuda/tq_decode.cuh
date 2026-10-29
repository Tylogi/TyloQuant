// Fixed Gaussian book; independent 1..8-bit row precision.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

template<class Kernel> int launchable(Kernel kernel,int threads) {
    int resident=0;cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,threads,0);
    return e!=cudaSuccess?int(e):resident?0:-2;
}

__device__ __forceinline__ uint32_t select_word(const uint32_t* chunks,int count,int index) {
    uint32_t value=0;
    #pragma unroll
    for(int j=0;j<6;++j) if(j<count && j==index)value=chunks[j];
    return value;
}

template<int V,bool Lookup,bool Columns,int Bits,class Consumer,bool RawBook=false>
__device__ __forceinline__ void visit_values(const uint32_t* words,uint64_t descriptor,
        const void* selectors,const half* relative,const half* lut,const half* coefficients,
        int row,int col,int width,int gslog,Consumer& consumer) {
    int bits=Bits?Bits:int((descriptor>>32)&7)+1;
    float anchor=__half2float(__ushort_as_half(uint16_t(descriptor>>40)));
    int ringwords=8*bits;
    static_assert(V>0 && 256%V==0,"Decode vectors must divide an independent ring");
    // Callers assign aligned V-weight vectors. Their bit offsets cannot have
    // every possible low five bits: even precisions with V=16 are word-aligned.
    int minimum=((256/V-1)-((col/V)%(256/V)))*(V*bits);
    int first=minimum>>5;
    const uint32_t* ring=words+uint32_t(descriptor)+(col>>8)*ringwords;
    constexpr int STEP=V*(Bits?Bits:8);
    constexpr int ALIGN=STEP&-STEP;
    constexpr int MAX_SHIFT=Bits?(ALIGN<32?32-ALIGN:0):31;
    constexpr int NW=(MAX_SHIFT+(V-1)*(Bits?Bits:8)+16+31)/32;
    uint32_t chunks[NW]={};
    #pragma unroll
    for(int j=0;j<NW;++j) {
        int at=first+j;
        if constexpr(Bits>0 && (Bits&(Bits-1))==0)at&=8*Bits-1;
        else if(at>=ringwords)at-=ringwords;
        chunks[j]=__ldg(ring+at);
    }
    #ifdef TRELLIS_NORMALIZED_WINDOW
    constexpr int NORM=((V-1)*(Bits?Bits:8)+16+31)/32;
    uint32_t normalized[NORM];
    #pragma unroll
    for(int j=0;j<NORM;++j)normalized[j]=__funnelshift_r(chunks[j],j+1<NW?chunks[j+1]:0,minimum&31);
    #endif
    float scale=anchor;
    if constexpr(!RawBook && !Columns) {
        int group=(row*((width+(1<<gslog)-1)>>gslog))+(col>>gslog);
        uint8_t packed=__ldg(static_cast<const uint8_t*>(selectors)+(group>>1));
        int code=(packed>>((group&1)*4))&15;
        scale=__fmul_rn(anchor,__half2float(__ldg(relative+code)));
    }
    float column_scales[V];
    if constexpr(!RawBook && Columns) {
        uint32_t packed_scale[V/2];
        // A compact RTRELL05 image aligns column scales to eight bytes;
        // pairs need four-byte alignment and also cover the compact tail.
        #pragma unroll
        for(int j=0;j<V/2;++j)packed_scale[j]=__ldg(
            reinterpret_cast<const uint32_t*>(static_cast<const half*>(selectors)+col)+j);
        #pragma unroll
        for(int j=0;j<V;++j)column_scales[j]=__fmul_rn(anchor,__half2float(__ushort_as_half(uint16_t(packed_scale[j/2]>>((j&1)*16)))));
    }
    #ifdef TRELLIS_PAIR_BOOK
    #pragma unroll
    for(int j=0;j<V;j+=2) {
        uint32_t sum[2];
        #pragma unroll
        for(int t=0;t<2;++t) {
            uint32_t state;
            if constexpr(Bits>0) {
                int pos=(V-1-j-t)*Bits,wi=pos/32,shift=pos&31;
                state=__funnelshift_r(normalized[wi],wi+1<NORM?normalized[wi+1]:0,shift)&65535;
            } else if constexpr(V==4) {
                const int pos=(V-1-j-t)*bits;
                state=__funnelshift_r(normalized[0],normalized[1],pos)&65535;
            } else if constexpr(V==8) {
                const int pos=(V-1-j-t)*bits;
                const uint32_t lo=pos<32?normalized[0]:normalized[1];
                const uint32_t hi=pos<32?normalized[1]:normalized[2];
                state=__funnelshift_r(lo,hi,pos&31)&65535;
            } else {
                int pos=(255-((col+j+t)&255))*bits-first*32,wi=pos/32;
                state=__funnelshift_r(select_word(chunks,NW,wi),select_word(chunks,NW,wi+1),pos&31)&65535;
            }
            sum[t]=__dp4a(state*0x83dcd12du,0x01010101u,0x6400u);
        }
        half2 base=__halves2half2(__ushort_as_half(sum[0]),__ushort_as_half(sum[1]));
        half2 q=__hfma2(base,__half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931)));
        float2 z=__half22float2(q);
        if constexpr(RawBook)consumer(j,z.x,z.y);
        else {
        float s0=scale,s1=scale;
        if constexpr(Columns){s0=column_scales[j];s1=column_scales[j+1];}
        const float w0=__half2float(__float2half_rn(__fmul_rn(z.x,s0)));
        const float w1=__half2float(__float2half_rn(__fmul_rn(z.y,s1)));
        consumer(j,w0,w1);
        }
    }
    #endif
}


template<int V> struct StoreValues {
    half* values;
    __device__ __forceinline__ void operator()(int j,float a,float b) {
        values[j]=__float2half_rn(a);values[j+1]=__float2half_rn(b);
    }
};

template<int V,bool Lookup,bool Columns,int Bits=0>
__device__ __forceinline__ void decode_vector(const uint32_t* words,uint64_t descriptor,
        const void* selectors,const half* relative,const half* lut,const half* coefficients,
        int row,int col,int width,int gslog,half (&values)[V]) {
    StoreValues<V> consumer{values};
    visit_values<V,Lookup,Columns,Bits>(words,descriptor,selectors,relative,lut,coefficients,
                                      row,col,width,gslog,consumer);
}
