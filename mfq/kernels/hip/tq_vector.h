// Vector decoder shared with the CUDA TQ implementation; ROCm dot-product mapping.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
namespace mfq_tq_vector {
using U4=uint32_t __attribute__((ext_vector_type(4)));
__device__ __forceinline__ uint32_t dot_bytes(uint32_t a,uint32_t b,uint32_t c) {
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_udot4)
 return __builtin_amdgcn_udot4(a,b,c,false);
#else
 return c+(a&255)+((a>>8)&255)+((a>>16)&255)+(a>>24);
#endif
}
#define TRELLIS_NORMALIZED_WINDOW 1
#define TRELLIS_PAIR_BOOK 1
__device__ __forceinline__ uint32_t select_word(const uint32_t* chunks,int count,int index) {
    uint32_t value=0;
    #pragma unroll
    for(int j=0;j<6;++j) if(j<count && j==index)value=chunks[j];
    return value;
}

template<int V,bool Lookup,bool Columns,int Bits,class Consumer,bool RawBook=false,int HalfOutput=0,bool Prepared=false>
__device__ __forceinline__ void visit_values(const uint32_t* words,uint64_t descriptor,
        const void* selectors,const half* relative,const half* lut,const half* coefficients,
        int row,int col,int width,int gslog,Consumer& consumer,U4 raw={},U4 column_pairs={}) {
    int bits=Bits?Bits:int((descriptor>>32)&7)+1;
    float anchor=__half2float(__ushort_as_half(uint16_t(descriptor>>40)));
    int ringwords=8*bits;
    static_assert(V>0 && 256%V==0,"Decode vectors must divide an independent ring");
    static_assert(!Prepared || (V==8 && !RawBook && HalfOutput==2),
                  "Prepared tiles contain eight aligned half weights");
    // Callers assign aligned V-weight vectors. Their bit offsets cannot have
    // every possible low five bits: even precisions with V=16 are word-aligned.
    int minimum=((256/V-1)-((col/V)%(256/V)))*(V*bits);
    int first=minimum>>5;
    const uint32_t* ring=words+uint32_t(descriptor)+(col>>8)*ringwords;
    constexpr int STEP=V*(Bits?Bits:8);
    constexpr int ALIGN=STEP&-STEP;
    constexpr int MAX_SHIFT=Bits?(ALIGN<32?32-ALIGN:0):(Prepared?(32-V):31);
    constexpr int NW=(MAX_SHIFT+(V-1)*(Bits?Bits:8)+16+31)/32;
    uint32_t chunks[NW]={};
    #pragma unroll
    for(int j=0;j<NW;++j) {
        int at=first+j;
        if constexpr(Bits>0 && (Bits&(Bits-1))==0)at&=8*Bits-1;
        else if(at>=ringwords)at-=ringwords;
        if constexpr(Prepared)chunks[j]=raw[j];
        else chunks[j]=__ldg(ring+at);
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
    if constexpr(!RawBook && Columns && !Prepared) {
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
            } else if constexpr(V==8 && Prepared) {
                const int pos=(V-1-j-t)*bits;
                const uint64_t window=uint64_t(normalized[0])|(uint64_t(normalized[1])<<32);
                if(j+t==0) {
                    const uint32_t lo=pos<32?normalized[0]:normalized[1];
                    const uint32_t hi=pos<32?normalized[1]:normalized[2];
                    state=__funnelshift_r(lo,hi,pos&31);
                } else if(j+t==4)state=__funnelshift_r(normalized[0],normalized[1],pos);
                else if(j+t>=5)state=normalized[0]>>pos;
                else state=uint32_t(window>>pos);
                state&=65535;
            } else if constexpr(V==8) {
                const int pos=(V-1-j-t)*bits;
                const uint32_t lo=pos<32?normalized[0]:normalized[1];
                const uint32_t hi=pos<32?normalized[1]:normalized[2];
                state=__funnelshift_r(lo,hi,pos&31)&65535;
            } else {
                int pos=(255-((col+j+t)&255))*bits-first*32,wi=pos/32;
                state=__funnelshift_r(select_word(chunks,NW,wi),select_word(chunks,NW,wi+1),pos&31)&65535;
            }
            // Prepared gfx942 PP tiles need a byte sum; SAD against zero
            // avoids the redundant multiply-by-one operand of dot4.
#if defined(__gfx942__)
            if constexpr(Prepared)sum[t]=__builtin_amdgcn_sad_u8(state*0x83dcd12du,0u,0x6400u);
            else
#endif
            sum[t]=dot_bytes(state*0x83dcd12du,0x01010101u,0x6400u);
        }
        half2 base=__halves2half2(__ushort_as_half(sum[0]),__ushort_as_half(sum[1]));
        half2 q=__hfma2(base,__half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931)));
        float2 z=__half22float2(q);
        if constexpr(RawBook)consumer(j,z.x,z.y);
        else {
        float s0=scale,s1=scale;
        if constexpr(Columns) {
            if constexpr(Prepared) {
                uint32_t packed=column_pairs[j/2];asm volatile("" : "+v"(packed));
                const auto columns=__half22float2(__builtin_bit_cast(half2,packed));
                s0=__fmul_rn(anchor,columns.x);s1=__fmul_rn(anchor,columns.y);
            } else {s0=column_scales[j];s1=column_scales[j+1];}
        }
        float scaled0=__fmul_rn(z.x,s0),scaled1=__fmul_rn(z.y,s1);
        asm volatile("" : "+v"(scaled0), "+v"(scaled1));
        if constexpr(HalfOutput) {
            if constexpr(HalfOutput==2)consumer(j,__float2half_rn(scaled0),__float2half_rn(scaled1));
            else consumer(j,__halves2half2(__float2half_rn(scaled0),__float2half_rn(scaled1)));
        } else {
            float w0=__half2float(__float2half_rn(scaled0));
            float w1=__half2float(__float2half_rn(scaled1));
            asm volatile("" : "+v"(w0), "+v"(w1));
            consumer(j,w0,w1);
        }
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

#undef TRELLIS_PAIR_BOOK
#undef TRELLIS_NORMALIZED_WINDOW
}
