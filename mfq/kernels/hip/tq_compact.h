#pragma once
#include <cstdint>

// RTRELL06 virtual ring words preserve the existing vector decoder while
// compressed CPU/GPU storage contains exactly the logical columns.
__device__ __forceinline__ uint32_t tq_ring_word(const uint32_t* words,
        uint64_t descriptor,int width,int col,int bits,int index) {
    const bool compact=(width&255)!=0;
    const uint32_t offset=uint32_t(descriptor)&(compact?0x7ffffffeU:0xffffffffU);
    const int ring=col>>8,ringwords=8*bits;
    if(!compact || ring<(width>>8))return __ldg(words+offset+ring*ringwords+index);
    const int skipped=(256-(width&255))*bits/32;
    if(index>=skipped)return __ldg(words+offset+ring*ringwords+index-skipped);
    if(index)return 0;
    return uint32_t(((descriptor>>35)&31)|(((descriptor>>56)&255)<<5)
        |((descriptor&1)<<13)|(((descriptor>>31)&1)<<14));
}

// Resolve the short-ring branch once for a whole decode window.
template<int N,int Bits=0>
__device__ __forceinline__ void tq_ring_window(const uint32_t* words,
        uint64_t descriptor,int width,int col,int bits,int first,uint32_t (&output)[N]) {
    const bool compact=(width&255)!=0;
    const uint32_t offset=uint32_t(descriptor)&(compact?0x7ffffffeU:0xffffffffU);
    const int ring=col>>8,ringwords=8*bits;
    const auto* source=words+offset+ring*ringwords;
    if(!compact || ring<(width>>8)) {
        #pragma unroll
        for(int j=0;j<N;++j) {
            int at=first+j;
            if constexpr(Bits>0 && (Bits&(Bits-1))==0)at&=8*Bits-1;
            else if(at>=ringwords)at-=ringwords;
            output[j]=__ldg(source+at);
        }
    } else {
        const int tailwords=(width&255)*bits/32;
        const int kept_first=first-ringwords+tailwords;
        const uint32_t seed=uint32_t(((descriptor>>35)&31)|(((descriptor>>56)&255)<<5)
            |((descriptor&1)<<13)|(((descriptor>>31)&1)<<14));
        // Valid decode windows start inside the retained tail. Only their end
        // can cross its boundary, followed by the seed and then zero words.
        #pragma unroll
        for(int j=0;j<N;++j) {
            const int at=kept_first+j;
            output[j]=at<tailwords?__ldg(source+at):(at==tailwords?seed:0u);
        }
    }
}
