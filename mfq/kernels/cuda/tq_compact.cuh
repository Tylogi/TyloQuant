#pragma once
#include <cstdint>

// A short final ring keeps only real symbols. Its wrap state uses reserved
// descriptor bits; synthesize the missing ring words in registers, never RAM.
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
