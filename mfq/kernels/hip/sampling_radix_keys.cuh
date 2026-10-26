#pragma once

// The low 32 bits of each sort key already encode the token ID. Keeping only
// keys removes the duplicate radix payload and the inter-stage re-encoding.
template<typename Scalar,int Items,bool CompactRows=false,bool Conditional=false>
__global__ void sample_radix_keys_first(const Scalar* logits,uint64_t* output,int vocab,int top_k,
        const volatile int32_t* enabled=nullptr) {
    if constexpr(Conditional)if(!*enabled)return;
    using Sort=hipcub::BlockRadixSort<uint64_t,SAMPLE_BD,Items>;
    __shared__ typename Sort::TempStorage temporary;
    if constexpr(CompactRows) {
        logits+=int64_t(blockIdx.y)*vocab;
        output+=int64_t(blockIdx.y)*gridDim.x*top_k;
    }
    uint64_t keys[Items];
    #pragma unroll
    for(int item=0;item<Items;++item) {
        const int index=(blockIdx.x*SAMPLE_BD+threadIdx.x)*Items+item;
        if constexpr(CompactRows) {
            // Match stable float top-k, including signed-zero equivalence and
            // radix ordering of NaNs. The final packet gathers original bits.
            float value=index<vocab?float(logits[index]):0.f;
            if(value==0.f)value=0.f;
            const uint32_t raw=__float_as_uint(value);
            const uint32_t ordered=raw^((raw&0x80000000u)?0xffffffffu:0x80000000u);
            keys[item]=index<vocab?(uint64_t(ordered)<<32)|(0xffffffffu-uint32_t(index)):0;
        } else keys[item]=index<vocab?sample_make_sort_key(float(logits[index]),index):0;
    }
    Sort(temporary).SortDescending(keys);
    #pragma unroll
    for(int item=0;item<Items;++item) {
        const int rank=threadIdx.x*Items+item;
        if(rank<top_k)output[blockIdx.x*top_k+rank]=keys[item];
    }
}

__device__ __forceinline__ uint64_t sample_bitonic_key(uint64_t key,uint64_t* shared) {
    const int tid=threadIdx.x;
    #pragma unroll
    for(int width=2;width<=SAMPLE_BD;width*=2) {
        #pragma unroll
        for(int stride=width/2;stride;stride/=2) {
            uint64_t other;
            if(stride>=32) {
                shared[tid]=key;__syncthreads();
                other=shared[tid^stride];__syncthreads();
            }else other=__shfl_xor_sync(0xffffffffu,key,stride);
            const bool larger=((tid&stride)==0)==((tid&width)==0);
            key=larger?(key>other?key:other):(key<other?key:other);
        }
    }
    return key;
}

template<bool Bitonic,bool CompactRows=false,bool Conditional=false>
__global__ void sample_radix_keys_reduce(const uint64_t* input,uint64_t* output,int count,int top_k,
        const volatile int32_t* enabled=nullptr) {
    if constexpr(Conditional)if(!*enabled)return;
    using Sort=hipcub::BlockRadixSort<uint64_t,SAMPLE_BD,1>;
    __shared__ typename Sort::TempStorage temporary;
    if constexpr(CompactRows) {
        input+=int64_t(blockIdx.y)*count;
        output+=int64_t(blockIdx.y)*gridDim.x*top_k;
    }
    const int index=blockIdx.x*SAMPLE_BD+threadIdx.x;
    uint64_t key[1]={index<count?input[index]:0};
    if constexpr(Bitonic) {
        __shared__ uint64_t staging[SAMPLE_BD];
        key[0]=sample_bitonic_key(key[0],staging);
    }else Sort(temporary).SortDescending(key);
    if(threadIdx.x<top_k)output[blockIdx.x*top_k+threadIdx.x]=key[0];
}

template<bool Bitonic>
__global__ void sample_radix_keys_finish(const uint64_t* input,const float* random,int64_t* out,
        int count,int top_k,float temperature,float top_p) {
    using Sort=hipcub::BlockRadixSort<uint64_t,SAMPLE_BD,1>;
    __shared__ typename Sort::TempStorage temporary;
    __shared__ float top_vals[SAMPLE_RADIX_MAX_TOP_K];
    __shared__ int top_idxs[SAMPLE_RADIX_MAX_TOP_K];
    uint64_t key[1]={threadIdx.x<count?input[threadIdx.x]:0};
    if constexpr(Bitonic) {
        __shared__ uint64_t staging[SAMPLE_BD];
        key[0]=sample_bitonic_key(key[0],staging);
    }else Sort(temporary).SortDescending(key);
    if(threadIdx.x<top_k) {
        top_vals[threadIdx.x]=sample_sort_key_value(key[0])/temperature;
        top_idxs[threadIdx.x]=key[0]?int(0xffffffffu-uint32_t(key[0])):INT_MAX;
    }
    __syncthreads();
    if(threadIdx.x!=0)return;
    const float max_v=top_vals[0];
    float probs[SAMPLE_RADIX_MAX_TOP_K];
    float sum=0.f;
    for(int j=0;j<top_k;++j) {
        probs[j]=expf(top_vals[j]-max_v);
        sum+=probs[j];
    }
    int keep=top_k;
    float keep_sum=sum;
    if(top_p>0.f && top_p<1.f) {
        const float cutoff=top_p*sum;
        float cumulative=0.f;
        for(int j=0;j<top_k;++j) {
            cumulative+=probs[j];
            if(cumulative>=cutoff){keep=j+1;keep_sum=cumulative;break;}
        }
    }
    const float u=fminf(fmaxf(random[0],0.f),0.99999994f);
    const float target=u*keep_sum;
    float cumulative=0.f;
    int chosen=top_idxs[keep-1];
    for(int j=0;j<keep;++j) {
        cumulative+=probs[j];
        if(cumulative>=target){chosen=top_idxs[j];break;}
    }
    out[0]=chosen;
}

// Compact MTP top-k packet: original f32 logit bits and uint32 token ID.
// Values/IDs travel together in one D2H transfer; sampling stays on the CPU.
template<typename Scalar,bool Bitonic,bool Conditional=false>
__global__ void sample_radix_compact_finish(const uint64_t* input,const Scalar* logits,
        uint64_t* output,int count,int vocab,int top_k,const volatile int32_t* enabled=nullptr) {
    if constexpr(Conditional)if(!*enabled)return;
    using Sort=hipcub::BlockRadixSort<uint64_t,SAMPLE_BD,1>;
    __shared__ typename Sort::TempStorage temporary;
    const int row=blockIdx.y;
    input+=int64_t(row)*count;
    uint64_t key[1]={threadIdx.x<count?input[threadIdx.x]:0};
    if constexpr(Bitonic) {
        __shared__ uint64_t staging[SAMPLE_BD];
        key[0]=sample_bitonic_key(key[0],staging);
    }else Sort(temporary).SortDescending(key);
    if(threadIdx.x<top_k) {
        const uint32_t id=0xffffffffu-uint32_t(key[0]);
        const uint32_t raw=__float_as_uint(float(logits[int64_t(row)*vocab+id]));
        output[int64_t(row)*top_k+threadIdx.x]=(uint64_t(raw)<<32)|id;
    }
}
