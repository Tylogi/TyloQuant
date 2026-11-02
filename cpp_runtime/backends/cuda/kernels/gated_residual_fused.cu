#include "../ops/include/float_projection.h"
#include "qwen4_exp.h"
#include "../ops/include/gated_residual_fused.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <type_traits>
#include <unordered_map>
#include "kernel_dispatch_trace.h"
#include "../../../../mfq/kernels/cuda/nint_whole_group_decode.cuh"


namespace {
struct MhcMatrix {
    const uint8_t *q,*bits,*scales,*minima;
    const int64_t* offsets;
    const float *row_scale,*row_minimum;
    const __half* zero_scale;
    int width,outputs,groups,gs;
    bool zero,aligned;
    const __nv_bfloat16* dense=nullptr;
    uint64_t bytes=0;
    bool fast_sigmoid=false;
};
struct MhcInput {
    const void *branch,*residual,*injection;
    const float* norm;
    const float* cached_gates;
    int hidden,streams;
    bool branch_half,residual_half,injection_half,output_half,after;
    bool fast_norm=false;
};
__device__ __forceinline__ float mhc_read(const void* p,size_t i,bool half) {
    return half?__half2float(static_cast<const __half*>(p)[i]):static_cast<const float*>(p)[i];
}
__device__ __forceinline__ float mhc_round(float v,bool half) {
    return half?__half2float(__float2half_rn(v)):v;
}
__device__ __forceinline__ void mhc_write(void* p,size_t i,float v,bool half) {
    if(half)static_cast<__half*>(p)[i]=__float2half_rn(v);else static_cast<float*>(p)[i]=v;
}
__device__ __forceinline__ float mhc_sigmoid(float value,bool fast) {
    // Strata's fused GR uses the CUDA exponential intrinsic. Keep weight dots
    // and rounding unchanged; only the Up gate's exponential is optional.
    return 1.f/(1.f+(fast?__expf(-value):expf(-value)));
}
__device__ __forceinline__ float mhc_norm_inverse(float sum,int width,float eps,bool fast) {
    const float variance=__fadd_rn(__fdiv_rn(sum,float(width)),eps);
    return fast?rsqrtf(variance):__fdiv_rn(1.f,__fsqrt_rn(variance));
}
__device__ __forceinline__ float mhc_updated(MhcInput p,int sample,int stream,int column) {
    const size_t at=(size_t(sample)*p.streams+stream)*p.hidden+column;
    const float residual=mhc_read(p.residual,at,p.residual_half);
    if(!p.after)return residual;
    const float branch=mhc_read(p.branch,size_t(sample)*p.hidden+column,p.branch_half);
    const float gate=p.cached_gates[stream];
    const float product=mhc_round(__fmul_rn(branch,gate),p.branch_half && p.injection_half);
    return mhc_round(__fadd_rn(residual,product),p.output_half);
}
__device__ __forceinline__ void mhc_normalized_slice(MhcInput p,const float* inverse,int sample,int begin,int end,float* slice) {
    const int first=begin/p.hidden,last=(end-1)/p.hidden;
    for(int s=first;s<=last;++s) {
        const int from=max(0,begin-s*p.hidden),until=min(p.hidden,end-s*p.hidden);
        for(int column=from+int(threadIdx.x);column<until;column+=blockDim.x) {
            const float value=mhc_updated(p,sample,s,column);
            slice[s*p.hidden+column-begin]=mhc_round(__fmul_rn(__fmul_rn(value,inverse[s]),__fadd_rn(p.norm[s*p.hidden+column],1.f)),p.output_half);
        }
    }
}
__device__ __forceinline__ float mhc_activated(float value,bool half,int streams,bool injection) {
    const float projected=mhc_round(value,half);
    const float low=mhc_round(__fdiv_rn(projected,float(streams)),half);
    const float gate=mhc_round(1.f/(1.f+expf(-low)),half);
    return mhc_round(__fmul_rn(injection?2.f:low,gate),half);
}

#include "gr_two_plane.cuh"

struct MhcGroupCodes {uint32_t word;int shift,bits;};
__device__ __forceinline__ MhcGroupCodes mhc_group_read(MhcMatrix w,uint64_t bit,int bits,int lane) {
    const uint64_t start=(bit>>3)&~uint64_t(3);
    const int shift=int(bit&31u),words=(shift+w.gs*bits+31)/32;
    uint32_t word=0;
    if(lane<words) {
        const uint64_t at=start+uint64_t(lane)*4;
        if(at+4<=w.bytes)word=__ldg(reinterpret_cast<const uint32_t*>(w.q+at));
        else for(int byte=0;byte<4;++byte)if(at+byte<w.bytes)
            word|=uint32_t(w.q[at+byte])<<(byte*8);
    }
    return {word,shift,bits};
}
__device__ __forceinline__ float mhc_group_code(MhcGroupCodes group,int element) {
    const int bit=group.shift+element*group.bits,index=bit>>5;
    const uint32_t lo=__shfl_sync(0xffffffffu,group.word,index);
    const uint32_t hi=__shfl_sync(0xffffffffu,group.word,index+1);
    return float(__funnelshift_r(lo,hi,bit&31)&((1u<<group.bits)-1));
}
__device__ __forceinline__ uint32_t mhc_checked_word(MhcMatrix w,uint64_t at) {
    if(at+4<=w.bytes)return __ldg(reinterpret_cast<const uint32_t*>(w.q+at));
    uint32_t value=0;
    for(int byte=0;byte<4;++byte)if(at+byte<w.bytes)value|=uint32_t(w.q[at+byte])<<(8*byte);
    return value;
}
__device__ __forceinline__ float mhc_word_code(MhcMatrix w,uint64_t bit,int bits) {
    const uint64_t at=(bit>>5)*4;const int shift=int(bit&31u);
    const uint32_t lo=mhc_checked_word(w,at);
    const uint32_t hi=shift+bits>32?mhc_checked_word(w,at+4):0;
    return float(__funnelshift_r(lo,hi,shift)&((1u<<bits)-1));
}

template<int FixedGroup=0,bool StaticUnsignedQ8=false,int Cooperative=0>
__device__ __forceinline__ float mhc_affine(MhcMatrix w,const float* normalized,int row,int first,int last,int lane) {
    if constexpr(!StaticUnsignedQ8)if(w.dense) {
        float sum=0.f;
        for(int group=first;group<last;++group) {
            const int column=group*w.gs+lane;
            if(column<w.width)sum=fmaf(__bfloat162float(w.dense[size_t(row)*w.width+column]),normalized[column-first*w.gs],sum);
        }
        for(int stride=16;stride;stride>>=1)sum+=__shfl_down_sync(0xffffffff,sum,stride);
        return sum;
    }
    if(StaticUnsignedQ8 || (w.aligned && !w.zero)) {
        const int gs=FixedGroup?FixedGroup:w.gs;
        const uint8_t* packed=w.q+(w.offsets[row]>>3);
        const float outer_scale=w.row_scale[row],outer_minimum=w.row_minimum[row];
        float sum=0.f;
        for(int group=first;group<last;++group) {
            const size_t meta=size_t(row)*w.groups+group;
            const float scale=__fmul_rn(outer_scale,float(w.scales[meta]));
            const float minimum=__fmul_rn(outer_minimum,float(w.minima[meta]));
            #pragma unroll
            for(int chunk=0;chunk<2;++chunk) {
                const int j=lane+chunk*32,column=group*gs+j;
                if(j<gs && column<w.width)
                    sum=fmaf(__fsub_rn(__fmul_rn(scale,float(packed[column])),minimum),normalized[column-first*gs],sum);
            }
        }
        for(int stride=16;stride;stride>>=1)sum+=__shfl_down_sync(0xffffffff,sum,stride);
        return sum;
    }
    if constexpr(!StaticUnsignedQ8) {
    const int bits=w.zero || w.aligned?8:w.bits[row];
    const uint64_t offset=w.zero?0:uint64_t(w.offsets[row]);
    const float outer_scale=w.zero?0.f:w.row_scale[row],outer_minimum=w.zero?0.f:w.row_minimum[row];
    float sum=0.f;
    for(int group=first;group<last;++group) {
        const size_t meta=size_t(row)*w.groups+group;
        const float scale=w.zero?__half2float(w.zero_scale[meta]):__fmul_rn(outer_scale,float(w.scales[meta]));
        const float minimum=w.zero?0.f:__fmul_rn(outer_minimum,float(w.minima[meta]));
        float codes[2]{};
        if constexpr(Cooperative==1)if(!w.zero && !w.aligned) {
            // All lanes participate, including inactive GS24/28 tails. Packed
            // words are read once by producer lanes and shared in registers;
            // the consumer's original FP32 FMA and reduction order is retained.
            const auto packed=mhc_group_read(w,offset+uint64_t(group)*w.gs*bits,bits,lane);
            codes[0]=mhc_group_code(packed,lane);
            if(w.gs>32)codes[1]=mhc_group_code(packed,lane+32);
        }
        #pragma unroll
        for(int chunk=0;chunk<2;++chunk) {
            const int j=lane+chunk*32;if(j>=w.gs)continue;
            const int column=group*w.gs+j;if(column>=w.width)continue;
            float q;
            if(w.zero)q=float(static_cast<const int8_t*>(static_cast<const void*>(w.q))[size_t(row)*w.groups*w.gs+column]);
            else if(w.aligned)q=float(w.q[(offset>>3)+column]);
            else {
                if constexpr(Cooperative==1)q=codes[chunk];
                else if constexpr(Cooperative==2)q=mhc_word_code(w,offset+uint64_t(column)*bits,bits);
                else {
                const uint64_t bit=offset+uint64_t(column)*bits;const int shift=bit&7;
                unsigned value=w.q[bit>>3];if(shift+bits>8)value|=unsigned(w.q[(bit>>3)+1])<<8;
                q=float((value>>shift)&((1u<<bits)-1));
                }
            }
            sum=fmaf(__fsub_rn(__fmul_rn(scale,q),minimum),normalized[column-first*w.gs],sum);
        }
    }
    for(int stride=16;stride;stride>>=1)sum+=__shfl_down_sync(0xffffffff,sum,stride);
    return sum;
    }else return 0.f; // The static aligned-q8 branch above always returns.
}

template<bool CacheUpdated=false,bool StatsOnly=false>
__global__ void gated_residual_norm_kernel(MhcInput input,void* updated,void* normalized,float* promoted_normalized,float eps,const float* prepared_right,float* saved_inverse=nullptr) {
    const int sample=blockIdx.y,tid=threadIdx.x;
    if constexpr(CacheUpdated)input.hidden=2560;
    float cached[CacheUpdated?10:1];
    constexpr int unroll_columns=CacheUpdated?10:1;
    __shared__ float partial[256],inverse[4],prior_gates[4];
    if(tid<input.streams)prior_gates[tid]=input.after?mhc_read(input.injection,size_t(sample)*input.streams+tid,input.injection_half):0.f;
    input.cached_gates=prior_gates;
    __syncthreads();
    for(int s=int(blockIdx.x);s<=int(blockIdx.x);++s) {
        float sum=0.f;
        #pragma unroll unroll_columns
        for(int item=0;item<(CacheUpdated?10:(input.hidden+255-tid)/256);++item) {
            const int column=tid+256*item;
            const float value=mhc_updated(input,sample,s,column);
            if constexpr(StatsOnly)mhc_write(updated,(size_t(sample)*input.streams+s)*input.hidden+column,value,input.output_half);
            if constexpr(CacheUpdated)cached[item]=value;
            sum=__fadd_rn(sum,__fmul_rn(value,value));
        }
        partial[tid]=sum;
        __syncthreads();
        if(tid<32) {
            const float a=__fadd_rn(partial[tid],partial[tid+128]);
            const float b=__fadd_rn(partial[tid+64],partial[tid+192]);
            const float c=__fadd_rn(partial[tid+32],partial[tid+160]);
            const float d=__fadd_rn(partial[tid+96],partial[tid+224]);
            float value=__fadd_rn(__fadd_rn(a,b),__fadd_rn(c,d));
            #pragma unroll
            for(int stride=16;stride>0;stride>>=1) {
                const float other=__shfl_down_sync(0xffffffff,value,stride);
                if(tid<stride)value=__fadd_rn(value,other);
            }
            if(tid==0)inverse[s]=mhc_norm_inverse(value,input.hidden,eps,input.fast_norm);
        }
        __syncthreads();
    }
    if constexpr(StatsOnly) {
        if(tid==0)saved_inverse[size_t(sample)*input.streams+blockIdx.x]=inverse[blockIdx.x];
    }else {
        for(int s=int(blockIdx.x);s<=int(blockIdx.x);++s) {
        #pragma unroll unroll_columns
        for(int item=0;item<(CacheUpdated?10:(input.hidden+255-tid)/256);++item) {
            const int column=tid+256*item;
            const size_t at=(size_t(sample)*input.streams+s)*input.hidden+column;
            const float value=CacheUpdated?cached[item]:mhc_updated(input,sample,s,column);
            mhc_write(updated,at,value,input.output_half);
            const float norm_value=mhc_round(__fmul_rn(__fmul_rn(value,inverse[s]),__fadd_rn(input.norm[s*input.hidden+column],1.f)),input.output_half);
            mhc_write(normalized,at,norm_value,input.output_half);
            if(promoted_normalized)promoted_normalized[at]=norm_value;
        }
        }

        if(prepared_right && blockIdx.x==0 && blockIdx.y==0) {
            const char* address=reinterpret_cast<const char*>(prepared_right);
            const size_t bytes=size_t(input.hidden)*input.streams*input.streams*sizeof(float);
            for(size_t at=size_t(tid)*128;at<bytes;at+=size_t(blockDim.x)*128)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(address+at));
        }
    }
}

#include "gr_norm_quant.cuh"
#include "gr_norm_wide.cuh"
#include "gr_norm_vector_stats.cuh"
#include "gr_norm_replicated.cuh"
// Strata's GR staging uses cp.async for aligned float tiles. Keep the
// activation bytes and lane arithmetic unchanged; unaligned slices use scalar
// copies. Only Float normalization output uses the asynchronous path.
__device__ __forceinline__ void mhc_copy_normalized(float* destination,const void* source,
        int count,bool half,bool asynchronous) {
    const int tid=threadIdx.x;
    const bool aligned=((reinterpret_cast<std::uintptr_t>(source)|
        reinterpret_cast<std::uintptr_t>(destination))&15u)==0;
    if(asynchronous && !half && aligned) {
        const auto* input=static_cast<const float*>(source);
        for(int column=tid*4;column+4<=count;column+=blockDim.x*4) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            const unsigned shared_address=unsigned(__cvta_generic_to_shared(destination+column));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                "r"(shared_address),"l"(input+column):"memory");
#else
            *reinterpret_cast<float4*>(destination+column)=*reinterpret_cast<const float4*>(input+column);
#endif
        }
        for(int column=(count&~3)+tid;column<count;column+=blockDim.x)
            destination[column]=input[column];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
    }else for(int column=tid;column<count;column+=blockDim.x)
        destination[column]=mhc_read(source,column,half);
}

#include "gr_part_vectors.cuh"
#include "gr_up_half_prepare.cuh"
#include "gr_down_prepare.cuh"
#include "gr_down_dp2a.cuh"

template<int FixedGroup=0,int ProjectionThreads=256,int InjectionKind=0,bool StaticFormat=false,bool VectorInjection=false,int Cooperative=0,int PlaneCount=0,bool PrepareUp=false,bool PackedPrepare=false,bool VectorParts=false,bool SplitMajor=false,bool PairedStats=false,bool MmaUp=false,bool HalfUp=false,int PackedDown=0>
__global__ void gated_residual_projection_kernel(MhcInput input,MhcMatrix down,MhcMatrix inject,
    const void* normalized,float* parts,int down_splits,int injection_splits,int row_tile,int down_capacity,
    const float* dense_input,bool asynchronous,GrTwoPlane global_quantized,
    GrTwoPlane prepared_up={},int prepare_counter_slot=-1) {
    const int sample=blockIdx.y,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int down_blocks=(down.outputs+row_tile-1)/row_tile;
    const int split=blockIdx.x%down_splits,row_base=(blockIdx.x/down_splits)*row_tile;
    const int first=down.groups*split/down_splits,last=down.groups*(split+1)/down_splits;
    const int begin_column=first*down.gs,end_column=min(last*down.gs,down.width);
    extern __shared__ __align__(16) float normalized_shared[];
    const size_t base=size_t(sample)*(down.outputs*down_splits+inject.outputs*injection_splits);
    {
        if constexpr(PlaneCount>0) {
            auto q=global_quantized;
            const size_t group_base=size_t(sample)*down.groups+first;
            q.high+=group_base*down.gs*(PackedDown?2:1);q.low+=group_base*down.gs;
            if constexpr(GrDotFlags<PlaneCount>::triple)q.middle+=group_base*down.gs;
            q.scales+=group_base;q.sums+=group_base;
            for(int row=row_base+warp;row<row_base+row_tile && row<down.outputs;row+=ProjectionThreads/32) {
                float sum;
                if constexpr(PackedDown)sum=gr_down_direct_s16<PackedDown==2>(down,q,row,first,last,lane);
                else if constexpr(FixedGroup==24)sum=gr_packed24_affine<PlaneCount>(down,q,row,first,last,lane);
                else sum=gr_dp4a2_affine<FixedGroup,GrDotFlags<PlaneCount>::triple,GrDotFlags<PlaneCount>::unsigned_low,
                    false,GrDotFlags<PlaneCount>::combined,GrDotFlags<PlaneCount>::unsigned_weights>(down,q,row,first,last,lane);
                if(lane==0)parts[base+(SplitMajor?split*down.outputs+row:row*down_splits+split)]=sum;
            }
        }else {
        const size_t byte_offset=(size_t(sample)*down.width+begin_column)*(input.output_half?sizeof(__half):sizeof(float));
        mhc_copy_normalized(normalized_shared,static_cast<const char*>(normalized)+byte_offset,
            end_column-begin_column,input.output_half,asynchronous);
        __syncthreads();
        for(int row=row_base+warp;row<row_base+row_tile && row<down.outputs;row+=ProjectionThreads/32) {
            const float sum=mhc_affine<FixedGroup,StaticFormat,Cooperative>(down,normalized_shared,row,first,last,lane);
            if(lane==0)parts[base+row*down_splits+split]=sum;
        }
        }
    }
    if constexpr(InjectionKind!=2)if(dense_input && tid<32) {
        const int per_block=32/inject.outputs;
        const int virtual_lane=int(blockIdx.x)*per_block+tid/inject.outputs;
        const int row=tid%inject.outputs;
        if(tid<per_block*inject.outputs && virtual_lane<32) {
            float sum=0.f;
            for(int begin=virtual_lane*4;begin<inject.width;begin+=128) {
                if constexpr(VectorInjection) {
                    const auto x=*reinterpret_cast<const float4*>(dense_input+size_t(sample)*inject.width+begin);
                    const auto w=*reinterpret_cast<const float4*>(inject.row_scale+size_t(begin)*inject.outputs+row*4);
                    sum=fmaf(x.x,w.x,sum);sum=fmaf(x.y,w.y,sum);
                    sum=fmaf(x.z,w.z,sum);sum=fmaf(x.w,w.w,sum);
                } else {
                #pragma unroll
                for(int item=0;item<4;++item)if(begin+item<inject.width)
                    sum=fmaf(dense_input[size_t(sample)*inject.width+begin+item],
                        inject.row_scale[(size_t(begin)+item)*inject.outputs+row],sum);
                }
            }
            parts[base+down.outputs*down_splits+row*injection_splits+virtual_lane]=sum;
        }
    }
    if constexpr(InjectionKind!=1)if(InjectionKind==2 || !inject.dense) {
        const int first_inject=((begin_column+inject.gs-1)/inject.gs*injection_splits+inject.groups-1)/inject.groups;
        const int last_inject=min(injection_splits,((end_column+inject.gs-1)/inject.gs*injection_splits+inject.groups-1)/inject.groups);
        for(int part=first_inject+int(blockIdx.x)/down_splits;part<last_inject;part+=down_blocks) {
        const int begin=inject.groups*part/injection_splits,end=inject.groups*(part+1)/injection_splits;
        float* slice=normalized_shared+down_capacity;
        const int injection_end=end*inject.gs<inject.width?end*inject.gs:inject.width;
        const size_t byte_offset=(size_t(sample)*down.width+begin*inject.gs)*(input.output_half?sizeof(__half):sizeof(float));
        mhc_copy_normalized(slice,static_cast<const char*>(normalized)+byte_offset,
            injection_end-begin*inject.gs,input.output_half,asynchronous);
        __syncthreads();
        if(warp<inject.outputs) {
            const float sum=mhc_affine<0,InjectionKind==2>(inject,slice,warp,begin,end,lane);
            if(lane==0)parts[base+down.outputs*down_splits+warp*injection_splits+part]=sum;
        }
        __syncthreads();
        }
    }
    if constexpr(PrepareUp)
        gr_down_finish_up_prepare<ProjectionThreads,PackedPrepare,VectorParts,SplitMajor,PairedStats,MmaUp,HalfUp>(parts,down_splits,row_tile,row_base,
            input.output_half,prepared_up,prepare_counter_slot);
}

template<int FixedGroup=0,int PlaneCount=0,bool PrepareUp=false,int FixedKind=0>
__global__ void __launch_bounds__(256,PlaneCount>0?4:1) gated_residual_down_kernel(MhcInput input,MhcMatrix down,MhcMatrix inject,
    void* updated,void* normalized,float* promoted_normalized,float* parts,int down_splits,int injection_splits,float eps,int row_tile,int down_capacity,
    GrTwoPlane prepared_up={},int prepare_counter_slot=-1) {
    if constexpr(PlaneCount>0) {
        input.hidden=2560;input.streams=4;
        down.width=10240;down.outputs=320;down.groups=214;down.gs=48;
        if constexpr(FixedKind>0) {
            input.residual_half=true;input.output_half=true;
            input.after=FixedKind==2;input.branch_half=FixedKind==2;input.injection_half=FixedKind==2;
        }
    }
    const int sample=blockIdx.y,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int down_blocks=(down.outputs+row_tile-1)/row_tile;
    const int split=blockIdx.x%down_splits,row_base=(blockIdx.x/down_splits)*row_tile;
    const int first=down.groups*split/down_splits,last=down.groups*(split+1)/down_splits;
    const int begin_column=first*down.gs,end_column=min(last*down.gs,down.width);
    const int first_inject=((begin_column+inject.gs-1)/inject.gs*injection_splits+inject.groups-1)/inject.groups;
    const int last_inject=min(injection_splits,((end_column+inject.gs-1)/inject.gs*injection_splits+inject.groups-1)/inject.groups);
    int first_stream=begin_column/input.hidden,last_stream=(end_column-1)/input.hidden;
    if(first_inject+int(blockIdx.x)/down_splits<last_inject)
        last_stream=max(last_stream,(min(inject.groups*last_inject/injection_splits*inject.gs,inject.width)-1)/input.hidden);
    if constexpr(PlaneCount==0)if(blockIdx.x==0) {first_stream=0;last_stream=input.streams-1;}
    __shared__ float partial[256],inverse[4],prior_gates[4];
    extern __shared__ float normalized_shared[];
    if(tid<input.streams)prior_gates[tid]=input.after?mhc_read(input.injection,size_t(sample)*input.streams+tid,input.injection_half):0.f;
    input.cached_gates=prior_gates;
    __syncthreads();
    for(int s=first_stream;s<=last_stream;++s) {
        float sum=0.f;
        for(int column=tid;column<input.hidden;column+=256) {
            const float value=mhc_updated(input,sample,s,column);
            sum=__fadd_rn(sum,__fmul_rn(value,value));
        }
        partial[tid]=sum;
        __syncthreads();
        if constexpr(PlaneCount==0)for(int stride=128;stride>=32;stride>>=1) {
            if(tid<stride)partial[tid]=__fadd_rn(partial[tid],partial[tid+stride]);
            __syncthreads();
        }
        if(tid<32) {
            float value;
            if constexpr(PlaneCount>0) {
                const float a=__fadd_rn(partial[tid],partial[tid+128]);
                const float b=__fadd_rn(partial[tid+64],partial[tid+192]);
                const float c=__fadd_rn(partial[tid+32],partial[tid+160]);
                const float d=__fadd_rn(partial[tid+96],partial[tid+224]);
                value=__fadd_rn(__fadd_rn(a,b),__fadd_rn(c,d));
            }else value=partial[tid];
            #pragma unroll
            for(int stride=16;stride>0;stride>>=1) {
                const float other=__shfl_down_sync(0xffffffff,value,stride);
                if(tid<stride)value=__fadd_rn(value,other);
            }
            if(tid==0)inverse[s]=mhc_norm_inverse(value,input.hidden,eps,input.fast_norm);
        }
        __syncthreads();
    }
    __syncthreads();
    if constexpr(PlaneCount==0)if(blockIdx.x==0)for(int s=0;s<input.streams;++s)for(int column=tid;column<input.hidden;column+=256) {
        const size_t at=(size_t(sample)*input.streams+s)*input.hidden+column;
        const float value=mhc_updated(input,sample,s,column);
        mhc_write(updated,at,value,input.output_half);
        const float norm_value=mhc_round(__fmul_rn(__fmul_rn(value,inverse[s]),__fadd_rn(input.norm[s*input.hidden+column],1.f)),input.output_half);
        mhc_write(normalized,at,norm_value,input.output_half);
        if(promoted_normalized)promoted_normalized[at]=norm_value;
    }
    const size_t base=size_t(sample)*(down.outputs*down_splits+inject.outputs*injection_splits);
    {
        if constexpr(PlaneCount>0) {
            for(int column=begin_column+tid;column<end_column;column+=256) {
                const int stream=column/2560,local=column%2560;
                const float value=mhc_updated(input,sample,stream,local);
                const float norm_value=mhc_round(__fmul_rn(__fmul_rn(value,inverse[stream]),
                    __fadd_rn(input.norm[column],1.f)),input.output_half);
                normalized_shared[column-begin_column]=norm_value;
                if(row_base==0) {
                    const size_t at=size_t(sample)*10240+column;
                    mhc_write(updated,at,value,input.output_half);
                    mhc_write(normalized,at,norm_value,input.output_half);
                }
            }
        }else mhc_normalized_slice(input,inverse,sample,begin_column,end_column,normalized_shared);
        __syncthreads();
        GrTwoPlane q{};
        if constexpr(PlaneCount>0) {
            __shared__ __align__(16) int8_t activation_codes[2*32*48];
            __shared__ float2 activation_scales[32];
            __shared__ int4 activation_sums[32];
            static_assert(FixedGroup==48 && PlaneCount==8);
            q={activation_codes,nullptr,activation_codes+32*48,activation_scales,activation_sums};
            MhcMatrix local{};local.width=end_column-begin_column;local.groups=last-first;
            for(int group=warp;group<local.groups;group+=8)
                gr_quantize_activation_group<48,false,true>(local,normalized_shared,false,q,group,0,lane);
            __syncthreads();
        }
        for(int row=row_base+warp;row<row_base+row_tile && row<down.outputs;row+=8) {
            float sum;
            if constexpr(PlaneCount>0)sum=gr_dp4a2_affine<48,false,true,false,false,true>(down,q,row,first,last,lane);
            else sum=mhc_affine<FixedGroup>(down,normalized_shared,row,first,last,lane);
            if(lane==0)parts[base+(PrepareUp?split*down.outputs+row:row*down_splits+split)]=sum;
        }
        __syncthreads();
    }
    if(!inject.dense)for(int part=first_inject+int(blockIdx.x)/down_splits;part<last_inject;part+=down_blocks) {
        const int begin=inject.groups*part/injection_splits,end=inject.groups*(part+1)/injection_splits;
        float* slice=normalized_shared+down_capacity;
        const int injection_end=end*inject.gs<inject.width?end*inject.gs:inject.width;
        mhc_normalized_slice(input,inverse,sample,begin*inject.gs,injection_end,slice);
        __syncthreads();
        if(warp<inject.outputs) {
            const float sum=mhc_affine(inject,slice,warp,begin,end,lane);
            if(lane==0)parts[base+down.outputs*down_splits+warp*injection_splits+part]=sum;
        }
        __syncthreads();
    }

    // A single existing CTA warms only prepared FP32 injection cache lines.
    // No additional kernel, projection, or weight arithmetic is introduced.
    if(inject.dense && inject.row_scale && blockIdx.x==0 && blockIdx.y==0) {
        const char* address=reinterpret_cast<const char*>(inject.row_scale);
        const size_t bytes=size_t(inject.width)*inject.outputs*sizeof(float);
        for(size_t at=size_t(tid)*128;at<bytes;at+=size_t(blockDim.x)*128)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(address+at));
    }
    if constexpr(PrepareUp)
        gr_down_finish_up_prepare<256,true,false,true,true>(parts,down_splits,row_tile,row_base,
            input.output_half,prepared_up,prepare_counter_slot);
}

// Two speculative rows share the original q8 weight decode. Each row keeps
// its scalar FMA order and the same split/reduction tree as mhc_affine.
// Read the small normalized batch through the cache instead of allocating a
// full shared-memory copy per row, keeping enough CTAs for short batches.
template<int GroupSize,int Capacity=2,int PlaneCount=0>
__global__ void __launch_bounds__(128) gated_residual_batch_projection_kernel(
    MhcInput input,MhcMatrix down,MhcMatrix inject,const void* normalized,
    float* parts,int down_splits,int injection_splits,int samples,bool vector_injection,GrTwoPlane quantized={}) {
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int split=blockIdx.x%down_splits,row=(blockIdx.x/down_splits)*4+warp;
    const int sample=int(blockIdx.y)*Capacity;
    const int first=down.groups*split/down_splits,last=down.groups*(split+1)/down_splits;
    const int stride=down.outputs*down_splits+inject.outputs*injection_splits;
    if(row<down.outputs) {
        float sums[Capacity]{};
        if constexpr(PlaneCount>0) {
            if constexpr(GroupSize==24)gr_packed24_affine_batch<Capacity,PlaneCount>(
                down,quantized,row,first,last,sample,samples,lane,sums);
            else gr_dp4a_affine_batch<GroupSize,Capacity,GrDotFlags<PlaneCount>::triple,GrDotFlags<PlaneCount>::unsigned_low,
                GrDotFlags<PlaneCount>::combined,GrDotFlags<PlaneCount>::unsigned_weights>(
                down,quantized,row,first,last,sample,samples,lane,sums);
        }else {
        const auto* packed=down.q+(down.offsets[row]>>3);
        const float outer_scale=down.row_scale[row],outer_minimum=down.row_minimum[row];
        for(int group=first;group<last;++group) {
            const size_t meta=size_t(row)*down.groups+group;
            const float scale=__fmul_rn(outer_scale,float(down.scales[meta]));
            const float minimum=__fmul_rn(outer_minimum,float(down.minima[meta]));
            #pragma unroll
            for(int chunk=0;chunk<2;++chunk) {
                const int j=lane+chunk*32,column=group*GroupSize+j;
                if(j<GroupSize && column<down.width) {
                    const float weight=__fsub_rn(__fmul_rn(scale,float(packed[column])),minimum);
                    #pragma unroll
                    for(int k=0;k<Capacity;++k)if(sample+k<samples)
                        sums[k]=fmaf(weight,mhc_read(normalized,size_t(sample+k)*down.width+column,input.output_half),sums[k]);
                }
            }
        }
        for(int shift=16;shift;shift>>=1) {
            #pragma unroll
            for(int k=0;k<Capacity;++k)sums[k]+=__shfl_down_sync(0xffffffffu,sums[k],shift);
        }
        }
        if(lane==0) {
            #pragma unroll
            for(int k=0;k<Capacity;++k)if(sample+k<samples)
                parts[size_t(sample+k)*stride+row*down_splits+split]=sums[k];
        }
    }
    if(inject.dense && tid<32) {
        const int per_block=32/inject.outputs;
        const int virtual_lane=int(blockIdx.x)*per_block+tid/inject.outputs;
        const int output=tid%inject.outputs;
        if(tid<per_block*inject.outputs && virtual_lane<32) {
            float sums[Capacity]{};
            for(int begin=virtual_lane*4;begin<inject.width;begin+=128) {
                #pragma unroll
                for(int item=0;item<4;++item)if(begin+item<inject.width) {
                    const size_t at=vector_injection?size_t(begin)*inject.outputs+output*4+item:
                        size_t(begin+item)*inject.outputs+output;
                    const float weight=inject.row_scale[at];
                    #pragma unroll
                    for(int k=0;k<Capacity;++k)if(sample+k<samples)
                        sums[k]=fmaf(mhc_read(normalized,size_t(sample+k)*inject.width+begin+item,input.output_half),weight,sums[k]);
                }
            }
            #pragma unroll
            for(int k=0;k<Capacity;++k)if(sample+k<samples)
                parts[size_t(sample+k)*stride+down.outputs*down_splits+output*injection_splits+virtual_lane]=sums[k];
        }
    }
    if(!inject.dense && split==0) {
        extern __shared__ float injection_input[];
        const int blocks=(down.outputs+3)/4;
        for(int part=int(blockIdx.x)/down_splits;part<injection_splits;part+=blocks) {
            const int begin=inject.groups*part/injection_splits,end=inject.groups*(part+1)/injection_splits;
            const int first_column=begin*inject.gs,last_column=min(end*inject.gs,inject.width);
            const int inject_capacity=((inject.groups+injection_splits-1)/injection_splits)*inject.gs;
            if constexpr(GroupSize==24) {
                #pragma unroll
                for(int k=0;k<Capacity;++k)if(sample+k<samples)
                    for(int column=first_column+tid;column<last_column;column+=blockDim.x)
                        injection_input[k*inject_capacity+column-first_column]=
                            mhc_read(normalized,size_t(sample+k)*inject.width+column,input.output_half);
                __syncthreads();
            }
            if(warp<inject.outputs) {
                const auto* packed=inject.q+(inject.offsets[warp]>>3);
                const float outer_scale=inject.row_scale[warp],outer_minimum=inject.row_minimum[warp];
                float sums[Capacity]{};
                for(int group=begin;group<end;++group) {
                    const size_t meta=size_t(warp)*inject.groups+group;
                    const float scale=__fmul_rn(outer_scale,float(inject.scales[meta]));
                    const float minimum=__fmul_rn(outer_minimum,float(inject.minima[meta]));
                    #pragma unroll
                    for(int chunk=0;chunk<2;++chunk) {
                        const int j=lane+chunk*32,column=group*inject.gs+j;
                        if(j<inject.gs && column<inject.width) {
                            const float weight=__fsub_rn(__fmul_rn(scale,float(packed[column])),minimum);
                            #pragma unroll
                            for(int k=0;k<Capacity;++k)if(sample+k<samples)
                                sums[k]=fmaf(weight,GroupSize==24?
                                    injection_input[k*inject_capacity+column-first_column]:
                                    mhc_read(normalized,size_t(sample+k)*inject.width+column,input.output_half),sums[k]);
                        }
                    }
                }
                for(int shift=16;shift;shift>>=1) {
                    #pragma unroll
                    for(int k=0;k<Capacity;++k)sums[k]+=__shfl_down_sync(0xffffffffu,sums[k],shift);
                }
                if(lane==0) {
                    #pragma unroll
                    for(int k=0;k<Capacity;++k)if(sample+k<samples)
                        parts[size_t(sample+k)*stride+down.outputs*down_splits+warp*injection_splits+part]=sums[k];
                }
            }
            if constexpr(GroupSize==24)__syncthreads();
        }
    }
}

// All Up output blocks consume the same 320-dimensional activated Down.
// Prepare its integer planes once per token rather than once per output CTA.
template<bool Uniform16=false,bool Packed16=false,bool VectorParts=false,bool SplitMajor=false,bool PairedStats=false>
__global__ void __launch_bounds__(256) gated_residual_prepare_up_integer_kernel(
        MhcMatrix down,MhcMatrix up,const float* parts,GrTwoPlane q,
        int down_splits,int injection_stride,int streams,bool half) {
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,sample=blockIdx.x;
    const size_t base=size_t(sample)*(down.outputs*down_splits+injection_stride);
    const int elements=up.groups*48;
    q.high+=sample*elements*(Packed16?2:1);
    if constexpr(!Uniform16)q.middle+=sample*elements;
    q.low+=sample*elements;
    q.scales+=sample*up.groups;q.sums+=sample*up.groups;
    if constexpr(Packed16) {
        // Each warp owns one GS48 group. Reduce and activate the original
        // Down parts directly in registers, avoiding a shared-memory round trip.
        if(warp<up.groups) {
            const auto activate=[&](int row) {
                if(row>=320)return 0.f;
                float sum;
                if constexpr(SplitMajor)sum=gr_reduce_split_major_parts<false>(parts,base,row,320,down_splits);
                else if constexpr(VectorParts)sum=gr_reduce_down_parts<false>(parts,base,row,down_splits);
                else {
                    sum=down_splits==1?parts[base+row]:0.f;
                    if(down_splits>1)for(int split=0;split<down_splits;++split)
                        sum+=parts[base+row*down_splits+split];
                }
                return mhc_activated(sum,half,streams,false);
            };
            const int column=warp*48+lane;
            const float x0=activate(column),x1=lane<16?activate(column+32):0.f;
            gr_quantize_activation_values<48,false,true,true,PairedStats>(up,x0,x1,q,warp,0,lane);
        }
    }else {
        __shared__ float low[320];
        for(int row=tid;row<320;row+=256) {
            float sum=down_splits==1?parts[base+row]:0.f;
            if(down_splits>1)for(int split=0;split<down_splits;++split)
                sum+=parts[base+row*down_splits+split];
            low[row]=mhc_activated(sum,half,streams,false);
        }
        __syncthreads();
        if(warp<up.groups)gr_quantize_activation_group<48,!Uniform16,Uniform16>(up,low,false,q,warp,0,lane);
    }
}

#include "gr_up_mma.cuh"
#include "gr_up_half_mma.cuh"
#include "gr_up_int16_mma.cuh"
#include "gr_down_mma.cuh"

template<int FixedGroup=0,int FixedStreams=0,int FixedWidth=0,int Capacity=1,int MetadataMode=0,int IntegerUp=0>
__global__ void __launch_bounds__(Capacity==4?256:512,Capacity==4?(IntegerUp==6?4:3):IntegerUp==7?2:(Capacity==2 && FixedWidth>0)?2:1)
gated_residual_up_kernel(MhcMatrix down,MhcMatrix up,MhcMatrix inject,
    const void* normalized,const float* parts,void* mixed,void* injection,
    int hidden,int streams,bool half,int down_splits,int injection_splits,const float* dense_projection,int samples,
    GrTwoPlane prepared_up={},const float2* group_coefficients=nullptr,
    const int4* row_records=nullptr,const uint16_t* group_records=nullptr) {
    extern __shared__ float low[];
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,sample_begin=int(blockIdx.y)*Capacity;
    if constexpr(FixedStreams>0)streams=FixedStreams;
    #pragma unroll
    for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
    const int sample=sample_begin+k;
    const size_t base=size_t(sample)*(down.outputs*down_splits+inject.outputs*injection_splits);
    if constexpr(IntegerUp!=7 && IntegerUp!=8 && IntegerUp!=10 && IntegerUp!=12 && IntegerUp!=15 && IntegerUp!=16 && IntegerUp!=17 && IntegerUp!=18 && IntegerUp!=19 && IntegerUp!=20 && IntegerUp!=21 && IntegerUp!=22 && IntegerUp!=23 && IntegerUp!=24)for(int row=tid;row<down.outputs;row+=blockDim.x) {
        float sum=down_splits==1?parts[base+row]:0.f;
        if(down_splits>1)for(int split=0;split<down_splits;++split)sum+=parts[base+row*down_splits+split];
        low[k*down.outputs+row]=mhc_activated(sum,half,streams,false);
    }
    if(inject.dense && !dense_projection && blockIdx.x==0 && tid<32) {
        #pragma unroll
        for(int row=0;row<4;++row)if(row<inject.outputs) {
            float sum=parts[base+down.outputs*down_splits+row*injection_splits+lane];
            #pragma unroll
            for(int shift=16;shift>0;shift>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffffu,sum,shift));
            if(lane==0)mhc_write(injection,size_t(sample)*inject.outputs+row,
                mhc_activated(__fadd_rn(sum,0.f),false,streams,true),false);
        }
    }
    if(blockIdx.x==0 && tid<inject.outputs && (!inject.dense || dense_projection)) {
        const size_t start=base+down.outputs*down_splits+tid*injection_splits;
        float sum=inject.dense?dense_projection[size_t(sample)*inject.outputs+tid]:(injection_splits==1?parts[start]:0.f);
        if(!inject.dense && injection_splits>1)for(int split=0;split<injection_splits;++split)sum+=parts[start+split];
        const bool injection_half=half && !inject.dense;
        mhc_write(injection,size_t(sample)*inject.outputs+tid,mhc_activated(sum,injection_half,streams,true),injection_half);
    }
    }
    if constexpr(IntegerUp!=7 && IntegerUp!=8 && IntegerUp!=10 && IntegerUp!=12 && IntegerUp!=15 && IntegerUp!=16 && IntegerUp!=17 && IntegerUp!=18 && IntegerUp!=19 && IntegerUp!=20 && IntegerUp!=21 && IntegerUp!=22 && IntegerUp!=23 && IntegerUp!=24)__syncthreads();
    constexpr int quantized_groups=IntegerUp?(FixedWidth+FixedGroup-1)/FixedGroup:1;
    constexpr int quantized_elements=quantized_groups*(IntegerUp?FixedGroup:1);
    constexpr bool three_up=IntegerUp==3 || IntegerUp==5 || IntegerUp==6 || IntegerUp==7 || IntegerUp==8 || (IntegerUp>=9 && IntegerUp!=13 && IntegerUp!=14 && IntegerUp!=15 && IntegerUp!=16 && IntegerUp!=17 && IntegerUp!=18 && IntegerUp!=19 && IntegerUp!=20 && IntegerUp!=21 && IntegerUp!=22 && IntegerUp!=23 && IntegerUp!=24);
    constexpr int up_plane_count=three_up?3:2;
    constexpr bool local_prepare=IntegerUp && IntegerUp!=8 && IntegerUp!=15 && IntegerUp!=16 && IntegerUp!=17 && IntegerUp!=18 && IntegerUp!=19 && IntegerUp!=20 && IntegerUp!=21 && IntegerUp!=22 && IntegerUp!=23 && IntegerUp!=24;
    __shared__ __align__(16) int8_t up_planes[local_prepare?Capacity*quantized_elements*up_plane_count:1];
    __shared__ float2 up_scales[local_prepare?Capacity*quantized_groups:1];
    __shared__ int4 up_sums[local_prepare?Capacity*quantized_groups:1];
    GrTwoPlane quantized_up{up_planes,three_up && local_prepare?up_planes+Capacity*quantized_elements:nullptr,
        up_planes+(local_prepare?(up_plane_count-1)*Capacity*quantized_elements:0),up_scales,up_sums};
    if constexpr(IntegerUp==15 || IntegerUp==16 || IntegerUp==17 || IntegerUp==18 || IntegerUp==19 || IntegerUp==20 || IntegerUp==21 || IntegerUp==22 || IntegerUp==23 || IntegerUp==24) {
        quantized_up=prepared_up;
        quantized_up.high+=sample_begin*quantized_elements*(IntegerUp>=19?2:1);
        quantized_up.low+=sample_begin*quantized_elements;
        quantized_up.scales+=sample_begin*quantized_groups;
        quantized_up.sums+=sample_begin*quantized_groups;
    }else if constexpr(IntegerUp==8) {
        quantized_up=prepared_up;
        quantized_up.high+=sample_begin*quantized_elements;
        quantized_up.middle+=sample_begin*quantized_elements;
        quantized_up.low+=sample_begin*quantized_elements;
        quantized_up.scales+=sample_begin*quantized_groups;
        quantized_up.sums+=sample_begin*quantized_groups;
    }else if constexpr(IntegerUp==7 || IntegerUp==10 || IntegerUp==12) {
        constexpr int words=quantized_elements/16;
        for(int word=tid;word<Capacity*words;word+=blockDim.x)if(sample_begin+word/words<samples) {
            const int source=sample_begin*words+word;
            reinterpret_cast<int4*>(quantized_up.high)[word]=reinterpret_cast<const int4*>(prepared_up.high)[source];
            reinterpret_cast<int4*>(quantized_up.middle)[word]=reinterpret_cast<const int4*>(prepared_up.middle)[source];
            reinterpret_cast<int4*>(quantized_up.low)[word]=reinterpret_cast<const int4*>(prepared_up.low)[source];
        }
        for(int at=tid;at<Capacity*quantized_groups;at+=blockDim.x)if(sample_begin+at/quantized_groups<samples) {
            const int source=sample_begin*quantized_groups+at;
            quantized_up.scales[at]=prepared_up.scales[source];
            quantized_up.sums[at]=prepared_up.sums[source];
        }
        __syncthreads();
    }else if constexpr(IntegerUp) {
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(sample_begin+k<samples && warp<quantized_groups)
            gr_quantize_activation_group<FixedGroup,three_up,!three_up>(up,low,false,quantized_up,warp,k,lane);
        __syncthreads();
    }
    constexpr int outputs_per_warp=IntegerUp==11 || IntegerUp==12 || IntegerUp==13 || IntegerUp==14 || IntegerUp==15 || IntegerUp==17 || IntegerUp==18 || IntegerUp==19 || IntegerUp==20 || IntegerUp==21 || IntegerUp==22 || IntegerUp==23?2:IntegerUp==24 || IntegerUp==9 || IntegerUp==10?4:1;
    const int column=blockIdx.x*(blockDim.x/32)*outputs_per_warp+warp*outputs_per_warp+lane/(32/outputs_per_warp);
    if(column>=hidden)return;
    if constexpr(IntegerUp==6) {
        gr_up_integer_chunks_mix<Capacity>(up,quantized_up,column,hidden,samples,sample_begin,lane,normalized,mixed,half);
        return;
    }
    float sums[Capacity][4]{};
    if constexpr(IntegerUp==24) {
        gr_up_direct_s16_four_outputs<Capacity>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums);
    }else if constexpr(IntegerUp==23) {
        gr_up_direct_s16<Capacity,2,true,true>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums);
    }else if constexpr(IntegerUp==22) {
        gr_up_direct_s16<Capacity,4,true>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums);
    }else if constexpr(IntegerUp==21) {
        gr_up_direct_s16<Capacity,2,true>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums);
    }else if constexpr(IntegerUp==20) {
        gr_up_direct_s16<Capacity,4>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums);
    }else if constexpr(IntegerUp==19) {
        gr_up_direct_s16<Capacity>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums);
    }else if constexpr(IntegerUp==18) {
        gr_up_uniform16_pair_preload<Capacity,2>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums,nullptr,row_records,group_records);
    }else if constexpr(IntegerUp==17) {
        gr_up_uniform16_pair_preload<Capacity,true>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums,group_coefficients);
    }else if constexpr(IntegerUp==16) {
        gr_up_uniform16_full_warp<Capacity>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums);
    }else if constexpr(IntegerUp==15) {
        gr_up_uniform16_pair_preload<Capacity>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums);
    }else if constexpr(IntegerUp==14) {
        gr_up_uniform16_preload<Capacity>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums);
    }else if constexpr(IntegerUp==13) {
        gr_up_uniform16_half_groups<Capacity>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums);
    }else if constexpr(IntegerUp==11 || IntegerUp==12) {
        gr_up_integer_half_groups<Capacity>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums);
    }else if constexpr(IntegerUp==9 || IntegerUp==10) {
        gr_up_integer_groups<Capacity>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums);
    }else if constexpr(IntegerUp==5 || IntegerUp==7 || IntegerUp==8) {
        gr_up_integer_chunks<Capacity>(up,quantized_up,column,hidden,samples,sample_begin,lane,sums);
    }else if constexpr(IntegerUp) {
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
            auto q=quantized_up;
            q.high+=k*quantized_elements;q.low+=k*quantized_elements;
            if constexpr(IntegerUp==3)q.middle+=k*quantized_elements;
            q.scales+=k*quantized_groups;q.sums+=k*quantized_groups;
            #pragma unroll
            for(int s=0;s<4;++s)
                sums[k][s]=gr_dp4a2_affine<FixedGroup,IntegerUp==3,IntegerUp!=3,IntegerUp==3>(up,q,s*hidden+column,0,up.groups,lane);
        }
    }else if(up.aligned && !up.zero) {
        constexpr bool SharedMetadata=MetadataMode==1;
        const int gs=FixedGroup?FixedGroup:up.gs;
        constexpr int fixed_groups=FixedWidth?(FixedWidth+FixedGroup-1)/FixedGroup:0;
        __shared__ float metadata[SharedMetadata && fixed_groups>0 && fixed_groups<=32?16*8*fixed_groups:1];
        const int groups=fixed_groups?fixed_groups:up.groups;
        const int width=FixedWidth?FixedWidth:up.width;
        const uint8_t* packed[4];
        float outer_scale[4],outer_minimum[4];
        float cached_scale[4]{},cached_minimum[4]{};
        int rows[4];
        #pragma unroll
        for(int s=0;s<4;++s)if(s<streams) {
            rows[s]=s*hidden+column;
            packed[s]=up.q+(up.offsets[rows[s]]>>3);
            outer_scale[s]=up.row_scale[rows[s]];
            outer_minimum[s]=up.row_minimum[rows[s]];
            if constexpr(fixed_groups>0 && fixed_groups<=32) {
                if(lane<fixed_groups) {
                    const size_t meta=size_t(rows[s])*fixed_groups+lane;
                    const float scale=__fmul_rn(outer_scale[s],float(up.scales[meta]));
                    const float minimum=__fmul_rn(outer_minimum[s],float(up.minima[meta]));
                    if constexpr(SharedMetadata) {
                        metadata[(warp*8+s*2)*fixed_groups+lane]=scale;
                        metadata[(warp*8+s*2+1)*fixed_groups+lane]=minimum;
                    }else {cached_scale[s]=scale;cached_minimum[s]=minimum;}
                }
            }
        }
        if constexpr(SharedMetadata && fixed_groups>0 && fixed_groups<=32)__syncwarp();
        // The fixed 320-wide projection has only seven GS48 groups. Preserve
        // the FMA order while making group offsets and the final tail constant.
        constexpr int unroll_groups=MetadataMode==2 && fixed_groups>0?fixed_groups:1;
        #pragma unroll unroll_groups
        for(int group=0;group<groups;++group) {
            float scale[4],minimum[4];
            #pragma unroll
            for(int s=0;s<4;++s)if(s<streams) {
                if constexpr(fixed_groups>0 && fixed_groups<=32) {
                    if constexpr(SharedMetadata) {
                        scale[s]=metadata[(warp*8+s*2)*fixed_groups+group];
                        minimum[s]=metadata[(warp*8+s*2+1)*fixed_groups+group];
                    }else {
                        scale[s]=__shfl_sync(0xffffffff,cached_scale[s],group);
                        minimum[s]=__shfl_sync(0xffffffff,cached_minimum[s],group);
                    }
                } else {
                    const size_t meta=size_t(rows[s])*up.groups+group;
                    scale[s]=__fmul_rn(outer_scale[s],float(up.scales[meta]));
                    minimum[s]=__fmul_rn(outer_minimum[s],float(up.minima[meta]));
                }
            }
            #pragma unroll
            for(int chunk=0;chunk<2;++chunk) {
                const int j=lane+chunk*32,at=group*gs+j;
                if(j<gs && at<width) {
                    #pragma unroll
                    for(int s=0;s<4;++s)if(s<streams) {
                        const float weight=__fsub_rn(__fmul_rn(scale[s],float(packed[s][at])),minimum[s]);
                        #pragma unroll
                        for(int k=0;k<Capacity;++k)if(sample_begin+k<samples)
                            sums[k][s]=fmaf(weight,low[k*down.outputs+at],sums[k][s]);
                    }
                }
            }
        }
        for(int stride=16;stride;stride>>=1) {
            #pragma unroll
            for(int s=0;s<4;++s)if(s<streams) {
                #pragma unroll
                for(int k=0;k<Capacity;++k)sums[k][s]+=__shfl_down_sync(0xffffffff,sums[k][s],stride);
            }
        }
    } else {
        #pragma unroll
        for(int s=0;s<4;++s)if(s<streams) {
            #pragma unroll
            for(int k=0;k<Capacity;++k)if(sample_begin+k<samples)
                sums[k][s]=mhc_affine(up,low+k*down.outputs,s*hidden+column,0,up.groups,lane);
        }
    }
    if(lane%(32/outputs_per_warp)==0) {
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
        const int sample=sample_begin+k;
        float value=0.f;
        #pragma unroll
        for(int s=0;s<4;++s)if(s<streams) {
            const float projected=mhc_round(sums[k][s],half);
            const float gate=mhc_round(mhc_sigmoid(projected,up.fast_sigmoid),half);
            const float product=mhc_round(__fmul_rn(gate,mhc_read(normalized,(size_t(sample)*streams+s)*hidden+column,half)),half);
            value=__fadd_rn(value,product);
        }
        mhc_write(mixed,size_t(sample)*hidden+column,__fdiv_rn(value,float(streams)),half);
        }
    }
}

MhcMatrix mhc_matrix(const NintWeight& w) {
    MhcMatrix matrix{w.q_packed.data_ptr<uint8_t>(),w.q8_zero?nullptr:w.row_q_bits.data_ptr<uint8_t>(),
        w.q8_zero?nullptr:w.sub_scale.data_ptr<uint8_t>(),w.q8_zero?nullptr:w.sub_min.data_ptr<uint8_t>(),
        w.q8_zero?nullptr:w.row_q_bit_offsets.data_ptr<int64_t>(),w.q8_zero?nullptr:w.neuron_scale.data_ptr<float>(),
        w.q8_zero?nullptr:w.neuron_min.data_ptr<float>(),w.q8_zero?w.q8_zero_scale.data_ptr<__half>():nullptr,
        int(w.neuron_len),int(w.out),int(w.ng),int(w.gs),w.q8_zero,w.aligned_q8};
    matrix.bytes=uint64_t(w.q_packed.numel());
    return matrix;
}

__global__ void gr_prepare_up_coefficients_kernel(MhcMatrix weight,float2* output) {
    const int at=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at>=weight.outputs*weight.groups)return;
    const int row=at/weight.groups;
    output[at]=make_float2(__fmul_rn(weight.row_scale[row],float(weight.scales[at])),
        __fmul_rn(weight.row_minimum[row],float(weight.minima[at])));
}

__global__ void gr_prepare_up_records_kernel(MhcMatrix weight,int4* rows,uint16_t* groups) {
    const int at=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at<weight.outputs) {
        const uint64_t offset=uint64_t(weight.offsets[at])>>3;
        rows[at]=make_int4(int(uint32_t(offset)),int(uint32_t(offset>>32)),
            __float_as_int(weight.row_scale[at]),__float_as_int(weight.row_minimum[at]));
    }
    if(at<weight.outputs*weight.groups)
        groups[at]=uint16_t(unsigned(weight.scales[at])|(unsigned(weight.minima[at])<<8));
}
}

mfq_tensor_backend::Tensor prepare_gr_dense_vector_right(const mfq_tensor_backend::Tensor& right) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(right.is_cuda() && right.dim()==2 && right.scalar_type()==tb::kFloat32 &&
        right.size(0)>0 && right.size(0)%4==0 && right.size(1)>0 && right.size(1)<=4,
        "vector residual injection requires F32 [4*k, 1..4] weights");
    return right.reshape({right.size(0)/4,4,right.size(1)}).transpose(1,2).contiguous();
}

mfq_tensor_backend::Tensor prepare_gr_up_packed_metadata(const NintWeight& weight) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(weight.aligned_q8 && !weight.q8_zero && weight.gs==48 &&
        weight.neuron_len==320 && weight.out==10240 && weight.ng==7,"GR Up metadata geometry invalid");
    MfqCudaGuard guard(weight.q_packed.device());
    auto result=tb::empty({weight.out*4+(weight.out*weight.ng+1)/2},weight.q_packed.options().dtype(tb::kInt32));
    auto* records=result.data_ptr<int32_t>();
    gr_prepare_up_records_kernel<<<unsigned((weight.out*weight.ng+255)/256),256,0,mfq_current_cuda_stream()>>>(
        mhc_matrix(weight),reinterpret_cast<int4*>(records),reinterpret_cast<uint16_t*>(records+weight.out*4));
    MFQ_CUDA_CHECK(cudaGetLastError());return result;
}

mfq_tensor_backend::Tensor prepare_gr_up_group_coefficients(const NintWeight& weight) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(weight.aligned_q8 && !weight.q8_zero && weight.gs==48 &&
        weight.neuron_len==320 && weight.out==10240 && weight.ng==7,"GR Up coefficient cache geometry invalid");
    MfqCudaGuard guard(weight.q_packed.device());
    auto result=tb::empty({weight.out,weight.ng,2},weight.q_packed.options().dtype(tb::kFloat32));
    gr_prepare_up_coefficients_kernel<<<unsigned((weight.out*weight.ng+255)/256),256,0,mfq_current_cuda_stream()>>>(
        mhc_matrix(weight),reinterpret_cast<float2*>(result.data_ptr<float>()));
    MFQ_CUDA_CHECK(cudaGetLastError());return result;
}

std::vector<mfq_tensor_backend::Tensor> gated_residual_two_stage_cuda(
    const mfq_tensor_backend::Tensor& branch,const mfq_tensor_backend::Tensor& residual,
    const mfq_tensor_backend::Tensor& injection,const mfq_tensor_backend::Tensor& norm,
    const NintWeight& down,const NintWeight& up,const NintWeight* inject,int64_t streams,double eps,bool after,
    const mfq_tensor_backend::Tensor& dense_injection,const mfq_tensor_backend::Tensor& prepared_injection_right,
    const std::function<void(const char*)>& marker,const mfq_tensor_backend::Tensor& up_group_coefficients) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK((inject!=nullptr)!=dense_injection.defined(),"exactly one residual injection weight is required");
    const auto matrix_valid=[](const NintWeight& w) {return w.out>0 && w.neuron_len>0 && w.gs>0 && w.gs<=64 &&
        w.ng==(w.neuron_len+w.gs-1)/w.gs;};
    MFQ_RUNTIME_CHECK(matrix_valid(down) && matrix_valid(up) && (!inject || matrix_valid(*inject)),"fused residual quantization geometry invalid");
    const auto valid=[](const tb::Tensor& x) {return x.is_cuda() && x.is_contiguous() &&
        (x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32);};
    MFQ_RUNTIME_CHECK(valid(residual) && streams>0 && streams<=4 && residual.dim()>0 && residual.size(-1)%streams==0,
        "fused residual residual geometry/dtype unsupported");
    const int hidden=residual.size(-1)/streams,rows=residual.numel()/residual.size(-1);
    MFQ_RUNTIME_CHECK(rows>0 && rows<=65535 && norm.is_cuda() && norm.is_contiguous() && norm.scalar_type()==tb::kFloat32 &&
        norm.numel()==residual.size(-1) && norm.device()==residual.device(),"fused residual norm unsupported");
    MFQ_RUNTIME_CHECK(!after || (valid(branch) && valid(injection) && branch.numel()==rows*hidden &&
        injection.numel()==rows*streams && branch.device()==residual.device() && injection.device()==residual.device()),
        "fused residual prior branch/injection disagree");
    MFQ_RUNTIME_CHECK(down.neuron_len==residual.size(-1) && up.neuron_len==down.out && up.out==residual.size(-1) &&
        down.q_packed.device()==residual.device() && up.q_packed.device()==residual.device() &&
        (dense_injection.defined()?(dense_injection.is_cuda() && dense_injection.is_contiguous() && dense_injection.scalar_type()==tb::kBFloat16 &&
        dense_injection.device()==residual.device() && dense_injection.dim()==2 && dense_injection.size(0)==streams && dense_injection.size(1)==down.neuron_len)
        :(inject->neuron_len==down.neuron_len && inject->out==streams && inject->q_packed.device()==residual.device())),"fused residual matrices disagree");
    MfqCudaGuard guard(residual.device());
    const int resident=nint_float_projection_resident_blocks(residual.get_device());
    const auto splits=[&](const NintWeight& w) {const int blocks=((w.out+3)/4)*rows;
        return int(std::min<int64_t>(w.ng,std::max(1,(resident+blocks-1)/blocks)));};
    MFQ_RUNTIME_CHECK(splits(up)==1,"fused residual requires original up projection unsplit");
    int ds=splits(down);
    const char* norm_setting=std::getenv("MFQ_GR_SHARE_NORM");
    const bool share_norm=!norm_setting || norm_setting[0]!='0';
    const char* group_setting=std::getenv("MFQ_GR_FIXED_GROUPS");
    const bool fixed_groups=!group_setting || group_setting[0]=='1';
    const char* format_setting=std::getenv("MFQ_GR_FIXED_FORMATS");
    const bool fixed_formats=!format_setting || format_setting[0]=='1';
    const auto* down_fused_option=std::getenv("MFQ_GR_DOWN_MMA_FUSED_UP");
    const bool down_fused_requested=rows==1 && down_fused_option && down_fused_option[0]!='0';
    const auto* down_mma_option=std::getenv("MFQ_GR_DOWN_MMA");
    const auto* combined_option=std::getenv("MFQ_GR_COMBINED_AFFINE");
    const bool combined_affine=combined_option?combined_option[0]!='0':rows==5;
    const auto* unsigned_dot_option=std::getenv("MFQ_GR_DOWN_UNSIGNED_DOT");
    const bool unsigned_dot=unsigned_dot_option?unsigned_dot_option[0]!='0':(rows==1 || rows==2 || rows==5);
    const auto* down_plane_option=std::getenv("MFQ_GR_INT_DOT");
    const auto* down_batch_option=std::getenv("MFQ_GR_BATCH_SHARED_DOWN");
    const bool down_mma_candidate=(down_mma_option?down_mma_option[0]!='0':rows==6 || down_fused_requested) &&
        share_norm && fixed_groups && down.aligned_q8 && !down.q8_zero && down.gs==48 &&
        down.neuron_len==10240 && down.out==320 && hidden==2560 && streams==4 && rows<=6 &&
        (!down_plane_option || std::atoi(down_plane_option)==3 || std::atoi(down_plane_option)==4) &&
        (!down_batch_option || down_batch_option[0]!='1') &&
        (dense_injection.defined() || (inject && inject->aligned_q8 && !inject->q8_zero));
    if(down_mma_candidate) {
        const auto* split_option=std::getenv("MFQ_GR_DOWN_MMA_SPLITS");
        ds=split_option?std::atoi(split_option):(rows==1 && !down_fused_requested?32:16);
        MFQ_RUNTIME_CHECK(ds>0 && ds<=down.ng,"GR Down MMA split count is invalid");
    }
    const auto* down_split_option=std::getenv("MFQ_GR_DOWN_SPLITS");
    if(rows==1 && !down_mma_candidate && share_norm &&
        down.aligned_q8 && !down.q8_zero && down.gs==48 &&
        down.neuron_len==10240 && down.out==320 && hidden==2560 && streams==4) {
        if(down_split_option)ds=std::atoi(down_split_option);
        else if(ds==13 && fixed_groups && fixed_formats && !dense_injection.defined() &&
            inject && inject->aligned_q8 && !inject->q8_zero && up.aligned_q8 && !up.q8_zero &&
            up.gs==48 && up.neuron_len==320)ds=8;
        MFQ_RUNTIME_CHECK(ds>0 && ds<=down.ng,"GR Down split count is invalid");
    }
    const auto* s16_down_option=std::getenv("MFQ_GR_DOWN_DIRECT_S16");
    const int s16_down_mode=s16_down_option?std::atoi(s16_down_option):0;
    const bool s16_down_requested=s16_down_mode!=0 && rows==1 &&
        share_norm && fixed_groups && fixed_formats && !down_mma_candidate &&
        hidden==2560 && streams==4 && down.aligned_q8 && !down.q8_zero && down.gs==48 &&
        down.neuron_len==10240 && down.out==320 && up.aligned_q8 && !up.q8_zero &&
        up.gs==48 && up.neuron_len==320 && !dense_injection.defined() &&
        inject && inject->aligned_q8 && !inject->q8_zero;
    if(s16_down_requested && !down_split_option)ds=s16_down_mode==2?16:8;
    const auto* group_read_option=std::getenv("MFQ_GR_GROUP_READ");
    const int group_read=group_read_option && !down.q8_zero && !down.aligned_q8 &&
        (reinterpret_cast<uintptr_t>(down.q_packed.data_ptr())&3u)==0?std::atoi(group_read_option):0;
    const char* thread_setting=std::getenv("MFQ_GR_UP_THREADS");
    const int up_threads=thread_setting?std::atoi(thread_setting):512;
    const auto* up_metadata_option=std::getenv("MFQ_GR_UP_METADATA");
    const int up_metadata=up_metadata_option?std::atoi(up_metadata_option):0;
    MFQ_RUNTIME_CHECK(up_threads==128 || up_threads==256 || up_threads==512,
        "residual Up threads must be 128, 256 or 512");
    const char* projection_setting=std::getenv("MFQ_GR_PROJECTION_THREADS");
    int projection_threads=projection_setting?std::atoi(projection_setting):256;
    MFQ_RUNTIME_CHECK(projection_threads==128 || projection_threads==256 || projection_threads==512,
        "residual projection threads must be 128, 256 or 512");
    if(!share_norm)projection_threads=256;
    const int tile=projection_threads/32;
    const char* warp_setting=std::getenv("MFQ_GR_DENSE_FUSION");
    const bool dense_warp=(!warp_setting || warp_setting[0]!='0') && share_norm &&
        rows<=8 && dense_injection.defined() && prepared_injection_right.defined() &&
        ((down.out+tile-1)/tile)*ds >= (32+(32/streams)-1)/(32/streams);
    const int is=dense_warp?32:(dense_injection.defined()?std::min<int64_t>((down.neuron_len+31)/32,resident):splits(*inject));
    const bool bh=after && branch.scalar_type()==tb::kFloat16,rh=residual.scalar_type()==tb::kFloat16,
        ih=after && injection.scalar_type()==tb::kFloat16,half=rh && (!after || (bh && ih));
    const auto options=residual.options().dtype(half?tb::kFloat16:tb::kFloat32);
    const bool packed_up_metadata=up_group_coefficients.defined() && up_group_coefficients.scalar_type()==tb::kInt32;
    MFQ_RUNTIME_CHECK(!up_group_coefficients.defined() || (up_group_coefficients.is_cuda() &&
        up_group_coefficients.device()==residual.device() && up_group_coefficients.is_contiguous() &&
        (packed_up_metadata?(up_group_coefficients.dim()==1 && up_group_coefficients.numel()==up.out*4+(up.out*up.ng+1)/2):
        (up_group_coefficients.scalar_type()==tb::kFloat32 &&
        up_group_coefficients.sizes().vec()==std::vector<int64_t>({up.out,up.ng,2})))),"GR Up coefficient cache tensor invalid");
    auto normalized=tb::empty(residual.sizes().vec(),options),updated=tb::empty(residual.sizes().vec(),options);
    auto shape=residual.sizes().vec();shape.back()=hidden;auto mixed=tb::empty(shape,options);
    shape.back()=streams;auto next_injection=tb::empty(shape,options.dtype(dense_injection.defined() || !half?tb::kFloat32:tb::kFloat16));
    const char* async_setting=std::getenv("MFQ_GR_ASYNC_NORMALIZED");
    const bool asynchronous=share_norm && !half && (!async_setting || async_setting[0]=='1');
    auto promoted_normalized=(dense_injection.defined() || asynchronous) && half?
        tb::empty(residual.sizes().vec(),options.dtype(tb::kFloat32)):tb::Tensor{};
    const bool vector_injection=prepared_injection_right.defined() && prepared_injection_right.dim()==3;
    if(prepared_injection_right.defined()) {
        const bool shape_matches=vector_injection?
            (down.neuron_len%4==0 && prepared_injection_right.size(0)==down.neuron_len/4 &&
             prepared_injection_right.size(1)==streams && prepared_injection_right.size(2)==4 &&
             (reinterpret_cast<std::uintptr_t>(prepared_injection_right.data_ptr())&15u)==0):
            (prepared_injection_right.dim()==2 && prepared_injection_right.size(0)==down.neuron_len &&
             prepared_injection_right.size(1)==streams);
        MFQ_RUNTIME_CHECK(dense_injection.defined() && prepared_injection_right.is_cuda() &&
            prepared_injection_right.device()==residual.device() && prepared_injection_right.is_contiguous() &&
            prepared_injection_right.scalar_type()==tb::kFloat32 && shape_matches,
            "prepared residual injection matrix shape/dtype/device disagrees");
    }
    auto parts=tb::empty({rows,down.out*ds+streams*is},residual.options().dtype(tb::kFloat32));
    MhcInput input{after?branch.data_ptr():nullptr,residual.data_ptr(),after?injection.data_ptr():nullptr,
        norm.data_ptr<float>(),nullptr,hidden,int(streams),bh,rh,ih,half,after};
    const auto* fast_norm_option=std::getenv("MFQ_GR_NORM_FAST_RSQRT");
    const bool verified_fast_gr=hidden==2560 && streams==4 && down.neuron_len==10240 && down.out==320 &&
        up.neuron_len==320 && up.aligned_q8 && !up.q8_zero && up.gs==48;
    input.fast_norm=fast_norm_option?fast_norm_option[0]!='0':verified_fast_gr &&
        ((rows==1 && (!half || !after)) || (rows==2 && !half) || rows==3 || rows==5 ||
         (rows==4 && (!half || !after)) || (rows==6 && (after?!half:half)));
    const auto* fixed_norm_option=std::getenv("MFQ_GR_NORM_FIXED_TYPES");
    const bool fixed_norm_types=fixed_norm_option && fixed_norm_option[0]!='0';
    const auto* register_norm_option=std::getenv("MFQ_GR_NORM_REGISTER_QUANT");
    const int register_norm_mode=register_norm_option?std::atoi(register_norm_option):0;
    auto projection_input=input;
    if(asynchronous)projection_input.output_half=false;
    const void* projection_normalized=asynchronous && promoted_normalized.defined()?
        promoted_normalized.data_ptr():normalized.data_ptr();
    const auto d=mhc_matrix(down);
    auto u=mhc_matrix(up);
    const auto* fast_sigmoid_option=std::getenv("MFQ_GR_UP_FAST_SIGMOID");
    u.fast_sigmoid=fast_sigmoid_option?fast_sigmoid_option[0]!='0':verified_fast_gr &&
        ((rows==1 && (!half || after)) || (rows==2 && half) ||
         (rows==5 && (half?after:!after)) || (!half && after && (rows==3 || rows==4 || rows==6)));
    MhcMatrix i{};
    if(dense_injection.defined()) {
        i.width=down.neuron_len;i.outputs=streams;i.gs=32;i.groups=(i.width+31)/32;
        i.dense=static_cast<const __nv_bfloat16*>(dense_injection.data_ptr());
        if(prepared_injection_right.defined())i.row_scale=prepared_injection_right.data_ptr<float>();
    } else i=mhc_matrix(*inject);
    const int down_capacity=((down.ng+ds-1)/ds)*down.gs,inject_capacity=dense_warp?0:((i.groups+is-1)/is)*i.gs;
    const auto* inline_norm_option=std::getenv("MFQ_GR_INLINE_NORM_DOWN");
    const bool inline_norm_down=inline_norm_option && inline_norm_option[0]!='0' &&
        verified_fast_gr && rows==1 && ds==8 && down.aligned_q8 && !down.q8_zero && down.gs==48 &&
        down.ng==214 && i.aligned && !i.zero && !dense_injection.defined() &&
        share_norm && fixed_groups && fixed_formats;
    if(inline_norm_down) {
        const int counter_slot=gr_norm_counter_slot();
        if(counter_slot>=0) {
            auto up_values=tb::empty({2,336},options.dtype(tb::kInt8));
            auto up_scales=tb::empty({7,2},options.dtype(tb::kFloat32));
            auto up_sums=tb::empty({7,4},options.dtype(tb::kInt32));
            GrTwoPlane prepared{up_values.data_ptr<int8_t>(),nullptr,up_values.data_ptr<int8_t>()+336,
                reinterpret_cast<float2*>(up_scales.data_ptr<float>()),reinterpret_cast<int4*>(up_sums.data_ptr<int32_t>())};
            if(marker){marker("norm");marker("quantize_input");}
            const int row_tile=std::atoi(inline_norm_option)==2?16:24;
            const auto launch=[&](auto kind) {
            gated_residual_down_kernel<48,8,true,decltype(kind)::value><<<dim3(unsigned((320+row_tile-1)/row_tile*ds)),256,
                unsigned((down_capacity+inject_capacity)*sizeof(float)),mfq_current_cuda_stream()>>>(
                input,d,i,updated.data_ptr(),normalized.data_ptr(),nullptr,parts.data_ptr<float>(),
                ds,is,float(eps),row_tile,down_capacity,prepared,counter_slot);
            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                "gr_inline_norm_down",reinterpret_cast<const void*>(gated_residual_down_kernel<48,8,true,decltype(kind)::value>),
                "M=1 half="+std::to_string(half)+" splits="+std::to_string(ds)+" tile="+std::to_string(row_tile)+" kind="+std::to_string(decltype(kind)::value));
            };
            if(half) {
                if(after)launch(std::integral_constant<int,2>{});else launch(std::integral_constant<int,1>{});
            }else launch(std::integral_constant<int,0>{});
            if(marker)marker("down");
            gated_residual_up_kernel<48,4,320,1,0,21><<<unsigned(hidden/16),256,0,mfq_current_cuda_stream()>>>(
                d,u,i,normalized.data_ptr(),parts.data_ptr<float>(),mixed.data_ptr(),next_injection.data_ptr(),
                hidden,int(streams),half,ds,is,nullptr,1,prepared,nullptr,nullptr,nullptr);
            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                "gr_up_mix_direct_s16_stats",reinterpret_cast<const void*>(gated_residual_up_kernel<48,4,320,1,0,21>),
                "M=1 half="+std::to_string(half)+" capacity=1");
            if(marker){marker("up_mix");marker("end");}
            MFQ_CUDA_CHECK(cudaGetLastError());
            return {mixed,updated,next_injection};
        }
    }
    const size_t shared=(down_capacity+inject_capacity)*sizeof(float);
    MFQ_RUNTIME_CHECK(shared+1056<=48*1024,"fused residual projection exceeds shared memory capacity");
    if(marker)marker("begin");
    const auto launch_norm=[&](auto cache) {
        const auto* wide_option=std::getenv("MFQ_GR_NORM_WIDE");
        if constexpr(decltype(cache)::value)if((!wide_option || wide_option[0]!='0') && streams==4 && rows<=6) {
            const auto* threads_option=std::getenv("MFQ_GR_NORM_THREADS");
            const int norm_threads=threads_option?std::atoi(threads_option):1024;
            MFQ_RUNTIME_CHECK(norm_threads==512 || norm_threads==640 || norm_threads==1024,"GR norm threads must be 512, 640 or 1024");
            const auto launch_wide=[&](auto threads) {
            gated_residual_norm_wide_kernel<0,0,false,decltype(threads)::value><<<dim3(unsigned(streams),unsigned(rows)),decltype(threads)::value,0,mfq_current_cuda_stream()>>>(
                input,updated.data_ptr(),normalized.data_ptr(),promoted_normalized.defined()?promoted_normalized.data_ptr<float>():nullptr,
                float(eps),dense_injection.defined()&&prepared_injection_right.defined()?prepared_injection_right.data_ptr<float>():nullptr);
            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                "gr_norm_wide",reinterpret_cast<const void*>(gated_residual_norm_wide_kernel<0,0,false,decltype(threads)::value>),
                "M="+std::to_string(rows)+" half="+std::to_string(half)+" threads="+std::to_string(decltype(threads)::value));
            };
            if(norm_threads==512)launch_wide(std::integral_constant<int,512>{});
            else if(norm_threads==640)launch_wide(std::integral_constant<int,640>{});
            else launch_wide(std::integral_constant<int,1024>{});
            return;
        }
        gated_residual_norm_kernel<decltype(cache)::value><<<dim3(unsigned(streams),unsigned(rows)),256,0,mfq_current_cuda_stream()>>>(
            input,updated.data_ptr(),normalized.data_ptr(),promoted_normalized.defined()?promoted_normalized.data_ptr<float>():nullptr,float(eps),dense_injection.defined()&&prepared_injection_right.defined()?prepared_injection_right.data_ptr<float>():nullptr);
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),decltype(cache)::value?"gr_norm_cache":"gr_norm_original",
            reinterpret_cast<const void*>(gated_residual_norm_kernel<decltype(cache)::value>),
            "M="+std::to_string(rows)+" hidden="+std::to_string(hidden)+" half="+std::to_string(half));
    };
    // Verified GR geometry uses uniform16 activation planes for M1/M3-M6;
    // M2 and larger rows retain int24. Weight bytes stay unchanged.
    const auto* packed24_option=std::getenv("MFQ_GR_PACKED24_INT_DOT");
    const int packed24_mode=packed24_option?std::atoi(packed24_option):(rows==1?1:2);
    const bool packed_down=(packed24_mode==1 || packed24_mode==2) &&
        share_norm && fixed_groups && !down.aligned_q8 && !down.q8_zero &&
        down.gs==24 && down.neuron_len==10240 && down.out==320 && streams==4 && rows<=6 &&
        inject && inject->aligned_q8 && !inject->q8_zero &&
        (reinterpret_cast<uintptr_t>(down.q_packed.data_ptr())&3u)==0;
    const auto* two_plane_option=std::getenv("MFQ_GR_INT_DOT");
    const int plane_count=packed_down?(packed24_mode==2?4:3):two_plane_option?std::atoi(two_plane_option):((rows==1 || (rows>=3 && rows<=6))?4:3);
    const auto* batch_down_option=std::getenv("MFQ_GR_BATCH_SHARED_DOWN");
    const bool two_plane=(plane_count==2 || plane_count==3 || plane_count==4) &&
        share_norm && fixed_groups && ((down.aligned_q8 && !down.q8_zero && down.gs==48) || packed_down) &&
        down.neuron_len==10240 && down.out==320 && streams==4 &&
        rows<=8 && (reinterpret_cast<uintptr_t>(down.q_packed.data_ptr())&15u)==0 &&
        (!batch_down_option || batch_down_option[0]!='1');
    tb::Tensor plane_values,plane_scales,plane_sums;
    GrTwoPlane global_quantized{};
    const auto* norm_quant_option=std::getenv("MFQ_GR_NORM_QUANT_FUSED");
    const auto* norm_group_option=std::getenv("MFQ_GR_NORM_GROUP_QUANT");
    const int norm_group_mode=norm_group_option?std::atoi(norm_group_option):0;
    const auto* wide_norm_option=std::getenv("MFQ_GR_NORM_WIDE");
    const auto* cached_norm_option=std::getenv("MFQ_GR_NORM_CACHE");
    const int norm_quant_mode=norm_quant_option?std::atoi(norm_quant_option):
        (rows>6 || norm_group_mode!=0 || (wide_norm_option && wide_norm_option[0]=='0') ||
        (cached_norm_option && cached_norm_option[0]=='0')?0:(rows==1 && half && after?3:2));
    const bool norm_quant_fused=two_plane && norm_quant_mode!=0 &&
        hidden==2560 && streams==4 && rows<=6;
    const bool packed_down_s16=s16_down_requested && norm_quant_fused && plane_count==4;
    const auto* replicated_norm_option=std::getenv("MFQ_GR_NORM_REPLICATED");
    const int replicated_norm_groups=replicated_norm_option?std::atoi(replicated_norm_option):8;
    const bool replicated_norm=(replicated_norm_groups==4 || replicated_norm_groups==8) &&
        norm_quant_fused && !packed_down_s16 && rows<=6 && half && input.residual_half &&
        (plane_count==3 || plane_count==4) && (down.gs==24 || down.gs==48) && !promoted_normalized.defined();
    bool norm_group_quant=two_plane && !norm_quant_fused &&
        norm_group_mode!=0 &&
        (!wide_norm_option || wide_norm_option[0]!='0') && (!cached_norm_option || cached_norm_option[0]!='0') &&
        hidden==2560 && streams==4 && rows<=6;
    tb::Tensor norm_inverses;
    const int norm_counter_slot=norm_group_quant && norm_group_mode==2?gr_norm_counter_slot():-1;
    if(norm_group_mode==2 && norm_counter_slot<0)norm_group_quant=false;
    if(share_norm && !norm_quant_fused && !norm_group_quant) {
        const auto* cache_option=std::getenv("MFQ_GR_NORM_CACHE");
        if(hidden==2560 && (!cache_option || cache_option[0]!='0'))launch_norm(std::true_type{});
        else launch_norm(std::false_type{});
        if(marker)marker("norm");
    }
    if(two_plane) {
        const int64_t elements=int64_t(rows)*down.ng*down.gs;
        const int storage_planes=plane_count==3?3:2;
        plane_values=tb::empty({storage_planes,elements},options.dtype(tb::kInt8));
        plane_scales=tb::empty({rows,down.ng,2},options.dtype(tb::kFloat32));
        plane_sums=tb::empty({rows,down.ng,4},options.dtype(tb::kInt32));
        global_quantized={plane_values.data_ptr<int8_t>(),plane_count==3?plane_values.data_ptr<int8_t>()+elements:nullptr,
            plane_values.data_ptr<int8_t>()+(storage_planes-1)*elements,
            reinterpret_cast<float2*>(plane_scales.data_ptr<float>()),reinterpret_cast<int4*>(plane_sums.data_ptr<int32_t>())};
        if(norm_quant_fused && !replicated_norm) {
            norm_inverses=tb::empty({rows,streams},options.dtype(tb::kFloat32));
            const auto* vector_stats_option=std::getenv("MFQ_GR_NORM_VECTOR_STATS");
            // M5 Float-before did not improve in the interleaved real-weight
            // probe. Keep its existing statistics kernel unless requested.
            const bool vector_stats=(vector_stats_option?vector_stats_option[0]!='0':
                !(rows==5 && !half && !after)) &&
                input.residual_half==input.output_half &&
                (reinterpret_cast<uintptr_t>(input.residual)%(half?8:16))==0 &&
                (!after || (reinterpret_cast<uintptr_t>(input.branch)%(bh?8:16))==0);
            if(vector_stats) {
                const auto launch=[&](auto half_type,auto after_type,auto branch_type) {
                    gated_residual_norm_vector_stats_kernel<decltype(half_type)::value,decltype(after_type)::value,decltype(branch_type)::value><<<
                        dim3(unsigned(streams),unsigned(rows)),256,0,mfq_current_cuda_stream()>>>(
                        input,updated.data_ptr(),norm_inverses.data_ptr<float>(),float(eps));
                    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                        "gr_norm_stats_vector",reinterpret_cast<const void*>(gated_residual_norm_vector_stats_kernel<
                            decltype(half_type)::value,decltype(after_type)::value,decltype(branch_type)::value>),
                        "M="+std::to_string(rows)+" half="+std::to_string(half)+" after="+std::to_string(after));
                };
                const auto select=[&](auto half_type) {
                    if(!after)launch(half_type,std::false_type{},std::false_type{});
                    else if(bh)launch(half_type,std::true_type{},std::true_type{});
                    else launch(half_type,std::true_type{},std::false_type{});
                };
                if(half)select(std::true_type{});else select(std::false_type{});
            }else if(norm_quant_mode==2 || norm_quant_mode==3) {
                const auto launch_stats=[&](auto threads) {
                const auto run=[&](auto kind) {
                    gated_residual_norm_wide_kernel<0,0,false,decltype(threads)::value,true,1,decltype(kind)::value><<<dim3(unsigned(streams),unsigned(rows)),decltype(threads)::value,0,mfq_current_cuda_stream()>>>(
                        input,updated.data_ptr(),nullptr,nullptr,float(eps),nullptr,{}, {},-1,norm_inverses.data_ptr<float>());
                    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                        decltype(kind)::value?"gr_norm_stats_fixed_types":"gr_norm_stats_wide",reinterpret_cast<const void*>(gated_residual_norm_wide_kernel<0,0,false,decltype(threads)::value,true,1,decltype(kind)::value>),
                        "M="+std::to_string(rows)+" half="+std::to_string(half)+" threads="+std::to_string(decltype(threads)::value)+" kind="+std::to_string(decltype(kind)::value));
                };
                if(fixed_norm_types && input.residual_half==input.output_half) {
                    if(half) {
                        if(!after)run(std::integral_constant<int,1>{});
                        else if(input.branch_half)run(std::integral_constant<int,2>{});
                        else run(std::integral_constant<int,3>{});
                    }else {
                        if(!after)run(std::integral_constant<int,4>{});
                        else if(input.branch_half)run(std::integral_constant<int,5>{});
                        else run(std::integral_constant<int,6>{});
                    }
                }else run(std::integral_constant<int,0>{});
                };
                if(norm_quant_mode==3)launch_stats(std::integral_constant<int,512>{});
                else launch_stats(std::integral_constant<int,1024>{});
            }else {
            gated_residual_norm_kernel<false,true><<<dim3(unsigned(streams),unsigned(rows)),256,0,mfq_current_cuda_stream()>>>(
                input,updated.data_ptr(),nullptr,nullptr,float(eps),nullptr,norm_inverses.data_ptr<float>());
            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                "gr_norm_stats",reinterpret_cast<const void*>(gated_residual_norm_kernel<false,true>),
                "M="+std::to_string(rows)+" half="+std::to_string(half));
            }
            if(marker)marker("norm");
        }
        const auto launch=[&](auto triple,auto unsigned_low,auto group) {
        if(norm_group_quant) {
            static constexpr int planes=decltype(triple)::value?3:decltype(unsigned_low)::value?4:2;
            const auto launch_group_norm=[&](auto inline_boundary) {
            const auto run_slices=[&](auto slices,auto threads) {
            gated_residual_norm_wide_kernel<decltype(group)::value,planes,decltype(inline_boundary)::value,decltype(threads)::value,false,decltype(slices)::value><<<
                dim3(unsigned(streams),unsigned(rows),unsigned(decltype(slices)::value)),decltype(threads)::value,0,mfq_current_cuda_stream()>>>(
                input,updated.data_ptr(),normalized.data_ptr(),promoted_normalized.defined()?promoted_normalized.data_ptr<float>():nullptr,
                float(eps),dense_injection.defined()&&prepared_injection_right.defined()?prepared_injection_right.data_ptr<float>():nullptr,
                d,global_quantized,norm_counter_slot);
            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                decltype(inline_boundary)::value?"gr_norm_group_quant_inline":"gr_norm_group_quant",
                reinterpret_cast<const void*>(gated_residual_norm_wide_kernel<decltype(group)::value,planes,decltype(inline_boundary)::value,decltype(threads)::value,false,decltype(slices)::value>),
                "M="+std::to_string(rows)+" half="+std::to_string(half)+" gs="+std::to_string(down.gs)+" slices="+std::to_string(decltype(slices)::value)+" threads="+std::to_string(decltype(threads)::value));
            };
            if constexpr(decltype(inline_boundary)::value) {
                const auto* slices=std::getenv("MFQ_GR_NORM_GROUP_SLICES");
                const int count=slices?std::atoi(slices):1;
                const auto* threads_option=std::getenv("MFQ_GR_NORM_GROUP_THREADS");
                const int threads=threads_option?std::atoi(threads_option):(count==8?256:1024);
                if(count==8 && threads==256){run_slices(std::integral_constant<int,8>{},std::integral_constant<int,256>{});return;}
                if(count==4 && threads==512){run_slices(std::integral_constant<int,4>{},std::integral_constant<int,512>{});return;}
                if(count==2 && threads==1024){run_slices(std::integral_constant<int,2>{},std::integral_constant<int,1024>{});return;}
                if(count==4 && threads==1024){run_slices(std::integral_constant<int,4>{},std::integral_constant<int,1024>{});return;}
                if(count==16 && threads==1024){run_slices(std::integral_constant<int,16>{},std::integral_constant<int,1024>{});return;}
                if(count==32 && threads==1024){run_slices(std::integral_constant<int,32>{},std::integral_constant<int,1024>{});return;}
            }
            run_slices(std::integral_constant<int,1>{},std::integral_constant<int,1024>{});
            };
            if(norm_counter_slot>=0)launch_group_norm(std::true_type{});
            else launch_group_norm(std::false_type{});
            if(marker)marker("norm");
            if(norm_counter_slot>=0)return;
            gated_residual_quantize_cross_stream_kernel<decltype(group)::value,decltype(triple)::value,decltype(unsigned_low)::value><<<
                dim3(1,unsigned(rows)),64,0,mfq_current_cuda_stream()>>>(d,projection_normalized,projection_input.output_half,global_quantized);
            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                "gr_quantize_cross_stream",reinterpret_cast<const void*>(gated_residual_quantize_cross_stream_kernel<
                    decltype(group)::value,decltype(triple)::value,decltype(unsigned_low)::value>),
                "M="+std::to_string(rows)+" gs="+std::to_string(down.gs));
            return;
        }
        if(norm_quant_fused) {
            if(replicated_norm) {
                const auto launch_replicated=[&](auto groups,auto after_type,auto branch_type) {
                    constexpr int gs=decltype(group)::value,groups_per_cta=decltype(groups)::value*48/gs;
                    constexpr bool triple_plane=decltype(triple)::value;
                    gated_residual_norm_replicated_kernel<gs,decltype(groups)::value,triple_plane,decltype(after_type)::value,decltype(branch_type)::value><<<
                        dim3(unsigned((down.ng+groups_per_cta-1)/groups_per_cta),unsigned(rows)),256,0,mfq_current_cuda_stream()>>>(
                        input,updated.data_ptr(),normalized.data_ptr(),global_quantized,float(eps));
                    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                        "gr_norm_quant_replicated",reinterpret_cast<const void*>(gated_residual_norm_replicated_kernel<
                            gs,decltype(groups)::value,triple_plane,decltype(after_type)::value,decltype(branch_type)::value>),
                        "M="+std::to_string(rows)+" half=1 gs="+std::to_string(gs)+" groups="+std::to_string(groups_per_cta));
                };
                const auto select=[&](auto groups) {
                    if(!after)launch_replicated(groups,std::false_type{},std::false_type{});
                    else if(bh)launch_replicated(groups,std::true_type{},std::true_type{});
                    else launch_replicated(groups,std::true_type{},std::false_type{});
                };
                if(replicated_norm_groups==4)select(std::integral_constant<int,4>{});
                else select(std::integral_constant<int,8>{});
                return;
            }
            if(packed_down_s16) {
                const auto launch_s16=[&](auto half_input) {
                    gated_residual_finish_norm_quant_s16_kernel<decltype(half_input)::value><<<
                        dim3(unsigned((down.ng+3)/4),unsigned(rows)),128,0,mfq_current_cuda_stream()>>>(
                        input,norm_inverses.data_ptr<float>(),updated.data_ptr(),normalized.data_ptr(),global_quantized);
                    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                        "gr_finish_norm_quant_s16",reinterpret_cast<const void*>(gated_residual_finish_norm_quant_s16_kernel<decltype(half_input)::value>),
                        "M=1 half="+std::to_string(half));
                };
                if(half)launch_s16(std::true_type{});else launch_s16(std::false_type{});
                return;
            }
            const auto run=[&](auto kind) {
                const auto launch=[&](auto reg,auto threads) {
                constexpr int warps=decltype(threads)::value/32;
                gated_residual_finish_norm_quant_kernel<decltype(group)::value,decltype(triple)::value,decltype(unsigned_low)::value,decltype(kind)::value,decltype(reg)::value,decltype(threads)::value><<<
                    dim3(unsigned((down.ng+warps-1)/warps),unsigned(rows)),decltype(threads)::value,0,mfq_current_cuda_stream()>>>(
                    input,d,norm_inverses.data_ptr<float>(),updated.data_ptr(),normalized.data_ptr(),
                    promoted_normalized.defined()?promoted_normalized.data_ptr<float>():nullptr,global_quantized);
                if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                    decltype(reg)::value?"gr_finish_norm_quant_register":decltype(kind)::value>=0?"gr_finish_norm_quant_fixed_types":"gr_finish_norm_quant",reinterpret_cast<const void*>(gated_residual_finish_norm_quant_kernel<
                        decltype(group)::value,decltype(triple)::value,decltype(unsigned_low)::value,decltype(kind)::value,decltype(reg)::value,decltype(threads)::value>),
                    "M="+std::to_string(rows)+" half="+std::to_string(half)+" gs="+std::to_string(down.gs)+" kind="+std::to_string(decltype(kind)::value)+" threads="+std::to_string(decltype(threads)::value));
                };
                if(register_norm_mode==2)launch(std::true_type{},std::integral_constant<int,128>{});
                else if(register_norm_mode==1)launch(std::true_type{},std::integral_constant<int,256>{});
                else launch(std::false_type{},std::integral_constant<int,256>{});
            };
            if(fixed_norm_types) {
                if(half)run(std::integral_constant<int,1>{});else run(std::integral_constant<int,0>{});
            }else run(std::integral_constant<int,-1>{});
            return;
        }
        gr_global_two_plane_kernel<decltype(group)::value,decltype(triple)::value,decltype(unsigned_low)::value><<<dim3(unsigned((down.ng+7)/8),unsigned(rows)),256,0,mfq_current_cuda_stream()>>>(
            d,projection_normalized,projection_input.output_half,global_quantized);
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),decltype(triple)::value?"gr_activation24":decltype(unsigned_low)::value?"gr_activation_uniform16":"gr_activation16",
            reinterpret_cast<const void*>(gr_global_two_plane_kernel<decltype(group)::value,decltype(triple)::value,decltype(unsigned_low)::value>),
            "M="+std::to_string(rows)+" half="+std::to_string(half));
        };
        if(packed_down && plane_count==4)launch(std::false_type{},std::true_type{},std::integral_constant<int,24>{});
        else if(packed_down)launch(std::true_type{},std::false_type{},std::integral_constant<int,24>{});
        else if(plane_count==3)launch(std::true_type{},std::false_type{},std::integral_constant<int,48>{});
        else if(plane_count==4)launch(std::false_type{},std::true_type{},std::integral_constant<int,48>{});
        else launch(std::false_type{},std::false_type{},std::integral_constant<int,48>{});
        if(marker)marker("quantize_input");
    }
    // M1 uses the same Uniform16 activation representation as the default
    // Up. Compare a single preparation launch with preparation by the last
    // Down CTA; original packed weights and all dot orders stay unchanged.
    const auto* uniform_prepare_option=std::getenv("MFQ_GR_UP_UNIFORM_PREPARE");
    int uniform_prepare_mode=uniform_prepare_option?std::atoi(uniform_prepare_option):3;
    const auto* direct_prepare_option=std::getenv("MFQ_GR_UP_DIRECT_S16");
    if(!uniform_prepare_option && direct_prepare_option && direct_prepare_option[0]=='0')uniform_prepare_mode=1;
    const auto* prepared_warp_option=std::getenv("MFQ_GR_UP_PREPARED_WARP");
    const bool prepared_full_warp=prepared_warp_option && prepared_warp_option[0]!='0';
    const auto* uniform_prepare_mma=std::getenv("MFQ_GR_UP_MMA");
    const auto* uniform_prepare_up=std::getenv("MFQ_GR_UP_UNIFORM16");
    const auto* m1_half_option=std::getenv("MFQ_GR_UP_MMA_F16_M1");
    const int m1_half_mode=m1_half_option?std::atoi(m1_half_option):0;
    bool m1_half_candidate=rows==1 && (m1_half_mode==1 || m1_half_mode==2 || (m1_half_mode==3 && half)) &&
        plane_count!=0 && fixed_groups && streams==4 && up_metadata==0 &&
        up.aligned_q8 && !up.q8_zero && up.gs==48 && up.neuron_len==320 && hidden==2560 &&
        down.neuron_len==10240 && down.out==320;
    bool uniform_prepare=(uniform_prepare_mode==1 || uniform_prepare_mode==2 || uniform_prepare_mode==3) &&
        (!m1_half_candidate || m1_half_mode==3) && rows==1 && hidden==2560 && streams==4 && two_plane && plane_count==4 &&
        (!down_mma_candidate || down_fused_requested) && !combined_affine && (!unsigned_dot || uniform_prepare_mode!=2) && fixed_formats &&
        projection_threads==256 && up_metadata==0 && !std::getenv("MFQ_GR_UP_INT_DOT") &&
        (!uniform_prepare_mma || std::atoi(uniform_prepare_mma)==0 || std::atoi(uniform_prepare_mma)==3) &&
        (!uniform_prepare_up || uniform_prepare_up[0]!='0') &&
        down.aligned_q8 && !down.q8_zero && down.gs==48 &&
        up.aligned_q8 && !up.q8_zero && up.gs==48 && up.neuron_len==320 &&
        !dense_injection.defined() && i.aligned && !i.zero;
    const int uniform_prepare_slot=uniform_prepare && uniform_prepare_mode>=2?gr_norm_counter_slot():-1;
    if(uniform_prepare_mode>=2 && uniform_prepare_slot<0)uniform_prepare=false;
    const auto* direct_s16_option=std::getenv("MFQ_GR_UP_DIRECT_S16");
    const int direct_s16_mode=direct_s16_option?std::atoi(direct_s16_option):1;
    const bool direct_s16=uniform_prepare && (uniform_prepare_mode==3 ||
        (uniform_prepare_mode==1 && (direct_s16_mode==1 || direct_s16_mode==2)));
    const auto* direct_stats_option=std::getenv("MFQ_GR_UP_DIRECT_STATS");
    const bool direct_stats=direct_s16 && (!direct_stats_option || direct_stats_option[0]!='0');
    const auto* split_major_option=std::getenv("MFQ_GR_UP_SPLIT_MAJOR");
    const bool split_major=uniform_prepare && direct_s16 && (uniform_prepare_mode==1 || uniform_prepare_mode==3) &&
        (split_major_option?split_major_option[0]!='0':uniform_prepare_mode==3);
    const bool fused_half_up=!down_mma_candidate && m1_half_candidate && m1_half_mode==3 && half &&
        uniform_prepare && uniform_prepare_mode==3 && direct_s16 && direct_stats && split_major;
    if(m1_half_mode==3 && !fused_half_up)m1_half_candidate=false;
    const auto* up_int_mma_option=std::getenv("MFQ_GR_UP_INT16_MMA");
    const int up_int_mma_mode=up_int_mma_option?std::atoi(up_int_mma_option):0;
    const bool up_int_mma=!fused_half_up && (up_int_mma_mode==1 || up_int_mma_mode==2 || up_int_mma_mode==3) &&
        uniform_prepare && uniform_prepare_mode==3 && direct_s16 && direct_stats && split_major;
    const bool down_mma_fused_up=down_fused_requested && down_mma_candidate && uniform_prepare &&
        uniform_prepare_mode==3 && direct_s16 && direct_stats && split_major && plane_count==4 && !up_int_mma;
    if(down_fused_requested && down_mma_candidate && !down_mma_fused_up)uniform_prepare=false;
    MFQ_RUNTIME_CHECK(!packed_down_s16 || (uniform_prepare && uniform_prepare_mode==3 &&
        split_major && direct_s16 && direct_stats && !fused_half_up && !up_int_mma),
        "direct S16 GR Down requires the existing split-major S16 Up preparation");
    tb::Tensor uniform_up_values,uniform_up_scales,uniform_up_sums;
    GrTwoPlane uniform_up{};
    if(uniform_prepare) {
        constexpr int elements=336;
        uniform_up_values=tb::empty({up_int_mma?4:2,elements},options.dtype(tb::kInt8));
        uniform_up_scales=tb::empty({7,2},options.dtype(tb::kFloat32));
        uniform_up_sums=tb::empty({7,4},options.dtype(tb::kInt32));
        uniform_up={uniform_up_values.data_ptr<int8_t>(),nullptr,uniform_up_values.data_ptr<int8_t>()+elements,
            reinterpret_cast<float2*>(uniform_up_scales.data_ptr<float>()),
            reinterpret_cast<int4*>(uniform_up_sums.data_ptr<int32_t>())};
    }
    const auto launch_down=[&](auto group) {
        static constexpr int fixed=decltype(group)::value;
        if constexpr(fixed==48) {
            if(down_mma_candidate && two_plane && (plane_count==3 || plane_count==4) &&
                (dense_warp || (!dense_injection.defined() && i.aligned && !i.zero))) {
                if(down_mma_fused_up) {
                    gated_residual_down_mma_kernel<1,4,true,true><<<
                        dim3(unsigned(5*ds)),128,unsigned(inject_capacity*sizeof(float)),mfq_current_cuda_stream()>>>(
                        d,i,projection_normalized,projection_input.output_half,parts.data_ptr<float>(),ds,is,vector_injection,
                        nullptr,global_quantized,uniform_up,uniform_prepare_slot);
                    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                        "gr_down_mma_prepare_direct_up",reinterpret_cast<const void*>(gated_residual_down_mma_kernel<1,4,true,true>),
                        "M=1 half="+std::to_string(half)+" splits="+std::to_string(ds));
                    return;
                }
                const auto run=[&](auto count,auto planes) {
                    gated_residual_down_mma_kernel<decltype(count)::value,decltype(planes)::value><<<
                        dim3(unsigned(5*ds)),128,unsigned(inject_capacity*sizeof(float)),mfq_current_cuda_stream()>>>(
                        d,i,projection_normalized,projection_input.output_half,parts.data_ptr<float>(),ds,is,vector_injection,
                        dense_warp?(promoted_normalized.defined()?promoted_normalized.data_ptr<float>():normalized.data_ptr<float>()):nullptr,
                        global_quantized);
                    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                        "gr_down_mma",reinterpret_cast<const void*>(gated_residual_down_mma_kernel<decltype(count)::value,decltype(planes)::value>),
                        "M="+std::to_string(rows)+" half="+std::to_string(half)+" splits="+std::to_string(ds));
                };
                const auto launch=[&](auto count) {
                    if(plane_count==3)run(count,std::integral_constant<int,3>{});
                    else run(count,std::integral_constant<int,4>{});
                };
                switch(rows) {
                    case 1:launch(std::integral_constant<int,1>{});break;
                    case 2:launch(std::integral_constant<int,2>{});break;
                    case 3:launch(std::integral_constant<int,3>{});break;
                    case 4:launch(std::integral_constant<int,4>{});break;
                    case 5:launch(std::integral_constant<int,5>{});break;
                    case 6:launch(std::integral_constant<int,6>{});break;
                }
                return;
            }
            const auto* pair=std::getenv("MFQ_GR_INT_PAIR_DOWN");
            if(two_plane && rows>=2 && rows<=6 && (!pair || pair[0]=='1') &&
                (dense_warp || (!dense_injection.defined() && i.aligned && !i.zero))) {
                const auto launch=[&](auto planes) {
                    gated_residual_batch_projection_kernel<48,2,decltype(planes)::value><<<
                        dim3(unsigned((down.out+3)/4*ds),unsigned((rows+1)/2)),128,0,mfq_current_cuda_stream()>>>(
                        projection_input,d,i,projection_normalized,parts.data_ptr<float>(),ds,is,rows,vector_injection,global_quantized);
                    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                        GrDotFlags<decltype(planes)::value>::unsigned_weights?
                            (GrDotFlags<decltype(planes)::value>::triple?"gr_down_unsigned_int24_pair":"gr_down_unsigned_uniform16_pair"):
                        decltype(planes)::value==5?"gr_down_int24_combined_pair":decltype(planes)::value==6?"gr_down_uniform16_combined_pair":
                        decltype(planes)::value==3?"gr_down_int24_pair":"gr_down_uniform16_pair",
                        reinterpret_cast<const void*>(gated_residual_batch_projection_kernel<48,2,decltype(planes)::value>),
                        "M="+std::to_string(rows)+" half="+std::to_string(half));
                };
                if(plane_count==3) {
                    if(unsigned_dot && combined_affine)launch(std::integral_constant<int,9>{});
                    else if(unsigned_dot)launch(std::integral_constant<int,7>{});
                    else if(combined_affine)launch(std::integral_constant<int,5>{});
                    else launch(std::integral_constant<int,3>{});
                }
                else if(plane_count==4) {
                    if(unsigned_dot && combined_affine)launch(std::integral_constant<int,10>{});
                    else if(unsigned_dot)launch(std::integral_constant<int,8>{});
                    else if(combined_affine)launch(std::integral_constant<int,6>{});
                    else launch(std::integral_constant<int,4>{});
                }
                else launch(std::integral_constant<int,2>{});
                return;
            }
        }
        const auto* batch_setting=std::getenv("MFQ_GR_BATCH_SHARED_DOWN");
        if constexpr(fixed>0)if(batch_setting && batch_setting[0]=='1' &&
                rows>=2 && rows<=8 && share_norm &&
                (dense_warp || (!dense_injection.defined() && i.aligned && !i.zero))) {
            gated_residual_batch_projection_kernel<fixed><<<
                dim3(unsigned((down.out+3)/4*ds),unsigned((rows+1)/2)),128,0,mfq_current_cuda_stream()>>>(
                projection_input,d,i,projection_normalized,parts.data_ptr<float>(),ds,is,rows,vector_injection);
            return;
        }
        if(share_norm) {
            const auto launch=[&](auto threads) {
              const auto format=[&](auto injection_kind,auto static_format) {
                const auto packed=[&](auto vector_layout) {
                  const auto run=[&](auto cooperative) {
                    const auto launch_plane=[&](auto plane) {
                    static constexpr int planes=decltype(plane)::value;
                    constexpr bool quantized=planes>0;
                    const int capacity=quantized?0:down_capacity;
                    const size_t shared_bytes=quantized?size_t(inject_capacity)*sizeof(float):shared;
                    if constexpr(fixed==48 && (planes==4 || planes==8) && std::decay_t<decltype(threads)>::value==256 &&
                            std::decay_t<decltype(injection_kind)>::value==2 && std::decay_t<decltype(static_format)>::value &&
                            !std::decay_t<decltype(vector_layout)>::value && std::decay_t<decltype(cooperative)>::value==0) {
                        if(uniform_prepare && uniform_prepare_mode==2) {
                            constexpr int prepare_row_tile=24;
                            gated_residual_projection_kernel<48,256,2,true,false,0,4,true><<<
                                dim3(unsigned((down.out+prepare_row_tile-1)/prepare_row_tile*ds)),256,unsigned(shared_bytes),mfq_current_cuda_stream()>>>(
                                projection_input,d,i,projection_normalized,parts.data_ptr<float>(),ds,is,prepare_row_tile,capacity,
                                nullptr,asynchronous,global_quantized,uniform_up,uniform_prepare_slot);
                            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                                "gr_down_uniform16_prepare_up",reinterpret_cast<const void*>(gated_residual_projection_kernel<48,256,2,true,false,0,4,true>),
                                "M=1 half="+std::to_string(half)+" splits="+std::to_string(ds));
                            return;
                        }
                        if(uniform_prepare && uniform_prepare_mode==3) {
                            const auto* tile_option=std::getenv("MFQ_GR_UP_FUSED_DIRECT_TILE");
                            const int prepare_row_tile=tile_option?std::atoi(tile_option):(ds==8?8:16);
                            MFQ_RUNTIME_CHECK(prepare_row_tile==8 || prepare_row_tile==16 || prepare_row_tile==24 || prepare_row_tile==48,
                                "fused direct GR Up row tile must divide GS48");
                            if(packed_down_s16) {
                                const auto launch_s16=[&](auto mode) {
                                gated_residual_projection_kernel<48,256,2,true,false,0,8,true,true,false,true,true,false,false,decltype(mode)::value><<<
                                    dim3(unsigned((down.out+prepare_row_tile-1)/prepare_row_tile*ds)),256,unsigned(shared_bytes),mfq_current_cuda_stream()>>>(
                                    projection_input,d,i,projection_normalized,parts.data_ptr<float>(),ds,is,prepare_row_tile,capacity,
                                    nullptr,asynchronous,global_quantized,uniform_up,uniform_prepare_slot);
                                if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                                    "gr_down_direct_s16_prepare_up",reinterpret_cast<const void*>(gated_residual_projection_kernel<48,256,2,true,false,0,8,true,true,false,true,true,false,false,decltype(mode)::value>),
                                    "M=1 half="+std::to_string(half)+" splits="+std::to_string(ds)+" tile="+std::to_string(prepare_row_tile)+" mode="+std::to_string(decltype(mode)::value));
                                };
                                if(s16_down_mode==2)launch_s16(std::integral_constant<int,2>{});
                                else launch_s16(std::integral_constant<int,1>{});
                                return;
                            }
                            if(fused_half_up) {
                                gated_residual_projection_kernel<48,256,2,true,false,0,planes,true,true,false,true,true,false,true><<<
                                    dim3(unsigned((down.out+prepare_row_tile-1)/prepare_row_tile*ds)),256,unsigned(shared_bytes),mfq_current_cuda_stream()>>>(
                                    projection_input,d,i,projection_normalized,parts.data_ptr<float>(),ds,is,prepare_row_tile,capacity,
                                    nullptr,asynchronous,global_quantized,uniform_up,uniform_prepare_slot);
                                if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                                    "gr_down_prepare_half_up",reinterpret_cast<const void*>(gated_residual_projection_kernel<48,256,2,true,false,0,planes,true,true,false,true,true,false,true>),
                                    "M=1 half=1 splits="+std::to_string(ds)+" tile="+std::to_string(prepare_row_tile));
                                return;
                            }
                            const auto launch_stats=[&](auto paired) {
                            if constexpr(decltype(paired)::value)if(up_int_mma) {
                                gated_residual_projection_kernel<48,256,2,true,false,0,planes,true,true,false,true,true,true><<<
                                    dim3(unsigned((down.out+prepare_row_tile-1)/prepare_row_tile*ds)),256,unsigned(shared_bytes),mfq_current_cuda_stream()>>>(
                                    projection_input,d,i,projection_normalized,parts.data_ptr<float>(),ds,is,prepare_row_tile,capacity,
                                    nullptr,asynchronous,global_quantized,uniform_up,uniform_prepare_slot);
                                if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                                    "gr_down_prepare_int16_mma_up",reinterpret_cast<const void*>(gated_residual_projection_kernel<48,256,2,true,false,0,planes,true,true,false,true,true,true>),
                                    "M=1 half="+std::to_string(half)+" splits="+std::to_string(ds)+" tile="+std::to_string(prepare_row_tile));
                                return;
                            }
                            if(split_major) {
                                gated_residual_projection_kernel<48,256,2,true,false,0,planes,true,true,false,true,decltype(paired)::value><<<
                                    dim3(unsigned((down.out+prepare_row_tile-1)/prepare_row_tile*ds)),256,unsigned(shared_bytes),mfq_current_cuda_stream()>>>(
                                    projection_input,d,i,projection_normalized,parts.data_ptr<float>(),ds,is,prepare_row_tile,capacity,
                                    nullptr,asynchronous,global_quantized,uniform_up,uniform_prepare_slot);
                                if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                                    "gr_down_uniform16_prepare_direct_up_split_major",reinterpret_cast<const void*>(gated_residual_projection_kernel<48,256,2,true,false,0,planes,true,true,false,true,decltype(paired)::value>),
                                    "M=1 half="+std::to_string(half)+" splits="+std::to_string(ds)+" tile="+std::to_string(prepare_row_tile)+" paired_stats="+std::to_string(decltype(paired)::value)+" unsigned="+std::to_string(GrDotFlags<planes>::unsigned_weights));
                                return;
                            }
                            const auto run=[&](auto vector_parts) {
                            gated_residual_projection_kernel<48,256,2,true,false,0,planes,true,true,decltype(vector_parts)::value,false,decltype(paired)::value><<<
                                dim3(unsigned((down.out+prepare_row_tile-1)/prepare_row_tile*ds)),256,unsigned(shared_bytes),mfq_current_cuda_stream()>>>(
                                projection_input,d,i,projection_normalized,parts.data_ptr<float>(),ds,is,prepare_row_tile,capacity,
                                nullptr,asynchronous,global_quantized,uniform_up,uniform_prepare_slot);
                            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                                decltype(vector_parts)::value?"gr_down_uniform16_prepare_direct_up_vector":"gr_down_uniform16_prepare_direct_up",
                                reinterpret_cast<const void*>(gated_residual_projection_kernel<48,256,2,true,false,0,planes,true,true,decltype(vector_parts)::value,false,decltype(paired)::value>),
                                "M=1 half="+std::to_string(half)+" splits="+std::to_string(ds)+" tile="+std::to_string(prepare_row_tile)+" paired_stats="+std::to_string(decltype(paired)::value)+" unsigned="+std::to_string(GrDotFlags<planes>::unsigned_weights));
                            };
                            const auto* vector_option=std::getenv("MFQ_GR_UP_PARTS_VECTOR");
                            if(vector_option && vector_option[0]!='0' && (ds==13 || ds==16))run(std::true_type{});
                            else run(std::false_type{});
                            };
                            if(direct_stats)launch_stats(std::true_type{});else launch_stats(std::false_type{});
                            return;
                        }
                        if(split_major) {
                            gated_residual_projection_kernel<48,256,2,true,false,0,planes,false,false,false,true><<<
                                dim3(unsigned((down.out+tile-1)/tile*ds)),256,unsigned(shared_bytes),mfq_current_cuda_stream()>>>(
                                projection_input,d,i,projection_normalized,parts.data_ptr<float>(),ds,is,tile,capacity,
                                nullptr,asynchronous,global_quantized);
                            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                                "gr_down_uniform16_split_major",reinterpret_cast<const void*>(gated_residual_projection_kernel<48,256,2,true,false,0,planes,false,false,false,true>),
                                "M=1 half="+std::to_string(half)+" splits="+std::to_string(ds));
                            return;
                        }
                    }
                    gated_residual_projection_kernel<fixed,std::decay_t<decltype(threads)>::value,std::decay_t<decltype(injection_kind)>::value,std::decay_t<decltype(static_format)>::value,decltype(vector_layout)::value,decltype(cooperative)::value,planes><<<dim3(unsigned((down.out+tile-1)/tile*ds),unsigned(rows)),unsigned(threads),unsigned(shared_bytes),mfq_current_cuda_stream()>>>(
                        projection_input,d,i,projection_normalized,parts.data_ptr<float>(),ds,is,tile,capacity,
                        dense_warp?(promoted_normalized.defined()?promoted_normalized.data_ptr<float>():normalized.data_ptr<float>()):nullptr,asynchronous,global_quantized);
                    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                        GrDotFlags<planes>::unsigned_weights?(GrDotFlags<planes>::triple?"gr_down_unsigned_int24":"gr_down_unsigned_uniform16"):
                        planes==5?"gr_down_int24_combined":planes==6?"gr_down_uniform16_combined":planes==3?"gr_down_int24":planes==4?"gr_down_uniform16":planes==2?"gr_down_int16":"gr_down_float",
                        reinterpret_cast<const void*>(gated_residual_projection_kernel<fixed,std::decay_t<decltype(threads)>::value,std::decay_t<decltype(injection_kind)>::value,std::decay_t<decltype(static_format)>::value,decltype(vector_layout)::value,decltype(cooperative)::value,planes>),
                        "M="+std::to_string(rows)+" half="+std::to_string(half)+" gs="+std::to_string(down.gs)+" groups="+std::to_string(down.ng));
                    };
                    if constexpr(fixed==48) {
                        if(two_plane && plane_count==3) {
                            if(unsigned_dot && combined_affine)launch_plane(std::integral_constant<int,9>{});
                            else if(unsigned_dot)launch_plane(std::integral_constant<int,7>{});
                            else if(combined_affine)launch_plane(std::integral_constant<int,5>{});
                            else launch_plane(std::integral_constant<int,3>{});
                        }
                        else if(two_plane && plane_count==4) {
                            if(unsigned_dot && combined_affine)launch_plane(std::integral_constant<int,10>{});
                            else if(unsigned_dot)launch_plane(std::integral_constant<int,8>{});
                            else if(combined_affine)launch_plane(std::integral_constant<int,6>{});
                            else launch_plane(std::integral_constant<int,4>{});
                        }
                        else if(two_plane)launch_plane(std::integral_constant<int,2>{});
                        else launch_plane(std::integral_constant<int,0>{});
                    }else launch_plane(std::integral_constant<int,0>{});
                  };
                  if constexpr(fixed==0) {
                      if(group_read==2)run(std::integral_constant<int,2>{});
                      else if(group_read==1)run(std::true_type{});else run(std::false_type{});
                  }else run(std::false_type{});
                };
                if constexpr(decltype(injection_kind)::value!=2) {
                    if(vector_injection)packed(std::true_type{});
                    else packed(std::false_type{});
                }else packed(std::false_type{});
              };
              if constexpr(fixed>0) {
                if(fixed_formats) {
                    if(dense_injection.defined())format(std::integral_constant<int,1>{},std::true_type{});
                    else if(inject->aligned_q8 && !inject->q8_zero)format(std::integral_constant<int,2>{},std::true_type{});
                    else format(std::integral_constant<int,0>{},std::true_type{});
                }else format(std::integral_constant<int,0>{},std::false_type{});
              }else format(std::integral_constant<int,0>{},std::false_type{});
            };
            if(projection_threads==128)launch(std::integral_constant<int,128>{});
            else if(projection_threads==512)launch(std::integral_constant<int,512>{});
            else launch(std::integral_constant<int,256>{});
        }else gated_residual_down_kernel<fixed><<<dim3(unsigned((down.out+tile-1)/tile*ds),unsigned(rows)),256,unsigned(shared),mfq_current_cuda_stream()>>>(
            input,d,i,updated.data_ptr(),normalized.data_ptr(),promoted_normalized.defined()?promoted_normalized.data_ptr<float>():nullptr,parts.data_ptr<float>(),ds,is,float(eps),tile,down_capacity);
    };
    if(packed_down && two_plane) {
        const auto launch_packed=[&](auto plane) {
        const auto* pair_setting=std::getenv("MFQ_GR_PACKED24_PAIR_DOWN");
        const bool pair_default=rows==2 || ((rows==5 || rows==6) && half && after);
        if(rows>=2 && (pair_setting?pair_setting[0]!='0':pair_default)) {
            gated_residual_batch_projection_kernel<24,2,decltype(plane)::value><<<
                dim3(unsigned((down.out+3)/4*ds),unsigned((rows+1)/2)),128,unsigned(2*inject_capacity*sizeof(float)),mfq_current_cuda_stream()>>>(
                projection_input,d,i,projection_normalized,parts.data_ptr<float>(),ds,is,rows,vector_injection,global_quantized);
            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                "gr_down_packed24_pair",reinterpret_cast<const void*>(gated_residual_batch_projection_kernel<24,2,decltype(plane)::value>),
                "M="+std::to_string(rows)+" half="+std::to_string(half)+" planes="+std::to_string(decltype(plane)::value));
            return;
        }
        gated_residual_projection_kernel<24,256,2,false,false,0,decltype(plane)::value><<<
            dim3(unsigned((down.out+7)/8*ds),unsigned(rows)),256,unsigned(inject_capacity*sizeof(float)),mfq_current_cuda_stream()>>>(
            projection_input,d,i,projection_normalized,parts.data_ptr<float>(),ds,is,8,0,
            nullptr,asynchronous,global_quantized);
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
            decltype(plane)::value==3?"gr_down_packed24_int24":"gr_down_packed24_uniform16",
            reinterpret_cast<const void*>(gated_residual_projection_kernel<24,256,2,false,false,0,decltype(plane)::value>),
            "M="+std::to_string(rows)+" half="+std::to_string(half)+" groups="+std::to_string(down.ng));
        };
        if(plane_count==4)launch_packed(std::integral_constant<int,4>{});else launch_packed(std::integral_constant<int,3>{});
    }
    else if(fixed_groups && down.aligned_q8 && !down.q8_zero && down.gs==32)launch_down(std::integral_constant<int,32>{});
    else if(fixed_groups && down.aligned_q8 && !down.q8_zero && down.gs==48)launch_down(std::integral_constant<int,48>{});
    else if(fixed_groups && down.aligned_q8 && !down.q8_zero && down.gs==64)launch_down(std::integral_constant<int,64>{});
    else launch_down(std::integral_constant<int,0>{});
    if(marker)marker("down");
    if(up_int_mma) {
        const auto run=[&](auto unroll,auto shared) {
        gated_residual_up_int16_mma_kernel<decltype(unroll)::value,decltype(shared)::value><<<hidden/8,128,0,mfq_current_cuda_stream()>>>(
            u,i,normalized.data_ptr(),parts.data_ptr<float>(),mixed.data_ptr(),next_injection.data_ptr(),ds,is,half,uniform_up);
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
            "gr_up_int16_digits_mma",reinterpret_cast<const void*>(gated_residual_up_int16_mma_kernel<decltype(unroll)::value,decltype(shared)::value>),
            "M=1 half="+std::to_string(half)+" tile=8 shared="+std::to_string(decltype(shared)::value)+" unroll="+std::to_string(decltype(unroll)::value));
        };
        if(up_int_mma_mode==2)run(std::integral_constant<int,1>{},std::true_type{});
        else if(up_int_mma_mode==3)run(std::integral_constant<int,7>{},std::true_type{});
        else run(std::integral_constant<int,7>{},std::false_type{});
        if(marker){marker("up_mix");marker("end");}
        MFQ_CUDA_CHECK(cudaGetLastError());return {mixed,updated,next_injection};
    }
    tb::Tensor dense_projection;
    if(dense_injection.defined() && !dense_warp) {
        const auto right=vector_injection?
            prepared_injection_right.transpose(1,2).reshape({down.neuron_len,streams}):
            prepared_injection_right;
        dense_projection=tb::matmul(promoted_normalized.defined()?promoted_normalized:normalized,
            right.defined()?right:dense_injection.transpose(-1,-2).to(tb::kFloat32));
    }
    if(marker && dense_projection.defined())marker("dense_projection");
    const auto* prepare_setting=std::getenv("MFQ_GR_UP_INT_DOT");
    const auto* half_groups_setting=std::getenv("MFQ_GR_UP_HALF_GROUPS");
    const bool half_groups_default=(!half_groups_setting || half_groups_setting[0]!='0') &&
        plane_count!=0 && rows==1 && half && after;
    const int prepare_mode=prepare_setting?std::atoi(prepare_setting):
        (plane_count!=0 && rows>=2 && rows<=6?7:5);
    const auto* mma_setting=std::getenv("MFQ_GR_UP_MMA");
    const int mma_mode=mma_setting?std::atoi(mma_setting):3;
    const bool mma_up=(mma_mode==1 || mma_mode==2 || (mma_mode==3 && rows>=2)) &&
        plane_count!=0 && rows<=6 && fixed_groups && streams==4 &&
        up.aligned_q8 && !up.q8_zero && up.gs==48 && up.neuron_len==320 && hidden==2560 &&
        down.neuron_len==10240 && down.out==320 && up_metadata==0;
    const auto* mma_uniform_option=std::getenv("MFQ_GR_UP_MMA_UNIFORM16");
    const bool mma_uniform16=mma_up && rows>=2 && mma_uniform_option && mma_uniform_option[0]!='0';
    const auto* mma_half_option=std::getenv("MFQ_GR_UP_MMA_F16_SPLIT");
    const int mma_half_mode=mma_half_option?std::atoi(mma_half_option):2;
    if(m1_half_candidate || (mma_up && rows>=2 &&
            (mma_half_mode==1 || (mma_half_mode==2 && (half || rows==2))))) {
        const bool local_half=rows==1 && m1_half_mode==2;
        const int64_t elements=int64_t(rows)*336;
        tb::Tensor values,statistics;
        GrUpHalf prepared{};
        if(fused_half_up)prepared={reinterpret_cast<__half*>(uniform_up.high),nullptr,uniform_up.scales};
        if(!local_half && !fused_half_up) {
            values=tb::empty({half?1:2,elements},options.dtype(tb::kFloat16));
            statistics=tb::empty({rows,7,2},options.dtype(tb::kFloat32));
            prepared={values.data_ptr<__half>(),half?nullptr:values.data_ptr<__half>()+elements,
                reinterpret_cast<float2*>(statistics.data_ptr<float>())};
        }
        const auto prepare=[&](auto one) {
            const auto run=[&](auto registers) {
                const auto kernel=[&](auto vector_parts) {
                gated_residual_prepare_up_half_kernel<decltype(one)::value,decltype(registers)::value,decltype(vector_parts)::value><<<rows,256,0,mfq_current_cuda_stream()>>>(
                    d,parts.data_ptr<float>(),prepared,ds,int(streams)*is,half);
                if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                    decltype(vector_parts)::value?"gr_prepare_up_half_vector":decltype(registers)::value?"gr_prepare_up_half_registers":"gr_prepare_up_half",
                    reinterpret_cast<const void*>(gated_residual_prepare_up_half_kernel<decltype(one)::value,decltype(registers)::value,decltype(vector_parts)::value>),
                    "M="+std::to_string(rows)+" half="+std::to_string(half));
                };
                const auto* vector_option=std::getenv("MFQ_GR_UP_PARTS_VECTOR");
                if constexpr(!decltype(registers)::value) {
                    const bool selected=vector_option?vector_option[0]!='0':half && rows==6;
                    if(selected && (ds==13 || ds==16))kernel(std::true_type{});
                    else kernel(std::false_type{});
                }else kernel(std::false_type{});
            };
            const auto* register_option=std::getenv("MFQ_GR_UP_MMA_F16_REGISTER_PREPARE");
            if(register_option && register_option[0]!='0')run(std::true_type{});else run(std::false_type{});
        };
        if(!local_half && !fused_half_up) {
            if(half)prepare(std::true_type{});else prepare(std::false_type{});
        }
        const auto launch=[&](auto count) {
            const auto storage=[&](auto local) {
            const auto run=[&](auto one) {
                const auto kernel=[&](auto unroll) {
                gated_residual_up_half_mma_kernel<std::decay_t<decltype(count)>::value,decltype(one)::value,std::decay_t<decltype(local)>::value,decltype(unroll)::value><<<
                    hidden/16,128,0,mfq_current_cuda_stream()>>>(d,u,i,normalized.data_ptr(),parts.data_ptr<float>(),
                    mixed.data_ptr(),next_injection.data_ptr(),ds,is,
                    dense_projection.defined()?dense_projection.data_ptr<float>():nullptr,half,prepared);
                if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                    "gr_up_half_mma",reinterpret_cast<const void*>(gated_residual_up_half_mma_kernel<std::decay_t<decltype(count)>::value,decltype(one)::value,std::decay_t<decltype(local)>::value,decltype(unroll)::value>),
                    "M="+std::to_string(rows)+" half="+std::to_string(half)+" planes="+std::to_string(decltype(one)::value?1:2)+
                    " local_prepare="+std::to_string(std::decay_t<decltype(local)>::value)+
                    " group_unroll="+std::to_string(decltype(unroll)::value));
                };
                const auto* unroll_option=std::getenv("MFQ_GR_UP_MMA_F16_GROUP_UNROLL");
                const int unroll=unroll_option?std::atoi(unroll_option):
                    half && (rows==2 || rows==3 || rows==4 || rows==6)?7:1;
                if(unroll==7)kernel(std::integral_constant<int,7>{});
                else if(unroll==2)kernel(std::integral_constant<int,2>{});
                else kernel(std::integral_constant<int,1>{});
            };
            if(half)run(std::true_type{});else run(std::false_type{});
            };
            if constexpr(decltype(count)::value==1)if(local_half) {storage(std::true_type{});return;}
            storage(std::false_type{});
        };
        switch(rows) {
            case 1:launch(std::integral_constant<int,1>{});break;
            case 2:launch(std::integral_constant<int,2>{});break;
            case 3:launch(std::integral_constant<int,3>{});break;
            case 4:launch(std::integral_constant<int,4>{});break;
            case 5:launch(std::integral_constant<int,5>{});break;
            case 6:launch(std::integral_constant<int,6>{});break;
        }
        if(marker){marker("up_mix");marker("end");}
        MFQ_CUDA_CHECK(cudaGetLastError());return {mixed,updated,next_injection};
    }
    const bool prepare_up=(prepare_mode==7 || prepare_mode==8 || (down_mma_candidate && rows==1 && !uniform_prepare) ||
        ((prepare_mode==9 || prepare_mode==11) && rows>=2) ||
        (mma_up && (rows>1 || mma_mode==2))) && rows<=6 &&
        fixed_groups && streams==4 && up.aligned_q8 && !up.q8_zero && up.gs==48 &&
        up.neuron_len==320 && hidden==2560 && down.neuron_len==10240 && down.out==320 && up_metadata==0;
    tb::Tensor up_plane_values,up_plane_scales,up_plane_sums;
    GrTwoPlane prepared_up=uniform_up;
    if(uniform_prepare && uniform_prepare_mode==1) {
        const auto prepare=[&](auto packed) {
            const auto run=[&](auto vector_parts,auto split_layout) {
            const auto statistics=[&](auto paired) {
            gated_residual_prepare_up_integer_kernel<true,decltype(packed)::value,decltype(vector_parts)::value,decltype(split_layout)::value,decltype(paired)::value><<<1,256,0,mfq_current_cuda_stream()>>>(
                d,u,parts.data_ptr<float>(),prepared_up,ds,int(streams)*is,int(streams),half);
            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                decltype(split_layout)::value?"gr_prepare_up_direct_s16_split_major":decltype(vector_parts)::value?"gr_prepare_up_direct_s16_vector":decltype(packed)::value?"gr_prepare_up_direct_s16":"gr_prepare_up_uniform16",
                reinterpret_cast<const void*>(gated_residual_prepare_up_integer_kernel<true,decltype(packed)::value,decltype(vector_parts)::value,decltype(split_layout)::value,decltype(paired)::value>),"M=1 half="+std::to_string(half)+" paired_stats="+std::to_string(decltype(paired)::value));
            };
            if constexpr(decltype(packed)::value) {
                if(direct_stats)statistics(std::true_type{});else statistics(std::false_type{});
            }else statistics(std::false_type{});
            };
            const auto* vector_option=std::getenv("MFQ_GR_UP_PARTS_VECTOR");
            if constexpr(decltype(packed)::value) {
                if(split_major)run(std::false_type{},std::true_type{});
                else if(vector_option && vector_option[0]!='0' && (ds==13 || ds==16))run(std::true_type{},std::false_type{});
                else run(std::false_type{},std::false_type{});
            }else run(std::false_type{},std::false_type{});
        };
        if(direct_s16)prepare(std::true_type{});else prepare(std::false_type{});
    }
    if(prepare_up) {
        const int64_t elements=int64_t(rows)*up.ng*48;
        const int storage_planes=mma_uniform16?2:3;
        up_plane_values=tb::empty({storage_planes,elements},options.dtype(tb::kInt8));
        up_plane_scales=tb::empty({rows,up.ng,2},options.dtype(tb::kFloat32));
        up_plane_sums=tb::empty({rows,up.ng,4},options.dtype(tb::kInt32));
        prepared_up={up_plane_values.data_ptr<int8_t>(),mma_uniform16?nullptr:up_plane_values.data_ptr<int8_t>()+elements,
            up_plane_values.data_ptr<int8_t>()+(storage_planes-1)*elements,reinterpret_cast<float2*>(up_plane_scales.data_ptr<float>()),
            reinterpret_cast<int4*>(up_plane_sums.data_ptr<int32_t>())};
        const auto prepare=[&](auto uniform) {
            gated_residual_prepare_up_integer_kernel<decltype(uniform)::value><<<rows,256,0,mfq_current_cuda_stream()>>>(
                d,u,parts.data_ptr<float>(),prepared_up,ds,int(streams)*is,int(streams),half);
            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                decltype(uniform)::value?"gr_prepare_up_uniform16":"gr_prepare_up_int24",
                reinterpret_cast<const void*>(gated_residual_prepare_up_integer_kernel<decltype(uniform)::value>),
                "M="+std::to_string(rows)+" half="+std::to_string(half));
        };
        if(mma_uniform16)prepare(std::true_type{});else prepare(std::false_type{});
    }
    if(mma_up) {
        const auto launch=[&](auto count,auto local) {
            const auto run=[&](auto uniform) {
            gated_residual_up_mma_kernel<std::decay_t<decltype(count)>::value,std::decay_t<decltype(local)>::value,decltype(uniform)::value><<<
                hidden/16,128,0,mfq_current_cuda_stream()>>>(d,u,i,normalized.data_ptr(),parts.data_ptr<float>(),
                mixed.data_ptr(),next_injection.data_ptr(),ds,is,
                dense_projection.defined()?dense_projection.data_ptr<float>():nullptr,half,prepared_up);
            if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
                decltype(uniform)::value?"gr_up_uniform16_mma":"gr_up_int24_mma",
                reinterpret_cast<const void*>(gated_residual_up_mma_kernel<std::decay_t<decltype(count)>::value,std::decay_t<decltype(local)>::value,decltype(uniform)::value>),
                "M="+std::to_string(rows)+" half="+std::to_string(half)+" local_prepare="+std::to_string(std::decay_t<decltype(local)>::value));
            };
            if constexpr(decltype(count)::value>=2 && !decltype(local)::value)
                if(mma_uniform16){run(std::true_type{});return;}
            run(std::false_type{});
        };
        switch(rows) {
            case 1:if(prepare_up)launch(std::integral_constant<int,1>{},std::false_type{});
                else launch(std::integral_constant<int,1>{},std::true_type{});break;
            case 2:launch(std::integral_constant<int,2>{},std::false_type{});break;
            case 3:launch(std::integral_constant<int,3>{},std::false_type{});break;
            case 4:launch(std::integral_constant<int,4>{},std::false_type{});break;
            case 5:launch(std::integral_constant<int,5>{},std::false_type{});break;
            case 6:launch(std::integral_constant<int,6>{},std::false_type{});break;
        }
        if(marker){marker("up_mix");marker("end");}
        MFQ_CUDA_CHECK(cudaGetLastError());return {mixed,updated,next_injection};
    }
    const auto launch_up=[&](auto group) {
        static constexpr int fixed=decltype(group)::value;
        const auto launch=[&](auto width) {
            const auto run=[&](auto capacity) {
                static constexpr int count=decltype(capacity)::value;
                // The paired accumulator crosses the 64-register occupancy
                // boundary at 512 threads on Ampere. Keep four 256-thread
                // blocks resident, with the fixed-width specialization bounded
                // to 64 registers, instead of one wide block.
                const int selected_threads=count>1 && !thread_setting?256:up_threads;
                const int threads=count==4?std::min(selected_threads,256):selected_threads;
                const auto launch_metadata=[&](auto shared_metadata) {
                const auto launch_integer=[&](auto integer_up) {
                constexpr int up_mode=int(decltype(integer_up)::value);
                const auto* group_threads_setting=std::getenv("MFQ_GR_UP_GROUP_THREADS");
                const int group_threads=group_threads_setting?std::atoi(group_threads_setting):
                    up_mode==24?128:(up_mode==13 || (up_mode==11 && rows==1 && half && after)?512:256);
                MFQ_RUNTIME_CHECK((group_threads==128 && up_mode==24) || group_threads==256 || group_threads==512,"GR Up grouped threads must be 256 or 512");
                const int launch_threads=up_mode>=9?(count==4?std::min(group_threads,256):group_threads):threads;
                const int columns_per_block=(launch_threads/32)*(up_mode==11 || up_mode==12 || up_mode==13 || up_mode==14 || up_mode==15 || up_mode==17 || up_mode==18 || up_mode==19 || up_mode==20 || up_mode==21 || up_mode==22 || up_mode==23?2:up_mode==24 || up_mode==9 || up_mode==10?4:1);
                gated_residual_up_kernel<fixed,fixed?4:0,std::decay_t<decltype(width)>::value,count,decltype(shared_metadata)::value,up_mode><<<dim3(unsigned((hidden+columns_per_block-1)/columns_per_block),unsigned((rows+count-1)/count)),unsigned(launch_threads),up_mode==7 || up_mode==8 || up_mode==10 || up_mode==12 || up_mode==15 || up_mode==16 || up_mode==17 || up_mode==18 || up_mode==19 || up_mode==20 || up_mode==21 || up_mode==22 || up_mode==23 || up_mode==24?0:unsigned(count*down.out*sizeof(float)),mfq_current_cuda_stream()>>>(
                    d,u,i,normalized.data_ptr(),parts.data_ptr<float>(),mixed.data_ptr(),next_injection.data_ptr(),hidden,int(streams),half,ds,is,dense_projection.defined()?dense_projection.data_ptr<float>():nullptr,rows,prepared_up,
                    up_group_coefficients.defined() && !packed_up_metadata?reinterpret_cast<const float2*>(up_group_coefficients.data_ptr<float>()):nullptr,
                    packed_up_metadata?reinterpret_cast<const int4*>(up_group_coefficients.data_ptr<int32_t>()):nullptr,
                    packed_up_metadata?reinterpret_cast<const uint16_t*>(up_group_coefficients.data_ptr<int32_t>()+up.out*4):nullptr);
                if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),up_mode==24?"gr_up_mix_direct_s16_four_outputs":up_mode==23?"gr_up_mix_direct_s16_stats_vector24":up_mode==22?"gr_up_mix_direct_s16_stats_preload4":up_mode==21?"gr_up_mix_direct_s16_stats":up_mode==20?"gr_up_mix_direct_s16_preload4":up_mode==19?"gr_up_mix_direct_s16":up_mode==18?"gr_up_mix_uniform16_records":up_mode==17?"gr_up_mix_uniform16_coefficients":up_mode==16?"gr_up_mix_uniform16_prepared_warp":up_mode==15?"gr_up_mix_uniform16_prepared":up_mode==14?"gr_up_mix_uniform16_preload":up_mode==13?"gr_up_mix_uniform16_half_groups":up_mode==11 || up_mode==12?"gr_up_mix_int24_half_groups":up_mode==9 || up_mode==10?"gr_up_mix_int24_groups":up_mode==8?"gr_up_mix_int24_prepared_global":up_mode==7?"gr_up_mix_int24_prepared":up_mode==6?"gr_up_mix_int24_stream":up_mode==5?"gr_up_mix_int24_reuse":up_mode==3?"gr_up_mix_int24_chunks":up_mode?"gr_up_mix_uniform16":"gr_up_mix",
                    reinterpret_cast<const void*>(gated_residual_up_kernel<fixed,fixed?4:0,std::decay_t<decltype(width)>::value,count,decltype(shared_metadata)::value,decltype(integer_up)::value>),
                    "M="+std::to_string(rows)+" half="+std::to_string(half)+" capacity="+std::to_string(count));
                };
                if constexpr(fixed==48 && std::decay_t<decltype(width)>::value==320 && decltype(shared_metadata)::value==0) {
                    const auto* integer_up=std::getenv("MFQ_GR_UP_INT_DOT");
                    const bool verified_geometry=hidden==2560 && down.neuron_len==10240 && down.out==320;
                    const auto* uniform_up=std::getenv("MFQ_GR_UP_UNIFORM16");
                    const auto* preload_up=std::getenv("MFQ_GR_UP_PRELOAD");
                    if(uniform_prepare) {
                        if constexpr(count==1) {
                            const auto* vector_weights=std::getenv("MFQ_GR_UP_VECTOR_WEIGHTS");
                            const auto* direct_groups=std::getenv("MFQ_GR_UP_DIRECT_GROUPS");
                            if(direct_stats && direct_groups && direct_groups[0]!='0')launch_integer(std::integral_constant<int,24>{});
                            else if(direct_stats && vector_weights && vector_weights[0]!='0')launch_integer(std::integral_constant<int,23>{});
                            else if(direct_stats && direct_s16_mode==2)launch_integer(std::integral_constant<int,22>{});
                            else if(direct_stats)launch_integer(std::integral_constant<int,21>{});
                            else if(direct_s16 && direct_s16_mode==2)launch_integer(std::integral_constant<int,20>{});
                            else if(direct_s16)launch_integer(std::integral_constant<int,19>{});
                            else if(packed_up_metadata)launch_integer(std::integral_constant<int,18>{});
                            else if(up_group_coefficients.defined())launch_integer(std::integral_constant<int,17>{});
                            else if(prepared_full_warp)launch_integer(std::integral_constant<int,16>{});
                            else launch_integer(std::integral_constant<int,15>{});
                        }else MFQ_RUNTIME_CHECK(false,"prepared GR Up requires one token");
                    }
                    else if(rows==1 && verified_geometry && !prepare_up && integer_up && std::atoi(integer_up)==14)
                        launch_integer(std::integral_constant<int,14>{});
                    else if(rows==1 && verified_geometry && !prepare_up &&
                        ((integer_up && std::atoi(integer_up)==13) || (!integer_up && (!uniform_up || uniform_up[0]!='0'))))
                        if(!integer_up && (!preload_up || preload_up[0]!='0'))launch_integer(std::integral_constant<int,14>{});
                        else launch_integer(std::integral_constant<int,13>{});
                    else if(rows<=6 && verified_geometry && ((integer_up && std::atoi(integer_up)==11) ||
                            (!integer_up && half_groups_default))) {
                        if(prepare_up)launch_integer(std::integral_constant<int,12>{});
                        else launch_integer(std::integral_constant<int,11>{});
                    }
                    else if(rows<=6 && verified_geometry && integer_up && integer_up[0]=='9') {
                        if(prepare_up)launch_integer(std::integral_constant<int,10>{});
                        else launch_integer(std::integral_constant<int,9>{});
                    }
                    else if(prepare_up && prepare_mode==8)launch_integer(std::integral_constant<int,8>{});
                    else if(prepare_up)launch_integer(std::integral_constant<int,7>{});
                    else if(rows<=6 && verified_geometry && integer_up && integer_up[0]=='6')launch_integer(std::integral_constant<int,6>{});
                    else if(rows<=6 && verified_geometry && ((!integer_up && plane_count!=0) || (integer_up && integer_up[0]=='5')))launch_integer(std::integral_constant<int,5>{});
                    else if(rows<=6 && integer_up && integer_up[0]=='3')launch_integer(std::integral_constant<int,3>{});
                    else if(rows<=6 && integer_up && std::atoi(integer_up)==1)launch_integer(std::true_type{});
                    else launch_integer(std::false_type{});
                }else launch_integer(std::false_type{});
                };
                if constexpr(fixed>0 && std::decay_t<decltype(width)>::value==320) {
                    if(up_metadata==2)launch_metadata(std::integral_constant<int,2>{});
                    else if(up_metadata==1)launch_metadata(std::integral_constant<int,1>{});
                    else launch_metadata(std::integral_constant<int,0>{});
                }else launch_metadata(std::integral_constant<int,0>{});
            };
            const auto* shared_up=std::getenv("MFQ_GR_BATCH_SHARED_UP");
            if constexpr(fixed>0)if(rows>=4 && rows<=32 && shared_up &&
                    std::strcmp(shared_up,"4")==0) {
                run(std::integral_constant<int,4>{});return;
            }
            if constexpr(fixed>0)if(rows>=2 && rows<=32 && (!shared_up ||
                (shared_up[0]!='0' && (shared_up[0]!='2' || rows%2==0)))) {
                run(std::integral_constant<int,2>{});return;
            }
            run(std::integral_constant<int,1>{});
        };
        if constexpr(fixed>0) {
            if(up.neuron_len==320)launch(std::integral_constant<int,320>{});
            else launch(std::integral_constant<int,0>{});
        } else launch(std::integral_constant<int,0>{});
    };
    if(fixed_groups && streams==4 && up.aligned_q8 && !up.q8_zero && up.gs==32)launch_up(std::integral_constant<int,32>{});
    else if(fixed_groups && streams==4 && up.aligned_q8 && !up.q8_zero && up.gs==48)launch_up(std::integral_constant<int,48>{});
    else if(fixed_groups && streams==4 && up.aligned_q8 && !up.q8_zero && up.gs==64)launch_up(std::integral_constant<int,64>{});
    else launch_up(std::integral_constant<int,0>{});
    if(marker)marker("up_mix");
    MFQ_CUDA_CHECK(cudaGetLastError());
    if(marker)marker("end");
    return {mixed,updated,next_injection};
}
