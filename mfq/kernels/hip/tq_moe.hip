#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <type_traits>
#include <cstring>
#include "mfq_hip_compat.h"
#include "tq_moe.h"
#include "tq_vector.h"
#define MFQ_TQ_ROW_HELPERS_ONLY
#include "tq_float_tokens.h"
#undef MFQ_TQ_ROW_HELPERS_ONLY

namespace mfq_tq_moe {
using namespace mfq_tq_cuda_exact;

// CUDA's routed row decode/activation order, with complete CDNA waves.
template<int DT,bool Pair,int V=8>
__global__ __launch_bounds__(256) void decode(ProjectionView weight,ProjectionView up,
    const S<DT>* input,S<DT>* output,int rows,TqMoeRoutePlan plan,bool routed,float limit) {
    constexpr int R=Pair?2:4;
    const int wave=threadIdx.x/64,lane=threadIdx.x&63;
    const int role=Pair?(wave&1):0,slot=Pair?wave/2:wave;
    const int pair=blockIdx.y,expert=plan.ids[pair]-plan.expert_offset,row=blockIdx.x*R+slot;
    if(expert<0 || expert>=plan.expert_count) {
        const int zero_row=blockIdx.x*R+threadIdx.x;
        if(threadIdx.x<R && zero_row<rows)
            output[size_t(pair)*rows+zero_row]=rounded<DT>(0.f);
        return;
    }
    auto m=role?up:weight;
    if(m.columns==2 && expert>=0 && expert<plan.expert_count)
        m.scales=static_cast<const half*>(m.scales)+size_t(expert)*m.stored_columns;
    float value[1]{};
    const int global_row=expert*rows+row;
    const auto* x=input+size_t(routed?pair:pair/plan.routes)*m.input_columns;
    if(expert>=0 && expert<plan.expert_count && row<rows) {
        if(m.columns)generic_dispatch<DT,true,1,1,V>(m,global_row,x,0,1,m.input_columns,value);
        else generic_dispatch<DT,false,1,1,V>(m,global_row,x,0,1,m.input_columns,value);
    }
    for(int d=32;d;d>>=1)value[0]+=__shfl_down(value[0],d,64);
    __shared__ float joined[R][2];
    if(!lane) {
        if(expert>=0 && expert<plan.expert_count && row<rows && m.bias)value[0]+=m.bias[global_row];
        if constexpr(Pair)joined[slot][role]=number<DT>(rounded<DT>(value[0]));
        else if(row<rows)output[size_t(pair)*rows+row]=rounded<DT>(value[0]);
    }
    if constexpr(Pair) {
        __syncthreads();
        if(!role && !lane && row<rows) {
            float gate=joined[slot][0],value_up=joined[slot][1];
            if(limit>0){gate=fminf(gate,limit);value_up=fmaxf(-limit,fminf(value_up,limit));}
            output[size_t(pair)*rows+row]=rounded<DT>(value_up*(gate/(1.f+expf(-gate))));
        }
    }
}


template<int Bits>__device__ __forceinline__ void paired_shared_row(ProjectionView gate,ProjectionView up,
        uint64_t dg,uint64_t du,int row,const uint16_t* x,float& gv,float& uv) {
    row=__builtin_amdgcn_readfirstlane(row);
    GenericConsumer<0,1,16> g{},u{};
    for(int col=(threadIdx.x&63)*16;col<gate.input_columns;col+=64*16) {
        #pragma unroll
        for(int i=0;i<2;++i) {
            const auto v=__ldg(reinterpret_cast<const uint4*>(x+col+i*8));
            g.packed[0][i*4]=u.packed[0][i*4]=v.x;
            g.packed[0][i*4+1]=u.packed[0][i*4+1]=v.y;
            g.packed[0][i*4+2]=u.packed[0][i*4+2]=v.z;
            g.packed[0][i*4+3]=u.packed[0][i*4+3]=v.w;
        }
        mfq_tq_vector::visit_values<16,true,true,Bits,GenericConsumer<0,1,16>,false,1>(gate.words,dg,gate.scales,
            static_cast<const half*>(gate.relative),nullptr,nullptr,row,col,gate.stored_columns,gate.gslog,g);
        mfq_tq_vector::visit_values<16,true,true,Bits,GenericConsumer<0,1,16>,false,1>(up.words,du,up.scales,
            static_cast<const half*>(up.relative),nullptr,nullptr,row,col,up.stored_columns,up.gslog,u);
    }
    gv=g.sum[0][0]+g.sum[0][1];uv=u.sum[0][0]+u.sum[0][1];
}
__global__ __launch_bounds__(256) void decode_pair_shared(ProjectionView gate,ProjectionView up,
        const uint16_t* input,uint16_t* output,int rows,TqMoeRoutePlan plan,float limit) {
    const int lane=threadIdx.x&63,row=blockIdx.x*4+threadIdx.x/64;
    const int pair=blockIdx.y,expert=plan.ids[pair]-plan.expert_offset;
    if(expert<0 || expert>=plan.expert_count) {
        const int z=blockIdx.x*4+threadIdx.x;
        if(threadIdx.x<4 && z<rows)output[size_t(pair)*rows+z]=rounded<0>(0.f);
        return;
    }
    gate.scales=static_cast<const half*>(gate.scales)+size_t(expert)*gate.stored_columns;
    up.scales=static_cast<const half*>(up.scales)+size_t(expert)*up.stored_columns;
    float gv=0,uv=0;const int global_row=__builtin_amdgcn_readfirstlane(expert*rows+row);
    const auto* x=input+size_t(pair/plan.routes)*gate.input_columns;
    if(row<rows) {
        const auto rg=__ldg(gate.descriptors+global_row),ru=__ldg(up.descriptors+global_row);
        const uint64_t dg=uint64_t(__builtin_amdgcn_readfirstlane(uint32_t(rg))) | (uint64_t(__builtin_amdgcn_readfirstlane(uint32_t(rg>>32)))<<32);
        const uint64_t du=uint64_t(__builtin_amdgcn_readfirstlane(uint32_t(ru))) | (uint64_t(__builtin_amdgcn_readfirstlane(uint32_t(ru>>32)))<<32);
        const int gb=int((dg>>32)&7),ub=int((du>>32)&7);
        if(gb==ub) {
            #define B(N) case N-1:paired_shared_row<N>(gate,up,dg,du,global_row,x,gv,uv);break
            switch(gb){B(1);B(2);B(3);B(4);B(5);B(6);B(7);B(8);}
            #undef B
        } else {
            float g[1]{},u[1]{};
            generic_dispatch<0,true,1,1,16>(gate,global_row,x,0,1,gate.input_columns,g);
            generic_dispatch<0,true,1,1,16>(up,global_row,x,0,1,up.input_columns,u);
            gv=g[0];uv=u[0];
        }
    }
    #pragma unroll
    for(int d=32;d;d>>=1){gv+=__shfl_down(gv,d,64);uv+=__shfl_down(uv,d,64);}
    if(lane==0 && row<rows) {
        if(gate.bias)gv+=gate.bias[global_row];if(up.bias)uv+=up.bias[global_row];
        float a=number<0>(rounded<0>(gv)),b=number<0>(rounded<0>(uv));
        if(limit>0){a=fminf(a,limit);b=fmaxf(-limit,fminf(b,limit));}
        output[size_t(pair)*rows+row]=rounded<0>(b*(a/(1.f+expf(-a))));
    }
}
struct TileStore {
    half* output;
    int valid;
    __device__ __forceinline__ void operator()(int j,half a,half b) {
        output[j]=j<valid?a:half{};
        output[j+1]=j+1<valid?b:half{};
    }
};

template<bool Columns,bool Uniform=false>
__device__ __forceinline__ void load_weight(ProjectionView m,half* tile,int expert,
    int rows,int n0,int k0,int tid,bool fast=false) {
    constexpr int BK=64,BN=64,Stride=72,V=8;
    if(m.columns==2)m.scales=static_cast<const half*>(m.scales)+size_t(expert)*m.stored_columns;
    for(int i=tid;i<BN*(BK/V);i+=256) {
        const int local_row=i/(BK/V),local_col=(i%(BK/V))*V;
        const int row=expert*rows+n0+local_row,col=k0+local_col;
        auto* destination=tile+local_row*Stride+local_col;
        if(n0+local_row<rows && col<m.input_columns) {
            TileStore store{destination,min(V,m.input_columns-col)};
            const auto descriptor=m.descriptors[row];
            if constexpr(Uniform) {
                if(fast) {
            mfq_tq_vector::visit_values<V,true,Columns,2,TileStore,false,2>(m.words,descriptor,m.scales,
                static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,store);
                } else {
            mfq_tq_vector::visit_values<V,true,Columns,0,TileStore,false,2>(m.words,descriptor,m.scales,
                static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,store);
                }
            } else {
            mfq_tq_vector::visit_values<V,true,Columns,0,TileStore,false,2>(m.words,descriptor,m.scales,
                static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,store);
            }
        } else *reinterpret_cast<uint4*>(destination)=uint4{};
    }
}

// Decode one weight tile in LDS, then share it across sixteen routed tokens.
// F32/BF16 activations use F32 MFMA and retain exact reconstructed FP16 weights.
template<int DT,bool Pair,int BM=16,bool Uniform=false>
__global__ __launch_bounds__(256) void grouped(ProjectionView weight,ProjectionView up,
    const S<DT>* input,S<DT>* output,int rows,TqMoeRoutePlan plan,bool routed,float limit) {
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    constexpr int BN=64,BK=64,Stride=72;
    using F4=float __attribute__((ext_vector_type(4)));
    using H4=_Float16 __attribute__((ext_vector_type(4)));
    using X=std::conditional_t<DT==0,half,float>;
    __shared__ half w[BN*Stride];
    __shared__ X x[BM*Stride];
    const int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
    const int ntiles=(rows+BN-1)/BN,total=plan.tiles[plan.expert_count]*ntiles;
    for(int task=blockIdx.x;task<total;task+=gridDim.x) {
        const int tile=task/ntiles,n0=(task%ntiles)*BN,expert=plan.experts[tile];
        const int first=plan.bounds[expert]+(tile-plan.tiles[expert])*BM;
        const int last=min(first+BM,plan.bounds[expert+1]);
        bool fast=false;
        if constexpr(Uniform) {
            if(n0+64<=rows) {
                const int check=expert*rows+n0+lane;
                const auto dg=__ldg(weight.descriptors+check);
                const auto du=Pair?__ldg(up.descriptors+check):dg;
                fast=__all(int((dg>>32)&7)==1 && int((du>>32)&7)==1);
            }
        }
        F4 acc[Pair?2:1][BM/16]{};
        for(int kb=0;kb<weight.input_columns;kb+=BK) {
            for(int i=tid;i<BM*BK;i+=256) {
                const int m=i/BK,k=i%BK,compact=first+m;
                const int pair=compact<last?plan.pairs[compact]:-1;
                const float value=pair>=0 && kb+k<weight.input_columns?
                    number<DT>(input[size_t(routed?pair:pair/plan.routes)*weight.input_columns+kb+k]):0.f;
                x[m*Stride+k]=static_cast<X>(value);
            }
            #pragma unroll
            for(int role=0;role<(Pair?2:1);++role) {
                const auto m=role?up:weight;
                if(m.columns)load_weight<true,Uniform>(m,w,expert,rows,n0,kb,tid,fast);
                else load_weight<false,Uniform>(m,w,expert,rows,n0,kb,tid,fast);
                __syncthreads();
                if constexpr(DT==0) {
                    #pragma unroll
                    for(int k=0;k<BK;k+=16) {
                        const auto a=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(w+(wave*16+(lane&15))*Stride+k+(lane>>4)*4));
                        #pragma unroll
                        for(int mt=0;mt<BM/16;++mt) {
                            const auto b=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(x+(mt*16+(lane&15))*Stride+k+(lane>>4)*4));
                            acc[role][mt]=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc[role][mt],0,0,0);
                        }
                    }
                } else {
                    #pragma unroll
                    for(int k=0;k<BK;k+=4) {
                        #pragma unroll
                        for(int mt=0;mt<BM/16;++mt)acc[role][mt]=__builtin_amdgcn_mfma_f32_16x16x4f32(
                            __half2float(w[(wave*16+(lane&15))*Stride+k+(lane>>4)]),
                            x[(mt*16+(lane&15))*Stride+k+(lane>>4)],acc[role][mt],0,0,0);
                    }
                }
                __syncthreads();
            }
        }
        #pragma unroll
        for(int mt=0;mt<BM/16;++mt) {
            const int compact=first+mt*16+(lane&15);
            if(compact<last) {
                const int pair=plan.pairs[compact];
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const int row=n0+wave*16+(lane>>4)*4+j;
                    if(row<rows) {
                        const int global_row=expert*rows+row;
                        float result=acc[0][mt][j]+(weight.bias?weight.bias[global_row]:0.f);
                        if constexpr(Pair) {
                            float gate=number<DT>(rounded<DT>(result));
                            float value_up=number<DT>(rounded<DT>(acc[1][mt][j]+(up.bias?up.bias[global_row]:0.f)));
                            if(limit>0){gate=fminf(gate,limit);value_up=fmaxf(-limit,fminf(value_up,limit));}
                            result=value_up*(gate/(1.f+expf(-gate)));
                        }
                        output[size_t(pair)*rows+row]=rounded<DT>(result);
                    }
                }
            }
        }
        __syncthreads();
    }
#endif
}


template<int BM>__global__ __launch_bounds__(512) void grouped_parallel_pair(
        ProjectionView g,ProjectionView u,const uint16_t* input,uint16_t* output,
        int rows,TqMoeRoutePlan plan,float limit) {
    constexpr int BN=64,BK=64,Stride=72;
    using F4=float __attribute__((ext_vector_type(4)));
    using H4=_Float16 __attribute__((ext_vector_type(4)));
    __shared__ half weights[2][BN*Stride];
    __shared__ half x[BM*Stride];
    const int full=threadIdx.x,tid=full&255,role=full>>8,lane=tid&63,wave=tid>>6;
    const auto m=role?u:g;
    const int ntiles=(rows+BN-1)/BN,total=plan.tiles[plan.expert_count]*ntiles;
    for(int task=blockIdx.x;task<total;task+=gridDim.x) {
        const int tile=task/ntiles,n0=(task%ntiles)*BN,expert=plan.experts[tile];
        const int first=plan.bounds[expert]+(tile-plan.tiles[expert])*BM,last=min(first+BM,plan.bounds[expert+1]);
        bool fast=false;
        if(n0+64<=rows)fast=__all(int((__ldg(m.descriptors+expert*rows+n0+lane)>>32)&7)==1);
        F4 acc[BM/16]{};
        for(int kb=0;kb<g.input_columns;kb+=BK) {
            for(int i=full;i<BM*BK;i+=512) {
                const int token=i/BK,k=i%BK,compact=first+token;
                const int pair=compact<last?plan.pairs[compact]:-1;
                x[token*Stride+k]=pair>=0 && kb+k<g.input_columns?__ushort_as_half(input[size_t(pair/plan.routes)*g.input_columns+kb+k]):half{};
            }
            if(m.columns)load_weight<true,true>(m,weights[role],expert,rows,n0,kb,tid,fast);
            else load_weight<false,true>(m,weights[role],expert,rows,n0,kb,tid,fast);
            __syncthreads();
            #pragma unroll
            for(int k=0;k<BK;k+=16) {
                const auto a=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(weights[role]+(wave*16+(lane&15))*Stride+k+(lane>>4)*4));
                #pragma unroll
                for(int mt=0;mt<BM/16;++mt) {
                    const auto b=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(x+(mt*16+(lane&15))*Stride+k+(lane>>4)*4));
                    acc[mt]=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc[mt],0,0,0);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for(int mt=0;mt<BM/16;++mt) {
            const int token=mt*16+(lane&15);
            #pragma unroll
            for(int j=0;j<4;++j) {
                const int row=wave*16+(lane>>4)*4+j;
                if(first+token<last && n0+row<rows)
                    weights[role][token*BN+row]=__float2half_rn(acc[mt][j]+(m.bias?m.bias[expert*rows+n0+row]:0.f));
            }
        }
        __syncthreads();
        for(int i=full;i<BM*BN;i+=512) {
            const int token=i/BN,row=i%BN;
            if(first+token<last && n0+row<rows) {
                const int pair=plan.pairs[first+token];
                float a=__half2float(weights[0][i]),b=__half2float(weights[1][i]);
                if(limit>0){a=fminf(a,limit);b=fmaxf(-limit,fminf(b,limit));}
                output[size_t(pair)*rows+n0+row]=rounded<0>(b*(a/(1.f+expf(-a))));
            }
        }
        __syncthreads();
    }
}

struct CachedTileStore {
    half* output;
    int valid;
    __device__ __forceinline__ void operator()(int j,half a,half b) {
        output[j]=j<valid?a:half{};
        output[j+1]=j+1<valid?b:half{};
    }
};

struct SwizzledCachedTileStore {
    half* output;
    int valid,column=0,mask=0;
    __device__ __forceinline__ void operator()(int j,half a,half b) {
        *reinterpret_cast<half2*>(output+((column+j)^mask))=
            __halves2half2(j<valid?a:half{},j+1<valid?b:half{});
    }
};

// Four reconstructed halves retain their original rounding and XOR layout.
struct PackedSwizzledTileStore {
    half* output;
    int valid,column=0,mask=0;
    uint32_t pending=0;
    __device__ __forceinline__ void operator()(int j,half a,half b) {
        const uint32_t packed=__builtin_bit_cast(uint32_t,__halves2half2(
            j<valid?a:half{},j+1<valid?b:half{}));
        if((j&2)==0)pending=packed;
        else *reinterpret_cast<uint2*>(output+((column+j-2)^mask))=make_uint2(pending,packed);
    }
};

// Only compressed words and packed scales are kept across K tiles.
// Explicit vector fields avoid private-memory arrays during prefetch.
using U4=mfq_tq_vector::U4;
struct PreparedTile {U4 first,second,columns;};
template<int Bits>__device__ __forceinline__ U4 prefetch_row(ProjectionView m,uint64_t descriptor,int col) {
    U4 output{};
    const int bits=Bits?Bits:int((descriptor>>32)&7)+1,ringwords=8*bits;
    const int minimum=(31-(col/8)%32)*(8*bits),first=minimum>>5;
    const auto* ring=m.words+uint32_t(descriptor)+(col>>8)*ringwords;
    #pragma unroll
    for(int j=0;j<(Bits?2:3);++j){int at=first+j;if(at>=ringwords)at-=ringwords;output[j]=__ldg(ring+at);}
    return output;
}
__device__ __forceinline__ void prefetch_tile(ProjectionView m,int expert,int kb,int tid,int choice,const uint64_t* descriptors,PreparedTile& tile) {
    const int col=kb+(tid%8)*8;
    const auto* scales=static_cast<const half*>(m.scales)+size_t(expert)*m.stored_columns+col;
    #pragma unroll
    for(int j=0;j<4;++j)tile.columns[j]=__ldg(reinterpret_cast<const uint32_t*>(scales)+j);
    if(choice==2){tile.first=prefetch_row<2>(m,descriptors[0],col);tile.second=prefetch_row<2>(m,descriptors[1],col);}
    else if(choice==3){tile.first=prefetch_row<3>(m,descriptors[0],col);tile.second=prefetch_row<3>(m,descriptors[1],col);}
    else{tile.first=prefetch_row<0>(m,descriptors[0],col);tile.second=prefetch_row<0>(m,descriptors[1],col);}
}

template<bool Columns,bool Uniform=false,bool ExpertColumns=false,bool Swizzle=false,bool Full=false,bool Prepared=false,bool Packed=false>
__device__ __forceinline__ void load_weight_cached(ProjectionView m,half* tile,int expert,
    int rows,int n0,int k0,int tid,int choice,const uint64_t* descriptors,const PreparedTile& prepared={}) {
    constexpr int BK=64,BN=64,Stride=Swizzle?64:72,V=8;
    using Store=std::conditional_t<Swizzle && Packed,PackedSwizzledTileStore,
        std::conditional_t<Swizzle,SwizzledCachedTileStore,CachedTileStore>>;
    if constexpr(ExpertColumns)m.scales=static_cast<const half*>(m.scales)+size_t(expert)*m.stored_columns;
    else if(m.columns==2)m.scales=static_cast<const half*>(m.scales)+size_t(expert)*m.stored_columns;
    #pragma unroll
    for(int i=tid;i<BN*(BK/V);i+=256) {
        const int local_row=i/(BK/V),local_col=(i%(BK/V))*V;
        const int row=expert*rows+n0+local_row,col=k0+local_col;
        auto* destination=tile+local_row*Stride+local_col;
        if(Full || (n0+local_row<rows && col<m.input_columns)) {
            Store store{Swizzle?tile+local_row*Stride:destination,Full?V:min(V,m.input_columns-col)};
            if constexpr(Swizzle){store.column=local_col;store.mask=(local_row&15)*4;}
            const auto descriptor=descriptors[i/256];
            U4 raw{};
            if constexpr(Prepared)raw=i<256?prepared.first:prepared.second;
            switch(choice) {
                case 2:mfq_tq_vector::visit_values<V,true,Columns,2,Store,false,2,Prepared>(m.words,descriptor,m.scales,static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,store,raw,prepared.columns);break;
                case 3:mfq_tq_vector::visit_values<V,true,Columns,3,Store,false,2,Prepared>(m.words,descriptor,m.scales,static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,store,raw,prepared.columns);break;
                default:mfq_tq_vector::visit_values<V,true,Columns,0,Store,false,2,Prepared>(m.words,descriptor,m.scales,static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,store,raw,prepared.columns);break;
            }
        } else {
            if constexpr(Swizzle) {
                Store store{tile+local_row*Stride,0,local_col,(local_row&15)*4};
                #pragma unroll
                for(int j=0;j<V;j+=2)store(j,half{},half{});
            } else *reinterpret_cast<uint4*>(destination)=uint4{};
        }
    }
}


template<int BM> __device__ __forceinline__ void load_vector_input(const uint16_t* input,half* x,int columns,int kb,TqMoeRoutePlan plan,int first,int last,bool routed) {
    #pragma unroll
    for(int i=threadIdx.x;i<BM*8;i+=blockDim.x) {
        const int token=i/8,k=(i%8)*8,compact=first+token;
        const int pair=compact<last?plan.pairs[compact]:-1;
        const uint4 value=pair>=0 && kb+k<columns?*reinterpret_cast<const uint4*>(input+size_t(routed?pair:pair/plan.routes)*columns+kb+k):uint4{};
        *reinterpret_cast<uint4*>(x+token*72+k)=value;
    }
}
template<int BM,int Threads,bool Skip>struct InputCache {
    static constexpr int Count=(BM*8+Threads-1)/Threads;
    const uint16_t* source[Count];
    half* output;
    int columns;
    __device__ __forceinline__ InputCache(const uint16_t* input,half* x,int width,TqMoeRoutePlan plan,int first,int last,bool routed):output(x),columns(width) {
        if constexpr(BM>=16) {
        #pragma unroll
        for(int j=0;j<Count;++j) {
            const int i=threadIdx.x+j*Threads,token=i/8,k=(i%8)*8,compact=first+token;
            const int pair=i<BM*8 && compact<last?plan.pairs[compact]:-1;
            source[j]=(columns&7)==0 && pair>=0?input+size_t(routed?pair:pair/plan.routes)*columns+k:nullptr;
            if constexpr(Skip)if(i<BM*8 && !source[j])*reinterpret_cast<uint4*>(output+token*72+k)=uint4{};
        }
        }
    }
    __device__ __forceinline__ void load(int kb) {
        #pragma unroll
        for(int j=0;j<Count;++j) {
            const int i=threadIdx.x+j*Threads,token=i/8,k=(i%8)*8;
            if(i<BM*8) {
                if constexpr(Skip) {if(!source[j])continue;}
                const uint4 value=source[j] && kb+k<columns?*reinterpret_cast<const uint4*>(source[j]+kb):uint4{};
                *reinterpret_cast<uint4*>(output+token*72+k)=value;
            }
        }
    }
};
// The PP input tile uses a four-half XOR layout for conflict-free MFMA reads.
// gfx942 can gather dwords straight into LDS while the compressed weight tile
// is reconstructed. Invalid padded token rows read a safe row and are discarded.
template<int BM,int Threads,bool Skip>struct SwizzledInputCache {
    static constexpr int Count=(BM*8+Threads-1)/Threads;
    const uint16_t* source[Count];
    half* output;
    int columns;
    __device__ __forceinline__ SwizzledInputCache(const uint16_t* input,half* x,int width,TqMoeRoutePlan plan,int first,int last,bool routed):output(x),columns(width) {
        if constexpr(BM>=16) {
        #pragma unroll
        for(int j=0;j<Count;++j) {
            const int i=threadIdx.x+j*Threads,token=i/8,k=(i%8)*8,compact=first+token;
            const int pair=i<BM*8 && compact<last?plan.pairs[compact]:-1;
            source[j]=(columns&7)==0 && pair>=0?input+size_t(routed?pair:pair/plan.routes)*columns+k:nullptr;

        }
        }
    }
    __device__ __forceinline__ void load(int kb) {
        #pragma unroll
        for(int j=0;j<Count;++j) {
            const int i=threadIdx.x+j*Threads,token=i/8,k=(i%8)*8;
            if(i<BM*8) {

                const uint4 value=source[j] && kb+k<columns?*reinterpret_cast<const uint4*>(source[j]+kb):uint4{};
                const int mask=(token&15)*4;
                *reinterpret_cast<uint2*>(output+token*64+(k^mask))=make_uint2(value.x,value.y);
                *reinterpret_cast<uint2*>(output+token*64+((k+4)^mask))=make_uint2(value.z,value.w);
            }
        }
    }
};
template<int BM,int Threads>struct DirectInputCache {
    static constexpr int Count=(BM*32+Threads-1)/Threads;
    using Resource=decltype(__builtin_amdgcn_make_buffer_rsrc((void*)nullptr,0,0,0));
    using LDS=uint32_t __attribute__((address_space(3)));
    Resource resource;
    uint32_t offsets[Count];
    half* output;
    __device__ __forceinline__ DirectInputCache(const uint16_t* input,half* x,int width,TqMoeRoutePlan plan,int first,int last,bool routed=false):output(x) {
        resource=__builtin_amdgcn_make_buffer_rsrc(const_cast<uint16_t*>(input),0,uint32_t(uint64_t(plan.tokens)*(routed?plan.routes:1)*width*2),0x00020000);
        #pragma unroll
        for(int j=0;j<Count;++j) {
            const int at=threadIdx.x+j*Threads,row=at/32,physical=(at%32)*2;
            const int col=physical^((row&15)*4),compact=first+row;
            const int pair=row<BM && compact<last?plan.pairs[compact]:-1;
            offsets[j]=pair>=0?uint32_t((size_t(routed?pair:pair/plan.routes)*width+col)*2):0;
        }
    }
    __device__ __forceinline__ void load(int kb) {
        const int wave=__builtin_amdgcn_readfirstlane(threadIdx.x>>6);
        #pragma unroll
        for(int j=0;j<Count;++j) {
            auto* destination=(LDS*)(output+(wave+j*(Threads/64))*128);
            __builtin_amdgcn_raw_ptr_buffer_load_lds(resource,destination,4,offsets[j]+kb*2,0,0,0);
        }
    }
    __device__ __forceinline__ void wait(){__builtin_amdgcn_s_waitcnt(0);}
};
template<int BM,bool Async=false,bool Full=false>__global__ __launch_bounds__(512) void grouped_parallel_pair_cached(
        ProjectionView g,ProjectionView u,const uint16_t* input,uint16_t* output,
        int rows,TqMoeRoutePlan plan,float limit) {
    constexpr int BN=64,BK=64,Stride=72;
#if defined(__gfx942__)
    constexpr bool Direct=Async;
#else
    constexpr bool Direct=false;
#endif
    using F4=float __attribute__((ext_vector_type(4)));
    using H4=_Float16 __attribute__((ext_vector_type(4)));
    __shared__ half weights[2][BN*(Direct?64:Stride)];
    __shared__ __align__(16) half x[BM*(Direct?64:Stride)];
    const int full=threadIdx.x,tid=full&255,role=full>>8,lane=tid&63,wave=tid>>6;
    const auto m=role?u:g;
    const int ntiles=(rows+BN-1)/BN,total=plan.tiles[plan.expert_count]*ntiles;
    for(int task=blockIdx.x;task<total;task+=gridDim.x) {
        const int tile=task/ntiles,n0=(task%ntiles)*BN,expert=plan.experts[tile];
        const int first=plan.bounds[expert]+(tile-plan.tiles[expert])*BM,last=min(first+BM,plan.bounds[expert+1]);
        uint64_t descriptors[2];
        #pragma unroll
        for(int i=0;i<2;++i) {
            const int row=n0+tid/8+i*32;
            descriptors[i]=(Full || row<rows)?__ldg(m.descriptors+expert*rows+row):0;
        }
        const int code0=int((descriptors[0]>>32)&7),code1=int((descriptors[1]>>32)&7),leader=__shfl(code0,0,64);
        const int choice=__all(code0==leader && code1==leader)?leader+1:0;
        using Cache=std::conditional_t<Direct,DirectInputCache<BM,512>,InputCache<BM,512,true>>;
        Cache input_cache(input,x,g.input_columns,plan,first,last,false);
        F4 acc[BM/16]{};
        for(int kb=0;kb<g.input_columns;kb+=BK) {
            if constexpr(Direct)input_cache.load(kb);
            else if((g.input_columns&7)==0)input_cache.load(kb);
            else {
            for(int i=full;i<BM*BK;i+=512) {
                const int token=i/BK,k=i%BK,compact=first+token;
                const int pair=compact<last?plan.pairs[compact]:-1;
                x[token*Stride+k]=pair>=0 && kb+k<g.input_columns?__ushort_as_half(input[size_t(pair/plan.routes)*g.input_columns+kb+k]):half{};
            }
            }
            // This cached entry is dispatched only for expert column-scale banks.
            load_weight_cached<true,true,true,Direct,Full>(m,weights[role],expert,rows,n0,kb,tid,choice,descriptors);
            if constexpr(Direct)input_cache.wait();
            __syncthreads();
            #pragma unroll
            for(int k=0;k<BK;k+=16) {
                const half* weight_source=Direct?
                    weights[role]+(wave*16+(lane&15))*64+((k+(lane>>4)*4)^((lane&15)*4)):
                    weights[role]+(wave*16+(lane&15))*Stride+k+(lane>>4)*4;
                const auto a=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(weight_source));
                #pragma unroll
                for(int mt=0;mt<BM/16;++mt) {
                    const half* source=Direct?
                        x+(mt*16+(lane&15))*64+((k+(lane>>4)*4)^((lane&15)*4)):
                        x+(mt*16+(lane&15))*Stride+k+(lane>>4)*4;
                    const auto b=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(source));
                    if(mt==0 || first+mt*16<last)acc[mt]=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc[mt],0,0,0);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for(int mt=0;mt<BM/16;++mt) {
            const int token=mt*16+(lane&15);
            #pragma unroll
            for(int j=0;j<4;++j) {
                const int row=wave*16+(lane>>4)*4+j;
                if(first+token<last && n0+row<rows)
                    weights[role][token*BN+row]=__float2half_rn(acc[mt][j]+(m.bias?m.bias[expert*rows+n0+row]:0.f));
            }
        }
        __syncthreads();
        for(int i=full;i<BM*BN;i+=512) {
            const int token=i/BN,row=i%BN;
            if(first+token<last && n0+row<rows) {
                const int pair=plan.pairs[first+token];
                float a=__half2float(weights[0][i]),b=__half2float(weights[1][i]);
                if(limit>0){a=fminf(a,limit);b=fmaxf(-limit,fminf(b,limit));}
                output[size_t(pair)*rows+n0+row]=rounded<0>(b*(a/(1.f+expf(-a))));
            }
        }
        __syncthreads();
    }
}


template<int BM,bool Async=false,bool Full=false,bool Pipeline=true>__global__ __launch_bounds__(512) void grouped_parallel_pair_prepared(
        ProjectionView g,ProjectionView u,const uint16_t* input,uint16_t* output,
        int rows,TqMoeRoutePlan plan,float limit) {
    static_assert(Async && Full,"Prepared projections require complete column-scale tiles");
    constexpr int BN=64,BK=64,Stride=72;
#if defined(__gfx942__)
    constexpr bool Direct=Async;
#else
    constexpr bool Direct=false;
#endif
    using F4=float __attribute__((ext_vector_type(4)));
    using H4=_Float16 __attribute__((ext_vector_type(4)));
    __shared__ half weights[2][BN*(Direct?64:Stride)];
    __shared__ __align__(16) half x[BM*(Direct?64:Stride)];
    const int full=threadIdx.x,tid=full&255,role=__builtin_amdgcn_readfirstlane(full>>8),lane=tid&63,wave=tid>>6;
    const auto m=role?u:g;
    const int ntiles=(rows+BN-1)/BN,total=plan.tiles[plan.expert_count]*ntiles;
    for(int task=blockIdx.x;task<total;task+=gridDim.x) {
        const int tile=task/ntiles,n0=(task%ntiles)*BN,expert=plan.experts[tile];
        const int first=plan.bounds[expert]+(tile-plan.tiles[expert])*BM,last=min(first+BM,plan.bounds[expert+1]);
        uint64_t descriptors[2];
        #pragma unroll
        for(int i=0;i<2;++i) {
            const int row=n0+tid/8+i*32;
            descriptors[i]=(Full || row<rows)?__ldg(m.descriptors+expert*rows+row):0;
        }
        const int code0=int((descriptors[0]>>32)&7),code1=int((descriptors[1]>>32)&7),leader=__shfl(code0,0,64);
        const int choice=__builtin_amdgcn_readfirstlane(__all(code0==leader && code1==leader)?leader+1:0);
        using Cache=std::conditional_t<Direct,DirectInputCache<BM,512>,InputCache<BM,512,true>>;
        Cache input_cache(input,x,g.input_columns,plan,first,last,false);
        PreparedTile prepared;
        if constexpr(Pipeline)prefetch_tile(m,expert,0,tid,choice,descriptors,prepared);
        F4 acc[BM/16]{};
        for(int kb=0;kb<g.input_columns;kb+=BK) {
            if constexpr(!Pipeline)prefetch_tile(m,expert,kb,tid,choice,descriptors,prepared);
            if constexpr(Direct)input_cache.load(kb);
            else if((g.input_columns&7)==0)input_cache.load(kb);
            else {
            for(int i=full;i<BM*BK;i+=512) {
                const int token=i/BK,k=i%BK,compact=first+token;
                const int pair=compact<last?plan.pairs[compact]:-1;
                x[token*Stride+k]=pair>=0 && kb+k<g.input_columns?__ushort_as_half(input[size_t(pair/plan.routes)*g.input_columns+kb+k]):half{};
            }
            }
            // This cached entry is dispatched only for expert column-scale banks.
            load_weight_cached<true,true,true,Direct,Full,true,BM==64 && Direct && Full>(m,weights[role],expert,rows,n0,kb,tid,choice,descriptors,prepared);
            if constexpr(Direct)input_cache.wait();
            __syncthreads();
            if constexpr(Pipeline)if(kb+BK<g.input_columns)prefetch_tile(m,expert,kb+BK,tid,choice,descriptors,prepared);
            #pragma unroll
            for(int k=0;k<BK;k+=16) {
                const half* weight_source=Direct?
                    weights[role]+(wave*16+(lane&15))*64+((k+(lane>>4)*4)^((lane&15)*4)):
                    weights[role]+(wave*16+(lane&15))*Stride+k+(lane>>4)*4;
                const auto a=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(weight_source));
                if constexpr(BM==64 && Direct && Full) {
                    // Load the four independent M fragments together so LDS reads
                    // can combine; each accumulator retains its original K order.
                    H4 b[4];
                    #pragma unroll
                    for(int mt=0;mt<4;++mt)b[mt]=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(
                        x+(mt*16+(lane&15))*64+((k+(lane>>4)*4)^((lane&15)*4))));
                    asm volatile(
                        "v_mfma_f32_16x16x16_f16 %0, %4, %5, %0\n\t"
                        "v_mfma_f32_16x16x16_f16 %1, %4, %6, %1\n\t"
                        "v_mfma_f32_16x16x16_f16 %2, %4, %7, %2\n\t"
                        "v_mfma_f32_16x16x16_f16 %3, %4, %8, %3"
                        : "+v"(acc[0]),"+v"(acc[1]),"+v"(acc[2]),"+v"(acc[3])
                        : "v"(a),"v"(b[0]),"v"(b[1]),"v"(b[2]),"v"(b[3]));
                } else {
                #pragma unroll
                for(int mt=0;mt<BM/16;++mt) {
                    const half* source=Direct?
                        x+(mt*16+(lane&15))*64+((k+(lane>>4)*4)^((lane&15)*4)):
                        x+(mt*16+(lane&15))*Stride+k+(lane>>4)*4;
                    const auto b=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(source));
                    if(mt==0 || first+mt*16<last)acc[mt]=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc[mt],0,0,0);
                    if constexpr(BM==64)asm volatile("" ::: "memory");
                }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for(int mt=0;mt<BM/16;++mt) {
            const int token=mt*16+(lane&15);
            if constexpr(Direct && Full) {
                const int row=wave*16+(lane>>4)*4;
                if(first+token<last) {
                    H4 packed;
                    #pragma unroll
                    for(int j=0;j<4;++j) {
                        const float value=acc[mt][j]+(m.bias?m.bias[expert*rows+n0+row+j]:0.f);
                        packed[j]=__builtin_bit_cast(_Float16,__float2half_rn(value));
                    }
                    const int store_row=row^((token&15)*4);
                    *reinterpret_cast<uint2*>(weights[role]+token*BN+store_row)=__builtin_bit_cast(uint2,packed);
                }
            } else {
            #pragma unroll
            for(int j=0;j<4;++j) {
                const int row=wave*16+(lane>>4)*4+j;
                if(first+token<last && n0+row<rows)
                    weights[role][token*BN+row]=__float2half_rn(acc[mt][j]+(m.bias?m.bias[expert*rows+n0+row]:0.f));
            }
            }
        }
        __syncthreads();
        for(int i=full;i<BM*BN;i+=512) {
            const int token=i/BN,row=i%BN;
            if(first+token<last && n0+row<rows) {
                const int pair=plan.pairs[first+token];
                const int physical=Direct && Full?token*BN+(row^((token&15)*4)):i;
                float a=__half2float(weights[0][physical]),b=__half2float(weights[1][physical]);
                if(limit>0){a=fminf(a,limit);b=fmaxf(-limit,fminf(b,limit));}
                output[size_t(pair)*rows+n0+row]=rounded<0>(b*(a/(1.f+expf(-a))));
            }
        }
        __syncthreads();
    }
}


template<bool Columns,bool Uniform=false,bool ExpertColumns=false,bool Swizzle=false,bool Full=false,bool Prepared=false,bool Packed=false>
__device__ __forceinline__ void load_weight_wide128(ProjectionView m,half* tile,int expert,
    int rows,int n0,int k0,int tid,int choice,const uint64_t* descriptors,const PreparedTile& prepared={}) {
    constexpr int BK=64,BN=64,Stride=Swizzle?64:72,V=8;
    using Store=std::conditional_t<Swizzle && Packed,PackedSwizzledTileStore,
        std::conditional_t<Swizzle,SwizzledCachedTileStore,CachedTileStore>>;
    if constexpr(ExpertColumns)m.scales=static_cast<const half*>(m.scales)+size_t(expert)*m.stored_columns;
    else if(m.columns==2)m.scales=static_cast<const half*>(m.scales)+size_t(expert)*m.stored_columns;
    #pragma unroll 1
    for(int i=tid;i<BN*(BK/V);i+=256) {
        const int local_row=i/(BK/V),local_col=(i%(BK/V))*V;
        const int row=expert*rows+n0+local_row,col=k0+local_col;
        auto* destination=tile+local_row*Stride+local_col;
        if(Full || (n0+local_row<rows && col<m.input_columns)) {
            Store store{Swizzle?tile+local_row*Stride:destination,Full?V:min(V,m.input_columns-col)};
            if constexpr(Swizzle){store.column=local_col;store.mask=(local_row&15)*4;}
            const auto descriptor=descriptors[i/256];
            U4 raw{};
            if constexpr(Prepared)raw=i<256?prepared.first:prepared.second;
            switch(choice) {
                case 2:mfq_tq_vector::visit_values<V,true,Columns,2,Store,false,2,Prepared>(m.words,descriptor,m.scales,static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,store,raw,prepared.columns);break;
                case 3:mfq_tq_vector::visit_values<V,true,Columns,3,Store,false,2,Prepared>(m.words,descriptor,m.scales,static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,store,raw,prepared.columns);break;
                default:mfq_tq_vector::visit_values<V,true,Columns,0,Store,false,2,Prepared>(m.words,descriptor,m.scales,static_cast<const half*>(m.relative),nullptr,nullptr,row,col,m.stored_columns,m.gslog,store,raw,prepared.columns);break;
            }
        } else {
            if constexpr(Swizzle) {
                Store store{tile+local_row*Stride,0,local_col,(local_row&15)*4};
                #pragma unroll
                for(int j=0;j<V;j+=2)store(j,half{},half{});
            } else *reinterpret_cast<uint4*>(destination)=uint4{};
        }
    }
}


template<int BM,bool Async=false,bool Full=false,bool Pipeline=true,bool Hybrid=false>__global__ __launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(4))) void grouped_pair_wide128(
        ProjectionView g,ProjectionView u,const uint16_t* input,uint16_t* output,
        int rows,TqMoeRoutePlan plan,float limit) {
    static_assert(Async && Full,"Prepared projections require complete column-scale tiles");
    constexpr int BN=64,BK=64,Stride=72;
#if defined(__gfx942__)
    constexpr bool Direct=Async;
#else
    constexpr bool Direct=false;
#endif
    using F4=float __attribute__((ext_vector_type(4)));
    using H4=_Float16 __attribute__((ext_vector_type(4)));
    __shared__ half weights[2][BN*(Direct?64:Stride)];
    __shared__ __align__(16) half x[BM*(Direct?64:Stride)];
    const int full=threadIdx.x,tid=full&255,role=__builtin_amdgcn_readfirstlane(full>>8),lane=tid&63,wave=tid>>6;
    const auto m=role?u:g;
    const int ntiles=(rows+BN-1)/BN,total=plan.tiles[plan.expert_count]*ntiles;
    for(int task=blockIdx.x;task<total;task+=gridDim.x) {
        const int tile=task/ntiles,n0=(task%ntiles)*BN,expert=plan.experts[tile];
        if constexpr(Hybrid){
            const int count=plan.bounds[expert+1]-plan.bounds[expert];
            const int small=(count+63)/64,wide=(count+95)/96;
            const bool use_wide=wide<small && wide*96<=small*64;
            if(use_wide!=(BM==96))continue;
        }
        const int first=plan.bounds[expert]+(tile-plan.tiles[expert])*BM,last=min(first+BM,plan.bounds[expert+1]);
        uint64_t descriptors[2];
        #pragma unroll
        for(int i=0;i<2;++i) {
            const int row=n0+tid/8+i*32;
            descriptors[i]=(Full || row<rows)?__ldg(m.descriptors+expert*rows+row):0;
        }
        const int code0=int((descriptors[0]>>32)&7),code1=int((descriptors[1]>>32)&7),leader=__shfl(code0,0,64);
        const int choice=__builtin_amdgcn_readfirstlane(__all(code0==leader && code1==leader)?leader+1:0);
        using Cache=std::conditional_t<Direct,DirectInputCache<BM,512>,InputCache<BM,512,true>>;
        Cache input_cache(input,x,g.input_columns,plan,first,last,false);
        PreparedTile prepared;
        if constexpr(Pipeline)prefetch_tile(m,expert,0,tid,choice,descriptors,prepared);
        F4 acc[BM/16]{};
        for(int kb=0;kb<g.input_columns;kb+=BK) {
            if constexpr(!Pipeline)prefetch_tile(m,expert,kb,tid,choice,descriptors,prepared);
            if constexpr(Direct)input_cache.load(kb);
            else if((g.input_columns&7)==0)input_cache.load(kb);
            else {
            for(int i=full;i<BM*BK;i+=512) {
                const int token=i/BK,k=i%BK,compact=first+token;
                const int pair=compact<last?plan.pairs[compact]:-1;
                x[token*Stride+k]=pair>=0 && kb+k<g.input_columns?__ushort_as_half(input[size_t(pair/plan.routes)*g.input_columns+kb+k]):half{};
            }
            }
            // This cached entry is dispatched only for expert column-scale banks.
            load_weight_wide128<true,true,true,Direct,Full,true,BM>=64 && Direct && Full>(m,weights[role],expert,rows,n0,kb,tid,choice,descriptors,prepared);
            if constexpr(Direct)input_cache.wait();
            __syncthreads();
            if constexpr(Pipeline)if(kb+BK<g.input_columns)prefetch_tile(m,expert,kb+BK,tid,choice,descriptors,prepared);
            #pragma unroll
            for(int k=0;k<BK;k+=16) {
                const half* weight_source=Direct?
                    weights[role]+(wave*16+(lane&15))*64+((k+(lane>>4)*4)^((lane&15)*4)):
                    weights[role]+(wave*16+(lane&15))*Stride+k+(lane>>4)*4;
                const auto a=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(weight_source));
                if constexpr(BM>=64 && Direct && Full) {
                    #pragma unroll
                    for(int group=0;group<BM/16;group+=4) {
                        H4 b[4];
                        #pragma unroll
                        for(int j=0;j<4;++j)b[j]=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(
                            x+((group+j)*16+(lane&15))*64+((k+(lane>>4)*4)^((lane&15)*4))));
                        #pragma unroll
                        for(int j=0;j<4;++j)acc[group+j]=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b[j],acc[group+j],0,0,0);
                        asm volatile("" ::: "memory");
                    }
                } else {
                #pragma unroll
                for(int mt=0;mt<BM/16;++mt) {
                    const half* source=Direct?
                        x+(mt*16+(lane&15))*64+((k+(lane>>4)*4)^((lane&15)*4)):
                        x+(mt*16+(lane&15))*Stride+k+(lane>>4)*4;
                    const auto b=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(source));
                    if(mt==0 || first+mt*16<last)acc[mt]=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc[mt],0,0,0);
                    if constexpr(BM>=64)asm volatile("" ::: "memory");
                }
                }
            }
            __syncthreads();
        }
        static_assert(BM==128 && Async && Full);
        // Reuse the64-row weight tiles for two successive output stripes.
        // Keep all128-row FP32 accumulators and the original K order.
        #pragma unroll
        for(int stripe=0;stripe<128;stripe+=64) {
            #pragma unroll
            for(int mt=0;mt<4;++mt) {
                const int token=stripe+mt*16+(lane&15),row=wave*16+(lane>>4)*4;
                if(first+token<last) {
                    H4 packed;
                    #pragma unroll
                    for(int j=0;j<4;++j)packed[j]=__builtin_bit_cast(_Float16,__float2half_rn(
                        acc[stripe/16+mt][j]+(m.bias?m.bias[expert*rows+n0+row+j]:0.f)));
                    const int store_row=row^((token&15)*4);
                    *reinterpret_cast<uint2*>(weights[role]+(token-stripe)*BN+store_row)=__builtin_bit_cast(uint2,packed);
                }
            }
            __syncthreads();
            for(int i=full;i<64*BN;i+=512) {
                const int token=stripe+i/BN,row=i%BN;
                if(first+token<last && n0+row<rows) {
                    const int pair=plan.pairs[first+token],physical=(token-stripe)*BN+(row^((token&15)*4));
                    float a=__half2float(weights[0][physical]),b=__half2float(weights[1][physical]);
                    if(limit>0){a=fminf(a,limit);b=fmaxf(-limit,fminf(b,limit));}
                    output[size_t(pair)*rows+n0+row]=rounded<0>(b*(a/(1.f+expf(-a))));
                }
            }
            __syncthreads();
        }
    }
}


template<int DT,bool Pair,int BM=16,bool Uniform=false,bool Async=false,bool Full=false,bool Prepared=false>
__global__ __launch_bounds__(256) void grouped_down_cached(ProjectionView weight,ProjectionView up,
    const S<DT>* input,S<DT>* output,int rows,TqMoeRoutePlan plan,bool routed,float limit) {
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    constexpr int BN=64,BK=64,Stride=72;
    static_assert(!Prepared || (DT==0 && !Pair && Async && Full));
    using F4=float __attribute__((ext_vector_type(4)));
    using H4=_Float16 __attribute__((ext_vector_type(4)));
    using X=std::conditional_t<DT==0,half,float>;
    #if defined(__gfx942__)
    constexpr bool SmallAsync=Async && BM==16;
#else
    constexpr bool SmallAsync=false;
#endif
    constexpr bool Swizzle=DT==0 && !Pair && (BM==64 || SmallAsync);
#if defined(__gfx942__)
    constexpr bool Direct=Swizzle && Async;
#else
    constexpr bool Direct=false;
#endif
    __shared__ half w[BN*(Swizzle?64:Stride)];
    __shared__ X x[BM*(Swizzle?64:Stride)];
    const int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
    const int ntiles=(rows+BN-1)/BN,total=plan.tiles[plan.expert_count]*ntiles;
    for(int task=blockIdx.x;task<total;task+=gridDim.x) {
        const int tile=task/ntiles,n0=(task%ntiles)*BN,expert=plan.experts[tile];
        const int first=plan.bounds[expert]+(tile-plan.tiles[expert])*BM;
        const int last=min(first+BM,plan.bounds[expert+1]);
        uint64_t descriptors[2];
        #pragma unroll
        for(int i=0;i<2;++i) {
            const int row=n0+tid/8+i*32;
            descriptors[i]=(Full || row<rows)?__ldg(weight.descriptors+expert*rows+row):0;
        }
        const int code0=int((descriptors[0]>>32)&7),code1=int((descriptors[1]>>32)&7),leader=__shfl(code0,0,64);
        const int selected=__all(code0==leader && code1==leader)?leader+1:0;
        const int choice=Prepared?__builtin_amdgcn_readfirstlane(selected):selected;
        using Cache=std::conditional_t<Direct,DirectInputCache<BM,256>,std::conditional_t<Swizzle,SwizzledInputCache<BM,256,true>,InputCache<BM,256,true>>>;
        Cache input_cache(reinterpret_cast<const uint16_t*>(input),reinterpret_cast<half*>(x),weight.input_columns,plan,first,last,routed);
        F4 acc[Pair?2:1][BM/16]{};
        for(int kb=0;kb<weight.input_columns;kb+=BK) {
            PreparedTile prepared;
            if constexpr(Prepared)prefetch_tile(weight,expert,kb,tid,choice,descriptors,prepared);
            if constexpr(DT==0 && (BM>=32 || Direct)) {
                if((weight.input_columns&7)==0)input_cache.load(kb);
                else {
            for(int i=tid;i<BM*BK;i+=256) {
                const int m=i/BK,k=i%BK,compact=first+m;
                const int pair=compact<last?plan.pairs[compact]:-1;
                const float value=pair>=0 && kb+k<weight.input_columns?
                    number<DT>(input[size_t(routed?pair:pair/plan.routes)*weight.input_columns+kb+k]):0.f;
                x[Swizzle?m*64+(k^((m&15)*4)):m*Stride+k]=static_cast<X>(value);
            }
                }
            } else {
            for(int i=tid;i<BM*BK;i+=256) {
                const int m=i/BK,k=i%BK,compact=first+m;
                const int pair=compact<last?plan.pairs[compact]:-1;
                const float value=pair>=0 && kb+k<weight.input_columns?
                    number<DT>(input[size_t(routed?pair:pair/plan.routes)*weight.input_columns+kb+k]):0.f;
                x[Swizzle?m*64+(k^((m&15)*4)):m*Stride+k]=static_cast<X>(value);
            }
            }
            #pragma unroll
            for(int role=0;role<(Pair?2:1);++role) {
                const auto m=role?up:weight;
                if constexpr(Prepared)load_weight_cached<true,Uniform,true,Swizzle,Full,true,DT==0 && !Pair && BM==64 && Direct && Full>(m,w,expert,rows,n0,kb,tid,choice,descriptors,prepared);
                else if constexpr(BM>=32 || Swizzle)load_weight_cached<true,Uniform,true,Swizzle,Full>(m,w,expert,rows,n0,kb,tid,choice,descriptors);
                else {
                if(m.columns)load_weight_cached<true,Uniform>(m,w,expert,rows,n0,kb,tid,choice,descriptors);
                else load_weight_cached<false,Uniform>(m,w,expert,rows,n0,kb,tid,choice,descriptors);
                }
                if constexpr(Direct)input_cache.wait();
                __syncthreads();
                if constexpr(DT==0) {
                    #pragma unroll
                    for(int k=0;k<BK;k+=16) {
                        const half* source=Swizzle?
                            w+(wave*16+(lane&15))*64+((k+(lane>>4)*4)^((lane&15)*4)):
                            w+(wave*16+(lane&15))*Stride+k+(lane>>4)*4;
                        const auto a=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(source));
                        if constexpr(!Pair && BM==64 && Direct && Full && Prepared) {
                            H4 b[4];
                            #pragma unroll
                            for(int mt=0;mt<4;++mt)b[mt]=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(
                                x+(mt*16+(lane&15))*64+((k+(lane>>4)*4)^((lane&15)*4))));
                            asm volatile(
                                "v_mfma_f32_16x16x16_f16 %0, %4, %5, %0\n\t"
                                "v_mfma_f32_16x16x16_f16 %1, %4, %6, %1\n\t"
                                "v_mfma_f32_16x16x16_f16 %2, %4, %7, %2\n\t"
                                "v_mfma_f32_16x16x16_f16 %3, %4, %8, %3"
                                : "+v"(acc[role][0]),"+v"(acc[role][1]),"+v"(acc[role][2]),"+v"(acc[role][3])
                                : "v"(a),"v"(b[0]),"v"(b[1]),"v"(b[2]),"v"(b[3]));
                        } else {
                        #pragma unroll
                        for(int mt=0;mt<BM/16;++mt) {
                            const half* input_source=Swizzle?
                                x+(mt*16+(lane&15))*64+((k+(lane>>4)*4)^((lane&15)*4)):
                                x+(mt*16+(lane&15))*Stride+k+(lane>>4)*4;
                            const auto b=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(input_source));
                            acc[role][mt]=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc[role][mt],0,0,0);
                        }
                        }
                    }
                } else {
                    #pragma unroll
                    for(int k=0;k<BK;k+=4) {
                        #pragma unroll
                        for(int mt=0;mt<BM/16;++mt)acc[role][mt]=__builtin_amdgcn_mfma_f32_16x16x4f32(
                            __half2float(w[(wave*16+(lane&15))*Stride+k+(lane>>4)]),
                            x[(mt*16+(lane&15))*Stride+k+(lane>>4)],acc[role][mt],0,0,0);
                    }
                }
                __syncthreads();
            }
        }
        #pragma unroll
        for(int mt=0;mt<BM/16;++mt) {
            const int compact=first+mt*16+(lane&15);
            if(compact<last) {
                const int pair=plan.pairs[compact];
                if constexpr(DT==0 && !Pair && Direct && Full) {
                    const int row=n0+wave*16+(lane>>4)*4;
                    H4 packed;
                    #pragma unroll
                    for(int j=0;j<4;++j) {
                        const float value=acc[0][mt][j]+(weight.bias?weight.bias[expert*rows+row+j]:0.f);
                        packed[j]=__builtin_bit_cast(_Float16,__float2half_rn(value));
                    }
                    *reinterpret_cast<uint2*>(output+size_t(pair)*rows+row)=__builtin_bit_cast(uint2,packed);
                } else {
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const int row=n0+wave*16+(lane>>4)*4+j;
                    if(row<rows) {
                        const int global_row=expert*rows+row;
                        float result=acc[0][mt][j]+(weight.bias?weight.bias[global_row]:0.f);
                        if constexpr(Pair) {
                            float gate=number<DT>(rounded<DT>(result));
                            float value_up=number<DT>(rounded<DT>(acc[1][mt][j]+(up.bias?up.bias[global_row]:0.f)));
                            if(limit>0){gate=fminf(gate,limit);value_up=fmaxf(-limit,fminf(value_up,limit));}
                            result=value_up*(gate/(1.f+expf(-gate)));
                        }
                        output[size_t(pair)*rows+row]=rounded<DT>(result);
                    }
                }
                }
            }
        }
        __syncthreads();
    }
#endif
}



template<int BM,int Threads,bool Skip>struct InputCacheSmall {
    static constexpr int Count=(BM*8+Threads-1)/Threads;
    const uint16_t* source[Count];
    half* output;
    int columns;
    __device__ __forceinline__ InputCacheSmall(const uint16_t* input,half* x,int width,TqMoeRoutePlan plan,int first,int last,bool routed):output(x),columns(width) {
        if constexpr(BM>=16) {
        #pragma unroll
        for(int j=0;j<Count;++j) {
            const int i=threadIdx.x+j*Threads,token=i/8,k=(i%8)*8,compact=first+token;
            const int pair=i<BM*8 && compact<last?plan.pairs[compact]:-1;
            source[j]=(columns&7)==0 && pair>=0?input+size_t(routed?pair:pair/plan.routes)*columns+k:nullptr;
            if constexpr(Skip)if(i<BM*8 && !source[j])*reinterpret_cast<uint4*>(output+token*72+k)=uint4{};
        }
        }
    }
    __device__ __forceinline__ void load(int kb) {
        #pragma unroll
        for(int j=0;j<Count;++j) {
            const int i=threadIdx.x+j*Threads,token=i/8,k=(i%8)*8;
            if(i<BM*8) {
                if constexpr(Skip) {if(!source[j])continue;}
                const uint4 value=source[j] && kb+k<columns?*reinterpret_cast<const uint4*>(source[j]+kb):uint4{};
                *reinterpret_cast<uint4*>(output+token*72+k)=value;
            }
        }
    }
};
template<int DT,bool Pair,int BM=16,bool Uniform=false>
__global__ __launch_bounds__(256) void grouped_pair_small_vector(ProjectionView weight,ProjectionView up,
    const S<DT>* input,S<DT>* output,int rows,TqMoeRoutePlan plan,bool routed,float limit) {
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    constexpr int BN=64,BK=64,Stride=72;
    using F4=float __attribute__((ext_vector_type(4)));
    using H4=_Float16 __attribute__((ext_vector_type(4)));
    using X=std::conditional_t<DT==0,half,float>;
    __shared__ half w[BN*Stride];
    __shared__ X x[BM*Stride];
    const int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
    const int ntiles=(rows+BN-1)/BN,total=plan.tiles[plan.expert_count]*ntiles;
    for(int task=blockIdx.x;task<total;task+=gridDim.x) {
        const int tile=task/ntiles,n0=(task%ntiles)*BN,expert=plan.experts[tile];
        const int first=plan.bounds[expert]+(tile-plan.tiles[expert])*BM;
        const int last=min(first+BM,plan.bounds[expert+1]);
        uint64_t descriptors[2][2];int choices[2];
        #pragma unroll
        for(int role=0;role<2;++role) {
            const auto m=role?up:weight;
            #pragma unroll
            for(int i=0;i<2;++i) {
                const int row=n0+tid/8+i*32;
                descriptors[role][i]=row<rows?__ldg(m.descriptors+expert*rows+row):0;
            }
            const int a=int((descriptors[role][0]>>32)&7),b=int((descriptors[role][1]>>32)&7),leader=__shfl(a,0,64);
            choices[role]=__all(a==leader && b==leader)?leader+1:0;
        }
        InputCacheSmall<BM,256,true> input_cache(reinterpret_cast<const uint16_t*>(input),reinterpret_cast<half*>(x),weight.input_columns,plan,first,last,routed);
        F4 acc[Pair?2:1][BM/16]{};
        for(int kb=0;kb<weight.input_columns;kb+=BK) {
            if constexpr(DT==0 && BM>=16) {
                if((weight.input_columns&7)==0)input_cache.load(kb);
                else {
            for(int i=tid;i<BM*BK;i+=256) {
                const int m=i/BK,k=i%BK,compact=first+m;
                const int pair=compact<last?plan.pairs[compact]:-1;
                const float value=pair>=0 && kb+k<weight.input_columns?
                    number<DT>(input[size_t(routed?pair:pair/plan.routes)*weight.input_columns+kb+k]):0.f;
                x[m*Stride+k]=static_cast<X>(value);
            }
                }
            } else {
            for(int i=tid;i<BM*BK;i+=256) {
                const int m=i/BK,k=i%BK,compact=first+m;
                const int pair=compact<last?plan.pairs[compact]:-1;
                const float value=pair>=0 && kb+k<weight.input_columns?
                    number<DT>(input[size_t(routed?pair:pair/plan.routes)*weight.input_columns+kb+k]):0.f;
                x[m*Stride+k]=static_cast<X>(value);
            }
            }
            #pragma unroll
            for(int role=0;role<(Pair?2:1);++role) {
                const auto m=role?up:weight;
                if(m.columns)load_weight_cached<true,Uniform>(m,w,expert,rows,n0,kb,tid,choices[role],descriptors[role]);
                else load_weight_cached<false,Uniform>(m,w,expert,rows,n0,kb,tid,choices[role],descriptors[role]);
                __syncthreads();
                if constexpr(DT==0) {
                    #pragma unroll
                    for(int k=0;k<BK;k+=16) {
                        const auto a=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(w+(wave*16+(lane&15))*Stride+k+(lane>>4)*4));
                        #pragma unroll
                        for(int mt=0;mt<BM/16;++mt) {
                            const auto b=__builtin_bit_cast(H4,*reinterpret_cast<const uint2*>(x+(mt*16+(lane&15))*Stride+k+(lane>>4)*4));
                            acc[role][mt]=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc[role][mt],0,0,0);
                        }
                    }
                } else {
                    #pragma unroll
                    for(int k=0;k<BK;k+=4) {
                        #pragma unroll
                        for(int mt=0;mt<BM/16;++mt)acc[role][mt]=__builtin_amdgcn_mfma_f32_16x16x4f32(
                            __half2float(w[(wave*16+(lane&15))*Stride+k+(lane>>4)]),
                            x[(mt*16+(lane&15))*Stride+k+(lane>>4)],acc[role][mt],0,0,0);
                    }
                }
                __syncthreads();
            }
        }
        #pragma unroll
        for(int mt=0;mt<BM/16;++mt) {
            const int compact=first+mt*16+(lane&15);
            if(compact<last) {
                const int pair=plan.pairs[compact];
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const int row=n0+wave*16+(lane>>4)*4+j;
                    if(row<rows) {
                        const int global_row=expert*rows+row;
                        float result=acc[0][mt][j]+(weight.bias?weight.bias[global_row]:0.f);
                        if constexpr(Pair) {
                            float gate=number<DT>(rounded<DT>(result));
                            float value_up=number<DT>(rounded<DT>(acc[1][mt][j]+(up.bias?up.bias[global_row]:0.f)));
                            if(limit>0){gate=fminf(gate,limit);value_up=fmaxf(-limit,fminf(value_up,limit));}
                            result=value_up*(gate/(1.f+expf(-gate)));
                        }
                        output[size_t(pair)*rows+row]=rounded<DT>(result);
                    }
                }
            }
        }
        __syncthreads();
    }
#endif
}



template<int DT>__global__ void reduce(const S<DT>* pairs,const float* weights,S<DT>* output,
    int tokens,int routes,int rows) {
    const int64_t total=int64_t(tokens)*rows;
    for(int64_t at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;at<total;at+=int64_t(blockDim.x)*gridDim.x) {
        const int64_t token=at/rows,col=at%rows;float sum=0.f;
        for(int r=0;r<routes;++r)sum=__fadd_rn(sum,__fmul_rn(number<DT>(pairs[(token*routes+r)*rows+col]),weights[token*routes+r]));
        output[at]=rounded<DT>(sum);
    }
}

// Retain the original route sum order while loading/storing sixteen F16 columns.
__global__ void reduce_vector16(const half* pairs,const float* weights,half* output,int tokens,int rows){
 constexpr int V=16;
 using Packed=uint16_t __attribute__((ext_vector_type(V)));
 const int vectors=rows/V;const int64_t total=int64_t(tokens)*vectors;
 for(int64_t at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;at<total;at+=int64_t(blockDim.x)*gridDim.x){
  const int64_t token=at/vectors,col=(at%vectors)*V;float sums[V]{};
  #pragma unroll 1
  for(int r=0;r<10;++r){
   const Packed values=*reinterpret_cast<const Packed*>(pairs+(token*10+r)*rows+col);
   const float coefficient=__ldg(weights+token*10+r);
   #pragma unroll
   for(int j=0;j<V;++j)sums[j]=__fadd_rn(sums[j],__fmul_rn(__half2float(__ushort_as_half(values[j])),coefficient));
  }
  Packed result;
  #pragma unroll
  for(int j=0;j<V;++j)result[j]=__half_as_ushort(__float2half_rn(sums[j]));
  *reinterpret_cast<Packed*>(output+token*rows+col)=result;
 }
}

// Split only experts whose128-row tiles reduce task count without padding
// more rows than the original64-row map. Down retains the original map.
__global__ void wide128_plans(TqMoeRoutePlan original,TqMoeRoutePlan small,TqMoeRoutePlan wide){
 const int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
 const int count=original.bounds[tid+1]-original.bounds[tid];
 const int t64=(count+63)/64,t128=(count+127)/128;
 const bool use=t128<t64&&t128*128<=t64*64;
 const int own0=use?0:t64,own1=use?t128:0;
 int p0=own0,p1=own1;
 #pragma unroll
 for(int step=1;step<64;step*=2){int a=__shfl_up(p0,step,64),b=__shfl_up(p1,step,64);if(lane>=step){p0+=a;p1+=b;}}
 __shared__ int totals[8][2];if(lane==63){totals[wave][0]=p0;totals[wave][1]=p1;}__syncthreads();
 for(int previous=0;previous<wave;++previous){p0+=totals[previous][0];p1+=totals[previous][1];}
 const_cast<int32_t*>(small.tiles)[tid+1]=p0;const_cast<int32_t*>(wide.tiles)[tid+1]=p1;
 if(!tid){const_cast<int32_t*>(small.tiles)[0]=0;const_cast<int32_t*>(wide.tiles)[0]=0;}
 for(int i=p0-own0;i<p0;++i)const_cast<int32_t*>(small.experts)[i]=tid;
 for(int i=p1-own1;i<p1;++i)const_cast<int32_t*>(wide.experts)[i]=tid;
}

template<int DT>struct EmbeddingStore {
    S<DT>* output;int valid;
    __device__ __forceinline__ void operator()(int j,float a,float b) {
        if(j<valid)output[j]=rounded<DT>(a);
        if(j+1<valid)output[j+1]=rounded<DT>(b);
    }
};
template<int DT,bool Separate>
__global__ void embedding(ProjectionView weight,const ProjectionView* views,const int64_t* ids,
    S<DT>* output,int count,int columns) {
    const int col=(blockIdx.x*blockDim.x+threadIdx.x)*8;
    if(col>=columns)return;
    for(int i=blockIdx.y;i<count;i+=gridDim.y) {
        const auto m=Separate?views[i]:weight;
        const int64_t row=Separate?0:(ids?ids[i]:i);
        auto* target=output+int64_t(i)*columns+col;
        const int valid=min(8,columns-col);
        if(row>=0 && row<m.rows) {
            EmbeddingStore<DT> store{target,valid};
            if(m.columns)mfq_tq_vector::visit_values<8,true,true,0>(m.words,m.descriptors[row],m.scales,
                static_cast<const half*>(m.relative),nullptr,nullptr,int(row),col,m.stored_columns,m.gslog,store);
            else mfq_tq_vector::visit_values<8,true,false,0>(m.words,m.descriptors[row],m.scales,
                static_cast<const half*>(m.relative),nullptr,nullptr,int(row),col,m.stored_columns,m.gslog,store);
        } else for(int j=0;j<valid;++j)target[j]=S<DT>{};
    }
}

template<int D,bool Pair>void launch_decode(ProjectionView m,ProjectionView up,const void* x,void* y,
    int rows,TqMoeRoutePlan plan,bool routed,float limit,hipStream_t stream) {
    if constexpr(D==0 && Pair) {
        static const bool pair_shared=[] {
            const char* value=std::getenv("MFQ_HIP_TQ_PAIR_SHARED");return !value || value[0]!='0';
        }();
        if(pair_shared && plan.tokens==1 && !routed && rows==2304 && m.input_columns==5120 &&
                m.columns==2 && up.columns==2 && (reinterpret_cast<uintptr_t>(x)&15)==0) {
            decode_pair_shared<<<dim3((rows+3)/4,plan.routes),256,0,stream>>>(m,up,
                static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),rows,plan,limit);
            static const bool log=std::getenv("MFQ_HIP_TQ_PAIR_SHARED_LOG")!=nullptr;
            static std::atomic_flag logged=ATOMIC_FLAG_INIT;
            if(log && !logged.test_and_set(std::memory_order_relaxed))
                std::fprintf(stderr,"native_tq_pair_shared rows=2304 width=5120 wave=64 vector=16\n");
            return;
        }
        // Wider vectors reduce decode-loop iterations on measured expert shapes.
        if(plan.tokens==1 && ((rows==640 && m.input_columns==2560) ||
                             (rows==2304 && m.input_columns==5120))) {
            decode<D,Pair,16><<<dim3((rows+1)/2,plan.routes),256,0,stream>>>(m,up,
                static_cast<const S<D>*>(x),static_cast<S<D>*>(y),rows,plan,routed,limit);
            return;
        }
    }
    decode<D,Pair,(D==0 && !Pair?16:8)><<<dim3((rows+(Pair?2:4)-1)/(Pair?2:4),
        plan.tokens*plan.routes),256,0,stream>>>(m,up,static_cast<const S<D>*>(x),
        static_cast<S<D>*>(y),rows,plan,routed,limit);
}

template<bool Pair>int launch(ProjectionView m,ProjectionView up,const void* x,void* y,
    int rows,TqMoeRoutePlan plan,int dtype,bool routed,float limit,hipStream_t stream) {
    if(!x || !y || x==y || !plan.ids || rows<1 || plan.tokens<1 || plan.routes<1 ||
        plan.expert_count<1 || int64_t(rows)*plan.expert_count!=m.rows || m.input_columns<1 ||
        dtype<0 || dtype>2 || !std::isfinite(limit) || limit<0 ||
        (Pair && (up.rows!=m.rows || up.input_columns!=m.input_columns)))return int(hipErrorInvalidValue);
    static const bool cached=[] {const char* value=std::getenv("MFQ_HIP_TQ_PREFILL_DESCRIPTORS");return !value || value[0]!='0';}();
    static const bool cached_log=std::getenv("MFQ_HIP_TQ_PREFILL_DESCRIPTORS_LOG")!=nullptr;
    static std::atomic_flag cached_logged=ATOMIC_FLAG_INIT;
    if constexpr(Pair) {
        if(cached && dtype==0 && plan.pairs && plan.tile_rows==16 && m.columns==2 && up.columns==2) {
            const bool direct=(m.input_columns&63)==0 && uint64_t(plan.tokens)*m.input_columns*2<=UINT32_MAX;
            if(direct)
                {
                    if((rows&63)==0)
                        grouped_parallel_pair_prepared<16,true,true,false><<<4096,512,0,stream>>>(m,up,static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),rows,plan,limit);
                    else
                        grouped_parallel_pair_cached<16,true><<<4096,512,0,stream>>>(m,up,static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),rows,plan,limit);
                }
            else
                grouped_pair_small_vector<0,true,16,true><<<4096,256,0,stream>>>(m,up,static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),rows,plan,false,limit);
            static std::atomic_flag logged=ATOMIC_FLAG_INIT;
            if(cached_log && !logged.test_and_set(std::memory_order_relaxed))
                std::fprintf(stderr,"native_tq_small_pair parallel_roles=%d async_eligible=%d threads=%d full_tiles=%d prepared=%d pipeline=0\n",direct?2:1,int(direct),direct?512:256,int(direct && (rows&63)==0),int(direct && (rows&63)==0));
            return int(hipGetLastError());
        }
        static const bool parallel=[] {const char* value=std::getenv("MFQ_HIP_TQ_GROUPED_PAIR_PARALLEL");return !value || value[0]!='0';}();
        if(parallel && dtype==0 && plan.pairs && plan.tile_rows==64 && m.columns==2 && up.columns==2) {
            if(cached) {
                if((m.input_columns&63)==0 && uint64_t(plan.tokens)*m.input_columns*2<=UINT32_MAX)
                    {
                    if((rows&63)==0)
                        {
                            grouped_parallel_pair_prepared<64,true,true,true><<<4096,512,0,stream>>>(m,up,static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),rows,plan,limit);
                            static std::atomic_flag prepared_logged=ATOMIC_FLAG_INIT;
                            if(cached_log && !prepared_logged.test_and_set(std::memory_order_relaxed))
                                std::fprintf(stderr,"native_tq_raw_prefetch bm=64 pipeline=1 compressed=1\n");
                        }
                    else
                        grouped_parallel_pair_cached<64,true><<<4096,512,0,stream>>>(m,up,static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),rows,plan,limit);
                }
                else
                    grouped_parallel_pair_cached<64><<<4096,512,0,stream>>>(m,up,static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),rows,plan,limit);
                if(cached_log && !cached_logged.test_and_set(std::memory_order_relaxed))
                    std::fprintf(stderr,"native_tq_prefill_descriptors cached_rows=2 bits=2,3\n");
            } else grouped_parallel_pair<64><<<4096,512,0,stream>>>(m,up,static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),rows,plan,limit);
            static const bool log=std::getenv("MFQ_HIP_TQ_GROUPED_PAIR_PARALLEL_LOG")!=nullptr;
            static std::atomic_flag logged=ATOMIC_FLAG_INIT;
            if(log && !logged.test_and_set(std::memory_order_relaxed))
                std::fprintf(stderr,"native_tq_grouped_parallel_pair bm=64 threads=512\n");
            return int(hipGetLastError());
        }
    }
    if constexpr(!Pair) {
        if(cached && dtype==0 && plan.pairs && m.columns==2) {
            // Full is selected only after the input-column and complete-row checks.
            #define CACHED_FULL(M,A,F) grouped_down_cached<0,false,M,true,A,F,A&&F><<<(A&&F?8192:4096),256,0,stream>>>(m,up,static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),rows,plan,routed,limit)
            #define CACHED(M,A) do {if(A && (rows&63)==0){CACHED_FULL(M,A,true);}else{CACHED_FULL(M,A,false);}}while(false)
            if(plan.tile_rows==64){
                if((m.input_columns&63)==0 && uint64_t(plan.tokens)*(routed?plan.routes:1)*m.input_columns*2<=UINT32_MAX){CACHED(64,true);}
                else{CACHED(64,false);}
                static std::atomic_flag swizzle_logged=ATOMIC_FLAG_INIT;
                if(cached_log && !swizzle_logged.test_and_set(std::memory_order_relaxed))
                    std::fprintf(stderr,"native_tq_down_swizzle bm=64 granule=4 input_swizzle=1 async_eligible=1\n");
            }else if(plan.tile_rows==32){CACHED(32,false);}
            else if((m.input_columns&63)==0 && uint64_t(plan.tokens)*(routed?plan.routes:1)*m.input_columns*2<=UINT32_MAX){
                CACHED(16,true);
                static std::atomic_flag small_logged=ATOMIC_FLAG_INIT;
                if(cached_log && !small_logged.test_and_set(std::memory_order_relaxed))
                    std::fprintf(stderr,"native_tq_small_down bm=16 input_swizzle=1 async_eligible=1\n");
            }else{CACHED(16,false);}
            #undef CACHED
            #undef CACHED_FULL
            if(cached_log && !cached_logged.test_and_set(std::memory_order_relaxed))
                std::fprintf(stderr,"native_tq_prefill_descriptors cached_rows=2 bits=2,3\n");
            return int(hipGetLastError());
        }
    }
    static const bool uniform=[] {const char* value=std::getenv("MFQ_HIP_TQ_GROUPED_BLOCK2");return !value || value[0]!='0';}();
    #define GROUP(D,M,U) grouped<D,Pair,M,U><<<4096,256,0,stream>>>(m,up,static_cast<const S<D>*>(x),static_cast<S<D>*>(y),rows,plan,routed,limit)
    #define TILES(D,U) if(plan.tile_rows==64){GROUP(D,64,U);}else if(plan.tile_rows==32){GROUP(D,32,U);}else{GROUP(D,16,U);}
    #define RUN(D) \
     if(plan.pairs) { \
      if constexpr(D==0) {if(uniform){TILES(D,true);}else{TILES(D,false);}} \
      else {TILES(D,false);} \
     } else launch_decode<D,Pair>(m,up,x,y,rows,plan,routed,limit,stream)
    if(dtype==0){RUN(0);}else if(dtype==1){RUN(1);}else{RUN(2);}
    #undef RUN
    #undef TILES
    #undef GROUP
    if(plan.pairs && dtype==0 && uniform) {
        static const bool log=std::getenv("MFQ_HIP_TQ_GROUPED_BLOCK2_LOG")!=nullptr;
        static std::atomic_flag logged=ATOMIC_FLAG_INIT;
        if(log && !logged.test_and_set(std::memory_order_relaxed))
            std::fprintf(stderr,"native_tq_grouped_block2 uniform_rows=64 bits=2\n");
    }
    return int(hipGetLastError());
}
}
extern "C" int mfq_tq_moe_projection(ProjectionView m,const void* x,void* y,int rows,
    TqMoeRoutePlan plan,int dtype,bool routed,hipStream_t stream) {
    return mfq_tq_moe::launch<false>(m,{},x,y,rows,plan,dtype,routed,0.f,stream);
}
extern "C" int mfq_tq_moe_swiglu(ProjectionView gate,ProjectionView up,const void* x,void* y,int rows,
    TqMoeRoutePlan plan,int dtype,float limit,hipStream_t stream) {
    return mfq_tq_moe::launch<true>(gate,up,x,y,rows,plan,dtype,false,limit,stream);
}
namespace {
bool tq_reduce_vector_supported() {
    int device=-1;
    if(hipGetDevice(&device)!=hipSuccess)return false;
    static thread_local int cached_device=-1;
    static thread_local bool supported=false;
    if(cached_device!=device) {
        hipDeviceProp_t properties{};
        supported=hipGetDeviceProperties(&properties,device)==hipSuccess &&
            std::strncmp(properties.gcnArchName,"gfx942",6)==0;
        cached_device=device;
    }
    return supported;
}
}
extern "C" int mfq_tq_moe_reduce(const void* pairs,const float* weights,void* output,int tokens,
    int routes,int rows,int dtype,hipStream_t stream) {
    if(!pairs || !weights || !output || tokens<1 || routes<1 || rows<1 || dtype<0 || dtype>2)return int(hipErrorInvalidValue);
    if(dtype==0 && tokens>=1024 && (tokens==2048 || tokens==4096) &&
       routes==10 && rows==2560 &&
       ((reinterpret_cast<uintptr_t>(pairs)|reinterpret_cast<uintptr_t>(output))&31u)==0 &&
       tq_reduce_vector_supported()) {
        const int blocks=(int64_t(tokens)*(rows/16)+255)/256;
        mfq_tq_moe::reduce_vector16<<<blocks,256,0,stream>>>(
            static_cast<const half*>(pairs),weights,static_cast<half*>(output),tokens,rows);
        return int(hipGetLastError());
    }
    const int blocks=int(std::min<int64_t>(65535,(int64_t(tokens)*rows+255)/256));
    #define RUN(D) mfq_tq_moe::reduce<D><<<blocks,256,0,stream>>>(static_cast<const mfq_tq_cuda_exact::S<D>*>(pairs),weights,static_cast<mfq_tq_cuda_exact::S<D>*>(output),tokens,routes,rows)
    if(dtype==0){RUN(0);}else if(dtype==1){RUN(1);}else{RUN(2);}
    #undef RUN
    return int(hipGetLastError());
}
namespace {
__global__ void tq_peer_join(const half* a,const half* b,half* output,int64_t elements) {
    const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<elements)output[index]=__float2half_rn(__half2float(a[index])+__half2float(b[index]));
}
}
extern "C" int mfq_tq_moe_peer_join(const void* a,const void* b,void* output,int64_t elements,hipStream_t stream) {
    if(!a || !b || !output || elements<1)return int(hipErrorInvalidValue);
    tq_peer_join<<<(elements+255)/256,256,0,stream>>>(static_cast<const half*>(a),static_cast<const half*>(b),static_cast<half*>(output),elements);
    return int(hipGetLastError());
}
namespace {
template<bool Separate>int launch_embedding(ProjectionView weight,const ProjectionView* views,const int64_t* ids,
    int count,int columns,void* output,int dtype,hipStream_t stream) {
    if(!output || count<1 || columns<1 || dtype<0 || dtype>2 || (Separate && !views))return int(hipErrorInvalidValue);
    const dim3 grid((int64_t(columns)+2047)/2048,std::min(count,65535));
    #define RUN(D) mfq_tq_moe::embedding<D,Separate><<<grid,256,0,stream>>>(weight,views,ids,static_cast<mfq_tq_cuda_exact::S<D>*>(output),count,columns)
    if(dtype==0){RUN(0);}else if(dtype==1){RUN(1);}else{RUN(2);}
    #undef RUN
    return int(hipGetLastError());
}
}
extern "C" int mfq_tq_embedding_hip(ProjectionView weight,const int64_t* ids,int count,void* output,int dtype,hipStream_t stream) {
    return launch_embedding<false>(weight,nullptr,ids,count,weight.input_columns,output,dtype,stream);
}
extern "C" int mfq_tq_embedding_rows_hip(const ProjectionView* views,int count,int columns,void* output,int dtype,hipStream_t stream) {
    return launch_embedding<true>({},views,nullptr,count,columns,output,dtype,stream);
}

extern "C" uint64_t mfq_tq_moe_wide128_workspace(ProjectionView g,ProjectionView u,int rows,
 TqMoeRoutePlan plan,int dtype,float limit){
 static const bool enabled=[] {
  const char*a=std::getenv("MFQ_HIP_TQ_PREFILL_DESCRIPTORS"),*b=std::getenv("MFQ_HIP_TQ_GROUPED_PAIR_PARALLEL");
  return (!a||a[0]!='0')&&(!b||b[0]!='0');
 }();
 if(!enabled||dtype!=0||limit!=0.f||rows!=640||plan.tokens!=4096||plan.routes!=10||plan.expert_count!=512||
  plan.tile_rows!=64||plan.expert_offset||!plan.pairs||!plan.bounds||!plan.tiles||!plan.experts||
  g.columns!=2||u.columns!=2||g.input_columns!=2560||u.input_columns!=2560||g.rows!=512*640||u.rows!=g.rows)return 0;
 int device=0;if(hipGetDevice(&device)!=hipSuccess||!mfq_tq_prefill_f32_supported(device))return 0;
 return uint64_t(2)*(plan.expert_count+1+plan.tokens*plan.routes)*sizeof(int32_t);
}
extern "C" int mfq_tq_moe_swiglu_wide128(ProjectionView g,ProjectionView u,const void*input,void*output,int rows,
 TqMoeRoutePlan original,int dtype,float limit,void*workspace,uint64_t bytes,hipStream_t stream){
 const auto required=mfq_tq_moe_wide128_workspace(g,u,rows,original,dtype,limit);
 if(!required||!workspace||bytes<required||!input||!output||input==output)return int(hipErrorInvalidValue);
 auto* p=static_cast<int32_t*>(workspace);auto small=original,wide=original;
 small.tiles=p;p+=original.expert_count+1;small.experts=p;p+=original.tokens*original.routes;
 wide.tiles=p;p+=original.expert_count+1;wide.experts=p;wide.tile_rows=128;
 mfq_tq_moe::wide128_plans<<<1,512,0,stream>>>(original,small,wide);
 mfq_tq_moe::grouped_parallel_pair_prepared<64,true,true,true><<<4096,512,0,stream>>>(g,u,static_cast<const uint16_t*>(input),static_cast<uint16_t*>(output),rows,small,limit);
 mfq_tq_moe::grouped_pair_wide128<128,true,true,true,false><<<4096,512,0,stream>>>(g,u,static_cast<const uint16_t*>(input),static_cast<uint16_t*>(output),rows,wide,limit);
 return int(hipGetLastError());
}
