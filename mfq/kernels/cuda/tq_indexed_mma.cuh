#pragma once
#include "tq_ops.h"
#include <cuda_fp16.h>
#include <cstdint>

template<bool Columns>
__device__ __forceinline__ uint32_t tq_mma_pair(TrellisProjectionView m,uint64_t d,int row,int col,float scale) {
    if(row>=m.rows || col>=m.input_columns)return 0;
    const int bits=int((d>>32)&7)+1,words=8*bits;
    const int pos=(254-(col&255))*bits,first=pos>>5;
    const auto* ring=m.words+uint32_t(d)+(col>>8)*words;
    const uint32_t lo=__ldg(ring+first),hi=__ldg(ring+(first+1==words?0:first+1));
    const uint32_t window=__funnelshift_r(lo,hi,pos&31);
    const uint32_t a=__dp4a(((window>>bits)&65535)*0x83dcd12du,0x01010101u,0x6400u);
    const uint32_t b=__dp4a((window&65535)*0x83dcd12du,0x01010101u,0x6400u);
    const half2 gaussian=__hfma2(__halves2half2(__ushort_as_half(uint16_t(a)),__ushort_as_half(uint16_t(b))),
        __half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931)));
    if constexpr(!Columns) {
        // Shared row/subgroup scales are applied after the group dot, as in
        // the short SIMT kernels. Keep Gaussian values in their FP16 bits.
        return uint32_t(__half_as_ushort(__low2half(gaussian)))|
            (uint32_t(__half_as_ushort(__high2half(gaussian)))<<16);
    }
    float s0=scale,s1=scale;
    if constexpr(Columns) {
        const auto s=__ldg(reinterpret_cast<const uint32_t*>(static_cast<const half*>(m.scales)+col));
        s0=__fmul_rn(scale,__half2float(__ushort_as_half(uint16_t(s))));
        s1=__fmul_rn(scale,__half2float(__ushort_as_half(uint16_t(s>>16))));
    }
    const float2 z=__half22float2(gaussian);
    return uint32_t(__half_as_ushort(__float2half_rn(__fmul_rn(z.x,s0))))|
        (col+1<m.input_columns?uint32_t(__half_as_ushort(__float2half_rn(__fmul_rn(z.y,s1)))):0u)<<16;
}
__device__ __forceinline__ uint32_t tq_mma_input(const uint16_t* p,int col,int width) {
    if(col+1<width && (reinterpret_cast<uintptr_t>(p+col)&3)==0)return __ldg(reinterpret_cast<const uint32_t*>(p+col));
    uint32_t v=col<width?__ldg(p+col):0;
    if(col+1<width)v|=uint32_t(__ldg(p+col+1))<<16;
    return v;
}
template<bool Columns,class Job>
__device__ __forceinline__ void tq_mma_grouped_loop(TrellisProjectionView m,
        const Job* job,const uint16_t* x,int tile,int lane,int row0,int row1,int part,
        uint64_t d0,uint64_t d1,float (&dot)[4],int first_k,int last_k) {
    const int count=min(tile,job->count);
    const float anchor0=__half2float(__ushort_as_half(uint16_t(d0>>40)));
    const float anchor1=__half2float(__ushort_as_half(uint16_t(d1>>40)));
    const int span=Columns?m.input_columns:(1<<m.gslog);
    const size_t groups=Columns?0:(size_t(m.stored_columns)+span-1)/span;
    for(int64_t group=int64_t(first_k/span)*span;group<last_k;group+=span) {
        const int64_t last=min(int64_t(last_k),group+span);float scale0=anchor0,scale1=anchor1;
        if constexpr(!Columns) {
            const auto load=[&](int row,float anchor) {
                if(row>=m.rows)return 0.f;
                const size_t index=size_t(row)*groups+(group>>m.gslog);
                const uint8_t code=__ldg(static_cast<const uint8_t*>(m.scales)+(index>>1));
                return __fmul_rn(anchor,__half2float(__ldg(static_cast<const half*>(m.relative)+((code>>((index&1)*4))&15))));
            };
            scale0=load(row0,anchor0);scale1=load(row1,anchor1);
        }
    float groupdot[4]{};
    float (&acc)[4]=Columns?dot:groupdot;
    for(int64_t at=max(int64_t(first_k),group);at<last;at+=16) {
        const int base=int(at);
        uint32_t a[4]={tq_mma_pair<Columns>(m,d0,row0,base+part,scale0),tq_mma_pair<Columns>(m,d1,row1,base+part,scale1),
            tq_mma_pair<Columns>(m,d0,row0,base+part+8,scale0),tq_mma_pair<Columns>(m,d1,row1,base+part+8,scale1)};
        uint32_t b[2]{};const int token=lane>>2;
        if(token<count) {
            const auto* input=x+job->input_offsets[token];
            b[0]=tq_mma_input(input,base+part,m.input_columns);b[1]=tq_mma_input(input,base+part+8,m.input_columns);
        }
#if __CUDA_ARCH__ >= 800
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#endif
    }
    if constexpr(!Columns) {
        dot[0]=__fmaf_rn(scale0,groupdot[0],dot[0]);
        dot[1]=__fmaf_rn(scale0,groupdot[1],dot[1]);
        dot[2]=__fmaf_rn(scale1,groupdot[2],dot[2]);
        dot[3]=__fmaf_rn(scale1,groupdot[3],dot[3]);
    }
    }
}
__global__ __launch_bounds__(128)
void tq_indexed_mma_kernel(const TrellisIndexedWideProjectionJob* jobs,const uint16_t* x,uint16_t* output,int tile) {
    const auto* job=jobs+blockIdx.y;const auto m=job->weight;
#if __CUDA_ARCH__ < 800
    // m16n8k16 FP16 MMA requires SM80. Explicit wide API calls on older
    // targets remain correct; the native planner retains its four-row tiles.
    const int lane=threadIdx.x&31,first=blockIdx.x*64+(threadIdx.x/32)*16;
    for(int r=0;r<16;++r) {
        const int row=first+r;if(row>=m.rows)continue;
        float value[8]{};
        const int count=min(tile,job->count);
        if(m.columns)multi_dispatch<8,true,true>(m,row,x,0,count,0,value,job->input_offsets);
        else multi_dispatch<8,false,true>(m,row,x,0,count,0,value,job->input_offsets);
        #pragma unroll
        for(int t=0;t<8;++t) {
            #pragma unroll
            for(int step=16;step;step/=2)value[t]+=__shfl_down_sync(0xffffffff,value[t],step);
            if(lane==0 && t<count) {
                if(m.bias)value[t]+=__ldg(m.bias+row);
                output[job->output_offsets[t]+row]=__half_as_ushort(__float2half_rn(value[t]));
            }
        }
    }
#else
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const int first=blockIdx.x*64+warp*16,row0=first+(lane>>2),row1=row0+8,part=(lane&3)*2;
    const uint64_t d0=row0<m.rows?__ldg(m.descriptors+row0):0,d1=row1<m.rows?__ldg(m.descriptors+row1):0;
    const int count=min(tile,job->count);float dot[4]{};
    if(m.columns)tq_mma_grouped_loop<true>(m,job,x,tile,lane,row0,row1,part,d0,d1,dot,0,m.input_columns);
    else tq_mma_grouped_loop<false>(m,job,x,tile,lane,row0,row1,part,d0,d1,dot,0,m.input_columns);
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int row=j<2?row0:row1,token=part+(j&1);
        if(row<m.rows && token<count) {
            float v=dot[j];if(m.bias)v+=__ldg(m.bias+row);
            output[job->output_offsets[token]+row]=__half_as_ushort(__float2half_rn(v));
        }
    }
#endif
}

template<bool Columns>
__device__ __forceinline__ float tq_mma_row_scale(TrellisProjectionView m,uint64_t d,int row,int col) {
    if(row>=m.rows)return 0.f;
    const float anchor=__half2float(__ushort_as_half(uint16_t(d>>40)));
    if constexpr(Columns)return anchor;
    const size_t groups=(size_t(m.stored_columns)+(1<<m.gslog)-1)>>m.gslog;
    const size_t index=size_t(row)*groups+(col>>m.gslog);
    const uint8_t code=__ldg(static_cast<const uint8_t*>(m.scales)+(index>>1));
    return __fmul_rn(anchor,__half2float(__ldg(static_cast<const half*>(m.relative)+((code>>((index&1)*4))&15))));
}

template<bool GateColumns,bool UpColumns>
__device__ __forceinline__ void tq_mma_pair_loop(const TrellisIndexedWidePairJob* job,const uint16_t* x,
        int tile,int lane,int row,int part,uint64_t gd,uint64_t ud,float (&dot)[4],int first,int last) {
    const auto g=job->weight,u=job->second;const int count=min(tile,job->count);
    for(int group=first;group<last;) {
        // Gate and Up may use different subgroup sizes or scale modes.
        const int64_t gend=GateColumns?last:((int64_t(group)>>g.gslog)+1)<<g.gslog;
        const int64_t uend=UpColumns?last:((int64_t(group)>>u.gslog)+1)<<u.gslog;
        const int end=int(min(int64_t(last),min(gend,uend)));
        const float gs=tq_mma_row_scale<GateColumns>(g,gd,row,group),us=tq_mma_row_scale<UpColumns>(u,ud,row,group);
        float groupdot[4]{};
        for(int base=group;base<end;base+=16) {
            // MMA rows0..7 are Gate; rows8..15 are Up. One lane retains both
            // results, sharing input fragments and the activation epilogue.
            uint32_t a[4]={tq_mma_pair<GateColumns>(g,gd,row,base+part,gs),tq_mma_pair<UpColumns>(u,ud,row,base+part,us),
                tq_mma_pair<GateColumns>(g,gd,row,base+part+8,gs),tq_mma_pair<UpColumns>(u,ud,row,base+part+8,us)};
            uint32_t b[2]{};const int token=lane>>2;
            if(token<count) {
                const auto* input=x+job->input_offsets[token];
                b[0]=tq_mma_input(input,base+part,g.input_columns);b[1]=tq_mma_input(input,base+part+8,g.input_columns);
            }
#if __CUDA_ARCH__ >= 800
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(groupdot[0]),"+f"(groupdot[1]),"+f"(groupdot[2]),"+f"(groupdot[3])
                : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#endif
        }
        dot[0]=__fmaf_rn(GateColumns?1.f:gs,groupdot[0],dot[0]);
        dot[1]=__fmaf_rn(GateColumns?1.f:gs,groupdot[1],dot[1]);
        dot[2]=__fmaf_rn(UpColumns?1.f:us,groupdot[2],dot[2]);
        dot[3]=__fmaf_rn(UpColumns?1.f:us,groupdot[3],dot[3]);
        group=end;
    }
}

__global__ __launch_bounds__(256)
void tq_indexed_wide_swiglu_kernel(const TrellisIndexedWidePairJob* jobs,
        const uint16_t* x,uint16_t* output,int tile,float limit) {
    const auto* job=jobs+blockIdx.y;
#if __CUDA_ARCH__ < 800
    // Explicit calls remain correct on older targets. The native planner uses
    // its existing <=4-row jobs when m16n8k16 FP16 MMA is unavailable.
    const int lane=threadIdx.x&31,row=blockIdx.x*8+threadIdx.x/32;
    if(row>=job->weight.rows)return;
    float g[8]{},u[8]{};const int count=min(tile,job->count);
    if(job->weight.columns)multi_dispatch<8,true,true>(job->weight,row,x,0,count,0,g,job->input_offsets);
    else multi_dispatch<8,false,true>(job->weight,row,x,0,count,0,g,job->input_offsets);
    if(job->second.columns)multi_dispatch<8,true,true>(job->second,row,x,0,count,0,u,job->input_offsets);
    else multi_dispatch<8,false,true>(job->second,row,x,0,count,0,u,job->input_offsets);
    #pragma unroll
    for(int t=0;t<8;++t) {
        #pragma unroll
        for(int step=16;step;step/=2){g[t]+=__shfl_down_sync(0xffffffff,g[t],step);u[t]+=__shfl_down_sync(0xffffffff,u[t],step);}
        if(lane==0 && t<count) {
            if(job->weight.bias)g[t]+=__ldg(job->weight.bias+row);
            if(job->second.bias)u[t]+=__ldg(job->second.bias+row);
            float a=__half2float(__float2half_rn(g[t])),b=__half2float(__float2half_rn(u[t]));
            if(limit>0){a=fminf(a,limit);b=fmaxf(-limit,fminf(b,limit));}
            output[job->output_offsets[t]+row]=__half_as_ushort(__float2half_rn(b*(a/(1.f+expf(-a)))));
        }
    }
#else
    constexpr int W=8;
    const int lane=threadIdx.x&31,warp=threadIdx.x/32,row=blockIdx.x*8+(lane>>2),part=(lane&3)*2;
    const uint64_t gd=row<job->weight.rows?__ldg(job->weight.descriptors+row):0,
        ud=row<job->second.rows?__ldg(job->second.descriptors+row):0;
    // Split only at complete MMA K fragments, including short/tail matrices.
    const int64_t steps=(int64_t(job->weight.input_columns)+15)/16;
    const int first_k=int((steps*warp/W)*16);
    const int last_k=int(min(int64_t(job->weight.input_columns),(steps*(warp+1)/W)*16));
    float dot[4]{};
    if(job->weight.columns) {
        if(job->second.columns)tq_mma_pair_loop<true,true>(job,x,tile,lane,row,part,gd,ud,dot,first_k,last_k);
        else tq_mma_pair_loop<true,false>(job,x,tile,lane,row,part,gd,ud,dot,first_k,last_k);
    }else if(job->second.columns)tq_mma_pair_loop<false,true>(job,x,tile,lane,row,part,gd,ud,dot,first_k,last_k);
    else tq_mma_pair_loop<false,false>(job,x,tile,lane,row,part,gd,ud,dot,first_k,last_k);
    __shared__ float partial[W][32][4];
    #pragma unroll
    for(int j=0;j<4;++j)partial[warp][lane][j]=dot[j];
    __syncthreads();
    if(warp)return;
    const int count=min(tile,job->count);
    #pragma unroll
    for(int j=0;j<2;++j) {
        const int token=part+j;
        if(row<job->weight.rows && token<count) {
            float g=partial[0][lane][j],u=partial[0][lane][j+2];
            #pragma unroll
            for(int w=1;w<W;++w){g+=partial[w][lane][j];u+=partial[w][lane][j+2];}
            if(job->weight.bias)g+=__ldg(job->weight.bias+row);
            if(job->second.bias)u+=__ldg(job->second.bias+row);
            g=__half2float(__float2half_rn(g));u=__half2float(__float2half_rn(u));
            if(limit>0){g=fminf(g,limit);u=fmaxf(-limit,fminf(u,limit));}
            output[job->output_offsets[token]+row]=__half_as_ushort(__float2half_rn(u*(g/(1.f+expf(-g)))));
        }
    }
#endif
}
extern "C" int mfq_tq_indexed_wide_swiglu(const TrellisIndexedWidePairJob* jobs,
        const void* x,void* y,int count,int rows,int dtype,int tile,float limit,cudaStream_t stream) {
    if(!jobs || !x || !y || x==y || count<1 || rows<1 || dtype!=0 || tile<1 || tile>8 ||
       !std::isfinite(limit) || limit<0)return -1;
    for(int64_t first=0;first<count;first+=65535) {
        tq_indexed_wide_swiglu_kernel<<<dim3((uint64_t(rows)+7)/8,min(int64_t(65535),int64_t(count)-first)),256,0,stream>>>(
            jobs+first,static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),tile,limit);
        const auto error=cudaGetLastError();if(error)return int(error);
    }
    return 0;
}
extern "C" int mfq_tq_indexed_wide(const TrellisIndexedWideProjectionJob* jobs,
        const void* x,void* y,int count,int rows,int dtype,int tile,cudaStream_t stream) {
    if(!jobs || !x || !y || x==y || count<1 || rows<1 || dtype!=0 || tile<5 || tile>8)return -1;
    for(int64_t first=0;first<count;first+=65535) {
        tq_indexed_mma_kernel<<<dim3((uint64_t(rows)+63)/64,min(int64_t(65535),int64_t(count)-first)),128,0,stream>>>(
            jobs+first,static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y),tile);
        const auto error=cudaGetLastError();if(error)return int(error);
    }
    return 0;
}
