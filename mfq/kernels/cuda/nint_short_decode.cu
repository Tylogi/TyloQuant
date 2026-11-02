#include "nint_short_decode.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <string>
#include <type_traits>
#include "../../../cpp_runtime/backends/cuda/kernels/kernel_dispatch_trace.h"

namespace {
#include "nint_whole_group.cuh"
#include "nint_whole_warp_rows.cuh"
#include "nint_row_bounds.cuh"

// Small contraction matrices: one original group per lane, one output per
// warp. Keep the dense kernel's single XOR tree and reuse the packed words.
template<int GS,int Bits,int Rows,bool Narrow>
__global__ void __launch_bounds__(128) nint_fixed_small_kernel(NintWholeArgs a) {
    const int row=int(blockIdx.x)*4+threadIdx.y,lane=threadIdx.x;
    if(row>=a.rows)return;
    using Offset=std::conditional_t<Narrow,uint32_t,uint64_t>;
    using Packed=NintWholeGroup<GS,Bits,false,Offset>;
    float acc[Rows]{};
    if(lane<a.groups) {
        const int group=lane;
        const Packed packed(a.weights,Offset(a.offsets[row])+Offset(group*GS)*Bits,Offset(a.bytes),a.group_load);
        int dots[Rows]{},sums[Rows]{};
        const auto consume=[&](auto constant) {
            constexpr int chunk=decltype(constant)::value;
            const int w=packed.template codes<chunk>();
            #pragma unroll
            for(int m=0;m<Rows;++m) {
                const int x=*reinterpret_cast<const int*>(a.activation+size_t(m)*a.groups*GS+group*GS+chunk*4);
                dots[m]=__dp4a(w,x,dots[m]);sums[m]=__dp4a(0x01010101,x,sums[m]);
            }
        };
        consume(std::integral_constant<int,0>{});consume(std::integral_constant<int,1>{});
        consume(std::integral_constant<int,2>{});consume(std::integral_constant<int,3>{});
        consume(std::integral_constant<int,4>{});consume(std::integral_constant<int,5>{});
        if constexpr(GS==28)consume(std::integral_constant<int,6>{});
        const float scale=a.outer_scale[row]*float(a.scale[size_t(row)*a.groups+group]);
        const float minimum=a.outer_minimum[row]*float(a.minimum[size_t(row)*a.groups+group]);
        #pragma unroll
        for(int m=0;m<Rows;++m)acc[m]+=a.activation_scale[size_t(m)*a.groups+group]*
            (scale*float(dots[m])-minimum*float(sums[m]));
    }
    #pragma unroll
    for(int m=0;m<Rows;++m) {
        #pragma unroll
        for(int offset=16;offset;offset>>=1)acc[m]+=__shfl_xor_sync(0xffffffffu,acc[m],offset);
        if(lane==0)a.output[size_t(m)*a.rows+row]=__float2half_rn(acc[m]);
    }
}

// Immutable, uniform NINT5 GS28 input projections. The loader proves this
// before capture; general and mutable row metadata never use these entries.
template<int Rows,bool CachedSum,int RowsPerBlock=1,bool Interleaved=false,bool NarrowOffsets=false,bool ExactRows=false>
__global__ void __launch_bounds__(96*RowsPerBlock,(RowsPerBlock==1 && Interleaved && Rows>=3)?14:(RowsPerBlock==2 && Rows<=2)?8:1) nint_fixed_five_kernel(NintWholeArgs a) {
    const int row=int(blockIdx.x)*RowsPerBlock+(RowsPerBlock==1?0:threadIdx.y/3);
    using Packed=NintWholeGroup<28,5,false,std::conditional_t<NarrowOffsets,uint32_t,uint64_t>>;
    nint_whole_row<28,5,Rows,3,false,CachedSum,RowsPerBlock,Interleaved,Packed,ExactRows>(a,uint64_t(a.offsets[row]));
}
template<int Rows,bool CachedSum,bool Interleaved=false,bool NarrowOffsets=false,bool ExactRows=false,bool FixedGroups=false,bool RowProof=false>
__global__ void __launch_bounds__(128) nint_fixed_five_warp_rows_kernel(NintWholeArgs a) {
    const int row=int(blockIdx.x)*4+threadIdx.y;
    using Offset=std::conditional_t<NarrowOffsets,uint32_t,uint64_t>;
    using Packed=NintWholeGroup<28,5,false,Offset>;
    if constexpr(FixedGroups)a.groups=92;
    else if(row>=a.rows)return;
    if constexpr(RowProof) {
        a.group_load=nint_row_reads_fit<28,5,Offset>(a,row);
        nint_whole_warp_row<28,5,Rows,CachedSum,Interleaved,NintRowBoundGroup<28,5,Offset>,ExactRows>(a,row);
        return;
    }
    nint_whole_warp_row<28,5,Rows,CachedSum,Interleaved,Packed,ExactRows>(a,row);
}

template<int GS,int Bits,int Rows,bool CachedSum,bool ExactRows=false,bool NarrowOffsets=false,bool RowProof=false>
__global__ void __launch_bounds__(128,Rows>=3?(GS==28?10:12):1) nint_fixed_output_kernel(NintWholeArgs a) {
    using Offset=std::conditional_t<NarrowOffsets,uint32_t,uint64_t>;
            using Packed=std::conditional_t<RowProof,NintRowBoundGroup<GS,Bits,Offset>,NintWholeGroup<GS,Bits,false,Offset>>;
    if constexpr(RowProof)a.group_load=nint_row_reads_fit<GS,Bits,std::conditional_t<NarrowOffsets,uint32_t,uint64_t>>(a,int(blockIdx.x));
    nint_whole_row<GS,Bits,Rows,4,false,CachedSum,1,false,Packed,ExactRows>(a,uint64_t(a.offsets[blockIdx.x]));
}

// Two output rows share four warps. Each pair preserves the old four
// independent 32-group subtrees, combining (p0+p2)+(p1+p3) in that order.
template<int GS,int Bits,int Rows,bool CachedSum,bool ExactRows=false,bool NarrowOffsets=false,bool RowProof=false>
__global__ void __launch_bounds__(128,Rows>=4?8:1) nint_fixed_two_warp_output_kernel(NintWholeArgs a) {
    const int local_row=int(threadIdx.y)>>1,warp=int(threadIdx.y)&1,lane=threadIdx.x;
    const int row=int(blockIdx.x)*2+local_row;
    if constexpr(RowProof)a.group_load=nint_row_reads_fit<GS,Bits,std::conditional_t<NarrowOffsets,uint32_t,uint64_t>>(a,row);
    const auto row_bit=uint64_t(a.offsets[row]);
    const float outer_scale=a.outer_scale[row],outer_minimum=a.outer_minimum[row];
    float total[Rows]{};
    #pragma unroll 1
    for(int part=0;part<2;++part) {
        float acc[Rows]{};
        for(int group=(warp+part*2)*32+lane;group<a.groups;group+=128) {
            using Offset=std::conditional_t<NarrowOffsets,uint32_t,uint64_t>;
            using Packed=std::conditional_t<RowProof,NintRowBoundGroup<GS,Bits,Offset>,NintWholeGroup<GS,Bits,false,Offset>>;
            const Packed packed(a.weights,row_bit+uint64_t(group*GS)*Bits,a.bytes,a.group_load);
            int dots[Rows]{},sums[CachedSum?1:Rows]{};
            const auto consume=[&](auto constant) {
                constexpr int chunk=decltype(constant)::value;
                const int column=group*GS+chunk*4;
                const int qw=packed.template codes<chunk>();
                #pragma unroll
                for(int m=0;m<Rows;++m)if(ExactRows || (CachedSum && Rows!=4 && Rows!=8) || m<a.activation_rows) {
                    const int qx=*reinterpret_cast<const int*>(a.activation+size_t(m)*a.groups*GS+column);
                    if constexpr(!CachedSum)sums[m]=__dp4a(0x01010101,qx,sums[m]);
                    dots[m]=__dp4a(qw,qx,dots[m]);
                }
            };
            consume(std::integral_constant<int,0>{});consume(std::integral_constant<int,1>{});
            consume(std::integral_constant<int,2>{});consume(std::integral_constant<int,3>{});
            consume(std::integral_constant<int,4>{});consume(std::integral_constant<int,5>{});
            if constexpr(GS==28)consume(std::integral_constant<int,6>{});
            const float scale=outer_scale*float(a.scale[size_t(row)*a.groups+group]);
            const float minimum=outer_minimum*float(a.minimum[size_t(row)*a.groups+group]);
            #pragma unroll
            for(int m=0;m<Rows;++m)if(ExactRows || (CachedSum && Rows!=4 && Rows!=8) || m<a.activation_rows) {
                const int sum=CachedSum?a.activation_sum[size_t(m)*a.groups+group]:sums[m];
                acc[m]+=a.activation_scale[size_t(m)*a.groups+group]*(scale*float(dots[m])-minimum*float(sum));
            }
        }
        #pragma unroll
        for(int m=0;m<Rows;++m) {
            #pragma unroll
            for(int offset=16;offset>0;offset>>=1)acc[m]+=__shfl_xor_sync(0xffffffffu,acc[m],offset);
            #pragma unroll
            for(int pad=0;pad<3;++pad)acc[m]=__fadd_rn(acc[m],0.f);
            total[m]=part==0?acc[m]:__fadd_rn(total[m],acc[m]);
        }
    }
    __shared__ float partial[Rows][2][2];
    #pragma unroll
    for(int m=0;m<Rows;++m)if(lane==0)partial[m][local_row][warp]=total[m];
    __syncthreads();
    if(warp==0 && lane==0) {
        #pragma unroll
        for(int m=0;m<Rows;++m)if(ExactRows || m<a.activation_rows)
            a.output[size_t(m)*a.rows+row]=__float2half_rn(__fadd_rn(partial[m][local_row][0],partial[m][local_row][1]));
    }
}

// Four output rows per CTA, preserving the original four independent trees
// and the (p0+p2)+(p1+p3) merge without a CTA barrier.
template<int GS,int Bits,int Rows,bool CachedSum,bool ExactRows,bool NarrowOffsets=false,int FixedGroups=0,bool RowProof=false>
__global__ void __launch_bounds__(128) nint_fixed_one_warp_output_kernel(NintWholeArgs a) {
    if constexpr(FixedGroups)a.groups=FixedGroups;
    const int row=int(blockIdx.x)*4+int(threadIdx.y),lane=threadIdx.x;
    if(row>=a.rows)return;
    if constexpr(RowProof)a.group_load=nint_row_reads_fit<GS,Bits,std::conditional_t<NarrowOffsets,uint32_t,uint64_t>>(a,row);
    const auto row_bit=uint64_t(a.offsets[row]);
    const float outer_scale=a.outer_scale[row],outer_minimum=a.outer_minimum[row];
    float total[Rows][2]{};
    #pragma unroll 1
    for(int part=0;part<4;++part) {
        const int warp=part==0?0:part==1?2:part==2?1:3;
        float acc[Rows]{};
        for(int group=warp*32+lane;group<a.groups;group+=128) {
            using Offset=std::conditional_t<NarrowOffsets,uint32_t,uint64_t>;
            using Packed=std::conditional_t<RowProof,NintRowBoundGroup<GS,Bits,Offset>,NintWholeGroup<GS,Bits,false,Offset>>;
            const Packed packed(a.weights,row_bit+uint64_t(group*GS)*Bits,a.bytes,a.group_load);
            int dots[Rows]{},sums[CachedSum?1:Rows]{};
            const auto consume=[&](auto constant) {
                constexpr int chunk=decltype(constant)::value;
                const int column=group*GS+chunk*4,qw=packed.template codes<chunk>();
                #pragma unroll
                for(int m=0;m<Rows;++m)if(ExactRows || (CachedSum && Rows!=4 && Rows!=8) || m<a.activation_rows) {
                    const int qx=*reinterpret_cast<const int*>(a.activation+size_t(m)*a.groups*GS+column);
                    if constexpr(!CachedSum)sums[m]=__dp4a(0x01010101,qx,sums[m]);
                    dots[m]=__dp4a(qw,qx,dots[m]);
                }
            };
            consume(std::integral_constant<int,0>{});consume(std::integral_constant<int,1>{});
            consume(std::integral_constant<int,2>{});consume(std::integral_constant<int,3>{});
            consume(std::integral_constant<int,4>{});consume(std::integral_constant<int,5>{});
            if constexpr(GS==28)consume(std::integral_constant<int,6>{});
            const float scale=outer_scale*float(a.scale[size_t(row)*a.groups+group]);
            const float minimum=outer_minimum*float(a.minimum[size_t(row)*a.groups+group]);
            #pragma unroll
            for(int m=0;m<Rows;++m)if(ExactRows || (CachedSum && Rows!=4 && Rows!=8) || m<a.activation_rows) {
                const int sum=CachedSum?a.activation_sum[size_t(m)*a.groups+group]:sums[m];
                acc[m]+=a.activation_scale[size_t(m)*a.groups+group]*(scale*float(dots[m])-minimum*float(sum));
            }
        }
        #pragma unroll
        for(int m=0;m<Rows;++m) {
            #pragma unroll
            for(int offset=16;offset;offset>>=1)acc[m]+=__shfl_xor_sync(0xffffffffu,acc[m],offset);
            #pragma unroll
            for(int pad=0;pad<3;++pad)acc[m]=__fadd_rn(acc[m],0.f);
            if(part%2==0)total[m][part/2]=acc[m];
            else total[m][part/2]=__fadd_rn(total[m][part/2],acc[m]);
        }
    }
    if(lane==0) {
        #pragma unroll
        for(int m=0;m<Rows;++m)if(ExactRows || m<a.activation_rows)
            a.output[size_t(m)*a.rows+row]=__float2half_rn(__fadd_rn(total[m][0],total[m][1]));
    }
}
// Compute the original p0/p2 and p1/p3 subtrees together. Independent packed
// decoders and integer dots provide instruction-level parallelism without
// changing either floating-point tree or the original weight storage.
template<int GS,int Bits,int Rows,bool CachedSum,bool ExactRows,bool NarrowOffsets>
__global__ void __launch_bounds__(128) nint_fixed_dual_group_output_kernel(NintWholeArgs a) {
    const int lane=threadIdx.x,local_row=threadIdx.y>>1,warp=threadIdx.y&1;
    const int row=int(blockIdx.x)*2+local_row;
    using Offset=std::conditional_t<NarrowOffsets,uint32_t,uint64_t>;
    using Packed=NintWholeGroup<GS,Bits,false,Offset>;
    const Offset row_bit=Offset(a.offsets[row]);
    const float outer_scale=a.outer_scale[row],outer_minimum=a.outer_minimum[row];
    float acc[Rows][2]{};
    for(int first=warp*32+lane;first<a.groups;first+=128) {
        const int second=first+64;
        const bool active=second<a.groups;
        const Packed p0(a.weights,row_bit+Offset(first*GS)*Bits,Offset(a.bytes),a.group_load);
        // The inactive second subtree has zero contributions. Reuse a valid
        // address for construction, keeping all guarded loads in bounds.
        const Packed p1(a.weights,row_bit+Offset((active?second:first)*GS)*Bits,Offset(a.bytes),a.group_load);
        int dot[Rows][2]{},sum[CachedSum?1:Rows][2]{};
        const auto consume=[&](auto chunk_constant) {
            constexpr int chunk=decltype(chunk_constant)::value;
            const int w0=p0.template codes<chunk>(),w1=p1.template codes<chunk>();
            #pragma unroll
            for(int m=0;m<Rows;++m)if(ExactRows || m<a.activation_rows) {
                const auto* x=a.activation+size_t(m)*a.groups*GS;
                const int x0=*reinterpret_cast<const int*>(x+first*GS+chunk*4);
                const int x1=active?*reinterpret_cast<const int*>(x+second*GS+chunk*4):0;
                dot[m][0]=__dp4a(w0,x0,dot[m][0]);
                dot[m][1]=__dp4a(w1,x1,dot[m][1]);
                if constexpr(!CachedSum) {
                    sum[m][0]=__dp4a(0x01010101,x0,sum[m][0]);
                    sum[m][1]=__dp4a(0x01010101,x1,sum[m][1]);
                }
            }
        };
        consume(std::integral_constant<int,0>{});consume(std::integral_constant<int,1>{});
        consume(std::integral_constant<int,2>{});consume(std::integral_constant<int,3>{});
        consume(std::integral_constant<int,4>{});consume(std::integral_constant<int,5>{});
        if constexpr(GS==28)consume(std::integral_constant<int,6>{});
        const float scale0=outer_scale*float(a.scale[size_t(row)*a.groups+first]);
        const float min0=outer_minimum*float(a.minimum[size_t(row)*a.groups+first]);
        const float scale1=active?outer_scale*float(a.scale[size_t(row)*a.groups+second]):0.f;
        const float min1=active?outer_minimum*float(a.minimum[size_t(row)*a.groups+second]):0.f;
        #pragma unroll
        for(int m=0;m<Rows;++m)if(ExactRows || m<a.activation_rows) {
            const int s0=CachedSum?a.activation_sum[size_t(m)*a.groups+first]:sum[m][0];
            acc[m][0]+=a.activation_scale[size_t(m)*a.groups+first]*(scale0*float(dot[m][0])-min0*float(s0));
            if(active) {
                const int s1=CachedSum?a.activation_sum[size_t(m)*a.groups+second]:sum[m][1];
                acc[m][1]+=a.activation_scale[size_t(m)*a.groups+second]*(scale1*float(dot[m][1])-min1*float(s1));
            }
        }
    }
    __shared__ float partial[Rows][2][2];
    #pragma unroll
    for(int m=0;m<Rows;++m) {
        #pragma unroll
        for(int part=0;part<2;++part) {
            #pragma unroll
            for(int offset=16;offset;offset>>=1)acc[m][part]+=__shfl_xor_sync(0xffffffffu,acc[m][part],offset);
            #pragma unroll
            for(int pad=0;pad<3;++pad)acc[m][part]=__fadd_rn(acc[m][part],0.f);
        }
        if(lane==0)partial[m][local_row][warp]=__fadd_rn(acc[m][0],acc[m][1]);
    }
    __syncthreads();
    if(warp==0 && lane==0) {
        #pragma unroll
        for(int m=0;m<Rows;++m)if(ExactRows || m<a.activation_rows)
            a.output[size_t(m)*a.rows+row]=__float2half_rn(__fadd_rn(partial[m][local_row][0],partial[m][local_row][1]));
    }
}
}

bool nint_launch_fixed_decode(NintWholeArgs args,int group_size,int uniform_bits,cudaStream_t stream) {
    const auto* fixed_bits=std::getenv("MFQ_NINT_FIXED_BITS");
    const auto* small_setting=std::getenv("MFQ_NINT_FIXED_SMALL_GROUPS");
    if((!small_setting || small_setting[0]!='0') && (!fixed_bits || fixed_bits[0]!='0') &&
            args.groups<=32 && args.activation_rows<=6 &&
            ((group_size==24 && uniform_bits==6) || (group_size==28 && uniform_bits==5))) {
        const auto launch=[&](auto rows) {
            using RowCount=decltype(rows);
            const auto geometry=[&](auto gs,auto bits) {
                using GroupSize=decltype(gs);
                using BitWidth=decltype(bits);
                const auto offset=[&](auto narrow) {
                    nint_fixed_small_kernel<GroupSize::value,BitWidth::value,RowCount::value,decltype(narrow)::value><<<
                        (args.rows+3)/4,dim3(32,4),0,stream>>>(args);
                    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,"nint_fixed_small_groups",
                        reinterpret_cast<const void*>(nint_fixed_small_kernel<GroupSize::value,BitWidth::value,RowCount::value,decltype(narrow)::value>),
                        "M="+std::to_string(args.activation_rows)+" N="+std::to_string(args.rows)+" gs="+std::to_string(group_size)+" groups="+std::to_string(args.groups));
                };
                if(args.bytes<(uint64_t(1)<<29))offset(std::true_type{});else offset(std::false_type{});
            };
            if(group_size==24)geometry(std::integral_constant<int,24>{},std::integral_constant<int,6>{});
            else geometry(std::integral_constant<int,28>{},std::integral_constant<int,5>{});
        };
        switch(args.activation_rows) {
        case 1:launch(std::integral_constant<int,1>{});break;
        case 2:launch(std::integral_constant<int,2>{});break;
        case 3:launch(std::integral_constant<int,3>{});break;
        case 4:launch(std::integral_constant<int,4>{});break;
        case 5:launch(std::integral_constant<int,5>{});break;
        case 6:launch(std::integral_constant<int,6>{});break;
        }
        return true;
    }
    const auto* fixed_output=std::getenv("MFQ_NINT_FIXED_OUTPUT");
    const bool five=group_size==28 && uniform_bits==5 && args.groups<=96 &&
        (!fixed_bits || fixed_bits[0]!='0');
    const auto* output_warps=std::getenv("MFQ_NINT_OUTPUT_TWO_WARPS");
    const auto* one_warp_setting=std::getenv("MFQ_NINT_OUTPUT_ONE_WARP");
    const bool one_warp=one_warp_setting?one_warp_setting[0]=='1':
        (args.activation_rows==1 && args.rows==2560 &&
            ((group_size==28 && uniform_bits==5 && args.groups==220) ||
             (group_size==24 && uniform_bits==6 && args.groups==256)));
    const bool two_warps=(output_warps?output_warps[0]=='1':(args.activation_rows==5 || args.activation_rows==6)) && args.rows%2==0;
    const bool output=(group_size==24 || group_size==28) &&
        (uniform_bits==6 || (group_size==28 && uniform_bits==5)) && args.groups>96 &&
        (fixed_output?fixed_output[0]!='0':(one_warp || two_warps || args.activation_rows!=4));
    if(!five && !output)return false;
    const bool cached_sums=args.activation_sum!=nullptr;
    const bool interleaved=args.activation_interleaved!=nullptr;
    const bool tracing=mfq::cuda::dispatch_trace::enabled();
    const auto* warp_rows=std::getenv("MFQ_NINT_WARP_ROWS");
    const auto* fixed_group_setting=std::getenv("MFQ_NINT_FIXED_92_GROUPS");
    const bool fixed_92=(fixed_group_setting?fixed_group_setting[0]=='1':
        (args.activation_rows>=2 && args.activation_rows<=6)) && args.groups==92 && args.rows%4==0;
    const auto* narrow_setting=std::getenv("MFQ_NINT_NARROW_OFFSETS");
    const bool narrow_offsets=(narrow_setting?narrow_setting[0]=='1':
        (args.activation_rows<=6 && args.activation_rows!=4 && args.activation_rows!=6)) && args.bytes<(uint64_t(1)<<29);
    const auto* row_setting=std::getenv("MFQ_NINT_FIXED_EXACT_ROWS");
    const bool exact_default=args.activation_rows>=2 || args.groups<=128;
    const bool exact_rows=(row_setting?row_setting[0]=='1':exact_default) && args.activation_rows<=6;
    const auto* output_narrow_setting=std::getenv("MFQ_NINT_OUTPUT_NARROW");
    const bool narrow_output_default=group_size==24 && uniform_bits==6 && args.rows==2560 &&
        args.groups==256 && args.activation_rows<=2;
    const bool output_narrow=(output_narrow_setting?output_narrow_setting[0]=='1':narrow_output_default) && args.bytes<(uint64_t(1)<<29);
    const auto launch=[&](auto rows) {
        // Keep outer tag parameters out of nested lambda captures on GCC 11.
        using RowCount=decltype(rows);
        const auto details=tracing?"M="+std::to_string(args.activation_rows)+" N="+std::to_string(args.rows)+
            " gs="+std::to_string(group_size)+" groups="+std::to_string(args.groups)+
            " capacity="+std::to_string(RowCount::value)+
            " exact="+std::to_string(exact_rows && args.activation_rows==RowCount::value)+
            " output_narrow="+std::to_string(output && output_narrow):std::string{};
        const auto launch_exact=[&](auto shape) {
        static constexpr bool Exact=decltype(shape)::value;
        const auto launch_sum=[&](auto sums) {
            using Sums=decltype(sums);
            if(output) {
                const auto launch_output=[&](auto gs,auto bits) {
                    using GroupSize=decltype(gs);
                    using BitWidth=decltype(bits);
                const auto launch_offset=[&](auto narrow) {
                static constexpr bool Narrow=decltype(narrow)::value;
                const auto launch_proof=[&](auto proof) {
                const auto* dual_setting=std::getenv("MFQ_NINT_OUTPUT_DUAL_GROUP");
                if(dual_setting && dual_setting[0]=='1' && args.rows%2==0 && args.activation_rows<=6) {
                    nint_fixed_dual_group_output_kernel<GroupSize::value,BitWidth::value,RowCount::value,Sums::value,Exact,Narrow><<<
                        args.rows/2,dim3(32,4),0,stream>>>(args);
                    if(tracing)mfq::cuda::dispatch_trace::record(stream,"nint_fixed_dual_group_output",
                        reinterpret_cast<const void*>(nint_fixed_dual_group_output_kernel<GroupSize::value,BitWidth::value,RowCount::value,Sums::value,Exact,Narrow>),details);
                    return;
                }
                if(one_warp) {
                    if constexpr(RowCount::value==1) {
                        const auto* setting=std::getenv("MFQ_NINT_FIXED_LONG_GROUPS");
                        constexpr int FixedGroups=GroupSize::value==24?256:220;
                        if(setting && setting[0]=='1' && args.groups==FixedGroups && args.rows==2560 &&
                                (GroupSize::value==24 || BitWidth::value==5)) {
                            nint_fixed_one_warp_output_kernel<GroupSize::value,BitWidth::value,1,Sums::value,Exact,Narrow,FixedGroups><<<
                                (args.rows+3)/4,dim3(32,4),0,stream>>>(args);
                            if(tracing)mfq::cuda::dispatch_trace::record(stream,"nint_fixed_long_groups",
                                reinterpret_cast<const void*>(nint_fixed_one_warp_output_kernel<GroupSize::value,BitWidth::value,1,Sums::value,Exact,Narrow,FixedGroups>),details);
                            return;
                        }
                    }
                    nint_fixed_one_warp_output_kernel<GroupSize::value,BitWidth::value,RowCount::value,Sums::value,Exact,Narrow,0,decltype(proof)::value><<<
                        (args.rows+3)/4,dim3(32,4),0,stream>>>(args);
                    if(tracing)mfq::cuda::dispatch_trace::record(stream,decltype(proof)::value?"nint_fixed_one_warp_output_bounds":"nint_fixed_one_warp_output",
                        reinterpret_cast<const void*>(nint_fixed_one_warp_output_kernel<GroupSize::value,BitWidth::value,RowCount::value,Sums::value,Exact,Narrow,0,decltype(proof)::value>),details);
                    return;
                }
                if(two_warps) {
                    nint_fixed_two_warp_output_kernel<GroupSize::value,BitWidth::value,RowCount::value,Sums::value,Exact,Narrow,decltype(proof)::value><<<args.rows/2,dim3(32,4),0,stream>>>(args);
                    if(tracing)mfq::cuda::dispatch_trace::record(stream,decltype(proof)::value?"nint_fixed_two_warp_output_bounds":uniform_bits==5?"nint_fixed5_two_warp_output":"nint_fixed6_two_warp_output",
                        reinterpret_cast<const void*>(nint_fixed_two_warp_output_kernel<GroupSize::value,BitWidth::value,RowCount::value,Sums::value,Exact,Narrow,decltype(proof)::value>),details);
                    return;
                }
                nint_fixed_output_kernel<GroupSize::value,BitWidth::value,RowCount::value,Sums::value,Exact,Narrow,decltype(proof)::value><<<args.rows,dim3(32,4),0,stream>>>(args);
                if(tracing)mfq::cuda::dispatch_trace::record(stream,decltype(proof)::value?"nint_fixed_output_bounds":uniform_bits==5?"nint_fixed5_long_output":"nint_fixed6_output",
                    reinterpret_cast<const void*>(nint_fixed_output_kernel<GroupSize::value,BitWidth::value,RowCount::value,Sums::value,Exact,Narrow,decltype(proof)::value>),details);
                };
                const auto* row_bounds=std::getenv("MFQ_NINT_ROW_BOUNDS");
                const bool proof_selected=row_bounds?row_bounds[0]!='0':args.activation_rows<=6 && args.activation_rows!=3;
                if(proof_selected)launch_proof(std::true_type{});
                else launch_proof(std::false_type{});
                };
                // All encoded bit offsets fit in 32 bits below 512 MiB.
                if(output_narrow)
                    launch_offset(std::true_type{});
                else launch_offset(std::false_type{});
                };
                if(group_size==24)launch_output(std::integral_constant<int,24>{},std::integral_constant<int,6>{});
                else if(uniform_bits==5)launch_output(std::integral_constant<int,28>{},std::integral_constant<int,5>{});
                else launch_output(std::integral_constant<int,28>{},std::integral_constant<int,6>{});
                return;
            }
            const auto launch_columns=[&](auto columns) {
                static constexpr bool Interleaved=decltype(columns)::value;
                const auto launch_offset=[&](auto narrow) {
                static constexpr bool Narrow=decltype(narrow)::value;
                const auto* row_pairs=std::getenv("MFQ_NINT_ROW_PAIRS");
                if constexpr(!Interleaved) {
                    if(args.rows%2==0 && row_pairs && row_pairs[0]=='1') {
                        nint_fixed_five_kernel<RowCount::value,Sums::value,2,false,Narrow,Exact><<<args.rows/2,dim3(32,6),0,stream>>>(args);
                        if(tracing)mfq::cuda::dispatch_trace::record(stream,"nint_fixed5_row_pairs",
                            reinterpret_cast<const void*>(nint_fixed_five_kernel<RowCount::value,Sums::value,2,false,Narrow,Exact>),details);
                        return;
                    }
                }
                if(warp_rows?warp_rows[0]=='1':args.activation_rows<=6) {
                    const auto launch_groups=[&](auto fixed_groups) {
                        using FixedGroups=decltype(fixed_groups);
                        const auto launch_bounds=[&](auto proof) {
                            nint_fixed_five_warp_rows_kernel<RowCount::value,Sums::value,Interleaved,Narrow,Exact,FixedGroups::value,decltype(proof)::value><<<(args.rows+3)/4,dim3(32,4),0,stream>>>(args);
                            if(tracing)mfq::cuda::dispatch_trace::record(stream,decltype(proof)::value?"nint_fixed5_warp_rows_bounds":FixedGroups::value?"nint_fixed5_warp_rows_92":Narrow?"nint_fixed5_warp_rows_narrow":Interleaved?"nint_fixed5_warp_rows_interleaved":"nint_fixed5_warp_rows",
                                reinterpret_cast<const void*>(nint_fixed_five_warp_rows_kernel<RowCount::value,Sums::value,Interleaved,Narrow,Exact,FixedGroups::value,decltype(proof)::value>),details);
                        };
                        const auto* row_bounds=std::getenv("MFQ_NINT_ROW_BOUNDS");
                        if(row_bounds?row_bounds[0]!='0':args.activation_rows<=6)launch_bounds(std::true_type{});
                        else launch_bounds(std::false_type{});
                    };
                    if(fixed_92)launch_groups(std::true_type{});else launch_groups(std::false_type{});
                }else {
                    nint_fixed_five_kernel<RowCount::value,Sums::value,1,Interleaved,Narrow,Exact><<<args.rows,dim3(32,3),0,stream>>>(args);
                    if(tracing)mfq::cuda::dispatch_trace::record(stream,Narrow?(Interleaved?"nint_fixed5_interleaved_narrow":"nint_fixed5_narrow"):Interleaved?"nint_fixed5_interleaved":"nint_fixed5",
                        reinterpret_cast<const void*>(nint_fixed_five_kernel<RowCount::value,Sums::value,1,Interleaved,Narrow,Exact>),details);
                }
                };
                if(narrow_offsets)launch_offset(std::true_type{});else launch_offset(std::false_type{});
            };
            if(interleaved)launch_columns(std::true_type{});else launch_columns(std::false_type{});
        };
        if(cached_sums)launch_sum(std::true_type{});else launch_sum(std::false_type{});
        };
        // Forced padded capacity must retain guards, including M5/M6 -> 8.
        if(exact_rows && args.activation_rows==RowCount::value)launch_exact(std::true_type{});
        else launch_exact(std::false_type{});
    };
    const auto* exact=std::getenv("MFQ_NINT_EXACT_BATCH");
    if(args.activation_rows==1)launch(std::integral_constant<int,1>{});
    else if(args.activation_rows==2)launch(std::integral_constant<int,2>{});
    else if(args.activation_rows==3)launch(std::integral_constant<int,3>{});
    else if(args.activation_rows==4)launch(std::integral_constant<int,4>{});
    else if(args.activation_rows==5 && (!exact || exact[0]!='0'))launch(std::integral_constant<int,5>{});
    else if(args.activation_rows==6 && (!exact || exact[0]!='0'))launch(std::integral_constant<int,6>{});
    else launch(std::integral_constant<int,8>{});
    return true;
}

bool nint_default_activation_sum(int group_size,int groups,int rows,int uniform_bits,bool interleaved) {
    const auto* long_sum=std::getenv("MFQ_NINT_LONG_ACTIVATION_SUM");
    return (group_size==28 && groups<=96 && (rows<=2 || rows==5 ||
        (uniform_bits==5 && interleaved && rows<=6))) ||
        (group_size==28 && groups>96 && uniform_bits==5 && rows<=6 && rows!=4 &&
         (!long_sum || long_sum[0]!='0')) ||
        (group_size==24 && groups>=128 && (rows==5 || rows==6));
}
