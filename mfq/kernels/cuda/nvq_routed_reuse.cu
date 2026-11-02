#include "nvq_routed_reuse.cuh"
#include "nvq_byte_signs.cuh"
#include "nvq_e8_window.cuh"
#include "nvq_e8_narrow.cuh"
#include <cstdlib>
#include <type_traits>

namespace mfq::cuda {
namespace {
using namespace packed_nvq;

template<int Format,int Lanes,int Rows,bool WordWindow=false,int SignVariant=-1,bool Narrow32=false>
__device__ __forceinline__ void reused_group_dot(
        const NvqDeviceWeight& w,int base,int valid_rows,const int8_t* input,
        const float* scales,float* output,bool byte_signs) {
    float sums[Rows]{};
    constexpr int shuffle_lanes=Lanes>32?32:Lanes;
    const int group_lane=Lanes>32?int(threadIdx.y)*32+int(threadIdx.x):int(threadIdx.x)&(Lanes-1);
    for(int group=group_lane;group<w.ng;group+=Lanes) {
        uint32_t state[Rows]{},signs[Rows]{},exec96[Rows][3]{};
        uint64_t packed[Rows]{};
        int delta[Rows]{},dot[Rows]{},activation_sum=0;
        const int8_t* bank[Rows]{};
        float scale[Rows]{};
#pragma unroll
        for(int r=0;r<Rows;++r)if(r<valid_rows) {
            const int row=base+r;
            const int64_t state_index=int64_t(row)*w.ng+group;
            // Extended execution records include the group's state. Reuse the
            // record for its bank, scale, indices and signs.
            if constexpr(Narrow32) {
                state[r]=load_e8_state32(w.sub_scale,row*w.ng+group);
            }else if constexpr(Format==kNvq2JscXLGroupExec) {
                packed[r]=load_group_exec64(w.indices,row,group,w.ng);
                state[r]=uint32_t(packed[r]>>60);
            }else if constexpr(Format==kNvq3JscLGroupExec) {
                load_group_exec96_words(w.indices,row,group,w.ng,exec96[r]);
                state[r]=(exec96[r][2]>>20)&15u;
            }else if constexpr(Format==kNvq1L || Format==kNvq1S) {
                if(w.aux_nbytes==0) {
                    packed[r]=load_nvq1_record64<Format>(w.indices,w.indices_nbytes,row,group,w.ng);
                    state[r]=uint32_t(packed[r]>>(Format==kNvq1S?27:33))&(Format==kNvq1S?15u:7u);
                }else state[r]=w.sub_bits==4?load_packed_4(w.sub_scale,state_index):
                    load_packed_bits(w.sub_scale,state_index*w.sub_bits,w.sub_bits,w.sub_scale_nbytes);
            }else state[r]=w.sub_bits==4?load_packed_4(w.sub_scale,state_index):
                load_packed_bits(w.sub_scale,state_index*w.sub_bits,w.sub_bits,w.sub_scale_nbytes);
            scale[r]=(Format==kNvq1L || Format==kNvq1S)?w.neuron_scale[row]*float(state[r]):
                format_scale<Format>(w.neuron_scale[row],state[r],w.codebook);
            bank[r]=active_codebook<Format>(w.codebook,state[r]);
            if constexpr(Format==kNvq1L || Format==kNvq1S) {
                const int negative=w.aux_nbytes==0?int((packed[r]>>(Format==kNvq1S?31:36))&1u):
                    load_packed_bits(w.aux,state_index,1,w.aux_nbytes);
                delta[r]=negative?-1:1;
                if constexpr(Format==kNvq1S)bank[r]=w.codebook+negative*kNvq1SBankBytes;
            } else if constexpr(Format!=kNvq2Exec && Format!=kNvq2JscExec && Format!=kNpq0L && Format!=kNpq0S &&
                               Format!=kNvq2JscXLGroupExec && Format!=kNvq3JscLGroupExec) {
                if constexpr(Narrow32)signs[r]=load_e8_signs32(w.aux,(row*w.nsign+group*3)*7,int(w.aux_nbytes));
                else signs[r]=load_packed_sign_group3(w.aux,(int64_t(row)*w.nsign+group*3)*7,w.aux_nbytes);
            }
            if constexpr(Format!=kNvq2Exec && Format!=kNvq2JscExec && Format!=kNpq0L && Format!=kNpq0S &&
                    Format!=kNvq2JscXLGroupExec && Format!=kNvq3JscLGroupExec) {
                constexpr int bits=format_index_bits(Format),vectors=is_d4_format(Format)?6:3;
                if constexpr(Narrow32) {
                    packed[r]=load_e8_indices32<bits,WordWindow>(w.indices,
                        row*w.nvec+group*vectors,int(w.indices_nbytes));
                }else if constexpr(WordWindow && (Format==5 || Format==13 || Format==14)) {
                    packed[r]=load_e8_index_window<bits>(w.indices,
                        int64_t(row)*w.nvec+group*vectors,w.indices_nbytes);
                }else if(!((Format==kNvq1L || Format==kNvq1S) && w.aux_nbytes==0))
                    packed[r]=load_packed_group_window<bits*vectors>(w.indices,
                        (int64_t(row)*w.nvec+group*vectors)*bits,w.indices_nbytes);
            }
        }
#pragma unroll
        for(int segment=0;segment<3;++segment) {
            const int2 x=*reinterpret_cast<const int2*>(input+group*24+segment*8);
            if constexpr(Format==kNvq1L || Format==kNvq1S)
                activation_sum=__dp4a(0x01010101,x.y,__dp4a(0x01010101,x.x,activation_sum));
#pragma unroll
            for(int r=0;r<Rows;++r)if(r<valid_rows) {
                NvqVec8Values<Format> weight{};
                if constexpr(Format!=4 && Format!=6 && Format!=7 && Format!=9 && Format<16) {
                    // The GS24 activation tail is zero. The checked packed
                    // window already bounds all metadata reads, including
                    // padded segments, so codebook vectors need no tail test.
                    constexpr int bits=format_index_bits(Format);
                    constexpr uint32_t mask=(1u<<bits)-1u;
                    int2 codes;
                    if constexpr(is_d4_format(Format)) {
                        const uint32_t a=uint32_t(packed[r]>>(segment*2*bits))&mask;
                        const uint32_t b=uint32_t(packed[r]>>((segment*2+1)*bits))&mask;
                        codes=make_int2(reinterpret_cast<const int*>(bank[r])[a],reinterpret_cast<const int*>(bank[r])[b]);
                    }else {
                        const uint32_t index=uint32_t(packed[r]>>(segment*bits))&mask;
                        codes=reinterpret_cast<const int2*>(bank[r])[index];
                    }
                    if constexpr(Format!=1 && Format!=8) {
                        const uint32_t sign=(signs[r]>>(segment*7))&127u;
                        int last=__popc(sign)&1;
                        if constexpr(!is_d4_format(Format))
                            if(w.sign_mode&1)last^=int((packed[r]>>(segment*bits+7))&1u);
                        const uint32_t mask8=sign|(uint32_t(last)<<7);
                        if constexpr(SignVariant==1)codes=nvq_apply_byte_signs8(codes,mask8);
                        else if constexpr(SignVariant==0)codes=apply_sign8(codes,mask8);
                        else codes=byte_signs?nvq_apply_byte_signs8(codes,mask8):apply_sign8(codes,mask8);
                    }
                    weight={codes,delta[r],true};
                }else weight=load_nvq_group_vec8<Format,true>(w.indices,w.indices_nbytes,w.aux,w.aux_nbytes,
                    w.codebook,bank[r],base+r,group,segment,w.ng,w.nvec,w.nsign,w.sign_mode,
                    state[r],signs[r],packed[r],exec96[r],delta[r]);
                if(weight.valid)dot[r]=__dp4a(weight.values.y,x.y,__dp4a(weight.values.x,x.x,dot[r]));
            }
        }
#pragma unroll
        for(int r=0;r<Rows;++r)if(r<valid_rows) {
            float value=float(dot[r]);
            if constexpr(Format==kNvq1L)value+=0.125f*float(delta[r]*activation_sum);
            if constexpr(Format==kNvq1S)value+=0.15625f*float(delta[r]*activation_sum);
            sums[r]=fmaf(scale[r]*scales[group],value,sums[r]);
        }
    }
#pragma unroll
    for(int r=0;r<Rows;++r) {
#pragma unroll
        for(int offset=shuffle_lanes/2;offset>0;offset>>=1)sums[r]+=__shfl_xor_sync(0xffffffffu,sums[r],offset,shuffle_lanes);
        output[r]=sums[r];
    }
}

template<int Format,int Lanes,int Rows,bool Heterogeneous,int Warps,bool WordWindow=false,int SignVariant=-1,bool Narrow32=false>
__global__ void __launch_bounds__(Warps*32) routed_reuse_kernel(NvqRoutedReuseParams args,int byte_signs) {
    constexpr int rows_per_warp=Lanes>32?Rows:(32/Lanes)*Rows;
    constexpr int rows_per_block=Lanes>32?Rows:Warps*rows_per_warp;
    const int pair=blockIdx.y,first=blockIdx.x*rows_per_block+
        (Lanes>32?0:threadIdx.y*rows_per_warp+(threadIdx.x/Lanes)*Rows);
    if(first>=args.rows)return;
    const int expert=args.ids[pair];if(unsigned(expert)>=unsigned(args.experts))return;
    int local;NvqDeviceWeight w=args.weight;int format=Format;
    if constexpr(Heterogeneous) {
        const int pool=args.active?args.active[1+pair]:args.expert_pool[expert];
        local=args.active?args.active[1+args.pairs+pair]:args.expert_local[expert];
        if(unsigned(pool)>=unsigned(args.pools) || local<0)return;
        const auto* p=args.pointers+int64_t(pool)*5;
        const auto* s=args.sizes+int64_t(pool)*3;
        const auto* g=args.params+int64_t(pool)*7;
        if(local>=g[0])return;
        w={reinterpret_cast<const uint8_t*>(p[0]),s[0],reinterpret_cast<const uint8_t*>(p[1]),s[1],
            reinterpret_cast<const uint8_t*>(p[2]),s[2],reinterpret_cast<const float*>(p[3]),reinterpret_cast<const int8_t*>(p[4]),
            0,g[0]*args.rows,g[1],g[2],g[3],g[4],g[5]};
        format=g[6];
    } else {
        local=args.expert_local[expert];if(unsigned(local)>=unsigned(args.local_experts))return;
    }
    const int source=args.down?pair:pair/args.routes;
    const int8_t* input=args.input+int64_t(source)*w.ng*24;
    const float* scales=args.scales+int64_t(source)*w.ng;
    float values[Rows]{};
#define MFQ_REUSE_DOT(F) reused_group_dot<F,Lanes,Rows,WordWindow,SignVariant,Narrow32>(w,local*args.rows+first,Rows,input,scales,values, \
    byte_signs==1 || (byte_signs<0 && (F==5 || F==13 || F==10 || F==11 || F==12 || F==15)))
    if constexpr(!Heterogeneous)MFQ_REUSE_DOT(Format);
    else switch(format) {
#define MFQ_REUSE_CASE(F) case F: MFQ_REUSE_DOT(F);break
        MFQ_REUSE_CASE(1);MFQ_REUSE_CASE(2);MFQ_REUSE_CASE(3);MFQ_REUSE_CASE(4);MFQ_REUSE_CASE(5);
        MFQ_REUSE_CASE(6);MFQ_REUSE_CASE(7);MFQ_REUSE_CASE(8);MFQ_REUSE_CASE(9);MFQ_REUSE_CASE(10);
        MFQ_REUSE_CASE(11);MFQ_REUSE_CASE(12);MFQ_REUSE_CASE(13);MFQ_REUSE_CASE(14);MFQ_REUSE_CASE(15);
        MFQ_REUSE_CASE(16);MFQ_REUSE_CASE(17);
#undef MFQ_REUSE_CASE
    }
#undef MFQ_REUSE_DOT
    if constexpr(Lanes>32) {
        __shared__ float partial[4][Rows];
        if(threadIdx.x==0) {
#pragma unroll
            for(int r=0;r<Rows;++r)partial[threadIdx.y][r]=values[r];
        }
        __syncthreads();
        if(threadIdx.x==0 && threadIdx.y==0) {
#pragma unroll
            for(int r=0;r<Rows;++r)args.output[int64_t(pair)*args.rows+first+r]=__float2half_rn(
                (partial[0][r]+partial[1][r])+(partial[2][r]+partial[3][r]));
        }
    }else if((threadIdx.x&(Lanes-1))==0) {
#pragma unroll
        for(int r=0;r<Rows;++r)if(first+r<args.rows)
            args.output[int64_t(pair)*args.rows+first+r]=__float2half_rn(values[r]);
    }
}
}

bool nvq_launch_routed_reuse(const NvqRoutedReuseParams& args,cudaStream_t stream) {
    const char* row_env=std::getenv("MFQ_NVQ_ROUTE_ROWS");
    const char* lane_env=std::getenv("MFQ_NVQ_ROUTE_LANES");
    const bool explicit_shape=row_env || lane_env;
    const bool canonical=args.pointers || args.format==1 || args.format==8 || args.format==5 ||
        args.format==13 || args.format==14 || args.format==10 || args.format==11 || args.format==12 || args.format==15 ||
        args.format==16 || args.format==17;
    if(!explicit_shape && !canonical)return false;
    const int automatic_rows=!args.pointers && (args.format==1 || args.format==8) && args.weight.ng<=32?2:1;
    const char* lanes8_env=std::getenv("MFQ_NVQ_ROUTE_LANES8");
    // Latest Metal uses eight K lanes for some single-row compressed decodes.
    // On SM86 only the measured short raw D4 Down geometry benefits; its
    // counterpart long projections retain sixteen lanes.
    const bool measured_d4_down=!args.pointers && !explicit_shape && args.down &&
        args.rows==2560 && args.weight.ng==27 &&
        (args.format==11 || args.format==12 || args.format==15);
    const bool lanes8=lanes8_env?lanes8_env[0]!='0':measured_d4_down;
    const int rows=row_env?std::atoi(row_env):(explicit_shape || lanes8?1:automatic_rows);
    const int lanes=lane_env?std::atoi(lane_env):(lanes8?8:16);
    const char* byte_signs=std::getenv("MFQ_NVQ_BYTE_SIGNS");
    const int fast_signs=byte_signs?(byte_signs[0]=='0'?0:1):-1;
    const char* warps_env=std::getenv("MFQ_NVQ_ROUTE_WARPS8");
    const bool raw_jsc=args.format==5 || args.format==10 || args.format==11 || args.format==12 ||
        args.format==13 || args.format==14 || args.format==15;
    const bool measured_long=!args.down && args.rows==640 && args.weight.ng==107 && raw_jsc;
    const bool measured_down=args.down && args.rows==2560 && args.weight.ng==27 && args.format==14;
    const bool automatic_warps=!args.pointers && !explicit_shape && (measured_long || measured_down);
    const bool eight_warps=warps_env?warps_env[0]!='0':automatic_warps;
    if(!((rows==2 && (lanes==16 || lanes==32)) || (rows==1 && (lanes==8 || lanes==16)) ||
            (rows==4 && lanes==128)) || args.rows%rows!=0)return false;
    const char* window_env=std::getenv("MFQ_NVQ_E8_WORD_WINDOW");
    // Two aligned words improve the measured ten-expert E8 projections except
    // short E8-4096 Down, whose original instruction schedule remains faster.
    const bool measured_window=!args.pointers && args.pairs==10 &&
        (measured_long || (args.down && args.rows==2560 && args.weight.ng==27 &&
            (args.format==5 || args.format==13)));
    const bool word_window=window_env?window_env[0]!='0':measured_window;
    const char* static_signs_env=std::getenv("MFQ_NVQ_E8_STATIC_SIGNS");
    // Select the sign algorithm once at launch instead of for every vector.
    const bool measured_signs=!args.pointers && args.pairs==10 &&
        (measured_long || (args.down && args.rows==2560 && args.weight.ng==27));
    const bool static_signs=static_signs_env?static_signs_env[0]!='0':measured_signs;
    const char* narrow_env=std::getenv("MFQ_NVQ_E8_NARROW");
    const bool narrow_requested=narrow_env?narrow_env[0]!='0':measured_signs;
    const bool narrow=narrow_requested && !args.pointers &&
        nvq_e8_narrow_eligible(args.weight,args.rows,args.local_experts,args.format);
    const auto launch=[&](auto format,auto l,auto r,auto heterogeneous,auto warps) {
        static constexpr int per_block=decltype(l)::value>32?decltype(r)::value:decltype(warps)::value*(32/decltype(l)::value)*decltype(r)::value;
        if constexpr(!decltype(heterogeneous)::value && decltype(l)::value==16 && decltype(r)::value==1 &&
                (decltype(format)::value==5 || decltype(format)::value==13 || decltype(format)::value==14)) {
            if(static_signs) {
                const bool use_byte_signs=fast_signs==1 || (fast_signs<0 &&
                    (decltype(format)::value==5 || decltype(format)::value==13));
                const auto specialized=[&](auto window,auto signs) {
                    if(narrow) {
                        routed_reuse_kernel<decltype(format)::value,16,1,false,decltype(warps)::value,
                                decltype(window)::value,decltype(signs)::value,true>
                            <<<dim3((args.rows+per_block-1)/per_block,args.pairs),dim3(32,decltype(warps)::value),0,stream>>>(args,fast_signs);
                        return;
                    }
                    routed_reuse_kernel<decltype(format)::value,16,1,false,decltype(warps)::value,
                            decltype(window)::value,decltype(signs)::value>
                        <<<dim3((args.rows+per_block-1)/per_block,args.pairs),dim3(32,decltype(warps)::value),0,stream>>>(args,fast_signs);
                };
                if(word_window) {
                    if(use_byte_signs)specialized(std::true_type{},std::integral_constant<int,1>{});
                    else specialized(std::true_type{},std::integral_constant<int,0>{});
                }else {
                    if(use_byte_signs)specialized(std::false_type{},std::integral_constant<int,1>{});
                    else specialized(std::false_type{},std::integral_constant<int,0>{});
                }
                return;
            }
            if(word_window) {
                routed_reuse_kernel<decltype(format)::value,16,1,false,decltype(warps)::value,true>
                    <<<dim3((args.rows+per_block-1)/per_block,args.pairs),dim3(32,decltype(warps)::value),0,stream>>>(args,fast_signs);
                return;
            }
        }
        routed_reuse_kernel<decltype(format)::value,decltype(l)::value,decltype(r)::value,decltype(heterogeneous)::value,decltype(warps)::value>
            <<<dim3((args.rows+per_block-1)/per_block,args.pairs),dim3(32,decltype(warps)::value),0,stream>>>(args,fast_signs);
    };
    const auto shape=[&](auto format,auto heterogeneous,auto warps) {
        if(lanes==128)launch(format,std::integral_constant<int,128>{},std::integral_constant<int,4>{},heterogeneous,std::integral_constant<int,4>{});
        else if(lanes==8)launch(format,std::integral_constant<int,8>{},std::integral_constant<int,1>{},heterogeneous,warps);
        else if(lanes==32)launch(format,std::integral_constant<int,32>{},std::integral_constant<int,2>{},heterogeneous,warps);
        else if(rows==2)launch(format,std::integral_constant<int,16>{},std::integral_constant<int,2>{},heterogeneous,warps);
        else launch(format,std::integral_constant<int,16>{},std::integral_constant<int,1>{},heterogeneous,warps);
    };
    const auto select=[&](auto format,auto heterogeneous) {
        if(eight_warps && lanes<=32)shape(format,heterogeneous,std::integral_constant<int,8>{});
        else shape(format,heterogeneous,std::integral_constant<int,4>{});
    };
    if(args.pointers)select(std::integral_constant<int,0>{},std::true_type{});
    else switch(args.format) {
#define MFQ_REUSE_LAUNCH(F) case F: select(std::integral_constant<int,F>{},std::false_type{});break
        MFQ_REUSE_LAUNCH(1);MFQ_REUSE_LAUNCH(2);MFQ_REUSE_LAUNCH(3);MFQ_REUSE_LAUNCH(4);MFQ_REUSE_LAUNCH(5);
        MFQ_REUSE_LAUNCH(6);MFQ_REUSE_LAUNCH(7);MFQ_REUSE_LAUNCH(8);MFQ_REUSE_LAUNCH(9);MFQ_REUSE_LAUNCH(10);
        MFQ_REUSE_LAUNCH(11);MFQ_REUSE_LAUNCH(12);MFQ_REUSE_LAUNCH(13);MFQ_REUSE_LAUNCH(14);MFQ_REUSE_LAUNCH(15);
        MFQ_REUSE_LAUNCH(16);MFQ_REUSE_LAUNCH(17);
#undef MFQ_REUSE_LAUNCH
        default:return false;
    }
    return true;
}
}
