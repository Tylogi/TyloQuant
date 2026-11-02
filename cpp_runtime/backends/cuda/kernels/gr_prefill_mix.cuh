#pragma once

// Four stream projections share the activation tile. Products overwrite the
// decoded tile after the last FMA, then reduce in canonical stream order.
namespace {
template<int BM,int BN,int BK> union GrPrefillMixShared {
    struct {float a[BM][BK+1];union {float w[BN][BK+1];float wt[BK][BN];};} tile;
    float products[BM][BN];
};
template<class Input,class Value,bool AlignedQ8,bool Zero,bool Swizzle=false,int BM=128,int HiddenN=16,int BK=32,bool CompactK=false,bool VectorDecode=false,bool TransposeWeight=false,int RegisterM=4>
__global__ __launch_bounds__(512,1) void gr_prefill_mix_kernel(const Input* input,
        const uint8_t* packed,const uint8_t* row_bits,const int64_t* row_offsets,
        const uint8_t* scales,const uint8_t* minima,const float* row_scales,
        const float* row_minima,const __half* zero_scales,const Value* normalized,
        std::conditional_t<std::is_same_v<Input,float> || std::is_same_v<Value,float>,float,__half>* output,
        int rows,int width,int hidden,int groups,int gs) {
    using Output=std::conditional_t<std::is_same_v<Input,float> || std::is_same_v<Value,float>,float,__half>;
    constexpr int BN=HiddenN*4,RN=4,RM=RegisterM,Threads=512,Columns=BN/RN;
    static_assert(BM*BN/(RN*RM)==Threads,"GR fused tile and register geometry mismatch");
    using Shared=GrPrefillMixShared<BM,BN,BK>;
    extern __shared__ __align__(16) unsigned char gr_mix_dynamic[];
    Shared* storage;
    if constexpr(BK>32)storage=reinterpret_cast<Shared*>(gr_mix_dynamic);
    else {__shared__ __align__(16) Shared fixed;storage=&fixed;}
    auto& shared=*storage;
    const int tid=threadIdx.x,m=tid/Columns,n=(tid%Columns)*RN;
    const int row0=int(blockIdx.y)*BM,col0=int(blockIdx.x)*HiddenN;
    float acc[RM][RN][4]{};
    for(int k0=0;k0<width;k0+=BK) {
        for(int i=tid;i<BM*BK;i+=Threads) {
            const int r=i/BK,c=i%BK;
            shared.tile.a[r][c]=row0+r<rows && k0+c<width?float(input[int64_t(row0+r)*width+k0+c]):0;
        }
        if constexpr(VectorDecode) {
            for(int i=tid;i<BN*(BK/4);i+=Threads) {
                const int r=i/(BK/4),c=(i%(BK/4))*4,hcol=col0+r%HiddenN;
                const int neuron=(r/HiddenN)*hidden+hcol;
                float values[4]{};
                if(hcol<hidden && k0+c<width)gr_prefill_decode_four<AlignedQ8,Zero>(
                    values,packed,row_bits,row_offsets,scales,minima,row_scales,row_minima,zero_scales,
                    neuron,k0+c,width,groups,gs);
                #pragma unroll
                for(int j=0;j<4;++j) {
                    if constexpr(TransposeWeight)shared.tile.wt[c+j][r]=values[j];
                    else shared.tile.w[r][c+j]=values[j];
                }
            }
        } else for(int i=tid;i<BN*BK;i+=Threads) {
            const int r=i/BK,c=i%BK,column=k0+c,hcol=col0+r%HiddenN;
            const int neuron=(r/HiddenN)*hidden+hcol;
            float value=0;
            if(hcol<hidden && column<width) {
                const int64_t meta=int64_t(neuron)*groups+column/gs;
                float q,scale,minimum;
                if constexpr(Zero) {
                    q=float(static_cast<int8_t>(packed[meta*gs+column%gs]));
                    scale=__half2float(zero_scales[meta]);minimum=0;
                } else {
                    const int bits=AlignedQ8?8:row_bits[neuron];
                    const uint64_t bit=uint64_t(row_offsets[neuron])+uint64_t(column)*bits;
                    unsigned word=packed[bit>>3];const int shift=int(bit&7);
                    if(shift+bits>8)word|=unsigned(packed[(bit>>3)+1])<<8;
                    q=float((word>>shift)&((1u<<bits)-1));
                    scale=row_scales[neuron]*float(scales[meta]);
                    minimum=row_minima[neuron]*float(minima[meta]);
                }
                value=__fsub_rn(__fmul_rn(scale,q),minimum);
            }
            const int stored_row=Swizzle?r^(r>>5):r;
            shared.tile.w[stored_row][c]=value;
        }
        __syncthreads();
        const auto accumulate=[&](int k,auto slot) {
            #pragma unroll
            for(int r=0;r<RM;++r) {
                const float x=shared.tile.a[m+r*(BM/RM)][k];
                #pragma unroll
                for(int j=0;j<RN;++j) {
                    const int stored_row=Swizzle?(n+j)^((n+j)>>5):n+j;
                    acc[r][j][decltype(slot)::value]=fmaf(x,shared.tile.w[stored_row][k],acc[r][j][decltype(slot)::value]);
                }
            }
        };
        if constexpr(TransposeWeight) {
            #pragma unroll
            for(int k=0;k<BK;++k) {
                const float4 vector=*reinterpret_cast<const float4*>(&shared.tile.wt[k][n]);
                const float weight[4]={vector.x,vector.y,vector.z,vector.w};
                #pragma unroll
                for(int r=0;r<RM;++r) {
                    const float x=shared.tile.a[m+r*(BM/RM)][k];
                    #pragma unroll
                    for(int j=0;j<RN;++j)acc[r][j][k&3]=fmaf(x,weight[j],acc[r][j][k&3]);
                }
            }
        } else if constexpr(CompactK) {
            #pragma unroll 1
            for(int k=0;k<BK;k+=4) {
                accumulate(k,std::integral_constant<int,0>{});
                accumulate(k+1,std::integral_constant<int,1>{});
                accumulate(k+2,std::integral_constant<int,2>{});
                accumulate(k+3,std::integral_constant<int,3>{});
            }
        } else {
            #pragma unroll
            for(int k=0;k<BK;++k) {
                #pragma unroll
                for(int r=0;r<RM;++r) {
                    const float x=shared.tile.a[m+r*(BM/RM)][k];
                    #pragma unroll
                    for(int j=0;j<RN;++j) {
                        const int stored_row=Swizzle?(n+j)^((n+j)>>5):n+j;
                        acc[r][j][k&3]=fmaf(x,shared.tile.w[stored_row][k],acc[r][j][k&3]);
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<RM;++r) {
        const int row=row0+m+r*(BM/RM);
        #pragma unroll
        for(int j=0;j<RN;++j) {
            const int hcol=col0+(n+j)%HiddenN,neuron=((n+j)/HiddenN)*hidden+hcol;
            float product=0;
            if(row<rows && hcol<hidden) {
                const Input projected=static_cast<Input>((acc[r][j][0]+acc[r][j][1])+(acc[r][j][2]+acc[r][j][3]));
                const Input gate=static_cast<Input>(1.0f/(1.0f+expf(-static_cast<float>(projected))));
                product=float(static_cast<Output>(__fmul_rn(float(gate),float(normalized[int64_t(row)*hidden*4+neuron]))));
            }
            shared.products[m+r*(BM/RM)][n+j]=product;
        }
    }
    __syncthreads();
    for(int i=tid;i<BM*HiddenN;i+=Threads) {
        const int r=i/HiddenN,c=i%HiddenN;
        if(row0+r<rows && col0+c<hidden) {
            float sum=0;
            #pragma unroll
            for(int s=0;s<4;++s)sum=__fadd_rn(sum,shared.products[r][s*HiddenN+c]);
            output[int64_t(row0+r)*hidden+col0+c]=static_cast<Output>(__fdiv_rn(sum,4.0f));
        }
    }
}

bool gr_prefill_mix_selected(const NintWeight& weight,const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& normalized,int64_t streams) {
    namespace tb=mfq_tensor_backend;
    const auto* option=std::getenv("MFQ_GR_PREFILL_FUSED_MIX");
    const auto* projection=std::getenv("MFQ_GR_PREFILL_MATMUL");
    if(projection && (std::atoi(projection)==2 || std::atoi(projection)==3))return false;
    if(!option && (weight.out<weight.neuron_len || input.numel()/input.size(-1)<128))return false;
    const auto supported=[](const tb::Tensor& t){return t.scalar_type()==tb::kFloat16 || t.scalar_type()==tb::kFloat32;};
    return (!option || std::atoi(option)!=0) && streams==4 && input.is_contiguous() && normalized.is_contiguous()
        && supported(input) && supported(normalized);
}

mfq_tensor_backend::Tensor gr_prefill_fused_mix(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& normalized) {
    namespace tb=mfq_tensor_backend;
    MfqCudaGuard guard(input.device());
    const auto rows=input.numel()/weight.neuron_len,hidden=weight.out/4;
    MFQ_RUNTIME_CHECK(rows<=65535*int64_t(128) && hidden<=std::numeric_limits<int>::max() &&
        weight.neuron_len<=std::numeric_limits<int>::max() && weight.ng<=std::numeric_limits<int>::max(),
        "fused GR prefill exceeds CUDA launch geometry");
    auto shape=input.sizes().vec();shape.back()=hidden;
    const auto dtype=input.scalar_type()==tb::kFloat32 || normalized.scalar_type()==tb::kFloat32?tb::kFloat32:tb::kFloat16;
    auto output=tb::empty(shape,input.options().dtype(dtype));
    const auto typed=[&](auto input_tag,auto value_tag,auto aligned_tag,auto zero_tag,auto swizzle_tag) {
        using Input=decltype(input_tag);using Value=decltype(value_tag);
        using Output=std::conditional_t<std::is_same_v<Input,float> || std::is_same_v<Value,float>,float,__half>;
        static constexpr bool aligned=decltype(aligned_tag)::value,zero=decltype(zero_tag)::value,swizzle=decltype(swizzle_tag)::value;
        {
            const auto* pipeline_option=std::getenv("MFQ_GR_PREFILL_FUSED_MIX");
            if((!pipeline_option || std::atoi(pipeline_option)==10 || std::atoi(pipeline_option)==12) && gr_prefill_pipeline_supported(input)) {
                launch_gr_prefill_pipeline<true,Input,Value,aligned,zero>(weight,input,normalized,output);return;
            }
        }
        const auto invoke=[&](auto tile_tag,auto k_tag,auto compact_tag,auto vector_tag,auto transpose_tag,auto register_tag) {
        constexpr int tile=decltype(tile_tag)::value,register_m=decltype(register_tag)::value;
        constexpr int columns=register_m==2 || tile==256?8:16,k=decltype(k_tag)::value;
        constexpr bool compact=decltype(compact_tag)::value;
        constexpr bool vector_decode=decltype(vector_tag)::value;
        constexpr bool transpose_weight=decltype(transpose_tag)::value;
        constexpr size_t shared_bytes=k>32?sizeof(GrPrefillMixShared<tile,columns*4,k>):0;
        if constexpr(shared_bytes>48*1024) {
            thread_local int configured_device=-1;
            if(configured_device!=input.device().index) {
                MFQ_CUDA_CHECK(cudaFuncSetAttribute(gr_prefill_mix_kernel<Input,Value,aligned,zero,swizzle,tile,columns,k,compact,vector_decode,transpose_weight,register_m>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared_bytes)));
                configured_device=input.device().index;
            }
        }
        gr_prefill_mix_kernel<Input,Value,aligned,zero,swizzle,tile,columns,k,compact,vector_decode,transpose_weight,register_m><<<dim3(unsigned((hidden+columns-1)/columns),unsigned((rows+tile-1)/tile)),512,shared_bytes,mfq_current_cuda_stream()>>>(
            input.data_ptr<Input>(),weight.q_packed.data_ptr<uint8_t>(),
            zero?nullptr:weight.row_q_bits.data_ptr<uint8_t>(),zero?nullptr:weight.row_q_bit_offsets.data_ptr<int64_t>(),
            zero?nullptr:weight.sub_scale.data_ptr<uint8_t>(),zero?nullptr:weight.sub_min.data_ptr<uint8_t>(),
            zero?nullptr:weight.neuron_scale.data_ptr<float>(),zero?nullptr:weight.neuron_min.data_ptr<float>(),
            zero?weight.q8_zero_scale.data_ptr<__half>():nullptr,normalized.data_ptr<Value>(),output.data_ptr<Output>(),
            int(rows),int(weight.neuron_len),int(hidden),int(weight.ng),int(weight.gs));
        };
        if constexpr(swizzle)invoke(std::integral_constant<int,128>{},std::integral_constant<int,32>{},std::false_type{},std::false_type{},std::false_type{},std::integral_constant<int,4>{});
        else {
            const auto* option=std::getenv("MFQ_GR_PREFILL_FUSED_MIX");
            const int requested_mode=option?std::atoi(option):(std::is_same<Input,__half>::value?12:10);
            const int mode=requested_mode==10 || requested_mode==12?6:requested_mode;
            if(mode==8)invoke(std::integral_constant<int,128>{},std::integral_constant<int,32>{},std::false_type{},std::true_type{},std::true_type{},std::integral_constant<int,2>{});
            else if(mode==7)invoke(std::integral_constant<int,256>{},std::integral_constant<int,32>{},std::false_type{},std::true_type{},std::true_type{},std::integral_constant<int,4>{});
            else if(mode==6)invoke(std::integral_constant<int,256>{},std::integral_constant<int,32>{},std::false_type{},std::true_type{},std::false_type{},std::integral_constant<int,4>{});
            else if(mode==4)invoke(std::integral_constant<int,256>{},std::integral_constant<int,64>{},std::false_type{},std::false_type{},std::false_type{},std::integral_constant<int,4>{});
            else if(mode==5)invoke(std::integral_constant<int,256>{},std::integral_constant<int,64>{},std::true_type{},std::false_type{},std::false_type{},std::integral_constant<int,4>{});
            else if(mode==3)invoke(std::integral_constant<int,256>{},std::integral_constant<int,32>{},std::false_type{},std::false_type{},std::false_type{},std::integral_constant<int,4>{});
            else invoke(std::integral_constant<int,128>{},std::integral_constant<int,32>{},std::false_type{},std::false_type{},std::false_type{},std::integral_constant<int,4>{});
        }
    };
    const auto launch=[&](auto input_tag,auto value_tag) {
        const auto dispatch=[&](auto swizzle_tag) {
            if(weight.q8_zero)typed(input_tag,value_tag,std::false_type{},std::true_type{},swizzle_tag);
            else if(weight.aligned_q8)typed(input_tag,value_tag,std::true_type{},std::false_type{},swizzle_tag);
            else typed(input_tag,value_tag,std::false_type{},std::false_type{},swizzle_tag);
        };
        const auto* option=std::getenv("MFQ_GR_PREFILL_FUSED_MIX");
        if(option && std::atoi(option)==2)dispatch(std::true_type{});else dispatch(std::false_type{});
    };
    if(input.scalar_type()==tb::kFloat16) {
        if(normalized.scalar_type()==tb::kFloat16)launch(__half{},__half{});else launch(__half{},float{});
    } else {
        if(normalized.scalar_type()==tb::kFloat16)launch(float{},__half{});else launch(float{},float{});
    }
    MFQ_CUDA_CHECK(cudaGetLastError());return output;
}
}
