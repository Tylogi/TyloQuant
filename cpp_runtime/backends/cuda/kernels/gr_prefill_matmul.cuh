#pragma once

// FP32 batched GR projection from compressed NINT weights. The only decoded
// weights are a small shared-memory tile reused by sixty-four token rows.
namespace {
bool gr_prefill_selected(const mfq_tensor_backend::Tensor& input) {
    const auto* option=std::getenv("MFQ_GR_PREFILL_MATMUL");
    return (!option || std::atoi(option)!=0) && input.dim()>0 && input.size(-1)>0 &&
        input.numel()/input.size(-1)>=32;
}

template<class Input,bool AlignedQ8,bool Zero,int BM=64,int RegisterN=4,int BK=32,int RegisterM=2,bool CompactK=false,bool VectorDecode=false,bool TransposeWeight=false,int TileN=32>
__global__ __launch_bounds__(BM*TileN/(RegisterN*RegisterM),1) void gr_prefill_matmul_kernel(const Input* input,
        const uint8_t* packed,const uint8_t* row_bits,const int64_t* row_offsets,
        const uint8_t* scales,const uint8_t* minima,const float* row_scales,
        const float* row_minima,const __half* zero_scales,float* output,
        int rows,int width,int outputs,int groups,int gs) {
    constexpr int BN=TileN,Pad=BK+1,Threads=BM*BN/(RegisterN*RegisterM),Columns=BN/RegisterN;
    static_assert(BK%4==0,"GR K tile must preserve the four FP32 accumulator streams");
    __shared__ float a[BM][Pad];
    union WeightTile {float row[BN][Pad];float col[BK][BN];};
    __shared__ __align__(16) WeightTile w;
    const int tid=threadIdx.x,m=tid/Columns,n=(tid%Columns)*RegisterN;
    const int row0=int(blockIdx.y)*BM,out0=int(blockIdx.x)*BN;
    float acc[RegisterM][RegisterN][4]{};
    for(int k0=0;k0<width;k0+=BK) {
        for(int i=tid;i<BM*BK;i+=Threads) {
            const int r=i/BK,c=i%BK;
            a[r][c]=row0+r<rows && k0+c<width ? float(input[(int64_t(row0+r)*width)+k0+c]) : 0;
        }
        if constexpr(VectorDecode) {
            for(int i=tid;i<BN*(BK/4);i+=Threads) {
                const int r=i/(BK/4),c=(i%(BK/4))*4,neuron=out0+r;
                float values[4]{};
                if(neuron<outputs && k0+c<width)gr_prefill_decode_four<AlignedQ8,Zero>(
                    values,packed,row_bits,row_offsets,scales,minima,row_scales,row_minima,zero_scales,
                    neuron,k0+c,width,groups,gs);
                #pragma unroll
                for(int j=0;j<4;++j) {
                    if constexpr(TransposeWeight)w.col[c+j][r]=values[j];
                    else w.row[TileN>32?r^(r>>5):r][c+j]=values[j];
                }
            }
        } else for(int i=tid;i<BN*BK;i+=Threads) {
            const int r=i/BK,c=i%BK,neuron=out0+r,column=k0+c;
            float value=0;
            if(neuron<outputs && column<width) {
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
            w.row[TileN>32?r^(r>>5):r][c]=value;
        }
        __syncthreads();
        if constexpr(TransposeWeight) {
            #pragma unroll
            for(int k=0;k<BK;++k) {
                float weight[RegisterN];
                #pragma unroll
                for(int j=0;j<RegisterN;j+=4) {
                    const float4 value=*reinterpret_cast<const float4*>(&w.col[k][n+j]);
                    weight[j]=value.x;weight[j+1]=value.y;weight[j+2]=value.z;weight[j+3]=value.w;
                }
                #pragma unroll
                for(int r=0;r<RegisterM;++r) {
                    const float x=a[m+r*(BM/RegisterM)][k];
                    #pragma unroll
                    for(int j=0;j<RegisterN;++j)acc[r][j][k&3]=fmaf(x,weight[j],acc[r][j][k&3]);
                }
            }
        } else if constexpr(CompactK) {
        // Explicit constant accumulator slots keep the arrays in registers;
        // only four K positions are live rather than the entire shared tile.
        #pragma unroll 1
        for(int kb=0;kb<BK;kb+=4) {
            #pragma unroll
            for(int slot=0;slot<4;++slot) {
                #pragma unroll
                for(int r=0;r<RegisterM;++r) {
                    const float x=a[m+r*(BM/RegisterM)][kb+slot];
                    #pragma unroll
                    for(int j=0;j<RegisterN;++j)
                        acc[r][j][slot]=fmaf(x,w.row[TileN>32?(n+j)^((n+j)>>5):n+j][kb+slot],acc[r][j][slot]);
                }
            }
        }
        } else {
        #pragma unroll
        for(int k=0;k<BK;++k) {
            #pragma unroll
            for(int r=0;r<RegisterM;++r) {
                const float x=a[m+r*(BM/RegisterM)][k];
                #pragma unroll
                for(int j=0;j<RegisterN;++j)acc[r][j][k&3]=fmaf(x,w.row[TileN>32?(n+j)^((n+j)>>5):n+j][k],acc[r][j][k&3]);
            }
        }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<RegisterM;++r)if(row0+m+r*(BM/RegisterM)<rows) {
        #pragma unroll
        for(int j=0;j<RegisterN;++j)if(out0+n+j<outputs)
            output[int64_t(row0+m+r*(BM/RegisterM))*outputs+out0+n+j]=
                (acc[r][j][0]+acc[r][j][1])+(acc[r][j][2]+acc[r][j][3]);
    }
}

mfq_tensor_backend::Tensor gr_prefill_matmul(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input) {
    namespace tb=mfq_tensor_backend;
    auto x=input.scalar_type()==tb::kFloat16 ? input.contiguous() : input.to(tb::kFloat32).contiguous();
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.size(-1)==weight.neuron_len && weight.q_packed.is_cuda() &&
        input.device()==weight.q_packed.device() && weight.gs>0 && weight.neuron_len>0 &&
        weight.neuron_len<=std::numeric_limits<int>::max() && weight.out>0 && weight.out<=std::numeric_limits<int>::max() &&
        weight.ng==(weight.neuron_len+weight.gs-1)/weight.gs,
        "GR prefill projection geometry mismatch");
    const auto rows=x.numel()/weight.neuron_len;
    if(gr_prefill_blas_selected(weight,input))return gr_prefill_blas_fp32(weight,x);
    const auto* trace_geometry=std::getenv("MFQ_GR_PREFILL_GEOMETRY_TRACE");
    if(trace_geometry && trace_geometry[0]=='1') {
        thread_local std::map<std::array<int64_t,5>,bool> seen;
        const std::array<int64_t,5> shape={weight.out,weight.neuron_len,weight.gs,int64_t(x.scalar_type()==tb::kFloat16),int64_t(weight.q8_zero)};
        if(seen.emplace(shape,true).second)std::fprintf(stderr,
            "gr_prefill_geometry rows=%lld outputs=%lld width=%lld gs=%lld half=%d zero=%d\n",
            (long long)rows,(long long)weight.out,(long long)weight.neuron_len,(long long)weight.gs,
            int(x.scalar_type()==tb::kFloat16),int(weight.q8_zero));
    }
    MFQ_RUNTIME_CHECK(rows<=65535*16,"GR prefill projection exceeds CUDA grid geometry");
    MfqCudaGuard guard(input.device());
    const auto* option=std::getenv("MFQ_GR_PREFILL_MATMUL");
    int major=0;
    const bool requested_tf32=option && std::atoi(option)==2;
    const bool requested_half2=option && std::atoi(option)==3;
    int mode=option?std::atoi(option):31;
    if(mode==32)mode=31;
    const bool wide_contraction=mode==24 && weight.out<weight.neuron_len && rows>=128;
    const bool transpose_weight=mode==22 || mode==23;
    const bool fewer_registers=mode==23;
    const bool pipeline_mode=mode==27 || mode==29 || mode==31;
    const bool vector_decode=mode==21 || mode==24 || pipeline_mode || transpose_weight;
    if(vector_decode)mode=17;
    const bool compact_k=mode==19 || mode==20;
    if(compact_k)mode=mode==19?6:17;
    if(mode==16 || mode==17 || mode==18) {
        const bool expansion=weight.out>=weight.neuron_len;
        mode=expansion?(mode>=17?(x.scalar_type()==tb::kFloat32?14:(mode==18?15:9)):(x.scalar_type()==tb::kFloat32?5:9))
            :(rows>=2048?9:rows>=128?(mode==17?11:13):1);
    }
    if(mode==12)mode=weight.out>=weight.neuron_len?(x.scalar_type()==tb::kFloat32?5:9):(rows>=2048?9:rows>=128?11:1);
    if(mode==6)mode=weight.out>=weight.neuron_len?(x.scalar_type()==tb::kFloat32?5:9):(rows>=2048?9:1);
    const int exact_tile=mode==4?128:mode==5?256:64;
    if(requested_tf32 || requested_half2)MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,input.get_device()));
    const bool tf32=requested_tf32 && major>=8;
    const bool half2=requested_half2 && major>=7;
    auto shape=input.sizes().vec();shape.back()=weight.out;
    auto output=tb::empty(shape,input.options().dtype(tb::kFloat32));
    const auto typed=[&](auto tag,auto aligned_tag,auto zero_tag) {
        using Input=decltype(tag);constexpr bool aligned=decltype(aligned_tag)::value;static constexpr bool zero=decltype(zero_tag)::value;
        if(pipeline_mode && rows>=128 && gr_prefill_pipeline_supported(x)) {
            launch_gr_prefill_pipeline<false,Input,float,aligned,zero>(weight,x,{},output);return;
        }
        const auto launch_kernel=[&](auto kernel,int tile_m,int threads,int tile_n=32) {
        kernel<<<dim3(unsigned((weight.out+tile_n-1)/tile_n),unsigned((rows+tile_m-1)/tile_m)),threads,0,mfq_current_cuda_stream()>>>(
            x.data_ptr<Input>(),weight.q_packed.data_ptr<uint8_t>(),
            zero?nullptr:weight.row_q_bits.data_ptr<uint8_t>(),zero?nullptr:weight.row_q_bit_offsets.data_ptr<int64_t>(),
            zero?nullptr:weight.sub_scale.data_ptr<uint8_t>(),zero?nullptr:weight.sub_min.data_ptr<uint8_t>(),
            zero?nullptr:weight.neuron_scale.data_ptr<float>(),zero?nullptr:weight.neuron_min.data_ptr<float>(),
            zero?weight.q8_zero_scale.data_ptr<__half>():nullptr,output.data_ptr<float>(),int(rows),
            int(weight.neuron_len),int(weight.out),int(weight.ng),int(weight.gs));
        };
        if(transpose_weight) {
            if(fewer_registers)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,32,2,false,true,true>,128,512);
            else if(mode==14)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,4,32,4,false,true,true>,256,512);
            else if(mode==11)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,64,2,false,true,true>,128,512);
            else if(mode==9)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,8,32,2,false,true,true>,256,512);
            else launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,64,4,32,2,false,true,true>,64,256);
            return;
        }
        if(vector_decode) {
            if(wide_contraction)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,8,32,2,false,true,false,64>,128,512,64);
            else if(mode==14)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,4,32,4,false,true>,256,512);
            else if(mode==11)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,64,2,false,true>,128,512);
            else if(mode==9)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,8,32,2,false,true>,256,512);
            else launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,64,4,32,2,false,true>,64,256);
            return;
        }
        if(compact_k) {
            if(mode==14)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,4,32,4,true>,256,512);
            else if(mode==11)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,64,2,true>,128,512);
            else if(mode==9)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,8,32,2,true>,256,512);
            else if(mode==5)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,4,32,2,true>,256,1024);
            else launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,64,4,32,2,true>,64,256);
            return;
        }
        if(half2)launch_kernel(gr_prefill_half2_kernel<Input,aligned,zero>,64,256);
        else if(tf32)launch_kernel(gr_prefill_tf32_kernel<Input,aligned,zero>,32,128);
        else if(mode==10)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,64,4,64>,64,256);
        else if(mode==11)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,64>,128,512);
        else if(mode==13)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,64,4>,128,256);
        else if(mode==14)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,4,32,4>,256,512);
        else if(mode==15)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,8,32,4>,256,256);
        else if(mode==9)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,8>,256,512);
        else if(mode==8)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,8>,128,256);
        else if(mode==7)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,64,8>,64,128);
        else if(exact_tile==256)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256>,256,1024);
        else if(exact_tile==128)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128>,128,512);
        else launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero>,64,256);
    };
    const auto launch=[&](auto tag) {
        if(weight.q8_zero)typed(tag,std::false_type{},std::true_type{});
        else if(weight.aligned_q8)typed(tag,std::true_type{},std::false_type{});
        else typed(tag,std::false_type{},std::false_type{});
    };
    if(x.scalar_type()==tb::kFloat16)launch(__half{});else launch(float{});
    MFQ_CUDA_CHECK(cudaGetLastError());return output;
}
} // namespace
