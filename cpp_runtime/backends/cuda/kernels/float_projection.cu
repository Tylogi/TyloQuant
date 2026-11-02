#include "../ops/include/float_projection.h"
#include "../ops/include/nint.h"
#include "qwen4_exp.h"

#include <limits>
#include <algorithm>
#include <map>
#include <mutex>
#include <type_traits>
#include <cstdlib>
#include <array>
#include <cstdio>
#include "gr_prefill_tf32.cuh"
#include "gr_prefill_half2.cuh"
#include "gr_prefill_decode.cuh"
#include "gr_prefill_pipeline.cuh"
#include "gr_prefill_blas.h"
#include "gr_prefill_matmul.cuh"
#include "gr_prefill_mix.cuh"
#include "kernel_dispatch_trace.h"

namespace {
template<class Input>
__device__ __forceinline__ float4 gr_dense_input4(const Input* input) {
    if constexpr(std::is_same_v<Input,float>)return *reinterpret_cast<const float4*>(input);
    else {
        const uint2 x=*reinterpret_cast<const uint2*>(input);
        return make_float4(__half2float(__ushort_as_half(x.x)),__half2float(__ushort_as_half(x.x>>16)),
            __half2float(__ushort_as_half(x.y)),__half2float(__ushort_as_half(x.y>>16)));
    }
}
__device__ __forceinline__ float4 gr_dense_weight4(const uint16_t* weight) {
    const uint2 w=*reinterpret_cast<const uint2*>(weight);
    return make_float4(__uint_as_float(w.x<<16),__uint_as_float(w.x&0xffff0000u),
        __uint_as_float(w.y<<16),__uint_as_float(w.y&0xffff0000u));
}
__device__ __forceinline__ float gr_dense_dot4(float sum,float4 w,float4 x) {
    sum=fmaf(w.x,x.x,sum);sum=fmaf(w.y,x.y,sum);
    sum=fmaf(w.z,x.z,sum);return fmaf(w.w,x.w,sum);
}
template<int Rows,class Input>
__global__ void __launch_bounds__(128) gr_bf16_down_kernel(
        const Input* input,const uint16_t* weight,float* output) {
    constexpr int Width=10240,Rank=320;
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    float sums[Rows]{};
    for(int column=tid*4;column<Width;column+=128*4) {
        const float4 w=gr_dense_weight4(weight+size_t(row)*Width+column);
        #pragma unroll
        for(int m=0;m<Rows;++m)sums[m]=gr_dense_dot4(sums[m],w,
            gr_dense_input4(input+size_t(m)*Width+column));
    }
    __shared__ float partial[Rows][4];
    #pragma unroll
    for(int m=0;m<Rows;++m) {
        #pragma unroll
        for(int offset=16;offset>0;offset>>=1)sums[m]+=__shfl_down_sync(0xffffffffu,sums[m],offset);
        if(lane==0)partial[m][warp]=sums[m];
    }
    __syncthreads();
    if(tid<Rows) {
        const float sum=(partial[tid][0]+partial[tid][2])+(partial[tid][1]+partial[tid][3]);
        const float low=__fdiv_rn(sum,4.f);
        const float gate=1.f/(1.f+expf(-low));
        output[size_t(tid)*Rank+row]=__fmul_rn(low,gate);
    }
}
template<int Rows,class Normalized>
__global__ void __launch_bounds__(128) gr_bf16_up_mix_kernel(
        const float* input,const Normalized* normalized,const uint16_t* weight,float* output) {
    constexpr int Rank=320,Hidden=2560,Streams=4;
    const int column=blockIdx.x,tid=threadIdx.x,lane=tid&31,stream=tid>>5;
    const int row=stream*Hidden+column;
    float sums[Rows]{};
    for(int begin=lane*4;begin<Rank;begin+=128) {
        const float4 w=gr_dense_weight4(weight+size_t(row)*Rank+begin);
        #pragma unroll
        for(int m=0;m<Rows;++m)sums[m]=gr_dense_dot4(sums[m],w,
            gr_dense_input4(input+size_t(m)*Rank+begin));
    }
    __shared__ float products[Rows][Streams];
    #pragma unroll
    for(int m=0;m<Rows;++m) {
        #pragma unroll
        for(int offset=16;offset>0;offset>>=1)sums[m]+=__shfl_down_sync(0xffffffffu,sums[m],offset);
        if(lane==0) {
            const float gate=1.f/(1.f+expf(-sums[m]));
            products[m][stream]=__fmul_rn(gate,static_cast<float>(normalized[size_t(m)*Hidden*Streams+row]));
        }
    }
    __syncthreads();
    if(tid<Rows) {
        float sum=0.f;
        #pragma unroll
        for(int s=0;s<Streams;++s)sum=__fadd_rn(sum,products[tid][s]);
        output[size_t(tid)*Hidden+column]=sum/float(Streams);
    }
}

template<bool AlignedQ8=false,int FixedGroup=0,class Input>
__device__ __forceinline__ float affine_sum(
    const Input* input, const unsigned char* values,
    const unsigned char* row_bits, const int64_t* row_offsets,
    const unsigned char* scales, const unsigned char* minima,
    const float* row_scales, const float* row_minima, const __half* zero_scales,
    int64_t width, int64_t groups, int group_size, bool zero,
    int64_t neuron,int64_t sample,int64_t first,int64_t last,int lane) {
    const int bits = AlignedQ8 || zero ? 8 : row_bits[neuron];
    float sum = 0;
    for (int64_t group = first; group < last; ++group) {
        const int64_t meta = neuron * groups + group;
        const float scale = FixedGroup==0 && zero ? __half2float(zero_scales[meta])
            : row_scales[neuron] * float(scales[meta]);
        const float minimum = FixedGroup==0 && zero ? 0 : row_minima[neuron] * float(minima[meta]);
        if constexpr(FixedGroup>0) {
            const auto* packed=values+(row_offsets[neuron]>>3)+group*FixedGroup;
            const auto* activation=input+sample*width+group*FixedGroup;
            if((group+1)*FixedGroup<=width) {
#pragma unroll
                for(int j=lane;j<FixedGroup;j+=32) {
                    const float weight=__fsub_rn(__fmul_rn(scale,float(packed[j])),minimum);
                    sum=fmaf(weight,static_cast<float>(activation[j]),sum);
                }
            } else {
#pragma unroll
                for(int j=lane;j<FixedGroup;j+=32) {
                    if(group*FixedGroup+j>=width)continue;
                    const float weight=__fsub_rn(__fmul_rn(scale,float(packed[j])),minimum);
                    sum=fmaf(weight,static_cast<float>(activation[j]),sum);
                }
            }
        } else {
            for (int j = lane; j < group_size; j += 32) {
                const int64_t column = group * group_size + j;
                if (column >= width) continue;
                float q;
                if (zero) {
                    q = float(static_cast<signed char>(values[meta * group_size + j]));
                } else if constexpr(AlignedQ8) {
                    q=float(values[(row_offsets[neuron]>>3)+column]);
                } else {
                    const uint64_t bit = uint64_t(row_offsets[neuron]) + uint64_t(column) * bits;
                    const int shift = int(bit & 7);
                    unsigned packed = values[bit >> 3];
                    if (shift + bits > 8) packed |= unsigned(values[(bit >> 3) + 1]) << 8;
                    q = float((packed >> shift) & ((1u << bits) - 1));
                }
                // Separate affine rounding matches the canonical FP32 decoder.
                const float weight = __fsub_rn(__fmul_rn(scale, q), minimum);
                sum = fmaf(weight, static_cast<float>(input[sample * width + column]), sum);
            }
        }
    }
    for (int step = 16; step; step >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, step);
    return sum;
}

template<class Output,int Activation>
__device__ __forceinline__ Output projection_activation(float sum,float streams) {
    const Output projected=static_cast<Output>(sum);
    const Output low=static_cast<Output>(__fdiv_rn(static_cast<float>(projected),streams));
    const Output gate=static_cast<Output>(1.0f/(1.0f+expf(-static_cast<float>(low))));
    const float multiplier=Activation==2?2.0f:static_cast<float>(low);
    return static_cast<Output>(__fmul_rn(multiplier,static_cast<float>(gate)));
}

template<class Input,bool AlignedQ8=false,int FixedGroup=0,class Output=float,int Activation=0>
__global__ void affine_projection(
    const Input* input,const unsigned char* values,
    const unsigned char* row_bits,const int64_t* row_offsets,
    const unsigned char* scales,const unsigned char* minima,
    const float* row_scales,const float* row_minima,const __half* zero_scales,
    Output* output,int64_t width,int64_t outputs,int64_t groups,int group_size,bool zero,
    int splits,float streams) {
    const int lane=threadIdx.x&31;
    const int64_t neuron=int64_t(blockIdx.x)*4+(threadIdx.x>>5);
    if(neuron>=outputs)return;
    const int64_t sample=blockIdx.y;
    const float sum=affine_sum<AlignedQ8,FixedGroup>(input,values,row_bits,row_offsets,scales,minima,row_scales,
        row_minima,zero_scales,width,groups,group_size,zero,neuron,sample,
        groups*blockIdx.z/splits,groups*(blockIdx.z+1)/splits,lane);
    if (!lane) {
        if constexpr(Activation==0)output[(sample * outputs + neuron)*splits+blockIdx.z]=sum;
        else output[sample*outputs+neuron]=projection_activation<Output,Activation>(sum,streams);
    }
}

template<class Gate,class Value,bool AlignedQ8,class Input,int FixedGroup=0>
__global__ void affine_projection_mix(const Input* input,const unsigned char* values,
    const unsigned char* row_bits,const int64_t* row_offsets,
    const unsigned char* scales,const unsigned char* minima,
    const float* row_scales,const float* row_minima,const __half* zero_scales,
    const Value* normalized,
    std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>* output,
    int64_t width,int64_t hidden,int64_t groups,int group_size,bool zero,int streams) {
    using Output=std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int stream=warp;
    const int64_t hidden_column=blockIdx.x;
    const int64_t neuron=int64_t(stream)*hidden+hidden_column,sample=blockIdx.y;
    __shared__ float products[32];
    if(hidden_column<hidden) {
        const float sum=affine_sum<AlignedQ8,FixedGroup>(input,values,row_bits,row_offsets,scales,minima,row_scales,
            row_minima,zero_scales,width,groups,group_size,zero,neuron,sample,0,groups,lane);
        if(!lane) {
            const Gate projection=static_cast<Gate>(sum);
            const Gate gate=static_cast<Gate>(1.0f/(1.0f+expf(-static_cast<float>(projection))));
            const Output product=static_cast<Output>(__fmul_rn(static_cast<float>(gate),
                static_cast<float>(normalized[sample*hidden*streams+neuron])));
            products[warp]=static_cast<float>(product);
        }
    }
    __syncthreads();
    if(!lane && !stream && hidden_column<hidden) {
        float mixed=0;
        for(int s=0;s<streams;++s)mixed=__fadd_rn(mixed,products[s]);
        output[sample*hidden+hidden_column]=static_cast<Output>(__fdiv_rn(mixed,float(streams)));
    }
}
template<class Gate,class Value,class Input,int FixedGroup,int Streams>
__global__ void affine_projection_warp_mix(const Input* input,const unsigned char* values,
    const int64_t* row_offsets,const unsigned char* scales,const unsigned char* minima,
    const float* row_scales,const float* row_minima,const Value* normalized,
    std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>* output,
    int64_t width,int64_t hidden,int64_t groups) {
    using Output=std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>;
    const int lane=threadIdx.x&31;
    const int64_t column=int64_t(blockIdx.x)*Streams+(threadIdx.x>>5),sample=blockIdx.y;
    if(column>=hidden)return;
    float sums[Streams]{};
    const unsigned char* packed[Streams];
    float row_scale[Streams],row_minimum[Streams];
#pragma unroll
    for(int s=0;s<Streams;++s) {
        const auto neuron=int64_t(s)*hidden+column;
        packed[s]=values+(row_offsets[neuron]>>3);
        row_scale[s]=row_scales[neuron];
        row_minimum[s]=row_minima[neuron];
    }
    for(int64_t group=0;group<groups;++group) {
        float group_scale[Streams],group_minimum[Streams];
#pragma unroll
        for(int s=0;s<Streams;++s) {
            const auto meta=(int64_t(s)*hidden+column)*groups+group;
            group_scale[s]=__fmul_rn(row_scale[s],float(scales[meta]));
            group_minimum[s]=__fmul_rn(row_minimum[s],float(minima[meta]));
        }
#pragma unroll
        for(int j=lane;j<FixedGroup;j+=32) {
            if(group*FixedGroup+j>=width)continue;
            const float activation=static_cast<float>(input[sample*width+group*FixedGroup+j]);
#pragma unroll
            for(int s=0;s<Streams;++s) {
                const float q=float(packed[s][group*FixedGroup+j]);
                const float weight=__fsub_rn(__fmul_rn(group_scale[s],q),group_minimum[s]);
                sums[s]=fmaf(weight,activation,sums[s]);
            }
        }
    }
    for(int step=16;step;step>>=1) {
#pragma unroll
        for(int s=0;s<Streams;++s)sums[s]+=__shfl_down_sync(0xffffffffu,sums[s],step);
    }
    if(!lane) {
        float mixed=0;
#pragma unroll
        for(int s=0;s<Streams;++s) {
            const Gate projection=static_cast<Gate>(sums[s]);
            const Gate gate=static_cast<Gate>(1.0f/(1.0f+expf(-static_cast<float>(projection))));
            const auto neuron=int64_t(s)*hidden+column;
            const Output product=static_cast<Output>(__fmul_rn(static_cast<float>(gate),
                static_cast<float>(normalized[sample*hidden*Streams+neuron])));
            mixed=__fadd_rn(mixed,static_cast<float>(product));
        }
        output[sample*hidden+column]=static_cast<Output>(__fdiv_rn(mixed,float(Streams)));
    }
}
__global__ void reduce_projection_splits(const float* partial,float* output,int64_t count,int splits) {
    for(int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<count;row+=int64_t(gridDim.x)*blockDim.x) {
        float sum=0;
        for(int split=0;split<splits;++split)sum+=partial[row*splits+split];
        output[row]=sum;
    }
}
template<class Output,int Activation>
__global__ void reduce_projection_activation(const float* partial,Output* output,int64_t count,int splits,float streams) {
    for(int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<count;row+=int64_t(gridDim.x)*blockDim.x) {
        float sum=0;
        for(int split=0;split<splits;++split)sum+=partial[row*splits+split];
        output[row]=projection_activation<Output,Activation>(sum,streams);
    }
}
int projection_resident_blocks(int device) {
    static std::mutex mutex;
    static std::map<int,int> devices;
    std::lock_guard lock(mutex);
    if(const auto found=devices.find(device);found!=devices.end())return found->second;
    cudaDeviceProp properties{};
    MFQ_CUDA_CHECK(cudaGetDeviceProperties(&properties,device));
    int active=0;
    MFQ_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,affine_projection<float,false>,128,0));
    const int blocks=active*properties.multiProcessorCount;
    devices.emplace(device,blocks);return blocks;
}
} // namespace

namespace {
mfq_tensor_backend::Tensor float_projection_impl(
        const NintWeight& weight, const mfq_tensor_backend::Tensor& input,bool parallel_groups,bool native_input,
        bool fixed_groups,int activation,int64_t streams) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(input.is_cuda() && weight.q_packed.is_cuda() &&
        input.device() == weight.q_packed.device() && input.dim() >= 1 &&
        input.size(-1) == weight.neuron_len && weight.neuron_len > 0 && weight.out > 0 &&
        weight.gs > 0 && weight.gs <= 64 &&
        weight.ng == (weight.neuron_len + weight.gs - 1) / weight.gs,
        "packed float projection dimensions/device disagree");
    MfqCudaGuard guard(input.device());
    auto x = native_input && input.scalar_type()==tb::kFloat16
        ? input.contiguous() : input.to(tb::kFloat32).contiguous();
    auto shape = x.sizes().vec();
    const auto rows = x.numel() / weight.neuron_len;
    MFQ_RUNTIME_CHECK(rows <= 65535 && (weight.out + 3) / 4 <= std::numeric_limits<int>::max(),
        "packed float projection exceeds CUDA launch geometry");
    shape.back() = weight.out;
    const auto output_dtype=activation && input.scalar_type()==tb::kFloat16?tb::kFloat16:tb::kFloat32;
    auto result = tb::empty(shape, x.options().dtype(output_dtype));
    if (!rows) return result;
    const int64_t blocks=((weight.out+3)/4)*rows;
    const int target=parallel_groups?projection_resident_blocks(x.get_device()):1;
    const int splits=int(std::min<int64_t>(weight.ng,std::max<int64_t>(1,(target+blocks-1)/blocks)));
    auto partial=splits>1?tb::empty({rows,weight.out,splits},result.options().dtype(tb::kFloat32)):result;
    const auto launch=[&](auto input_tag,auto aligned_tag,auto group_tag) {
        using Input=decltype(input_tag);static constexpr bool aligned=decltype(aligned_tag)::value;
        static constexpr int group=decltype(group_tag)::value;
        const auto invoke=[&](auto output_tag,auto activation_tag,const tb::Tensor& target) {
        using Output=decltype(output_tag);constexpr int action=decltype(activation_tag)::value;
        affine_projection<Input,aligned,group,Output,action><<<dim3(unsigned((weight.out + 3) / 4), unsigned(rows),unsigned(splits)), 128, 0,
        mfq_current_cuda_stream()>>>(x.data_ptr<Input>(), weight.q_packed.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.row_q_bits.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.row_q_bit_offsets.data_ptr<int64_t>(),
        weight.q8_zero ? nullptr : weight.sub_scale.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.sub_min.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.neuron_scale.data_ptr<float>(),
        weight.q8_zero ? nullptr : weight.neuron_min.data_ptr<float>(),
        weight.q8_zero ? weight.q8_zero_scale.data_ptr<__half>() : nullptr,
        target.data_ptr<Output>(), weight.neuron_len, weight.out, weight.ng, int(weight.gs),
        weight.q8_zero,splits,float(streams));
        };
        if(!activation || splits>1)invoke(float{},std::integral_constant<int,0>{},partial);
        else {
            const auto activated=[&](auto output_tag) {
                if(activation==2)invoke(output_tag,std::integral_constant<int,2>{},result);
                else invoke(output_tag,std::integral_constant<int,1>{},result);
            };
            if(output_dtype==tb::kFloat16)activated(__half{});else activated(float{});
        }
    };
    const auto dispatch=[&](auto input_tag) {
        if(native_input && weight.aligned_q8) {
            if(fixed_groups && !weight.q8_zero) {
                switch(weight.gs) {
                    case 32:launch(input_tag,std::true_type{},std::integral_constant<int,32>{});return;
                    case 48:launch(input_tag,std::true_type{},std::integral_constant<int,48>{});return;
                    case 64:launch(input_tag,std::true_type{},std::integral_constant<int,64>{});return;
                }
            }
            launch(input_tag,std::true_type{},std::integral_constant<int,0>{});
        } else launch(input_tag,std::false_type{},std::integral_constant<int,0>{});
    };
    if(x.scalar_type()==tb::kFloat16)dispatch(__half{});else dispatch(float{});
    MFQ_CUDA_CHECK(cudaGetLastError());
    if(splits>1) {
        const int64_t count=rows*weight.out;
        if(!activation)reduce_projection_splits<<<unsigned((count+255)/256),256,0,mfq_current_cuda_stream()>>>(
                partial.data_ptr<float>(),result.data_ptr<float>(),count,splits);
        else {
            const auto invoke=[&](auto output_tag,auto activation_tag) {
                using Output=decltype(output_tag);constexpr int action=decltype(activation_tag)::value;
                reduce_projection_activation<Output,action><<<unsigned((count+255)/256),256,0,mfq_current_cuda_stream()>>>(
                    partial.data_ptr<float>(),result.data_ptr<Output>(),count,splits,float(streams));
            };
            const auto activated=[&](auto output_tag) {
                if(activation==2)invoke(output_tag,std::integral_constant<int,2>{});
                else invoke(output_tag,std::integral_constant<int,1>{});
            };
            if(output_dtype==tb::kFloat16)activated(__half{});else activated(float{});
        }
        MFQ_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}
} // namespace

mfq_tensor_backend::Tensor try_gr_bf16_down_cuda(const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& weight,int64_t streams) {
    namespace tb=mfq_tensor_backend;
    if(streams!=4 || !input.is_cuda() || !input.is_contiguous() || input.dim()<1 ||
            input.size(-1)!=10240 || input.numel()/10240<1 || input.numel()/10240>6 ||
            (input.scalar_type()!=tb::kFloat16 && input.scalar_type()!=tb::kFloat32) ||
            !weight.is_cuda() || !weight.is_contiguous() || weight.device()!=input.device() ||
            weight.scalar_type()!=tb::kBFloat16 || weight.sizes().vec()!=std::vector<int64_t>({320,10240}))return {};
    const MfqCudaGuard guard(input.device());
    const int rows=int(input.numel()/10240);
    auto shape=input.sizes().vec();shape.back()=320;
    auto output=tb::empty(shape,input.options().dtype(tb::kFloat32));
    const auto launch=[&](auto capacity,auto type) {
        using Input=decltype(type);
        gr_bf16_down_kernel<decltype(capacity)::value,Input><<<320,128,0,mfq_current_cuda_stream()>>>(
            input.data_ptr<Input>(),static_cast<const uint16_t*>(weight.data_ptr()),output.data_ptr<float>());
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
            "gr_bf16_down_silu",reinterpret_cast<const void*>(gr_bf16_down_kernel<decltype(capacity)::value,Input>),
            "M="+std::to_string(rows)+" N=320 K=10240");
    };
    const auto types=[&](auto capacity) {
        if(input.scalar_type()==tb::kFloat16)launch(capacity,__half{});else launch(capacity,float{});
    };
    switch(rows) {
        case 1:types(std::integral_constant<int,1>{});break;
        case 2:types(std::integral_constant<int,2>{});break;
        case 3:types(std::integral_constant<int,3>{});break;
        case 4:types(std::integral_constant<int,4>{});break;
        case 5:types(std::integral_constant<int,5>{});break;
        case 6:types(std::integral_constant<int,6>{});break;
    }
    MFQ_CUDA_CHECK(cudaGetLastError());return output;
}

mfq_tensor_backend::Tensor try_gr_bf16_up_mix_cuda(const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& normalized,const mfq_tensor_backend::Tensor& weight,int64_t streams) {
    namespace tb=mfq_tensor_backend;
    if(streams!=4 || !input.is_cuda() || !input.is_contiguous() || input.dim()<1 ||
            input.size(-1)!=320 || input.numel()/320<1 || input.numel()/320>6 ||
            input.scalar_type()!=tb::kFloat32 || !normalized.is_cuda() || !normalized.is_contiguous() ||
            normalized.device()!=input.device() ||
            (normalized.scalar_type()!=tb::kFloat16 && normalized.scalar_type()!=tb::kFloat32) ||
            !weight.is_cuda() || !weight.is_contiguous() || weight.device()!=input.device() ||
            weight.scalar_type()!=tb::kBFloat16 || weight.sizes().vec()!=std::vector<int64_t>({10240,320}))return {};
    auto expected=input.sizes().vec();expected.back()=10240;
    if(normalized.sizes().vec()!=expected)return {};
    const MfqCudaGuard guard(input.device());
    const int rows=int(input.numel()/320);
    auto shape=input.sizes().vec();shape.back()=2560;
    auto output=tb::empty(shape,input.options());
    const auto launch=[&](auto capacity,auto type) {
        using Normalized=decltype(type);
        gr_bf16_up_mix_kernel<decltype(capacity)::value,Normalized><<<2560,128,0,mfq_current_cuda_stream()>>>(
            input.data_ptr<float>(),normalized.data_ptr<Normalized>(),static_cast<const uint16_t*>(weight.data_ptr()),output.data_ptr<float>());
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
            "gr_bf16_up_mix",reinterpret_cast<const void*>(gr_bf16_up_mix_kernel<decltype(capacity)::value,Normalized>),
            "M="+std::to_string(rows)+" N=10240 K=320");
    };
    const auto types=[&](auto capacity) {
        if(normalized.scalar_type()==tb::kFloat16)launch(capacity,__half{});else launch(capacity,float{});
    };
    switch(rows) {
        case 1:types(std::integral_constant<int,1>{});break;
        case 2:types(std::integral_constant<int,2>{});break;
        case 3:types(std::integral_constant<int,3>{});break;
        case 4:types(std::integral_constant<int,4>{});break;
        case 5:types(std::integral_constant<int,5>{});break;
        case 6:types(std::integral_constant<int,6>{});break;
    }
    MFQ_CUDA_CHECK(cudaGetLastError());return output;
}

mfq_tensor_backend::Tensor nint_float_projection_cuda(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input,bool parallel_groups,bool native_input,bool fixed_groups) {
    if(gr_prefill_selected(input))return gr_prefill_matmul(weight,input);
    return float_projection_impl(weight,input,parallel_groups,native_input,fixed_groups,0,1);
}

int nint_float_projection_resident_blocks(int device) {
    return projection_resident_blocks(device);
}

mfq_tensor_backend::Tensor nint_float_projection_activation_cuda(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input,int64_t streams,bool injection,
        bool parallel_groups,bool native_input,bool fixed_groups) {
    MFQ_RUNTIME_CHECK(streams>0,"packed residual activation requires positive stream count");
    if(gr_prefill_selected(input)) {
        auto projected=gr_prefill_matmul(weight,input);
        if(input.scalar_type()==mfq_tensor_backend::kFloat16)projected=projected.to(mfq_tensor_backend::kFloat16);
        return injection ? mfq_qwen4_exp::gated_residual_injection(projected,streams)
                         : mfq_qwen4_exp::gated_residual_bottleneck(projected,streams);
    }
    return float_projection_impl(weight,input,parallel_groups,native_input,fixed_groups,injection?2:1,streams);
}

mfq_tensor_backend::Tensor nint_float_projection_mix_cuda(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& normalized,
        int64_t streams,bool native_input,bool fixed_groups,bool compact) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(input.dim()>0 && streams>0 && weight.out>0 && weight.out%streams==0 &&
        normalized.dim()==input.dim(),"packed residual mixing dimensions disagree");
    auto expected=input.sizes().vec();expected.back()=weight.out;
    MFQ_RUNTIME_CHECK(normalized.sizes().vec()==expected && input.is_cuda() && normalized.is_cuda() &&
        input.device()==normalized.device() && input.device()==weight.q_packed.device() &&
        input.size(-1)==weight.neuron_len && weight.ng>0 && weight.gs>0 && weight.gs<=64 &&
        weight.ng==(weight.neuron_len+weight.gs-1)/weight.gs,"packed residual mixing geometry/device disagree");
    const auto rows=input.numel()/weight.neuron_len,hidden=weight.out/streams;
    if(gr_prefill_selected(input)) {
        if(gr_prefill_blas_selected(weight,input) && normalized.is_contiguous() &&
                (normalized.scalar_type()==tb::kFloat16 || normalized.scalar_type()==tb::kFloat32))
            return gr_prefill_blas_mix(weight,input,normalized,streams);
        if(gr_prefill_mix_selected(weight,input,normalized,streams))return gr_prefill_fused_mix(weight,input,normalized);
        auto projected=gr_prefill_matmul(weight,input);
        if(input.scalar_type()==tb::kFloat16)projected=projected.to(tb::kFloat16);
        return mfq_qwen4_exp::gated_residual_mix(projected,normalized,streams);
    }
    const auto ordinary_blocks=((weight.out+3)/4)*rows;
    MfqCudaGuard guard(input.device());
    // Retain the original group partition whenever the ordinary projection splits.
    const bool unsplit=weight.ng==1 || !rows || ordinary_blocks>=projection_resident_blocks(input.get_device());
    const auto half_or_float=[](const tb::Tensor& value) {
        return value.scalar_type()==tb::kFloat16 || value.scalar_type()==tb::kFloat32;
    };
    if(!unsplit || streams>32 || !input.is_contiguous() || !normalized.is_contiguous() ||
        !half_or_float(input) || !half_or_float(normalized)) {
        auto projection=nint_float_projection_cuda(weight,input,true,native_input,fixed_groups);
        if(input.scalar_type()==tb::kFloat16)projection=projection.to(tb::kFloat16);
        return mfq_qwen4_exp::gated_residual_mix(projection,normalized,streams);
    }
    MFQ_RUNTIME_CHECK(rows<=65535 && hidden<=std::numeric_limits<int>::max(),
        "packed residual mixing exceeds CUDA launch geometry");
    auto x=native_input && input.scalar_type()==tb::kFloat16?input:input.to(tb::kFloat32);
    auto shape=input.sizes().vec();shape.back()=hidden;
    const auto dtype=input.scalar_type()==tb::kFloat32 || normalized.scalar_type()==tb::kFloat32
        ?tb::kFloat32:tb::kFloat16;
    auto output=tb::empty(shape,input.options().dtype(dtype));
    if(!rows)return output;
    const auto launch=[&](auto gate_tag,auto value_tag,auto aligned_tag,auto input_tag,auto group_tag) {
        using Gate=decltype(gate_tag);using Value=decltype(value_tag);
        using Input=decltype(input_tag);
        using Output=std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>;
        constexpr bool aligned=decltype(aligned_tag)::value;
        static constexpr int group=decltype(group_tag)::value;
        if constexpr(aligned && group>0) {
            if(compact && (streams==3 || streams==4)) {
                const auto warp_launch=[&](auto stream_tag) {
                    constexpr int count=decltype(stream_tag)::value;
                    affine_projection_warp_mix<Gate,Value,Input,group,count><<<
                        dim3(unsigned((hidden+count-1)/count),unsigned(rows)),count*32,0,mfq_current_cuda_stream()>>>(
                        x.data_ptr<Input>(),weight.q_packed.data_ptr<uint8_t>(),weight.row_q_bit_offsets.data_ptr<int64_t>(),
                        weight.sub_scale.data_ptr<uint8_t>(),weight.sub_min.data_ptr<uint8_t>(),
                        weight.neuron_scale.data_ptr<float>(),weight.neuron_min.data_ptr<float>(),
                        normalized.data_ptr<Value>(),output.data_ptr<Output>(),weight.neuron_len,hidden,weight.ng);
                };
                if(streams==3)warp_launch(std::integral_constant<int,3>{});
                else warp_launch(std::integral_constant<int,4>{});
                return;
            }
        }
        affine_projection_mix<Gate,Value,aligned,Input,group><<<dim3(unsigned(hidden),unsigned(rows)),unsigned(streams*32),0,
            mfq_current_cuda_stream()>>>(x.data_ptr<Input>(),weight.q_packed.data_ptr<uint8_t>(),
            weight.q8_zero?nullptr:weight.row_q_bits.data_ptr<uint8_t>(),
            weight.q8_zero?nullptr:weight.row_q_bit_offsets.data_ptr<int64_t>(),
            weight.q8_zero?nullptr:weight.sub_scale.data_ptr<uint8_t>(),
            weight.q8_zero?nullptr:weight.sub_min.data_ptr<uint8_t>(),
            weight.q8_zero?nullptr:weight.neuron_scale.data_ptr<float>(),
            weight.q8_zero?nullptr:weight.neuron_min.data_ptr<float>(),
            weight.q8_zero?weight.q8_zero_scale.data_ptr<__half>():nullptr,
            normalized.data_ptr<Value>(),output.data_ptr<Output>(),weight.neuron_len,hidden,weight.ng,
            int(weight.gs),weight.q8_zero,int(streams));
    };
    const auto dispatch=[&](auto gate_tag,auto value_tag) {
        const auto typed=[&](auto input_tag) {
            if(weight.aligned_q8) {
                if(fixed_groups && !weight.q8_zero) {
                    switch(weight.gs) {
                        case 32:launch(gate_tag,value_tag,std::true_type{},input_tag,std::integral_constant<int,32>{});return;
                        case 48:launch(gate_tag,value_tag,std::true_type{},input_tag,std::integral_constant<int,48>{});return;
                        case 64:launch(gate_tag,value_tag,std::true_type{},input_tag,std::integral_constant<int,64>{});return;
                    }
                }
                launch(gate_tag,value_tag,std::true_type{},input_tag,std::integral_constant<int,0>{});
            } else launch(gate_tag,value_tag,std::false_type{},input_tag,std::integral_constant<int,0>{});
        };
        if(x.scalar_type()==tb::kFloat16)typed(__half{});else typed(float{});
    };
    if(input.scalar_type()==tb::kFloat16 && normalized.scalar_type()==tb::kFloat16)dispatch(__half{},__half{});
    else if(input.scalar_type()==tb::kFloat16)dispatch(__half{},float{});
    else if(normalized.scalar_type()==tb::kFloat16)dispatch(float{},__half{});
    else dispatch(float{},float{});
    MFQ_CUDA_CHECK(cudaGetLastError());
    return output;
}
