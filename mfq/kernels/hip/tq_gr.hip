#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <type_traits>
#include "mfq_hip_compat.h"
#include "tq_projection.h"
#include "tq_vector.h"
#include "tq_float_tokens.h"

namespace mfq_tq_gr_impl {
using namespace mfq_tq_cuda_exact;
template<int DT,int W,int Mode>
__global__ __launch_bounds__(256) void activation(ProjectionView weight,const S<DT>* input,
    S<DT>* output,int streams) {
    constexpr int R=4/W;
    const int wave=threadIdx.x/64,lane=threadIdx.x&63,slot=wave/W,row=blockIdx.x*R+slot;
    float value[1]{};
    if(row<weight.rows) {
        const auto* source=input+size_t(blockIdx.y)*weight.input_columns;
        if(weight.columns)generic_dispatch<DT,true,W,1,16>(weight,row,source,0,1,weight.input_columns,value);
        else generic_dispatch<DT,false,W,1,16>(weight,row,source,0,1,weight.input_columns,value);
    }
    for(int shift=32;shift;shift/=2)value[0]+=__shfl_down(value[0],shift,64);
    __shared__ float partial[R][W];
    if constexpr(W>1) {
        if(!lane)partial[slot][wave%W]=value[0];
        __syncthreads();
        value[0]=lane<W?partial[slot][lane]:0.f;
        for(int shift=32;shift;shift/=2)value[0]+=__shfl_down(value[0],shift,64);
    }
    if(threadIdx.x%(W*64)==0 && row<weight.rows) {
        const float projected=number<DT>(rounded<DT>(value[0]+(weight.bias?weight.bias[row]:0.f)));
        const float low=number<DT>(rounded<DT>(__fdiv_rn(projected,float(streams))));
        const float gate=number<DT>(rounded<DT>(1.f/(1.f+expf(-low))));
        output[size_t(blockIdx.y)*weight.rows+row]=rounded<DT>(__fmul_rn(Mode==1?low:2.f,gate));
    }
}

template<int DT>
__global__ __launch_bounds__(256) void mix(ProjectionView weight,const S<DT>* input,
    const S<DT>* normalized,S<DT>* output,int hidden,int streams) {
    const int wave=threadIdx.x/64,lane=threadIdx.x&63,col=blockIdx.x,row=wave*hidden+col;
    float value[1]{};
    if(wave<streams) {
        const auto* source=input+size_t(blockIdx.y)*weight.input_columns;
        if(weight.columns)generic_dispatch<DT,true,1,1,8>(weight,row,source,0,1,weight.input_columns,value);
        else generic_dispatch<DT,false,1,1,8>(weight,row,source,0,1,weight.input_columns,value);
    }
    for(int shift=32;shift;shift/=2)value[0]+=__shfl_down(value[0],shift,64);
    __shared__ float products[4];
    if(!lane && wave<streams) {
        const float projected=number<DT>(rounded<DT>(value[0]+(weight.bias?weight.bias[row]:0.f)));
        const float gate=number<DT>(rounded<DT>(1.f/(1.f+expf(-projected))));
        products[wave]=number<DT>(rounded<DT>(__fmul_rn(gate,number<DT>(normalized[size_t(blockIdx.y)*weight.rows+row]))));
    }
    __syncthreads();
    if(!threadIdx.x) {
        float sum=0;for(int s=0;s<streams;++s)sum=__fadd_rn(sum,products[s]);
        output[size_t(blockIdx.y)*hidden+col]=rounded<DT>(__fdiv_rn(sum,float(streams)));
    }
}
template<int DT,int Mode>
void launch(ProjectionView weight,const void* input,const void* norm,void* output,int tokens,int streams,hipStream_t stream) {
    const auto* x=static_cast<const S<DT>*>(input);auto* y=static_cast<S<DT>*>(output);
    if constexpr(Mode==3)mix<DT><<<dim3(weight.rows/streams,tokens),256,0,stream>>>(weight,x,static_cast<const S<DT>*>(norm),y,weight.rows/streams,streams);
    else {
        #define RUN(W) activation<DT,W,Mode><<<dim3((weight.rows+4/W-1)/(4/W),tokens),256,0,stream>>>(weight,x,y,streams)
        if(weight.rows<=64 || (tokens==1 && weight.rows==320 && weight.input_columns==10240)){RUN(4);}
        else if(weight.input_columns>=8192 && weight.rows<weight.input_columns){RUN(2);}else{RUN(1);}
        #undef RUN
    }
}
}
extern "C" int mfq_tq_gr(ProjectionView weight,const void* input,const void* normalized,void* output,
    int tokens,int streams,int mode,int dtype,hipStream_t stream) {
    if(!input || !output || !weight.words || !weight.descriptors || !weight.scales || !weight.relative ||
        tokens<1 || tokens>8 || streams<1 || mode<1 || mode>3 || (dtype!=0 && dtype!=2) ||
        (mode==3 && (!normalized || streams>4 || weight.rows%streams)))return int(hipErrorInvalidValue);
    #define RUN(D,M) mfq_tq_gr_impl::launch<D,M>(weight,input,normalized,output,tokens,streams,stream)
    #define MODES(D) if(mode==1){RUN(D,1);}else if(mode==2){RUN(D,2);}else{RUN(D,3);}
    if(dtype==0){MODES(0);}else{MODES(2);}
    #undef MODES
    #undef RUN
    return int(hipGetLastError());
}
