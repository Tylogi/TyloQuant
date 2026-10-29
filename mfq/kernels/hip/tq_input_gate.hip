#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>
#include <type_traits>
namespace mfq_tq_gate_detail {
template<int D>using Scalar=std::conditional_t<D==2,float,uint16_t>;
template<int D>__device__ __forceinline__ float number(Scalar<D> x) {
    if constexpr(D==2)return x;
    else if constexpr(D==0)return __half2float(__ushort_as_half(x));
    else return __bfloat162float(__ushort_as_bfloat16(x));
}
template<int D>__device__ __forceinline__ Scalar<D> cast(float x) {
    if constexpr(D==2)return x;
    else if constexpr(D==0)return __half_as_ushort(__float2half_rn(x));
    else return __bfloat16_as_ushort(__float2bfloat16(x));
}
template<int D,int G>__global__ void apply(const Scalar<D>* x,const Scalar<G>* gate,Scalar<D>* y,
        int64_t count,int mode,bool double_math) {
    const int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;
    if(i>=count)return;
    const float z=number<D>(cast<D>(number<G>(gate[i])));
    float activation;
    if(double_math) {
        const double denominator=1.+exp(-double(z));
        activation=float(mode==1?1./denominator:double(z)/denominator);
    } else {
        const float denominator=1.f+expf(-z);
        activation=mode==1?1.f/denominator:z/denominator;
    }
    activation=number<D>(cast<D>(activation));
    y[i]=cast<D>(number<D>(x[i])*activation);
}
template<int D>int launch(const void* x,const void* gate,void* y,int64_t count,int g,int mode,
        bool double_math,hipStream_t stream) {
    #define RUN(G) apply<D,G><<<(count+255)/256,256,0,stream>>>(static_cast<const Scalar<D>*>(x),static_cast<const Scalar<G>*>(gate),static_cast<Scalar<D>*>(y),count,mode,double_math)
    if(g==0){RUN(0);}else if(g==1){RUN(1);}else{RUN(2);}
    return int(hipGetLastError());
}
}
extern "C" int mfq_tq_input_gate(const void* x,const void* gate,void* y,int64_t count,
        int dtype,int gate_dtype,int mode,int double_math,hipStream_t stream) {
    if(count<0 || dtype<0 || dtype>2 || gate_dtype<0 || gate_dtype>2 || (mode!=1 && mode!=2))return int(hipErrorInvalidValue);
    if(!count)return 0;
    if(!x || !gate || !y)return int(hipErrorInvalidValue);
    if(dtype==0)return mfq_tq_gate_detail::launch<0>(x,gate,y,count,gate_dtype,mode,double_math,stream);
    if(dtype==1)return mfq_tq_gate_detail::launch<1>(x,gate,y,count,gate_dtype,mode,double_math,stream);
    return mfq_tq_gate_detail::launch<2>(x,gate,y,count,gate_dtype,mode,double_math,stream);
}
