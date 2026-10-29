#pragma once
#include "tq_projection.h"
#include "tq_vector.h"
#include <type_traits>
namespace mfq_tq_decode_matrix {
template<int DT>using S=std::conditional_t<DT==2,float,uint16_t>;
template<int V>struct Store {
    float values[V];
    __device__ __forceinline__ void operator()(int j,float a,float b) {
        values[j]=a;values[j+1]=b;
    }
};
template<int DT,int V,int B,bool Columns>__device__ __forceinline__ void row(
        ProjectionView m,S<DT>* output,uint64_t descriptor,int r,int col) {
    Store<V> store{};
    mfq_tq_vector::visit_values<V,true,Columns,B>(m.words,descriptor,m.scales,
        static_cast<const half*>(m.relative),nullptr,nullptr,r,col,m.stored_columns,m.gslog,store);
    auto* target=output+int64_t(r)*m.input_columns+col;
    if((m.input_columns%4)==0 && col+V<=m.input_columns) {
        #pragma unroll
        for(int j=0;j<V;j+=4) {
            if constexpr(DT==2)reinterpret_cast<float4*>(target)[j/4]=
                make_float4(store.values[j],store.values[j+1],store.values[j+2],store.values[j+3]);
            else {
                const uint32_t a=__half_as_ushort(__float2half_rn(store.values[j]))|
                    (uint32_t(__half_as_ushort(__float2half_rn(store.values[j+1])))<<16);
                const uint32_t b=__half_as_ushort(__float2half_rn(store.values[j+2]))|
                    (uint32_t(__half_as_ushort(__float2half_rn(store.values[j+3])))<<16);
                reinterpret_cast<uint2*>(target)[j/4]=make_uint2(a,b);
            }
        }
    } else {
        #pragma unroll
        for(int j=0;j<V;++j)if(col+j<m.input_columns) {
            if constexpr(DT==2)target[j]=store.values[j];
            else target[j]=__half_as_ushort(__float2half_rn(store.values[j]));
        }
    }
}
template<int DT,int V,bool Columns>__global__ void materialize(ProjectionView m,S<DT>* output) {
    const int col=(blockIdx.x*blockDim.x+threadIdx.x)*V;
    if(col>=m.input_columns)return;
    for(int r=blockIdx.y;r<m.rows;r+=gridDim.y) {
        const auto descriptor=m.descriptors[r];
        #define CASE(B) case B-1:row<DT,V,B,Columns>(m,output,descriptor,r,col);break
        switch((descriptor>>32)&7){CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);}
        #undef CASE
    }
}
template<int DT,int V>int launch(ProjectionView m,void* output,hipStream_t stream) {
    const dim3 grid((m.input_columns+256*V-1)/(256*V),m.rows<65535?m.rows:65535);
    if(m.columns) {
        materialize<DT,V,true><<<grid,256,0,stream>>>(m,static_cast<S<DT>*>(output));
    } else {
        materialize<DT,V,false><<<grid,256,0,stream>>>(m,static_cast<S<DT>*>(output));
    }
    return int(hipGetLastError());
}
}
