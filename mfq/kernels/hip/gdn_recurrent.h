#pragma once
namespace mfq_hip_gdn {
// Four independent value columns share each physical wave. The state stays
// in FP32 registers for the entire token block, including cached prefill.
template<int Waves,bool Transposed> __global__ __launch_bounds__(64*Waves,2)
void recurrent4(const float* q,const float* k,const float* v,const float* g,
        const float* beta,const float* initial,float* output,float* final_state,
        int query_heads,int value_heads,int tokens,bool tiled_heads) {
    constexpr int D=128,Lanes=16,Rows=D/Lanes;
    const int tid=threadIdx.x,lane=tid%Lanes,wave=tid/64,column_in_wave=(tid%64)/Lanes;
    const int column=(blockIdx.y*Waves+wave)*4+column_in_wave;
    const int group=blockIdx.x,batch=group/value_heads,head=group%value_heads;
    const int query_head=tiled_heads?head%query_heads:head/(value_heads/query_heads);
    const size_t q_base=size_t(batch*query_heads+query_head)*tokens*D;
    const size_t v_base=size_t(group)*tokens*D,gate_base=size_t(group)*tokens;
    const size_t state_base=size_t(group)*D*D;
    float state[Rows];
    #pragma unroll
    for(int r=0;r<Rows;++r) {
        const int row=r*Lanes+lane;
        const size_t at=Transposed?size_t(column)*D+row:size_t(row)*D+column;
        state[r]=initial?initial[state_base+at]:0.f;
    }
    for(int t=0;t<tokens;++t) {
        float key[Rows],query[Rows],projection=0.f;
        #pragma unroll
        for(int r=0;r<Rows;++r) {
            const size_t at=q_base+size_t(t)*D+r*Lanes+lane;
            key[r]=k[at];query[r]=q[at];projection+=state[r]*key[r];
        }
        #pragma unroll
        for(int shift=8;shift;shift/=2)projection+=__shfl_xor(projection,shift,Lanes);
        const float decay=expf(g[gate_base+t]);
        const float delta=(v[v_base+size_t(t)*D+column]-decay*projection)*beta[gate_base+t];
        float result=0.f;
        #pragma unroll
        for(int r=0;r<Rows;++r){state[r]=decay*state[r]+key[r]*delta;result+=state[r]*query[r];}
        #pragma unroll
        for(int shift=8;shift;shift/=2)result+=__shfl_xor(result,shift,Lanes);
        if(!lane)output[v_base+size_t(t)*D+column]=result*0.08838834764831845f;
    }
    #pragma unroll
    for(int r=0;r<Rows;++r) {
        const int row=r*Lanes+lane;
        const size_t at=Transposed?size_t(column)*D+row:size_t(row)*D+column;
        final_state[state_base+at]=state[r];
    }
}
}
