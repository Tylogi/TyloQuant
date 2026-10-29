#pragma once

// A row is decoded once for each pair of input offsets. Routing is expressed
// entirely by the execution plan; weights remain ordinary TQ row tensors.
template<int DT,bool Pair,int T>
__global__ __launch_bounds__(128)
void tq_indexed_kernel(const TrellisIndexedProjectionJob* jobs,
        const Scalar<DT>* input,Scalar<DT>* output,float limit) {
    const auto job=jobs[blockIdx.y];
    const int warp=threadIdx.x/32,role=Pair?(warp&1):0,slot=Pair?warp/2:warp;
    const auto m=role?job.second:job.weight;
    constexpr int R=Pair?2:4;
    const int row=blockIdx.x*R+slot;
    float value[T]{};
    if(row<m.rows) {
        if constexpr(DT==0) {
            if(m.columns)multi_dispatch<T,true,true>(m,row,input,0,job.count,0,value,job.input_offsets);
            else multi_dispatch<T,false,true>(m,row,input,0,job.count,0,value,job.input_offsets);
        }else if constexpr(T>1) {
            // Bound live activations while decoding each compressed row once
            // for the entire job, including BF16/FP32 verification windows.
            constexpr int V=T>2?8:16;
            if(m.columns)tq_float_tokens::generic_dispatch<DT,true,1,T,V,true>(
                m,row,input,0,job.count,0,value,job.input_offsets);
            else tq_float_tokens::generic_dispatch<DT,false,1,T,V,true>(
                m,row,input,0,job.count,0,value,job.input_offsets);
        }else {
            #pragma unroll
            for(int t=0;t<T;++t)if(t<job.count)
                value[t]=matrix_dot<DT,1,false>(m,row,input+job.input_offsets[t]);
        }
    }
    __shared__ float joined[R][T][2];
    #pragma unroll
    for(int t=0;t<T;++t) {
        #pragma unroll
        for(int step=16;step;step/=2)value[t]+=__shfl_down_sync(0xffffffff,value[t],step);
        if((threadIdx.x&31)==0) {
            if(row<m.rows && m.bias)value[t]+=__ldg(m.bias+row);
            if constexpr(Pair)joined[slot][t][role]=scalar_float<DT>(scalar_round<DT>(value[t]));
            else if(t<job.count && row<m.rows)
                output[job.output_offsets[t]+row]=scalar_round<DT>(value[t]);
        }
    }
    if constexpr(Pair) {
        __syncthreads();
        if(!role && (threadIdx.x&31)==0 && row<m.rows) {
            #pragma unroll
            for(int t=0;t<T;++t)if(t<job.count) {
                float g=joined[slot][t][0],u=joined[slot][t][1];
                if(limit>0) {g=fminf(g,limit);u=fmaxf(-limit,fminf(u,limit));}
                output[job.output_offsets[t]+row]=scalar_round<DT>(u*(g/(1.f+expf(-g))));
            }
        }
    }
}

template<bool Pair,int T>
int tq_indexed_launch(const TrellisIndexedProjectionJob* jobs,const void* x,
        void* y,int count,int rows,int dtype,float limit,cudaStream_t stream) {
    if(!jobs || !x || !y || x==y || count<1 || rows<1 || dtype<0 || dtype>2 ||
       !std::isfinite(limit) || limit<0)return -1;
    for(int64_t first=0;first<count;first+=65535) {
        constexpr int R=Pair?2:4;
        const dim3 grid((uint64_t(rows)+R-1)/R,min(int64_t(65535),int64_t(count)-first));
        #define INDEXED(D) tq_indexed_kernel<D,Pair,T><<<grid,128,0,stream>>>( \
            jobs+first,static_cast<const Scalar<D>*>(x),static_cast<Scalar<D>*>(y),limit)
        if(dtype==0){INDEXED(0);}
        else if(dtype==1){INDEXED(1);}
        else {INDEXED(2);}
        #undef INDEXED
        const auto error=cudaGetLastError();if(error)return int(error);
    }
    return 0;
}

extern "C" int mfq_tq_indexed(const TrellisIndexedProjectionJob* jobs,const void* x,
        void* y,int count,int rows,int dtype,int tile,cudaStream_t stream) {
    switch(tile) {
    #define TILE(T) case T:return tq_indexed_launch<false,T>(jobs,x,y,count,rows,dtype,0,stream)
    TILE(1);TILE(2);TILE(3);TILE(4);
    #undef TILE
    default:return -1;
    }
}
extern "C" int mfq_tq_indexed_swiglu(const TrellisIndexedProjectionJob* jobs,const void* x,
        void* y,int count,int rows,int dtype,int tile,float limit,cudaStream_t stream) {
    switch(tile) {
    #define TILE(T) case T:return tq_indexed_launch<true,T>(jobs,x,y,count,rows,dtype,limit,stream)
    TILE(1);TILE(2);TILE(3);TILE(4);
    #undef TILE
    default:return -1;
    }
}
