#pragma once

template<int DT,int T>
__global__ __launch_bounds__(128)
void tq_parallel_token_kernel(const TrellisProjectionJob* jobs,int count,
        const Scalar<DT>* x,Scalar<DT>* y,int tokens,int stride) {
    int lo=0,hi=count;
    while(lo<hi) {
        const int mid=lo+(hi-lo)/2;
        if(blockIdx.x<unsigned(jobs[mid].block_end))hi=mid;
        else lo=mid+1;
    }
    if(lo==count)return;
    const auto job=jobs[lo];const auto m=job.weight;
    const int begin=lo?jobs[lo-1].block_end:0;
    const int row=(blockIdx.x-begin)*TRELLIS_GROUP_ROWS+threadIdx.x/32,first=blockIdx.y*T;
    if(row>=m.rows)return;
    float value[T];
    if constexpr(DT==0) {
        if(m.columns)multi_dispatch<T,true>(m,row,x,first,tokens,stride,value);
        else multi_dispatch<T,false>(m,row,x,first,tokens,stride,value);
    }else {
        constexpr int V=DT==2 && T>2?8:16;
        if(m.columns)tq_float_tokens::generic_dispatch<DT,true,1,T,V>(m,row,x,first,tokens,stride,value);
        else tq_float_tokens::generic_dispatch<DT,false,1,T,V>(m,row,x,first,tokens,stride,value);
    }
    #pragma unroll
    for(int t=0;t<T;++t) {
        #pragma unroll
        for(int step=16;step;step/=2)value[t]+=__shfl_down_sync(0xffffffff,value[t],step);
        if((threadIdx.x&31)==0 && first+t<tokens) {
            if(m.bias)value[t]+=__ldg(m.bias+row);
            y[std::size_t(job.output_offset)*tokens+std::size_t(first+t)*m.rows+row]=scalar_round<DT>(value[t]);
        }
    }
}

extern "C" int mfq_tq_parallel_tokens(const TrellisProjectionJob* jobs,int count,int blocks,
        const void* x,void* y,int tokens,int stride,int dtype,cudaStream_t stream) {
    if(!jobs || !x || !y || x==y || count<1 || blocks<1 || tokens<1 || tokens>65535 || stride<1)return -1;
    #define TQ_PARALLEL(D,T) tq_parallel_token_kernel<D,T><<<dim3(blocks,(tokens+T-1)/T),128,0,stream>>>( \
        jobs,count,static_cast<const Scalar<D>*>(x),static_cast<Scalar<D>*>(y),tokens,stride)
    if(dtype==0) {
        if(tokens==3){TQ_PARALLEL(0,3);}
        else if(tokens<8 && tokens!=4){TQ_PARALLEL(0,2);}
        else {TQ_PARALLEL(0,4);}
    }else if(dtype==1){TQ_PARALLEL(1,2);}
    else if(dtype==2) {
        if(tokens<=2){TQ_PARALLEL(2,2);}
        else if(tokens==3 || tokens==5 || tokens==6){TQ_PARALLEL(2,3);}
        else {TQ_PARALLEL(2,4);}
    }
    else return -1;
    #undef TQ_PARALLEL
    return int(cudaGetLastError());
}
