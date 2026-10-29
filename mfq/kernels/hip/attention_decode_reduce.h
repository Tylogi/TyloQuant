#pragma once
namespace mfq_hip_attention {
// Thirty-two output columns and eight independent segment lanes per CTA.
// Parallel segment statistics avoid recomputing every exponential per column.
template<class Value>__global__ __launch_bounds__(256) void split_reduce(
        const float* partial_o,const float* partial_m,const float* partial_l,
        const int64_t* lengths,Value* out,int total,int heads,int parts,
        int workspace_parts,int width,int dynamic) {
    const int row=blockIdx.x,tid=threadIdx.x,column=blockIdx.y*32+(tid&31),segment_lane=tid/32;
    const int active=attention_decode_active_parts(int(lengths[row/heads]),parts,dynamic);
    const size_t base=size_t(row)*workspace_parts;
    __shared__ float weights[256],statistics[4],sums[256];
    float maximum=-1e30f;
    for(int p=tid;p<active;p+=256)maximum=fmaxf(maximum,partial_m[base+p]);
    for(int d=32;d;d/=2)maximum=fmaxf(maximum,__shfl_xor(maximum,d,64));
    if(!(tid&63))statistics[tid/64]=maximum;
    __syncthreads();
    maximum=fmaxf(fmaxf(statistics[0],statistics[1]),fmaxf(statistics[2],statistics[3]));
    __syncthreads();
    float denominator=0;
    for(int p=tid;p<active;p+=256) {
        const float l=partial_l[base+p],w=l>0?expf(partial_m[base+p]-maximum):0.f;
        weights[p]=w;denominator+=w*l;
    }
    for(int d=32;d;d/=2)denominator+=__shfl_xor(denominator,d,64);
    if(!(tid&63))statistics[tid/64]=denominator;
    __syncthreads();
    denominator=statistics[0]+statistics[1]+statistics[2]+statistics[3];
    float value=0;
    if(column<width) {
        #pragma unroll 4
        for(int p=segment_lane;p<active;p+=8)
            value+=weights[p]*partial_o[(base+p)*width+column];
    }
    sums[tid]=value;__syncthreads();
    if(segment_lane==0 && column<width) {
        value=0;
        #pragma unroll
        for(int p=0;p<8;++p)value+=sums[p*32+(tid&31)];
        out[size_t(row)*width+column]=Value(value/(denominator>0?denominator:1.f));
    }
}
}
