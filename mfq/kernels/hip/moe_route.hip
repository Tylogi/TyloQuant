#include "moe_route.h"
#include "mfq_hip_compat.h"
#include <atomic>

namespace {
bool has_wave64(int device) {
    static std::atomic<int> sizes[32]{};
    if(device<0 || device>=32)return false;
    int size=sizes[device].load(std::memory_order_relaxed);
    if(size==0) {
        if(hipDeviceGetAttribute(&size,hipDeviceAttributeWarpSize,device)!=hipSuccess)return false;
        sizes[device].store(size,std::memory_order_relaxed);
    }
    return size==64;
}

__global__ void scan_counts_parallel(MfqMoeRouteScanView p) {
    const int lane=threadIdx.x;
    int carry0=0,carry1=0,carry2=0,carry3=0;
    if(lane==0) {
        p.bounds[0]=0;p.tiles[0]=0;
        if(p.secondary)p.secondary[0]=0;
        if(p.tertiary)p.tertiary[0]=0;
    }
    for(int base=0;base<p.experts;base+=64) {
        const int expert=base+lane;
        const int count=expert<p.experts?p.counts[expert]:0;
        int p0=count,p1=(count+p.tile_m-1)/p.tile_m;
        int p2=p.secondary?(count+p.secondary_m-1)/p.secondary_m:0;
        int p3=p.tertiary?(count+p.tertiary_m-1)/p.tertiary_m:0;
        #pragma unroll
        for(int step=1;step<64;step*=2) {
            const int a=__shfl_up(p0,step,64),b=__shfl_up(p1,step,64);
            const int c=__shfl_up(p2,step,64),d=__shfl_up(p3,step,64);
            if(lane>=step){p0+=a;p1+=b;p2+=c;p3+=d;}
        }
        p0+=carry0;p1+=carry1;p2+=carry2;p3+=carry3;
        if(expert<p.experts) {
            p.cursors[expert]=0;p.bounds[expert+1]=p0;p.tiles[expert+1]=p1;
            if(p.secondary)p.secondary[expert+1]=p2;
            if(p.tertiary)p.tertiary[expert+1]=p3;
        }
        carry0=__shfl(p0,63,64);carry1=__shfl(p1,63,64);
        carry2=__shfl(p2,63,64);carry3=__shfl(p3,63,64);
    }
}

template<int Threads>__global__ __launch_bounds__(Threads) void route_map_single(MfqMoeRouteMapView v) {
    constexpr int MaxExperts=1024,Waves=Threads/64;
    __shared__ int counts[MaxExperts],cursors[MaxExperts],offsets[MaxExperts];
    __shared__ int sums[Waves][4];
    const int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
    const auto p=v.scan;
    for(int e=tid;e<p.experts;e+=Threads){counts[e]=0;cursors[e]=0;}
    __syncthreads();
    for(int pair=tid;pair<v.pair_count;pair+=Threads) {
        const int e=v.ids[pair];
        if(unsigned(e)<unsigned(p.experts))atomicAdd(counts+e,1);
    }
    __syncthreads();
    const int own0=tid<p.experts?counts[tid]:0;
    const int own1=(own0+p.tile_m-1)/p.tile_m;
    const int own2=p.secondary?(own0+p.secondary_m-1)/p.secondary_m:0;
    const int own3=p.tertiary?(own0+p.tertiary_m-1)/p.tertiary_m:0;
    int p0=own0,p1=own1,p2=own2,p3=own3;
    #pragma unroll
    for(int step=1;step<64;step*=2) {
        const int a=__shfl_up(p0,step,64),b=__shfl_up(p1,step,64);
        const int c=__shfl_up(p2,step,64),d=__shfl_up(p3,step,64);
        if(lane>=step){p0+=a;p1+=b;p2+=c;p3+=d;}
    }
    if(lane==63){sums[wave][0]=p0;sums[wave][1]=p1;sums[wave][2]=p2;sums[wave][3]=p3;}
    __syncthreads();
    for(int i=0;i<wave;++i){p0+=sums[i][0];p1+=sums[i][1];p2+=sums[i][2];p3+=sums[i][3];}
    if(tid<p.experts) {
        p.counts[tid]=own0;offsets[tid]=p0-own0;
        p.bounds[tid+1]=p0;p.tiles[tid+1]=p1;
        if(p.secondary)p.secondary[tid+1]=p2;
        if(p.tertiary)p.tertiary[tid+1]=p3;
    }
    if(tid==0){p.bounds[0]=0;p.tiles[0]=0;if(p.secondary)p.secondary[0]=0;if(p.tertiary)p.tertiary[0]=0;}
    __syncthreads();
    for(int pair=tid;pair<v.pair_count;pair+=Threads) {
        const int e=v.ids[pair];
        if(unsigned(e)<unsigned(p.experts)) {
            const int position=atomicAdd(cursors+e,1);
            v.ids_dst[offsets[e]+position]=pair;
        }
    }
    __syncthreads();
    if(tid<p.experts) {
        p.cursors[tid]=cursors[tid];
        for(int i=p1-own1;i<p1;++i)v.tile_experts[i]=tid;
        if(p.secondary)for(int i=p2-own2;i<p2;++i)v.secondary_experts[i]=tid;
        if(p.tertiary)for(int i=p3-own3;i<p3;++i)v.tertiary_experts[i]=tid;
    }
}
} // namespace

bool mfq_moe_route_map_fast_cuda(const MfqMoeRouteMapView& view,int device,hipStream_t stream) {
    // Bound single-CU work; larger maps retain the distributed count/scatter.
    if(view.pair_count<=0 || view.pair_count>16384 || view.scan.experts>1024 || !has_wave64(device))return false;
    if(view.pair_count<=2048 && view.scan.experts<=512)
        route_map_single<512><<<1,512,0,stream>>>(view);
    else route_map_single<1024><<<1,1024,0,stream>>>(view);
    return true;
}

bool mfq_moe_route_scan_parallel_cuda(const MfqMoeRouteScanView& view,int device,hipStream_t stream) {
    if(!has_wave64(device))return false;
    scan_counts_parallel<<<1,64,0,stream>>>(view);
    return true;
}
