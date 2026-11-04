#pragma once
#include "mfe_ffn_prepare.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mfe_expert_plan_test {
template<class T> struct Buffer {
    T* data=nullptr;
    std::size_t count;
    explicit Buffer(std::size_t size):count(size) {
        MFQ_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data),count*sizeof(T)));
    }
    ~Buffer(){cudaFree(data);}
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    void upload(const std::vector<T>& values,cudaStream_t stream) {
        MFQ_CUDA_CHECK(cudaMemcpyAsync(data,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::vector<T> read(cudaStream_t stream)const {
        std::vector<T> result(count);
        MFQ_CUDA_CHECK(cudaMemcpyAsync(result.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost,stream));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
        return result;
    }
};
struct Mapped {
    void* host=nullptr;
    void* device=nullptr;
    explicit Mapped(std::size_t bytes) {
        MFQ_CUDA_CHECK(cudaHostAlloc(&host,bytes,cudaHostAllocMapped));
        MFQ_CUDA_CHECK(cudaHostGetDevicePointer(&device,host,0));
        std::memset(host,0,bytes);
    }
    ~Mapped(){cudaFreeHost(host);}
};
inline void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
template<class Body> double time(Body body,cudaStream_t stream,int iterations=32,int replays=64) {
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    MFQ_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<iterations;++i)body();
    MFQ_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
    MFQ_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
    for(int i=0;i<8;++i)MFQ_CUDA_CHECK(cudaGraphLaunch(executable,stream));
    cudaEvent_t begin,end;
    MFQ_CUDA_CHECK(cudaEventCreate(&begin));
    MFQ_CUDA_CHECK(cudaEventCreate(&end));
    MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));
    for(int i=0;i<replays;++i)MFQ_CUDA_CHECK(cudaGraphLaunch(executable,stream));
    MFQ_CUDA_CHECK(cudaEventRecord(end,stream));
    MFQ_CUDA_CHECK(cudaEventSynchronize(end));
    float ms=0;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
    MFQ_CUDA_CHECK(cudaEventDestroy(begin));
    MFQ_CUDA_CHECK(cudaEventDestroy(end));
    MFQ_CUDA_CHECK(cudaGraphExecDestroy(executable));
    MFQ_CUDA_CHECK(cudaGraphDestroy(graph));
    return ms*1000/(iterations*replays);
}
inline void verify(bool timings,cudaStream_t stream) {
    using namespace mfq::cuda;
    int cases=0;
    for(int tokens:{2,8,16,32})for(int routes:{8,22})for(int pattern:{0,1,2}) {
        const int entries=tokens*routes,experts=768,capacity=tokens<=8?2:8;
        const int groups=107,width=2560,all_routes=routes+1;
        MfeFfnBatch b;
        b.tokens=tokens;
        b.routes=routes;
        b.experts=experts;
        b.input_width=width;
        b.scale_stride=27;
        b.token_group_capacity=capacity;
        b.token_groups=entries+(tokens+capacity-1)/capacity;
        MfePackedProjection shared;
        b.shared=&shared;
        // Prepare uses only the shared slot's presence.
        Buffer<int32_t> ids(entries),counts(b.token_groups),slots(b.token_groups*8),buckets(4*(b.token_groups+1));
        Buffer<int32_t> sorted(entries),ranks(entries),offsets(experts),sizes(experts),unique(entries),unique_count(1);
        Buffer<uint32_t> completion(tokens*all_routes*b.scale_stride);
        Buffer<__half> input(tokens*width);
        Buffer<int8_t> codes(tokens*groups*24);
        Buffer<float> scales(tokens*groups);
        Buffer<MfeInputQuantization> quantizers(1);
        quantizers.upload({{codes.data,scales.data,groups,24}},stream);
        std::vector<__half> values(tokens*width);
        for(std::size_t i=0;i<values.size();++i)values[i]=__float2half(float(int(i%47)-23)/31);
        input.upload(values,stream);
        std::vector<int32_t> selected(entries);
        for(int pair=0;pair<entries;++pair)
            selected[pair]=pattern==0?pair%routes:pattern==1?(pair*37)%experts:pair%7==0?-1:experts;
        ids.upload(selected,stream);
        b.ids=ids.data;
        b.input=input.data;
        b.completion=completion.data;
        b.token_group_counts=counts.data;
        b.token_group_slots=slots.data;
        b.token_group_buckets=buckets.data;
        const MfeExpertPlan plan{sorted.data,ranks.data,offsets.data,sizes.data,unique.data,unique_count.data};
        std::vector<int32_t> expected_counts,expected_slots;
        std::vector<int8_t> expected_codes;
        std::vector<float> expected_scales;
        for(bool planned:{false,true}) {
            MFQ_CUDA_CHECK(cudaMemsetAsync(completion.data,0x7f,completion.count*sizeof(uint32_t),stream));
            MFQ_CUDA_CHECK(cudaMemsetAsync(slots.data,0xff,slots.count*sizeof(int32_t),stream));
            mfe_ffn_prepare_warp(b,quantizers.data,1,groups,stream,planned?plan:MfeExpertPlan{});
            const auto actual_counts=counts.read(stream),actual_slots=slots.read(stream);
            if(!planned) {
                expected_counts=actual_counts;
                expected_slots=actual_slots;
                expected_codes=codes.read(stream);
                expected_scales=scales.read(stream);
            }else {
                require(actual_counts==expected_counts,"expert plan changed group lengths");
                require(actual_slots==expected_slots,"expert plan changed group slots or route ownership");
                require(codes.read(stream)==expected_codes && scales.read(stream)==expected_scales,
                    "expert plan changed quantization");
                const auto actual_ranks=ranks.read(stream),actual_sorted=sorted.read(stream);
                const auto actual_offsets=offsets.read(stream),actual_sizes=sizes.read(stream);
                int total=0;
                for(int expert=0;expert<experts;++expert) {
                    int rank=0;
                    for(int pair=0;pair<entries;++pair)if(selected[pair]==expert) {
                        require(actual_ranks[pair]==rank,"expert plan rank differs from original order");
                        require(actual_sorted[actual_offsets[expert]+rank]==pair,"expert-major pair list differs");
                        ++rank;
                    }
                    require(actual_sizes[expert]==rank,"expert plan count differs");
                    if(rank)++total;
                }
                require(unique_count.read(stream)[0]==total,"expert plan unique count differs");
                auto leaders=unique.read(stream);
                leaders.resize(total);
                std::sort(leaders.begin(),leaders.end());
                require(std::adjacent_find(leaders.begin(),leaders.end())==leaders.end(),"duplicate unique owner");
                for(int pair:leaders)require(actual_ranks[pair]==0,"unique lookup does not own first route");
                const auto done=completion.read(stream);
                require(std::all_of(done.begin(),done.end(),[](uint32_t v){return v==0;}),"completion not reset");
            }
            if(timings) {
                const double us=time([&]{mfe_ffn_prepare_warp(b,quantizers.data,1,groups,stream,
                    planned?plan:MfeExpertPlan{});},stream);
                std::cout<<"mfe_expert_prepare tokens="<<tokens<<" routes="<<routes<<" pattern="<<pattern
                    <<" planned="<<planned<<" us="<<us<<'\n';
            }
        }
        ++cases;
    }
    std::cout<<"mfe_expert_plan_group_cases="<<cases<<" PASS\n";
}
inline void verify_cache(bool timings,cudaStream_t stream) {
    using namespace mfq::cuda;
    int cases=0;
    for(int family:{1,2,3})for(int tokens:{2,8,16,32})for(int routes:{8,22}) {
        const int entries=tokens*routes,experts=41;
        if(entries>256)continue;
        // Preserve the existing cached-dispatch contract.
        const int groups=4,width=96,capacity=tokens<=8?2:8;
        MfeFfnBatch b;
        b.tokens=tokens;
        b.routes=routes;
        b.experts=experts;
        b.input_width=width;
        b.scale_stride=4;
        b.token_group_capacity=capacity;
        b.token_groups=entries;
        b.resident_plan_overlap=true;
        b.expert_activation=family==2?1:0;
        Buffer<int32_t> ids(entries),counts(entries),slots(entries*8),local(experts),mapping(3*experts);
        Buffer<int32_t> primary(3*experts),kinds(experts),indices(experts);
        Buffer<int32_t> sorted(entries),ranks(entries),offsets(experts),sizes(experts),unique(entries),unique_count(1);
        Buffer<uint32_t> completion(entries*4),aborted(1);
        Buffer<uint64_t> tags(experts),weight_storage(experts*3*8*8);
        Buffer<__half> input(tokens*width);
        Buffer<int8_t> codes(tokens*groups*24);
        Buffer<float> scales(tokens*groups);
        Buffer<MfeInputQuantization> quantizers(1);
        quantizers.upload({{codes.data,scales.data,groups,24}},stream);
        input.upload(std::vector<__half>(tokens*width,__float2half(.25f)),stream);
        std::vector<int32_t> selected(entries),locals(experts,-1),primaries(3*experts,0);
        for(int expert=0;expert<experts;++expert)if(expert%4==0)locals[expert]=0;
        for(int pair=0;pair<entries;++pair)selected[pair]=(pair*7)%experts;
        selected.back()=-1;
        ids.upload(selected,stream);
        local.upload(locals,stream);
        primary.upload(primaries,stream);
        b.ids=ids.data;
        b.input=input.data;
        b.completion=completion.data;
        b.kinds=kinds.data;
        b.transfer_index=indices.data;
        b.aborted=aborted.data;
        b.token_group_counts=counts.data;
        b.token_group_slots=slots.data;
        Mapped publication((entries+4)*sizeof(int32_t)),cache_image(experts*sizeof(MfeCachedExpert));
        auto* published=static_cast<int32_t*>(publication.device);
        auto* flags=reinterpret_cast<uint32_t*>(published+entries);
        auto* host_flags=static_cast<uint32_t*>(publication.host)+entries;
        b.plan_ready=flags;
        b.host_aborted=flags+2;
        std::vector<MfePackedProjection> prototypes(3*(1+experts));
        for(int p=0;p<3;++p) {
            b.projection_offsets[p]=p*(1+experts);
            b.projection_counts[p]=p==0 && b.expert_activation==1?0:1+experts;
            auto& d=prototypes[b.projection_offsets[p]];
            d.family=family;
            d.local_experts=1;
            d.expert_local=local.data;
            for(int expert=0;expert<experts;++expert) {
                auto& cached=prototypes[b.projection_offsets[p]+1+expert];
                cached.family=family;
                cached.local_experts=1;
            }
        }
        Buffer<MfePackedProjection> descriptors(prototypes.size());
        b.projections=descriptors.data;
        b.cohort_by_expert=mapping.data;
        MfeCachedDispatch cache;
        cache.mapped=static_cast<const MfeCachedExpert*>(cache_image.device);
        cache.primary_cohorts=primary.data;
        cache.projections=descriptors.data;
        cache.snapshot_tags=tags.data;
        cache.snapshot_ready=flags+1;
        cache.primary_incomplete=true;
        for(int p=0;p<3;++p)cache.primary_counts[p]=p==0 && b.expert_activation==1?0:1;
        const MfeExpertPlan plan{sorted.data,ranks.data,offsets.data,sizes.data,unique.data,unique_count.data};
        struct Result {
            std::vector<int32_t> mapping,kinds,counts,slots;
            std::vector<uint64_t> tags;
            std::vector<MfePackedProjection> descriptors;
        };
        std::vector<Result> reference;
        for(bool planned:{false,true}) {
            descriptors.upload(prototypes,stream);
            MFQ_CUDA_CHECK(cudaMemsetAsync(tags.data,0,tags.count*sizeof(uint64_t),stream));
            for(int step=0;step<6;++step) {
                std::memset(publication.host,0,(entries+4)*sizeof(int32_t));
                host_flags[2]=step==5;
                std::vector<MfeCachedExpert> image(experts);
                for(int expert=0;expert<experts;++expert) {
                    const unsigned mask=step==0 || step==4?0:expert%3==0?3:7;
                    image[expert].ready_mask=(step>=5?24:step>=3?16:8)|mask;
                    for(int p=0;p<3;++p)for(int f=0;f<8;++f)
                        image[expert].fields[p][f]=reinterpret_cast<uint64_t>(weight_storage.data+
                            ((expert*3+p)*8+f)*8+step);
                }
                std::memcpy(cache_image.host,image.data(),image.size()*sizeof(MfeCachedExpert));
                mapping.upload(primaries,stream);
                MFQ_CUDA_CHECK(cudaMemsetAsync(kinds.data,0xff,kinds.count*sizeof(int32_t),stream));
                MFQ_CUDA_CHECK(cudaMemsetAsync(slots.data,0xff,slots.count*sizeof(int32_t),stream));
                const auto body=[&]{mfe_ffn_prepare_publish_cached(b,quantizers.data,1,groups,published,
                    flags,cache,stream,planned?plan:MfeExpertPlan{});};
                body();
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
                while(!*reinterpret_cast<volatile uint32_t*>(host_flags))
                    require(std::chrono::steady_clock::now()<deadline,"route ready publication timed out");
                std::atomic_thread_fence(std::memory_order_acquire);
                require(!std::memcmp(publication.host,selected.data(),entries*sizeof(int32_t)),
                    "ready publication preceded original route IDs");
                while(!*reinterpret_cast<volatile uint32_t*>(host_flags+1))
                    require(std::chrono::steady_clock::now()<deadline,"snapshot ready publication timed out");
                std::atomic_thread_fence(std::memory_order_acquire);
                // Once released, the host may recycle every mapped descriptor.
                std::memset(cache_image.host,0xa5,image.size()*sizeof(MfeCachedExpert));
                MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                Result result{mapping.read(stream),kinds.read(stream),counts.read(stream),slots.read(stream),
                    tags.read(stream),descriptors.read(stream)};
                if(!planned)reference.push_back(result);
                else {
                    const auto& old=reference[step];
                    require(result.mapping==old.mapping && result.kinds==old.kinds,"cached route ownership changed");
                    require(result.counts==old.counts && result.slots==old.slots,"cached grouping changed");
                    require(result.tags==old.tags,"snapshot tags changed");
                    require(!std::memcmp(result.descriptors.data(),old.descriptors.data(),
                        result.descriptors.size()*sizeof(MfePackedProjection)),"snapshot fields changed");
                }
                std::vector<bool> active(experts);
                for(int expert:selected)if(expert>=0)active[expert]=true;
                for(int expert=0;expert<experts;++expert)if(active[expert]) {
                    const int expected=step==5?-1:expert%4==0?1:step==0 || step==4?2:expert%3==0?6:1;
                    if(result.kinds[expert]!=expected)
                        throw std::runtime_error("cached kind differs: step="+std::to_string(step)+
                            " expert="+std::to_string(expert)+" actual="+std::to_string(result.kinds[expert])+
                            " expected="+std::to_string(expected));
                    for(int p=0;p<3;++p) {
                        if(p==0 && b.expert_activation==1)continue;
                        const bool hit=expert%4!=0 && step!=0 && step!=4 && (expert%3!=0 || p<2);
                        const int expected_pool=hit?1+expert:0;
                        require(result.mapping[p*experts+expert]==expected_pool,"cache cohort ownership differs");
                        if(hit) {
                            const int source_step=step==2?1:step;
                            const auto* expected_field=weight_storage.data+((expert*3+p)*8)*8+source_step;
                            require(result.descriptors[b.projection_offsets[p]+expected_pool].fields[0]==expected_field,
                                "snapshot generation refresh or reuse failed");
                        }
                    }
                }
                if(timings && step==2) {
                    std::memcpy(cache_image.host,image.data(),image.size()*sizeof(MfeCachedExpert));
                    const double us=time(body,stream);
                    std::cout<<"mfe_expert_cache family="<<family<<" tokens="<<tokens<<" routes="<<routes
                        <<" planned="<<planned<<" us="<<us<<'\n';
                }
                ++cases;
            }
        }
    }
    std::cout<<"mfe_expert_plan_cache_cases="<<cases<<" PASS\n";
}
} // namespace mfe_expert_plan_test
