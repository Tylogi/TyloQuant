#include "moe_residency.h"
#include "runtime/decode_window.h"
#include "runtime/moe_pipeline.h"
#include "runtime/native_dma_batch.h"
#include "storage/moe_cached_source_internal.h"
#include "mfq/cpu_expert_pool.h"
#include "mfq/runtime.h"
#include "host_memory_copy.h"
#include "mfq_cuda_resident_allocations.h"
#include "mfq/moe_residency_ranker.h"
#include <algorithm>
#include <array>
#include <exception>
#include <map>
#include <set>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>
#include <fstream>
#endif

namespace tb=mfq_tensor_backend;
namespace {
class PhaseClock {
    std::atomic<std::uint64_t>& total_;
    std::atomic<std::uint64_t>* decode_;
    std::chrono::steady_clock::time_point start_=std::chrono::steady_clock::now();
public:
    PhaseClock(std::atomic<std::uint64_t>& total,std::atomic<std::uint64_t>* decode)
        :total_(total),decode_(decode) {}
    ~PhaseClock() {
        const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-start_).count();
        total_.fetch_add(elapsed,std::memory_order_relaxed);
        if(decode_)decode_->fetch_add(elapsed,std::memory_order_relaxed);
    }
};
bool native_residency_dma_enabled() {
    const auto* residency=std::getenv("MFQ_MOE_RESIDENCY_NATIVE_DMA");
    if(residency)return residency[0]=='1' && mfq::cuda::NativeDmaBatch::available();
#if defined(_WIN32)
    // Follow the measured Windows transfer default; preserve the global opt-out.
    const auto* pipeline=std::getenv("MFQ_MOE_NATIVE_DMA_BATCH");
    return (!pipeline || std::atoi(pipeline)>0) && mfq::cuda::NativeDmaBatch::available();
#else
    return false;
#endif
}
MoeResidencyManager::MemorySample process_memory() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS process{};MEMORYSTATUSEX system{};system.dwLength=sizeof(system);
    if(!K32GetProcessMemoryInfo(GetCurrentProcess(),&process,sizeof(process)) || !GlobalMemoryStatusEx(&system))
        throw std::runtime_error("cannot measure physical memory for expert residency transaction");
    return {process.WorkingSetSize,system.ullAvailPhys,system.ullAvailPageFile};
#elif defined(__linux__)
    std::uint64_t virtual_pages=0,resident_pages=0;std::ifstream status("/proc/self/statm");
    struct sysinfo system{};
    if(!(status>>virtual_pages>>resident_pages) || sysinfo(&system)!=0)
        throw std::runtime_error("cannot measure physical memory for expert residency transaction");
    return {resident_pages*static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE)),
        std::uint64_t(system.freeram+system.bufferram)*system.mem_unit};
#else
    throw std::runtime_error("physical memory measurement unavailable on this CUDA host");
#endif
}
}
struct MoeResidencyManager::Impl {
    struct Move {
        int incoming_source=0,outgoing_source=0,cohort=0,incoming=0,outgoing=0,slot=-1;
        int incoming_bundle=0,incoming_projection=0;
        MoeHostExpertCache::Lease lease;
        mfq::MoeCacheSlotBook::Replacement replacement;
        // Canonical expert formats have at most seven physical fields. Keep
        // transaction metadata with the move instead of allocating three tiny
        // vectors for every promoted projection on every adaptation round.
        std::array<int64_t,8> offsets{},preserve_offsets{};
        std::array<const void*,8> loan_fields{};
        MoeHostExpertCache::Lease shadow_owner;
        std::array<const uint8_t*,8> shadow_fields{};
        bool shadow_indexed=false;
        std::size_t field_count=0;
        int64_t uncached_bytes=0;
        std::size_t touched_fields=0;
        std::size_t backed_fields=0;
        MoeGpuArena* cached_arena=nullptr;
        int cached_slot=-1;
        int prefill_ram_slot=-1;
    };
    struct Map {int source=0,cohort=0;std::vector<int32_t> next;};
    MoeExpertCache* cache;
    Observer observer;
    MemoryProbe memory_probe;
    int first=0,last=0;
    int64_t rounds=0;
    int pending_tokens=0;
    std::vector<std::vector<float>> heat;
    std::vector<std::vector<uint8_t>> projection_marks;
    const float heat_retention=mfq::cuda::runtime_options::residency_heat_retention();
    const int interval=mfq::cuda::runtime_options::residency_interval();
    const bool exchange_defaults=[this] {
        // The measured combination needs fixed GPU routing, registered RAM,
        // transfer-cache reuse and no background CPU calibration. Keep it a
        // coherent default: an explicit opt-out restores legacy defaults for
        // the other switches, while every explicit setting remains respected.
        if(!cache || !cache->complete_residency_ || !cache->config_.moe_residency_adapt ||
           !cache->config_.moe_residency_warm || !cache->config_.moe_residency_projection_heat ||
           !cache->config_.moe_direct_ram || !cache->pipeline_ram_registered_bytes_ ||
           !cache->pipeline_transfer_cache_ || !cache->config_.moe_ram_pcie_fraction ||
           cache->ram_pcie_fraction_!=1.0 || cache->pipeline_shared_cpu_cost_)return false;
        for(const auto* name:{"MFQ_MOE_RESIDENCY_DIRECT_EXCHANGE","MFQ_MOE_RESIDENCY_EXPERT_FENCE",
                "MFQ_MOE_RESIDENCY_CACHED_REFILL","MFQ_MOE_RESIDENCY_BATCHED",
                "MFQ_MOE_RESIDENCY_ASYNC_PUBLISH","MFQ_MOE_RESIDENCY_WINDOW_PREPARE"}) {
            const auto* value=std::getenv(name);if(value && value[0]!='1')return false;
        }
        const auto* promotion=std::getenv("MFQ_MOE_RESIDENCY_PROMOTE_TRANSFER");
        return !promotion || promotion[0]=='2';
    }();
    const bool promote_transfer=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_PROMOTE_TRANSFER");
        return flag?(flag[0]=='1' || flag[0]=='2'):exchange_defaults;
    }();
    const bool reuse_transfer=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_PROMOTE_TRANSFER");
        return flag?flag[0]=='2':exchange_defaults;
    }();
    std::vector<std::vector<float>> observations;
    struct SpeculativeRoutes {
        std::vector<int32_t> ids;
        std::vector<uint64_t> heat_seen;
        int rows=0;
    };
    std::vector<SpeculativeRoutes> speculative_routes;
    std::vector<mfq::MoeCacheKey> speculative_host_keys;
    int speculative_rows=0;
    // One streamed expert serves every matching row of a verification window.
    // Rank its avoided transfer once, rather than counting each routed token.
    const int speculative_heat_mode=[] {
        const auto* option=std::getenv("MFQ_MOE_MTP_WINDOW_HEAT");
        return option?std::clamp(std::atoi(option),0,2):2;
    }();
    const bool speculative_window_heat=speculative_heat_mode!=0;
    bool traced_speculative_heat=false;
    std::atomic<std::uint64_t> speculative_route_windows{0},speculative_routes_committed{0};
    std::atomic<std::uint64_t> speculative_routes_discarded{0},speculative_route_cancels{0};
    std::atomic<std::uint64_t> speculative_heat_routes{0};
    std::unique_ptr<mfq::MoeResidencyRanker> ranking;
    struct TransferStage {std::string key;MoeGpuArena* arena=nullptr;int projection=0;};
    std::vector<std::vector<TransferStage>> transfer_stages;
    bool direct=false;
    bool batched=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_BATCHED");
        return flag?flag[0]!='0':exchange_defaults;
    }();
    bool async_publish=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_ASYNC_PUBLISH");
        return flag?flag[0]!='0':exchange_defaults;
    }();
    bool window_prepare=[] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_WINDOW_PREPARE");
        return !flag || flag[0]!='0';
    }();
    bool expert_fence=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_EXPERT_FENCE");
        return flag?flag[0]=='1':exchange_defaults;
    }();
    bool cached_refill=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_CACHED_REFILL");
        return flag?flag[0]=='1':exchange_defaults;
    }();
    bool direct_exchange=[this] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_DIRECT_EXCHANGE");
        return flag?flag[0]=='1':exchange_defaults;
    }();
    std::uint64_t refill_dma_min=[] {
        const auto* value=std::getenv("MFQ_MOE_RESIDENCY_REFILL_DMA_MIN_BYTES");
        return value?std::uint64_t(std::max<long long>(0,std::atoll(value))):std::uint64_t(0);
    }();
    std::uint64_t refill_chunk_bytes=[] {
        const auto* value=std::getenv("MFQ_MOE_RESIDENCY_REFILL_CHUNK_BYTES");
        if(!value)return std::uint64_t(65536);
        const auto parsed=std::atoll(value);
        return parsed<=0?std::uint64_t(0):std::uint64_t(std::clamp<long long>(parsed,4096,1048576))&~std::uint64_t(15);
    }();
    bool parallel_host=[] {
        const auto* flag=std::getenv("MFQ_MOE_RESIDENCY_PARALLEL_HOST");
        return flag && flag[0]!='0';
    }();
    const bool retire_kv_duplicates=[] {
        const auto* flag=std::getenv("MFQ_MOE_RETIRE_KV_DUPLICATES");
        return !flag || flag[0]!='0';
    }();
    void retire_promoted_kv() {
        if(!retire_kv_duplicates || !cache->kv_expert_loan_)return;
        for(const auto& move:pending)cache->kv_expert_loan_->retire_expert(
            move.incoming_source,move.incoming,
            int(cache->sources_[move.incoming_source]->cohorts_[move.cohort].bytes_per_expert.size()));
    }
    std::atomic<std::uint64_t> native_dma_batches{0},native_dma_copies{0},native_dma_fallbacks{0};
    std::atomic<std::uint64_t> shadow_backup_bytes{0},shadow_backup_fields{0};
    std::atomic<std::uint64_t> shadow_index_rebuilds{0},shadow_index_matches{0},shadow_index_backup_bytes{0};
    struct ShadowIndex {
        struct Row {mfq::MoeCacheKey key;int64_t row;};
        std::weak_ptr<const MoeHostExpert> owner;
        uint64_t generation=0;
        int64_t slots=0;
        std::vector<Row> rows;
        static bool less(const mfq::MoeCacheKey& a,const mfq::MoeCacheKey& b) {
            if(a.source!=b.source)return a.source<b.source;
            if(a.cohort!=b.cohort)return a.cohort<b.cohort;
            return a.expert<b.expert;
        }
    };
    std::map<const MoeGpuArena*,ShadowIndex> shadow_indices;
    bool shadow_index_reported=false;
    int64_t find_shadow_row(const MoeGpuArena& arena,const MoeHostExpertCache::Lease& owner,
                           const mfq::MoeCacheKey& key) {
        try {
            auto& index=shadow_indices[&arena];
            const auto& identities=owner->weights.expert_local;const auto slots=identities.size(0);
            const auto* data=identities.data_ptr<int32_t>();
            if(index.owner.lock()!=owner || index.generation!=arena.prefill_backup_generation || index.slots!=slots) {
                index.rows.clear();index.rows.reserve(std::size_t(slots));
                for(int64_t row=0;row<slots;++row) {
                    const mfq::MoeCacheKey candidate{data[row*3],data[row*3+1],data[row*3+2]};
                    if(candidate.source>=0 && candidate.cohort>=0 && candidate.expert>=0)index.rows.push_back({candidate,row});
                }
                std::sort(index.rows.begin(),index.rows.end(),[](const auto& a,const auto& b){return ShadowIndex::less(a.key,b.key);});
                index.owner=owner;index.generation=arena.prefill_backup_generation;index.slots=slots;
                ++shadow_index_rebuilds;
            }
            const auto found=std::lower_bound(index.rows.begin(),index.rows.end(),key,
                [](const auto& row,const auto& key){return ShadowIndex::less(row.key,key);});
            if(found==index.rows.end() || !(found->key==key))return -1;
            const auto row=found->row;
            // Validate the live metadata as well as the cached generation.
            if(data[row*3]!=key.source || data[row*3+1]!=key.cohort || data[row*3+2]!=key.expert)return -1;
            ++shadow_index_matches;return row;
        }catch(const std::bad_alloc&) {return -1;}
    }
    std::atomic<std::uint64_t> observed_routes{0},scheduled_rounds{0},candidate_bundles{0};
    std::atomic<std::uint64_t> planned_bundles{0},committed_bundles{0},memory_rejections{0};
    std::atomic<std::uint64_t> host_pack_bytes{0},direct_upload_bytes{0},backup_peak_bytes{0},shared_backup_bytes{0};
    std::atomic<std::uint64_t> candidate_projections{0},planned_projections{0},committed_projections{0},projection_memory_rejections{0};
    std::atomic<std::uint64_t> staging_budget_skips{0};
    std::atomic<std::uint64_t> ssd_load_requests{0},ssd_load_bytes{0};
    std::atomic<std::uint64_t> ssd_gpu_image_bytes{0},ssd_gpu_images{0},ssd_gpu_image_ns{0},ssd_gpu_image_copies{0};
    std::atomic<std::uint64_t> plan_ns{0},decode_plan_ns{0};
    std::uint64_t last_plan_ns=0,last_staging_budget_skips=0;
    std::atomic<std::uint64_t> prepare_ns{0},join_ns{0},dma_wait_ns{0},map_ns{0},host_exchange_ns{0},apply_ns{0};
    std::atomic<std::uint64_t> decode_prepare_ns{0},decode_join_ns{0},decode_dma_wait_ns{0},decode_map_ns{0},decode_host_exchange_ns{0},decode_apply_ns{0};
    std::atomic<std::uint64_t> backup_copies{0},upload_copies{0};
    std::atomic<std::uint64_t> parallel_host_bytes{0},parallel_host_batches{0};
    std::atomic<std::uint64_t> batch_device_bytes{0},batch_descriptor_copies{0};
    std::atomic<std::uint64_t> async_publish_ns{0},decode_async_publish_ns{0};
    std::atomic<std::uint64_t> window_prepares{0};
    std::atomic<std::uint64_t> cached_copy_bytes{0},cached_copy_fields{0};
    std::atomic<std::uint64_t> expert_fence_waits{0};
    std::atomic<std::uint64_t> cached_copy_batches{0};
    std::atomic<std::uint64_t> expert_fence_wait_ns{0},backup_submit_ns{0},refill_submit_ns{0};
    std::atomic<std::uint64_t> cached_copy_dma_bytes{0},cached_copy_dma_fields{0};
    std::atomic<std::uint64_t> direct_exchange_rounds{0},direct_exchange_bytes{0},direct_exchange_ns{0};
    std::atomic<std::uint64_t> cached_copy_descriptors{0},cached_copy_blocks{0};
    std::atomic<std::uint64_t> kv_refill_bytes{0};
    unsigned traced_decode_waits=0;
    bool last_gpu_preserve=false;
    int64_t last_preserve_bytes=0;
    int64_t last_backup_bytes=0,last_small_backup_bytes=0;
    std::size_t last_backup_fields=0,last_small_backup_fields=0;
    const int64_t small_backup_limit=[] {
        const auto* value=std::getenv("MFQ_MOE_RESIDENCY_SMALL_BACKUP_BYTES");
        return value?std::clamp<int64_t>(std::atoll(value),0,1048576):int64_t(65536);
    }();
    std::unique_ptr<mfq::cuda::HostBuffer> backup_descriptors;
    mfq::MoeCacheMappedCopyDescriptor* backup_descriptors_device=nullptr;
    uint8_t* backup_ram_device=nullptr;
    struct PrefillRoutes {int tokens=0,routes=0;};
    std::vector<PrefillRoutes> prefill_routes;
    tb::Tensor prefill_route_device;
    std::unique_ptr<mfq::cuda::HostBuffer> prefill_route_host;
    int prefill_route_stride=0;
    cudaEvent_t prefill_done=nullptr;
    bool prefill_pending=false;
    std::vector<Move> pending;
    std::size_t pending_complete_bundles=0;
    std::vector<Map> maps;
    tb::Tensor upload,backup;
    std::unique_ptr<mfq::cuda::HostBuffer> map_upload,map_descriptors;
    uint8_t* map_upload_device=nullptr;
    mfq::MoeCacheMappedCopyDescriptor* map_descriptors_device=nullptr;
    tb::Tensor device_work,copy_descriptors,device_descriptors;
    int work_tokens=0;
    cudaStream_t stream=nullptr,backup_stream=nullptr;
    cudaEvent_t backup_done=nullptr;
    std::vector<cudaEvent_t> preserve_events;
    cudaEvent_t done=nullptr;
    cudaEvent_t window_done=nullptr;
    std::unique_ptr<mfq::cuda::Event> retired_dma_done;
    std::vector<MoeHostExpertCache::Lease> retired_dma_leases;
    std::unique_ptr<mfq::cuda::HostBuffer> expert_done_host;
    uint32_t* expert_done_device=nullptr;
    cudaStream_t expert_compute_stream=nullptr;
    bool pending_window=false;
    bool pending_expert_fence=false;
    static constexpr std::size_t refill_capacity=65535;
    std::unique_ptr<mfq::cuda::HostBuffer> refill_descriptors;
    mfq::MoeCacheMappedCopyDescriptor* refill_device=nullptr;
    std::thread worker;
    std::mutex worker_mutex;
    std::condition_variable work_available,work_finished;
    std::atomic_bool work_pending{false};
    bool work_requested=false,work_ready=false,stopping=false,window_inflight=false;
    const bool predictor_overlap=[] {
        const auto* option=std::getenv("MFQ_MOE_PREDICTOR_EXCHANGE_OVERLAP");
        return !option || option[0]!='0';
    }();
    bool predictor_overlap_scope=false,predictor_overlap_allowed=false;
    bool ssd_predictor_isolated=false;
    bool ssd_copies_permitted=false;
    std::size_t ssd_stage_capacity=0;
    std::optional<mfq::cuda::KvExpertLoan::ExchangeReadGuard> exchange_reads;
    std::atomic<std::uint64_t> predictor_overlap_rounds{0},predictor_overlap_serves{0};
    std::exception_ptr error;
    explicit Impl(MoeExpertCache* value,Observer observe,MemoryProbe probe)
        :cache(value),observer(std::move(observe)),memory_probe(probe?std::move(probe):process_memory) {}
    void observe(const char* phase,std::size_t index) {if(observer)observer(phase,index);}
    void upload_maps() {
        if(maps.empty())return;
        if(!map_upload) {
            std::size_t capacity=0,count=0;
            for(const auto& source:cache->sources_)for(const auto& cohort:source->cohorts_)
                {capacity+=cohort.host_map.size()*sizeof(int32_t);++count;}
            map_upload=std::make_unique<mfq::cuda::HostBuffer>(capacity,true);
            map_descriptors=std::make_unique<mfq::cuda::HostBuffer>(count*sizeof(mfq::MoeCacheMappedCopyDescriptor),true);
            MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&map_upload_device),map_upload->data(),0));
            MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&map_descriptors_device),map_descriptors->data(),0));
        }
        // Publish every small map in one launch after the old graph retires.
        // One H2D submission per projection otherwise costs more than the
        // entire map payload, especially during the first routing changes.
        auto* descriptors=static_cast<mfq::MoeCacheMappedCopyDescriptor*>(map_descriptors->data());
        std::size_t offset=0;
        for(std::size_t i=0;i<maps.size();++i) {
            const auto& map=maps[i];auto& cohort=cache->sources_[map.source]->cohorts_[map.cohort];
            const auto bytes=map.next.size()*sizeof(int32_t);
            if(offset+bytes>map_upload->size())throw std::logic_error("residency map staging exceeded topology");
            auto* source=static_cast<uint8_t*>(map_upload->data())+offset;
            descriptors[i]={reinterpret_cast<uint64_t>(cohort.active.expert_local.data_ptr<int32_t>()),
                reinterpret_cast<uint64_t>(map_upload_device+offset),bytes};
            std::memcpy(source,map.next.data(),bytes);offset+=bytes;
            observe("map",i);
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        mfq::moe_cache_mapped_gather_cuda(map_descriptors_device,int(maps.size()),1,stream);
    }
    unsigned projection_mask(int bundle,int projection) const {
        if(projection<0)return 7;
        const auto& sources=cache->pipeline_bundles_[bundle];
        return projection==0 && sources[0]==sources[1]?3:1u<<projection;
    }
    void clear_projection_marks() {
        for(auto& row:projection_marks)std::fill(row.begin(),row.end(),uint8_t(0));
    }
    bool mark_projection_complete(int bundle,int expert,int projection) {
        auto& mask=projection_marks[bundle][expert];const auto before=mask;
        mask|=projection_mask(bundle,projection);
        return before!=7 && mask==7;
    }
    std::size_t complete_pending_bundles() {
        clear_projection_marks();std::size_t count=0;
        for(const auto& move:pending)
            count+=mark_projection_complete(move.incoming_bundle,move.incoming,move.incoming_projection);
        return count;
    }
    struct CachedRefill {
        Impl& s;
        const bool native_ssd_dma;
        std::vector<mfq::cuda::DmaCopy> native_copies;
        std::size_t count=0;
        std::uint64_t largest=0;
        explicit CachedRefill(Impl& owner):s(owner),
            native_ssd_dma(s.cache->ssd_experts() && native_residency_dma_enabled()) {}
        void copy(uint8_t* destination,const uint8_t* source,std::uint64_t bytes) {
            const auto chunk=s.refill_chunk_bytes?s.refill_chunk_bytes:bytes;
            const auto pieces=1+(bytes-1)/chunk;
            if(s.cache->ssd_experts()) {
                // Copy-engine work can pass a graph's expert publication wait.
                if(native_ssd_dma)native_copies.push_back({destination,source,std::size_t(bytes)});
                else MFQ_CUDA_CHECK(cudaMemcpyAsync(destination,source,std::size_t(bytes),cudaMemcpyDeviceToDevice,s.stream));
                s.cached_copy_dma_bytes+=bytes;++s.cached_copy_dma_fields;
            } else if(s.cached_refill && (!s.refill_dma_min || bytes<s.refill_dma_min) && pieces<=refill_capacity-count) {
                auto* descriptors=static_cast<mfq::MoeCacheMappedCopyDescriptor*>(s.refill_descriptors->data());
                for(std::uint64_t begin=0;begin<bytes;begin+=chunk) {
                    const auto size=std::min(chunk,bytes-begin);
                    descriptors[count++]={reinterpret_cast<uint64_t>(destination+begin),reinterpret_cast<uint64_t>(source+begin),size};
                    largest=std::max(largest,size);
                }
            } else {
                MFQ_CUDA_CHECK(cudaMemcpyAsync(destination,source,std::size_t(bytes),cudaMemcpyDeviceToDevice,s.stream));
                s.cached_copy_dma_bytes+=bytes;++s.cached_copy_dma_fields;
            }
            s.cached_copy_bytes+=bytes;++s.cached_copy_fields;
        }
        bool submit() {
            if(!native_copies.empty()) {
                if(mfq::cuda::NativeDmaBatch::submit(native_copies,s.stream,1,
                        [](const void* pointer,std::size_t){return reinterpret_cast<CUdeviceptr>(pointer);})) {
                    ++s.native_dma_batches;s.native_dma_copies+=native_copies.size();
                } else {
                    ++s.native_dma_fallbacks;
                    for(const auto& copy:native_copies)MFQ_CUDA_CHECK(cudaMemcpyAsync(
                        copy.destination,copy.source,copy.bytes,cudaMemcpyDeviceToDevice,s.stream));
                }
            }
            if(!count)return !native_copies.empty();
            // Chunked descriptors receive one block each. Work grows with
            // actual bytes, avoiding maximum-size grids for tiny fields.
            const int blocks=s.refill_chunk_bytes?1:static_cast<int>(std::min<std::uint64_t>(128,(largest+4095)/4096));
            std::atomic_thread_fence(std::memory_order_seq_cst);
            mfq::moe_cache_mapped_gather_cuda(s.refill_device,int(count),blocks,s.stream);
            ++s.cached_copy_batches;s.cached_copy_descriptors+=count;s.cached_copy_blocks+=count*blocks;
            return true;
        }
    };
};

MoeResidencyManager::MemorySample MoeResidencyManager::sample_host_memory() {return process_memory();}

MoeResidencyManager::MoeResidencyManager(MoeExpertCache* cache,Observer observer,MemoryProbe memory_probe)
    :impl_(std::make_unique<Impl>(cache,std::move(observer),std::move(memory_probe))) {
    if(!cache || !cache->pipeline_ready() || cache->pipeline_bundles_.empty())
        throw std::invalid_argument("adaptive tier requires resident expert bundles");
    auto& s=*impl_;s.first=std::numeric_limits<int>::max();s.last=std::numeric_limits<int>::min();
    std::cerr<<"moe_residency_policy retention="<<s.heat_retention<<" interval="<<s.interval
        <<" projection_heat="<<(cache->config_.moe_residency_warm && cache->config_.moe_residency_projection_heat)<<'\n';
    s.transfer_stages.resize(cache->sources_.size());
    for(const auto& bundle:cache->pipeline_bundles_) {
        const auto& source=cache->sources_.at(bundle[0]);s.heat.emplace_back(std::size_t(source->n_experts()),0.0f);
        s.projection_marks.emplace_back(std::size_t(source->n_experts()),uint8_t(0));
        s.observations.emplace_back(std::size_t(source->n_experts()),0.0f);
        s.speculative_routes.emplace_back();
        s.speculative_routes.back().heat_seen.resize(
            (std::size_t(source->n_experts())+63)/64);
        for(int id:bundle)if(cache->sources_.at(id)->layer_id_!=source->layer_id_ ||
                cache->sources_.at(id)->n_experts()!=source->n_experts())
            throw std::invalid_argument("adaptive expert bundle geometry differs");
        if(source->layer_id_>=0) {
            s.first=std::min(s.first,source->layer_id_);s.last=std::max(s.last,source->layer_id_);
        }
        for(int p=0;p<3;++p) {
            if(p==1 && bundle[0]==bundle[1])continue;
            const auto& projection=*cache->sources_[bundle[p]];
            auto& stages=s.transfer_stages[bundle[p]];
            for(const auto& cohort:projection.cohorts_)
                stages.push_back({std::to_string(p)+":"+cohort.arena->signature,nullptr,p});
        }
    }
    if(s.first==std::numeric_limits<int>::max())for(const auto& bundle:cache->pipeline_bundles_) {
        const auto layer=cache->sources_[bundle[0]]->layer_id_;
        s.first=std::min(s.first,layer);s.last=std::max(s.last,layer);
    }
    using Expert=mfq::MoeResidencyRanker::Expert;
    std::vector<std::vector<Expert>> layout;
    if(cache->config_.moe_residency_warm && cache->config_.moe_residency_projection_heat) {
        std::map<std::pair<MoeGpuArena*,int>,std::vector<Expert>> groups;
        for(std::size_t b=0;b<cache->pipeline_bundles_.size();++b) {
            const auto& bundle=cache->pipeline_bundles_[b];
            for(int e=0;e<cache->sources_[bundle[0]]->n_experts();++e)for(int p=0;p<3;++p) {
                // Combined Gate/Up is one physical projection. Rank it once
                // alongside Down so predictor experts can adapt as well.
                if(p==1 && bundle[0]==bundle[1])continue;
                const auto& source=*cache->sources_[bundle[p]];
                // Field-major prefill images keep each source's slot count
                // and geometry. Exchanges within that source retain the image.
                groups[{source.cohorts_[source.expert_to_cohort_[e]].arena,
                    cache->pipeline_prefill_layer_layout_?source.id_:-1}].push_back({int(b),e,p});
            }
        }
        for(auto& item:groups)layout.push_back(std::move(item.second));
    } else {
        std::map<std::array<std::uintptr_t,4>,std::vector<Expert>> groups;
        for(std::size_t b=0;b<cache->pipeline_bundles_.size();++b) {
            const auto& bundle=cache->pipeline_bundles_[b];
            if(bundle[0]==bundle[1])continue;
            for(int e=0;e<cache->sources_[bundle[0]]->n_experts();++e) {
                std::array<std::uintptr_t,4> signature{};
                signature[3]=cache->pipeline_prefill_layer_layout_?b+1:0;
                for(int p=0;p<3;++p) {
                    const auto& source=*cache->sources_[bundle[p]];
                    signature[p]=reinterpret_cast<std::uintptr_t>(source.cohorts_[source.expert_to_cohort_[e]].arena);
                }
                groups[signature].push_back({int(b),e,-1});
            }
        }
        for(auto& item:groups)layout.push_back(std::move(item.second));
    }
    s.ranking=std::make_unique<mfq::MoeResidencyRanker>(std::move(layout));
    if(s.cached_refill) {
        // Allocate before any consumer graph runs. Only descriptors are mapped;
        // weights stay in their original GPU fields. This avoids a per-round
        // allocation or descriptor upload delaying the early D2H backups.
        s.refill_descriptors=std::make_unique<mfq::cuda::HostBuffer>(
            Impl::refill_capacity*sizeof(mfq::MoeCacheMappedCopyDescriptor),true);
        MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&s.refill_device),s.refill_descriptors->data(),0));
    }
    if(s.small_backup_limit && cache->pipeline_ram_registered_bytes_ && cache->pipeline_ram_complement_) {
        s.backup_descriptors=std::make_unique<mfq::cuda::HostBuffer>(
            Impl::refill_capacity*sizeof(mfq::MoeCacheMappedCopyDescriptor),true);
        MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&s.backup_descriptors_device),s.backup_descriptors->data(),0));
        MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&s.backup_ram_device),cache->pipeline_ram_complement_->data_ptr(),0));
    }
    if(native_residency_dma_enabled() && cache->pipeline_ram_registered_bytes_ && !cache->pipeline_ram_alias_)
        MFQ_CUDA_CHECK(cudaHostGetDevicePointer(&cache->pipeline_ram_alias_,cache->pipeline_ram_complement_->data_ptr(),0));
    if(cache->ssd_experts()) {
        // cudaHostAlloc can synchronize with a graph waiting for host-side
        // expert publication. Reserve both directions before any replay and
        // keep adaptation batches within this fixed workspace thereafter.
        const auto memory=s.memory_probe();
        auto capacity=std::min<std::uint64_t>({cache->host_experts_->stats().budget_bytes/4,
            std::uint64_t(moe_ssd_residency_stage_limit_bytes),memory.available_host_bytes()/8});
        if(cache->config_.moe_host_physical_bytes)
            capacity=std::min(capacity,memory.resident_bytes>=cache->config_.moe_host_physical_bytes?std::uint64_t(0):
                (cache->config_.moe_host_physical_bytes-memory.resident_bytes)/8);
        const auto options=tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true);
        for(capacity=capacity/4096*4096;capacity;capacity=capacity/2/4096*4096) {
            try {
                // Optional exchange buffers must leave pinned capacity for
                // the decode/verification packets prepared after this stage.
                // The driver may impose a tighter quota than free host RAM.
                auto headroom=tb::empty({std::int64_t(std::max<std::uint64_t>(
                    std::uint64_t(256)<<20,2*capacity))},options);
                s.backup=tb::empty({std::int64_t(capacity)},options);
                s.upload=tb::empty({std::int64_t(capacity)},options);
                s.ssd_stage_capacity=std::size_t(capacity);break;
            } catch(const MfqResourceExhausted&) {s.backup={};s.upload={};}
        }
    }
    MFQ_CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream,cudaStreamNonBlocking));
    const auto event=cudaEventCreateWithFlags(&s.done,cudaEventDisableTiming);
    if(event!=cudaSuccess){cudaStreamDestroy(s.stream);s.stream=nullptr;MFQ_CUDA_CHECK(event);}
    const auto window_event=cudaEventCreateWithFlags(&s.window_done,cudaEventDisableTiming);
    if(window_event!=cudaSuccess){cudaEventDestroy(s.done);s.done=nullptr;cudaStreamDestroy(s.stream);s.stream=nullptr;MFQ_CUDA_CHECK(window_event);}
}
MoeResidencyManager::~MoeResidencyManager() {
    if(impl_->prefill_done) {
        (void)cudaEventSynchronize(impl_->prefill_done);
        (void)cudaEventDestroy(impl_->prefill_done);
    }
    if(impl_->worker.joinable()) {
        {std::lock_guard<std::mutex> lock(impl_->worker_mutex);impl_->stopping=true;}
        impl_->work_available.notify_one();impl_->work_finished.notify_all();impl_->worker.join();
    }
    if(!impl_->retired_dma_leases.empty()) {
        (void)cudaEventSynchronize(impl_->retired_dma_done->get());
        (void)cudaEventSynchronize(impl_->window_done);
        impl_->retired_dma_leases.clear();
    }
    try {if(!impl_->pending.empty())rollback();}catch(...) {}
    // Batch storage is allocated on the adaptation stream. Its async frees
    // must be issued while that stream is still alive.
    impl_->device_work={};impl_->device_descriptors={};
    impl_->copy_descriptors={};impl_->upload={};impl_->backup={};
    if(impl_->stream)(void)cudaStreamSynchronize(impl_->stream);
    if(impl_->done)cudaEventDestroy(impl_->done);
    if(impl_->window_done)cudaEventDestroy(impl_->window_done);
    if(impl_->backup_stream)(void)cudaStreamSynchronize(impl_->backup_stream);
    for(auto event:impl_->preserve_events)cudaEventDestroy(event);
    if(impl_->backup_done)cudaEventDestroy(impl_->backup_done);
    if(impl_->backup_stream)cudaStreamDestroy(impl_->backup_stream);
    if(impl_->stream)cudaStreamDestroy(impl_->stream);
}
void MoeResidencyManager::record_prefill_routes(int layer,const int32_t* ids,int tokens,int routes,void* raw_stream) {
    auto& s=*impl_;
    if(layer<0 || !s.cache->config_.moe_residency_adapt || !s.cache->config_.moe_residency_warm)return;
    const auto* option=std::getenv("MFQ_MOE_PREFILL_RESIDENCY");
    if(option && option[0]=='0')return;
    const auto stream=static_cast<cudaStream_t>(raw_stream);
    if(!s.prefill_done)MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&s.prefill_done,cudaEventDisableTiming));
    if(s.prefill_routes.empty())s.prefill_routes.resize(s.cache->pipeline_bundles_.size());
    if(routes>s.prefill_route_stride) {
        if(s.prefill_pending)MFQ_CUDA_CHECK(cudaEventSynchronize(s.prefill_done));
        const int64_t elements=int64_t(s.prefill_routes.size())*32*routes;
        // The samples are consumed after prefill returns the borrowed expert
        // storage. They must not be allocated from that temporary workspace.
        mfq::cuda::ResidentAllocationScope resident;
        auto next=tb::empty({elements},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
        // Preserve samples if a model uses different routing widths by layer.
        if(s.prefill_route_device.defined())for(std::size_t b=0;b<s.prefill_routes.size();++b) {
            const auto& sample=s.prefill_routes[b];if(!sample.tokens)continue;
            MFQ_CUDA_CHECK(cudaMemcpyAsync(next.data_ptr<int32_t>()+b*32*routes,
                s.prefill_route_device.data_ptr<int32_t>()+b*32*s.prefill_route_stride,
                sample.tokens*sample.routes*sizeof(int32_t),cudaMemcpyDeviceToDevice,stream));
        }
        s.prefill_route_device=std::move(next);s.prefill_route_stride=routes;
        s.prefill_route_host=std::make_unique<mfq::cuda::HostBuffer>(elements*sizeof(int32_t),false);
    }
    for(std::size_t b=0;b<s.prefill_routes.size();++b) {
        if(s.cache->sources_[s.cache->pipeline_bundles_[b][0]]->layer_id_!=layer)continue;
        auto& sample=s.prefill_routes[b];
        const auto tail=std::min(tokens,32),bytes=tail*routes*int(sizeof(int32_t));
        sample.tokens=tail;sample.routes=routes;
        // Keep route samples on-device during prefill. Small D2H copies at
        // every layer interfere with the overlapping expert-weight transfers.
        MFQ_CUDA_CHECK(cudaMemcpyAsync(s.prefill_route_device.data_ptr<int32_t>()+b*32*s.prefill_route_stride,
            ids+int64_t(tokens-tail)*routes,bytes,cudaMemcpyDeviceToDevice,stream));
        MFQ_CUDA_CHECK(cudaEventRecord(s.prefill_done,stream));
        s.prefill_pending=true;
    }
}
void MoeResidencyManager::prepare_decode() {
    auto& s=*impl_;
    if(!s.prefill_pending)return;
    apply_pending();
    MFQ_CUDA_CHECK(cudaStreamWaitEvent(s.stream,s.prefill_done,0));
    MFQ_CUDA_CHECK(cudaMemcpyAsync(s.prefill_route_host->data(),s.prefill_route_device.data_ptr(),
        s.prefill_route_device.numel()*sizeof(int32_t),cudaMemcpyDeviceToHost,s.stream));
    MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
    const auto started=std::chrono::steady_clock::now();
    for(std::size_t b=0;b<s.prefill_routes.size();++b) {
        auto& sample=s.prefill_routes[b];if(!sample.tokens)continue;
        std::fill(s.heat[b].begin(),s.heat[b].end(),0.0f);
        std::fill(s.observations[b].begin(),s.observations[b].end(),0.0f);
        const auto* ids=static_cast<const int32_t*>(s.prefill_route_host->data())+b*32*s.prefill_route_stride;
        for(int t=0;t<sample.tokens;++t) {
            // About six recent token-equivalents: inform the first decode
            // without letting a long prompt dominate later routing changes.
            const float weight=std::exp2(-float(sample.tokens-1-t)/4.0f);
            for(int r=0;r<sample.routes;++r)s.heat[b].at(ids[t*sample.routes+r])+=weight;
        }
        sample.tokens=0;
    }
    s.prefill_pending=false;
    const auto before=s.committed_projections.load();
    // This boundary is outside the prefill workspace loan. All expert slots
    // contain weights again and the usual transactional guards still apply.
    after_layer(s.last,{},32);
    apply_pending();
    s.traced_decode_waits=0;
    std::cerr<<"runtime_moe_prefill_residency projections="<<s.committed_projections.load()-before
        <<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<<'\n';
}
void MoeResidencyManager::release_transfer_stages() {
    apply_pending();
    for(auto& stages:impl_->transfer_stages)for(auto& stage:stages)stage.arena=nullptr;
}
void MoeResidencyManager::before_layer(int layer) {
    auto& s=*impl_;if(layer!=s.first && layer>=0)return;
    if(layer<0 && s.predictor_overlap_scope){prepare_predictor();return;}
    s.predictor_overlap_scope=false;
    // Eager MTP enters the predictor before the next backbone layer zero.
    // Finish backbone exchanges before predictor DMA reuses shared staging or
    // acquires RAM leases held by the exchange worker.
    apply_pending();
    if(s.expert_done_host) {
        *static_cast<volatile uint32_t*>(s.expert_done_host->data())=0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
}
void MoeResidencyManager::prepare_predictor() {
    auto& s=*impl_;s.predictor_overlap_scope=true;
    if(s.predictor_overlap && s.work_pending.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lock(s.worker_mutex);
        s.work_finished.wait(lock,[&]{return s.predictor_overlap_allowed || s.work_ready || s.stopping;});
        if(s.predictor_overlap_allowed && !s.work_ready && !s.stopping) {
            ++s.predictor_overlap_serves;return;
        }
    }
    apply_pending();
    // The replay's reset callback enters the same scoped predictor boundary.
    s.predictor_overlap_scope=true;
}
bool MoeResidencyManager::prepare_during_window() const noexcept {
    return impl_->window_prepare;
}
void MoeResidencyManager::record_window_expert_fence(int layer,void* raw_stream) {
    auto& s=*impl_;
    if(!s.expert_fence || !s.window_prepare || layer!=s.last)return;
    const auto stream=static_cast<cudaStream_t>(raw_stream);
    cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
    MFQ_CUDA_CHECK(cudaStreamIsCapturing(stream,&status));
    if(!s.expert_done_host) {
        if(status!=cudaStreamCaptureStatusNone)
            throw std::logic_error("adaptive expert fence was not warmed before graph capture");
        s.expert_done_host=std::make_unique<mfq::cuda::HostBuffer>(128,true);
        std::memset(s.expert_done_host->data(),0,128);
        MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&s.expert_done_device),s.expert_done_host->data(),0));
    }
    // The final expert output is complete here; the LM head and residual work
    // never access expert weights or transfer slots. Only GPU backup/upload may
    // start early. RAM ownership and mapping publication retain finish_window.
    // Match the mapped publication protocol used by Strata/MFQ routing. A
    // replayed external CUDA event did not release side-stream copies before
    // a blocked graph tail in the local WDDM regression test.
    s.expert_compute_stream=stream;
    mfq::cuda::signal_mapped_flag(s.expert_done_device,stream);
}
void MoeResidencyManager::finish_window(int layer) {
    auto& s=*impl_;if(layer!=s.last)return;
    if(s.window_prepare)s.cache->record_compute_use();
    {std::lock_guard<std::mutex> lock(s.worker_mutex);s.window_inflight=false;}
    s.work_finished.notify_all();
}
void MoeResidencyManager::begin_speculative_routes(int rows) {
    auto& s=*impl_;
    if(rows<1 || rows>32)throw std::invalid_argument("speculative route row count is out of range");
    if(s.speculative_rows)throw std::logic_error("speculative routes already pending");
    for(auto& row:s.speculative_routes){row.ids.clear();row.rows=0;}
    if(s.cache->ssd_experts())s.cache->host_experts_->begin_speculative_learning();
    s.speculative_rows=rows;
}
void MoeResidencyManager::commit_speculative_routes(int rows) {
    auto& s=*impl_;if(!s.speculative_rows)return;
    if(rows<0 || rows>s.speculative_rows)throw std::invalid_argument("committed route prefix is out of range");
    // Validate the entire prefix before changing observations. Graph setup may
    // replay a layer more than once; only its latest route publication counts.
    for(const auto& record:s.speculative_routes)
        if(record.rows && (record.rows<rows || record.ids.size()%record.rows))
            throw std::logic_error("speculative route publication lost row alignment");
    if(s.cache->ssd_experts()) {
        s.speculative_host_keys.clear();
        // Commit in ordinary decode order: each accepted token visits all
        // layers. Frequency and recency must not depend on verify batch size.
        for(int row=0;row<rows;++row)for(std::size_t b=0;b<s.speculative_routes.size();++b) {
            auto& record=s.speculative_routes[b];if(!record.rows)continue;
            const auto routes=record.ids.size()/record.rows;
            const auto& bundle=s.cache->pipeline_bundles_[b];
            for(std::size_t i=0;i<routes;++i) {
                const auto expert=record.ids[std::size_t(row)*routes+i];
                for(int p=0;p<3;++p) {
                    if(p && bundle[p]==bundle[p-1])continue;
                    const auto& source=*s.cache->sources_[bundle[p]];
                    if(source.quant_source_->gpu_resident(expert))continue;
                    s.speculative_host_keys.push_back({source.id_,source.expert_to_cohort_[expert],expert});
                }
            }
        }
        s.cache->host_experts_->commit_speculative_learning(s.speculative_host_keys);
    }
    uint64_t committed=0,heat_routes=0;
    for(std::size_t b=0;b<s.speculative_routes.size();++b) {
        auto& record=s.speculative_routes[b];if(!record.rows)continue;
        const auto count=std::size_t(rows)*(record.ids.size()/record.rows);
        if(s.speculative_window_heat)std::fill(record.heat_seen.begin(),record.heat_seen.end(),uint64_t(0));
        std::size_t distinct=0;
        for(std::size_t i=0;i<count;++i) {
            const auto expert=unsigned(record.ids[i]);
            if(s.speculative_window_heat) {
                auto& word=record.heat_seen[expert/64];const auto bit=uint64_t(1)<<(expert%64);
                if(word&bit)continue;
                word|=bit;
            }
            ++distinct;
        }
        // Keep the old total heat (and its hysteresis scale) while removing
        // within-window multiplicity from individual expert priorities.
        // An all-repeated route set is unchanged; mixed route sets give every
        // unique expert the same avoided-transfer value in this window.
        const float weight=s.speculative_heat_mode==2 && distinct?float(count)/float(distinct):1.0f;
        for(std::size_t i=0;i<count;++i) {
            const auto expert=unsigned(record.ids[i]);
            if(s.speculative_window_heat) {
                auto& word=record.heat_seen[expert/64];const auto bit=uint64_t(1)<<(expert%64);
                if(!(word&bit))continue;
                word&=~bit;
            }
            s.observations[b][expert]+=weight;
        }
        heat_routes+=distinct;
        committed+=count;
        s.observed_routes+=count;s.speculative_routes_committed+=count;
        s.speculative_routes_discarded+=record.ids.size()-count;
    }
    s.speculative_heat_routes+=heat_routes;
    if(!s.traced_speculative_heat && committed) {
        std::fprintf(stderr,"moe_mtp_heat_policy scope=first_committed_window window_union=%d normalized=%d rows=%d routes=%llu heat_routes=%llu\n",
            int(s.speculative_window_heat),int(s.speculative_heat_mode==2),rows,static_cast<unsigned long long>(committed),
            static_cast<unsigned long long>(heat_routes));
        s.traced_speculative_heat=true;
    }
    ++s.speculative_route_windows;s.speculative_rows=0;
}
void MoeResidencyManager::discard_speculative_routes() noexcept {
    auto& s=*impl_;if(!s.speculative_rows)return;
    if(s.cache->ssd_experts())s.cache->host_experts_->discard_speculative_learning();
    for(auto& record:s.speculative_routes){s.speculative_routes_discarded+=record.ids.size();record.ids.clear();record.rows=0;}
    ++s.speculative_route_cancels;s.speculative_rows=0;
}
void MoeResidencyManager::after_layer(int layer,const std::vector<int32_t>& ids,int tokens,bool window_inflight,bool independent_sequences) {
    auto& s=*impl_;
    if(s.cache->pipeline_residency_suspended_)return;
    for(std::size_t b=0;b<s.cache->pipeline_bundles_.size();++b)
        if(s.cache->sources_[s.cache->pipeline_bundles_[b][0]]->layer_id_==layer) {
            if(s.speculative_rows && layer>=0) {
                if(independent_sequences || tokens<1 || tokens>s.speculative_rows || ids.size()%tokens)
                    throw std::logic_error("speculative routes do not describe one aligned sequence");
                for(const auto expert:ids)if(unsigned(expert)>=s.observations[b].size())
                    throw std::out_of_range("speculative expert route is out of range");
                auto& record=s.speculative_routes[b];record.ids=ids;record.rows=tokens;
            }else for(int expert:ids) {
                if (expert < 0 && independent_sequences) continue;
                s.observations[b].at(expert)+=1.0f;
                ++s.observed_routes;
            }
        }
    if(!s.cache->config_.moe_residency_adapt)return;
    if(layer!=s.last || s.work_pending.load(std::memory_order_acquire))return;
    if(!s.cache->config_.moe_residency_warm && (tokens>8 || (++s.rounds%4)!=0))return;
    // Strata adjusts its cache every few completed windows. Keep collecting
    // all routes between rounds; prompt batches still prepare immediately.
    // Speculative observations become visible when verification commits its
    // accepted prefix. Prepare from the previous committed observations here,
    // retaining overlap between expert exchange and the current graph tail.
    // Rank and exchange at the configured window interval for both paths;
    // exchanging after every speculative verification serializes the next
    // draft behind a multi-millisecond backbone cache transaction.
    const auto* short_interval=std::getenv("MFQ_MOE_SHORT_DECODE_INTERVAL");
    const bool interval_decode=tokens==1 || (tokens<=8 &&
        (!independent_sequences || !short_interval || short_interval[0]!='0'));
    if(s.cache->config_.moe_residency_warm && interval_decode && (++s.rounds%s.interval)!=0)return;
    if(!s.pending.empty())throw std::logic_error("adaptive tier has an unpublished transaction");
    for(std::size_t b=0;b<s.heat.size();++b)for(std::size_t e=0;e<s.heat[b].size();++e) {
        s.heat[b][e]+=s.observations[b][e];s.observations[b][e]=0;
    }
    ++s.scheduled_rounds;
    s.pending_tokens=tokens;
    s.error=nullptr;
    // During replay the entire graph is already queued on the compute stream.
    // Planning can proceed now; resident writes must follow its completion.
    s.pending_window=window_inflight;
    s.pending_expert_fence=window_inflight && s.expert_fence && bool(s.expert_done_host);
    // Capture this transaction's producer fence now. compute_done_ is also
    // recorded by later predictor calls and cannot be read asynchronously by
    // the worker without risking a dependency on its own downstream DMA.
    MFQ_CUDA_CHECK(cudaEventRecord(s.window_done,mfq_get_current_cuda_stream().stream()));
    if(window_inflight)++s.window_prepares;
    if(!s.worker.joinable())s.worker=std::thread([this] {
        auto& state=*impl_;
        bool device_ready=false;
        std::unique_lock<std::mutex> lock(state.worker_mutex);
        while(true) {
            state.work_available.wait(lock,[&]{return state.work_requested || state.stopping;});
            if(!state.work_requested)break;
            state.work_requested=false;lock.unlock();
            try {
                // A same-device guard only calls cudaGetDevice. A pure
                // cache-hit round may reach the Driver API without any CUDA
                // Runtime operation establishing this worker's context.
                if(!device_ready) {
                    MFQ_CUDA_CHECK(cudaSetDevice(state.cache->sources_.front()->cohorts_.front().active.expert_local.get_device()));
                    device_ready=true;
                }
                prepare();
                if(state.async_publish && !state.ssd_predictor_isolated) {
                    std::unique_lock<std::mutex> finish_lock(state.worker_mutex);
                    state.work_finished.wait(finish_lock,[&]{return !state.window_inflight || state.stopping;});
                    const bool publish=!state.stopping;finish_lock.unlock();
                    const PhaseClock publish_clock(state.async_publish_ns,
                        state.pending_tokens<=8?&state.decode_async_publish_ns:nullptr);
                    if(publish)publish_pending();
                }
            }
            catch(...) {
                auto failure=std::current_exception();
                try {rollback();}catch(...) {failure=std::current_exception();}
                state.error=failure;
            }
            // Keep borrowed cache addresses protected through rollback as well
            // as successful DMA and publication. Predictor misses may reclaim
            // them only after no transaction can still read those addresses.
            if(!state.cache->ssd_experts() || state.pending.empty())state.exchange_reads.reset();
            lock.lock();state.work_ready=true;state.work_finished.notify_one();
        }
    });
    if(!window_inflight && !s.cache->pipeline_dma_leases_.empty()) {
        // Eager direct-RAM uploads otherwise retain their readers until the
        // next fetch, so every hot incoming expert looks busy to this planner.
        // The recorded consumer fence also covers mapped-copy/direct-read
        // kernels that can keep reading host fields after weight DMA finishes.
        // Snapshot uploads separately: predictive prefetch may be unconsumed.
        if(!s.retired_dma_done)s.retired_dma_done=std::make_unique<mfq::cuda::Event>();
        s.retired_dma_done->record(s.cache->weight_stream_);
        s.retired_dma_leases.swap(s.cache->pipeline_dma_leases_);
    }
    s.work_pending.store(true,std::memory_order_release);
    {std::lock_guard<std::mutex> lock(s.worker_mutex);s.window_inflight=window_inflight;s.work_requested=true;
        s.ssd_predictor_isolated=false;s.ssd_copies_permitted=false;
        s.work_ready=false;s.predictor_overlap_allowed=false;}
    s.work_available.notify_one();
}
void MoeResidencyManager::prepare() {
    auto& s=*impl_;auto& cache=*s.cache;
    const auto plan_started=std::chrono::steady_clock::now();
    const auto prior_skips=s.staging_budget_skips.load();
    const auto finish_plan=[&] {
        s.last_plan_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-plan_started).count();
        s.last_staging_budget_skips=s.staging_budget_skips.load()-prior_skips;
        s.plan_ns+=s.last_plan_ns;
        if(s.pending_tokens<=8)s.decode_plan_ns+=s.last_plan_ns;
    };
    const PhaseClock clock(s.prepare_ns,s.pending_tokens<=8?&s.decode_prepare_ns:nullptr);
    const MfqCudaGuard guard(cache.sources_.front()->cohorts_.front().active.expert_local.device());
    using Expert=mfq::MoeResidencyRanker::Expert;
    if(cache.pipeline_transfer_cache_)for(std::size_t source_id=0;source_id<s.transfer_stages.size();++source_id)
      for(auto& stage:s.transfer_stages[source_id])
        if(!stage.arena) {
            const auto found=cache.pipeline_stages_.find(stage.key);
            if(found!=cache.pipeline_stages_.end())stage.arena=found->second.get();
            // Decode arenas include their route capacity in the cache key.
            // Prefer the smallest compatible arena if multiple widths exist.
            for(const auto& item:cache.pipeline_stages_)
                if((item.first.starts_with(stage.key+":") ||
                    item.first.starts_with(moe_packed_transfer_prefix(stage.projection,cache.sources_[source_id]->layer_id_<0))) &&
                    (!stage.arena || item.second->slots<stage.arena->slots))
                    stage.arena=item.second.get();
        }
    auto& pairs=s.ranking->rank([&](const Expert& e){return s.heat[e.bundle][e.expert];},
        [&](const Expert& e) {
            const auto& bundle=cache.pipeline_bundles_[e.bundle];
            if(e.projection>=0) {
                const auto& source=*cache.sources_[bundle[e.projection]];
                if(source.quant_source_->gpu_resident(e.expert))return 1;
                // A transfer-cache hit is temporary residency. Let a measured
                // hot projection compete for permanent space when requested;
                // the normal lease, inflight and publication guards still
                // protect the RAM/GPU exchange.
                if(cache.pipeline_transfer_cache_ && !s.promote_transfer) {
                    const int c=source.expert_to_cohort_[e.expert];
                    const auto* arena=s.transfer_stages[bundle[e.projection]][c].arena;
                    if(arena && arena->book && arena->book->slot_for({source.id_,c,e.expert})>=0)return -1;
                }
                return 0;
            }
            bool hot=true,cold=true;
            for(int p=0;p<3;++p) {
                const bool gpu=cache.sources_[bundle[p]]->quant_source_->gpu_resident(e.expert);
                hot&=gpu;cold&=!gpu;
            }
            return hot?1:cold?0:-1;
        });
    s.clear_projection_marks();
    for(const auto& pair:pairs) {
        s.candidate_bundles+=s.mark_projection_complete(
            pair.incoming.bundle,pair.incoming.expert,pair.incoming.projection);
        s.candidate_projections+=pair.incoming.projection<0?3:1;
    }
    if(!cache.config_.moe_residency_warm && pairs.size()>32)pairs.resize(32);
    if(!s.retired_dma_leases.empty()) {
        s.retired_dma_done->synchronize();
        MFQ_CUDA_CHECK(cudaEventSynchronize(s.window_done));
        s.retired_dma_leases.clear();
    }
    s.direct=cache.config_.moe_direct_ram && cache.pipeline_ram_registered_bytes_>0;
    bool copies_ready=false;
    const auto allow_copies=[&] {
        if(copies_ready)return;
        if(s.ssd_predictor_isolated) {
            std::unique_lock lock(s.worker_mutex);
            s.work_finished.wait(lock,[&]{return s.ssd_copies_permitted || s.stopping;});
            if(s.stopping)throw std::runtime_error("SSD expert exchange cancelled before DMA");
        }
        if(s.pending_window && s.pending_expert_fence) {
            const PhaseClock wait_clock(s.expert_fence_wait_ns,nullptr);
            mfq::cuda::wait_route_publication(static_cast<const uint32_t*>(s.expert_done_host->data()),s.expert_compute_stream,60000);
            ++s.expert_fence_waits;
        }else MFQ_CUDA_CHECK(cudaStreamWaitEvent(s.stream,s.window_done,0));
        copies_ready=true;
    };
    const bool shared_backup=cache.config_.moe_residency_warm && s.direct && cache.pipeline_host_stage_.defined() &&
        !cache.ssd_experts();
    // The next layer waits for publication before using the transfer workspace.
    // Reuse its pinned storage between windows instead of allocating a second
    // model-sized backup alongside graph preparation.
    if(shared_backup) {
        s.backup=cache.pipeline_host_stage_;
        s.shared_backup_bytes=s.backup.numel();
    }
    std::uint64_t stage_limit=std::numeric_limits<std::uint64_t>::max();
    if(cache.config_.moe_residency_warm) {
        const auto memory=s.memory_probe();auto available=memory.available_host_bytes();
        if(cache.config_.moe_host_physical_bytes)
            available=std::min(available,memory.resident_bytes>=cache.config_.moe_host_physical_bytes
                ? std::uint64_t(0):cache.config_.moe_host_physical_bytes-memory.resident_bytes);
        const auto reusable=std::uint64_t(s.backup.defined()?s.backup.numel():0)+
            std::uint64_t(s.upload.defined()?s.upload.numel():0);
        stage_limit=(available+reusable)/(s.direct?1:2);
        if(shared_backup)stage_limit=std::min(stage_limit,std::uint64_t(s.backup.numel()));
        else if(cache.ssd_experts() && cache.pipeline_host_stage_.defined())
            stage_limit=std::min(stage_limit,std::uint64_t(cache.pipeline_host_stage_.numel()));
    }
    std::map<std::pair<int,int>,MoeHostExpertCache::Lease> ssd_images;
    if(cache.ssd_experts()) {
        stage_limit=std::min(stage_limit,std::uint64_t(s.ssd_stage_capacity));
        // A small LRU mostly contains the last layers after a decode window.
        // Rank all cold experts, then read only the selected replacements;
        // otherwise early-layer hotspots can never become GPU residents.
        // Leave most of the managed cache available to concurrent demand IO.
        stage_limit=std::min(stage_limit,std::uint64_t(cache.host_experts_->stats().budget_bytes/4));
        std::vector<mfq::MoeResidencyRanker::Pair> selected;
        std::vector<MoeExpertCache::SsdExpertRequest> requests;
        std::vector<std::pair<int,int>> keys;
        std::vector<std::pair<int,int>> gpu_images;
        std::uint64_t selected_bytes=0;
        for(const auto& pair:pairs) {
            const int count=pair.incoming.projection<0?3:1;
            std::uint64_t required=selected_bytes;bool valid=true;
            for(int i=0;i<count;++i) {
                const int p=count==3?i:pair.incoming.projection;
                const int q=count==3?i:pair.outgoing.projection;
                const auto& source=*cache.sources_[cache.pipeline_bundles_[pair.incoming.bundle][p]];
                const auto& old=*cache.sources_[cache.pipeline_bundles_[pair.outgoing.bundle][q]];
                const auto& cohort=source.cohorts_[source.expert_to_cohort_[pair.incoming.expert]];
                const auto slot=cohort.arena->book->slot_for(old.quant_source_->host_key(pair.outgoing.expert));
                if(slot<0 || cohort.arena->book->inflight(slot)){valid=false;break;}
                for(const auto bytes:cohort.bytes_per_expert) {
                    const auto padding=(16-(required&15))&15;
                    if(required>stage_limit || padding>stage_limit-required ||
                            std::uint64_t(bytes)>stage_limit-required-padding){valid=false;break;}
                    required+=padding+bytes;
                }
                if(!valid)break;
            }
            if(!valid)continue;
            selected_bytes=required;selected.push_back(pair);
            for(int i=0;i<count;++i) {
                const int p=count==3?i:pair.incoming.projection;
                const int id=cache.pipeline_bundles_[pair.incoming.bundle][p];
                const auto key=std::make_pair(id,pair.incoming.expert);
                if(ssd_images.try_emplace(key).second) {
                    const auto& source=*cache.sources_[id];
                    auto held=cache.host_experts_->try_find(source.quant_source_->host_key(key.second),false);
                    if(held){ssd_images.at(key)=std::move(held);continue;}
                    const auto& cohort=source.cohorts_[source.expert_to_cohort_[key.second]];
                    if(s.direct && cache.kv_expert_loan_ &&
                            cache.kv_expert_loan_->contains_projection(id,key.second,cohort.bytes_per_expert)) {
                        gpu_images.push_back(key);continue;
                    }
                    requests.emplace_back(source.quant_source_.get(),pair.incoming.expert);
                    keys.push_back(key);
                }
            }
        }
        pairs.swap(selected);
        // Only independent predictor arenas may run beside a backbone
        // transaction. Keep publication on the caller after that predictor;
        // it may still own the current packet and its host/DMA leases.
        s.ssd_predictor_isolated=s.predictor_overlap && s.pending_tokens<=8 && !pairs.empty();
        if(s.ssd_predictor_isolated) {
            std::unordered_set<MoeGpuArena*> predictor_arenas;
            for(const auto& source:cache.sources_)if(source->layer_id_<0)
                for(const auto& cohort:source->cohorts_)predictor_arenas.insert(cohort.arena);
            if(predictor_arenas.empty())s.ssd_predictor_isolated=false;
            for(const auto& pair:pairs) {
                const int count=pair.incoming.projection<0?3:1;
                for(int i=0;i<count;++i)for(const auto& expert:{pair.incoming,pair.outgoing}) {
                    const int p=count==3?i:expert.projection;
                    const auto& source=*cache.sources_[cache.pipeline_bundles_[expert.bundle][p]];
                    if(source.layer_id_<0 || predictor_arenas.contains(source.cohorts_[source.expert_to_cohort_[expert.expert]].arena))
                        s.ssd_predictor_isolated=false;
                }
            }
        }
        if(s.ssd_predictor_isolated) {
            {std::lock_guard<std::mutex> lock(s.worker_mutex);s.predictor_overlap_allowed=true;}
            s.work_finished.notify_all();
            if(++s.predictor_overlap_rounds==1)
                std::fprintf(stderr,"moe_ssd_predictor_exchange_overlap separate_readers=1 separate_backup=1 separate_arenas=1 deferred_publication=1\n");
            s.observe("ssd_predictor_overlap_ready",0);
        }
        s.ssd_load_requests+=requests.size()+gpu_images.size();s.ssd_load_bytes+=selected_bytes;
        auto images=cache.read_ssd_experts(requests,true);
        for(std::size_t i=0;i<images.size();++i)ssd_images.at(keys[i])=std::move(images[i]);
        if(!gpu_images.empty()) {
            // Predictor packets may evict these hints. Retain and revalidate
            // their GPU addresses only after the existing copy/read fence.
            // The RAM transaction still receives a complete immutable image.
            allow_copies();
            s.exchange_reads.emplace(cache.kv_expert_loan_->retain_exchange_reads(false,gpu_images.size()));
            for(const auto& key:gpu_images) {
                auto& source=*cache.sources_[key.first];
                const auto& cohort=source.cohorts_[source.expert_to_cohort_[key.second]];
                std::array<const void*,8> fields{};bool complete=true;
                for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f)if(cohort.bytes_per_expert[f]) {
                    fields[f]=cache.kv_expert_loan_->find_for_exchange({key.first,key.second,int64_t(f)},
                        std::size_t(cohort.bytes_per_expert[f]),&*s.exchange_reads);
                    complete&=fields[f]!=nullptr;
                }
                if(!complete) {ssd_images.at(key)=source.quant_source_->acquire_expert(key.second);continue;}
                const auto bytes=source.quant_source_->expert_field_bytes(key.second);
                ssd_images.at(key)=cache.host_experts_->acquire(source.quant_source_->host_key(key.second),bytes,[&] {
                    const PhaseClock copy_clock(s.ssd_gpu_image_ns,nullptr);
                    auto image=source.quant_source_->allocate_expert(key.second);
                    struct RetireImage {
                        cudaStream_t stream;
                        ~RetireImage(){if(stream)(void)cudaStreamSynchronize(stream);}
                    } retire{s.stream};
                    const auto target=moe_cache_field_refs(image);
                    std::array<mfq::cuda::DmaCopy,8> copies{};std::size_t count=0;
                    for(std::size_t f=0;f<target.size();++f)if(cohort.bytes_per_expert[f]) {
                        auto* destination=target[f].data_ptr();const auto bytes=std::size_t(cohort.bytes_per_expert[f]);
                        if(count) {
                            auto& previous=copies[count-1];
                            const auto gap=reinterpret_cast<uintptr_t>(fields[f])-reinterpret_cast<uintptr_t>(previous.source);
                            if(gap>=previous.bytes && gap-previous.bytes<=15 &&
                                    reinterpret_cast<uintptr_t>(destination)-reinterpret_cast<uintptr_t>(previous.destination)==gap) {
                                previous.bytes=gap+bytes;continue;
                            }
                        }
                        copies[count++]={destination,fields[f],bytes};
                    }
                    for(std::size_t i=0;i<count;++i)MFQ_CUDA_CHECK(cudaMemcpyAsync(
                        copies[i].destination,copies[i].source,copies[i].bytes,cudaMemcpyDeviceToHost,s.stream));
                    s.observe("gpu_image",std::size_t(key.second));
                    MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
                    retire.stream=nullptr;
                    s.ssd_gpu_image_bytes+=bytes;s.ssd_gpu_image_copies+=count;
                    if(++s.ssd_gpu_images==1)std::fprintf(stderr,
                        "moe_ssd_gpu_image_dispatch scope=first_image enabled=1 bytes=%zu copies=%zu\n",bytes,count);
                    return image;
                });
            }
        }
    }
    s.pending.reserve(pairs.size()*3);int64_t bytes=0;
    std::size_t memory_rejections=0;
    for(const auto& pair:pairs) {
        const int count=pair.incoming.projection<0?3:1;
        // Geometry is immutable. Reject candidates that cannot fit before
        // scanning the RAM image, acquiring leases or allocating replacement
        // nodes. A full workspace can otherwise pay those costs for every
        // lower-ranked candidate even though none can be exchanged.
        int64_t required=bytes;
        for(int i=0;i<count;++i) {
            const int projection=count==3?i:pair.incoming.projection;
            const auto& source=*cache.sources_[cache.pipeline_bundles_[pair.incoming.bundle][projection]];
            const auto& cohort=source.cohorts_[source.expert_to_cohort_[pair.incoming.expert]];
            for(const auto field_bytes:cohort.bytes_per_expert) {
                if(required>INT64_MAX-15)throw std::overflow_error("adaptive staging alignment overflow");
                required=(required+15)&~int64_t(15);
                if(field_bytes>INT64_MAX-required)throw std::overflow_error("adaptive staging size overflow");
                required+=field_bytes;
            }
        }
        if(std::uint64_t(required)>stage_limit) {
            // Candidate counting has finished. Reuse one spare bit to count
            // rejected bundles once, without allocating a set node per expert.
            auto& mark=s.projection_marks[pair.incoming.bundle][pair.incoming.expert];
            if(!(mark&8)){mark|=8;++memory_rejections;}
            s.projection_memory_rejections+=count;s.staging_budget_skips+=count;
            continue;
        }
        std::array<Impl::Move,3> bundle;bool valid=true;
        for(int i=0;i<count;++i) {
            const int incoming_projection=count==3?i:pair.incoming.projection;
            const int outgoing_projection=count==3?i:pair.outgoing.projection;
            auto& move=bundle[i];move.incoming_source=cache.pipeline_bundles_[pair.incoming.bundle][incoming_projection];
            move.outgoing_source=cache.pipeline_bundles_[pair.outgoing.bundle][outgoing_projection];
            move.incoming_bundle=pair.incoming.bundle;move.incoming_projection=incoming_projection;
            move.incoming=pair.incoming.expert;move.outgoing=pair.outgoing.expert;
            auto& incoming=*cache.sources_[move.incoming_source];auto& outgoing=*cache.sources_[move.outgoing_source];
            move.cohort=incoming.expert_to_cohort_[move.incoming];auto& cohort=incoming.cohorts_[move.cohort];
            move.field_count=cohort.bytes_per_expert.size();
            if(move.field_count>move.offsets.size())
                throw std::logic_error("adaptive expert field count exceeds transaction capacity");
            if(cache.pipeline_prefill_layer_layout_) {
                if(move.incoming_source!=move.outgoing_source)
                    throw std::logic_error("prefill residency exchange crosses source geometry");
                const auto row=std::find(cohort.prefill_ram_experts.begin(),cohort.prefill_ram_experts.end(),move.incoming);
                if(row==cohort.prefill_ram_experts.end())
                    throw std::logic_error("prefill residency exchange has no cold image row");
                move.prefill_ram_slot=int(std::distance(cohort.prefill_ram_experts.begin(),row));
            }
            const auto old_key=outgoing.quant_source_->host_key(move.outgoing);move.slot=cohort.arena->book->slot_for(old_key);
            if(move.slot<0 || cohort.arena->book->inflight(move.slot) ||
                    !cache.host_experts_->contains(incoming.quant_source_->host_key(move.incoming))){valid=false;break;}
            if(cache.ssd_experts())move.lease=std::move(ssd_images.at({move.incoming_source,move.incoming}));
            else move.lease=incoming.quant_source_->acquire_expert(move.incoming);
            if(move.lease.use_count()!=2){valid=false;break;}
            // Streamed RAM images may outgrow the pinned arena. The ordinary
            // SSD transaction accepts pageable images through staged CUDA DMA;
            // only the full-RAM in-place exchange requires registered fields.
            if(s.direct && !cache.ssd_experts() && cache.pipeline_ram_registered_bytes_<cache.pipeline_ram_complement_->numel()) {
                const auto begin=reinterpret_cast<uintptr_t>(cache.pipeline_ram_complement_->data_ptr());
                const auto end=begin+cache.pipeline_ram_registered_bytes_;
                for(const auto& field:moe_cache_field_refs(move.lease->weights)) {
                    if(!tensor_nbytes(field))continue;
                    const auto address=reinterpret_cast<uintptr_t>(field.data_ptr());
                    if(address<begin || address>end || uint64_t(tensor_nbytes(field))>end-address){valid=false;break;}
                }
                if(!valid)break;
            }
            if(s.reuse_transfer && s.direct && cache.pipeline_transfer_cache_ && !s.ssd_predictor_isolated) {
                auto* arena=s.transfer_stages[move.incoming_source][move.cohort].arena;
                const int slot=arena && arena->book ? arena->book->slot_for(
                    incoming.quant_source_->host_key(move.incoming)) : -1;
                const bool aliased=slot>=0 && std::any_of(cache.pipeline_transfer_slots_.begin(),
                    cache.pipeline_transfer_slots_.end(),[&](const auto& held) {
                        return held.aliased && held.book==arena->book.get() && held.slot==slot;
                    });
                if(slot>=0 && !aliased){move.cached_arena=arena;move.cached_slot=slot;}
            }
        }
        if(!valid)continue;
        int64_t next_bytes=bytes;
        for(int i=0;i<count;++i) {
            auto& move=bundle[i];
            auto& incoming=*cache.sources_[move.incoming_source];
            auto& outgoing=*cache.sources_[move.outgoing_source];
            auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
            for(std::size_t f=0;f<move.field_count;++f) {
                const auto bytes=cohort.bytes_per_expert[f];
                next_bytes=(next_bytes+15)&~int64_t(15);move.offsets[f]=next_bytes;
                if(bytes>INT64_MAX-next_bytes)throw std::overflow_error("adaptive staging size overflow");next_bytes+=bytes;
            }
            move.replacement=cohort.arena->book->prepare_replace(move.slot,
                outgoing.quant_source_->host_key(move.outgoing),incoming.quant_source_->host_key(move.incoming));
        }
        bytes=next_bytes;for(int i=0;i<count;++i)s.pending.push_back(std::move(bundle[i]));
    }
    s.memory_rejections+=memory_rejections;
    for(auto& heat:s.heat)for(float& value:heat)value*=s.heat_retention;
    if(s.pending.empty()){finish_plan();return;}
    const auto options=tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true);
    if(cache.config_.moe_residency_warm &&
        (!s.backup.defined() || s.backup.numel()<bytes || (!s.direct && (!s.upload.defined() || s.upload.numel()<bytes)))) {
        s.backup={};s.upload={};
    }
    if(!s.backup.defined() || s.backup.numel()<bytes)s.backup=tb::empty({bytes},options);
    if(!s.direct && (!s.upload.defined() || s.upload.numel()<bytes))s.upload=tb::empty({bytes},options);
    s.backup_peak_bytes=std::max(s.backup_peak_bytes.load(),std::uint64_t(s.backup.numel()));
    s.pending_complete_bundles=s.complete_pending_bundles();
    s.planned_bundles+=s.pending_complete_bundles;s.planned_projections+=s.pending.size();
    auto update_map=[&](int source,int cohort,int expert,int slot) {
        auto found=std::find_if(s.maps.begin(),s.maps.end(),[&](const auto& item){return item.source==source && item.cohort==cohort;});
        if(found==s.maps.end()){s.maps.push_back({source,cohort,cache.sources_[source]->cohorts_[cohort].host_map});found=std::prev(s.maps.end());}
        found->next[expert]=slot;
    };
    for(auto& move:s.pending) {
        auto& source=*cache.sources_[move.incoming_source];auto& cohort=source.cohorts_[move.cohort];
        auto& outgoing=*cache.sources_[move.outgoing_source];
        update_map(move.incoming_source,move.cohort,move.incoming,move.slot);
        update_map(move.outgoing_source,outgoing.expert_to_cohort_[move.outgoing],move.outgoing,-1);
        cohort.arena->book->mark_inflight(move.slot);
        if(!s.direct) {
            const auto fields=moe_cache_field_refs(move.lease->weights);
            for(std::size_t f=0;f<fields.size();++f) {
                std::memcpy(s.upload.data_ptr<uint8_t>()+move.offsets[f],fields[f].data_ptr(),std::size_t(cohort.bytes_per_expert[f]));
                s.host_pack_bytes+=cohort.bytes_per_expert[f];
            }
        }
    }
    finish_plan();
    allow_copies();
    // Only fixed GPU routing without background CPU calibration uses this
    // path. The host-cache lock spans the exchange, so window completion must
    // never require a new host-cache lookup. Existing readers reject it.
    if(s.direct_exchange && s.direct && s.reuse_transfer && !cache.ssd_experts() &&
            (!s.pending_window || s.pending_expert_fence) &&
            cache.config_.moe_ram_pcie_fraction && cache.ram_pcie_fraction_==1.0 &&
            !cache.pipeline_shared_cpu_cost_ &&
            std::all_of(cache.pipeline_dispatch_replay_.begin(),cache.pipeline_dispatch_replay_.end(),
                [](const auto& record){return record.cpu.empty();}) &&
            !s.pending.empty()) {
        prepare_direct_exchange();return;
    }
    // The packed path starts with a GPU gather. On the tested WDDM runtime,
    // that side-stream work cannot pass a running graph's tail; even a fully
    // preallocated gather reproduces the stall. Queue D2H backups on the copy
    // engine for an early expert fence, preserving the existing GPU-cache reuse
    // and RAM/publication transaction. Normal windows retain packed batching.
    if(s.batched && !s.pending_expert_fence && !s.cached_refill) {
        prepare_batched(bytes);
        MFQ_CUDA_CHECK(cudaEventRecord(s.done,s.stream));
        if(s.observer) {MFQ_CUDA_CHECK(cudaEventSynchronize(s.done));s.observe("copies_complete",0);}
        return;
    }
    if(cache.ssd_experts() && cache.kv_expert_loan_) {
        const auto* option=std::getenv("MFQ_MOE_RESIDENCY_KV_REFILL");
        if(!option || option[0]!='0') {
            // CPU images still participate in the bounded RAM exchange, but
            // an immutable extra GPU copy need not cross PCIe a second time.
            // Copies are already permitted here, after predictor overlap.
            s.exchange_reads.emplace(cache.kv_expert_loan_->retain_exchange_reads(false,s.pending.size()));
            bool found=false;
            for(auto& move:s.pending)if(!move.cached_arena) {
                const auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
                for(std::size_t f=0;f<move.field_count;++f)if(cohort.bytes_per_expert[f]) {
                    move.loan_fields[f]=cache.kv_expert_loan_->find_for_exchange(
                        {move.incoming_source,move.incoming,int64_t(f)},std::size_t(cohort.bytes_per_expert[f]),&*s.exchange_reads);
                    found|=move.loan_fields[f]!=nullptr;
                }
            }
            if(!found)s.exchange_reads.reset();
        }
    }
    const bool native_ssd_dma=cache.ssd_experts() && s.direct && native_residency_dma_enabled();
    void* backup_alias=nullptr;
    if(native_ssd_dma)MFQ_CUDA_CHECK(cudaHostGetDevicePointer(&backup_alias,s.backup.data_ptr(),0));
    const auto submit_ssd_dma=[&](const std::vector<mfq::cuda::DmaCopy>& copies,cudaMemcpyKind kind) {
        if(copies.empty())return;
        auto aliases=copies;
        const auto ram_begin=reinterpret_cast<uintptr_t>(cache.pipeline_ram_complement_->data_ptr());
        const auto backup_begin=reinterpret_cast<uintptr_t>(s.backup.data_ptr());
        bool mapped=true;
        for(auto& copy:aliases) {
            const bool upload=kind==cudaMemcpyHostToDevice;
            const auto address=reinterpret_cast<uintptr_t>(upload?copy.source:copy.destination);
            const auto begin=upload?ram_begin:backup_begin;
            const auto bytes=upload?std::size_t(cache.pipeline_ram_registered_bytes_):std::size_t(s.backup.numel());
            auto* base=static_cast<uint8_t*>(upload?cache.pipeline_ram_alias_:backup_alias);
            if(!base || address<begin || address-begin>bytes || copy.bytes>bytes-(address-begin)) {
                mapped=false;break;
            }
            if(upload)copy.source=base+(address-begin);else copy.destination=base+(address-begin);
        }
        if(mapped && mfq::cuda::NativeDmaBatch::submit(aliases,s.stream,1,
                [](const void* p,std::size_t){return reinterpret_cast<CUdeviceptr>(p);})) {
            ++s.native_dma_batches;s.native_dma_copies+=copies.size();
        } else {
            ++s.native_dma_fallbacks;
            for(const auto& copy:copies)MFQ_CUDA_CHECK(cudaMemcpyAsync(copy.destination,copy.source,copy.bytes,kind,s.stream));
        }
    };
    {
    const PhaseClock backup_clock(s.backup_submit_ns,nullptr);
    std::vector<mfq::cuda::DmaCopy> copies;
    for(const auto& move:s.pending) {
        auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
        for(std::size_t f=0;f<move.field_count;++f) {
            const auto count=cohort.bytes_per_expert[f];if(!count)continue;
            auto* gpu=static_cast<uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(move.slot)*count;
            auto* backup=s.backup.data_ptr<uint8_t>()+move.offsets[f];
            if(native_ssd_dma)copies.push_back({backup,gpu,std::size_t(count)});
            else MFQ_CUDA_CHECK(cudaMemcpyAsync(backup,gpu,std::size_t(count),cudaMemcpyDeviceToHost,s.stream));
            ++s.backup_copies;
        }
    }
    submit_ssd_dma(copies,cudaMemcpyDeviceToHost);
    }
    {
    const PhaseClock refill_clock(s.refill_submit_ns,nullptr);
    Impl::CachedRefill refill(s);
    std::vector<mfq::cuda::DmaCopy> copies;
    for(std::size_t i=0;i<s.pending.size();++i) {
        auto& move=s.pending[i];auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
        const auto incoming=moe_cache_field_refs(move.lease->weights);
        for(std::size_t f=0;f<move.field_count;++f) {
            const auto count=cohort.bytes_per_expert[f];if(!count)continue;
            auto* gpu=static_cast<uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(move.slot)*count;
            move.touched_fields=f+1;
            if(move.cached_arena) {
                const auto* cached=move.cached_arena->field_pointer(move.cached_slot,f,cohort.bytes_per_expert);
                // The window event fences the final transfer scatter. The
                // next window joins this transaction before reusing its slots.
                refill.copy(gpu,cached,std::uint64_t(count));
            } else if(move.loan_fields[f]) {
                refill.copy(gpu,static_cast<const uint8_t*>(move.loan_fields[f]),std::uint64_t(count));
                s.kv_refill_bytes+=count;
            } else {
                const auto* host=s.direct?static_cast<const uint8_t*>(incoming[f].data_ptr()):s.upload.data_ptr<uint8_t>()+move.offsets[f];
                if(native_ssd_dma)copies.push_back({gpu,host,std::size_t(count)});
                else MFQ_CUDA_CHECK(cudaMemcpyAsync(gpu,host,std::size_t(count),cudaMemcpyHostToDevice,s.stream));
                ++s.upload_copies;
                if(s.direct)s.direct_upload_bytes+=count;
            }
        }
        s.observe("upload",i);
    }
    // All backups precede this kernel; publication still waits for done.
    submit_ssd_dma(copies,cudaMemcpyHostToDevice);
    refill.submit();
    }
    MFQ_CUDA_CHECK(cudaEventRecord(s.done,s.stream));
    if(s.observer) {MFQ_CUDA_CHECK(cudaEventSynchronize(s.done));s.observe("copies_complete",0);}
}
void MoeResidencyManager::prepare_direct_exchange() {
    auto& s=*impl_;auto& cache=*s.cache;
    const PhaseClock direct_clock(s.direct_exchange_ns,nullptr);
    // Preserve only uncached incoming fields. Cached ones already have an
    // immutable GPU copy until publication. One cache miss must not force a
    // backup-to-staging and CPU copy of every victim in this transaction.
    // The expert fence has retired the wire's previous consumers, and the next
    // decode joins this exchange before reusing it. Preserve its control header
    // and borrow only its payload. This avoids a CPU copy without growing VRAM.
    const bool native_dma=native_residency_dma_enabled();
    const auto native_host_alias=[&](const void* pointer,std::size_t count) -> void* {
        const auto address=reinterpret_cast<uintptr_t>(pointer);
        const auto begin=reinterpret_cast<uintptr_t>(cache.pipeline_ram_complement_->data_ptr());
        const auto bytes=uint64_t(cache.pipeline_ram_registered_bytes_);
        auto* alias=static_cast<uint8_t*>(cache.pipeline_ram_alias_?cache.pipeline_ram_alias_:s.backup_ram_device);
        return alias && address>=begin && address-begin<=bytes && count<=bytes-(address-begin)
            ?alias+(address-begin):nullptr;
    };
    const auto submit_native=[&](std::vector<mfq::cuda::DmaCopy>& copies,cudaMemcpyKind kind,cudaStream_t stream) {
        if(copies.empty())return;
        std::vector<mfq::cuda::DmaCopy> aliases=copies;
        bool mapped=true;
        for(auto& copy:aliases) {
            if(kind==cudaMemcpyHostToDevice) {
                const auto* source=native_host_alias(copy.source,copy.bytes);
                if(!source){mapped=false;break;}copy.source=source;
            } else {
                auto* destination=native_host_alias(copy.destination,copy.bytes);
                if(!destination){mapped=false;break;}copy.destination=destination;
            }
        }
        if(mapped && mfq::cuda::NativeDmaBatch::submit(aliases,stream,1,
                [](const void* pointer,std::size_t){return reinterpret_cast<CUdeviceptr>(pointer);})) {
            ++s.native_dma_batches;s.native_dma_copies+=copies.size();
        } else {
            ++s.native_dma_fallbacks;
            for(const auto& copy:copies)MFQ_CUDA_CHECK(cudaMemcpyAsync(copy.destination,copy.source,copy.bytes,kind,stream));
        }
        copies.clear();
    };
    constexpr int64_t wire_control_bytes=256;
    const auto* loan_option=std::getenv("MFQ_MOE_RESIDENCY_KV_REFILL");
    const bool reuse_loan=(!loan_option || loan_option[0]!='0') && bool(cache.kv_expert_loan_);
    const auto* keyed_option=std::getenv("MFQ_MOE_KEYED_EXCHANGE");
    bool predictor_disjoint=s.predictor_overlap && s.pending_tokens<=8 && reuse_loan && (!keyed_option || keyed_option[0]=='1') &&
        std::all_of(s.pending.begin(),s.pending.end(),[&](const auto& move) {
            return !move.cached_arena && cache.sources_[move.incoming_source]->layer_id_>=0 &&
                cache.sources_[move.outgoing_source]->layer_id_>=0;
        });
    if(predictor_disjoint)for(const auto& source:cache.sources_)if(source->layer_id_<0)
        for(const auto& cohort:source->cohorts_)for(const auto& move:s.pending)
            if(cohort.arena==cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena)
                predictor_disjoint=false;
    if(predictor_disjoint)s.exchange_reads.emplace(cache.kv_expert_loan_->retain_exchange_reads(false,s.pending.size()));
    const auto cached_field=[&](const Impl::Move& move,std::size_t f) -> const uint8_t* {
        const auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
        if(move.cached_arena)return move.cached_arena->field_pointer(move.cached_slot,f,cohort.bytes_per_expert);
        return f<move.field_count?static_cast<const uint8_t*>(move.loan_fields[f]):nullptr;
    };
    int64_t incoming_bytes=0;
    for(auto& move:s.pending)if(!move.cached_arena) {
        const auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
        if(reuse_loan) {
            for(std::size_t f=0;f<move.field_count;++f)if(cohort.bytes_per_expert[f])
                move.loan_fields[f]=cache.kv_expert_loan_->find_for_exchange(
                    {move.incoming_source,move.incoming,int64_t(f)},std::size_t(cohort.bytes_per_expert[f]),
                    s.exchange_reads?&*s.exchange_reads:nullptr);
        }
        for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f) {
            const auto bytes=cohort.bytes_per_expert[f];
            if(cached_field(move,f)){move.preserve_offsets[f]=-1;continue;}
            incoming_bytes=(incoming_bytes+15)&~int64_t(15);
            move.preserve_offsets[f]=incoming_bytes;
            if(bytes>INT64_MAX-incoming_bytes)throw std::overflow_error("incoming expert preservation size overflow");
            incoming_bytes+=bytes;move.uncached_bytes+=bytes;
        }
    }
    const auto* shadow_option=std::getenv("MFQ_MOE_RESIDENCY_PREFILL_SHADOW");
    const auto* shadow_index_option=std::getenv("MFQ_MOE_RESIDENCY_PREFILL_SHADOW_INDEX");
    const bool indexed_shadow=!shadow_index_option || shadow_index_option[0]!='0';
    if(!shadow_option || shadow_option[0]!='0')for(auto& move:s.pending) {
        const auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
        const auto& arena=*cohort.arena;
        if(!arena.prefill_backup_key)continue;
        bool reusable=false;
        for(std::size_t f=0;f<move.field_count;++f)
            reusable|=cohort.bytes_per_expert[f]>0 && cached_field(move,f)!=nullptr;
        if(!reusable)continue;
        auto owner=cache.host_experts_->find(*arena.prefill_backup_key);
        if(!owner)continue;
        const auto& keys=owner->weights.expert_local;
        if(!keys.defined() || keys.is_cuda() || keys.scalar_type()!=tb::kInt32 ||
                keys.dim()!=2 || keys.size(1)!=3 || !keys.is_contiguous())continue;
        const auto image_slots=keys.size(0);auto row=int64_t(move.slot)+image_slots-arena.slots;
        const auto old_key=cache.sources_[move.outgoing_source]->quant_source_->host_key(move.outgoing);
        const auto matches=[&](int64_t row) {
            if(row<0 || row>=image_slots)return false;
            const auto* identity=keys.data_ptr<int32_t>()+row*3;
            return identity[0]==old_key.source && identity[1]==old_key.cohort && identity[2]==old_key.expert;
        };
        if(!matches(row)) {
            if(!indexed_shadow)continue;
            row=s.find_shadow_row(arena,owner,old_key);if(row<0)continue;
            move.shadow_indexed=true;
        }
        const auto fields=moe_cache_field_refs(owner->weights);
        if(fields.size()!=move.field_count)continue;
        for(std::size_t f=0;f<move.field_count;++f) {
            const auto bytes=cohort.bytes_per_expert[f];
            if(bytes>0 && cached_field(move,f) && !fields[f].is_cuda() && fields[f].is_contiguous() &&
                    tensor_nbytes(fields[f])/bytes==image_slots && tensor_nbytes(fields[f])%bytes==0)
                move.shadow_fields[f]=static_cast<const uint8_t*>(fields[f].data_ptr())+row*bytes;
        }
        move.shadow_owner=std::move(owner);
    }
    const bool gpu_preserve=incoming_bytes>0 && cache.pipeline_wire_gpu_.defined() &&
        incoming_bytes<=cache.pipeline_wire_gpu_.numel()-wire_control_bytes;
    // With every incoming field already cached, this transaction neither
    // borrows the shared wire nor reads shared pinned transfer staging. Its
    // primary arenas and RAM keys are disjoint from the predictor's. The read
    // guard prevents a predictor packet from reclaiming cached DMA sources.
    const bool overlap_candidate=predictor_disjoint && incoming_bytes==0;
    s.last_gpu_preserve=gpu_preserve;s.last_preserve_bytes=incoming_bytes;
    s.last_backup_bytes=0;s.last_small_backup_bytes=0;
    s.last_backup_fields=0;s.last_small_backup_fields=0;
    const auto* duplex_option=std::getenv("MFQ_MOE_RESIDENCY_DUPLEX");
    const bool duplex=gpu_preserve && (!duplex_option || duplex_option[0]!='0');
    const auto* group_option=std::getenv("MFQ_MOE_RESIDENCY_DUPLEX_CHUNK_BYTES");
    const int64_t group_limit=group_option?std::clamp<int64_t>(std::atoll(group_option),4096,int64_t(64)<<20):int64_t(8)<<20;
    std::vector<std::vector<std::size_t>> preserve_groups;
    if(duplex) {
        int64_t group_bytes=0;
        for(std::size_t i=0;i<s.pending.size();++i)if(s.pending[i].uncached_bytes) {
            const auto bytes=s.pending[i].uncached_bytes;
            if(preserve_groups.empty() || group_bytes+bytes>group_limit) {
                preserve_groups.emplace_back();group_bytes=0;
            }
            preserve_groups.back().push_back(i);group_bytes+=bytes;
        }
    }
    if(duplex) {
        if(!s.backup_stream)MFQ_CUDA_CHECK(cudaStreamCreateWithFlags(&s.backup_stream,cudaStreamNonBlocking));
        while(s.preserve_events.size()<preserve_groups.size()) {
            cudaEvent_t event=nullptr;
            MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
            s.preserve_events.push_back(event);
        }
        if(!s.backup_done)MFQ_CUDA_CHECK(cudaEventCreateWithFlags(&s.backup_done,cudaEventDisableTiming));
        if(!s.pending_expert_fence)MFQ_CUDA_CHECK(cudaStreamWaitEvent(s.backup_stream,s.window_done,0));
    }
    const auto backup_stream=duplex?s.backup_stream:s.stream;
    // Also retire backup copies when an observer/CUDA failure occurs before
    // the cross-stream event is recorded. Normal completion has already waited.
    struct RetireBackup {
        cudaStream_t stream;
        ~RetireBackup(){if(stream)(void)cudaStreamSynchronize(stream);}
    } retire_backup{duplex?backup_stream:nullptr};
    const auto incoming_copy=[&](const Impl::Move& move,std::size_t field) {
        return gpu_preserve?cache.pipeline_wire_gpu_.data_ptr<uint8_t>()+wire_control_bytes+move.preserve_offsets[field]
            :s.backup.data_ptr<uint8_t>()+move.offsets[field];
    };
    {
        mfq::cpu::HostMemoryCopyBatch copies;
        std::vector<mfq::cuda::DmaCopy> preserve_copies;
        const auto flush_preserve=[&]{submit_native(preserve_copies,cudaMemcpyHostToDevice,s.stream);};
        const auto preserve=[&](const Impl::Move& move) {
            const auto fields=moe_cache_field_refs(move.lease->weights);
            const auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
            for(std::size_t f=0;f<fields.size();++f) {
                const auto bytes=cohort.bytes_per_expert[f];if(!bytes)continue;
                if(cached_field(move,f))continue;
                if(gpu_preserve) {
                    if(native_dma) {
                        preserve_copies.push_back({incoming_copy(move,f),fields[f].data_ptr(),std::size_t(bytes)});
                        if(preserve_copies.size()==128)flush_preserve();
                    } else MFQ_CUDA_CHECK(cudaMemcpyAsync(incoming_copy(move,f),fields[f].data_ptr(),
                        std::size_t(bytes),cudaMemcpyHostToDevice,s.stream));
                } else copies.copy(incoming_copy(move,f),fields[f].data_ptr(),std::size_t(bytes));
            }
        };
        if(duplex)for(std::size_t g=0;g<preserve_groups.size();++g) {
            for(auto i:preserve_groups[g])preserve(s.pending[i]);
            flush_preserve();
            MFQ_CUDA_CHECK(cudaEventRecord(s.preserve_events[g],s.stream));
        } else {
            for(const auto& move:s.pending)if(!move.cached_arena)preserve(move);
            flush_preserve();
        }
    }
    // This callback runs before the cache lock, allowing an arriving reader to
    // retain its lease. exchange_batch must then reject without changing bytes.
    s.observe("before_direct_exchange",0);
    std::vector<MoeHostExpertCache::Exchange> changes;changes.reserve(s.pending.size());
    for(std::size_t i=0;i<s.pending.size();++i) {
        auto& move=s.pending[i];
        auto& incoming=*cache.sources_[move.incoming_source];
        auto& outgoing=*cache.sources_[move.outgoing_source];
        changes.push_back({incoming.quant_source_->host_key(move.incoming),
            outgoing.quant_source_->host_key(move.outgoing),&move.lease,
            [&,i](const MixedMoePool& reused) {
                const auto& item=s.pending[i];auto& old=*cache.sources_[item.outgoing_source];
                const auto& cohort=old.cohorts_[old.expert_to_cohort_[item.outgoing]];
                s.observe("host",i);auto result=*cohort.cpu;result.local_experts=1;result.expert_local={};
                return moe_replace_quant_fields(std::move(result),moe_cache_field_refs(reused));
            },[&,i](const MixedMoePool& reused) {
                auto& item=s.pending[i];
                const auto& cohort=cache.sources_[item.incoming_source]->cohorts_[item.cohort];
                const auto fields=moe_cache_field_refs(reused);
                MFQ_CUDA_CHECK(cudaStreamSynchronize(backup_stream));
                MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
                // RAM now contains the victim. Restore its GPU bytes before
                // restoring the incoming RAM payload from its retained cache.
                for(std::size_t f=0;f<item.touched_fields;++f) {
                    const auto count=cohort.bytes_per_expert[f];if(!count)continue;
                    auto* gpu=static_cast<uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(item.slot)*count;
                    MFQ_CUDA_CHECK(cudaMemcpyAsync(gpu,fields[f].data_ptr(),std::size_t(count),cudaMemcpyHostToDevice,s.stream));
                }
                for(std::size_t f=0;f<item.backed_fields;++f) {
                    const auto count=cohort.bytes_per_expert[f];if(!count)continue;
                    if(const auto* cached=cached_field(item,f)) {
                        MFQ_CUDA_CHECK(cudaMemcpyAsync(fields[f].data_ptr(),cached,std::size_t(count),cudaMemcpyDeviceToHost,s.stream));
                    } else if(gpu_preserve) {
                        MFQ_CUDA_CHECK(cudaMemcpyAsync(fields[f].data_ptr(),incoming_copy(item,f),
                            std::size_t(count),cudaMemcpyDeviceToHost,s.stream));
                    } else {
                        // Finish restoring the old GPU owner before replacing
                        // the RAM source that restoration is reading.
                        MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
                        std::memcpy(fields[f].data_ptr(),incoming_copy(item,f),std::size_t(count));
                    }
                }
                MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
                item.touched_fields=0;item.backed_fields=0;
            }});
    }
    const bool committed=cache.host_experts_->exchange_batch(changes,[&] {
        {
            const PhaseClock wait_clock(s.dma_wait_ns,s.pending_tokens<=8?&s.decode_dma_wait_ns:nullptr);
            MFQ_CUDA_CHECK(cudaEventSynchronize(s.done));
            s.observe("copies_complete",0);
        }
        // Final GPU expert consumers have finished before prepare() enters
        // this path. Keep registry and GPU map publication at window finish.
        {
            std::unique_lock<std::mutex> lock(s.worker_mutex);
            s.work_finished.wait(lock,[&]{return !s.window_inflight || s.stopping;});
            if(s.stopping)throw std::runtime_error("adaptive direct exchange cancelled before window finish");
        }
        {
            const PhaseClock map_clock(s.map_ns,s.pending_tokens<=8?&s.decode_map_ns:nullptr);
            for(const auto& move:s.pending)
                if(!cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->replacement_valid(move.replacement))
                    throw std::runtime_error("adaptive GPU slot changed before direct publication");
            s.upload_maps();
            MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
        }
        s.observe("publish",0);
        try {
            for(auto& move:s.pending)
                if(!cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->commit_replace(move.replacement))
                    throw std::runtime_error("adaptive direct replacement lost its generation");
        } catch(...) {
            for(auto& move:s.pending)
                cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->rollback_replace(move.replacement);
            throw;
        }
        for(auto& map:s.maps) {
            auto& cohort=cache.sources_[map.source]->cohorts_[map.cohort];
            cohort.host_map.swap(map.next);cohort.map_dirty=false;
        }
        for(auto& move:s.pending) {
            cache.sources_[move.outgoing_source]->quant_source_->publish_gpu_resident(move.outgoing,false);
            cache.sources_[move.incoming_source]->quant_source_->publish_gpu_resident(move.incoming,true);
        }
    },[&] {
        if(overlap_candidate) {
            // exchange_batch has reserved just its own incoming/outgoing keys
            // and released the global host-cache mutex before invoking us.
            {std::lock_guard<std::mutex> lock(s.worker_mutex);s.predictor_overlap_allowed=true;}
            if(++s.predictor_overlap_rounds==1)std::fprintf(stderr,
                "moe_predictor_exchange_overlap_dispatch enabled=1 cached_fields=1 separate_arenas=1 scoped_pins=1 projections=%zu\n",s.pending.size());
            s.work_finished.notify_all();s.observe("predictor_overlap_ready",0);
        }
        {
            const PhaseClock backup_clock(s.backup_submit_ns,nullptr);
            std::size_t small_count=0;
            std::vector<mfq::cuda::DmaCopy> backup_copies;
            const auto flush_backup=[&]{submit_native(backup_copies,cudaMemcpyDeviceToHost,backup_stream);};
            const auto ram_begin=reinterpret_cast<uintptr_t>(cache.pipeline_ram_complement_->data_ptr());
            const auto ram_bytes=uint64_t(cache.pipeline_ram_registered_bytes_);
            const auto backup=[&](Impl::Move& move) {
                const auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
                const auto fields=moe_cache_field_refs(move.lease->weights);
                for(std::size_t f=0;f<fields.size();++f) {
                    const auto count=cohort.bytes_per_expert[f];if(!count)continue;
                    const auto* gpu=static_cast<const uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(move.slot)*count;
                    move.backed_fields=f+1;
                    // Cached incoming bytes have no preservation read from
                    // this RAM destination. Reader exclusion is already held,
                    // so an immutable, identity-checked prefill image can
                    // replace the victim's D2H without another GPU fence.
                    // Uncached incoming fields must keep their queued backup.
                    if(const auto* shadow=move.shadow_fields[f]) {
                        std::memcpy(fields[f].data_ptr(),shadow,std::size_t(count));
                        s.shadow_backup_bytes+=count;++s.shadow_backup_fields;
                        if(move.shadow_indexed) {
                            s.shadow_index_backup_bytes+=count;
                            if(!s.shadow_index_reported) {
                                s.shadow_index_reported=true;
                                std::cerr<<"runtime_moe_shadow_lookup enabled=1 identity_lookup=1 source="
                                    <<move.outgoing_source<<" expert="<<move.outgoing<<'\n';
                            }
                        }
                        s.observe("direct_backup",f);continue;
                    }
                    const auto address=reinterpret_cast<uintptr_t>(fields[f].data_ptr());
                    if(s.backup_descriptors && count<=s.small_backup_limit && small_count<Impl::refill_capacity &&
                            address>=ram_begin && address-ram_begin<=ram_bytes && uint64_t(count)<=ram_bytes-(address-ram_begin)) {
                        auto* descriptors=static_cast<mfq::MoeCacheMappedCopyDescriptor*>(s.backup_descriptors->data());
                        descriptors[small_count++]={reinterpret_cast<uint64_t>(s.backup_ram_device+(address-ram_begin)),
                            reinterpret_cast<uint64_t>(gpu),uint64_t(count)};
                        s.last_small_backup_bytes+=count;++s.last_small_backup_fields;
                    } else {
                        if(native_dma) {
                            backup_copies.push_back({fields[f].data_ptr(),gpu,std::size_t(count)});
                            if(backup_copies.size()==128)flush_backup();
                        } else MFQ_CUDA_CHECK(cudaMemcpyAsync(fields[f].data_ptr(),gpu,std::size_t(count),cudaMemcpyDeviceToHost,backup_stream));
                        ++s.backup_copies;
                    }
                    s.last_backup_bytes+=count;++s.last_backup_fields;s.observe("direct_backup",f);
                }
            };
            if(duplex) {
                // Overlap opposite PCIe directions. Each uncached RAM range
                // becomes writable only after its own preservation group;
                // later H2D groups can still run while earlier victims drain.
                for(auto& move:s.pending)if(!move.uncached_bytes)backup(move);
                flush_backup();
                for(std::size_t g=0;g<preserve_groups.size();++g) {
                    MFQ_CUDA_CHECK(cudaStreamWaitEvent(backup_stream,s.preserve_events[g],0));
                    for(auto i:preserve_groups[g])backup(s.pending[i]);
                    flush_backup();
                }
            } else {
                for(auto& move:s.pending)backup(move);
                flush_backup();
            }
            // Large victims use the copy engine as soon as their inputs are
            // preserved. Metadata shares one mapped write kernel after those
            // transfers, without hundreds of tiny DMA submissions. Waiting on
            // every preservation group above also protects deferred RAM writes.
            if(small_count) {
                std::atomic_thread_fence(std::memory_order_seq_cst);
                mfq::moe_cache_mapped_gather_cuda(s.backup_descriptors_device,int(small_count),1,backup_stream);
                s.observe("direct_backup_batch",0);
            }
            if(duplex) {
                MFQ_CUDA_CHECK(cudaEventRecord(s.backup_done,backup_stream));
                MFQ_CUDA_CHECK(cudaStreamWaitEvent(s.stream,s.backup_done,0));
            }
        }
        {
            const PhaseClock refill_clock(s.refill_submit_ns,nullptr);
            Impl::CachedRefill refill(s);
            for(std::size_t i=0;i<s.pending.size();++i) {
                auto& move=s.pending[i];const auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
                for(std::size_t f=0;f<cohort.bytes_per_expert.size();++f) {
                    const auto count=cohort.bytes_per_expert[f];if(!count)continue;
                    auto* gpu=static_cast<uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(move.slot)*count;
                    move.touched_fields=f+1;
                    // All victims have already been backed up into final RAM.
                    // Mark queued fields before launch so either submission or
                    // publication failures restore the complete transaction.
                    if(const auto* cached=cached_field(move,f)) {
                        refill.copy(gpu,cached,std::uint64_t(count));
                        if(!move.cached_arena)s.kv_refill_bytes+=count;
                    } else if(gpu_preserve) {
                        refill.copy(gpu,incoming_copy(move,f),std::uint64_t(count));
                    } else {
                        MFQ_CUDA_CHECK(cudaMemcpyAsync(gpu,incoming_copy(move,f),
                            std::size_t(count),cudaMemcpyHostToDevice,s.stream));
                        ++s.upload_copies;s.direct_upload_bytes+=count;
                    }
                    s.observe("direct_refill",f);
                }
                s.observe("upload",i);
            }
            if(refill.submit())s.observe("direct_refill_batch",0);
        }
        MFQ_CUDA_CHECK(cudaEventRecord(s.done,s.stream));
    },overlap_candidate);
    if(!committed){rollback();return;}
    s.retire_promoted_kv();
    for(const auto& move:s.pending)if(move.prefill_ram_slot>=0)
        cache.sources_[move.incoming_source]->cohorts_[move.cohort].prefill_ram_experts[move.prefill_ram_slot]=move.outgoing;
    s.committed_bundles+=s.pending_complete_bundles;s.committed_projections+=s.pending.size();
    ++s.direct_exchange_rounds;
    for(const auto& move:s.pending) {
        ++cache.stats_.evictions;
        const auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
        cache.stats_.gpu_demote_bytes+=cohort.arena->slot_bytes;
        for(auto count:cohort.bytes_per_expert)s.direct_exchange_bytes+=count;
    }
    s.pending.clear();s.maps.clear();
}

void MoeResidencyManager::prepare_batched(std::int64_t bytes) {
    auto& s=*impl_;auto& cache=*s.cache;
    // cudaMallocAsync must be ordered with the copy kernels using its storage,
    // independently of a model stream that may still be executing its tail.
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    const MfqCudaStreamGuard stream_guard(MfqCudaStream(mfq_current_cuda_device(),s.stream));
#else
    const MfqCudaStreamGuard stream_guard(at::cuda::getStreamFromExternal(s.stream,mfq_current_cuda_device()));
#endif
    struct Field {const uint8_t* host;uint8_t* gpu;int64_t bytes,backup_offset,upload_offset;const uint8_t* cached;};
    struct Interval {const uint8_t* host;int64_t bytes,offset;};
    std::vector<Field> fields;
    int64_t largest=0;
    for(const auto& move:s.pending) {
        const auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
        const auto incoming=moe_cache_field_refs(move.lease->weights);
        for(std::size_t f=0;f<move.field_count;++f) {
            const auto count=cohort.bytes_per_expert[f];if(!count)continue;
            auto* gpu=static_cast<uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(move.slot)*count;
            const auto* cached=move.cached_arena ? move.cached_arena->field_pointer(move.cached_slot,f,cohort.bytes_per_expert) : nullptr;
            fields.push_back({s.direct?static_cast<const uint8_t*>(incoming[f].data_ptr()):nullptr,
                gpu,count,move.offsets[f],move.offsets[f],cached});
            largest=std::max(largest,count);
        }
    }
    if(fields.empty() || fields.size()>65535)
        throw std::overflow_error("adaptive copy descriptors exceed CUDA grid geometry");
    std::vector<Interval> intervals;
    int64_t upload_bytes=bytes;
    if(s.direct) {
        std::sort(fields.begin(),fields.end(),[](const Field& a,const Field& b) {
            return reinterpret_cast<uintptr_t>(a.host)<reinterpret_cast<uintptr_t>(b.host);
        });
        upload_bytes=0;
        for(std::size_t first=0;first<fields.size();) {
            if(fields[first].cached){++first;continue;}
            const auto address=reinterpret_cast<uintptr_t>(fields[first].host);
            auto end=address+fields[first].bytes;auto last=first+1;
            while(last<fields.size()) {
                if(fields[last].cached)break;
                const auto next=reinterpret_cast<uintptr_t>(fields[last].host);
                if(next<end || next-end>15)break;
                end=next+fields[last].bytes;++last;
            }
            upload_bytes=(upload_bytes+15)&~int64_t(15);
            const auto count=static_cast<int64_t>(end-address);
            if(count>INT64_MAX-upload_bytes)throw std::overflow_error("adaptive upload interval overflow");
            for(auto i=first;i<last;++i)fields[i].upload_offset=upload_bytes+
                static_cast<int64_t>(reinterpret_cast<uintptr_t>(fields[i].host)-address);
            intervals.push_back({fields[first].host,count,upload_bytes});
            upload_bytes+=count;first=last;
        }
    }
    const auto gpu_options=s.cache->sources_.front()->cohorts_.front().active.expert_local.options().dtype(tb::kUInt8);
    const auto host_options=tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true);
    const int64_t work_bytes=std::max(bytes,upload_bytes);
    const int64_t descriptor_bytes=int64_t(fields.size())*sizeof(mfq::MoeCacheMappedCopyDescriptor);
    // These copies are outside captured graphs. Release a larger prefill
    // workspace at a row-count transition; steady decode keeps reusing it.
    if(s.work_tokens!=s.pending_tokens) {
        if(s.device_work.defined() && s.device_work.numel()>work_bytes)s.device_work={};
        if(s.copy_descriptors.defined() && s.copy_descriptors.numel()>2*descriptor_bytes) {
            s.copy_descriptors={};s.device_descriptors={};
        }
    }
    s.work_tokens=s.pending_tokens;
    s.observe("workspace_begin",0);
    if(!s.device_work.defined() || s.device_work.numel()<work_bytes)s.device_work=tb::empty({work_bytes},gpu_options);
    s.observe("workspace_device",0);
    if(!s.copy_descriptors.defined() || s.copy_descriptors.numel()<2*descriptor_bytes) {
        s.copy_descriptors=tb::empty({2*descriptor_bytes},host_options);
        s.observe("workspace_host_descriptors",0);
        s.device_descriptors=tb::empty({2*descriptor_bytes},gpu_options);
    }
    s.observe("workspace_ready",0);
    s.batch_device_bytes=s.device_work.numel()+s.device_descriptors.numel();
    auto* work=s.device_work.data_ptr<uint8_t>();
    auto* host=reinterpret_cast<mfq::MoeCacheMappedCopyDescriptor*>(s.copy_descriptors.data_ptr<uint8_t>());
    auto* device=reinterpret_cast<mfq::MoeCacheMappedCopyDescriptor*>(s.device_descriptors.data_ptr<uint8_t>());
    for(std::size_t f=0;f<fields.size();++f) {
        const auto& field=fields[f];
        host[f]={reinterpret_cast<uint64_t>(work+field.backup_offset),reinterpret_cast<uint64_t>(field.gpu),uint64_t(field.bytes)};
        host[fields.size()+f]={reinterpret_cast<uint64_t>(field.gpu),
            reinterpret_cast<uint64_t>(field.cached?field.cached:work+field.upload_offset),uint64_t(field.bytes)};
        if(field.cached){s.cached_copy_bytes+=field.bytes;++s.cached_copy_fields;}
    }
    MFQ_CUDA_CHECK(cudaMemcpyAsync(device,host,std::size_t(2*descriptor_bytes),cudaMemcpyHostToDevice,s.stream));
    ++s.batch_descriptor_copies;
    // One device workspace serves both directions. The stream completes the
    // outgoing backup before reusing its bytes or changing any resident field.
    const int blocks=static_cast<int>(std::min<int64_t>(128,(largest+4095)/4096));
    mfq::moe_cache_mapped_gather_cuda(device,int(fields.size()),blocks,s.stream);
    MFQ_CUDA_CHECK(cudaMemcpyAsync(s.backup.data_ptr(),work,std::size_t(bytes),cudaMemcpyDeviceToHost,s.stream));
    ++s.backup_copies;
    s.observe("backup",0);
    if(s.direct) {
        for(const auto& interval:intervals) {
            MFQ_CUDA_CHECK(cudaMemcpyAsync(work+interval.offset,interval.host,std::size_t(interval.bytes),cudaMemcpyHostToDevice,s.stream));
            ++s.upload_copies;s.direct_upload_bytes+=interval.bytes;
        }
    } else {
        MFQ_CUDA_CHECK(cudaMemcpyAsync(work,s.upload.data_ptr(),std::size_t(bytes),cudaMemcpyHostToDevice,s.stream));
        ++s.upload_copies;
    }
    for(auto& move:s.pending)move.touched_fields=move.field_count;
    mfq::moe_cache_mapped_gather_cuda(device+fields.size(),int(fields.size()),blocks,s.stream);
    for(std::size_t i=0;i<s.pending.size();++i)s.observe("upload",i);
}
void MoeResidencyManager::rollback() {
    auto& s=*impl_;auto& cache=*s.cache;
    const MfqCudaGuard guard(cache.sources_.front()->cohorts_.front().active.expert_local.device());
    MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
    for(auto& move:s.pending) {
        auto& cohort=cache.sources_[move.incoming_source]->cohorts_[move.cohort];
        for(std::size_t f=0;f<move.touched_fields;++f) {
            const auto count=cohort.bytes_per_expert[f];auto* gpu=static_cast<uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(move.slot)*count;
            MFQ_CUDA_CHECK(cudaMemcpyAsync(gpu,s.backup.data_ptr<uint8_t>()+move.offsets[f],std::size_t(count),cudaMemcpyHostToDevice,s.stream));
        }
    }
    for(const auto& map:s.maps) {
        auto& cohort=cache.sources_[map.source]->cohorts_[map.cohort];
        MFQ_CUDA_CHECK(cudaMemcpyAsync(cohort.active.expert_local.data_ptr<int32_t>(),cohort.host_map.data(),
            cohort.host_map.size()*sizeof(int32_t),cudaMemcpyHostToDevice,s.stream));
    }
    MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
    for(auto& move:s.pending)cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->rollback_replace(move.replacement);
    s.pending.clear();s.maps.clear();
}
void MoeResidencyManager::apply_pending() {
    auto& s=*impl_;s.predictor_overlap_scope=false;
    if(!s.work_pending.load(std::memory_order_acquire) && s.pending.empty() && !s.error)return;
    {std::lock_guard lock(s.worker_mutex);s.ssd_copies_permitted=true;}
    s.work_finished.notify_all();
    const bool decode=s.pending_tokens<=8;
    const PhaseClock apply_clock(s.apply_ns,decode?&s.decode_apply_ns:nullptr);
    const auto started=std::chrono::steady_clock::now();
    const auto joined=s.join_ns.load(),dma=s.dma_wait_ns.load(),maps=s.map_ns.load();
    {
        const PhaseClock join_clock(s.join_ns,decode?&s.decode_join_ns:nullptr);
        if(s.work_pending.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(s.worker_mutex);
            s.work_finished.wait(lock,[&]{return s.work_ready;});s.work_ready=false;
            s.work_pending.store(false,std::memory_order_release);
        }
    }
    // SSD prepares may return with device copies still queued. Keep borrowed
    // addresses pinned until publication (or rollback) retires their reads.
    struct ReleaseReads {
        decltype(s.exchange_reads)& guard;
        ~ReleaseReads(){guard.reset();}
    } release_reads{s.exchange_reads};
    if(s.error){auto failure=s.error;s.error=nullptr;std::rethrow_exception(failure);}
    publish_pending();
    const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    if(decode && elapsed>=4.0 && s.traced_decode_waits++<32)
        std::cerr<<"runtime_moe_exchange_wait tokens="<<s.pending_tokens<<" ms="<<elapsed
            <<" join_ms="<<(s.join_ns.load()-joined)/1e6
            <<" plan_ms="<<s.last_plan_ns/1e6<<" staging_budget_skips="<<s.last_staging_budget_skips
            <<" dma_ms="<<(s.dma_wait_ns.load()-dma)/1e6
            <<" map_ms="<<(s.map_ns.load()-maps)/1e6
            <<" gpu_preserve="<<s.last_gpu_preserve<<" preserve_mb="<<s.last_preserve_bytes/1048576.0
            <<" backup_mb="<<s.last_backup_bytes/1048576.0<<" backup_fields="<<s.last_backup_fields
            <<" small_backup_mb="<<s.last_small_backup_bytes/1048576.0<<" small_backup_fields="<<s.last_small_backup_fields<<'\n';
}
void MoeResidencyManager::publish_pending() {
    auto& s=*impl_;
    if(s.pending.empty())return;
    const bool decode=s.pending_tokens<=8;
    auto& cache=*s.cache;const MfqCudaGuard guard(cache.sources_.front()->cohorts_.front().active.expert_local.device());
    try {
        {
        const PhaseClock dma_clock(s.dma_wait_ns,decode?&s.decode_dma_wait_ns:nullptr);
        MFQ_CUDA_CHECK(cudaEventSynchronize(s.done));
        if(!cache.pipeline_dma_leases_.empty()) {
            MFQ_CUDA_CHECK(cudaEventSynchronize(cache.transfer_ready_));
            cache.pipeline_dma_leases_.clear();
        }
        }
        {
        const PhaseClock map_clock(s.map_ns,decode?&s.decode_map_ns:nullptr);
        for(const auto& move:s.pending)if(!cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->replacement_valid(move.replacement))
            throw std::runtime_error("adaptive GPU slot changed before publication");
        s.upload_maps();
        MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
        }
        const PhaseClock host_clock(s.host_exchange_ns,decode?&s.decode_host_exchange_ns:nullptr);
        std::vector<MoeHostExpertCache::Exchange> changes;changes.reserve(s.pending.size());
        struct HostCopy {uint8_t* destination;const uint8_t* source;int64_t begin,end;};
        std::vector<HostCopy> copies;int64_t copy_bytes=0;
        for(std::size_t i=0;i<s.pending.size();++i) {
            auto& move=s.pending[i];auto& incoming=*cache.sources_[move.incoming_source];auto& outgoing=*cache.sources_[move.outgoing_source];
            {
                const auto fields=moe_cache_field_refs(move.lease->weights);
                const auto& cohort=incoming.cohorts_[move.cohort];
                for(std::size_t f=0;f<fields.size();++f) {
                    const auto count=cohort.bytes_per_expert[f];if(!count)continue;
                    copies.push_back({static_cast<uint8_t*>(fields[f].data_ptr()),
                        s.backup.data_ptr<uint8_t>()+move.offsets[f],copy_bytes,copy_bytes+count});
                    copy_bytes+=count;
                }
            }
            changes.push_back({incoming.quant_source_->host_key(move.incoming),outgoing.quant_source_->host_key(move.outgoing),&move.lease,
                [&,i](const MixedMoePool& reused) {
                    const auto& item=s.pending[i];auto& old=*cache.sources_[item.outgoing_source];
                    const auto& cohort=old.cohorts_[old.expert_to_cohort_[item.outgoing]];auto fields=moe_cache_field_refs(reused);
                    s.observe("host",i);auto result=*cohort.cpu;result.local_experts=1;result.expert_local={};
                    return moe_replace_quant_fields(std::move(result),fields);
                },[&,i](const MixedMoePool& reused) {
                    const auto& item=s.pending[i];const auto& cohort=cache.sources_[item.incoming_source]->cohorts_[item.cohort];auto fields=moe_cache_field_refs(reused);
                    for(std::size_t f=0;f<fields.size();++f) {
                        const auto count=cohort.bytes_per_expert[f];
                        if(s.direct) {
                            const auto* gpu=static_cast<const uint8_t*>(cohort.arena->fields[f].data_ptr())+int64_t(item.slot)*count;
                            MFQ_CUDA_CHECK(cudaMemcpyAsync(fields[f].data_ptr(),gpu,std::size_t(count),cudaMemcpyDeviceToHost,s.stream));
                        } else std::memcpy(fields[f].data_ptr(),s.upload.data_ptr<uint8_t>()+item.offsets[f],std::size_t(count));
                    }
                    if(s.direct)MFQ_CUDA_CHECK(cudaStreamSynchronize(s.stream));
                }});
        }
        std::function<void()> prepare_host;
        if(s.parallel_host)prepare_host=[&] {
            mfq::cpu::expert_pool().rows(copy_bytes,[&](int64_t begin,int64_t end) {
                auto copy=std::upper_bound(copies.begin(),copies.end(),begin,
                    [](int64_t position,const HostCopy& item){return position<item.end;});
                while(begin<end) {
                    const auto count=std::min(end,copy->end)-begin;
                    std::memcpy(copy->destination+(begin-copy->begin),copy->source+(begin-copy->begin),std::size_t(count));
                    begin+=count;++copy;
                }
            });
            s.parallel_host_bytes+=copy_bytes;++s.parallel_host_batches;
        };
        if(!s.parallel_host)prepare_host=[&] {
            mfq::cpu::HostMemoryCopyBatch batch;
            for(const auto& copy:copies)
                batch.copy(copy.destination,copy.source,std::size_t(copy.end-copy.begin));
        };
        const bool committed=cache.host_experts_->exchange_batch(changes,[&] {
            s.observe("publish",0);
            try {
                for(auto& move:s.pending)if(!cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->commit_replace(move.replacement))
                    throw std::runtime_error("adaptive GPU replacement lost its generation");
            } catch(...) {
                for(auto& move:s.pending)cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->book->rollback_replace(move.replacement);
                throw;
            }
            for(auto& map:s.maps){auto& cohort=cache.sources_[map.source]->cohorts_[map.cohort];cohort.host_map.swap(map.next);cohort.map_dirty=false;}
            for(auto& move:s.pending) {
                cache.sources_[move.outgoing_source]->quant_source_->publish_gpu_resident(move.outgoing,false);
                cache.sources_[move.incoming_source]->quant_source_->publish_gpu_resident(move.incoming,true);
            }
        },prepare_host);
        if(!committed){rollback();return;}
        s.retire_promoted_kv();
        for(const auto& move:s.pending)if(move.prefill_ram_slot>=0)
            cache.sources_[move.incoming_source]->cohorts_[move.cohort].prefill_ram_experts[move.prefill_ram_slot]=move.outgoing;
        s.committed_bundles+=s.pending_complete_bundles;s.committed_projections+=s.pending.size();
        for(const auto& move:s.pending){++cache.stats_.evictions;cache.stats_.gpu_demote_bytes+=cache.sources_[move.incoming_source]->cohorts_[move.cohort].arena->slot_bytes;}
        s.pending.clear();s.maps.clear();
    } catch(...) {const auto failure=std::current_exception();rollback();std::rethrow_exception(failure);}
}
MoeResidencyManager::Stats MoeResidencyManager::stats() const noexcept {
    const auto& s=*impl_;return {s.observed_routes.load(),s.scheduled_rounds.load(),s.candidate_bundles.load(),
        s.planned_bundles.load(),s.committed_bundles.load(),s.memory_rejections.load(),
        s.host_pack_bytes.load(),s.direct_upload_bytes.load(),s.backup_peak_bytes.load(),s.shared_backup_bytes.load(),
        s.candidate_projections.load(),s.planned_projections.load(),s.committed_projections.load(),s.projection_memory_rejections.load(),
        s.prepare_ns.load(),s.join_ns.load(),s.dma_wait_ns.load(),s.map_ns.load(),s.host_exchange_ns.load(),s.apply_ns.load(),
        s.decode_prepare_ns.load(),s.decode_join_ns.load(),s.decode_dma_wait_ns.load(),s.decode_map_ns.load(),s.decode_host_exchange_ns.load(),s.decode_apply_ns.load(),
        s.backup_copies.load(),s.upload_copies.load(),s.parallel_host,s.parallel_host_bytes.load(),s.parallel_host_batches.load(),
        s.batched,s.batch_device_bytes.load(),s.batch_descriptor_copies.load(),
        s.async_publish,s.async_publish_ns.load(),s.decode_async_publish_ns.load(),s.window_prepares.load(),
        s.cached_copy_bytes.load(),s.cached_copy_fields.load(),s.expert_fence_waits.load(),s.cached_copy_batches.load(),
        s.expert_fence_wait_ns.load(),s.backup_submit_ns.load(),s.refill_submit_ns.load(),
        s.cached_copy_dma_bytes.load(),s.cached_copy_dma_fields.load(),
        s.direct_exchange_rounds.load(),s.direct_exchange_bytes.load(),s.direct_exchange_ns.load(),
        s.cached_copy_descriptors.load(),s.cached_copy_blocks.load(),s.exchange_defaults,s.kv_refill_bytes.load(),s.native_dma_batches.load(),s.native_dma_copies.load(),s.native_dma_fallbacks.load(),s.staging_budget_skips.load(),s.plan_ns.load(),s.decode_plan_ns.load(),
        s.speculative_route_windows.load(),s.speculative_routes_committed.load(),s.speculative_routes_discarded.load(),s.speculative_route_cancels.load(),
        s.shadow_backup_bytes.load(),s.shadow_backup_fields.load(),s.shadow_index_rebuilds.load(),
        s.shadow_index_matches.load(),s.shadow_index_backup_bytes.load(),s.speculative_heat_routes.load(),
        s.predictor_overlap_rounds.load(),s.predictor_overlap_serves.load(),
        s.ssd_load_requests.load(),s.ssd_load_bytes.load(),s.ssd_gpu_image_bytes.load(),s.ssd_gpu_images.load(),s.ssd_gpu_image_ns.load(),s.ssd_gpu_image_copies.load()};
}
void MoeResidencyManager::trace_rank_summary(std::ostream& out) const {
    const auto& s=*impl_;
    const auto* enabled=std::getenv("MFQ_TRACE_MOE_RANK_SUMMARY");
    const auto* ssd_profile=std::getenv("MFQ_PROFILE_SSD_EXPERTS");
    if(enabled?enabled[0]!='1':!(s.cache->ssd_experts() && ssd_profile && ssd_profile[0]=='1'))return;
    // Called only after exchanges finish. Estimate the placement restriction
    // from existing decayed observations, without instrumenting GPU work.
    struct Group {std::vector<double> heat;std::size_t slots=0;double bytes=0;};
    std::map<std::pair<MoeGpuArena*,int>,Group> local;
    std::map<MoeGpuArena*,Group> global;
    double total=0,resident=0,total_bytes=0,resident_bytes=0;
    for(std::size_t b=0;b<s.cache->pipeline_bundles_.size();++b) {
        const auto& bundle=s.cache->pipeline_bundles_[b];
        if(bundle[0]==bundle[1])continue;
        for(int p=0;p<3;++p) {
            const auto& source=*s.cache->sources_[bundle[p]];
            for(int e=0;e<source.n_experts();++e) {
                auto* arena=source.cohorts_[source.expert_to_cohort_[e]].arena;
                const double heat=s.heat[b][e]+s.observations[b][e];
                const bool hot=source.quant_source_->gpu_resident(e);
                const auto bytes=double(source.bytes_for_expert(e));
                total+=heat;total_bytes+=heat*bytes;
                if(hot){resident+=heat;resident_bytes+=heat*bytes;}
                for(auto* group:{&local[{arena,source.id_}],&global[arena]}) {
                    group->heat.push_back(heat);group->slots+=hot;group->bytes=bytes;
                }
            }
        }
    }
    const auto optimum=[](auto& groups) {
        std::pair<double,double> sum{};
        for(auto& [key,g]:groups) {
            std::partial_sort(g.heat.begin(),g.heat.begin()+g.slots,g.heat.end(),std::greater<double>{});
            for(std::size_t i=0;i<g.slots;++i){sum.first+=g.heat[i];sum.second+=g.heat[i]*g.bytes;}
        }
        return sum;
    };
    const auto fixed=optimum(local),free=optimum(global);
    out<<"moe_rank_summary heat="<<total<<" current_hot_heat="<<resident
        <<" layer_optimal_hot_heat="<<fixed.first<<" arena_optimal_hot_heat="<<free.first
        <<" weighted_bytes="<<total_bytes<<" current_hot_weighted_bytes="<<resident_bytes
        <<" layer_optimal_hot_weighted_bytes="<<fixed.second
        <<" arena_optimal_hot_weighted_bytes="<<free.second<<'\n';
}
