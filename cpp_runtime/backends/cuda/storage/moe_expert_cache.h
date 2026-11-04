#pragma once

#include "moe_cache_profile.h"

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include <utility>

class MoeExpertCache;
struct MoePrefillWorkspaceBudget {
    std::size_t lendable_bytes = 0;
    std::size_t transfer_bytes = 0;
};
MoePrefillWorkspaceBudget moe_prefill_workspace_budget(
    const std::shared_ptr<MoeExpertCache>& cache, bool allow_borrow);
// Bound each target/predictor transfer arena across single-token and batched decode.
inline int moe_decode_transfer_slots(int rows, int routes) {
    return std::max(80, rows * routes);
}
inline constexpr std::size_t moe_ssd_residency_stage_limit_bytes=std::size_t(256)<<20;
inline bool moe_predictor_transfer_cache_enabled() {
    const auto* option=std::getenv("MFQ_MOE_PREDICTOR_TRANSFER_CACHE");
    return !option || option[0]!='0';
}
inline std::string moe_packed_transfer_prefix(int projection,bool predictor) {
    return std::string("decodepacked:")+(predictor && moe_predictor_transfer_cache_enabled()?"mtp:":"")+
        std::to_string(projection)+":";
}
namespace mfq::cuda {class KvExpertLoan;}
void set_moe_kv_expert_loan(const std::shared_ptr<MoeExpertCache>& cache,std::shared_ptr<mfq::cuda::KvExpertLoan> loan);

// The owner outlives every temporary tensor allocated from the borrowed tails.
// Destruction restores experts on cancellation/exception as well as success.
class MoePrefillWorkspaceLoan {
public:
    explicit MoePrefillWorkspaceLoan(std::shared_ptr<MoeExpertCache> cache):cache_(std::move(cache)){}
    ~MoePrefillWorkspaceLoan();
    std::size_t begin(std::size_t bytes);
    void suspend();
    void finish();
private:
    std::shared_ptr<MoeExpertCache> cache_;
    bool active_=false;
};
// A phase can prefer experts from [first,last). Scope exit restores normal
// eviction on success, cancellation and exceptions, without moving weights.
class MoePrefillLayerPriority {
public:
    MoePrefillLayerPriority(std::shared_ptr<MoeExpertCache> cache,int first,int last);
    ~MoePrefillLayerPriority();
    void finish();
    MoePrefillLayerPriority(const MoePrefillLayerPriority&)=delete;
    MoePrefillLayerPriority& operator=(const MoePrefillLayerPriority&)=delete;
private:
    std::shared_ptr<MoeExpertCache> cache_;
    std::vector<int> previous_;
};
std::size_t moe_prefill_transfer_bytes(const std::shared_ptr<MoeExpertCache>& cache,bool predictor=false);
class MoeQuantRangeSource;
struct CudaExecutionConfig;

// One cache per Engine load, shared by its target and optional predictor weights.
// Registered weights retain the cache; resetting the Engine's execution resources
// drops only its ownership. Cache APIs receive the resource, never the execution.
std::shared_ptr<MoeExpertCache> make_moe_expert_cache(
    std::int64_t bytes,
    const CudaExecutionConfig& config);
bool moe_expert_cache_has_sources(
    const std::shared_ptr<MoeExpertCache>& cache);
bool moe_expert_cache_finalized(
    const std::shared_ptr<MoeExpertCache>& cache);
bool moe_expert_cache_layer_prefill_ready(const std::shared_ptr<MoeExpertCache>& cache);
bool moe_expert_cache_ssd_experts(const std::shared_ptr<MoeExpertCache>& cache);
bool moe_expert_cache_ssd_router_prefetch(const std::shared_ptr<MoeExpertCache>& cache);
std::vector<std::pair<std::string, double>> moe_expert_memory_metrics(
    const std::shared_ptr<MoeExpertCache>& cache);
void finalize_moe_expert_cache(
    const std::shared_ptr<MoeExpertCache>& cache, std::size_t kv_growth_reserve = 0, std::size_t kv_budget_charge = 0);
std::size_t moe_expert_cache_minimum_bytes(const std::shared_ptr<MoeExpertCache>& cache);
std::size_t moe_expert_cache_budget_bytes(const std::shared_ptr<MoeExpertCache>& cache);
void resize_moe_expert_cache(const std::shared_ptr<MoeExpertCache>& cache, std::size_t bytes);
// Request boundary only: callers retire graphs before moving primary expert arenas.
void reclaim_moe_expert_workspace(const std::shared_ptr<MoeExpertCache>& cache, std::size_t bytes);
// Reclaim lower-priority cache storage before reducing prefill transfer overlap.
void set_moe_workspace_reservation(const std::shared_ptr<MoeExpertCache>& cache,
    std::function<void(std::size_t)> reserve,std::function<void(std::size_t)> reserve_ram={});
void reserve_moe_workspace(const std::shared_ptr<MoeExpertCache>& cache,std::size_t bytes);
std::size_t reclaim_moe_optional_host(const std::shared_ptr<MoeExpertCache>& cache,
    std::size_t allocation_bytes=0);
double moe_expert_cache_ram_pcie_fraction(const std::shared_ptr<MoeExpertCache>& cache);
std::uint64_t moe_expert_streamed_bytes(const std::shared_ptr<MoeExpertCache>& cache);
// Successful private FFN captures change this counter for every model family.
std::uint64_t moe_expert_graph_captures(const std::shared_ptr<MoeExpertCache>& cache);
// Finish the adaptive transaction before a session reset or a prefill loan.
void finish_moe_expert_exchanges(const std::shared_ptr<MoeExpertCache>& cache);
void prepare_moe_predictor_exchange(const std::shared_ptr<MoeExpertCache>& cache);
void begin_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>& cache,int rows);
void commit_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>& cache,int rows);
void discard_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>& cache) noexcept;
void prepare_moe_decode_residency(const std::shared_ptr<MoeExpertCache>& cache);
// Target, predictor and request batches share the MFE pool on their compute stream.
bool prepare_moe_graph_pool(const std::shared_ptr<MoeExpertCache>& cache);
// Request boundary only, after target/predictor graphs have been destroyed.
// Lend decode staging/workspace capacity to the upcoming long prefill.
void release_moe_decode_buffers(const std::shared_ptr<MoeExpertCache>& cache);
// Private graphs with indirect expert addresses can return cache pages while
// retaining their captured kernels. False leaves full retirement to the caller.
bool lend_moe_decode_cache_for_prefill(const std::shared_ptr<MoeExpertCache>& cache,
    std::size_t workspace_bytes);
bool suspend_moe_residency(const std::shared_ptr<MoeExpertCache>& cache,bool suspended);
bool set_moe_reuse_learning(const std::shared_ptr<MoeExpertCache>& cache,bool enabled);
// Return prompt-only ring and workspace storage before preparing decode graphs.
void release_moe_prefill_buffers(MoeExpertCache* cache);
void release_moe_prefill_buffers(const std::shared_ptr<MoeExpertCache>& cache);
// Idle diagnostic passes retain primary weights, clear transfer-cache entries,
// and replay the CPU assignments captured by the first pass.
void prepare_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>& cache, bool replay);
// The CPU policy pass uses natural routing and measured dispatch. Baseline
// passes retain their exact CPU assignments; report any policy route changes.
void prepare_moe_cpu_budget_comparison(const std::shared_ptr<MoeExpertCache>& cache,
    bool replay,bool transfer_budget);
// Compare synchronous/background calibration with fresh cost samples each pass.
void prepare_moe_cpu_calibration_comparison(const std::shared_ptr<MoeExpertCache>& cache,bool replay);
std::size_t finish_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>& cache);
void print_moe_expert_cache_stats(
    const std::shared_ptr<MoeExpertCache>& cache,
    std::ostream& output);
void set_moe_expert_cache_profile(
    MoeExpertCache& cache,
    mfq::MoeCacheProfile profile);

struct MfeWeight;
struct MixedMoeRuntime;
namespace mfq::cuda { class MfeMxfp4ExpertStore; }

MfeWeight cache_moe_weight(
    const std::shared_ptr<MoeExpertCache>& cache,
    const std::string& name,
    const std::shared_ptr<MixedMoeRuntime>& runtime,
    int minimum_slots,
    int layer_id,
    const std::string& projection_role,
    std::shared_ptr<mfq::cuda::MfeMxfp4ExpertStore> range_store = {});

MfeWeight cache_quant_moe_weight(const std::shared_ptr<MoeExpertCache>& cache,
    const std::string& name,std::shared_ptr<MoeQuantRangeSource> source,
    int minimum_slots,int layer_id,const std::string& projection_role);
