#pragma once

#include "mlx_moe_ops.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

class MfqContainer;
class MlxMfeWeight;
struct MlxMfeCachedProjection;
class MlxNintSwiGluPair;
class MlxNintWeight;

struct MlxMfeKernelProfile {
    std::uint32_t family_mask = 0;
    std::uint32_t vq_profile_mask = 0;
    std::uint64_t vq_layouts = 0;
    std::shared_ptr<const std::vector<int>> nint_group_sizes;
    bool vq_execution_layout = false;
    bool nint_q_dispatch = false;
};

struct MlxMfeProjectionInfo {
    int experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<std::int32_t> available_experts;
};

struct MlxMfeCacheReuseStats {
    std::size_t expert_loads = 0;
    std::size_t expert_hits = 0;
    std::size_t packed_hits = 0;
    std::size_t load_waits = 0;
    std::size_t bytes_read = 0;
    std::size_t read_calls = 0;
    std::size_t evictions = 0;
    double io_seconds = 0;
    double wait_seconds = 0;
    double prepare_seconds = 0;
    double view_seconds = 0;
    std::size_t cached_route_hits = 0;
    std::size_t cached_route_misses = 0;
    std::size_t overlapped_read_batches = 0;
    std::size_t overlapped_read_submissions = 0;
    std::size_t parallel_body_read_batches = 0;
};

// Optional bounded per-expert residency for MFE records.
//
// This is an offload policy, not the MFE container itself. Full-resident
// MlxMfeWeight is the default model path. The cache is constructed only
// after an explicit disk-offload request. Apple Silicon has unified memory,
// so there is no separate CUDA-style CPU-RAM versus device-RAM mode. Disk
// backing uses read_range(); the LRU limit accounts for packed expert bytes
// staged in unified memory for active dispatches.
class MlxMfeOffloadCache {
public:
    MlxMfeOffloadCache(
        const MfqContainer& model,
        std::size_t cache_limit_bytes,
        int experts = 0);
    ~MlxMfeOffloadCache();

    MlxMfeOffloadCache(
        const MlxMfeOffloadCache&) = delete;
    MlxMfeOffloadCache& operator=(
        const MlxMfeOffloadCache&) = delete;

    // Returns false only for a valid non-streamable representation (for
    // example legacy NINTv1, dense cohorts, or unsupported VQ layouts).
    // Malformed streamable records still raise.
    bool can_offload(const std::string& name);
    bool can_stream(const std::string& name) {
        return can_offload(name);
    }
    bool can_group_mfe(const std::string& name);
    void activate_record(const std::string& name);
    void prepare_projection(
        const std::string& name, bool shared_input, bool fused_swiglu = false);
    void prepare_swiglu_pair(
        const std::string& gate, const std::string& up, int routes);
    std::size_t backing_payload_bytes() const;

    MlxMfeProjectionInfo projection_info(
        const std::string& name);
    std::vector<std::uint8_t> availability(
        const std::string& name);

    // Materialize only the requested experts from a native MFE record and
    // return one ordinary heterogeneous execution weight whose local expert
    // order matches active_experts. NINTv2, NVQ1-S/L, NVQ-JSC, MXFP4, and aligned
    // MXFP8 reuse their existing packed kernels after paging.
    MlxMfeWeight grouped_mfe(
        const std::string& name,
        const std::vector<std::int32_t>& active_experts,
        std::span<const std::uint32_t> activation_counts = {},
        bool prefill_scan = false);

    std::vector<MlxMfeWeight> grouped_mfe(
        std::span<const std::string> names,
        const std::vector<std::int32_t>& active_experts,
        std::span<const std::uint32_t> activation_counts = {},
        bool prefill_scan = false,
        const std::function<void(const std::vector<MlxMfeWeight>&)>& prefix_ready = {});

    std::optional<std::vector<MlxMfeWeight>> layer_mfe(
        std::span<const std::string> names,
        std::span<const std::int32_t> active_experts = {});

    std::shared_ptr<const MlxMfeCachedProjection> cached_projection(const std::string& name);
    bool resolve_cached_projections(std::span<const std::shared_ptr<const MlxMfeCachedProjection>> views,
        std::span<const std::int32_t> experts);

    std::size_t cache_limit_bytes() const noexcept;
    std::size_t resident_packed_bytes() const;
    std::size_t cached_expert_count() const;
    MlxMfeCacheReuseStats reuse_stats() const;
    std::size_t set_cache_limit(std::size_t bytes);
    void set_prefill_priority(std::span<const std::string> names);

    // Atomically forget one parsed projection and all of its resident
    // experts.  Already returned routed weights retain their own Metal
    // arrays and remain executable.
    void discard_record(
        const std::string& name) noexcept;
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// A sorted routed-MoE row block list.  The plan is built once on the GPU and
// shared by gate/up and down projections so every populated row block can be
// scheduled as an independent Metal threadgroup.
struct MlxGroupedMmqPlan {
    mlx::core::array block_meta;
    mlx::core::array block_count;
    int max_blocks = 0;
    int block_rows = 0;
    int route_count = 0;
    int experts = 0;
};

// Temporary source-compatibility alias; new code uses the family-neutral
// name because the same plan serves NINT, VQ, MX, and dense expert cohorts.
using MlxGroupedVqMmqPlan = MlxGroupedMmqPlan;

// Native packed MFE routed-expert weight.
//
// NINT, VQ-family, MXFP4/MXFP8, and BF16/F16 cohorts use the common
// heterogeneous Metal dispatch. Canonical NINT q+k row metadata is consumed
// directly by both the decode and grouped-prefill kernels; it is not split
// into per-q execution pools. Native-QAT SQ cohorts remain composed at the
// routing layer until their own heterogeneous grouped kernels are available.
// Expert IDs retain the global ordering from the MFE container while each
// descriptor or standalone cohort map points at its local rows.
class MlxMfeWeight {
public:
    static MlxMfeWeight from_blob(
        std::span<const std::uint8_t> blob);
    static MlxMfeWeight from_blob(const mlx::core::array& blob);
    static MlxMfeWeight from_nint_blob(
        std::span<const std::uint8_t> blob);
    static MlxMfeWeight from_nint_blob(const mlx::core::array& blob);

    // Load compatible packed projections into one final arena, without
    // separate Gate/Up arenas or model-sized staging vectors. Standalone
    // SQ, reference and rotated cohorts retain their established dispatch.
    static MlxMfeWeight from_projection_blobs(
        std::span<const std::span<const std::uint8_t>> blobs);

    // Gate/up and other shape-compatible projections share one packed routed
    // call when their formats participate in the common dispatch. The
    // returned last dimension is
    // projections() * out_per_expert(), in projection-major order.
    static MlxMfeWeight concatenate_projections(
        const std::vector<MlxMfeWeight>& weights);

    // Finalize native two-projection storage at load time and release the
    // separate source pools. Unsupported split/rotated layouts are unchanged.
    MlxMfeWeight materialize_packed_projections() const;

    // Assemble independently resident single-expert pages into one routed
    // dispatch.  This changes only the packed storage view: every source must
    // use the ordinary heterogeneous MFE execution streams, and the resulting
    // expert IDs are the source-vector indices.  No format-specific kernel is
    // introduced for SSD residency.
    static MlxMfeWeight concatenate_experts(
        const std::vector<MlxMfeWeight>& weights);

    MlxMfeWeight select_experts(
        std::span<const std::int32_t> experts) const;

    // Return a shared-storage view with automatic large-M MLX gather-QMM
    // selection enabled or disabled. Explicit environment forcing remains
    // available for operator experiments.
    MlxMfeWeight with_automatic_mxfp4_nax_prefill(bool enabled) const;

    // Build a native MXFP4 routed view over a shared slot arena without
    // copying packed values or scales. slot_for_expert maps each global
    // expert ID to one arena row.
    static MlxMfeWeight from_mxfp4_slots(
        int experts,
        int out_per_expert,
        int neuron_len,
        const std::vector<std::int32_t>& slot_for_expert,
        mlx::core::array packed_values,
        mlx::core::array block_scales,
        int logical_experts = 0);

    mlx::core::array routed_matmul(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array* route_groups = nullptr) const;
    mlx::core::array routed_matmul_mapped(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& expert_map) const;
    mlx::core::array routed_matmul_packed(
        const mlx::core::array& input,
        const mlx::core::array& packed_expert_ids) const;
    mlx::core::array routed_swiglu(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        float limit = 0.0f,
        const mlx::core::array* route_groups = nullptr) const;
    mlx::core::array routed_swiglu_mapped(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& expert_map,
        float limit = 0.0f) const;
    mlx::core::array routed_swiglu_packed(
        const mlx::core::array& input,
        const mlx::core::array& packed_expert_ids,
        float limit = 0.0f) const;
    mlx::core::array routed_swiglu_pair(
        const MlxMfeWeight& up,
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        float limit = 0.0f,
        const mlx::core::array* route_groups = nullptr,
        const mlx::core::array* gate_map = nullptr,
        const mlx::core::array* up_map = nullptr) const;
    // One through six rows of heterogeneous MoE in two Metal launches after
    // routing. The first runs routed and shared Gate/Up; the second runs
    // routed and shared Down in parallel and combines them in-threadgroup.
    // Unsupported geometry returns nullopt.
    std::optional<mlx::core::array> decode_nint_shared(
        const MlxMfeWeight& down,
        const MlxNintSwiGluPair& shared_gate_up,
        const MlxNintWeight& shared_down,
        const mlx::core::array& shared_gate_weight,
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& route_weights,
        bool shared_gate_is_logits = false,
        const mlx::core::array* route_groups = nullptr,
        const mlx::core::array* softmax_router_logits = nullptr) const;
    // Decode/small-M fast path. For one through six tokens, project
    // every selected expert and apply its routing weight in one Metal
    // dispatch, avoiding the transient [M,routes,hidden] down-projection
    // tensor. Unsupported shapes and representations transparently use the
    // ordinary projection followed by moe_weighted_reduce().
    mlx::core::array routed_matmul_reduce(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& route_weights,
        const mlx::core::array* route_groups = nullptr) const;
    // Whether this representation has a fused projection/reduction kernel.
    // Callers may otherwise fuse reduction with their following epilogue.
    bool supports_fused_routed_reduce(const mlx::core::array* route_groups = nullptr) const noexcept;
    mlx::core::array routed_matmul_reduce_packed(
        const mlx::core::array& input,
        const mlx::core::array& packed_expert_ids,
        const mlx::core::array& route_weights) const;
    bool supports_grouped_mmq() const noexcept;
    bool supports_grouped_vq_mmq() const noexcept {
        return supports_grouped_mmq();
    }
    bool prefers_mxfp4_smallm_nax(
        const mlx::core::array& expert_ids) const noexcept;
    bool prefers_mxfp4_nax_prefill(int route_count) const noexcept;
    bool supports_mxfp4_blocks() const noexcept;
    bool supports_mxfp4_pair_blocks(
        const MlxMfeWeight& other) const noexcept;
    mlx::core::array mxfp4_block_matmul_sorted(
        const mlx::core::array& sorted_input,
        const MlxGroupedMmqPlan& plan) const;
    mlx::core::array mxfp4_pair_swiglu_sorted(
        const MlxMfeWeight& other,
        const mlx::core::array& sorted_input,
        const MlxGroupedMmqPlan& plan,
        float limit = 0.0f) const;
    // Largest aligned token chunk within the native sorted-MXFP4 prefill row
    // limit, or zero when this representation/device cannot use that path.
    // The decision is derived from operator geometry rather than model ID.
    int recommended_mxfp4_nax_prefill_tokens(
        int routes_per_token) const noexcept;
    int recommended_grouped_mmq_block_rows(
        int route_count,
        bool fused_swiglu = false) const noexcept;
    MlxGroupedMmqPlan build_grouped_mmq_plan(
        const mlx::core::array& expert_ids,
        const mlx::core::array& route_order,
        int block_rows = 32) const;
    MlxGroupedMmqPlan build_grouped_vq_mmq_plan(
        const mlx::core::array& expert_ids,
        const mlx::core::array& route_order) const {
        return build_grouped_mmq_plan(expert_ids, route_order);
    }
    mlx::core::array routed_matmul_sorted(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& route_order,
        bool input_is_sorted,
        bool fused_swiglu = false,
        float swiglu_limit = 0.0f,
        const MlxGroupedMmqPlan* plan = nullptr,
        bool force_mxfp4_nax = false) const;
    mlx::core::array operator()(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids) const {
        return routed_matmul(input, expert_ids);
    }

    int experts() const noexcept;
    int out_per_expert() const noexcept;
    int neuron_len() const noexcept;
    int projections() const noexcept;
    std::size_t packed_nbytes() const noexcept;
    static std::size_t packed_copy_count() noexcept;
    static std::size_t indirect_dispatch_count() noexcept;
    static std::size_t uncached_packed_bytes() noexcept;
    static std::size_t pinned_packed_bytes() noexcept;
    static std::size_t packed_pin_failures() noexcept;
    static std::size_t pooled_staging_count() noexcept;

private:
    struct Impl;

    friend class MlxMfeOffloadCache;
    static MlxMfeWeight concatenate_expert_pages(const std::vector<MlxMfeWeight>& weights,
        std::shared_ptr<const MlxMfeKernelProfile> profile = nullptr);
    std::shared_ptr<void> pin_packed_storage() const noexcept;
    MlxMfeWeight with_kernel_profile(
        std::shared_ptr<const MlxMfeKernelProfile> profile) const;
    MlxMfeWeight with_prepared_page_resources() const;

    static MlxMfeWeight from_projection_blobs_impl(
        std::span<const std::span<const std::uint8_t>> blobs,
        bool nint_tensor,
        const mlx::core::array* storage = nullptr,
        const mlx::core::array* destination = nullptr);

    explicit MlxMfeWeight(std::shared_ptr<const Impl> impl);

    mlx::core::array routed_matmul_impl(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        bool fused_swiglu,
        float swiglu_limit,
        const mlx::core::array* expert_map = nullptr,
        bool packed_expert_ids = false,
        const mlx::core::array* route_groups = nullptr,
        const mlx::core::array* reduce_weights = nullptr) const;
    mlx::core::array routed_bf16_reference(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        bool fused_swiglu,
        float swiglu_limit) const;
    mlx::core::array mxfp4_blocks_sorted_impl(
        const MlxMfeWeight* other,
        const mlx::core::array& sorted_input,
        const MlxGroupedMmqPlan& plan) const;

    std::shared_ptr<const Impl> impl_;
};

using MlxMoeWeight = MlxMfeWeight;

struct MlxMfeCachedProjection {
    MlxMfeWeight weight;
    mlx::core::array expert_map;
    std::vector<std::int32_t> host_expert_map;
    std::string name;
    std::size_t metadata_bytes = 0;
};

struct MlxMfeCachedRoute {
    std::shared_ptr<MlxMfeOffloadCache> cache;
    std::vector<std::shared_ptr<const MlxMfeCachedProjection>> projections;
    std::optional<mlx::core::array> ids;

    mlx::core::array forward(const mlx::core::array& source,
        const mlx::core::array& routes, float swiglu_limit = 0.0f,
        const mlx::core::array* reduce_weights = nullptr);
    std::span<const std::int32_t> host_ids();
    bool all_resident();
    bool resolve();
};

// Resolve canonical split Gate/Up projections into one packed dispatch.
// Legacy fused ``gate_up.weight`` remains a read-only compatibility input.
MlxMoeWeight load_routed_gate_up_weight(
    const MfqContainer& model,
    const std::string& mlp_prefix);

// One explicit-[tokens,routes] routed projection.
class MlxRoutedLinear {
public:
    explicit MlxRoutedLinear(MlxMoeWeight weight);

    static MlxRoutedLinear from_blob(
        std::span<const std::uint8_t> blob);

    mlx::core::array forward(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids) const;
    mlx::core::array forward_mapped(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& expert_map) const;
    mlx::core::array forward_packed(
        const mlx::core::array& input,
        const mlx::core::array& packed_expert_ids) const;
    mlx::core::array swiglu(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        float limit = 0.0f) const;
    mlx::core::array swiglu_mapped(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& expert_map,
        float limit = 0.0f) const;
    mlx::core::array swiglu_packed(
        const mlx::core::array& input,
        const mlx::core::array& packed_expert_ids,
        float limit = 0.0f) const;
    mlx::core::array swiglu_pair(
        const MlxRoutedLinear& up,
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        float limit = 0.0f) const;
    mlx::core::array combine(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& route_weights) const;
    mlx::core::array combine_mapped(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& expert_map,
        const mlx::core::array& route_weights) const;
    mlx::core::array combine_packed(
        const mlx::core::array& input,
        const mlx::core::array& packed_expert_ids,
        const mlx::core::array& route_weights) const;
    bool supports_grouped_mmq() const noexcept {
        return weight_.supports_grouped_mmq();
    }
    bool supports_grouped_vq_mmq() const noexcept {
        return supports_grouped_mmq();
    }
    bool prefers_mxfp4_smallm_nax(
        const mlx::core::array& expert_ids) const noexcept {
        return weight_.prefers_mxfp4_smallm_nax(expert_ids);
    }
    bool prefers_mxfp4_nax_prefill(int route_count) const noexcept {
        return weight_.prefers_mxfp4_nax_prefill(route_count);
    }
    bool supports_mxfp4_blocks() const noexcept {
        return weight_.supports_mxfp4_blocks();
    }
    bool supports_mxfp4_pair_blocks(
        const MlxRoutedLinear& other) const noexcept {
        return weight_.supports_mxfp4_pair_blocks(other.weight_);
    }
    mlx::core::array mxfp4_block_matmul_sorted(
        const mlx::core::array& sorted_input,
        const MlxGroupedMmqPlan& plan) const {
        return weight_.mxfp4_block_matmul_sorted(sorted_input, plan);
    }
    mlx::core::array mxfp4_pair_swiglu_sorted(
        const MlxRoutedLinear& other,
        const mlx::core::array& sorted_input,
        const MlxGroupedMmqPlan& plan,
        float limit = 0.0f) const {
        return weight_.mxfp4_pair_swiglu_sorted(
            other.weight_, sorted_input, plan, limit);
    }
    int recommended_mxfp4_nax_prefill_tokens(
        int routes_per_token) const noexcept {
        return weight_.recommended_mxfp4_nax_prefill_tokens(
            routes_per_token);
    }
    int recommended_grouped_mmq_block_rows(
        int route_count,
        bool fused_swiglu = false) const noexcept {
        return weight_.recommended_grouped_mmq_block_rows(
            route_count,
            fused_swiglu);
    }
    MlxGroupedMmqPlan build_grouped_mmq_plan(
        const mlx::core::array& expert_ids,
        const mlx::core::array& route_order,
        int block_rows = 32) const {
        return weight_.build_grouped_mmq_plan(
            expert_ids,
            route_order,
            block_rows);
    }
    MlxGroupedMmqPlan build_grouped_vq_mmq_plan(
        const mlx::core::array& expert_ids,
        const mlx::core::array& route_order) const {
        return build_grouped_mmq_plan(expert_ids, route_order);
    }
    mlx::core::array forward_sorted(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& route_order,
        bool input_is_sorted,
        const MlxGroupedMmqPlan* plan = nullptr,
        bool force_mxfp4_nax = false) const {
        return weight_.routed_matmul_sorted(
            input,
            expert_ids,
            route_order,
            input_is_sorted,
            false,
            0.0f,
            plan,
            force_mxfp4_nax);
    }
    mlx::core::array swiglu_sorted(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& route_order,
        float limit = 0.0f,
        bool force_mxfp4_nax = false) const {
        return weight_.routed_matmul_sorted(
            input,
            expert_ids,
            route_order,
            false,
            true,
            limit,
            nullptr,
            force_mxfp4_nax);
    }
    mlx::core::array operator()(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids) const {
        return forward(input, expert_ids);
    }

    const MlxMoeWeight& weight() const noexcept {
        return weight_;
    }

private:
    MlxMoeWeight weight_;
};

struct MlxRoutedFfnResult {
    mlx::core::array output;
    mlx::core::array ids;
    mlx::core::array weights;
};

// Fused routed SwiGLU flow:
//   one gate/up packed dispatch -> SwiGLU -> one down packed dispatch
//   -> route-weighted reduction.
class MlxRoutedSwiGluFfn {
public:
    MlxRoutedSwiGluFfn(
        MlxMoeWeight gate,
        MlxMoeWeight up,
        MlxMoeWeight down);

    static MlxRoutedSwiGluFfn from_blobs(
        std::span<const std::uint8_t> gate,
        std::span<const std::uint8_t> up,
        std::span<const std::uint8_t> down);

    mlx::core::array forward(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& route_weights) const;

    MlxRoutedFfnResult forward_from_logits(
        const mlx::core::array& input,
        const mlx::core::array& router_logits,
        int top_k,
        bool use_sigmoid = false,
        bool use_sqrt_softplus = false,
        bool normalize = false,
        bool delayed_softmax = false,
        const std::optional<mlx::core::array>& bias = std::nullopt,
        const std::optional<mlx::core::array>& available = std::nullopt,
        float norm_floor = 1e-20f,
        float scale = 1.0f) const;

    mlx::core::array operator()(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& route_weights) const {
        return forward(input, expert_ids, route_weights);
    }

    const MlxMoeWeight& gate_up_weight() const noexcept {
        return gate_up_;
    }
    const MlxMoeWeight& down_weight() const noexcept {
        return down_;
    }

private:
    MlxMoeWeight gate_up_;
    MlxMoeWeight down_;
};

} // namespace mfq::metal
