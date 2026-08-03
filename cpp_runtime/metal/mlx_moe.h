#pragma once

#include "mlx_moe_ops.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

class MfqContainer;

struct MlxNintMoeProjectionInfo {
    int experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::size_t shared_codebook_nbytes = 0;
    std::vector<std::int32_t> available_experts;
};

// One transient single-dispatch view over the currently active CCCP experts.
// Expert IDs remain global.  Packed indices are assembled only for the active
// set, while every cohort codebook is retained once by the residency object
// and shared by all expert views from that projection.
class MlxCccpRoutedWeight {
public:
    mlx::core::array routed_matmul(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids) const;
    int experts() const noexcept;
    int out_per_expert() const noexcept;
    int neuron_len() const noexcept;
    std::size_t packed_nbytes() const noexcept;
    std::size_t shared_codebook_nbytes() const noexcept;

private:
    struct Impl;

    explicit MlxCccpRoutedWeight(
        std::shared_ptr<const Impl> impl);

    std::shared_ptr<const Impl> impl_;

    friend class MlxNintMoeOffloadCache;
};

// Optional bounded per-expert residency for NINTM records.
//
// This is an offload policy, not the NINTM container itself. Full-resident
// MlxNintMoeWeight is the default model path. The cache is constructed only
// after an explicit disk-offload request. Apple Silicon has unified memory,
// so there is no separate CUDA-style CPU-RAM versus device-RAM mode. Disk
// backing uses read_range(); the LRU limit accounts for packed expert bytes
// staged in unified memory for active dispatches.
class MlxNintMoeOffloadCache {
public:
    MlxNintMoeOffloadCache(
        const MfqContainer& model,
        std::size_t cache_limit_bytes,
        int experts);
    ~MlxNintMoeOffloadCache();

    MlxNintMoeOffloadCache(
        const MlxNintMoeOffloadCache&) = delete;
    MlxNintMoeOffloadCache& operator=(
        const MlxNintMoeOffloadCache&) = delete;

    // Returns false only for a valid non-streamable representation (for
    // example NIM1 or a mixed non-CCCP NIM2 record).  Malformed CCCP records
    // still raise.
    bool can_offload(const std::string& name);
    bool can_stream(const std::string& name) {
        return can_offload(name);
    }

    MlxNintMoeProjectionInfo projection_info(
        const std::string& name);
    std::vector<std::uint8_t> availability(
        const std::string& name);

    MlxCccpRoutedWeight grouped(
        const std::string& name,
        const std::vector<std::int32_t>& active_experts);

    std::size_t cache_limit_bytes() const noexcept;
    std::size_t resident_packed_bytes() const;
    std::size_t cached_expert_count() const;

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

// Compatibility names for callers that used the original CCCP-specific API.
// New model/runtime code must use the generic NINTM names above.
using MlxCccpProjectionInfo = MlxNintMoeProjectionInfo;
using MlxCccpExpertResidency = MlxNintMoeOffloadCache;

// Native packed NINTM routed-expert weight.
//
// NINT1-NINT8 and NINT8-0 cohorts are decoded directly by one heterogeneous
// Metal dispatch.  Expert IDs retain the global ordering from the NINTM
// container while each descriptor points at its cohort-local packed rows.
class MlxNintMoeWeight {
public:
    static MlxNintMoeWeight from_blob(
        std::span<const std::uint8_t> blob);

    // Gate/up and other shape-compatible projections can share one dispatch.
    // The returned last dimension is
    // projections() * out_per_expert(), in projection-major order.
    static MlxNintMoeWeight concatenate_projections(
        const std::vector<MlxNintMoeWeight>& weights);

    mlx::core::array routed_matmul(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids) const;
    mlx::core::array routed_swiglu(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        float limit = 0.0f) const;
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

private:
    struct Impl;

    explicit MlxNintMoeWeight(std::shared_ptr<const Impl> impl);

    mlx::core::array routed_matmul_impl(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        bool fused_swiglu,
        float swiglu_limit) const;

    std::shared_ptr<const Impl> impl_;
};

using MlxMoeWeight = MlxNintMoeWeight;

// One explicit-[tokens,routes] routed projection.
class MlxRoutedLinear {
public:
    explicit MlxRoutedLinear(MlxMoeWeight weight);

    static MlxRoutedLinear from_blob(
        std::span<const std::uint8_t> blob);

    mlx::core::array forward(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids) const;
    mlx::core::array swiglu(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        float limit = 0.0f) const;
    mlx::core::array combine(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& route_weights) const;
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
