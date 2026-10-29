#pragma once

#include "deepseek_v41_model.h"
#include "mlx_grouped_linear.h"
#include "mlx_moe.h"
#include "mlx_ssd_expert_cache.h"
#include "mlx_tq_expert_cache.h"
#include "mlx_tensor.h"

#include <cstddef>
#include <array>
#include <memory>
#include <optional>
#include <string>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV41MoeResult {
    mlx::core::array routed;
    mlx::core::array shared;
};

// DeepSeek-V4.1 routed + shared expert block.  The architecture owns router
// semantics; only the heterogeneous MFE projection primitive is shared.
class MlxDeepseekV41Moe {
public:
    static MlxDeepseekV41Moe load(
        const MfqContainer& model,
        const DeepseekV41Config& config,
        const std::string& prefix,
        bool predictor = false,
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache = nullptr,
        std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache = nullptr,
        std::size_t expert_cache_layer = 0,
        std::shared_ptr<MlxTqExpertCache> tq_expert_cache = nullptr);

    MlxDeepseekV41MoeResult forward(
        const mlx::core::array& input,
        const std::optional<mlx::core::array>& image_mask = std::nullopt,
        MlxSsdPrefetchedExpertLayer* prefetched = nullptr,
        MlxMfeCachedRoute* deferred_route = nullptr) const;

    std::optional<MlxSsdPrefetchedExpertLayer> prefetch_routed(
        std::size_t rows) const;

    int recommended_prefill_chunk_size() const noexcept;
    std::size_t resident_routed_bytes() const noexcept;
    void validate_offload(const std::shared_ptr<MlxMfeOffloadCache>& cache) const;
    void use_offload(const std::shared_ptr<MlxMfeOffloadCache>& cache);
    void validate_tq_offload(const std::shared_ptr<MlxTqExpertCache>& cache) const;
    void use_tq_offload(const std::shared_ptr<MlxTqExpertCache>& cache);
    std::optional<MlxMfeCachedRoute> cached_route() const;

private:
    using TqWeights = MlxTqExpertWeights;
    MlxDeepseekV41Moe(
        int hidden,
        int intermediate,
        int experts,
        int top_k,
        bool normalize,
        float scale,
        float swiglu_limit,
        MlxLinear router,
        mlx::core::array router_bias,
        std::optional<mlx::core::array> vision_bias,
        MlxLinear shared_gate,
        MlxLinear shared_up,
        MlxLinear shared_down,
        std::optional<MlxMoeWeight> routed_gate_up,
        std::optional<MlxRoutedLinear> routed_gate,
        std::optional<MlxRoutedLinear> routed_up,
        std::optional<MlxRoutedLinear> routed_down,
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache,
        std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache,
        std::string streamed_gate_up_name,
        std::optional<std::string> streamed_up_name,
        std::string streamed_down_name,
        std::size_t expert_cache_layer,
        std::optional<TqWeights> tq = std::nullopt,
        std::shared_ptr<MlxTqExpertCache> tq_expert_cache = nullptr);

    int hidden_;
    int intermediate_;
    int experts_;
    int top_k_;
    bool normalize_;
    float scale_;
    float swiglu_limit_;
    MlxLinear router_;
    mlx::core::array router_bias_;
    std::optional<mlx::core::array> vision_bias_;
    MlxLinear shared_gate_;
    MlxLinear shared_up_;
    MlxLinear shared_down_;
    std::optional<MlxProjectionBatch> projection_batch_;
    std::optional<MlxProjectionBatch> shared_gate_up_batch_;
    // Canonical split Gate/Up records keep their original storage pools;
    // routed_swiglu_pair provides the common zero-copy execution path. The
    // legacy fused gate_up record remains a read-only compatibility input.
    std::optional<MlxMoeWeight> routed_gate_up_;
    std::optional<MlxRoutedLinear> routed_gate_;
    std::optional<MlxRoutedLinear> routed_up_;
    std::optional<MlxRoutedLinear> routed_down_;
    std::optional<TqWeights> tq_;
    std::shared_ptr<MlxTqExpertCache> tq_expert_cache_;
    std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache_;
    std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache_;
    std::string streamed_gate_up_name_;
    std::optional<std::string> streamed_up_name_;
    std::string streamed_down_name_;
    std::size_t expert_cache_layer_ = 0;
};

} // namespace mfq::metal
