#pragma once

#include "deepseek_v41_model.h"
#include "mlx_deepseek_v41_mhc.h"
#include "mlx_deepseek_v41_moe.h"
#include "mlx_mtp.h"
#include "mlx_tensor.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

class MlxDeepseekV41DSparkState {
public:
    static MlxDeepseekV41DSparkState allocate(
        int stages,
        int batch,
        int window,
        int head_dim,
        mlx::core::Dtype dtype = mlx::core::float16);

    int position() const noexcept { return position_; }
    int batch() const noexcept;
    int window() const noexcept;
    std::size_t stages() const noexcept { return rings_.size(); }
    std::size_t nbytes() const noexcept;
    const mlx::core::array& ring(std::size_t stage) const;

    MlxDeepseekV41DSparkState snapshot(bool copy_arrays = true) const;
    static MlxDeepseekV41DSparkState from_snapshot(
        std::vector<mlx::core::array> rings, int position);
    void restore_snapshot(MlxDeepseekV41DSparkState snapshot);

private:
    MlxDeepseekV41DSparkState(
        std::vector<mlx::core::array> rings,
        int position);

    std::vector<mlx::core::array> rings_;
    int position_ = 0;

    friend class MlxDeepseekV41DSpark;
};

struct MlxDeepseekV41DSparkDraft {
    mlx::core::array tokens;
    mlx::core::array logits;
    mlx::core::array confidence;
};

// Released V4.1 predictor: three complete DSpark stages with delayed
// Mega-mHC, a shared target embedding/head, and a sequential Markov bias.
// Speculative scheduling, sampling, verification and adaptive depth remain in
// the backend-wide MTP engine.
class MlxDeepseekV41DSpark {
public:
    static std::optional<MlxDeepseekV41DSpark> load_if_present(
        const MfqContainer& model,
        const DeepseekV41Config& config,
        const MlxEmbedding& embedding,
        const MlxLinear& output,
        int max_context,
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache = nullptr,
        std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache = nullptr,
        std::size_t expert_layer_base = 0,
        std::shared_ptr<MlxTqExpertCache> tq_expert_cache = nullptr);

    MlxDeepseekV41DSparkState make_state(
        int batch = 1,
        mlx::core::Dtype dtype = mlx::core::float16) const;

    void append_context(
        const mlx::core::array& target_hidden,
        MlxDeepseekV41DSparkState& state,
        int start_position,
        bool replay_window = false) const;

    MlxDeepseekV41DSparkDraft draft(
        const mlx::core::array& anchor_ids,
        MlxDeepseekV41DSparkState& state,
        const MlxMtpTokenSelector& select_token,
        int width = 0) const;

    void propose(
        const mlx::core::array& anchor_ids,
        MlxDeepseekV41DSparkState& state,
        const MlxMtpTokenSelector& select_token,
        int width = 0) const;

    MlxDeepseekV41DSparkDraft draft_greedy(
        const mlx::core::array& anchor_ids,
        MlxDeepseekV41DSparkState& state,
        int width = 0) const;

    int block_size() const noexcept;
    MlxMtpPredictorDescriptor mtp_descriptor() const noexcept {
        return MlxMtpPredictorDescriptor::dspark_block(block_size());
    }
    std::size_t stage_count() const noexcept;
    std::size_t resident_routed_bytes() const noexcept;
    void validate_offload(const std::shared_ptr<MlxMfeOffloadCache>& cache) const;
    void use_offload(const std::shared_ptr<MlxMfeOffloadCache>& cache);
    void validate_tq_offload(const std::shared_ptr<MlxTqExpertCache>& cache) const;
    void use_tq_offload(const std::shared_ptr<MlxTqExpertCache>& cache);
    void offload_until(const std::shared_ptr<MlxMfeOffloadCache>& cache,
        std::size_t& resident, std::size_t budget);

private:
    std::optional<MlxDeepseekV41DSparkDraft> draft_impl(
        const mlx::core::array& anchor_ids,
        MlxDeepseekV41DSparkState& state,
        const MlxMtpTokenSelector& select_token,
        int width,
        bool collect_diagnostics) const;

    struct Impl;
    explicit MlxDeepseekV41DSpark(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_;
};

} // namespace mfq::metal
