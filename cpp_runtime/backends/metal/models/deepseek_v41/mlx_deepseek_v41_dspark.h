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
    const mlx::core::array& ring(std::size_t stage) const;

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
        int max_context);

    MlxDeepseekV41DSparkState make_state(
        int batch = 1,
        mlx::core::Dtype dtype = mlx::core::float16) const;

    void append_context(
        const mlx::core::array& target_hidden,
        MlxDeepseekV41DSparkState& state,
        int start_position) const;

    MlxDeepseekV41DSparkDraft draft(
        const mlx::core::array& anchor_ids,
        MlxDeepseekV41DSparkState& state,
        const MlxMtpTokenSelector& select_token,
        int width = 0) const;

    MlxDeepseekV41DSparkDraft draft_greedy(
        const mlx::core::array& anchor_ids,
        MlxDeepseekV41DSparkState& state,
        int width = 0) const;

    int block_size() const noexcept;
    std::size_t stage_count() const noexcept;

private:
    struct Impl;
    explicit MlxDeepseekV41DSpark(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_;
};

} // namespace mfq::metal
