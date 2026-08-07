#pragma once

#include "deepseek_v4_model.h"
#include "mlx_deepseek_v4_sparse.h"
#include "mlx_tensor.h"

#include <memory>
#include <optional>
#include <utility>

#include <mlx/mlx.h>

namespace mfq::metal {

// Build adjacent-pair RoPE tables with the same Yarn frequency correction as
// the DeepSeek-V4 reference runtime.
std::pair<mlx::core::array, mlx::core::array>
deepseek_v4_yarn_tables(
    int dimension,
    int length,
    float theta,
    const DeepseekV4RopeScaling& scaling = {});

mlx::core::array deepseek_v4_rope_adjacent(
    const mlx::core::array& value,
    const mlx::core::array& cosine,
    const mlx::core::array& sine,
    bool inverse = false);

mlx::core::array deepseek_v4_unweighted_rms(
    const mlx::core::array& value,
    float eps);

class MlxDeepseekV4PoolState {
public:
    static MlxDeepseekV4PoolState allocate(
        int ratio,
        int head_dim,
        bool overlap,
        int batch,
        int max_context,
        mlx::core::Dtype dtype = mlx::core::float16);

    void update(
        const mlx::core::array& kv_token,
        const mlx::core::array& gate_token,
        const mlx::core::array& ape,
        const mlx::core::array& norm,
        int length,
        const mlx::core::array& compressed_cosine,
        const mlx::core::array& compressed_sine,
        int quant_mode,
        float eps);

    // Build complete compressor windows in parallel during prefill, then
    // retain only the bounded tail state required by subsequent decode.
    // The starting position must be ratio-aligned; callers can fall back to
    // update() for an already-partial window.
    void prefill(
        const mlx::core::array& kv,
        const mlx::core::array& gate,
        const mlx::core::array& ape,
        const mlx::core::array& norm,
        int start_position,
        const mlx::core::array& compressed_cosine,
        const mlx::core::array& compressed_sine,
        int quant_mode,
        float eps);

    int ratio() const noexcept {
        return ratio_;
    }
    int head_dim() const noexcept {
        return head_dim_;
    }
    bool overlap() const noexcept {
        return overlap_;
    }
    int batch() const noexcept {
        return batch_;
    }
    int capacity() const noexcept {
        return capacity_;
    }
    int pool_len() const noexcept {
        return pool_len_;
    }
    int remainder() const noexcept {
        return remainder_;
    }

    const mlx::core::array& pool() const noexcept {
        return pool_;
    }
    const mlx::core::array& state_kv() const noexcept {
        return state_kv_;
    }
    const mlx::core::array& state_gate() const noexcept {
        return state_gate_;
    }
    const std::optional<mlx::core::array>&
    prev_kv() const noexcept {
        return prev_kv_;
    }
    const std::optional<mlx::core::array>&
    prev_gate() const noexcept {
        return prev_gate_;
    }

    // Take a rollback snapshot. Rolling compressor state is copied because
    // decode replaces it, while the fixed pool may share storage: it is
    // append-only and restored metadata hides rows beyond the checkpoint.
    MlxDeepseekV4PoolState snapshot() const;

private:
    MlxDeepseekV4PoolState(
        int ratio,
        int head_dim,
        bool overlap,
        int batch,
        int capacity,
        mlx::core::Dtype dtype,
        mlx::core::array pool,
        mlx::core::array state_kv,
        mlx::core::array state_gate,
        std::optional<mlx::core::array> prev_kv,
        std::optional<mlx::core::array> prev_gate);

    int ratio_;
    int head_dim_;
    bool overlap_;
    int batch_;
    int capacity_;
    mlx::core::Dtype dtype_;
    mlx::core::array pool_;
    mlx::core::array state_kv_;
    mlx::core::array state_gate_;
    std::optional<mlx::core::array> prev_kv_;
    std::optional<mlx::core::array> prev_gate_;
    int pool_len_ = 0;
    int remainder_ = 0;
};

class MlxDeepseekV4LayerState {
public:
    static MlxDeepseekV4LayerState allocate(
        const DeepseekV4Config& config,
        int ratio,
        int batch,
        int max_context,
        mlx::core::Dtype dtype = mlx::core::float16);

    int batch() const noexcept {
        return local_.shape(0);
    }
    int position() const noexcept {
        return position_;
    }
    const mlx::core::array& local_state() const noexcept {
        return local_;
    }
    mlx::core::array local_positions() const;
    const std::optional<MlxDeepseekV4PoolState>&
    main() const noexcept {
        return main_;
    }
    const std::optional<MlxDeepseekV4PoolState>&
    indexer() const noexcept {
        return indexer_;
    }

    MlxDeepseekV4LayerState snapshot() const;

private:
    friend class MlxDeepseekV4Attention;

    MlxDeepseekV4LayerState(
        mlx::core::array local,
        std::optional<MlxDeepseekV4PoolState> main,
        std::optional<MlxDeepseekV4PoolState> indexer);

    mlx::core::array local_;
    std::optional<MlxDeepseekV4PoolState> main_;
    std::optional<MlxDeepseekV4PoolState> indexer_;
    int position_ = 0;
};

// Injectable construction keeps the attention graph testable independently of
// container I/O while using the same native MlxLinear objects as production.
struct MlxDeepseekV4AttentionComponents {
    MlxLinear q_a;
    MlxLinear kv;
    MlxLinear q_b;
    MlxLinear wo_a;
    MlxLinear wo_b;
    mlx::core::array q_norm;
    mlx::core::array kv_norm;
    mlx::core::array sinks;

    std::optional<MlxLinear> main_kv;
    std::optional<MlxLinear> main_gate;
    std::optional<mlx::core::array> main_ape;
    std::optional<mlx::core::array> main_norm;

    std::optional<MlxLinear> index_q_b;
    std::optional<MlxLinear> index_kv;
    std::optional<MlxLinear> index_gate;
    std::optional<MlxLinear> index_weights;
    std::optional<mlx::core::array> index_ape;
    std::optional<mlx::core::array> index_norm;
};

class MlxDeepseekV4Attention {
public:
    static MlxDeepseekV4Attention load(
        const MfqContainer& model,
        const DeepseekV4Config& config,
        int layer,
        int ratio,
        int max_context);

    static MlxDeepseekV4Attention load(
        const MfqContainer& model,
        const DeepseekV4Config& config,
        int layer,
        int ratio,
        int max_context,
        std::pair<mlx::core::array, mlx::core::array>
            rope_base,
        std::pair<mlx::core::array, mlx::core::array>
            rope_compressed);

    MlxDeepseekV4Attention(
        DeepseekV4Config config,
        int layer,
        int ratio,
        int max_context,
        MlxDeepseekV4AttentionComponents components,
        std::pair<mlx::core::array, mlx::core::array>
            rope_base,
        std::pair<mlx::core::array, mlx::core::array>
            rope_compressed);

    mlx::core::array operator()(
        const mlx::core::array& input,
        MlxDeepseekV4LayerState& state,
        int pos0) const;

    int ratio() const noexcept;
    int layer() const noexcept;
    int max_context() const noexcept;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace mfq::metal
