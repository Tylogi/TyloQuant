#pragma once

#include "deepseek_v41_model.h"
#include "mlx_tensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV41AttentionSpeculation;

// Persistent storage belongs to the layer that produces it. Consumers only
// observe a source through MlxDeepseekV41SharedAttentionState while layers are
// evaluated in order. This mirrors CSA2's source schedule without coupling it
// to the older DeepSeek-V4 cache layout.
struct MlxDeepseekV41AttentionState {
    mlx::core::array local_kv = mlx::core::array(0.0f);
    std::optional<mlx::core::array> compressed_kv;
    std::optional<mlx::core::array> index_k;
    std::optional<mlx::core::array> partial_kv;
    std::optional<mlx::core::array> partial_score;
    int position = 0;
    int compressed_length = 0;
    int partial_length = 0;
    std::shared_ptr<MlxDeepseekV41AttentionSpeculation> speculation;

    static MlxDeepseekV41AttentionState allocate(
        const DeepseekV41Config& config,
        int layer,
        int batch,
        int max_context,
        mlx::core::Dtype dtype = mlx::core::float16);
};

struct MlxDeepseekV41SharedAttentionState {
    std::optional<mlx::core::array> compressed_kv;
    std::optional<mlx::core::array> index_k;
    std::optional<mlx::core::array> topk;
    std::optional<mlx::core::array> candidate_blocks;
    int compressed_length = 0;
    int ratio = 0;

    void reset() noexcept;
};

struct MlxDeepseekV41AttentionComponents {
    MlxLinear query_a;
    MlxLinear query_b;
    MlxLinear key_value;
    MlxLinear output_a;
    MlxLinear output_b;
    mlx::core::array query_a_norm;
    mlx::core::array key_value_norm;
    mlx::core::array sinks;

    std::optional<MlxLinear> compressor_key_value;
    std::optional<MlxLinear> compressor_gate;
    std::optional<mlx::core::array> compressor_norm;

    std::optional<MlxLinear> index_query;
    std::optional<MlxLinear> index_key;
    std::optional<mlx::core::array> index_key_norm;
    std::optional<MlxLinear> index_score;
};

// Independent CED/CSA2 attention adapter for DeepSeek-V4.1. Only primitive
// RoPE, persistent cache writes, and sparse-MLA execution are shared with the
// Metal backend.
class MlxDeepseekV41Attention {
public:
    static MlxDeepseekV41Attention load(
        const MfqContainer& model,
        const DeepseekV41Config& config,
        int layer,
        int max_context,
        std::pair<mlx::core::array, mlx::core::array> rope_base,
        std::pair<mlx::core::array, mlx::core::array> rope_compressed);

    MlxDeepseekV41Attention(
        DeepseekV41Config config,
        int layer,
        int max_context,
        MlxDeepseekV41AttentionComponents components,
        std::pair<mlx::core::array, mlx::core::array> rope_base,
        std::pair<mlx::core::array, mlx::core::array> rope_compressed);

    mlx::core::array forward(
        const mlx::core::array& input,
        MlxDeepseekV41AttentionState& state,
        MlxDeepseekV41SharedAttentionState& shared,
        int pos0) const;

    // The common MTP engine owns scheduling and verification. These methods
    // expose only this attention layout's compact cache transaction.
    std::vector<mlx::core::array> begin_speculative(
        MlxDeepseekV41AttentionState& state,
        int confirmed_tokens,
        int total_tokens) const;
    void commit_speculative(
        MlxDeepseekV41AttentionState& state) const noexcept;
    std::vector<mlx::core::array> rollback_speculative(
        MlxDeepseekV41AttentionState& state,
        int accepted_drafts) const;

    int layer() const noexcept { return layer_; }
    int ratio() const noexcept { return ratio_; }
    int max_context() const noexcept { return max_context_; }

private:
    DeepseekV41Config config_;
    int layer_;
    int ratio_;
    int max_context_;
    MlxDeepseekV41AttentionComponents components_;
    std::pair<mlx::core::array, mlx::core::array> rope_;
};

// Exact released activation fake-quantization boundaries. Values remain F16
// after quantize/dequantize because the sparse kernels consume unpacked cache
// rows; no persistent full-weight dequantization is involved.
mlx::core::array deepseek_v41_mxfp8_e4m3_sim(
    const mlx::core::array& input);
mlx::core::array deepseek_v41_mxfp4_e4m3_scale_sim(
    const mlx::core::array& input);

} // namespace mfq::metal
