#pragma once

#include "mlx_linear_attention.h"
#include "mlx_qwen35_full_attention.h"
#include "qwen35_model.h"

#include <cstddef>
#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

// Qwen3.5/3.6 Gated DeltaNet decoder block.
//
// The block accepts either the combined qkv projection used by GGUF MFQ
// models, or the split qk/v projections used by some Hugging Face MFQ
// models. Cached calls append convolution and recurrent states
// contiguously; uncached calls always start from zero state.
class MlxQwen35LinearAttentionBlock {
public:
    static MlxQwen35LinearAttentionBlock load(
        const MfqContainer& model,
        const Qwen35Config& config,
        const Qwen35TensorNames& names,
        std::size_t layer_index);

    MlxQwen35LinearAttentionBlock(
        Qwen35Config config,
        bool gguf_layout,
        MlxRmsNorm attention_norm,
        std::optional<MlxLinear> combined_qkv,
        std::optional<MlxLinear> split_qk,
        std::optional<MlxLinear> split_value,
        MlxLinear z,
        MlxLinear alpha,
        MlxLinear beta,
        mlx::core::array convolution_weight,
        std::optional<mlx::core::array> convolution_bias,
        mlx::core::array dt_bias,
        mlx::core::array a,
        MlxRmsNorm linear_norm,
        MlxLinear output,
        MlxRmsNorm ffn_norm,
        MlxQwen35DenseSwiGlu ffn);

    mlx::core::array forward(
        const mlx::core::array& input,
        bool use_cache);

    mlx::core::array forward(
        const mlx::core::array& input,
        int position_offset,
        bool use_cache);

    // Gated DeltaNet has no RoPE dependency. These overloads preserve the
    // mixed-layer model interface while intentionally ignoring position
    // values after validating their token-axis shape.
    mlx::core::array forward(
        const mlx::core::array& input,
        const mlx::core::array& positions,
        bool use_cache);

    mlx::core::array forward(
        const mlx::core::array& input,
        const mlx::core::array& positions,
        int position_offset,
        bool use_cache);

    mlx::core::array operator()(
        const mlx::core::array& input,
        bool use_cache) {
        return forward(input, use_cache);
    }

    void reset_cache(int batch);
    void materialize_cache();
    void clear_cache() noexcept;

    int cache_position() const noexcept {
        return cache_position_;
    }
    int cache_batch() const noexcept {
        return cache_batch_;
    }
    bool split_input() const noexcept {
        return split_input_;
    }
    bool gguf_layout() const noexcept {
        return gguf_layout_;
    }
    bool uses_grouped_ffn() const noexcept {
        return ffn_.uses_grouped_gate_up();
    }
    bool uses_combined_alpha_beta() const noexcept {
        return alpha_beta_.has_value();
    }

    const std::optional<mlx::core::array>& convolution_state()
        const noexcept {
        return convolution_state_;
    }
    const std::optional<mlx::core::array>& recurrent_state()
        const noexcept {
        return recurrent_state_;
    }
    const Qwen35Config& config() const noexcept {
        return config_;
    }

private:
    void validate_components() const;

    Qwen35Config config_;
    bool gguf_layout_ = false;
    bool split_input_ = false;
    MlxRmsNorm attention_norm_;
    std::optional<MlxLinear> combined_qkv_;
    std::optional<MlxLinear> split_qk_;
    std::optional<MlxLinear> split_value_;
    MlxLinear z_;
    std::optional<MlxLinear> alpha_;
    std::optional<MlxLinear> beta_;
    std::optional<MlxLinear> alpha_beta_;
    mlx::core::array convolution_weight_;
    std::optional<mlx::core::array> convolution_bias_;
    mlx::core::array dt_bias_;
    mlx::core::array a_;
    MlxRmsNorm linear_norm_;
    MlxLinear output_;
    MlxRmsNorm ffn_norm_;
    MlxQwen35DenseSwiGlu ffn_;
    std::optional<mlx::core::array> convolution_state_;
    std::optional<mlx::core::array> recurrent_state_;
    std::optional<mlx::core::array> zero_convolution_state_;
    std::optional<mlx::core::array> zero_recurrent_state_;
    int zero_cache_batch_ = 0;
    int cache_position_ = 0;
    int cache_batch_ = 0;
};

} // namespace mfq::metal
