#pragma once

#include "qwen35_model.h"

#include <cstddef>
#include <memory>
#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

namespace detail {

// Decode keeps ordinary packed GEMVs because the heterogeneous grouped
// projection kernel is slower for a single flattened row. The caller still
// checks the grouped kernel's supported row range for multi-row inputs.
inline bool qwen35_use_grouped_projection_rows(
    std::size_t input_elements,
    int input_width) noexcept {
    return input_width > 0 &&
        input_elements / static_cast<std::size_t>(input_width) != 1;
}

} // namespace detail

// Dense or packed Qwen-style SwiGLU feed-forward network:
// down(silu(gate(x)) * up(x)).
class MlxQwen35DenseSwiGlu {
public:
    static MlxQwen35DenseSwiGlu load(
        const MfqContainer& model,
        const Qwen35ResolvedLayerNames& names);

    MlxQwen35DenseSwiGlu(
        MlxLinear gate,
        MlxLinear up,
        MlxLinear down,
        std::shared_ptr<MlxQwen35DenseSwiGlu>
            important_neurons = nullptr);

    mlx::core::array operator()(
        const mlx::core::array& input) const;

    int input_size() const noexcept {
        return input_size_;
    }
    int intermediate_size() const noexcept {
        return intermediate_size_;
    }
    int output_size() const noexcept {
        return output_size_;
    }
    bool uses_grouped_gate_up() const noexcept {
        return gate_up_.has_value();
    }

private:
    mlx::core::array forward_branch(
        const mlx::core::array& input) const;

    MlxLinear gate_;
    MlxLinear up_;
    MlxLinear down_;
    std::optional<MlxGroupedLinear> gate_up_;
    std::shared_ptr<MlxQwen35DenseSwiGlu> important_neurons_;
    int input_size_ = 0;
    int intermediate_size_ = 0;
    int output_size_ = 0;
};

// Correctness-oriented reusable implementation of a Qwen3.5
// full-attention + dense/SwiGLU decoder block.
//
// The append-only KV cache position is independent from explicit multimodal
// RoPE coordinates. Ordinary text prefill/decode uses the contiguous-offset
// overload; three-axis Qwen mRoPE uses the explicit-position overload.
class MlxQwen35FullAttentionBlock {
public:
    static MlxQwen35FullAttentionBlock load(
        const MfqContainer& model,
        const Qwen35Config& config,
        const Qwen35TensorNames& names,
        std::size_t layer_index);

    MlxQwen35FullAttentionBlock(
        Qwen35Config config,
        MlxRmsNorm attention_norm,
        MlxLinear query,
        MlxLinear key,
        MlxLinear value,
        MlxLinear output,
        std::optional<MlxRmsNorm> query_norm,
        std::optional<MlxRmsNorm> key_norm,
        MlxRmsNorm ffn_norm,
        MlxQwen35DenseSwiGlu ffn);

    // Uses offset zero without a cache, or the current append position with
    // a cache. Call reset_cache() explicitly when the batch size changes.
    mlx::core::array forward(
        const mlx::core::array& input,
        bool use_cache);

    // Explicit contiguous position range [position_offset,
    // position_offset + tokens). Cached calls must append exactly at the
    // current cache position.
    mlx::core::array forward(
        const mlx::core::array& input,
        int position_offset,
        bool use_cache);

    // Explicit [tokens], [1,tokens], or [3,tokens] RoPE coordinates. The
    // cache remains append-only and uses its current token position.
    mlx::core::array forward(
        const mlx::core::array& input,
        const mlx::core::array& positions,
        bool use_cache);

    // Explicit RoPE coordinates plus the expected append-only cache
    // position. Cached calls must append exactly at position_offset.
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

    void reset_cache(
        int batch,
        int initial_capacity = 16);
    void materialize_cache();
    void clear_cache() noexcept;
    MlxKvCacheSnapshot snapshot_cache() const;
    void restore_cache(const MlxKvCacheSnapshot& snapshot);

    int cache_position() const noexcept;
    int cache_batch() const noexcept {
        return cache_batch_;
    }

    const Qwen35Config& config() const noexcept {
        return config_;
    }
    bool uses_grouped_qkv() const noexcept {
        return qkv_.has_value();
    }
    bool uses_grouped_ffn() const noexcept {
        return ffn_.uses_grouped_gate_up();
    }

private:
    void validate_components() const;
    mlx::core::array forward_impl(
        const mlx::core::array& input,
        const mlx::core::array* positions,
        int position_offset,
        bool use_cache);

    Qwen35Config config_;
    MlxRmsNorm attention_norm_;
    MlxLinear query_;
    MlxLinear key_;
    MlxLinear value_;
    MlxLinear output_;
    std::optional<MlxGroupedLinear> qkv_;
    std::optional<MlxRmsNorm> query_norm_;
    std::optional<MlxRmsNorm> key_norm_;
    MlxRmsNorm ffn_norm_;
    MlxQwen35DenseSwiGlu ffn_;
    std::unique_ptr<MlxKvCache> cache_;
    int cache_batch_ = 0;
};

} // namespace mfq::metal
