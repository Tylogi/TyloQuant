#pragma once

#include "mlx_qwen35_full_attention.h"
#include "mlx_qwen35_linear_attention.h"
#include "mlx_sampling.h"
#include "qwen35_model.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

namespace detail {

// Generation only needs a dense vocabulary-sized count vector when a
// configured penalty consumes it. Returning nullopt for the common
// no-penalty path also prevents an otherwise-unused per-token graph chain.
std::optional<mlx::core::array>
qwen35_generation_token_counts(
    const MlxSamplingParams& sampling,
    const mlx::core::array& prompt_ids,
    int vocab);

} // namespace detail

using MlxQwen35Layer = std::variant<
    MlxQwen35FullAttentionBlock,
    MlxQwen35LinearAttentionBlock>;

using MlxTokenCallback = std::function<bool(std::int64_t)>;

class MlxQwen35CausalLm {
public:
    static MlxQwen35CausalLm load(const MfqContainer& model);

    static MlxQwen35CausalLm load(
        const MfqContainer& model,
        const Qwen35Config& config,
        const Qwen35TensorNames& names);

    MlxQwen35CausalLm(
        Qwen35Config config,
        Qwen35Embedding embedding,
        std::vector<MlxQwen35Layer> layers,
        MlxRmsNorm output_norm,
        std::optional<Qwen35Linear> output,
        mlx::core::Dtype activation_dtype = mlx::core::float16);

    // token_ids must be a non-empty [batch,tokens] array. Cached calls append
    // contiguously; uncached calls do not mutate existing cache state.
    mlx::core::array forward(
        const mlx::core::array& token_ids,
        bool use_cache = true);

    // Explicit text or three-axis multimodal RoPE coordinates. positions
    // accepts [tokens], [1,tokens], or [3,tokens]. Cache storage still
    // advances contiguously by token count.
    mlx::core::array forward(
        const mlx::core::array& token_ids,
        const mlx::core::array& positions,
        bool use_cache = true);

    mlx::core::array operator()(
        const mlx::core::array& token_ids,
        bool use_cache = true) {
        return forward(token_ids, use_cache);
    }

    mlx::core::array operator()(
        const mlx::core::array& token_ids,
        const mlx::core::array& positions,
        bool use_cache = true) {
        return forward(token_ids, positions, use_cache);
    }

    void reset_cache(int batch = 1);
    void clear_cache() noexcept;

    // Returns the number of sampled tokens. A token is counted before its
    // callback is invoked, so callback=false still returns a count including
    // that token and stops immediately without an extra decode.
    std::int32_t generate(
        const std::vector<std::int64_t>& prompt,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const MlxTokenCallback& callback = {},
        const std::function<void(std::size_t, double)>&
            prefill_callback = {});

    const Qwen35Config& config() const noexcept {
        return config_;
    }
    std::size_t layer_count() const noexcept {
        return layers_.size();
    }
    std::string_view layer_type(std::size_t index) const;
    int cache_position() const noexcept {
        return cache_position_;
    }
    int cache_batch() const noexcept {
        return cache_batch_;
    }

private:
    void validate_components() const;
    void prepare_cache_for_prefill(
        int batch,
        int prompt_tokens);
    mlx::core::array forward_impl(
        const mlx::core::array& token_ids,
        const mlx::core::array* positions,
        bool use_cache);

    Qwen35Config config_;
    Qwen35Embedding embedding_;
    std::vector<MlxQwen35Layer> layers_;
    MlxRmsNorm output_norm_;
    std::optional<Qwen35Linear> output_;
    mlx::core::Dtype activation_dtype_;
    int cache_position_ = 0;
    int cache_batch_ = 0;
};

} // namespace mfq::metal
