#pragma once

#include "../mfq_token_constraint.h"
#include "mfq_container.h"
#include "mlx_sampling.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxMiniCPMO45Inputs {
    mlx::core::array input_ids;
    std::optional<mlx::core::array> position_ids;
    std::optional<mlx::core::array> attention_mask;
    std::optional<mlx::core::array> pixel_values;
    std::optional<mlx::core::array> patch_mask;
    std::optional<mlx::core::array> target_sizes;
    std::optional<mlx::core::array> image_bounds;
    std::optional<mlx::core::array> audio_features;
    std::optional<mlx::core::array> audio_lengths;
    std::optional<mlx::core::array> audio_bounds;
};

struct MlxMiniCPMO45ForwardResult {
    std::optional<mlx::core::array> vision_states;
    std::optional<mlx::core::array> image_embeddings;
    std::optional<mlx::core::array> audio_embeddings;
    mlx::core::array input_embeddings;
    mlx::core::array hidden_states;
    mlx::core::array logits;
};

struct MlxMiniCPMO45TtsResult {
    mlx::core::array codes;
    std::vector<mlx::core::array> logits;
    bool finished = false;
};

struct MlxMiniCPMO45DuplexSpecialIds {
    std::int64_t unit_start = -1;
    std::int64_t unit_end = -1;
    std::int64_t image_start = -1;
    std::int64_t image_end = -1;
    std::int64_t slice_start = -1;
    std::int64_t slice_end = -1;
    std::int64_t listen = -1;
    std::int64_t speak = -1;
    std::int64_t tts_bos = -1;
    std::int64_t tts_eos = -1;
    std::int64_t chunk_eos = -1;
    std::int64_t chunk_tts_eos = -1;
    std::int64_t turn_eos = -1;
    std::int64_t tts_pad = -1;
    std::int64_t audio_bos = -1;
};

struct MlxMiniCPMO45DuplexConfig {
    MlxMiniCPMO45DuplexSpecialIds special_ids;
    std::vector<std::int64_t> forbidden_ids;
    bool greedy = false;
    double temperature = 0.7;
    std::int32_t top_k = 100;
    double top_p = 0.8;
    double listen_probability_scale = 1.0;
    double repetition_penalty = 1.05;
    std::int32_t repetition_window = 512;
    double length_penalty = 1.0;
    double tts_temperature = 0.8;
    double tts_repetition_penalty = 1.05;
    std::uint64_t seed = 0;
};

struct MlxMiniCPMO45DuplexInputs {
    std::optional<mlx::core::array> pixel_values;
    std::optional<mlx::core::array> patch_mask;
    std::optional<mlx::core::array> target_sizes;
    std::optional<mlx::core::array> image_slice_counts;
    std::optional<mlx::core::array> audio_features;
    std::int64_t audio_prefix_extra_frames = 0;
    std::int64_t audio_suffix_extra_frames = 2;
    std::optional<mlx::core::array> text_ids;
    std::int32_t max_new_speak_tokens = 20;
    bool force_listen = false;
    bool force_speak = false;
};

struct MlxMiniCPMO45DuplexResult {
    mlx::core::array decision_logits;
    std::optional<mlx::core::array> audio_embeddings;
    mlx::core::array generated_ids;
    mlx::core::array tts_codes;
    bool is_listen = false;
    bool end_of_turn = false;
    bool tts_force_flush = false;
    std::int64_t audio_chunk_index = 0;
    std::int64_t language_cache_position = 0;
    std::int64_t audio_cache_position = 0;
    std::int64_t tts_cache_position = 0;
};

// Native C++/MLX implementation of the official MiniCPM-o 4.5 composite
// graph. Processor-owned image/audio tensors use the same layouts as the
// CUDA runtime documented in docs/minicpmo45.md. Token2wav waveform rendering
// remains outside this weight graph; this runtime emits the official S3 codes.
class MlxMiniCPMO45Runtime {
public:
    static MlxMiniCPMO45Runtime load(
        const MfqContainer& model,
        std::int64_t context_size = 0,
        bool load_modalities = true);

    MlxMiniCPMO45Runtime(MlxMiniCPMO45Runtime&&) noexcept;
    MlxMiniCPMO45Runtime& operator=(MlxMiniCPMO45Runtime&&) noexcept;
    ~MlxMiniCPMO45Runtime();

    MlxMiniCPMO45Runtime(const MlxMiniCPMO45Runtime&) = delete;
    MlxMiniCPMO45Runtime& operator=(const MlxMiniCPMO45Runtime&) = delete;

    MlxMiniCPMO45ForwardResult forward(
        const MlxMiniCPMO45Inputs& inputs);

    mlx::core::array tts_condition(
        const mlx::core::array& text_ids,
        const mlx::core::array& language_hidden) const;

    mlx::core::array tts_duplex_condition(
        const mlx::core::array& text_ids,
        const mlx::core::array& language_hidden,
        std::int64_t audio_bos_token) const;

    MlxMiniCPMO45TtsResult generate_tts(
        const mlx::core::array& condition_embeddings,
        std::int32_t steps,
        std::uint64_t seed = 0);

    mlx::core::array encode_audio_streaming(
        const mlx::core::array& features,
        std::int64_t prefix_extra_frames,
        std::int64_t suffix_extra_frames);

    // Start a stateful MiniCPM-o duplex session. The optional reference audio
    // is embedded between the system prefix and suffix, then the streaming
    // audio cache is reset before the first live unit.
    void prepare_duplex(
        const MlxMiniCPMO45DuplexConfig& config,
        const std::optional<mlx::core::array>& system_prefix_ids = std::nullopt,
        const std::optional<mlx::core::array>& reference_audio_features =
            std::nullopt,
        const std::optional<mlx::core::array>& system_suffix_ids = std::nullopt);

    MlxMiniCPMO45DuplexResult duplex_step(
        const MlxMiniCPMO45DuplexInputs& inputs);

    bool duplex_prepared() const noexcept;

    void reset();

    std::int32_t generate(
        const std::vector<std::int64_t>& prompt,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const std::function<bool(std::int64_t)>& callback = {},
        const std::function<void(std::size_t, double)>&
            prefill_callback = {},
        const MfqTokenConstraintPtr& token_constraint = {});

    std::size_t layer_count() const noexcept;
    std::int64_t maximum_context() const noexcept;
    std::int64_t vocabulary_size() const noexcept;
    int cache_position() const noexcept;

private:
    class Impl;
    explicit MlxMiniCPMO45Runtime(std::unique_ptr<Impl> implementation);
    std::unique_ptr<Impl> implementation_;
};

namespace detail {

// Production-block numerical regression used by the native Metal test suite.
// It compares causal prefill with token-by-token BF16 KV-cache execution.
void test_minicpmo45_qwen3_cache_equivalence();

} // namespace detail

} // namespace mfq::metal
