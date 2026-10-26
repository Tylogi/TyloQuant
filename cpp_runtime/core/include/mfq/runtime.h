#pragma once

#include "mfq/token_constraint.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class MfqResourceExhausted : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A backend has lost execution state and cannot safely accept another request.
// Input/encoder failures use ordinary exceptions and leave admission healthy.
class MfqBackendFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct MfqSamplingParams {
    int32_t max_tokens = 4096;
    double temperature = 1.0;
    int32_t top_k = 100;
    double top_p = 0.95;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    double repetition_penalty = 1.0;
    bool enable_thinking = true;
    bool enable_vision = true;
    bool enable_mtp = true;
    int32_t mtp_max_draft_tokens = 3;
    uint64_t seed = 0;
};

// Describes the portion of a rendered prompt whose KV state is stable across
// requests. A runtime may retain this exact token prefix after generation and
// reuse it only when the next request starts with the same token sequence.
struct MfqPromptCachePlan {
    std::string session_id;
    size_t stable_prefix_tokens = 0;
    bool cache_output_tokens = false;
};

enum class MfqMultimodalProcessor {
    minicpmo,
    deepseek_v4,
    deepseek_v41,
    grid_vision,
};

// Architecture-neutral named media payload. Family-specific fields are
// validated before this reaches a runtime; prompt-position dependent metadata
// is produced after tokenizing.
struct MfqMultimodalInput {
    MfqMultimodalProcessor processor = MfqMultimodalProcessor::minicpmo;
    std::string processor_name;
    std::vector<float> pixel_values;
    std::vector<int64_t> pixel_shape;
    std::vector<uint8_t> patch_mask;
    std::vector<int64_t> patch_mask_shape;
    std::vector<int32_t> target_sizes;
    std::vector<int64_t> target_sizes_shape;
    std::vector<int32_t> vision_grid;
    std::vector<int64_t> vision_grid_shape;
    std::vector<int32_t> vision_types;
    std::vector<int32_t> image_grid;
    std::vector<int64_t> image_grid_shape;
    std::vector<int32_t> video_grid;
    std::vector<int64_t> video_grid_shape;
    std::vector<int64_t> image_bounds;
    std::vector<int64_t> image_permutation;
    std::vector<int64_t> image_permutation_offsets;
    std::vector<float> audio_features;
    std::vector<int64_t> audio_features_shape;
    std::vector<int64_t> audio_lengths;
    std::vector<int64_t> audio_bounds;
};

// Source compatibility for out-of-tree MiniCPM-o integrations.
using MfqVisionInput = MfqMultimodalInput;

struct MfqDuplexSessionParams {
    std::vector<int64_t> system_prefix;
    std::vector<int64_t> system_suffix;
    std::vector<float> reference_audio_features;
    int32_t reference_audio_frames = 0;
    std::vector<int64_t> special_ids;
    std::vector<int64_t> forbidden_ids;
    bool greedy = false;
    double temperature = 0.7;
    int32_t top_k = 100;
    double top_p = 0.8;
    double listen_probability_scale = 1.0;
    double repetition_penalty = 1.05;
    int32_t repetition_window = 512;
    double length_penalty = 1.0;
    double tts_temperature = 0.8;
    double tts_repetition_penalty = 1.05;
    uint64_t seed = 0;
};

struct MfqDuplexStepInput {
    std::vector<float> audio_features;
    std::vector<int64_t> text_tokens;
    int32_t audio_frames = 0;
    int64_t audio_prefix_extra_frames = 0;
    int64_t audio_suffix_extra_frames = 0;
    int32_t max_new_speak_tokens = 20;
    bool force_listen = false;
    bool force_speak = false;
};

struct MfqDuplexStepResult {
    std::vector<int64_t> generated_tokens;
    std::vector<int32_t> audio_tokens;
    bool is_listen = false;
    bool end_of_turn = false;
    bool tts_force_flush = false;
    int64_t audio_chunk_index = 0;
    int64_t language_cache_position = 0;
    int64_t audio_cache_position = 0;
    int64_t tts_cache_position = 0;
    double inference_ms = 0.0;
};

struct MfqPrefillTiming {
    size_t prompt_tokens = 0;
    // Language-model prompt evaluation only.
    double llm_ms = 0.0;
    // Vision/audio encoder and multimodal projector work preceding the LLM.
    double multimodal_ms = 0.0;
    // Complete model-side prefill wall time.
    double model_ms = 0.0;
};
