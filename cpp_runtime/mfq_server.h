#pragma once

#include "mfq_token_constraint.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct MfqSamplingParams {
    int32_t max_tokens = 4096;
    double temperature = 1.0;
    int32_t top_k = 100;
    double top_p = 0.95;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    double repetition_penalty = 1.0;
    bool enable_thinking = true;
    uint64_t seed = 0;
};

struct MfqChatSamplingProfile {
    std::optional<int32_t> max_tokens;
    std::optional<double> temperature;
    std::optional<int32_t> top_k;
    std::optional<double> top_p;
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<double> repetition_penalty;
    std::optional<bool> enable_thinking;
};

struct MfqDuplexSamplingProfile {
    std::optional<std::string> system_prompt;
    std::optional<std::string> decode_mode;
    std::optional<double> temperature;
    std::optional<int32_t> top_k;
    std::optional<double> top_p;
    std::optional<double> text_repetition_penalty;
    std::optional<int32_t> text_repetition_window_size;
    std::optional<double> length_penalty;
    std::optional<double> listen_prob_scale;
    std::optional<int32_t> force_listen_count;
    std::optional<int32_t> max_new_speak_tokens_per_chunk;
};

struct MfqTtsSamplingProfile {
    std::optional<double> temperature;
    std::optional<double> repetition_penalty;
    std::optional<int32_t> token2wav_steps;
};

struct MfqRuntimeProfile {
    MfqChatSamplingProfile chat;
    MfqDuplexSamplingProfile duplex;
    MfqTtsSamplingProfile tts;
    std::string source = "generic-defaults";
};

// Describes the portion of a rendered prompt whose KV state is stable across
// requests.  A runtime may retain this exact token prefix after generation and
// reuse it only when the next request starts with the same token sequence.
struct MfqPromptCachePlan {
    std::string session_id;
    size_t stable_prefix_tokens = 0;
};

struct MfqVisionInput {
    std::vector<float> pixel_values;
    std::vector<int64_t> pixel_shape;
    std::vector<uint8_t> patch_mask;
    std::vector<int64_t> patch_mask_shape;
    std::vector<int32_t> target_sizes;
    std::vector<int64_t> target_sizes_shape;
    std::vector<int64_t> image_bounds;
};

struct MfqServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_name = "mfq-model";
    std::string model_type;
    std::vector<uint8_t> tokenizer_gguf;
    std::string tokenizer_model;
    std::string api_key;
    int64_t max_context = 0;
    int64_t context_capacity = 0;
    int64_t vocab_size = 0;
    MfqRuntimeProfile runtime_profile;
};

struct MfqTokenizerProbe {
    int32_t vocab_size = 0;
    int32_t bos_token = -1;
    int32_t eos_token = -1;
    int32_t eot_token = -1;
    int32_t pad_token = -1;
    bool add_bos = false;
    bool add_eos = false;
    std::string chat_template;
    std::vector<int64_t> tokens;
};

// Internal stateful backend contract for MiniCPM-o full-duplex sessions.
// Public clients send raw PCM to mfqd; the media gateway converts each audio
// unit to exact Whisper log-Mel frames before forwarding it to this backend.
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

struct MfqDuplexBackend {
    std::string name = "native";
    std::function<void(const MfqDuplexSessionParams &)> start;
    std::function<MfqDuplexStepResult(const MfqDuplexStepInput &)> step;
    std::function<void()> stop;

    explicit operator bool() const noexcept {
        return static_cast<bool>(start) && static_cast<bool>(step) &&
            static_cast<bool>(stop);
    }
};

using MfqTokenCallback = std::function<bool(int64_t token)>;

struct MfqPrefillTiming {
    size_t prompt_tokens = 0;
    // Language-model prompt evaluation only. This is the field comparable to
    // llama.cpp's prompt-eval timing.
    double llm_ms = 0.0;
    // Vision/audio encoder and multimodal projector work preceding the LLM.
    double multimodal_ms = 0.0;
    // Complete model-side prefill wall time, excluding request parsing,
    // tokenization, media decoding/preprocessing, queueing and sampling.
    double model_ms = 0.0;
};

using MfqPrefillCallback =
    std::function<void(const MfqPrefillTiming & timing)>;
using MfqGenerateFn = std::function<int32_t(
    const std::vector<int64_t> & prompt,
    const MfqSamplingParams & sampling,
    const MfqTokenCallback & on_token,
    const MfqPrefillCallback & on_prefill,
    const MfqPromptCachePlan & cache_plan,
    const MfqTokenConstraintPtr & token_constraint)>;
using MfqMultimodalGenerateFn = std::function<int32_t(
    const std::vector<int64_t> & prompt,
    const MfqVisionInput & vision,
    const MfqSamplingParams & sampling,
    const MfqTokenCallback & on_token,
    const MfqPrefillCallback & on_prefill,
    const MfqTokenConstraintPtr & token_constraint)>;
using MfqReloadFn = std::function<int64_t(int64_t context_size)>;
using MfqRuntimeMetricsFn =
    std::function<std::vector<std::pair<std::string, double>>() >;

struct MfqSessionControl {
    std::function<size_t(
        const std::string & source_session_id,
        const std::string & target_session_id)> fork;
    std::function<size_t(const std::string & session_id)> close;
    std::function<std::vector<std::pair<std::string, double>>()> metrics;
    std::function<size_t()> clear;
};

int run_mfq_server(
    const MfqServerConfig & config,
    const MfqGenerateFn & generate,
    const MfqReloadFn & reload = {},
    const MfqDuplexBackend & duplex = {},
    const MfqSessionControl & session_control = {},
    const MfqMultimodalGenerateFn & multimodal_generate = {},
    const MfqRuntimeMetricsFn & runtime_metrics = {});
MfqTokenizerProbe probe_mfq_tokenizer(
    const std::vector<uint8_t> & tokenizer_gguf,
    const std::string & text,
    bool add_special = false,
    bool parse_special = true);
MfqRuntimeProfile resolve_mfq_runtime_profile(
    const std::string & mfq_path,
    const std::string & model_architecture,
    const std::string & model_type,
    const std::string & model_name,
    const std::string & embedded_profile_json = {},
    const std::string & model_config_json = {},
    const std::string & explicit_profile_path = {});
MfqTokenizerProbe probe_mfq_tokenizer(
    const std::string & tokenizer_model,
    const std::string & text,
    bool add_special = false,
    bool parse_special = true);
