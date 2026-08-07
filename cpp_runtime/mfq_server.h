#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct MfqSamplingParams {
    int32_t max_tokens = 256;
    double temperature = 1.0;
    int32_t top_k = 20;
    double top_p = 0.95;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    double repetition_penalty = 1.0;
    uint64_t seed = 0;
};

// Describes the portion of a rendered prompt whose KV state is stable across
// requests.  A runtime may retain this exact token prefix after generation and
// reuse it only when the next request starts with the same token sequence.
struct MfqPromptCachePlan {
    size_t stable_prefix_tokens = 0;
};

struct MfqServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_name = "mfq-model";
    std::string model_type;
    std::vector<uint8_t> tokenizer_gguf;
    std::string tokenizer_model;
    std::string api_key;
    std::string web_root;
    int64_t max_context = 0;
    int64_t context_capacity = 0;
    int64_t vocab_size = 0;
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

using MfqTokenCallback = std::function<bool(int64_t token)>;
using MfqPrefillCallback =
    std::function<void(size_t prompt_tokens, double prefill_ms)>;
using MfqGenerateFn = std::function<int32_t(
    const std::vector<int64_t> & prompt,
    const MfqSamplingParams & sampling,
    const MfqTokenCallback & on_token,
    const MfqPrefillCallback & on_prefill,
    const MfqPromptCachePlan & cache_plan)>;
using MfqReloadFn = std::function<int64_t(int64_t context_size)>;

int run_mfq_server(
    const MfqServerConfig & config,
    const MfqGenerateFn & generate,
    const MfqReloadFn & reload = {});
MfqTokenizerProbe probe_mfq_tokenizer(
    const std::vector<uint8_t> & tokenizer_gguf,
    const std::string & text,
    bool add_special = false,
    bool parse_special = true);
MfqTokenizerProbe probe_mfq_tokenizer(
    const std::string & tokenizer_model,
    const std::string & text,
    bool add_special = false,
    bool parse_special = true);
