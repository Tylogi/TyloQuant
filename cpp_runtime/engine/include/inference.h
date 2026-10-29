#pragma once

#include "chat.h"
#include "mfq/runtime.h"
#include "mtp_policy.h"
#include "tokenizer.h"
#include "text_emitter.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace mfq::engine {

std::unordered_set<std::int64_t> model_stop_token_ids(std::string_view config, std::int64_t vocabulary);

struct InferenceRequest {
    // Synthetic benchmark requests always execute the requested output length.
    bool benchmark_fixed_tokens = false;
    bool chat = true;
    bool stream = false;
    bool include_usage = false;
    common_chat_parser_params chat_parser;
    std::unordered_set<std::int64_t> preserved_tokens;
    std::vector<std::int64_t> prompt;
    std::vector<std::string> stops;
    std::unordered_set<std::int64_t> stop_token_ids;
    MfqSamplingParams sampling;
    MfqPromptCachePlan cache_plan;
    MfqTokenConstraintPtr token_constraint;
    std::optional<MfqMultimodalInput> vision;
};

struct InferenceMetrics {
    using Clock = std::chrono::steady_clock;

    Clock::time_point started = Clock::now();
    Clock::time_point first_token;
    Clock::time_point last_token;
    std::size_t prefill_tokens = 0;
    double prefill_ms = 0.0;
    double multimodal_ms = 0.0;
    double model_prefill_ms = 0.0;
    bool saw_token = false;
    bool saw_prefill = false;
    mtp::GenerationStats mtp;

    void mark_prefill(const MfqPrefillTiming& timing);
    void mark_token();
};

struct InferenceResult {
    std::string text;
    std::string reasoning_text;
    std::vector<common_chat_tool_call> tool_calls;
    std::string finish_reason = "length";
    std::int32_t completion_tokens = 0;
    bool client_connected = true;
    bool cancelled = false;
};

struct TokenOutput {
    std::vector<std::int64_t> token_ids;
    std::vector<common_chat_msg_diff> diffs;
};

// Engine-owned parser state. append/finish only return values.
class InferenceOutput {
public:
    InferenceOutput(const InferenceRequest& request,
                    const MfqTokenizer* tokenizer, std::string request_id);
    TokenOutput append(const std::vector<std::int64_t>& tokens);
    TokenOutput finish();
    bool stopped() const noexcept;
    InferenceResult result;
    InferenceMetrics metrics;
    // Internal teardown result, checked before the terminal event is built.
    std::exception_ptr cleanup_failure;
private:
    void parse(std::string piece, bool partial, TokenOutput& output);
    const InferenceRequest& request_;
    const MfqTokenizer* tokenizer_;
    std::string id_, generated_;
    TextEmitter emitter_;
    common_chat_msg message_;
    std::vector<std::string> tool_ids_;
    bool stopped_ = false, finished_ = false;
};

} // namespace mfq::engine
