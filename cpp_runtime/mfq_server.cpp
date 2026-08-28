#include "mfq_server.h"

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "ggml.h"
#include "mfq_text.h"
#include "mfq_grammar.h"
#include "chat.h"
#include "common.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

class ApiError final : public std::runtime_error {
public:
    ApiError(int status, std::string type, std::string message, std::string param = {})
        : std::runtime_error(std::move(message)), status(status), type(std::move(type)), param(std::move(param)) {}

    int status;
    std::string type;
    std::string param;
};

static json error_body(const std::string & message, const std::string & type, const std::string & param = {}) {
    return {
        {"error", {
            {"message", message},
            {"type", type},
            {"param", param.empty() ? json(nullptr) : json(param)},
            {"code", json(nullptr)},
        }},
    };
}

static void set_json(httplib::Response & res, const json & body, int status = 200) {
    res.status = status;
    res.set_content(body.dump(), "application/json; charset=utf-8");
}

static int64_t unix_time_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string request_id(const char * prefix) {
    static std::atomic<uint64_t> sequence{0};
    const uint64_t n = sequence.fetch_add(1, std::memory_order_relaxed);
    return std::string(prefix) + std::to_string(unix_time_seconds()) + "-" + std::to_string(n);
}

static void mfq_text_log_quiet(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::cerr << "mfq-text: " << text;
    }
}

class MfqTokenizer {
public:
    explicit MfqTokenizer(const std::string & path) {
        load_from_file(path);
        finish_init();
    }

    explicit MfqTokenizer(const std::vector<uint8_t> & gguf) {
        if (gguf.empty()) {
            throw std::runtime_error("embedded tokenizer GGUF is empty");
        }
        ggml_log_set(mfq_text_log_quiet, nullptr);
        context_ = mfq_text_load_buffer(gguf.data(), gguf.size());
        if (context_ == nullptr) {
            throw std::runtime_error(
                "cannot initialize tokenizer from embedded GGUF metadata");
        }
        finish_init();
    }

    ~MfqTokenizer() {
        mfq_text_free(context_);
    }

    MfqTokenizer(const MfqTokenizer &) = delete;
    MfqTokenizer & operator=(const MfqTokenizer &) = delete;

    int32_t vocab_size() const {
        return mfq_text_vocab_n_tokens(vocab_);
    }

    std::string chat_template() const {
        const char * value = mfq_text_get_chat_template(context_, nullptr);
        return value == nullptr ? std::string() : std::string(value);
    }

    const mfq_text_context * context() const {
        return context_;
    }

    int32_t bos_token() const {
        return mfq_text_vocab_bos(vocab_);
    }

    int32_t eos_token() const {
        return mfq_text_vocab_eos(vocab_);
    }

    int32_t eot_token() const {
        return mfq_text_vocab_eot(vocab_);
    }

    int32_t pad_token() const {
        return mfq_text_vocab_pad(vocab_);
    }

    bool add_bos() const {
        return mfq_text_vocab_get_add_bos(vocab_);
    }

    bool add_eos() const {
        return mfq_text_vocab_get_add_eos(vocab_);
    }

    std::vector<int64_t> tokenize(
            const std::string & text,
            bool parse_special,
            bool add_special = false) const {
        int32_t n = mfq_text_tokenize(vocab_, text.data(), static_cast<int32_t>(text.size()),
                                   nullptr, 0, add_special, parse_special);
        if (n == std::numeric_limits<int32_t>::min()) {
            throw std::runtime_error("tokenized prompt exceeds the tokenizer limit");
        }
        if (n == 0) return {};
        if (n > 0) {
            throw std::runtime_error("tokenizer returned an invalid sizing result");
        }
        std::vector<mfq_text_token> tokens(static_cast<size_t>(-n));
        n = mfq_text_tokenize(vocab_, text.data(), static_cast<int32_t>(text.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()), add_special, parse_special);
        if (n < 0) throw std::runtime_error("tokenizer buffer sizing changed unexpectedly");
        std::vector<int64_t> out;
        out.reserve(static_cast<size_t>(n));
        for (int32_t i = 0; i < n; ++i) out.push_back(tokens[static_cast<size_t>(i)]);
        return out;
    }

    int64_t special_token_id(const std::string & text) const {
        const auto tokens = tokenize(text, true, false);
        if (tokens.size() != 1) {
            throw std::runtime_error(
                "tokenizer does not map the required special token to one ID: " +
                text);
        }
        return tokens.front();
    }

    bool is_eog(int64_t token) const {
        return mfq_text_vocab_is_eog(vocab_, static_cast<mfq_text_token>(token));
    }

    std::string piece(int64_t token, bool special = false) const {
        char local[128];
        int32_t n = mfq_text_token_to_piece(vocab_, static_cast<mfq_text_token>(token),
                                         local, static_cast<int32_t>(sizeof(local)), 0, special);
        if (n >= 0) return std::string(local, local + n);
        std::string out(static_cast<size_t>(-n), '\0');
        n = mfq_text_token_to_piece(vocab_, static_cast<mfq_text_token>(token),
                                 out.data(), static_cast<int32_t>(out.size()), 0, special);
        if (n < 0) throw std::runtime_error("token piece buffer sizing changed unexpectedly");
        out.resize(static_cast<size_t>(n));
        return out;
    }

private:
    void load_from_file(const std::string & path) {
        ggml_log_set(mfq_text_log_quiet, nullptr);
        context_ = mfq_text_load_file(path.c_str());
        if (context_ == nullptr) {
            throw std::runtime_error(
                "cannot load tokenizer metadata from GGUF: " + path);
        }
    }

    void finish_init() {
        vocab_ = mfq_text_get_vocab(context_);
        if (vocab_ == nullptr) {
            mfq_text_free(context_);
            context_ = nullptr;
            throw std::runtime_error(
                "GGUF does not contain a tokenizer vocabulary");
        }
    }

    mfq_text_context * context_ = nullptr;
    const mfq_text_vocab * vocab_ = nullptr;
};

class MfqGrammarConstraint {
public:
    MfqGrammarConstraint(
            const MfqTokenizer & tokenizer,
            const common_chat_params & params)
        : vocab_(mfq_text_get_vocab(tokenizer.context())),
          vocab_size_(tokenizer.vocab_size()) {
        if (vocab_ == nullptr || params.grammar.empty()) {
            throw std::invalid_argument(
                "cannot create an empty chat-template grammar");
        }

        std::vector<std::string> trigger_patterns;
        std::vector<mfq_text_token> trigger_tokens;
        trigger_patterns.reserve(params.grammar_triggers.size());
        trigger_tokens.reserve(params.grammar_triggers.size());
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                    trigger_patterns.push_back(
                        regex_escape(trigger.value));
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                    trigger_patterns.push_back(trigger.value);
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
                    const auto & pattern = trigger.value;
                    trigger_patterns.push_back(
                        pattern.empty()
                            ? "^$"
                            : (pattern.front() == '^' ? "" : "^") +
                                pattern +
                                (pattern.back() == '$' ? "" : "$"));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                    trigger_tokens.push_back(trigger.token);
                    break;
                default:
                    throw std::runtime_error(
                        "unknown chat-template grammar trigger type");
            }
        }

        std::vector<const char *> trigger_pattern_ptrs;
        trigger_pattern_ptrs.reserve(trigger_patterns.size());
        for (const auto & pattern : trigger_patterns) {
            trigger_pattern_ptrs.push_back(pattern.c_str());
        }

        grammar_ = mfq_text_grammar_init_impl(
            vocab_, params.grammar.c_str(), "root", params.grammar_lazy,
            trigger_pattern_ptrs.data(), trigger_pattern_ptrs.size(),
            trigger_tokens.data(), trigger_tokens.size());
        if (grammar_ == nullptr) {
            throw std::runtime_error(
                "failed to initialize chat-template grammar");
        }

        if (!params.grammar_lazy &&
            !params.generation_prompt.empty()) {
            for (const auto token : tokenizer.tokenize(
                     params.generation_prompt, true)) {
                mfq_text_grammar_accept_impl(
                    *grammar_, static_cast<mfq_text_token>(token));
            }
        }
    }

    ~MfqGrammarConstraint() {
        if (grammar_ != nullptr) {
            mfq_text_grammar_free_impl(grammar_);
        }
    }

    MfqGrammarConstraint(const MfqGrammarConstraint &) = delete;
    MfqGrammarConstraint & operator=(
        const MfqGrammarConstraint &) = delete;

    bool allows(std::int64_t token) {
        if (token < 0 || token >= vocab_size_) return false;
        mfq_text_token_data candidate = {
            static_cast<mfq_text_token>(token), 0.0f, 0.0f};
        mfq_text_token_data_array candidates = {
            &candidate, 1, -1, false};
        mfq_text_grammar_apply_impl(*grammar_, &candidates);
        return std::isfinite(candidate.logit);
    }

    void apply(float * logits, std::size_t count) {
        if (logits == nullptr ||
            count != static_cast<std::size_t>(vocab_size_)) {
            throw std::invalid_argument(
                "grammar logits do not match tokenizer vocabulary");
        }
        candidates_.resize(count);
        for (std::size_t index = 0; index < count; ++index) {
            candidates_[index] = {
                static_cast<mfq_text_token>(index), logits[index], 0.0f};
        }
        mfq_text_token_data_array candidates = {
            candidates_.data(), candidates_.size(), -1, false};
        mfq_text_grammar_apply_impl(*grammar_, &candidates);
        bool has_candidate = false;
        for (std::size_t index = 0; index < count; ++index) {
            logits[index] = candidates_[index].logit;
            has_candidate = has_candidate ||
                std::isfinite(candidates_[index].logit);
        }
        if (!has_candidate) {
            throw std::runtime_error(
                "chat-template grammar rejected every token");
        }
    }

    void accept(std::int64_t token) {
        if (token < 0 || token >= vocab_size_) {
            throw std::out_of_range(
                "grammar accepted token is out of range");
        }
        mfq_text_grammar_accept_impl(
            *grammar_, static_cast<mfq_text_token>(token));
    }

private:
    const mfq_text_vocab * vocab_ = nullptr;
    int32_t vocab_size_ = 0;
    mfq_text_grammar * grammar_ = nullptr;
    std::vector<mfq_text_token_data> candidates_;
};

static MfqTokenConstraintPtr make_token_constraint(
        const MfqTokenizer & tokenizer,
        const common_chat_params & params) {
    if (params.grammar.empty()) return {};
    auto implementation =
        std::make_shared<MfqGrammarConstraint>(tokenizer, params);
    auto constraint = std::make_shared<MfqTokenConstraint>();
    constraint->allows = [implementation](std::int64_t token) {
        return implementation->allows(token);
    };
    constraint->apply = [implementation](float * logits, std::size_t count) {
        implementation->apply(logits, count);
    };
    constraint->accept = [implementation](std::int64_t token) {
        implementation->accept(token);
    };
    return constraint;
}

static bool request_enable_thinking(const json & body, bool fallback) {
    if (body.contains("enable_thinking") && !body["enable_thinking"].is_null()) {
        if (!body["enable_thinking"].is_boolean()) {
            throw ApiError(400, "invalid_request_error", "enable_thinking must be boolean", "enable_thinking");
        }
        return body["enable_thinking"].get<bool>();
    }
    if (body.contains("chat_template_kwargs") && !body["chat_template_kwargs"].is_null()) {
        const auto & kwargs = body["chat_template_kwargs"];
        if (!kwargs.is_object()) {
            throw ApiError(400, "invalid_request_error", "chat_template_kwargs must be an object", "chat_template_kwargs");
        }
        if (kwargs.contains("enable_thinking")) {
            if (!kwargs["enable_thinking"].is_boolean()) {
                throw ApiError(400, "invalid_request_error", "enable_thinking must be boolean",
                               "chat_template_kwargs.enable_thinking");
            }
            return kwargs["enable_thinking"].get<bool>();
        }
    }
    return fallback;
}

static common_reasoning_format request_reasoning_format(const json & body) {
    if (!body.contains("reasoning_format") || body["reasoning_format"].is_null()) {
        return COMMON_REASONING_FORMAT_AUTO;
    }
    if (!body["reasoning_format"].is_string()) {
        throw ApiError(
            400, "invalid_request_error",
            "reasoning_format must be a string", "reasoning_format");
    }
    try {
        return common_reasoning_format_from_name(
            body["reasoning_format"].get<std::string>());
    } catch (const std::exception & error) {
        throw ApiError(
            400, "invalid_request_error", error.what(), "reasoning_format");
    }
}

static bool boolean_field(
        const json & body, const char * name, bool fallback) {
    if (!body.contains(name) || body[name].is_null()) return fallback;
    if (!body[name].is_boolean()) {
        throw ApiError(
            400, "invalid_request_error",
            std::string(name) + " must be boolean", name);
    }
    return body[name].get<bool>();
}

static std::string request_json_schema(const json & body) {
    const bool has_direct =
        body.contains("json_schema") && !body["json_schema"].is_null();
    const bool has_response_format =
        body.contains("response_format") &&
        !body["response_format"].is_null();
    if (has_direct && has_response_format) {
        throw ApiError(
            400, "invalid_request_error",
            "json_schema and response_format cannot both be specified",
            "response_format");
    }

    json schema;
    if (has_direct) {
        schema = body["json_schema"];
        if (schema.is_string()) {
            try {
                schema = json::parse(schema.get<std::string>());
            } catch (const std::exception &) {
                throw ApiError(
                    400, "invalid_request_error",
                    "json_schema string must contain valid JSON",
                    "json_schema");
            }
        }
    } else if (has_response_format) {
        const auto & response_format = body["response_format"];
        if (!response_format.is_object() ||
            !response_format.contains("type") ||
            !response_format["type"].is_string()) {
            throw ApiError(
                400, "invalid_request_error",
                "response_format must contain a string type",
                "response_format");
        }
        const std::string type = response_format["type"];
        if (type == "text") return {};
        if (type == "json_object") {
            schema = {{"type", "object"}};
        } else if (type == "json_schema") {
            if (!response_format.contains("json_schema") ||
                !response_format["json_schema"].is_object()) {
                throw ApiError(
                    400, "invalid_request_error",
                    "response_format.json_schema must be an object",
                    "response_format.json_schema");
            }
            const auto & envelope = response_format["json_schema"];
            schema = envelope.contains("schema")
                ? envelope["schema"]
                : envelope;
        } else {
            throw ApiError(
                400, "invalid_request_error",
                "response_format.type must be text, json_object, or json_schema",
                "response_format.type");
        }
    } else {
        return {};
    }

    if (!schema.is_object()) {
        throw ApiError(
            400, "invalid_request_error",
            "structured output JSON schema must be an object",
            has_direct ? "json_schema" : "response_format.json_schema.schema");
    }
    return schema.dump();
}

static common_chat_params apply_chat_template(
        const json & body, const common_chat_templates * templates,
        bool enable_thinking_default) {
    if (!body.contains("messages") || !body["messages"].is_array() ||
        body["messages"].empty()) {
        throw ApiError(
            400, "invalid_request_error",
            "messages must be a non-empty array", "messages");
    }

    try {
        common_chat_templates_inputs inputs;
        inputs.messages =
            common_chat_msgs_parse_oaicompat(body["messages"]);
        inputs.json_schema = request_json_schema(body);
        inputs.reasoning_format = request_reasoning_format(body);
        inputs.enable_thinking = request_enable_thinking(
            body, enable_thinking_default);
        inputs.use_jinja = true;
        inputs.add_generation_prompt =
            boolean_field(body, "add_generation_prompt", true);

        if (body.contains("continue_final_message") &&
            !body["continue_final_message"].is_null()) {
            inputs.continue_final_message =
                common_chat_continuation_parse(
                    body["continue_final_message"]);
        }
        if (inputs.continue_final_message !=
                COMMON_CHAT_CONTINUATION_NONE &&
            inputs.add_generation_prompt) {
            throw ApiError(
                400, "invalid_request_error",
                "add_generation_prompt and continue_final_message "
                "cannot both be enabled",
                "continue_final_message");
        }

        const auto caps =
            common_chat_templates_get_caps(templates);
        inputs.parallel_tool_calls = boolean_field(
            body, "parallel_tool_calls",
            caps.at("supports_parallel_tool_calls"));

        if (body.contains("tools") && !body["tools"].is_null()) {
            inputs.tools =
                common_chat_tools_parse_oaicompat(body["tools"]);
        }
        const json tool_choice =
            body.contains("tool_choice") &&
                    !body["tool_choice"].is_null()
                ? body["tool_choice"]
                : json("auto");
        if (tool_choice.is_string()) {
            inputs.tool_choice =
                common_chat_tool_choice_parse_oaicompat(
                    tool_choice.get<std::string>());
        } else if (tool_choice.is_object()) {
            if (!tool_choice.contains("type") ||
                tool_choice["type"] != "function" ||
                !tool_choice.contains("function") ||
                !tool_choice["function"].is_object() ||
                !tool_choice["function"].contains("name") ||
                !tool_choice["function"]["name"].is_string()) {
                throw ApiError(
                    400, "invalid_request_error",
                    "named tool_choice must select a function name",
                    "tool_choice");
            }
            const std::string selected_name =
                tool_choice["function"]["name"];
            const auto selected = std::find_if(
                inputs.tools.begin(), inputs.tools.end(),
                [&](const common_chat_tool & tool) {
                    return tool.name == selected_name;
                });
            if (selected == inputs.tools.end()) {
                throw ApiError(
                    400, "invalid_request_error",
                    "named tool_choice does not match any supplied tool",
                    "tool_choice");
            }
            inputs.tools = {*selected};
            inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        } else {
            throw ApiError(
                400, "invalid_request_error",
                "tool_choice must be a string or function selector object",
                "tool_choice");
        }
        if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED &&
            inputs.tools.empty()) {
            throw ApiError(
                400, "invalid_request_error",
                "tool_choice required needs at least one tool",
                "tool_choice");
        }

        if (body.contains("chat_template_kwargs") &&
            !body["chat_template_kwargs"].is_null()) {
            if (!body["chat_template_kwargs"].is_object()) {
                throw ApiError(
                    400, "invalid_request_error",
                    "chat_template_kwargs must be an object",
                    "chat_template_kwargs");
            }
            for (const auto & item :
                 body["chat_template_kwargs"].items()) {
                inputs.chat_template_kwargs[item.key()] =
                    item.value().dump();
            }
        }
        return common_chat_templates_apply(templates, inputs);
    } catch (const ApiError &) {
        throw;
    } catch (const std::exception & error) {
        throw ApiError(
            400, "invalid_request_error",
            std::string("chat template application failed: ") +
                error.what(),
            "messages");
    }
}

static int64_t integer_field(const json & body, const char * name, int64_t fallback) {
    if (!body.contains(name) || body[name].is_null()) return fallback;
    if (!body[name].is_number_integer()) {
        throw ApiError(400, "invalid_request_error", std::string(name) + " must be an integer", name);
    }
    return body[name].get<int64_t>();
}

static double number_field(const json & body, const char * name, double fallback) {
    if (!body.contains(name) || body[name].is_null()) return fallback;
    if (!body[name].is_number()) {
        throw ApiError(400, "invalid_request_error", std::string(name) + " must be a number", name);
    }
    return body[name].get<double>();
}

static MfqSamplingParams default_sampling_params(
        const MfqServerConfig & config) {
    MfqSamplingParams defaults;
    const auto & profile = config.runtime_profile.chat;
    if (profile.max_tokens) defaults.max_tokens = *profile.max_tokens;
    if (profile.temperature) defaults.temperature = *profile.temperature;
    if (profile.top_k) defaults.top_k = *profile.top_k;
    if (profile.top_p) defaults.top_p = *profile.top_p;
    if (profile.presence_penalty) {
        defaults.presence_penalty = *profile.presence_penalty;
    }
    if (profile.frequency_penalty) {
        defaults.frequency_penalty = *profile.frequency_penalty;
    }
    if (profile.repetition_penalty) {
        defaults.repetition_penalty = *profile.repetition_penalty;
    }
    if (profile.enable_thinking) {
        defaults.enable_thinking = *profile.enable_thinking;
    }
    return defaults;
}

static json sampling_params_json(const MfqSamplingParams & sampling) {
    return {
        {"max_tokens", sampling.max_tokens},
        {"temperature", sampling.temperature},
        {"top_k", sampling.top_k},
        {"top_p", sampling.top_p},
        {"presence_penalty", sampling.presence_penalty},
        {"frequency_penalty", sampling.frequency_penalty},
        {"repetition_penalty", sampling.repetition_penalty},
        {"enable_thinking", sampling.enable_thinking},
    };
}

template <typename Value>
static void merge_optional(std::optional<Value> & target,
                           const std::optional<Value> & source) {
    if (source) target = source;
}

static void merge_runtime_profile(MfqRuntimeProfile & target,
                                  const MfqRuntimeProfile & source) {
#define MFQ_MERGE(section, field) \
    merge_optional(target.section.field, source.section.field)
    MFQ_MERGE(chat, max_tokens);
    MFQ_MERGE(chat, temperature);
    MFQ_MERGE(chat, top_k);
    MFQ_MERGE(chat, top_p);
    MFQ_MERGE(chat, presence_penalty);
    MFQ_MERGE(chat, frequency_penalty);
    MFQ_MERGE(chat, repetition_penalty);
    MFQ_MERGE(chat, enable_thinking);
    MFQ_MERGE(duplex, system_prompt);
    MFQ_MERGE(duplex, decode_mode);
    MFQ_MERGE(duplex, temperature);
    MFQ_MERGE(duplex, top_k);
    MFQ_MERGE(duplex, top_p);
    MFQ_MERGE(duplex, text_repetition_penalty);
    MFQ_MERGE(duplex, text_repetition_window_size);
    MFQ_MERGE(duplex, length_penalty);
    MFQ_MERGE(duplex, listen_prob_scale);
    MFQ_MERGE(duplex, force_listen_count);
    MFQ_MERGE(duplex, max_new_speak_tokens_per_chunk);
    MFQ_MERGE(tts, temperature);
    MFQ_MERGE(tts, repetition_penalty);
    MFQ_MERGE(tts, token2wav_steps);
#undef MFQ_MERGE
    if (source.source != "generic-defaults") target.source = source.source;
}

static std::string normalized_identity(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return std::isalnum(c) ? static_cast<char>(std::tolower(c)) : '_';
    });
    return value;
}

static bool identity_matches(const std::vector<std::string> & identities,
                             const std::string & needle) {
    return std::any_of(identities.begin(), identities.end(), [&](const auto & value) {
        return normalized_identity(value).find(needle) != std::string::npos;
    });
}

static MfqRuntimeProfile architecture_runtime_profile(
        const std::vector<std::string> & identities) {
    MfqRuntimeProfile result;
    if (identity_matches(identities, "minicpmo")) {
        result.chat.temperature = 0.7;
        result.chat.top_k = 100;
        result.chat.top_p = 0.8;
        result.chat.repetition_penalty = 1.02;
        result.chat.enable_thinking = false;
        result.duplex.system_prompt = "Streaming Omni Conversation.";
        result.duplex.decode_mode = "sampling";
        result.duplex.temperature = 0.7;
        result.duplex.top_k = 100;
        result.duplex.top_p = 0.8;
        result.duplex.text_repetition_penalty = 1.05;
        result.duplex.text_repetition_window_size = 512;
        result.duplex.length_penalty = 1.0;
        result.duplex.listen_prob_scale = 1.0;
        result.duplex.force_listen_count = 0;
        result.duplex.max_new_speak_tokens_per_chunk = 20;
        result.tts.temperature = 0.8;
        result.tts.repetition_penalty = 1.05;
        result.tts.token2wav_steps = 10;
        result.source = "architecture-registry:minicpmo";
    } else if (identity_matches(identities, "deepseek_v4")) {
        result.chat.temperature = 1.0;
        result.chat.top_p = 0.8;
        result.chat.repetition_penalty = 1.05;
        result.chat.presence_penalty = 0.0;
        result.source = "architecture-registry:deepseek_v4";
    }
    return result;
}

struct MfqModelCapabilityProfile {
    std::string family = "unknown";
    bool text = true;
    bool image_input = false;
    bool video_input = false;
    bool audio_input = false;
    bool audio_output = false;
    bool full_duplex = false;
};

static MfqModelCapabilityProfile architecture_capability_profile(
        const std::string & model_type) {
    const std::string identity = normalized_identity(model_type);
    if (identity == "minicpmo") {
        return {
            "minicpmo", true, true, true, true, true, true,
        };
    }
    if (identity == "minicpmtts") {
        return {
            "minicpmo_tts", false, false, false, false, true, false,
        };
    }
    if (identity == "deepseek_v4") {
        return {"deepseek_v4"};
    }
    if (identity == "glm_moe_dsa") {
        return {"glm_dsa"};
    }
    if (identity == "gemma4" || identity == "gemma4_text") {
        return {"gemma4"};
    }
    if (identity == "qwen3_5" || identity == "qwen3_5_text") {
        return {"qwen3_5"};
    }
    MfqModelCapabilityProfile result;
    if (!identity.empty()) result.family = identity;
    return result;
}

static json model_capability_profile_json(
        const MfqModelCapabilityProfile & profile) {
    return {
        {"architecture_family", profile.family},
        {"source", "architecture-registry:" + profile.family},
        {"features", {
            {"text", profile.text},
            {"image_input", profile.image_input},
            {"video_input", profile.video_input},
            {"audio_input", profile.audio_input},
            {"audio_output", profile.audio_output},
            {"full_duplex", profile.full_duplex},
        }},
    };
}

static MfqRuntimeProfile exact_model_runtime_profile(
        const std::vector<std::string> & identities) {
    MfqRuntimeProfile result;
    if (identity_matches(identities, "minicpm_o_4_5")) {
        result = architecture_runtime_profile({"minicpmo"});
        result.source = "model-registry:minicpm-o-4_5";
    } else if (identity_matches(identities, "deepseek_v4_flash_0731")) {
        result = architecture_runtime_profile({"deepseek_v4"});
        result.source = "model-registry:deepseek-v4-flash-0731";
    }
    return result;
}

static double profile_number(const json & section, const char * name) {
    if (!section[name].is_number()) {
        throw std::runtime_error(std::string("runtime profile ") + name + " must be numeric");
    }
    const double value = section[name].get<double>();
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string("runtime profile ") + name + " must be finite");
    }
    return value;
}

static int32_t profile_integer(const json & section, const char * name) {
    if (!section[name].is_number_integer()) {
        throw std::runtime_error(std::string("runtime profile ") + name + " must be an integer");
    }
    const auto value = section[name].get<int64_t>();
    if (value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string("runtime profile ") + name + " is out of range");
    }
    return static_cast<int32_t>(value);
}

static MfqRuntimeProfile parse_runtime_profile(const std::string & text,
                                               const std::string & source) {
    const json root = json::parse(text);
    if (!root.is_object()) throw std::runtime_error("runtime profile must be a JSON object");
    if (root.contains("schema") && root["schema"] != "mfq.runtime.sampling") {
        throw std::runtime_error("unsupported runtime profile schema");
    }
    if (root.contains("version") &&
        (!root["version"].is_number_integer() || root["version"] != 1)) {
        throw std::runtime_error("unsupported runtime profile version");
    }
    MfqRuntimeProfile result;
    result.source = source;
    if (root.contains("chat")) {
        const auto & value = root["chat"];
        if (!value.is_object()) throw std::runtime_error("runtime profile chat must be an object");
#define MFQ_CHAT_NUMBER(field) if (value.contains(#field)) result.chat.field = profile_number(value, #field)
        if (value.contains("max_tokens")) result.chat.max_tokens = profile_integer(value, "max_tokens");
        MFQ_CHAT_NUMBER(temperature);
        if (value.contains("top_k")) result.chat.top_k = profile_integer(value, "top_k");
        MFQ_CHAT_NUMBER(top_p);
        MFQ_CHAT_NUMBER(presence_penalty);
        MFQ_CHAT_NUMBER(frequency_penalty);
        MFQ_CHAT_NUMBER(repetition_penalty);
        if (value.contains("enable_thinking")) {
            if (!value["enable_thinking"].is_boolean()) {
                throw std::runtime_error(
                    "runtime profile chat.enable_thinking must be boolean");
            }
            result.chat.enable_thinking =
                value["enable_thinking"].get<bool>();
        }
#undef MFQ_CHAT_NUMBER
    }
    if (root.contains("duplex")) {
        const auto & value = root["duplex"];
        if (!value.is_object()) throw std::runtime_error("runtime profile duplex must be an object");
        if (value.contains("system_prompt")) {
            if (!value["system_prompt"].is_string()) {
                throw std::runtime_error(
                    "runtime profile duplex.system_prompt must be a string");
            }
            result.duplex.system_prompt =
                value["system_prompt"].get<std::string>();
        }
        if (value.contains("decode_mode")) {
            if (!value["decode_mode"].is_string()) throw std::runtime_error("runtime profile decode_mode must be a string");
            result.duplex.decode_mode = value["decode_mode"].get<std::string>();
            if (*result.duplex.decode_mode != "sampling" && *result.duplex.decode_mode != "greedy") {
                throw std::runtime_error("runtime profile decode_mode is invalid");
            }
        }
#define MFQ_DUPLEX_NUMBER(field) if (value.contains(#field)) result.duplex.field = profile_number(value, #field)
#define MFQ_DUPLEX_INTEGER(field) if (value.contains(#field)) result.duplex.field = profile_integer(value, #field)
        MFQ_DUPLEX_NUMBER(temperature);
        MFQ_DUPLEX_INTEGER(top_k);
        MFQ_DUPLEX_NUMBER(top_p);
        MFQ_DUPLEX_NUMBER(text_repetition_penalty);
        MFQ_DUPLEX_INTEGER(text_repetition_window_size);
        MFQ_DUPLEX_NUMBER(length_penalty);
        MFQ_DUPLEX_NUMBER(listen_prob_scale);
        MFQ_DUPLEX_INTEGER(force_listen_count);
        MFQ_DUPLEX_INTEGER(max_new_speak_tokens_per_chunk);
#undef MFQ_DUPLEX_NUMBER
#undef MFQ_DUPLEX_INTEGER
    }
    if (root.contains("tts")) {
        const auto & value = root["tts"];
        if (!value.is_object()) throw std::runtime_error("runtime profile tts must be an object");
        if (value.contains("temperature")) result.tts.temperature = profile_number(value, "temperature");
        if (value.contains("repetition_penalty")) result.tts.repetition_penalty = profile_number(value, "repetition_penalty");
        if (value.contains("token2wav_steps")) result.tts.token2wav_steps = profile_integer(value, "token2wav_steps");
    }
    const auto bounded = [](const std::optional<double> & value,
                            double low, double high,
                            const char * name) {
        if (value && (*value < low || *value > high)) {
            throw std::runtime_error(
                std::string("runtime profile ") + name + " is out of range");
        }
    };
    const auto positive = [](const auto & value, const char * name) {
        if (value && *value <= 0) {
            throw std::runtime_error(
                std::string("runtime profile ") + name + " must be positive");
        }
    };
    bounded(result.chat.temperature, 0.0, 10.0, "chat.temperature");
    bounded(result.chat.top_p, 0.0, 1.0, "chat.top_p");
    bounded(result.duplex.temperature, 0.0, 10.0, "duplex.temperature");
    bounded(result.duplex.top_p, 0.0, 1.0, "duplex.top_p");
    bounded(result.tts.temperature, 0.0, 10.0, "tts.temperature");
    if (result.chat.top_k && *result.chat.top_k < 0) {
        throw std::runtime_error("runtime profile chat.top_k must be non-negative");
    }
    if (result.duplex.top_k && *result.duplex.top_k < 0) {
        throw std::runtime_error("runtime profile duplex.top_k must be non-negative");
    }
    positive(result.chat.max_tokens, "chat.max_tokens");
    positive(result.chat.repetition_penalty, "chat.repetition_penalty");
    positive(result.duplex.text_repetition_penalty,
             "duplex.text_repetition_penalty");
    positive(result.duplex.text_repetition_window_size,
             "duplex.text_repetition_window_size");
    positive(result.duplex.length_penalty, "duplex.length_penalty");
    positive(result.duplex.max_new_speak_tokens_per_chunk,
             "duplex.max_new_speak_tokens_per_chunk");
    positive(result.tts.repetition_penalty, "tts.repetition_penalty");
    positive(result.tts.token2wav_steps, "tts.token2wav_steps");
    if (result.duplex.listen_prob_scale &&
        *result.duplex.listen_prob_scale < 0.0) {
        throw std::runtime_error(
            "runtime profile duplex.listen_prob_scale must be non-negative");
    }
    if (result.duplex.force_listen_count &&
        (*result.duplex.force_listen_count < 0 ||
         *result.duplex.force_listen_count > 60)) {
        throw std::runtime_error(
            "runtime profile duplex.force_listen_count is out of range");
    }
    return result;
}

static std::string read_profile_file(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open runtime profile: " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

static std::vector<std::filesystem::path> profile_sidecar_paths(
        const std::filesystem::path & mfq_path) {
    std::vector<std::filesystem::path> result;
    static const std::regex split_pattern(R"(^(.*)-[0-9]{5}-of-[0-9]{5}\.mfq$)");
    std::smatch match;
    const auto filename = mfq_path.filename().string();
    if (std::regex_match(filename, match, split_pattern)) {
        result.push_back(mfq_path.parent_path() / (match[1].str() + ".runtime.json"));
    } else {
        auto family = mfq_path;
        family.replace_extension(".runtime.json");
        result.push_back(std::move(family));
    }
    auto exact = std::filesystem::path(mfq_path.string() + ".runtime.json");
    if (exact != result.front()) result.push_back(std::move(exact));
    return result;
}

static json duplex_profile_json(const MfqDuplexSamplingProfile & value) {
    json result = json::object();
#define MFQ_SET(field) if (value.field) result[#field] = *value.field
    MFQ_SET(system_prompt);
    MFQ_SET(decode_mode);
    MFQ_SET(temperature);
    MFQ_SET(top_k);
    MFQ_SET(top_p);
    MFQ_SET(text_repetition_penalty);
    MFQ_SET(text_repetition_window_size);
    MFQ_SET(length_penalty);
    MFQ_SET(listen_prob_scale);
    MFQ_SET(force_listen_count);
    MFQ_SET(max_new_speak_tokens_per_chunk);
#undef MFQ_SET
    return result;
}

static json tts_profile_json(const MfqTtsSamplingProfile & value) {
    json result = json::object();
    if (value.temperature) result["temperature"] = *value.temperature;
    if (value.repetition_penalty) result["repetition_penalty"] = *value.repetition_penalty;
    if (value.token2wav_steps) result["token2wav_steps"] = *value.token2wav_steps;
    return result;
}

static json chat_template_capabilities_json(
        const std::string & chat_template) {
    const bool supports_thinking =
        chat_template.find("enable_thinking") != std::string::npos;
    json reasoning_effort_values = json::array();
    if (chat_template.find("reasoning_effort") != std::string::npos) {
        const auto supports_value = [&](const char * value) {
            return chat_template.find(
                       std::string("'") + value + "'") !=
                       std::string::npos ||
                   chat_template.find(
                       std::string("\"") + value + "\"") !=
                       std::string::npos;
        };
        for (const char * value : {"high", "max"}) {
            if (supports_value(value)) {
                reasoning_effort_values.push_back(value);
            }
        }
    }
    return {
        {"thinking", {
            {"supported", supports_thinking},
        }},
        {"reasoning_effort", {
            {"supported", !reasoning_effort_values.empty()},
            {"values", std::move(reasoning_effort_values)},
        }},
    };
}

static std::vector<std::string> parse_stops(const json & body) {
    std::vector<std::string> stops;
    if (!body.contains("stop") || body["stop"].is_null()) return stops;
    if (body["stop"].is_string()) {
        stops.push_back(body["stop"].get<std::string>());
    } else if (body["stop"].is_array()) {
        if (body["stop"].size() > 16) {
            throw ApiError(400, "invalid_request_error", "stop accepts at most 16 strings", "stop");
        }
        for (const auto & stop : body["stop"]) {
            if (!stop.is_string()) {
                throw ApiError(400, "invalid_request_error", "stop entries must be strings", "stop");
            }
            stops.push_back(stop.get<std::string>());
        }
    } else {
        throw ApiError(400, "invalid_request_error", "stop must be a string or an array of strings", "stop");
    }
    for (const auto & stop : stops) {
        if (stop.empty()) throw ApiError(400, "invalid_request_error", "stop strings cannot be empty", "stop");
    }
    return stops;
}

struct RequestWork {
    bool chat = true;
    bool stream = false;
    bool include_usage = false;
    common_chat_parser_params chat_parser;
    std::unordered_set<int64_t> preserved_tokens;
    std::vector<int64_t> prompt;
    std::vector<std::string> stops;
    MfqSamplingParams sampling;
    MfqPromptCachePlan cache_plan;
    MfqTokenConstraintPtr token_constraint;
    std::optional<MfqVisionInput> vision;
};

static bool valid_mfq_session_id(const std::string & session_id) {
    return !session_id.empty() && session_id.size() <= 128 &&
        std::all_of(
            session_id.begin(), session_id.end(),
            [](unsigned char value) {
                return std::isalnum(value) != 0 || value == '-' ||
                    value == '_' || value == '.' || value == ':';
            });
}

class RequestCancellationRegistry {
public:
    struct Lease {
        std::shared_ptr<std::atomic<bool>> flag;
        std::function<void()> release;

        ~Lease() {
            if (release) release();
        }
    };

    std::shared_ptr<Lease> activate(const std::string & session_id) {
        auto flag = std::make_shared<std::atomic<bool>>(false);
        if (session_id.empty()) {
            auto lease = std::make_shared<Lease>();
            lease->flag = std::move(flag);
            return lease;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto found = active_.find(session_id);
            if (found != active_.end()) {
                found->second->store(true, std::memory_order_release);
            }
            active_[session_id] = flag;
        }
        auto release = [this, session_id, flag]() {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto found = active_.find(session_id);
            if (found != active_.end() && found->second == flag) {
                active_.erase(found);
            }
        };
        auto lease = std::make_shared<Lease>();
        lease->flag = std::move(flag);
        lease->release = std::move(release);
        return lease;
    }

    bool cancel(const std::string & session_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = active_.find(session_id);
        if (found == active_.end()) return false;
        found->second->store(true, std::memory_order_release);
        return true;
    }

private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<std::atomic<bool>>> active_;
};

static RequestWork parse_work(const json & body, bool chat, const MfqTokenizer & tokenizer,
                              const common_chat_templates * templates,
                              int64_t max_context,
                              const std::string & model_type,
                              const MfqSamplingParams & defaults) {
    if (!body.is_object()) throw ApiError(400, "invalid_request_error", "request body must be a JSON object");
    if (integer_field(body, "n", 1) != 1) {
        throw ApiError(400, "unsupported_parameter", "only n=1 is supported", "n");
    }
    if (body.contains("logprobs") && !body["logprobs"].is_null()) {
        if (!body["logprobs"].is_boolean()) {
            throw ApiError(400, "invalid_request_error", "logprobs must be boolean", "logprobs");
        }
        if (body["logprobs"].get<bool>()) {
            throw ApiError(400, "unsupported_parameter", "logprobs are not implemented", "logprobs");
        }
    }
    if (number_field(body, "min_p", 0.0) != 0.0) {
        throw ApiError(400, "unsupported_parameter", "min_p is not implemented", "min_p");
    }

    RequestWork work;
    work.chat = chat;
    if (body.contains("stream") && !body["stream"].is_null() && !body["stream"].is_boolean()) {
        throw ApiError(400, "invalid_request_error", "stream must be boolean", "stream");
    }
    work.stream = body.contains("stream") && !body["stream"].is_null()
        ? body["stream"].get<bool>()
        : false;
    if (body.contains("stream_options") && !body["stream_options"].is_null()) {
        if (!body["stream_options"].is_object()) {
            throw ApiError(400, "invalid_request_error", "stream_options must be an object", "stream_options");
        }
        const auto & stream_options = body["stream_options"];
        if (stream_options.contains("include_usage") && !stream_options["include_usage"].is_null() &&
            !stream_options["include_usage"].is_boolean()) {
            throw ApiError(400, "invalid_request_error", "include_usage must be boolean",
                           "stream_options.include_usage");
        }
        work.include_usage = stream_options.contains("include_usage") && !stream_options["include_usage"].is_null()
            ? stream_options["include_usage"].get<bool>()
            : false;
    }

    const int64_t max_tokens = body.contains("max_completion_tokens")
        ? integer_field(body, "max_completion_tokens", defaults.max_tokens)
        : integer_field(body, "max_tokens", defaults.max_tokens);
    if (max_tokens < 1 || max_tokens > std::numeric_limits<int32_t>::max()) {
        throw ApiError(400, "invalid_request_error", "max_tokens must be positive", "max_tokens");
    }
    work.sampling.max_tokens = static_cast<int32_t>(max_tokens);
    work.sampling.temperature = number_field(
        body, "temperature", defaults.temperature);
    work.sampling.top_p = number_field(body, "top_p", defaults.top_p);
    work.sampling.top_k = static_cast<int32_t>(integer_field(
        body, "top_k", defaults.top_k));
    work.sampling.presence_penalty = number_field(
        body, "presence_penalty", defaults.presence_penalty);
    work.sampling.frequency_penalty = number_field(
        body, "frequency_penalty", defaults.frequency_penalty);
    work.sampling.repetition_penalty = number_field(
        body, "repetition_penalty", defaults.repetition_penalty);
    work.sampling.enable_thinking = request_enable_thinking(
        body, defaults.enable_thinking);
    if (work.sampling.temperature < 0.0 || work.sampling.temperature > 10.0) {
        throw ApiError(400, "invalid_request_error", "temperature must be in [0, 10]", "temperature");
    }
    if (work.sampling.top_p <= 0.0 || work.sampling.top_p > 1.0) {
        throw ApiError(400, "invalid_request_error", "top_p must be in (0, 1]", "top_p");
    }
    if (work.sampling.top_k < 0 || work.sampling.top_k > 1024) {
        throw ApiError(400, "invalid_request_error", "top_k must be in [0, 1024]", "top_k");
    }
    if (work.sampling.temperature > 0.0 && work.sampling.top_k == 0 && work.sampling.top_p < 1.0) {
        throw ApiError(400, "invalid_request_error", "top_p below 1 requires top_k above 0 in this sampler", "top_p");
    }
    if (work.sampling.presence_penalty < -2.0 || work.sampling.presence_penalty > 2.0) {
        throw ApiError(400, "invalid_request_error", "presence_penalty must be in [-2, 2]", "presence_penalty");
    }
    if (work.sampling.frequency_penalty < -2.0 || work.sampling.frequency_penalty > 2.0) {
        throw ApiError(400, "invalid_request_error", "frequency_penalty must be in [-2, 2]", "frequency_penalty");
    }
    if (work.sampling.repetition_penalty <= 0.0 || work.sampling.repetition_penalty > 10.0) {
        throw ApiError(400, "invalid_request_error", "repetition_penalty must be in (0, 10]", "repetition_penalty");
    }
    if (body.contains("seed") && !body["seed"].is_null()) {
        const int64_t seed = integer_field(body, "seed", 0);
        work.sampling.seed = static_cast<uint64_t>(seed);
    } else {
        std::random_device device;
        work.sampling.seed = (static_cast<uint64_t>(device()) << 32) ^ device();
    }

    std::string prompt;
    bool parse_special = false;
    if (chat) {
        const common_chat_params chat_params =
            apply_chat_template(
                body, templates, work.sampling.enable_thinking);
        work.token_constraint =
            make_token_constraint(tokenizer, chat_params);
        prompt = chat_params.prompt;
        parse_special = true;
        work.chat_parser.format = chat_params.format;
        work.chat_parser.reasoning_format =
            request_reasoning_format(body);
        work.chat_parser.reasoning_in_content =
            work.stream &&
            work.chat_parser.reasoning_format ==
                COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
        work.chat_parser.generation_prompt =
            chat_params.generation_prompt;
        work.chat_parser.parse_tool_calls = true;
        if (!chat_params.parser.empty()) {
            work.chat_parser.parser.load(chat_params.parser);
        }
        if (body.contains("continue_final_message") &&
            !body["continue_final_message"].is_null()) {
            work.chat_parser.is_continuation =
                common_chat_continuation_parse(
                    body["continue_final_message"]) !=
                COMMON_CHAT_CONTINUATION_NONE;
        }
        for (const auto & text : chat_params.preserved_tokens) {
            const auto tokens = tokenizer.tokenize(text, true);
            work.preserved_tokens.insert(tokens.begin(), tokens.end());
        }
        work.stops.insert(
            work.stops.end(), chat_params.additional_stops.begin(),
            chat_params.additional_stops.end());
    } else {
        if (!body.contains("prompt") || !body["prompt"].is_string()) {
            throw ApiError(400, "invalid_request_error", "prompt must be a string", "prompt");
        }
        prompt = body["prompt"].get<std::string>();
    }
    work.prompt = tokenizer.tokenize(prompt, parse_special);
    if (work.prompt.empty()) throw ApiError(400, "invalid_request_error", "prompt tokenized to an empty sequence", "prompt");
    if (body.contains("mfq_session_id") && !body["mfq_session_id"].is_null()) {
        if (!body["mfq_session_id"].is_string()) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_session_id must be a string", "mfq_session_id");
        }
        work.cache_plan.session_id =
            body["mfq_session_id"].get<std::string>();
        if (!valid_mfq_session_id(work.cache_plan.session_id)) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_session_id must contain 1 to 128 safe identifier bytes",
                "mfq_session_id");
        }
        work.cache_plan.stable_prefix_tokens = work.prompt.size();
    }
    if (chat && model_type == "deepseek_v4" &&
        boolean_field(body, "add_generation_prompt", true)) {
        const std::string stable_marker =
            work.sampling.enable_thinking ? "<think>" : "</think>";
        const auto marker_tokens =
            tokenizer.tokenize(stable_marker, true);
        if (!marker_tokens.empty() &&
            marker_tokens.size() < work.prompt.size() &&
            std::equal(
                marker_tokens.rbegin(),
                marker_tokens.rend(),
                work.prompt.rbegin())) {
            work.cache_plan.stable_prefix_tokens =
                work.prompt.size() - marker_tokens.size();
        }
    }
    if (max_context > 0 &&
        static_cast<int64_t>(work.prompt.size()) + max_tokens > max_context) {
        throw ApiError(400, "context_length_exceeded",
                       "prompt tokens plus max_tokens exceed the model context window", "max_tokens");
    }
    const auto requested_stops = parse_stops(body);
    work.stops.insert(
        work.stops.end(), requested_stops.begin(),
        requested_stops.end());
    std::vector<std::string> unique_stops;
    unique_stops.reserve(work.stops.size());
    for (const auto & stop : work.stops) {
        if (std::find(
                unique_stops.begin(), unique_stops.end(), stop) ==
            unique_stops.end()) {
            unique_stops.push_back(stop);
        }
    }
    work.stops = std::move(unique_stops);
    return work;
}

static size_t complete_utf8_prefix(const std::string & value, size_t limit) {
    size_t i = 0;
    size_t complete = 0;
    limit = std::min(limit, value.size());
    while (i < limit) {
        const unsigned char lead = static_cast<unsigned char>(value[i]);
        size_t width = 1;
        if ((lead & 0x80u) == 0) width = 1;
        else if ((lead & 0xE0u) == 0xC0u) width = 2;
        else if ((lead & 0xF0u) == 0xE0u) width = 3;
        else if ((lead & 0xF8u) == 0xF0u) width = 4;
        else break;
        if (i + width > limit) break;
        bool valid = true;
        for (size_t j = 1; j < width; ++j) {
            if ((static_cast<unsigned char>(value[i + j]) & 0xC0u) != 0x80u) {
                valid = false;
                break;
            }
        }
        if (!valid) break;
        i += width;
        complete = i;
    }
    return complete;
}

class TextEmitter {
public:
    using Emit = std::function<bool(const std::string &)>;

    TextEmitter(std::vector<std::string> stops, Emit emit)
        : stops_(std::move(stops)), emit_(std::move(emit)) {}

    bool append(const std::string & piece) {
        pending_ += piece;
        size_t stop_pos = std::string::npos;
        for (const auto & stop : stops_) {
            const size_t pos = pending_.find(stop);
            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) stop_pos = pos;
        }
        if (stop_pos != std::string::npos) {
            if (!emit_prefix(stop_pos)) return false;
            pending_.clear();
            stopped_ = true;
            return false;
        }

        size_t retain = 0;
        for (const auto & stop : stops_) {
            const size_t max_prefix = std::min(stop.size() - 1, pending_.size());
            for (size_t n = 1; n <= max_prefix; ++n) {
                if (pending_.compare(pending_.size() - n, n, stop, 0, n) == 0) retain = std::max(retain, n);
            }
        }
        return emit_prefix(pending_.size() - retain);
    }

    bool flush() {
        const size_t complete = complete_utf8_prefix(pending_, pending_.size());
        if (complete > 0 && !emit_bytes(complete)) return false;
        if (!pending_.empty()) {
            pending_.clear();
            return emit_("\xEF\xBF\xBD");
        }
        return true;
    }

    bool stopped() const { return stopped_; }

private:
    bool emit_prefix(size_t limit) {
        const size_t complete = complete_utf8_prefix(pending_, limit);
        return complete == 0 || emit_bytes(complete);
    }

    bool emit_bytes(size_t count) {
        std::string text = pending_.substr(0, count);
        pending_.erase(0, count);
        return text.empty() || emit_(text);
    }

    std::vector<std::string> stops_;
    Emit emit_;
    std::string pending_;
    bool stopped_ = false;
};

class ChatOutputParser {
public:
    using Emit = std::function<bool(const common_chat_msg_diff &)>;

    ChatOutputParser(
            const common_chat_parser_params & params, Emit emit)
        : params_(params), emit_(std::move(emit)) {
        if (params_.is_continuation && !params_.echo) {
            message_ = common_chat_parse("", true, params_);
        }
    }

    bool append(const std::string & piece) {
        generated_ += piece;
        return update(true);
    }

    bool flush() { return update(false); }

    const common_chat_msg & message() const {
        return message_;
    }

private:
    bool update(bool partial) {
        common_chat_msg parsed =
            common_chat_parse(generated_, partial, params_);
        if (parsed.empty()) return true;
        // A partial PEG parse may already know the tool name while its JSON
        // arguments are still incomplete. Do not expose that half-call to an
        // OpenAI-compatible client; emit the complete call on flush instead.
        if (partial) {
            parsed.tool_calls.clear();
        }
        parsed.set_tool_call_ids(
            tool_call_ids_,
            []() { return request_id("call_"); });
        const auto diffs =
            common_chat_msg_diff::compute_diffs(message_, parsed);
        message_ = std::move(parsed);
        for (const auto & diff : diffs) {
            if (!emit_(diff)) return false;
        }
        return true;
    }

    common_chat_parser_params params_;
    Emit emit_;
    std::string generated_;
    common_chat_msg message_;
    std::vector<std::string> tool_call_ids_;
};

struct RequestMetrics {
    using Clock = std::chrono::steady_clock;

    Clock::time_point started = Clock::now();
    Clock::time_point first_token;
    size_t prefill_tokens = 0;
    double prefill_ms = 0.0;
    double multimodal_ms = 0.0;
    double model_prefill_ms = 0.0;
    bool saw_token = false;
    bool saw_prefill = false;

    void mark_prefill(const MfqPrefillTiming & timing) {
        prefill_tokens = timing.prompt_tokens;
        prefill_ms = timing.llm_ms;
        multimodal_ms = timing.multimodal_ms;
        model_prefill_ms = timing.model_ms;
        saw_prefill = timing.llm_ms > 0.0 || timing.model_ms > 0.0;
    }

    void mark_token() {
        if (saw_token) return;
        first_token = Clock::now();
        saw_token = true;
    }
};

struct CompletionResult {
    std::string text;
    std::string reasoning_text;
    std::vector<common_chat_tool_call> tool_calls;
    std::string finish_reason = "length";
    int32_t completion_tokens = 0;
    bool client_connected = true;
    bool cancelled = false;
};

struct RequestMetricValues {
    size_t prefill_tokens = 0;
    double generation_ms = 0.0;
    double ttft_ms = 0.0;
    double prefill_ms = 0.0;
    double prefill_tps = 0.0;
    double multimodal_ms = 0.0;
    double model_prefill_ms = 0.0;
    double decode_ms = 0.0;
    double generation_tps = 0.0;
    double decode_tps = 0.0;
};

static json request_metric_values_json(
        const RequestMetricValues & values,
        const MfqSamplingParams & sampling) {
    return {
        {"prefill_tokens", values.prefill_tokens},
        {"ttft_ms", values.ttft_ms},
        {"prefill_ms", values.prefill_ms},
        {"prefill_tps", values.prefill_tps},
        {"multimodal_ms", values.multimodal_ms},
        {"model_prefill_ms", values.model_prefill_ms},
        {"decode_ms", values.decode_ms},
        {"decode_tps", values.decode_tps},
        {"generation_ms", values.generation_ms},
        {"generation_tps", values.generation_tps},
        {"sampling", {
            {"max_tokens", sampling.max_tokens},
            {"temperature", sampling.temperature},
            {"top_k", sampling.top_k},
            {"top_p", sampling.top_p},
            {"presence_penalty", sampling.presence_penalty},
            {"frequency_penalty", sampling.frequency_penalty},
            {"repetition_penalty", sampling.repetition_penalty},
            {"seed", sampling.seed},
            {"enable_thinking", sampling.enable_thinking},
        }},
    };
}

static RequestMetricValues request_metric_values(
        const CompletionResult & result, const RequestMetrics & metrics) {
    const auto finished = RequestMetrics::Clock::now();
    RequestMetricValues values;
    values.prefill_tokens = metrics.saw_prefill
        ? metrics.prefill_tokens
        : 0;
    values.generation_ms =
        std::chrono::duration<double, std::milli>(finished - metrics.started).count();
    values.ttft_ms = metrics.saw_token
        ? std::chrono::duration<double, std::milli>(
              metrics.first_token - metrics.started).count()
        : values.generation_ms;
    values.prefill_ms = metrics.saw_prefill ? metrics.prefill_ms : 0.0;
    values.multimodal_ms = metrics.saw_prefill
        ? metrics.multimodal_ms
        : 0.0;
    values.model_prefill_ms = metrics.saw_prefill
        ? metrics.model_prefill_ms
        : 0.0;
    values.prefill_tps = metrics.saw_prefill && metrics.prefill_ms > 0.0
        ? 1000.0 * metrics.prefill_tokens / metrics.prefill_ms
        : 0.0;
    values.decode_ms = metrics.saw_token
        ? std::chrono::duration<double, std::milli>(
              finished - metrics.first_token).count()
        : 0.0;
    values.generation_tps = values.generation_ms > 0.0
        ? 1000.0 * result.completion_tokens / values.generation_ms
        : 0.0;
    values.decode_tps =
        result.completion_tokens > 1 && values.decode_ms > 0.0
        ? 1000.0 * (result.completion_tokens - 1) / values.decode_ms
        : 0.0;
    return values;
}

static void log_request_metrics(const std::string & id, bool chat, bool stream,
                                size_t prompt_tokens, const MfqSamplingParams & sampling,
                                const CompletionResult & result,
                                const RequestMetricValues & values) {
    const char * enabled = std::getenv("MFQ_SERVER_REQUEST_METRICS");
    if (enabled != nullptr && std::atoi(enabled) == 0) return;

    const bool penalties = sampling.presence_penalty != 0.0 ||
        sampling.frequency_penalty != 0.0 || sampling.repetition_penalty != 1.0;

    std::ostringstream line;
    line << std::fixed << std::setprecision(3)
         << "request_metrics"
         << " id=" << id
         << " endpoint=" << (chat ? "chat" : "completion")
         << " stream=" << (stream ? 1 : 0)
         << " prompt_tokens=" << prompt_tokens
         << " prefill_tokens=" << values.prefill_tokens
         << " completion_tokens=" << result.completion_tokens
         << " max_tokens=" << sampling.max_tokens
         << " ttft_ms=" << values.ttft_ms
         << " prefill_ms=" << values.prefill_ms
         << " prefill_tps=" << values.prefill_tps
         << " multimodal_ms=" << values.multimodal_ms
         << " model_prefill_ms=" << values.model_prefill_ms
         << " decode_ms=" << values.decode_ms
         << " decode_tps=" << values.decode_tps
         << " generation_ms=" << values.generation_ms
         << " generation_tps=" << values.generation_tps
         << " temperature=" << sampling.temperature
         << " top_k=" << sampling.top_k
         << " top_p=" << sampling.top_p
         << " presence_penalty=" << sampling.presence_penalty
         << " frequency_penalty=" << sampling.frequency_penalty
         << " repetition_penalty=" << sampling.repetition_penalty
         << " seed=" << sampling.seed
         << " penalties=" << (penalties ? 1 : 0)
         << " finish_reason=" << result.finish_reason
         << " client_connected=" << (result.client_connected ? 1 : 0);
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cout << line.str() << std::endl;
}

class ServerMetrics {
public:
    ServerMetrics()
        : started_steady_(std::chrono::steady_clock::now()),
          started_unix_(unix_time_seconds()) {}

    void begin() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++total_requests_;
        ++active_requests_;
    }

    void complete(
            const std::string & id, bool chat, bool stream,
            size_t prompt_tokens, const CompletionResult & result,
            const RequestMetricValues & values) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_requests_ > 0) --active_requests_;
        total_prompt_tokens_ += prompt_tokens;
        total_completion_tokens_ +=
            static_cast<uint64_t>(std::max<int32_t>(result.completion_tokens, 0));
        last_request_ = {
            {"id", id},
            {"endpoint", chat ? "chat" : "completion"},
            {"stream", stream},
            {"prompt_tokens", prompt_tokens},
            {"prefill_tokens", values.prefill_tokens},
            {"completion_tokens", result.completion_tokens},
            {"ttft_ms", values.ttft_ms},
            {"prefill_ms", values.prefill_ms},
            {"prefill_tps", values.prefill_tps},
            {"multimodal_ms", values.multimodal_ms},
            {"model_prefill_ms", values.model_prefill_ms},
            {"decode_ms", values.decode_ms},
            {"decode_tps", values.decode_tps},
            {"generation_ms", values.generation_ms},
            {"generation_tps", values.generation_tps},
            {"finish_reason", result.finish_reason},
            {"client_connected", result.client_connected},
            {"completed_at", unix_time_seconds()},
        };
    }

    void fail() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_requests_ > 0) --active_requests_;
        ++failed_requests_;
    }

    uint64_t active_requests() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_requests_;
    }

    json snapshot(
            const MfqServerConfig & config,
            int64_t max_context,
            bool reloading) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const double uptime_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started_steady_).count();
        return {
            {"status", "ok"},
            {"model", config.model_name},
            {"model_type", config.model_type},
            {"max_context", max_context},
            {"context_capacity", config.context_capacity},
            {"reloading", reloading},
            {"vocab_size", config.vocab_size},
            {"started_at", started_unix_},
            {"uptime_seconds", uptime_seconds},
            {"active_requests", active_requests_},
            {"total_requests", total_requests_},
            {"failed_requests", failed_requests_},
            {"total_prompt_tokens", total_prompt_tokens_},
            {"total_completion_tokens", total_completion_tokens_},
            {"last_request", last_request_},
        };
    }

private:
    mutable std::mutex mutex_;
    std::chrono::steady_clock::time_point started_steady_;
    int64_t started_unix_ = 0;
    uint64_t active_requests_ = 0;
    uint64_t total_requests_ = 0;
    uint64_t failed_requests_ = 0;
    uint64_t total_prompt_tokens_ = 0;
    uint64_t total_completion_tokens_ = 0;
    json last_request_ = nullptr;
};

class ActiveRequest {
public:
    explicit ActiveRequest(ServerMetrics & metrics)
        : metrics_(metrics) {
        metrics_.begin();
    }

    ~ActiveRequest() {
        if (!completed_) metrics_.fail();
    }

    void complete(
            const std::string & id, bool chat, bool stream,
            size_t prompt_tokens, const CompletionResult & result,
            const RequestMetricValues & values) {
        metrics_.complete(
            id, chat, stream, prompt_tokens, result, values);
        completed_ = true;
    }

private:
    ServerMetrics & metrics_;
    bool completed_ = false;
};

static CompletionResult generate_text(const RequestWork & work, const MfqTokenizer & tokenizer,
                                      const MfqGenerateFn & generate,
                                      const MfqMultimodalGenerateFn & multimodal_generate,
                                      const std::shared_ptr<std::atomic<bool>> & cancel_requested,
                                      const std::function<bool(const common_chat_msg_diff &)> & emit,
                                      RequestMetrics * metrics,
                                      bool defer_token_parsing) {
    CompletionResult result;
    auto emit_parsed = [&](const common_chat_msg_diff & diff) {
        result.client_connected = emit(diff);
        return result.client_connected;
    };
    std::unique_ptr<ChatOutputParser> chat_parser;
    if (work.chat) {
        chat_parser = std::make_unique<ChatOutputParser>(
            work.chat_parser, emit_parsed);
    }
    TextEmitter emitter(work.stops, [&](const std::string & text) {
        if (chat_parser) {
            return chat_parser->append(text);
        }
        result.text += text;
        common_chat_msg_diff diff;
        diff.content_delta = text;
        return emit_parsed(diff);
    });

    const bool defer_tokens = defer_token_parsing && work.stops.empty();
    std::vector<int64_t> deferred_tokens;
    if (defer_tokens) {
        deferred_tokens.reserve(
            static_cast<std::size_t>(std::max(work.sampling.max_tokens, 0)));
    }
    const auto on_token = [&](int64_t token) {
            if (cancel_requested &&
                cancel_requested->load(std::memory_order_acquire)) {
                result.cancelled = true;
                result.finish_reason = "cancelled";
                return false;
            }
            if (metrics != nullptr) metrics->mark_token();
            if (tokenizer.is_eog(token)) {
                result.finish_reason = "stop";
                return false;
            }
            if (defer_tokens) {
                deferred_tokens.push_back(token);
                return true;
            }
            const bool preserve =
                work.preserved_tokens.find(token) !=
                work.preserved_tokens.end();
            if (!emitter.append(tokenizer.piece(token, preserve))) {
                if (emitter.stopped()) result.finish_reason = "stop";
                return false;
            }
            return true;
        };
    const auto on_prefill = [&](const MfqPrefillTiming & timing) {
            if (metrics != nullptr) metrics->mark_prefill(timing);
        };
    result.completion_tokens = work.vision
        ? multimodal_generate(
              work.prompt, *work.vision, work.sampling,
              on_token, on_prefill, work.token_constraint)
        : generate(
              work.prompt, work.sampling, on_token, on_prefill,
              work.cache_plan, work.token_constraint);
    if (cancel_requested &&
        cancel_requested->load(std::memory_order_acquire)) {
        result.cancelled = true;
        result.finish_reason = "cancelled";
    }
    if (defer_tokens) {
        std::string deferred_text;
        deferred_text.reserve(deferred_tokens.size() * 8);
        for (const auto token : deferred_tokens) {
            const bool preserve =
                work.preserved_tokens.find(token) !=
                work.preserved_tokens.end();
            deferred_text += tokenizer.piece(token, preserve);
        }
        if (!deferred_text.empty() && !emitter.append(deferred_text)) {
            if (emitter.stopped()) result.finish_reason = "stop";
        }
    }
    if (result.client_connected && !emitter.stopped()) emitter.flush();
    if (result.client_connected && chat_parser) {
        chat_parser->flush();
        const auto & message = chat_parser->message();
        result.text = message.content;
        result.reasoning_text = message.reasoning_content;
        result.tool_calls = message.tool_calls;
    }
    if (emitter.stopped() && !result.cancelled) result.finish_reason = "stop";
    if (!result.cancelled && !result.tool_calls.empty()) {
        result.finish_reason = "tool_calls";
    }
    return result;
}

static json usage_json(size_t prompt_tokens, int32_t completion_tokens) {
    return {
        {"prompt_tokens", prompt_tokens},
        {"completion_tokens", completion_tokens},
        {"total_tokens", prompt_tokens + static_cast<size_t>(completion_tokens)},
    };
}

static json chat_chunk(const std::string & id, int64_t created, const std::string & model,
                       json delta, json finish_reason, json usage = nullptr) {
    json out = {
        {"id", id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model},
        {"choices", json::array({{
            {"index", 0},
            {"delta", std::move(delta)},
            {"logprobs", nullptr},
            {"finish_reason", std::move(finish_reason)},
        }})},
    };
    if (!usage.is_null()) out["usage"] = std::move(usage);
    return out;
}

static json completion_chunk(const std::string & id, int64_t created, const std::string & model,
                             const std::string & text, json finish_reason, json usage = nullptr) {
    json out = {
        {"id", id},
        {"object", "text_completion"},
        {"created", created},
        {"model", model},
        {"choices", json::array({{
            {"index", 0},
            {"text", text},
            {"logprobs", nullptr},
            {"finish_reason", std::move(finish_reason)},
        }})},
    };
    if (!usage.is_null()) out["usage"] = std::move(usage);
    return out;
}

static json chat_diff_json(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call = {{"index", diff.tool_call_index}};
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"] = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        if (!diff.tool_call_delta.name.empty() ||
            !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                function["arguments"] =
                    diff.tool_call_delta.arguments;
            }
            tool_call["function"] = std::move(function);
        }
        delta["tool_calls"] =
            json::array({std::move(tool_call)});
    }
    return delta;
}

static json chat_tool_calls_json(
        const std::vector<common_chat_tool_call> & tool_calls) {
    json out = json::array();
    for (const auto & tool_call : tool_calls) {
        out.push_back({
            {"id", tool_call.id},
            {"type", "function"},
            {"function", {
                {"name", tool_call.name},
                {"arguments", tool_call.arguments},
            }},
        });
    }
    return out;
}

static json stream_usage_chunk(const std::string & id, int64_t created, const std::string & model,
                               bool chat, json usage) {
    return {
        {"id", id},
        {"object", chat ? "chat.completion.chunk" : "text_completion"},
        {"created", created},
        {"model", model},
        {"choices", json::array()},
        {"usage", std::move(usage)},
    };
}

static bool write_sse(httplib::DataSink & sink, const json & value) {
    const std::string event = "data: " + value.dump() + "\n\n";
    return sink.write(event.data(), event.size());
}

static bool authorized(const httplib::Request & req, httplib::Response & res, const std::string & api_key) {
    if (api_key.empty()) return true;
    const std::string expected = "Bearer " + api_key;
    if (req.get_header_value("Authorization") == expected) return true;
    res.set_header("WWW-Authenticate", "Bearer");
    set_json(res, error_body("invalid API key", "authentication_error"), 401);
    return false;
}

static int base64_digit(unsigned char value) {
    if (value >= 'A' && value <= 'Z') return value - 'A';
    if (value >= 'a' && value <= 'z') return value - 'a' + 26;
    if (value >= '0' && value <= '9') return value - '0' + 52;
    if (value == '+') return 62;
    if (value == '/') return 63;
    return -1;
}

static std::vector<uint8_t> decode_base64(
        const std::string & encoded,
        const std::string & parameter = "audio_features") {
    std::string compact;
    compact.reserve(encoded.size());
    for (const unsigned char value : encoded) {
        if (value == ' ' || value == '\t' || value == '\r' ||
            value == '\n') {
            continue;
        }
        compact.push_back(static_cast<char>(value));
    }
    if (compact.empty() || compact.size() % 4 != 0) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " must be padded base64", parameter);
    }

    std::vector<uint8_t> output;
    output.reserve(compact.size() / 4 * 3);
    for (size_t offset = 0; offset < compact.size(); offset += 4) {
        const bool pad2 = compact[offset + 2] == '=';
        const bool pad3 = compact[offset + 3] == '=';
        if (pad2 && !pad3) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " has invalid base64 padding", parameter);
        }
        if ((pad2 || pad3) && offset + 4 != compact.size()) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " has interior base64 padding", parameter);
        }
        const int a = base64_digit(compact[offset]);
        const int b = base64_digit(compact[offset + 1]);
        const int c = pad2 ? 0 : base64_digit(compact[offset + 2]);
        const int d = pad3 ? 0 : base64_digit(compact[offset + 3]);
        if (a < 0 || b < 0 || c < 0 || d < 0) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " contains invalid base64 data", parameter);
        }
        const uint32_t merged =
            (static_cast<uint32_t>(a) << 18) |
            (static_cast<uint32_t>(b) << 12) |
            (static_cast<uint32_t>(c) << 6) |
            static_cast<uint32_t>(d);
        output.push_back(static_cast<uint8_t>(merged >> 16));
        if (!pad2) output.push_back(static_cast<uint8_t>(merged >> 8));
        if (!pad3) output.push_back(static_cast<uint8_t>(merged));
    }
    return output;
}

static size_t tensor_element_count(
        const std::vector<int64_t> & shape,
        const std::string & parameter) {
    if (shape.empty() || shape.size() > 4) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " shape must have between 1 and 4 dimensions",
            parameter);
    }
    size_t count = 1;
    for (const int64_t dimension : shape) {
        if (dimension <= 0 ||
            static_cast<uint64_t>(dimension) >
                static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            count > std::numeric_limits<size_t>::max() /
                static_cast<size_t>(dimension)) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " has an invalid tensor shape", parameter);
        }
        count *= static_cast<size_t>(dimension);
    }
    return count;
}

static uint8_t hex_digit_value(char value, const std::string & parameter) {
    if (value >= '0' && value <= '9') {
        return static_cast<uint8_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
        return static_cast<uint8_t>(value - 'a' + 10);
    }
    if (value >= 'A' && value <= 'F') {
        return static_cast<uint8_t>(value - 'A' + 10);
    }
    throw ApiError(
        400, "invalid_request_error",
        parameter + " contains an invalid hexadecimal token", parameter);
}

static std::array<uint8_t, 32> decode_file_token(
        const std::string & encoded,
        const std::string & parameter) {
    if (encoded.size() != 64) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " must contain a 32-byte token", parameter);
    }
    std::array<uint8_t, 32> result{};
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<uint8_t>(
            (hex_digit_value(encoded[2 * index], parameter) << 4) |
            hex_digit_value(encoded[2 * index + 1], parameter));
    }
    return result;
}

class TensorFileReader final {
public:
    explicit TensorFileReader(const json & tensors) {
        const std::string parameter = "mfq_multimodal.binary_file";
        if (!tensors.contains("binary_file") ||
            !tensors["binary_file"].is_object()) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " must be an object", parameter);
        }
        const auto & spec = tensors["binary_file"];
        if (!spec.contains("path") || !spec["path"].is_string() ||
            !spec.contains("token") || !spec["token"].is_string() ||
            !spec.contains("size") || !spec["size"].is_number_integer()) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " must include path, token, and size", parameter);
        }
        const auto declared_size = spec["size"].get<int64_t>();
        if (declared_size < 64 || declared_size > 1024LL * 1024LL * 1024LL) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " has an invalid size", parameter);
        }
        size_ = static_cast<size_t>(declared_size);
        const std::filesystem::path path(spec["path"].get<std::string>());
        const std::string filename = path.filename().string();
        if (!path.is_absolute() ||
            filename.rfind("mfq-multimodal-", 0) != 0 ||
            path.extension() != ".bin") {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " path is not an MFQ temporary tensor file", parameter);
        }
        const auto token = decode_file_token(
            spec["token"].get<std::string>(), parameter + ".token");
#ifdef _WIN32
        (void) token;
        throw ApiError(
            400, "invalid_request_error",
            parameter + " is unavailable on this platform", parameter);
#else
        int flags = O_RDONLY;
#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
        flags |= O_NOFOLLOW;
#endif
        descriptor_ = ::open(path.c_str(), flags);
        if (descriptor_ < 0) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " cannot be opened", parameter);
        }
        struct stat metadata {};
        if (::fstat(descriptor_, &metadata) != 0 ||
            !S_ISREG(metadata.st_mode) || metadata.st_nlink != 1 ||
            metadata.st_uid != ::geteuid() ||
            (metadata.st_mode & 0077) != 0 ||
            metadata.st_size < 0 ||
            static_cast<uint64_t>(metadata.st_size) != size_) {
            close_descriptor();
            throw ApiError(
                400, "invalid_request_error",
                parameter + " failed ownership, mode, or size validation", parameter);
        }
        std::array<uint8_t, 64> header{};
        read_exact(0, header.data(), header.size(), parameter);
        constexpr std::array<uint8_t, 8> magic{
            'M', 'F', 'Q', 'M', 'M', '0', '1', 0};
        if (!std::equal(magic.begin(), magic.end(), header.begin()) ||
            !std::equal(token.begin(), token.end(), header.begin() + 8)) {
            close_descriptor();
            throw ApiError(
                400, "invalid_request_error",
                parameter + " header or token does not match", parameter);
        }
#endif
    }

    ~TensorFileReader() {
        close_descriptor();
    }

    TensorFileReader(const TensorFileReader &) = delete;
    TensorFileReader & operator=(const TensorFileReader &) = delete;

    void read(
            size_t offset,
            void * destination,
            size_t length,
            const std::string & parameter) const {
        if (offset < 64 || offset % 64 != 0 ||
            length > size_ || offset > size_ - length) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " byte range is outside the tensor file", parameter);
        }
        read_exact(offset, destination, length, parameter);
    }

private:
    void close_descriptor() noexcept {
#ifndef _WIN32
        if (descriptor_ >= 0) {
            ::close(descriptor_);
            descriptor_ = -1;
        }
#endif
    }

    void read_exact(
            size_t offset,
            void * destination,
            size_t length,
            const std::string & parameter) const {
#ifdef _WIN32
        (void) offset;
        (void) destination;
        (void) length;
        throw ApiError(
            400, "invalid_request_error",
            parameter + " is unavailable on this platform", parameter);
#else
        auto * output = static_cast<uint8_t *>(destination);
        size_t completed = 0;
        while (completed < length) {
            const ssize_t count = ::pread(
                descriptor_, output + completed, length - completed,
                static_cast<off_t>(offset + completed));
            if (count > 0) {
                completed += static_cast<size_t>(count);
                continue;
            }
            if (count < 0 && errno == EINTR) continue;
            throw ApiError(
                400, "invalid_request_error",
                parameter + " could not be read completely", parameter);
        }
#endif
    }

    size_t size_ = 0;
#ifndef _WIN32
    mutable int descriptor_ = -1;
#endif
};

template <typename Value>
static std::pair<std::vector<Value>, std::vector<int64_t>> decode_tensor(
        const json & tensors,
        const std::string & name,
        const std::string & expected_dtype,
        const TensorFileReader * file_reader) {
    const std::string parameter = "mfq_multimodal." + name;
    if (!tensors.contains(name) || !tensors[name].is_object()) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " must be a tensor object", parameter);
    }
    const auto & tensor = tensors[name];
    if (!tensor.contains("dtype") || !tensor["dtype"].is_string() ||
        tensor["dtype"].get<std::string>() != expected_dtype) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " must use dtype " + expected_dtype, parameter);
    }
    if (!tensor.contains("shape") || !tensor["shape"].is_array()) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " must include an integer shape", parameter);
    }
    std::vector<int64_t> shape;
    shape.reserve(tensor["shape"].size());
    for (const auto & dimension : tensor["shape"]) {
        if (!dimension.is_number_integer()) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " shape must contain integers", parameter);
        }
        shape.push_back(dimension.get<int64_t>());
    }
    const size_t count = tensor_element_count(shape, parameter);
    if (count > std::numeric_limits<size_t>::max() / sizeof(Value)) {
        throw ApiError(
            400, "invalid_request_error",
            parameter + " byte length does not match its shape", parameter);
    }
    const size_t expected_bytes = count * sizeof(Value);
    std::vector<Value> values(count);
    if (tensor.contains("data_base64") &&
        tensor["data_base64"].is_string()) {
        const auto bytes = decode_base64(
            tensor["data_base64"].get<std::string>(), parameter);
        if (bytes.size() != expected_bytes) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " byte length does not match its shape", parameter);
        }
        std::memcpy(values.data(), bytes.data(), bytes.size());
    } else {
        if (file_reader == nullptr ||
            !tensor.contains("data_offset") ||
            !tensor["data_offset"].is_number_integer() ||
            !tensor.contains("data_length") ||
            !tensor["data_length"].is_number_integer()) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " must include data_base64 or a binary file range",
                parameter);
        }
        const int64_t raw_offset = tensor["data_offset"].get<int64_t>();
        const int64_t raw_length = tensor["data_length"].get<int64_t>();
        if (raw_offset < 0 || raw_length < 0 ||
            static_cast<uint64_t>(raw_length) != expected_bytes) {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " byte length does not match its shape", parameter);
        }
        file_reader->read(
            static_cast<size_t>(raw_offset), values.data(), expected_bytes,
            parameter);
    }
    return {std::move(values), std::move(shape)};
}

static int64_t single_special_token(
        const MfqTokenizer & tokenizer,
        const std::string & marker) {
    const auto tokens = tokenizer.tokenize(marker, true, false);
    if (tokens.size() != 1) {
        throw ApiError(
            500, "server_error",
            "MiniCPM-o tokenizer does not encode " + marker +
                " as one special token");
    }
    return tokens.front();
}

static MfqVisionInput parse_mfq_vision(
        const json & value,
        const std::vector<int64_t> & prompt,
        const MfqTokenizer & tokenizer) {
    if (!value.is_object()) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal must be an object", "mfq_multimodal");
    }
    if (!value.contains("version") || !value["version"].is_number_integer() ||
        value["version"].get<int>() != 1) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal.version must be 1", "mfq_multimodal.version");
    }

    MfqVisionInput result;
    std::unique_ptr<TensorFileReader> file_reader;
    if (value.contains("binary_file")) {
        file_reader = std::make_unique<TensorFileReader>(value);
    }
    const bool has_any_image_tensor =
        value.contains("pixel_values") || value.contains("patch_mask") ||
        value.contains("target_sizes");
    const bool has_all_image_tensors =
        value.contains("pixel_values") && value.contains("patch_mask") &&
        value.contains("target_sizes");
    if (has_any_image_tensor != has_all_image_tensors) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal image tensors must be provided together",
            "mfq_multimodal");
    }
    const bool has_any_audio_tensor =
        value.contains("audio_features") || value.contains("audio_lengths");
    const bool has_all_audio_tensors =
        value.contains("audio_features") && value.contains("audio_lengths");
    if (has_any_audio_tensor != has_all_audio_tensors) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal audio tensors must be provided together",
            "mfq_multimodal");
    }
    if (!has_all_image_tensors && !has_all_audio_tensors) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal contains no media tensors", "mfq_multimodal");
    }

    if (has_all_image_tensors) {
        auto pixels = decode_tensor<float>(
            value, "pixel_values", "float32", file_reader.get());
        result.pixel_values = std::move(pixels.first);
        result.pixel_shape = std::move(pixels.second);
        auto mask = decode_tensor<uint8_t>(
            value, "patch_mask", "uint8", file_reader.get());
        result.patch_mask = std::move(mask.first);
        result.patch_mask_shape = std::move(mask.second);
        auto sizes = decode_tensor<int32_t>(
            value, "target_sizes", "int32", file_reader.get());
        result.target_sizes = std::move(sizes.first);
        result.target_sizes_shape = std::move(sizes.second);

        if (result.pixel_shape.size() != 4 || result.pixel_shape[1] != 3 ||
            result.pixel_shape[2] != 14 ||
            result.patch_mask_shape.size() != 2 ||
            result.target_sizes_shape.size() != 2 ||
            result.target_sizes_shape[1] != 2 ||
            result.pixel_shape[0] != result.patch_mask_shape[0] ||
            result.pixel_shape[0] != result.target_sizes_shape[0] ||
            result.pixel_shape[3] % 14 != 0 ||
            result.patch_mask_shape[1] != result.pixel_shape[3] / 14) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_multimodal image tensor geometry is invalid",
                "mfq_multimodal");
        }
        const int64_t source_count = result.pixel_shape[0];
        if (source_count <= 0 || source_count > 576) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_multimodal contains an invalid number of image slices",
                "mfq_multimodal.pixel_values");
        }
        for (int64_t source = 0; source < source_count; ++source) {
            const int64_t rows = result.target_sizes[2 * source];
            const int64_t columns = result.target_sizes[2 * source + 1];
            if (rows <= 0 || columns <= 0 ||
                rows * columns > result.patch_mask_shape[1]) {
                throw ApiError(
                    400, "invalid_request_error",
                    "mfq_multimodal target size disagrees with pixel geometry",
                    "mfq_multimodal.target_sizes");
            }
            for (int64_t patch = 0; patch < result.patch_mask_shape[1]; ++patch) {
                const bool expected = patch < rows * columns;
                const uint8_t actual = result.patch_mask[
                    static_cast<size_t>(source * result.patch_mask_shape[1] + patch)];
                if (actual > 1 || static_cast<bool>(actual) != expected) {
                    throw ApiError(
                        400, "invalid_request_error",
                        "mfq_multimodal patch mask is not a contiguous active prefix",
                        "mfq_multimodal.patch_mask");
                }
            }
        }
        if (!std::all_of(
                result.pixel_values.begin(), result.pixel_values.end(),
                [](float item) { return std::isfinite(item); })) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_multimodal pixel_values contains a non-finite value",
                "mfq_multimodal.pixel_values");
        }

        const int64_t image_start = single_special_token(tokenizer, "<image>");
        const int64_t image_end = single_special_token(tokenizer, "</image>");
        const int64_t slice_start = single_special_token(tokenizer, "<slice>");
        const int64_t slice_end = single_special_token(tokenizer, "</slice>");
        for (size_t index = 0; index < prompt.size(); ++index) {
            const int64_t token = prompt[index];
            if (token != image_start && token != slice_start) continue;
            const int64_t end_token = token == image_start ? image_end : slice_end;
            const auto found = std::find(
                prompt.begin() + static_cast<std::ptrdiff_t>(index + 1),
                prompt.end(), end_token);
            if (found == prompt.end()) {
                throw ApiError(
                    400, "invalid_request_error",
                    "MiniCPM-o image placeholder is missing its end token",
                    "messages");
            }
            const size_t end = static_cast<size_t>(found - prompt.begin());
            if (end - index - 1 != 64) {
                throw ApiError(
                    400, "invalid_request_error",
                    "MiniCPM-o image placeholder must contain 64 query tokens",
                    "messages");
            }
            const int64_t source =
                static_cast<int64_t>(result.image_bounds.size() / 4);
            result.image_bounds.insert(
                result.image_bounds.end(),
                {0, source, static_cast<int64_t>(index + 1),
                 static_cast<int64_t>(end)});
            index = end;
        }
        if (result.image_bounds.size() / 4 !=
            static_cast<size_t>(source_count)) {
            throw ApiError(
                400, "invalid_request_error",
                "MiniCPM-o image placeholders do not match processed image slices",
                "messages");
        }
    }

    if (has_all_audio_tensors) {
        auto features = decode_tensor<float>(
            value, "audio_features", "float32", file_reader.get());
        result.audio_features = std::move(features.first);
        result.audio_features_shape = std::move(features.second);
        auto lengths = decode_tensor<int64_t>(
            value, "audio_lengths", "int64", file_reader.get());
        result.audio_lengths = std::move(lengths.first);
        const auto & length_shape = lengths.second;
        if (result.audio_features_shape.size() != 3 ||
            result.audio_features_shape[0] <= 0 ||
            result.audio_features_shape[0] > 128 ||
            result.audio_features_shape[1] != 80 ||
            result.audio_features_shape[2] < 9 ||
            result.audio_features_shape[2] > 3000 ||
            length_shape.size() != 1 ||
            length_shape[0] != result.audio_features_shape[0]) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_multimodal audio tensor geometry is invalid",
                "mfq_multimodal");
        }
        if (!std::all_of(
                result.audio_features.begin(), result.audio_features.end(),
                [](float item) { return std::isfinite(item); })) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_multimodal audio_features contains a non-finite value",
                "mfq_multimodal.audio_features");
        }
        std::vector<int64_t> pooled_lengths;
        pooled_lengths.reserve(result.audio_lengths.size());
        for (const int64_t length : result.audio_lengths) {
            if (length < 9 || length > result.audio_features_shape[2]) {
                throw ApiError(
                    400, "invalid_request_error",
                    "mfq_multimodal audio length is out of range",
                    "mfq_multimodal.audio_lengths");
            }
            const int64_t after_convolution = (length - 1) / 2 + 1;
            const int64_t pooled = (after_convolution - 5) / 5 + 1;
            if (pooled <= 0) {
                throw ApiError(
                    400, "invalid_request_error",
                    "mfq_multimodal audio input is too short",
                    "mfq_multimodal.audio_lengths");
            }
            pooled_lengths.push_back(pooled);
        }

        const int64_t audio_start =
            single_special_token(tokenizer, "<|audio_start|>");
        const int64_t audio_end =
            single_special_token(tokenizer, "<|audio_end|>");
        for (size_t index = 0; index < prompt.size(); ++index) {
            if (prompt[index] != audio_start) continue;
            const auto found = std::find(
                prompt.begin() + static_cast<std::ptrdiff_t>(index + 1),
                prompt.end(), audio_end);
            if (found == prompt.end()) {
                throw ApiError(
                    400, "invalid_request_error",
                    "MiniCPM-o audio placeholder is missing its end token",
                    "messages");
            }
            const size_t end = static_cast<size_t>(found - prompt.begin());
            const size_t source = result.audio_bounds.size() / 4;
            if (source >= pooled_lengths.size() ||
                static_cast<int64_t>(end - index - 1) != pooled_lengths[source]) {
                throw ApiError(
                    400, "invalid_request_error",
                    "MiniCPM-o audio placeholder does not match pooled audio length",
                    "messages");
            }
            result.audio_bounds.insert(
                result.audio_bounds.end(),
                {0, static_cast<int64_t>(source),
                 static_cast<int64_t>(index + 1), static_cast<int64_t>(end)});
            index = end;
        }
        if (result.audio_bounds.size() / 4 != pooled_lengths.size()) {
            throw ApiError(
                400, "invalid_request_error",
                "MiniCPM-o audio placeholders do not match processed audio chunks",
                "messages");
        }
    }
    return result;
}

static std::vector<float> decode_audio_features(
        const std::string & encoded,
        int32_t frames) {
    if (frames < 3 || frames > 4096) {
        throw ApiError(
            400, "invalid_request_error",
            "audio_frames must be in [3, 4096]", "audio_frames");
    }
    const auto bytes = decode_base64(encoded);
    const size_t expected = static_cast<size_t>(frames) * 80 * sizeof(float);
    if (bytes.size() != expected) {
        throw ApiError(
            400, "invalid_request_error",
            "audio_features byte length does not match audio_frames",
            "audio_features");
    }
    std::vector<float> features(static_cast<size_t>(frames) * 80);
    std::memcpy(features.data(), bytes.data(), bytes.size());
    if (!std::all_of(features.begin(), features.end(), [](float value) {
            return std::isfinite(value);
        })) {
        throw ApiError(
            400, "invalid_request_error",
            "audio_features contains a non-finite value",
            "audio_features");
    }
    return features;
}

static json parse_body(const httplib::Request & req) {
    try {
        return json::parse(req.body);
    } catch (const json::parse_error & error) {
        throw ApiError(400, "invalid_request_error", std::string("invalid JSON: ") + error.what());
    }
}

static void handle_api_error(httplib::Response & res, const ApiError & error) {
    set_json(res, error_body(error.what(), error.type, error.param), error.status);
}

} // namespace

MfqRuntimeProfile resolve_mfq_runtime_profile(
        const std::string & mfq_path,
        const std::string & model_architecture,
        const std::string & model_type,
        const std::string & model_name,
        const std::string & embedded_profile_json,
        const std::string & model_config_json,
        const std::string & explicit_profile_path) {
    std::vector<std::string> identities{
        model_architecture, model_type, model_name,
    };
    if (!model_config_json.empty()) {
        const json config = json::parse(model_config_json);
        if (!config.is_object()) {
            throw std::runtime_error("embedded model config must be a JSON object");
        }
        for (const char * key : {"model_type", "_name_or_path", "name_or_path"}) {
            if (config.contains(key) && config[key].is_string()) {
                identities.push_back(config[key].get<std::string>());
            }
        }
        if (config.contains("architectures") && config["architectures"].is_array()) {
            for (const auto & value : config["architectures"]) {
                if (value.is_string()) identities.push_back(value.get<std::string>());
            }
        }
    }

    // Low to high priority. Every merge is field-wise.
    MfqRuntimeProfile result = architecture_runtime_profile(identities);
    merge_runtime_profile(result, exact_model_runtime_profile(identities));
    if (!embedded_profile_json.empty()) {
        merge_runtime_profile(result, parse_runtime_profile(
            embedded_profile_json, "embedded-mfq"));
    }
    if (!mfq_path.empty()) {
        for (const auto & sidecar : profile_sidecar_paths(mfq_path)) {
            std::error_code error;
            if (std::filesystem::is_regular_file(sidecar, error) && !error) {
                merge_runtime_profile(result, parse_runtime_profile(
                    read_profile_file(sidecar), "sidecar:" + sidecar.filename().string()));
            }
        }
    }
    if (!explicit_profile_path.empty()) {
        const std::filesystem::path path(explicit_profile_path);
        merge_runtime_profile(result, parse_runtime_profile(
            read_profile_file(path), "server-explicit:" + path.filename().string()));
    }
    return result;
}

MfqTokenizerProbe probe_mfq_tokenizer(
        const std::vector<uint8_t> & tokenizer_gguf,
        const std::string & text,
        bool add_special,
        bool parse_special) {
    MfqTokenizer tokenizer(tokenizer_gguf);
    return {
        tokenizer.vocab_size(),
        tokenizer.bos_token(),
        tokenizer.eos_token(),
        tokenizer.eot_token(),
        tokenizer.pad_token(),
        tokenizer.add_bos(),
        tokenizer.add_eos(),
        tokenizer.chat_template(),
        tokenizer.tokenize(text, parse_special, add_special),
    };
}

MfqTokenizerProbe probe_mfq_tokenizer(
        const std::string & tokenizer_model,
        const std::string & text,
        bool add_special,
        bool parse_special) {
    MfqTokenizer tokenizer(tokenizer_model);
    return {
        tokenizer.vocab_size(),
        tokenizer.bos_token(),
        tokenizer.eos_token(),
        tokenizer.eot_token(),
        tokenizer.pad_token(),
        tokenizer.add_bos(),
        tokenizer.add_eos(),
        tokenizer.chat_template(),
        tokenizer.tokenize(text, parse_special, add_special),
    };
}

int run_mfq_server(
        const MfqServerConfig & config,
        const MfqGenerateFn & generate,
        const MfqReloadFn & reload,
        const MfqDuplexBackend & duplex,
        const MfqSessionControl & session_control,
        const MfqMultimodalGenerateFn & multimodal_generate,
        const MfqRuntimeMetricsFn & runtime_metrics) {
    if (config.tokenizer_gguf.empty() &&
        config.tokenizer_model.empty()) {
        throw std::runtime_error(
            "MFQ server requires an embedded or external tokenizer GGUF");
    }
    if (!config.tokenizer_gguf.empty() &&
        !config.tokenizer_model.empty()) {
        throw std::runtime_error(
            "MFQ server tokenizer source is ambiguous");
    }
    if (config.port < 1 || config.port > 65535) throw std::runtime_error("server port must be in [1, 65535]");

    std::unique_ptr<MfqTokenizer> tokenizer =
        config.tokenizer_gguf.empty()
        ? std::make_unique<MfqTokenizer>(
              config.tokenizer_model)
        : std::make_unique<MfqTokenizer>(
              config.tokenizer_gguf);
    if (config.vocab_size > 0 && tokenizer->vocab_size() != config.vocab_size) {
        throw std::runtime_error("tokenizer/model vocabulary mismatch: tokenizer=" +
                                 std::to_string(tokenizer->vocab_size()) + " model=" +
                                 std::to_string(config.vocab_size));
    }
    if (tokenizer->chat_template().empty()) {
        throw std::runtime_error(
            "MFQ server requires tokenizer.chat_template");
    }
    common_chat_templates_ptr chat_templates =
        common_chat_templates_init(tokenizer->context(), "");
    if (!chat_templates) {
        throw std::runtime_error(
            "cannot initialize tokenizer.chat_template");
    }
    const MfqSamplingParams sampling_defaults =
        default_sampling_params(config);
    const json duplex_sampling_defaults =
        duplex_profile_json(config.runtime_profile.duplex);
    const json tts_sampling_defaults =
        tts_profile_json(config.runtime_profile.tts);
    const std::string duplex_backend_name =
        duplex.name.empty() ? "native" : duplex.name;
    const json chat_template_capabilities =
        chat_template_capabilities_json(
            tokenizer->chat_template());
    const auto model_capability_profile =
        architecture_capability_profile(config.model_type);
    const json model_capabilities =
        model_capability_profile_json(model_capability_profile);

    httplib::Server server;
    ServerMetrics server_metrics;
    RequestCancellationRegistry request_cancellations;
    std::atomic<int64_t> active_context{config.max_context};
    std::atomic<bool> reloading{false};
    std::mutex reload_gate;
    std::mutex duplex_gate;
    std::string duplex_session_id;
    httplib::ws::WebSocket * duplex_socket = nullptr;
    bool duplex_backend_started = false;

    const auto duplex_is_active = [&]() {
        std::lock_guard<std::mutex> lock(duplex_gate);
        return !duplex_session_id.empty();
    };
    const auto stop_duplex_session = [&](const std::string & session_id,
                                         bool close_socket) {
        httplib::ws::WebSocket * socket = nullptr;
        bool stop_backend = false;
        {
            std::lock_guard<std::mutex> lock(duplex_gate);
            if (duplex_session_id.empty() ||
                duplex_session_id != session_id) {
                return false;
            }
            socket = duplex_socket;
            stop_backend = duplex_backend_started;
            duplex_session_id.clear();
            duplex_socket = nullptr;
            duplex_backend_started = false;
        }
        if (stop_backend) duplex.stop();
        if (close_socket && socket != nullptr && socket->is_open()) {
            socket->close(
                httplib::ws::CloseStatus::Normal, "session closed");
        }
        return true;
    };
    server.set_payload_max_length(
        (multimodal_generate ? 512ULL : 16ULL) * 1024ULL * 1024ULL);
    server.set_read_timeout(300, 0);
    server.set_write_timeout(300, 0);
    server.set_keep_alive_max_count(100);
    server.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
        {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"},
        {"X-Content-Type-Options", "nosniff"},
    });

    server.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
    });

    if (duplex) {
        server.WebSocket("/backend", [&](const httplib::Request & req,
                                          httplib::ws::WebSocket & ws) {
            const std::string expected = "Bearer " + config.api_key;
            if (!config.api_key.empty() &&
                req.get_header_value("Authorization") != expected) {
                ws.close(
                    httplib::ws::CloseStatus::PolicyViolation,
                    "authentication failed");
                return;
            }

            std::string owned_session;
            std::unordered_set<int64_t> session_controls;
            const auto send_event = [&](json event) {
                event["server_send_ts"] =
                    std::chrono::duration<double>(
                        std::chrono::system_clock::now()
                            .time_since_epoch()).count();
                return ws.send(event.dump());
            };
            try {
                std::string message;
                while (ws.is_open()) {
                    const auto read_result = ws.read(message);
                    if (read_result == httplib::ws::ReadResult::Fail) break;
                    if (read_result != httplib::ws::ReadResult::Text) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "duplex backend accepts JSON text frames only");
                    }
                    json body;
                    try {
                        body = json::parse(message);
                    } catch (const json::parse_error & error) {
                        throw ApiError(
                            400, "invalid_request_error",
                            std::string("invalid duplex JSON: ") +
                                error.what());
                    }
                    if (!body.is_object() || !body.contains("type") ||
                        !body["type"].is_string()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "duplex message requires a string type");
                    }
                    const std::string type = body["type"].get<std::string>();

                    if (type == "session.init") {
                        if (!owned_session.empty()) {
                            throw ApiError(
                                409, "conflict",
                                "duplex session is already initialized");
                        }
                        const json payload = body.value(
                            "payload", json::object());
                        if (!payload.is_object()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "session.init payload must be an object");
                        }
                        const std::string mode = payload.value(
                            "mode", std::string("full_duplex"));
                        if (mode != "full_duplex") {
                            throw ApiError(
                                400, "unsupported_operation",
                                duplex_backend_name +
                                    " duplex backend supports full_duplex only");
                        }

                        owned_session = request_id("sess-");
                        {
                            std::lock_guard<std::mutex> lock(duplex_gate);
                            if (!duplex_session_id.empty()) {
                                owned_session.clear();
                                throw ApiError(
                                    409, "conflict",
                                    "the " + duplex_backend_name +
                                        " worker already owns a duplex session");
                            }
                            duplex_session_id = owned_session;
                            duplex_socket = &ws;
                        }

                        MfqDuplexSessionParams parameters;
                        const std::string system_prompt = payload.value(
                            "system_prompt",
                            config.runtime_profile.duplex.system_prompt.value_or(
                                "Streaming Omni Conversation."));
                        const std::string rendered_prefix =
                            "<|im_start|>system\n" + system_prompt +
                            "\n<|audio_start|>";
                        parameters.system_prefix = tokenizer->tokenize(
                            rendered_prefix, true, false);
                        parameters.system_suffix = tokenizer->tokenize(
                            "<|audio_end|><|im_end|>", true, false);
                        if (payload.contains("reference_audio_features")) {
                            if (!payload["reference_audio_features"].is_string()) {
                                throw ApiError(
                                    400, "invalid_request_error",
                                    "reference_audio_features must be base64 float32 Mel data",
                                    "reference_audio_features");
                            }
                            parameters.reference_audio_frames =
                                static_cast<int32_t>(integer_field(
                                    payload, "reference_audio_frames", 0));
                            parameters.reference_audio_features =
                                decode_audio_features(
                                    payload["reference_audio_features"].get<std::string>(),
                                    parameters.reference_audio_frames);
                        }
                        parameters.special_ids = {
                            tokenizer->special_token_id("<unit>"),
                            tokenizer->special_token_id("</unit>"),
                            tokenizer->special_token_id("<image>"),
                            tokenizer->special_token_id("</image>"),
                            tokenizer->special_token_id("<slice>"),
                            tokenizer->special_token_id("</slice>"),
                            tokenizer->special_token_id("<|listen|>"),
                            tokenizer->special_token_id("<|speak|>"),
                            tokenizer->special_token_id("<|tts_bos|>"),
                            tokenizer->special_token_id("<|tts_eos|>"),
                            tokenizer->special_token_id("<|chunk_eos|>"),
                            tokenizer->special_token_id("<|chunk_tts_eos|>"),
                            tokenizer->special_token_id("<|turn_eos|>"),
                            tokenizer->special_token_id("<|tts_pad|>"),
                            151687,
                        };
                        parameters.forbidden_ids = {
                            tokenizer->special_token_id("<|tts_pad|>"),
                        };
                        session_controls = std::unordered_set<int64_t>(
                            parameters.special_ids.begin(),
                            parameters.special_ids.end());
                        const json generation = payload.value(
                            "config", json::object());
                        if (!generation.is_object()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "session config must be an object", "config");
                        }
                        const auto & duplex_defaults = config.runtime_profile.duplex;
                        parameters.greedy = generation.value(
                            "decode_mode", duplex_defaults.decode_mode.value_or("sampling")) ==
                            "greedy";
                        parameters.temperature = number_field(
                            generation, "temperature", duplex_defaults.temperature.value_or(0.7));
                        parameters.top_k = static_cast<int32_t>(integer_field(
                            generation, "top_k", duplex_defaults.top_k.value_or(100)));
                        parameters.top_p = number_field(
                            generation, "top_p", duplex_defaults.top_p.value_or(0.8));
                        parameters.listen_probability_scale = number_field(
                            generation, "listen_prob_scale", duplex_defaults.listen_prob_scale.value_or(1.0));
                        parameters.repetition_penalty = number_field(
                            generation, "text_repetition_penalty", duplex_defaults.text_repetition_penalty.value_or(1.05));
                        parameters.repetition_window =
                            static_cast<int32_t>(integer_field(
                                generation,
                                "text_repetition_window_size",
                                duplex_defaults.text_repetition_window_size.value_or(512)));
                        parameters.length_penalty = number_field(
                            generation, "length_penalty", duplex_defaults.length_penalty.value_or(1.0));
                        parameters.tts_temperature = number_field(
                            generation, "tts_temperature",
                            config.runtime_profile.tts.temperature.value_or(0.8));
                        parameters.tts_repetition_penalty = number_field(
                            generation, "tts_repetition_penalty",
                            config.runtime_profile.tts.repetition_penalty.value_or(1.05));
                        if (generation.contains("seed")) {
                            parameters.seed = static_cast<uint64_t>(
                                integer_field(generation, "seed", 0));
                        } else {
                            std::random_device random;
                            parameters.seed =
                                (static_cast<uint64_t>(random()) << 32) ^
                                static_cast<uint64_t>(random());
                        }

                        try {
                            duplex.start(parameters);
                            std::lock_guard<std::mutex> lock(duplex_gate);
                            if (duplex_session_id == owned_session) {
                                duplex_backend_started = true;
                            }
                        } catch (...) {
                            std::lock_guard<std::mutex> lock(duplex_gate);
                            if (duplex_session_id == owned_session) {
                                duplex_session_id.clear();
                                duplex_socket = nullptr;
                            }
                            owned_session.clear();
                            throw;
                        }
                        send_event({
                            {"type", "session.created"},
                            {"session_id", owned_session},
                            {"mode", "full_duplex"},
                            {"metrics", {{"backend", duplex_backend_name}}},
                        });
                        continue;
                    }

                    if (type != "input.append") {
                        throw ApiError(
                            400, "invalid_request_error",
                            "unsupported duplex message type: " + type);
                    }
                    if (owned_session.empty()) {
                        throw ApiError(
                            409, "conflict",
                            "session.init must precede input.append");
                    }
                    const json input = body.value("input", json::object());
                    if (!input.is_object()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "input must be an object", "input");
                    }
                    const bool has_audio = input.contains("audio_features");
                    const bool has_text = input.contains("text");
                    if (!has_audio && !has_text) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "input requires audio_features or text", "input");
                    }
                    MfqDuplexStepInput step;
                    if (has_audio) {
                        if (!input["audio_features"].is_string()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "input.audio_features must be base64 float32 Mel data",
                                "audio_features");
                        }
                        step.audio_frames = static_cast<int32_t>(integer_field(
                            input, "audio_frames", 0));
                        step.audio_features = decode_audio_features(
                            input["audio_features"].get<std::string>(),
                            step.audio_frames);
                        step.audio_prefix_extra_frames = integer_field(
                            input, "audio_prefix_extra_frames", 0);
                        step.audio_suffix_extra_frames = integer_field(
                            input, "audio_suffix_extra_frames", 0);
                    }
                    if (has_text) {
                        if (!input["text"].is_string() ||
                            input["text"].get_ref<const std::string&>().empty()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "input.text must be a non-empty string", "text");
                        }
                        step.text_tokens = tokenizer->tokenize(
                            input["text"].get<std::string>(), false, false);
                        if (step.text_tokens.empty()) {
                            throw ApiError(
                                400, "invalid_request_error",
                                "input.text produced no tokens", "text");
                        }
                    }
                    step.max_new_speak_tokens = static_cast<int32_t>(
                        integer_field(
                            input,
                            "max_new_speak_tokens",
                            config.runtime_profile.duplex
                                .max_new_speak_tokens_per_chunk
                                .value_or(20)));
                    if (input.contains("force_listen") &&
                        !input["force_listen"].is_boolean()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "force_listen must be boolean", "force_listen");
                    }
                    step.force_listen = input.value("force_listen", false);
                    if (input.contains("force_speak") &&
                        !input["force_speak"].is_boolean()) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "force_speak must be boolean", "force_speak");
                    }
                    step.force_speak = input.value("force_speak", false);
                    if (step.force_listen && step.force_speak) {
                        throw ApiError(
                            400, "invalid_request_error",
                            "force_listen and force_speak are mutually exclusive");
                    }

                    const auto result = duplex.step(step);
                    const std::string response_id = request_id("resp-");
                    json metrics = {
                        {"backend", duplex_backend_name},
                        {"wall_clock_ms", result.inference_ms},
                        {"kv_cache_length", result.language_cache_position},
                        {"audio_cache_length", result.audio_cache_position},
                        {"tts_cache_length", result.tts_cache_position},
                        {"audio_chunk_index", result.audio_chunk_index},
                    };

                    std::string text_delta;
                    for (const int64_t token : result.generated_tokens) {
                        if (session_controls.count(token) == 0) {
                            text_delta += tokenizer->piece(token, false);
                        }
                    }
                    if (!text_delta.empty()) {
                        send_event({
                            {"type", "response.output.delta"},
                            {"kind", "text"},
                            {"text", text_delta},
                            {"session_id", owned_session},
                            {"response_id", response_id},
                            {"end_of_turn", result.end_of_turn},
                            {"metrics", metrics},
                        });
                    }
                    if (!result.audio_tokens.empty() ||
                        (result.end_of_turn && !result.is_listen)) {
                        send_event({
                            {"type", "response.output.delta"},
                            {"kind", "audio_tokens"},
                            {"audio_tokens", result.audio_tokens},
                            {"session_id", owned_session},
                            {"response_id", response_id},
                            {"end_of_turn", result.end_of_turn},
                            {"force_flush", result.tts_force_flush},
                            {"metrics", metrics},
                        });
                    }
                    if (result.is_listen) {
                        send_event({
                            {"type", "response.output.delta"},
                            {"kind", "listen"},
                            {"session_id", owned_session},
                            {"response_id", response_id},
                            {"metrics", metrics},
                        });
                    }
                    send_event({
                        {"type", "response.step.done"},
                        {"session_id", owned_session},
                        {"response_id", response_id},
                        {"end_of_turn", result.end_of_turn},
                        {"metrics", metrics},
                    });
                }
            } catch (const std::exception & error) {
                send_event({
                    {"type", "session.closed"},
                    {"session_id", owned_session},
                    {"reason", "backend_error"},
                    {"diagnostic", {{"message", error.what()}}},
                });
                if (ws.is_open()) {
                    ws.close(
                        httplib::ws::CloseStatus::InternalError,
                        "duplex backend error");
                }
            }
            if (!owned_session.empty()) {
                stop_duplex_session(owned_session, false);
            }
        });

        server.Post(R"(/sessions/([A-Za-z0-9_-]+)/close)",
            [&](const httplib::Request & req, httplib::Response & res) {
                if (!authorized(req, res, config.api_key)) return;
                const std::string session_id = req.matches[1].str();
                if (!stop_duplex_session(session_id, true)) {
                    set_json(res, error_body(
                        "duplex session was not found", "not_found"), 404);
                    return;
                }
                set_json(res, {
                    {"ok", true},
                    {"session_id", session_id},
                    {"closed", true},
                });
            });
    }

    server.Get("/", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        set_json(res, {
            {"name", "MFQ C++ inference server"},
            {"model", config.model_name},
            {"endpoints", {
                "/v1/chat/completions", "/v1/completions", "/v1/models",
                "/health", "/api/status", "/api/reload", "/backend",
                "/api/runtime/cache/clear",
                "/api/runtime/sessions/fork",
                "/api/runtime/sessions/{id}",
                "/api/runtime/sessions/{id}/cancel",
            }},
        });
    });

    const auto add_runtime_metrics = [&](json & value) {
        if (!runtime_metrics) return;
        for (const auto & item : runtime_metrics()) {
            value[item.first] = item.second;
        }
    };
    const auto add_session_metrics = [&](json & value) {
        if (!session_control.metrics) return;
        for (const auto & item : session_control.metrics()) {
            value[item.first] = item.second;
        }
    };

    server.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        json health = {
            {"status", reloading.load() ? "loading" : "ok"},
            {"model", config.model_name},
            {"model_type", config.model_type},
            {"model_capabilities", model_capabilities},
            {"max_context", active_context.load()},
            {"duplex_available", static_cast<bool>(duplex)},
            {"duplex_active", duplex_is_active()},
            {"sampling_defaults", sampling_params_json(sampling_defaults)},
            {"duplex_sampling_defaults", duplex_sampling_defaults},
            {"tts_sampling_defaults", tts_sampling_defaults},
            {"runtime_profile_source", config.runtime_profile.source},
            {"chat_template_capabilities", chat_template_capabilities},
        };
        add_runtime_metrics(health);
        add_session_metrics(health);
        set_json(res, health);
    });

    server.Get("/api/status", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        json status = server_metrics.snapshot(
            config, active_context.load(), reloading.load());
        status["sampling_defaults"] = sampling_params_json(
            sampling_defaults);
        status["duplex_sampling_defaults"] = duplex_sampling_defaults;
        status["tts_sampling_defaults"] = tts_sampling_defaults;
        status["runtime_profile_source"] = config.runtime_profile.source;
        status["chat_template_capabilities"] =
            chat_template_capabilities;
        status["model_capabilities"] = model_capabilities;
        status["duplex_available"] = static_cast<bool>(duplex);
        status["duplex_active"] = duplex_is_active();
        add_runtime_metrics(status);
        add_session_metrics(status);
        set_json(res, status);
    });

    server.Post("/api/runtime/cache/clear", [&] (
            const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        if (!session_control.clear) {
            set_json(res, error_body(
                "this runtime does not expose a prefix cache",
                "unsupported_operation"), 501);
            return;
        }
        {
            std::lock_guard<std::mutex> gate(reload_gate);
            bool expected = false;
            if (!reloading.compare_exchange_strong(expected, true)) {
                set_json(res, error_body(
                    "a runtime control operation is already in progress",
                    "conflict"), 409);
                return;
            }
            if (server_metrics.active_requests() != 0 ||
                duplex_is_active()) {
                reloading.store(false);
                set_json(res, error_body(
                    "cannot clear the prefix cache while a generation or "
                    "duplex session is active",
                    "conflict"), 409);
                return;
            }
        }
        try {
            const size_t released = session_control.clear();
            json result = {
                {"status", "ok"},
                {"released_snapshots", released},
            };
            add_session_metrics(result);
            reloading.store(false);
            set_json(res, result);
        } catch (const std::exception & error) {
            reloading.store(false);
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    server.Post(
        R"(/api/runtime/sessions/([A-Za-z0-9._:-]{1,128})/cancel)",
        [&] (const httplib::Request & req, httplib::Response & res) {
            if (!authorized(req, res, config.api_key)) return;
            const std::string session_id = req.matches[1].str();
            set_json(res, {
                {"status", "ok"},
                {"cancelled", request_cancellations.cancel(session_id)},
            });
        });

    server.Post("/api/runtime/sessions/fork", [&] (
            const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        if (!session_control.fork) {
            set_json(res, error_body(
                "this runtime does not support session forks",
                "unsupported_operation"), 501);
            return;
        }
        try {
            const json body = parse_body(req);
            const auto read_session_id = [&](const char * field) {
                if (!body.contains(field) || !body[field].is_string()) {
                    throw ApiError(
                        400, "invalid_request_error",
                        std::string(field) + " must be a string", field);
                }
                auto session_id = body[field].get<std::string>();
                if (!valid_mfq_session_id(session_id)) {
                    throw ApiError(
                        400, "invalid_request_error",
                        std::string(field) +
                            " must contain 1 to 128 safe identifier bytes",
                        field);
                }
                return session_id;
            };
            const std::string source_session_id =
                read_session_id("source_session_id");
            const std::string target_session_id =
                read_session_id("target_session_id");
            if (source_session_id == target_session_id) {
                throw ApiError(
                    400, "invalid_request_error",
                    "source and target sessions must differ",
                    "target_session_id");
            }
            const size_t copied = session_control.fork(
                source_session_id, target_session_id);
            set_json(res, {
                {"status", "ok"},
                {"copied_snapshots", copied},
            });
        } catch (const ApiError & error) {
            handle_api_error(res, error);
        } catch (const std::exception & error) {
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    server.Delete(
        R"(/api/runtime/sessions/([A-Za-z0-9._:-]{1,128}))",
        [&] (const httplib::Request & req, httplib::Response & res) {
            if (!authorized(req, res, config.api_key)) return;
            if (!session_control.close) {
                set_json(res, error_body(
                    "this runtime does not support session close",
                    "unsupported_operation"), 501);
                return;
            }
            try {
                const std::string session_id = req.matches[1].str();
                const size_t released = session_control.close(session_id);
                set_json(res, {
                    {"status", "ok"},
                    {"released_snapshots", released},
                });
            } catch (const std::exception & error) {
                set_json(res, error_body(error.what(), "server_error"), 500);
            }
        });

    server.Post("/api/reload", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        if (!reload) {
            set_json(res, error_body(
                "this runtime does not support model reload",
                "unsupported_operation"), 501);
            return;
        }
        {
            std::lock_guard<std::mutex> gate(reload_gate);
            bool expected = false;
            if (!reloading.compare_exchange_strong(expected, true)) {
                set_json(res, error_body(
                    "model reload is already in progress", "conflict"), 409);
                return;
            }
            if (server_metrics.active_requests() != 0 ||
                duplex_is_active()) {
                reloading.store(false);
                set_json(res, error_body(
                    "cannot reload while a generation or duplex session is active",
                    "conflict"), 409);
                return;
            }
        }
        const auto finish_reload = [&] {
            reloading.store(false);
        };
        try {
            const json body = parse_body(req);
            const int64_t context_size = integer_field(
                body, "context_size", active_context.load());
            const int64_t capacity = config.context_capacity > 0
                ? config.context_capacity
                : config.max_context;
            if (context_size < 1 ||
                (capacity > 0 && context_size > capacity)) {
                throw ApiError(
                    400, "invalid_request_error",
                    "context_size must be within the model context capacity",
                    "context_size");
            }
            const int64_t loaded_context = reload(context_size);
            if (loaded_context < 1 ||
                (capacity > 0 && loaded_context > capacity)) {
                throw std::runtime_error(
                    "runtime reload returned an invalid context size");
            }
            active_context.store(loaded_context);
            finish_reload();
            set_json(res, {
                {"status", "ok"},
                {"model", config.model_name},
                {"max_context", loaded_context},
                {"context_capacity", capacity},
            });
        } catch (const ApiError & error) {
            finish_reload();
            handle_api_error(res, error);
        } catch (const std::exception & error) {
            finish_reload();
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    });

    server.Get("/v1/models", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        set_json(res, {
            {"object", "list"},
            {"data", json::array({{
                {"id", config.model_name},
                {"object", "model"},
                {"created", 0},
                {"owned_by", "mfq"},
            }})},
        });
    });

    auto completion_handler = [&](bool chat, const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        if (duplex_is_active()) {
            set_json(res, error_body(
                "the model is reserved by an active duplex session",
                "conflict"), 409);
            return;
        }
        if (reloading.load()) {
            set_json(res, error_body(
                "model reload is in progress", "service_unavailable"), 503);
            return;
        }
        try {
            const json body = parse_body(req);
            RequestWork work = parse_work(
                body, chat, *tokenizer, chat_templates.get(),
                active_context.load(), config.model_type,
                sampling_defaults);
            if (body.contains("mfq_multimodal")) {
                if (!chat) {
                    throw ApiError(
                        400, "invalid_request_error",
                        "mfq_multimodal is only valid for chat completions",
                        "mfq_multimodal");
                }
                if (!multimodal_generate) {
                    throw ApiError(
                        501, "unsupported_parameter",
                        "the loaded model has no native vision runtime",
                        "mfq_multimodal");
                }
                work.vision = parse_mfq_vision(
                    body["mfq_multimodal"], work.prompt, *tokenizer);
                work.cache_plan.stable_prefix_tokens = 0;
            }
            const std::string id = request_id(chat ? "chatcmpl-" : "cmpl-");
            const int64_t created = unix_time_seconds();
            std::shared_ptr<ActiveRequest> active_request;
            {
                std::lock_guard<std::mutex> gate(reload_gate);
                if (reloading.load()) {
                    set_json(res, error_body(
                        "model reload is in progress",
                        "service_unavailable"), 503);
                    return;
                }
                active_request =
                    std::make_shared<ActiveRequest>(server_metrics);
            }
            auto cancellation = request_cancellations.activate(
                work.cache_plan.session_id);

            if (!work.stream) {
                RequestMetrics metrics;
                CompletionResult result = generate_text(
                    work, *tokenizer, generate, multimodal_generate,
                    cancellation->flag,
                    [](const common_chat_msg_diff &) {
                        return true;
                    },
                    &metrics,
                    true);
                const RequestMetricValues metric_values =
                    request_metric_values(result, metrics);
                log_request_metrics(
                    id, chat, false, work.prompt.size(), work.sampling,
                    result, metric_values);
                active_request->complete(
                    id, chat, false, work.prompt.size(), result, metric_values);
                json response;
                if (chat) {
                    json message = {{"role", "assistant"}, {"content", result.text}};
                    if (!result.reasoning_text.empty()) {
                        message["reasoning_content"] = result.reasoning_text;
                    }
                    if (!result.tool_calls.empty()) {
                        message["tool_calls"] =
                            chat_tool_calls_json(result.tool_calls);
                    }
                    response = {
                        {"id", id},
                        {"object", "chat.completion"},
                        {"created", created},
                        {"model", config.model_name},
                        {"choices", json::array({{
                            {"index", 0},
                            {"message", std::move(message)},
                            {"logprobs", nullptr},
                            {"finish_reason", result.finish_reason},
                        }})},
                        {"usage", usage_json(work.prompt.size(), result.completion_tokens)},
                    };
                } else {
                    response = completion_chunk(id, created, config.model_name, result.text,
                                                result.finish_reason,
                                                usage_json(work.prompt.size(), result.completion_tokens));
                }
                response["mfq_metrics"] =
                    request_metric_values_json(metric_values, work.sampling);
                set_json(res, response);
                return;
            }

            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider(
                "text/event-stream; charset=utf-8",
                [work = std::move(work), id, created, &tokenizer, &generate,
                 &multimodal_generate, &config, active_request, cancellation,
                 chat]
                (size_t offset, httplib::DataSink & sink) mutable -> bool {
                    if (offset != 0) {
                        sink.done();
                        return false;
                    }
                    try {
                        if (chat && !write_sse(sink, chat_chunk(id, created, config.model_name,
                                                                {{"role", "assistant"}, {"content", ""}}, nullptr))) {
                            return false;
                        }
                        RequestMetrics metrics;
                        CompletionResult result = generate_text(
                            work, *tokenizer, generate,
                            multimodal_generate, cancellation->flag,
                            [&](const common_chat_msg_diff & diff) {
                                if (!chat) {
                                    if (diff.content_delta.empty()) {
                                        return true;
                                    }
                                    return write_sse(
                                        sink, completion_chunk(
                                            id, created, config.model_name,
                                            diff.content_delta, nullptr));
                                }
                                json delta = chat_diff_json(diff);
                                return delta.empty() ||
                                    write_sse(
                                        sink, chat_chunk(
                                            id, created, config.model_name,
                                            std::move(delta), nullptr));
                        }, &metrics, false);
                        const RequestMetricValues metric_values =
                            request_metric_values(result, metrics);
                        log_request_metrics(
                            id, chat, true, work.prompt.size(), work.sampling,
                            result, metric_values);
                        active_request->complete(
                            id, chat, true, work.prompt.size(), result,
                            metric_values);
                        if (!result.client_connected) return false;
                        const json final_chunk = chat
                            ? chat_chunk(id, created, config.model_name, json::object(), result.finish_reason)
                            : completion_chunk(id, created, config.model_name, "", result.finish_reason);
                        json enriched_final_chunk = final_chunk;
                        enriched_final_chunk["mfq_metrics"] =
                            request_metric_values_json(
                                metric_values, work.sampling);
                        if (!write_sse(sink, enriched_final_chunk)) return false;
                        if (work.include_usage) {
                            const json usage = usage_json(work.prompt.size(), result.completion_tokens);
                            const json usage_chunk = stream_usage_chunk(
                                id, created, config.model_name, chat, usage);
                            if (!write_sse(sink, usage_chunk)) return false;
                        }
                        static constexpr char done[] = "data: [DONE]\n\n";
                        if (!sink.write(done, sizeof(done) - 1)) return false;
                    } catch (const std::exception & error) {
                        write_sse(sink, error_body(error.what(), "server_error"));
                    }
                    sink.done();
                    return false;
                });
        } catch (const ApiError & error) {
            handle_api_error(res, error);
        } catch (const std::exception & error) {
            set_json(res, error_body(error.what(), "server_error"), 500);
        }
    };

    server.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        completion_handler(true, req, res);
    });
    server.Post("/v1/completions", [&](const httplib::Request & req, httplib::Response & res) {
        completion_handler(false, req, res);
    });

    server.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string message = "unhandled server exception";
        try {
            if (ep) std::rethrow_exception(ep);
        } catch (const std::exception & error) {
            message = error.what();
        }
        set_json(res, error_body(message, "server_error"), 500);
    });

    if (!server.bind_to_port(config.host, config.port)) {
        throw std::runtime_error("failed to bind " + config.host + ":" + std::to_string(config.port));
    }
    std::cout << "MFQ server ready: http://" << config.host << ":" << config.port
              << " model=" << config.model_name
              << " context=" << config.max_context
              << " vocab=" << tokenizer->vocab_size() << std::endl;
    if (!server.listen_after_bind()) {
        throw std::runtime_error("server stopped after binding " + config.host + ":" + std::to_string(config.port));
    }
    return 0;
}
