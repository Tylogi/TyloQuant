#include "mfq_server.h"

#include "httplib.h"
#include "json.hpp"
#include "gguf.h"
#include "llama.h"
#include "common/chat.h"
#include "common/common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

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

static void llama_log_quiet(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::cerr << "llama-tokenizer: " << text;
    }
}

class LlamaTokenizer {
public:
    explicit LlamaTokenizer(const std::string & path) {
        load_from_file(path);
        finish_init();
    }

    explicit LlamaTokenizer(const std::vector<uint8_t> & gguf) {
        if (gguf.empty()) {
            throw std::runtime_error("embedded tokenizer GGUF is empty");
        }
        llama_log_set(llama_log_quiet, nullptr);
        gguf_init_params metadata_params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ nullptr,
        };
        metadata_ = gguf_init_from_buffer(
            gguf.data(), gguf.size(), metadata_params);
        if (metadata_ == nullptr) {
            throw std::runtime_error(
                "cannot parse embedded tokenizer GGUF metadata");
        }
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.use_mmap = false;
        params.progress_callback = [](float, void *) { return true; };
        model_ = llama_model_init_from_user(
            metadata_, nullptr, nullptr, params);
        if (model_ == nullptr) {
            gguf_free(metadata_);
            metadata_ = nullptr;
            throw std::runtime_error(
                "cannot initialize tokenizer from embedded GGUF metadata");
        }
        finish_init();
    }

    ~LlamaTokenizer() {
        if (model_ != nullptr) llama_model_free(model_);
        if (metadata_ != nullptr) gguf_free(metadata_);
    }

    LlamaTokenizer(const LlamaTokenizer &) = delete;
    LlamaTokenizer & operator=(const LlamaTokenizer &) = delete;

    int32_t vocab_size() const {
        return llama_vocab_n_tokens(vocab_);
    }

    std::string chat_template() const {
        const char * value = llama_model_chat_template(model_, nullptr);
        return value == nullptr ? std::string() : std::string(value);
    }

    const llama_model * model() const {
        return model_;
    }

    int32_t bos_token() const {
        return llama_vocab_bos(vocab_);
    }

    int32_t eos_token() const {
        return llama_vocab_eos(vocab_);
    }

    int32_t eot_token() const {
        return llama_vocab_eot(vocab_);
    }

    int32_t pad_token() const {
        return llama_vocab_pad(vocab_);
    }

    bool add_bos() const {
        return llama_vocab_get_add_bos(vocab_);
    }

    bool add_eos() const {
        return llama_vocab_get_add_eos(vocab_);
    }

    std::vector<int64_t> tokenize(
            const std::string & text,
            bool parse_special,
            bool add_special = false) const {
        int32_t n = llama_tokenize(vocab_, text.data(), static_cast<int32_t>(text.size()),
                                   nullptr, 0, add_special, parse_special);
        if (n == std::numeric_limits<int32_t>::min()) {
            throw std::runtime_error("tokenized prompt exceeds the tokenizer limit");
        }
        if (n == 0) return {};
        if (n > 0) {
            throw std::runtime_error("tokenizer returned an invalid sizing result");
        }
        std::vector<llama_token> tokens(static_cast<size_t>(-n));
        n = llama_tokenize(vocab_, text.data(), static_cast<int32_t>(text.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()), add_special, parse_special);
        if (n < 0) throw std::runtime_error("tokenizer buffer sizing changed unexpectedly");
        std::vector<int64_t> out;
        out.reserve(static_cast<size_t>(n));
        for (int32_t i = 0; i < n; ++i) out.push_back(tokens[static_cast<size_t>(i)]);
        return out;
    }

    bool is_eog(int64_t token) const {
        return llama_vocab_is_eog(vocab_, static_cast<llama_token>(token));
    }

    std::string piece(int64_t token, bool special = false) const {
        char local[128];
        int32_t n = llama_token_to_piece(vocab_, static_cast<llama_token>(token),
                                         local, static_cast<int32_t>(sizeof(local)), 0, special);
        if (n >= 0) return std::string(local, local + n);
        std::string out(static_cast<size_t>(-n), '\0');
        n = llama_token_to_piece(vocab_, static_cast<llama_token>(token),
                                 out.data(), static_cast<int32_t>(out.size()), 0, special);
        if (n < 0) throw std::runtime_error("token piece buffer sizing changed unexpectedly");
        out.resize(static_cast<size_t>(n));
        return out;
    }

private:
    void load_from_file(const std::string & path) {
        llama_log_set(llama_log_quiet, nullptr);
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.use_mmap = true;
        params.progress_callback = [](float, void *) { return true; };
        model_ = llama_model_load_from_file(path.c_str(), params);
        if (model_ == nullptr) {
            throw std::runtime_error(
                "cannot load tokenizer metadata from GGUF: " + path);
        }
    }

    void finish_init() {
        vocab_ = llama_model_get_vocab(model_);
        if (vocab_ == nullptr) {
            llama_model_free(model_);
            model_ = nullptr;
            if (metadata_ != nullptr) {
                gguf_free(metadata_);
                metadata_ = nullptr;
            }
            throw std::runtime_error(
                "GGUF does not contain a tokenizer vocabulary");
        }
    }

    llama_model * model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    gguf_context * metadata_ = nullptr;
};

class LlamaGrammarConstraint {
public:
    LlamaGrammarConstraint(
            const LlamaTokenizer & tokenizer,
            const common_chat_params & params)
        : vocab_(llama_model_get_vocab(tokenizer.model())),
          vocab_size_(tokenizer.vocab_size()) {
        if (vocab_ == nullptr || params.grammar.empty()) {
            throw std::invalid_argument(
                "cannot create an empty chat-template grammar");
        }

        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
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

        grammar_ = params.grammar_lazy
            ? llama_sampler_init_grammar_lazy_patterns(
                  vocab_, params.grammar.c_str(), "root",
                  trigger_pattern_ptrs.data(),
                  trigger_pattern_ptrs.size(),
                  trigger_tokens.data(), trigger_tokens.size())
            : llama_sampler_init_grammar(
                  vocab_, params.grammar.c_str(), "root");
        if (grammar_ == nullptr) {
            throw std::runtime_error(
                "failed to initialize chat-template grammar");
        }

        if (!params.grammar_lazy &&
            !params.generation_prompt.empty()) {
            for (const auto token : tokenizer.tokenize(
                     params.generation_prompt, true)) {
                llama_sampler_accept(
                    grammar_, static_cast<llama_token>(token));
            }
        }
    }

    ~LlamaGrammarConstraint() {
        if (grammar_ != nullptr) {
            llama_sampler_free(grammar_);
        }
    }

    LlamaGrammarConstraint(const LlamaGrammarConstraint &) = delete;
    LlamaGrammarConstraint & operator=(
        const LlamaGrammarConstraint &) = delete;

    bool allows(std::int64_t token) {
        if (token < 0 || token >= vocab_size_) return false;
        llama_token_data candidate = {
            static_cast<llama_token>(token), 0.0f, 0.0f};
        llama_token_data_array candidates = {
            &candidate, 1, -1, false};
        llama_sampler_apply(grammar_, &candidates);
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
                static_cast<llama_token>(index), logits[index], 0.0f};
        }
        llama_token_data_array candidates = {
            candidates_.data(), candidates_.size(), -1, false};
        llama_sampler_apply(grammar_, &candidates);
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
        llama_sampler_accept(
            grammar_, static_cast<llama_token>(token));
    }

private:
    const llama_vocab * vocab_ = nullptr;
    int32_t vocab_size_ = 0;
    llama_sampler * grammar_ = nullptr;
    std::vector<llama_token_data> candidates_;
};

static MfqTokenConstraintPtr make_token_constraint(
        const LlamaTokenizer & tokenizer,
        const common_chat_params & params) {
    if (params.grammar.empty()) return {};
    auto implementation =
        std::make_shared<LlamaGrammarConstraint>(tokenizer, params);
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

static bool request_enable_thinking(const json & body) {
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
    return true;
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
        const json & body, const common_chat_templates * templates) {
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
        inputs.enable_thinking = request_enable_thinking(body);
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
    if (config.model_type == "deepseek_v4") {
        defaults.temperature = 1.0;
        defaults.top_p = 0.8;
        defaults.repetition_penalty = 1.05;
        defaults.presence_penalty = 0.0;
    }
    return defaults;
}

static json sampling_params_json(const MfqSamplingParams & sampling) {
    return {
        {"temperature", sampling.temperature},
        {"top_k", sampling.top_k},
        {"top_p", sampling.top_p},
        {"presence_penalty", sampling.presence_penalty},
        {"frequency_penalty", sampling.frequency_penalty},
        {"repetition_penalty", sampling.repetition_penalty},
    };
}

static json chat_template_capabilities_json(
        const std::string & chat_template) {
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
};

static RequestWork parse_work(const json & body, bool chat, const LlamaTokenizer & tokenizer,
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
        ? integer_field(body, "max_completion_tokens", 4096)
        : integer_field(body, "max_tokens", 4096);
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
            apply_chat_template(body, templates);
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
    if (chat && model_type == "deepseek_v4" &&
        boolean_field(body, "add_generation_prompt", true)) {
        const std::string stable_marker =
            request_enable_thinking(body) ? "<think>" : "</think>";
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
    bool saw_token = false;
    bool saw_prefill = false;

    void mark_prefill(size_t prompt_token_count, double elapsed_ms) {
        prefill_tokens = prompt_token_count;
        prefill_ms = elapsed_ms;
        saw_prefill = elapsed_ms > 0.0;
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
};

struct RequestMetricValues {
    size_t prefill_tokens = 0;
    double generation_ms = 0.0;
    double ttft_ms = 0.0;
    double prefill_ms = 0.0;
    double prefill_tps = 0.0;
    double decode_ms = 0.0;
    double generation_tps = 0.0;
    double decode_tps = 0.0;
};

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

static CompletionResult generate_text(const RequestWork & work, const LlamaTokenizer & tokenizer,
                                      const MfqGenerateFn & generate,
                                      const std::function<bool(const common_chat_msg_diff &)> & emit,
                                      RequestMetrics * metrics) {
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

    result.completion_tokens = generate(
        work.prompt, work.sampling,
        [&](int64_t token) {
            if (metrics != nullptr) metrics->mark_token();
            if (tokenizer.is_eog(token)) {
                result.finish_reason = "stop";
                return false;
            }
            const bool preserve =
                work.preserved_tokens.find(token) !=
                work.preserved_tokens.end();
            if (!emitter.append(tokenizer.piece(token, preserve))) {
                if (emitter.stopped()) result.finish_reason = "stop";
                return false;
            }
            return true;
        },
        [&](size_t prompt_tokens, double prefill_ms) {
            if (metrics != nullptr) {
                metrics->mark_prefill(prompt_tokens, prefill_ms);
            }
        },
        work.cache_plan,
        work.token_constraint);
    if (result.client_connected && !emitter.stopped()) emitter.flush();
    if (result.client_connected && chat_parser) {
        chat_parser->flush();
        const auto & message = chat_parser->message();
        result.text = message.content;
        result.reasoning_text = message.reasoning_content;
        result.tool_calls = message.tool_calls;
    }
    if (emitter.stopped()) result.finish_reason = "stop";
    if (!result.tool_calls.empty()) result.finish_reason = "tool_calls";
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

MfqTokenizerProbe probe_mfq_tokenizer(
        const std::vector<uint8_t> & tokenizer_gguf,
        const std::string & text,
        bool add_special,
        bool parse_special) {
    LlamaTokenizer tokenizer(tokenizer_gguf);
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
    LlamaTokenizer tokenizer(tokenizer_model);
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
        const MfqReloadFn & reload) {
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

    std::unique_ptr<LlamaTokenizer> tokenizer =
        config.tokenizer_gguf.empty()
        ? std::make_unique<LlamaTokenizer>(
              config.tokenizer_model)
        : std::make_unique<LlamaTokenizer>(
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
        common_chat_templates_init(tokenizer->model(), "");
    if (!chat_templates) {
        throw std::runtime_error(
            "cannot initialize tokenizer.chat_template");
    }
    const MfqSamplingParams sampling_defaults =
        default_sampling_params(config);
    const json chat_template_capabilities =
        chat_template_capabilities_json(
            tokenizer->chat_template());

    httplib::Server server;
    ServerMetrics server_metrics;
    std::atomic<int64_t> active_context{config.max_context};
    std::atomic<bool> reloading{false};
    server.set_payload_max_length(16 * 1024 * 1024);
    server.set_read_timeout(300, 0);
    server.set_write_timeout(300, 0);
    server.set_keep_alive_max_count(100);
    server.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"X-Content-Type-Options", "nosniff"},
    });

    server.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
    });

    bool web_ui_available = false;
    if (!config.web_root.empty()) {
        std::error_code error;
        const auto web_root = std::filesystem::absolute(
            std::filesystem::path(config.web_root), error);
        if (!error &&
            std::filesystem::is_regular_file(web_root / "index.html", error)) {
            web_ui_available = server.set_mount_point(
                "/admin", web_root.string(), {
                    {"Cache-Control", "no-cache"},
                    {"Referrer-Policy", "no-referrer"},
                    {"X-Frame-Options", "DENY"},
                });
        }
        if (!web_ui_available) {
            std::cerr << "MFQ web UI disabled: cannot read "
                      << config.web_root << "/index.html" << std::endl;
        }
    }

    server.Get("/admin", [web_ui_available](
            const httplib::Request &, httplib::Response & res) {
        if (web_ui_available) {
            res.set_redirect("/admin/");
        } else {
            set_json(res, error_body(
                "MFQ web UI assets are unavailable", "not_found"), 404);
        }
    });

    server.Get("/", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        if (web_ui_available &&
            req.get_header_value("Accept").find("text/html") !=
                std::string::npos) {
            res.set_redirect("/admin/");
            return;
        }
        set_json(res, {
            {"name", "MFQ C++ inference server"},
            {"model", config.model_name},
            {"endpoints", {
                "/v1/chat/completions", "/v1/completions", "/v1/models",
                "/health", "/api/status", "/api/reload", "/admin/",
            }},
        });
    });

    server.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        set_json(res, {
            {"status", reloading.load() ? "loading" : "ok"},
            {"model", config.model_name},
            {"model_type", config.model_type},
            {"max_context", active_context.load()},
            {"sampling_defaults", sampling_params_json(sampling_defaults)},
            {"chat_template_capabilities", chat_template_capabilities},
        });
    });

    server.Get("/api/status", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        json status = server_metrics.snapshot(
            config, active_context.load(), reloading.load());
        status["sampling_defaults"] = sampling_params_json(
            sampling_defaults);
        status["chat_template_capabilities"] =
            chat_template_capabilities;
        set_json(res, status);
    });

    server.Post("/api/reload", [&](const httplib::Request & req, httplib::Response & res) {
        if (!authorized(req, res, config.api_key)) return;
        if (!reload) {
            set_json(res, error_body(
                "this runtime does not support model reload",
                "unsupported_operation"), 501);
            return;
        }
        bool expected = false;
        if (!reloading.compare_exchange_strong(expected, true)) {
            set_json(res, error_body(
                "model reload is already in progress", "conflict"), 409);
            return;
        }
        const auto finish_reload = [&] {
            reloading.store(false);
        };
        try {
            if (server_metrics.active_requests() != 0) {
                throw ApiError(
                    409, "conflict",
                    "cannot reload while a generation request is active");
            }
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
            const std::string id = request_id(chat ? "chatcmpl-" : "cmpl-");
            const int64_t created = unix_time_seconds();

            if (!work.stream) {
                ActiveRequest active_request(server_metrics);
                RequestMetrics metrics;
                CompletionResult result = generate_text(
                    work, *tokenizer, generate,
                    [](const common_chat_msg_diff &) {
                        return true;
                    },
                    &metrics);
                const RequestMetricValues metric_values =
                    request_metric_values(result, metrics);
                log_request_metrics(
                    id, chat, false, work.prompt.size(), work.sampling,
                    result, metric_values);
                active_request.complete(
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
                set_json(res, response);
                return;
            }

            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider(
                "text/event-stream; charset=utf-8",
                [work = std::move(work), id, created, &tokenizer, &generate,
                 &config, &server_metrics, &reloading, chat]
                (size_t offset, httplib::DataSink & sink) mutable -> bool {
                    if (offset != 0) {
                        sink.done();
                        return false;
                    }
                    if (reloading.load()) {
                        write_sse(sink, error_body(
                            "model reload is in progress",
                            "service_unavailable"));
                        sink.done();
                        return false;
                    }
                    try {
                        ActiveRequest active_request(server_metrics);
                        if (chat && !write_sse(sink, chat_chunk(id, created, config.model_name,
                                                                {{"role", "assistant"}, {"content", ""}}, nullptr))) {
                            return false;
                        }
                        RequestMetrics metrics;
                        CompletionResult result = generate_text(
                            work, *tokenizer, generate,
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
                        }, &metrics);
                        const RequestMetricValues metric_values =
                            request_metric_values(result, metrics);
                        log_request_metrics(
                            id, chat, true, work.prompt.size(), work.sampling,
                            result, metric_values);
                        active_request.complete(
                            id, chat, true, work.prompt.size(), result,
                            metric_values);
                        if (!result.client_connected) return false;
                        const json final_chunk = chat
                            ? chat_chunk(id, created, config.model_name, json::object(), result.finish_reason)
                            : completion_chunk(id, created, config.model_name, "", result.finish_reason);
                        if (!write_sse(sink, final_chunk)) return false;
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
