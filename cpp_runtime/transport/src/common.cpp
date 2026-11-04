#include "common.h"

#include "nlohmann/json.hpp"
#include "chat.h"
#include "mtp_metrics.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
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
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <aclapi.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mfq::transport_detail {


int64_t unix_time_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string request_id(const char * prefix) {
    static std::atomic<uint64_t> sequence{0};
    const uint64_t n = sequence.fetch_add(1, std::memory_order_relaxed);
    return std::string(prefix) + std::to_string(unix_time_seconds()) + "-" + std::to_string(n);
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

static std::optional<std::string> request_preformatted_prompt(
    const json & body);

static mfq::engine::ChatInput parse_chat_input(
        const json & body,
        bool enable_thinking_default) {
    if (!body.contains("messages") || !body["messages"].is_array() ||
        body["messages"].empty()) {
        throw ApiError(
            400, "invalid_request_error",
            "messages must be a non-empty array", "messages");
    }

    mfq::engine::ChatInput result;
    auto & inputs = result.template_inputs;
    result.preformatted_prompt = request_preformatted_prompt(body);
    result.reasoning_format = request_reasoning_format(body);
    inputs.json_schema = request_json_schema(body);
    inputs.reasoning_format = result.reasoning_format;
    inputs.enable_thinking = request_enable_thinking(
        body, enable_thinking_default);
    inputs.use_jinja = true;
    inputs.add_generation_prompt =
        boolean_field(body, "add_generation_prompt", true);
    if (result.preformatted_prompt) return result;

    try {
        inputs.messages = common_chat_msgs_parse_oaicompat(body["messages"]);
        if (body.contains("continue_final_message") &&
            !body["continue_final_message"].is_null()) {
            inputs.continue_final_message = common_chat_continuation_parse(
                body["continue_final_message"]);
        }
        if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE &&
            inputs.add_generation_prompt) {
            throw ApiError(
                400, "invalid_request_error",
                "add_generation_prompt and continue_final_message "
                "cannot both be enabled",
                "continue_final_message");
        }
        if (body.contains("parallel_tool_calls") &&
            !body["parallel_tool_calls"].is_null()) {
            result.parallel_tool_calls = boolean_field(
                body, "parallel_tool_calls", false);
        }
        if (body.contains("tools") && !body["tools"].is_null()) {
            inputs.tools = common_chat_tools_parse_oaicompat(body["tools"]);
        }
        const json tool_choice =
            body.contains("tool_choice") && !body["tool_choice"].is_null()
                ? body["tool_choice"] : json("auto");
        if (tool_choice.is_string()) {
            inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(
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
            for (const auto & item : body["chat_template_kwargs"].items()) {
                inputs.chat_template_kwargs[item.key()] = item.value().dump();
            }
        }
    } catch (const ApiError &) {
        throw;
    } catch (const std::exception & error) {
        throw ApiError(
            400, "invalid_request_error", error.what(), "messages");
    }
    return result;
}

static std::optional<std::string> request_preformatted_prompt(
        const json & body) {
    if (!body.contains("mfq_preformatted_prompt") ||
        body["mfq_preformatted_prompt"].is_null()) {
        return std::nullopt;
    }
    if (!body["mfq_preformatted_prompt"].is_string() ||
        body["mfq_preformatted_prompt"].get_ref<
            const std::string&>().empty()) {
        throw ApiError(
            400,
            "invalid_request_error",
            "mfq_preformatted_prompt must be a non-empty string",
            "mfq_preformatted_prompt");
    }
    return body["mfq_preformatted_prompt"].get<std::string>();
}

int64_t integer_field(const json & body, const char * name, int64_t fallback) {
    if (!body.contains(name) || body[name].is_null()) return fallback;
    if (!body[name].is_number_integer()) {
        throw ApiError(400, "invalid_request_error", std::string(name) + " must be an integer", name);
    }
    return body[name].get<int64_t>();
}

double number_field(const json & body, const char * name, double fallback) {
    if (!body.contains(name) || body[name].is_null()) return fallback;
    if (!body[name].is_number()) {
        throw ApiError(400, "invalid_request_error", std::string(name) + " must be a number", name);
    }
    return body[name].get<double>();
}

MfqSamplingParams default_sampling_params(
        const MfqRuntimeTransportConfig & config) {
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
    if (profile.enable_vision) defaults.enable_vision = *profile.enable_vision;
    if (profile.enable_mtp) defaults.enable_mtp = *profile.enable_mtp;
    if (profile.mtp_max_draft_tokens) {
        defaults.mtp_max_draft_tokens = *profile.mtp_max_draft_tokens;
    }
    return defaults;
}

json sampling_params_json(const MfqSamplingParams & sampling) {
    return {
        {"max_tokens", sampling.max_tokens},
        {"temperature", sampling.temperature},
        {"top_k", sampling.top_k},
        {"top_p", sampling.top_p},
        {"presence_penalty", sampling.presence_penalty},
        {"frequency_penalty", sampling.frequency_penalty},
        {"repetition_penalty", sampling.repetition_penalty},
        {"enable_thinking", sampling.enable_thinking},
        {"enable_vision", sampling.enable_vision},
        {"enable_mtp", sampling.enable_mtp},
        {"mtp_max_draft_tokens", sampling.mtp_max_draft_tokens},
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
    MFQ_MERGE(chat, enable_vision);
    MFQ_MERGE(chat, enable_mtp);
    MFQ_MERGE(chat, mtp_max_draft_tokens);
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
    } else if (identity_matches(identities, "deepseek_v41")) {
        result.chat.temperature = 1.0;
        result.chat.top_p = 0.95;
        result.chat.mtp_max_draft_tokens = 5;
        result.source = "architecture-registry:deepseek_v41";
    } else if (identity_matches(identities, "deepseek_v4")) {
        result.chat.temperature = 1.0;
        result.chat.top_p = 0.8;
        result.chat.repetition_penalty = 1.05;
        result.chat.presence_penalty = 0.0;
        result.chat.mtp_max_draft_tokens = 5;
        result.source = "architecture-registry:deepseek_v4";
    }
    return result;
}

struct MfqModelCapabilityRegistration {
    std::array<const char *, 3> aliases{};
    MfqModelCapabilityProfile profile;
};

static const std::array<MfqModelCapabilityRegistration, 12>
    kModelCapabilityRegistry{{
        {{{"minicpmo", nullptr, nullptr}},
         {"minicpmo", true, true, true, true, true, true, false}},
        {{{"minicpmtts", nullptr, nullptr}},
         {"minicpmo_tts", false, false, false, false, true, false, false}},
        {{{"deepseek_v41", "deepseek_v41_text", nullptr}},
         {"deepseek_v41", true, false, false, false, false, false, true}},
        {{{"deepseek_v41_vision", nullptr, nullptr}},
         {"deepseek_v41", true, true, false, false, false, false, true}},
        {{{"deepseek_v4", nullptr, nullptr}},
         {"deepseek_v4", true, false, false, false, false, false, true}},
        {{{"deepseek_v4_vision", nullptr, nullptr}},
         {"deepseek_v4", true, true, false, false, false, false, true}},
        {{{"glm_moe_dsa", nullptr, nullptr}}, {"glm_dsa"}},
        {{{"gemma4", "gemma4_text", nullptr}}, {"gemma4"}},
        {{{"qwen3_5", "qwen35", nullptr}},
         {"qwen3_5", true, true, true, false, false, false, true}},
        {{{"qwen3_5_text", nullptr, nullptr}},
         {"qwen3_5", true, false, false, false, false, false, true}},
        {{{"qwen4_exp", "qwen4_exp_text", nullptr}},
         {"qwen4_exp", true, true, true, false, false, false, true}},
        {{{"glm5_next", "glm5_next_text", nullptr}},
         {"glm5_next", true, true, true, false, false, false, true}},
    }};

MfqModelCapabilityProfile architecture_capability_profile(
        const std::string & model_type) {
    const std::string identity = normalized_identity(model_type);
    for (const auto & registration : kModelCapabilityRegistry) {
        for (const char * alias : registration.aliases) {
            if (alias != nullptr && identity == alias) {
                return registration.profile;
            }
        }
    }
    MfqModelCapabilityProfile result;
    if (!identity.empty()) result.family = identity;
    return result;
}

json model_capability_profile_json(
        const MfqModelCapabilityProfile & profile) {
    return {
        {"architecture_family", profile.family},
        {"source", profile.source.empty()
            ? "architecture-registry:" + profile.family
            : profile.source},
        {"features", {
            {"text", profile.text},
            {"image_input", profile.image_input},
            {"video_input", profile.video_input},
            {"audio_input", profile.audio_input},
            {"audio_output", profile.audio_output},
            {"full_duplex", profile.full_duplex},
            {"mtp", profile.mtp},
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
        if (value.contains("enable_vision")) {
            if (!value["enable_vision"].is_boolean()) {
                throw std::runtime_error(
                    "runtime profile chat.enable_vision must be boolean");
            }
            result.chat.enable_vision = value["enable_vision"].get<bool>();
        }
        if (value.contains("enable_mtp")) {
            if (!value["enable_mtp"].is_boolean()) {
                throw std::runtime_error(
                    "runtime profile chat.enable_mtp must be boolean");
            }
            result.chat.enable_mtp = value["enable_mtp"].get<bool>();
        }
        if (value.contains("mtp_max_draft_tokens")) {
            result.chat.mtp_max_draft_tokens =
                profile_integer(value, "mtp_max_draft_tokens");
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
    if (result.chat.mtp_max_draft_tokens &&
        (*result.chat.mtp_max_draft_tokens < 1 ||
         *result.chat.mtp_max_draft_tokens > 5)) {
        throw std::runtime_error(
            "runtime profile chat.mtp_max_draft_tokens must be in [1, 5]");
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
    static const std::regex split_pattern(R"(^(.*)-[0-9]{5}-of-[0-9]{5}\.[mM][fF][qQ]$)");
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

json duplex_profile_json(const MfqDuplexSamplingProfile & value) {
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

json tts_profile_json(const MfqTtsSamplingProfile & value) {
    json result = json::object();
    if (value.temperature) result["temperature"] = *value.temperature;
    if (value.repetition_penalty) result["repetition_penalty"] = *value.repetition_penalty;
    if (value.token2wav_steps) result["token2wav_steps"] = *value.token2wav_steps;
    return result;
}

json chat_template_capabilities_json(
        const mfq::engine::ChatTemplateCapabilities & capabilities) {
    return {
        {"thinking", {
            {"supported", capabilities.thinking},
        }},
        {"reasoning_effort", {
            {"supported", !capabilities.reasoning_effort_values.empty()},
            {"values", capabilities.reasoning_effort_values},
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


bool valid_mfq_session_id(const std::string & session_id) {
    return !session_id.empty() && session_id.size() <= 128 &&
        std::all_of(
            session_id.begin(), session_id.end(),
            [](unsigned char value) {
                return std::isalnum(value) != 0 || value == '-' ||
                    value == '_' || value == '.' || value == ':';
            });
}

mfq::engine::CountTokens parse_token_count_request(const json& body) {
    if (!body.is_object() || body.size() != 1 || !body.contains("prompt") ||
        !body["prompt"].is_string() || body["prompt"].get_ref<const std::string&>().empty() ||
        body["prompt"].get_ref<const std::string&>().size() > 262144)
        throw ApiError(400, "invalid_request_error", "token count requires a nonempty prompt of at most 262144 bytes");
    return {body["prompt"].get<std::string>()};
}

json token_count_json(const mfq::engine::TokenCount& result) {
    return {{"prompt_tokens", result.prompt_tokens}, {"context_size", result.context_size},
        {"tokenization", "raw-no-special-tokens"}};
}

mfq::engine::ScoreText parse_score_request(const json& body) {
    if (!body.is_object() || !body.contains("prompt") || !body["prompt"].is_string() ||
        body["prompt"].get_ref<const std::string&>().empty() || body["prompt"].get_ref<const std::string&>().size() > 262144 ||
        !body.contains("continuations") || !body["continuations"].is_array() || body["continuations"].empty() || body["continuations"].size() > 64)
        throw ApiError(400, "invalid_request_error", "score requires a prompt and 1 to 64 continuations");
    if (body.contains("mode") && !body["mode"].is_string())
        throw ApiError(400, "invalid_request_error", "score mode must be a string", "mode");
    for (const auto& item : body.items())
        if (item.key() != "prompt" && item.key() != "continuations" && item.key() != "mode")
            throw ApiError(400, "invalid_request_error", "unknown score parameter", item.key());
    const auto mode = body.value("mode", std::string("continuation"));
    if (mode != "continuation" && mode != "next_token")
        throw ApiError(400, "invalid_request_error", "score mode must be continuation or next_token", "mode");
    mfq::engine::ScoreText request;
    request.prompt = body["prompt"].get<std::string>();
    request.next_token = mode == "next_token";
    for (const auto& item : body["continuations"]) {
        if (!item.is_string() || item.get_ref<const std::string&>().empty() || item.get_ref<const std::string&>().size() > 65536)
            throw ApiError(400, "invalid_request_error", "score continuations must be nonempty bounded strings", "continuations");
        request.continuations.push_back(item.get<std::string>());
    }
    return request;
}

json likelihood_result_json(const mfq::engine::LikelihoodResult& result) {
    json scores = json::array();
    for (const auto& score : result.scores)
        scores.push_back({{"token_ids", score.token_ids}, {"token_logprobs", score.token_logprobs}, {"log_likelihood", score.log_likelihood}});
    return {{"prompt_tokens", result.prompt_tokens}, {"scores", scores}, {"log_base", "e"}, {"tokenization", "raw-no-special-tokens"}};
}

RequestSchedule parse_schedule(const json & body) {
    RequestSchedule result;
    if (body.contains("priority") && !body["priority"].is_null()) {
        if (!body["priority"].is_number_integer()) {
            throw ApiError(
                400, "invalid_request_error",
                "priority must be an integer", "priority");
        }
        const int64_t priority = body["priority"].get<int64_t>();
        if (priority < std::numeric_limits<int>::min() ||
            priority > std::numeric_limits<int>::max()) {
            throw ApiError(
                400, "invalid_request_error",
                "priority must be a 32-bit integer", "priority");
        }
        result.priority = static_cast<int>(priority);
    }
    if (body.contains("timeout_seconds") && !body["timeout_seconds"].is_null()) {
        const double seconds = number_field(body, "timeout_seconds", 0.0);
        if (!std::isfinite(seconds) || seconds <= 0.0 ||
            seconds > max_request_timeout_seconds) {
            throw ApiError(
                400, "invalid_request_error",
                "timeout_seconds must be in (0, 86400]", "timeout_seconds");
        }
        const auto remaining = std::chrono::duration_cast<mfq::engine::Clock::duration>(
            std::chrono::duration<double>(seconds));
        result.deadline = mfq::engine::Clock::now() + remaining;
    }
    return result;
}

json runtime_generate_body(const json & params) {
    if (!params.is_object()) {
        throw ApiError(
            400, "invalid_request_error",
            "runtime generation parameters must be a JSON object");
    }
    if (!params.contains("input") || !params["input"].is_object()) {
        throw ApiError(
            400, "invalid_request_error",
            "runtime generation requires an input object", "input");
    }
    const auto & input = params["input"];
    json body = json::object();
    if (params.contains("model")) body["model"] = params["model"];
    if (input.contains("messages")) body["messages"] = input["messages"];
    if (input.contains("preformatted_prompt")) {
        body["mfq_preformatted_prompt"] = input["preformatted_prompt"];
    }
    if (input.contains("benchmark_prompt_tokens")) {
        body["mfq_benchmark_prompt_tokens"] = input["benchmark_prompt_tokens"];
    }
    if (params.contains("cache")) {
        if (!params["cache"].is_object()) {
            throw ApiError(400, "invalid_request_error", "cache must be an object", "cache");
        }
        if (params["cache"].contains("enabled"))
            body["mfq_prefix_cache_enabled"] = params["cache"]["enabled"];
    }
    if (params.contains("sampling")) {
        if (!params["sampling"].is_object()) {
            throw ApiError(
                400, "invalid_request_error",
                "sampling must be an object", "sampling");
        }
        const auto & sampling = params["sampling"];
        if (sampling.contains("max_new_tokens")) {
            body["max_tokens"] = sampling["max_new_tokens"];
        }
        constexpr std::array<const char *, 10> fields{
            "temperature", "top_k", "top_p", "presence_penalty",
            "frequency_penalty", "repetition_penalty", "enable_vision",
            "enable_mtp", "mtp_max_draft_tokens", "seed",
        };
        for (const char * field : fields) {
            if (sampling.contains(field)) body[field] = sampling[field];
        }
    }
    if (params.contains("stream")) body["stream"] = params["stream"];
    if (params.contains("include_usage")) {
        body["stream_options"] = {
            {"include_usage", params["include_usage"]},
        };
    }
    if (params.contains("template")) {
        body["chat_template_kwargs"] = params["template"];
    }
    if (params.contains("output")) {
        if (!params["output"].is_object()) {
            throw ApiError(
                400, "invalid_request_error",
                "output must be an object", "output");
        }
        if (params["output"].contains("reasoning_format")) {
            body["reasoning_format"] =
                params["output"]["reasoning_format"];
        }
    }
    constexpr std::array<const char *, 3> constraint_fields{
        "tools", "tool_choice", "response_format",
    };
    for (const char * field : constraint_fields) {
        if (params.contains(field)) body[field] = params[field];
    }
    if (params.contains("session_id")) {
        body["mfq_session_id"] = params["session_id"];
    }
    if (params.contains("priority")) body["priority"] = params["priority"];
    if (params.contains("timeout_seconds")) {
        body["timeout_seconds"] = params["timeout_seconds"];
    }
    if (params.contains("media")) body["mfq_multimodal"] = params["media"];
    return body;
}

RequestInput parse_input(
        const json & body,
        bool chat,
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

    RequestInput work;
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
    work.sampling.enable_vision = boolean_field(
        body, "enable_vision", defaults.enable_vision);
    work.sampling.enable_mtp = boolean_field(
        body, "enable_mtp", defaults.enable_mtp);
    work.sampling.mtp_max_draft_tokens = static_cast<int32_t>(integer_field(
        body,
        "mtp_max_draft_tokens",
        defaults.mtp_max_draft_tokens));
    if (work.sampling.temperature < 0.0 || work.sampling.temperature > 10.0) {
        throw ApiError(400, "invalid_request_error", "temperature must be in [0, 10]", "temperature");
    }
    if (work.sampling.top_p < 0.0 || work.sampling.top_p > 1.0) {
        throw ApiError(400, "invalid_request_error", "top_p must be in [0, 1]", "top_p");
    }
    if (work.sampling.top_k < 0 || work.sampling.top_k > 1024) {
        throw ApiError(400, "invalid_request_error", "top_k must be in [0, 1024]", "top_k");
    }
    if (work.sampling.mtp_max_draft_tokens < 1 ||
        work.sampling.mtp_max_draft_tokens > 5) {
        throw ApiError(
            400,
            "invalid_request_error",
            "mtp_max_draft_tokens must be in [1, 5]",
            "mtp_max_draft_tokens");
    }
    if (work.sampling.temperature > 0.0 && work.sampling.top_k == 0 &&
        work.sampling.top_p > 0.0 && work.sampling.top_p < 1.0) {
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

    if (chat) {
        work.chat_input = parse_chat_input(
            body, work.sampling.enable_thinking);
    } else {
        if (!body.contains("prompt") || !body["prompt"].is_string()) {
            throw ApiError(
                400, "invalid_request_error",
                "prompt must be a string", "prompt");
        }
        work.prompt = body["prompt"].get<std::string>();
    }
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
    }
    work.stops = parse_stops(body);
    work.prefix_cache_enabled = boolean_field(body, "mfq_prefix_cache_enabled", true);
    if (body.contains("mfq_benchmark_prompt_tokens")) {
        const auto count = integer_field(body, "mfq_benchmark_prompt_tokens", 0);
        if (count < 1 || count > 1048576) {
            throw ApiError(400, "invalid_request_error", "benchmark prompt tokens must be in [1, 1048576]", "mfq_benchmark_prompt_tokens");
        }
        if (work.prefix_cache_enabled || !work.chat_input || !work.chat_input->preformatted_prompt) {
            throw ApiError(400, "invalid_request_error", "benchmark token sizing requires a preformatted prompt and disabled prefix cache");
        }
        work.benchmark_prompt_tokens = static_cast<std::size_t>(count);
    }
    return work;
}


json request_metric_values_json(
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
            {"enable_mtp", sampling.enable_mtp},
            {"mtp_max_draft_tokens", sampling.mtp_max_draft_tokens},
        }},
    };
}

void add_request_runtime_metrics(json& value, const RequestMetrics& metrics) {
    value["mtp_available"] = metrics.mtp.available ? 1.0 : 0.0;
    if (!metrics.mtp.available) return;
    mfq::engine::Metrics values;
    mfq::engine::mtp::append_generation_metrics(values, metrics.mtp);
    for (const auto& [name, number] : values) value[name] = number;
}

RequestMetricValues request_metric_values(
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
              metrics.last_token - metrics.first_token).count()
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

void log_request_metrics(const std::string & id, bool chat, bool stream,
                                size_t prompt_tokens, const MfqSamplingParams & sampling,
                                const CompletionResult & result,
                                const RequestMetricValues & values) {
    const char * enabled = std::getenv("MFQ_RUNTIME_REQUEST_METRICS");
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
         << " mtp=" << (sampling.enable_mtp ? 1 : 0)
         << " mtp_max_draft_tokens=" << sampling.mtp_max_draft_tokens
         << " seed=" << sampling.seed
         << " penalties=" << (penalties ? 1 : 0)
         << " finish_reason=" << result.finish_reason
         << " client_connected=" << (result.client_connected ? 1 : 0);
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cout << line.str() << std::endl;
}

RuntimeRequestMetrics::RuntimeRequestMetrics()
    : started_steady_(std::chrono::steady_clock::now()),
      started_unix_(unix_time_seconds()) {}

void RuntimeRequestMetrics::begin() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++total_requests_;
    ++active_requests_;
}

void RuntimeRequestMetrics::complete(
        const std::string & id,
        bool chat,
        bool stream,
        size_t prompt_tokens,
        const CompletionResult & result,
        const RequestMetricValues & values) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_requests_ > 0) --active_requests_;
    live_requests_.erase(id);
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

void RuntimeRequestMetrics::progress(const std::string & id, const json & value) {
    std::lock_guard<std::mutex> lock(mutex_);
    live_requests_[id] = value;
}

void RuntimeRequestMetrics::fail(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex_);
    live_requests_.erase(id);
    if (active_requests_ > 0) --active_requests_;
    ++failed_requests_;
}

uint64_t RuntimeRequestMetrics::active_requests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_requests_;
}

json RuntimeRequestMetrics::snapshot(
        const MfqRuntimeTransportConfig & config,
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
        {"live_requests", live_requests_},
    };
}

ActiveRequest::ActiveRequest(RuntimeRequestMetrics & metrics)
    : metrics_(metrics) {
    metrics_.begin();
}

ActiveRequest::~ActiveRequest() {
    if (!completed_) metrics_.fail(id_);
}

void ActiveRequest::progress(const std::string & id, const json & value) {
    id_ = id;
    metrics_.progress(id, value);
}

void ActiveRequest::complete(
        const std::string & id,
        bool chat,
        bool stream,
        size_t prompt_tokens,
        const CompletionResult & result,
        const RequestMetricValues & values) {
    metrics_.complete(id, chat, stream, prompt_tokens, result, values);
    completed_ = true;
}

json usage_json(size_t prompt_tokens, int32_t completion_tokens) {
    return {
        {"prompt_tokens", prompt_tokens},
        {"completion_tokens", completion_tokens},
        {"total_tokens", prompt_tokens + static_cast<size_t>(completion_tokens)},
    };
}

static json chat_tool_calls_json(
    const std::vector<common_chat_tool_call> & tool_calls);

json runtime_generation_event(
        const std::string & event,
        const std::string & request_id,
        int64_t created,
        const std::string & model) {
    return {
        {"event", event},
        {"request_id", request_id},
        {"created", created},
        {"model", model},
    };
}

json runtime_generation_result(
        const std::string & request_id,
        int64_t created,
        const std::string & model,
        const CompletionResult & result,
        json usage,
        json metrics) {
    json output = {{"text", result.text}};
    if (!result.reasoning_text.empty()) {
        output["reasoning"] = result.reasoning_text;
    }
    if (!result.tool_calls.empty()) {
        output["tool_calls"] = chat_tool_calls_json(result.tool_calls);
    }
    return {
        {"request_id", request_id},
        {"created", created},
        {"model", model},
        {"output", std::move(output)},
        {"finish_reason", result.finish_reason},
        {"usage", std::move(usage)},
        {"metrics", std::move(metrics)},
    };
}

json chat_diff_json(const common_chat_msg_diff & diff) {
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
        // Preserve UTF-8 paths on Windows by building the path from UTF-8 bytes.
        const std::string utf8_path = spec["path"].get<std::string>();
        const auto path = std::filesystem::path(
            std::u8string(utf8_path.begin(), utf8_path.end()));
        const auto filename = path.filename().native();
        if (!path.is_absolute() ||
            filename.rfind(std::filesystem::path("mfq-multimodal-").native(), 0) != 0 ||
            path.extension() != ".bin") {
            throw ApiError(
                400, "invalid_request_error",
                parameter + " path is not an MFQ temporary tensor file", parameter);
        }
        const auto token = decode_file_token(
            spec["token"].get<std::string>(), parameter + ".token");
#ifdef _WIN32
        descriptor_=CreateFileW(path.c_str(),GENERIC_READ|READ_CONTROL,FILE_SHARE_READ,
            nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        BY_HANDLE_FILE_INFORMATION metadata{};
        bool valid=descriptor_!=INVALID_HANDLE_VALUE && GetFileType(descriptor_)==FILE_TYPE_DISK &&
            GetFileInformationByHandle(descriptor_,&metadata) && metadata.nNumberOfLinks==1 &&
            !(metadata.dwFileAttributes&(FILE_ATTRIBUTE_DIRECTORY|FILE_ATTRIBUTE_REPARSE_POINT)) &&
            ((uint64_t(metadata.nFileSizeHigh)<<32)|metadata.nFileSizeLow)==size_;
        if(valid) {
            PSID owner=nullptr;PSECURITY_DESCRIPTOR security=nullptr;HANDLE process_token=nullptr;
            valid=GetSecurityInfo(descriptor_,SE_FILE_OBJECT,OWNER_SECURITY_INFORMATION,
                &owner,nullptr,nullptr,nullptr,&security)==ERROR_SUCCESS;
            if(valid)valid=OpenProcessToken(GetCurrentProcess(),TOKEN_QUERY,&process_token)!=0;
            if(valid) {
                DWORD bytes=0;
                GetTokenInformation(process_token,TokenOwner,nullptr,0,&bytes);
                std::vector<uint8_t> info(bytes);
                valid=bytes && GetTokenInformation(process_token,TokenOwner,info.data(),bytes,&bytes) &&
                    EqualSid(owner,reinterpret_cast<TOKEN_OWNER*>(info.data())->Owner);
            }
            if(process_token)CloseHandle(process_token);
            if(security)LocalFree(security);
        }
        if(!valid) {
            close_descriptor();
            throw ApiError(400,"invalid_request_error",
                parameter+" failed ownership, type, or size validation",parameter);
        }
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
#endif
        std::array<uint8_t, 64> header{};
        try {read_exact(0, header.data(), header.size(), parameter);}
        catch(...) {close_descriptor();throw;}
        constexpr std::array<uint8_t, 8> magic{
            'M', 'F', 'Q', 'M', 'M', '0', '1', 0};
        if (!std::equal(magic.begin(), magic.end(), header.begin()) ||
            !std::equal(token.begin(), token.end(), header.begin() + 8)) {
            close_descriptor();
            throw ApiError(
                400, "invalid_request_error",
                parameter + " header or token does not match", parameter);
        }
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
#ifdef _WIN32
        if(descriptor_!=INVALID_HANDLE_VALUE) {
            CloseHandle(descriptor_);descriptor_=INVALID_HANDLE_VALUE;
        }
#else
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
        LARGE_INTEGER position{};position.QuadPart=static_cast<LONGLONG>(offset);
        if(!SetFilePointerEx(descriptor_,position,nullptr,FILE_BEGIN))
            throw ApiError(400,"invalid_request_error",parameter+" could not be positioned",parameter);
        size_t completed=0;
        while(completed<length) {
            DWORD count=0;
            if(!ReadFile(descriptor_,static_cast<uint8_t*>(destination)+completed,
                    static_cast<DWORD>(length-completed),&count,nullptr) || !count)
                throw ApiError(400,"invalid_request_error",parameter+" could not be read completely",parameter);
            completed+=count;
        }
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
#ifdef _WIN32
    HANDLE descriptor_=INVALID_HANDLE_VALUE;
#else
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

static MfqMultimodalInput parse_deepseek_multimodal(
        const json & value,
        TensorFileReader * file_reader,
        bool v41) {
    const std::string label = v41 ? "DeepSeek-V4.1" : "DeepSeek-V4";
    for (const char * name : {"pixel_values", "patch_mask", "vision_grid"}) {
        if (!value.contains(name)) {
            throw ApiError(
                400, "invalid_request_error",
                label + " multimodal payload is missing " + name,
                "mfq_multimodal");
        }
    }
    MfqMultimodalInput result;
    result.processor = v41
        ? MfqMultimodalProcessor::deepseek_v41
        : MfqMultimodalProcessor::deepseek_v4;
    result.processor_name = v41 ? "deepseek_v41" : "deepseek_v4";
    auto pixels = decode_tensor<float>(
        value, "pixel_values", "float32", file_reader);
    result.pixel_values = std::move(pixels.first);
    result.pixel_shape = std::move(pixels.second);
    auto mask = decode_tensor<uint8_t>(
        value, "patch_mask", "uint8", file_reader);
    result.patch_mask = std::move(mask.first);
    result.patch_mask_shape = std::move(mask.second);
    auto grid = decode_tensor<int32_t>(
        value, "vision_grid", "int32", file_reader);
    result.vision_grid = std::move(grid.first);
    result.vision_grid_shape = std::move(grid.second);

    if (result.pixel_shape.size() != 3 ||
        result.pixel_shape[0] <= 0 || result.pixel_shape[0] > 128 ||
        result.pixel_shape[2] != 3 * 14 * 14 ||
        result.patch_mask_shape.size() != 2 ||
        result.vision_grid_shape.size() != 2 ||
        result.vision_grid_shape[1] != 4 ||
        result.pixel_shape[0] != result.patch_mask_shape[0] ||
        result.pixel_shape[0] != result.vision_grid_shape[0] ||
        result.pixel_shape[1] != result.patch_mask_shape[1]) {
        throw ApiError(
            400, "invalid_request_error",
            label + " multimodal tensor geometry is invalid",
            "mfq_multimodal");
    }
    if (!std::all_of(
            result.pixel_values.begin(), result.pixel_values.end(),
            [](float item) { return std::isfinite(item); })) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal pixel_values contains a non-finite value",
            "mfq_multimodal.pixel_values");
    }
    const int64_t images = result.pixel_shape[0];
    const int64_t padded_patches = result.pixel_shape[1];
    for (int64_t image = 0; image < images; ++image) {
        const int64_t vit_h = result.vision_grid[4 * image];
        const int64_t vit_w = result.vision_grid[4 * image + 1];
        const int64_t llm_h = result.vision_grid[4 * image + 2];
        const int64_t llm_w = result.vision_grid[4 * image + 3];
        if (vit_h <= 0 || vit_w <= 0 ||
            vit_h > std::numeric_limits<int64_t>::max() / vit_w ||
            vit_h * vit_w > padded_patches ||
            llm_h != (vit_h + 2) / 3 ||
            llm_w != (vit_w + 2) / 3) {
            throw ApiError(
                400, "invalid_request_error",
                label + " vision grid disagrees with patch geometry",
                "mfq_multimodal.vision_grid");
        }
        const int64_t active = vit_h * vit_w;
        for (int64_t patch = 0; patch < padded_patches; ++patch) {
            const uint8_t actual = result.patch_mask[
                static_cast<size_t>(image * padded_patches + patch)];
            if (actual > 1 || static_cast<bool>(actual) != (patch < active)) {
                throw ApiError(
                    400, "invalid_request_error",
                    label + " patch mask is not a contiguous active prefix",
                    "mfq_multimodal.patch_mask");
            }
        }
    }

    return result;
}

static bool valid_processor_identity(const std::string & value) {
    if (value.empty() || value.size() > 64) return false;
    return std::all_of(
        value.begin(), value.end(), [](unsigned char item) {
            return std::isalnum(item) || item == '_' || item == '-' || item == '.';
        });
}

static MfqMultimodalInput parse_grid_vision_multimodal(
        const json & value,
        TensorFileReader * file_reader) {
    if (!value.contains("processor") || !value["processor"].is_string()) {
        throw ApiError(
            400, "invalid_request_error",
            "grid-vision payload requires a processor identity",
            "mfq_multimodal.processor");
    }
    const auto processor = value["processor"].get<std::string>();
    if (!valid_processor_identity(processor)) {
        throw ApiError(
            400, "invalid_request_error",
            "grid-vision processor identity is invalid",
            "mfq_multimodal.processor");
    }

    auto pixels = decode_tensor<float>(
        value, "pixel_values", "float32", file_reader);
    auto grid = decode_tensor<int32_t>(
        value, "vision_grid_thw", "int32", file_reader);
    auto types = decode_tensor<int32_t>(
        value, "vision_types", "int32", file_reader);
    if (pixels.second.size() != 2 || pixels.second[0] <= 0 ||
        pixels.second[1] <= 0 || grid.second.size() != 2 ||
        grid.second[0] <= 0 || grid.second[1] != 3 ||
        types.second != std::vector<int64_t>{grid.second[0]}) {
        throw ApiError(
            400, "invalid_request_error",
            "grid-vision tensor geometry is invalid",
            "mfq_multimodal");
    }
    if (!std::all_of(
            pixels.first.begin(), pixels.first.end(),
            [](float item) { return std::isfinite(item); })) {
        throw ApiError(
            400, "invalid_request_error",
            "grid-vision pixels contain a non-finite value",
            "mfq_multimodal.pixel_values");
    }

    std::uint64_t patches = 0;
    std::vector<int32_t> derived_images;
    std::vector<int32_t> derived_videos;
    for (int64_t item = 0; item < grid.second[0]; ++item) {
        const auto offset = static_cast<size_t>(3 * item);
        const int32_t temporal = grid.first[offset];
        const int32_t height = grid.first[offset + 1];
        const int32_t width = grid.first[offset + 2];
        const int32_t type = types.first[static_cast<size_t>(item)];
        if (temporal <= 0 || height <= 0 || width <= 0 ||
            (type != 1 && type != 2)) {
            throw ApiError(
                400, "invalid_request_error",
                "grid-vision dimensions or modality type are invalid",
                "mfq_multimodal.vision_grid_thw");
        }
        const auto frame = static_cast<std::uint64_t>(temporal);
        const auto rows = static_cast<std::uint64_t>(height);
        const auto columns = static_cast<std::uint64_t>(width);
        if (rows > std::numeric_limits<std::uint64_t>::max() / columns ||
            frame > std::numeric_limits<std::uint64_t>::max() / (rows * columns) ||
            patches > std::numeric_limits<std::uint64_t>::max() -
                frame * rows * columns) {
            throw ApiError(
                400, "invalid_request_error",
                "grid-vision patch count overflows",
                "mfq_multimodal.vision_grid_thw");
        }
        patches += frame * rows * columns;
        auto & destination = type == 1 ? derived_images : derived_videos;
        destination.insert(
            destination.end(),
            grid.first.begin() + static_cast<std::ptrdiff_t>(offset),
            grid.first.begin() + static_cast<std::ptrdiff_t>(offset + 3));
        if (type == 1 && temporal != 1) {
            throw ApiError(
                400, "invalid_request_error",
                "grid-vision images must have temporal size one",
                "mfq_multimodal.vision_grid_thw");
        }
    }
    if (patches != static_cast<std::uint64_t>(pixels.second[0])) {
        throw ApiError(
            400, "invalid_request_error",
            "grid-vision patch count disagrees with pixel_values",
            "mfq_multimodal.pixel_values");
    }

    const auto optional_grid = [&](const char * name) {
        if (!value.contains(name)) {
            return std::pair<std::vector<int32_t>, std::vector<int64_t>>{
                {}, {0, 3}};
        }
        auto result = decode_tensor<int32_t>(
            value, name, "int32", file_reader);
        if (result.second.size() != 2 || result.second[1] != 3) {
            throw ApiError(
                400, "invalid_request_error",
                std::string(name) + " must have [items,3] shape",
                std::string("mfq_multimodal.") + name);
        }
        return result;
    };
    auto images = optional_grid("image_grid_thw");
    auto videos = optional_grid("video_grid_thw");
    if (images.first != derived_images || videos.first != derived_videos) {
        throw ApiError(
            400, "invalid_request_error",
            "grid-vision split grids disagree with combined media order",
            "mfq_multimodal");
    }

    MfqMultimodalInput result;
    result.processor = MfqMultimodalProcessor::grid_vision;
    result.processor_name = processor;
    result.pixel_values = std::move(pixels.first);
    result.pixel_shape = std::move(pixels.second);
    result.vision_grid = std::move(grid.first);
    result.vision_grid_shape = std::move(grid.second);
    result.vision_types = std::move(types.first);
    result.image_grid = std::move(images.first);
    result.image_grid_shape = std::move(images.second);
    result.video_grid = std::move(videos.first);
    result.video_grid_shape = std::move(videos.second);
    return result;
}

MfqMultimodalInput parse_mfq_vision(const json & value) {
    if (!value.is_object()) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal must be an object", "mfq_multimodal");
    }
    if (!value.contains("version") || !value["version"].is_number_integer()) {
        throw ApiError(
            400, "invalid_request_error",
            "mfq_multimodal.version must be an integer", "mfq_multimodal.version");
    }

    std::unique_ptr<TensorFileReader> file_reader;
    if (value.contains("binary_file")) {
        file_reader = std::make_unique<TensorFileReader>(value);
    }
    const int version = value["version"].get<int>();
    if (version == 3) {
        return parse_grid_vision_multimodal(value, file_reader.get());
    }
    if (version == 2) {
        if (!value.contains("processor") ||
            !value["processor"].is_string()) {
            throw ApiError(
                400, "invalid_request_error",
                "mfq_multimodal version 2 requires a DeepSeek processor",
                "mfq_multimodal.processor");
        }
        const auto processor = value["processor"].get<std::string>();
        if (processor != "deepseek_v4" && processor != "deepseek_v41") {
            throw ApiError(
                400, "invalid_request_error",
                "unsupported DeepSeek multimodal processor",
                "mfq_multimodal.processor");
        }
        return parse_deepseek_multimodal(
            value, file_reader.get(), processor == "deepseek_v41");
    }
    if (version != 1) {
        throw ApiError(
            400, "invalid_request_error",
            "unsupported mfq_multimodal.version", "mfq_multimodal.version");
    }

    MfqMultimodalInput result;
    result.processor = MfqMultimodalProcessor::minicpmo;
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
        for (const int64_t length : result.audio_lengths) {
            if (length < 9 || length > result.audio_features_shape[2]) {
                throw ApiError(
                    400, "invalid_request_error",
                    "mfq_multimodal audio length is out of range",
                    "mfq_multimodal.audio_lengths");
            }
        }
    }
    return result;
}

std::vector<float> decode_audio_features(
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

} // namespace mfq::transport_detail

MfqRuntimeProfile resolve_mfq_runtime_profile(
        const std::string & mfq_path,
        const std::string & model_architecture,
        const std::string & model_type,
        const std::string & model_name,
        const std::string & embedded_profile_json,
        const std::string & model_config_json,
        const std::string & explicit_profile_path) {
    using namespace mfq::transport_detail;
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
            read_profile_file(path), "runtime-explicit:" + path.filename().string()));
    }
    return result;
}

namespace mfq::transport_detail {
CompletionStream::CompletionStream(const MfqScheduler& scheduler, const RequestWork& work, std::string id,
                                   RequestSchedule schedule)
    : scheduler_(scheduler), id_(std::move(id)),
      request_(scheduler_.submit(mfq::engine::EngineRequest{
          id_, work, std::move(schedule.deadline), schedule.priority})) {}
CompletionStream::~CompletionStream() {
    if (!terminal_) try { cancel(); } catch (...) {}
}
void CompletionStream::cancel() {
    scheduler_.cancel_request(id_);
    result.client_connected = false;
}
std::optional<std::vector<common_chat_msg_diff>> CompletionStream::next() {
    if (terminal_) return {};
    std::vector<common_chat_msg_diff> diffs;
    for (auto& event : request_->wait()) {
        std::visit([&](auto& data) {
            using T = std::decay_t<decltype(data)>;
            if constexpr (std::is_same_v<T, mfq::engine::OutputDelta>) {
                if (!data.token_ids.empty()) {
                    if (!generated_tokens_) first_token_ = std::chrono::steady_clock::now();
                    generated_tokens_ += data.token_ids.size();
                }
                for (const auto& diff : data.diffs) {
                    result.text += diff.content_delta;
                    result.reasoning_text += diff.reasoning_content_delta;
                    if (diff.tool_call_index == std::string::npos) continue;
                    if (result.tool_calls.size() <= diff.tool_call_index)
                        result.tool_calls.resize(diff.tool_call_index + 1);
                    auto& call = result.tool_calls[diff.tool_call_index];
                    call.id += diff.tool_call_delta.id;
                    call.name += diff.tool_call_delta.name;
                    call.arguments += diff.tool_call_delta.arguments;
                }
                diffs.insert(diffs.end(), std::make_move_iterator(data.diffs.begin()),
                             std::make_move_iterator(data.diffs.end()));
            } else if constexpr (std::is_same_v<T, mfq::engine::PrefillProgress>) {
                const auto & timing = data.timing;
                const double interval = timing.llm_ms - prefill_ms_;
                if (interval > 0 && timing.prompt_tokens >= processed_tokens_)
                    prefill_tps_ = 1000.0 * (timing.prompt_tokens - processed_tokens_) / interval;
                processed_tokens_ = timing.prompt_tokens;
                prefill_ms_ = timing.llm_ms;
            } else if constexpr (std::is_same_v<T, mfq::engine::Completed> ||
                                 std::is_same_v<T, mfq::engine::Cancelled>) {
                result.completion_tokens = static_cast<std::int32_t>(data.usage.completion_tokens);
                prompt_tokens = data.usage.prompt_tokens;
                result.cancelled = std::is_same_v<T, mfq::engine::Cancelled>;
                if constexpr (std::is_same_v<T, mfq::engine::Completed>) result.finish_reason = std::move(data.finish_reason);
                else result.finish_reason = "cancelled";
                metrics = data.metrics; terminal_ = true;
            } else if constexpr (std::is_same_v<T, mfq::engine::Failed>) {
                terminal_ = true;
                if (data.code == "invalid_request") throw ApiError(400, "invalid_request_error", data.message);
                if (data.code == "unsupported_input") throw ApiError(501, "unsupported_parameter", data.message);
                if (data.code == "resource_exhausted") throw ApiError(503, "resource_exhausted", data.message);
                // A request deadline is absolute: the client cannot satisfy it
                // by retrying, so it is a distinct terminal instead of 502.
                if (data.code == "deadline_exceeded") throw ApiError(504, "deadline_exceeded", data.message);
                throw std::runtime_error(data.message);
            }
        }, event.data);
    }
    return diffs;
}

json CompletionStream::progress() const {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - started_).count();
    const double decode = generated_tokens_ ? std::chrono::duration<double>(now - first_token_).count() : 0;
    return {{"phase", generated_tokens_ ? "generating" : "prefilling"},
        {"processed_tokens", processed_tokens_}, {"generated_tokens", generated_tokens_},
        {"prefill_tps", prefill_tps_}, {"elapsed_seconds", elapsed},
        {"decode_tps", decode > 0 ? (generated_tokens_ - 1) / decode : 0},
        {"updated_at", unix_time_seconds()}};
}

} // namespace mfq::transport_detail
