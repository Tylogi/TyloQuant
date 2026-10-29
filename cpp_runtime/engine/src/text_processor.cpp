#include "text_processor.h"
#include "mfq/model_source.h"

#include "token_constraint.h"
#include "tokenizer.h"

#include "chat/common.h"
#include "json-schema-to-grammar.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <mutex>
#include <utility>

namespace mfq::engine {
namespace {

using json = nlohmann::ordered_json;

std::string normalized_identity(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return std::isalnum(c)
                ? static_cast<char>(std::tolower(c))
                : '_';
        });
    return value;
}

std::int64_t special_token(
        const MfqTokenizer& tokenizer,
        const std::string& marker) {
    const auto tokens = tokenizer.tokenize(marker, true, false);
    if (tokens.size() != 1) {
        throw std::runtime_error(
            "model tokenizer does not encode " + marker +
            " as one special token");
    }
    return tokens.front();
}

struct DeepseekImageBlock {
    std::vector<std::int64_t> types;
    std::vector<std::int64_t> permutation;
};

DeepseekImageBlock deepseek_v4_image_block(
        std::int64_t height,
        std::int64_t width,
        std::size_t start_position) {
    constexpr std::int64_t image_start = 0;
    constexpr std::int64_t image_pad = 1;
    constexpr std::int64_t image = 2;
    constexpr std::int64_t image_newline = 3;
    constexpr std::int64_t image_end = 4;
    const std::int64_t padded_height = height + height % 2;
    const std::int64_t row_length = width + 1;
    const std::int64_t compress_pad =
        3 - static_cast<std::int64_t>(start_position % 4);
    const std::int64_t trailing_pad =
        ((padded_height / 2 * row_length) % 2) * 2;
    DeepseekImageBlock result;
    result.types.insert(
        result.types.end(), static_cast<std::size_t>(compress_pad), image_pad);
    result.types.push_back(image_start);
    for (std::int64_t pair = 0; pair < padded_height / 2; ++pair) {
        for (std::int64_t column = 0; column < row_length; ++column) {
            for (std::int64_t within = 0; within < 2; ++within) {
                const std::int64_t row = pair * 2 + within;
                if (row < height && column < width) {
                    result.types.push_back(image);
                    result.permutation.push_back(row * width + column);
                } else if (row < height) {
                    result.types.push_back(image_newline);
                } else {
                    result.types.push_back(image_pad);
                }
            }
        }
    }
    result.types.insert(
        result.types.end(), static_cast<std::size_t>(trailing_pad), image_pad);
    result.types.push_back(image_end);
    return result;
}

DeepseekImageBlock deepseek_v41_image_block(
        std::int64_t height,
        std::int64_t width) {
    constexpr std::int64_t image_start = 0;
    constexpr std::int64_t image = 1;
    constexpr std::int64_t image_newline = 2;
    constexpr std::int64_t image_end = 3;
    DeepseekImageBlock result;
    result.types.push_back(image_start);
    for (std::int64_t row = 0; row < height; ++row) {
        result.types.insert(
            result.types.end(), static_cast<std::size_t>(width), image);
        result.types.push_back(image_newline);
    }
    result.types.push_back(image_end);
    return result;
}

void prepare_deepseek_media(
        const MfqTokenizer& tokenizer,
        std::int32_t vocabulary,
        std::vector<std::int64_t>& prompt,
        MfqMultimodalInput& media,
        bool v41) {
    const std::int64_t placeholder = special_token(
        tokenizer, "<｜deepseek_image｜>");
    const std::int64_t images = media.pixel_shape.at(0);
    std::vector<std::int64_t> expanded;
    expanded.reserve(
        prompt.size() + static_cast<std::size_t>(images) *
            (v41 ? 1024 : 384));
    media.image_bounds.clear();
    media.image_permutation.clear();
    media.image_permutation_offsets = {0};
    std::int64_t source = 0;
    for (const std::int64_t token : prompt) {
        if (token != placeholder) {
            expanded.push_back(token);
            continue;
        }
        if (source >= images) {
            throw InferenceInputError(
                InferenceInputErrorCode::Invalid,
                "image placeholders exceed processed images", "messages");
        }
        const std::int64_t height = media.vision_grid[4 * source + 2];
        const std::int64_t width = media.vision_grid[4 * source + 3];
        auto block = v41
            ? deepseek_v41_image_block(height, width)
            : deepseek_v4_image_block(height, width, expanded.size());
        const std::int64_t begin =
            static_cast<std::int64_t>(expanded.size());
        for (const std::int64_t type : block.types) {
            expanded.push_back(v41 ? placeholder : vocabulary + type);
        }
        media.image_bounds.insert(
            media.image_bounds.end(),
            {0, source, begin, static_cast<std::int64_t>(expanded.size())});
        media.image_permutation.insert(
            media.image_permutation.end(),
            block.permutation.begin(), block.permutation.end());
        media.image_permutation_offsets.push_back(
            static_cast<std::int64_t>(media.image_permutation.size()));
        ++source;
    }
    if (source != images) {
        throw InferenceInputError(
            InferenceInputErrorCode::Invalid,
            "image placeholders do not match processed images", "messages");
    }
    prompt = std::move(expanded);
}

void prepare_minicpmo_media(
        const MfqTokenizer& tokenizer,
        const std::vector<std::int64_t>& prompt,
        MfqMultimodalInput& media) {
    media.image_bounds.clear();
    if (!media.pixel_shape.empty()) {
        const auto image_start = special_token(tokenizer, "<image>");
        const auto image_end = special_token(tokenizer, "</image>");
        const auto slice_start = special_token(tokenizer, "<slice>");
        const auto slice_end = special_token(tokenizer, "</slice>");
        for (std::size_t index = 0; index < prompt.size(); ++index) {
            const auto token = prompt[index];
            if (token != image_start && token != slice_start) continue;
            const auto end_token = token == image_start ? image_end : slice_end;
            const auto found = std::find(
                prompt.begin() + static_cast<std::ptrdiff_t>(index + 1),
                prompt.end(), end_token);
            if (found == prompt.end()) {
                throw InferenceInputError(
                    InferenceInputErrorCode::Invalid,
                    "MiniCPM-o image placeholder is missing its end token",
                    "messages");
            }
            const auto end = static_cast<std::size_t>(found - prompt.begin());
            if (end - index - 1 != 64) {
                throw InferenceInputError(
                    InferenceInputErrorCode::Invalid,
                    "MiniCPM-o image placeholder must contain 64 query tokens",
                    "messages");
            }
            const auto source = static_cast<std::int64_t>(
                media.image_bounds.size() / 4);
            media.image_bounds.insert(
                media.image_bounds.end(),
                {0, source, static_cast<std::int64_t>(index + 1),
                 static_cast<std::int64_t>(end)});
            index = end;
        }
        if (media.image_bounds.size() / 4 !=
                static_cast<std::size_t>(media.pixel_shape.at(0))) {
            throw InferenceInputError(
                InferenceInputErrorCode::Invalid,
                "MiniCPM-o image placeholders do not match processed image slices",
                "messages");
        }
    }

    media.audio_bounds.clear();
    if (!media.audio_features_shape.empty()) {
        std::vector<std::int64_t> pooled_lengths;
        pooled_lengths.reserve(media.audio_lengths.size());
        for (const auto length : media.audio_lengths) {
            const auto after_convolution = (length - 1) / 2 + 1;
            pooled_lengths.push_back((after_convolution - 5) / 5 + 1);
        }
        const auto audio_start = special_token(tokenizer, "<|audio_start|>");
        const auto audio_end = special_token(tokenizer, "<|audio_end|>");
        for (std::size_t index = 0; index < prompt.size(); ++index) {
            if (prompt[index] != audio_start) continue;
            const auto found = std::find(
                prompt.begin() + static_cast<std::ptrdiff_t>(index + 1),
                prompt.end(), audio_end);
            if (found == prompt.end()) {
                throw InferenceInputError(
                    InferenceInputErrorCode::Invalid,
                    "MiniCPM-o audio placeholder is missing its end token",
                    "messages");
            }
            const auto end = static_cast<std::size_t>(found - prompt.begin());
            const auto source = media.audio_bounds.size() / 4;
            if (source >= pooled_lengths.size() ||
                    static_cast<std::int64_t>(end - index - 1) !=
                        pooled_lengths[source]) {
                throw InferenceInputError(
                    InferenceInputErrorCode::Invalid,
                    "MiniCPM-o audio placeholder does not match pooled audio length",
                    "messages");
            }
            media.audio_bounds.insert(
                media.audio_bounds.end(),
                {0, static_cast<std::int64_t>(source),
                 static_cast<std::int64_t>(index + 1),
                 static_cast<std::int64_t>(end)});
            index = end;
        }
        if (media.audio_bounds.size() / 4 != pooled_lengths.size()) {
            throw InferenceInputError(
                InferenceInputErrorCode::Invalid,
                "MiniCPM-o audio placeholders do not match processed audio chunks",
                "messages");
        }
    }
}

void prepare_media(
        const MfqTokenizer& tokenizer,
        std::int32_t vocabulary,
        std::vector<std::int64_t>& prompt,
        MfqMultimodalInput& media) {
    switch (media.processor) {
        case MfqMultimodalProcessor::minicpmo:
            prepare_minicpmo_media(tokenizer, prompt, media);
            return;
        case MfqMultimodalProcessor::deepseek_v4:
            prepare_deepseek_media(
                tokenizer, vocabulary, prompt, media, false);
            return;
        case MfqMultimodalProcessor::deepseek_v41:
            prepare_deepseek_media(
                tokenizer, vocabulary, prompt, media, true);
            return;
        case MfqMultimodalProcessor::grid_vision:
            return;
    }
}

} // namespace

InferenceInputError::InferenceInputError(
        InferenceInputErrorCode input_code,
        std::string message,
        std::string input_field)
    : std::runtime_error(std::move(message)),
      code(input_code),
      field(std::move(input_field)) {}

struct TextProcessor::Impl {
    Impl(
            std::unique_ptr<MfqTokenizer> input_tokenizer,
            std::int32_t expected_vocabulary,
            std::string input_model_type)
        : tokenizer(std::move(input_tokenizer)),
          model_type(std::move(input_model_type)) {
        if (expected_vocabulary > 0 &&
                tokenizer->vocab_size() != expected_vocabulary) {
            throw std::runtime_error(
                "tokenizer/model vocabulary mismatch: tokenizer=" +
                std::to_string(tokenizer->vocab_size()) + " model=" +
                std::to_string(expected_vocabulary));
        }
        const auto source = tokenizer->chat_template();
        if (!source.empty()) {
            templates = common_chat_templates_init(tokenizer->context(), "");
            if (!templates) {
                throw std::runtime_error(
                    "cannot initialize tokenizer.chat_template");
            }
        }
        capabilities.thinking =
            source.find("enable_thinking") != std::string::npos;
        if (source.find("reasoning_effort") != std::string::npos) {
            for (const char* value : {"high", "max"}) {
                if (source.find(std::string("'") + value + "'") !=
                        std::string::npos ||
                    source.find(std::string("\"") + value + "\"") !=
                        std::string::npos) {
                    capabilities.reasoning_effort_values.emplace_back(value);
                }
            }
        }
    }

    std::unique_ptr<MfqTokenizer> tokenizer;
    common_chat_templates_ptr templates;
    std::string model_type;
    ChatTemplateCapabilities capabilities;
    std::map<std::string, std::pair<std::string, bool>> prefix_policies;
    std::mutex prefix_policy_mutex;

    std::pair<std::string, bool> prefix_policy(const common_chat_templates_inputs& inputs) {
        std::lock_guard<std::mutex> lock(prefix_policy_mutex);
        json key{{"kwargs", inputs.chat_template_kwargs}, {"thinking", inputs.enable_thinking},
            {"tool_choice", inputs.tool_choice}, {"parallel", inputs.parallel_tool_calls}};
        for (const auto& tool : inputs.tools)
            key["tools"].push_back({tool.name, tool.description, tool.parameters});
        const auto signature = key.dump();
        if (const auto found = prefix_policies.find(signature); found != prefix_policies.end())
            return found->second;
        std::pair<std::string, bool> policy;
        try {
            auto probe = inputs;
            probe.messages = {common_chat_msg{"user", "probe"}};
            probe.continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
            probe.add_generation_prompt = true;
            const auto with = common_chat_templates_apply(templates.get(), probe).prompt;
            probe.add_generation_prompt = false;
            const auto without = common_chat_templates_apply(templates.get(), probe).prompt;
            if (with.starts_with(without)) {
                policy.first = with.substr(without.size());
                probe.messages.push_back(common_chat_msg{"assistant", "reply"});
                probe.messages.push_back(common_chat_msg{"user", "next"});
                policy.second = common_chat_templates_apply(templates.get(), probe).prompt.starts_with(with);
            }
        } catch (const std::exception&) {
        }
        if (prefix_policies.size() >= 16) prefix_policies.clear();
        prefix_policies.emplace(signature, policy);
        return policy;
    }
};

std::unique_ptr<TextProcessor> TextProcessor::load(const ModelSource& source,
        const std::string& tokenizer_path, std::int32_t vocabulary, const std::string& model_type) {
    if (source.has_asset(kTokenizerGgufAsset)) {
        const auto bytes = source.read_asset(kTokenizerGgufAsset);
        return std::make_unique<TextProcessor>(std::vector<std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(bytes.data()),
            reinterpret_cast<const std::uint8_t*>(bytes.data()) + bytes.size()), vocabulary, model_type);
    }
    return std::make_unique<TextProcessor>(tokenizer_path, vocabulary, model_type);
}

TextProcessor::TextProcessor(
        const std::string& tokenizer_path,
        std::int32_t expected_vocabulary,
        std::string model_type)
    : impl_(std::make_unique<Impl>(
          std::make_unique<MfqTokenizer>(tokenizer_path),
          expected_vocabulary, std::move(model_type))) {}

TextProcessor::TextProcessor(
        const std::vector<std::uint8_t>& tokenizer_gguf,
        std::int32_t expected_vocabulary,
        std::string model_type)
    : impl_(std::make_unique<Impl>(
          std::make_unique<MfqTokenizer>(tokenizer_gguf),
          expected_vocabulary, std::move(model_type))) {}

TextProcessor::~TextProcessor() = default;

std::int32_t TextProcessor::vocab_size() const {
    return impl_->tokenizer->vocab_size();
}

ChatTemplateCapabilities TextProcessor::chat_template_capabilities() const {
    return impl_->capabilities;
}

InferenceRequest TextProcessor::prepare(
        InferenceInput input,
        std::int64_t max_context) const {
    InferenceRequest work;
    work.chat = input.chat;
    work.stream = input.stream;
    work.include_usage = input.include_usage;
    work.sampling = input.sampling;
    work.cache_plan = std::move(input.cache_plan);
    work.stops = std::move(input.stops);

    std::string prompt = std::move(input.prompt);
    bool parse_special = false;
    if (input.chat) {
        if (!input.chat_input) {
            throw InferenceInputError(
                InferenceInputErrorCode::Invalid,
                "chat input is missing", "messages");
        }
        auto& chat = *input.chat_input;
        work.chat_parser.reasoning_format = chat.reasoning_format;
        work.chat_parser.reasoning_in_content =
            input.stream && chat.reasoning_format ==
                COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
        if (chat.preformatted_prompt) {
            prompt = *chat.preformatted_prompt;
            work.cache_plan.cache_output_tokens = true;
            work.chat_parser.parse_tool_calls = false;
            if (!chat.template_inputs.json_schema.empty()) {
                common_chat_params constraint;
                constraint.grammar = json_schema_to_grammar(
                    json::parse(chat.template_inputs.json_schema));
                work.token_constraint = make_chat_token_constraint(
                    *impl_->tokenizer, constraint);
            }
        } else {
            if (!impl_->templates) {
                throw InferenceInputError(
                    InferenceInputErrorCode::Unsupported,
                    "the tokenizer has no chat template; provide input.preformatted_prompt",
                    "messages");
            }
            auto inputs = std::move(chat.template_inputs);
            if (chat.parallel_tool_calls) {
                inputs.parallel_tool_calls = *chat.parallel_tool_calls;
            } else {
                inputs.parallel_tool_calls = common_chat_templates_get_caps(
                    impl_->templates.get()).at("supports_parallel_tool_calls");
            }
            common_chat_params params;
            try {
                params = common_chat_templates_apply(
                    impl_->templates.get(), inputs);
            } catch (const std::exception& error) {
                throw InferenceInputError(
                    InferenceInputErrorCode::Invalid,
                    std::string("chat template application failed: ") +
                        error.what(),
                    "messages");
            }
            prompt = params.prompt;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_NONE && inputs.add_generation_prompt) {
                const auto policy = impl_->prefix_policy(inputs);
                const bool open_think = !params.thinking_start_tag.empty() &&
                    policy.first.find(params.thinking_start_tag) != std::string::npos &&
                    (params.thinking_end_tag.empty() || policy.first.find(params.thinking_end_tag) == std::string::npos);
                work.cache_plan.cache_output_tokens = policy.second && !open_think;
                if (!work.cache_plan.cache_output_tokens && !policy.first.empty() && prompt.ends_with(policy.first))
                    work.cache_plan.stable_prefix_tokens = impl_->tokenizer->tokenize(
                        prompt.substr(0, prompt.size() - policy.first.size()), true).size();
            }
            work.token_constraint = make_chat_token_constraint(
                *impl_->tokenizer, params);
            work.chat_parser.format = params.format;
            work.chat_parser.generation_prompt = params.generation_prompt;
            work.chat_parser.parse_tool_calls = true;
            work.chat_parser.is_continuation =
                inputs.continue_final_message !=
                COMMON_CHAT_CONTINUATION_NONE;
            if (!params.parser.empty()) {
                work.chat_parser.parser.load(params.parser);
            }
            for (const auto& text : params.preserved_tokens) {
                const auto tokens = impl_->tokenizer->tokenize(text, true);
                work.preserved_tokens.insert(tokens.begin(), tokens.end());
            }
            work.stops.insert(
                work.stops.end(),
                params.additional_stops.begin(), params.additional_stops.end());
        }
        parse_special = true;
    }

    work.prompt = impl_->tokenizer->tokenize(prompt, parse_special);
    if (work.prompt.empty()) {
        throw InferenceInputError(
            InferenceInputErrorCode::Invalid,
            "prompt tokenized to an empty sequence", "prompt");
    }
    if (input.benchmark_prompt_tokens) {
        work.benchmark_fixed_tokens = true;
        const auto count = *input.benchmark_prompt_tokens;
        if (count > static_cast<std::size_t>(max_context) ||
            count + work.sampling.max_tokens > static_cast<std::size_t>(max_context)) {
            throw InferenceInputError(InferenceInputErrorCode::Invalid,
                "benchmark prompt and output exceed the loaded context capacity", "prompt");
        }
        const auto source = work.prompt;
        work.prompt.resize(count);
        for (std::size_t i = source.size(); i < count; ++i)
            work.prompt[i] = source[i % source.size()];
    }
    if (work.cache_plan.stable_prefix_tokens == 0)
        work.cache_plan.stable_prefix_tokens = work.prompt.size();
    if (!input.chat) work.cache_plan.cache_output_tokens = true;
    if (input.chat && input.chat_input &&
            normalized_identity(impl_->model_type).rfind(
                "deepseek_v4", 0) == 0 &&
            input.chat_input->template_inputs.add_generation_prompt) {
        const std::string marker = work.sampling.enable_thinking
            ? "<think>" : "</think>";
        const auto tokens = impl_->tokenizer->tokenize(marker, true);
        if (!tokens.empty() && tokens.size() < work.prompt.size() &&
                std::equal(
                    tokens.rbegin(), tokens.rend(), work.prompt.rbegin())) {
            work.cache_plan.stable_prefix_tokens =
                work.prompt.size() - tokens.size();
        }
    }

    if (!input.prefix_cache_enabled) {
        work.cache_plan.stable_prefix_tokens = 0;
        work.cache_plan.cache_output_tokens = false;
    }
    if (input.media) {
        const auto suffix_tokens = work.prompt.size() - std::min(
            work.cache_plan.stable_prefix_tokens, work.prompt.size());
        prepare_media(
            *impl_->tokenizer, impl_->tokenizer->vocab_size(),
            work.prompt, *input.media);
        if (work.cache_plan.stable_prefix_tokens > 0)
            work.cache_plan.stable_prefix_tokens = work.prompt.size() - suffix_tokens;
        work.vision = std::move(input.media);
    }
    if (max_context > 0 &&
            static_cast<std::int64_t>(work.prompt.size()) +
                    work.sampling.max_tokens > max_context) {
        throw InferenceInputError(
            InferenceInputErrorCode::ContextLength,
            input.media
                ? "expanded multimodal prompt plus max_tokens exceed the model context window"
                : "prompt tokens plus max_tokens exceed the model context window",
            "max_tokens");
    }

    std::vector<std::string> unique_stops;
    unique_stops.reserve(work.stops.size());
    for (auto& stop : work.stops) {
        if (std::find(unique_stops.begin(), unique_stops.end(), stop) ==
                unique_stops.end()) {
            unique_stops.push_back(std::move(stop));
        }
    }
    work.stops = std::move(unique_stops);
    return work;
}

std::size_t TextProcessor::count_tokens(const std::string& prompt) const {
    if (prompt.empty() || prompt.size() > 262144)
        throw std::invalid_argument("token count requires a nonempty prompt of at most 262144 bytes");
    return impl_->tokenizer->tokenize(prompt, false, false).size();
}

const MfqTokenizer& TextProcessor::tokenizer() const {
    return *impl_->tokenizer;
}

void TextProcessor::prepare_duplex_session(
        const std::string& system_prompt,
        MfqDuplexSessionParams& parameters) const {
    parameters.system_prefix = impl_->tokenizer->tokenize(
        "<|im_start|>system\n" + system_prompt + "\n<|audio_start|>",
        true, false);
    parameters.system_suffix = impl_->tokenizer->tokenize(
        "<|audio_end|><|im_end|>", true, false);
    parameters.special_ids = {
        impl_->tokenizer->special_token_id("<unit>"),
        impl_->tokenizer->special_token_id("</unit>"),
        impl_->tokenizer->special_token_id("<image>"),
        impl_->tokenizer->special_token_id("</image>"),
        impl_->tokenizer->special_token_id("<slice>"),
        impl_->tokenizer->special_token_id("</slice>"),
        impl_->tokenizer->special_token_id("<|listen|>"),
        impl_->tokenizer->special_token_id("<|speak|>"),
        impl_->tokenizer->special_token_id("<|tts_bos|>"),
        impl_->tokenizer->special_token_id("<|tts_eos|>"),
        impl_->tokenizer->special_token_id("<|chunk_eos|>"),
        impl_->tokenizer->special_token_id("<|chunk_tts_eos|>"),
        impl_->tokenizer->special_token_id("<|turn_eos|>"),
        impl_->tokenizer->special_token_id("<|tts_pad|>"),
        151687,
    };
    parameters.forbidden_ids = {
        impl_->tokenizer->special_token_id("<|tts_pad|>"),
    };
}

void TextProcessor::prepare_duplex_step(
        const std::string& text,
        MfqDuplexStepInput& step) const {
    step.text_tokens = impl_->tokenizer->tokenize(text, false, false);
    if (step.text_tokens.empty()) {
        throw InferenceInputError(
            InferenceInputErrorCode::Invalid,
            "input.text produced no tokens", "text");
    }
}

std::string TextProcessor::decode_tokens(
        const std::vector<std::int64_t>& tokens,
        const std::unordered_set<std::int64_t>& excluded) const {
    std::string result;
    for (const auto token : tokens) {
        if (excluded.count(token) == 0) {
            result += impl_->tokenizer->piece(token, false);
        }
    }
    return result;
}

} // namespace mfq::engine
