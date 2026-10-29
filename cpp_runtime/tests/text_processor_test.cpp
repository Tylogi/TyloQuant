#include "text_processor.h"
#include "gguf.h"
#include "mfq_text.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct TokenizerFixture {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("mfq-text-prefix-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    int vocabulary = 0;

    TokenizerFixture() {
        std::vector<std::string> names{"<unk>", "<s>", "</s>"};
        std::vector<std::int32_t> types{MFQ_TEXT_TOKEN_TYPE_UNKNOWN,
            MFQ_TEXT_TOKEN_TYPE_CONTROL, MFQ_TEXT_TOKEN_TYPE_CONTROL};
        for (int value = 0; value < 256; ++value) {
            char name[7];
            std::snprintf(name, sizeof(name), "<0x%02X>", value);
            names.emplace_back(name);
            types.push_back(MFQ_TEXT_TOKEN_TYPE_BYTE);
        }
        for (const auto* name : {"<｜deepseek_image｜>", "<think>", "</think>"}) {
            names.emplace_back(name);
            types.push_back(MFQ_TEXT_TOKEN_TYPE_CONTROL);
        }
        vocabulary = static_cast<int>(names.size());
        std::vector<const char*> tokens;
        for (const auto& name : names) tokens.push_back(name.c_str());
        std::vector<float> scores(names.size());
        auto* metadata = gguf_init_empty();
        require(metadata != nullptr, "cannot create tokenizer fixture");
        gguf_set_val_str(metadata, "general.architecture", "mfq-test");
        gguf_set_val_str(metadata, "tokenizer.ggml.model", "llama");
        gguf_set_arr_str(metadata, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
        gguf_set_arr_data(metadata, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
        gguf_set_arr_data(metadata, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
        gguf_set_val_u32(metadata, "tokenizer.ggml.bos_token_id", 1);
        gguf_set_val_u32(metadata, "tokenizer.ggml.eos_token_id", 2);
        gguf_set_val_bool(metadata, "tokenizer.ggml.add_space_prefix", false);
        gguf_set_val_str(metadata, "tokenizer.chat_template", "{{ messages[0]['content'] }}");
        const bool written = gguf_write_to_file(metadata, path.string().c_str(), true);
        gguf_free(metadata);
        require(written, "cannot write tokenizer fixture");
    }
    ~TokenizerFixture() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

} // namespace

int main() try {
    using namespace mfq::engine;
    TokenizerFixture fixture;
    TextProcessor processor(fixture.path.string(), fixture.vocabulary, "deepseek_v41");
    for (const auto& prompt : std::vector<std::string>{"abc", "<s>abc</s>", std::string(4096, 'a')})
        require(processor.count_tokens(prompt) == processor.tokenizer().tokenize(prompt, false, false).size(),
            "token count differs from raw scoring tokenization");
    require(processor.count_tokens("<s>") > 1, "token count parsed a special token");
    for (const std::string prompt : {std::string(), std::string(262145, 'a')}) {
        bool rejected = false;
        try { (void)processor.count_tokens(prompt); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "token count accepted an unbounded prompt");
    }
    for (int images : {0, 1, 2}) for (bool enabled : {false, true}) {
        InferenceInput input;
        input.chat = true;
        input.sampling.max_tokens = 1;
        input.sampling.enable_thinking = false;
        input.prefix_cache_enabled = enabled;
        input.chat_input.emplace();
        std::string prompt = "a";
        for (int i = 0; i < images; ++i) prompt += "<｜deepseek_image｜>b";
        prompt += "</think>";
        input.chat_input->preformatted_prompt = prompt;
        input.chat_input->template_inputs.add_generation_prompt = true;
        if (images) {
            input.media.emplace();
            input.media->processor = MfqMultimodalProcessor::deepseek_v41;
            input.media->processor_name = "deepseek_v41";
            input.media->pixel_shape = {images, 18, 588};
            input.media->pixel_values.resize(images * 18 * 588);
            input.media->vision_grid_shape = {images, 4};
            for (int i = 0; i < images; ++i)
                input.media->vision_grid.insert(input.media->vision_grid.end(), {3, 6, 1, 2});
        }
        const auto result = processor.prepare(std::move(input), 1024);
        require(result.cache_plan.stable_prefix_tokens == (enabled ? result.prompt.size() - 1 : 0),
            "image expansion moved the stable prefix away from its token boundary");
        if (images) {
            require(result.vision && result.vision->image_bounds.size() == static_cast<std::size_t>(images * 4),
                "image expansion lost image bounds");
            require(result.cache_plan.stable_prefix_tokens == 0 ||
                    result.cache_plan.stable_prefix_tokens >= static_cast<std::size_t>(result.vision->image_bounds.back()),
                "stable prefix does not include expanded image spans");
        }
    }
    for(bool benchmark:{false,true}) {
        InferenceInput input;input.chat=false;input.prompt="abc";
        input.sampling.max_tokens=4;input.prefix_cache_enabled=false;
        if(benchmark)input.benchmark_prompt_tokens=32;
        auto prepared=processor.prepare(std::move(input),1024);
        require(prepared.benchmark_fixed_tokens==benchmark,"fixed output length leaked outside synthetic benchmark");
        if(benchmark)require(prepared.prompt.size()==32,"benchmark prompt length differs");
    }
    std::cout << "MFQ text processor media-prefix boundaries passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
