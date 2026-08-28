// SPDX-License-Identifier: Apache-2.0

#include "chat.h"
#include "gguf.h"
#include "mfq_text.h"
#include "mfq_grammar.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

std::filesystem::path write_tokenizer_fixture() {
    std::vector<std::string> token_storage = {"<unk>", "<s>", "</s>"};
    std::vector<int32_t> token_types = {
        MFQ_TEXT_TOKEN_TYPE_UNKNOWN,
        MFQ_TEXT_TOKEN_TYPE_CONTROL,
        MFQ_TEXT_TOKEN_TYPE_CONTROL,
    };
    for (int value = 0; value < 256; ++value) {
        char token[7];
        std::snprintf(token, sizeof(token), "<0x%02X>", value);
        token_storage.emplace_back(token);
        token_types.push_back(MFQ_TEXT_TOKEN_TYPE_BYTE);
    }
    const int32_t hello_token = static_cast<int32_t>(token_storage.size());
    token_storage.emplace_back("hello");
    token_types.push_back(MFQ_TEXT_TOKEN_TYPE_NORMAL);

    std::vector<const char *> tokens;
    tokens.reserve(token_storage.size());
    for (const auto & token : token_storage) tokens.push_back(token.c_str());
    std::vector<float> scores(token_storage.size(), 0.0f);
    scores[static_cast<size_t>(hello_token)] = 10.0f;

    gguf_context * metadata = gguf_init_empty();
    require(metadata != nullptr, "cannot create GGUF metadata");
    gguf_set_val_str(metadata, "general.architecture", "mfq-test");
    gguf_set_val_str(metadata, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(
        metadata, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    gguf_set_arr_data(
        metadata, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32,
        scores.data(), scores.size());
    gguf_set_arr_data(
        metadata, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,
        token_types.data(), token_types.size());
    gguf_set_val_u32(metadata, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(metadata, "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_bool(metadata, "tokenizer.ggml.add_space_prefix", false);
    gguf_set_val_str(
        metadata,
        "tokenizer.chat_template",
        "{% for message in messages %}{{ message['role'] + ': ' + "
        "message['content'] + '\\n' }}{% endfor %}{% if "
        "add_generation_prompt %}assistant: {% endif %}");

    const auto path = std::filesystem::temp_directory_path() /
        "mfq-integrated-tokenizer-test.gguf";
    require(
        gguf_write_to_file(metadata, path.string().c_str(), true),
        "cannot write GGUF fixture");
    gguf_free(metadata);
    return path;
}

}  // namespace

int main() {
    const auto path = write_tokenizer_fixture();
    std::ifstream input(path, std::ios::binary);
    std::vector<uint8_t> encoded(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    require(!encoded.empty(), "cannot read GGUF fixture");

    mfq_text_context * context = mfq_text_load_file(path.string().c_str());
    std::filesystem::remove(path);
    require(context != nullptr, "cannot load integrated tokenizer");

    const mfq_text_vocab * vocab = mfq_text_get_vocab(context);
    require(vocab != nullptr, "missing vocabulary");
    require(mfq_text_vocab_n_tokens(vocab) == 260, "vocabulary size mismatch");

    mfq_text_token token = MFQ_TEXT_TOKEN_NULL;
    const int32_t count = mfq_text_tokenize(
        vocab, "h", 1, &token, 1, false, false);
    require(count == 1 && token == 107, "SPM tokenization mismatch");

    const char * source = mfq_text_get_chat_template(context, nullptr);
    require(source != nullptr, "missing chat template");
    auto templates = common_chat_templates_init(context, "");
    common_chat_templates_inputs inputs;
    inputs.messages.push_back({"user", "hello"});
    const auto applied = common_chat_templates_apply(templates.get(), inputs);
    require(
        applied.prompt == "user: hello\nassistant: ",
        "chat template output mismatch");

    mfq_text_grammar * grammar = mfq_text_grammar_init_impl(
        vocab, "root ::= \"h\"", "root", false, nullptr, 0, nullptr, 0);
    require(grammar != nullptr, "cannot create grammar");
    mfq_text_token_data candidates[] = {
        {107, 0.0f, 0.0f},
        {108, 0.0f, 0.0f},
    };
    mfq_text_token_data_array candidate_array = {
        candidates, 2, -1, false};
    mfq_text_grammar_apply_impl(*grammar, &candidate_array);
    require(
        std::isfinite(candidates[0].logit) &&
        !std::isfinite(candidates[1].logit),
        "grammar filtering mismatch");
    mfq_text_grammar_accept_impl(*grammar, 107);
    mfq_text_grammar_free_impl(grammar);
    mfq_text_free(context);

    mfq_text_context * embedded = mfq_text_load_buffer(
        encoded.data(), encoded.size());
    require(embedded != nullptr, "cannot load embedded tokenizer");
    require(
        mfq_text_vocab_n_tokens(mfq_text_get_vocab(embedded)) == 260,
        "embedded vocabulary size mismatch");
    mfq_text_free(embedded);
    return 0;
}
