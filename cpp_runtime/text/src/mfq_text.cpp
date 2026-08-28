// MFQ-owned integration for GGUF tokenizer metadata and chat templates.
// Tokenization internals are derived from ggml-org/llama.cpp.
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "mfq_text.h"

#include "mfq_chat.h"
#include "mfq_gguf_loader.h"
#include "mfq_vocab.h"

#include "ggml-impl.h"
#include "gguf.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct mfq_text_context {
    std::unique_ptr<mfq_text_vocab> vocab;
    gguf_context * metadata = nullptr;
    std::string chat_template;
    std::string tool_use_template;
};

namespace {

std::string metadata_string(
        const gguf_context * metadata,
        const char * key) {
    const int index = gguf_find_key(metadata, key);
    if (index < 0 || gguf_get_kv_type(metadata, index) != GGUF_TYPE_STRING) {
        return {};
    }
    return gguf_get_val_str(metadata, index);
}

mfq_text_context * make_text_context(gguf_context * metadata) {
    if (metadata == nullptr) return nullptr;
    try {
        auto context = std::make_unique<mfq_text_context>();
        context->metadata = metadata;
        context->chat_template = metadata_string(
            metadata, "tokenizer.chat_template");
        context->tool_use_template = metadata_string(
            metadata, "tokenizer.chat_template.tool_use");

        std::string architecture;
        const int architecture_index = gguf_find_key(
            metadata, "general.architecture");
        if (architecture_index >= 0 &&
            gguf_get_kv_type(metadata, architecture_index) == GGUF_TYPE_STRING) {
            architecture = gguf_get_val_str(metadata, architecture_index);
        }
        mfq_gguf_keys keys(architecture);
        mfq_gguf_loader loader(metadata, keys);
        context->vocab = std::make_unique<mfq_text_vocab>();
        context->vocab->load(loader, keys);
        return context.release();
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("MFQ tokenizer initialization failed: %s\n", error.what());
        gguf_free(metadata);
        return nullptr;
    }
}

}  // namespace

mfq_text_context * mfq_text_load_file(const char * path) {
    if (path == nullptr || path[0] == '\0') return nullptr;
    const gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };
    return make_text_context(gguf_init_from_file(path, params));
}

mfq_text_context * mfq_text_load_buffer(const void * data, size_t size) {
    if (data == nullptr || size == 0) return nullptr;
    const gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };
    return make_text_context(gguf_init_from_buffer(data, size, params));
}

void mfq_text_free(mfq_text_context * context) {
    if (context == nullptr) return;
    if (context->metadata != nullptr) {
        gguf_free(context->metadata);
        context->metadata = nullptr;
    }
    delete context;
}

const mfq_text_vocab * mfq_text_get_vocab(const mfq_text_context * context) {
    return context == nullptr ? nullptr : context->vocab.get();
}

const char * mfq_text_get_chat_template(
        const mfq_text_context * context,
        const char * name) {
    if (context == nullptr) return nullptr;
    const std::string * value = nullptr;
    if (name == nullptr) {
        value = &context->chat_template;
    } else if (std::string(name) == "tool_use") {
        value = &context->tool_use_template;
    } else {
        return nullptr;
    }
    return value->empty() ? nullptr : value->c_str();
}

int32_t mfq_text_chat_apply_template(
        const char * template_source,
        const mfq_text_chat_message * messages,
        size_t message_count,
        bool add_assistant,
        char * output,
        int32_t output_size) {
    const std::string source(
        template_source == nullptr ? "chatml" : template_source);
    std::vector<const mfq_text_chat_message *> message_pointers(message_count);
    for (size_t index = 0; index < message_count; ++index) {
        message_pointers[index] = &messages[index];
    }
    std::string formatted;
    const mfq_text_chat_template detected = mfq_text_chat_detect_template(source);
    if (detected == MFQ_TEXT_CHAT_TEMPLATE_UNKNOWN) return -1;
    const int32_t result = mfq_text_chat_apply_template(
        detected, message_pointers, formatted, add_assistant);
    if (result < 0) return result;
    if (output != nullptr && output_size > 0) {
        std::strncpy(output, formatted.c_str(), static_cast<size_t>(output_size));
    }
    return result;
}
