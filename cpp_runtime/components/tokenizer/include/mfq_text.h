#ifndef MFQ_TEXT_API_H
#define MFQ_TEXT_API_H

// MFQ's integrated GGUF tokenizer and chat-template API.
// Portions derived from ggml-org/llama.cpp.
// SPDX-License-Identifier: MIT

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define MFQ_TEXT_TOKEN_NULL (-1)

#ifdef __cplusplus
extern "C" {
#endif

struct mfq_text_context;
struct mfq_text_vocab;

typedef int32_t mfq_text_token;

enum mfq_text_vocab_type {
    MFQ_TEXT_VOCAB_TYPE_NONE   = 0,
    MFQ_TEXT_VOCAB_TYPE_SPM    = 1,
    MFQ_TEXT_VOCAB_TYPE_BPE    = 2,
    MFQ_TEXT_VOCAB_TYPE_WPM    = 3,
    MFQ_TEXT_VOCAB_TYPE_UGM    = 4,
    MFQ_TEXT_VOCAB_TYPE_RWKV   = 5,
    MFQ_TEXT_VOCAB_TYPE_PLAMO2 = 6,
};

enum mfq_text_token_type {
    MFQ_TEXT_TOKEN_TYPE_UNDEFINED    = 0,
    MFQ_TEXT_TOKEN_TYPE_NORMAL       = 1,
    MFQ_TEXT_TOKEN_TYPE_UNKNOWN      = 2,
    MFQ_TEXT_TOKEN_TYPE_CONTROL      = 3,
    MFQ_TEXT_TOKEN_TYPE_USER_DEFINED = 4,
    MFQ_TEXT_TOKEN_TYPE_UNUSED       = 5,
    MFQ_TEXT_TOKEN_TYPE_BYTE         = 6,
};

enum mfq_text_token_attr {
    MFQ_TEXT_TOKEN_ATTR_UNDEFINED     = 0,
    MFQ_TEXT_TOKEN_ATTR_UNKNOWN       = 1 << 0,
    MFQ_TEXT_TOKEN_ATTR_UNUSED        = 1 << 1,
    MFQ_TEXT_TOKEN_ATTR_NORMAL        = 1 << 2,
    MFQ_TEXT_TOKEN_ATTR_CONTROL       = 1 << 3,
    MFQ_TEXT_TOKEN_ATTR_USER_DEFINED  = 1 << 4,
    MFQ_TEXT_TOKEN_ATTR_BYTE          = 1 << 5,
    MFQ_TEXT_TOKEN_ATTR_NORMALIZED    = 1 << 6,
    MFQ_TEXT_TOKEN_ATTR_LSTRIP        = 1 << 7,
    MFQ_TEXT_TOKEN_ATTR_RSTRIP        = 1 << 8,
    MFQ_TEXT_TOKEN_ATTR_SINGLE_WORD   = 1 << 9,
};

typedef struct mfq_text_token_data {
    mfq_text_token id;
    float logit;
    float p;
} mfq_text_token_data;

typedef struct mfq_text_token_data_array {
    mfq_text_token_data * data;
    size_t size;
    int64_t selected;
    bool sorted;
} mfq_text_token_data_array;

typedef struct mfq_text_chat_message {
    const char * role;
    const char * content;
} mfq_text_chat_message;

struct mfq_text_context * mfq_text_load_file(const char * path);
struct mfq_text_context * mfq_text_load_buffer(
    const void * data,
    size_t size);
void mfq_text_free(struct mfq_text_context * context);
const struct mfq_text_vocab * mfq_text_get_vocab(
    const struct mfq_text_context * context);
const char * mfq_text_get_chat_template(
    const struct mfq_text_context * context,
    const char * name);

int32_t mfq_text_vocab_n_tokens(const struct mfq_text_vocab * vocab);
enum mfq_text_vocab_type mfq_text_vocab_type(const struct mfq_text_vocab * vocab);
const char * mfq_text_vocab_get_text(
    const struct mfq_text_vocab * vocab,
    mfq_text_token token);
float mfq_text_vocab_get_score(
    const struct mfq_text_vocab * vocab,
    mfq_text_token token);
enum mfq_text_token_attr mfq_text_vocab_get_attr(
    const struct mfq_text_vocab * vocab,
    mfq_text_token token);
bool mfq_text_vocab_is_eog(
    const struct mfq_text_vocab * vocab,
    mfq_text_token token);
bool mfq_text_vocab_is_control(
    const struct mfq_text_vocab * vocab,
    mfq_text_token token);
mfq_text_token mfq_text_vocab_bos(const struct mfq_text_vocab * vocab);
mfq_text_token mfq_text_vocab_eos(const struct mfq_text_vocab * vocab);
mfq_text_token mfq_text_vocab_eot(const struct mfq_text_vocab * vocab);
mfq_text_token mfq_text_vocab_sep(const struct mfq_text_vocab * vocab);
mfq_text_token mfq_text_vocab_nl(const struct mfq_text_vocab * vocab);
mfq_text_token mfq_text_vocab_pad(const struct mfq_text_vocab * vocab);
bool mfq_text_vocab_get_add_bos(const struct mfq_text_vocab * vocab);
bool mfq_text_vocab_get_add_eos(const struct mfq_text_vocab * vocab);
bool mfq_text_vocab_get_add_sep(const struct mfq_text_vocab * vocab);

int32_t mfq_text_tokenize(
    const struct mfq_text_vocab * vocab,
    const char * text,
    int32_t text_length,
    mfq_text_token * tokens,
    int32_t token_capacity,
    bool add_special,
    bool parse_special);
int32_t mfq_text_token_to_piece(
    const struct mfq_text_vocab * vocab,
    mfq_text_token token,
    char * output,
    int32_t output_size,
    int32_t left_strip,
    bool special);
int32_t mfq_text_detokenize(
    const struct mfq_text_vocab * vocab,
    const mfq_text_token * tokens,
    int32_t token_count,
    char * output,
    int32_t output_size,
    bool remove_special,
    bool unparse_special);

int32_t mfq_text_chat_apply_template(
    const char * template_source,
    const struct mfq_text_chat_message * messages,
    size_t message_count,
    bool add_assistant,
    char * output,
    int32_t output_size);

#ifdef __cplusplus
}
#endif

#endif
