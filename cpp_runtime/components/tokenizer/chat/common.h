#pragma once

// Portions derived from ggml-org/llama.cpp.
// SPDX-License-Identifier: MIT

#include "mfq_text.h"

#include <sstream>
#include <string>
#include <string_view>
#include <vector>

enum common_grammar_trigger_type {
    COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
};

struct common_grammar_trigger {
    common_grammar_trigger_type type;
    std::string value;
    mfq_text_token token = MFQ_TEXT_TOKEN_NULL;
};

enum common_reasoning_format {
    COMMON_REASONING_FORMAT_NONE,
    COMMON_REASONING_FORMAT_AUTO,
    COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,
    COMMON_REASONING_FORMAT_DEEPSEEK,
};

using mfq_text_tokens = std::vector<mfq_text_token>;

std::string string_join(
    const std::vector<std::string> & values,
    const std::string & separator);
std::vector<std::string> string_split(
    const std::string & str,
    const std::string & delimiter);
std::string string_repeat(const std::string & str, size_t n);
void string_replace_all(
    std::string & value,
    const std::string & search,
    const std::string & replacement);
std::string regex_escape(const std::string & value);

inline bool string_starts_with(
        std::string_view value,
        std::string_view prefix) {
    return value.size() >= prefix.size() &&
        value.compare(0, prefix.size(), prefix) == 0;
}

inline bool string_ends_with(
        std::string_view value,
        std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<mfq_text_token> common_tokenize(
    const mfq_text_vocab * vocab,
    const std::string & text,
    bool add_special,
    bool parse_special);
std::string common_token_to_piece(
    const mfq_text_vocab * vocab,
    mfq_text_token token,
    bool special);
