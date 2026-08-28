// Portions derived from ggml-org/llama.cpp.
// SPDX-License-Identifier: MIT

#include "common.h"

#include "ggml.h"

#include <limits>
#include <regex>
#include <stdexcept>

std::string string_join(
        const std::vector<std::string> & values,
        const std::string & separator) {
    std::string result;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) result += separator;
        result += values[index];
    }
    return result;
}

std::vector<std::string> string_split(
        const std::string & value,
        const std::string & delimiter) {
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = value.find(delimiter);
    while (end != std::string::npos) {
        result.push_back(value.substr(start, end - start));
        start = end + delimiter.size();
        end = value.find(delimiter, start);
    }
    result.push_back(value.substr(start));
    return result;
}

std::string string_repeat(const std::string & value, size_t count) {
    std::string result;
    result.reserve(value.size() * count);
    for (size_t index = 0; index < count; ++index) result += value;
    return result;
}

void string_replace_all(
        std::string & value,
        const std::string & search,
        const std::string & replacement) {
    if (search.empty()) return;
    size_t position = 0;
    while ((position = value.find(search, position)) != std::string::npos) {
        value.replace(position, search.size(), replacement);
        position += replacement.size();
    }
}

std::string regex_escape(const std::string & value) {
    static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
    return std::regex_replace(value, special_chars, "\\$&");
}

std::vector<mfq_text_token> common_tokenize(
        const mfq_text_vocab * vocab,
        const std::string & text,
        bool add_special,
        bool parse_special) {
    int32_t count = mfq_text_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()),
        nullptr, 0, add_special, parse_special);
    if (count == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("tokenization result exceeds int32_t");
    }
    if (count == 0) return {};
    if (count > 0) {
        throw std::runtime_error("invalid tokenizer sizing result");
    }
    std::vector<mfq_text_token> result(static_cast<size_t>(-count));
    count = mfq_text_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()),
        result.data(), static_cast<int32_t>(result.size()),
        add_special, parse_special);
    if (count < 0) {
        throw std::runtime_error("tokenizer sizing changed unexpectedly");
    }
    result.resize(static_cast<size_t>(count));
    return result;
}

std::string common_token_to_piece(
        const mfq_text_vocab * vocab,
        mfq_text_token token,
        bool special) {
    char local[128];
    int32_t count = mfq_text_token_to_piece(
        vocab, token, local, static_cast<int32_t>(sizeof(local)), 0, special);
    if (count >= 0) return std::string(local, local + count);
    std::string result(static_cast<size_t>(-count), '\0');
    count = mfq_text_token_to_piece(
        vocab, token, result.data(), static_cast<int32_t>(result.size()), 0,
        special);
    if (count < 0) {
        throw std::runtime_error("token piece sizing changed unexpectedly");
    }
    result.resize(static_cast<size_t>(count));
    return result;
}
