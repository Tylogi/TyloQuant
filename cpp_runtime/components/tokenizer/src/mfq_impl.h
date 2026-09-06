#pragma once

// Shared support utilities for MFQ's integrated tokenizer and grammar code.
// Portions derived from ggml-org/llama.cpp.
// SPDX-License-Identifier: MIT

#include "ggml-impl.h"

#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

#define MFQ_TEXT_LOG(...)       GGML_LOG(__VA_ARGS__)
#define MFQ_TEXT_LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define MFQ_TEXT_LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define MFQ_TEXT_LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)
#define MFQ_TEXT_LOG_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define MFQ_TEXT_LOG_CONT(...)  GGML_LOG_CONT(__VA_ARGS__)

inline void replace_all(
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

inline std::string format(const char * format_string, ...) {
    va_list arguments;
    va_start(arguments, format_string);
    va_list copy;
    va_copy(copy, arguments);
    const int size = std::vsnprintf(nullptr, 0, format_string, copy);
    va_end(copy);
    if (size < 0) {
        va_end(arguments);
        return {};
    }
    std::vector<char> buffer(static_cast<size_t>(size) + 1);
    std::vsnprintf(buffer.data(), buffer.size(), format_string, arguments);
    va_end(arguments);
    return std::string(buffer.data(), static_cast<size_t>(size));
}
