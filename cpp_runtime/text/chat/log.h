#pragma once

// MFQ logging bridge for the integrated chat-template implementation.
// SPDX-License-Identifier: Apache-2.0

#include "ggml-impl.h"

#define LOG_DBG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define LOG_INF(...) GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WRN(...) GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERR(...) GGML_LOG_ERROR(__VA_ARGS__)
#define LOG_DBGV(verbosity, ...) GGML_LOG_DEBUG(__VA_ARGS__)
#define LOG_INFV(verbosity, ...) GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WRNV(verbosity, ...) GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERRV(verbosity, ...) GGML_LOG_ERROR(__VA_ARGS__)
