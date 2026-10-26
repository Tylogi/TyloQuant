#pragma once

namespace mfq {

// Install before loading models or starting workers. Fatal reports use bounded
// buffers and write directly to stderr, independently of the stdio transport.
void install_fatal_diagnostics() noexcept;
void log_runtime_error(const char* scope, const char* message) noexcept;

} // namespace mfq
