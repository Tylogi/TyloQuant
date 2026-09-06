#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

// Runtime-independent token constraint used to carry a chat-template grammar
// from the HTTP server into the backend sampler.
struct MfqTokenConstraint {
    std::function<bool(std::int64_t)> allows;
    std::function<void(float *, std::size_t)> apply;
    std::function<void(std::int64_t)> accept;

    explicit operator bool() const noexcept {
        return static_cast<bool>(apply);
    }
};

using MfqTokenConstraintPtr = std::shared_ptr<MfqTokenConstraint>;
