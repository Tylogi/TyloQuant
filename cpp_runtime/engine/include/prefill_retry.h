#pragma once

#include "mfq/runtime.h"
#include "generation_step.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <new>
#include <string>
#include <string_view>

namespace mfq::engine {

// Only capacity failures are retryable. Kernel, input and teardown errors keep
// their original type and must never be hidden by a prefill replay.
inline bool prefill_resource_failure(const std::exception_ptr& failure) {
    try { std::rethrow_exception(failure); }
    catch (const MfqResourceExhausted&) { return true; }
    catch (const std::bad_alloc&) { return true; }
    catch (const std::exception& error) {
        const std::string_view message=error.what();
        return message.starts_with("CUDA pinned allocation failed:") ||
            message.starts_with("CUDA process VRAM limit exceeded:") ||
            message.starts_with("CUDA VRAM limit exceeded:") ||
            message.starts_with("insufficient VRAM for context growth;") ||
            message.starts_with("kv_memory_insufficient:");
    }
    catch (...) { return false; }
}

inline std::string prefill_failure_message(const std::exception_ptr& failure) {
    try { std::rethrow_exception(failure); }
    catch (const std::exception& error) { return error.what(); }
    catch (...) { return "unknown resource failure"; }
}

inline int64_t smaller_prefill_chunk(int64_t current, const std::exception_ptr& failure) {
    if (!prefill_resource_failure(failure)) std::rethrow_exception(failure);
    if (current <= 1)
        throw MfqResourceExhausted("prefill resource exhausted at minimum chunk=1: "+
            prefill_failure_message(failure));
    return std::max<int64_t>(1,current/2);
}

// Restart only before token publication. Backend recovery destroys physical
// state and shrinks the tile; this loop owns coroutine teardown before reclaim.
template<class Ops> Generation retry_prefill_request(Ops ops) {
    for(;;) {
        auto sequence=ops.start();
        std::exception_ptr failure;
        for(;;) {
            mfq::StepResult<EventData> step;
            try { step=sequence.next(); }
            catch(...) { failure=std::current_exception(); break; }
            if(!step)co_return;
            co_yield std::move(step);
        }
        sequence={};
        ops.recover(failure);
    }
}

} // namespace mfq::engine
