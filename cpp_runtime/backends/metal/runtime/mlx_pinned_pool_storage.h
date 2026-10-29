#pragma once

#include <cerrno>
#include <cstddef>
#include <sys/mman.h>
#include <unistd.h>

namespace mfq::metal::detail {

struct MlxPinnedPoolStorage {
    void* data = MAP_FAILED;
    std::size_t bytes = 0;
    std::size_t locked_bytes = 0;
    int lock_error = 0;

    MlxPinnedPoolStorage() = default;
    MlxPinnedPoolStorage(const MlxPinnedPoolStorage&) = delete;
    MlxPinnedPoolStorage& operator=(const MlxPinnedPoolStorage&) = delete;

    bool pin_prefix(std::size_t target) noexcept {
        if (data == MAP_FAILED || target > bytes || target % std::size_t(getpagesize())) {
            lock_error = EINVAL;
            return false;
        }
        if (target == locked_bytes) {
            lock_error = 0;
            return true;
        }
        const auto offset = target < locked_bytes ? target : locked_bytes;
        const auto length = target < locked_bytes ? locked_bytes - target : target - locked_bytes;
        auto* address = static_cast<std::byte*>(data) + offset;
        const int result = target < locked_bytes ? munlock(address, length) : mlock(address, length);
        if (result != 0) {
            lock_error = errno;
            return false;
        }
        locked_bytes = target;
        lock_error = 0;
        return true;
    }

    ~MlxPinnedPoolStorage() {
        if (data != MAP_FAILED) {
            if (locked_bytes) munlock(data, locked_bytes);
            munmap(data, bytes);
        }
    }
};

}
