#pragma once

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

#include <sys/mman.h>

namespace mfq::metal::detail {

// Large model-load scratch buffers must bypass Darwin malloc's large-object
// depot: freed multi-GiB vector capacities can otherwise remain resident and
// count against the model footprint. Anonymous mappings are returned to the
// kernel immediately when a staging vector dies.
template <typename T>
class MmapAllocator {
public:
    using value_type = T;
    using is_always_equal = std::true_type;

    MmapAllocator() noexcept = default;

    template <typename U>
    MmapAllocator(const MmapAllocator<U>&) noexcept {}

    T* allocate(std::size_t count) {
        if (count == 0) {
            return nullptr;
        }
        if (count > max_size()) {
            throw std::bad_array_new_length();
        }
        const auto bytes = count * sizeof(T);
        void* mapping = ::mmap(
            nullptr,
            bytes,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0);
        if (mapping == MAP_FAILED) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(mapping);
    }

    void deallocate(T* pointer, std::size_t count) noexcept {
        if (pointer != nullptr && count != 0) {
            ::munmap(pointer, count * sizeof(T));
        }
    }

    constexpr std::size_t max_size() const noexcept {
        return std::numeric_limits<std::size_t>::max() / sizeof(T);
    }
};

template <typename T, typename U>
bool operator==(
    const MmapAllocator<T>&,
    const MmapAllocator<U>&) noexcept {
    return true;
}

template <typename T>
using StagingVector = std::vector<T, MmapAllocator<T>>;

}  // namespace mfq::metal::detail
