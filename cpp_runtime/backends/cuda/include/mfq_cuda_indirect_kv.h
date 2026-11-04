#pragma once

#include <cstdint>
#include "mfq_cuda_kv_hot_cache.h"

namespace mfq::cuda {

// A view of BatchDecodeInputs, never a second request-pointer allocation.
// Each field has [row, {address, sequence capacity}] entries.
struct IndirectKvView {
    const int64_t* states = nullptr;
    const int64_t* enabled = nullptr;
    int rows = 0;
    int field = 0;
    int hot_field = -1;
#ifdef __CUDACC__
    __device__ int64_t capacity(int row, int offset = 0) const {
        return states[2 * (int64_t(field + offset) * rows + row) + 1];
    }
    __device__ void* address(int row, int offset = 0) const {
        return reinterpret_cast<void*>(states[2 * (int64_t(field + offset) * rows + row)]);
    }
    __device__ bool active(int row) const {
        return enabled[row] && address(row);
    }
    __device__ bool exclusive(int row) const {
        return hot_field >= 0 && tier_address(row, HotKvStateFields::Mapping);
    }
    __device__ void* tier_address(int row, int offset) const {
        return reinterpret_cast<void*>(states[2 * (int64_t(hot_field + offset) * rows + row)]);
    }
    __device__ int64_t tier_capacity(int row, int offset) const {
        return states[2 * (int64_t(hot_field + offset) * rows + row) + 1];
    }
    // Logical positions select a tier first; head strides use only its physical extent.
    template<class Value>
    __device__ Value* resolve(int row, int64_t token, int64_t head, int64_t width,
            int offset = 0, bool value_tier = false) const {
        if (!enabled[row] || !address(row, offset) || token < 0 || token >= capacity(row, offset))
            return nullptr;
        void* base = address(row, offset);
        int64_t extent = capacity(row, offset);
        if (exclusive(row)) {
            const bool value = offset != 0 || value_tier;
            const int slot = static_cast<const int32_t*>(tier_address(row, HotKvStateFields::Mapping))[token];
            const int source = slot >= 0 ? (value ? HotKvStateFields::Values : HotKvStateFields::Keys)
                : (value ? HotKvStateFields::ColdValues : HotKvStateFields::ColdKeys);
            base = tier_address(row, source);
            extent = tier_capacity(row, source);
            token = slot >= 0 ? slot :
                static_cast<const int32_t*>(tier_address(row, HotKvStateFields::ColdMapping))[token];
        }
        return base && token >= 0 && token < extent
            ? static_cast<Value*>(base) + (int64_t(head) * extent + token) * width : nullptr;
    }
#endif
};

} // namespace mfq::cuda
