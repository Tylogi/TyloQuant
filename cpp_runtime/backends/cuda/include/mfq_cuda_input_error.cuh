#pragma once

#include "mfq_cuda_context.h"

namespace mfq::cuda {

// Invalid inputs never reach memory addressing. The host reads this record only
// after stream completion, so normal operations need no extra copy or barrier.
__device__ inline bool valid_device_index(
        int64_t index, int64_t extent, DeviceInputError* error, int operation) {
    if (index >= 0 && index < extent) return true;
    if (atomicCAS(&error->operation, 0, operation) == 0) {
        error->index = index;
        error->extent = extent;
    }
    return false;
}

} // namespace mfq::cuda
