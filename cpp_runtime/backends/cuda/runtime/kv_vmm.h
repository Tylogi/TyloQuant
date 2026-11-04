// Adapted from Strata (MIT); see LICENSES/Strata-MIT.txt.
#pragma once
// Stable CUDA virtual address ranges with physical pages mapped on demand.
// MFQ currently retains growth headroom in the expert budget; direct physical
// page exchange with expert arenas is not enabled. Driver entry points are
// resolved through the runtime API, without linking the driver library.

#include <cstdint>
#include <vector>

namespace mfq::cuda::kv_vmm {

/// A physical chunk: the driver's allocation handle (CUmemGenericAllocationHandle); 0 = none.
using VmmChunk = unsigned long long;

/// The current device supports it and the entry points resolved.
bool vmm_available();
/// The chunk size (0 when not available).
uint64_t vmm_granularity();
/// A new physical chunk on the current device (0: out of memory), and its release.
VmmChunk vmm_chunk_new();
void vmm_chunk_free(VmmChunk h);

class VmmRange {
public:
    VmmRange() = default;
    ~VmmRange();
    VmmRange(const VmmRange&) = delete;
    VmmRange& operator=(const VmmRange&) = delete;

    /// Reserves `bytes` (rounded up to whole chunks) of addresses, nothing mapped.
    bool reserve(uint64_t bytes);
    /// Failed unmaps retain their handles and addresses for a later retry.
    bool release();
    uint8_t* base() const { return (uint8_t*) (uintptr_t) base_; }
    int64_t chunks() const { return (int64_t) h_.size(); }
    bool mapped(int64_t i) const {
        return i >= 0 && i < chunks() && h_[(std::size_t) i] != 0 && accessible_[(std::size_t) i];
    }
    int64_t mapped_count() const;
    /// Maps chunks [lo, hi) that are not mapped yet, each to `take()` or, when that returns 0, to a new chunk.
    /// True means readable and writable; false retains ownership of any failed rollback mappings.
    template <class Take> bool map_range(int64_t lo, int64_t hi, Take take);
    /// Returns 0 on an absent mapping or driver failure; failures keep ownership in this range.
    VmmChunk unmap(int64_t i);

private:
    bool map_one(int64_t i, VmmChunk h);
    bool set_access(int64_t lo, int64_t hi);
    /// Roll back failed access changes, retaining failed unmaps for retry.
    bool commit_run(int64_t lo, int64_t hi);
    unsigned long long base_ = 0;
    std::vector<VmmChunk> h_;   // per chunk: its physical chunk, 0 = unmapped
    std::vector<bool> accessible_;
};

template <class Take> bool VmmRange::map_range(int64_t lo, int64_t hi, Take take) {
    lo = lo < 0 ? 0 : lo;
    hi = hi > chunks() ? chunks() : hi;
    int64_t run = -1;   // access is set once per run of newly mapped chunks (it costs as much as the map)
    for (int64_t i = lo; i < hi; ++i) {
        if (h_[(std::size_t) i] != 0) {
            if (run >= 0 && !commit_run(run, i)) return false;
            run = -1;
            if (!accessible_[(std::size_t) i] && !commit_run(i, i + 1)) return false;
            continue;
        }
        VmmChunk h = take();
        if (h == 0) h = vmm_chunk_new();
        if (h == 0 || !map_one(i, h)) {
            if (h != 0) vmm_chunk_free(h);
            if (run >= 0) commit_run(run, i);
            return false;
        }
        if (run < 0) run = i;
    }
    return run < 0 || commit_run(run, hi);
}

}  // namespace mfq::cuda::kv_vmm
