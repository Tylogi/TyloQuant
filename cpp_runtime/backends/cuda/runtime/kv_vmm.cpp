// Adapted from Strata (MIT); see LICENSES/Strata-MIT.txt.
#include "kv_vmm.h"

#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>
#include <cstdio>
#include <utility>

namespace mfq::cuda::kv_vmm {
namespace {

struct Api {
    bool ok = false;
    uint64_t gran = 0;
    int dev = -1;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemAddressFree) free_va = nullptr;
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
};

template <class F> bool resolve(const char* name, F& f) {
    cudaDriverEntryPointQueryResult q{};
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else   // the versioned query arrived in CUDA 12.5
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        (void) cudaGetLastError();
        return false;
    }
    f = (F) p;
    return true;
}

// never destroyed: the K/V pools and the cache release their chunks from static destructors at exit
const Api& api() {
    static Api& a = *new Api;
    static std::once_flag once;
    std::call_once(once, [] {
        decltype(&cuDeviceGetAttribute) attr = nullptr;
        decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
        if (cudaGetDevice(&a.dev) != cudaSuccess) return;
        if (!resolve("cuDeviceGetAttribute", attr) || !resolve("cuMemGetAllocationGranularity", granularity) ||
            !resolve("cuMemAddressReserve", a.reserve) || !resolve("cuMemAddressFree", a.free_va) ||
            !resolve("cuMemCreate", a.create) || !resolve("cuMemRelease", a.release) || !resolve("cuMemMap", a.map) ||
            !resolve("cuMemUnmap", a.unmap) || !resolve("cuMemSetAccess", a.access))
            return;
        int vmm = 0;
        if (attr(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, (CUdevice) a.dev) != CUDA_SUCCESS || !vmm)
            return;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = a.dev;
        size_t g = 0;
        if (granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || g == 0) return;
        a.gran = g;
        a.ok = true;
    });
    return a;
}

struct RetiredRange {
    unsigned long long base;
    std::vector<VmmChunk> handles;
};
std::mutex& retired_mutex() {
    static auto* mutex = new std::mutex;
    return *mutex;
}
// Keep ownership through persistent driver failures, including destruction of
// the original cache. Subsequent allocations retry these releases.
auto& retired_ranges() {
    static auto* ranges = new std::vector<RetiredRange>;
    return *ranges;
}
bool release_range(unsigned long long base, std::vector<VmmChunk>& handles) {
    const Api& a = api();
    bool complete = true;
    for (std::size_t i = 0; i < handles.size(); ++i) {
        if (!handles[i]) continue;
        if (a.unmap(base + i * a.gran, a.gran) != CUDA_SUCCESS) {
            complete = false;
            continue;
        }
        vmm_chunk_free(std::exchange(handles[i], 0));
    }
    return complete && a.free_va(base, handles.size() * a.gran) == CUDA_SUCCESS;
}
void retry_retired_ranges() {
    std::lock_guard lock(retired_mutex());
    auto& ranges = retired_ranges();
    for (auto it = ranges.begin(); it != ranges.end();) {
        if (release_range(it->base, it->handles)) it = ranges.erase(it);
        else ++it;
    }
}

}  // namespace

bool vmm_available() { return api().ok; }
uint64_t vmm_granularity() { return api().ok ? api().gran : 0; }

VmmChunk vmm_chunk_new() {
    retry_retired_ranges();
    const Api& a = api();
    if (!a.ok) return 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = a.dev;
    CUmemGenericAllocationHandle h = 0;
    if (a.create(&h, (size_t) a.gran, &prop, 0) != CUDA_SUCCESS) return 0;
    return (VmmChunk) h;
}

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) api().release((CUmemGenericAllocationHandle) h);
}

bool VmmRange::reserve(uint64_t bytes) {
    retry_retired_ranges();
    if (!release()) return false;
    const Api& a = api();
    if (!a.ok || bytes == 0) return false;
    const uint64_t n = (bytes + a.gran - 1) / a.gran;
    CUdeviceptr p = 0;
    if (a.reserve(&p, (size_t) (n * a.gran), 0, 0, 0) != CUDA_SUCCESS) return false;
    base_ = (unsigned long long) p;
    h_.assign((size_t) n, 0);
    accessible_.assign((size_t) n, false);
    return true;
}

VmmRange::~VmmRange() {
    if (!release()) {
        std::lock_guard lock(retired_mutex());
        retired_ranges().push_back({base_, std::move(h_)});
    }
}

bool VmmRange::release() {
    if (base_ == 0) return true;
    if (!release_range(base_, h_)) {
        std::fprintf(stderr, "kv_vmm release failed; retaining virtual range and physical handles for retry\n");
        return false;
    }
    base_ = 0;
    h_.clear();
    accessible_.clear();
    return true;
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    const Api& a = api();
    if (a.map((CUdeviceptr) (base_ + (uint64_t) i * a.gran), (size_t) a.gran, 0, (CUmemGenericAllocationHandle) h, 0) !=
        CUDA_SUCCESS)
        return false;
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    CUmemAccessDesc d{};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = a.dev;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return a.access((CUdeviceptr) (base_ + (uint64_t) lo * a.gran), (size_t) ((uint64_t) (hi - lo) * a.gran), &d, 1) ==
           CUDA_SUCCESS;
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) {
        for (auto i = lo; i < hi; ++i) accessible_[(size_t) i] = true;
        return true;
    }
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (i < 0 || i >= chunks() || !h_[(size_t) i]) return 0;
    const Api& a = api();
    const VmmChunk h = h_[(size_t) i];
    if (a.unmap((CUdeviceptr) (base_ + (uint64_t) i * a.gran), (size_t) a.gran) != CUDA_SUCCESS) return 0;
    h_[(size_t) i] = 0;
    accessible_[(size_t) i] = false;
    return h;
}

}  // namespace mfq::cuda::kv_vmm

#else  // HIP: no virtual memory here; the elastic K/V stays off

namespace mfq::cuda::kv_vmm {
bool vmm_available() { return false; }
uint64_t vmm_granularity() { return 0; }
VmmChunk vmm_chunk_new() { return 0; }
void vmm_chunk_free(VmmChunk) {}
bool VmmRange::reserve(uint64_t) { return false; }
VmmRange::~VmmRange() = default;
bool VmmRange::release() { return true; }
int64_t VmmRange::mapped_count() const { return 0; }
bool VmmRange::map_one(int64_t, VmmChunk) { return false; }
bool VmmRange::set_access(int64_t, int64_t) { return false; }
bool VmmRange::commit_run(int64_t, int64_t) { return false; }
VmmChunk VmmRange::unmap(int64_t) { return 0; }
}  // namespace mfq::cuda::kv_vmm

#endif
