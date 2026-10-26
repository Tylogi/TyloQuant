// Compile the implementation into this test so driver failures can be injected
// without production environment switches or touching real device allocations.
#include "../runtime/kv_vmm.cpp"

#include <set>
#include <stdexcept>
#include <iostream>

using namespace mfq::cuda::kv_vmm;
namespace {
std::set<VmmChunk> owned;
std::set<CUdeviceptr> mapped_addresses;
int fail_unmap = 0, fail_access = 0, free_addresses = 0;
VmmChunk next_handle = 1;
CUdeviceptr next_address = 4096;
void require(bool value) { if (!value) throw std::runtime_error("VMM failure ownership regression"); }
}

int main() {
    auto& driver = const_cast<Api&>(api());
    driver.ok = true;
    driver.gran = 4096;
    driver.reserve = [](CUdeviceptr* address, size_t size, size_t, CUdeviceptr, unsigned long long) {
        *address = next_address;
        next_address += size;
        return CUDA_SUCCESS;
    };
    driver.free_va = [](CUdeviceptr, size_t) { ++free_addresses; return CUDA_SUCCESS; };
    driver.create = [](CUmemGenericAllocationHandle* handle, size_t, const CUmemAllocationProp*, unsigned long long) {
        *handle = next_handle++;
        owned.insert(*handle);
        return CUDA_SUCCESS;
    };
    driver.release = [](CUmemGenericAllocationHandle handle) {
        require(owned.erase(handle) == 1);
        return CUDA_SUCCESS;
    };
    driver.map = [](CUdeviceptr address, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) {
        require(mapped_addresses.insert(address).second);
        return CUDA_SUCCESS;
    };
    driver.unmap = [](CUdeviceptr address, size_t) {
        if (fail_unmap) { --fail_unmap; return CUDA_ERROR_UNKNOWN; }
        require(mapped_addresses.erase(address) == 1);
        return CUDA_SUCCESS;
    };
    driver.access = [](CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) {
        if (fail_access) { --fail_access; return CUDA_ERROR_UNKNOWN; }
        return CUDA_SUCCESS;
    };
    {
        VmmRange range;
        require(range.reserve(8192));
        require(range.map_range(0, 2, [] { return VmmChunk{0}; }));
        fail_unmap = 1;
        require(!range.release());
        require(range.base() && range.mapped_count() == 1 && owned.size() == 1 && !free_addresses);
        fail_unmap = 1;
        require(!range.reserve(4096)); // reserve cannot overwrite the failed range
        require(range.mapped_count() == 1 && owned.size() == 1);
        require(range.release());
        require(owned.empty() && mapped_addresses.empty() && free_addresses == 1);
    }
    {
        VmmRange range;
        require(range.reserve(4096));
        fail_access = fail_unmap = 1;
        require(!range.map_range(0, 1, [] { return VmmChunk{0}; }));
        require(range.mapped_count() == 1 && !range.mapped(0) && owned.size() == 1);
        fail_access = 1;
        require(!range.map_range(0, 1, [] { return VmmChunk{0}; })); // inaccessible mapping is retried
        require(range.mapped_count() == 0 && owned.empty());
        require(range.map_range(0, 1, [] { return VmmChunk{0}; }));
        fail_unmap = 1; // destructor transfers ownership to the retry list
    }
    require(owned.size() == 1 && retired_ranges().size() == 1);
    const auto handle = vmm_chunk_new();
    require(retired_ranges().empty() && mapped_addresses.empty() && owned.size() == 1);
    vmm_chunk_free(handle);
    require(owned.empty());
    std::cout << "vmm_failure_ownership release/reserve/rollback/destructor_retry PASS\n";
}
