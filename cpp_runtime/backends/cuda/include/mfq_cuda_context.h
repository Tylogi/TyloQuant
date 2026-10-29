#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mfq::cuda {

class Error final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void check(cudaError_t status, const char* expression, const char* file, int line);
void check(cublasStatus_t status, const char* expression, const char* file, int line);
void record_gpu_timestamp(uint64_t* timestamps, std::size_t index, cudaStream_t stream);

#define MFQ_NATIVE_CUDA_CHECK(expression) \
    ::mfq::cuda::check((expression), #expression, __FILE__, __LINE__)

class DeviceGuard final {
public:
    explicit DeviceGuard(int device);
    ~DeviceGuard() noexcept;

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int previous_ = 0;
    bool restore_ = false;
};

class Event final {
public:
    explicit Event(unsigned flags = cudaEventDisableTiming);
    ~Event() noexcept;

    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t get() const noexcept { return event_; }
    void record(cudaStream_t stream);
    bool ready() const;
    void synchronize() const;

private:
    int device_ = 0;
    cudaEvent_t event_ = nullptr;
};

class Stream final {
public:
    explicit Stream(int device, unsigned flags = cudaStreamNonBlocking);
    ~Stream() noexcept;

    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other) noexcept;
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    int device() const noexcept { return device_; }
    cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const;
    void wait(const Event& event) const;

private:
    int device_ = 0;
    cudaStream_t stream_ = nullptr;
};

// Copyable stream handle used by the runtime graph.  Owning streams retain the
// underlying Stream through shared ownership; the per-device default stream is
// retained by its Context.  This mirrors the value semantics of ATen's
// CUDAStream without exposing ATen in the native runtime ABI.
class StreamHandle final {
public:
    StreamHandle() = default;
    StreamHandle(int device, cudaStream_t stream, std::shared_ptr<Stream> owner = {})
        : device_(device), stream_(stream), owner_(std::move(owner)) {}

    int device_index() const noexcept { return device_; }
    int device() const noexcept { return device_; }
    cudaStream_t stream() const noexcept { return stream_; }
    operator cudaStream_t() const noexcept { return stream_; }
    explicit operator bool() const noexcept { return stream_ != nullptr; }

private:
    int device_ = 0;
    cudaStream_t stream_ = nullptr;
    std::shared_ptr<Stream> owner_;
};

StreamHandle current_stream(int device = -1);
StreamHandle stream_from_pool(bool high_priority = false, int device = -1);

class StreamGuard final {
public:
    explicit StreamGuard(StreamHandle stream);
    ~StreamGuard() noexcept;

    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;

private:
    int device_ = 0;
    std::optional<StreamHandle> previous_;
};

class Context;

class Graph final {
public:
    Graph() = default;
    ~Graph() noexcept;

    Graph(Graph&& other) noexcept;
    Graph& operator=(Graph&& other) noexcept;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    void prepare_memory();
    void prepare_memory(
        const std::vector<StreamHandle>& participant_streams);
    void capture_begin();
    // Record in a pool already owned by a decoder or expert-cache lifetime.
    void capture_begin_shared();
    void capture_end();
    void replay();
    void reset() noexcept;
    bool valid() const noexcept { return executable_ != nullptr; }
    cudaGraphExec_t native_executable()const noexcept{return executable_;}
    std::size_t nodes() const;

private:
    int device_ = 0;
    StreamHandle stream_;
    std::shared_ptr<Context> context_;
    std::vector<StreamHandle> pool_streams_;
    std::vector<std::shared_ptr<Context>> pool_contexts_;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    bool shared_pool_ = false;
};

class BlasHandle final {
public:
    explicit BlasHandle(int device);
    ~BlasHandle() noexcept;

    BlasHandle(BlasHandle&& other) noexcept;
    BlasHandle& operator=(BlasHandle&& other) noexcept;
    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;

    cublasHandle_t get() const noexcept { return handle_; }
    void set_stream(cudaStream_t stream);

private:
    int device_ = 0;
    cublasHandle_t handle_ = nullptr;
};

struct AllocationBudget;
struct DeviceMemoryStats {
    std::size_t limit = 0;
    std::size_t allocated = 0;
    std::size_t peak = 0;
};

class Context final : public std::enable_shared_from_this<Context> {
public:
    explicit Context(int device);

    int device() const noexcept { return device_; }
    Stream& stream() noexcept { return stream_; }
    const Stream& stream() const noexcept { return stream_; }
    BlasHandle& blas() noexcept { return blas_; }
    bool supports_async_allocations() const noexcept { return async_allocations_; }
    DeviceMemoryStats memory_stats() const noexcept;
    std::pair<std::uint64_t,std::uint64_t> pool_memory_stats() const;
    std::size_t local_memory_usage() const;
    // Process-owned pages, including idle allocator pages and CUDA libraries.
    std::size_t resident_memory_usage() const;
    std::size_t available_memory() const;
    void reclaim_idle_memory();
    // Temporarily reuse already charged expert-cache storage on one ordered
    // compute stream. Graph/resident allocations never enter this workspace.
    void begin_workspace_loan(const std::vector<std::pair<void*,std::size_t>>& regions,
                              cudaStream_t stream);
    void suspend_workspace_loan();
    void end_workspace_loan();
    std::size_t workspace_available() const;
    std::vector<std::pair<void*,std::size_t>> workspace_touched_regions() const;
    void check_memory_limit(std::size_t additional = 0) const;
    void set_memory_limit(std::size_t bytes);

    void* allocate(std::size_t bytes, cudaStream_t stream = nullptr);
    void release(
        void* pointer,
        std::size_t bytes,
        cudaStream_t stream = nullptr) noexcept;
    void begin_graph_pool(cudaStream_t stream);
    void begin_graph_warmup(cudaStream_t stream);
    void end_graph_warmup(cudaStream_t stream) noexcept;
    void begin_graph_capture(cudaStream_t stream);
    void end_graph_capture(cudaStream_t stream) noexcept;
    void end_graph_pool(cudaStream_t stream) noexcept;
    void trim();

private:
    struct GraphPool {
        cudaMemPool_t memory = nullptr;
        bool capturing = false;
        bool warming = false;
        std::map<std::size_t, std::vector<void*>> available;
        std::unordered_set<void*> owned;
    };

    int device_ = 0;
    Stream stream_;
    BlasHandle blas_;
    cudaMemPool_t pool_ = nullptr;
    bool async_allocations_ = false;
    std::shared_ptr<AllocationBudget> allocation_budget_;
    void trim_pool_pages() const;
    mutable std::mutex graph_pool_mutex_;
    // A tensor may use a larger cached graph block and outlive its pool.
    // Account/free the allocation's capacity, not that tensor's logical bytes.
    std::unordered_map<void*,std::size_t> graph_allocation_sizes_;
    struct WorkspaceLoan {
        cudaStream_t stream=nullptr;
        std::thread::id thread;
        bool enabled=true;
        std::map<std::uintptr_t,std::size_t> free;
        std::unordered_map<void*,std::size_t> live;
        std::vector<std::pair<std::uintptr_t,std::size_t>> touched;
    };
    std::unique_ptr<WorkspaceLoan> workspace_loan_;
    std::unordered_map<cudaStream_t, GraphPool> graph_pools_;
};

std::shared_ptr<Context> default_context(int device = 0);
class GraphWarmupScope {
public:
    GraphWarmupScope(std::shared_ptr<Context> context,cudaStream_t stream):context_(std::move(context)),stream_(stream) {
        context_->begin_graph_warmup(stream_);
    }
    ~GraphWarmupScope(){context_->end_graph_warmup(stream_);}
    GraphWarmupScope(const GraphWarmupScope&)=delete;
private:
    std::shared_ptr<Context> context_;cudaStream_t stream_;
};

class Buffer final {
public:
    Buffer() = default;
    Buffer(std::shared_ptr<Context> context, std::size_t bytes);
    ~Buffer() noexcept;

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return bytes_; }
    explicit operator bool() const noexcept { return data_ != nullptr; }

private:
    void reset() noexcept;

    std::shared_ptr<Context> context_;
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
    StreamHandle stream_;
};

// cudaHostAlloc reports an exhausted pinned-memory limit as
// cudaErrorMemoryAllocation on Linux, but WDDM can reject a large mapped
// allocation with a mapping error instead ("resource already mapped").
// Callers that retry smaller sizes or fall back to pageable memory treat all
// of these as recoverable resource exhaustion.
inline bool pinned_allocation_rejected(cudaError_t status) noexcept {
    return status == cudaErrorMemoryAllocation || status == cudaErrorAlreadyMapped ||
           status == cudaErrorMapBufferObjectFailed;
}

class HostBuffer final {
public:
    explicit HostBuffer(std::size_t bytes, bool mapped = false);
    ~HostBuffer() noexcept;

    HostBuffer(HostBuffer&& other) noexcept;
    HostBuffer& operator=(HostBuffer&& other) noexcept;
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return bytes_; }

private:
    void reset() noexcept;

    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

}  // namespace mfq::cuda
