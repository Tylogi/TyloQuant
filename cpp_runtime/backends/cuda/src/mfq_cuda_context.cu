#include "mfq_cuda_context.h"
#include "mfq_cuda_resident_allocations.h"
#include "mfq/runtime.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#endif

namespace mfq::cuda {
namespace {
__global__ void record_gpu_timestamp_kernel(uint64_t* timestamps, std::size_t index) {
    unsigned long long nanoseconds;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanoseconds));
    timestamps[index]=nanoseconds;
}
}
void record_gpu_timestamp(uint64_t* timestamps, std::size_t index, cudaStream_t stream) {
    record_gpu_timestamp_kernel<<<1,1,0,stream>>>(timestamps,index);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
}
namespace { thread_local bool resident_allocations=false; }
ResidentAllocationScope::ResidentAllocationScope(bool enabled) noexcept
    :previous_(resident_allocations) {resident_allocations=enabled;}
void ResidentAllocationScope::finish() noexcept {
    if(active_){resident_allocations=previous_;active_=false;}
}

struct AllocationBudget {
    std::mutex graph_memory_mutex;
    // Retired pools remain visible until their persistent request tensors die.
    std::unordered_map<cudaMemPool_t,bool> graph_memory_pools;
    // Drained graph pools, trimmed and kept for the next graph. A pool's used
    // bytes drop when cudaFreeAsync is enqueued, and the driver finishes a
    // free's bookkeeping lazily, on a later call for that stream. Destroying
    // a drained pool first made the driver fault inside cudaStreamSynchronize
    // on a busy GPU, so graph pools are recycled instead of destroyed. Their
    // number is bounded by the peak number of live graph pools.
    std::vector<cudaMemPool_t> idle_graph_pools;
    cudaMemPool_t resident_memory_pool=nullptr;
    std::size_t limit = 0;
    std::atomic<std::size_t> allocated{},peak{};
#ifdef _WIN32
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;
#endif
    void acquire(std::size_t bytes) {
        auto before=allocated.load(std::memory_order_relaxed);
        do {
            if(bytes>std::numeric_limits<std::size_t>::max()-before ||
                (limit && (before>limit || bytes>limit-before)))
                throw MfqResourceExhausted("CUDA VRAM limit exceeded: requested="+std::to_string(bytes)+
                    " allocated="+std::to_string(before)+" limit="+std::to_string(limit));
        } while(!allocated.compare_exchange_weak(before,before+bytes,std::memory_order_relaxed));
        auto maximum=peak.load(std::memory_order_relaxed);
        while(maximum<before+bytes && !peak.compare_exchange_weak(maximum,before+bytes,std::memory_order_relaxed)){}
    }
    void release(std::size_t bytes) noexcept {allocated.fetch_sub(bytes,std::memory_order_relaxed);}
};
namespace {

// Returns a drained graph pool's pages and keeps the pool for reuse. The
// caller holds graph_memory_mutex.
void park_graph_pool(AllocationBudget& budget, cudaMemPool_t memory) noexcept {
    if(cudaMemPoolTrimTo(memory,0)!=cudaSuccess)(void)cudaGetLastError();
    try {
        budget.idle_graph_pools.push_back(memory);
    } catch(...) {
        // Losing track of the handle leaks it, which is safe; destroying it is not.
    }
}

std::mutex allocation_budgets_mutex;
std::unordered_map<int,std::shared_ptr<AllocationBudget>> allocation_budgets;
std::shared_ptr<AllocationBudget> device_allocation_budget(int device) {
    std::lock_guard lock(allocation_budgets_mutex);
    auto& budget=allocation_budgets[device];
    if(!budget) {
        auto candidate=std::make_shared<AllocationBudget>();
        if(const auto* value=std::getenv("MFQ_CUDA_VRAM_LIMIT_GIB")) {
            char* end=nullptr;const double gib=std::strtod(value,&end);
            const long double bytes=static_cast<long double>(gib)*1024*1024*1024;
            if(end==value || *end || !std::isfinite(gib) || gib<0 ||
                bytes>std::numeric_limits<std::size_t>::max() || (gib>0 && bytes<1))
                throw Error("MFQ_CUDA_VRAM_LIMIT_GIB must be a finite non-negative GiB value");
            candidate->limit=static_cast<std::size_t>(bytes);
        }
#ifdef _WIN32
        {
            Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
            cudaDeviceProp properties{};MFQ_NATIVE_CUDA_CHECK(cudaGetDeviceProperties(&properties,device));
            if(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
                for(UINT index=0;;++index) {
                    Microsoft::WRL::ComPtr<IDXGIAdapter1> item;
                    if(factory->EnumAdapters1(index,&item)==DXGI_ERROR_NOT_FOUND)break;
                    DXGI_ADAPTER_DESC1 description{};
                    if(SUCCEEDED(item->GetDesc1(&description)) &&
                        !std::memcmp(properties.luid,&description.AdapterLuid,8)) {item.As(&candidate->adapter);break;}
                }
            }
            if(candidate->limit && !candidate->adapter)throw Error("CUDA VRAM limit requires the matching WDDM adapter");
        }
#endif
        budget=std::move(candidate);
    }
    return budget;
}

std::mutex default_context_mutex;
std::unordered_map<int, std::weak_ptr<Context>> default_contexts;
// ponytail: global lock is sufficient; shard by thread if stream switching contends.
std::mutex active_streams_mutex;
std::unordered_map<
    std::thread::id,
    std::unordered_map<int, StreamHandle>> active_streams;

template <typename Function>
void on_device_noexcept(int device, Function&& function) noexcept {
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess) return;
    const bool restore = previous != device;
    if (restore && cudaSetDevice(device) != cudaSuccess) return;
    try {
        function();
    } catch (...) {
        // Resource destruction must never terminate inference during unwinding.
    }
    if (restore) (void)cudaSetDevice(previous);
}

std::string location_message(
    const char* category,
    const char* detail,
    const char* expression,
    const char* file,
    int line) {
    std::ostringstream message;
    message << category << " error: " << detail << " (" << expression << ") at "
            << file << ':' << line;
    return message.str();
}

const char* cublas_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "unknown cuBLAS status";
}

}  // namespace

std::shared_ptr<Context> default_context(int device) {
    std::lock_guard lock(default_context_mutex);
    if (const auto found = default_contexts.find(device); found != default_contexts.end()) {
        if (auto context = found->second.lock()) {
            return context;
        }
    }
    auto context = std::make_shared<Context>(device);
    default_contexts[device] = context;
    return context;
}

void check(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        throw Error(location_message(
            "CUDA", cudaGetErrorString(status), expression, file, line));
    }
}

void check(cublasStatus_t status, const char* expression, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw Error(location_message(
            "cuBLAS", cublas_status_name(status), expression, file, line));
    }
}

DeviceGuard::DeviceGuard(int device) {
    MFQ_NATIVE_CUDA_CHECK(cudaGetDevice(&previous_));
    if (previous_ != device) {
        MFQ_NATIVE_CUDA_CHECK(cudaSetDevice(device));
        restore_ = true;
    }
}

DeviceGuard::~DeviceGuard() noexcept {
    if (restore_) {
        (void)cudaSetDevice(previous_);
    }
}

Event::Event(unsigned flags) {
    MFQ_NATIVE_CUDA_CHECK(cudaGetDevice(&device_));
    MFQ_NATIVE_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
}

Event::~Event() noexcept {
    if (event_ != nullptr) {
        on_device_noexcept(device_, [&] { (void)cudaEventDestroy(event_); });
    }
}

Event::Event(Event&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

Event& Event::operator=(Event&& other) noexcept {
    if (this != &other) {
        if (event_ != nullptr) {
            on_device_noexcept(device_, [&] { (void)cudaEventDestroy(event_); });
        }
        device_ = other.device_;
        event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
}

void Event::record(cudaStream_t stream) {
    DeviceGuard guard(device_);
    MFQ_NATIVE_CUDA_CHECK(cudaEventRecord(event_, stream));
}

bool Event::ready() const {
    DeviceGuard guard(device_);
    const auto status = cudaEventQuery(event_);
    if (status == cudaSuccess) {
        return true;
    }
    if (status == cudaErrorNotReady) {
        (void)cudaGetLastError();
        return false;
    }
    MFQ_NATIVE_CUDA_CHECK(status);
    return false;
}

void Event::synchronize() const {
    DeviceGuard guard(device_);
    MFQ_NATIVE_CUDA_CHECK(cudaEventSynchronize(event_));
}

Stream::Stream(int device, unsigned flags) : device_(device) {
    DeviceGuard guard(device_);
    MFQ_NATIVE_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
}

Stream::~Stream() noexcept {
    if (stream_ != nullptr) {
        on_device_noexcept(device_, [&] { (void)cudaStreamDestroy(stream_); });
    }
}

Stream::Stream(Stream&& other) noexcept
    : device_(other.device_), stream_(std::exchange(other.stream_, nullptr)) {}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        if (stream_ != nullptr) {
            on_device_noexcept(device_, [&] { (void)cudaStreamDestroy(stream_); });
        }
        device_ = other.device_;
        stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
}

StreamHandle current_stream(int device) {
    if (device < 0) {
        MFQ_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
    }
    {
        std::lock_guard lock(active_streams_mutex);
        const auto thread = active_streams.find(std::this_thread::get_id());
        if (thread != active_streams.end()) {
            const auto found = thread->second.find(device);
            if (found != thread->second.end() && found->second) {
                return found->second;
            }
        }
    }
    auto context = default_context(device);
    auto* stream = &context->stream();
    auto owner = std::shared_ptr<Stream>(context, stream);
    return StreamHandle(device, stream->get(), std::move(owner));
}

StreamHandle stream_from_pool(bool high_priority, int device) {
    if (device < 0) {
        MFQ_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
    }
    int least = 0;
    int greatest = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    const int priority = high_priority ? greatest : 0;
    DeviceGuard guard(device);
    cudaStream_t raw = nullptr;
    MFQ_NATIVE_CUDA_CHECK(cudaStreamCreateWithPriority(
        &raw, cudaStreamNonBlocking, priority));
    // Stream's ordinary constructor cannot adopt a raw stream, so retain it
    // through an independent shared owner with the same value semantics.
    auto owner = std::shared_ptr<Stream>();
    auto raw_owner = std::shared_ptr<void>(
        reinterpret_cast<void*>(raw), [device](void* value) {
            on_device_noexcept(device, [&] {
                (void)cudaStreamDestroy(reinterpret_cast<cudaStream_t>(value));
            });
        });
    // Alias the raw lifetime to a dummy Stream pointer.  StreamHandle only
    // needs shared ownership; it never dereferences owner_.
    owner = std::shared_ptr<Stream>(raw_owner, static_cast<Stream*>(nullptr));
    return StreamHandle(device, raw, std::move(owner));
}

StreamGuard::StreamGuard(StreamHandle stream) : device_(stream.device_index()) {
    std::lock_guard lock(active_streams_mutex);
    auto& streams = active_streams[std::this_thread::get_id()];
    if (const auto found = streams.find(device_); found != streams.end()) {
        previous_ = found->second;
    }
    streams[device_] = std::move(stream);
}

StreamGuard::~StreamGuard() noexcept {
    std::lock_guard lock(active_streams_mutex);
    const auto thread_id = std::this_thread::get_id();
    auto& streams = active_streams[thread_id];
    if (previous_.has_value()) {
        streams[device_] = std::move(*previous_);
    } else {
        streams.erase(device_);
        if (streams.empty()) active_streams.erase(thread_id);
    }
}

Graph::~Graph() noexcept {
    reset();
}

Graph::Graph(Graph&& other) noexcept
    : device_(other.device_),
      stream_(std::move(other.stream_)),
      context_(std::move(other.context_)),
      pool_streams_(std::move(other.pool_streams_)),
      pool_contexts_(std::move(other.pool_contexts_)),
      graph_(std::exchange(other.graph_, nullptr)),
      executable_(std::exchange(other.executable_, nullptr)),
      shared_pool_(std::exchange(other.shared_pool_, false)) {}

Graph& Graph::operator=(Graph&& other) noexcept {
    if (this != &other) {
        reset();
        device_ = other.device_;
        stream_ = std::move(other.stream_);
        context_ = std::move(other.context_);
        pool_streams_ = std::move(other.pool_streams_);
        pool_contexts_ = std::move(other.pool_contexts_);
        graph_ = std::exchange(other.graph_, nullptr);
        executable_ = std::exchange(other.executable_, nullptr);
        shared_pool_ = std::exchange(other.shared_pool_, false);
    }
    return *this;
}

void Graph::prepare_memory() {
    prepare_memory({});
}

void Graph::prepare_memory(
        const std::vector<StreamHandle>& participant_streams) {
    reset();
    stream_ = current_stream();
    device_ = stream_.device_index();
    context_ = default_context(device_);
    const auto add_pool = [&](const StreamHandle& candidate) {
        if (!candidate) return;
        const auto duplicate = std::find_if(
            pool_streams_.begin(), pool_streams_.end(),
            [&](const StreamHandle& existing) {
                return existing.device_index() == candidate.device_index() &&
                    existing.stream() == candidate.stream();
            });
        if (duplicate != pool_streams_.end()) return;
        auto participant_context =
            default_context(candidate.device_index());
        if (!participant_context->supports_async_allocations()) {
            throw Error(
                "CUDA graph participant does not support async allocations");
        }
        participant_context->begin_graph_pool(candidate.stream());
        participant_context->begin_graph_warmup(candidate.stream());
        pool_streams_.push_back(candidate);
        pool_contexts_.push_back(std::move(participant_context));
    };
    try {
        add_pool(stream_);
        for (const auto& participant : participant_streams) {
            add_pool(participant);
        }
    } catch (...) {
        reset();
        throw;
    }
}

void Graph::capture_begin() {
    if (!stream_) {
        prepare_memory();
    }
    const auto active = current_stream();
    if (active.device_index() != device_ ||
            active.stream() != stream_.stream()) {
        throw Error(
            "CUDA graph capture stream differs from its prepared memory stream");
    }
    DeviceGuard guard(device_);
    try {
        for (std::size_t index = 0; index < pool_streams_.size(); ++index) {
            pool_contexts_[index]->begin_graph_capture(
                pool_streams_[index].stream());
        }
        MFQ_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(
            stream_.stream(), cudaStreamCaptureModeGlobal));
    } catch (...) {
        for (std::size_t index = 0; index < pool_streams_.size(); ++index) {
            pool_contexts_[index]->end_graph_capture(
                pool_streams_[index].stream());
        }
        throw;
    }
}

void Graph::capture_begin_shared() {
    if (stream_ || graph_ || executable_) throw Error("graph already owns a capture");
    stream_ = current_stream();
    device_ = stream_.device_index();
    context_ = default_context(device_);
    shared_pool_ = true;
    context_->begin_graph_capture(stream_.stream());
    try {
        MFQ_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(
            stream_.stream(), cudaStreamCaptureModeThreadLocal));
    } catch (...) {
        context_->end_graph_capture(stream_.stream());
        throw;
    }
}

std::size_t Graph::nodes() const {
    if (!graph_) return 0;
    DeviceGuard guard(device_);
    std::size_t count = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &count));
    return count;
}

void Graph::capture_end() {
    if (!stream_) {
        throw Error("CUDA graph capture_end called without capture_begin");
    }
    DeviceGuard guard(device_);
    const auto capture_status =
        cudaStreamEndCapture(stream_.stream(), &graph_);
    if (shared_pool_) context_->end_graph_capture(stream_.stream());
    for (std::size_t index = 0; index < pool_streams_.size(); ++index) {
        pool_contexts_[index]->end_graph_capture(
            pool_streams_[index].stream());
    }
    MFQ_NATIVE_CUDA_CHECK(capture_status);
    if (const char* dot_path = std::getenv("MFQ_NATIVE_CUDA_GRAPH_DOT");
        dot_path != nullptr && dot_path[0] != '\0') {
        MFQ_NATIVE_CUDA_CHECK(cudaGraphDebugDotPrint(
            graph_, dot_path, cudaGraphDebugDotFlagsVerbose));
    }
    if (const char* trace = std::getenv("MFQ_TRACE_NATIVE_CUDA_GRAPH");
        trace != nullptr && std::atoi(trace) != 0) {
        std::size_t count = 0;
        MFQ_NATIVE_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &count));
        std::vector<cudaGraphNode_t> nodes(count);
        MFQ_NATIVE_CUDA_CHECK(cudaGraphGetNodes(graph_, nodes.data(), &count));
        std::unordered_set<void*> allocations;
        std::unordered_set<void*> frees;
        std::unordered_set<void*> reported_kernels;
        const char* resources=std::getenv("MFQ_TRACE_NATIVE_CUDA_KERNEL_RESOURCES");
        std::size_t kernels = 0;
        std::size_t copies = 0;
        for (auto node : nodes) {
            cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
            MFQ_NATIVE_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
            if (type == cudaGraphNodeTypeKernel) {
                ++kernels;
                if(resources && std::atoi(resources)!=0) {
                    cudaKernelNodeParams params{};
                    MFQ_NATIVE_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&params));
                    if(reported_kernels.insert(params.func).second) {
                        cudaFuncAttributes attributes{};
                        MFQ_NATIVE_CUDA_CHECK(cudaFuncGetAttributes(&attributes,params.func));
                        std::cerr<<"native_cuda_kernel function="<<params.func
                            <<" grid="<<params.gridDim.x<<','<<params.gridDim.y<<','<<params.gridDim.z
                            <<" block="<<params.blockDim.x<<','<<params.blockDim.y<<','<<params.blockDim.z
                            <<" registers="<<attributes.numRegs<<" local_bytes="<<attributes.localSizeBytes
                            <<" static_shared_bytes="<<attributes.sharedSizeBytes
                            <<" dynamic_shared_bytes="<<params.sharedMemBytes<<'\n';
                    }
                }
            } else if (type == cudaGraphNodeTypeMemcpy) {
                ++copies;
            } else if (type == cudaGraphNodeTypeMemAlloc) {
                cudaMemAllocNodeParams params{};
                MFQ_NATIVE_CUDA_CHECK(cudaGraphMemAllocNodeGetParams(node, &params));
                allocations.insert(reinterpret_cast<void*>(params.dptr));
            } else if (type == cudaGraphNodeTypeMemFree) {
                void* pointer = nullptr;
                MFQ_NATIVE_CUDA_CHECK(cudaGraphMemFreeNodeGetParams(node, &pointer));
                frees.insert(pointer);
            }
        }
        std::size_t allocation_only = 0;
        std::size_t free_only = 0;
        for (auto pointer : allocations) {
            allocation_only += frees.contains(pointer) ? 0 : 1;
        }
        for (auto pointer : frees) {
            free_only += allocations.contains(pointer) ? 0 : 1;
        }
        std::cerr << "native_cuda_graph nodes=" << count
                  << " kernels=" << kernels
                  << " copies=" << copies
                  << " allocations=" << allocations.size()
                  << " frees=" << frees.size()
                  << " allocation_only=" << allocation_only
                  << " free_only=" << free_only << '\n';
    }
    MFQ_NATIVE_CUDA_CHECK(cudaGraphInstantiate(&executable_, graph_, 0));
    // A first replay may block in the driver while it uploads a large graph.
    // Prepare it before replay can wait for CPU-produced mapped flags.
    if (const char* upload = std::getenv("MFQ_NATIVE_CUDA_GRAPH_UPLOAD");
        upload == nullptr || std::atoi(upload) != 0) {
        MFQ_NATIVE_CUDA_CHECK(cudaGraphUpload(executable_, stream_.stream()));
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));
        default_context(stream_.device_index())->check_memory_limit();
    }
}

void Graph::replay() {
    if (executable_ == nullptr || !stream_) {
        throw Error("CUDA graph replay called before capture");
    }
    DeviceGuard guard(device_);
    MFQ_NATIVE_CUDA_CHECK(cudaGraphLaunch(executable_, stream_.stream()));
}

void Graph::reset() noexcept {
    on_device_noexcept(device_, [&] {
        if (stream_) {
            cudaStreamCaptureStatus capture_status =
                cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(
                    stream_.stream(), &capture_status) == cudaSuccess &&
                    capture_status != cudaStreamCaptureStatusNone) {
                cudaGraph_t abandoned = nullptr;
                if (cudaStreamEndCapture(
                        stream_.stream(), &abandoned) == cudaSuccess &&
                        abandoned != nullptr) {
                    (void)cudaGraphDestroy(abandoned);
                } else {
                    (void)cudaGetLastError();
                }
            }
            (void)cudaStreamSynchronize(stream_.stream());
        }
        if (executable_ != nullptr) {
            (void)cudaGraphExecDestroy(executable_);
            executable_ = nullptr;
        }
        if (graph_ != nullptr) {
            (void)cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        for (std::size_t index = 0; index < pool_streams_.size(); ++index) {
            pool_contexts_[index]->end_graph_capture(
                pool_streams_[index].stream());
            pool_contexts_[index]->end_graph_pool(
                pool_streams_[index].stream());
        }
        if (shared_pool_ && context_) context_->end_graph_capture(stream_.stream());
    });
    pool_contexts_.clear();
    pool_streams_.clear();
    context_.reset();
    stream_ = {};
    shared_pool_ = false;
}

void Stream::synchronize() const {
    DeviceGuard guard(device_);
    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Stream::wait(const Event& event) const {
    DeviceGuard guard(device_);
    MFQ_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(stream_, event.get(), 0));
}

BlasHandle::BlasHandle(int device) : device_(device) {
    DeviceGuard guard(device_);
    MFQ_NATIVE_CUDA_CHECK(cublasCreate(&handle_));
    // Match the established runtime's explicit setAllowTF32CuBLAS(false).
    // Individual GEMM launchers select their exact accumulation type.
    MFQ_NATIVE_CUDA_CHECK(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
}

BlasHandle::~BlasHandle() noexcept {
    if (handle_ != nullptr) {
        on_device_noexcept(device_, [&] { (void)cublasDestroy(handle_); });
    }
}

BlasHandle::BlasHandle(BlasHandle&& other) noexcept
    : device_(other.device_), handle_(std::exchange(other.handle_, nullptr)) {}

BlasHandle& BlasHandle::operator=(BlasHandle&& other) noexcept {
    if (this != &other) {
        if (handle_ != nullptr) {
            on_device_noexcept(device_, [&] { (void)cublasDestroy(handle_); });
        }
        device_ = other.device_;
        handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
}

void BlasHandle::set_stream(cudaStream_t stream) {
    MFQ_NATIVE_CUDA_CHECK(cublasSetStream(handle_, stream));
}

Context::Context(int device)
    : device_(device), stream_(device), blas_(device) {
    DeviceGuard guard(device_);
    allocation_budget_=device_allocation_budget(device_);
    int pools_supported = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(
        &pools_supported, cudaDevAttrMemoryPoolsSupported, device_));
    async_allocations_ = pools_supported != 0;
    if (async_allocations_) {
        MFQ_NATIVE_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool_, device_));
        std::uint64_t threshold = allocation_budget_->limit ? 0 : std::numeric_limits<std::uint64_t>::max();
        MFQ_NATIVE_CUDA_CHECK(cudaMemPoolSetAttribute(
            pool_, cudaMemPoolAttrReleaseThreshold, &threshold));
    }
    blas_.set_stream(stream_.get());
}

DeviceMemoryStats Context::memory_stats() const noexcept {
    return {allocation_budget_->limit,allocation_budget_->allocated.load(std::memory_order_relaxed),
        allocation_budget_->peak.load(std::memory_order_relaxed)};
}

std::size_t Context::local_memory_usage() const {
#ifdef _WIN32
    if(allocation_budget_->adapter) {
        DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
        if(FAILED(allocation_budget_->adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&memory)))
            throw Error("Cannot read WDDM process VRAM usage for the configured limit");
        return static_cast<std::size_t>(memory.CurrentUsage);
    }
#endif
    return memory_stats().allocated;
}

std::size_t Context::resident_memory_usage() const {
    DeviceGuard guard(device_);
    auto bytes=memory_stats().allocated;
    const auto [used,reserved]=pool_memory_stats();
    // Allocated bytes already include live pool and graph allocations.
    // Add only unused pages, never device-wide usage from other processes.
    if(reserved>used)bytes+=reserved-used;
    return std::max(bytes,local_memory_usage());
}

std::optional<std::size_t> shared_memory_available(int device) {
#ifdef _WIN32
    const auto budget=device_allocation_budget(device);
    if(budget->adapter) {
        DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
        if(SUCCEEDED(budget->adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,&memory)))
            return memory.Budget>memory.CurrentUsage?std::size_t(memory.Budget-memory.CurrentUsage):0;
    }
#endif
    return std::nullopt;
}

std::size_t Context::graph_memory_pool_count() const {
    std::lock_guard lock(allocation_budget_->graph_memory_mutex);
    return allocation_budget_->graph_memory_pools.size()+allocation_budget_->idle_graph_pools.size();
}

std::pair<std::uint64_t,std::uint64_t> Context::pool_memory_stats() const {
    DeviceGuard guard(device_);
    std::uint64_t used=0,reserved=0;
    const auto add=[&](cudaMemPool_t memory) {
        if(!memory)return;
        std::uint64_t current_used=0,current_reserved=0;
        MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(memory,cudaMemPoolAttrUsedMemCurrent,&current_used));
        MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(memory,cudaMemPoolAttrReservedMemCurrent,&current_reserved));
        used+=current_used;reserved+=current_reserved;
    };
    add(pool_);
    std::lock_guard lock(allocation_budget_->graph_memory_mutex);
    for(const auto& [memory,retired]:allocation_budget_->graph_memory_pools)add(memory);
    for(const auto memory:allocation_budget_->idle_graph_pools)add(memory);
    return {used,reserved};
}

void Context::trim_pool_pages() const {
    if(pool_)MFQ_NATIVE_CUDA_CHECK(cudaMemPoolTrimTo(pool_,0));
    std::lock_guard lock(allocation_budget_->graph_memory_mutex);
    for(auto entry=allocation_budget_->graph_memory_pools.begin();entry!=allocation_budget_->graph_memory_pools.end();) {
        const auto [memory,retired]=*entry;
        MFQ_NATIVE_CUDA_CHECK(cudaMemPoolTrimTo(memory,0));
        std::uint64_t used=0;
        if(retired)MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(memory,cudaMemPoolAttrUsedMemCurrent,&used));
        if(retired && !used) {
            entry=allocation_budget_->graph_memory_pools.erase(entry);
            park_graph_pool(*allocation_budget_,memory);
        } else ++entry;
    }
    for(const auto memory:allocation_budget_->idle_graph_pools)
        MFQ_NATIVE_CUDA_CHECK(cudaMemPoolTrimTo(memory,0));
}

std::size_t Context::available_memory() const {
    DeviceGuard guard(device_);
    std::size_t free=0,total=0;
    MFQ_NATIVE_CUDA_CHECK(cudaMemGetInfo(&free,&total));
#ifdef _WIN32
    // CUDA may allocate beyond WDDM's current local-memory budget and let the
    // driver page live weights through system RAM. Price revocable caches and
    // new workspace against physical residency, before that silent slowdown.
    if(allocation_budget_->adapter) {
        DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
        if(SUCCEEDED(allocation_budget_->adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&memory))) {
            const auto available=memory.Budget>memory.CurrentUsage?std::size_t(memory.Budget-memory.CurrentUsage):0;
            if(available<free) {
                static std::atomic<bool> reported{false};
                if(std::getenv("MFQ_PROFILE_SSD_EXPERTS") && !reported.exchange(true,std::memory_order_relaxed))
                    std::fprintf(stderr,"cuda_residency_budget scope=first_limit cuda_free_bytes=%zu budget_bytes=%llu usage_bytes=%llu available_bytes=%zu\n",
                        free,static_cast<unsigned long long>(memory.Budget),static_cast<unsigned long long>(memory.CurrentUsage),available);
                free=available;
            }
        }
    }
#endif
    const auto budget=memory_stats();
    if(budget.limit) {
        const auto used=resident_memory_usage();
        free=std::min(free,used<budget.limit?budget.limit-used:0);
    }
    return free;
}

void Context::reclaim_idle_memory() {
    DeviceGuard guard(device_);
    // Phase transitions must retire frees on all compute/transfer streams.
    // Trimming unused pages leaves live graph-pool allocations untouched.
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceSynchronize());
    trim_pool_pages();
    if(const auto* trace=std::getenv("MFQ_TRACE_RESIDENCY_MEMORY");trace && trace[0]=='1') {
        std::lock_guard lock(graph_pool_mutex_);
        std::size_t cached=0,blocks=0;
        for(const auto& [stream,pool]:graph_pools_)for(const auto& [bytes,pointers]:pool.available) {
            cached+=bytes*pointers.size();blocks+=pointers.size();
        }
        std::fprintf(stderr,"graph_workspace_memory cached_bytes=%zu cached_blocks=%zu\n",cached,blocks);
    }
}

void Context::set_memory_limit(std::size_t bytes) {
    // Called by the engine only while no requests or expert transfers are active.
    reclaim_idle_memory();
    if(bytes && bytes < std::max(memory_stats().allocated,local_memory_usage()))
        throw Error("resident memory budget is below live CUDA allocations");
    allocation_budget_->limit=bytes;
}

void Context::check_memory_limit(std::size_t additional) const {
    if(!allocation_budget_->limit)return;
    auto usage=local_memory_usage();
    // Completed asynchronous frees can leave reusable pool pages charged to
    // WDDM. Release those pages before pricing a new allocation against the
    // process limit; the logical allocation budget remains checked separately.
    if(async_allocations_ && pool_ && (usage>allocation_budget_->limit || additional>allocation_budget_->limit-usage)) {
        trim_pool_pages();
        usage=local_memory_usage();
    }
    if(usage>allocation_budget_->limit || additional>allocation_budget_->limit-usage) {
        // A layer transition may still have frees pending on transfer/compute
        // streams. Retire them before rejecting a request against physical use.
        MFQ_NATIVE_CUDA_CHECK(cudaDeviceSynchronize());
        trim_pool_pages();
        usage=local_memory_usage();
    }
    if(usage>allocation_budget_->limit || additional>allocation_budget_->limit-usage) {
        const auto [pool_used,pool_reserved]=pool_memory_stats();
        throw MfqResourceExhausted("CUDA process VRAM limit exceeded: local_usage="+std::to_string(usage)+
            " requested="+std::to_string(additional)+" limit="+std::to_string(allocation_budget_->limit)+
            " owned="+std::to_string(memory_stats().allocated)+" pool_used="+std::to_string(pool_used)+
            " pool_reserved="+std::to_string(pool_reserved));
    }
}

void Context::begin_workspace_loan(const std::vector<std::pair<void*,std::size_t>>& regions,
                                  cudaStream_t stream) {
    std::lock_guard lock(graph_pool_mutex_);
    if(workspace_loan_)throw Error("CUDA workspace loan is already active");
    auto loan=std::make_unique<WorkspaceLoan>();
    loan->stream=stream;loan->thread=std::this_thread::get_id();
    for(const auto& [pointer,bytes]:regions) {
        const auto raw=reinterpret_cast<std::uintptr_t>(pointer);
        const auto begin=(raw+255)&~std::uintptr_t(255);
        if(begin-raw>=bytes)continue;
        const auto size=(bytes-(begin-raw))&~std::size_t(255);
        if(size && !loan->free.emplace(begin,size).second)throw Error("duplicate CUDA workspace loan region");
    }
    for(auto it=loan->free.begin();it!=loan->free.end();++it) {
        const auto next=std::next(it);
        if(next==loan->free.end())break;
        if(it->first+it->second>next->first)throw Error("overlapping CUDA workspace loan regions");
        // Keep a gap between distinct allocations even when the allocator
        // happened to place them adjacently. Never coalesce across their bounds.
        if(it->first+it->second==next->first)it->second-=256;
    }
    workspace_loan_=std::move(loan);
}
void Context::suspend_workspace_loan() {
    std::lock_guard lock(graph_pool_mutex_);
    if(workspace_loan_)workspace_loan_->enabled=false;
}
void Context::end_workspace_loan() {
    std::lock_guard lock(graph_pool_mutex_);
    if(workspace_loan_ && !workspace_loan_->live.empty()) {
        std::size_t bytes=0;for(const auto& [p,n]:workspace_loan_->live)bytes+=n;
        throw Error("CUDA workspace loan still has live tensors: count="+
            std::to_string(workspace_loan_->live.size())+" bytes="+std::to_string(bytes));
    }
    workspace_loan_.reset();
}
std::size_t Context::workspace_available() const {
    std::lock_guard lock(graph_pool_mutex_);
    std::size_t bytes=0;
    if(workspace_loan_ && workspace_loan_->enabled && workspace_loan_->thread==std::this_thread::get_id())
        for(const auto& [p,n]:workspace_loan_->free)bytes+=n;
    return bytes;
}

std::vector<std::pair<void*,std::size_t>> Context::workspace_touched_regions() const {
    std::lock_guard lock(graph_pool_mutex_);
    std::vector<std::pair<void*,std::size_t>> result;
    if(!workspace_loan_)return result;
    auto ranges=workspace_loan_->touched;
    std::sort(ranges.begin(),ranges.end());
    for(const auto& [begin,size]:ranges) {
        if(!result.empty()) {
            auto& last=result.back();
            const auto previous=reinterpret_cast<std::uintptr_t>(last.first);
            if(begin<=previous+last.second) {
                last.second=std::max(previous+last.second,begin+size)-previous;
                continue;
            }
        }
        result.emplace_back(reinterpret_cast<void*>(begin),size);
    }
    return result;
}

void* Context::allocate(std::size_t bytes, cudaStream_t stream) {
    if (bytes == 0) {
        return nullptr;
    }
    DeviceGuard guard(device_);
    const auto allocation_stream =
        stream != nullptr ? stream : stream_.get();
    bool capture_pool_miss = false,pooled_allocation=false;
    cudaMemPool_t allocation_pool=pool_;
    {
        std::lock_guard lock(graph_pool_mutex_);
        if (auto pool = graph_pools_.find(allocation_stream);
            pool != graph_pools_.end() && (pool->second.warming || pool->second.capturing)) {
            if(pool->second.memory)allocation_pool=pool->second.memory;
            pooled_allocation=true;
            // Stable size classes let serial graph shapes share scratch. The
            // same request must use the same class in warmup and capture; a
            // free-form best-fit choice can consume a later operation's slot.
            if(bytes>=4096) {
                std::size_t capacity=4096;
                while(capacity<bytes) {
                    if(capacity>SIZE_MAX/2)throw Error("CUDA graph workspace size overflows");
                    capacity*=2;
                }
                bytes=capacity;
            }
            auto available = pool->second.available.find(bytes);
            if (available != pool->second.available.end() &&
                    !available->second.empty()) {
                void* pointer = available->second.back();
                available->second.pop_back();
                return pointer;
            }
            capture_pool_miss = pool->second.capturing;
        }
        if(!pooled_allocation && !resident_allocations && workspace_loan_ &&
                workspace_loan_->enabled && workspace_loan_->stream==allocation_stream &&
                workspace_loan_->thread==std::this_thread::get_id() && bytes<=SIZE_MAX-255) {
            auto& loan=*workspace_loan_;
            const auto aligned=(bytes+255)&~std::size_t(255);
            // Best fit keeps the large regions available for transfer images.
            auto best=loan.free.end();
            for(auto it=loan.free.begin();it!=loan.free.end();++it)
                if(it->second>=aligned && (best==loan.free.end() || it->second<best->second))best=it;
            if(best!=loan.free.end()) {
                const auto base=best->first,size=best->second;
                auto* result=reinterpret_cast<void*>(base);
                // Record conservatively before publishing the allocation.
                // A failed bookkeeping allocation can only cause extra restore.
                loan.touched.emplace_back(base,aligned);
                loan.live.emplace(result,aligned);
                loan.free.erase(best);
                if(size>aligned)loan.free.emplace(base+aligned,size-aligned);
                return result;
            }
        }
    }
    if (capture_pool_miss) {
        throw Error(
            "CUDA graph capture requested an un-warmed allocation of " +
            std::to_string(bytes) + " bytes");
    }
    void* pointer = nullptr;
    const bool async=async_allocations_;
    if(async && resident_allocations && !pooled_allocation) {
        // Pack the many small recurrent tensors into their own pages instead
        // of paying cudaMalloc's WDDM allocation granularity for every layer.
        std::lock_guard lock(allocation_budget_->graph_memory_mutex);
        auto& memory=allocation_budget_->resident_memory_pool;
        if(!memory) {
            cudaMemPoolProps properties{};
            properties.allocType=cudaMemAllocationTypePinned;
            properties.location.type=cudaMemLocationTypeDevice;
            properties.location.id=device_;
            MFQ_NATIVE_CUDA_CHECK(cudaMemPoolCreate(&memory,&properties));
            try {
                std::uint64_t threshold=std::numeric_limits<std::uint64_t>::max();
                MFQ_NATIVE_CUDA_CHECK(cudaMemPoolSetAttribute(memory,cudaMemPoolAttrReleaseThreshold,&threshold));
                allocation_budget_->graph_memory_pools.emplace(memory,false);
            } catch(...) {(void)cudaMemPoolDestroy(memory);memory=nullptr;throw;}
        }
        allocation_pool=memory;
    }
    bool verify_pool_reuse=false;
    if(allocation_budget_->limit) {
        cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
        MFQ_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(allocation_stream,&capture));
        if(capture==cudaStreamCaptureStatusNone) {
            const auto usage=local_memory_usage();
            if(async && allocation_pool && usage<=allocation_budget_->limit &&
                    bytes>allocation_budget_->limit-usage) {
                std::uint64_t used=0,reserved=0;
                MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(allocation_pool,cudaMemPoolAttrUsedMemCurrent,&used));
                MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(allocation_pool,cudaMemPoolAttrReservedMemCurrent,&reserved));
                // Reusing existing pages need not increase WDDM's charge.
                // Check the actual charge after malloc and roll back if the
                // allocator had to grow the pool instead.
                verify_pool_reuse=reserved>=used && bytes<=reserved-used;
            }
            if(!verify_pool_reuse && (usage>allocation_budget_->limit || bytes>allocation_budget_->limit-usage))
                check_memory_limit(bytes);
        }
    }
    allocation_budget_->acquire(bytes);
    try {
        auto result=async ? cudaMallocFromPoolAsync(&pointer,bytes,allocation_pool,allocation_stream)
            : cudaMalloc(&pointer,bytes);
        if(result==cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            reclaim_idle_memory();
            result=async ? cudaMallocFromPoolAsync(&pointer,bytes,allocation_pool,allocation_stream)
                : cudaMalloc(&pointer,bytes);
        }
        if(result==cudaErrorMemoryAllocation)
            throw MfqResourceExhausted("CUDA allocation failed: requested="+std::to_string(bytes)+
                " allocated="+std::to_string(memory_stats().allocated));
        MFQ_NATIVE_CUDA_CHECK(result);
        if(verify_pool_reuse)check_memory_limit();
        if(pooled_allocation) {
            std::lock_guard lock(graph_pool_mutex_);
            graph_allocation_sizes_.emplace(pointer,bytes);
            graph_pools_.at(allocation_stream).owned.insert(pointer);
        }
    } catch(...) {
        if(pooled_allocation && pointer) {
            std::lock_guard lock(graph_pool_mutex_);
            graph_allocation_sizes_.erase(pointer);
            graph_pools_.at(allocation_stream).owned.erase(pointer);
        }
        if(pointer) {if(async_allocations_)(void)cudaFreeAsync(pointer,allocation_stream);else (void)cudaFree(pointer);}
        allocation_budget_->release(bytes);throw;
    }
    if(const auto* trace=std::getenv("MFQ_TRACE_LARGE_ALLOCATIONS");trace && trace[0]=='1' && bytes>=(std::size_t(1)<<20))
        std::cerr<<"cuda_large_allocate pointer="<<pointer<<" bytes="<<bytes
            <<" graph="<<pooled_allocation<<" resident="<<resident_allocations<<'\n';
    return pointer;
}

void Context::release(
        void* pointer,
        std::size_t bytes,
        cudaStream_t stream) noexcept {
    if (pointer == nullptr) {
        return;
    }
    const auto allocation_stream =
        stream != nullptr ? stream : stream_.get();
    try {
        std::lock_guard lock(graph_pool_mutex_);
        if(workspace_loan_) {
            auto& loan=*workspace_loan_;
            if(auto it=loan.live.find(pointer);it!=loan.live.end()) {
                const auto base=reinterpret_cast<std::uintptr_t>(pointer),size=it->second;
                auto entry=loan.free.emplace(base,size).first;
                loan.live.erase(it);
                if(entry!=loan.free.begin()) {
                    auto before=std::prev(entry);
                    if(before->first+before->second==entry->first) {
                        before->second+=entry->second;loan.free.erase(entry);entry=before;
                    }
                }
                auto after=std::next(entry);
                if(after!=loan.free.end() && entry->first+entry->second==after->first) {
                    entry->second+=after->second;loan.free.erase(after);
                }
                return;
            }
        }
        if(auto capacity=graph_allocation_sizes_.find(pointer);capacity!=graph_allocation_sizes_.end())
            bytes=capacity->second;
        if (auto pool = graph_pools_.find(allocation_stream);
            pool != graph_pools_.end() && pool->second.owned.count(pointer)) {
            pool->second.available[bytes].push_back(pointer);
            return;
        }
        graph_allocation_sizes_.erase(pointer);
    } catch (...) {
        // Tensor destruction is noexcept. A warm-up bookkeeping allocation
        // failure must release the CUDA allocation instead of terminating or
        // leaking it. During capture the size bucket already exists and the
        // preceding pop leaves enough vector capacity for this push.
        std::lock_guard lock(graph_pool_mutex_);
        // A host bookkeeping failure must never cudaFree an expert-cache
        // suballocation. Keeping the live record makes return fail safely.
        if(workspace_loan_ && workspace_loan_->live.count(pointer))return;
        if (auto pool = graph_pools_.find(allocation_stream); pool != graph_pools_.end())
            pool->second.owned.erase(pointer);
        graph_allocation_sizes_.erase(pointer);
    }
    on_device_noexcept(device_, [&] {
        if (async_allocations_) {
            (void)cudaFreeAsync(pointer, allocation_stream);
        } else {
            (void)cudaFree(pointer);
        }
    });
    if(const auto* trace=std::getenv("MFQ_TRACE_LARGE_ALLOCATIONS");trace && trace[0]=='1' && bytes>=(std::size_t(1)<<20))
        std::cerr<<"cuda_large_release pointer="<<pointer<<" bytes="<<bytes<<'\n';
    allocation_budget_->release(bytes);
}

void Context::begin_graph_pool(cudaStream_t stream) {
    if (stream == nullptr) {
        throw std::invalid_argument("CUDA graph memory pool requires a stream");
    }
    std::lock_guard lock(graph_pool_mutex_);
    const auto [_, inserted] = graph_pools_.try_emplace(stream);
    if (!inserted) {
        throw Error("CUDA graph memory pool is already active on this stream");
    }
    if(async_allocations_) {
        // Graph scratch has a short lifetime compared with resident model
        // weights. Keep its backing pages separate so retiring a window can
        // return all of them despite live allocations in the default pool.
        DeviceGuard guard(device_);
        cudaMemPoolProps properties{};
        properties.allocType=cudaMemAllocationTypePinned;
        properties.location.type=cudaMemLocationTypeDevice;
        properties.location.id=device_;
        cudaMemPool_t memory=nullptr;
        bool created=false;
        try {
            // A workspace may outlive its graph by one operation. Recycle its
            // retired pool once that allocation is freed; pool handles are a
            // finite driver resource even when their pages have been trimmed.
            {
                std::lock_guard shared_lock(allocation_budget_->graph_memory_mutex);
                for(auto entry=allocation_budget_->graph_memory_pools.begin();
                        entry!=allocation_budget_->graph_memory_pools.end();) {
                    std::uint64_t used=0;
                    if(entry->second)MFQ_NATIVE_CUDA_CHECK(cudaMemPoolGetAttribute(
                        entry->first,cudaMemPoolAttrUsedMemCurrent,&used));
                    if(entry->second && !used) {
                        const auto drained=entry->first;
                        entry=allocation_budget_->graph_memory_pools.erase(entry);
                        park_graph_pool(*allocation_budget_,drained);
                    } else ++entry;
                }
                auto& idle=allocation_budget_->idle_graph_pools;
                if(!idle.empty()) {memory=idle.back();idle.pop_back();}
            }
            if(!memory) {
                MFQ_NATIVE_CUDA_CHECK(cudaMemPoolCreate(&memory,&properties));
                created=true;
                std::uint64_t threshold=std::numeric_limits<std::uint64_t>::max();
                MFQ_NATIVE_CUDA_CHECK(cudaMemPoolSetAttribute(memory,cudaMemPoolAttrReleaseThreshold,&threshold));
            }
            std::lock_guard shared_lock(allocation_budget_->graph_memory_mutex);
            allocation_budget_->graph_memory_pools.emplace(memory,false);
            graph_pools_.at(stream).memory=memory;
        } catch(...) {
            if(memory && created)(void)cudaMemPoolDestroy(memory);
            else if(memory) {
                std::lock_guard shared_lock(allocation_budget_->graph_memory_mutex);
                park_graph_pool(*allocation_budget_,memory);
            }
            graph_pools_.erase(stream);throw;
        }
    }
}

void Context::begin_graph_capture(cudaStream_t stream) {
    std::lock_guard lock(graph_pool_mutex_);
    const auto pool = graph_pools_.find(stream);
    if (pool == graph_pools_.end()) {
        throw Error("CUDA graph capture requires a prepared memory pool");
    }
    if (pool->second.capturing) {
        throw Error("CUDA graph memory pool is already capturing");
    }
    pool->second.capturing = true;
}
void Context::begin_graph_warmup(cudaStream_t stream) {
    std::lock_guard lock(graph_pool_mutex_);auto& pool=graph_pools_.at(stream);
    if(pool.warming || pool.capturing)throw Error("nested CUDA graph warmup");pool.warming=true;
}
void Context::end_graph_warmup(cudaStream_t stream) noexcept {
    std::lock_guard lock(graph_pool_mutex_);const auto pool=graph_pools_.find(stream);
    if(pool!=graph_pools_.end())pool->second.warming=false;
}

void Context::end_graph_capture(cudaStream_t stream) noexcept {
    std::lock_guard lock(graph_pool_mutex_);
    if (const auto pool = graph_pools_.find(stream);
        pool != graph_pools_.end()) {
        pool->second.capturing = false;
        pool->second.warming = false;
    }
}

void Context::end_graph_pool(cudaStream_t stream) noexcept {
    GraphPool pool;
    {
        std::lock_guard lock(graph_pool_mutex_);
        const auto found = graph_pools_.find(stream);
        if (found == graph_pools_.end()) {
            return;
        }
        for(const auto& [bytes,pointers]:found->second.available)
            for(void* pointer:pointers)graph_allocation_sizes_.erase(pointer);
        pool = std::move(found->second);
        graph_pools_.erase(found);
    }
    on_device_noexcept(device_, [&] {
        for (const auto& [bytes, pointers] : pool.available) {
            for (void* pointer : pointers) {
                if (async_allocations_) {
                    (void)cudaFreeAsync(pointer, stream);
                } else {
                    (void)cudaFree(pointer);
                }
                allocation_budget_->release(bytes);
            }
        }
        if (async_allocations_) {
            (void)cudaStreamSynchronize(stream);
        }
        // Request workspaces can outlive their graph. Keep that CUDA pool
        // registered until their asynchronous releases finish as well.
        if(pool.memory) {
            std::lock_guard lock(allocation_budget_->graph_memory_mutex);
            std::uint64_t used=0;
            if(cudaMemPoolGetAttribute(pool.memory,cudaMemPoolAttrUsedMemCurrent,&used)==cudaSuccess && !used) {
                allocation_budget_->graph_memory_pools.erase(pool.memory);
                park_graph_pool(*allocation_budget_,pool.memory);
            } else allocation_budget_->graph_memory_pools.at(pool.memory)=true;
        }
    });
}

void Context::trim() {
    {
        std::lock_guard lock(graph_pool_mutex_);
        if (!graph_pools_.empty()) {
            throw Error("cannot trim CUDA memory while a graph pool is active");
        }
    }
    stream_.synchronize();
    if (async_allocations_) {
        DeviceGuard guard(device_);
        trim_pool_pages();
    }
    check_memory_limit();
}

Buffer::Buffer(std::shared_ptr<Context> context, std::size_t bytes)
    : context_(std::move(context)), bytes_(bytes) {
    if (!context_) {
        throw std::invalid_argument("CUDA buffer requires a context");
    }
    stream_ = current_stream(context_->device());
    data_ = context_->allocate(bytes_, stream_.stream());
}

Buffer::~Buffer() noexcept {
    reset();
}

Buffer::Buffer(Buffer&& other) noexcept
    : context_(std::move(other.context_)),
      data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)),
      stream_(std::exchange(other.stream_, {})) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        reset();
        context_ = std::move(other.context_);
        data_ = std::exchange(other.data_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
        stream_ = std::exchange(other.stream_, {});
    }
    return *this;
}

void Buffer::reset() noexcept {
    if (context_ && data_ != nullptr) {
        context_->release(data_, bytes_, stream_.stream());
    }
    data_ = nullptr;
    bytes_ = 0;
    stream_ = {};
    context_.reset();
}

HostBuffer::HostBuffer(std::size_t bytes, bool mapped) : bytes_(bytes) {
    if (bytes_ == 0) {
        return;
    }
    const auto flags = mapped ? cudaHostAllocMapped : cudaHostAllocDefault;
    const auto allocation=cudaHostAlloc(&data_,bytes_,flags);
    if(pinned_allocation_rejected(allocation)) {
        (void)cudaGetLastError();
        data_=nullptr;
        throw MfqResourceExhausted("CUDA pinned allocation failed: requested="+std::to_string(bytes_)+
            " error="+cudaGetErrorName(allocation));
    }
    MFQ_NATIVE_CUDA_CHECK(allocation);
}

HostBuffer::~HostBuffer() noexcept {
    reset();
}

HostBuffer::HostBuffer(HostBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)) {}

HostBuffer& HostBuffer::operator=(HostBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = std::exchange(other.data_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
}

void HostBuffer::reset() noexcept {
    if (data_ != nullptr) {
        (void)cudaFreeHost(data_);
    }
    data_ = nullptr;
    bytes_ = 0;
}

}  // namespace mfq::cuda
