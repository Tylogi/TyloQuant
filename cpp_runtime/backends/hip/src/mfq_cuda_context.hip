#include "hip/hip_runtime.h"
#include "mfq_hip_context.h"
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
struct AllocationBudget {
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
                throw Error("CUDA VRAM limit exceeded: requested="+std::to_string(bytes)+
                    " allocated="+std::to_string(before)+" limit="+std::to_string(limit));
        } while(!allocated.compare_exchange_weak(before,before+bytes,std::memory_order_relaxed));
        auto maximum=peak.load(std::memory_order_relaxed);
        while(maximum<before+bytes && !peak.compare_exchange_weak(maximum,before+bytes,std::memory_order_relaxed)){}
    }
    void release(std::size_t bytes) noexcept {allocated.fetch_sub(bytes,std::memory_order_relaxed);}
};
namespace {

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
        if(candidate->limit) {
            Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
            hipDeviceProp_t properties{};MFQ_NATIVE_CUDA_CHECK(hipGetDeviceProperties(&properties,device));
            if(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
                for(UINT index=0;;++index) {
                    Microsoft::WRL::ComPtr<IDXGIAdapter1> item;
                    if(factory->EnumAdapters1(index,&item)==DXGI_ERROR_NOT_FOUND)break;
                    DXGI_ADAPTER_DESC1 description{};
                    if(SUCCEEDED(item->GetDesc1(&description)) &&
                        !std::memcmp(properties.luid,&description.AdapterLuid,8)) {item.As(&candidate->adapter);break;}
                }
            }
            if(!candidate->adapter)throw Error("CUDA VRAM limit requires the matching WDDM adapter");
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
    if (hipGetDevice(&previous) != hipSuccess) return;
    const bool restore = previous != device;
    if (restore && hipSetDevice(device) != hipSuccess) return;
    try {
        function();
    } catch (...) {
        // Resource destruction must never terminate inference during unwinding.
    }
    if (restore) (void)hipSetDevice(previous);
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

const char* cublas_status_name(hipblasStatus_t status) {
    switch (status) {
        case HIPBLAS_STATUS_SUCCESS: return "HIPBLAS_STATUS_SUCCESS";
        case HIPBLAS_STATUS_NOT_INITIALIZED: return "HIPBLAS_STATUS_NOT_INITIALIZED";
        case HIPBLAS_STATUS_ALLOC_FAILED: return "HIPBLAS_STATUS_ALLOC_FAILED";
        case HIPBLAS_STATUS_INVALID_VALUE: return "HIPBLAS_STATUS_INVALID_VALUE";
        case HIPBLAS_STATUS_ARCH_MISMATCH: return "HIPBLAS_STATUS_ARCH_MISMATCH";
        case HIPBLAS_STATUS_MAPPING_ERROR: return "HIPBLAS_STATUS_MAPPING_ERROR";
        case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
        case HIPBLAS_STATUS_INTERNAL_ERROR: return "HIPBLAS_STATUS_INTERNAL_ERROR";
        case HIPBLAS_STATUS_NOT_SUPPORTED: return "HIPBLAS_STATUS_NOT_SUPPORTED";
        case HIPBLAS_STATUS_UNKNOWN: return "HIPBLAS_STATUS_UNKNOWN";
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

void check(hipError_t status, const char* expression, const char* file, int line) {
    if (status != hipSuccess) {
        if (status == hipErrorIllegalAddress || status == hipErrorAssert ||
            status == hipErrorLaunchFailure || status == hipErrorECCNotCorrectable ||
            status == hipErrorContextIsDestroyed) {
            throw MfqBackendFailure(location_message(
                "HIP", hipGetErrorString(status), expression, file, line));
        }
        if (status == hipErrorOutOfMemory)
            throw MfqResourceExhausted(location_message(
                "HIP", hipGetErrorString(status), expression, file, line));
        throw Error(location_message(
            "CUDA", hipGetErrorString(status), expression, file, line));
    }
}

void check(hipblasStatus_t status, const char* expression, const char* file, int line) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        if (status == HIPBLAS_STATUS_ALLOC_FAILED)
            throw MfqResourceExhausted(location_message(
                "hipBLAS", cublas_status_name(status), expression, file, line));
        throw Error(location_message(
            "cuBLAS", cublas_status_name(status), expression, file, line));
    }
}

DeviceGuard::DeviceGuard(int device) {
    MFQ_NATIVE_CUDA_CHECK(hipGetDevice(&previous_));
    if (previous_ != device) {
        MFQ_NATIVE_CUDA_CHECK(hipSetDevice(device));
        restore_ = true;
    }
}

DeviceGuard::~DeviceGuard() noexcept {
    if (restore_) {
        (void)hipSetDevice(previous_);
    }
}

Event::Event(unsigned flags) {
    MFQ_NATIVE_CUDA_CHECK(hipGetDevice(&device_));
    MFQ_NATIVE_CUDA_CHECK(hipEventCreateWithFlags(&event_, flags));
}

Event::~Event() noexcept {
    if (event_ != nullptr) {
        on_device_noexcept(device_, [&] { (void)hipEventDestroy(event_); });
    }
}

Event::Event(Event&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

Event& Event::operator=(Event&& other) noexcept {
    if (this != &other) {
        if (event_ != nullptr) {
            on_device_noexcept(device_, [&] { (void)hipEventDestroy(event_); });
        }
        device_ = other.device_;
        event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
}

void Event::record(hipStream_t stream) {
    DeviceGuard guard(device_);
    MFQ_NATIVE_CUDA_CHECK(hipEventRecord(event_, stream));
}

bool Event::ready() const {
    DeviceGuard guard(device_);
    const auto status = hipEventQuery(event_);
    if (status == hipSuccess) {
        return true;
    }
    if (status == hipErrorNotReady) {
        (void)hipGetLastError();
        return false;
    }
    MFQ_NATIVE_CUDA_CHECK(status);
    return false;
}

void Event::synchronize() const {
    DeviceGuard guard(device_);
    MFQ_NATIVE_CUDA_CHECK(hipEventSynchronize(event_));
}

Stream::Stream(int device, unsigned flags) : device_(device) {
    DeviceGuard guard(device_);
    MFQ_NATIVE_CUDA_CHECK(hipStreamCreateWithFlags(&stream_, flags));
}

Stream::~Stream() noexcept {
    if (stream_ != nullptr) {
        on_device_noexcept(device_, [&] { (void)hipStreamDestroy(stream_); });
    }
}

Stream::Stream(Stream&& other) noexcept
    : device_(other.device_), stream_(std::exchange(other.stream_, nullptr)) {}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        if (stream_ != nullptr) {
            on_device_noexcept(device_, [&] { (void)hipStreamDestroy(stream_); });
        }
        device_ = other.device_;
        stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
}

StreamHandle current_stream(int device) {
    if (device < 0) {
        MFQ_NATIVE_CUDA_CHECK(hipGetDevice(&device));
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
        MFQ_NATIVE_CUDA_CHECK(hipGetDevice(&device));
    }
    int least = 0;
    int greatest = 0;
    MFQ_NATIVE_CUDA_CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
    const int priority = high_priority ? greatest : 0;
    DeviceGuard guard(device);
    hipStream_t raw = nullptr;
    MFQ_NATIVE_CUDA_CHECK(hipStreamCreateWithPriority(
        &raw, hipStreamNonBlocking, priority));
    // Stream's ordinary constructor cannot adopt a raw stream, so retain it
    // through an independent shared owner with the same value semantics.
    auto owner = std::shared_ptr<Stream>();
    auto raw_owner = std::shared_ptr<void>(
        reinterpret_cast<void*>(raw), [device](void* value) {
            on_device_noexcept(device, [&] {
                (void)hipStreamDestroy(reinterpret_cast<hipStream_t>(value));
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
        MFQ_NATIVE_CUDA_CHECK(hipStreamBeginCapture(
            stream_.stream(), hipStreamCaptureModeGlobal));
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
        MFQ_NATIVE_CUDA_CHECK(hipStreamBeginCapture(
            stream_.stream(), hipStreamCaptureModeThreadLocal));
    } catch (...) {
        context_->end_graph_capture(stream_.stream());
        throw;
    }
}

std::size_t Graph::nodes() const {
    if (!graph_) return 0;
    DeviceGuard guard(device_);
    std::size_t count = 0;
    MFQ_NATIVE_CUDA_CHECK(hipGraphGetNodes(graph_, nullptr, &count));
    return count;
}

void Graph::capture_end() {
    if (!stream_) {
        throw Error("CUDA graph capture_end called without capture_begin");
    }
    DeviceGuard guard(device_);
    const auto capture_status =
        hipStreamEndCapture(stream_.stream(), &graph_);
    if (shared_pool_) context_->end_graph_capture(stream_.stream());
    for (std::size_t index = 0; index < pool_streams_.size(); ++index) {
        pool_contexts_[index]->end_graph_capture(
            pool_streams_[index].stream());
    }
    MFQ_NATIVE_CUDA_CHECK(capture_status);
    if (const char* dot_path = std::getenv("MFQ_NATIVE_CUDA_GRAPH_DOT");
        dot_path != nullptr && dot_path[0] != '\0') {
        MFQ_NATIVE_CUDA_CHECK(hipGraphDebugDotPrint(
            graph_, dot_path, hipGraphDebugDotFlagsVerbose));
    }
    if (const char* trace = std::getenv("MFQ_TRACE_NATIVE_CUDA_GRAPH");
        trace != nullptr && std::atoi(trace) != 0) {
        std::size_t count = 0;
        MFQ_NATIVE_CUDA_CHECK(hipGraphGetNodes(graph_, nullptr, &count));
        std::vector<hipGraphNode_t> nodes(count);
        MFQ_NATIVE_CUDA_CHECK(hipGraphGetNodes(graph_, nodes.data(), &count));
        std::unordered_set<void*> allocations;
        std::unordered_set<void*> frees;
        std::size_t kernels = 0;
        std::size_t copies = 0;
        for (auto node : nodes) {
            hipGraphNodeType type = hipGraphNodeTypeEmpty;
            MFQ_NATIVE_CUDA_CHECK(hipGraphNodeGetType(node, &type));
            if (type == hipGraphNodeTypeKernel) {
                ++kernels;
            } else if (type == hipGraphNodeTypeMemcpy) {
                ++copies;
            } else if (type == hipGraphNodeTypeMemAlloc) {
                hipMemAllocNodeParams params{};
                MFQ_NATIVE_CUDA_CHECK(hipGraphMemAllocNodeGetParams(node, &params));
                allocations.insert(reinterpret_cast<void*>(params.dptr));
            } else if (type == hipGraphNodeTypeMemFree) {
                void* pointer = nullptr;
                MFQ_NATIVE_CUDA_CHECK(hipGraphMemFreeNodeGetParams(node, &pointer));
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
    MFQ_NATIVE_CUDA_CHECK(hipGraphInstantiateWithFlags(&executable_, graph_, 0));
    // A first replay may block in the driver while it uploads a large graph.
    // Prepare it before replay can wait for CPU-produced mapped flags.
    if (const char* upload = std::getenv("MFQ_NATIVE_CUDA_GRAPH_UPLOAD");
        upload == nullptr || std::atoi(upload) != 0) {
        MFQ_NATIVE_CUDA_CHECK(hipGraphUpload(executable_, stream_.stream()));
        MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(stream_.stream()));
        default_context(stream_.device_index())->check_memory_limit();
    }
}

void Graph::replay() {
    if (executable_ == nullptr || !stream_) {
        throw Error("CUDA graph replay called before capture");
    }
    DeviceGuard guard(device_);
    MFQ_NATIVE_CUDA_CHECK(hipGraphLaunch(executable_, stream_.stream()));
}

void Graph::reset() noexcept {
    on_device_noexcept(device_, [&] {
        if (stream_) {
            hipStreamCaptureStatus capture_status =
                hipStreamCaptureStatusNone;
            if (hipStreamIsCapturing(
                    stream_.stream(), &capture_status) == hipSuccess &&
                    capture_status != hipStreamCaptureStatusNone) {
                hipGraph_t abandoned = nullptr;
                if (hipStreamEndCapture(
                        stream_.stream(), &abandoned) == hipSuccess &&
                        abandoned != nullptr) {
                    (void)hipGraphDestroy(abandoned);
                } else {
                    (void)hipGetLastError();
                }
            }
            (void)hipStreamSynchronize(stream_.stream());
        }
        if (executable_ != nullptr) {
            (void)hipGraphExecDestroy(executable_);
            executable_ = nullptr;
        }
        if (graph_ != nullptr) {
            (void)hipGraphDestroy(graph_);
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
    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(stream_));
}

void Stream::wait(const Event& event) const {
    DeviceGuard guard(device_);
    MFQ_NATIVE_CUDA_CHECK(hipStreamWaitEvent(stream_, event.get(), 0));
}

BlasHandle::BlasHandle(int device) : device_(device) {
    DeviceGuard guard(device_);
    MFQ_NATIVE_CUDA_CHECK(hipblasCreate(&handle_));
    // Match the established runtime's explicit setAllowTF32CuBLAS(false).
    // Individual GEMM launchers select their exact accumulation type.
    MFQ_NATIVE_CUDA_CHECK(hipblasSetMathMode(handle_, HIPBLAS_DEFAULT_MATH));
}

BlasHandle::~BlasHandle() noexcept {
    if (handle_ != nullptr) {
        on_device_noexcept(device_, [&] { (void)hipblasDestroy(handle_); });
    }
}

BlasHandle::BlasHandle(BlasHandle&& other) noexcept
    : device_(other.device_), handle_(std::exchange(other.handle_, nullptr)) {}

BlasHandle& BlasHandle::operator=(BlasHandle&& other) noexcept {
    if (this != &other) {
        if (handle_ != nullptr) {
            on_device_noexcept(device_, [&] { (void)hipblasDestroy(handle_); });
        }
        device_ = other.device_;
        handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
}

void BlasHandle::set_stream(hipStream_t stream) {
    MFQ_NATIVE_CUDA_CHECK(hipblasSetStream(handle_, stream));
}

Context::Context(int device)
    : device_(device), stream_(device), blas_(device) {
    DeviceGuard guard(device_);
    allocation_budget_=device_allocation_budget(device_);
    int pools_supported = 0;
    MFQ_NATIVE_CUDA_CHECK(hipDeviceGetAttribute(
        &pools_supported, hipDeviceAttributeMemoryPoolsSupported, device_));
    async_allocations_ = pools_supported != 0;
    if (async_allocations_) {
        MFQ_NATIVE_CUDA_CHECK(hipDeviceGetDefaultMemPool(&pool_, device_));
        std::uint64_t threshold = allocation_budget_->limit ? 0 : std::numeric_limits<std::uint64_t>::max();
        MFQ_NATIVE_CUDA_CHECK(hipMemPoolSetAttribute(
            pool_, hipMemPoolAttrReleaseThreshold, &threshold));
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

void Context::check_memory_limit(std::size_t additional) const {
    if(!allocation_budget_->limit)return;
    const auto usage=local_memory_usage();
    if(usage>allocation_budget_->limit || additional>allocation_budget_->limit-usage)
        throw Error("CUDA process VRAM limit exceeded: local_usage="+std::to_string(usage)+
            " requested="+std::to_string(additional)+" limit="+std::to_string(allocation_budget_->limit));
}

void* Context::allocate(std::size_t bytes, hipStream_t stream) {
    if (bytes == 0) {
        return nullptr;
    }
    DeviceGuard guard(device_);
    const auto allocation_stream =
        stream != nullptr ? stream : stream_.get();
    bool capture_pool_miss = false,pooled_allocation=false;
    {
        std::lock_guard lock(graph_pool_mutex_);
        if (auto pool = graph_pools_.find(allocation_stream);
            pool != graph_pools_.end() && (pool->second.warming || pool->second.capturing)) {
            pooled_allocation=true;
            auto available = pool->second.available.find(bytes);
            if (available != pool->second.available.end() &&
                    !available->second.empty()) {
                void* pointer = available->second.back();
                available->second.pop_back();
                return pointer;
            }
            capture_pool_miss = pool->second.capturing;
        }
    }
    if (capture_pool_miss) {
        throw Error(
            "CUDA graph capture requested an un-warmed allocation of " +
            std::to_string(bytes) + " bytes");
    }
    void* pointer = nullptr;
    if(allocation_budget_->limit) {
        hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
        MFQ_NATIVE_CUDA_CHECK(hipStreamIsCapturing(allocation_stream,&capture));
        if(capture==hipStreamCaptureStatusNone)check_memory_limit(bytes);
    }
    allocation_budget_->acquire(bytes);
    try {
        if (async_allocations_) {
            MFQ_NATIVE_CUDA_CHECK(hipMallocAsync(&pointer,bytes,allocation_stream));
        } else {
            MFQ_NATIVE_CUDA_CHECK(hipMalloc(&pointer,bytes));
        }
        if(pooled_allocation) {
            std::lock_guard lock(graph_pool_mutex_);graph_pools_.at(allocation_stream).owned.insert(pointer);
        }
    } catch(...) {
        if(pointer) {if(async_allocations_)(void)hipFreeAsync(pointer,allocation_stream);else (void)hipFree(pointer);}
        allocation_budget_->release(bytes);throw;
    }
    return pointer;
}

void Context::release(
        void* pointer,
        std::size_t bytes,
        hipStream_t stream) noexcept {
    if (pointer == nullptr) {
        return;
    }
    const auto allocation_stream =
        stream != nullptr ? stream : stream_.get();
    try {
        std::lock_guard lock(graph_pool_mutex_);
        if (auto pool = graph_pools_.find(allocation_stream);
            pool != graph_pools_.end() && pool->second.owned.count(pointer)) {
            pool->second.available[bytes].push_back(pointer);
            return;
        }
    } catch (...) {
        // Tensor destruction is noexcept. A warm-up bookkeeping allocation
        // failure must release the CUDA allocation instead of terminating or
        // leaking it. During capture the size bucket already exists and the
        // preceding pop leaves enough vector capacity for this push.
        std::lock_guard lock(graph_pool_mutex_);
        if (auto pool = graph_pools_.find(allocation_stream); pool != graph_pools_.end())
            pool->second.owned.erase(pointer);
    }
    on_device_noexcept(device_, [&] {
        if (async_allocations_) {
            (void)hipFreeAsync(pointer, allocation_stream);
        } else {
            (void)hipFree(pointer);
        }
    });
    allocation_budget_->release(bytes);
}

void Context::begin_graph_pool(hipStream_t stream) {
    if (stream == nullptr) {
        throw std::invalid_argument("CUDA graph memory pool requires a stream");
    }
    std::lock_guard lock(graph_pool_mutex_);
    const auto [_, inserted] = graph_pools_.try_emplace(stream);
    if (!inserted) {
        throw Error("CUDA graph memory pool is already active on this stream");
    }
}

void Context::begin_graph_capture(hipStream_t stream) {
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
void Context::begin_graph_warmup(hipStream_t stream) {
    std::lock_guard lock(graph_pool_mutex_);auto& pool=graph_pools_.at(stream);
    if(pool.warming || pool.capturing)throw Error("nested CUDA graph warmup");pool.warming=true;
}
void Context::end_graph_warmup(hipStream_t stream) noexcept {
    std::lock_guard lock(graph_pool_mutex_);const auto pool=graph_pools_.find(stream);
    if(pool!=graph_pools_.end())pool->second.warming=false;
}

void Context::end_graph_capture(hipStream_t stream) noexcept {
    std::lock_guard lock(graph_pool_mutex_);
    if (const auto pool = graph_pools_.find(stream);
        pool != graph_pools_.end()) {
        pool->second.capturing = false;
        pool->second.warming = false;
    }
}

void Context::end_graph_pool(hipStream_t stream) noexcept {
    GraphPool pool;
    {
        std::lock_guard lock(graph_pool_mutex_);
        const auto found = graph_pools_.find(stream);
        if (found == graph_pools_.end()) {
            return;
        }
        pool = std::move(found->second);
        graph_pools_.erase(found);
    }
    on_device_noexcept(device_, [&] {
        for (const auto& [bytes, pointers] : pool.available) {
            for (void* pointer : pointers) {
                if (async_allocations_) {
                    (void)hipFreeAsync(pointer, stream);
                } else {
                    (void)hipFree(pointer);
                }
                allocation_budget_->release(bytes);
            }
        }
        if (async_allocations_) {
            (void)hipStreamSynchronize(stream);
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
        MFQ_NATIVE_CUDA_CHECK(hipMemPoolTrimTo(pool_, 0));
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
    const auto flags = mapped ? hipHostMallocMapped : hipHostMallocDefault;
    MFQ_NATIVE_CUDA_CHECK(hipHostAlloc(&data_, bytes_, flags));
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
        (void)hipHostFree(data_);
    }
    data_ = nullptr;
    bytes_ = 0;
}

}  // namespace mfq::cuda
