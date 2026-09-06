#include "mfq_cuda_context.h"

#include <limits>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mfq::cuda {
namespace {

std::mutex default_context_mutex;
std::unordered_map<int, std::weak_ptr<Context>> default_contexts;
thread_local std::unordered_map<int, StreamHandle> active_streams;

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
    if (const auto found = active_streams.find(device);
        found != active_streams.end() && found->second) {
        return found->second;
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
    if (const auto found = active_streams.find(device_); found != active_streams.end()) {
        previous_ = found->second;
    }
    active_streams[device_] = std::move(stream);
}

StreamGuard::~StreamGuard() noexcept {
    if (previous_.has_value()) {
        active_streams[device_] = std::move(*previous_);
    } else {
        active_streams.erase(device_);
    }
}

Graph::~Graph() noexcept {
    reset();
}

Graph::Graph(Graph&& other) noexcept
    : device_(other.device_),
      stream_(std::move(other.stream_)),
      context_(std::move(other.context_)),
      graph_(std::exchange(other.graph_, nullptr)),
      executable_(std::exchange(other.executable_, nullptr)) {}

Graph& Graph::operator=(Graph&& other) noexcept {
    if (this != &other) {
        reset();
        device_ = other.device_;
        stream_ = std::move(other.stream_);
        context_ = std::move(other.context_);
        graph_ = std::exchange(other.graph_, nullptr);
        executable_ = std::exchange(other.executable_, nullptr);
    }
    return *this;
}

void Graph::prepare_memory() {
    reset();
    stream_ = current_stream();
    device_ = stream_.device_index();
    context_ = default_context(device_);
    context_->begin_graph_pool(stream_.stream());
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
    context_->begin_graph_capture(stream_.stream());
    MFQ_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(
        stream_.stream(), cudaStreamCaptureModeGlobal));
}

void Graph::capture_end() {
    if (!stream_) {
        throw Error("CUDA graph capture_end called without capture_begin");
    }
    DeviceGuard guard(device_);
    const auto capture_status =
        cudaStreamEndCapture(stream_.stream(), &graph_);
    context_->end_graph_capture(stream_.stream());
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
        std::size_t kernels = 0;
        std::size_t copies = 0;
        for (auto node : nodes) {
            cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
            MFQ_NATIVE_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
            if (type == cudaGraphNodeTypeKernel) {
                ++kernels;
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
    if (const char* upload = std::getenv("MFQ_NATIVE_CUDA_GRAPH_UPLOAD");
        upload != nullptr && std::atoi(upload) != 0) {
        MFQ_NATIVE_CUDA_CHECK(cudaGraphUpload(executable_, stream_.stream()));
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));
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
        if (context_ && stream_) {
            context_->end_graph_capture(stream_.stream());
            context_->end_graph_pool(stream_.stream());
        }
    });
    context_.reset();
    stream_ = {};
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
    int pools_supported = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(
        &pools_supported, cudaDevAttrMemoryPoolsSupported, device_));
    async_allocations_ = pools_supported != 0;
    if (async_allocations_) {
        MFQ_NATIVE_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool_, device_));
        std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
        MFQ_NATIVE_CUDA_CHECK(cudaMemPoolSetAttribute(
            pool_, cudaMemPoolAttrReleaseThreshold, &threshold));
    }
    blas_.set_stream(stream_.get());
}

void* Context::allocate(std::size_t bytes, cudaStream_t stream) {
    if (bytes == 0) {
        return nullptr;
    }
    DeviceGuard guard(device_);
    const auto allocation_stream =
        stream != nullptr ? stream : stream_.get();
    bool capture_pool_miss = false;
    {
        std::lock_guard lock(graph_pool_mutex_);
        if (auto pool = graph_pools_.find(allocation_stream);
            pool != graph_pools_.end()) {
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
    if (async_allocations_) {
        MFQ_NATIVE_CUDA_CHECK(cudaMallocAsync(
            &pointer, bytes, allocation_stream));
    } else {
        MFQ_NATIVE_CUDA_CHECK(cudaMalloc(&pointer, bytes));
    }
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
        if (auto pool = graph_pools_.find(allocation_stream);
            pool != graph_pools_.end()) {
            pool->second.available[bytes].push_back(pointer);
            return;
        }
    } catch (...) {
        // Tensor destruction is noexcept. A warm-up bookkeeping allocation
        // failure must release the CUDA allocation instead of terminating or
        // leaking it. During capture the size bucket already exists and the
        // preceding pop leaves enough vector capacity for this push.
    }
    on_device_noexcept(device_, [&] {
        if (async_allocations_) {
            (void)cudaFreeAsync(pointer, allocation_stream);
        } else {
            (void)cudaFree(pointer);
        }
    });
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

void Context::end_graph_capture(cudaStream_t stream) noexcept {
    std::lock_guard lock(graph_pool_mutex_);
    if (const auto pool = graph_pools_.find(stream);
        pool != graph_pools_.end()) {
        pool->second.capturing = false;
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
        pool = std::move(found->second);
        graph_pools_.erase(found);
    }
    on_device_noexcept(device_, [&] {
        for (const auto& [_, pointers] : pool.available) {
            for (void* pointer : pointers) {
                if (async_allocations_) {
                    (void)cudaFreeAsync(pointer, stream);
                } else {
                    (void)cudaFree(pointer);
                }
            }
        }
        if (async_allocations_) {
            (void)cudaStreamSynchronize(stream);
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
        MFQ_NATIVE_CUDA_CHECK(cudaMemPoolTrimTo(pool_, 0));
    }
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
    MFQ_NATIVE_CUDA_CHECK(cudaHostAlloc(&data_, bytes_, flags));
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
