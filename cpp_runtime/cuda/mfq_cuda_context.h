#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::cuda {

class Error final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void check(cudaError_t status, const char* expression, const char* file, int line);
void check(cublasStatus_t status, const char* expression, const char* file, int line);

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

class Graph final {
public:
    Graph() = default;
    ~Graph() noexcept;

    Graph(Graph&& other) noexcept;
    Graph& operator=(Graph&& other) noexcept;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    void capture_begin();
    void capture_end();
    void replay();
    void reset() noexcept;

private:
    int device_ = 0;
    StreamHandle stream_;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
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

class Context final : public std::enable_shared_from_this<Context> {
public:
    explicit Context(int device);

    int device() const noexcept { return device_; }
    Stream& stream() noexcept { return stream_; }
    const Stream& stream() const noexcept { return stream_; }
    BlasHandle& blas() noexcept { return blas_; }
    bool supports_async_allocations() const noexcept { return async_allocations_; }

    void* allocate(std::size_t bytes, cudaStream_t stream = nullptr);
    void release(void* pointer, cudaStream_t stream = nullptr) noexcept;
    void trim();

private:
    int device_ = 0;
    Stream stream_;
    BlasHandle blas_;
    cudaMemPool_t pool_ = nullptr;
    bool async_allocations_ = false;
};

std::shared_ptr<Context> default_context(int device = 0);

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
