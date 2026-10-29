#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <cuda_runtime_api.h>

namespace mfq::cuda {
// Host registration changes may synchronize the device. Keep them outside
// graph capture and replays that wait for CPU expert-loading callbacks.
class HostRegistrationBlock {
    std::shared_ptr<void> state_;
public:
    HostRegistrationBlock();
};
std::unique_lock<std::mutex> try_host_registration();
struct HostRegistrationRetirement {
    void* owner=nullptr;
    void (*dispose)(void*) noexcept=nullptr;
    HostRegistrationRetirement* next=nullptr;
};
// Intrusive retirement allocates nothing when optional cached RAM is evicted.
void retire_host_registration(HostRegistrationRetirement*)noexcept;

class CapturedGraph {
public:
    ~CapturedGraph();
    CapturedGraph(CapturedGraph&&) noexcept;
    CapturedGraph& operator=(CapturedGraph&&) noexcept;
    CapturedGraph(const CapturedGraph&) = delete;
    CapturedGraph& operator=(const CapturedGraph&) = delete;
    bool launch(void* stream, std::string& error) const;
    bool wait_ms(int timeout_ms) const;
    std::size_t nodes() const;
    bool valid() const;
    void set_kernel_rows(cudaGraphNode_t node,unsigned rows)const;
private:
    friend class GraphRegistry;
    CapturedGraph(void* stream, const std::function<void()>& body);
    explicit CapturedGraph(void* stream);
    void finish();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class GraphRegistry {
public:
    explicit GraphRegistry(void* stream) : stream_(stream) {}
    bool record(std::uint64_t identity, int tokens,
                const std::function<void()>& body, std::string& error);
    const CapturedGraph* find(std::uint64_t identity, int tokens) const;
    bool launch(std::uint64_t identity, int tokens, int timeout_ms, std::string& error) const;
    std::size_t size() const { return graphs_.size(); }
    std::size_t captures() const { return captures_; }
    void clear() { graphs_.clear(); }
    void begin(std::uint64_t identity, int tokens);
    void end(std::uint64_t identity, int tokens);
private:
    void* stream_;
    std::map<std::pair<std::uint64_t, int>, CapturedGraph> graphs_;
    std::size_t captures_ = 0;
};
}
