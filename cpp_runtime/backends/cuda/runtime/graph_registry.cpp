#include "graph_registry.h"
#include "mfq_cuda_context.h"
#include <chrono>
#include <thread>
#include <unordered_map>
#include <condition_variable>

namespace mfq::cuda {
namespace {
struct HostRegistrationSchedule {
    std::mutex mutex;
    std::condition_variable changed;
    unsigned active=0;
    HostRegistrationRetirement* retired=nullptr;
    void drain()noexcept {
        while(retired) {
            auto* item=retired;retired=item->next;
            item->dispose(item->owner);
        }
    }
};
HostRegistrationSchedule& host_registration_schedule() {
    static HostRegistrationSchedule state;return state;
}
struct HostRegistrationState {
    HostRegistrationState() {
        auto& state=host_registration_schedule();std::unique_lock lock(state.mutex);
        // A pending release gets a finite gap between overlapping windows;
        // continuously arriving requests cannot retain evicted pinned RAM.
        state.changed.wait(lock,[&]{return !state.retired || !state.active;});
        if(!state.active)state.drain();
        ++state.active;
    }
    ~HostRegistrationState()noexcept {
        auto& state=host_registration_schedule();std::lock_guard lock(state.mutex);
        if(!--state.active){state.drain();state.changed.notify_all();}
    }
};
}
HostRegistrationBlock::HostRegistrationBlock() {
    // Nested capture segments share a token. The shared owner also permits
    // cleanup on a different CPU thread after an aborted capture.
    static thread_local std::weak_ptr<void> current;
    state_=current.lock();
    if(!state_){state_=std::make_shared<HostRegistrationState>();current=state_;}
}
std::unique_lock<std::mutex> try_host_registration() {
    auto& state=host_registration_schedule();std::unique_lock lock(state.mutex,std::try_to_lock);
    if(!lock.owns_lock() || state.active)return {};
    state.drain();return lock;
}
void retire_host_registration(HostRegistrationRetirement* item)noexcept {
    auto& state=host_registration_schedule();std::lock_guard lock(state.mutex);
    item->next=state.retired;state.retired=item;
    if(!state.active)state.drain();
}
struct CapturedGraph::Impl {
    std::unique_ptr<HostRegistrationBlock> registration;
    Graph graph;
    Event completion;
    StreamHandle stream;
    struct KernelLaunch {cudaKernelNodeParams params{};bool enabled=true;};
    std::unordered_map<cudaGraphNode_t,KernelLaunch> kernels;
    explicit Impl(void* native):registration(std::make_unique<HostRegistrationBlock>()) {
        int device = 0;
        MFQ_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
        stream = StreamHandle(device, static_cast<cudaStream_t>(native));
        StreamGuard selected(stream);
        graph.capture_begin_shared();
    }
};
CapturedGraph::CapturedGraph(void* stream, const std::function<void()>& body)
    : CapturedGraph(stream) { body(); finish(); }
CapturedGraph::CapturedGraph(void* stream) : impl_(std::make_unique<Impl>(stream)) {}
void CapturedGraph::finish() {
    StreamGuard selected(impl_->stream);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    impl_->graph.capture_end();
    impl_->registration.reset();
    if (!impl_->graph.nodes()) throw Error("CUDA graph body produced no work");
}
CapturedGraph::~CapturedGraph() = default;
CapturedGraph::CapturedGraph(CapturedGraph&&) noexcept = default;
CapturedGraph& CapturedGraph::operator=(CapturedGraph&&) noexcept = default;
bool CapturedGraph::valid() const { return impl_ && impl_->graph.valid(); }
void CapturedGraph::set_kernel_rows(cudaGraphNode_t node,unsigned rows)const {
#if CUDART_VERSION >= 11060
    if(!valid() || !node || rows>65535)throw std::invalid_argument("invalid graph kernel grid.y update");
    StreamGuard selected(impl_->stream);
    auto [entry,inserted]=impl_->kernels.try_emplace(node);
    auto& state=entry->second;
    if(inserted)MFQ_NATIVE_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&state.params));
    const auto executable=impl_->graph.native_executable();
    if(!rows) {
        if(state.enabled)MFQ_NATIVE_CUDA_CHECK(cudaGraphNodeSetEnabled(executable,node,0));
        state.enabled=false;return;
    }
    if(state.params.gridDim.y!=rows) {
        auto params=state.params;params.gridDim.y=rows;
        MFQ_NATIVE_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(executable,node,&params));
        state.params=params;
    }
    if(!state.enabled)MFQ_NATIVE_CUDA_CHECK(cudaGraphNodeSetEnabled(executable,node,1));
    state.enabled=true;
#else
    (void)node;(void)rows;
    throw Error("graph kernel enable/disable requires CUDA11.6 or newer");
#endif
}
std::size_t CapturedGraph::nodes() const { return impl_ ? impl_->graph.nodes() : 0; }
bool CapturedGraph::launch(void* stream, std::string& error) const {
    try {
        if (!valid() || impl_->stream.stream() != static_cast<cudaStream_t>(stream))
            throw Error("CUDA graph replay has no matching capture stream");
        StreamGuard selected(impl_->stream);
        impl_->graph.replay();
        impl_->completion.record(impl_->stream.stream());
        return true;
    } catch (const std::exception& failure) { error = failure.what(); return false; }
}
bool CapturedGraph::wait_ms(int timeout_ms) const {
    if (!valid() || timeout_ms < 0) return false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    do {
        if (impl_->completion.ready()) return true;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}
bool GraphRegistry::record(std::uint64_t identity, int tokens,
        const std::function<void()>& body, std::string& error) {
    if (find(identity, tokens)) return true;
    try {
        CapturedGraph graph(stream_, body);
        graphs_.emplace(std::make_pair(identity, tokens), std::move(graph));
        ++captures_;
        return true;
    } catch (const std::exception& failure) { error = failure.what(); return false; }
}
void GraphRegistry::begin(std::uint64_t identity,int tokens) {
    if(find(identity,tokens))throw Error("CUDA graph segment already exists");
    graphs_.emplace(std::make_pair(identity,tokens),CapturedGraph(stream_));
}
void GraphRegistry::end(std::uint64_t identity,int tokens) {
    graphs_.at({identity,tokens}).finish();++captures_;
}
const CapturedGraph* GraphRegistry::find(std::uint64_t identity, int tokens) const {
    const auto found = graphs_.find({identity, tokens});
    return found == graphs_.end() ? nullptr : &found->second;
}
bool GraphRegistry::launch(std::uint64_t identity, int tokens, int timeout_ms, std::string& error) const {
    const HostRegistrationBlock registration;
    const auto* graph = find(identity, tokens);
    if (!graph) { error = "CUDA graph was not recorded for this operator and shape"; return false; }
    if (!graph->launch(stream_, error)) return false;
    if (!graph->wait_ms(timeout_ms)) { error = "CUDA graph completion exceeded its deadline"; return false; }
    return true;
}
}
