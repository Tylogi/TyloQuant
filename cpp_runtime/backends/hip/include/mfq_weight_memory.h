#pragma once
#include <cstddef>
#include <memory>
#include <vector>

namespace mfq::cuda {
struct TensorStorage;

// Weak references count live weight allocations without retaining replaced
// projections, allocator pools, prefill workspaces or a whole PLE table.
class WeightMemoryLedger {
    struct Payload { std::weak_ptr<void> owner; std::size_t bytes; };
    std::vector<std::weak_ptr<TensorStorage>> weights_;
    std::vector<Payload> ple_;
public:
    void observe(const std::shared_ptr<TensorStorage>& storage);
    void observe_ple(const std::shared_ptr<void>& owner,std::size_t bytes);
    std::size_t resident_bytes(int device = -1) const;
    std::size_t ple_bytes() const;
};

class WeightMemoryLoadScope {
    WeightMemoryLedger* previous_;
public:
    explicit WeightMemoryLoadScope(WeightMemoryLedger& ledger);
    ~WeightMemoryLoadScope();
    WeightMemoryLoadScope(const WeightMemoryLoadScope&)=delete;
    WeightMemoryLoadScope& operator=(const WeightMemoryLoadScope&)=delete;
};
class WeightAllocationScope {
    bool previous_;
public:
    WeightAllocationScope();
    ~WeightAllocationScope();
    WeightAllocationScope(const WeightAllocationScope&)=delete;
    WeightAllocationScope& operator=(const WeightAllocationScope&)=delete;
};
void observe_weight_allocation(const std::shared_ptr<TensorStorage>& storage);
void observe_streamed_ple(const std::shared_ptr<void>& owner,std::size_t bytes);
}
