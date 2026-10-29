#pragma once

#include <cstdint>
#include <functional>
#include <memory>

namespace mfq {

// Optional observations of one synchronous dispatch. Work sums span parallel
// participants; wake delays overlap compute and must not be added to wall time.
struct HostParallelProfile {
    std::uint64_t wall_ns=0, dispatch_wait_ns=0, setup_ns=0;
    std::uint64_t work_ns=0, caller_work_ns=0, caller_wait_ns=0;
    std::uint64_t worker_wake_sum_ns=0, worker_wake_max_ns=0;
    std::uint64_t callbacks=0, rows=0, worker_arrivals=0;
    int participants=0;
    bool failed=false;
};

// Independent worker groups let file reads overlap CPU expert computation.
class HostParallelPool {
public:
    explicit HostParallelPool(std::function<void(std::size_t)> initialize = {},bool wait_idle_workers=true);
    ~HostParallelPool();
    HostParallelPool(const HostParallelPool&) = delete;
    HostParallelPool& operator=(const HostParallelPool&) = delete;
    void run(std::int64_t begin, std::int64_t end, std::int64_t grain,
             int threads,
             const std::function<void(std::int64_t, std::int64_t)>& function,
             HostParallelProfile* profile=nullptr);
    void run_staged(std::int64_t end, std::int64_t ready_end, int threads,
             const std::function<void(std::int64_t, std::int64_t)>& function,
             const std::function<void()>& ready);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Persistent host workers. Concurrent callers are serialized; nested work
// executes on its calling worker so it cannot wait for the same pool.
void host_parallel_for(std::int64_t begin, std::int64_t end,
    std::int64_t grain, int threads,
    const std::function<void(std::int64_t, std::int64_t)>& function);

} // namespace mfq
