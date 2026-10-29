#include "mfq/host_parallel.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mfq {
namespace {

thread_local bool inside_host_work = false;
using ProfileClock=std::chrono::steady_clock;
std::uint64_t elapsed_ns(ProfileClock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        ProfileClock::now()-start).count());
}

class WorkerScope {
public:
    WorkerScope() : previous_(inside_host_work) { inside_host_work = true; }
    ~WorkerScope() { inside_host_work = previous_; }
private:
    bool previous_;
};

} // namespace

struct HostParallelPool::Impl {
public:
    explicit Impl(std::function<void(std::size_t)> initialize,bool wait_idle)
        : initialize_(std::move(initialize)),wait_idle_workers_(wait_idle) {}
    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    void run(std::int64_t begin, std::int64_t end, std::int64_t grain,
             int threads, const std::function<void(std::int64_t, std::int64_t)>& function,
             HostParallelProfile* profile, std::int64_t ready_end = 0,
             const std::function<void()>* ready = nullptr) {
        // Own the callback until every participant has acknowledged completion.
        const auto started=profile ? ProfileClock::now() : ProfileClock::time_point{};
        std::unique_lock<std::mutex> dispatch(dispatch_mutex_);
        const auto setup_started=profile ? ProfileClock::now() : ProfileClock::time_point{};
        const int participants = static_cast<int>(std::min<std::int64_t>(
            threads, 1 + (end - begin - 1) / grain));
        // More row ranges let active CPU expert workers share the tail when
        // another worker arrives late. Pools waiting for every worker retain
        // their original balanced partitioning.
        const auto partitions = static_cast<std::int64_t>(participants) *
            (wait_idle_workers_ ? 1 : 4);
        const auto balanced = 1 + (end - begin - 1) / partitions;
        const auto chunk = ready ? 1 : std::max(grain, balanced);
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // Each new worker starts before the current epoch is published.
            while (workers_.size() < static_cast<std::size_t>(participants - 1)) {
                const auto epoch = epoch_;
                const auto index = workers_.size();
                workers_.emplace_back([this, index, epoch] { worker_loop(index, epoch); });
            }
            if(!wait_idle_workers_) {
                initialized_.wait(lock,[&]{return initialized_workers_==workers_.size();});
                if(initialization_failure_)std::rethrow_exception(initialization_failure_);
            }
            function_ = &function;
            next_.store(begin, std::memory_order_relaxed);
            end_ = end;
            chunk_ = chunk;
            stage_end_ = ready_end;
            stage_remaining_.store(ready ? ready_end - begin : 0, std::memory_order_relaxed);
            staged_ = ready != nullptr;
            active_workers_ = participants - 1;
            remaining_ = active_workers_;
            failure_ = nullptr;
            cancelled_.store(false, std::memory_order_relaxed);
            profiling_=profile!=nullptr;
            if(profile) {
                work_ns_=0; callbacks_=0; rows_=0;
                wake_sum_ns_=0; wake_max_ns_=0; worker_arrivals_=0;
                published_=ProfileClock::now();
            }
            ++epoch_;
        }
        const auto setup_ns=profile ? elapsed_ns(setup_started) : 0;
        ready_.notify_all();
        auto caller_work=drain(ready != nullptr); // The caller contributes compute.
        if (ready) {
            std::unique_lock<std::mutex> lock(mutex_);
            done_.wait(lock, [&] { return stage_remaining_.load(std::memory_order_acquire) == 0 || failure_; });
            if (!failure_) {
                lock.unlock();
                try {
                    WorkerScope scope;
                    (*ready)();
                } catch (...) {
                    cancelled_.store(true, std::memory_order_relaxed);
                    lock.lock();
                    if (!failure_) failure_ = std::current_exception();
                    lock.unlock();
                }
            } else lock.unlock();
            caller_work += drain();
        }
        const auto wait_started=profile ? ProfileClock::now() : ProfileClock::time_point{};
        std::unique_lock<std::mutex> lock(mutex_);
        if(!wait_idle_workers_) {
            // Unclaimed chunks have been drained or cancelled. Retire workers
            // still asleep; only callbacks already running own the function.
            active_workers_=0;
            remaining_=running_workers_;
        }
        done_.wait(lock, [&] { return remaining_ == 0; });
        function_ = nullptr;
        if(profile) {
            *profile={};
            profile->wall_ns=elapsed_ns(started);
            profile->dispatch_wait_ns=static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(setup_started-started).count());
            profile->setup_ns=setup_ns;
            profile->work_ns=work_ns_.load(); profile->caller_work_ns=caller_work;
            profile->caller_wait_ns=elapsed_ns(wait_started);
            profile->worker_wake_sum_ns=wake_sum_ns_.load();
            profile->worker_wake_max_ns=wake_max_ns_.load();
            profile->callbacks=callbacks_.load(); profile->rows=rows_.load();
            profile->worker_arrivals=worker_arrivals_.load();
            profile->participants=participants; profile->failed=bool(failure_);
        }
        if (failure_) std::rethrow_exception(failure_);
    }

private:
    std::uint64_t drain(bool prefix_only = false) {
        WorkerScope scope;
        std::uint64_t work=0, callbacks=0, rows=0;
        try {
            while (!cancelled_.load(std::memory_order_relaxed)) {
                std::int64_t begin;
                if (prefix_only) {
                    begin = next_.load(std::memory_order_relaxed);
                    while (begin < stage_end_ && !next_.compare_exchange_weak(
                            begin, begin + 1, std::memory_order_relaxed)) {}
                    if (begin >= stage_end_) break;
                } else begin = next_.fetch_add(chunk_, std::memory_order_relaxed);
                if (begin >= end_) break;
                const auto end=begin+std::min(chunk_,end_-begin);
                if(profiling_) {
                    const auto started=ProfileClock::now();
                    ++callbacks;
                    try { (*function_)(begin,end); }
                    catch(...) { work+=elapsed_ns(started); throw; }
                    work+=elapsed_ns(started); rows+=end-begin;
                } else (*function_)(begin,end);
                if (staged_ && begin < stage_end_ && stage_remaining_.fetch_sub(
                        end - begin, std::memory_order_acq_rel) == end - begin) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    done_.notify_all();
                }
            }
        } catch (...) {
            cancelled_.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(mutex_);
            if (!failure_) failure_ = std::current_exception();
            done_.notify_all();
        }
        if(profiling_) {
            work_ns_.fetch_add(work,std::memory_order_relaxed);
            callbacks_.fetch_add(callbacks,std::memory_order_relaxed);
            rows_.fetch_add(rows,std::memory_order_relaxed);
        }
        return work;
    }

    void worker_loop(std::size_t index, std::uint64_t observed) {
        std::exception_ptr initialization_error;
        try { if (initialize_) initialize_(index); }
        catch (...) { initialization_error = std::current_exception(); }
        std::unique_lock<std::mutex> lock(mutex_);
        if(initialization_error && !initialization_failure_)initialization_failure_=initialization_error;
        ++initialized_workers_;initialized_.notify_one();
        for (;;) {
            ready_.wait(lock, [&] { return stopping_ || observed != epoch_; });
            if (stopping_) return;
            observed = epoch_;
            if (index >= static_cast<std::size_t>(active_workers_)) continue;
            if (initialization_error) {
                if (!failure_) failure_ = initialization_error;
                cancelled_.store(true, std::memory_order_relaxed);
            } else {
                ++running_workers_;
                lock.unlock();
                if(profiling_) {
                    const auto wake=elapsed_ns(published_);
                    wake_sum_ns_.fetch_add(wake,std::memory_order_relaxed);
                    worker_arrivals_.fetch_add(1,std::memory_order_relaxed);
                    auto maximum=wake_max_ns_.load(std::memory_order_relaxed);
                    while(maximum<wake && !wake_max_ns_.compare_exchange_weak(
                            maximum,wake,std::memory_order_relaxed)) {}
                }
                drain();
                lock.lock();
                --running_workers_;
            }
            if (--remaining_ == 0) done_.notify_one();
        }
    }

    std::mutex dispatch_mutex_, mutex_;
    std::condition_variable ready_, done_,initialized_;
    std::vector<std::thread> workers_;
    const std::function<void(std::int64_t, std::int64_t)>* function_ = nullptr;
    std::atomic<std::int64_t> next_{0};
    std::atomic<bool> cancelled_{false};
    std::int64_t end_ = 0, chunk_ = 1;
    std::int64_t stage_end_ = 0;
    std::atomic<std::int64_t> stage_remaining_{0};
    bool staged_ = false;
    std::uint64_t epoch_ = 0;
    int active_workers_ = 0, remaining_ = 0;
    int running_workers_=0;
    std::size_t initialized_workers_=0;
    std::exception_ptr initialization_failure_;
    bool stopping_ = false;
    bool profiling_=false;
    ProfileClock::time_point published_;
    std::atomic<std::uint64_t> work_ns_{0},callbacks_{0},rows_{0};
    std::atomic<std::uint64_t> wake_sum_ns_{0},wake_max_ns_{0},worker_arrivals_{0};
    std::exception_ptr failure_;
    std::function<void(std::size_t)> initialize_;
    bool wait_idle_workers_=true;
};

HostParallelPool::HostParallelPool(std::function<void(std::size_t)> initialize,bool wait_idle)
    : impl_(std::make_unique<Impl>(std::move(initialize),wait_idle)) {}
HostParallelPool::~HostParallelPool() = default;

void HostParallelPool::run_staged(std::int64_t end, std::int64_t ready_end, int threads,
    const std::function<void(std::int64_t, std::int64_t)>& function,
    const std::function<void()>& ready) {
    if (ready_end < 0 || end < ready_end) throw std::invalid_argument("invalid staged host work range");
    if (threads <= 1 || inside_host_work || end <= 1) {
        WorkerScope scope;
        if (ready_end) function(0, ready_end);
        ready();
        if (ready_end < end) function(ready_end, end);
    } else impl_->run(0, end, 1, threads, function, nullptr, ready_end, &ready);
}

void HostParallelPool::run(std::int64_t begin, std::int64_t end,
    std::int64_t grain, int threads,
    const std::function<void(std::int64_t, std::int64_t)>& function,
    HostParallelProfile* profile) {
    if(profile)*profile={};
    if (end <= begin) return;
    grain = std::max<std::int64_t>(grain, 1);
    if (threads <= 1 || inside_host_work || end - begin <= grain) {
        WorkerScope scope;
        if(!profile) { function(begin,end); return; }
        const auto started=ProfileClock::now();
        profile->participants=1;profile->callbacks=1;
        try { function(begin,end); profile->rows=end-begin; }
        catch(...) {
            profile->wall_ns=profile->work_ns=profile->caller_work_ns=elapsed_ns(started);
            profile->failed=true;throw;
        }
        profile->wall_ns=profile->work_ns=profile->caller_work_ns=elapsed_ns(started);
    } else impl_->run(begin, end, grain, threads, function, profile);
}

void host_parallel_for(std::int64_t begin, std::int64_t end,
    std::int64_t grain, int threads,
    const std::function<void(std::int64_t, std::int64_t)>& function) {
    if (end <= begin) return;
    grain = std::max<std::int64_t>(grain, 1);
    if (threads <= 1 || inside_host_work || end - begin <= grain) {
        WorkerScope scope;
        function(begin, end);
        return;
    }
    static HostParallelPool pool;
    pool.run(begin, end, grain, threads, function);
}

} // namespace mfq
