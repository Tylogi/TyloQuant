#pragma once

#include <mlx/array.h>
#include <mlx/backend/metal/device.h>
#include <mlx/ops.h>
#include <mlx/stream.h>
#include <mlx/transforms.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

namespace mfq::metal {

class MlxWeightResidency {
    template <typename Data>
    static auto allocation_buffer(Data& data) {
        if constexpr (requires { data.buffer(); }) return data.buffer();
        else return data.buffer;
    }

    struct Capture {
        std::mutex mutex;
        std::vector<std::weak_ptr<mlx::core::array::Data>> allocations;
        std::size_t pending_bytes = 0;
    };
    struct Allocation {
        std::weak_ptr<mlx::core::array::Data> owner;
        NS::SharedPtr<MTL::Buffer> buffer;
        std::size_t set = 0;
        std::size_t bytes = 0;
    };

public:
    static void begin_load() {
        std::atomic_store_explicit(&capture(), std::make_shared<Capture>(), std::memory_order_release);
    }

    static void track(const mlx::core::array& value) {
        const auto current = std::atomic_load_explicit(&capture(), std::memory_order_acquire);
        if (!current || !value.data_shared_ptr()) return;
        std::unique_lock lock(current->mutex);
        current->allocations.push_back(value.data_shared_ptr());
        current->pending_bytes += value.buffer_size();
        if (current->pending_bytes < (std::size_t{4} << 30)) return;
        current->pending_bytes = 0;
        lock.unlock();
        activate();
    }

    static std::unique_ptr<MlxWeightResidency> finish_load(std::size_t limit) {
        const auto current = std::atomic_exchange_explicit(
            &capture(), std::shared_ptr<Capture>{}, std::memory_order_acq_rel);
        if (!current || limit == 0 || !supported()) return {};
        std::vector<std::shared_ptr<mlx::core::array::Data>> live;
        std::unordered_set<const void*> seen;
        {
            std::lock_guard lock(current->mutex);
            for (const auto& weak : current->allocations) {
                if (auto data = weak.lock(); data && allocation_buffer(*data).ptr()
                    && seen.insert(allocation_buffer(*data).ptr()).second) live.push_back(std::move(data));
            }
        }
        if (live.empty()) return {};
        if (current->pending_bytes != 0) activate();
        return std::unique_ptr<MlxWeightResidency>(new MlxWeightResidency(live, limit));
    }

    static std::unique_ptr<MlxWeightResidency> retain_arrays(
        const std::vector<mlx::core::array>& values, std::size_t limit) {
        if (limit == 0 || !supported()) return {};
        std::vector<std::shared_ptr<mlx::core::array::Data>> live;
        std::unordered_set<const void*> seen;
        for (const auto& value : values) {
            auto data = value.data_shared_ptr();
            if (data && allocation_buffer(*data).ptr()
                && seen.insert(allocation_buffer(*data).ptr()).second) live.push_back(std::move(data));
        }
        if (live.empty()) return {};
        return std::unique_ptr<MlxWeightResidency>(new MlxWeightResidency(live, limit));
    }

    ~MlxWeightResidency() {
        {
            std::lock_guard lock(mutex_);
            stopped_ = true;
        }
        wake_.notify_one();
        if (heartbeat_.joinable()) heartbeat_.join();
        for (auto& set : sets_) set->endResidency();
    }

    MlxWeightResidency(const MlxWeightResidency&) = delete;
    MlxWeightResidency& operator=(const MlxWeightResidency&) = delete;

    std::size_t bytes() const noexcept { return bytes_.load(std::memory_order_relaxed); }

private:
    static void activate() {
        const auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
        const mlx::core::array input({0, 1}, mlx::core::int32);
        mlx::core::eval(mlx::core::add(input, input, stream));
        mlx::core::synchronize(stream);
    }

    explicit MlxWeightResidency(
        const std::vector<std::shared_ptr<mlx::core::array::Data>>& live,
        std::size_t limit) {
        constexpr std::size_t set_limit = 8ULL << 30;
        auto& device = mlx::core::metal::device(mlx::core::Device::gpu);
        std::size_t set_bytes = 0;
        for (const auto& data : live) {
            auto* buffer = static_cast<MTL::Buffer*>(allocation_buffer(*data).ptr());
            const auto bytes = buffer->allocatedSize();
            if (bytes > limit - bytes_.load(std::memory_order_relaxed)) {
                throw std::runtime_error("Metal weight residency exceeds the configured budget");
            }
            if (sets_.empty() || (set_bytes != 0 && bytes > set_limit - std::min(set_bytes, set_limit))) {
                if (!sets_.empty()) sets_.back()->commit();
                auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
                descriptor->setInitialCapacity(live.size());
                NS::Error* error = nullptr;
                auto set = NS::TransferPtr(device.mtl_device()->newResidencySet(descriptor.get(), &error));
                if (!set) throw std::runtime_error(error
                    ? error->localizedDescription()->utf8String() : "Metal weight residency set unavailable");
                sets_.push_back(std::move(set));
                set_bytes = 0;
            }
            const MTL::Allocation* allocation = buffer;
            sets_.back()->addAllocations(&allocation, 1);
            allocations_.push_back({data, NS::RetainPtr(buffer), sets_.size() - 1, bytes});
            bytes_.fetch_add(bytes, std::memory_order_relaxed);
            set_bytes += bytes;
        }
        sets_.back()->commit();
        for (auto& set : sets_) set->requestResidency();
        heartbeat_ = std::thread([this] {
            std::unique_lock lock(mutex_);
            while (!wake_.wait_for(lock, std::chrono::milliseconds(500), [this] { return stopped_; })) {
                std::vector<bool> changed(sets_.size(), false);
                allocations_.erase(std::remove_if(allocations_.begin(), allocations_.end(), [&](const auto& allocation) {
                    if (!allocation.owner.expired()) return false;
                    sets_[allocation.set]->removeAllocation(allocation.buffer.get());
                    changed[allocation.set] = true;
                    bytes_.fetch_sub(allocation.bytes, std::memory_order_relaxed);
                    return true;
                }), allocations_.end());
                for (std::size_t index = 0; index < sets_.size(); ++index) {
                    if (changed[index]) sets_[index]->commit();
                }
                for (auto& set : sets_) set->requestResidency();
            }
        });
    }

    static bool supported() {
        if (__builtin_available(macOS 15, *)) return true;
        return false;
    }

    static std::shared_ptr<Capture>& capture() {
        static std::shared_ptr<Capture> value;
        return value;
    }

    std::vector<NS::SharedPtr<MTL::ResidencySet>> sets_;
    std::vector<Allocation> allocations_;
    std::thread heartbeat_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopped_ = false;
    std::atomic<std::size_t> bytes_{0};
};

}
