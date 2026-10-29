#pragma once

#include <mlx/array.h>
#include <mlx/backend/metal/device.h>
#include <mlx/ops.h>
#include <mlx/stream.h>
#include <mlx/transforms.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <unordered_map>
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
        auto residency = std::unique_ptr<MlxWeightResidency>(new MlxWeightResidency(live, limit));
        activate();
        return residency;
    }

    static std::unique_ptr<MlxWeightResidency> retain_arrays(
        const std::vector<mlx::core::array>& values, std::size_t limit, bool indirect = false) {
        if (limit == 0 || !supported()) return {};
        std::vector<std::shared_ptr<mlx::core::array::Data>> live;
        std::unordered_set<const void*> seen;
        for (const auto& value : values) {
            auto data = value.data_shared_ptr();
            if (data && allocation_buffer(*data).ptr()
                && seen.insert(allocation_buffer(*data).ptr()).second) live.push_back(std::move(data));
        }
        if (live.empty()) return {};
        return std::unique_ptr<MlxWeightResidency>(new MlxWeightResidency(live, limit, indirect));
    }

    void update_arrays(const std::vector<mlx::core::array>& values,std::size_t limit) {
        std::vector<std::shared_ptr<mlx::core::array::Data>> live;
        std::unordered_set<const void*> wanted;
        std::size_t bytes=0;
        for(const auto& value:values) {
            auto data=value.data_shared_ptr();
            if(!data||!allocation_buffer(*data).ptr()||!wanted.insert(allocation_buffer(*data).ptr()).second)continue;
            const auto count=static_cast<MTL::Buffer*>(allocation_buffer(*data).ptr())->allocatedSize();
            if(bytes>limit||count>limit-bytes)throw std::runtime_error("Metal weight residency exceeds the configured budget");
            bytes+=count;live.push_back(std::move(data));
        }
        std::lock_guard lock(mutex_);
        const std::size_t set_limit=indirect_ ? std::max<std::size_t>(8ULL<<30,limit/16+(limit%16!=0)) : 8ULL<<30;
        std::vector<std::size_t> occupied(sets_.size(),0);
        std::unordered_map<const void*,std::size_t> existing;
        for(const auto& allocation:allocations_)if(wanted.contains(allocation.buffer.get())) {
            occupied[allocation.set]+=allocation.bytes;
            existing.emplace(allocation.buffer.get(),allocation.set);
        }
        std::vector<Allocation> next;next.reserve(live.size());
        for(const auto& data:live) {
            auto* buffer=static_cast<MTL::Buffer*>(allocation_buffer(*data).ptr());
            const auto count=buffer->allocatedSize();
            std::size_t slot;
            if(const auto found=existing.find(buffer);found!=existing.end())slot=found->second;
            else {
                slot=0;
                while(slot<occupied.size()&&occupied[slot]&&(count>set_limit-std::min(occupied[slot],set_limit)))++slot;
                if(slot==occupied.size()) {
                    if(occupied.size()<32)occupied.push_back(0);
                    else slot=std::min_element(occupied.begin(),occupied.end())-occupied.begin();
                }
                occupied[slot]+=count;
            }
            next.push_back({data,NS::RetainPtr(buffer),slot,count});
        }
        std::vector<NS::SharedPtr<MTL::ResidencySet>> additions;
        const auto existing_sets = sets_.size();
        while(sets_.size()+additions.size()<occupied.size()) {
            auto descriptor=NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
            descriptor->setInitialCapacity(live.size());
            NS::Error* error=nullptr;
            auto set=NS::TransferPtr(mlx::core::metal::device(mlx::core::Device::gpu).mtl_device()->newResidencySet(descriptor.get(),&error));
            if(!set)throw std::runtime_error(error?error->localizedDescription()->utf8String():"Metal weight residency set unavailable");
            additions.push_back(std::move(set));
        }
        sets_.reserve(occupied.size());
        std::vector<bool> changed(occupied.size(),false);
        for(auto& set:additions)sets_.push_back(std::move(set));
        for(const auto& allocation:allocations_)if(!wanted.contains(allocation.buffer.get())) {
            sets_[allocation.set]->removeAllocation(allocation.buffer.get());changed[allocation.set]=true;
        }
        for(const auto& allocation:next)if(!existing.contains(allocation.buffer.get())) {
            const MTL::Allocation* item=allocation.buffer.get();
            sets_[allocation.set]->addAllocations(&item,1);changed[allocation.set]=true;
        }
        for(std::size_t slot=0;slot<sets_.size();++slot)if(changed[slot]) {
            sets_[slot]->commit();
            if (slot >= existing_sets) sets_[slot]->requestResidency();
        }
        allocations_=std::move(next);
        if(indirect_)indirect_owners_=std::move(live);
        bytes_.store(bytes,std::memory_order_relaxed);
    }

    void use_command_buffer(MTL::CommandBuffer* buffer) {
        std::lock_guard lock(mutex_);
        if(!indirect_)throw std::logic_error("weight residency is not configured for indirect access");
        if(sets_.size()>32)throw std::runtime_error("TQ indirect residency exceeds command-buffer capacity");
        std::array<const MTL::ResidencySet*,32> sets;
        for(std::size_t i=0;i<sets_.size();++i)sets[i]=sets_[i].get();
        buffer->useResidencySets(sets.data(),sets_.size());
    }
    ~MlxWeightResidency() {
        {
            std::lock_guard lock(mutex_);
            stopped_ = true;
        }
        wake_.notify_one();
        if (heartbeat_.joinable()) heartbeat_.join();
        for (auto& set : sets_) {
            set->endResidency();
            set->removeAllAllocations();
            set->commit();
        }
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
        std::size_t limit, bool indirect = false) : indirect_(indirect) {
        if(indirect_)indirect_owners_=live;
        const std::size_t set_limit = indirect_ ? std::max<std::size_t>(8ULL << 30, limit / 16 + (limit % 16 != 0)) : 8ULL << 30;
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
        if (indirect_) return;
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
    std::vector<std::shared_ptr<mlx::core::array::Data>> indirect_owners_;
    std::vector<Allocation> allocations_;
    bool indirect_ = false;
    std::thread heartbeat_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopped_ = false;
    std::atomic<std::size_t> bytes_{0};
};

}
