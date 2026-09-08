#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace mfq::cuda::continuous {

class PagedKvPageAllocator {
public:
    explicit PagedKvPageAllocator(std::size_t maximum_pages)
        : maximum_pages_(maximum_pages) {
        if (maximum_pages_ == 0 ||
                maximum_pages_ > static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument(
                "Paged KV physical page count must fit int32");
        }
    }

    std::vector<std::int32_t> allocate(std::size_t count) {
        if (count > maximum_pages_ - live_pages_) {
            throw std::runtime_error("Paged KV physical page pool exhausted");
        }
        std::vector<std::int32_t> pages;
        pages.reserve(count);
        try {
            while (pages.size() < count && !free_pages_.empty()) {
                const auto page = free_pages_.back();
                free_pages_.pop_back();
                allocated_[static_cast<std::size_t>(page)] = true;
                pages.push_back(page);
                ++reuse_count_;
            }
            while (pages.size() < count) {
                if (high_watermark_ >= maximum_pages_) {
                    throw std::runtime_error(
                        "Paged KV physical page pool exhausted");
                }
                const auto page = static_cast<std::int32_t>(high_watermark_++);
                allocated_.push_back(true);
                pages.push_back(page);
            }
        } catch (...) {
            release_unchecked(pages);
            throw;
        }
        live_pages_ += pages.size();
        allocation_count_ += pages.size();
        peak_live_pages_ = std::max(peak_live_pages_, live_pages_);
        return pages;
    }

    void release(const std::vector<std::int32_t> & pages) {
        std::unordered_set<std::int32_t> unique;
        unique.reserve(pages.size());
        for (const auto page : pages) {
            if (page < 0 ||
                    static_cast<std::size_t>(page) >= allocated_.size() ||
                    !allocated_[static_cast<std::size_t>(page)] ||
                    !unique.insert(page).second) {
                throw std::logic_error(
                    "Paged KV attempted to release an invalid physical page");
            }
        }
        for (const auto page : pages) {
            allocated_[static_cast<std::size_t>(page)] = false;
            free_pages_.push_back(page);
        }
        live_pages_ -= pages.size();
        release_count_ += pages.size();
    }

    std::size_t maximum_pages() const noexcept { return maximum_pages_; }
    std::size_t live_pages() const noexcept { return live_pages_; }
    std::size_t peak_live_pages() const noexcept { return peak_live_pages_; }
    std::size_t high_watermark() const noexcept { return high_watermark_; }
    std::size_t allocation_count() const noexcept { return allocation_count_; }
    std::size_t reuse_count() const noexcept { return reuse_count_; }
    std::size_t release_count() const noexcept { return release_count_; }
    bool owns(std::int32_t page) const noexcept {
        return page >= 0 &&
            static_cast<std::size_t>(page) < allocated_.size() &&
            allocated_[static_cast<std::size_t>(page)];
    }

private:
    void release_unchecked(const std::vector<std::int32_t> & pages) noexcept {
        for (const auto page : pages) {
            const auto index = static_cast<std::size_t>(page);
            if (index < allocated_.size() && allocated_[index]) {
                allocated_[index] = false;
                free_pages_.push_back(page);
            }
        }
    }

    std::size_t maximum_pages_ = 0;
    std::size_t high_watermark_ = 0;
    std::size_t live_pages_ = 0;
    std::size_t peak_live_pages_ = 0;
    std::size_t allocation_count_ = 0;
    std::size_t reuse_count_ = 0;
    std::size_t release_count_ = 0;
    std::vector<bool> allocated_;
    std::vector<std::int32_t> free_pages_;
};

} // namespace mfq::cuda::continuous
