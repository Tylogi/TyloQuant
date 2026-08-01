#include "moe_cache_policy.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_lru_replaces_oldest_non_inflight_slot() {
    mfq::MoeCacheSlotBook book(2);
    const mfq::MoeCacheKey first{1, 0, 3};
    const mfq::MoeCacheKey second{1, 0, 4};
    const mfq::MoeCacheKey third{2, 0, 9};

    const auto first_lease = book.acquire(first);
    const auto second_lease = book.acquire(second);
    require(!first_lease.replaced.has_value(), "first lease replaced a key");
    require(!second_lease.replaced.has_value(), "second lease replaced a key");

    book.touch(first_lease.slot);
    const auto third_lease = book.acquire(third);
    require(
        third_lease.replaced == std::optional<mfq::MoeCacheKey>(second),
        "LRU did not replace the oldest key");
    require(book.slot_for(first) == first_lease.slot, "touched key was evicted");
    require(book.slot_for(third) == third_lease.slot, "new key was not installed");
}

void test_inflight_slot_is_not_replaced() {
    mfq::MoeCacheSlotBook book(2);
    const mfq::MoeCacheKey first{0, 0, 0};
    const mfq::MoeCacheKey second{0, 0, 1};
    const mfq::MoeCacheKey third{0, 0, 2};

    const auto first_lease = book.acquire(first);
    const auto second_lease = book.acquire(second);
    book.mark_inflight(first_lease.slot);
    book.touch(second_lease.slot);

    const auto third_lease = book.acquire(third);
    require(
        third_lease.replaced == std::optional<mfq::MoeCacheKey>(second),
        "in-flight slot was selected for replacement");
    require(book.slot_for(first) == first_lease.slot, "in-flight key was evicted");
}

void test_all_inflight_slots_reject_replacement() {
    mfq::MoeCacheSlotBook book(1);
    const auto lease = book.acquire({0, 0, 0});
    book.mark_inflight(lease.slot);
    bool rejected = false;
    try {
        (void)book.acquire({0, 0, 1});
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "all-inflight cache accepted a replacement");
}

void test_budget_planner_honors_minimums_and_hard_limit() {
    const std::vector<mfq::MoeArenaDemand> demands{
        {"nint4-gu", 256, 8, 64},
        {"nint4-down", 128, 8, 64},
    };
    const auto plan = mfq::plan_moe_arena_slots(4096, demands);
    require(plan.at("nint4-gu") >= 8, "gate/up minimum was not satisfied");
    require(plan.at("nint4-down") >= 8, "down minimum was not satisfied");
    const int64_t used =
        static_cast<int64_t>(plan.at("nint4-gu")) * 256 +
        static_cast<int64_t>(plan.at("nint4-down")) * 128;
    require(used <= 4096, "planner exceeded its hard byte budget");
}

void test_budget_planner_rejects_insufficient_budget() {
    const std::vector<mfq::MoeArenaDemand> demands{
        {"nint4-gu", 256, 8, 64},
        {"nint4-down", 128, 8, 64},
    };
    bool rejected = false;
    try {
        (void)mfq::plan_moe_arena_slots(3071, demands);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "planner accepted less than the minimum working set");
}

void test_budget_planner_caps_registered_experts() {
    const std::vector<mfq::MoeArenaDemand> demands{
        {"small", 64, 2, 3},
        {"large", 128, 2, 4},
    };
    const auto plan = mfq::plan_moe_arena_slots(1 << 20, demands);
    require(plan.at("small") == 3, "planner exceeded small registered count");
    require(plan.at("large") == 4, "planner exceeded large registered count");
}

}  // namespace

int main() {
    try {
        test_lru_replaces_oldest_non_inflight_slot();
        test_inflight_slot_is_not_replaced();
        test_all_inflight_slots_reject_replacement();
        test_budget_planner_honors_minimums_and_hard_limit();
        test_budget_planner_rejects_insufficient_budget();
        test_budget_planner_caps_registered_experts();
        std::cout << "moe_cache_policy_tests=6 passed=6\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "moe_cache_policy_test failure=" << error.what() << "\n";
        return 1;
    }
}
