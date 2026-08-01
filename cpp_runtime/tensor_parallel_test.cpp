#include "tensor_parallel.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_equal_partition() {
    const auto slices = mfq::plan_tensor_parallel_slices(
        5120, 128, {0, 1}, {});
    mfq::validate_tensor_parallel_slices(slices, 5120, 128);
    require(slices.size() == 2, "equal split count");
    require(slices[0].size() == 2560, "equal first split");
    require(slices[1].size() == 2560, "equal second split");
}

void test_weighted_partition() {
    const auto slices = mfq::plan_tensor_parallel_slices(
        5120, 128, {0, 2}, {1.0, 3.0});
    mfq::validate_tensor_parallel_slices(slices, 5120, 128);
    require(slices[0].size() == 1280, "weighted first split");
    require(slices[1].size() == 3840, "weighted second split");
}

void test_weighted_partition_is_stable_across_layers() {
    const auto first = mfq::plan_tensor_parallel_slices(
        5120, 128, {0, 2}, {1.0, 3.0});
    const auto second = mfq::plan_tensor_parallel_slices(
        5120, 128, {0, 2}, {1.0, 3.0});
    require(
        first[0].device == second[0].device &&
        first[0].size() == second[0].size(),
        "weighted split must remain attached to its CUDA device");
    require(
        first[1].device == second[1].device &&
        first[1].size() == second[1].size(),
        "weighted split must not rotate between CUDA devices");
}

void test_padded_tail_partition() {
    const auto slices = mfq::plan_tensor_parallel_slices(
        17408, 128, {0, 1, 2}, {});
    mfq::validate_tensor_parallel_slices(slices, 17408, 128);
    int64_t total = 0;
    for (const auto & slice : slices) {
        total += slice.size();
    }
    require(total == 17408, "tail split coverage");
}

void test_invalid_small_extent() {
    bool failed = false;
    try {
        (void)mfq::plan_tensor_parallel_slices(
            128, 128, {0, 1}, {});
    } catch (const std::runtime_error &) {
        failed = true;
    }
    require(failed, "small extent must fail");
}

}  // namespace

int main() {
    try {
        test_equal_partition();
        test_weighted_partition();
        test_weighted_partition_is_stable_across_layers();
        test_padded_tail_partition();
        test_invalid_small_extent();
        std::cout << "tensor-parallel partition tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
