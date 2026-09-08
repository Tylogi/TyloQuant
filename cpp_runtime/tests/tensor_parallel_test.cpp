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

void test_four_and_eight_rank_partitions() {
    const auto four = mfq::plan_tensor_parallel_slices(
        5120, 128, {0, 1, 2, 3}, {});
    mfq::validate_tensor_parallel_slices(four, 5120, 128);
    require(four.size() == 4, "four-rank split count");
    for (const auto & slice : four) {
        require(slice.size() == 1280, "four-rank equal split");
    }

    const auto eight = mfq::plan_tensor_parallel_slices(
        5120, 128, {0, 1, 2, 3, 4, 5, 6, 7}, {});
    mfq::validate_tensor_parallel_slices(eight, 5120, 128);
    require(eight.size() == 8, "eight-rank split count");
    for (const auto & slice : eight) {
        require(slice.size() == 640, "eight-rank equal split");
    }
}

void test_four_and_eight_rank_expert_partitions() {
    const auto four = mfq::plan_tensor_parallel_slices(
        288, 1, {0, 1, 2, 3}, {});
    mfq::validate_tensor_parallel_slices(four, 288, 1);
    for (const auto & slice : four) {
        require(slice.size() == 72, "four-rank expert split");
    }

    const auto eight = mfq::plan_tensor_parallel_slices(
        288, 1, {0, 1, 2, 3, 4, 5, 6, 7}, {});
    mfq::validate_tensor_parallel_slices(eight, 288, 1);
    for (const auto & slice : eight) {
        require(slice.size() == 36, "eight-rank expert split");
    }

    const auto weighted = mfq::plan_tensor_parallel_slices(
        512, 1, {0, 1, 2, 3}, {1.0, 1.0, 2.0, 4.0});
    mfq::validate_tensor_parallel_slices(weighted, 512, 1);
    require(weighted[0].size() == 64, "weighted expert rank zero");
    require(weighted[1].size() == 64, "weighted expert rank one");
    require(weighted[2].size() == 128, "weighted expert rank two");
    require(weighted[3].size() == 256, "weighted expert rank three");
}

void test_peer_first_launch_order() {
    std::vector<size_t> four;
    std::vector<size_t> eight;
    for (size_t position = 0; position < 4; ++position) {
        four.push_back(
            mfq::peer_first_parallel_launch_index(position, 4));
    }
    for (size_t position = 0; position < 8; ++position) {
        eight.push_back(
            mfq::peer_first_parallel_launch_index(position, 8));
    }
    require(
        four == std::vector<size_t>({1, 2, 3, 0}),
        "four-rank peer-first launch order");
    require(
        eight == std::vector<size_t>({1, 2, 3, 4, 5, 6, 7, 0}),
        "eight-rank peer-first launch order");
    require(
        mfq::peer_first_parallel_launch_index(3, 4, 2) == 2,
        "nonzero primary rank launches last");
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
        test_four_and_eight_rank_partitions();
        test_four_and_eight_rank_expert_partitions();
        test_peer_first_launch_order();
        test_padded_tail_partition();
        test_invalid_small_extent();
        std::cout << "tensor-parallel partition tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
