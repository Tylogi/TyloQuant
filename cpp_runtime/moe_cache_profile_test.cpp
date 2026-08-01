#include "moe_cache_profile.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

std::filesystem::path write_profile(
        const std::string & name,
        const std::string & contents) {
    const auto path =
        std::filesystem::temp_directory_path() / name;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << contents;
    output.close();
    return path;
}

template <typename Function>
void require_rejected(Function && function, const char * message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, message);
}

void test_parses_partial_source_independent_profile() {
    const auto path = write_profile(
        "mfq-moe-profile-valid.json",
        R"JSON({
          "version": 1,
          "metadata": {"source": "manual"},
          "layers": {
            "0": {
              "ranking": [2, 0],
              "frequencies": {"2": 0.75, "0": 0.25}
            },
            "1": {"ranking": [1]},
            "2": {}
          }
        })JSON");
    const auto profile = mfq::load_moe_cache_profile(path.string());
    require(profile.layers.size() == 3, "layer count differs");
    require(
        profile.layers.at(0).ranking ==
            std::vector<int>({2, 0}),
        "ranking differs");
    require(
        profile.layers.at(0).frequencies.at(2) == 0.75,
        "frequency differs");
    require(
        profile.layers.at(2).ranking.empty() &&
            profile.layers.at(2).frequencies.empty(),
        "empty layer was not preserved");
}

void test_orders_frequency_globally_and_ranking_deterministically() {
    const auto path = write_profile(
        "mfq-moe-profile-order.json",
        R"JSON({
          "version": 1,
          "layers": {
            "0": {
              "ranking": [2, 1],
              "frequencies": {"2": 0.5, "1": 0.25}
            },
            "1": {"frequencies": {"0": 0.75}},
            "2": {"ranking": [3, 4]}
          }
        })JSON");
    const auto profile = mfq::load_moe_cache_profile(path.string());
    const auto hot =
        mfq::order_moe_profile_candidates(profile, false);
    require(hot.size() == 5, "candidate count differs");
    require(
        hot[0].layer == 1 && hot[0].expert == 0,
        "highest global frequency was not first");
    require(
        hot[1].layer == 0 && hot[1].expert == 2,
        "second global frequency differs");
    require(
        hot[2].layer == 0 && hot[2].expert == 1,
        "third global frequency differs");
    require(
        hot[3].layer == 2 && hot[3].expert == 3 &&
            hot[4].expert == 4,
        "ranking-only order differs");

    const auto cold =
        mfq::order_moe_profile_candidates(profile, true);
    require(
        cold.front().layer == 2 && cold.front().expert == 4,
        "cold-to-hot ordering differs");
    require(
        cold.back().layer == 1 && cold.back().expert == 0,
        "hottest candidate was not last");
}

void test_validates_model_dimensions() {
    const auto path = write_profile(
        "mfq-moe-profile-bounds.json",
        R"JSON({
          "version": 1,
          "layers": {
            "3": {"ranking": [7]}
          }
        })JSON");
    const auto profile = mfq::load_moe_cache_profile(path.string());
    require_rejected(
        [&]() {
            mfq::validate_moe_cache_profile(
                profile, std::unordered_map<int, int>{{0, 4}, {3, 4}});
        },
        "out-of-range expert was accepted");
    require_rejected(
        [&]() {
            mfq::validate_moe_cache_profile(
                profile, std::unordered_map<int, int>{{0, 8}});
        },
        "unknown layer was accepted");
}

void test_rejects_malformed_profiles() {
    const auto duplicate = write_profile(
        "mfq-moe-profile-duplicate.json",
        R"JSON({
          "version": 1,
          "layers": {"0": {"ranking": [1, 1]}}
        })JSON");
    require_rejected(
        [&]() { (void)mfq::load_moe_cache_profile(duplicate.string()); },
        "duplicate ranking was accepted");

    const auto negative = write_profile(
        "mfq-moe-profile-negative.json",
        R"JSON({
          "version": 1,
          "layers": {"0": {"frequencies": {"1": -0.1}}}
        })JSON");
    require_rejected(
        [&]() { (void)mfq::load_moe_cache_profile(negative.string()); },
        "negative frequency was accepted");

    const auto unknown = write_profile(
        "mfq-moe-profile-unknown.json",
        R"JSON({
          "version": 1,
          "layers": {"0": {"frequncy": {"1": 0.1}}}
        })JSON");
    require_rejected(
        [&]() { (void)mfq::load_moe_cache_profile(unknown.string()); },
        "unknown field was accepted");
}

}  // namespace

int main() {
    try {
        test_parses_partial_source_independent_profile();
        test_orders_frequency_globally_and_ranking_deterministically();
        test_validates_model_dimensions();
        test_rejects_malformed_profiles();
        std::cout << "moe_cache_profile_tests=4 passed=4\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr
            << "moe_cache_profile_test failure="
            << error.what() << "\n";
        return 1;
    }
}
