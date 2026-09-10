#include "nintm_expert_store.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void append_little(std::vector<std::uint8_t>& output, T value) {
    using Unsigned = std::make_unsigned_t<T>;
    const auto bits = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        output.push_back(static_cast<std::uint8_t>(bits >> (8 * index)));
    }
}

std::vector<std::uint8_t> mxfp4_payload(
    const std::vector<std::int32_t>& experts,
    std::uint32_t output,
    std::uint32_t input) {
    const std::uint64_t rows = experts.size() * output;
    std::vector<std::uint8_t> payload{'M', 'X', 'T', '1', 1, 4, 0, 0};
    append_little(payload, rows);
    append_little(payload, static_cast<std::uint64_t>(input));
    append_little(payload, rows);
    append_little(payload, static_cast<std::uint64_t>(input / 2));
    append_little(payload, rows);
    append_little(payload, static_cast<std::uint64_t>(input / 32));
    for (const auto expert : experts) {
        for (std::uint32_t row = 0; row < output; ++row) {
            for (std::uint32_t column = 0; column < input / 2; ++column) {
                payload.push_back(static_cast<std::uint8_t>(
                    17 * expert + 3 * row + column));
            }
        }
    }
    for (const auto expert : experts) {
        for (std::uint32_t row = 0; row < output; ++row) {
            for (std::uint32_t column = 0; column < input / 32; ++column) {
                payload.push_back(static_cast<std::uint8_t>(
                    127 + expert + row + column));
            }
        }
    }
    return payload;
}

void append_pool(
    std::vector<std::uint8_t>& record,
    const std::vector<std::int32_t>& experts,
    std::uint32_t output,
    std::uint32_t input,
    const std::string& dtype = "MXFP4") {
    const auto payload = mxfp4_payload(experts, output, input);
    append_little(record, static_cast<std::uint32_t>(experts.size()));
    append_little(record, static_cast<std::uint32_t>(dtype.size()));
    append_little(record, static_cast<std::uint64_t>(payload.size()));
    append_little(record, std::uint64_t{0});
    for (const auto expert : experts) append_little(record, expert);
    record.insert(record.end(), dtype.begin(), dtype.end());
    record.insert(record.end(), payload.begin(), payload.end());
}

std::vector<std::uint8_t> make_record(const std::string& dtype = "MXFP4") {
    std::vector<std::uint8_t> record{'N', 'I', 'M', '2'};
    append_little(record, std::uint32_t{3});
    append_little(record, std::uint32_t{2});
    append_little(record, std::uint32_t{32});
    append_little(record, std::uint32_t{2});
    append_pool(record, {2, 0}, 2, 32, dtype);
    append_pool(record, {1}, 2, 32, dtype);
    return record;
}

struct TempFile {
    std::filesystem::path path;

    explicit TempFile(const std::vector<std::uint8_t>& record) {
        path = std::filesystem::temp_directory_path() /
            "mfq-cuda-nintm-expert-store-test.bin";
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        const std::array<std::uint8_t, 7> prefix{9, 8, 7, 6, 5, 4, 3};
        stream.write(
            reinterpret_cast<const char*>(prefix.data()),
            static_cast<std::streamsize>(prefix.size()));
        stream.write(
            reinterpret_cast<const char*>(record.data()),
            static_cast<std::streamsize>(record.size()));
        if (!stream) throw std::runtime_error("failed writing test fixture");
    }

    ~TempFile() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

void test_exact_ranges() {
    const auto blob = make_record();
    TempFile file(blob);
    mfq::cuda::NintMxfp4ExpertStore store({
        "model.block.0.mlp.experts.gate.weight",
        "NINTM",
        file.path.string(),
        7,
        blob.size(),
    });
    require(store.num_experts() == 3, "expert count was lost");
    require(store.out_per_expert() == 2, "output geometry was lost");
    require(store.neuron_len() == 32, "input geometry was lost");
    require(store.values_bytes_per_expert() == 32, "value size is wrong");
    require(store.scales_bytes_per_expert() == 2, "scale size is wrong");
    require(store.payload_bytes() == 102, "payload accounting is wrong");
    require(store.read_blob() == blob, "record-range base offset was ignored");

    for (int expert = 0; expert < 3; ++expert) {
        std::vector<std::uint8_t> values(32);
        std::vector<std::uint8_t> scales(2);
        store.read_part_into(
            store.part(expert, mfq::cuda::NintMxfp4ExpertStore::values),
            values);
        store.read_part_into(
            store.part(expert, mfq::cuda::NintMxfp4ExpertStore::scales),
            scales);
        require(values.front() == 17 * expert, "expert value range is wrong");
        require(values[16] == 17 * expert + 3, "expert row range is wrong");
        require(scales[0] == 127 + expert, "expert scale range is wrong");
        require(scales[1] == 128 + expert, "expert scale row is wrong");
    }
}

void test_unsupported_cohort() {
    const auto blob = make_record("NINT4");
    TempFile file(blob);
    bool rejected = false;
    try {
        (void)mfq::cuda::NintMxfp4ExpertStore({
            "experts.up.weight", "NINTM", file.path.string(), 7, blob.size()});
    } catch (const mfq::cuda::NintMxfp4Unsupported&) {
        rejected = true;
    }
    require(rejected, "non-MXFP4 cohort was accepted by the range store");
}

void test_parallel_read_batch() {
    const auto blob = make_record();
    TempFile file(blob);
    mfq::cuda::NintMxfp4ExpertStore store({
        "experts.down.weight", "NINTM", file.path.string(), 7, blob.size()});
    std::array<std::vector<std::uint8_t>, 3> values;
    std::array<std::vector<std::uint8_t>, 3> scales;
    std::vector<mfq::cuda::NintMxfp4ReadRequest> requests;
    requests.reserve(6);
    for (int expert = 0; expert < 3; ++expert) {
        values[expert].resize(32);
        scales[expert].resize(2);
        requests.push_back({
            &store,
            &store.part(expert, mfq::cuda::NintMxfp4ExpertStore::values),
            values[expert],
        });
        requests.push_back({
            &store,
            &store.part(expert, mfq::cuda::NintMxfp4ExpertStore::scales),
            scales[expert],
        });
    }
    mfq::cuda::NintMxfp4ReadPool pool(3);
    const auto stats = pool.read(requests);
    require(pool.workers() == 3, "parallel worker count was lost");
    require(stats.calls == 6, "parallel range-call accounting is wrong");
    require(stats.bytes == 102, "parallel range-byte accounting is wrong");
    require(stats.wall_nanoseconds > 0, "parallel read timing is missing");
    for (int expert = 0; expert < 3; ++expert) {
        require(values[expert].front() == 17 * expert,
                "parallel expert value range is wrong");
        require(scales[expert].front() == 127 + expert,
                "parallel expert scale range is wrong");
    }
}

} // namespace

int main() {
    try {
        test_exact_ranges();
        test_unsupported_cohort();
        test_parallel_read_batch();
        std::cout << "cuda_nintm_expert_store_tests=3 passed=3\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cuda_nintm_expert_store_test failure="
                  << error.what() << "\n";
        return 1;
    }
}
