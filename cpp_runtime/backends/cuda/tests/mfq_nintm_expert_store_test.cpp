#include "nintm_expert_store.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
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
    auto ticket = pool.submit(requests);
    require(ticket.valid(), "asynchronous read ticket is empty");
    const auto stats = ticket.wait();
    require(!ticket.valid(), "completed read ticket remained valid");
    require(pool.workers() == 3, "parallel worker count was lost");
    require(stats.calls == 6, "parallel range-call accounting is wrong");
    require(stats.bytes == 102, "parallel range-byte accounting is wrong");
    require(stats.file_opens > 0 && stats.file_opens <= 3,
            "parallel file-open accounting is wrong");
    require(stats.wall_nanoseconds > 0, "parallel read timing is missing");
    for (int expert = 0; expert < 3; ++expert) {
        require(values[expert].front() == 17 * expert,
                "parallel expert value range is wrong");
        require(scales[expert].front() == 127 + expert,
                "parallel expert scale range is wrong");
    }

    mfq::cuda::NintMxfp4ReadPool serial(1);
    const auto first = serial.read(requests);
    const auto second = serial.read(requests);
    require(first.file_opens == 1, "serial worker did not open its source once");
    require(second.file_opens == 0, "serial worker did not reuse its file handle");

    std::array<std::uint8_t, 32> destructor_values{};
    const std::array<mfq::cuda::NintMxfp4ReadRequest, 1>
        destructor_requests{{
            {
                &store,
                &store.part(
                    2,
                    mfq::cuda::NintMxfp4ExpertStore::values),
                destructor_values,
            },
        }};
    {
        auto abandoned = serial.submit(destructor_requests);
        require(abandoned.valid(), "abandoned read ticket is empty");
    }
    require(destructor_values.front() == 34,
            "read ticket destructor did not preserve destination lifetime");
}

void test_concurrent_read_batches() {
    const auto blob = make_record();
    TempFile file(blob);
    mfq::cuda::NintMxfp4ExpertStore store({
        "experts.gate.weight", "NINTM", file.path.string(), 7, blob.size()});
    std::array<std::uint8_t, 32> first_values{};
    std::array<std::uint8_t, 2> first_scales{};
    std::array<std::uint8_t, 32> second_values{};
    std::array<std::uint8_t, 2> second_scales{};
    const std::array<mfq::cuda::NintMxfp4ReadRequest, 2> first_requests{{
        {
            &store,
            &store.part(0, mfq::cuda::NintMxfp4ExpertStore::values),
            first_values,
        },
        {
            &store,
            &store.part(0, mfq::cuda::NintMxfp4ExpertStore::scales),
            first_scales,
        },
    }};
    const std::array<mfq::cuda::NintMxfp4ReadRequest, 2> second_requests{{
        {
            &store,
            &store.part(1, mfq::cuda::NintMxfp4ExpertStore::values),
            second_values,
        },
        {
            &store,
            &store.part(1, mfq::cuda::NintMxfp4ExpertStore::scales),
            second_scales,
        },
    }};
    mfq::cuda::NintMxfp4ReadPool pool(3);
    auto first = std::async(std::launch::async, [&] {
        return pool.read(first_requests);
    });
    auto second = std::async(std::launch::async, [&] {
        return pool.read(second_requests);
    });
    require(first.get().calls == 2, "first concurrent batch was incomplete");
    require(second.get().calls == 2, "second concurrent batch was incomplete");
    require(first_values.front() == 0, "first concurrent values are wrong");
    require(first_scales.front() == 127, "first concurrent scales are wrong");
    require(second_values.front() == 17, "second concurrent values are wrong");
    require(second_scales.front() == 128, "second concurrent scales are wrong");
}

} // namespace

int main() {
    try {
        test_exact_ranges();
        test_unsupported_cohort();
        test_parallel_read_batch();
        test_concurrent_read_batches();
        std::cout << "cuda_nintm_expert_store_tests=4 passed=4\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cuda_nintm_expert_store_test failure="
                  << error.what() << "\n";
        return 1;
    }
}
