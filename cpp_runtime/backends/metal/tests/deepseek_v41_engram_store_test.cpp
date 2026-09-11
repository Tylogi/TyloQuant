#include "deepseek_v41_engram_store.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <unistd.h>

namespace {

using json = nlohmann::json;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void append_little(std::vector<std::uint8_t>& output, T value) {
    using Unsigned = std::make_unsigned_t<T>;
    const auto bits = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        output.push_back(static_cast<std::uint8_t>(bits >> (index * 8)));
    }
}

void write_u64(std::ofstream& stream, std::uint64_t value) {
    std::vector<std::uint8_t> bytes;
    append_little(bytes, value);
    stream.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
}

void write_asset(const std::filesystem::path& path) {
    std::vector<std::uint8_t> bytes{'D', '4', '1', 'T'};
    append_little<std::uint32_t>(bytes, 2); // version
    append_little<std::uint32_t>(bytes, 5); // tokenizer vocab
    append_little<std::uint32_t>(bytes, 5); // compressed vocab
    append_little<std::uint32_t>(bytes, 1); // layers
    append_little<std::uint32_t>(bytes, 3); // max n-gram
    append_little<std::uint32_t>(bytes, 2); // heads
    for (std::uint32_t value = 0; value < 5; ++value) {
        append_little(bytes, value);
    }
    append_little<std::int32_t>(bytes, 1);
    append_little<std::uint64_t>(bytes, 3);
    append_little<std::uint64_t>(bytes, 5);
    append_little<std::uint64_t>(bytes, 7);
    for (const std::uint32_t prime : {11u, 13u, 17u, 19u}) {
        append_little(bytes, prime);
    }
    std::ofstream stream(path, std::ios::binary);
    stream.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(stream), "cannot write Engram asset fixture");
}

void write_safetensors(const std::filesystem::path& root) {
    constexpr std::int64_t rows = 60;
    constexpr std::int64_t width = 32;
    const std::string shard_name = "model-00001-of-00001.safetensors";
    const std::string weight_name = "layers.1.engram.embed.weight";
    const std::string scale_name = "layers.1.engram.embed.scale";
    const std::uint64_t weight_bytes = rows * width;
    const std::uint64_t scale_bytes = rows;
    json header = {
        {
            weight_name,
            {
                {"dtype", "F8_E4M3"},
                {"shape", {rows, width}},
                {"data_offsets", {0, weight_bytes}},
            },
        },
        {
            scale_name,
            {
                {"dtype", "F8_E8M0"},
                {"shape", {rows, 1}},
                {"data_offsets", {weight_bytes, weight_bytes + scale_bytes}},
            },
        },
    };
    auto header_text = header.dump();
    while ((8 + header_text.size()) % 8 != 0) header_text.push_back(' ');
    std::ofstream shard(root / shard_name, std::ios::binary);
    write_u64(shard, header_text.size());
    shard.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::array<std::uint8_t, width> values = [&] {
            std::array<std::uint8_t, width> result{};
            result.fill(row % 2 == 0 ? 0x38u : 0x40u); // 1.0 or 2.0
            return result;
        }();
        shard.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size()));
    }
    const std::array<std::uint8_t, rows> scales = [] {
        std::array<std::uint8_t, rows> result{};
        result.fill(127u); // E8M0 1.0
        return result;
    }();
    shard.write(
        reinterpret_cast<const char*>(scales.data()),
        static_cast<std::streamsize>(scales.size()));
    require(static_cast<bool>(shard), "cannot write Engram Safetensors fixture");

    const json index = {
        {"metadata", {{"total_size", weight_bytes + scale_bytes}}},
        {"weight_map", {{weight_name, shard_name}, {scale_name, shard_name}}},
    };
    std::ofstream index_stream(root / "model.safetensors.index.json");
    index_stream << index.dump();
    require(static_cast<bool>(index_stream), "cannot write Engram index fixture");
}

void test_hash(const std::filesystem::path& root) {
    const auto asset = root / "engram.bin";
    write_asset(asset);
    const std::array<std::int64_t, 1> layers{1};
    const std::array<std::int64_t, 1> rows{60};
    auto state = mfq::metal::DeepseekV41EngramHashState::load(
        asset, 5, 5, 3, 2, 2, layers, rows);
    const std::array<std::int32_t, 3> tokens{0, 1, 2};
    const auto full = state.update(tokens, 1, 3, 0);
    require(
        full.size() == 1 &&
            full[0] == std::vector<std::uint64_t>{
                10, 21, 28, 45,
                3, 14, 37, 54,
                3, 14, 27, 44,
            },
        "Engram full-prefill hashes mismatch");

    state.reset();
    const auto prefix = state.update(
        std::span<const std::int32_t>(tokens.data(), 2), 1, 2, 0);
    const auto suffix = state.update(
        std::span<const std::int32_t>(tokens.data() + 2, 1), 1, 1, 2);
    require(
        prefix[0] == std::vector<std::uint64_t>(full[0].begin(), full[0].begin() + 8) &&
            suffix[0] == std::vector<std::uint64_t>(full[0].begin() + 8, full[0].end()),
        "Engram chunked hashes do not match one-shot prefill");
}

void test_ssd_cache(const std::filesystem::path& root) {
    write_safetensors(root);
    const std::array<std::int64_t, 1> layers{1};
    const std::array<std::int64_t, 1> rows{60};
    auto checkpoint =
        std::make_shared<mfq::metal::HfSafetensorStore>(root);
    mfq::metal::DeepseekV41EngramSsdStore store(
        checkpoint,
        layers,
        rows,
        32,
        (32 + 1 + 96) * 2,
        2);
    const std::array<std::uint64_t, 3> request{0, 1, 0};
    const auto values = store.lookup_bf16(0, request);
    require(values.size() == 96, "Engram lookup shape mismatch");
    require(
        std::all_of(values.begin(), values.begin() + 32,
                    [](std::uint16_t value) { return value == 0x3f80u; }) &&
            std::all_of(values.begin() + 32, values.begin() + 64,
                        [](std::uint16_t value) { return value == 0x4000u; }) &&
            std::all_of(values.begin() + 64, values.end(),
                        [](std::uint16_t value) { return value == 0x3f80u; }),
        "Engram FP8 row dequantization mismatch");
    auto stats = store.stats();
    require(
        stats.rows_loaded == 2 && stats.read_calls == 4 &&
            stats.bytes_read == 66 && stats.resident_rows == 2,
        "Engram cold-read/cache statistics mismatch");
    (void)store.lookup_bf16(
        0, std::span<const std::uint64_t>(request.data(), 1));
    stats = store.stats();
    require(
        stats.rows_loaded == 2 && stats.cache_hits >= 1,
        "Engram resident row did not hit the LRU cache");

    store.clear();
    const std::array<std::uint64_t, 1> even_request{2};
    const std::array<std::uint64_t, 1> odd_request{3};
    auto even_future = std::async(std::launch::async, [&] {
        return store.lookup_bf16(0, even_request);
    });
    auto odd_future = std::async(std::launch::async, [&] {
        return store.lookup_bf16(0, odd_request);
    });
    const auto even = even_future.get();
    const auto odd = odd_future.get();
    require(
        std::all_of(
            even.begin(),
            even.end(),
            [](std::uint16_t value) { return value == 0x3f80u; }) &&
            std::all_of(
                odd.begin(),
                odd.end(),
                [](std::uint16_t value) { return value == 0x4000u; }),
        "concurrent Engram lookups returned wrong rows");
}

} // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("mfq-dsv41-engram-test-" + std::to_string(::getpid()));
    try {
        std::filesystem::create_directories(root);
        test_hash(root);
        test_ssd_cache(root);
        std::filesystem::remove_all(root);
        std::cout << "DeepSeek-V4.1 Engram SSD store passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(root);
        std::cerr << "DeepSeek-V4.1 Engram SSD store failed: "
                  << error.what() << '\n';
        return 1;
    }
}
