#include "mlx_quantized_row_store.h"
#include "mlx_nint.h"
#include "mlx_tq.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void append_little(std::string& output, T value) {
    using Unsigned = std::make_unsigned_t<T>;
    const auto bits = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        output.push_back(static_cast<char>(bits >> (index * 8)));
    }
}

void append_string(std::string& output, const std::string& value) {
    append_little<std::uint32_t>(
        output, static_cast<std::uint32_t>(value.size()));
    output.append(value);
}

std::string row_table_payload() {
    constexpr std::uint64_t rows = 4;
    constexpr std::uint64_t width = 32;
    std::string result{"MXT1", 4};
    result.push_back(1);
    result.push_back(8);
    result.push_back(0);
    result.push_back(0);
    for (const auto value : {rows, width, rows, width, rows, width / 32}) {
        append_little<std::uint64_t>(result, value);
    }
    for (std::uint64_t row = 0; row < rows; ++row) {
        result.append(
            static_cast<std::size_t>(width),
            static_cast<char>((row & 1u) == 0 ? 0x38 : 0x40));
    }
    result.append(static_cast<std::size_t>(rows), static_cast<char>(127));
    return result;
}

void write_fixture(const std::filesystem::path& path,
                   const std::string& dtype = "MXFP8",
                   const std::string& payload = row_table_payload(),
                   const std::string& name = "table.weight", int split_no = -1) {
    std::string bytes{"MFQ1", 4};
    append_little<std::uint32_t>(bytes, 2);
    append_string(bytes, "unit-test");
    append_little<std::uint32_t>(bytes, split_no < 0 ? 0 : 4);
    if (split_no >= 0) {
        for (const auto& field : std::vector<std::pair<std::string, std::string>>{
                {"split.no", std::to_string(split_no)}, {"split.count", "3"},
                {"split.tensors.count", "3"}, {"split.records.count", "3"}}) {
            append_string(bytes, field.first);
            append_string(bytes, field.second);
        }
    }
    append_little<std::uint32_t>(bytes, 1);
    append_string(bytes, name);
    append_string(bytes, dtype);
    append_little<std::uint64_t>(bytes, payload.size());
    bytes.append(payload);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(stream), "cannot write MXFP8 row fixture");
}

std::string tq_payload(int seed, bool columns) {
    constexpr int rows = 4, width = 768;
    const std::array<int, 4> bits{1, 3, 5, 8};
    std::string result{"RTRELL05", 8};
    for (const auto value : {5, rows, width, int(columns), columns ? 0 : 16,
            width * (1 + 3 + 5 + 8) / 8}) append_little<std::uint32_t>(result, value);
    result.append(32, '\0');
    std::uint64_t words = 0;
    for (int row = 0; row < rows; ++row) {
        append_little<std::uint64_t>(result, words | (std::uint64_t(bits[row] - 1) << 32)
            | (std::uint64_t(0x2800 + row * 0x100) << 40));
        words += width * bits[row] / 32;
    }
    if (columns) for (int col = 0; col < width; ++col) append_little<std::uint16_t>(result, 0x3800 + col % 32);
    else result.append(rows * (width / 16) / 2, char(0x87));
    std::uint32_t state = seed;
    for (std::size_t word = 0; word < words; ++word) {
        state = state * 1664525u + 1013904223u;
        append_little<std::uint32_t>(result, state);
    }
    return result;
}

void write_tq_shards(const std::filesystem::path& path, const nlohmann::json& manifest,
        bool wrong_dtype = false, bool physical_alias = false) {
    const std::array<std::string, 2> payloads{tq_payload(1, false), tq_payload(2, true)};
    std::string bytes{"MFQ1", 4};
    append_little<std::uint32_t>(bytes, 2);
    append_string(bytes, "unit-test");
    append_little<std::uint32_t>(bytes, 1);
    append_string(bytes, "tq_row_shards"); append_string(bytes, manifest.dump());
    append_little<std::uint32_t>(bytes, physical_alias ? 3 : 2);
    for (int index = 0; index < (physical_alias ? 3 : 2); ++index) {
        append_string(bytes, index == 2 ? "table.weight" : "table.weight.tq_row_shard.0000" + std::to_string(index));
        append_string(bytes, wrong_dtype && index == 1 ? "F16" : "TQ");
        append_little<std::uint64_t>(bytes, payloads[index % 2].size());
    }
    bytes += payloads[0]; bytes += payloads[1];
    if (physical_alias) bytes += payloads[0];
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(bytes.data(), bytes.size());
    require(bool(stream), "cannot write TQ row shard fixture");
}

void test_tq_shards(const std::filesystem::path& root) {
    using namespace mlx::core;
    using namespace mfq::metal;
    std::filesystem::create_directories(root);
    const nlohmann::json manifest = {{"table.weight",
        {"table.weight.tq_row_shard.00000", "table.weight.tq_row_shard.00001"}}};
    const auto path = root / "rows.mfq";
    write_tq_shards(path, manifest);
    const MfqContainer model(path);
    MlxQuantizedRowStore store(model, "table.weight", 8, 513, 3, 2);
    std::vector<array> expected;
    for (int index = 0; index < 2; ++index) {
        const auto payload = tq_payload(index + 1, index == 1);
        const auto blob = std::span(reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size());
        expected.push_back(MlxTqWeight::from_blob(blob, 513).embedding(arange(4, int32)));
    }
    const std::array<std::int64_t, 6> requested{3, 4, 7, 0, 4, 3};
    const auto reference = take(concatenate(expected, 0), array(requested.data(), {6}, int64), 0);
    require(max(abs(astype(store.gather(requested), float32) - astype(reference, float32))).item<float>() == 0.f,
        "TQ row shard boundaries or request order differ");
    const auto before = store.stats();
    require(before.rows_loaded == 4 && before.cache_hits == 2 && before.resident_rows == 3
        && before.resident_payload_bytes <= before.cache_limit_bytes, "TQ row shard LRU accounting differs");
    const std::array<std::int64_t, 1> cached{4};
    store.gather(cached).eval();
    require(store.stats().rows_loaded == before.rows_loaded, "TQ shard boundary row did not hit its cache");
    store.clear();
    const std::array<std::int64_t, 1> left{3}, right{4};
    auto first = std::async(std::launch::async, [&] { return store.prepare_rows(left); });
    auto second = std::async(std::launch::async, [&] { return store.prepare_rows(right); });
    auto actual = concatenate({first.get().decode(), second.get().decode()}, 0);
    require(max(abs(astype(actual, float32) - astype(take(concatenate(expected, 0), array({3, 4}), 0), float32))).item<float>() == 0.f,
        "TQ asynchronous shard reads differ");
    for (const auto invalid : {-1, 8}) {
        bool rejected = false;
        const std::array<std::int64_t, 1> ids{invalid};
        try { store.prepare_rows(ids); } catch (const std::out_of_range&) { rejected = true; }
        require(rejected, "TQ logical row bounds are not enforced");
    }
    const std::vector<nlohmann::json> invalid_manifests{
        nlohmann::json::array(), {{"table.weight", nlohmann::json::array()}},
        {{"table.weight", {"table.weight.tq_row_shard.00000", "table.weight.tq_row_shard.00000"}}},
        {{"table.weight", {"missing"}}}, {{"table.weight", {1}}}};
    for (std::size_t index = 0; index < invalid_manifests.size() + 3; ++index) {
        const auto invalid_path = root / ("invalid-" + std::to_string(index) + ".mfq");
        write_tq_shards(invalid_path, index < invalid_manifests.size() ? invalid_manifests[index] : manifest,
            index == invalid_manifests.size(), index == invalid_manifests.size() + 1);
        bool rejected = false;
        try {
            MlxQuantizedRowStore invalid(MfqContainer(invalid_path), "table.weight",
                index == invalid_manifests.size() + 2 ? 7 : 8, 513, 3);
        } catch (const std::exception&) { rejected = true; }
        require(rejected, "TQ invalid row-shard manifest was accepted");
    }
}

std::string nint_payload(bool adaptive, int rows = 4) {
    constexpr int width = 32, group = 16, groups = 2;
    const std::array<int, 4> qs{1, 2, 4, 8}, ks{1, 2, 3, 4};
    std::string result;
    result.push_back(static_cast<char>(4 | (adaptive ? 128 : 0)));
    result.push_back(static_cast<char>(adaptive ? 2 : 4));
    append_little<std::int32_t>(result, group);
    append_little<std::int32_t>(result, 0);
    append_little<std::int32_t>(result, width);
    append_little<std::uint32_t>(result, 2);
    append_little<std::int64_t>(result, rows);
    append_little<std::int64_t>(result, width);
    append_little<std::uint32_t>(result, rows);
    append_little<std::uint32_t>(result, groups);
    for (int row = 0; row < rows; ++row) append_little<std::uint16_t>(result, 0x3800);
    for (int row = 0; row < rows; ++row) append_little<std::uint16_t>(result, 0x3400);
    const auto pack = [&](const std::vector<unsigned>& values, int bits) {
        std::string bytes((values.size() * bits + 7) / 8, '\0');
        for (std::size_t i = 0; i < values.size(); ++i) {
            for (int bit = 0; bit < bits; ++bit) {
                if (values[i] & (1u << bit)) {
                    const auto offset = i * bits + bit;
                    bytes[offset / 8] = static_cast<char>(
                        static_cast<unsigned char>(bytes[offset / 8]) | (1u << (offset % 8)));
                }
            }
        }
        result += bytes;
    };
    if (adaptive) {
        std::vector<unsigned> selectors;
        for (int row = 0; row < rows; ++row) selectors.push_back(row % 4);
        pack(selectors, 2);
    }
    for (int bits = 1; bits <= 8; ++bits) {
        if (!adaptive && bits != 4) continue;
        for (int field = 0; field < 2; ++field) {
            std::vector<unsigned> values;
            for (int row = 0; row < rows; ++row) {
                if (adaptive && ks[row % 4] != bits) continue;
                for (int g = 0; g < groups; ++g)
                    values.push_back(field ? g : ((g + 1) & ((1u << bits) - 1)));
            }
            pack(values, bits);
        }
    }
    if (adaptive) {
        std::vector<unsigned> selectors;
        for (int row = 0; row < rows; ++row) selectors.push_back(qs[row % 4] - 1);
        pack(selectors, 3);
    }
    for (int bits = 1; bits <= 8; ++bits) {
        if (!adaptive && bits != 4) continue;
        std::vector<unsigned> values;
        for (int row = 0; row < rows; ++row) {
            if (adaptive && qs[row % 4] != bits) continue;
            for (int column = 0; column < width; ++column)
                values.push_back((row + column) & ((1u << bits) - 1));
        }
        pack(values, bits);
    }
    return result;
}

std::vector<float> values(mlx::core::array value);

void test_nint_rows(const std::filesystem::path& path, bool adaptive) {
    const auto payload = nint_payload(adaptive);
    write_fixture(path, "NINT", payload);
    mfq::metal::MfqContainer model(path);
    const auto blob = std::span(reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size());
    auto reference = values(mfq::metal::MlxNintWeight::from_blob(blob)
        .dequantize(mlx::core::float16));
    mfq::metal::MlxQuantizedRowStore store(model, "table.weight", 4, 32, 2, 2);
    const std::array<std::int64_t, 5> request{3, 0, 3, 1, 2};
    const auto actual = values(store.gather(request));
    for (std::size_t row = 0; row < request.size(); ++row)
        for (int column = 0; column < 32; ++column)
            require(actual[row * 32 + column] == reference[request[row] * 32 + column],
                "streamed NINT row differs from resident decoder");
    const auto stats = store.stats();
    require(stats.row_requests == 5 && stats.rows_loaded == 4 && stats.cache_hits == 1
        && stats.read_calls == 1 && stats.bytes_read <= payload.size()
        && stats.bytes_read != 0 && stats.resident_rows == 2
        && stats.resident_payload_bytes <= stats.cache_limit_bytes,
        "NINT row-store accounting mismatch");
    const std::array<std::int64_t, 1> first{3}, second{0};
    auto a = std::async(std::launch::async, [&] { return store.prepare_rows(first); });
    auto b = std::async(std::launch::async, [&] { return store.prepare_rows(second); });
    require(values(a.get().decode()) == std::vector<float>(reference.begin() + 96, reference.end()),
        "concurrent NINT first row changed");
    require(values(b.get().decode()) == std::vector<float>(reference.begin(), reference.begin() + 32),
        "concurrent NINT second row changed");
    store.clear();
    require(store.cached_rows() == 0, "NINT row cache was not cleared");
    bool rejected = false;
    try { const std::array<std::int64_t, 1> invalid{4}; (void)store.gather(invalid); }
    catch (const std::out_of_range&) { rejected = true; }
    require(rejected, "NINT row store accepted an invalid row");
    rejected = false;
    try { mfq::metal::MlxQuantizedRowStore wrong(model, "table.weight", 5, 32, 2, 2); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "NINT row store accepted mismatched table geometry");

    for (const auto workers : {std::size_t{0}, std::numeric_limits<std::size_t>::max()}) {
        rejected = false;
        try { mfq::metal::MlxQuantizedRowStore wrong(model, "table.weight", 4, 32, 2, workers); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "NINT row store accepted an invalid worker count");
    }

    auto invalid_payload = payload;
    invalid_payload[48] = 0;
    invalid_payload[49] = static_cast<char>(0x7e);
    write_fixture(path, "NINT", invalid_payload);
    mfq::metal::MfqContainer invalid_model(path);
    mfq::metal::MlxQuantizedRowStore invalid_store(invalid_model, "table.weight", 4, 32, 2, 2);
    const std::array<std::int64_t, 2> invalid_batch{0, 3};
    rejected = false;
    try { (void)invalid_store.prepare_rows(invalid_batch); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected && invalid_store.cached_rows() == 0,
        "NINT row store published a partially validated batch");
    require(values(invalid_store.gather(second)) == std::vector<float>(reference.begin(), reference.begin() + 32),
        "NINT row store did not recover after an invalid anchor");

    write_fixture(path, "NINT", payload);
    mfq::metal::MfqContainer truncated_model(path);
    mfq::metal::MlxQuantizedRowStore truncated_store(truncated_model, "table.weight", 4, 32, 2, 2);
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    rejected = false;
    try { (void)truncated_store.prepare_rows(invalid_batch); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected && truncated_store.cached_rows() == 0,
        "NINT row store published a short-read batch");
    write_fixture(path, "NINT", payload);
    require(values(truncated_store.gather(second)) == std::vector<float>(reference.begin(), reference.begin() + 32),
        "NINT row store did not recover after a short read");
}

std::vector<float> values(mlx::core::array value) {
    value = mlx::core::astype(value, mlx::core::float32);
    value.eval();
    return {value.data<float>(), value.data<float>() + value.size()};
}

void test_multi_page_nint_rows(const std::filesystem::path& path, int rows, std::size_t workers) {
    constexpr int width = 32;
    const auto payload = nint_payload(true, rows);
    write_fixture(path, "NINT", payload);
    mfq::metal::MfqContainer model(path);
    const auto blob = std::span(reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size());
    const auto reference = values(mfq::metal::MlxNintWeight::from_blob(blob)
        .dequantize(mlx::core::float16));
    mfq::metal::MlxQuantizedRowStore store(model, "table.weight", rows, width, 8, workers);
    std::vector<std::int64_t> request;
    for (int index = 0; index < rows / 2; ++index) {
        request.push_back((index * 2053) % rows);
        if (index % 7 == 0) request.push_back(request.back());
    }
    const auto check = [&](const auto& selected, auto prepared) {
        const auto actual = values(prepared.decode());
        for (std::size_t row = 0; row < selected.size(); ++row)
            for (int column = 0; column < width; ++column)
                require(actual[row * width + column] == reference[selected[row] * width + column],
                    "multi-page NINT read changed row order or precision");
    };
    check(request, store.prepare_rows(request));
    const auto stats = store.stats();
    require(stats.rows_loaded == rows / 2 && stats.read_calls > 1
        && stats.read_calls < stats.rows_loaded && stats.bytes_read <= payload.size()
        && stats.resident_payload_bytes <= stats.cache_limit_bytes,
        "multi-page NINT reads were not bounded and coalesced");
    std::vector<std::int64_t> reversed(request.rbegin(), request.rend());
    auto a = std::async(std::launch::async, [&] { return store.prepare_rows(request); });
    auto b = std::async(std::launch::async, [&] { return store.prepare_rows(reversed); });
    check(request, a.get());
    check(reversed, b.get());
}

void require_row(
    const std::vector<float>& source,
    std::size_t row,
    float expected) {
    const auto begin = source.begin() + static_cast<std::ptrdiff_t>(row * 32);
    require(
        std::all_of(begin, begin + 32, [expected](float value) {
            return std::fabs(value - expected) < 1e-6f;
        }),
        "decoded MXFP8 row mismatch");
}

void test_shared_engram_rows(const std::filesystem::path& root) {
    const std::string name = "model.block.1.associative_memory.embedding.weight";
    const auto shared = root / "Engram" / "model-EW-Engram-MXFP8-00002-of-00003.mfq";
    std::filesystem::create_directories(shared.parent_path());
    write_fixture(shared, "MXFP8", row_table_payload(), name, 1);
    for (const std::string precision : {"Q3", "Q2.5"}) {
        const auto directory = root / precision;
        std::filesystem::create_directories(directory);
        const auto first = directory / ("model-EW-" + precision + "-00001-of-00003.mfq");
        const auto last = directory / ("model-EW-" + precision + "-00003-of-00003.mfq");
        write_fixture(first, "MXFP8", row_table_payload(), "model.input.weight", 0);
        write_fixture(last, "MXFP8", row_table_payload(), "model.output.weight", 2);
        for (const auto& entry : {first, last}) {
            mfq::metal::MfqContainer model(entry);
            require(model.source_paths().size() == 3 &&
                model.record(name).source_path == std::filesystem::canonical(shared),
                "Metal did not resolve the original shared Engram file");
            mfq::metal::MlxQuantizedRowStore store(model, name, 4, 32, 2, 2);
            const std::array<std::int64_t, 4> request{3, 0, 3, 2};
            const auto output = values(store.gather(request));
            for (std::size_t row = 0; row < request.size(); ++row)
                require_row(output, row, request[row] % 2 ? 2.0f : 1.0f);
            require(store.stats().rows_loaded == 3 && store.cached_rows() == 2,
                "shared MXFP8 Engram did not keep bounded row residency");
            store.clear();
            require_row(values(store.prepare_rows(std::span(request).first(1)).decode()), 0, 2.0f);
            require(!std::filesystem::exists(directory / shared.filename()),
                "shared Engram was copied into a precision directory");
        }
    }
}

} // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("mfq-mxfp8-row-store-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directories(root);
        test_tq_shards(root / "tq");
        test_shared_engram_rows(root / "shared");
        const auto path = root / "fixture.mfq";
        write_fixture(path);
        mfq::metal::MfqContainer model(path);
        mfq::metal::MlxQuantizedRowStore store(
            model, "table.weight", 4, 32, 2, 2);

        const std::array<std::int64_t, 3> request{0, 1, 0};
        auto decoded = store.gather(request);
        require(
            decoded.shape() == mlx::core::Shape{3, 32},
            "MXFP8 gather shape mismatch");
        const auto output = values(std::move(decoded));
        require_row(output, 0, 1.0f);
        require_row(output, 1, 2.0f);
        require_row(output, 2, 1.0f);
        auto stats = store.stats();
        require(
            stats.row_requests == 3 && stats.cache_hits == 1 &&
            stats.cache_misses == 2 && stats.rows_loaded == 2 &&
            stats.bytes_read == 66 && stats.read_calls == 4 &&
            stats.resident_rows == 2 &&
            stats.resident_payload_bytes == 128 &&
            stats.cache_limit_bytes == 128,
            "MXFP8 row cache statistics mismatch");

        (void)store.gather(std::span<const std::int64_t>(request.data(), 1));
        stats = store.stats();
        require(
            stats.rows_loaded == 2 && stats.cache_hits == 2,
            "resident MXFP8 row did not hit the cache");

        store.clear();
        const std::array<std::int64_t, 1> even_request{2};
        const std::array<std::int64_t, 1> odd_request{3};
        auto even_future = std::async(std::launch::async, [&] {
            return store.prepare_rows(even_request);
        });
        auto odd_future = std::async(std::launch::async, [&] {
            return store.prepare_rows(odd_request);
        });
        require_row(values(even_future.get().decode()), 0, 1.0f);
        require_row(values(odd_future.get().decode()), 0, 2.0f);

        const auto previous_stream = mlx::core::default_stream(mlx::core::Device::gpu);
        const auto runtime_stream = mlx::core::new_thread_unsafe_stream(mlx::core::Device::gpu);
        mlx::core::set_default_stream(runtime_stream);
        test_nint_rows(root / "nint.mfq", false);
        test_nint_rows(root / "adaptive-nint.mfq", true);
        test_multi_page_nint_rows(root / "multi-page-nint.mfq", 4096, 4);
        test_multi_page_nint_rows(root / "multi-page-nint32.mfq", 32768, 32);
        mlx::core::synchronize();
        mlx::core::set_default_stream(previous_stream);

        std::filesystem::remove_all(root);
        std::cout << "MXFP8/NINT/TQ row store passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
        std::cerr << "MXFP8 row store failed: " << error.what() << '\n';
        return 1;
    }
}
