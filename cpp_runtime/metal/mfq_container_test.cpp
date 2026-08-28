#include "mfq_container.h"

#include "../json/nlohmann/json.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;

struct RecordFixture {
    std::string name;
    std::string dtype;
    std::string payload;
};

using MetadataFixture =
    std::vector<std::pair<std::string, std::string>>;

template <typename T>
void write_scalar(std::ostream& stream, T value) {
    stream.write(
        reinterpret_cast<const char*>(&value),
        sizeof(value));
}

void write_string(
    std::ostream& stream,
    const std::string& value) {
    write_scalar<std::uint32_t>(
        stream,
        static_cast<std::uint32_t>(value.size()));
    stream.write(
        value.data(),
        static_cast<std::streamsize>(value.size()));
}

void write_u64_little(std::ostream& stream, std::uint64_t value) {
    std::array<char, 8> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<char>((value >> (index * 8)) & 0xffu);
    }
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_mfq(
    const std::filesystem::path& path,
    std::uint32_t version,
    const std::string& architecture,
    const MetadataFixture& metadata,
    const std::vector<RecordFixture>& records) {
    std::ofstream stream(
        path,
        std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error(
            "cannot create MFQ test fixture");
    }
    stream.write("MFQ1", 4);
    write_scalar<std::uint32_t>(stream, version);
    write_string(stream, architecture);
    if (version >= 2) {
        write_scalar<std::uint32_t>(
            stream,
            static_cast<std::uint32_t>(
                metadata.size()));
        for (const auto& [key, value] : metadata) {
            write_string(stream, key);
            write_string(stream, value);
        }
    }
    write_scalar<std::uint32_t>(
        stream,
        static_cast<std::uint32_t>(
            records.size()));
    for (const auto& record : records) {
        write_string(stream, record.name);
        write_string(stream, record.dtype);
        write_scalar<std::uint64_t>(
            stream,
            record.payload.size());
    }
    for (const auto& record : records) {
        stream.write(
            record.payload.data(),
            static_cast<std::streamsize>(
                record.payload.size()));
    }
    if (!stream) {
        throw std::runtime_error(
            "cannot write MFQ test fixture");
    }
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_rejected(
    Function&& function,
    const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, message);
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto nonce =
            std::chrono::steady_clock::now()
                .time_since_epoch()
                .count();
        path_ =
            std::filesystem::temp_directory_path()
            / (
                "mfq-metal-container-test-"
                + std::to_string(nonce)
            );
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

class CurrentPathGuard {
public:
    explicit CurrentPathGuard(
        const std::filesystem::path& path)
        : previous_(std::filesystem::current_path()) {
        std::filesystem::current_path(path);
    }

    ~CurrentPathGuard() {
        std::error_code ignored;
        std::filesystem::current_path(
            previous_,
            ignored);
    }

private:
    std::filesystem::path previous_;
};

void test_basic_container(
    const std::filesystem::path& root) {
    const auto path = root / "basic.mfq";
    write_mfq(
        path,
        2,
        "unit-test",
        {{"source", "\"metal-test\""}},
        {{"weight", "BLOB", "data"}});

    const mfq::metal::MfqContainer model(path);
    require(model.header().version == 2, "version mismatch");
    require(
        model.header().architecture == "unit-test",
        "architecture mismatch");
    require(model.records().size() == 1, "record count mismatch");
    require(model.contains("weight"), "missing record");
    require(
        model.record("weight").source_path.is_absolute(),
        "record source path was not made absolute");
    require(model.read_text("weight") == "data", "payload mismatch");
    const auto mapped = model.map_record("weight");
    require(
        mapped.size() == 4
            && std::string(
                   mapped.view().begin(),
                   mapped.view().end()) == "data",
        "mapped payload mismatch");
    const auto middle =
        model.read_range("weight", 1, 2);
    require(
        std::string(
            middle.begin(),
            middle.end()) == "at",
        "record range mismatch");
    require(
        model.read_range("weight", 4, 0).empty(),
        "empty record range mismatch");
    require_rejected(
        [&] {
            (void)model.read_range("weight", 3, 2);
        },
        "out-of-bounds record range was accepted");
}

void test_malformed_tables(
    const std::filesystem::path& root) {
    const auto huge_string =
        root / "huge-string.mfq";
    {
        std::ofstream stream(
            huge_string,
            std::ios::binary | std::ios::trunc);
        stream.write("MFQ1", 4);
        write_scalar<std::uint32_t>(stream, 1);
        write_scalar<std::uint32_t>(
            stream,
            std::numeric_limits<std::uint32_t>::max());
    }
    require_rejected(
        [&] {
            (void)mfq::metal::MfqContainer(
                huge_string);
        },
        "huge MFQ string length was accepted");

    const auto huge_metadata =
        root / "huge-metadata-count.mfq";
    {
        std::ofstream stream(
            huge_metadata,
            std::ios::binary | std::ios::trunc);
        stream.write("MFQ1", 4);
        write_scalar<std::uint32_t>(stream, 2);
        write_string(stream, "unit-test");
        write_scalar<std::uint32_t>(
            stream,
            std::numeric_limits<std::uint32_t>::max());
    }
    require_rejected(
        [&] {
            (void)mfq::metal::MfqContainer(
                huge_metadata);
        },
        "huge MFQ metadata count was accepted");

    const auto huge_records =
        root / "huge-record-count.mfq";
    {
        std::ofstream stream(
            huge_records,
            std::ios::binary | std::ios::trunc);
        stream.write("MFQ1", 4);
        write_scalar<std::uint32_t>(stream, 1);
        write_string(stream, "unit-test");
        write_scalar<std::uint32_t>(
            stream,
            std::numeric_limits<std::uint32_t>::max());
    }
    require_rejected(
        [&] {
            (void)mfq::metal::MfqContainer(
                huge_records);
        },
        "huge MFQ record count was accepted");

    const auto truncated_table =
        root / "truncated-record-table.mfq";
    {
        std::ofstream stream(
            truncated_table,
            std::ios::binary | std::ios::trunc);
        stream.write("MFQ1", 4);
        write_scalar<std::uint32_t>(stream, 1);
        write_string(stream, "unit-test");
        write_scalar<std::uint32_t>(stream, 1);
        write_scalar<std::uint32_t>(stream, 8);
        stream.write("x", 1);
    }
    require_rejected(
        [&] {
            (void)mfq::metal::MfqContainer(
                truncated_table);
        },
        "truncated MFQ record table was accepted");

    const auto huge_payload =
        root / "huge-record-length.mfq";
    {
        std::ofstream stream(
            huge_payload,
            std::ios::binary | std::ios::trunc);
        stream.write("MFQ1", 4);
        write_scalar<std::uint32_t>(stream, 1);
        write_string(stream, "unit-test");
        write_scalar<std::uint32_t>(stream, 1);
        write_string(stream, "weight");
        write_string(stream, "BLOB");
        write_scalar<std::uint64_t>(
            stream,
            std::numeric_limits<std::uint64_t>::max());
    }
    require_rejected(
        [&] {
            (void)mfq::metal::MfqContainer(
                huge_payload);
        },
        "huge MFQ record length was accepted");
}

MetadataFixture shard_metadata(int shard) {
    return {
        {"split.no", std::to_string(shard)},
        {"split.count", "2"},
        {"split.records.count", "2"},
        {"split.tensors.count", "2"},
    };
}

void test_sharded_ranges_and_source_lifetime(
    const std::filesystem::path& root) {
    const auto first =
        root / "sharded-00001-of-00002.mfq";
    const auto second =
        root / "sharded-00002-of-00002.mfq";
    write_mfq(
        first,
        2,
        "sharded-test",
        shard_metadata(0),
        {{"first", "BLOB", "alpha"}});
    write_mfq(
        second,
        2,
        "sharded-test",
        shard_metadata(1),
        {{"second", "BLOB", "bravo"}});

    std::unique_ptr<mfq::metal::MfqContainer> model;
    {
        CurrentPathGuard cwd(root);
        model =
            std::make_unique<
                mfq::metal::MfqContainer>(
                first.filename());
    }
    require(
        model->source_paths().size() == 2,
        "sharded source path count mismatch");
    for (const auto& path : model->source_paths()) {
        require(
            path.is_absolute(),
            "sharded source path was not made absolute");
    }
    const auto first_range =
        model->read_range("first", 1, 3);
    const auto second_range =
        model->read_range("second", 1, 3);
    require(
        std::string(
            first_range.begin(),
            first_range.end()) == "lph",
        "first-shard range mismatch");
    require(
        std::string(
            second_range.begin(),
            second_range.end()) == "rav",
        "second-shard range mismatch");

    const auto shortened =
        std::filesystem::file_size(second) - 2;
    std::filesystem::resize_file(
        second,
        shortened);
    require_rejected(
        [&] {
            (void)model->read_range(
                "second",
                0,
                5);
        },
        "truncated MFQ record source was accepted");
    require(
        model->read_text("first") == "alpha",
        "truncating one shard broke another source");
}

void test_hf_virtual_full_precision_container(
    const std::filesystem::path& root) {
    const auto hf = root / "hf-model";
    std::filesystem::create_directories(hf);
    const std::string config =
        R"({"model_type":"qwen3_5","hidden_size":2})";
    {
        std::ofstream stream(hf / "config.json", std::ios::binary);
        stream << config;
    }
    {
        std::ofstream stream(
            hf / "generation_config.json", std::ios::binary);
        stream << R"({"temperature":1.0,"top_k":20,"top_p":0.95})";
    }
    json header = {
        {"model.embed_tokens.weight", {
            {"dtype", "BF16"},
            {"shape", {2, 2}},
            {"data_offsets", {0, 8}},
        }},
        {"model.layers.0.mlp.experts.0.gate_proj.weight", {
            {"dtype", "I8"},
            {"shape", {1, 16}},
            {"data_offsets", {8, 24}},
        }},
        {"model.layers.0.mlp.experts.0.gate_proj.scale", {
            {"dtype", "F8_E8M0"},
            {"shape", {1, 1}},
            {"data_offsets", {24, 25}},
        }},
    };
    auto header_text = header.dump();
    while ((8 + header_text.size()) % 8 != 0) {
        header_text.push_back(' ');
    }
    {
        std::ofstream stream(hf / "model.safetensors", std::ios::binary);
        write_u64_little(stream, header_text.size());
        stream.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
        const std::array<unsigned char, 25> payload{
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
            0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
            0x7f,
        };
        stream.write(
            reinterpret_cast<const char*>(payload.data()),
            static_cast<std::streamsize>(payload.size()));
    }

    const mfq::metal::MfqContainer model(hf);
    require(model.header().version == 2, "HF virtual version mismatch");
    require(
        model.header().architecture == "qwen3_5-hf-full-mfq",
        "HF virtual architecture mismatch");
    require(
        model.header().extra_json.at("runtime.sampling.v1").find(
            "\"top_k\":20") != std::string::npos,
        "HF virtual runtime profile mismatch");
    require(model.source_paths().size() == 1, "HF virtual shard count mismatch");
    require(
        model.read_text("__mfq_asset__/model_config.json") == config,
        "HF virtual config asset mismatch");

    const auto dense = model.read("model.embed_tokens.weight");
    require(dense.size() == 28, "HF virtual dense record size mismatch");
    require(dense[0] == 2 && dense[4] == 2 && dense[12] == 2,
            "HF virtual dense shape prefix mismatch");
    for (std::size_t index = 0; index < 8; ++index) {
        require(
            dense[20 + index] == index + 1,
            "HF virtual dense payload mismatch");
    }
    const auto dense_cross = model.read_range(
        "model.embed_tokens.weight", 18, 4);
    require(
        dense_cross == std::vector<std::uint8_t>({0, 0, 1, 2}),
        "HF virtual prefix/payload range mismatch");
    const auto dense_mapped = model.map_record("model.embed_tokens.weight");
    require(
        std::vector<std::uint8_t>(
            dense_mapped.view().begin(), dense_mapped.view().end()) == dense,
        "HF virtual mapped record mismatch");

    const auto mx = model.read(
        "model.layers.0.mlp.experts.0.gate_proj.weight");
    require(mx.size() == 73, "HF virtual MX record size mismatch");
    require(
        std::string(mx.begin(), mx.begin() + 4) == "MXT1" &&
            mx[4] == 1 && mx[5] == 4,
        "HF virtual MX header mismatch");
    for (std::size_t index = 0; index < 16; ++index) {
        require(
            mx[56 + index] == 0x10 + index,
            "HF virtual MX values mismatch");
    }
    require(mx.back() == 0x7f, "HF virtual MX scale mismatch");
    require(
        !model.contains("model.layers.0.mlp.experts.0.gate_proj.scale"),
        "HF virtual container exposed a consumed scale");
}

} // namespace

int main() {
    try {
        const TemporaryDirectory root;
        test_basic_container(root.path());
        test_malformed_tables(root.path());
        test_sharded_ranges_and_source_lifetime(
            root.path());
        test_hf_virtual_full_precision_container(
            root.path());
        std::cout << "MFQ Metal container test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
