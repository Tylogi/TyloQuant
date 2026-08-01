#include "mfq_container.h"

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

} // namespace

int main() {
    try {
        const TemporaryDirectory root;
        test_basic_container(root.path());
        test_malformed_tables(root.path());
        test_sharded_ranges_and_source_lifetime(
            root.path());
        std::cout << "MFQ Metal container test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
