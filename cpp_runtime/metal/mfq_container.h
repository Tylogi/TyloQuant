#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace mfq::metal {

class MfqMappedBytes {
public:
    MfqMappedBytes() = default;

    std::span<const std::uint8_t> view() const noexcept {
        return {data_, size_};
    }

    const std::uint8_t* data() const noexcept {
        return data_;
    }

    std::size_t size() const noexcept {
        return size_;
    }

private:
    friend class MfqContainer;

    MfqMappedBytes(
        std::shared_ptr<void> mapping,
        const std::uint8_t* data,
        std::size_t size)
        : mapping_(std::move(mapping)),
          data_(data),
          size_(size) {}

    std::shared_ptr<void> mapping_;
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
};

struct MfqRecord {
    std::string name;
    std::string dtype;
    // Stable absolute path: streamed reads remain valid if the process changes
    // its working directory after the container table has been loaded.
    std::filesystem::path source_path;
    std::uint64_t offset = 0;
    std::uint64_t nbytes = 0;
};

struct MfqHeader {
    std::uint32_t version = 0;
    std::string architecture;
    std::unordered_map<std::string, std::string> extra_json;
    std::uint32_t record_count = 0;
};

class MfqContainer {
public:
    explicit MfqContainer(std::filesystem::path path);

    const MfqHeader& header() const noexcept {
        return header_;
    }

    const std::vector<std::filesystem::path>& source_paths() const noexcept {
        return source_paths_;
    }

    const std::unordered_map<std::string, MfqRecord>& records() const noexcept {
        return records_;
    }

    bool contains(const std::string& name) const;
    const MfqRecord& record(const std::string& name) const;
    // Read one exact byte range relative to a record.  This is the native
    // streaming primitive used by bounded expert residency; it never
    // materializes bytes outside [relative_offset, relative_offset+nbytes).
    std::vector<std::uint8_t> read_range(
        const std::string& name,
        std::uint64_t relative_offset,
        std::uint64_t nbytes) const;
    std::vector<std::uint8_t> read(const std::string& name) const;
    // Read-only mmap view used by full-resident model loading. The mapping is
    // released as soon as the parser has copied the record into its final MLX
    // buffers, so record staging never enters the process malloc depot.
    MfqMappedBytes map_record(const std::string& name) const;
    std::string read_text(const std::string& name) const;

private:
    using RecordMap = std::unordered_map<std::string, MfqRecord>;

    static MfqHeader load_records(
        const std::filesystem::path& path,
        RecordMap& destination);
    static std::vector<std::filesystem::path> resolve_shards(
        const std::filesystem::path& path,
        std::uint64_t split_no,
        std::uint64_t split_count);
    static std::uint64_t metadata_uint(
        const MfqHeader& header,
        const std::string& key,
        std::uint64_t default_value);

    MfqHeader header_;
    std::vector<std::filesystem::path> source_paths_;
    RecordMap records_;
};

} // namespace mfq::metal
