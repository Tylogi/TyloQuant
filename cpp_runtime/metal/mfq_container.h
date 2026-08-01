#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace mfq::metal {

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

    bool contains(const std::string& name) const noexcept;
    const MfqRecord& record(const std::string& name) const;
    // Read one exact byte range relative to a record.  This is the native
    // streaming primitive used by bounded expert residency; it never
    // materializes bytes outside [relative_offset, relative_offset+nbytes).
    std::vector<std::uint8_t> read_range(
        const std::string& name,
        std::uint64_t relative_offset,
        std::uint64_t nbytes) const;
    std::vector<std::uint8_t> read(const std::string& name) const;
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
