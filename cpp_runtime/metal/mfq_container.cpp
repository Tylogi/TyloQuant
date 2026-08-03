#include "mfq_container.h"

#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace mfq::metal {
namespace {

constexpr std::uint64_t kMaxStringBytes =
    std::uint64_t{64} << 20;
constexpr std::uint32_t kMaxMetadataEntries =
    std::uint32_t{1} << 16;
constexpr std::uint32_t kMaxRecordEntries =
    std::uint32_t{1} << 20;
constexpr std::uint64_t kMinMetadataEntryBytes =
    2 * sizeof(std::uint32_t);
constexpr std::uint64_t kMinRecordEntryBytes =
    2 * sizeof(std::uint32_t) + sizeof(std::uint64_t);

std::string deepseek_v4_ew_alias(std::string_view name) {
    static const std::unordered_map<std::string_view, std::string_view>
        globals{
            {"embed.weight", "token_embd.weight"},
            {"norm.weight", "output_norm.weight"},
            {"head.weight", "output.weight"},
            {"hc_head_fn", "output_hc_fn.weight"},
            {"hc_head_base", "output_hc_base.weight"},
            {"hc_head_scale", "output_hc_scale.weight"},
        };
    if (const auto found = globals.find(name);
        found != globals.end()) {
        return std::string(found->second);
    }

    constexpr std::string_view prefix = "layers.";
    if (!name.starts_with(prefix)) {
        return {};
    }
    const auto suffix_start = name.find('.', prefix.size());
    if (suffix_start == std::string_view::npos ||
        suffix_start == prefix.size()) {
        return {};
    }
    const auto layer = name.substr(
        prefix.size(), suffix_start - prefix.size());
    for (const char value : layer) {
        if (value < '0' || value > '9') {
            return {};
        }
    }
    const auto suffix = name.substr(suffix_start + 1);
    static const std::unordered_map<std::string_view, std::string_view>
        suffixes{
            {"attn.wq_a.weight", "attn_q_a.weight"},
            {"attn.q_norm.weight", "attn_q_a_norm.weight"},
            {"attn.wq_b.weight", "attn_q_b.weight"},
            {"attn.wkv.weight", "attn_kv.weight"},
            {"attn.kv_norm.weight", "attn_kv_a_norm.weight"},
            {"attn.attn_sink", "attn_sinks.weight"},
            {"attn.wo_a.weight", "attn_output_a.weight"},
            {"attn.wo_b.weight", "attn_output_b.weight"},
            {"attn_norm.weight", "attn_norm.weight"},
            {"ffn_norm.weight", "ffn_norm.weight"},
            {"ffn.gate.weight", "ffn_gate_inp.weight"},
            {"ffn.shared_experts.w1.weight", "ffn_gate_shexp.weight"},
            {"ffn.shared_experts.w3.weight", "ffn_up_shexp.weight"},
            {"ffn.shared_experts.w2.weight", "ffn_down_shexp.weight"},
            {"hc_attn_fn", "hc_attn_fn.weight"},
            {"hc_attn_base", "hc_attn_base.weight"},
            {"hc_attn_scale", "hc_attn_scale.weight"},
            {"hc_ffn_fn", "hc_ffn_fn.weight"},
            {"hc_ffn_base", "hc_ffn_base.weight"},
            {"hc_ffn_scale", "hc_ffn_scale.weight"},
            {"ffn.experts.gate_up.weight", "ffn_gate_up_exps.weight"},
            {"ffn.experts.down.weight", "ffn_down_exps.weight"},
            {"ffn.gate.tid2eid", "ffn_gate_tid2eid.weight"},
            {"ffn.gate.bias", "exp_probs_b.bias"},
            {"attn.compressor.wkv.weight", "attn_compressor_kv.weight"},
            {"attn.compressor.wgate.weight", "attn_compressor_gate.weight"},
            {"attn.compressor.ape", "attn_compressor_ape.weight"},
            {"attn.compressor.norm.weight", "attn_compressor_norm.weight"},
            {"attn.indexer.wq_b.weight", "indexer.attn_q_b.weight"},
            {"attn.indexer.weights_proj.weight", "indexer.proj.weight"},
            {"attn.indexer.compressor.wkv.weight", "indexer_compressor_kv.weight"},
            {"attn.indexer.compressor.wgate.weight", "indexer_compressor_gate.weight"},
            {"attn.indexer.compressor.ape", "indexer_compressor_ape.weight"},
            {"attn.indexer.compressor.norm.weight", "indexer_compressor_norm.weight"},
        };
    const auto mapped = suffixes.find(suffix);
    if (mapped == suffixes.end()) {
        return {};
    }
    return "blk." + std::string(layer) + "." +
        std::string(mapped->second);
}

void require_regular_file(const std::filesystem::path& path) {
    std::error_code error;
    const bool regular = std::filesystem::is_regular_file(path, error);
    if (error || !regular) {
        throw std::runtime_error("cannot open MFQ file: " + path.string());
    }
}

std::filesystem::path stable_source_path(
    const std::filesystem::path& path) {
    require_regular_file(path);
    std::error_code error;
    auto result = std::filesystem::weakly_canonical(path, error);
    if (error) {
        error.clear();
        result = std::filesystem::absolute(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot resolve MFQ file path: " + path.string());
        }
        result = result.lexically_normal();
    }
    if (!result.is_absolute()) {
        throw std::runtime_error(
            "resolved MFQ file path is not absolute: " + path.string());
    }
    return result;
}

std::uint64_t checked_file_size(
    const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (
        error
        || size > static_cast<std::uintmax_t>(
            std::numeric_limits<std::uint64_t>::max())
        || size > static_cast<std::uintmax_t>(
            std::numeric_limits<std::streamoff>::max())
    ) {
        throw std::runtime_error(
            "cannot determine usable MFQ file size: " + path.string());
    }
    return static_cast<std::uint64_t>(size);
}

class BoundedInput {
public:
    BoundedInput(
        std::istream& stream,
        std::uint64_t file_size,
        const std::filesystem::path& path)
        : stream_(stream),
          file_size_(file_size),
          path_(path) {}

    std::uint64_t position() const noexcept {
        return position_;
    }

    std::uint64_t remaining() const noexcept {
        return file_size_ - position_;
    }

    template <typename T>
    T scalar(const char* what) {
        T value{};
        read_exact(
            reinterpret_cast<char*>(&value),
            sizeof(value),
            what);
        return value;
    }

    std::string string(const char* what) {
        const auto length =
            scalar<std::uint32_t>("string length");
        if (
            static_cast<std::uint64_t>(length)
                > kMaxStringBytes
        ) {
            throw std::runtime_error(
                std::string("MFQ ") + what
                + " exceeds the supported string length: "
                + path_.string());
        }
        require_remaining(length, what);
        std::string value(
            static_cast<std::size_t>(length),
            '\0');
        if (length != 0) {
            read_exact(
                value.data(),
                length,
                what);
        }
        return value;
    }

private:
    void require_remaining(
        std::uint64_t count,
        const char* what) const {
        if (count > remaining()) {
            throw std::runtime_error(
                std::string("unexpected EOF reading ")
                + what + ": " + path_.string());
        }
    }

    void read_exact(
        char* destination,
        std::uint64_t count,
        const char* what) {
        require_remaining(count, what);
        if (
            count > static_cast<std::uint64_t>(
                std::numeric_limits<std::streamsize>::max())
        ) {
            throw std::runtime_error(
                std::string("MFQ ") + what
                + " is too large to read: " + path_.string());
        }
        stream_.read(
            destination,
            static_cast<std::streamsize>(count));
        if (!stream_) {
            throw std::runtime_error(
                std::string("failed reading MFQ ")
                + what + ": " + path_.string());
        }
        position_ += count;
    }

    std::istream& stream_;
    const std::uint64_t file_size_;
    const std::filesystem::path& path_;
    std::uint64_t position_ = 0;
};

void validate_entry_count(
    std::uint32_t count,
    std::uint32_t hard_limit,
    std::uint64_t minimum_entry_bytes,
    std::uint64_t remaining,
    const char* what,
    const std::filesystem::path& path) {
    if (
        count > hard_limit
        || static_cast<std::uint64_t>(count)
            > remaining / minimum_entry_bytes
    ) {
        throw std::runtime_error(
            std::string("invalid MFQ ") + what
            + ": " + path.string());
    }
}

} // namespace

MfqHeader MfqContainer::load_records(
    const std::filesystem::path& path,
    RecordMap& destination) {
    require_regular_file(path);
    const auto file_size = checked_file_size(path);
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open MFQ file: " + path.string());
    }
    BoundedInput input(stream, file_size, path);

    char magic[4]{};
    for (std::size_t index = 0; index < sizeof(magic); ++index) {
        magic[index] = input.scalar<char>("MFQ magic");
    }
    if (std::memcmp(magic, "MFQ1", sizeof(magic)) != 0) {
        throw std::runtime_error("bad MFQ magic: " + path.string());
    }

    MfqHeader header;
    header.version = input.scalar<std::uint32_t>("MFQ version");
    if (header.version == 0 || header.version > 2) {
        throw std::runtime_error(
            "unsupported MFQ version " + std::to_string(header.version) +
            ": " + path.string());
    }
    header.architecture = input.string("architecture");
    if (header.version >= 2) {
        const auto extra_count =
            input.scalar<std::uint32_t>("metadata count");
        validate_entry_count(
            extra_count,
            kMaxMetadataEntries,
            kMinMetadataEntryBytes,
            input.remaining(),
            "metadata count",
            path);
        for (std::uint32_t index = 0; index < extra_count; ++index) {
            auto key = input.string("metadata key");
            auto value = input.string("metadata value");
            if (!header.extra_json.emplace(std::move(key), std::move(value)).second) {
                throw std::runtime_error(
                    "duplicate MFQ metadata key: " + path.string());
            }
        }
    }

    header.record_count =
        input.scalar<std::uint32_t>("record count");
    validate_entry_count(
        header.record_count,
        kMaxRecordEntries,
        kMinRecordEntryBytes,
        input.remaining(),
        "record count",
        path);
    std::vector<MfqRecord> records;
    records.reserve(header.record_count);
    for (std::uint32_t index = 0; index < header.record_count; ++index) {
        MfqRecord value;
        value.name = input.string("record name");
        value.dtype = input.string("record dtype");
        value.source_path = path;
        value.nbytes =
            input.scalar<std::uint64_t>("record length");
        records.push_back(std::move(value));
    }

    std::uint64_t offset = input.position();
    for (auto& value : records) {
        value.offset = offset;
        if (value.nbytes > file_size - offset) {
            throw std::runtime_error(
                "MFQ file length does not match record table: "
                + path.string());
        }
        offset += value.nbytes;
    }
    if (offset != file_size) {
        throw std::runtime_error(
            "MFQ file length does not match record table: " + path.string());
    }
    for (auto& value : records) {
        if (!destination.emplace(value.name, std::move(value)).second) {
            throw std::runtime_error(
                "duplicate MFQ tensor record: " + path.string());
        }
    }
    return header;
}

std::uint64_t MfqContainer::metadata_uint(
    const MfqHeader& header,
    const std::string& key,
    std::uint64_t default_value) {
    const auto found = header.extra_json.find(key);
    if (found == header.extra_json.end()) {
        return default_value;
    }
    const auto& text = found->second;
    std::size_t parsed = 0;
    std::uint64_t value = 0;
    try {
        value = std::stoull(text, &parsed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error(
            "invalid MFQ integer metadata " + key + ": " + text);
    }
    while (parsed < text.size() &&
           std::isspace(static_cast<unsigned char>(text[parsed]))) {
        ++parsed;
    }
    if (parsed != text.size()) {
        throw std::runtime_error(
            "invalid MFQ integer metadata " + key + ": " + text);
    }
    return value;
}

std::vector<std::filesystem::path> MfqContainer::resolve_shards(
    const std::filesystem::path& path,
    std::uint64_t split_no,
    std::uint64_t split_count) {
    static const std::regex pattern(
        R"(^(.*)-([0-9]{5})-of-([0-9]{5})\.mfq$)");
    std::smatch match;
    const auto filename = path.filename().string();
    if (!std::regex_match(filename, match, pattern)) {
        throw std::runtime_error(
            "sharded MFQ path lacks -00001-of-00000 suffix: " +
            path.string());
    }
    const auto file_no = std::stoull(match[2].str());
    const auto file_count = std::stoull(match[3].str());
    if (file_no != split_no + 1 || file_count != split_count) {
        throw std::runtime_error(
            "MFQ shard filename/metadata mismatch: " + path.string());
    }

    std::vector<std::filesystem::path> result;
    result.reserve(static_cast<std::size_t>(split_count));
    for (std::uint64_t index = 1; index <= split_count; ++index) {
        std::ostringstream name;
        name << match[1].str() << "-" << std::setfill('0') << std::setw(5)
             << index << "-of-" << std::setw(5) << split_count << ".mfq";
        auto shard = stable_source_path(
            path.parent_path() / name.str());
        result.push_back(std::move(shard));
    }
    return result;
}

MfqContainer::MfqContainer(std::filesystem::path path) {
    path = stable_source_path(path);
    header_ = load_records(path, records_);
    const auto split_no = metadata_uint(header_, "split.no", 0);
    const auto split_count = metadata_uint(header_, "split.count", 1);
    if (split_count == 0 || split_count > 99999 || split_no >= split_count) {
        throw std::runtime_error(
            "invalid MFQ split metadata: " + path.string());
    }

    if (split_count == 1) {
        source_paths_.push_back(std::move(path));
        return;
    }

    const auto paths = resolve_shards(path, split_no, split_count);
    records_.clear();
    std::uint64_t actual_records = 0;
    std::uint64_t actual_tensors = 0;
    auto expected_records = std::numeric_limits<std::uint64_t>::max();
    auto expected_tensors = std::numeric_limits<std::uint64_t>::max();

    for (std::uint64_t index = 0; index < split_count; ++index) {
        const auto current = load_records(paths[index], records_);
        if (current.version != header_.version ||
            current.architecture != header_.architecture ||
            metadata_uint(current, "split.no", split_count) != index ||
            metadata_uint(current, "split.count", 0) != split_count) {
            throw std::runtime_error(
                "MFQ shard metadata mismatch: " + paths[index].string());
        }
        const auto current_expected_records = metadata_uint(
            current,
            "split.records.count",
            std::numeric_limits<std::uint64_t>::max());
        const auto current_expected_tensors = metadata_uint(
            current,
            "split.tensors.count",
            std::numeric_limits<std::uint64_t>::max());
        if (expected_records == std::numeric_limits<std::uint64_t>::max()) {
            expected_records = current_expected_records;
        } else if (
            current_expected_records !=
                std::numeric_limits<std::uint64_t>::max() &&
            current_expected_records != expected_records) {
            throw std::runtime_error("MFQ shard record count mismatch");
        }
        if (expected_tensors == std::numeric_limits<std::uint64_t>::max()) {
            expected_tensors = current_expected_tensors;
        } else if (
            current_expected_tensors !=
                std::numeric_limits<std::uint64_t>::max() &&
            current_expected_tensors != expected_tensors) {
            throw std::runtime_error("MFQ shard tensor count mismatch");
        }
        actual_records += current.record_count;
    }

    for (const auto& item : records_) {
        if (item.first.rfind("__mfq_asset__/", 0) != 0) {
            ++actual_tensors;
        }
    }
    if (expected_records != std::numeric_limits<std::uint64_t>::max() &&
        actual_records != expected_records) {
        throw std::runtime_error("MFQ shard record total mismatch");
    }
    if (expected_tensors != std::numeric_limits<std::uint64_t>::max() &&
        actual_tensors != expected_tensors) {
        throw std::runtime_error("MFQ shard tensor total mismatch");
    }
    source_paths_ = paths;
}

bool MfqContainer::contains(const std::string& name) const {
    if (records_.find(name) != records_.end()) {
        return true;
    }
    if (header_.architecture != "deepseek_v4-ew-mfq") {
        return false;
    }
    const auto alias = deepseek_v4_ew_alias(name);
    return !alias.empty() && records_.find(alias) != records_.end();
}

const MfqRecord& MfqContainer::record(const std::string& name) const {
    auto found = records_.find(name);
    if (found == records_.end() &&
        header_.architecture == "deepseek_v4-ew-mfq") {
        const auto alias = deepseek_v4_ew_alias(name);
        if (!alias.empty()) {
            found = records_.find(alias);
        }
    }
    if (found == records_.end()) {
        throw std::runtime_error("missing MFQ record: " + name);
    }
    return found->second;
}

std::vector<std::uint8_t> MfqContainer::read(
    const std::string& name) const {
    const auto& value = record(name);
    return read_range(name, 0, value.nbytes);
}

MfqMappedBytes MfqContainer::map_record(
    const std::string& name) const {
    const auto& value = record(name);
    if (value.nbytes == 0) {
        return {};
    }
    if (
        value.nbytes > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())
        || value.offset > static_cast<std::uint64_t>(
            std::numeric_limits<off_t>::max())
    ) {
        throw std::runtime_error(
            "MFQ record is too large to map: " + name);
    }
    const auto page_size = static_cast<std::uint64_t>(
        ::getpagesize());
    const auto mapped_offset =
        value.offset - value.offset % page_size;
    const auto delta = value.offset - mapped_offset;
    if (
        delta > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())
            - value.nbytes
        || mapped_offset > static_cast<std::uint64_t>(
            std::numeric_limits<off_t>::max())
    ) {
        throw std::runtime_error(
            "MFQ mapped record range overflows: " + name);
    }
    const auto mapped_size = static_cast<std::size_t>(
        delta + value.nbytes);
    const int descriptor = ::open(
        value.source_path.c_str(),
        O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        throw std::runtime_error(
            "cannot open MFQ record for mmap: " + name
            + ": " + std::strerror(errno));
    }
    void* mapping = ::mmap(
        nullptr,
        mapped_size,
        PROT_READ,
        MAP_PRIVATE,
        descriptor,
        static_cast<off_t>(mapped_offset));
    const int map_error = errno;
    ::close(descriptor);
    if (mapping == MAP_FAILED) {
        throw std::runtime_error(
            "cannot mmap MFQ record: " + name
            + ": " + std::strerror(map_error));
    }
    auto owner = std::shared_ptr<void>(
        mapping,
        [mapped_size](void* address) {
            ::munmap(address, mapped_size);
        });
    return MfqMappedBytes(
        std::move(owner),
        static_cast<const std::uint8_t*>(mapping)
            + static_cast<std::size_t>(delta),
        static_cast<std::size_t>(value.nbytes));
}

std::vector<std::uint8_t> MfqContainer::read_range(
    const std::string& name,
    std::uint64_t relative_offset,
    std::uint64_t nbytes) const {
    const auto& value = record(name);
    if (
        relative_offset > value.nbytes
        || nbytes > value.nbytes - relative_offset
    ) {
        throw std::out_of_range(
            "MFQ record byte range is out of bounds: " + name);
    }
    if (nbytes >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())
        || nbytes >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::streamsize>::max())
        || value.offset >
            std::numeric_limits<std::uint64_t>::max()
                - relative_offset
    ) {
        throw std::runtime_error(
            "MFQ record byte range is too large: " + name);
    }
    if (nbytes == 0) {
        return {};
    }
    const auto absolute_offset =
        value.offset + relative_offset;
    if (
        absolute_offset >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::streamoff>::max())
    ) {
        throw std::runtime_error(
            "MFQ record byte offset is too large: " + name);
    }
    std::ifstream stream(value.source_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "cannot open MFQ record source: " + name);
    }
    stream.seekg(0, std::ios::end);
    const auto source_end = stream.tellg();
    if (
        source_end < 0
        || absolute_offset
            > static_cast<std::uint64_t>(source_end)
        || nbytes
            > static_cast<std::uint64_t>(source_end)
                - absolute_offset
    ) {
        throw std::runtime_error(
            "MFQ record source was truncated: " + name);
    }
    stream.seekg(
        static_cast<std::streamoff>(absolute_offset),
        std::ios::beg);
    if (!stream) {
        throw std::runtime_error(
            "failed seeking MFQ record: " + name);
    }
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(nbytes));
    stream.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(result.size()));
    if (!stream) {
        throw std::runtime_error(
            "failed reading MFQ record byte range: " + name);
    }
    return result;
}

std::string MfqContainer::read_text(const std::string& name) const {
    const auto bytes = read(name);
    return std::string(bytes.begin(), bytes.end());
}

} // namespace mfq::metal
