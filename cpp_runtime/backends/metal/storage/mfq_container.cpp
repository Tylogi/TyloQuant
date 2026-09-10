#include "mfq_container.h"

#include "mlx_legacy_tensor_compat.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_set>
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
constexpr std::string_view kModelConfigAsset =
    "__mfq_asset__/model_config.json";
constexpr std::string_view kMinicpmoResamplerAsset =
    "__mfq_asset__/minicpmo45-resampler-pos-embed-v1.bf16";
constexpr std::uint64_t kMxHeaderBytes = 56;

using json = nlohmann::json;

std::string read_file_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path.string());
    }
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < 0) {
        throw std::runtime_error("cannot size " + path.string());
    }
    std::string result(static_cast<std::size_t>(end), '\0');
    stream.seekg(0, std::ios::beg);
    stream.read(result.data(), static_cast<std::streamsize>(result.size()));
    if (!stream && !result.empty()) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return result;
}

template <typename T>
void append_little(std::vector<std::uint8_t>& destination, T value) {
    using Unsigned = std::make_unsigned_t<T>;
    const auto bits = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        destination.push_back(static_cast<std::uint8_t>(
            bits >> (index * 8)));
    }
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right,
    std::string_view what) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(std::string(what) + " byte size overflow");
    }
    return left + right;
}

std::uint64_t checked_product(
    const std::vector<std::int64_t>& shape,
    std::uint64_t item_size,
    const std::string& name) {
    std::uint64_t result = item_size;
    // Safetensors represents a scalar with an empty shape. Its dense payload
    // still contains exactly one item, so the multiplicative identity above
    // is already the correct byte count.
    for (const auto dimension : shape) {
        if (dimension <= 0 ||
            static_cast<std::uint64_t>(dimension) >
                std::numeric_limits<std::uint64_t>::max() / result) {
            throw std::runtime_error(
                "invalid Safetensors tensor shape: " + name);
        }
        result *= static_cast<std::uint64_t>(dimension);
    }
    return result;
}

std::uint64_t dense_item_size(std::string_view dtype) {
    if (dtype == "BF16" || dtype == "F16") {
        return 2;
    }
    if (dtype == "F32" || dtype == "I32") {
        return 4;
    }
    if (dtype == "I64") {
        return 8;
    }
    return 0;
}

std::vector<std::uint8_t> dense_prefix(
    const std::vector<std::int64_t>& shape) {
    std::vector<std::uint8_t> result;
    result.reserve(4 + shape.size() * 8);
    append_little<std::uint32_t>(
        result, static_cast<std::uint32_t>(shape.size()));
    for (const auto dimension : shape) {
        append_little<std::int64_t>(result, dimension);
    }
    return result;
}

std::vector<std::uint8_t> mx_prefix(
    std::string_view dtype,
    const std::vector<std::int64_t>& logical_shape,
    const std::vector<std::int64_t>& storage_shape,
    const std::vector<std::int64_t>& scale_shape,
    const std::string& name) {
    if (logical_shape.size() != 2 || storage_shape.size() != 2 ||
        scale_shape.size() != 2) {
        throw std::runtime_error("native MX tensor must be rank two: " + name);
    }
    const auto rows = logical_shape[0];
    const auto columns = logical_shape[1];
    if (rows <= 0 || columns <= 0) {
        throw std::runtime_error("invalid native MX tensor shape: " + name);
    }
    std::vector<std::int64_t> expected_storage;
    std::vector<std::int64_t> expected_scales;
    std::uint8_t kind = 0;
    if (dtype == "MXFP4") {
        if (columns % 32 != 0) {
            throw std::runtime_error("MXFP4 columns are not divisible by 32: " + name);
        }
        kind = 4;
        expected_storage = {rows, columns / 2};
        expected_scales = {rows, columns / 32};
    } else {
        if (columns % 32 != 0) {
            throw std::runtime_error("MXFP8 columns are not divisible by 32: " + name);
        }
        kind = 8;
        expected_storage = {rows, columns};
        const std::vector<std::int64_t> block128_scales{
            (rows + 127) / 128, columns / 128};
        const std::vector<std::int64_t> block32_scales{
            (rows + 31) / 32, columns / 32};
        const std::vector<std::int64_t> row_scales{
            rows, columns / 32};
        if (storage_shape != expected_storage ||
            (scale_shape != block128_scales &&
             scale_shape != block32_scales &&
             scale_shape != row_scales)) {
            throw std::runtime_error(
                "invalid native MX storage geometry: " + name);
        }
        expected_scales = scale_shape;
    }
    if (storage_shape != expected_storage || scale_shape != expected_scales) {
        throw std::runtime_error("invalid native MX storage geometry: " + name);
    }
    std::vector<std::uint8_t> result;
    result.reserve(kMxHeaderBytes);
    result.insert(result.end(), {'M', 'X', 'T', '1'});
    result.push_back(1);
    result.push_back(kind);
    append_little<std::uint16_t>(result, 0);
    for (const auto dimension : logical_shape) {
        append_little<std::uint64_t>(
            result, static_cast<std::uint64_t>(dimension));
    }
    for (const auto dimension : storage_shape) {
        append_little<std::uint64_t>(
            result, static_cast<std::uint64_t>(dimension));
    }
    for (const auto dimension : scale_shape) {
        append_little<std::uint64_t>(
            result, static_cast<std::uint64_t>(dimension));
    }
    return result;
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

struct MfqContainer::RandomAccessFiles {
    struct File {
        explicit File(std::filesystem::path source)
            : path(std::move(source)), size(checked_file_size(path)) {}

        ~File() {
            if (descriptor >= 0) {
                ::close(descriptor);
            }
        }

        int open() const {
            std::scoped_lock lock(mutex);
            if (descriptor >= 0) {
                return descriptor;
            }
            descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (descriptor < 0) {
                throw std::runtime_error(
                    "cannot open MFQ record source: " + path.string() +
                    ": " + std::strerror(errno));
            }
#if defined(__APPLE__) && defined(F_NOCACHE)
            if (::fcntl(descriptor, F_NOCACHE, 1) != 0) {
                const auto error = errno;
                ::close(descriptor);
                descriptor = -1;
                throw std::runtime_error(
                    "cannot enable direct MFQ record reads: " + path.string() +
                    ": " + std::strerror(error));
            }
#endif
            return descriptor;
        }

        std::filesystem::path path;
        std::uint64_t size = 0;
        mutable std::mutex mutex;
        mutable int descriptor = -1;
    };

    void read(
        const std::filesystem::path& path,
        std::uint64_t offset,
        std::span<std::byte> destination) {
        std::shared_ptr<File> source;
        {
            std::scoped_lock lock(mutex);
            const auto found = files.find(path);
            if (found != files.end()) {
                source = found->second;
            } else {
                source = std::make_shared<File>(path);
                files.emplace(path, source);
            }
        }
        if (offset > source->size ||
            destination.size() > source->size - offset) {
            throw std::runtime_error(
                "MFQ record source was truncated: " + path.string());
        }
        std::size_t done = 0;
        while (done < destination.size()) {
            constexpr std::size_t maximum_read = std::size_t{32} << 20;
            const auto requested = std::min(
                destination.size() - done, maximum_read);
            const auto count = ::pread(
                source->open(),
                destination.data() + done,
                requested,
                static_cast<off_t>(offset + done));
            if (count < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(
                    "failed reading MFQ record source: " + path.string() +
                    ": " + std::strerror(errno));
            }
            if (count == 0) {
                throw std::runtime_error(
                    "MFQ record source was truncated: " + path.string());
            }
            done += static_cast<std::size_t>(count);
        }
    }

    std::mutex mutex;
    std::map<std::filesystem::path, std::shared_ptr<File>> files;
};

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

void MfqContainer::load_hf_directory(
    const std::filesystem::path& requested_path) {
    std::error_code error;
    const auto root = std::filesystem::canonical(requested_path, error);
    if (error || !std::filesystem::is_directory(root)) {
        throw std::runtime_error(
            "cannot open HF model directory: " + requested_path.string());
    }
    const auto config_path = root / "config.json";
    const auto config_text = read_file_text(config_path);
    json config;
    try {
        config = json::parse(config_text);
    } catch (const json::exception& exception) {
        throw std::runtime_error(
            "invalid HF config.json: " + std::string(exception.what()));
    }
    const auto model_type = config.find("model_type");
    if (model_type == config.end() || !model_type->is_string() ||
        model_type->get<std::string>().empty()) {
        throw std::runtime_error("HF config.json has no model_type");
    }

    hf_store_ = std::make_shared<HfSafetensorStore>(root, false);
    header_.version = 2;
    header_.architecture = model_type->get<std::string>() + "-hf-full-mfq";
    header_.extra_json.emplace("source.format", "hf-safetensors");
    header_.extra_json.emplace("source.precision", "native");
    const auto generation_path = root / "generation_config.json";
    if (std::filesystem::is_regular_file(generation_path)) {
        const auto generation_text = read_file_text(generation_path);
        json generation;
        try {
            generation = json::parse(generation_text);
        } catch (const json::exception& exception) {
            throw std::runtime_error(
                "invalid HF generation_config.json: " +
                std::string(exception.what()));
        }
        if (!generation.is_object()) {
            throw std::runtime_error(
                "HF generation_config.json must be an object");
        }
        static const std::array<std::pair<std::string_view, std::string_view>, 7>
            aliases{{
                {"max_new_tokens", "max_tokens"},
                {"temperature", "temperature"},
                {"top_k", "top_k"},
                {"top_p", "top_p"},
                {"presence_penalty", "presence_penalty"},
                {"frequency_penalty", "frequency_penalty"},
                {"repetition_penalty", "repetition_penalty"},
            }};
        json chat = json::object();
        for (const auto& [source, target] : aliases) {
            const auto found = generation.find(std::string(source));
            if (found != generation.end() && !found->is_null()) {
                chat[std::string(target)] = *found;
            }
        }
        if (!chat.empty()) {
            header_.extra_json.emplace(
                "runtime.sampling.v1",
                json({
                    {"schema", "mfq.runtime.sampling"},
                    {"version", 1},
                    {"chat", std::move(chat)},
                    {"provenance", {{"source", "hf:generation_config.json"}}},
                }).dump());
        }
    }
    for (std::size_t shard = 0; shard < hf_store_->shard_count(); ++shard) {
        source_paths_.push_back(hf_store_->shard_path(shard));
    }

    hf_assets_.emplace(
        std::string(kModelConfigAsset),
        std::vector<std::uint8_t>(config_text.begin(), config_text.end()));
    MfqRecord config_record;
    config_record.name = std::string(kModelConfigAsset);
    config_record.dtype = "BLOB";
    config_record.source_path = config_path;
    config_record.nbytes = config_text.size();
    records_.emplace(config_record.name, std::move(config_record));

    if (model_type->get<std::string>().rfind("minicpmo", 0) == 0) {
        std::filesystem::path position_path;
        if (const auto* configured = std::getenv(
                "MFQ_MINICPMO45_RESAMPLER_POSITION_ASSET");
            configured != nullptr && *configured != '\0') {
            position_path = configured;
        } else {
            position_path = root /
                "minicpmo45-resampler-pos-embed-v1.bf16";
        }
        if (std::filesystem::is_regular_file(position_path)) {
            const auto bytes = read_file_text(position_path);
            hf_assets_.emplace(
                std::string(kMinicpmoResamplerAsset),
                std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
            MfqRecord asset_record;
            asset_record.name = std::string(kMinicpmoResamplerAsset);
            asset_record.dtype = "BLOB";
            asset_record.source_path = std::filesystem::canonical(position_path);
            asset_record.nbytes = bytes.size();
            records_.emplace(asset_record.name, std::move(asset_record));
        }
    }

    std::unordered_set<std::string> consumed_scales;
    std::vector<std::string> names;
    names.reserve(hf_store_->tensors().size());
    for (const auto& [name, unused] : hf_store_->tensors()) {
        static_cast<void>(unused);
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    for (const auto& name : names) {
        const auto& values = hf_store_->tensor(name);
        if (values.dtype == "F8_E8M0") {
            continue;
        }

        HfVirtualRecord virtual_record;
        virtual_record.values_name = name;
        MfqRecord record;
        record.name = name;
        record.source_path = hf_store_->shard_path(values.shard);
        const auto item_size = dense_item_size(values.dtype);
        if (item_size != 0) {
            if (checked_product(values.shape, item_size, name) != values.nbytes) {
                throw std::runtime_error(
                    "dense Safetensors byte size mismatch: " + name);
            }
            record.dtype = values.dtype;
            virtual_record.prefix = dense_prefix(values.shape);
            virtual_record.values_offset = virtual_record.prefix.size();
            record.nbytes = checked_add(
                virtual_record.values_offset,
                values.nbytes,
                "dense tensor");
        } else if (values.dtype == "I8" ||
                   values.dtype == "F8_E4M3" ||
                   values.dtype == "F8_E4M3FN") {
            if (!name.ends_with(".weight")) {
                throw std::runtime_error(
                    "native MX tensor is not named *.weight: " + name);
            }
            const auto scale_name =
                name.substr(0, name.size() - std::string_view(".weight").size()) +
                ".scale";
            const auto& scales = hf_store_->tensor(scale_name);
            if (scales.dtype != "F8_E8M0") {
                throw std::runtime_error(
                    "native MX scale is not F8_E8M0: " + scale_name);
            }
            std::vector<std::int64_t> logical_shape = values.shape;
            if (values.dtype == "I8") {
                if (logical_shape.size() != 2 ||
                    logical_shape[1] >
                        std::numeric_limits<std::int64_t>::max() / 2) {
                    throw std::runtime_error(
                        "invalid native MXFP4 tensor shape: " + name);
                }
                logical_shape[1] *= 2;
                record.dtype = "MXFP4";
            } else {
                record.dtype = "MXFP8";
            }
            virtual_record.scales_name = scale_name;
            virtual_record.prefix = mx_prefix(
                record.dtype,
                logical_shape,
                values.shape,
                scales.shape,
                name);
            virtual_record.values_offset = virtual_record.prefix.size();
            virtual_record.scales_offset = checked_add(
                virtual_record.values_offset,
                values.nbytes,
                "native MX tensor");
            record.nbytes = checked_add(
                virtual_record.scales_offset,
                scales.nbytes,
                "native MX tensor");
            consumed_scales.insert(scale_name);
        } else {
            throw std::runtime_error(
                "unsupported full-precision HF dtype " + values.dtype +
                ": " + name);
        }
        if (!records_.emplace(name, std::move(record)).second ||
            !hf_records_.emplace(name, std::move(virtual_record)).second) {
            throw std::runtime_error("duplicate HF tensor: " + name);
        }
    }
    for (const auto& name : names) {
        const auto& record = hf_store_->tensor(name);
        if (record.dtype == "F8_E8M0" &&
            consumed_scales.find(name) == consumed_scales.end()) {
            throw std::runtime_error("orphan E8M0 scale tensor: " + name);
        }
    }
    if (records_.size() > kMaxRecordEntries) {
        throw std::runtime_error("HF checkpoint has too many tensor records");
    }
    header_.record_count = static_cast<std::uint32_t>(records_.size());
}

MfqContainer::MfqContainer(std::filesystem::path path)
    : random_access_files_(std::make_shared<RandomAccessFiles>()) {
    std::error_code directory_error;
    if (std::filesystem::is_directory(path, directory_error) &&
        !directory_error) {
        load_hf_directory(path);
        install_legacy_tensor_compatibility(*this);
        return;
    }
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
        install_legacy_tensor_compatibility(*this);
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
        if (index == 0) {
            // Only shard zero owns global metadata. Canonicalize the header so
            // starting from any shard yields the same runtime profile/assets.
            header_ = current;
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
    install_legacy_tensor_compatibility(*this);
}

bool MfqContainer::contains(const std::string& name) const {
    if (records_.find(name) != records_.end()) {
        return true;
    }
    const auto alias = legacy_aliases_.find(name);
    return alias != legacy_aliases_.end() &&
        records_.find(alias->second) != records_.end();
}

const MfqRecord& MfqContainer::record(const std::string& name) const {
    auto found = records_.find(name);
    if (found == records_.end()) {
        const auto alias = legacy_aliases_.find(name);
        if (alias != legacy_aliases_.end()) {
            found = records_.find(alias->second);
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
    if (hf_store_) {
        auto bytes = std::make_shared<std::vector<std::uint8_t>>(read(name));
        const auto* data = bytes->data();
        const auto size = bytes->size();
        std::shared_ptr<void> owner = bytes;
        return MfqMappedBytes(std::move(owner), data, size);
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
    if (nbytes > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            "MFQ record byte range is too large: " + name);
    }
    std::vector<std::uint8_t> result(static_cast<std::size_t>(nbytes));
    read_range_into(
        name,
        relative_offset,
        std::as_writable_bytes(std::span<std::uint8_t>(result)));
    return result;
}

void MfqContainer::read_range_into(
    const std::string& name,
    std::uint64_t relative_offset,
    std::span<std::byte> destination) const {
    const auto& value = record(name);
    const auto nbytes = static_cast<std::uint64_t>(destination.size());
    if (
        relative_offset > value.nbytes
        || nbytes > value.nbytes - relative_offset
    ) {
        throw std::out_of_range(
            "MFQ record byte range is out of bounds: " + name);
    }
    if (nbytes >
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
        return;
    }
    if (hf_store_) {
        read_hf_range_into(value.name, relative_offset, destination);
        return;
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
    random_access_files_->read(
        value.source_path, absolute_offset, destination);
}

std::vector<std::uint8_t> MfqContainer::read_hf_range(
    const std::string& name,
    std::uint64_t relative_offset,
    std::uint64_t nbytes) const {
    std::vector<std::uint8_t> result(static_cast<std::size_t>(nbytes));
    read_hf_range_into(
        name,
        relative_offset,
        std::as_writable_bytes(std::span<std::uint8_t>(result)));
    return result;
}

void MfqContainer::read_hf_range_into(
    const std::string& name,
    std::uint64_t relative_offset,
    std::span<std::byte> destination) const {
    const auto nbytes = static_cast<std::uint64_t>(destination.size());
    if (const auto asset = hf_assets_.find(name); asset != hf_assets_.end()) {
        std::copy_n(
            asset->second.begin() + static_cast<std::ptrdiff_t>(relative_offset),
            static_cast<std::size_t>(nbytes),
            reinterpret_cast<std::uint8_t*>(destination.data()));
        return;
    }
    const auto found = hf_records_.find(name);
    if (found == hf_records_.end()) {
        throw std::runtime_error("missing virtual HF record: " + name);
    }
    const auto& logical = found->second;
    const auto request_end = relative_offset + nbytes;
    const auto copy_memory = [&](std::uint64_t logical_offset,
                                 std::span<const std::uint8_t> source) {
        const auto segment_end = logical_offset + source.size();
        const auto begin = std::max(relative_offset, logical_offset);
        const auto end = std::min(request_end, segment_end);
        if (begin < end) {
            std::copy_n(
                source.begin() + static_cast<std::ptrdiff_t>(begin - logical_offset),
                static_cast<std::size_t>(end - begin),
                reinterpret_cast<std::uint8_t*>(destination.data()) +
                    static_cast<std::ptrdiff_t>(begin - relative_offset));
        }
    };
    const auto copy_tensor = [&](std::uint64_t logical_offset,
                                 const std::string& tensor_name,
                                 std::uint64_t tensor_offset = 0,
                                 std::uint64_t tensor_nbytes = 0) {
        const auto& source = hf_store_->tensor(tensor_name);
        const auto source_bytes = tensor_nbytes == 0
            ? source.nbytes - tensor_offset
            : tensor_nbytes;
        if (tensor_offset > source.nbytes ||
            source_bytes > source.nbytes - tensor_offset) {
            throw std::runtime_error(
                "virtual HF tensor segment is out of bounds: " + name);
        }
        const auto segment_end = checked_add(
            logical_offset, source_bytes, "virtual HF segment");
        const auto begin = std::max(relative_offset, logical_offset);
        const auto end = std::min(request_end, segment_end);
        if (begin >= end) {
            return;
        }
        auto target = destination.subspan(
            static_cast<std::size_t>(begin - relative_offset),
            static_cast<std::size_t>(end - begin));
        hf_store_->read_range(
            source.shard,
            source.offset + tensor_offset + begin - logical_offset,
            target);
    };
    if (!logical.segments.empty()) {
        const auto segment_size = [](const HfVirtualRecord::Segment& segment) {
            return segment.inline_bytes.empty()
                ? segment.nbytes
                : static_cast<std::uint64_t>(segment.inline_bytes.size());
        };
        const auto first = std::lower_bound(
            logical.segments.begin(),
            logical.segments.end(),
            relative_offset,
            [&](const HfVirtualRecord::Segment& segment,
                std::uint64_t offset) {
                return checked_add(
                    segment.logical_offset,
                    segment_size(segment),
                    "virtual HF segment") <= offset;
            });
        for (auto item = first; item != logical.segments.end(); ++item) {
            const auto& segment = *item;
            if (segment.logical_offset >= request_end) break;
            if (!segment.inline_bytes.empty()) {
                copy_memory(segment.logical_offset, segment.inline_bytes);
            } else if (!segment.tensor_name.empty()) {
                copy_tensor(
                    segment.logical_offset,
                    segment.tensor_name,
                    segment.tensor_offset,
                    segment.nbytes);
            } else if (segment.nbytes != 0) {
                throw std::runtime_error(
                    "virtual HF segment has no backing storage: " + name);
            }
        }
        return;
    }
    copy_memory(0, logical.prefix);
    copy_tensor(logical.values_offset, logical.values_name);
    if (!logical.scales_name.empty()) {
        copy_tensor(logical.scales_offset, logical.scales_name);
    }
}

std::string MfqContainer::read_text(const std::string& name) const {
    const auto bytes = read(name);
    return std::string(bytes.begin(), bytes.end());
}

std::optional<mfq::MfqModelGraph> MfqContainer::model_graph() const {
    const std::string asset(mfq::kMfqModelGraphAsset);
    if (!contains(asset)) return std::nullopt;
    return mfq::MfqModelGraph::from_json(read_text(asset));
}

void MfqContainer::install_legacy_aliases(
        std::unordered_map<std::string, std::string> canonical_to_stored,
        mfq::MfqLegacyTensorLayout layout) {
    if (model_graph()) {
        throw std::invalid_argument(
            "canonical MFQ model graphs cannot install legacy tensor aliases");
    }
    for (const auto& [canonical, stored] : canonical_to_stored) {
        if (canonical.empty() || stored.empty() || canonical == stored) {
            throw std::invalid_argument("invalid legacy MFQ tensor alias");
        }
        if (records_.find(canonical) != records_.end()) {
            throw std::invalid_argument(
                "legacy MFQ alias shadows a stored canonical tensor: " + canonical);
        }
        if (records_.find(stored) == records_.end()) {
            throw std::invalid_argument(
                "legacy MFQ alias refers to a missing tensor: " + stored);
        }
        if (canonical_to_stored.find(stored) != canonical_to_stored.end()) {
            throw std::invalid_argument(
                "legacy MFQ tensor alias chains are forbidden: " + canonical);
        }
        const auto existing = legacy_aliases_.find(canonical);
        if (existing != legacy_aliases_.end() && existing->second != stored) {
            throw std::invalid_argument(
                "legacy MFQ tensor alias is ambiguous: " + canonical);
        }
    }
    legacy_aliases_.insert(
        std::make_move_iterator(canonical_to_stored.begin()),
        std::make_move_iterator(canonical_to_stored.end()));
    legacy_tensor_layout_ = layout;
}

void MfqContainer::install_hf_nintm_views(
    const std::unordered_map<std::string, std::string>&
        canonical_to_stored) {
    if (!hf_store_) {
        return;
    }

    struct Projection {
        std::map<std::size_t, std::string> experts;
    };
    std::map<std::string, Projection> projections;
    const auto collect = [&](std::string_view canonical,
                             const std::string& stored) {
        constexpr std::string_view marker = ".mlp.experts.";
        const auto marker_offset = canonical.find(marker);
        if (marker_offset == std::string_view::npos) {
            return;
        }
        const auto expert_begin = marker_offset + marker.size();
        const auto expert_end = canonical.find('.', expert_begin);
        if (expert_end == std::string_view::npos ||
            expert_end == expert_begin) {
            return;
        }
        std::size_t expert = 0;
        const auto id = canonical.substr(
            expert_begin, expert_end - expert_begin);
        const auto parsed = std::from_chars(
            id.data(), id.data() + id.size(), expert);
        if (parsed.ec != std::errc{} || parsed.ptr != id.data() + id.size()) {
            return;
        }
        const auto projection_end = canonical.find('.', expert_end + 1);
        if (projection_end == std::string_view::npos ||
            canonical.substr(projection_end) != ".weight") {
            return;
        }
        const auto projection = canonical.substr(
            expert_end + 1,
            projection_end - expert_end - 1);
        if (projection != "gate" && projection != "up" &&
            projection != "down") {
            return;
        }
        const auto aggregate = std::string(canonical.substr(0, marker_offset)) +
            std::string(marker) + std::string(projection) + ".weight";
        projections[aggregate].experts.emplace(expert, stored);
    };
    for (const auto& [canonical, stored] : canonical_to_stored) {
        collect(canonical, stored);
    }
    for (const auto& [name, _record] : records_) {
        collect(name, name);
    }

    for (const auto& [name, projection] : projections) {
        if (records_.find(name) != records_.end() ||
            projection.experts.empty()) {
            continue;
        }
        const auto expert_count = projection.experts.rbegin()->first + 1;
        if (projection.experts.size() != expert_count) {
            throw std::runtime_error(
                "native HF expert projection has missing expert IDs: " + name);
        }

        std::string dtype;
        std::uint64_t output = 0;
        std::uint64_t input = 0;
        std::uint64_t logical_offset = 0;
        HfVirtualRecord aggregate;
        const auto append_inline = [&](std::vector<std::uint8_t> bytes) {
            if (bytes.empty()) return;
            const auto count = static_cast<std::uint64_t>(bytes.size());
            aggregate.segments.push_back({
                .logical_offset = logical_offset,
                .inline_bytes = std::move(bytes),
                .nbytes = count,
            });
            logical_offset = checked_add(
                logical_offset, count, "virtual NINTM metadata");
        };
        const auto append_tensor = [&](const std::string& tensor_name) {
            const auto& tensor = hf_store_->tensor(tensor_name);
            aggregate.segments.push_back({
                .logical_offset = logical_offset,
                .tensor_name = tensor_name,
                .nbytes = tensor.nbytes,
            });
            logical_offset = checked_add(
                logical_offset, tensor.nbytes, "virtual NINTM tensor");
        };

        std::vector<std::uint8_t> header;
        header.insert(header.end(), {'N', 'I', 'M', '2'});
        append_little<std::uint32_t>(
            header, static_cast<std::uint32_t>(expert_count));
        // Output/input dimensions are filled after validating the first
        // native expert and patched before the segment is published.
        append_little<std::uint32_t>(header, 0);
        append_little<std::uint32_t>(header, 0);
        append_little<std::uint32_t>(
            header, static_cast<std::uint32_t>(expert_count));

        for (const auto& [expert, stored] : projection.experts) {
            const auto record = records_.find(stored);
            const auto virtual_record = hf_records_.find(stored);
            if (record == records_.end() ||
                virtual_record == hf_records_.end() ||
                (record->second.dtype != "MXFP4" &&
                 record->second.dtype != "MXFP8")) {
                throw std::runtime_error(
                    "native HF expert is not an MX tensor: " + stored);
            }
            const auto& source = virtual_record->second;
            if (!source.segments.empty() || source.prefix.size() != kMxHeaderBytes ||
                std::memcmp(source.prefix.data(), "MXT1", 4) != 0 ||
                source.values_name.empty() || source.scales_name.empty()) {
                throw std::runtime_error(
                    "invalid native HF MX expert view: " + stored);
            }
            std::uint64_t source_output = 0;
            std::uint64_t source_input = 0;
            std::memcpy(&source_output, source.prefix.data() + 8, sizeof(source_output));
            std::memcpy(&source_input, source.prefix.data() + 16, sizeof(source_input));
            if (source_output == 0 || source_input == 0 ||
                source_output > std::numeric_limits<std::uint32_t>::max() ||
                source_input > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "native HF expert geometry exceeds NINTM: " + stored);
            }
            if (dtype.empty()) {
                dtype = record->second.dtype;
                output = source_output;
                input = source_input;
                const auto output32 = static_cast<std::uint32_t>(output);
                const auto input32 = static_cast<std::uint32_t>(input);
                std::memcpy(
                    header.data() + 8, &output32, sizeof(output32));
                std::memcpy(
                    header.data() + 12, &input32, sizeof(input32));
                append_inline(std::move(header));
            } else if (dtype != record->second.dtype || output != source_output ||
                       input != source_input) {
                throw std::runtime_error(
                    "native HF expert projection has mixed geometry: " + name);
            }

            std::vector<std::uint8_t> pool;
            append_little<std::uint32_t>(pool, 1);
            append_little<std::uint32_t>(
                pool, static_cast<std::uint32_t>(dtype.size()));
            append_little<std::uint64_t>(pool, record->second.nbytes);
            append_little<std::uint64_t>(pool, 0);
            append_little<std::int32_t>(
                pool, static_cast<std::int32_t>(expert));
            pool.insert(pool.end(), dtype.begin(), dtype.end());
            pool.insert(pool.end(), source.prefix.begin(), source.prefix.end());
            append_inline(std::move(pool));
            append_tensor(source.values_name);
            append_tensor(source.scales_name);
        }

        MfqRecord record;
        record.name = name;
        record.dtype = "NINTM";
        record.source_path = hf_store_->root();
        record.nbytes = logical_offset;
        if (!records_.emplace(name, std::move(record)).second ||
            !hf_records_.emplace(name, std::move(aggregate)).second) {
            throw std::runtime_error(
                "duplicate virtual native NINTM projection: " + name);
        }
    }
    if (records_.size() > kMaxRecordEntries) {
        throw std::runtime_error(
            "HF virtual NINTM projections exceed the record limit");
    }
    header_.record_count = static_cast<std::uint32_t>(records_.size());
}

} // namespace mfq::metal
