#include "mfq_container.h"

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
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
    if (shape.empty()) {
        throw std::runtime_error("Safetensors tensor has empty shape: " + name);
    }
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
        if (columns % 128 != 0) {
            throw std::runtime_error("MXFP8 columns are not divisible by 128: " + name);
        }
        kind = 8;
        expected_storage = {rows, columns};
        expected_scales = {(rows + 127) / 128, columns / 128};
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
            {"ffn.experts.gate.weight", "ffn_gate_exps.weight"},
            {"ffn.experts.up.weight", "ffn_up_exps.weight"},
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
            if (values.shard != scales.shard) {
                throw std::runtime_error(
                    "native MX weight and scale are in different shards: " + name);
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

MfqContainer::MfqContainer(std::filesystem::path path) {
    std::error_code directory_error;
    if (std::filesystem::is_directory(path, directory_error) &&
        !directory_error) {
        load_hf_directory(path);
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
    if (hf_store_) {
        return read_hf_range(name, relative_offset, nbytes);
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

std::vector<std::uint8_t> MfqContainer::read_hf_range(
    const std::string& name,
    std::uint64_t relative_offset,
    std::uint64_t nbytes) const {
    std::vector<std::uint8_t> result(static_cast<std::size_t>(nbytes));
    if (const auto asset = hf_assets_.find(name); asset != hf_assets_.end()) {
        std::copy_n(
            asset->second.begin() + static_cast<std::ptrdiff_t>(relative_offset),
            static_cast<std::size_t>(nbytes),
            result.begin());
        return result;
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
                result.begin() + static_cast<std::ptrdiff_t>(begin - relative_offset));
        }
    };
    const auto copy_tensor = [&](std::uint64_t logical_offset,
                                 const std::string& tensor_name) {
        const auto& source = hf_store_->tensor(tensor_name);
        const auto segment_end = logical_offset + source.nbytes;
        const auto begin = std::max(relative_offset, logical_offset);
        const auto end = std::min(request_end, segment_end);
        if (begin >= end) {
            return;
        }
        auto destination = std::span<std::uint8_t>(result).subspan(
            static_cast<std::size_t>(begin - relative_offset),
            static_cast<std::size_t>(end - begin));
        hf_store_->read_range(
            source.shard,
            source.offset + begin - logical_offset,
            std::as_writable_bytes(destination));
    };
    copy_memory(0, logical.prefix);
    copy_tensor(logical.values_offset, logical.values_name);
    if (!logical.scales_name.empty()) {
        copy_tensor(logical.scales_offset, logical.scales_name);
    }
    return result;
}

std::string MfqContainer::read_text(const std::string& name) const {
    const auto bytes = read(name);
    return std::string(bytes.begin(), bytes.end());
}

} // namespace mfq::metal
