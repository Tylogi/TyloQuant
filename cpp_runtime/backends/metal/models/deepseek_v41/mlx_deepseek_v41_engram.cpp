#include "mlx_deepseek_v41_engram.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

constexpr std::string_view kEngramAsset =
    "__mfq_asset__/deepseek-v41-engram-v1.bin";
constexpr std::array<std::uint8_t, 8> kEngramMagic{
    'M', 'F', 'Q', 'E', 'N', 'G', 'R', '1'};
constexpr std::size_t kMxHeaderBytes = 56;
constexpr std::int64_t kDeadToken = -1;

class ByteCursor {
public:
    explicit ByteCursor(std::span<const std::uint8_t> bytes)
        : bytes_(bytes) {}

    std::span<const std::uint8_t> take(
        std::size_t count,
        std::string_view description) {
        if (count > bytes_.size() - offset_) {
            throw std::runtime_error(
                "truncated DeepSeek-V4.1 Engram " +
                std::string(description));
        }
        auto result = bytes_.subspan(offset_, count);
        offset_ += count;
        return result;
    }

    std::uint32_t u32(std::string_view description) {
        const auto bytes = take(4, description);
        std::uint32_t result = 0;
        for (int index = 0; index < 4; ++index) {
            result |= static_cast<std::uint32_t>(bytes[index]) << (8 * index);
        }
        return result;
    }

    std::uint64_t u64(std::string_view description) {
        const auto bytes = take(8, description);
        std::uint64_t result = 0;
        for (int index = 0; index < 8; ++index) {
            result |= static_cast<std::uint64_t>(bytes[index]) << (8 * index);
        }
        return result;
    }

    std::int32_t i32(std::string_view description) {
        const auto raw = u32(description);
        std::int32_t result = 0;
        std::memcpy(&result, &raw, sizeof(result));
        return result;
    }

    std::int64_t i64(std::string_view description) {
        const auto raw = u64(description);
        std::int64_t result = 0;
        std::memcpy(&result, &raw, sizeof(result));
        return result;
    }

    void require_end() const {
        if (offset_ != bytes_.size()) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram asset has trailing bytes");
        }
    }

private:
    std::span<const std::uint8_t> bytes_;
    std::size_t offset_ = 0;
};

template <typename T, typename Reader>
std::vector<T> read_vector(
    ByteCursor& cursor,
    std::size_t count,
    Reader&& reader,
    std::string_view description) {
    std::vector<T> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        result.push_back(reader(cursor, description));
    }
    return result;
}

array dense_array(
    const MfqContainer& model,
    const std::string& name,
    mlx::core::Dtype dtype = mlx::core::float32) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" && record.dtype != "F16" &&
        record.dtype != "F32") {
        throw std::runtime_error(
            "DeepSeek-V4.1 Engram requires dense control tensor " + name);
    }
    const auto mapped = model.map_record(name);
    auto value = load_dense_array(record.dtype, mapped.view());
    if (value.dtype() != dtype) value = mlx::core::astype(value, dtype);
    return mlx::core::contiguous(std::move(value));
}

float decode_e4m3(std::uint8_t raw) {
    if ((raw & 0x7fu) == 0x7fu) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float sign = (raw & 0x80u) ? -1.0f : 1.0f;
    const int exponent = (raw >> 3u) & 0x0fu;
    const int mantissa = raw & 0x07u;
    return sign * (exponent == 0
        ? std::ldexp(static_cast<float>(mantissa), -9)
        : std::ldexp(
              1.0f + static_cast<float>(mantissa) / 8.0f,
              exponent - 7));
}

std::size_t cache_capacity_rows() {
    constexpr std::size_t fallback = 16'384;
    const char* raw = std::getenv("MFQ_DEEPSEEK_V41_ENGRAM_CACHE_ROWS");
    if (raw == nullptr || *raw == '\0') return fallback;
    char* end = nullptr;
    const auto parsed = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "invalid MFQ_DEEPSEEK_V41_ENGRAM_CACHE_ROWS");
    }
    return static_cast<std::size_t>(parsed);
}

std::uint64_t checked_product(
    std::uint64_t left,
    std::uint64_t right,
    std::string_view description) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::runtime_error(
            "DeepSeek-V4.1 Engram " + std::string(description) +
            " byte size overflows");
    }
    return left * right;
}

std::uint64_t read_u64_at(
    std::span<const std::uint8_t> bytes,
    std::size_t offset) {
    if (offset > bytes.size() || 8 > bytes.size() - offset) {
        throw std::runtime_error("truncated DeepSeek-V4.1 Engram MX header");
    }
    std::uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(bytes[offset + index]) << (8 * index);
    }
    return value;
}

} // namespace

std::span<const std::int64_t> DeepseekV41EngramHashBatch::layer(
    int index) const {
    if (index < 0 || index >= layers || batch <= 0 || tokens <= 0 ||
        hash_columns <= 0) {
        throw std::out_of_range("invalid DeepSeek-V4.1 Engram hash layer");
    }
    const auto layer_size = static_cast<std::size_t>(batch) *
        static_cast<std::size_t>(tokens) *
        static_cast<std::size_t>(hash_columns);
    const auto begin = static_cast<std::size_t>(index) * layer_size;
    if (begin > values.size() || layer_size > values.size() - begin) {
        throw std::runtime_error("truncated DeepSeek-V4.1 Engram hash batch");
    }
    return std::span<const std::int64_t>(values).subspan(begin, layer_size);
}

MlxDeepseekV41EngramHashState MlxDeepseekV41EngramHashState::load(
    const MfqContainer& model,
    const DeepseekV41Config& config) {
    if (!model.contains(std::string(kEngramAsset))) {
        throw std::runtime_error(
            "DeepSeek-V4.1 MFQ has no embedded Engram hash asset");
    }
    const auto asset = model.read(std::string(kEngramAsset));
    ByteCursor cursor(asset);
    const auto magic = cursor.take(kEngramMagic.size(), "asset magic");
    if (!std::equal(magic.begin(), magic.end(), kEngramMagic.begin())) {
        throw std::runtime_error("invalid DeepSeek-V4.1 Engram asset magic");
    }
    const auto vocabulary = cursor.u32("vocabulary");
    const auto compressed = cursor.u32("compressed vocabulary");
    const auto layer_count = cursor.u32("layer count");
    const auto max_ngram = cursor.u32("maximum ngram size");
    const auto heads = cursor.u32("head count");
    const auto pad_id = cursor.i32("compressed pad token");
    if (vocabulary > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        compressed > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        layer_count == 0 ||
        layer_count > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        max_ngram < 2 ||
        max_ngram > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        heads == 0 ||
        heads > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("invalid DeepSeek-V4.1 Engram asset dimensions");
    }
    const auto layers = static_cast<std::size_t>(layer_count);
    const auto orders = static_cast<std::size_t>(max_ngram - 1);
    const auto head_count = static_cast<std::size_t>(heads);
    const auto bucket_count = checked_product(
        checked_product(layers, orders, "bucket"),
        head_count,
        "bucket");
    const auto multiplier_count = checked_product(
        layers,
        static_cast<std::size_t>(max_ngram),
        "multiplier");
    auto layer_ids = read_vector<std::int64_t>(
        cursor, layers,
        [](ByteCursor& value, std::string_view what) {
            return static_cast<std::int64_t>(value.i32(what));
        },
        "layer ids");
    auto table_rows = read_vector<std::int64_t>(
        cursor, layers,
        [](ByteCursor& value, std::string_view what) {
            return value.i64(what);
        },
        "table rows");
    auto primes = read_vector<std::int64_t>(
        cursor, static_cast<std::size_t>(bucket_count),
        [](ByteCursor& value, std::string_view what) {
            return value.i64(what);
        },
        "bucket primes");
    auto offsets = read_vector<std::int64_t>(
        cursor, static_cast<std::size_t>(bucket_count),
        [](ByteCursor& value, std::string_view what) {
            return value.i64(what);
        },
        "bucket offsets");
    auto multipliers = read_vector<std::int64_t>(
        cursor, static_cast<std::size_t>(multiplier_count),
        [](ByteCursor& value, std::string_view what) {
            return value.i64(what);
        },
        "hash multipliers");
    auto token_map = read_vector<std::int32_t>(
        cursor, vocabulary,
        [](ByteCursor& value, std::string_view what) {
            return value.i32(what);
        },
        "compressed token map");
    cursor.require_end();

    if (vocabulary != static_cast<std::uint32_t>(config.vocab) ||
        compressed != static_cast<std::uint32_t>(config.engram_compressed_vocab_size) ||
        max_ngram != static_cast<std::uint32_t>(config.engram_max_ngram_size) ||
        heads != static_cast<std::uint32_t>(config.engram_n_heads) ||
        layer_ids != config.engram_layer_ids ||
        table_rows != config.engram_num_embeddings ||
        pad_id < 0 || pad_id >= static_cast<std::int64_t>(compressed)) {
        throw std::runtime_error(
            "DeepSeek-V4.1 Engram asset disagrees with model config");
    }
    for (std::size_t layer = 0; layer < layers; ++layer) {
        const auto last = layer * orders * head_count + orders * head_count - 1;
        if (primes[last] <= 0 || offsets[last] < 0 ||
            primes[last] + offsets[last] != table_rows[layer]) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram bucket layout disagrees with table rows");
        }
    }
    if (std::any_of(
            token_map.begin(), token_map.end(),
            [compressed](std::int32_t value) {
                return value < 0 ||
                    value >= static_cast<std::int32_t>(compressed);
            }) ||
        std::any_of(
            multipliers.begin(), multipliers.end(),
            [](std::int64_t value) { return value <= 0 || !(value & 1); })) {
        throw std::runtime_error(
            "DeepSeek-V4.1 Engram hash metadata is invalid");
    }
    return MlxDeepseekV41EngramHashState(
        static_cast<int>(vocabulary),
        static_cast<int>(compressed),
        static_cast<int>(max_ngram),
        static_cast<int>(heads),
        pad_id,
        std::move(layer_ids),
        std::move(table_rows),
        std::move(primes),
        std::move(offsets),
        std::move(multipliers),
        std::move(token_map));
}

MlxDeepseekV41EngramHashState::MlxDeepseekV41EngramHashState(
    int vocabulary,
    int compressed_vocabulary,
    int max_ngram_size,
    int heads,
    std::int64_t pad_id,
    std::vector<std::int64_t> layer_ids,
    std::vector<std::int64_t> table_rows,
    std::vector<std::int64_t> primes,
    std::vector<std::int64_t> offsets,
    std::vector<std::int64_t> multipliers,
    std::vector<std::int32_t> token_map)
    : vocabulary_(vocabulary),
      compressed_vocabulary_(compressed_vocabulary),
      max_ngram_size_(max_ngram_size),
      heads_(heads),
      pad_id_(pad_id),
      layer_ids_(std::move(layer_ids)),
      table_rows_(std::move(table_rows)),
      primes_(std::move(primes)),
      offsets_(std::move(offsets)),
      multipliers_(std::move(multipliers)),
      token_map_(std::move(token_map)) {}

void MlxDeepseekV41EngramHashState::reset(int batch) {
    if (batch <= 0) {
        throw std::invalid_argument("DeepSeek-V4.1 Engram batch must be positive");
    }
    batch_ = batch;
    position_ = 0;
    context_.assign(
        static_cast<std::size_t>(batch * (max_ngram_size_ - 1)),
        pad_id_);
}

void MlxDeepseekV41EngramHashState::clear() noexcept {
    batch_ = 0;
    position_ = 0;
    context_.clear();
}

DeepseekV41EngramHashBatch MlxDeepseekV41EngramHashState::forward(
    const array& token_ids,
    int pos0,
    const std::optional<array>& participation_mask,
    bool use_cache) {
    if (token_ids.ndim() != 2 || token_ids.shape(0) <= 0 ||
        token_ids.shape(1) <= 0 || pos0 < 0) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 Engram token IDs must have nonempty [B,T] shape");
    }
    const int batch = token_ids.shape(0);
    const int tokens = token_ids.shape(1);
    if (!use_cache && pos0 != 0) {
        throw std::invalid_argument(
            "stateless DeepSeek-V4.1 Engram hashing must begin at position zero");
    }
    if (use_cache) {
        if (pos0 == 0) reset(batch);
        if (batch_ != batch || position_ != pos0 || context_.empty()) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram cache position disagrees");
        }
    }

    const bool ids_i32 = token_ids.dtype() == mlx::core::int32;
    auto host_ids = ids_i32 || token_ids.dtype() == mlx::core::int64
        ? mlx::core::contiguous(token_ids)
        : mlx::core::contiguous(
              mlx::core::astype(token_ids, mlx::core::int64));
    host_ids.eval();
    const auto* ids32 = ids_i32 ? host_ids.data<std::int32_t>() : nullptr;
    const auto* ids64 = ids_i32 ? nullptr : host_ids.data<std::int64_t>();

    std::vector<std::uint8_t> participates(
        static_cast<std::size_t>(batch * tokens), 1);
    if (participation_mask) {
        if (participation_mask->shape() != token_ids.shape()) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 Engram participation mask shape disagrees");
        }
        auto host_mask = mlx::core::contiguous(
            mlx::core::astype(*participation_mask, mlx::core::bool_));
        host_mask.eval();
        const auto* source = host_mask.data<bool>();
        std::transform(
            source,
            source + participates.size(),
            participates.begin(),
            [](bool value) { return static_cast<std::uint8_t>(value); });
    }

    std::vector<std::int64_t> compressed(
        static_cast<std::size_t>(batch * tokens));
    for (std::size_t index = 0; index < compressed.size(); ++index) {
        const auto token = ids_i32
            ? static_cast<std::int64_t>(ids32[index])
            : ids64[index];
        if (token < 0 || token >= vocabulary_) {
            throw std::out_of_range(
                "DeepSeek-V4.1 Engram token ID is outside the vocabulary");
        }
        compressed[index] = participates[index]
            ? static_cast<std::int64_t>(token_map_[static_cast<std::size_t>(token)])
            : kDeadToken;
    }

    const int prefix = max_ngram_size_ - 1;
    const int combined_length = prefix + tokens;
    const int layers = static_cast<int>(layer_ids_.size());
    const int columns = hash_columns();
    DeepseekV41EngramHashBatch result;
    result.batch = batch;
    result.tokens = tokens;
    result.layers = layers;
    result.hash_columns = columns;
    result.values.resize(
        static_cast<std::size_t>(layers) * batch * tokens * columns);
    std::vector<std::int64_t> history(
        static_cast<std::size_t>(batch * combined_length), pad_id_);

    for (int bi = 0; bi < batch; ++bi) {
        auto* row = history.data() + static_cast<std::size_t>(bi * combined_length);
        if (use_cache) {
            std::copy_n(
                context_.data() + static_cast<std::size_t>(bi * prefix),
                prefix,
                row);
        }
        std::copy_n(
            compressed.data() + static_cast<std::size_t>(bi * tokens),
            tokens,
            row + prefix);
        for (int token = 0; token < tokens; ++token) {
            std::array<std::int64_t, 32> ngram_tokens{};
            if (max_ngram_size_ > static_cast<int>(ngram_tokens.size())) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 Engram ngram size exceeds native bound");
            }
            bool blocked = false;
            for (int shift = 0; shift < max_ngram_size_; ++shift) {
                const auto source = row[prefix + token - shift];
                blocked = blocked || source == kDeadToken;
                ngram_tokens[static_cast<std::size_t>(shift)] =
                    blocked ? pad_id_ : source;
            }
            for (int layer = 0; layer < layers; ++layer) {
                const auto multiplier_base =
                    static_cast<std::size_t>(layer * max_ngram_size_);
                std::uint64_t rolling =
                    static_cast<std::uint64_t>(ngram_tokens[0]) *
                    static_cast<std::uint64_t>(multipliers_[multiplier_base]);
                for (int shift = 1; shift < max_ngram_size_; ++shift) {
                    rolling ^=
                        static_cast<std::uint64_t>(ngram_tokens[shift]) *
                        static_cast<std::uint64_t>(
                            multipliers_[multiplier_base + shift]);
                    std::int64_t signed_hash = 0;
                    std::memcpy(&signed_hash, &rolling, sizeof(signed_hash));
                    for (int head = 0; head < heads_; ++head) {
                        const int column = (shift - 1) * heads_ + head;
                        const auto bucket = static_cast<std::size_t>(
                            layer * columns + column);
                        const auto prime = primes_[bucket];
                        auto remainder = signed_hash % prime;
                        if (remainder < 0) remainder += prime;
                        const auto hash = remainder + offsets_[bucket];
                        if (hash < 0 || hash >= table_rows_[layer]) {
                            throw std::runtime_error(
                                "DeepSeek-V4.1 Engram hash exceeds table rows");
                        }
                        const auto output =
                            ((static_cast<std::size_t>(layer) * batch + bi) * tokens +
                             token) * columns + column;
                        result.values[output] = hash;
                    }
                }
            }
        }
    }
    if (use_cache) {
        for (int bi = 0; bi < batch; ++bi) {
            const auto* row = history.data() +
                static_cast<std::size_t>(bi * combined_length);
            std::copy_n(
                row + tokens,
                prefix,
                context_.data() + static_cast<std::size_t>(bi * prefix));
        }
        position_ += tokens;
    }
    return result;
}

DeepseekV41EngramHashSnapshot MlxDeepseekV41EngramHashState::snapshot() const {
    return {batch_, position_, context_};
}

void MlxDeepseekV41EngramHashState::restore(
    DeepseekV41EngramHashSnapshot snapshot) {
    const auto expected = snapshot.batch <= 0
        ? std::size_t{0}
        : static_cast<std::size_t>(
              snapshot.batch * (max_ngram_size_ - 1));
    if (snapshot.position < 0 || snapshot.context.size() != expected) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 Engram hash snapshot");
    }
    batch_ = snapshot.batch;
    position_ = snapshot.position;
    context_ = std::move(snapshot.context);
}

class MlxDeepseekV41Engram::Table {
public:
    static std::shared_ptr<Table> load(
        const MfqContainer& model,
        const std::string& name,
        std::int64_t expected_rows,
        int expected_width) {
        const auto& record = model.record(name);
        if (record.dtype != "MXFP8") {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram table must use native MXFP8: " + name);
        }
        const auto bytes = model.read_range(name, 0, kMxHeaderBytes);
        if (bytes.size() < kMxHeaderBytes ||
            std::memcmp(bytes.data(), "MXT1", 4) != 0 ||
            bytes[4] != 1 || bytes[5] != 8 || bytes[6] != 0 || bytes[7] != 0) {
            throw std::runtime_error(
                "invalid DeepSeek-V4.1 Engram MXFP8 header: " + name);
        }
        const auto rows = read_u64_at(bytes, 8);
        const auto columns = read_u64_at(bytes, 16);
        const auto storage_rows = read_u64_at(bytes, 24);
        const auto storage_columns = read_u64_at(bytes, 32);
        const auto scale_rows = read_u64_at(bytes, 40);
        const auto scale_columns = read_u64_at(bytes, 48);
        if (rows != static_cast<std::uint64_t>(expected_rows) ||
            columns != static_cast<std::uint64_t>(expected_width) ||
            storage_rows != rows || storage_columns != columns ||
            scale_rows != rows || scale_columns != columns / 32 ||
            columns == 0 || columns % 32) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram table is not row-wise MXFP8: " + name);
        }
        const auto value_bytes = checked_product(rows, columns, "value");
        const auto scale_bytes = checked_product(
            scale_rows, scale_columns, "scale");
        if (value_bytes > std::numeric_limits<std::uint64_t>::max() -
                kMxHeaderBytes ||
            scale_bytes > std::numeric_limits<std::uint64_t>::max() -
                kMxHeaderBytes - value_bytes ||
            kMxHeaderBytes + value_bytes + scale_bytes != record.nbytes) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram MXFP8 payload size disagrees: " + name);
        }
        return std::shared_ptr<Table>(new Table(
            model,
            name,
            kMxHeaderBytes,
            kMxHeaderBytes + value_bytes,
            static_cast<std::int64_t>(rows),
            static_cast<int>(columns),
            cache_capacity_rows()));
    }

    array gather(
        std::span<const std::int64_t> rows,
        int batch,
        int tokens,
        int hash_columns) const {
        const auto expected = static_cast<std::size_t>(batch) * tokens * hash_columns;
        if (batch <= 0 || tokens <= 0 || hash_columns <= 0 ||
            rows.size() != expected) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 Engram lookup geometry disagrees");
        }
        std::vector<std::pair<std::int64_t, std::size_t>> requests;
        requests.reserve(rows.size());
        for (std::size_t index = 0; index < rows.size(); ++index) {
            if (rows[index] < 0 || rows[index] >= rows_) {
                throw std::out_of_range(
                    "DeepSeek-V4.1 Engram lookup row is outside the table");
            }
            requests.emplace_back(rows[index], index);
        }
        std::sort(
            requests.begin(), requests.end(),
            [](const auto& left, const auto& right) {
                return left.first < right.first ||
                    (left.first == right.first && left.second < right.second);
            });
        std::vector<mlx::core::float16_t> output(
            expected * static_cast<std::size_t>(width_));
        std::lock_guard lock(mutex_);
        std::size_t begin = 0;
        while (begin < requests.size()) {
            std::size_t end = begin + 1;
            while (end < requests.size() &&
                   requests[end].first == requests[begin].first) {
                ++end;
            }
            const auto& decoded = row_locked(requests[begin].first);
            for (std::size_t request = begin; request < end; ++request) {
                std::copy_n(
                    decoded.data(),
                    width_,
                    output.data() + requests[request].second *
                        static_cast<std::size_t>(width_));
            }
            begin = end;
        }
        return array(
            output.begin(),
            Shape{batch, tokens, hash_columns * width_});
    }

    std::size_t cached_rows() const noexcept {
        std::lock_guard lock(mutex_);
        return cache_.size();
    }

private:
    struct CacheEntry {
        std::vector<mlx::core::float16_t> values;
        std::list<std::int64_t>::iterator recency;
    };

    Table(
        MfqContainer model,
        std::string name,
        std::uint64_t values_offset,
        std::uint64_t scales_offset,
        std::int64_t rows,
        int width,
        std::size_t capacity)
        : model_(std::move(model)),
          name_(std::move(name)),
          values_offset_(values_offset),
          scales_offset_(scales_offset),
          rows_(rows),
          width_(width),
          capacity_(capacity) {
        for (std::size_t raw = 0; raw < e4m3_.size(); ++raw) {
            e4m3_[raw] = decode_e4m3(static_cast<std::uint8_t>(raw));
        }
        for (std::size_t raw = 0; raw < e8m0_.size(); ++raw) {
            e8m0_[raw] = raw == 255
                ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(1.0f, static_cast<int>(raw) - 127);
        }
        scratch_.resize(static_cast<std::size_t>(width_));
        raw_values_.resize(static_cast<std::size_t>(width_));
        raw_scales_.resize(static_cast<std::size_t>(width_ / 32));
    }

    void decode_row(
        std::int64_t row,
        std::vector<mlx::core::float16_t>& output) const {
        model_.read_range_into(
            name_,
            values_offset_ + static_cast<std::uint64_t>(row) *
                static_cast<std::uint64_t>(width_),
            std::as_writable_bytes(std::span<std::uint8_t>(raw_values_)));
        model_.read_range_into(
            name_,
            scales_offset_ + static_cast<std::uint64_t>(row) *
                static_cast<std::uint64_t>(width_ / 32),
            std::as_writable_bytes(std::span<std::uint8_t>(raw_scales_)));
        for (int group = 0; group < width_ / 32; ++group) {
            const auto scale = e8m0_[raw_scales_[static_cast<std::size_t>(group)]];
            if (!std::isfinite(scale)) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 Engram row contains an invalid E8M0 scale");
            }
            for (int column = 0; column < 32; ++column) {
                const auto value = e4m3_[raw_values_[static_cast<std::size_t>(
                    group * 32 + column)]];
                if (!std::isfinite(value)) {
                    throw std::runtime_error(
                        "DeepSeek-V4.1 Engram row contains an E4M3 NaN");
                }
                output[static_cast<std::size_t>(group * 32 + column)] =
                    static_cast<mlx::core::float16_t>(value * scale);
            }
        }
    }

    const std::vector<mlx::core::float16_t>& row_locked(
        std::int64_t row) const {
        const auto found = cache_.find(row);
        if (found != cache_.end()) {
            recency_.splice(recency_.begin(), recency_, found->second.recency);
            return found->second.values;
        }
        if (capacity_ == 0) {
            decode_row(row, scratch_);
            return scratch_;
        }
        if (cache_.size() >= capacity_) {
            const auto victim = recency_.back();
            recency_.pop_back();
            cache_.erase(victim);
        }
        recency_.push_front(row);
        CacheEntry entry{
            std::vector<mlx::core::float16_t>(static_cast<std::size_t>(width_)),
            recency_.begin(),
        };
        decode_row(row, entry.values);
        return cache_.emplace(row, std::move(entry)).first->second.values;
    }

    MfqContainer model_;
    std::string name_;
    std::uint64_t values_offset_;
    std::uint64_t scales_offset_;
    std::int64_t rows_;
    int width_;
    std::size_t capacity_;
    std::array<float, 256> e4m3_{};
    std::array<float, 256> e8m0_{};
    mutable std::mutex mutex_;
    mutable std::list<std::int64_t> recency_;
    mutable std::unordered_map<std::int64_t, CacheEntry> cache_;
    mutable std::vector<mlx::core::float16_t> scratch_;
    mutable std::vector<std::uint8_t> raw_values_;
    mutable std::vector<std::uint8_t> raw_scales_;
};

MlxDeepseekV41Engram MlxDeepseekV41Engram::load(
    const MfqContainer& model,
    const DeepseekV41Config& config,
    int layer) {
    const auto found = std::find(
        config.engram_layer_ids.begin(),
        config.engram_layer_ids.end(),
        layer);
    if (found == config.engram_layer_ids.end()) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 layer has no Engram component");
    }
    const auto hash_layer = static_cast<int>(std::distance(
        config.engram_layer_ids.begin(), found));
    const auto prefix = DeepseekV41TensorNames::layer(
        static_cast<std::size_t>(layer), "associative_memory");
    auto query = dense_array(model, prefix + ".query.weight");
    auto key = dense_array(model, prefix + ".key.weight");
    if (query.shape() != Shape{static_cast<int>(config.hc_mult),
                               static_cast<int>(config.hidden)} ||
        key.shape() != query.shape()) {
        throw std::runtime_error(
            "DeepSeek-V4.1 Engram query/key geometry disagrees");
    }
    auto query_key = mlx::core::contiguous(query * key);
    query_key.eval();
    auto projection = MlxLinear::load(model, prefix + ".projection.weight");
    const int hash_columns = static_cast<int>(
        (config.engram_max_ngram_size - 1) * config.engram_n_heads);
    if (projection.input_size() != hash_columns * config.engram_head_dim ||
        projection.output_size() != (config.hc_mult + 1) * config.hidden) {
        throw std::runtime_error(
            "DeepSeek-V4.1 Engram projection geometry disagrees");
    }
    return MlxDeepseekV41Engram(
        config,
        layer,
        hash_layer,
        Table::load(
            model,
            prefix + ".embedding.weight",
            config.engram_num_embeddings[static_cast<std::size_t>(hash_layer)],
            static_cast<int>(config.engram_head_dim)),
        std::move(projection),
        std::move(query_key));
}

MlxDeepseekV41Engram::MlxDeepseekV41Engram(
    DeepseekV41Config config,
    int layer,
    int hash_layer_index,
    std::shared_ptr<Table> table,
    MlxLinear projection,
    array query_key_weight)
    : config_(std::move(config)),
      layer_(layer),
      hash_layer_index_(hash_layer_index),
      table_(std::move(table)),
      projection_(std::move(projection)),
      query_key_weight_(std::move(query_key_weight)) {}

array MlxDeepseekV41Engram::forward(
    const array& hidden_streams,
    const DeepseekV41EngramHashBatch& hashes,
    const std::optional<array>& participation_mask) const {
    const int batch = hashes.batch;
    const int tokens = hashes.tokens;
    const int streams = static_cast<int>(config_.hc_mult);
    const int hidden = static_cast<int>(config_.hidden);
    if (hidden_streams.shape() != Shape{batch, tokens, streams, hidden} ||
        hashes.layers != static_cast<int>(config_.engram_layer_ids.size()) ||
        hashes.hash_columns !=
            (config_.engram_max_ngram_size - 1) * config_.engram_n_heads) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 Engram activation geometry disagrees");
    }
    auto embeddings = table_->gather(
        hashes.layer(hash_layer_index_),
        batch,
        tokens,
        hashes.hash_columns);
    auto key_value = projection_(embeddings);
    const int key_width = streams * hidden;
    auto key = mlx::core::slice(
        key_value,
        Shape{0, 0, 0},
        Shape{batch, tokens, key_width});
    key = mlx::core::reshape(
        mlx::core::astype(std::move(key), mlx::core::float32),
        Shape{batch, tokens, streams, hidden});
    auto value = mlx::core::slice(
        key_value,
        Shape{0, 0, key_width},
        Shape{batch, tokens, key_width + hidden});
    value = mlx::core::reshape(
        mlx::core::astype(std::move(value), mlx::core::float32),
        Shape{batch, tokens, 1, hidden});
    auto source = mlx::core::astype(hidden_streams, mlx::core::float32);
    auto inverse = mlx::core::rsqrt(
        mlx::core::mean(source * source, -1) +
        static_cast<float>(config_.rms_eps));
    inverse = inverse * mlx::core::rsqrt(
        mlx::core::mean(key * key, -1) +
        static_cast<float>(config_.rms_eps));
    auto score = mlx::core::sum(
        source * key * query_key_weight_, -1) * inverse /
        std::sqrt(static_cast<float>(hidden));
    auto sign = mlx::core::astype(
        mlx::core::less(array(0.0f), score), mlx::core::float32) -
        mlx::core::astype(
            mlx::core::less(score, array(0.0f)), mlx::core::float32);
    auto gate = mlx::core::sigmoid(
        sign * mlx::core::sqrt(
            mlx::core::maximum(mlx::core::abs(score), array(1e-6f))));
    if (participation_mask) {
        if (participation_mask->shape() != Shape{batch, tokens}) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 Engram gate mask shape disagrees");
        }
        gate = gate * mlx::core::expand_dims(
            mlx::core::astype(*participation_mask, mlx::core::float32), -1);
    }
    auto output = source + mlx::core::expand_dims(gate, -1) * value;
    return hidden_streams.dtype() == mlx::core::float32
        ? output
        : mlx::core::astype(std::move(output), hidden_streams.dtype());
}

std::size_t MlxDeepseekV41Engram::cached_rows() const noexcept {
    return table_->cached_rows();
}

} // namespace mfq::metal
