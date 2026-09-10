// Included by mfq_decode.cpp after its storage and linear loaders.  Engram's
// token hashing stays on the CPU and each table faults in only selected MXFP8
// rows; the multi-hundred-billion-row tables are never materialized.

namespace mfq::cuda::deepseek_v41_runtime {

using EngramTensor = mfq_tensor_backend::Tensor;
using EngramConfig = mfq::models::deepseek_v41::Config;

namespace engram_detail {

constexpr std::array<std::uint8_t, 8> kMagic{
    'M', 'F', 'Q', 'E', 'N', 'G', 'R', '1'};
constexpr std::size_t kMxHeaderBytes = 56;
constexpr std::int64_t kDeadToken = -1;

class ByteCursor {
public:
    explicit ByteCursor(const std::vector<std::uint8_t>& bytes)
        : bytes_(bytes) {}

    const std::uint8_t* take(
        std::size_t count,
        std::string_view description) {
        if (offset_ > bytes_.size() || count > bytes_.size() - offset_) {
            throw std::runtime_error(
                "truncated DeepSeek-V4.1 Engram " +
                std::string(description));
        }
        const auto* result = bytes_.data() + offset_;
        offset_ += count;
        return result;
    }

    std::uint32_t u32(std::string_view description) {
        const auto* bytes = take(4, description);
        std::uint32_t result = 0;
        for (int index = 0; index < 4; ++index) {
            result |= static_cast<std::uint32_t>(bytes[index]) << (8 * index);
        }
        return result;
    }

    std::uint64_t u64(std::string_view description) {
        const auto* bytes = take(8, description);
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
    const std::vector<std::uint8_t>& bytes_;
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

std::uint64_t checked_product(
    std::uint64_t left,
    std::uint64_t right,
    std::string_view description) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::runtime_error(
            "DeepSeek-V4.1 Engram " + std::string(description) +
            " byte size overflows");
    }
    return left * right;
}

std::uint64_t read_u64_at(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset) {
    if (offset > bytes.size() || 8 > bytes.size() - offset) {
        throw std::runtime_error(
            "truncated DeepSeek-V4.1 Engram MX header");
    }
    std::uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(bytes[offset + index]) <<
            (8 * index);
    }
    return value;
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

} // namespace engram_detail

struct EngramHashBatch {
    int batch = 0;
    int tokens = 0;
    int layers = 0;
    int columns = 0;
    std::vector<std::int64_t> values;

    const std::int64_t* layer_data(int index) const {
        if (index < 0 || index >= layers || batch <= 0 || tokens <= 0 ||
            columns <= 0) {
            throw std::out_of_range(
                "invalid DeepSeek-V4.1 Engram hash layer");
        }
        const auto layer_size = static_cast<std::size_t>(batch) *
            static_cast<std::size_t>(tokens) *
            static_cast<std::size_t>(columns);
        const auto begin = static_cast<std::size_t>(index) * layer_size;
        if (begin > values.size() || layer_size > values.size() - begin) {
            throw std::runtime_error(
                "truncated DeepSeek-V4.1 Engram hash batch");
        }
        return values.data() + begin;
    }
};

class EngramHashState {
public:
    static std::unique_ptr<EngramHashState> load(
        const MfqFile& model,
        const EngramConfig& config) {
        constexpr const char* asset_name =
            "__mfq_asset__/deepseek-v41-engram-v1.bin";
        if (!model.has_record(asset_name)) {
            throw std::runtime_error(
                "DeepSeek-V4.1 MFQ has no embedded Engram hash asset");
        }
        const auto asset = model.read_asset(asset_name);
        engram_detail::ByteCursor cursor(asset);
        const auto* magic = cursor.take(
            engram_detail::kMagic.size(), "asset magic");
        if (!std::equal(
                engram_detail::kMagic.begin(),
                engram_detail::kMagic.end(),
                magic)) {
            throw std::runtime_error(
                "invalid DeepSeek-V4.1 Engram asset magic");
        }
        const auto vocabulary = cursor.u32("vocabulary");
        const auto compressed = cursor.u32("compressed vocabulary");
        const auto layer_count = cursor.u32("layer count");
        const auto max_ngram = cursor.u32("maximum ngram size");
        const auto heads = cursor.u32("head count");
        const auto pad_id = cursor.i32("compressed pad token");
        if (vocabulary > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()) ||
            compressed > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()) ||
            layer_count == 0 ||
            layer_count > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()) ||
            max_ngram < 2 ||
            max_ngram > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()) ||
            heads == 0 ||
            heads > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "invalid DeepSeek-V4.1 Engram asset dimensions");
        }
        const auto layers = static_cast<std::size_t>(layer_count);
        const auto orders = static_cast<std::size_t>(max_ngram - 1);
        const auto head_count = static_cast<std::size_t>(heads);
        const auto bucket_count = engram_detail::checked_product(
            engram_detail::checked_product(layers, orders, "bucket"),
            head_count,
            "bucket");
        const auto multiplier_count = engram_detail::checked_product(
            layers,
            static_cast<std::size_t>(max_ngram),
            "multiplier");
        auto layer_ids = engram_detail::read_vector<std::int64_t>(
            cursor, layers,
            [](auto& value, auto what) {
                return static_cast<std::int64_t>(value.i32(what));
            },
            "layer ids");
        auto table_rows = engram_detail::read_vector<std::int64_t>(
            cursor, layers,
            [](auto& value, auto what) { return value.i64(what); },
            "table rows");
        auto primes = engram_detail::read_vector<std::int64_t>(
            cursor, static_cast<std::size_t>(bucket_count),
            [](auto& value, auto what) { return value.i64(what); },
            "bucket primes");
        auto offsets = engram_detail::read_vector<std::int64_t>(
            cursor, static_cast<std::size_t>(bucket_count),
            [](auto& value, auto what) { return value.i64(what); },
            "bucket offsets");
        auto multipliers = engram_detail::read_vector<std::int64_t>(
            cursor, static_cast<std::size_t>(multiplier_count),
            [](auto& value, auto what) { return value.i64(what); },
            "hash multipliers");
        auto token_map = engram_detail::read_vector<std::int32_t>(
            cursor, vocabulary,
            [](auto& value, auto what) { return value.i32(what); },
            "compressed token map");
        cursor.require_end();

        if (vocabulary != static_cast<std::uint32_t>(config.vocab) ||
            compressed != static_cast<std::uint32_t>(
                config.engram_compressed_vocab_size) ||
            max_ngram != static_cast<std::uint32_t>(
                config.engram_max_ngram_size) ||
            heads != static_cast<std::uint32_t>(config.engram_n_heads) ||
            layer_ids != config.engram_layer_ids ||
            table_rows != config.engram_num_embeddings ||
            pad_id < 0 || pad_id >= static_cast<std::int64_t>(compressed)) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram asset disagrees with model config");
        }
        for (std::size_t layer = 0; layer < layers; ++layer) {
            const auto last = layer * orders * head_count +
                orders * head_count - 1;
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
                [](std::int64_t value) {
                    return value <= 0 || !(value & 1);
                })) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram hash metadata is invalid");
        }
        return std::unique_ptr<EngramHashState>(new EngramHashState(
            static_cast<int>(vocabulary),
            static_cast<int>(max_ngram),
            static_cast<int>(heads),
            pad_id,
            std::move(layer_ids),
            std::move(table_rows),
            std::move(primes),
            std::move(offsets),
            std::move(multipliers),
            std::move(token_map)));
    }

    void reset(int batch) {
        if (batch <= 0) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 Engram batch must be positive");
        }
        batch_ = batch;
        position_ = 0;
        context_.assign(
            static_cast<std::size_t>(batch * (max_ngram_size_ - 1)),
            pad_id_);
    }

    EngramHashBatch forward(
        const EngramTensor& token_ids,
        std::int64_t pos0) {
        if (token_ids.dim() != 2 || token_ids.size(0) <= 0 ||
            token_ids.size(1) <= 0 || pos0 < 0 ||
            token_ids.size(0) > std::numeric_limits<int>::max() ||
            token_ids.size(1) > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 Engram token IDs must have nonempty [B,T] shape");
        }
        const int batch = static_cast<int>(token_ids.size(0));
        const int tokens = static_cast<int>(token_ids.size(1));
        if (pos0 == 0) reset(batch);
        if (batch_ != batch || position_ != pos0 || context_.empty()) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram cache position disagrees");
        }
        auto host_ids = token_ids
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
            .contiguous();
        const auto* ids = host_ids.data_ptr<std::int64_t>();
        const auto element_count = static_cast<std::size_t>(batch) * tokens;
        std::vector<std::int64_t> compressed(element_count);
        for (std::size_t index = 0; index < element_count; ++index) {
            const auto token = ids[index];
            if (token < 0 ||
                token >= static_cast<std::int64_t>(token_map_.size())) {
                throw std::out_of_range(
                    "DeepSeek-V4.1 Engram token ID is outside the vocabulary");
            }
            compressed[index] = token_map_[static_cast<std::size_t>(token)];
        }

        const int prefix = max_ngram_size_ - 1;
        const int combined_length = prefix + tokens;
        const int layers = static_cast<int>(layer_ids_.size());
        const int columns = (max_ngram_size_ - 1) * heads_;
        EngramHashBatch result;
        result.batch = batch;
        result.tokens = tokens;
        result.layers = layers;
        result.columns = columns;
        result.values.resize(
            static_cast<std::size_t>(layers) * batch * tokens * columns);
        std::vector<std::int64_t> history(
            static_cast<std::size_t>(batch) * combined_length,
            pad_id_);
        for (int bi = 0; bi < batch; ++bi) {
            auto* row = history.data() +
                static_cast<std::size_t>(bi) * combined_length;
            std::copy_n(
                context_.data() + static_cast<std::size_t>(bi) * prefix,
                prefix,
                row);
            std::copy_n(
                compressed.data() + static_cast<std::size_t>(bi) * tokens,
                tokens,
                row + prefix);
            for (int token = 0; token < tokens; ++token) {
                std::array<std::int64_t, 32> ngram_tokens{};
                if (max_ngram_size_ >
                    static_cast<int>(ngram_tokens.size())) {
                    throw std::runtime_error(
                        "DeepSeek-V4.1 Engram ngram size exceeds native bound");
                }
                bool blocked = false;
                for (int shift = 0; shift < max_ngram_size_; ++shift) {
                    const auto source = row[prefix + token - shift];
                    blocked = blocked || source == engram_detail::kDeadToken;
                    ngram_tokens[static_cast<std::size_t>(shift)] =
                        blocked ? pad_id_ : source;
                }
                for (int layer = 0; layer < layers; ++layer) {
                    const auto multiplier_base = static_cast<std::size_t>(
                        layer * max_ngram_size_);
                    std::uint64_t rolling =
                        static_cast<std::uint64_t>(ngram_tokens[0]) *
                        static_cast<std::uint64_t>(
                            multipliers_[multiplier_base]);
                    for (int shift = 1; shift < max_ngram_size_; ++shift) {
                        rolling ^=
                            static_cast<std::uint64_t>(ngram_tokens[shift]) *
                            static_cast<std::uint64_t>(
                                multipliers_[multiplier_base + shift]);
                        std::int64_t signed_hash = 0;
                        std::memcpy(
                            &signed_hash, &rolling, sizeof(signed_hash));
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
                                ((static_cast<std::size_t>(layer) * batch + bi) *
                                     tokens + token) *
                                    columns + column;
                            result.values[output] = hash;
                        }
                    }
                }
            }
        }
        for (int bi = 0; bi < batch; ++bi) {
            const auto* row = history.data() +
                static_cast<std::size_t>(bi) * combined_length;
            std::copy_n(
                row + tokens,
                prefix,
                context_.data() + static_cast<std::size_t>(bi) * prefix);
        }
        position_ += tokens;
        return result;
    }

    static void self_check() {
        const std::vector<std::int64_t> expected{
            3, 8, 15, 26,
            3, 8, 15, 26,
            3, 8, 16, 27,
            3, 8, 14, 23,
        };
        const auto make_state = [] {
            return EngramHashState(
                8,
                3,
                2,
                0,
                {1},
                {36},
                {5, 7, 11, 13},
                {0, 5, 12, 23},
                {3, 5, 7},
                {0, 1, 2, 3, 4, 5, 6, 7});
        };
        const auto options = mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCPU)
            .dtype(mfq_tensor_backend::kInt64);
        auto all_ids = mfq_tensor_backend::tensor(
            std::vector<std::int64_t>{1, 2, 3, 4}, options)
                           .reshape({1, 4});
        auto full_state = make_state();
        const auto full = full_state.forward(all_ids, 0);
        MFQ_RUNTIME_CHECK(
            full.values == expected,
            "DeepSeek-V4.1 Engram full hash contract failed");

        auto chunked_state = make_state();
        const auto first = chunked_state.forward(
            all_ids.narrow(1, 0, 2), 0);
        const auto second = chunked_state.forward(
            all_ids.narrow(1, 2, 2), 2);
        std::vector<std::int64_t> chunked = first.values;
        chunked.insert(
            chunked.end(), second.values.begin(), second.values.end());
        MFQ_RUNTIME_CHECK(
            chunked == expected,
            "DeepSeek-V4.1 Engram chunked hash contract failed");
        MFQ_RUNTIME_CHECK(
            engram_detail::decode_e4m3(0x00) == 0.0f &&
                engram_detail::decode_e4m3(0x38) == 1.0f &&
                engram_detail::decode_e4m3(0x40) == 2.0f &&
                engram_detail::decode_e4m3(0xb8) == -1.0f &&
                engram_detail::decode_e4m3(0x01) ==
                    std::ldexp(1.0f, -9),
            "DeepSeek-V4.1 Engram E4M3 decode contract failed");
    }

private:
    EngramHashState(
        int vocabulary,
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
          max_ngram_size_(max_ngram_size),
          heads_(heads),
          pad_id_(pad_id),
          layer_ids_(std::move(layer_ids)),
          table_rows_(std::move(table_rows)),
          primes_(std::move(primes)),
          offsets_(std::move(offsets)),
          multipliers_(std::move(multipliers)),
          token_map_(std::move(token_map)) {}

    int vocabulary_ = 0;
    int max_ngram_size_ = 0;
    int heads_ = 0;
    std::int64_t pad_id_ = 0;
    std::vector<std::int64_t> layer_ids_;
    std::vector<std::int64_t> table_rows_;
    std::vector<std::int64_t> primes_;
    std::vector<std::int64_t> offsets_;
    std::vector<std::int64_t> multipliers_;
    std::vector<std::int32_t> token_map_;
    int batch_ = 0;
    std::int64_t position_ = 0;
    std::vector<std::int64_t> context_;
};

class EngramTable {
public:
    static std::shared_ptr<EngramTable> load(
        const MfqFile& model,
        const std::string& name,
        std::int64_t expected_rows,
        int expected_width) {
        const auto record = model.record(name);
        if (record.dtype != "MXFP8") {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram table must use native MXFP8: " + name);
        }
        auto header = read_record_range(record, name, 0,
                                        engram_detail::kMxHeaderBytes);
        if (header.size() < engram_detail::kMxHeaderBytes ||
            std::memcmp(header.data(), "MXT1", 4) != 0 ||
            header[4] != 1 || header[5] != 8 ||
            header[6] != 0 || header[7] != 0) {
            throw std::runtime_error(
                "invalid DeepSeek-V4.1 Engram MXFP8 header: " + name);
        }
        const auto rows = engram_detail::read_u64_at(header, 8);
        const auto columns = engram_detail::read_u64_at(header, 16);
        const auto storage_rows = engram_detail::read_u64_at(header, 24);
        const auto storage_columns = engram_detail::read_u64_at(header, 32);
        const auto scale_rows = engram_detail::read_u64_at(header, 40);
        const auto scale_columns = engram_detail::read_u64_at(header, 48);
        if (rows != static_cast<std::uint64_t>(expected_rows) ||
            columns != static_cast<std::uint64_t>(expected_width) ||
            storage_rows != rows || storage_columns != columns ||
            scale_rows != rows || scale_columns != columns / 32 ||
            columns == 0 || columns % 32) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram table is not row-wise MXFP8: " + name);
        }
        const auto value_bytes = engram_detail::checked_product(
            rows, columns, "value");
        const auto scale_bytes = engram_detail::checked_product(
            scale_rows, scale_columns, "scale");
        if (value_bytes > std::numeric_limits<std::uint64_t>::max() -
                engram_detail::kMxHeaderBytes ||
            scale_bytes > std::numeric_limits<std::uint64_t>::max() -
                engram_detail::kMxHeaderBytes - value_bytes ||
            engram_detail::kMxHeaderBytes + value_bytes + scale_bytes !=
                record.nbytes) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram MXFP8 payload size disagrees: " + name);
        }
        return std::shared_ptr<EngramTable>(new EngramTable(
            record,
            name,
            static_cast<std::int64_t>(rows),
            static_cast<int>(columns),
            value_bytes,
            engram_detail::cache_capacity_rows()));
    }

    EngramTensor gather(
        const std::int64_t* rows,
        int batch,
        int tokens,
        int hash_columns,
        const mfq_tensor_backend::Device& device) const {
        const auto expected = static_cast<std::size_t>(batch) * tokens *
            hash_columns;
        if (batch <= 0 || tokens <= 0 || hash_columns <= 0 ||
            rows == nullptr) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 Engram lookup geometry disagrees");
        }
        std::vector<std::pair<std::int64_t, std::size_t>> requests;
        requests.reserve(expected);
        for (std::size_t index = 0; index < expected; ++index) {
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
        std::vector<float> output(
            expected * static_cast<std::size_t>(width_));
        std::lock_guard<std::mutex> lock(mutex_);
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
        return mfq_tensor_backend::from_blob(
                   output.data(),
                   {batch, tokens,
                    static_cast<std::int64_t>(hash_columns) * width_},
                   mfq_tensor_backend::TensorOptions()
                       .device(mfq_tensor_backend::kCPU)
                       .dtype(mfq_tensor_backend::kFloat32))
            .clone()
            .to(device, mfq_tensor_backend::kFloat16)
            .contiguous();
    }

private:
    struct CacheEntry {
        std::vector<float> values;
        std::list<std::int64_t>::iterator recency;
    };

    static std::vector<std::uint8_t> read_record_range(
        const Record& record,
        const std::string& name,
        std::uint64_t relative_offset,
        std::uint64_t count) {
        if (relative_offset > record.nbytes ||
            count > record.nbytes - relative_offset ||
            count > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()) ||
            count > static_cast<std::uint64_t>(
                std::numeric_limits<std::streamsize>::max()) ||
            record.offset > std::numeric_limits<std::uint64_t>::max() -
                relative_offset) {
            throw std::out_of_range(
                "MFQ record byte range is out of bounds: " + name);
        }
        std::vector<std::uint8_t> result(static_cast<std::size_t>(count));
        std::ifstream stream(record.source_path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error(
                "cannot open DeepSeek-V4.1 Engram table: " + name);
        }
        stream.seekg(static_cast<std::streamoff>(
            record.offset + relative_offset));
        stream.read(
            reinterpret_cast<char*>(result.data()),
            static_cast<std::streamsize>(result.size()));
        if (!stream) {
            throw std::runtime_error(
                "failed reading DeepSeek-V4.1 Engram table: " + name);
        }
        return result;
    }

    EngramTable(
        Record record,
        std::string name,
        std::int64_t rows,
        int width,
        std::uint64_t value_bytes,
        std::size_t capacity)
        : record_(std::move(record)),
          name_(std::move(name)),
          rows_(rows),
          width_(width),
          value_bytes_(value_bytes),
          capacity_(capacity),
          stream_(record_.source_path, std::ios::binary) {
        if (!stream_) {
            throw std::runtime_error(
                "cannot open DeepSeek-V4.1 Engram table: " + name_);
        }
        for (std::size_t raw = 0; raw < e4m3_.size(); ++raw) {
            e4m3_[raw] = engram_detail::decode_e4m3(
                static_cast<std::uint8_t>(raw));
        }
        for (std::size_t raw = 0; raw < e8m0_.size(); ++raw) {
            e8m0_[raw] = raw == 255
                ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(1.0f, static_cast<int>(raw) - 127);
        }
        scratch_.resize(static_cast<std::size_t>(width_));
        row_values_.resize(static_cast<std::size_t>(width_));
        row_scales_.resize(static_cast<std::size_t>(width_ / 32));
    }

    void read_into(
        std::uint64_t relative_offset,
        std::uint8_t* destination,
        std::size_t count) const {
        if (relative_offset > record_.nbytes ||
            count > record_.nbytes - relative_offset ||
            record_.offset > std::numeric_limits<std::uint64_t>::max() -
                relative_offset) {
            throw std::out_of_range(
                "DeepSeek-V4.1 Engram row range is invalid: " + name_);
        }
        stream_.clear();
        stream_.seekg(static_cast<std::streamoff>(
            record_.offset + relative_offset));
        stream_.read(
            reinterpret_cast<char*>(destination),
            static_cast<std::streamsize>(count));
        if (!stream_) {
            throw std::runtime_error(
                "failed reading DeepSeek-V4.1 Engram row: " + name_);
        }
    }

    void decode_row(
        std::int64_t row,
        std::vector<float>& output) const {
        read_into(
            engram_detail::kMxHeaderBytes +
                static_cast<std::uint64_t>(row) * width_,
            row_values_.data(),
            row_values_.size());
        read_into(
            engram_detail::kMxHeaderBytes + value_bytes_ +
                static_cast<std::uint64_t>(row) * (width_ / 32),
            row_scales_.data(),
            row_scales_.size());
        for (int group = 0; group < width_ / 32; ++group) {
            const auto scale = e8m0_[row_scales_[group]];
            if (!std::isfinite(scale)) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 Engram row contains an invalid E8M0 scale");
            }
            for (int column = 0; column < 32; ++column) {
                const auto value = e4m3_[
                    row_values_[static_cast<std::size_t>(group * 32 + column)]];
                if (!std::isfinite(value)) {
                    throw std::runtime_error(
                        "DeepSeek-V4.1 Engram row contains an E4M3 NaN");
                }
                output[static_cast<std::size_t>(group * 32 + column)] =
                    value * scale;
            }
        }
    }

    const std::vector<float>& row_locked(std::int64_t row) const {
        const auto found = cache_.find(row);
        if (found != cache_.end()) {
            recency_.splice(
                recency_.begin(), recency_, found->second.recency);
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
            std::vector<float>(static_cast<std::size_t>(width_)),
            recency_.begin()};
        decode_row(row, entry.values);
        return cache_.emplace(row, std::move(entry)).first->second.values;
    }

    Record record_;
    std::string name_;
    std::int64_t rows_ = 0;
    int width_ = 0;
    std::uint64_t value_bytes_ = 0;
    std::size_t capacity_ = 0;
    std::array<float, 256> e4m3_{};
    std::array<float, 256> e8m0_{};
    mutable std::mutex mutex_;
    mutable std::list<std::int64_t> recency_;
    mutable std::unordered_map<std::int64_t, CacheEntry> cache_;
    mutable std::vector<float> scratch_;
    mutable std::vector<std::uint8_t> row_values_;
    mutable std::vector<std::uint8_t> row_scales_;
    mutable std::ifstream stream_;
};

class Engram {
public:
    static std::unique_ptr<Engram> load(
        const MfqFile& model,
        const EngramConfig& config,
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
        const auto prefix = "model.block." + std::to_string(layer) +
            ".associative_memory.";
        auto query = load_dense_gpu(model, prefix + "query.weight")
            .to(mfq_tensor_backend::kFloat32)
            .contiguous();
        auto key = load_dense_gpu(model, prefix + "key.weight")
            .to(mfq_tensor_backend::kFloat32)
            .contiguous();
        if (query.dim() != 2 || query.size(0) != config.hc_mult ||
            query.size(1) != config.hidden || key.sizes() != query.sizes()) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram query/key geometry disagrees");
        }
        auto projection = load_quant_linear(
            model, prefix + "projection.weight");
        const auto hash_columns =
            (config.engram_max_ngram_size - 1) * config.engram_n_heads;
        if (projection.neuron_len() !=
                hash_columns * config.engram_head_dim ||
            projection.out() != (config.hc_mult + 1) * config.hidden) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram projection geometry disagrees");
        }
        return std::unique_ptr<Engram>(new Engram(
            config,
            hash_layer,
            EngramTable::load(
                model,
                prefix + "embedding.weight",
                config.engram_num_embeddings[
                    static_cast<std::size_t>(hash_layer)],
                static_cast<int>(config.engram_head_dim)),
            std::move(projection),
            (query * key).contiguous()));
    }

    EngramTensor forward(
        const EngramTensor& hidden,
        const EngramHashBatch& hashes) const {
        const auto batch = hashes.batch;
        const auto tokens = hashes.tokens;
        if (hidden.dim() != 4 || hidden.size(0) != batch ||
            hidden.size(1) != tokens || hidden.size(2) != config_.hc_mult ||
            hidden.size(3) != config_.hidden ||
            hashes.layers != static_cast<int>(
                config_.engram_layer_ids.size()) ||
            hashes.columns !=
                (config_.engram_max_ngram_size - 1) *
                    config_.engram_n_heads) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 Engram activation geometry disagrees");
        }
        auto embeddings = table_->gather(
            hashes.layer_data(hash_layer_),
            batch,
            tokens,
            hashes.columns,
            hidden.device());
        auto key_value = projection_.forward(embeddings);
        const auto key_width = config_.hc_mult * config_.hidden;
        auto key = key_value.narrow(-1, 0, key_width)
            .reshape({batch, tokens, config_.hc_mult, config_.hidden})
            .to(mfq_tensor_backend::kFloat32);
        auto value = key_value.narrow(-1, key_width, config_.hidden)
            .reshape({batch, tokens, 1, config_.hidden})
            .to(mfq_tensor_backend::kFloat32);
        auto source = hidden.to(mfq_tensor_backend::kFloat32);
        auto inverse = mfq_tensor_backend::rsqrt(
            source.square().mean(-1) + config_.rms_eps) *
            mfq_tensor_backend::rsqrt(
                key.square().mean(-1) + config_.rms_eps);
        auto score = (source * key * query_key_weight_).sum(-1) * inverse /
            std::sqrt(static_cast<double>(config_.hidden));
        auto sign = (score > 0.0).to(mfq_tensor_backend::kFloat32) -
            (score < 0.0).to(mfq_tensor_backend::kFloat32);
        auto gate = mfq_tensor_backend::sigmoid(
            sign * score.abs().clamp_min(1e-6).sqrt());
        return (source + gate.unsqueeze(-1) * value)
            .to(hidden.scalar_type())
            .contiguous();
    }

private:
    Engram(
        EngramConfig config,
        int hash_layer,
        std::shared_ptr<EngramTable> table,
        QuantLinear projection,
        EngramTensor query_key_weight)
        : config_(std::move(config)),
          hash_layer_(hash_layer),
          table_(std::move(table)),
          projection_(std::move(projection)),
          query_key_weight_(std::move(query_key_weight)) {}

    EngramConfig config_;
    int hash_layer_ = -1;
    std::shared_ptr<EngramTable> table_;
    QuantLinear projection_;
    EngramTensor query_key_weight_;
};

} // namespace mfq::cuda::deepseek_v41_runtime
