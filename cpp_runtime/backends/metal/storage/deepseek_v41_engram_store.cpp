#include "deepseek_v41_engram_store.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace mfq::metal {
namespace {

constexpr std::array<char, 4> kAssetMagic{'D', '4', '1', 'T'};
constexpr std::uint32_t kAssetVersion = 2;
constexpr std::int64_t kDeadToken = -1;

class AssetCursor {
public:
    explicit AssetCursor(std::vector<std::uint8_t> bytes)
        : bytes_(std::move(bytes)) {}

    template <typename T>
    T scalar(const char* name) {
        if (sizeof(T) > remaining()) {
            throw std::runtime_error(
                std::string("truncated DeepSeek-V4.1 Engram ") + name);
        }
        using Unsigned = std::make_unsigned_t<T>;
        Unsigned value = 0;
        for (std::size_t index = 0; index < sizeof(T); ++index) {
            value |= static_cast<Unsigned>(bytes_[offset_ + index])
                << (index * 8);
        }
        offset_ += sizeof(T);
        T result{};
        std::memcpy(&result, &value, sizeof(T));
        return result;
    }

    std::array<char, 4> magic() {
        if (4 > remaining()) {
            throw std::runtime_error(
                "truncated DeepSeek-V4.1 Engram asset magic");
        }
        std::array<char, 4> result{};
        std::memcpy(result.data(), bytes_.data() + offset_, 4);
        offset_ += 4;
        return result;
    }

    std::size_t remaining() const noexcept {
        return bytes_.size() - offset_;
    }

private:
    std::vector<std::uint8_t> bytes_;
    std::size_t offset_ = 0;
};

std::vector<std::uint8_t> read_asset(
    const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "cannot open DeepSeek-V4.1 Engram asset: " + path.string());
    }
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) >
            std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "cannot size DeepSeek-V4.1 Engram asset: " + path.string());
    }
    std::vector<std::uint8_t> result(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    stream.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(result.size()));
    if (!stream && !result.empty()) {
        throw std::runtime_error(
            "cannot read DeepSeek-V4.1 Engram asset: " + path.string());
    }
    return result;
}

std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* name) {
    if (left != 0 && right >
            std::numeric_limits<std::size_t>::max() / left) {
        throw std::runtime_error(
            std::string("DeepSeek-V4.1 Engram ") + name + " overflows");
    }
    return left * right;
}

std::uint64_t positive_remainder(
    std::uint64_t twos_complement,
    std::uint32_t divisor) {
    if ((twos_complement >> 63u) == 0u) {
        return twos_complement % divisor;
    }
    const auto magnitude = ~twos_complement + 1u;
    const auto remainder = magnitude % divisor;
    return remainder == 0 ? 0 : divisor - remainder;
}

float fp8_e4m3fn(std::uint8_t raw) {
    const bool negative = (raw & 0x80u) != 0;
    const auto exponent = static_cast<int>((raw >> 3u) & 0x0fu);
    const auto mantissa = static_cast<int>(raw & 0x07u);
    float value = 0.0f;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
    } else if (exponent == 15 && mantissa == 7) {
        value = std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                           exponent - 7);
    }
    return negative ? -value : value;
}

float fp8_e8m0(std::uint8_t raw) {
    if (raw == 0xffu) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const std::uint32_t bits = raw == 0u
        ? 0x00400000u
        : static_cast<std::uint32_t>(raw) << 23u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        return static_cast<std::uint16_t>(
            (bits >> 16u) | ((bits & 0x007fffffu) != 0u ? 0x0040u : 0u));
    }
    const std::uint32_t rounding = 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>((bits + rounding) >> 16u);
}

const std::array<std::uint16_t, 1u << 16u>&
engram_fp8_bf16_table() {
    static const auto table = [] {
        std::array<std::uint16_t, 1u << 16u> result{};
        for (std::uint32_t scale = 0; scale < 256u; ++scale) {
            const auto decoded_scale = fp8_e8m0(
                static_cast<std::uint8_t>(scale));
            for (std::uint32_t value = 0; value < 256u; ++value) {
                result[(scale << 8u) | value] = float_to_bf16(
                    fp8_e4m3fn(static_cast<std::uint8_t>(value)) *
                    decoded_scale);
            }
        }
        return result;
    }();
    return table;
}

} // namespace

DeepseekV41EngramHashState::DeepseekV41EngramHashState(
    std::vector<std::uint32_t> token_map,
    std::uint32_t compressed_vocab_size,
    std::uint32_t max_ngram_size,
    std::uint32_t n_heads,
    std::uint32_t pad_id,
    std::vector<DeepseekV41EngramHashLayer> layers)
    : token_map_(std::move(token_map)),
      compressed_vocab_size_(compressed_vocab_size),
      max_ngram_size_(max_ngram_size),
      n_heads_(n_heads),
      pad_id_(pad_id),
      layers_(std::move(layers)) {}

DeepseekV41EngramHashState DeepseekV41EngramHashState::load(
    const std::filesystem::path& asset,
    std::uint32_t expected_vocab_size,
    std::uint32_t expected_compressed_vocab_size,
    std::uint32_t expected_max_ngram_size,
    std::uint32_t expected_n_heads,
    std::int32_t pad_token_id,
    std::span<const std::int64_t> expected_layer_ids,
    std::span<const std::int64_t> expected_table_rows) {
    AssetCursor cursor(read_asset(asset));
    const auto magic = cursor.magic();
    const auto version = cursor.scalar<std::uint32_t>("asset version");
    const auto vocab_size = cursor.scalar<std::uint32_t>("vocabulary size");
    const auto compressed_vocab_size =
        cursor.scalar<std::uint32_t>("compressed vocabulary size");
    const auto layer_count = cursor.scalar<std::uint32_t>("layer count");
    const auto max_ngram_size =
        cursor.scalar<std::uint32_t>("maximum n-gram size");
    const auto n_heads = cursor.scalar<std::uint32_t>("head count");
    if (magic != kAssetMagic || version != kAssetVersion ||
        vocab_size != expected_vocab_size ||
        compressed_vocab_size != expected_compressed_vocab_size ||
        layer_count != expected_layer_ids.size() ||
        layer_count != expected_table_rows.size() ||
        max_ngram_size != expected_max_ngram_size ||
        n_heads != expected_n_heads || vocab_size == 0 ||
        compressed_vocab_size == 0 || max_ngram_size < 2 || n_heads == 0 ||
        pad_token_id < 0 ||
        static_cast<std::uint32_t>(pad_token_id) >= vocab_size) {
        throw std::runtime_error(
            "DeepSeek-V4.1 Engram asset does not match model config");
    }

    std::vector<std::uint32_t> token_map(vocab_size);
    for (auto& value : token_map) {
        value = cursor.scalar<std::uint32_t>("token map");
        if (value >= compressed_vocab_size) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram token map is out of range");
        }
    }
    std::vector<std::int32_t> layer_ids(layer_count);
    for (auto& value : layer_ids) {
        value = cursor.scalar<std::int32_t>("layer id");
    }
    const auto primes_per_layer = checked_product(
        max_ngram_size - 1u,
        n_heads,
        "prime count");
    std::vector<DeepseekV41EngramHashLayer> layers(layer_count);
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        if (layer_ids[layer] != expected_layer_ids[layer] ||
            expected_table_rows[layer] <= 0) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram asset layer schedule mismatch");
        }
        layers[layer].layer_id = layer_ids[layer];
        layers[layer].table_rows =
            static_cast<std::uint64_t>(expected_table_rows[layer]);
        layers[layer].multipliers.resize(max_ngram_size);
        for (auto& value : layers[layer].multipliers) {
            value = cursor.scalar<std::uint64_t>("hash multiplier");
        }
    }
    for (auto& layer : layers) {
        layer.primes.resize(primes_per_layer);
        layer.offsets.resize(primes_per_layer);
        std::uint64_t offset = 0;
        for (std::size_t index = 0; index < primes_per_layer; ++index) {
            const auto prime = cursor.scalar<std::uint32_t>("prime bucket");
            if (prime < 2 || offset >
                    std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "invalid DeepSeek-V4.1 Engram prime layout");
            }
            layer.primes[index] = prime;
            layer.offsets[index] = static_cast<std::uint32_t>(offset);
            offset += prime;
        }
        if (offset != layer.table_rows) {
            throw std::runtime_error(
                "DeepSeek-V4.1 Engram prime layout has wrong table size");
        }
    }
    if (cursor.remaining() != 0) {
        throw std::runtime_error(
            "DeepSeek-V4.1 Engram asset has trailing bytes");
    }
    const auto pad_id = token_map[static_cast<std::size_t>(pad_token_id)];
    return DeepseekV41EngramHashState(
        std::move(token_map),
        compressed_vocab_size,
        max_ngram_size,
        n_heads,
        pad_id,
        std::move(layers));
}

std::vector<std::vector<std::uint64_t>>
DeepseekV41EngramHashState::update(
    std::span<const std::int32_t> input_ids,
    int batch,
    int tokens,
    int start_position,
    std::span<const std::uint8_t> token_mask) {
    if (batch <= 0 || tokens <= 0 || start_position < 0 ||
        input_ids.size() != checked_product(
            static_cast<std::size_t>(batch),
            static_cast<std::size_t>(tokens),
            "input size") ||
        (!token_mask.empty() && token_mask.size() != input_ids.size())) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 Engram hash input");
    }
    if (start_position == 0) {
        history_.assign(static_cast<std::size_t>(batch), {});
        batch_ = batch;
        position_ = 0;
    } else if (batch_ != batch || position_ != start_position ||
               history_.size() != static_cast<std::size_t>(batch)) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 Engram hash update is not contiguous");
    }

    for (int b = 0; b < batch; ++b) {
        auto& history = history_[static_cast<std::size_t>(b)];
        history.reserve(static_cast<std::size_t>(start_position + tokens));
        for (int token = 0; token < tokens; ++token) {
            const auto flat = static_cast<std::size_t>(b * tokens + token);
            const auto id = input_ids[flat];
            const bool live = (token_mask.empty() || token_mask[flat] != 0) &&
                id >= 0 && static_cast<std::size_t>(id) < token_map_.size();
            history.push_back(live
                ? static_cast<std::int64_t>(token_map_[static_cast<std::size_t>(id)])
                : kDeadToken);
        }
    }

    const auto columns = n_hash_columns();
    const auto rows = checked_product(
        static_cast<std::size_t>(batch),
        static_cast<std::size_t>(tokens),
        "hash row count");
    std::vector<std::vector<std::uint64_t>> result(
        layers_.size(),
        std::vector<std::uint64_t>(checked_product(rows, columns, "hash output")));
    std::vector<std::uint64_t> ngram_tokens(max_ngram_size_);
    for (int b = 0; b < batch; ++b) {
        const auto& history = history_[static_cast<std::size_t>(b)];
        for (int token = 0; token < tokens; ++token) {
            const auto position = start_position + token;
            bool blocked = false;
            for (std::size_t shift = 0; shift < max_ngram_size_; ++shift) {
                const bool before_start = position < static_cast<int>(shift);
                const auto source = before_start
                    ? kDeadToken
                    : history[static_cast<std::size_t>(position) - shift];
                blocked = blocked || before_start || source == kDeadToken;
                ngram_tokens[shift] = blocked
                    ? pad_id_
                    : static_cast<std::uint64_t>(source);
            }
            const auto output_row = static_cast<std::size_t>(b * tokens + token);
            for (std::size_t layer_index = 0;
                 layer_index < layers_.size(); ++layer_index) {
                const auto& layer = layers_[layer_index];
                std::uint64_t rolling =
                    ngram_tokens[0] * layer.multipliers[0];
                for (std::size_t ngram = 1;
                     ngram < max_ngram_size_; ++ngram) {
                    rolling ^= ngram_tokens[ngram] *
                        layer.multipliers[ngram];
                    for (std::size_t head = 0; head < n_heads_; ++head) {
                        const auto column =
                            (ngram - 1) * n_heads_ + head;
                        result[layer_index][output_row * columns + column] =
                            layer.offsets[column] + positive_remainder(
                                rolling,
                                layer.primes[column]);
                    }
                }
            }
        }
    }
    position_ += tokens;
    return result;
}

void DeepseekV41EngramHashState::reset() noexcept {
    history_.clear();
    batch_ = 0;
    position_ = 0;
}

void DeepseekV41EngramHashState::truncate(int position) {
    if (position < 0 || position > position_ ||
        (position != 0 && batch_ <= 0)) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 Engram hash rollback position");
    }
    if (position == 0) {
        reset();
        return;
    }
    for (auto& history : history_) {
        history.resize(static_cast<std::size_t>(position));
    }
    position_ = position;
}

int DeepseekV41EngramHashState::batch() const noexcept { return batch_; }
int DeepseekV41EngramHashState::position() const noexcept { return position_; }

std::size_t DeepseekV41EngramHashState::n_hash_columns() const noexcept {
    return static_cast<std::size_t>(max_ngram_size_ - 1u) * n_heads_;
}

const std::vector<DeepseekV41EngramHashLayer>&
DeepseekV41EngramHashState::layers() const noexcept {
    return layers_;
}

struct DeepseekV41EngramSsdStore::Impl {
    struct Table {
        HfSafetensorRecord weight;
        HfSafetensorRecord scale;
        std::uint64_t rows = 0;
    };

    struct Key {
        std::uint32_t table = 0;
        std::uint64_t row = 0;

        bool operator==(const Key&) const noexcept = default;
    };

    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept {
            auto value = key.row ^
                (static_cast<std::uint64_t>(key.table) << 56u);
            value ^= value >> 33u;
            value *= 0xff51afd7ed558ccdu;
            value ^= value >> 33u;
            return static_cast<std::size_t>(value);
        }
    };

    struct Entry {
        std::vector<std::uint8_t> raw;
        std::list<Key>::iterator lru;
    };

    struct ColdRow {
        Key key;
        std::vector<std::uint8_t> raw;
    };

    Impl(
        std::shared_ptr<HfSafetensorStore> selected_checkpoint,
        std::span<const std::int64_t> layer_ids,
        std::span<const std::int64_t> table_rows,
        std::size_t selected_head_dim,
        std::size_t selected_cache_bytes,
        std::size_t selected_io_workers)
        : checkpoint(std::move(selected_checkpoint)),
          head_dim(selected_head_dim),
          scale_columns(selected_head_dim / 32),
          row_bytes(selected_head_dim + selected_head_dim / 32),
          cache_limit(selected_cache_bytes),
          io_workers(selected_io_workers),
          max_entries(selected_cache_bytes / (row_bytes + 96)) {
        if (!checkpoint || layer_ids.empty() ||
            layer_ids.size() != table_rows.size() || head_dim == 0 ||
            head_dim % 32 != 0 || cache_limit < row_bytes + 96 ||
            io_workers == 0 ||
            layer_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                "invalid DeepSeek-V4.1 Engram SSD cache configuration");
        }
        tables.reserve(layer_ids.size());
        for (std::size_t index = 0; index < layer_ids.size(); ++index) {
            if (layer_ids[index] < 0 || table_rows[index] <= 0) {
                throw std::invalid_argument(
                    "invalid DeepSeek-V4.1 Engram table schedule");
            }
            const auto prefix = "layers." + std::to_string(layer_ids[index]) +
                ".engram.embed.";
            const auto& weight = checkpoint->tensor(prefix + "weight");
            const auto& scale = checkpoint->tensor(prefix + "scale");
            const std::vector<std::int64_t> weight_shape{
                table_rows[index], static_cast<std::int64_t>(head_dim)};
            const std::vector<std::int64_t> scale_shape{
                table_rows[index], static_cast<std::int64_t>(scale_columns)};
            if ((weight.dtype != "F8_E4M3" &&
                 weight.dtype != "F8_E4M3FN") ||
                scale.dtype != "F8_E8M0" ||
                weight.shape != weight_shape || scale.shape != scale_shape ||
                weight.nbytes != checked_product(
                    static_cast<std::size_t>(table_rows[index]), head_dim,
                    "weight bytes") ||
                scale.nbytes != checked_product(
                    static_cast<std::size_t>(table_rows[index]), scale_columns,
                    "scale bytes")) {
                throw std::runtime_error(
                    "unexpected DeepSeek-V4.1 Engram tensor geometry");
            }
            tables.push_back({
                weight,
                scale,
                static_cast<std::uint64_t>(table_rows[index]),
            });
        }
        cache.reserve(max_entries);
    }

    void read_row(ColdRow& row) const {
        const auto& table = tables.at(row.key.table);
        row.raw.resize(row_bytes);
        checkpoint->read_range(
            table.weight.shard,
            table.weight.offset + row.key.row * head_dim,
            std::span<std::byte>(
                reinterpret_cast<std::byte*>(row.raw.data()),
                head_dim));
        checkpoint->read_range(
            table.scale.shard,
            table.scale.offset + row.key.row * scale_columns,
            std::span<std::byte>(
                reinterpret_cast<std::byte*>(row.raw.data() + head_dim),
                scale_columns));
    }

    void decode(
        std::span<const std::uint8_t> raw,
        std::span<std::uint16_t> output) const {
        if (raw.size() != row_bytes || output.size() != head_dim) {
            throw std::logic_error("invalid Engram row decode geometry");
        }
        const auto& table = engram_fp8_bf16_table();
        for (std::size_t group = 0; group < scale_columns; ++group) {
            const auto lookup_base =
                static_cast<std::size_t>(raw[head_dim + group]) << 8u;
            const auto begin = group * 32u;
            for (std::size_t column = begin; column < begin + 32u; ++column) {
                output[column] = table[lookup_base | raw[column]];
            }
        }
    }

    void touch(typename std::unordered_map<Key, Entry, KeyHash>::iterator found) {
        lru.splice(lru.begin(), lru, found->second.lru);
        found->second.lru = lru.begin();
    }

    void admit(ColdRow row) {
        auto found = cache.find(row.key);
        if (found != cache.end()) {
            touch(found);
            return;
        }
        while (cache.size() >= max_entries) {
            const auto victim = lru.back();
            cache.erase(victim);
            lru.pop_back();
        }
        lru.push_front(row.key);
        cache.emplace(
            row.key,
            Entry{std::move(row.raw), lru.begin()});
    }

    std::shared_ptr<HfSafetensorStore> checkpoint;
    std::vector<Table> tables;
    std::size_t head_dim;
    std::size_t scale_columns;
    std::size_t row_bytes;
    std::size_t cache_limit;
    std::size_t io_workers;
    std::size_t max_entries;
    mutable std::mutex mutex;
    std::list<Key> lru;
    std::unordered_map<Key, Entry, KeyHash> cache;
    DeepseekV41EngramSsdStats counters;
};

DeepseekV41EngramSsdStore::DeepseekV41EngramSsdStore(
    std::shared_ptr<HfSafetensorStore> checkpoint,
    std::span<const std::int64_t> layer_ids,
    std::span<const std::int64_t> table_rows,
    std::size_t head_dim,
    std::size_t cache_bytes,
    std::size_t io_workers)
    : impl_(std::make_unique<Impl>(
          std::move(checkpoint),
          layer_ids,
          table_rows,
          head_dim,
          cache_bytes,
          io_workers)) {}

DeepseekV41EngramSsdStore::~DeepseekV41EngramSsdStore() = default;

std::vector<std::uint16_t> DeepseekV41EngramSsdStore::lookup_bf16(
    std::size_t table_index,
    std::span<const std::uint64_t> row_ids) {
    if (table_index >= impl_->tables.size()) {
        throw std::out_of_range("DeepSeek-V4.1 Engram table index out of range");
    }
    if (row_ids.empty()) {
        return {};
    }
    std::unique_lock lock(impl_->mutex);
    const auto& table = impl_->tables[table_index];
    std::vector<Impl::Key> request_keys;
    request_keys.reserve(row_ids.size());
    std::unordered_map<Impl::Key, std::size_t, Impl::KeyHash> unique_index;
    unique_index.reserve(row_ids.size());
    std::vector<Impl::ColdRow> unique_rows;
    std::vector<bool> cold;
    for (const auto row : row_ids) {
        if (row >= table.rows) {
            throw std::out_of_range(
                "DeepSeek-V4.1 Engram row index out of range");
        }
        const Impl::Key key{
            static_cast<std::uint32_t>(table_index), row};
        request_keys.push_back(key);
        if (unique_index.contains(key)) {
            continue;
        }
        const auto index = unique_rows.size();
        unique_index.emplace(key, index);
        auto found = impl_->cache.find(key);
        if (found != impl_->cache.end()) {
            impl_->touch(found);
            unique_rows.push_back({key, found->second.raw});
            cold.push_back(false);
        } else {
            unique_rows.push_back({key, {}});
            cold.push_back(true);
        }
    }

    std::vector<std::size_t> misses;
    misses.reserve(unique_rows.size());
    for (std::size_t index = 0; index < cold.size(); ++index) {
        if (cold[index]) misses.push_back(index);
    }
    // The request owns copies of all hot rows, so the shared LRU does not
    // need to remain locked while cold rows are read and the complete result
    // is decoded.  In particular this lets the two independent V4.1 Engram
    // tables prefetch concurrently without weakening cache consistency.
    lock.unlock();
    const auto begin = std::chrono::steady_clock::now();
    const auto workers = std::min(impl_->io_workers, misses.size());
    std::vector<std::future<void>> futures;
    futures.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
        futures.push_back(std::async(std::launch::async, [&, worker] {
            for (std::size_t item = worker; item < misses.size(); item += workers) {
                impl_->read_row(unique_rows[misses[item]]);
            }
        }));
    }
    for (auto& future : futures) future.get();
    const auto end = std::chrono::steady_clock::now();

    std::vector<std::uint16_t> result(
        checked_product(row_ids.size(), impl_->head_dim, "lookup output"));
    for (std::size_t request = 0; request < request_keys.size(); ++request) {
        const auto source = unique_index.at(request_keys[request]);
        impl_->decode(
            unique_rows[source].raw,
            std::span<std::uint16_t>(
                result.data() + request * impl_->head_dim,
                impl_->head_dim));
    }
    lock.lock();
    for (const auto index : misses) {
        impl_->admit(std::move(unique_rows[index]));
    }

    impl_->counters.row_requests += row_ids.size();
    impl_->counters.cache_misses += misses.size();
    impl_->counters.cache_hits += row_ids.size() - misses.size();
    impl_->counters.rows_loaded += misses.size();
    impl_->counters.bytes_read += misses.size() * impl_->row_bytes;
    impl_->counters.read_calls += misses.size() * 2;
    impl_->counters.io_seconds +=
        std::chrono::duration<double>(end - begin).count();
    return result;
}

std::size_t DeepseekV41EngramSsdStore::table_count() const noexcept {
    return impl_->tables.size();
}

std::size_t DeepseekV41EngramSsdStore::head_dim() const noexcept {
    return impl_->head_dim;
}

std::size_t DeepseekV41EngramSsdStore::cache_limit_bytes() const noexcept {
    return impl_->cache_limit;
}

DeepseekV41EngramSsdStats DeepseekV41EngramSsdStore::stats() const {
    std::scoped_lock lock(impl_->mutex);
    auto result = impl_->counters;
    result.resident_rows = impl_->cache.size();
    result.resident_payload_bytes = impl_->cache.size() * impl_->row_bytes;
    result.cache_limit_bytes = impl_->cache_limit;
    return result;
}

void DeepseekV41EngramSsdStore::clear() {
    std::scoped_lock lock(impl_->mutex);
    impl_->cache.clear();
    impl_->lru.clear();
}

} // namespace mfq::metal
