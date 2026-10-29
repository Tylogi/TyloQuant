#pragma once

#include "storage/weight_loader.h"

#include "mfe_expert_store.h"
#include "models/deepseek_v41/config.h"
#include "cpp_runtime/models/deepseek_v41/engram.h"
#include "models/deepseek_v41/causal_lm.h"
#include "quant_linear.h"
#include "mfq/kernels/hip/deepseek_v41.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mfq::cuda::deepseek_v41_runtime {

using EngramTensor = mfq_tensor_backend::Tensor;
using EngramConfig = mfq::models::deepseek_v41::Config;

namespace engram_detail {

constexpr std::array<std::uint8_t, 8> kMagic{'M', 'F', 'Q', 'E', 'N', 'G', 'R', '1'};
constexpr std::size_t kMxHeaderBytes = 56;
constexpr std::int64_t kDeadToken = -1;

class ByteCursor {
  public:
    explicit ByteCursor(const std::vector<std::uint8_t> &bytes) : bytes_(bytes) {}

    const std::uint8_t *take(std::size_t count, std::string_view description) {
        if (offset_ > bytes_.size() || count > bytes_.size() - offset_) {
            throw std::runtime_error("truncated DeepSeek-V4.1 Engram " + std::string(description));
        }
        const auto *result = bytes_.data() + offset_;
        offset_ += count;
        return result;
    }

    std::uint32_t u32(std::string_view description) {
        const auto *bytes = take(4, description);
        std::uint32_t result = 0;
        for (int index = 0; index < 4; ++index) {
            result |= static_cast<std::uint32_t>(bytes[index]) << (8 * index);
        }
        return result;
    }

    std::uint64_t u64(std::string_view description) {
        const auto *bytes = take(8, description);
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
            throw std::runtime_error("DeepSeek-V4.1 Engram asset has trailing bytes");
        }
    }

  private:
    const std::vector<std::uint8_t> &bytes_;
    std::size_t offset_ = 0;
};

template <typename T, typename Reader>
std::vector<T> read_vector(ByteCursor &cursor, std::size_t count, Reader &&reader,
                           std::string_view description) {
    std::vector<T> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        result.push_back(reader(cursor, description));
    }
    return result;
}

inline std::uint64_t checked_product(std::uint64_t left, std::uint64_t right,
                                     std::string_view description) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::runtime_error("DeepSeek-V4.1 Engram " + std::string(description) +
                                 " byte size overflows");
    }
    return left * right;
}

inline std::uint64_t read_u64_at(const std::vector<std::uint8_t> &bytes, std::size_t offset) {
    if (offset > bytes.size() || 8 > bytes.size() - offset) {
        throw std::runtime_error("truncated DeepSeek-V4.1 Engram MX header");
    }
    std::uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(bytes[offset + index]) << (8 * index);
    }
    return value;
}

inline float decode_e4m3(std::uint8_t raw) {
    if ((raw & 0x7fu) == 0x7fu) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float sign = (raw & 0x80u) ? -1.0f : 1.0f;
    const int exponent = (raw >> 3u) & 0x0fu;
    const int mantissa = raw & 0x07u;
    return sign * (exponent == 0
                       ? std::ldexp(static_cast<float>(mantissa), -9)
                       : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7));
}

} // namespace engram_detail

using EngramHashBatch = mfq::models::deepseek_v41::EngramHashBatch;
class EngramHashState : public mfq::models::deepseek_v41::EngramHashState {
    using HashState = mfq::models::deepseek_v41::EngramHashState;

  public:
    using HashState::HashState;
    static std::unique_ptr<EngramHashState> load(const mfq::ModelSource &model,
                                                 const EngramConfig &config) {
        constexpr const char *asset_name = "__mfq_asset__/deepseek-v41-engram-v1.bin";
        if (!model.has_asset(asset_name)) {
            throw std::runtime_error("DeepSeek-V4.1 model source has no Engram hash asset");
        }
        const auto asset = read_asset(model, asset_name);
        engram_detail::ByteCursor cursor(asset);
        const auto *magic = cursor.take(engram_detail::kMagic.size(), "asset magic");
        if (!std::equal(engram_detail::kMagic.begin(), engram_detail::kMagic.end(), magic)) {
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
            max_ngram > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) || heads == 0 ||
            heads > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("invalid DeepSeek-V4.1 Engram asset dimensions");
        }
        const auto layers = static_cast<std::size_t>(layer_count);
        const auto orders = static_cast<std::size_t>(max_ngram - 1);
        const auto head_count = static_cast<std::size_t>(heads);
        const auto bucket_count = engram_detail::checked_product(
            engram_detail::checked_product(layers, orders, "bucket"), head_count, "bucket");
        const auto multiplier_count = engram_detail::checked_product(
            layers, static_cast<std::size_t>(max_ngram), "multiplier");
        auto layer_ids = engram_detail::read_vector<std::int64_t>(
            cursor, layers,
            [](auto &value, auto what) { return static_cast<std::int64_t>(value.i32(what)); },
            "layer ids");
        auto table_rows = engram_detail::read_vector<std::int64_t>(
            cursor, layers, [](auto &value, auto what) { return value.i64(what); }, "table rows");
        auto primes = engram_detail::read_vector<std::int64_t>(
            cursor, static_cast<std::size_t>(bucket_count),
            [](auto &value, auto what) { return value.i64(what); }, "bucket primes");
        auto offsets = engram_detail::read_vector<std::int64_t>(
            cursor, static_cast<std::size_t>(bucket_count),
            [](auto &value, auto what) { return value.i64(what); }, "bucket offsets");
        auto multipliers = engram_detail::read_vector<std::int64_t>(
            cursor, static_cast<std::size_t>(multiplier_count),
            [](auto &value, auto what) { return value.i64(what); }, "hash multipliers");
        auto token_map = engram_detail::read_vector<std::int32_t>(
            cursor, vocabulary, [](auto &value, auto what) { return value.i32(what); },
            "compressed token map");
        cursor.require_end();

        if (vocabulary != static_cast<std::uint32_t>(config.vocab) ||
            compressed != static_cast<std::uint32_t>(config.engram_compressed_vocab_size) ||
            max_ngram != static_cast<std::uint32_t>(config.engram_max_ngram_size) ||
            heads != static_cast<std::uint32_t>(config.engram_n_heads) ||
            layer_ids != config.engram_layer_ids || table_rows != config.engram_num_embeddings ||
            pad_id < 0 || pad_id >= static_cast<std::int64_t>(compressed)) {
            throw std::runtime_error("DeepSeek-V4.1 Engram asset disagrees with model config");
        }
        for (std::size_t layer = 0; layer < layers; ++layer) {
            const auto last = layer * orders * head_count + orders * head_count - 1;
            if (primes[last] <= 0 || offsets[last] < 0 ||
                primes[last] + offsets[last] != table_rows[layer]) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 Engram bucket layout disagrees with table rows");
            }
        }
        if (std::any_of(token_map.begin(), token_map.end(),
                        [compressed](std::int32_t value) {
                            return value < 0 || value >= static_cast<std::int32_t>(compressed);
                        }) ||
            std::any_of(multipliers.begin(), multipliers.end(),
                        [](std::int64_t value) { return value <= 0 || !(value & 1); })) {
            throw std::runtime_error("DeepSeek-V4.1 Engram hash metadata is invalid");
        }
        return std::unique_ptr<EngramHashState>(new EngramHashState(
            static_cast<int>(vocabulary), static_cast<int>(max_ngram), static_cast<int>(heads),
            pad_id, std::move(layer_ids), std::move(table_rows), std::move(primes),
            std::move(offsets), std::move(multipliers), std::move(token_map)));
    }

    EngramHashBatch forward(const EngramTensor &token_ids, std::int64_t pos0,
                            const std::uint8_t* image_mask = nullptr) {
        if (token_ids.dim() != 2 || token_ids.size(0) <= 0 || token_ids.size(1) <= 0 || pos0 < 0 ||
            token_ids.size(0) > std::numeric_limits<int>::max() ||
            token_ids.size(1) > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 Engram token IDs must have nonempty [B,T] shape");
        }
        const int batch = static_cast<int>(token_ids.size(0));
        const int tokens = static_cast<int>(token_ids.size(1));
        auto host_ids =
            token_ids.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
        return HashState::forward(host_ids.data_ptr<int64_t>(), batch, tokens, pos0,image_mask);
    }

    static void self_check() {
        const std::vector<std::int64_t> expected{
            3, 8, 15, 26, 3, 8, 15, 26, 3, 8, 16, 27, 3, 8, 14, 23,
        };
        const auto make_state = [] {
            return EngramHashState(8, 3, 2, 0, {1}, {36}, {5, 7, 11, 13}, {0, 5, 12, 23}, {3, 5, 7},
                                   {0, 1, 2, 3, 4, 5, 6, 7});
        };
        const auto options = mfq_tensor_backend::TensorOptions()
                                 .device(mfq_tensor_backend::kCPU)
                                 .dtype(mfq_tensor_backend::kInt64);
        auto all_ids = mfq_tensor_backend::tensor(std::vector<std::int64_t>{1, 2, 3, 4}, options)
                           .reshape({1, 4});
        auto full_state = make_state();
        const auto full = full_state.forward(all_ids, 0);
        MFQ_RUNTIME_CHECK(full.values == expected,
                          "DeepSeek-V4.1 Engram full hash contract failed");

        auto chunked_state = make_state();
        const auto first = chunked_state.forward(all_ids.narrow(1, 0, 2), 0);
        const auto second = chunked_state.forward(all_ids.narrow(1, 2, 2), 2);
        std::vector<std::int64_t> chunked = first.values;
        chunked.insert(chunked.end(), second.values.begin(), second.values.end());
        MFQ_RUNTIME_CHECK(chunked == expected, "DeepSeek-V4.1 Engram chunked hash contract failed");

        auto speculative_state = make_state();
        (void)speculative_state.forward(all_ids.narrow(1, 0, 2), 0);
        speculative_state.begin_speculative();
        const auto speculative = speculative_state.forward(all_ids.narrow(1, 2, 2), 2);
        speculative_state.rollback_speculative();
        const auto replayed = speculative_state.forward(all_ids.narrow(1, 2, 2), 2);
        MFQ_RUNTIME_CHECK(speculative.values == replayed.values,
                          "DeepSeek-V4.1 Engram speculative rollback failed");
        MFQ_RUNTIME_CHECK(engram_detail::decode_e4m3(0x00) == 0.0f &&
                              engram_detail::decode_e4m3(0x38) == 1.0f &&
                              engram_detail::decode_e4m3(0x40) == 2.0f &&
                              engram_detail::decode_e4m3(0xb8) == -1.0f &&
                              engram_detail::decode_e4m3(0x01) == std::ldexp(1.0f, -9),
                          "DeepSeek-V4.1 Engram E4M3 decode contract failed");
    }
};

class EngramTable {
  public:
    static std::shared_ptr<EngramTable> load(const mfq::ModelSource&, const std::string&,
        std::int64_t expected_rows, int expected_width, std::size_t cache_capacity_rows);
    ~EngramTable();
    void prefetch(const std::int64_t* rows, std::size_t count) const;
    EngramTensor gather(const std::int64_t* rows, int batch, int tokens, int hash_columns,
                        const mfq_tensor_backend::Device& device) const;
  private:
    EngramTable(const mfq::ModelSource&, std::string, std::uint64_t nbytes,
                std::int64_t rows, int width, std::uint64_t value_bytes, std::size_t capacity,
                std::shared_ptr<mfq::NintRows> nint={});
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Engram {
  public:
    static std::unique_ptr<Engram> load(CudaExecutionContext &execution,
                                        const mfq::ModelSource &model, const EngramConfig &config,
                                        int layer) {
        const auto found =
            std::find(config.engram_layer_ids.begin(), config.engram_layer_ids.end(), layer);
        if (found == config.engram_layer_ids.end()) {
            throw std::invalid_argument("DeepSeek-V4.1 layer has no Engram component");
        }
        const auto hash_layer =
            static_cast<int>(std::distance(config.engram_layer_ids.begin(), found));
        const auto prefix = "model.block." + std::to_string(layer) + ".associative_memory.";
        auto query = load_dense_gpu(execution, model, prefix + "query.weight")
                         .to(mfq_tensor_backend::kFloat32)
                         .contiguous();
        auto key = load_dense_gpu(execution, model, prefix + "key.weight")
                       .to(mfq_tensor_backend::kFloat32)
                       .contiguous();
        if (query.dim() != 2 || query.size(0) != config.hc_mult || query.size(1) != config.hidden ||
            key.sizes() != query.sizes()) {
            throw std::runtime_error("DeepSeek-V4.1 Engram query/key geometry disagrees");
        }
        auto projection = load_quant_linear(execution, model, prefix + "projection.weight");
        const auto hash_columns = (config.engram_max_ngram_size - 1) * config.engram_n_heads;
        if (projection.neuron_len() != hash_columns * config.engram_head_dim ||
            projection.out() != (config.hc_mult + 1) * config.hidden) {
            throw std::runtime_error("DeepSeek-V4.1 Engram projection geometry disagrees");
        }
        return std::unique_ptr<Engram>(new Engram(
            config, hash_layer,
            EngramTable::load(model, prefix + "embedding.weight",
                              config.engram_num_embeddings[static_cast<std::size_t>(hash_layer)],
                              static_cast<int>(config.engram_head_dim),
                              execution.config.deepseek_v41_engram_cache_rows),
            std::move(projection), (query * key).contiguous()));
    }

    int64_t layer_id() const { return config_.engram_layer_ids[hash_layer_]; }

    void prefetch(const EngramHashBatch& hashes,const std::vector<int64_t>* text_rows=nullptr) const {
        if(text_rows) {
            if(text_rows->empty())return;
            const auto selected=select_hash_rows(hashes,*text_rows);
            table_->prefetch(selected.layer_data(hash_layer_),std::size_t(selected.tokens)*selected.columns);
            return;
        }
        table_->prefetch(hashes.layer_data(hash_layer_),
            std::size_t(hashes.batch) * hashes.tokens * hashes.columns);
    }

    EngramTensor forward(CudaExecutionContext &execution, const EngramTensor &hidden,
                         const EngramHashBatch &hashes,const EngramTensor& text_rows={},
                         const std::vector<int64_t>* host_rows=nullptr) const {
        if(text_rows.defined()) {
            if(!text_rows.numel())return hidden;
            MFQ_RUNTIME_CHECK(host_rows && host_rows->size()==std::size_t(text_rows.numel()),
                              "DeepSeek-V4.1 Engram image participation is incomplete");
            auto flat=hidden.reshape({1,-1,config_.hc_mult,config_.hidden});
            auto selected=forward(execution,flat.index_select(1,text_rows).contiguous(),
                                  select_hash_rows(hashes,*host_rows));
            auto result=flat.clone();result.index_copy_(1,text_rows,selected);
            return result.reshape(hidden.sizes());
        }
        const auto batch = hashes.batch;
        const auto tokens = hashes.tokens;
        if (hidden.dim() != 4 || hidden.size(0) != batch || hidden.size(1) != tokens ||
            hidden.size(2) != config_.hc_mult || hidden.size(3) != config_.hidden ||
            hashes.layers != static_cast<int>(config_.engram_layer_ids.size()) ||
            hashes.columns != (config_.engram_max_ngram_size - 1) * config_.engram_n_heads) {
            throw std::invalid_argument("DeepSeek-V4.1 Engram activation geometry disagrees");
        }
        auto embeddings = execution.profiler.measure("deepseek_v41.engram.rows", [&] {
            return table_->gather(hashes.layer_data(hash_layer_), batch, tokens,
                                  hashes.columns, hidden.device());
        });
        auto projected = execution.profiler.measure("deepseek_v41.engram.projection", [&] {
            return projection_.forward(execution, embeddings);
        });
        return execution.profiler.measure("deepseek_v41.engram.gate_add", [&] {
            return deepseek_v41_engram_gate_add_cuda(hidden, projected.contiguous(),
                                                       query_key_weight_, config_.rms_eps);
        });
    }

  private:
    EngramHashBatch select_hash_rows(const EngramHashBatch& hashes,const std::vector<int64_t>& rows) const {
        EngramHashBatch selected;
        selected.layers=hashes.layers;selected.columns=hashes.columns;
        selected.batch=1;selected.tokens=int(rows.size());
        selected.values.assign(std::size_t(hashes.layers)*rows.size()*hashes.columns,0);
        const auto* source=hashes.layer_data(hash_layer_);
        auto* target=selected.values.data()+std::size_t(hash_layer_)*rows.size()*hashes.columns;
        for(const auto row:rows) {
            MFQ_RUNTIME_CHECK(row>=0 && row<int64_t(hashes.batch)*hashes.tokens,
                              "DeepSeek-V4.1 Engram text row is out of range");
            std::copy_n(source+row*hashes.columns,hashes.columns,target);target+=hashes.columns;
        }
        return selected;
    }
    Engram(EngramConfig config, int hash_layer, std::shared_ptr<EngramTable> table,
           QuantLinear projection, EngramTensor query_key_weight)
        : config_(std::move(config)), hash_layer_(hash_layer), table_(std::move(table)),
          projection_(std::move(projection)), query_key_weight_(std::move(query_key_weight)) {}

    EngramConfig config_;
    int hash_layer_ = -1;
    std::shared_ptr<EngramTable> table_;
    QuantLinear projection_;
    EngramTensor query_key_weight_;
};

} // namespace mfq::cuda::deepseek_v41_runtime
