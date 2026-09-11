#include "mlx_deepseek_v41_engram.h"

#include <mlx/allocator.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

int checked_int(std::int64_t value, const char* name) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            std::string("invalid DeepSeek-V4.1 Engram ") + name);
    }
    return static_cast<int>(value);
}

array bf16_array(
    std::span<const std::uint16_t> bits,
    Shape shape) {
    const auto expected = static_cast<std::size_t>(
        std::accumulate(
            shape.begin(), shape.end(), std::int64_t{1},
            [](std::int64_t left, int right) { return left * right; }));
    if (bits.size() != expected) {
        throw std::logic_error(
            "DeepSeek-V4.1 Engram BF16 array shape mismatch");
    }
    auto buffer = mlx::core::allocator::malloc(bits.size() * sizeof(std::uint16_t));
    std::memcpy(buffer.raw_ptr(), bits.data(), bits.size() * sizeof(std::uint16_t));
    return array(std::move(buffer), std::move(shape), mlx::core::bfloat16);
}

array float32(const array& value) {
    return value.dtype() == mlx::core::float32
        ? mlx::core::contiguous(value)
        : mlx::core::contiguous(
              mlx::core::astype(value, mlx::core::float32));
}

} // namespace

struct MlxDeepseekV41HfEngram::Impl {
    struct Layer {
        std::size_t model_layer = 0;
        std::size_t table_index = 0;
        MlxLinear wkv;
        array q_weight;
        array k_weight;
    };

    Impl(
        DeepseekV4Config selected_config,
        DeepseekV41EngramHashState selected_hash,
        std::shared_ptr<DeepseekV41EngramSsdStore> selected_store,
        std::vector<Layer> selected_layers)
        : config(std::move(selected_config)),
          hash(std::move(selected_hash)),
          store(std::move(selected_store)),
          layers(std::move(selected_layers)) {
        layer_lookup.assign(
            static_cast<std::size_t>(config.n_layers),
            std::numeric_limits<std::size_t>::max());
        for (std::size_t index = 0; index < layers.size(); ++index) {
            layer_lookup[layers[index].model_layer] = index;
        }
    }

    const Layer& layer(std::size_t model_layer) const {
        if (model_layer >= layer_lookup.size() ||
            layer_lookup[model_layer] == std::numeric_limits<std::size_t>::max()) {
            throw std::out_of_range(
                "DeepSeek-V4.1 layer has no Engram module");
        }
        return layers[layer_lookup[model_layer]];
    }

    DeepseekV4Config config;
    DeepseekV41EngramHashState hash;
    std::shared_ptr<DeepseekV41EngramSsdStore> store;
    std::vector<Layer> layers;
    std::vector<std::size_t> layer_lookup;
};

MlxDeepseekV41HfEngram MlxDeepseekV41HfEngram::load_hf(
    const MlxHfTensorStore& model,
    const DeepseekV4Config& config,
    const std::filesystem::path& hash_asset,
    std::size_t cache_bytes,
    std::size_t io_workers) {
    config.validate();
    if (!config.is_v41() || !config.has_engram()) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 Engram loader requires an Engram model");
    }
    auto hash = DeepseekV41EngramHashState::load(
        hash_asset,
        static_cast<std::uint32_t>(config.vocab),
        static_cast<std::uint32_t>(config.engram_compressed_vocab_size),
        static_cast<std::uint32_t>(config.engram_max_ngram_size),
        static_cast<std::uint32_t>(config.engram_n_heads),
        static_cast<std::int32_t>(config.engram_pad_token_id),
        config.engram_layer_ids,
        config.engram_num_embeddings);
    auto store = std::make_shared<DeepseekV41EngramSsdStore>(
        model.shared_checkpoint(),
        config.engram_layer_ids,
        config.engram_num_embeddings,
        static_cast<std::size_t>(config.engram_head_dim),
        cache_bytes,
        io_workers);
    std::vector<Impl::Layer> layers;
    layers.reserve(config.engram_layer_ids.size());
    for (std::size_t table = 0; table < config.engram_layer_ids.size(); ++table) {
        const auto model_layer =
            static_cast<std::size_t>(config.engram_layer_ids[table]);
        const auto prefix = DeepseekV4TensorNames::layer(
            model_layer, "engram.");
        layers.push_back({
            model_layer,
            table,
            model.load_linear(prefix + "key_value.weight"),
            float32(model.load_dense(prefix + "query.weight")),
            float32(model.load_dense(prefix + "key.weight")),
        });
    }
    return MlxDeepseekV41HfEngram(std::make_shared<Impl>(
        config,
        std::move(hash),
        std::move(store),
        std::move(layers)));
}

MlxDeepseekV41HfEngram::MlxDeepseekV41HfEngram(
    std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MlxDeepseekV41HfEngramBatch MlxDeepseekV41HfEngram::prepare(
    const array& token_ids,
    int start_position,
    bool prefetch_rows) {
    if (token_ids.ndim() != 2 || token_ids.shape(0) <= 0 ||
        token_ids.shape(1) <= 0 || token_ids.dtype() != mlx::core::int32) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 Engram expects int32 token IDs [batch,tokens]");
    }
    auto ids = mlx::core::contiguous(token_ids);
    mlx::core::eval(ids);
    const auto count = ids.size();
    std::vector<std::int32_t> host_ids(count);
    std::memcpy(host_ids.data(), ids.data<std::int32_t>(),
                count * sizeof(std::int32_t));
    MlxDeepseekV41HfEngramBatch result;
    result.batch = ids.shape(0);
    result.tokens = ids.shape(1);
    result.start_position = start_position;
    result.token_mask.resize(count);
    std::transform(
        host_ids.begin(), host_ids.end(), result.token_mask.begin(),
        [vocab = impl_->config.vocab](std::int32_t token) {
            return static_cast<std::uint8_t>(
                token >= 0 && token < vocab);
        });
    result.row_ids = impl_->hash.update(
        host_ids,
        result.batch,
        result.tokens,
        start_position,
        result.token_mask);
    if (prefetch_rows) {
        result.prefetched_rows.reserve(result.row_ids.size());
        for (std::size_t table = 0; table < result.row_ids.size(); ++table) {
            auto requested = result.row_ids[table];
            result.prefetched_rows.push_back(
                std::async(
                    std::launch::async,
                    [store = impl_->store,
                     table,
                     requested = std::move(requested)] {
                        return store->lookup_bf16(table, requested);
                    })
                    .share());
        }
    }
    return result;
}

bool MlxDeepseekV41HfEngram::has_layer(std::size_t layer) const noexcept {
    return layer < impl_->layer_lookup.size() &&
        impl_->layer_lookup[layer] != std::numeric_limits<std::size_t>::max();
}

array MlxDeepseekV41HfEngram::apply(
    std::size_t model_layer,
    const array& hidden,
    const MlxDeepseekV41HfEngramBatch& batch) const {
    const auto& layer = impl_->layer(model_layer);
    const int connections = checked_int(impl_->config.hc_mult, "HC count");
    const int dimension = checked_int(impl_->config.hidden, "hidden size");
    const int columns = checked_int(
        impl_->config.engram_n_heads *
            (impl_->config.engram_max_ngram_size - 1),
        "hash columns");
    const int row_width = checked_int(
        impl_->config.engram_head_dim,
        "embedding width");
    if (hidden.shape() != Shape{
            batch.batch, batch.tokens, connections, dimension} ||
        batch.row_ids.size() != impl_->layers.size() ||
        batch.token_mask.size() !=
            static_cast<std::size_t>(batch.batch * batch.tokens) ||
        batch.row_ids[layer.table_index].size() !=
            static_cast<std::size_t>(batch.batch * batch.tokens * columns)) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 Engram batch geometry mismatch");
    }
    std::vector<std::uint16_t> loaded_rows;
    std::span<const std::uint16_t> rows;
    if (batch.prefetched_rows.empty()) {
        loaded_rows = impl_->store->lookup_bf16(
            layer.table_index,
            batch.row_ids[layer.table_index]);
        rows = loaded_rows;
    } else {
        if (batch.prefetched_rows.size() != impl_->layers.size() ||
            !batch.prefetched_rows[layer.table_index].valid()) {
            throw std::logic_error(
                "DeepSeek-V4.1 Engram prefetch state is malformed");
        }
        rows = batch.prefetched_rows[layer.table_index].get();
    }
    auto embeddings = bf16_array(
        rows,
        Shape{batch.batch, batch.tokens, columns, row_width});
    embeddings = mlx::core::reshape(
        std::move(embeddings),
        Shape{batch.batch, batch.tokens, columns * row_width});
    auto kv = layer.wkv(embeddings);
    auto key = mlx::core::reshape(
        mlx::core::slice(
            kv,
            Shape{0, 0, 0},
            Shape{batch.batch, batch.tokens, connections * dimension}),
        Shape{batch.batch, batch.tokens, connections, dimension});
    auto value = mlx::core::slice(
        kv,
        Shape{0, 0, connections * dimension},
        Shape{batch.batch, batch.tokens, (connections + 1) * dimension});

    auto h = mlx::core::astype(hidden, mlx::core::float32);
    key = mlx::core::astype(key, mlx::core::float32);
    auto weight = mlx::core::reshape(
        layer.q_weight * layer.k_weight,
        Shape{1, 1, connections, dimension});
    const auto eps = static_cast<float>(impl_->config.rms_eps);
    auto inverse = mlx::core::rsqrt(
        mlx::core::mean(h * h, -1) + eps) *
        mlx::core::rsqrt(mlx::core::mean(key * key, -1) + eps);
    auto dot = mlx::core::sum(h * weight * key, -1) * inverse /
        std::sqrt(static_cast<float>(dimension));
    auto magnitude = mlx::core::sqrt(
        mlx::core::maximum(mlx::core::abs(dot), array(1e-6f)));
    auto gate = mlx::core::sigmoid(mlx::core::where(
        mlx::core::less(dot, array(0.0f)),
        -magnitude,
        magnitude));
    std::vector<float> mask(batch.token_mask.begin(), batch.token_mask.end());
    auto mask_array = mlx::core::reshape(
        array(mask.begin(), Shape{batch.batch, batch.tokens}),
        Shape{batch.batch, batch.tokens, 1});
    gate = gate * mask_array;
    auto result = h +
        mlx::core::expand_dims(gate, -1) *
        mlx::core::expand_dims(
            mlx::core::astype(value, mlx::core::float32), -2);
    return hidden.dtype() == mlx::core::float32
        ? result
        : mlx::core::astype(result, hidden.dtype());
}

void MlxDeepseekV41HfEngram::reset_hash() noexcept {
    impl_->hash.reset();
}

void MlxDeepseekV41HfEngram::truncate_hash(int position) {
    impl_->hash.truncate(position);
}

void MlxDeepseekV41HfEngram::restore_text_hash(
    std::span<const std::int64_t> token_ids) {
    impl_->hash.reset();
    if (token_ids.empty()) return;
    std::vector<std::int32_t> ids;
    ids.reserve(token_ids.size());
    for (const auto token : token_ids) {
        if (token < 0 || token > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 Engram session token is out of range");
        }
        ids.push_back(static_cast<std::int32_t>(token));
    }
    (void)impl_->hash.update(
        ids,
        1,
        static_cast<int>(ids.size()),
        0);
}

DeepseekV41EngramSsdStats MlxDeepseekV41HfEngram::stats() const {
    return impl_->store->stats();
}

void MlxDeepseekV41HfEngram::clear_rows() {
    impl_->store->clear();
}

} // namespace mfq::metal
