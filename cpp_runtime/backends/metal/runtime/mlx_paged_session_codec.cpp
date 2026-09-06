#include "mlx_paged_session_codec.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <mlx/mlx.h>

namespace mfq::metal {
namespace {

using mlx::core::Dtype;
using mlx::core::Shape;
using mlx::core::array;

constexpr std::array<std::uint8_t, 8> kMagic{
    'M', 'F', 'Q', 'M', 'L', 'X', '1', 0};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint8_t kKvLayer = 1;
constexpr std::uint8_t kRecurrentLayer = 2;
constexpr std::uint32_t kMiniRuntime = 1;
constexpr std::uint32_t kQwen35Runtime = 2;

class Writer {
public:
    template <typename T>
    void scalar(T value) {
        using Unsigned = std::make_unsigned_t<T>;
        const auto converted = static_cast<Unsigned>(value);
        for (std::size_t index = 0; index < sizeof(T); ++index) {
            bytes_.push_back(static_cast<std::uint8_t>(
                converted >> (index * 8)));
        }
    }

    void raw(const void* data, std::size_t size) {
        const auto* begin = static_cast<const std::uint8_t*>(data);
        bytes_.insert(bytes_.end(), begin, begin + size);
    }

    void tensor(const array& source) {
        auto value = mlx::core::contiguous(source);
        value.eval();
        scalar<std::uint8_t>(
            static_cast<std::uint8_t>(value.dtype().val()));
        if (value.ndim() == 0 || value.ndim() > 8) {
            throw std::runtime_error("invalid MLX cache tensor rank");
        }
        scalar<std::uint8_t>(static_cast<std::uint8_t>(value.ndim()));
        scalar<std::uint16_t>(0);
        for (const auto dimension : value.shape()) {
            if (dimension <= 0) {
                throw std::runtime_error("invalid MLX cache tensor shape");
            }
            scalar<std::int32_t>(dimension);
        }
        scalar<std::uint64_t>(value.nbytes());
        raw(value.data<std::uint8_t>(), value.nbytes());
    }

    std::shared_ptr<const std::vector<std::uint8_t>> finish() && {
        return std::make_shared<const std::vector<std::uint8_t>>(
            std::move(bytes_));
    }

private:
    std::vector<std::uint8_t> bytes_;
};

class Reader {
public:
    explicit Reader(const std::vector<std::uint8_t>& bytes)
        : cursor_(bytes.data()), end_(bytes.data() + bytes.size()) {}

    template <typename T>
    T scalar(const char* name) {
        if (remaining() < sizeof(T)) {
            throw std::runtime_error(
                std::string("truncated MLX cache ") + name);
        }
        using Unsigned = std::make_unsigned_t<T>;
        Unsigned converted = 0;
        for (std::size_t index = 0; index < sizeof(T); ++index) {
            converted |= static_cast<Unsigned>(cursor_[index]) << (index * 8);
        }
        cursor_ += sizeof(T);
        return static_cast<T>(converted);
    }

    void raw(void* destination, std::size_t size, const char* name) {
        if (remaining() < size) {
            throw std::runtime_error(
                std::string("truncated MLX cache ") + name);
        }
        std::memcpy(destination, cursor_, size);
        cursor_ += size;
    }

    array tensor() {
        const auto dtype_value = scalar<std::uint8_t>("dtype");
        const auto rank = scalar<std::uint8_t>("rank");
        (void)scalar<std::uint16_t>("tensor flags");
        if (rank == 0 || rank > 8 ||
            dtype_value > static_cast<std::uint8_t>(Dtype::Val::complex64)) {
            throw std::runtime_error("invalid MLX cache tensor header");
        }
        Shape shape;
        shape.reserve(rank);
        std::size_t elements = 1;
        for (std::uint8_t index = 0; index < rank; ++index) {
            const auto dimension = scalar<std::int32_t>("shape");
            if (dimension <= 0 ||
                elements > std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(dimension)) {
                throw std::runtime_error("invalid MLX cache tensor shape");
            }
            shape.push_back(dimension);
            elements *= static_cast<std::size_t>(dimension);
        }
        const auto dtype = dtype_from_value(
            static_cast<Dtype::Val>(dtype_value));
        const auto bytes = scalar<std::uint64_t>("tensor size");
        if (bytes != elements * dtype.size() || bytes > remaining()) {
            throw std::runtime_error("invalid MLX cache tensor size");
        }
        auto result = array(
            mlx::core::allocator::malloc(static_cast<std::size_t>(bytes)),
            std::move(shape),
            dtype);
        raw(result.data<std::uint8_t>(), static_cast<std::size_t>(bytes), "tensor");
        return result;
    }

    void expect_end() const {
        if (cursor_ != end_) {
            throw std::runtime_error("trailing MLX cache payload bytes");
        }
    }

    std::size_t remaining() const noexcept {
        return static_cast<std::size_t>(end_ - cursor_);
    }

private:
    static Dtype dtype_from_value(Dtype::Val value) {
        switch (value) {
            case Dtype::Val::bool_: return mlx::core::bool_;
            case Dtype::Val::uint8: return mlx::core::uint8;
            case Dtype::Val::uint16: return mlx::core::uint16;
            case Dtype::Val::uint32: return mlx::core::uint32;
            case Dtype::Val::uint64: return mlx::core::uint64;
            case Dtype::Val::int8: return mlx::core::int8;
            case Dtype::Val::int16: return mlx::core::int16;
            case Dtype::Val::int32: return mlx::core::int32;
            case Dtype::Val::int64: return mlx::core::int64;
            case Dtype::Val::float16: return mlx::core::float16;
            case Dtype::Val::float32: return mlx::core::float32;
            case Dtype::Val::float64: return mlx::core::float64;
            case Dtype::Val::bfloat16: return mlx::core::bfloat16;
            case Dtype::Val::complex64: return mlx::core::complex64;
        }
        throw std::runtime_error("unsupported MLX cache tensor dtype");
    }

    const std::uint8_t* cursor_;
    const std::uint8_t* end_;
};

struct DecodedLayer {
    std::uint8_t kind = 0;
    int batch = 0;
    int heads = 0;
    int maximum_sequence = 0;
    int head_dimension = 0;
    int capacity = 0;
    int position = 0;
    Dtype dtype = mlx::core::float16;
    std::optional<array> first;
    std::optional<array> second;
};

struct DecodedBlock {
    std::uint32_t runtime = 0;
    std::uint32_t start = 0;
    std::uint32_t count = 0;
    std::vector<DecodedLayer> layers;
};

void write_header(
    Writer& writer,
    std::uint32_t runtime,
    std::uint32_t start,
    std::uint32_t count,
    std::size_t layers) {
    writer.raw(kMagic.data(), kMagic.size());
    writer.scalar<std::uint32_t>(kVersion);
    writer.scalar<std::uint32_t>(runtime);
    writer.scalar<std::uint32_t>(start);
    writer.scalar<std::uint32_t>(count);
    writer.scalar<std::uint32_t>(static_cast<std::uint32_t>(layers));
}

void write_kv_layer(
    Writer& writer,
    const MlxKvCacheSnapshot& snapshot,
    int start,
    int count) {
    if (snapshot.position < start + count || snapshot.batch != 1) {
        throw std::runtime_error("invalid MLX KV cache block range");
    }
    writer.scalar<std::uint8_t>(kKvLayer);
    writer.scalar<std::uint8_t>(
        static_cast<std::uint8_t>(snapshot.dtype.val()));
    writer.scalar<std::uint16_t>(0);
    writer.scalar<std::int32_t>(snapshot.batch);
    writer.scalar<std::int32_t>(snapshot.heads);
    writer.scalar<std::int32_t>(snapshot.maximum_sequence);
    writer.scalar<std::int32_t>(snapshot.head_dimension);
    writer.scalar<std::int32_t>(snapshot.capacity);
    writer.scalar<std::int32_t>(start + count);
    const Shape begin{0, 0, start, 0};
    const Shape end{
        snapshot.batch,
        snapshot.heads,
        start + count,
        snapshot.head_dimension,
    };
    writer.tensor(mlx::core::slice(snapshot.key, begin, end));
    writer.tensor(mlx::core::slice(snapshot.value, begin, end));
}

void write_recurrent_layer(
    Writer& writer,
    const MlxQwen35LinearAttentionCacheSnapshot& snapshot,
    int boundary) {
    if (snapshot.batch != 1 || snapshot.position < boundary) {
        throw std::runtime_error("invalid recurrent cache boundary");
    }
    writer.scalar<std::uint8_t>(kRecurrentLayer);
    writer.scalar<std::uint8_t>(0);
    writer.scalar<std::uint16_t>(0);
    writer.scalar<std::int32_t>(snapshot.batch);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(0);
    writer.scalar<std::int32_t>(boundary);
    writer.tensor(snapshot.convolution_state);
    writer.tensor(snapshot.recurrent_state);
}

DecodedBlock read_block(const std::vector<std::uint8_t>& payload) {
    Reader reader(payload);
    std::array<std::uint8_t, 8> magic{};
    reader.raw(magic.data(), magic.size(), "magic");
    const auto version = reader.scalar<std::uint32_t>("version");
    if (magic != kMagic || version != kVersion) {
        throw std::runtime_error("unsupported MLX cache payload");
    }
    DecodedBlock block;
    block.runtime = reader.scalar<std::uint32_t>("runtime");
    block.start = reader.scalar<std::uint32_t>("start");
    block.count = reader.scalar<std::uint32_t>("count");
    const auto layer_count = reader.scalar<std::uint32_t>("layer count");
    if (block.count == 0 || layer_count == 0 || layer_count > 4096) {
        throw std::runtime_error("invalid MLX cache block geometry");
    }
    block.layers.reserve(layer_count);
    for (std::uint32_t index = 0; index < layer_count; ++index) {
        DecodedLayer layer;
        layer.kind = reader.scalar<std::uint8_t>("layer kind");
        const auto dtype_value = reader.scalar<std::uint8_t>("layer dtype");
        (void)reader.scalar<std::uint16_t>("layer flags");
        layer.batch = reader.scalar<std::int32_t>("batch");
        layer.heads = reader.scalar<std::int32_t>("heads");
        layer.maximum_sequence =
            reader.scalar<std::int32_t>("maximum sequence");
        layer.head_dimension =
            reader.scalar<std::int32_t>("head dimension");
        layer.capacity = reader.scalar<std::int32_t>("capacity");
        layer.position = reader.scalar<std::int32_t>("position");
        layer.first.emplace(reader.tensor());
        layer.second.emplace(reader.tensor());
        if (layer.kind == kKvLayer) {
            if (dtype_value >
                static_cast<std::uint8_t>(Dtype::Val::complex64)) {
                throw std::runtime_error("invalid MLX KV layer dtype");
            }
            layer.dtype = layer.first->dtype();
            if (layer.dtype.val() != static_cast<Dtype::Val>(dtype_value)) {
                throw std::runtime_error("MLX KV layer dtype mismatch");
            }
        } else if (layer.kind != kRecurrentLayer) {
            throw std::runtime_error("unknown MLX cache layer kind");
        }
        block.layers.push_back(std::move(layer));
    }
    reader.expect_end();
    return block;
}

std::vector<DecodedBlock> decode_blocks(
    const std::vector<std::vector<std::uint8_t>>& payloads,
    std::uint32_t expected_runtime,
    std::size_t token_count,
    std::size_t block_size) {
    if (payloads.empty() || token_count != payloads.size() * block_size) {
        throw std::runtime_error("MLX cache block chain length mismatch");
    }
    std::vector<DecodedBlock> blocks;
    blocks.reserve(payloads.size());
    std::size_t expected_start = 0;
    std::size_t expected_layers = 0;
    for (const auto& payload : payloads) {
        auto block = read_block(payload);
        if (block.runtime != expected_runtime || block.start != expected_start ||
            block.count != block_size ||
            (expected_layers != 0 && block.layers.size() != expected_layers)) {
            throw std::runtime_error("incompatible MLX cache block chain");
        }
        expected_start += block.count;
        expected_layers = block.layers.size();
        blocks.push_back(std::move(block));
    }
    return blocks;
}

MlxKvCacheSnapshot rebuild_kv(
    const std::vector<DecodedBlock>& blocks,
    std::size_t layer_index,
    std::size_t token_count) {
    std::vector<array> keys;
    std::vector<array> values;
    keys.reserve(blocks.size());
    values.reserve(blocks.size());
    const auto& final = blocks.back().layers.at(layer_index);
    for (const auto& block : blocks) {
        const auto& layer = block.layers.at(layer_index);
        if (layer.kind != kKvLayer || layer.batch != final.batch ||
            layer.heads != final.heads ||
            layer.maximum_sequence != final.maximum_sequence ||
            layer.head_dimension != final.head_dimension ||
            layer.dtype != final.dtype || !layer.first || !layer.second) {
            throw std::runtime_error("inconsistent MLX KV cache block topology");
        }
        keys.push_back(*layer.first);
        values.push_back(*layer.second);
    }
    auto key = mlx::core::concatenate(std::move(keys), 2);
    auto value = mlx::core::concatenate(std::move(values), 2);
    mlx::core::eval(key, value);
    const int position = static_cast<int>(token_count);
    return MlxKvCacheSnapshot{
        final.batch,
        final.heads,
        final.maximum_sequence,
        final.head_dimension,
        std::max(final.capacity, position),
        position,
        final.dtype,
        std::move(key),
        std::move(value),
    };
}

MlxQwen35LinearAttentionCacheSnapshot rebuild_recurrent(
    const std::vector<DecodedBlock>& blocks,
    std::size_t layer_index,
    std::size_t token_count) {
    const auto& final = blocks.back().layers.at(layer_index);
    if (final.kind != kRecurrentLayer || final.batch != 1 ||
        !final.first || !final.second) {
        throw std::runtime_error("invalid recurrent cache boundary block");
    }
    return MlxQwen35LinearAttentionCacheSnapshot{
        *final.first,
        *final.second,
        static_cast<int>(token_count),
        final.batch,
    };
}

template <typename State, typename LayerWriter>
std::vector<MlxPagedPayload> encode_state(
    const State& state,
    std::size_t block_size,
    std::size_t first_block,
    std::uint32_t runtime,
    LayerWriter write_layer) {
    if (block_size == 0 || state.cache_batch != 1 ||
        state.cache_position <= 0 ||
        state.tokens.size() != static_cast<std::size_t>(state.cache_position) ||
        state.layers.empty()) {
        throw std::runtime_error("invalid MLX session state for paging");
    }
    const auto full_blocks = state.tokens.size() / block_size;
    if (first_block > full_blocks) {
        throw std::runtime_error("invalid MLX cache first block");
    }
    std::vector<MlxPagedPayload> result;
    result.reserve(full_blocks - first_block);
    for (std::size_t block = first_block; block < full_blocks; ++block) {
        const auto start = block * block_size;
        Writer writer;
        write_header(
            writer,
            runtime,
            static_cast<std::uint32_t>(start),
            static_cast<std::uint32_t>(block_size),
            state.layers.size());
        for (const auto& layer : state.layers) {
            write_layer(writer, layer, start, block_size);
        }
        result.push_back(std::move(writer).finish());
    }
    return result;
}

} // namespace

std::vector<MlxPagedPayload>
MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::encode(
    const MlxMiniCPMO45TextSessionState& state,
    std::size_t block_size,
    std::size_t first_block) {
    return encode_state(
        state,
        block_size,
        first_block,
        kMiniRuntime,
        [](Writer& writer, const MlxKvCacheSnapshot& layer,
           std::size_t start, std::size_t count) {
            write_kv_layer(
                writer,
                layer,
                static_cast<int>(start),
                static_cast<int>(count));
        });
}

MlxMiniCPMO45TextSessionState
MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::decode(
    const std::vector<std::vector<std::uint8_t>>& payloads,
    const std::vector<std::int64_t>& tokens,
    std::size_t block_size) {
    auto blocks = decode_blocks(
        payloads, kMiniRuntime, tokens.size(), block_size);
    MlxMiniCPMO45TextSessionState state;
    state.tokens = tokens;
    state.cache_position = static_cast<int>(tokens.size());
    state.cache_batch = 1;
    state.layers.reserve(blocks.front().layers.size());
    for (std::size_t index = 0; index < blocks.front().layers.size(); ++index) {
        auto layer = rebuild_kv(blocks, index, tokens.size());
        state.bytes += layer.nbytes();
        state.layers.push_back(std::move(layer));
    }
    return state;
}

std::vector<MlxPagedPayload>
MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode(
    const MlxQwen35TextSessionState& state,
    std::size_t block_size,
    std::size_t first_block) {
    return encode_state(
        state,
        block_size,
        first_block,
        kQwen35Runtime,
        [](Writer& writer, const MlxQwen35LayerCacheSnapshot& layer,
           std::size_t start, std::size_t count) {
            std::visit(
                [&](const auto& snapshot) {
                    using Snapshot = std::decay_t<decltype(snapshot)>;
                    if constexpr (std::is_same_v<Snapshot, MlxKvCacheSnapshot>) {
                        write_kv_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start),
                            static_cast<int>(count));
                    } else {
                        write_recurrent_layer(
                            writer,
                            snapshot,
                            static_cast<int>(start + count));
                    }
                },
                layer);
        });
}

MlxQwen35TextSessionState
MlxPagedSessionCodec<MlxQwen35TextSessionState>::decode(
    const std::vector<std::vector<std::uint8_t>>& payloads,
    const std::vector<std::int64_t>& tokens,
    std::size_t block_size) {
    auto blocks = decode_blocks(
        payloads, kQwen35Runtime, tokens.size(), block_size);
    MlxQwen35TextSessionState state;
    state.tokens = tokens;
    state.cache_position = static_cast<int>(tokens.size());
    state.cache_batch = 1;
    state.layers.reserve(blocks.front().layers.size());
    for (std::size_t index = 0; index < blocks.front().layers.size(); ++index) {
        if (blocks.front().layers[index].kind == kKvLayer) {
            auto layer = rebuild_kv(blocks, index, tokens.size());
            state.bytes += layer.nbytes();
            state.layers.emplace_back(std::move(layer));
        } else {
            auto layer = rebuild_recurrent(blocks, index, tokens.size());
            state.bytes += layer.nbytes();
            state.layers.emplace_back(std::move(layer));
        }
    }
    return state;
}

} // namespace mfq::metal
