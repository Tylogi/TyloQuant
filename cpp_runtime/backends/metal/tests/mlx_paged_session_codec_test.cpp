#include "mlx_paged_session_codec.h"

#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

mlx::core::array fp16_values(int count, int offset = 0) {
    std::vector<float> values(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        values[static_cast<std::size_t>(index)] =
            static_cast<float>(index + offset) / 16.0f;
    }
    return mlx::core::astype(
        mlx::core::array(
            values.begin(), mlx::core::Shape{1, 2, count / 4, 2}),
        mlx::core::float16);
}

bool byte_equal(const mlx::core::array& left, const mlx::core::array& right) {
    auto evaluated_left = left;
    auto evaluated_right = right;
    evaluated_left.eval();
    evaluated_right.eval();
    return evaluated_left.shape() == evaluated_right.shape() &&
        evaluated_left.dtype() == evaluated_right.dtype() &&
        evaluated_left.nbytes() == evaluated_right.nbytes() &&
        std::memcmp(
            evaluated_left.data<std::uint8_t>(),
            evaluated_right.data<std::uint8_t>(),
            evaluated_left.nbytes()) == 0;
}

void run_codec_benchmark() {
    using namespace mfq::metal;
    constexpr int tokens = 1536;
    constexpr int block_size = 256;
    constexpr int layers = 24;
    constexpr int heads = 8;
    constexpr int head_dim = 64;
    auto key = mlx::core::zeros(
        mlx::core::Shape{1, heads, tokens, head_dim}, mlx::core::float16);
    auto value = mlx::core::ones(
        mlx::core::Shape{1, heads, tokens, head_dim}, mlx::core::float16);
    mlx::core::eval(key, value);
    MlxMiniCPMO45TextSessionState state;
    state.tokens.resize(tokens);
    for (int token = 0; token < tokens; ++token) state.tokens[token] = token;
    state.cache_position = tokens;
    state.cache_batch = 1;
    for (int layer = 0; layer < layers; ++layer) {
        state.layers.push_back(MlxKvCacheSnapshot{
            1,
            heads,
            tokens,
            head_dim,
            tokens,
            tokens,
            mlx::core::float16,
            key,
            value,
        });
        state.bytes += state.layers.back().nbytes();
    }
    const auto encoded =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::encode(
            state, block_size);
    const auto started = std::chrono::steady_clock::now();
    auto decoded =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::decode(
            encoded, state.tokens, block_size);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    require(decoded.bytes == state.bytes, "codec benchmark byte count mismatch");
    std::cout << "paged_codec_benchmark bytes=" << state.bytes
              << " blocks=" << encoded.size()
              << " layers=" << layers
              << " elapsed_ms=" << elapsed << '\n';

    auto copied =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::decode(
            encoded, state.tokens, block_size);
    const auto copy_started = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<MlxKvCache>> copied_caches;
    for (const auto& layer : copied.layers) {
        auto cache = std::make_unique<MlxKvCache>(
            layer.batch,
            layer.heads,
            layer.maximum_sequence,
            layer.head_dimension,
            layer.position,
            layer.dtype);
        cache->restore_snapshot(layer);
        copied_caches.push_back(std::move(cache));
    }
    const auto copy_elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - copy_started).count();
    const auto adopt_started = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<MlxKvCache>> adopted_caches;
    for (auto& layer : decoded.layers) {
        auto cache = std::make_unique<MlxKvCache>(
            layer.batch,
            layer.heads,
            layer.maximum_sequence,
            layer.head_dimension,
            layer.position,
            layer.dtype);
        cache->restore_snapshot(std::move(layer));
        adopted_caches.push_back(std::move(cache));
    }
    const auto adopt_elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - adopt_started).count();
    std::cout << "paged_codec_restore_benchmark bytes=" << state.bytes
              << " copy_ms=" << copy_elapsed
              << " adopt_ms=" << adopt_elapsed << '\n';
}

} // namespace

int main() {
    using namespace mfq::metal;
    MlxMiniCPMO45TextSessionState mini;
    mini.tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    mini.cache_position = 8;
    mini.cache_batch = 1;
    mini.layers.push_back(MlxKvCacheSnapshot{
        1,
        2,
        32,
        2,
        16,
        8,
        mlx::core::float16,
        fp16_values(32),
        fp16_values(32, 100),
    });
    mini.bytes = mini.layers.front().nbytes();
    const auto encoded =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::encode(mini, 4);
    require(encoded.size() == 2, "MiniCPM block encode count mismatch");
    const auto encoded_second =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::encode_block(
            mini, 4, 1);
    require(*encoded_second == *encoded[1],
            "MiniCPM one-block encoding changed bytes");
    const auto decoded =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::decode(
            encoded, mini.tokens, 4);
    require(decoded.tokens == mini.tokens, "MiniCPM decoded tokens mismatch");
    require(decoded.layers.size() == 1, "MiniCPM decoded layers mismatch");
    require(byte_equal(decoded.layers[0].key, mini.layers[0].key),
            "MiniCPM decoded key bytes changed");
    require(byte_equal(decoded.layers[0].value, mini.layers[0].value),
            "MiniCPM decoded value bytes changed");
    auto adopted =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::decode(
            encoded, mini.tokens, 4);
    const auto adopted_key = adopted.layers[0].key.buffer().ptr();
    MlxKvCache resumed(1, 2, 32, 2, 1, mlx::core::float16);
    resumed.restore_snapshot(std::move(adopted.layers[0]));
    require(resumed.key_storage().buffer().ptr() == adopted_key &&
                resumed.position() == 8 && resumed.capacity() == 8,
            "decoded KV cache was not adopted directly");
    auto appended = resumed.append(fp16_values(4, 200), fp16_values(4, 300));
    mlx::core::eval(appended.first, appended.second);
    require(resumed.position() == 9 && resumed.capacity() >= 9,
            "adopted KV cache did not grow on append");

    MlxQwen35TextSessionState hybrid;
    hybrid.tokens = mini.tokens;
    hybrid.cache_position = 8;
    hybrid.cache_batch = 1;
    hybrid.layers.emplace_back(mini.layers.front());
    auto convolution = mlx::core::astype(
        mlx::core::array({1.0f, 2.0f}, mlx::core::Shape{1, 2}),
        mlx::core::float16);
    auto recurrent = mlx::core::astype(
        mlx::core::array({3.0f, 4.0f}, mlx::core::Shape{1, 2}),
        mlx::core::float16);
    hybrid.layers.emplace_back(MlxQwen35LinearAttentionCacheSnapshot{
        convolution,
        recurrent,
        8,
        1,
    });
    const auto hybrid_encoded =
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode(hybrid, 4);
    const auto hybrid_second =
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::encode_block(
            hybrid, 4, 1);
    require(*hybrid_second == *hybrid_encoded[1],
            "hybrid one-block encoding changed bytes");
    const auto hybrid_decoded =
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::decode(
            hybrid_encoded, hybrid.tokens, 4);
    require(hybrid_decoded.layers.size() == 2,
            "hybrid decoded layer count mismatch");
    const auto& decoded_recurrent =
        std::get<MlxQwen35LinearAttentionCacheSnapshot>(
            hybrid_decoded.layers[1]);
    require(byte_equal(decoded_recurrent.convolution_state, convolution),
            "recurrent convolution bytes changed");
    require(byte_equal(decoded_recurrent.recurrent_state, recurrent),
            "recurrent state bytes changed");
    require(decoded_recurrent.position == 8,
            "recurrent cache position mismatch");

    if (const auto* enabled = std::getenv("MFQ_PAGED_CODEC_BENCHMARK");
        enabled != nullptr && enabled[0] == '1') {
        run_codec_benchmark();
    }

    std::cout << "MFQ MLX paged session codec tests passed\n";
    return 0;
}
