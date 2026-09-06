#include "mlx_paged_session_codec.h"

#include <cstring>
#include <iostream>
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
    std::vector<std::vector<std::uint8_t>> payloads;
    for (const auto& payload : encoded) payloads.push_back(*payload);
    const auto decoded =
        MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState>::decode(
            payloads, mini.tokens, 4);
    require(decoded.tokens == mini.tokens, "MiniCPM decoded tokens mismatch");
    require(decoded.layers.size() == 1, "MiniCPM decoded layers mismatch");
    require(byte_equal(decoded.layers[0].key, mini.layers[0].key),
            "MiniCPM decoded key bytes changed");
    require(byte_equal(decoded.layers[0].value, mini.layers[0].value),
            "MiniCPM decoded value bytes changed");

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
    payloads.clear();
    for (const auto& payload : hybrid_encoded) payloads.push_back(*payload);
    const auto hybrid_decoded =
        MlxPagedSessionCodec<MlxQwen35TextSessionState>::decode(
            payloads, hybrid.tokens, 4);
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

    std::cout << "MFQ MLX paged session codec tests passed\n";
    return 0;
}
