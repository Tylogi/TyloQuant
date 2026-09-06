#pragma once

#include "mlx_minicpmo45.h"
#include "mlx_qwen35_causal_lm.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace mfq::metal {

using MlxPagedPayload =
    std::shared_ptr<const std::vector<std::uint8_t>>;

template <typename SessionState>
struct MlxPagedSessionCodec {
    static constexpr bool available = false;
};

template <>
struct MlxPagedSessionCodec<MlxMiniCPMO45TextSessionState> {
    static constexpr bool available = true;
    static constexpr std::string_view name = "minicpmo45-qwen3-kv-v1";

    static std::vector<MlxPagedPayload> encode(
        const MlxMiniCPMO45TextSessionState& state,
        std::size_t block_size,
        std::size_t first_block = 0);
    static MlxMiniCPMO45TextSessionState decode(
        const std::vector<std::vector<std::uint8_t>>& payloads,
        const std::vector<std::int64_t>& tokens,
        std::size_t block_size);
};

template <>
struct MlxPagedSessionCodec<MlxQwen35TextSessionState> {
    // Hybrid recurrent checkpoints must be captured at each block boundary.
    // A final-state snapshot cannot safely reconstruct an earlier boundary.
    static constexpr bool available = false;
    static constexpr std::string_view name = "qwen35-hybrid-kv-v1";

    static std::vector<MlxPagedPayload> encode(
        const MlxQwen35TextSessionState& state,
        std::size_t block_size,
        std::size_t first_block = 0);
    static MlxQwen35TextSessionState decode(
        const std::vector<std::vector<std::uint8_t>>& payloads,
        const std::vector<std::int64_t>& tokens,
        std::size_t block_size);
};

} // namespace mfq::metal
