#pragma once

#include "hf_safetensors_store.h"
#include "mlx_moe.h"
#include "mlx_mx.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV4SsdExpertWeights {
    MlxRoutedLinear gate_up;
    MlxRoutedLinear down;
};

// Four-bank unified-memory arena shared by SSD prefill buffers and the decode
// expert cache. The buffers are allocated by MLX, remain CPU writable, and are
// consumed directly by Metal without a staging copy. Gate and Up are adjacent
// in each slot so one routed MXFP4 kernel can project and apply SwiGLU.
class MlxDeepseekV4SsdExpertArena {
public:
    explicit MlxDeepseekV4SsdExpertArena(std::size_t slots);

    std::size_t slots() const noexcept;
    std::size_t bytes_per_slot() const noexcept;
    std::size_t nbytes() const noexcept;

    // Bind every large native bank to Metal once so resource mapping and
    // first-use kernel compilation are paid during model load, not TTFT.
    void prewarm_metal();

    DeepseekV4NativeExpertDestination destination(std::size_t slot);

    MlxDeepseekV4SsdExpertWeights routed_weights(
        const std::vector<std::int32_t>& slot_for_expert,
        std::span<const std::int32_t> active_experts) const;

    // Immutable identity view used by decode after the six routed expert IDs
    // have been remapped to arena slots on the host. It avoids rebuilding two
    // descriptor tables at every layer boundary.
    const MlxDeepseekV4SsdExpertWeights& slot_weights() const noexcept;

    MlxMxWeight expert_weight(
        std::size_t slot,
        char projection) const;

private:
    struct Bank {
        mlx::core::array array = mlx::core::array(0.0f);
        std::byte* data = nullptr;
        std::size_t bytes_per_slot = 0;
    };

    static Bank allocate_bank(
        std::size_t slots,
        std::size_t bytes_per_slot);
    std::span<std::byte> bank_slot(Bank& bank, std::size_t slot);
    mlx::core::array bank_slot_array(
        const Bank& bank,
        std::size_t slot,
        std::size_t offset = 0,
        std::size_t bytes = 0) const;

    std::size_t slots_ = 0;
    Bank gate_up_scale_;
    Bank w2_scale_;
    Bank gate_up_weight_;
    Bank w2_weight_;
    std::unique_ptr<MlxDeepseekV4SsdExpertWeights> slot_weights_;
};

} // namespace mfq::metal
