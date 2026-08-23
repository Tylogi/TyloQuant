#pragma once

#include "hf_safetensors_store.h"
#include "mlx_moe.h"
#include "mlx_mx.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV4SsdExpertWeights {
    MlxRoutedLinear gate;
    MlxRoutedLinear up;
    MlxRoutedLinear down;
};

// Six-bank unified-memory arena shared by SSD prefill buffers and the decode
// expert cache. The buffers are allocated by MLX, remain CPU writable, and are
// consumed directly by Metal without a staging copy.
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
        std::size_t slot) const;

    std::size_t slots_ = 0;
    Bank w1_scale_;
    Bank w2_scale_;
    Bank w3_scale_;
    Bank w1_weight_;
    Bank w2_weight_;
    Bank w3_weight_;
};

} // namespace mfq::metal
