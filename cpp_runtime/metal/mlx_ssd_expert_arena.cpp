#include "mlx_ssd_expert_arena.h"

#include <mlx/allocator.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

constexpr std::size_t kValueBytes = 2048ull * 2048ull;
constexpr std::size_t kScaleBytes = 2048ull * 128ull;

int checked_dimension(std::size_t value, const char* name) {
    if (value == 0 || value > static_cast<std::size_t>(
        std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(
            std::string("invalid SSD expert arena ") + name);
    }
    return static_cast<int>(value);
}

int checked_index(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(
        std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(
            std::string("invalid SSD expert arena ") + name);
    }
    return static_cast<int>(value);
}

std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* name) {
    if (left == 0 || right == 0 ||
        left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::invalid_argument(
            std::string("invalid SSD expert arena ") + name);
    }
    return left * right;
}

} // namespace

MlxDeepseekV4SsdExpertArena::Bank
MlxDeepseekV4SsdExpertArena::allocate_bank(
    std::size_t slots,
    std::size_t bytes_per_slot) {
    const auto bytes = checked_product(slots, bytes_per_slot, "bank size");
    auto buffer = mlx::core::allocator::malloc(bytes);
    auto* data = static_cast<std::byte*>(buffer.raw_ptr());
    if (data == nullptr) {
        mlx::core::allocator::free(buffer);
        throw std::bad_alloc();
    }
    auto value = array(
        buffer,
        Shape{
            checked_dimension(slots, "slot count"),
            checked_dimension(bytes_per_slot, "slot stride"),
        },
        mlx::core::uint8);
    return {
        .array = std::move(value),
        .data = data,
        .bytes_per_slot = bytes_per_slot,
    };
}

MlxDeepseekV4SsdExpertArena::MlxDeepseekV4SsdExpertArena(
    std::size_t slots)
    : slots_(slots),
      gate_up_scale_(allocate_bank(slots, 2 * kScaleBytes)),
      w2_scale_(allocate_bank(slots, kScaleBytes)),
      gate_up_weight_(allocate_bank(slots, 2 * kValueBytes)),
      w2_weight_(allocate_bank(slots, kValueBytes)) {
    std::vector<std::int32_t> identity(slots);
    std::iota(identity.begin(), identity.end(), 0);
    const auto experts = checked_dimension(slots, "slot count");
    slot_weights_ = std::make_unique<MlxDeepseekV4SsdExpertWeights>(
        MlxDeepseekV4SsdExpertWeights{
            .gate_up = MlxRoutedLinear(
                MlxNintMoeWeight::from_mxfp4_slots(
                    experts,
                    4096,
                    4096,
                    identity,
                    gate_up_weight_.array,
                    gate_up_scale_.array)),
            .down = MlxRoutedLinear(
                MlxNintMoeWeight::from_mxfp4_slots(
                    experts,
                    4096,
                    2048,
                    identity,
                    w2_weight_.array,
                    w2_scale_.array)),
        });
}

std::size_t MlxDeepseekV4SsdExpertArena::slots() const noexcept {
    return slots_;
}

std::size_t MlxDeepseekV4SsdExpertArena::bytes_per_slot() const noexcept {
    return 3 * (kValueBytes + kScaleBytes);
}

std::size_t MlxDeepseekV4SsdExpertArena::nbytes() const noexcept {
    return slots_ * bytes_per_slot();
}

void MlxDeepseekV4SsdExpertArena::prewarm_metal() {
    auto empty = destination(0);
    for (const auto bytes : {
             empty.w1_scale,
             empty.w2_scale,
             empty.w3_scale,
             empty.w1_weight,
             empty.w2_weight,
             empty.w3_weight,
         }) {
        std::memset(bytes.data(), 0, bytes.size());
    }
    constexpr std::array<std::int32_t, 1> active{0};
    const auto& weights = slot_weights();
    const mlx::core::array expert_ids(
        active.data(), Shape{1, 1}, mlx::core::int32);
    auto gate_up = weights.gate_up.swiglu(
        mlx::core::zeros(Shape{1, 4096}, mlx::core::float16),
        expert_ids,
        0.0f);
    auto down = weights.down.forward(
        mlx::core::zeros(Shape{1, 2048}, mlx::core::float16),
        expert_ids);
    mlx::core::eval({std::move(gate_up), std::move(down)});
}

const MlxDeepseekV4SsdExpertWeights&
MlxDeepseekV4SsdExpertArena::slot_weights() const noexcept {
    return *slot_weights_;
}

std::span<std::byte> MlxDeepseekV4SsdExpertArena::bank_slot(
    Bank& bank,
    std::size_t slot) {
    if (slot >= slots_) {
        throw std::out_of_range("SSD expert arena slot out of range");
    }
    return {
        bank.data + slot * bank.bytes_per_slot,
        bank.bytes_per_slot,
    };
}

DeepseekV4NativeExpertDestination
MlxDeepseekV4SsdExpertArena::destination(std::size_t slot) {
    auto gate_up_scale = bank_slot(gate_up_scale_, slot);
    auto gate_up_weight = bank_slot(gate_up_weight_, slot);
    return {
        .w1_scale = gate_up_scale.first(kScaleBytes),
        .w2_scale = bank_slot(w2_scale_, slot),
        .w3_scale = gate_up_scale.subspan(kScaleBytes, kScaleBytes),
        .w1_weight = gate_up_weight.first(kValueBytes),
        .w2_weight = bank_slot(w2_weight_, slot),
        .w3_weight = gate_up_weight.subspan(kValueBytes, kValueBytes),
    };
}

MlxDeepseekV4SsdExpertWeights
MlxDeepseekV4SsdExpertArena::routed_weights(
    const std::vector<std::int32_t>& slot_for_expert,
    std::span<const std::int32_t> active_experts) const {
    if (slot_for_expert.empty() || active_experts.empty()) {
        throw std::invalid_argument(
            "SSD expert routed view requires active experts");
    }
    std::int32_t fallback = -1;
    for (const auto expert : active_experts) {
        if (expert < 0 || static_cast<std::size_t>(expert) >=
                slot_for_expert.size()) {
            throw std::out_of_range("active SSD expert ID out of range");
        }
        const auto slot = slot_for_expert[static_cast<std::size_t>(expert)];
        if (slot < 0 || static_cast<std::size_t>(slot) >= slots_) {
            throw std::runtime_error("active SSD expert is not resident");
        }
        fallback = slot;
    }
    std::vector<std::int32_t> complete = slot_for_expert;
    for (auto& slot : complete) {
        if (slot < 0) {
            slot = fallback;
        } else if (static_cast<std::size_t>(slot) >= slots_) {
            throw std::runtime_error("SSD expert slot map is invalid");
        }
    }
    const int experts = checked_dimension(complete.size(), "expert count");
    return {
        .gate_up = MlxRoutedLinear(
            MlxNintMoeWeight::from_mxfp4_slots(
                experts,
                4096,
                4096,
                complete,
                gate_up_weight_.array,
                gate_up_scale_.array)),
        .down = MlxRoutedLinear(
            MlxNintMoeWeight::from_mxfp4_slots(
                experts,
                4096,
                2048,
                complete,
                w2_weight_.array,
                w2_scale_.array)),
    };
}

array MlxDeepseekV4SsdExpertArena::bank_slot_array(
    const Bank& bank,
    std::size_t slot,
    std::size_t offset,
    std::size_t bytes) const {
    if (slot >= slots_) {
        throw std::out_of_range("SSD expert arena slot out of range");
    }
    if (bytes == 0) {
        bytes = bank.bytes_per_slot - offset;
    }
    if (offset > bank.bytes_per_slot ||
        bytes > bank.bytes_per_slot - offset) {
        throw std::out_of_range("SSD expert arena bank slice out of range");
    }
    return mlx::core::reshape(
        mlx::core::slice(
            bank.array,
            Shape{
                checked_index(slot, "slot index"),
                checked_index(offset, "slot byte offset"),
            },
            Shape{
                checked_index(slot + 1, "slot end"),
                checked_index(offset + bytes, "slot byte end"),
            }),
        Shape{checked_dimension(bytes, "slot slice")});
}

MlxMxWeight MlxDeepseekV4SsdExpertArena::expert_weight(
    std::size_t slot,
    char projection) const {
    if (projection == '1') {
        return MlxMxWeight::from_arrays(
            "MXFP4",
            bank_slot_array(gate_up_weight_, slot, 0, kValueBytes),
            bank_slot_array(gate_up_scale_, slot, 0, kScaleBytes),
            4096,
            2048);
    }
    if (projection == '2') {
        return MlxMxWeight::from_arrays(
            "MXFP4",
            bank_slot_array(w2_weight_, slot),
            bank_slot_array(w2_scale_, slot),
            2048,
            4096);
    }
    if (projection == '3') {
        return MlxMxWeight::from_arrays(
            "MXFP4",
            bank_slot_array(
                gate_up_weight_, slot, kValueBytes, kValueBytes),
            bank_slot_array(
                gate_up_scale_, slot, kScaleBytes, kScaleBytes),
            4096,
            2048);
    }
    throw std::invalid_argument("MXFP4 projection must be '1', '2', or '3'");
}

} // namespace mfq::metal
