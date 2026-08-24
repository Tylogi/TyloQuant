#include "mlx_ssd_expert_arena.h"
#include "mlx_moe_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using mlx::core::Shape;
using mlx::core::array;

void fill_slot(
    mfq::metal::MlxDeepseekV4SsdExpertArena& arena,
    std::size_t slot,
    std::uint8_t code) {
    auto destination = arena.destination(slot);
    for (const auto bytes : {
             destination.w1_scale,
             destination.w2_scale,
             destination.w3_scale,
         }) {
        std::fill(bytes.begin(), bytes.end(), std::byte{127});
    }
    for (const auto bytes : {
             destination.w1_weight,
             destination.w2_weight,
             destination.w3_weight,
         }) {
        std::fill(bytes.begin(), bytes.end(), std::byte{code});
    }
}

void compare(const array& left, const array& right, const char* name) {
    auto lhs = mlx::core::contiguous(
        mlx::core::astype(left, mlx::core::float32));
    auto rhs = mlx::core::contiguous(
        mlx::core::astype(right, mlx::core::float32));
    mlx::core::eval({lhs, rhs});
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error(std::string(name) + " shape mismatch");
    }
    const auto* a = lhs.data<float>();
    const auto* b = rhs.data<float>();
    float maximum = 0.0f;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        maximum = std::max(maximum, std::abs(a[index] - b[index]));
    }
    if (maximum != 0.0f) {
        throw std::runtime_error(
            std::string(name) + " max error " + std::to_string(maximum));
    }
}

array make_input(int rows, int columns) {
    std::vector<float> values(
        static_cast<std::size_t>(rows) * columns);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<float>(
            static_cast<int>(index % 17) - 8) / 16.0f;
    }
    return mlx::core::astype(
        array(values.begin(), Shape{rows, columns}),
        mlx::core::float16);
}

} // namespace

int main() {
    try {
        mfq::metal::MlxDeepseekV4SsdExpertArena arena(2);
        // 0x22 is (+1,+1); 0x44 is (+2,+2) in E2M1 MXFP4.
        fill_slot(arena, 0, 0x22);
        fill_slot(arena, 1, 0x44);
        const std::vector<std::int32_t> slot_map{0, 1};
        const std::vector<std::int32_t> active{0, 1};
        auto weights = arena.routed_weights(slot_map, active);
        const array ids(
            std::vector<std::int32_t>{0, 1}.begin(),
            Shape{2, 1});

        auto input = make_input(2, 4096);
        auto gate_up = weights.gate_up.swiglu(input, ids, 0.0f);
        auto slot_gate_up = arena.slot_weights().gate_up.swiglu(
            input,
            ids,
            0.0f);
        auto gate0 = arena.expert_weight(0, '1').matmul(
            mlx::core::slice(input, Shape{0, 0}, Shape{1, 4096}));
        auto gate1 = arena.expert_weight(1, '1').matmul(
            mlx::core::slice(input, Shape{1, 0}, Shape{2, 4096}));
        auto up0 = arena.expert_weight(0, '3').matmul(
            mlx::core::slice(input, Shape{0, 0}, Shape{1, 4096}));
        auto up1 = arena.expert_weight(1, '3').matmul(
            mlx::core::slice(input, Shape{1, 0}, Shape{2, 4096}));
        auto gate_reference = mlx::core::expand_dims(
            mlx::core::concatenate({gate0, gate1}, 0),
            1);
        auto up_reference = mlx::core::expand_dims(
            mlx::core::concatenate({up0, up1}, 0),
            1);
        auto swiglu_reference =
            mfq::metal::moe_limited_swiglu_split(
                mlx::core::concatenate(
                    {gate_reference, up_reference},
                    -1),
                0.0f);
        compare(gate_up, swiglu_reference, "SSD MXFP4 Gate/Up SwiGLU");
        compare(slot_gate_up, gate_up, "SSD identity-slot Gate/Up SwiGLU");

        auto down_input = make_input(2, 2048);
        auto routed_down_input = mlx::core::expand_dims(down_input, 1);
        auto down = weights.down.forward(routed_down_input, ids);
        auto slot_down = arena.slot_weights().down.forward(
            routed_down_input,
            ids);
        auto down0 = arena.expert_weight(0, '2').matmul(
            mlx::core::slice(down_input, Shape{0, 0}, Shape{1, 2048}));
        auto down1 = arena.expert_weight(1, '2').matmul(
            mlx::core::slice(down_input, Shape{1, 0}, Shape{2, 2048}));
        auto down_reference = mlx::core::expand_dims(
            mlx::core::concatenate({down0, down1}, 0),
            1);
        compare(down, down_reference, "SSD MXFP4 down");
        compare(slot_down, down, "SSD identity-slot down");

        std::cout << "MLX SSD expert arena passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MLX SSD expert arena failed: " << error.what() << '\n';
        return 1;
    }
}
