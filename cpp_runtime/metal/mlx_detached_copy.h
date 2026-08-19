#pragma once

#include <mlx/mlx.h>

namespace mfq::metal {

// MLX may elide copy() when the result is otherwise identical to its input.
// Cache snapshots must own a distinct allocation because live cache updates
// can write through aliases. Appending and then slicing off one sentinel forces
// a new backing allocation without applying arithmetic to the tensor values.
inline mlx::core::array detached_copy(const mlx::core::array& value) {
    auto flat = mlx::core::reshape(value, {-1});
    auto storage = mlx::core::concatenate(
        {flat, mlx::core::zeros({1}, value.dtype())},
        0);
    return mlx::core::reshape(
        mlx::core::slice(
            storage,
            {0},
            {static_cast<int>(value.size())}),
        value.shape());
}

} // namespace mfq::metal
