#pragma once

#include <mlx/array.h>
#include <mlx/allocator.h>
#include <mlx/version.h>
#include <memory>

namespace mfq::metal {

template <typename Owner>
mlx::core::array mlx_buffer_view(mlx::core::allocator::Buffer buffer,
    mlx::core::Shape shape, mlx::core::Dtype dtype, std::shared_ptr<Owner> owner) {
    auto retain = [owner = std::move(owner)](mlx::core::allocator::Buffer) {};
#if MLX_VERSION_NUMERIC >= 32004
    return mlx::core::array(mlx::core::allocator::Data(buffer, std::move(retain)), std::move(shape), dtype);
#else
    return mlx::core::array(buffer, std::move(shape), dtype, std::move(retain));
#endif
}

} // namespace mfq::metal
