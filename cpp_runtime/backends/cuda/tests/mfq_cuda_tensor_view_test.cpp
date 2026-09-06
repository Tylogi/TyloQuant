#include "mfq_cuda_tensor_view.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>

int main() {
    using mfq::cuda::Device;
    using mfq::cuda::DeviceType;
    using mfq::cuda::ScalarType;

    std::array<std::uint16_t, 24> storage{};
    const std::array<std::int64_t, 3> shape = {2, 3, 4};
    const auto source = mfq::cuda::make_contiguous_view(
        storage.data(), shape, ScalarType::float16, Device{DeviceType::cuda, 1});
    assert(source.defined());
    assert(source.rank == 3);
    assert(source.size(-1) == 4);
    assert(source.stride(0) == 12);
    assert(source.numel() == 24);
    assert(source.nbytes() == 48);
    assert(source.is_contiguous());

    const std::array<std::int64_t, 2> inferred = {-1, 6};
    const auto reshaped = mfq::cuda::reshape_view(source, inferred);
    assert(reshaped.size(0) == 4);
    assert(reshaped.size(1) == 6);
    assert(reshaped.is_contiguous());

    const auto transposed = mfq::cuda::transpose_view(reshaped, 0, 1);
    assert(transposed.size(0) == 6);
    assert(transposed.size(1) == 4);
    assert(!transposed.is_contiguous());

    bool rejected = false;
    try {
        const std::array<std::int64_t, 2> incompatible = {5, 5};
        (void)mfq::cuda::reshape_view(source, incompatible);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    rejected = false;
    try {
        const std::array<std::int64_t, 2> ambiguous = {-1, -1};
        (void)mfq::cuda::reshape_view(source, ambiguous);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}
