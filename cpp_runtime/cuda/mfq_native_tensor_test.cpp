#include "mfq_native_tensor.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <stdexcept>

int main() {
    using namespace mfq::cuda;

    auto values = zeros({2, 3, 4}, TensorOptions{}.dtype(kFloat32));
    assert(values.defined());
    assert(values.is_cpu());
    assert(values.is_contiguous());
    assert(values.numel() == 24);
    values.fill_(3.0);
    assert(values.data_ptr<float>()[17] == 3.0f);

    const auto reshaped = values.reshape({4, 6});
    assert(reshaped.size(0) == 4);
    assert(reshaped.size(1) == 6);
    assert(reshaped.data_ptr() == values.data_ptr());

    const auto transposed = reshaped.transpose(0, 1);
    assert(transposed.size(0) == 6);
    assert(transposed.size(1) == 4);
    assert(!transposed.is_contiguous());

    const auto selected = values.narrow(1, 1, 2).select(2, 3);
    assert(selected.size(0) == 2);
    assert(selected.size(1) == 2);
    assert(selected.byte_offset() == (4 + 3) * sizeof(float));

    auto copy = values.clone();
    assert(copy.data_ptr() != values.data_ptr());
    assert(copy.data_ptr<float>()[17] == 3.0f);
    copy.zero_();
    assert(copy.data_ptr<float>()[17] == 0.0f);
    assert(values.data_ptr<float>()[17] == 3.0f);

    auto scalar = full({1}, 9.0, TensorOptions{}.dtype(kInt64));
    assert(scalar.item<std::int64_t>() == 9);

    auto sequence = tensor<float>({0, 1, 2, 3, 4, 5}).reshape({2, 3});
    auto materialized = sequence.transpose(0, 1).reshape({6});
    const std::array<float, 6> expected{0, 3, 1, 4, 2, 5};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        assert(materialized.data_ptr<float>()[index] == expected[index]);
    }

    auto half = tensor<float>({1.5f, -2.25f}).to(kFloat16);
    assert(std::abs(half.select(0, 0).item<float>() - 1.5f) == 0.0f);
    assert(std::abs(half.select(0, 1).item<float>() + 2.25f) == 0.0f);
    auto restored = half.to(kFloat32);
    assert(restored.data_ptr<float>()[0] == 1.5f);
    assert(restored.data_ptr<float>()[1] == -2.25f);

    auto dtype_only = sequence.to(TensorOptions{}.dtype(kFloat64));
    assert(dtype_only.is_cpu());
    assert(dtype_only.scalar_type() == kFloat64);
    auto like_options = TensorOptions{}.dtype(kInt32);
    auto like_dtype_only = empty_like(sequence, &like_options);
    assert(like_dtype_only.device() == sequence.device());
    assert(like_dtype_only.scalar_type() == kInt32);

    const auto serialized = pickle_save(sequence);
    const auto roundtrip = pickle_load(serialized).toTensor();
    assert(roundtrip.sizes() == sequence.sizes());
    assert(roundtrip.scalar_type() == sequence.scalar_type());
    assert(roundtrip.data_ptr<float>()[5] == 5.0f);
}
