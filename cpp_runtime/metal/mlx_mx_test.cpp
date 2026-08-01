#include "mlx_mx.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

template <typename T>
void append(std::vector<std::uint8_t>& target, T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    target.insert(target.end(), bytes, bytes + sizeof(T));
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<std::uint8_t> make_blob(int bits, int outputs, int inputs) {
    std::vector<std::uint8_t> blob{'M', 'X', 'T', '1'};
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, static_cast<std::uint8_t>(bits));
    append<std::uint16_t>(blob, 0);
    append<std::uint64_t>(blob, outputs);
    append<std::uint64_t>(blob, inputs);
    append<std::uint64_t>(blob, outputs);
    append<std::uint64_t>(blob, bits == 4 ? inputs / 2 : inputs);
    append<std::uint64_t>(blob, bits == 4 ? outputs : (outputs + 127) / 128);
    append<std::uint64_t>(blob, bits == 4 ? inputs / 32 : inputs / 128);
    const auto values = static_cast<std::size_t>(outputs) *
        static_cast<std::size_t>(bits == 4 ? inputs / 2 : inputs);
    blob.insert(blob.end(), values, bits == 4 ? 0x22 : 0x38);
    const auto scales = static_cast<std::size_t>(
        bits == 4 ? outputs : (outputs + 127) / 128) *
        static_cast<std::size_t>(bits == 4 ? inputs / 32 : inputs / 128);
    blob.insert(blob.end(), scales, 127);
    return blob;
}

void test_matmul(const std::string& dtype, int inputs, int rows) {
    using namespace mlx::core;
    constexpr int outputs = 5;
    const int bits = dtype == "MXFP4" ? 4 : 8;
    const auto weight = mfq::metal::MlxMxWeight::from_blob(
        dtype, make_blob(bits, outputs, inputs));
    std::vector<float> source_values(
        static_cast<std::size_t>(rows) * inputs);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < inputs; ++column) {
            source_values[static_cast<std::size_t>(row) * inputs + column] =
                static_cast<float>((column % 11) - 5) / 32.0f;
        }
    }
    auto output = astype(
        weight.matmul(array(source_values.begin(), Shape{rows, inputs})),
        float32);
    eval(output);
    for (int row = 0; row < rows; ++row) {
        float expected = 0.0f;
        for (int column = 0; column < inputs; ++column) {
            expected += source_values[
                static_cast<std::size_t>(row) * inputs + column];
        }
        for (int out = 0; out < outputs; ++out) {
            const auto actual = output.data<float>()[row * outputs + out];
            require(
                std::isfinite(actual) && std::fabs(actual - expected) < 2e-4f,
                dtype + " Metal matmul mismatch");
        }
    }
}

void test_embedding(const std::string& dtype, int inputs) {
    using namespace mlx::core;
    constexpr int outputs = 5;
    const int bits = dtype == "MXFP4" ? 4 : 8;
    const auto weight = mfq::metal::MlxMxWeight::from_blob(
        dtype, make_blob(bits, outputs, inputs));
    const std::vector<std::int32_t> indices{4, 1};
    auto output = astype(
        weight.embedding(
            array(indices.begin(), Shape{static_cast<int>(indices.size())}),
            float16),
        float32);
    eval(output);
    require(
        output.shape() == Shape{2, inputs},
        dtype + " Metal embedding shape mismatch");
    for (std::size_t index = 0; index < output.size(); ++index) {
        require(
            output.data<float>()[index] == 1.0f,
            dtype + " Metal embedding value mismatch");
    }
}

} // namespace

int main() {
    try {
        test_matmul("MXFP4", 96, 1);
        test_matmul("MXFP4", 96, 7);
        test_matmul("MXFP4", 96, 64);
        test_matmul("MXFP8", 128, 1);
        test_matmul("MXFP8", 128, 7);
        test_matmul("MXFP8", 128, 64);
        test_embedding("MXFP4", 96);
        test_embedding("MXFP8", 128);
        std::cout << "MFQ MXFP4/MXFP8 Metal GEMV/MMQ/GEMM/embedding passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
