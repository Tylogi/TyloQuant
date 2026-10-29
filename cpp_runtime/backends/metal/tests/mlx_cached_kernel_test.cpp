#include "mlx_cached_kernel.h"
#include "mlx_kernel_prepare.h"
#include <mlx/mlx.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace mlx::core;
using mfq::metal::cached_metal_kernel;

namespace {
void assert_same(const array& actual, const array& expected) {
    if (actual.shape() != expected.shape() || actual.dtype() != expected.dtype())
        throw std::runtime_error("cached kernel changed output shape or type");
    if (!array_equal(actual, expected).item<bool>()) {
        const auto error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
        throw std::runtime_error("cached kernel changed numerical output: " + std::to_string(error));
    }
}
}

int main() {
    try {
        set_default_device(Device::gpu);
        const std::string header = R"METAL(
float read_value(const constant float& x, uint i) { return x; }
float read_value(const constant float* x, uint i) { return x[i]; }
float read_value(const device float* x, uint i) { return x[i]; }
)METAL";
        const std::string body = "uint i = thread_position_in_grid.x; out[i] = read_value(x, i) * FACTOR;";
        auto cached = cached_metal_kernel("tinfer_cached_binding_test", {"x"}, {"out"}, body, header);
        auto stock = fast::metal_kernel("tinfer_cached_binding_test", {"x"}, {"out"}, body, header);
        for (int size : {1, 7, 8, 13, 256, 17, 7, 8}) {
            const auto x = arange(size, float32);
            for (int factor : {1, 3}) {
                auto actual = cached({x}, {{size}}, {float32}, {size, 1, 1}, {32, 1, 1},
                    {{"FACTOR", factor}}, std::nullopt, false, {});
                auto expected = stock({x}, {{size}}, {float32}, {size, 1, 1}, {32, 1, 1},
                    {{"FACTOR", factor}}, std::nullopt, false, {});
                assert_same(actual.front(), expected.front());
            }
        }
        for (float scalar : {2.f, -4.f})
            assert_same(cached({array(scalar)}, {{4}}, {float32}, {4, 1, 1}, {4, 1, 1},
                {{"FACTOR", 3}}, std::nullopt, false, {}).front(), full({4}, scalar * 3, float32));

        const std::string typed_body = R"METAL(
uint i = thread_position_in_grid.x;
if (i < uint(x_shape[0] * x_shape[1])) {
    int row = i / x_shape[1], col = i % x_shape[1];
    a[i] = U(float(x[row * x_strides[0] + col * x_strides[1]]) + x_ndim);
    b[i] = float(x[row * x_strides[0] + col * x_strides[1]]) * (NEGATE ? -1.f : 1.f);
}
)METAL";
        for (bool contiguous : {false, true}) {
            auto f = cached_metal_kernel("tinfer_cached_metadata_test", {"x"}, {"a", "b"}, typed_body, "", contiguous);
            auto g = fast::metal_kernel("tinfer_cached_metadata_test", {"x"}, {"a", "b"}, typed_body, "", contiguous);
            for (auto dtype : {float32, bfloat16, float16}) {
                for (int rows : {2, 7, 11}) {
                    const auto x = transpose(reshape(astype(arange(3 * rows, float32), dtype), {3, rows}));
                    for (bool negate : {false, true}) {
                        const std::vector<std::pair<std::string, fast::TemplateArg>> args{{"U", dtype}, {"NEGATE", negate}};
                        const auto stream = new_stream(Device::gpu);
                        const auto a = f({x}, {{rows, 3}, {rows, 3}}, {dtype, float32},
                            {rows * 3, 1, 1}, {32, 1, 1}, args, std::nullopt, false, stream);
                        const auto b = g({x}, {{rows, 3}, {rows, 3}}, {dtype, float32},
                            {rows * 3, 1, 1}, {32, 1, 1}, args, std::nullopt, false, stream);
                        assert_same(a[0], b[0]); assert_same(a[1], b[1]);
                    }
                }
            }
        }

        auto flat = cached_metal_kernel("tinfer_cached_contiguous_view_test", {"x"}, {"out"},
            "uint i=thread_position_in_grid.x; out[i]=x[i];", "", true);
        auto flat_stock = fast::metal_kernel("tinfer_cached_contiguous_view_test", {"x"}, {"out"},
            "uint i=thread_position_in_grid.x; out[i]=x[i];", "", true);
        for (const auto dtype : {float32, bfloat16, float16, int32, uint8})
            for (const bool ready : {false, true}) for (const bool strided : {false, true}) {
            auto parent = reshape(astype(arange(64 * 1024, float32), dtype), {64, 1024});
            if (ready) eval(parent);
            auto input = slice(parent, {16, 0}, {48, 1024});
            if (strided) input = transpose(input);
            const auto view = mfq::metal::mlx_row_contiguous_view(input, default_stream(Device::gpu));
            eval(view);
            if (!view.flags().row_contiguous) throw std::runtime_error("row-major view retained strided input");
            if (!strided && (view.buffer().ptr() != input.buffer().ptr() || view.offset() != input.offset()))
                throw std::runtime_error("row-major view copied an offset inside a larger allocation");
            assert_same(view, input);
            const auto output = flat({input}, {input.shape()}, {dtype}, {int(input.size()), 1, 1},
                {256, 1, 1}, {}, std::nullopt, false, {}).front();
            assert_same(output, input);
        }
        for (auto dtype : {float32, bfloat16, float16, uint8, int32, int64})
            for (int rows : {1, 2, 3, 7}) for (int width : {1, 2, 9}) for (bool ready : {false, true}) {
            const auto parent = reshape(astype(arange(rows * 17, float32), dtype), {rows, 17});
            if (ready) eval(parent);
            const auto view = slice(parent, {0, 4}, {rows, 4 + width});
            auto actual = flat({view}, {view.shape()}, {dtype}, {rows * width, 1, 1},
                {32, 1, 1}, {}, std::nullopt, false, {}).front();
            auto expected = flat_stock({view}, {view.shape()}, {dtype}, {rows * width, 1, 1},
                {32, 1, 1}, {}, std::nullopt, false, {}).front();
            try {
                assert_same(actual, expected);
                assert_same(actual, view);
            } catch (const std::exception& error) {
                throw std::runtime_error("contiguous view dtype=" + std::to_string(int(dtype.val())) +
                    " rows=" + std::to_string(rows) + " width=" + std::to_string(width) +
                    " parent_ready=" + std::to_string(ready) + ": " + error.what());
            }
        }

        auto compiled_view = compile([&](const std::vector<array>& values) {
            auto view = slice(values[0], {0, 2}, {values[0].shape(0), 11});
            return flat({view}, {view.shape()}, {view.dtype()}, {int(view.size()), 1, 1},
                {32, 1, 1}, {}, std::nullopt, false, {});
        });
        for (int round = 0; round < 4; ++round) {
            const auto input = reshape(arange(3 * 17, float32) + round * 100, {3, 17});
            assert_same(compiled_view({input}).front(), slice(input, {0, 2}, {3, 11}));
        }

        auto atomic = cached_metal_kernel("tinfer_cached_atomic_test", {"x"}, {"out"},
            "atomic_fetch_add_explicit(out, x[thread_position_in_grid.x], memory_order_relaxed);", "", true, true);
        for (float initial : {0.f, 5.f}) {
            const auto x = ones({32}, int32);
            const auto value = atomic({x}, {{1}}, {int32}, {32, 1, 1}, {32, 1, 1}, {}, initial, false, {}).front();
            eval(value);
            if (value.item<int>() != 32 + int(initial)) throw std::runtime_error("cached kernel lost output initialization");
        }

        auto compiled = compile([&](const std::vector<array>& values) {
            return cached(values, {values[0].shape()}, {float32}, {int(values[0].size()), 1, 1},
                {32, 1, 1}, {{"FACTOR", 3}}, std::nullopt, false, {});
        });
        for (int round = 1; round <= 5; ++round) {
            auto x = arange(13, float32) + round;
            assert_same(compiled({x}).front(), x * 3);
        }
        auto x = zeros({19}, float32);
        auto output = cached({x}, {{19}}, {float32}, {19, 1, 1}, {32, 1, 1},
            {{"FACTOR", 3}}, std::nullopt, false, {}).front();
        mfq::metal::MlxKernelPreparation preparation(128, 128);
        preparation.add(output);
        if (preparation.size() != 1) throw std::runtime_error("cached kernel not collected for preparation");
        preparation.finish();
        if (x.is_available() || output.is_available()) throw std::runtime_error("preparation executed the real input graph");
        assert_same(output, x * 3);

        std::weak_ptr<array::Data> allocation;
        {
            auto input = ones({97}, float32);
            eval(input);
            allocation = input.data_shared_ptr();
            auto result = cached({input}, {{97}}, {float32}, {97, 1, 1}, {32, 1, 1},
                {{"FACTOR", 3}}, std::nullopt, false, {});
            eval(result);
        }
        synchronize();
        if (!allocation.expired()) throw std::runtime_error("cached shader description retained input weights");
        {
            auto input = ones({2, 17}, float32);
            eval(input);
            allocation = input.data_shared_ptr();
            auto view = slice(input, {0, 2}, {2, 11});
            auto result = flat({view}, {view.shape()}, {float32}, {18, 1, 1},
                {32, 1, 1}, {}, std::nullopt, false, {});
            eval(result);
        }
        synchronize();
        if (!allocation.expired()) throw std::runtime_error("cached layout conversion retained its input allocation");
        bool rejected = false;
        try { cached({}, {{1}}, {float32}, {1, 1, 1}, {1, 1, 1}, {}, {}, false, {}); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("cached kernel accepted invalid input count");
        std::cout << "Cached Metal kernels preserve bindings, strides, templates, initialization, streams, replay and ownership\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
