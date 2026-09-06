#include "mfq_cuda_context.h"
#include "mfq_cuda_kernels.h"
#include "mfq_native_tensor.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {
using namespace mfq::cuda;

void close(float actual, float expected, float atol, float rtol) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > atol + rtol * std::abs(expected)) {
        std::cerr << "activation actual=" << actual << " expected=" << expected << '\n';
        throw std::runtime_error("activation numerical mismatch");
    }
}

void check_glu(std::int64_t elements, ScalarType dtype, bool gelu) {
    const Device gpu{DeviceType::cuda, 0};
    auto host = empty({2, elements}, TensorOptions{}.dtype(kFloat32));
    for (std::int64_t i = 0; i < elements; ++i) {
        host.data_ptr<float>()[i] = static_cast<float>((i * 13) % 1024 - 512) / 97.0f;
        host.data_ptr<float>()[elements + i] = static_cast<float>((i * 17) % 101 - 50) / 23.0f;
    }
    auto input = host.to(gpu).to(dtype);
    auto gate = input.select(0, 0);
    auto up = input.select(0, 1);
    auto output = empty({elements}, TensorOptions{}.dtype(dtype).device(gpu));
    const auto context = default_context(0);
    auto fn = gelu ? kernels::gelu_mul : kernels::silu_mul;
    fn(gate.view_descriptor(), up.view_descriptor(), output.view_descriptor(),
       context->stream().get());
    context->stream().synchronize();
    auto got = output.to(kFloat32).cpu();
    auto rounded = input.to(kFloat32).cpu();
    for (std::int64_t i = 0; i < elements; ++i) {
        const double g = rounded.data_ptr<float>()[i];
        const double u = rounded.data_ptr<float>()[elements + i];
        float expected = static_cast<float>(gelu
            ? u * 0.5 * g * (1.0 + std::tanh(0.7978845608028654 * g * (1.0 + 0.044715 * g * g)))
            : u * g / (1.0 + std::exp(-g)));
        if (dtype == kFloat16) expected = __half2float(__float2half_rn(expected));
        close(got.data_ptr<float>()[i], expected,
              dtype == kFloat16 ? 1.e-3f : 2.e-6f,
              dtype == kFloat16 ? 1.e-3f : 2.e-6f);
    }
}

void check_gate_beta(std::int64_t tokens, ScalarType dtype) {
    const Device gpu{DeviceType::cuda, 0};
    constexpr std::int64_t batch = 2, width = 7;
    const auto elements = batch * tokens * width;
    auto alpha_host = empty({batch, width, tokens}, TensorOptions{}.dtype(kFloat32));
    auto beta_host = empty({batch, tokens, width}, TensorOptions{}.dtype(kFloat32));
    auto dt_host = empty({width}, TensorOptions{}.dtype(kFloat32));
    auto log_host = empty({width}, TensorOptions{}.dtype(kFloat32));
    for (std::int64_t i = 0; i < elements; ++i) {
        alpha_host.data_ptr<float>()[i] = static_cast<float>((i * 17) % 101 - 50);
        beta_host.data_ptr<float>()[i] = static_cast<float>((i * 13) % 101 - 50) / 7.f;
    }
    for (std::int64_t i = 0; i < width; ++i) {
        dt_host.data_ptr<float>()[i] = static_cast<float>(i - 3) / 5.f;
        log_host.data_ptr<float>()[i] = static_cast<float>(i - 3) / 7.f;
    }
    // Deliberately different alpha/beta strides, with FP16 inputs and FP32 parameters/output.
    auto alpha = alpha_host.to(gpu).to(dtype).transpose(1, 2);
    auto beta = beta_host.to(gpu).to(dtype);
    auto dt = dt_host.to(gpu);
    auto log_a = log_host.to(gpu);
    auto gates = empty({batch, width, tokens}, TensorOptions{}.dtype(kFloat32).device(gpu));
    auto betas = empty({batch, width, tokens}, TensorOptions{}.dtype(kFloat32).device(gpu));
    const auto context = default_context(0);
    kernels::linear_gate_beta(alpha.view_descriptor(), beta.view_descriptor(),
        dt.view_descriptor(), log_a.view_descriptor(), gates.view_descriptor(),
        betas.view_descriptor(), context->stream().get());
    context->stream().synchronize();
    auto a_ref = alpha.contiguous().to(kFloat32).cpu();
    auto b_ref = beta.contiguous().to(kFloat32).cpu();
    auto g_got = gates.cpu();
    auto b_got = betas.cpu();
    for (std::int64_t b = 0; b < batch; ++b) {
        for (std::int64_t t = 0; t < tokens; ++t) {
            for (std::int64_t v = 0; v < width; ++v) {
                const auto src = (b * tokens + t) * width + v;
                const auto dst = (b * width + v) * tokens + t;
                const double a = a_ref.data_ptr<float>()[src] + dt_host.data_ptr<float>()[v];
                const double sp = a > 20.0 ? a : std::log1p(std::exp(a));
                const float expected_gate = static_cast<float>(-sp * std::exp(log_host.data_ptr<float>()[v]));
                const float expected_beta = static_cast<float>(1. / (1. + std::exp(-double(b_ref.data_ptr<float>()[src]))));
                close(g_got.data_ptr<float>()[dst], expected_gate, 2.e-6f, 2.e-6f);
                close(b_got.data_ptr<float>()[dst], expected_beta, 2.e-6f, 2.e-6f);
            }
        }
    }
}

void check_half_saturation() {
    const Device gpu{DeviceType::cuda, 0};
    auto host = empty({2, 3}, TensorOptions{}.dtype(kFloat32));
    const float values[] = {200.f, 200.f, -200.f, 400.f, -400.f, 400.f};
    std::copy(std::begin(values), std::end(values), host.data_ptr<float>());
    auto input = host.to(gpu).to(kFloat16);
    auto output = empty({3}, TensorOptions{}.dtype(kFloat16).device(gpu));
    const auto context = default_context(0);
    kernels::gelu_mul(input.select(0, 0).view_descriptor(), input.select(0, 1).view_descriptor(),
        output.view_descriptor(), context->stream().get());
    context->stream().synchronize();
    const auto got = output.to(kFloat32).cpu();
    close(got.data_ptr<float>()[0], 65504.f, 0.f, 0.f);
    close(got.data_ptr<float>()[1], -65504.f, 0.f, 0.f);
    close(got.data_ptr<float>()[2], 0.f, 0.f, 0.f);
}
}  // namespace

int main() {
    int devices = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        return 77;
    }

    int checked = 0;
    for (auto dtype : {mfq::cuda::kFloat16, mfq::cuda::kFloat32}) {
        for (auto count : {0, 1, 7, 32, 255, 1024, 1031}) {
            check_glu(count, dtype, false);
            check_glu(count, dtype, true);
            checked += 2;
        }
        for (int tokens = 1; tokens <= 6; ++tokens) {
            check_gate_beta(tokens, dtype);
            ++checked;
        }
    }
    check_half_saturation();
    std::cout << "activation cases=" << checked + 1 << " passed (Release checks enabled)\n";
}
