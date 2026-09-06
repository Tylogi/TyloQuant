#include "mfq_cuda_context.h"
#include "mfq_cuda_kernels.h"
#include "mfq_native_tensor.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

int main() {
    int devices = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        return 77;
    }

    using namespace mfq::cuda;
    constexpr std::int64_t elements = 1024;
    auto gate_host = empty({elements}, TensorOptions{}.dtype(kFloat32));
    auto up_host = empty({elements}, TensorOptions{}.dtype(kFloat32));
    for (std::int64_t index = 0; index < elements; ++index) {
        gate_host.data_ptr<float>()[index] = static_cast<float>(index - 512) / 97.0f;
        up_host.data_ptr<float>()[index] = static_cast<float>((index * 17) % 101) / 23.0f;
    }

    const Device cuda_device{DeviceType::cuda, 0};
    auto gate = gate_host.to(cuda_device);
    auto up = up_host.to(cuda_device);
    auto output = empty({elements}, TensorOptions{}.dtype(kFloat32).device(cuda_device));
    auto context = default_context(0);
    kernels::silu_mul(
        gate.view_descriptor(),
        up.view_descriptor(),
        output.view_descriptor(),
        context->stream().get());

    auto output_host = output.cpu();
    context->stream().synchronize();
    for (std::int64_t index = 0; index < elements; ++index) {
        const float gate_value = gate_host.data_ptr<float>()[index];
        const float expected =
            (gate_value / (1.0f + std::exp(-gate_value))) * up_host.data_ptr<float>()[index];
        assert(std::abs(output_host.data_ptr<float>()[index] - expected) <= 2.0e-6f);
    }
}
