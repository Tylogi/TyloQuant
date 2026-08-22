#include "mfq_cuda_context.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <memory>

int main() {
    int devices = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        return 77;
    }

    auto context = std::make_shared<mfq::cuda::Context>(0);
    mfq::cuda::Buffer device(context, sizeof(std::uint32_t) * 4);
    mfq::cuda::HostBuffer host(sizeof(std::uint32_t) * 4);
    auto* values = static_cast<std::uint32_t*>(host.data());
    values[0] = 1;
    values[1] = 2;
    values[2] = 3;
    values[3] = 4;

    MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(
        device.data(),
        host.data(),
        host.size(),
        cudaMemcpyHostToDevice,
        context->stream().get()));
    mfq::cuda::Event copied;
    copied.record(context->stream().get());
    copied.synchronize();
    assert(copied.ready());

    values[0] = 0;
    values[1] = 0;
    values[2] = 0;
    values[3] = 0;
    MFQ_NATIVE_CUDA_CHECK(cudaMemcpyAsync(
        host.data(),
        device.data(),
        host.size(),
        cudaMemcpyDeviceToHost,
        context->stream().get()));
    context->stream().synchronize();
    assert(values[0] == 1);
    assert(values[1] == 2);
    assert(values[2] == 3);
    assert(values[3] == 4);

    context->trim();
}
