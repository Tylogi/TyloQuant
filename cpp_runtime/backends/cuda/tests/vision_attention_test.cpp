#include "core/vision_block.h"
#include "mfq_cuda_context.h"
#include "mfq/runtime.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace tb = mfq_tensor_backend;

int main() {
    for (bool blas : {false, true}) {
        bool resource_error = false;
        try {
            if (blas) mfq::cuda::check(CUBLAS_STATUS_ALLOC_FAILED, "test allocation", __FILE__, __LINE__);
            else mfq::cuda::check(cudaErrorMemoryAllocation, "test allocation", __FILE__, __LINE__);
        } catch (const MfqResourceExhausted&) { resource_error = true; }
        assert(resource_error);
    }
    bool backend_failure = false;
    try { mfq::cuda::check(cudaErrorIllegalAddress, "test context", __FILE__, __LINE__); }
    catch (const MfqBackendFailure&) { backend_failure = true; }
    assert(backend_failure);
    const auto options = tb::TensorOptions().dtype(tb::kFloat32).device(tb::kCUDA);
    for (const int64_t tokens : {127, 129, 513, 2048}) {
        auto q = tb::sin(tb::arange(4 * tokens * 16, options) * 0.001).reshape({1, 4, tokens, 16});
        auto k = (q * 0.7).contiguous();
        auto v = (q * 0.3).contiguous();
        auto mask = (tb::sin(tb::arange(tokens * tokens, options) * 0.037) * 0.25).reshape({tokens, tokens});
        const auto reference = [&] {
            auto scores = tb::matmul(q, k.transpose(-2, -1)) * 0.25 + mask;
            return tb::matmul(tb::softmax(scores, -1, tb::kFloat32), v);
        };
        const auto tiled = [&] {
            return mfq::cuda::vision_non_causal_attention(q, k, v, mask, 0.25);
        };
        auto expected = reference().cpu().contiguous();
        auto actual = tiled().cpu().contiguous();
        for (int64_t i = 0; i < expected.numel(); ++i)
            assert(std::abs(expected.data_ptr<float>()[i] - actual.data_ptr<float>()[i]) < 2e-5f);
        // Warm both paths, then report synchronized wall time on the same
        // synthetic operands. No model weights or user GPU state are touched.
        const auto measure = [&](auto run) {
            run();
            MFQ_NATIVE_CUDA_CHECK(cudaDeviceSynchronize());
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 5; ++i) run();
            MFQ_NATIVE_CUDA_CHECK(cudaDeviceSynchronize());
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 5;
        };
        std::cout << "vision_attention tokens=" << tokens << " dense_ms=" << measure(reference)
                  << " tiled_ms=" << measure(tiled) << '\n';
    }
}
