#include "mfq_cuda_paged_kv.h"
#include "mfq_native_tensor.h"
#include "../runtime/paged_kv_allocator.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mfq::cuda::Device;
using mfq::cuda::DeviceType;
using mfq::cuda::Tensor;
using mfq::cuda::TensorOptions;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<float> float_values(const Tensor & tensor) {
    auto host = tensor.to(mfq::cuda::kCPU, mfq::cuda::kFloat32)
        .contiguous();
    return std::vector<float>(
        host.data_ptr<float>(), host.data_ptr<float>() + host.numel());
}

void check_allocator() {
    using mfq::cuda::continuous::PagedKvPageAllocator;
    PagedKvPageAllocator allocator(8);
    auto first = allocator.allocate(3);
    require(first == std::vector<std::int32_t>({0, 1, 2}),
        "Paged KV allocator did not issue monotonic pages");
    allocator.release({1});
    auto reused = allocator.allocate(1);
    require(reused == std::vector<std::int32_t>({1}) &&
        allocator.reuse_count() == 1 && allocator.live_pages() == 3,
        "Paged KV allocator did not reuse a retired page");
    bool rejected = false;
    try {
        allocator.release({7});
    } catch (const std::logic_error &) {
        rejected = true;
    }
    require(rejected && allocator.live_pages() == 3,
        "Paged KV allocator accepted an invalid release");
    allocator.release({0, 2, 1});
    require(allocator.live_pages() == 0 &&
        allocator.peak_live_pages() == 3,
        "Paged KV allocator accounting mismatch");
}

void check_kernels() {
    constexpr int B = 2;
    constexpr int Hq = 8;
    constexpr int Hk = 2;
    constexpr int T = 7;
    constexpr int D = 256;
    constexpr int Page = 4;
    constexpr int PagesPerChunk = 4;
    constexpr int LogicalPages = 4;
    const Device gpu{DeviceType::cuda, 0};

    std::vector<float> k_host(B * Hk * T * D);
    std::vector<float> v_host(k_host.size());
    std::vector<float> q_host(B * Hq * D);
    for (size_t index = 0; index < k_host.size(); ++index) {
        k_host[index] = static_cast<float>(
            static_cast<int>((index * 17 + 3) % 97) - 48) / 83.0f;
        v_host[index] = static_cast<float>(
            static_cast<int>((index * 29 + 5) % 89) - 44) / 71.0f;
    }
    for (size_t index = 0; index < q_host.size(); ++index) {
        q_host[index] = static_cast<float>(
            static_cast<int>((index * 11 + 7) % 79) - 39) / 67.0f;
    }
    auto k = mfq::cuda::tensor<float>(k_host).reshape({B, Hk, T, D})
        .to(gpu).to(mfq::cuda::kFloat16).contiguous();
    auto v = mfq::cuda::tensor<float>(v_host).reshape({B, Hk, T, D})
        .to(gpu).to(mfq::cuda::kFloat16).contiguous();
    auto q = mfq::cuda::tensor<float>(q_host).reshape({B, Hq, 1, D})
        .to(gpu).to(mfq::cuda::kFloat16).contiguous();
    auto k_chunk_0 = mfq::cuda::empty(
        {PagesPerChunk, Hk, Page, D},
        TensorOptions().device(gpu).dtype(mfq::cuda::kFloat16));
    auto v_chunk_0 = mfq::cuda::empty_like(k_chunk_0);
    auto k_chunk_1 = mfq::cuda::empty_like(k_chunk_0);
    auto v_chunk_1 = mfq::cuda::empty_like(k_chunk_0);
    const auto k_pointer_0 = static_cast<std::int64_t>(
        reinterpret_cast<std::intptr_t>(k_chunk_0.data_ptr()));
    const auto v_pointer_0 = static_cast<std::int64_t>(
        reinterpret_cast<std::intptr_t>(v_chunk_0.data_ptr()));
    const auto k_pointer_1 = static_cast<std::int64_t>(
        reinterpret_cast<std::intptr_t>(k_chunk_1.data_ptr()));
    const auto v_pointer_1 = static_cast<std::int64_t>(
        reinterpret_cast<std::intptr_t>(v_chunk_1.data_ptr()));
    auto k_chunks = mfq::cuda::tensor<std::int64_t>({
        k_pointer_0, k_pointer_1}).to(gpu);
    auto v_chunks = mfq::cuda::tensor<std::int64_t>({
        v_pointer_0, v_pointer_1}).to(gpu);
    // Logical pages deliberately map to non-contiguous physical pages and
    // cross the chunk boundary at physical page 4.
    auto page_table = mfq::cuda::tensor<std::int32_t>({
        2, 4, -1, -1,
        5, 3, -1, -1,
    }).reshape({B, LogicalPages}).to(gpu).contiguous();
    auto positions = mfq::cuda::tensor<std::int64_t>({
        0, 1, 2, 3, 4, 5, 6,
        0, 1, 2, 3, 4, 5, 6,
    }).reshape({B, T}).to(gpu).contiguous();
    paged_kv_cache_write_cuda(
        k_chunks, v_chunks, page_table, k, v, positions,
        Page, PagesPerChunk);

    auto lengths = mfq::cuda::tensor<std::int64_t>({7, 5}).to(gpu);
    auto partial_o = mfq::cuda::empty(
        {B * Hq, 4, D},
        TensorOptions().device(gpu).dtype(mfq::cuda::kFloat32));
    auto partial_m = mfq::cuda::empty(
        {B * Hq, 4},
        TensorOptions().device(gpu).dtype(mfq::cuda::kFloat32));
    auto partial_l = mfq::cuda::empty_like(partial_m);
    constexpr double Scale = 0.0625;
    auto direct = attention_paged_cache_decode_cuda(
        q, k_chunks, v_chunks, page_table, lengths, Scale,
        Page, PagesPerChunk, Hk,
        partial_o, partial_m, partial_l, 1, false);
    auto split = attention_paged_cache_decode_cuda(
        q, k_chunks, v_chunks, page_table, lengths, Scale,
        Page, PagesPerChunk, Hk,
        partial_o, partial_m, partial_l, 3, false);
    auto dynamic = attention_paged_cache_decode_cuda(
        q, k_chunks, v_chunks, page_table, lengths, Scale,
        Page, PagesPerChunk, Hk,
        partial_o, partial_m, partial_l, 4, true);
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceSynchronize());

    const auto rounded_k = float_values(k);
    const auto rounded_v = float_values(v);
    const auto rounded_q = float_values(q);
    const auto direct_values = float_values(direct);
    const auto split_values = float_values(split);
    const auto dynamic_values = float_values(dynamic);
    for (int batch = 0; batch < B; ++batch) {
        const int token_count = batch == 0 ? 7 : 5;
        for (int query_head = 0; query_head < Hq; ++query_head) {
            const int kv_head = query_head / (Hq / Hk);
            std::vector<double> scores(token_count);
            double maximum = -1e300;
            for (int token = 0; token < token_count; ++token) {
                double dot = 0.0;
                for (int dim = 0; dim < D; ++dim) {
                    const auto q_index =
                        (static_cast<size_t>(batch) * Hq + query_head) * D + dim;
                    const auto kv_index =
                        ((static_cast<size_t>(batch) * Hk + kv_head) * T +
                         token) * D + dim;
                    dot += static_cast<double>(rounded_q[q_index]) *
                        rounded_k[kv_index];
                }
                scores[token] = dot * Scale;
                maximum = std::max(maximum, scores[token]);
            }
            double denominator = 0.0;
            for (const auto score : scores) {
                denominator += std::exp(score - maximum);
            }
            for (int dim = 0; dim < D; ++dim) {
                double expected = 0.0;
                for (int token = 0; token < token_count; ++token) {
                    const auto kv_index =
                        ((static_cast<size_t>(batch) * Hk + kv_head) * T +
                         token) * D + dim;
                    expected += std::exp(scores[token] - maximum) *
                        rounded_v[kv_index];
                }
                expected /= denominator;
                const auto output_index =
                    (static_cast<size_t>(batch) * Hq + query_head) * D + dim;
                const auto error = std::abs(
                    direct_values[output_index] - expected);
                if (!std::isfinite(direct_values[output_index]) ||
                        error > 2.5e-3 ||
                        std::abs(split_values[output_index] -
                            direct_values[output_index]) > 2.5e-3 ||
                        std::abs(dynamic_values[output_index] -
                            direct_values[output_index]) > 2.5e-3) {
                    throw std::runtime_error(
                        "Paged KV attention numerical mismatch at " +
                        std::to_string(output_index));
                }
            }
        }
    }
}

} // namespace

int main() {
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) {
        (void)cudaGetLastError();
        return 77;
    }
    check_allocator();
    check_kernels();
    std::cout << "paged_kv_test PASS page_size=4 chunks=2 gqa=4 split=1\n";
    return 0;
}
