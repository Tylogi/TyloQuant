#include "mfq_cuda_paged_kv.h"
#include "mfq_native_tensor.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

mfq_tensor_backend::Tensor attention_cache_decode_dynamic_cuda(
    mfq_tensor_backend::Tensor q,
    mfq_tensor_backend::Tensor k_cache,
    mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor seq_len,
    double scale,
    mfq_tensor_backend::Tensor partial_o,
    mfq_tensor_backend::Tensor partial_m,
    mfq_tensor_backend::Tensor partial_l,
    int64_t max_parts);

namespace {

using mfq::cuda::Device;
using mfq::cuda::DeviceType;
using mfq::cuda::Tensor;
using mfq::cuda::TensorOptions;

constexpr int B = 4;
constexpr int Hq = 16;
constexpr int Hk = 4;
constexpr int D = 256;
constexpr int Page = 16;
constexpr int PagesPerChunk = 64;
constexpr int MaxParts = 64;
constexpr int MaxSeq = 8750;
constexpr double Scale = 0.0625;
constexpr int Lengths[B] = {2222, 4398, 6574, 8750};

void check_cuda(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <typename Function>
std::vector<float> measure(
        Function && function, int warmup, int iterations, int samples) {
    std::vector<Tensor> warmup_outputs;
    warmup_outputs.reserve(static_cast<size_t>(warmup));
    for (int index = 0; index < warmup; ++index) {
        warmup_outputs.push_back(function());
    }
    check_cuda(cudaDeviceSynchronize(), "benchmark warmup");
    warmup_outputs.clear();

    std::vector<float> values;
    values.reserve(static_cast<size_t>(samples));
    for (int sample = 0; sample < samples; ++sample) {
        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
        check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        check_cuda(cudaEventCreate(&end), "cudaEventCreate(end)");
        std::vector<Tensor> outputs;
        outputs.reserve(static_cast<size_t>(iterations));
        check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
        for (int iteration = 0; iteration < iterations; ++iteration) {
            outputs.push_back(function());
        }
        check_cuda(cudaEventRecord(end), "cudaEventRecord(end)");
        check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
        float elapsed_ms = 0.0f;
        check_cuda(
            cudaEventElapsedTime(&elapsed_ms, start, end),
            "cudaEventElapsedTime");
        values.push_back(elapsed_ms * 1000.0f / iterations);
        check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
        check_cuda(cudaEventDestroy(end), "cudaEventDestroy(end)");
    }
    return values;
}

std::vector<float> host_float_values(const Tensor & tensor) {
    auto host = tensor.to(mfq::cuda::kCPU, mfq::cuda::kFloat32).contiguous();
    return std::vector<float>(
        host.data_ptr<float>(), host.data_ptr<float>() + host.numel());
}

void print_values(const std::vector<float> & values) {
    std::cout << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

} // namespace

int main(int argc, char ** argv) {
    int iterations = 200;
    int samples = 7;
    if (argc > 1) iterations = std::max(1, std::atoi(argv[1]));
    if (argc > 2) samples = std::max(1, std::atoi(argv[2]));

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        (void)cudaGetLastError();
        return 77;
    }
    const Device gpu{DeviceType::cuda, 0};
    const auto fp16 = TensorOptions().device(gpu).dtype(mfq::cuda::kFloat16);
    const auto fp32 = TensorOptions().device(gpu).dtype(mfq::cuda::kFloat32);

    auto q = mfq::cuda::randn({B, Hq, 1, D}, fp32)
        .to(mfq::cuda::kFloat16).contiguous();
    auto k = mfq::cuda::randn({B, Hk, MaxSeq, D}, fp32)
        .to(mfq::cuda::kFloat16).contiguous();
    auto v = mfq::cuda::randn({B, Hk, MaxSeq, D}, fp32)
        .to(mfq::cuda::kFloat16).contiguous();
    auto seq_len = mfq::cuda::tensor<std::int64_t>({
        Lengths[0], Lengths[1], Lengths[2], Lengths[3]}).to(gpu);

    constexpr int LogicalPages = (MaxSeq + Page - 1) / Page;
    std::vector<std::int32_t> host_page_table(
        static_cast<size_t>(B) * LogicalPages, -1);
    int physical_pages = 0;
    for (int batch = 0; batch < B; ++batch) {
        const int pages = (Lengths[batch] + Page - 1) / Page;
        for (int logical = 0; logical < pages; ++logical) {
            host_page_table[static_cast<size_t>(batch) * LogicalPages + logical] =
                physical_pages++;
        }
    }
    const int chunks =
        (physical_pages + PagesPerChunk - 1) / PagesPerChunk;
    std::vector<Tensor> k_chunks;
    std::vector<Tensor> v_chunks;
    std::vector<std::int64_t> k_pointers;
    std::vector<std::int64_t> v_pointers;
    k_chunks.reserve(chunks);
    v_chunks.reserve(chunks);
    k_pointers.reserve(chunks);
    v_pointers.reserve(chunks);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        k_chunks.push_back(mfq::cuda::empty(
            {PagesPerChunk, Hk, Page, D}, fp16));
        v_chunks.push_back(mfq::cuda::empty_like(k_chunks.back()));
        k_pointers.push_back(static_cast<std::int64_t>(
            reinterpret_cast<std::intptr_t>(k_chunks.back().data_ptr())));
        v_pointers.push_back(static_cast<std::int64_t>(
            reinterpret_cast<std::intptr_t>(v_chunks.back().data_ptr())));
    }
    auto k_chunk_ptrs = mfq::cuda::tensor<std::int64_t>(k_pointers).to(gpu);
    auto v_chunk_ptrs = mfq::cuda::tensor<std::int64_t>(v_pointers).to(gpu);
    auto page_table = mfq::cuda::tensor<std::int32_t>(host_page_table)
        .reshape({B, LogicalPages}).to(gpu).contiguous();
    std::vector<std::int64_t> host_positions(
        static_cast<size_t>(B) * MaxSeq, -1);
    for (int batch = 0; batch < B; ++batch) {
        for (int token = 0; token < Lengths[batch]; ++token) {
            host_positions[static_cast<size_t>(batch) * MaxSeq + token] = token;
        }
    }
    auto positions = mfq::cuda::tensor<std::int64_t>(host_positions)
        .reshape({B, MaxSeq}).to(gpu).contiguous();
    paged_kv_cache_write_cuda(
        k_chunk_ptrs, v_chunk_ptrs, page_table,
        k, v, positions, Page, PagesPerChunk);

    auto contiguous_o = mfq::cuda::empty({B * Hq, MaxParts, D}, fp32);
    auto contiguous_m = mfq::cuda::empty({B * Hq, MaxParts}, fp32);
    auto contiguous_l = mfq::cuda::empty_like(contiguous_m);
    auto paged_o = mfq::cuda::empty_like(contiguous_o);
    auto paged_m = mfq::cuda::empty_like(contiguous_m);
    auto paged_l = mfq::cuda::empty_like(contiguous_m);

    auto run_contiguous = [&]() {
        return attention_cache_decode_dynamic_cuda(
            q, k, v, seq_len, Scale,
            contiguous_o, contiguous_m, contiguous_l, MaxParts);
    };
    auto run_paged = [&]() {
        return attention_paged_cache_decode_cuda(
            q, k_chunk_ptrs, v_chunk_ptrs, page_table, seq_len,
            Scale, Page, PagesPerChunk, Hk,
            paged_o, paged_m, paged_l, MaxParts, true);
    };

    auto reference = host_float_values(run_contiguous());
    auto candidate = host_float_values(run_paged());
    float maximum_error = 0.0f;
    for (size_t index = 0; index < reference.size(); ++index) {
        if (!std::isfinite(candidate[index])) {
            throw std::runtime_error("Paged KV benchmark produced non-finite output");
        }
        maximum_error = std::max(
            maximum_error, std::abs(reference[index] - candidate[index]));
    }
    if (maximum_error > 2.5e-3f) {
        throw std::runtime_error(
            "Paged KV benchmark output exceeds the existing tolerance: " +
            std::to_string(maximum_error));
    }

    const auto contiguous_us = measure(
        run_contiguous, 20, iterations, samples);
    const auto paged_us = measure(run_paged, 20, iterations, samples);
    const float contiguous_median = median(contiguous_us);
    const float paged_median = median(paged_us);

    std::cout << std::fixed << std::setprecision(6)
              << "{\"shape\":{\"batch\":" << B
              << ",\"query_heads\":" << Hq
              << ",\"kv_heads\":" << Hk
              << ",\"head_dim\":" << D
              << ",\"page_size\":" << Page
              << ",\"pages_per_chunk\":" << PagesPerChunk
              << ",\"parts\":" << MaxParts
              << ",\"lengths\":[" << Lengths[0] << ',' << Lengths[1]
              << ',' << Lengths[2] << ',' << Lengths[3] << "]}"
              << ",\"iterations\":" << iterations
              << ",\"samples\":" << samples
              << ",\"maximum_absolute_error\":" << maximum_error
              << ",\"contiguous_us\":";
    print_values(contiguous_us);
    std::cout << ",\"paged_us\":";
    print_values(paged_us);
    std::cout << ",\"contiguous_median_us\":" << contiguous_median
              << ",\"paged_median_us\":" << paged_median
              << ",\"paged_change_percent\":"
              << (paged_median / contiguous_median - 1.0f) * 100.0f
              << "}\n";
    return 0;
}
