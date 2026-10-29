#include "mfq_weight_memory.h"
#include "mfq_cuda_context.h"
#include "mfq_native_tensor.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(float actual, float expected, float tolerance, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            std::string(message) + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
    }
}

std::vector<float> host_values(const mfq::cuda::Tensor& value) {
    auto host = value.contiguous().to(mfq::cuda::kCPU, mfq::cuda::kFloat32);
    return std::vector<float>(
        host.data_ptr<float>(), host.data_ptr<float>() + host.numel());
}

std::vector<std::int64_t> host_int64_values(const mfq::cuda::Tensor& value) {
    auto host = value.contiguous().to(mfq::cuda::kCPU, mfq::cuda::kInt64);
    return std::vector<std::int64_t>(
        host.data_ptr<std::int64_t>(),
        host.data_ptr<std::int64_t>() + host.numel());
}

void set_test_environment(const char* name, const char* value) {
#ifdef _WIN32
    if (_putenv_s(name, value) != 0) {
        throw std::runtime_error("failed to set test environment variable");
    }
#else
    if (setenv(name, value, 1) != 0) {
        throw std::runtime_error("failed to set test environment variable");
    }
#endif
}

}  // namespace

void check_weight_memory() {
    using namespace mfq::cuda;
    WeightMemoryLedger ledger,nested;
    Tensor weight,view;
    auto table=std::make_shared<int>(1);
    {
        const WeightMemoryLoadScope loading(ledger);
        auto workspace=zeros({16384},TensorOptions{}.device(kCUDA));
        {
            const WeightAllocationScope weights;
            weight=zeros({16},TensorOptions{}.dtype(kFloat32).device(kCUDA));
            view=weight.narrow(0,0,4);
            auto replaced=zeros({32},weight.options());
            require(ledger.resident_bytes()==192,"weight ledger excludes unrelated workspace and shared views");
            {
                const WeightMemoryLoadScope inner(nested);
                auto other=zeros({8},weight.options());
                require(nested.resident_bytes()==32 && ledger.resident_bytes()==192,"nested weight ledger isolates engines");
            }
            require(nested.resident_bytes()==0,"weight ledger releases expired allocations");
        }
        observe_streamed_ple(table,4096);
        require(ledger.resident_bytes()==64 && ledger.ple_bytes()==4096,"weight ledger drops replaced projections and counts PLE payload");
    }
    weight={};
    require(ledger.resident_bytes()==64,"weight view retains its full allocation");
    view={};table.reset();
    require(ledger.resident_bytes()==0 && ledger.ple_bytes()==0,"telemetry must not retain weights or tables");
}

int main() try {
    int devices = 0;
    const auto device_status = cudaGetDeviceCount(&devices);
    if (device_status != cudaSuccess || devices == 0) {
        (void)cudaGetLastError();
        return 77;
    }

    using namespace mfq::cuda;
    const Device cuda_device{DeviceType::cuda, 0};
    check_weight_memory();

    auto empty_host_blob = from_blob(
        static_cast<void*>(nullptr), {0}, TensorOptions{}.dtype(kFloat32));
    auto empty_device_blob = empty_host_blob.clone().to(cuda_device);
    require(
        empty_device_blob.defined() && empty_device_blob.numel() == 0,
        "empty null blob CUDA transfer");

    auto input = tensor<float>({1, 2, 3, 4, 5, 6})
        .reshape({2, 3})
        .to(cuda_device);
#ifdef __HIP_PLATFORM_AMD__
    // Cross-device peer copies through the backend's async-pool allocations
    // fault in ROCr's copy kernel on this stack (reproducer characterized;
    // see docs/hip-native-runtime-validation.md). Single-device coverage
    // below remains active.
    if (false) {
#else
    if (devices > 1) {
#endif
        const Device second_device{DeviceType::cuda, 1};
        for (const auto [source_device, destination_device] :
             {std::pair{0, 1}, std::pair{1, 0}}) {
            int can_access = 0;
            MFQ_NATIVE_CUDA_CHECK(cudaDeviceCanAccessPeer(
                &can_access, source_device, destination_device));
            if (can_access) {
                DeviceGuard source_guard(source_device);
                const auto status = cudaDeviceEnablePeerAccess(
                    destination_device, 0);
                if (status == cudaErrorPeerAccessAlreadyEnabled) {
                    (void)cudaGetLastError();
                } else {
                    MFQ_NATIVE_CUDA_CHECK(status);
                }
            }
        }
            auto round_trip = input.to(second_device).to(cuda_device);
            require(
            host_values(round_trip) ==
                std::vector<float>({1, 2, 3, 4, 5, 6}),
            "cross-device peer copy values");
            auto strided_round_trip = input.narrow(1, 1, 2)
            .to(second_device)
            .to(cuda_device);
            require(
            host_values(strided_round_trip) ==
                std::vector<float>({2, 3, 5, 6}),
            "cross-device strided peer copy values");
    }
    auto elementwise = (input + 2.0) * 3.0;
    const auto elementwise_host = host_values(elementwise);
    require_close(elementwise_host[0], 9.0f, 0.0f, "elementwise first value");
    require_close(elementwise_host[5], 24.0f, 0.0f, "elementwise final value");

    const auto reduced = host_values(elementwise.sum(1));
    require_close(reduced[0], 36.0f, 0.0f, "row reduction zero");
    require_close(reduced[1], 63.0f, 0.0f, "row reduction one");
    const auto row_means = host_values(elementwise.mean(1, true));
    require_close(row_means[0], 12.0f, 0.0f, "row mean zero");
    require_close(row_means[1], 21.0f, 0.0f, "row mean one");

    auto bf16_argmax_input = tensor<float>({1, 5, 5, 2, 7, 3, 7, 7})
        .reshape({2, 4})
        .to(cuda_device)
        .to(kBFloat16);
    require(
        host_int64_values(bf16_argmax_input.argmax(-1)) ==
            std::vector<std::int64_t>({1, 0}),
        "contiguous BF16 argmax first-index ties");
    require(
        host_int64_values(bf16_argmax_input.transpose(0, 1).argmax(-1)) ==
            std::vector<std::int64_t>({1, 0, 1, 1}),
        "non-contiguous BF16 argmax fallback");

    const auto transposed = host_values(input.transpose(0, 1).contiguous());
    const std::vector<float> expected_transposed{1, 4, 2, 5, 3, 6};
    require(transposed == expected_transposed, "non-contiguous materialization order");

    std::vector<float> head_major_values(1 * 2 * 3 * 128);
    for (std::size_t index = 0; index < head_major_values.size(); ++index) {
        head_major_values[index] = static_cast<float>(index % 64);
    }
    auto head_major = tensor<float>(head_major_values)
        .reshape({1, 2, 3, 128})
        .to(cuda_device)
        .to(kBFloat16);
    const auto token_major = host_values(
        head_major.transpose(1, 2).contiguous());
    std::vector<float> expected_token_major;
    expected_token_major.reserve(head_major_values.size());
    for (std::int64_t token = 0; token < 3; ++token) {
        for (std::int64_t head = 0; head < 2; ++head) {
            const auto begin = (head * 3 + token) * 128;
            expected_token_major.insert(
                expected_token_major.end(),
                head_major_values.begin() + begin,
                head_major_values.begin() + begin + 128);
        }
    }
    require(
        token_major == expected_token_major,
        "BF16 head-to-token materialization order");

    auto right = tensor<float>({7, 8, 9, 10, 11, 12})
        .reshape({3, 2})
        .to(cuda_device);
    const auto product = host_values(matmul(input, right));
    const std::vector<float> expected_product{58, 64, 139, 154};
    require(product == expected_product, "cuBLAS row-major matmul");

    for (const auto dtype : {kFloat32, kFloat16, kBFloat16, kFloat64}) {
        auto padded = tensor<float>({1, 2, 3, 99, 4, 5, 6, 99})
            .reshape({2, 4}).to(cuda_device).to(dtype);
        auto sliced = padded.narrow(1, 0, 3);
        auto ones_column = ones({3, 1}, sliced.options());
        require(host_values(matmul(sliced, ones_column)) == std::vector<float>({6, 15}),
                "matmul narrowed left leading dimension");
        require(host_values(matmul(ones_column.transpose(0, 1), sliced.transpose(0, 1))) ==
                    std::vector<float>({6, 15}), "matmul transposed narrowed right");
        require(host_values(matmul(sliced, sliced.transpose(0, 1))) ==
                    std::vector<float>({14, 32, 32, 77}), "matmul both padded operands");
        auto stepped = padded.slice(1, 0, 4, 2);
        require(host_values(matmul(stepped, stepped.transpose(0, 1))) ==
                    std::vector<float>({10, 22, 22, 52}), "matmul strided inner materialization");
        require(host_values(matmul(sliced.select(0, 1), ones_column.squeeze(-1))) ==
                    std::vector<float>({15}), "matmul vector promotion");
        auto expanded = sliced.narrow(0, 0, 1).expand({2, 3});
        require(host_values(matmul(expanded, ones_column)) == std::vector<float>({6, 6}),
                "matmul overlapping broadcast matrix");
        require(host_values(matmul(empty({2, 0}, sliced.options()), empty({0, 3}, sliced.options()))) ==
                    std::vector<float>(6, 0), "matmul empty contraction");
        auto batched = padded.unsqueeze(0).expand({4, 2, 4}).narrow(-1, 0, 3);
        require(host_values(matmul(batched, ones_column)) ==
                    std::vector<float>({6, 15, 6, 15, 6, 15, 6, 15}),
                "matmul strided batched broadcast with padding");
    }

    auto batched_left = tensor<float>({1, 2, 3, 4, 5, 6, 7, 8})
        .reshape({2, 2, 2})
        .to(cuda_device);
    auto broadcast_right = tensor<float>({1, 2})
        .reshape({1, 2, 1})
        .to(cuda_device);
    require(
        host_values(matmul(batched_left, broadcast_right)) ==
            std::vector<float>({5, 11, 17, 23}),
        "batched matmul broadcasting");

    std::vector<float> attention_left_values(4 * 32 * 128);
    std::vector<float> attention_right_values(4 * 64 * 128);
    for (std::size_t index = 0; index < attention_left_values.size(); ++index) {
        attention_left_values[index] =
            static_cast<float>(static_cast<int>(index % 31) - 15) / 32.0f;
    }
    for (std::size_t index = 0; index < attention_right_values.size(); ++index) {
        attention_right_values[index] =
            static_cast<float>(static_cast<int>(index % 29) - 14) / 32.0f;
    }
    auto attention_left = tensor<float>(attention_left_values)
        .reshape({1, 4, 32, 128}).to(cuda_device).to(kBFloat16);
    auto attention_right = tensor<float>(attention_right_values)
        .reshape({1, 4, 64, 128}).to(cuda_device).to(kBFloat16)
        .transpose(-2, -1);
    set_test_environment("MFQ_DISABLE_NATIVE_PARALLEL_BATCH_MATMUL", "1");
    set_test_environment("MFQ_DISABLE_NATIVE_STRIDED_BATCH_MATMUL", "1");
    const auto loop_batched_product = host_values(
        matmul(attention_left, attention_right));
    set_test_environment("MFQ_DISABLE_NATIVE_PARALLEL_BATCH_MATMUL", "0");
    const auto parallel_batched_product = host_values(
        matmul(attention_left, attention_right));
    set_test_environment("MFQ_DISABLE_NATIVE_PARALLEL_BATCH_MATMUL", "1");
    require(
        parallel_batched_product == loop_batched_product,
        "parallel batched matmul exactness");
    set_test_environment("MFQ_DISABLE_NATIVE_STRIDED_BATCH_MATMUL", "0");
    const auto strided_batched_product = host_values(
        matmul(attention_left, attention_right));
    for (std::size_t index = 0; index < strided_batched_product.size(); ++index) {
        const auto reference = loop_batched_product[index];
        require_close(
            strided_batched_product[index], reference,
            0.01f + 0.01f * std::abs(reference),
            "strided batched matmul parity");
    }

    std::vector<float> selection_values(3 * 2051);
    for (std::size_t i = 0; i < selection_values.size(); ++i) {
        selection_values[i] = static_cast<float>((i * 17) % 127) - 63;
    }
    for (const auto dtype : {kFloat32, kFloat16, kBFloat16}) {
        auto selection = tensor<float>(selection_values).reshape({3, 2051})
            .to(cuda_device).to(dtype);
        for (const bool largest : {false, true}) {
            auto [sorted_values, sorted_indices] = sort(selection, -1, largest);
            for (const int k : {0, 1, 7, 256}) {
                auto [values, indices] = topk(selection, k, -1, largest, true);
                require(host_values(values) == host_values(sorted_values.narrow(-1, 0, k)),
                        "tiled topk values");
                require(host_int64_values(indices) == host_int64_values(sorted_indices.narrow(-1, 0, k)),
                        "tiled topk stable indices");
            }
        }
    }
    require(std::get<0>(sort(empty({2, 0}, input.options()), -1, false)).numel() == 0,
            "sort empty last dimension");

    auto mixed_left = tensor<float>({1, 2, 3, 4}).reshape({2, 1, 1, 2}).to(cuda_device);
    auto mixed_right = tensor<float>({5, 6, 7, 8, 9, 10}).reshape({1, 3, 2, 1}).to(cuda_device);
    require(host_values(matmul(mixed_left, mixed_right)) ==
                std::vector<float>({17, 23, 29, 39, 53, 67}), "mixed batch broadcasting");
    if (default_context(0)->supports_async_allocations()) {
        Graph matrix_graph;
        Tensor captured;
        matrix_graph.prepare_memory();
        captured = matmul(attention_left, attention_right);
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream(0).stream()));
        captured = {};
        matrix_graph.capture_begin();
        captured = matmul(attention_left, attention_right);
        matrix_graph.capture_end();
        matrix_graph.replay();
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream(0).stream()));
        require(
            host_values(captured) == strided_batched_product,
            "strided batched CUDA graph");
    }

    const auto probabilities = host_values(softmax(input, -1));
    require_close(probabilities[0], 0.09003057f, 1.0e-6f, "softmax first value");
    require_close(probabilities[2], 0.66524096f, 1.0e-6f, "softmax final value");
    require_close(
        probabilities[3] + probabilities[4] + probabilities[5],
        1.0f, 2.0e-6f, "softmax row normalization");

    auto query = tensor<float>({1, 0, 0, 1})
        .reshape({1, 1, 2, 2})
        .to(cuda_device);
    auto key = query.clone();
    auto value = tensor<float>({10, 20, 30, 40})
        .reshape({1, 1, 2, 2})
        .to(cuda_device);
    const auto attended = host_values(scaled_dot_product_attention(
        query, key, value, std::nullopt, 0.0, false, 1.0, false));
    for (const auto dtype : {kFloat32, kFloat16, kBFloat16}) {
        auto q = (arange(4 * 259 * 16, input.options()).remainder(17) / 32.0)
            .reshape({1, 4, 259, 16}).to(dtype);
        auto k = (arange(2 * 263 * 16, input.options()).remainder(13) / 32.0)
            .reshape({1, 2, 263, 16}).to(dtype);
        auto v = (arange(2 * 263 * 16, input.options()).remainder(19) / 32.0)
            .reshape({1, 2, 263, 16}).to(dtype);
        for (int mask_mode = 0; mask_mode < 4; ++mask_mode) {
            const bool causal = mask_mode == 1;
            std::optional<Tensor> mask;
            if (mask_mode == 2) mask = arange(259 * 263, input.options()).remainder(7)
                .reshape({259, 263}).ne(0);
            if (mask_mode == 3) mask = (arange(263, input.options()).remainder(7) / -4.0)
                .reshape({1, 263}).to(dtype);
            set_test_environment("MFQ_DISABLE_NATIVE_TILED_SDPA", "1");
            auto reference = host_values(scaled_dot_product_attention(q,
                k.repeat_interleave(2, 1), v.repeat_interleave(2, 1),
                mask, 0.0, causal, std::nullopt, false));
            set_test_environment("MFQ_DISABLE_NATIVE_TILED_SDPA", "0");
            auto actual = host_values(scaled_dot_product_attention(q, k, v,
                mask, 0.0, causal, std::nullopt, true));
            const float tolerance = dtype == kFloat32 ? 2e-5f : dtype == kFloat16 ? 0.001f : 0.004f;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                require_close(actual[i], reference[i], tolerance, "tiled grouped-query attention");
            }
        }
    }
    const float high = std::exp(1.0f) / (std::exp(1.0f) + 1.0f);
    require_close(attended[0], high * 10.0f + (1.0f - high) * 30.0f,
                  2.0e-5f, "attention query zero");
    require_close(attended[3], (1.0f - high) * 20.0f + high * 40.0f,
                  2.0e-5f, "attention query one");

    set_test_environment("MFQ_DISABLE_NATIVE_FUSED_CAUSAL_SCALE", "1");
    const auto causal_scale_reference = host_values(
        scaled_dot_product_attention(
            attention_left, attention_left, attention_left,
            std::nullopt, 0.0, true, 0.08838834764831845, false));
    set_test_environment("MFQ_DISABLE_NATIVE_FUSED_CAUSAL_SCALE", "0");
    const auto causal_scale_candidate = host_values(
        scaled_dot_product_attention(
            attention_left, attention_left, attention_left,
            std::nullopt, 0.0, true, 0.08838834764831845, false));
    set_test_environment("MFQ_DISABLE_NATIVE_FUSED_CAUSAL_SCALE", "1");
    require(
        causal_scale_candidate == causal_scale_reference,
        "fused BF16 causal scale exactness");

    const auto conversion_input = tensor<float>({
        -65504.0f, -3.1415927f, -0.0f, 0.0f,
        0.00006103515625f, 0.33333334f, 1.0f, 65504.0f})
        .to(cuda_device).to(kBFloat16).contiguous();
    set_test_environment("MFQ_DISABLE_NATIVE_CONTIGUOUS_BF16_TO_F16", "1");
    const auto conversion_reference = host_values(
        conversion_input.to(kFloat16));
    set_test_environment("MFQ_DISABLE_NATIVE_CONTIGUOUS_BF16_TO_F16", "0");
    const auto conversion_candidate = host_values(
        conversion_input.to(kFloat16));
    require(
        conversion_candidate == conversion_reference,
        "contiguous BF16 to F16 conversion exactness");

    set_test_environment("MFQ_DISABLE_NATIVE_EXACT_BF16_SOFTMAX", "1");
    const auto softmax_reference = host_values(
        scaled_dot_product_attention(
            attention_left, attention_left, attention_left,
            std::nullopt, 0.0, true, 0.08838834764831845, false));
    set_test_environment("MFQ_DISABLE_NATIVE_EXACT_BF16_SOFTMAX", "0");
    const auto softmax_candidate = host_values(
        scaled_dot_product_attention(
            attention_left, attention_left, attention_left,
            std::nullopt, 0.0, true, 0.08838834764831845, false));
    set_test_environment("MFQ_DISABLE_NATIVE_EXACT_BF16_SOFTMAX", "1");
    require(
        softmax_candidate == softmax_reference,
        "exact BF16 softmax specialization");
    auto signal = tensor<float>({1, 2, 3, 4}).reshape({1, 1, 4}).to(cuda_device);
    auto filter = tensor<float>({1, 1}).reshape({1, 1, 2}).to(cuda_device);
    constexpr std::array<std::int64_t, 1> stride1{1};
    constexpr std::array<std::int64_t, 1> padding1{0};
    constexpr std::array<std::int64_t, 1> dilation1{1};
    const auto convolved = host_values(conv1d(
        signal, filter, Tensor{}, stride1, padding1, dilation1, 1));
    require(convolved == std::vector<float>({3, 5, 7}), "conv1d values");

    auto image = tensor<float>({1, 2, 3, 4, 5, 6, 7, 8, 9})
        .reshape({1, 1, 3, 3})
        .to(cuda_device);
    auto kernel = ones({1, 1, 2, 2}, TensorOptions{}.dtype(kFloat32).device(cuda_device));
    constexpr std::array<std::int64_t, 2> stride2{1, 1};
    constexpr std::array<std::int64_t, 2> padding2{0, 0};
    constexpr std::array<std::int64_t, 2> dilation2{1, 1};
    const auto image_convolved = host_values(conv2d(
        image, kernel, Tensor{}, stride2, padding2, dilation2, 1));
    require(
        image_convolved == std::vector<float>({12, 16, 24, 28}),
        "conv2d values");

    constexpr std::array<std::int64_t, 1> pool_kernel{2};
    constexpr std::array<std::int64_t, 1> pool_stride{2};
    const auto pooled = host_values(avg_pool1d(
        signal, pool_kernel, pool_stride, padding1));
    require(pooled == std::vector<float>({1.5f, 3.5f}), "avg_pool1d values");

    auto scatter_indices = tensor<std::int64_t>({0, 1, 1, 3, 0, 3})
        .reshape({2, 3})
        .to(cuda_device);
    auto scatter_source = tensor<float>({1, 2, 3, 4, 5, 6})
        .reshape({2, 3})
        .to(cuda_device);
    auto scatter_output = zeros(
        {2, 4}, TensorOptions{}.dtype(kFloat32).device(cuda_device));
    scatter_output.scatter_add_(1, scatter_indices, scatter_source);
    require(
        host_values(scatter_output) ==
            std::vector<float>({1, 5, 0, 0, 5, 0, 0, 10}),
        "scatter_add float values");

    constexpr std::int64_t exact_large = 9007199254740993LL;
    auto integer_scatter_indices = tensor<std::int64_t>({0, 0}).to(cuda_device);
    auto integer_scatter_source = tensor<std::int64_t>({exact_large, 2}).to(cuda_device);
    auto integer_scatter_output = zeros(
        {1}, TensorOptions{}.dtype(kInt64).device(cuda_device));
    integer_scatter_output.scatter_add_(
        0, integer_scatter_indices, integer_scatter_source);
    require(
        host_int64_values(integer_scatter_output) ==
            std::vector<std::int64_t>({exact_large + 2}),
        "scatter_add int64 exactness");

    auto cumulative_input = tensor<std::int64_t>({exact_large, 1, 1})
        .reshape({1, 3})
        .to(cuda_device);
    require(
        host_int64_values(cumsum(cumulative_input, -1)) ==
            std::vector<std::int64_t>({exact_large, exact_large + 1, exact_large + 2}),
        "cumsum int64 exactness");
    require(
        host_int64_values(cumulative_input.sum(-1)) ==
            std::vector<std::int64_t>({exact_large + 2}),
        "sum int64 exactness");

    auto fractional_cumulative = cumsum(
        tensor<float>({0.5f, 0.25f, 0.125f}).to(cuda_device).to(kBFloat16), -1);
    const auto fractional_values = host_values(fractional_cumulative);
    require_close(fractional_values[0], 0.5f, 0.0f, "cumsum BF16 first value");
    require_close(fractional_values[2], 0.875f, 0.0f, "cumsum BF16 final value");

    auto bf16_scatter_output = zeros(
        {1}, TensorOptions{}.dtype(kBFloat16).device(cuda_device));
    bf16_scatter_output.scatter_add_(
        0,
        tensor<std::int64_t>({0, 0}).to(cuda_device),
        tensor<float>({1.5f, 1.5f}).to(cuda_device).to(kBFloat16));
    require_close(
        host_values(bf16_scatter_output)[0], 3.0f, 0.0f,
        "scatter_add BF16 value");

    auto unsorted = tensor<float>({3, 1, 4, 2}).reshape({1, 4}).to(cuda_device);
    auto [largest, largest_indices] = topk(unsorted, 2, -1, true, true);
    require(host_values(largest) == std::vector<float>({4, 3}), "topk values");
    auto index_host = largest_indices.to(kCPU).contiguous();
    require(index_host.data_ptr<std::int64_t>()[0] == 2, "topk first index");
    require(index_host.data_ptr<std::int64_t>()[1] == 0, "topk second index");

    auto usage_stream = stream_from_pool(false, 0);
    auto cross_stream = input.square();
    Event producer_ready;
    producer_ready.record(current_stream(0).stream());
    Tensor consumed;
    {
        StreamGuard stream_guard(usage_stream);
        MFQ_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(usage_stream.stream(), producer_ready.get(), 0));
        cross_stream.record_stream(
            reinterpret_cast<std::uintptr_t>(usage_stream.stream()));
        MFQ_NATIVE_CUDA_CHECK(cudaLaunchHostFunc(usage_stream.stream(), [](void*) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }, nullptr));
        consumed = cross_stream.square();
        cross_stream = {}; // release before the registered consumer executes
    }
    MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(usage_stream.stream()));
    require_close(host_values(consumed)[5], 1296.0f, 0.0f, "cross-stream lifetime");

    if (default_context(0)->supports_async_allocations()) {
        auto graph_stream = stream_from_pool(false, 0);
        StreamGuard graph_guard(graph_stream);
        {
            Graph cold_graph;
            bool rejected_unwarmed_allocation = false;
            try {
                cold_graph.capture_begin();
                (void)input.square();
                cold_graph.capture_end();
            } catch (const Error& error) {
                rejected_unwarmed_allocation =
                    std::string(error.what()).find("un-warmed allocation") !=
                    std::string::npos;
                cold_graph.reset();
            }
            require(
                rejected_unwarmed_allocation,
                "graph capture must reject an un-warmed allocation");
            require_close(
                host_values(input.square())[5], 36.0f, 0.0f,
                "stream remains usable after rejected graph capture");
        }
        Graph graph;
        Tensor graph_output;
        graph.prepare_memory();
        graph_output = input.square();
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
        graph_output = Tensor{};
        graph.capture_begin();
        graph_output = input.square();
        graph.capture_end();
        graph.replay();
        graph.replay();
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
        require_close(
            host_values(graph_output)[5], 36.0f, 0.0f,
            "back-to-back graph replay value");
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
