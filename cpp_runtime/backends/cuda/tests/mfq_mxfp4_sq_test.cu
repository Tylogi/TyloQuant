#include "mfq/kernels/cuda/mxfp4_sq.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {
using namespace mfq::cuda;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template<typename F> void rejects(F function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid input was accepted");
}

std::vector<std::uint8_t> read(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "fixture file missing");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void compare(const Tensor& result, const std::vector<float>& expected,
             const std::vector<float>& activation, const mfq::sq::Layout& q, int m) {
    const auto host = result.to(kCPU, kFloat32).contiguous();
    require(host.numel() == std::int64_t(m) * q.outputs, "matmul shape mismatch");
    for (int row = 0; row < m; ++row) {
        for (int n = 0; n < q.outputs; ++n) {
            double sum = 0;
            for (int k = 0; k < q.width; ++k)
                sum += double(activation[row * q.width + k]) * expected[n * q.width + k];
            const float actual = host.data_ptr<float>()[row * q.outputs + n];
            if (!std::isfinite(actual) || std::abs(actual - sum) > .02 + .006 * std::abs(sum)) {
                std::cerr << "M=" << m << " n=" << n << " actual=" << actual << " ref=" << sum << '\n';
                throw std::runtime_error("SQ matmul mismatch");
            }
        }
    }
}

void compare_backward(const Tensor& result, const std::vector<float>& expected,
                      const std::vector<float>& gradient,
                      const mfq::sq::Layout& q, int m) {
    const auto host = result.to(kCPU, kFloat32).contiguous();
    require(host.numel() == std::int64_t(m) * q.width, "backward shape mismatch");
    for (int row = 0; row < m; ++row) {
        for (int k = 0; k < q.width; ++k) {
            double sum = 0;
            for (int n = 0; n < q.outputs; ++n)
                sum += double(gradient[row * q.outputs + n]) * expected[n * q.width + k];
            const float actual = host.data_ptr<float>()[row * q.width + k];
            if (!std::isfinite(actual) || std::abs(actual - sum) > .02 + .006 * std::abs(sum)) {
                std::cerr << "backward M=" << m << " k=" << k
                          << " actual=" << actual << " ref=" << sum << '\n';
                throw std::runtime_error("SQ backward mismatch");
            }
        }
    }
}

void check_fixture(const std::filesystem::path& path, int& matmuls,
                   int& backwards, int& graphs) {
    const Device gpu{DeviceType::cuda, 0};
    auto raw = read(path);
    const auto q = mfq::sq::parse(raw.data(), raw.size());
    auto ref_path = path;
    ref_path.replace_extension(".f32");
    auto reference = read(ref_path);
    require(reference.size() == std::size_t(q.outputs) * q.width * sizeof(float), "golden size mismatch");
    std::vector<float> expected(std::size_t(q.outputs) * q.width);
    std::memcpy(expected.data(), reference.data(), reference.size());

    // CPU wire validation runs with assertions enabled even in Release.
    rejects([&] { mfq::sq::parse(raw.data(), raw.size() - 1); });
    auto damaged = raw;
    damaged.push_back(0);
    rejects([&] { mfq::sq::parse(damaged.data(), damaged.size()); });
    for (int field : {0, 3, 4, 5, 6, 7, 15, 23}) {
        damaged = raw;
        damaged[field] = 255;
        rejects([&] { mfq::sq::parse(damaged.data(), damaged.size()); });
    }
    auto blob_host = from_blob(raw.data(), {std::int64_t(raw.size())}, TensorOptions{}.dtype(kUInt8));
    auto blob = blob_host.to(gpu);
    rejects([&] { mxfp4_sq_dequant_cuda(blob_host, q.bits, q.outputs, q.width, q.base, true); });
    rejects([&] { mxfp4_sq_dequant_cuda(blob, q.bits, q.outputs, q.width + 1, q.base, true); });
    rejects([&] { mxfp4_sq_dequant_cuda(blob, q.bits, q.outputs, q.width, 252, true); });

    for (bool fp32 : {true, false}) {
        auto dense = mxfp4_sq_dequant_cuda(blob, q.bits, q.outputs, q.width, q.base, fp32).cpu();
        if (fp32) require(std::memcmp(dense.data_ptr<float>(), reference.data(), reference.size()) == 0,
                          "FP32 decode is not bit-exact");
        else {
            const auto* actual = reinterpret_cast<const std::uint16_t*>(dense.data_ptr<__half>());
            for (std::size_t i = 0; i < expected.size(); ++i) {
                const __half half_value = __float2half_rn(expected[i]);
                std::uint16_t bits;
                std::memcpy(&bits, &half_value, sizeof(bits));
                require(actual[i] == bits, "FP16 decode is not bit-exact");
            }
        }
    }
    // Extreme exponents test decode only; bounded base120 is the matmul gate.
    if (q.base != 120) return;
    for (const auto dtype : {kFloat16, kFloat32}) {
        for (int m : {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 32, 65}) {
            std::vector<float> values(std::size_t(m) * q.width);
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = float(int((i * 17 + 11) % 65) - 32) / 32;
            auto x = from_blob(values.data(), {m, q.width}, TensorOptions{}.dtype(kFloat32)).to(gpu).to(dtype);
            auto invoke = [&] { return mxfp4_sq_matmul_cuda(blob, x, q.bits, q.outputs, q.width, q.base); };
            compare(invoke(), expected, values, q, m);
            ++matmuls;
            std::vector<float> gradients(std::size_t(m) * q.outputs);
            for (std::size_t i = 0; i < gradients.size(); ++i)
                gradients[i] = float(int((i * 13 + 7) % 49) - 24) / 24;
            auto gradient = from_blob(
                gradients.data(), {m, q.outputs}, TensorOptions{}.dtype(kFloat32)
            ).to(gpu).to(dtype);
            auto invoke_backward = [&] {
                return mxfp4_sq_backward_input_cuda(
                    blob, gradient, q.bits, q.outputs, q.width, q.base);
            };
            compare_backward(invoke_backward(), expected, gradients, q, m);
            ++backwards;
            if (m >= 2 && m <= 6 && q.outputs == 33 && q.width == 96) {
                MFQ_NATIVE_CUDA_CHECK(cudaDeviceSynchronize());
                auto stream = stream_from_pool(false, 0);
                StreamGuard guard(stream);
                compare(invoke(), expected, values, q, m);
                require(default_context(0)->supports_async_allocations(), "graph gate requires async allocation");
                Graph graph;
                graph.prepare_memory();
                Tensor y = invoke();
                MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
                y = Tensor{};
                graph.capture_begin();
                y = invoke();
                graph.capture_end();
                graph.replay();
                graph.replay();
                MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
                compare(y, expected, values, q, m);
                ++graphs;
            }
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: mfq-mxfp4-sq-test FIXTURE_DIRECTORY\n"; return 2; }
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || !devices) return 77;
    try {
        int fixtures = 0, matmuls = 0, backwards = 0, graphs = 0;
        for (const auto& entry : std::filesystem::directory_iterator(argv[1])) {
            if (entry.path().extension() != ".sq") continue;
            check_fixture(entry.path(), matmuls, backwards, graphs);
            ++fixtures;
            std::cout << "PASS " << entry.path().filename().string() << std::endl;
        }
        require(
            fixtures == 30 && matmuls == 576 && backwards == 576 && graphs == 20,
            "incomplete gate coverage");
        std::cout << "PASS fixtures=" << fixtures << " matmuls=" << matmuls
                  << " backwards=" << backwards << " graphs=" << graphs << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
