#include "mfq_tensor_backend.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

using mfq_tensor_backend::Tensor;
Tensor nint4_gs24_small_m_f32_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);
Tensor nint6_gs24_small_m_f32_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);
Tensor nint_small_m_f32_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int64_t);
Tensor nint_gemv_packed_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    int64_t, Tensor, Tensor, Tensor);
Tensor nint_gemv_packed_int6_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    int64_t, Tensor, Tensor, Tensor);
Tensor nint_gemv_packed_u8_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    int64_t, Tensor, Tensor, Tensor);
Tensor nint_gemv_packed_bits_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    int64_t, int64_t, Tensor, Tensor, Tensor);

int main(int argc, char** argv) {
    using namespace mfq::cuda;
    try {
        if (argc != 5) throw std::runtime_error("usage: mfq-nint-small-m-bench bits N K f32|f16");
        const int bits = std::stoi(argv[1]), n = std::stoi(argv[2]), k = std::stoi(argv[3]);
        const std::string boundary = argv[4];
        int gs = 0, scale_bits = 0;
        switch (bits) {
            case 2: gs = 16; scale_bits = 5; break;
            case 3: gs = 24; scale_bits = 5; break;
            case 4: gs = 24; scale_bits = 6; break;
            case 5: gs = 28; scale_bits = 7; break;
            case 6: gs = 24; scale_bits = 7; break;
            case 8: gs = 48; scale_bits = 7; break;
            default: throw std::runtime_error("unsupported canonical NINT profile");
        }
        if (n < 16 || n > 16384 || k < 32 || k > 16384 ||
            (boundary != "f32" && boundary != "f16")) throw std::runtime_error("invalid benchmark geometry");
        auto stream = stream_from_pool(false, 0);
        StreamGuard guard(stream);
        const Device gpu{DeviceType::cuda, 0};
        const auto cpu = TensorOptions{}.device(kCPU);
        const auto options = TensorOptions{}.device(gpu);
        const int groups = (k + gs - 1) / gs, kpad = groups * gs;
        const int bytes = (gs * bits + 7) / 8, max_scale = (1 << scale_bits) - 1;
        auto qh = empty({n, groups, bytes}, cpu.dtype(kUInt8));
        auto sh = empty({n, groups}, cpu.dtype(kUInt8));
        auto mh = empty({n, groups}, cpu.dtype(kUInt8));
        auto nh = empty({n}, cpu.dtype(kFloat32));
        auto bh = empty({n}, cpu.dtype(kFloat32));
        std::mt19937 rng(20260907);
        for (int64_t i = 0; i < qh.numel(); ++i) qh.data_ptr<uint8_t>()[i] = rng() & 255;
        if ((gs * bits) % 8) for (int i = 0; i < n * groups; ++i)
            qh.data_ptr<uint8_t>()[i * bytes + bytes - 1] &= (1 << ((gs * bits) % 8)) - 1;
        for (int i = 0; i < n * groups; ++i) {
            sh.data_ptr<uint8_t>()[i] = rng() & max_scale;
            mh.data_ptr<uint8_t>()[i] = rng() & max_scale;
        }
        for (int row = 0; row < n; ++row) {
            nh.data_ptr<float>()[row] = 1.f / (((1 << bits) - 1) * max_scale);
            bh.data_ptr<float>()[row] = .5f / max_scale;
        }
        auto xh = empty({6, k}, cpu.dtype(kFloat32));
        for (int i = 0; i < 6 * k; ++i)
            xh.data_ptr<float>()[i] = float(int(rng() & 65535) - 32768) / 32768.f;
        auto q = qh.to(gpu), s = sh.to(gpu), sm = mh.to(gpu);
        auto ns = nh.to(gpu), nm = bh.to(gpu), x = xh.to(gpu);
        if (boundary == "f16") x = x.to(kFloat16);
        auto actual_input = x.to(kFloat32).cpu();
        auto qx = empty({6, kpad}, options.dtype(kInt8));
        auto xs = empty({6, groups}, options.dtype(kFloat32));
        auto xm = empty({6, groups}, options.dtype(kInt32));
        const uint64_t weight_bytes = q.nbytes() + s.nbytes() + sm.nbytes() + ns.nbytes() + nm.nbytes();
        std::cout << std::setprecision(10) << std::unitbuf;
        const char* fuse6_env = std::getenv("MFQ_NINT6_SMALL_M_F32_FUSE");
        const bool fuse6 = bits == 6 && boundary == "f32" && fuse6_env != nullptr && fuse6_env[0] == '1';
        const char* fuse_env = std::getenv("MFQ_NINT_SMALL_M_F32_FUSE");
        const bool fuse = bits != 4 && bits != 6 && boundary == "f32" && fuse_env != nullptr && fuse_env[0] == '1';
        const char* entry = fuse || fuse6 ? "small_m_f32_ws" : bits == 4 ? (boundary == "f32" ? "small_m_f32_ws" : "packed_ws")
            : bits == 6 ? "packed_int6_ws" : bits == 8 ? "packed_u8_ws" : "packed_bits_ws";
        std::cout << "profile=NINT" << bits << " gs=" << gs << " scale_bits=" << scale_bits
            << " N=" << n << " K=" << k << " input_output=" << boundary
            << " entry=" << entry
            << " weight_gpu_bytes=" << weight_bytes << " gpu_bpw=" << 8. * weight_bytes / (double(n) * k)
            << " seed=20260907 boundary_casts_included=1 input_quantization_included=1\n";
        for (int m = 2; m <= 6; ++m) {
            auto input = x.narrow(0, 0, m);
            auto invoke = [&]() {
                if (fuse)
                    return nint_small_m_f32_ws_cuda(q, s, sm, ns, nm, input, qx, xs, xm, bits);
                if (fuse6)
                    return nint6_gs24_small_m_f32_ws_cuda(q, s, sm, ns, nm, input, qx, xs, xm);
                if (bits == 4 && boundary == "f32")
                    return nint4_gs24_small_m_f32_ws_cuda(q, s, sm, ns, nm, input, qx, xs, xm);
                auto half_input = input.to(kFloat16);
                auto half_output = bits == 4
                    ? nint_gemv_packed_ws_cuda(q, s, sm, ns, nm, half_input, gs, qx, xs, xm)
                    : bits == 6
                    ? nint_gemv_packed_int6_ws_cuda(q, s, sm, ns, nm, half_input, gs, qx, xs, xm)
                    : bits == 8
                    ? nint_gemv_packed_u8_ws_cuda(q, s, sm, ns, nm, half_input, gs, qx, xs, xm)
                    : nint_gemv_packed_bits_ws_cuda(q, s, sm, ns, nm, half_input, gs, bits, qx, xs, xm);
                return half_output.to(input.scalar_type());
            };
            auto values = invoke().to(kFloat32).cpu();
            double squared = 0., norm = 0.;
            for (int token = 0; token < m; ++token) for (int sample = 0; sample < 16; ++sample) {
                const int row = sample * (n - 1) / 15;
                double reference = 0.;
                for (int column = 0; column < k; ++column) {
                    const int group = column / gs, index = column % gs;
                    unsigned code = 0;
                    for (int bit = 0; bit < bits; ++bit) {
                        const int offset = index * bits + bit;
                        code |= ((qh.data_ptr<uint8_t>()[(row * groups + group) * bytes + offset / 8]
                            >> (offset % 8)) & 1) << bit;
                    }
                    const double weight = double(nh.data_ptr<float>()[row]) * sh.data_ptr<uint8_t>()[row * groups + group] * code
                        - double(bh.data_ptr<float>()[row]) * mh.data_ptr<uint8_t>()[row * groups + group];
                    reference += weight * actual_input.data_ptr<float>()[token * k + column];
                }
                const double actual = values.data_ptr<float>()[token * n + row];
                if (!std::isfinite(actual)) throw std::runtime_error("nonfinite benchmark output");
                squared += (actual - reference) * (actual - reference);
                norm += reference * reference;
            }
            const double nmse = squared / std::max(norm, 1.e-30);
            if (nmse > 5.e-4) throw std::runtime_error("benchmark CPU oracle NMSE exceeds 5e-4");
            const uint64_t op_bytes = weight_bytes + uint64_t(m) * (n + k) * input.element_size();
            const int operations = int(std::min<uint64_t>(8192, (32ull << 30) / op_bytes + 1));
            Graph graph;
            graph.prepare_memory();
            Tensor output;
            auto repeat = [&]() {
                for (int i = 0; i < operations; ++i) {
                    output = Tensor{};
                    output = invoke();
                }
            };
            // Warm every temporary allocation before capture, then reuse the
            // same outputs and weights like test-backend-ops' resident graph.
            repeat();
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            output = Tensor{};
            graph.capture_begin();
            repeat();
            graph.capture_end();
            graph.replay();
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            uint64_t repeats = 0;
            double seconds = 0.;
            do {
                const auto start = std::chrono::steady_clock::now();
                graph.replay();
                MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
                seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                ++repeats;
            } while (seconds < 1.);
            std::cout << "M=" << m << " cpu_oracle_rows=16 nmse=" << nmse
                << " graph_operations=" << operations << " graph_replays=" << repeats
                << " us_per_op=" << seconds * 1.e6 / (repeats * operations) << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
