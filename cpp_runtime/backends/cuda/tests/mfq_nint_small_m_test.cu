#include "mfq_tensor_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using mfq_tensor_backend::Tensor;
Tensor nint4_gs24_small_m_f32_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);
Tensor nint6_gs24_small_m_f32_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);
Tensor nint_small_m_f32_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int64_t);

#define DECLARE_GLU(name) \
Tensor name(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, Tensor, Tensor, Tensor)
DECLARE_GLU(nint_gemv_packed_swiglu_ws_cuda);
DECLARE_GLU(nint_gemv_packed_geglu_ws_cuda);
#undef DECLARE_GLU
#define DECLARE_BITS_GLU(name) \
Tensor name(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, int64_t, Tensor, Tensor, Tensor)
DECLARE_BITS_GLU(nint_gemv_packed_bits_swiglu_ws_cuda);
DECLARE_BITS_GLU(nint_gemv_packed_bits_geglu_ws_cuda);
#undef DECLARE_BITS_GLU
Tensor nint_gemv_packed_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    int64_t, Tensor, Tensor, Tensor);
Tensor nint_gemv_packed_int6_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    int64_t, Tensor, Tensor, Tensor);
Tensor nint_gemv_packed_u8_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    int64_t, Tensor, Tensor, Tensor);
Tensor nint_gemv_packed_bits_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    int64_t, int64_t, Tensor, Tensor, Tensor);
Tensor nint_gemv_packed_gate_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    int64_t, int64_t, Tensor, Tensor, Tensor);
Tensor nint_gemv_packed_bits_gate_ws_cuda(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    int64_t, int64_t, int64_t, Tensor, Tensor, Tensor);

namespace {
using namespace mfq::cuda;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void exact(const Tensor& actual, const Tensor& expected) {
    auto a = actual.contiguous().cpu();
    auto e = expected.contiguous().cpu();
    require(a.sizes() == e.sizes() && a.scalar_type() == e.scalar_type(), "small-M shape or dtype mismatch");
    require(std::memcmp(a.data_ptr(), e.data_ptr(), a.nbytes()) == 0,
        "small-M output differs from serial M=1 bits");
}

void check(int bits, int gs, int scale_bits, int width, int& cases, int& graphs, int& f32_cases, int rows = 17) {
    constexpr int batch = 6;
    const int groups = (width + gs - 1) / gs, kpad = groups * gs;
    const int bytes = (gs * bits + 7) / 8;
    const auto cpu = TensorOptions{}.device(kCPU);
    const Device gpu{DeviceType::cuda, 0};
    auto qh = zeros({2 * rows, groups, bytes}, cpu.dtype(kUInt8));
    auto sh = empty({2 * rows, groups}, cpu.dtype(kUInt8));
    auto mh = empty({2 * rows, groups}, cpu.dtype(kUInt8));
    auto nh = empty({2 * rows}, cpu.dtype(kFloat32));
    auto bh = empty({2 * rows}, cpu.dtype(kFloat32));
    for (int r = 0; r < 2 * rows; ++r) {
        nh.data_ptr<float>()[r] = std::ldexp(float(1 + r % 3), -bits - 15);
        bh.data_ptr<float>()[r] = float(1 + r % 5) / 4096.f;
        for (int g = 0; g < groups; ++g) {
            sh.data_ptr<uint8_t>()[r * groups + g] = (r * 7 + g * 11) % (1 << scale_bits);
            mh.data_ptr<uint8_t>()[r * groups + g] = (r * 13 + g * 3) % 9;
            for (int k = 0; k < gs; ++k) {
                const unsigned code = (r * 17 + g * 7 + k * 13) & ((1 << bits) - 1);
                for (int bit = 0; bit < bits; ++bit) {
                    const int offset = k * bits + bit;
                    qh.data_ptr<uint8_t>()[(r * groups + g) * bytes + offset / 8] |=
                        ((code >> bit) & 1) << (offset % 8);
                }
            }
        }
    }
    auto xh = empty({batch, width}, cpu.dtype(kFloat16));
    for (int m = 0; m < batch; ++m)
        for (int k = 0; k < width; ++k)
            xh.data_ptr<__half>()[m * width + k] = __float2half_rn(
                float((m * 29 + k * 17) % 255 - 127) / 64.f);
    auto q = qh.to(gpu), s = sh.to(gpu), sm = mh.to(gpu);
    auto ns = nh.to(gpu), nm = bh.to(gpu), x = xh.to(gpu);
    auto qx = empty({batch, kpad}, TensorOptions{}.device(gpu).dtype(kInt8));
    auto xs = empty({batch, groups}, TensorOptions{}.device(gpu).dtype(kFloat32));
    auto xm = empty({batch, groups}, TensorOptions{}.device(gpu).dtype(kInt32));
    auto invoke = [&](const Tensor& input, int operation) {
        if (operation == 2 && (bits == 2 || bits == 3 || bits == 5))
            return nint_gemv_packed_bits_ws_cuda(q, s, sm, ns, nm, input, gs, bits, qx, xs, xm);
        if (operation == 2 && bits == 8)
            return nint_gemv_packed_u8_ws_cuda(q, s, sm, ns, nm, input, gs, qx, xs, xm);
        if (operation == 2) return (bits == 4 ? nint_gemv_packed_ws_cuda
            : nint_gemv_packed_int6_ws_cuda)(q, s, sm, ns, nm, input,
            gs, qx, xs, xm);
        if (operation >= 3 && bits != 4) return nint_gemv_packed_bits_gate_ws_cuda(
            q, s, sm, ns, nm, input, input, gs, bits, operation - 2, qx, xs, xm);
        if (operation >= 3) return nint_gemv_packed_gate_ws_cuda(q, s, sm, ns, nm,
            input, input, gs, operation - 2, qx, xs, xm);
        if (bits == 4) return (operation == 0 ? nint_gemv_packed_swiglu_ws_cuda
            : nint_gemv_packed_geglu_ws_cuda)(q, s, sm, ns, nm, input, gs, qx, xs, xm);
        return (operation == 0 ? nint_gemv_packed_bits_swiglu_ws_cuda
            : nint_gemv_packed_bits_geglu_ws_cuda)(q, s, sm, ns, nm, input, gs, bits, qx, xs, xm);
    };
    const int operations = bits == 4 || bits == 6 ? 5 : 3;
    for (int operation = 0; operation < operations; ++operation) {
        std::vector<Tensor> serial;
        for (int m = 0; m < batch; ++m) serial.push_back(invoke(x.narrow(0, m, 1), operation));
        auto reference = cat(serial, 0);
        for (int m = 1; m <= batch; ++m) {
            exact(invoke(x.narrow(0, 0, m), operation), reference.narrow(0, 0, m));
            ++cases;
        }
        // Independent CPU packed-weight/INT8-input oracle for fused GLU.
        // Uses actual CUDA input codes/scales, isolating the changed projection
        // and GLU arithmetic from the unchanged input quantizer.
        if (operation <= 1) {
            auto y = invoke(x, operation).to(kFloat32).cpu();
            auto aq = qx.cpu();
            auto as = xs.cpu();
            for (int m = 0; m < batch; ++m) {
                for (int r = 0; r < rows; ++r) {
                    double projection[2]{};
                    for (int p = 0; p < 2; ++p) {
                        const int row = r + p * rows;
                        for (int g = 0; g < groups; ++g) {
                            for (int k = 0; k < gs; ++k) {
                                unsigned code = 0;
                                for (int bit = 0; bit < bits; ++bit) {
                                    const int offset = k * bits + bit;
                                    code |= ((qh.data_ptr<uint8_t>()[(row * groups + g) * bytes + offset / 8]
                                        >> (offset % 8)) & 1) << bit;
                                }
                                const double weight = double(nh.data_ptr<float>()[row])
                                    * sh.data_ptr<uint8_t>()[row * groups + g] * code
                                    - double(bh.data_ptr<float>()[row]) * mh.data_ptr<uint8_t>()[row * groups + g];
                                projection[p] += weight * as.data_ptr<float>()[m * groups + g]
                                    * aq.data_ptr<int8_t>()[m * kpad + g * gs + k];
                            }
                        }
                        projection[p] = __half2float(__float2half_rn(float(projection[p])));
                    }
                    const double gate = projection[0], up = projection[1];
                    const double value = operation == 0 ? up * gate / (1. + std::exp(-gate))
                        : up * .5 * gate * (1. + std::tanh(.7978845608028654 * gate * (1. + .044715 * gate * gate)));
                    const double expected = __half2float(__float2half_rn(float(value)));
                    const double actual = y.data_ptr<float>()[m * rows + r];
                    require(std::isfinite(actual) && std::abs(actual - expected) <= .002 + .002 * std::abs(expected),
                        "small-M GLU CPU projection oracle mismatch");
                }
            }
        }
        if (operation == 2 && bits != 4) {
            auto actual = invoke(x, operation).to(kFloat32).cpu();
            auto aq = qx.cpu(), as = xs.cpu();
            for (int m = 0; m < batch; ++m) {
                for (int row = 0; row < 2 * rows; ++row) {
                    double value = 0.;
                    for (int g = 0; g < groups; ++g) {
                        for (int k = 0; k < gs; ++k) {
                            unsigned code = 0;
                            for (int bit = 0; bit < bits; ++bit) {
                                const int offset = k * bits + bit;
                                code |= ((qh.data_ptr<uint8_t>()[(row * groups + g) * bytes + offset / 8]
                                    >> (offset % 8)) & 1) << bit;
                            }
                            const double weight = double(nh.data_ptr<float>()[row])
                                * sh.data_ptr<uint8_t>()[row * groups + g] * code
                                - double(bh.data_ptr<float>()[row]) * mh.data_ptr<uint8_t>()[row * groups + g];
                            value += weight * as.data_ptr<float>()[m * groups + g]
                                * aq.data_ptr<int8_t>()[m * kpad + g * gs + k];
                        }
                    }
                    const double expected = __half2float(__float2half_rn(float(value)));
                    const double got = actual.data_ptr<float>()[m * (2 * rows) + row];
                    require(std::isfinite(got) && std::abs(got - expected) <= .002 + .002 * std::abs(expected),
                        "small-M packed plain CPU projection oracle mismatch");
                }
            }
        }
        if (width == 257) {
            Graph graph;
            graph.prepare_memory();
            Tensor output = invoke(x, operation);
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
            output = Tensor{};
            graph.capture_begin();
            output = invoke(x, operation);
            graph.capture_end();
            graph.replay();
            graph.replay();
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
            exact(output, reference);
            if (bits == 8 && operation == 2) {
                x.copy_(x + .25);
                std::vector<Tensor> changed_serial;
                for (int m = 0; m < batch; ++m) changed_serial.push_back(invoke(x.narrow(0, m, 1), operation));
                auto changed_reference = cat(changed_serial, 0);
                graph.replay();
                exact(output, changed_reference);
            }
            ++graphs;
        }
    }
    {
        auto float_input = x.to(kFloat32) + .000123;
        auto reference = invoke(float_input.to(kFloat16), 2).to(kFloat32);
        auto reference_qx = qx.cpu(), reference_xs = xs.cpu();
        auto reference_xm = bits == 8 ? xm.cpu() : Tensor{};
        auto invoke_float = [&](const Tensor& input) {
            if (bits != 4 && bits != 6)
                return nint_small_m_f32_ws_cuda(q, s, sm, ns, nm, input, qx, xs, xm, bits);
            auto function = bits == 4 ? nint4_gs24_small_m_f32_ws_cuda : nint6_gs24_small_m_f32_ws_cuda;
            return function(q, s, sm, ns, nm, input, qx, xs, xm);
        };
        for (int m = 2; m <= batch; ++m) {
            exact(invoke_float(float_input.narrow(0, 0, m)), reference.narrow(0, 0, m));
            exact(qx.narrow(0, 0, m), reference_qx.narrow(0, 0, m));
            exact(xs.narrow(0, 0, m), reference_xs.narrow(0, 0, m));
            if (bits == 8) exact(xm.narrow(0, 0, m), reference_xm.narrow(0, 0, m));
            ++f32_cases;
        }
        bool rejected = false;
        try { (void)invoke_float(float_input.narrow(0, 0, 1)); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "F32 small-M accepted unsupported M1");
        if (width == 257) {
            Graph graph;
            graph.prepare_memory();
            Tensor output = invoke_float(float_input);
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
            output = Tensor{};
            graph.capture_begin();
            output = invoke_float(float_input);
            graph.capture_end();
            graph.replay();
            graph.replay();
            exact(output, reference);
            float_input.copy_(float_input + .25);
            auto changed = invoke(float_input.to(kFloat16), 2).to(kFloat32);
            graph.replay();
            exact(output, changed);
            ++graphs;
        }
    }
    std::cout << "PASS NINT" << bits << " GS=" << gs << " N=" << rows * 2 << " K=" << width << '\n';
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        auto stream = mfq::cuda::stream_from_pool(false, 0);
        mfq::cuda::StreamGuard guard(stream);
        int cases = 0, graphs = 0, f32_cases = 0;
        for (const auto profile : std::vector<std::vector<int>>{{2,16,5},{3,24,5},{4,24,6},{5,28,7},{6,24,7},{8,48,7}})
            for (int width : {47, 257, 4096}) check(profile[0], profile[1], profile[2], width, cases, graphs, f32_cases);
        for (int width : {47, 257, 4096}) check(8, 48, 7, width, cases, graphs, f32_cases, 33);
        require(cases == 450 && graphs == 32 && f32_cases == 105, "incomplete small-M coverage");
        std::cout << "PASS small_m_cases=" << cases << " f32_boundary_cases=" << f32_cases
            << " graphs=" << graphs << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
