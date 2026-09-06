"""Cross-platform production-extension gate for canonical small-M NINT kernels."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from mfq.kernels.cuda._ext import ext


PROFILES = [(2, 16, 5), (3, 24, 5), (4, 24, 6),
            (5, 28, 7), (6, 24, 7), (8, 48, 7)]


def _exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    integer = torch.int16 if actual.dtype == torch.float16 else torch.int32
    assert torch.equal(actual.contiguous().view(integer),
                       expected.contiguous().view(integer))


def _fixture(bits, gs, scale_bits, width):
    groups = (width + gs - 1) // gs
    r = torch.arange(34)[:, None]
    g = torch.arange(groups)[None, :]
    codes = (r[:, :, None] * 17 + g[:, :, None] * 7
             + torch.arange(gs)[None, None, :] * 13) % (1 << bits)
    packed = torch.zeros((34, groups, (gs * bits + 7) // 8), dtype=torch.uint8)
    for k in range(gs):
        byte, shift = divmod(k * bits, 8)
        packed[:, :, byte] |= ((codes[:, :, k] << shift) & 255).byte()
        if shift + bits > 8:
            packed[:, :, byte + 1] |= (codes[:, :, k] >> (8 - shift)).byte()
    scales = ((r * 7 + g * 11) % (1 << scale_bits)).byte()
    minima = ((r * 13 + g * 3) % 9).byte()
    ns = (1 + torch.arange(34) % 3).float() * (2. ** (-bits - 15))
    nm = (1 + torch.arange(34) % 5).float() / 4096.
    weights = (ns.double()[:, None, None] * scales[:, :, None] * codes
               - nm.double()[:, None, None] * minima[:, :, None]).reshape(34, -1)
    m = torch.arange(6)[:, None]
    k = torch.arange(width)[None, :]
    x = (((m * 29 + k * 17) % 255 - 127).float() / 64.).half().cuda()
    tensors = [value.cuda() for value in (packed, scales, minima, ns, nm)]
    qx = torch.empty((6, groups * gs), dtype=torch.int8, device="cuda")
    xs = torch.empty((6, groups), dtype=torch.float32, device="cuda")
    xm = torch.empty((6, groups), dtype=torch.int32, device="cuda")
    return tensors, x, qx, xs, xm, weights


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("width", [47, 257, 4096])
@torch.inference_mode()
def test_nint_small_m_bridge(profile, width, monkeypatch):
    bits, gs, scale_bits = profile
    module = ext()
    monkeypatch.setenv("MFQ_NINT4_SMALL_M_XSUM", "0")
    tensors, x, qx, xs, xm, weights = _fixture(bits, gs, scale_bits, width)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def invoke(value, operation):
        if operation == 2:
            if bits in (2, 3):
                return module.nint_gemv_packed_bits_ws_cuda(
                    *tensors, value, gs, bits, qx, xs, xm)
            function = (module.nint_gemv_packed_ws_cuda if bits == 4
                        else module.nint_gemv_packed_int6_ws_cuda)
            return function(*tensors, value, gs, qx, xs, xm)
        if operation >= 3:
            if bits == 4:
                return module.nint_gemv_packed_gate_ws_cuda(
                    *tensors, value, value, gs, operation - 2, qx, xs, xm)
            return module.nint_gemv_packed_bits_gate_ws_cuda(
                *tensors, value, value, gs, bits, operation - 2, qx, xs, xm)
        name = "swiglu" if operation == 0 else "geglu"
        if bits == 4:
            return getattr(module, f"nint_gemv_packed_{name}_ws_cuda")(
                *tensors, value, gs, qx, xs, xm)
        return getattr(module, f"nint_gemv_packed_bits_{name}_ws_cuda")(
            *tensors, value, gs, bits, qx, xs, xm)

    with torch.cuda.stream(stream):
        for operation in range(5 if bits in (4, 6) else (3 if bits in (2, 3) else 2)):
            reference = torch.cat([invoke(x[m:m + 1], operation) for m in range(6)])
            for m in range(1, 7):
                _exact(invoke(x[:m], operation), reference[:m])
            if operation <= 1:
                actual = invoke(x, operation).double().cpu()
                grouped = qx.double().reshape(6, -1, gs) * xs.double()[:, :, None]
                projected = (grouped.flatten(1).cpu() @ weights.T).float().half().double()
                gate, up = projected[:, :17], projected[:, 17:]
                if operation == 0:
                    value = up * gate / (1. + torch.exp(-gate))
                else:
                    value = up * .5 * gate * (1. + torch.tanh(
                        .7978845608028654 * gate * (1. + .044715 * gate * gate)))
                expected = value.float().half().double()
                torch.testing.assert_close(actual, expected, atol=.002, rtol=.002)
            if operation == 2 and bits in (2, 3):
                actual = invoke(x, operation).double().cpu()
                grouped = qx.double().reshape(6, -1, gs) * xs.double()[:, :, None]
                expected = (grouped.flatten(1).cpu() @ weights.T).float().half().double()
                torch.testing.assert_close(actual, expected, atol=.002, rtol=.002)
            if width == 257:
                for _ in range(3):
                    invoke(x, operation)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = invoke(x, operation)
                graph.replay()
                graph.replay()
                _exact(output, reference)

        if bits == 4:
            float_input = x.float() + .000123
            reference = invoke(float_input.half(), 2).float()

            def invoke_float(value):
                return module.nint4_gs24_small_m_f32_ws_cuda(
                    *tensors, value, qx, xs, xm)

            for sum_flag in ("0", "1"):
                monkeypatch.setenv("MFQ_NINT4_SMALL_M_XSUM", sum_flag)
                for m in range(2, 7):
                    _exact(invoke_float(float_input[:m]), reference[:m])
            monkeypatch.setenv("MFQ_NINT4_SMALL_M_XSUM", "0")
            with pytest.raises(RuntimeError):
                invoke_float(float_input[:1])
            with pytest.raises(RuntimeError):
                module.nint4_gs24_small_m_f32_ws_cuda(
                    *tensors, float_input, qx, xs, xm.float())
            if width == 257:
                for _ in range(3):
                    invoke_float(float_input)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = invoke_float(float_input)
                graph.replay()
                _exact(output, reference)
                float_input.add_(.25)
                expected = invoke(float_input.half(), 2).float()
                graph.replay()
                _exact(output, expected)
    stream.synchronize()
