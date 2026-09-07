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


def _fixture(bits, gs, scale_bits, width, rows=34):
    groups = (width + gs - 1) // gs
    r = torch.arange(rows)[:, None]
    g = torch.arange(groups)[None, :]
    codes = (r[:, :, None] * 17 + g[:, :, None] * 7
             + torch.arange(gs)[None, None, :] * 13) % (1 << bits)
    packed = torch.zeros((rows, groups, (gs * bits + 7) // 8), dtype=torch.uint8)
    for k in range(gs):
        byte, shift = divmod(k * bits, 8)
        packed[:, :, byte] |= ((codes[:, :, k] << shift) & 255).byte()
        if shift + bits > 8:
            packed[:, :, byte + 1] |= (codes[:, :, k] >> (8 - shift)).byte()
    scales = ((r * 7 + g * 11) % (1 << scale_bits)).byte()
    minima = ((r * 13 + g * 3) % 9).byte()
    ns = (1 + torch.arange(rows) % 3).float() * (2. ** (-bits - 15))
    nm = (1 + torch.arange(rows) % 5).float() / 4096.
    weights = (ns.double()[:, None, None] * scales[:, :, None] * codes
               - nm.double()[:, None, None] * minima[:, :, None]).reshape(rows, -1)
    m = torch.arange(6)[:, None]
    k = torch.arange(width)[None, :]
    x = (((m * 29 + k * 17) % 255 - 127).float() / 64.).half().cuda()
    tensors = [value.cuda() for value in (packed, scales, minima, ns, nm)]
    qx = torch.empty((6, groups * gs), dtype=torch.int8, device="cuda")
    xs = torch.empty((6, groups), dtype=torch.float32, device="cuda")
    xm = torch.empty((6, groups), dtype=torch.int32, device="cuda")
    return tensors, x, qx, xs, xm, weights


@pytest.mark.parametrize("profile,rows", [(profile, 34) for profile in PROFILES] + [((8, 48, 7), 66)])
@pytest.mark.parametrize("width", [47, 257, 4096])
@torch.inference_mode()
def test_nint_small_m_bridge(profile, rows, width, monkeypatch):
    bits, gs, scale_bits = profile
    module = ext()
    monkeypatch.setenv("MFQ_NINT4_SMALL_M_XSUM", "0")
    tensors, x, qx, xs, xm, weights = _fixture(bits, gs, scale_bits, width, rows)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def invoke(value, operation):
        if operation == 2:
            if bits == 8:
                return module.nint_gemv_packed_u8_ws_cuda(*tensors, value, gs, qx, xs, xm)
            if bits in (2, 3, 5):
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
        for operation in range(5 if bits in (4, 6) else 3):
            reference = torch.cat([invoke(x[m:m + 1], operation) for m in range(6)])
            for m in range(1, 7):
                _exact(invoke(x[:m], operation), reference[:m])
            if operation <= 1:
                actual = invoke(x, operation).double().cpu()
                grouped = qx.double().reshape(6, -1, gs) * xs.double()[:, :, None]
                projected = (grouped.flatten(1).cpu() @ weights.T).float().half().double()
                gate, up = projected[:, :rows // 2], projected[:, rows // 2:]
                if operation == 0:
                    value = up * gate / (1. + torch.exp(-gate))
                else:
                    value = up * .5 * gate * (1. + torch.tanh(
                        .7978845608028654 * gate * (1. + .044715 * gate * gate)))
                expected = value.float().half().double()
                torch.testing.assert_close(actual, expected, atol=.002, rtol=.002)
            if operation == 2 and bits != 4:
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
                if bits == 8 and operation == 2:
                    x.add_(.25)
                    changed_reference = torch.cat([invoke(x[m:m + 1], operation) for m in range(6)])
                    graph.replay()
                    _exact(output, changed_reference)

        if bits in (2, 3, 4, 5, 6, 8):
            float_input = x.float() + .000123
            reference = invoke(float_input.half(), 2).float()
            reference_qx, reference_xs = qx.clone(), xs.clone()
            reference_xm = xm.clone() if bits == 8 else None
            float_function = (getattr(module, f"nint{bits}_gs24_small_m_f32_ws_cuda")
                              if bits in (4, 6) else None)

            def invoke_float(value):
                if float_function is not None:
                    return float_function(*tensors, value, qx, xs, xm)
                return module.nint_small_m_f32_ws_cuda(*tensors, value, qx, xs, xm, bits)

            for sum_flag in (("0", "1") if bits == 4 else ("0",)):
                monkeypatch.setenv("MFQ_NINT4_SMALL_M_XSUM", sum_flag)
                for m in range(2, 7):
                    _exact(invoke_float(float_input[:m]), reference[:m])
                    _exact(qx[:m], reference_qx[:m])
                    _exact(xs[:m], reference_xs[:m])
                    if bits == 8:
                        _exact(xm[:m], reference_xm[:m])
            monkeypatch.setenv("MFQ_NINT4_SMALL_M_XSUM", "0")
            with pytest.raises(RuntimeError):
                invoke_float(float_input[:1])
            with pytest.raises(RuntimeError):
                if float_function is not None:
                    float_function(*tensors, float_input, qx, xs, xm.float())
                else:
                    module.nint_small_m_f32_ws_cuda(*tensors, float_input, qx, xs, xm.float(), bits)
            if float_function is None:
                with pytest.raises(RuntimeError):
                    module.nint_small_m_f32_ws_cuda(*tensors, float_input, qx, xs, xm, 7)
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


@pytest.mark.parametrize("width", [31440, 31441])
@torch.inference_mode()
def test_nint8_large_k_and_padding(width):
    """Preserve long-K and adjacent GS48 padding regressions after shared-W rejection."""
    groups = (width + 47) // 48
    assert groups * 50 in (32750, 32800)
    module = ext()
    tensors, x, qx, xs, xm, weights = _fixture(8, 48, 7, width, 66)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def invoke(value):
        return module.nint_gemv_packed_u8_ws_cuda(*tensors, value, 48, qx, xs, xm)

    with torch.cuda.stream(stream):
        reference = torch.cat([invoke(x[m:m + 1]) for m in range(6)])
        for m in range(2, 7):
            _exact(invoke(x[:m]), reference[:m])
        actual = invoke(x).double().cpu()
        grouped = qx.double().reshape(6, -1, 48) * xs.double()[:, :, None]
        expected = (grouped.flatten(1).cpu() @ weights.T).float().half().double()
        torch.testing.assert_close(actual, expected, atol=.002, rtol=.002)
        for _ in range(3):
            invoke(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = invoke(x)
        graph.replay()
        _exact(output, reference)
        x.add_(.25)
        changed = torch.cat([invoke(x[m:m + 1]) for m in range(6)])
        graph.replay()
        _exact(output, changed)
        float_input = x.float() + .000123
        float_reference = invoke(float_input.half()).float()
        reference_qx, reference_xs, reference_xm = qx.clone(), xs.clone(), xm.clone()
        for m in range(2, 7):
            actual = module.nint_small_m_f32_ws_cuda(*tensors, float_input[:m], qx, xs, xm, 8)
            _exact(actual, float_reference[:m])
            _exact(qx[:m], reference_qx[:m])
            _exact(xs[:m], reference_xs[:m])
            _exact(xm[:m], reference_xm[:m])
    stream.synchronize()
