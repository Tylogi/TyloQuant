"""Small cross-platform numerical gate for the production Torch activation bridge."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from mfq.kernels.cuda._ext import ext


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("kind", ["silu", "gelu"])
@pytest.mark.parametrize("shape", [(0,), (33, 71)])
def test_activation_bridge(dtype, kind, shape):
    torch.manual_seed(20260907)
    gate = torch.randn(shape, device="cuda", dtype=dtype)
    up = torch.randn_like(gate)
    module = ext()
    operation = getattr(module, f"{kind}_mul_cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = operation(gate, up)
    torch.cuda.current_stream().wait_stream(stream)
    g = gate.float()
    activation = (torch.nn.functional.gelu(g, approximate="tanh")
                  if kind == "gelu" else torch.nn.functional.silu(g))
    expected = (activation * up.float()).to(dtype)
    tolerance = 1e-3 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_gelu_half_saturation():
    gate = torch.tensor([200., 200., -200.], device="cuda", dtype=torch.float16)
    up = torch.tensor([400., -400., 400.], device="cuda", dtype=torch.float16)
    expected = torch.tensor([65504., -65504., 0.], device="cuda", dtype=torch.float16)
    actual = ext().gelu_mul_cuda(gate, up)
    torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
