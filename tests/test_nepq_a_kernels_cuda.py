from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from mfq.formats.nepq import (  # noqa: E402
    NEPQ0_A,
    NEPQ1_A,
    dequantize_nepq,
    rotation_signs,
)
from mfq.kernels.cuda.moe import MoeRoutePlan  # noqa: E402
from mfq.kernels.cuda.nepq_matmul import (  # noqa: E402
    nepq_dequantize,
    nepq_gemv,
    nepq_grouped_matmul,
    nepq_backward_input,
    nepq_matmul,
    to_gpu_nepq,
)
from tests.test_formats.test_nepq_a import _a_tensor  # noqa: E402
from tests.test_nepq_kernels_cuda import _fwht_input  # noqa: E402


@pytest.mark.parametrize("spec", [NEPQ0_A, NEPQ1_A])
def test_nepq_a_cuda_dequant_and_gemv(spec):
    tensor, _ = _a_tensor(spec)
    gpu = to_gpu_nepq(tensor)
    assert gpu["residual_first"].dtype == torch.int16
    assert gpu["residual_second"].dtype == torch.int16
    weight = torch.as_tensor(
        dequantize_nepq(tensor).reshape(-1, tensor.neuron_len),
        device="cuda",
        dtype=torch.float16,
    )
    torch.testing.assert_close(nepq_dequantize(gpu).reshape_as(weight), weight, atol=2.1e-3, rtol=0)
    rng = np.random.default_rng(20260809 + spec.profile_id)
    source = rng.normal(0.0, 0.1, size=(3, tensor.neuron_len)).astype(np.float16)
    signs = rotation_signs(tensor.neuron_len, tensor.rotation_block, tensor.rotation_seed)
    rotated = torch.as_tensor(_fwht_input(source, tensor.rotation_block, signs), device="cuda")
    expected = rotated @ weight.T
    actual = nepq_gemv(gpu, torch.as_tensor(source, device="cuda", dtype=torch.float16))
    relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative < 0.013


@pytest.mark.parametrize("spec", [NEPQ0_A, NEPQ1_A])
@pytest.mark.parametrize("m", [3, 8, 16])
def test_nepq_a_packed_backward_and_autograd_match_dequant(spec, m):
    tensor, _ = _a_tensor(spec)
    gpu = to_gpu_nepq(tensor)
    dense = torch.as_tensor(
        dequantize_nepq(tensor).reshape(-1, tensor.neuron_len),
        device="cuda",
        dtype=torch.float16,
    )
    output_gradient = torch.randn(
        m, dense.shape[0], device="cuda", dtype=torch.float16
    )
    stored_gradient = output_gradient @ dense
    signs = torch.as_tensor(
        rotation_signs(
            tensor.neuron_len, tensor.rotation_block, tensor.rotation_seed
        ),
        device="cuda",
        dtype=torch.float16,
    )
    blocks = stored_gradient.reshape(-1, tensor.rotation_block).clone()
    stride = 1
    while stride < tensor.rotation_block:
        paired = blocks.reshape(-1, 2, stride)
        first = paired[:, 0].clone()
        second = paired[:, 1].clone()
        paired[:, 0] = first + second
        paired[:, 1] = first - second
        stride *= 2
    expected = (
        blocks.reshape_as(stored_gradient)
        * np.float32(1.0 / np.sqrt(tensor.rotation_block))
        * signs
    ).to(torch.float16)
    torch.testing.assert_close(
        nepq_backward_input(gpu, output_gradient),
        expected,
        rtol=0.008,
        atol=0.008,
    )
    source = torch.randn(
        m,
        tensor.neuron_len,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    (nepq_matmul(gpu, source).reshape(m, -1) * output_gradient).sum().backward()
    torch.testing.assert_close(source.grad, expected, rtol=0.008, atol=0.008)


@pytest.mark.parametrize("spec", [NEPQ0_A, NEPQ1_A])
def test_nepq_a_cuda_grouped_route(spec):
    tensor, _ = _a_tensor(spec)
    gpu = to_gpu_nepq(tensor)
    ids = torch.tensor([[0, 1], [1, 0]], device="cuda", dtype=torch.int32)
    route = MoeRoutePlan.build(ids, tensor.n_experts)
    rng = np.random.default_rng(20260819 + spec.profile_id)
    source = rng.normal(0.0, 0.1, size=(2, tensor.neuron_len)).astype(np.float16)
    signs = rotation_signs(tensor.neuron_len, tensor.rotation_block, tensor.rotation_seed)
    rotated = _fwht_input(source, tensor.rotation_block, signs).astype(np.float32)
    weight = dequantize_nepq(tensor)
    expected = np.empty((2, 2, tensor.out_per_expert), dtype=np.float32)
    for token in range(2):
        for route_index in range(2):
            expected[token, route_index] = rotated[token] @ weight[int(ids[token, route_index])].T
    actual = nepq_grouped_matmul(
        gpu,
        torch.as_tensor(source, device="cuda", dtype=torch.float16),
        route,
    )
    torch.testing.assert_close(
        actual.float(),
        torch.as_tensor(expected, device="cuda"),
        atol=0.03,
        rtol=0.015,
    )
