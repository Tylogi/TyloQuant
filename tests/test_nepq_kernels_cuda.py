from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from mfq.formats.nepq import (  # noqa: E402
    NEPQ0_L,
    NEPQ0_S,
    NEPQ1_L,
    NEPQ1_S,
    dequantize_nepq,
    rotation_signs,
)
from mfq.kernels.cuda.nepq_matmul import (  # noqa: E402
    nepq_dequantize,
    nepq_gemm_f16,
    nepq_gemv,
    nepq_grouped_matmul,
    nepq_mmq,
    to_gpu_nepq,
)
from mfq.kernels.cuda.moe import MoeRoutePlan  # noqa: E402
from tests.test_formats.test_nepq import _tensor  # noqa: E402


def _dynamic_tensor(spec):
    tensor = _tensor(spec)
    for supergroup in range(tensor.bank_ids.shape[-1]):
        tensor.bank_ids[:, :, supergroup] = (
            np.arange(tensor.n_experts * tensor.out_per_expert).reshape(
                tensor.n_experts, tensor.out_per_expert
            )
            + supergroup
        ) % tensor.bank_count
    return tensor


def _fwht_input(value: np.ndarray, block: int, signs: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32) * signs
    blocks = result.reshape(-1, block).copy()
    stride = 1
    while stride < block:
        paired = blocks.reshape(-1, 2, stride)
        first = paired[:, 0].copy()
        second = paired[:, 1].copy()
        paired[:, 0] = first + second
        paired[:, 1] = first - second
        stride *= 2
    blocks *= np.float32(1.0 / math.sqrt(block))
    return blocks.reshape(value.shape).astype(np.float16)


def test_nepq0_s_runtime_uses_dual_table_layouts():
    gpu = to_gpu_nepq(_dynamic_tensor(NEPQ0_S))
    assert tuple(gpu["table_pool"].shape) == (2, 320)
    assert tuple(gpu["grouped_table_pool"].shape) == (2, 2112)
    assert gpu["table_pool"].data_ptr() != gpu["grouped_table_pool"].data_ptr()


def test_nepq0_s_cuda_reads_bank_255():
    tensor = _dynamic_tensor(NEPQ0_S)
    tables = np.repeat(tensor.table_payloads[:1], 256, axis=0)
    tables[255] = tensor.table_payloads[1]
    tensor.table_payloads = tables
    tensor.bank_ids.fill(0)
    tensor.bank_ids[:, :, -1] = 255
    gpu = to_gpu_nepq(tensor)
    expected = torch.as_tensor(
        dequantize_nepq(tensor), device="cuda", dtype=torch.float16
    )
    torch.testing.assert_close(nepq_dequantize(gpu), expected, atol=2e-4, rtol=0)


@pytest.mark.parametrize("spec", [NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L])
def test_nepq_cuda_dequant_uses_per_supergroup_banks(spec):
    tensor = _dynamic_tensor(spec)
    gpu = to_gpu_nepq(tensor)
    expected = torch.as_tensor(
        dequantize_nepq(tensor), device="cuda", dtype=torch.float16
    )
    actual = nepq_dequantize(gpu)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)


@pytest.mark.parametrize("spec", [NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L])
@pytest.mark.parametrize("m", [1, 8, 16])
def test_nepq_cuda_rotated_gemv_matches_stored_weight(spec, m):
    tensor = _dynamic_tensor(spec)
    gpu = to_gpu_nepq(tensor)
    weight = torch.as_tensor(
        dequantize_nepq(tensor).reshape(-1, tensor.neuron_len),
        device="cuda",
        dtype=torch.float16,
    )
    rng = np.random.default_rng(20260724 + spec.profile_id * 100 + m)
    source = rng.normal(0.0, 0.1, size=(m, tensor.neuron_len)).astype(np.float16)
    signs = rotation_signs(
        tensor.neuron_len, tensor.rotation_block, tensor.rotation_seed
    )
    rotated = _fwht_input(source, tensor.rotation_block, signs)
    x = torch.as_tensor(source, device="cuda", dtype=torch.float16)
    expected = torch.as_tensor(rotated, device="cuda") @ weight.T
    actual = nepq_gemv(gpu, x)
    relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative < 0.012, f"{spec.label} M={m} relative error {relative}"


@pytest.mark.parametrize("spec", [NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L])
@pytest.mark.parametrize("m", [4, 32, 64])
def test_nepq_cuda_mmq_matches_stored_weight(spec, m):
    tensor = _dynamic_tensor(spec)
    tensor.rotation_block = 0
    tensor.rotation_seed = 0
    gpu = to_gpu_nepq(tensor)
    weight = torch.as_tensor(
        dequantize_nepq(tensor).reshape(-1, tensor.neuron_len),
        device="cuda",
        dtype=torch.float16,
    )
    torch.manual_seed(20260725 + spec.profile_id * 100 + m)
    x = (torch.randn(m, tensor.neuron_len, device="cuda") * 0.1).to(torch.float16)
    expected = x @ weight.T
    actual = nepq_mmq(gpu, x)
    relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative < 0.012, f"{spec.label} M={m} relative error {relative}"


@pytest.mark.parametrize("spec", [NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L])
@pytest.mark.parametrize("m", [16, 64, 256])
def test_nepq_cuda_online_f16_matches_stored_weight(spec, m):
    tensor = _dynamic_tensor(spec)
    tensor.rotation_block = 0
    tensor.rotation_seed = 0
    gpu = to_gpu_nepq(tensor)
    weight = torch.as_tensor(
        dequantize_nepq(tensor).reshape(-1, tensor.neuron_len),
        device="cuda",
        dtype=torch.float16,
    )
    torch.manual_seed(20260726 + spec.profile_id * 100 + m)
    x = (torch.randn(m, tensor.neuron_len, device="cuda") * 0.1).to(torch.float16)
    expected = x @ weight.T
    actual = nepq_gemm_f16(gpu, x)
    relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative < 0.006, f"{spec.label} M={m} relative error {relative}"


@pytest.mark.parametrize("spec", [NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L])
@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("routed_input", [False, True])
def test_nepq_cuda_grouped_route_matches_selected_experts(
    spec, tokens, routed_input
):
    tensor = _dynamic_tensor(spec)
    gpu = to_gpu_nepq(tensor)
    routes = 2
    ids = torch.as_tensor(
        (np.arange(tokens * routes).reshape(tokens, routes) + spec.profile_id) %
        tensor.n_experts,
        device="cuda",
        dtype=torch.int32,
    )
    route = MoeRoutePlan.build(ids, tensor.n_experts)
    rng = np.random.default_rng(
        20260727 + spec.profile_id * 100 + tokens + 1000 * routed_input
    )
    input_shape = (
        (tokens, routes, tensor.neuron_len)
        if routed_input
        else (tokens, tensor.neuron_len)
    )
    source = rng.normal(0.0, 0.1, size=input_shape).astype(np.float16)
    signs = rotation_signs(
        tensor.neuron_len, tensor.rotation_block, tensor.rotation_seed
    )
    rotated = _fwht_input(source, tensor.rotation_block, signs)
    weight = torch.as_tensor(
        dequantize_nepq(tensor), device="cuda", dtype=torch.float16
    )
    rotated_gpu = torch.as_tensor(rotated, device="cuda", dtype=torch.float16)
    expected = torch.empty(
        (tokens, routes, tensor.out_per_expert),
        device="cuda",
        dtype=torch.float16,
    )
    independent = torch.empty_like(expected)
    for token in range(tokens):
        for route_index in range(routes):
            expert = int(ids[token, route_index])
            source_row = (
                rotated_gpu[token, route_index]
                if routed_input
                else rotated_gpu[token]
            )
            expected[token, route_index] = source_row @ weight[expert].T
            raw_row = (
                source[token, route_index]
                if routed_input
                else source[token]
            )
            all_experts = nepq_gemv(
                gpu,
                torch.as_tensor(raw_row[None], device="cuda", dtype=torch.float16),
            ).reshape(tensor.n_experts, tensor.out_per_expert)
            independent[token, route_index] = all_experts[expert]
    actual = nepq_grouped_matmul(
        gpu,
        torch.as_tensor(source, device="cuda", dtype=torch.float16),
        route,
    )
    route_relative = (
        (actual.float() - independent.float()).norm() / independent.float().norm()
    ).item()
    quant_relative = (
        (actual.float() - expected.float()).norm() / expected.float().norm()
    ).item()
    assert route_relative < 0.002, (
        f"{spec.label} T={tokens} routed={routed_input} route error {route_relative}"
    )
    assert quant_relative < 0.02, (
        f"{spec.label} T={tokens} routed={routed_input} quant error {quant_relative}"
    )
