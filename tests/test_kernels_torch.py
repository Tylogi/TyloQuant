"""GPU tests for kernels.torch_backend and runtime.torch_linear, skipped without CUDA."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA 不可用", allow_module_level=True)

from mfq.formats.nint import RUNTIME_PROFILE_CATALOG, NintSpec  # noqa: E402
from mfq.kernels import torch_backend  # noqa: E402
from mfq.kernels.cuda.activation import silu_mul  # noqa: E402
from mfq.kernels.cuda.moe import MoeRoutePlan  # noqa: E402
from mfq.quantize import nint_quant  # noqa: E402
from mfq.runtime.torch_linear import (  # noqa: E402
    TorchNintEmbedding,
    TorchNintLinear,
    TorchNintLinearGroup,
    TorchSwiGLUFFN,
)

DEV = "cuda"


def _W(seed: int, shape: tuple[int, ...], scale: float = 0.05) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, scale, size=shape).astype(np.float32)


@pytest.mark.parametrize("bits,gs", RUNTIME_PROFILE_CATALOG)
def test_dequant_matches_numpy_all_profiles(bits, gs):
    W = _W(hash((bits, gs)) & 0xFFFF, (32, 96))
    t = nint_quant.quantize(W, NintSpec(bits, gs, 6), axis=0)
    g = torch_backend.to_gpu(t, DEV)
    dq = torch_backend.dequantize(g).cpu().numpy()
    np.testing.assert_allclose(dq, nint_quant.dequantize(t), atol=3e-3)


def test_matmul_matches_dequant_path():
    # Decomposed matmul should match (dequant fp16)@x, validating the arithmetic of llama.cpp-style decomposition
    W = _W(1, (32, 96))
    x = _W(2, (4, 96)) * 3
    t = nint_quant.quantize(W, NintSpec(4, 24, 6))
    g = torch_backend.to_gpu(t, DEV)
    xt = torch.as_tensor(x, device=DEV, dtype=torch.float16)
    y1 = torch_backend.matmul(g, xt)
    y2 = xt @ torch_backend.dequantize(g).T
    np.testing.assert_allclose(y1.cpu().numpy(), y2.cpu().numpy(), atol=1e-2)


def test_to_gpu_default_is_deploy_layout():
    W = _W(11, (32, 96))
    t = nint_quant.quantize(W, NintSpec(4, 24, 6))
    g = torch_backend.to_gpu(t, DEV)
    assert set(g) >= {
        "q_packed",
        "sub_scale",
        "sub_min",
        "neuron_scale",
        "neuron_min",
        "out",
        "ng",
        "gs",
        "neuron_len",
    }
    for key in (
        "q",
        "eff_pair_h",
        "d_eff",
        "m_eff",
        "d_eff_h",
        "m_eff_h",
        "q_mmq_packed",
        "q_prefill_u8_mmq",
    ):
        assert key not in g


def test_linear_forward_shape_and_close_to_numpy():
    W = _W(3, (32, 96))
    x = _W(4, (4, 96)) * 3
    lin = TorchNintLinear(nint_quant.quantize(W, NintSpec(4, 24, 6)), DEV)
    y = lin(torch.as_tensor(x, device=DEV)).cpu().numpy()
    ref = (x @ nint_quant.dequantize(nint_quant.quantize(W, NintSpec(4, 24, 6))).T).astype(np.float32)
    assert y.shape == (4, 32)
    np.testing.assert_allclose(y, ref, rtol=0.1, atol=0.05)


@pytest.mark.parametrize(
    "spec",
    [
        NintSpec(3, 24, 5),
        NintSpec(5, 24, 6),
        NintSpec(6, 22, 6),
        NintSpec(8, 32, 6),
    ],
)
def test_linear_forward_non4_bits_matches_dequant(spec):
    W = _W(31 + spec.bits, (24, 88))
    x = _W(41 + spec.bits, (3, 88)) * 3
    nt = nint_quant.quantize(W, spec)
    lin = TorchNintLinear(nt, DEV)
    xt = torch.as_tensor(x, device=DEV, dtype=torch.float16)
    y = lin(xt)
    ref = xt @ torch_backend.dequantize(torch_backend.to_gpu(nt, DEV)).T
    torch.testing.assert_close(y, ref, atol=2e-3, rtol=3e-3)


def test_linear_forward_m9_uses_prefill_path():
    W = _W(13, (32, 96))
    x = _W(14, (9, 96)) * 3
    lin = TorchNintLinear(nint_quant.quantize(W, NintSpec(4, 24, 6)), DEV)
    y = lin(torch.as_tensor(x, device=DEV)).cpu().numpy()
    ref = (x @ nint_quant.dequantize(nint_quant.quantize(W, NintSpec(4, 24, 6))).T).astype(np.float32)
    assert y.shape == (9, 32)
    np.testing.assert_allclose(y, ref, rtol=0.1, atol=0.05)



def test_linear_group_matches_separate_linears():
    din = 96
    specs = [NintSpec(4, 24, 6)] * 3
    tensors = [
        nint_quant.quantize(_W(21, (32, din)), specs[0]),
        nint_quant.quantize(_W(22, (16, din)), specs[1]),
        nint_quant.quantize(_W(23, (8, din)), specs[2]),
    ]
    group = TorchNintLinearGroup(tensors, DEV)
    linears = [TorchNintLinear(t, DEV) for t in tensors]
    x = torch.as_tensor(_W(24, (2, din)) * 3, device=DEV, dtype=torch.float16)
    y_group = torch.cat(group(x), dim=-1)
    y_ref = torch.cat([lin(x) for lin in linears], dim=-1)
    torch.testing.assert_close(y_group, y_ref, atol=0, rtol=0)


def test_silu_ffn_forward():
    din, dinter = 64, 96
    ffn = TorchSwiGLUFFN(
        TorchNintLinear(nint_quant.quantize(_W(5, (dinter, din)), NintSpec(4, 24, 6)), DEV),
        TorchNintLinear(nint_quant.quantize(_W(6, (dinter, din)), NintSpec(4, 24, 6)), DEV),
        TorchNintLinear(nint_quant.quantize(_W(7, (din, dinter)), NintSpec(4, 24, 6)), DEV),
    )
    x = torch.as_tensor(_W(8, (2, din)) * 3, device=DEV, dtype=torch.float16)
    y = ffn(x)
    assert tuple(y.shape) == (2, din)
    assert torch.isfinite(y).all()


def test_silu_ffn_fused_down_matches_materialized_decode():
    din, dinter = 64, 96
    gate_t = nint_quant.quantize(_W(25, (dinter, din)), NintSpec(4, 24, 6))
    up_t = nint_quant.quantize(_W(26, (dinter, din)), NintSpec(4, 24, 6))
    down_t = nint_quant.quantize(_W(27, (din, dinter)), NintSpec(4, 24, 6))
    ffn = TorchSwiGLUFFN.from_tensors(gate_t, up_t, down_t, DEV)
    gate = TorchNintLinear(gate_t, DEV)
    up = TorchNintLinear(up_t, DEV)
    down = TorchNintLinear(down_t, DEV)
    x = torch.as_tensor(_W(28, (1, din)) * 3, device=DEV, dtype=torch.float16)
    y_fused = ffn(x)
    y_ref = down(silu_mul(gate(x), up(x)))
    torch.testing.assert_close(y_fused, y_ref, atol=2e-3, rtol=3e-3)


def test_nint_embedding_forward():
    W = _W(9, (32, 48))
    t = nint_quant.quantize(W, NintSpec(4, 24, 6))
    emb = TorchNintEmbedding(t, DEV)
    ids = torch.tensor([[0, 3, 31], [2, 8, 5]], device=DEV, dtype=torch.int64)
    y = emb(ids)
    ref = torch.as_tensor(nint_quant.dequantize(t), device=DEV, dtype=torch.float16)[ids]
    torch.testing.assert_close(y, ref, atol=1.3e-4, rtol=0)
