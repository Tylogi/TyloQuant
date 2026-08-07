"""CUDA kernel 测试（GDN + NINT fused GEMM；需 nvcc + MSVC cl 在 PATH，否则跳过）。"""

from __future__ import annotations

import shutil
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA 不可用", allow_module_level=True)
if shutil.which("cl") is None and shutil.which("cl.exe") is None:
    pytest.skip("MSVC cl 不在 PATH（source vcvars64 或用 Developer Prompt 运行）", allow_module_level=True)

from mfq.kernels.cuda.gated_delta_net import gated_delta_net as gdn_cuda  # noqa: E402
from mfq.kernels.gated_delta_net import gated_delta_net as gdn_ref  # noqa: E402
from mfq.kernels.cuda.nint_matmul import (  # noqa: E402
    nint5_q5_exec_argmax,
    nint5_q5_exec_dequant,
    nint5_q5_exec_matmul,
    nint5_q5_exec_repack,
    nint_argmax,
    nint_matmul as fused_matmul,
    nint_matmul_input_mul,
    _workspace,
)
from mfq.kernels.cuda._ext import ext  # noqa: E402
from mfq.kernels.torch_backend import to_gpu, matmul as tb_matmul  # noqa: E402
from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.formats.nint8_one import quantize_nint8_one  # noqa: E402
from mfq.formats.nint8_zero import (  # noqa: E402
    dequantize_nint8_zero,
    quantize_nint8_zero,
)
from mfq.quantize.nint_quant import quantize as nint_quantize, dequantize as nint_dequant  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from mfq.kernels.cuda.norm import rms_norm, l2_norm  # noqa: E402
from mfq.kernels.cuda.acc import acc  # noqa: E402
from mfq.kernels.cuda.rope import rope  # noqa: E402
from mfq.kernels.cuda.attention import (  # noqa: E402
    attention,
    sliding_window_attention,
    sliding_window_attention_cached,
)
from mfq.kernels.cuda.kv_cache import (  # noqa: E402
    KVCache,
    SlidingWindowKVCache,
    kv_cache_write,
    kv_cache_write_ring,
)
from mfq.kernels.cuda.activation import gelu_mul, silu_mul  # noqa: E402
from mfq.kernels.cuda.embedding import embedding, nint_embedding  # noqa: E402
from mfq.kernels.cuda.sampling import sample, sample_greedy  # noqa: E402
from mfq.kernels.cuda.ssm_conv import ssm_conv_silu  # noqa: E402

DEV = "cuda"


def test_nint8_one_cuda_matches_cpu_q8_1_oracle_with_tail():
    values = np.zeros((2, 35), dtype=np.float32)
    values[0, :6] = [127.0, 63.5, -63.5, 0.5, -0.5, 1.25]
    values[1] = np.linspace(-2.0, 2.0, 35, dtype=np.float32)
    x = torch.from_numpy(values).to(device=DEV, dtype=torch.float16)
    oracle = quantize_nint8_one(
        x.cpu().numpy().astype(np.float32, copy=False)
    )

    q, d, s, reconstructed = ext().nint8_one_quantize_reconstruct_cuda(x)

    np.testing.assert_array_equal(q.cpu().numpy(), oracle.q)
    np.testing.assert_array_equal(d.cpu().numpy(), oracle.d)
    np.testing.assert_array_equal(s.cpu().numpy(), oracle.s)
    np.testing.assert_array_equal(
        reconstructed.cpu().numpy(), oracle.reconstructed
    )


@pytest.mark.parametrize(
    "spec",
    [
        NintSpec(2, 16, 5),
        NintSpec(3, 24, 5),
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 7),
        NintSpec(6, 24, 7),
        NintSpec(8, 48, 7),
    ],
)
def test_common_f16_packed_mmq_matches_dequantized_weight(spec):
    torch.manual_seed(901 + spec.bits)
    np.random.seed(901 + spec.bits)
    out, width, rows = 71, 280, 512
    tensor, gpu = _gpu_g(
        np.random.randn(out, width).astype(np.float32) * 0.05,
        spec,
    )
    x = (
        torch.randn(rows, width, device=DEV, dtype=torch.float16)
        * 0.1
    ).contiguous()

    actual = ext().nint_mmq_f16_packed_cuda(
        gpu["q_packed"],
        gpu["sub_scale"],
        gpu["sub_min"],
        gpu["neuron_scale"],
        gpu["neuron_min"],
        x,
        spec.groupsize,
        spec.bits,
    )
    expected = _ref_out(tensor, x)

    relative = ((actual - expected).norm() / expected.norm()).item()
    assert relative < 3e-3, (
        f"NINT{spec.bits} gs{spec.groupsize} fp16 relative={relative}"
    )


def test_common_f16_nint8_zero_mmq_matches_dequantized_weight():
    torch.manual_seed(919)
    np.random.seed(919)
    out, width, rows = 71, 288, 512
    tensor = quantize_nint8_zero(
        np.random.randn(out, width).astype(np.float32) * 0.05
    )
    q = torch.from_numpy(tensor.q.view(np.uint8)).to(DEV).contiguous()
    scale = torch.from_numpy(tensor.scale).to(DEV).contiguous()
    x = (
        torch.randn(rows, width, device=DEV, dtype=torch.float16)
        * 0.1
    ).contiguous()

    actual = ext().nint8_zero_mmq_f16_packed_cuda(q, scale, x, width)
    weight = torch.from_numpy(
        dequantize_nint8_zero(tensor)
    ).to(device=DEV, dtype=torch.float32)
    expected = (x.float() @ weight.T).half()

    relative = ((actual - expected).norm() / expected.norm()).item()
    assert relative < 3e-3, f"NINT8-0 fp16 relative={relative}"


def _dec_g(*shape):
    """真实衰减门控（g<0，exp(g)<1），避免状态爆炸。"""
    return -(torch.rand(*shape, device=DEV) * 3 + 0.5)


def _inputs(B, H, T, D, kda=False, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H, T, D, device=DEV) * 0.1
    k = torch.randn(B, H, T, D, device=DEV) * 0.1
    v = torch.randn(B, H, T, D, device=DEV) * 0.1
    g = _dec_g(B, H, T, D) if kda else _dec_g(B, H, T)
    beta = torch.sigmoid(torch.randn(B, H, T, device=DEV)) * 0.5
    return q, k, v, g, beta


def test_gdn_cuda_scalar_gate_matches_ref():
    q, k, v, g, beta = _inputs(2, 4, 32, 32)
    yc, sc = gdn_cuda(q, k, v, g, beta)
    yr, sr = gdn_ref(q, k, v, g, beta)
    assert ((yc - yr).norm() / yr.norm()).item() < 1e-5
    assert (sc - sr).abs().max().item() < 1e-5


def test_gdn_cuda_kda_matches_ref():
    q, k, v, g, beta = _inputs(1, 2, 16, 32, kda=True)
    yc, _ = gdn_cuda(q, k, v, g, beta)
    yr, _ = gdn_ref(q, k, v, g, beta)
    assert ((yc - yr).norm() / yr.norm()).item() < 1e-5


def test_gdn_cuda_decode_column_path_matches_ref():
    q, k, v, g, beta = _inputs(1, 8, 1, 64, seed=33)
    state = torch.randn(1, 8, 64, 64, device=DEV) * 0.01
    old = os.environ.get("MFQ_GDN_COLUMN")
    os.environ["MFQ_GDN_COLUMN"] = "1"
    try:
        yc, sc = gdn_cuda(q, k, v, g, beta, state=state)
        yr, sr = gdn_ref(q, k, v, g, beta, state=state)
        assert ((yc - yr).norm() / yr.norm()).item() < 1e-5
        assert (sc - sr).abs().max().item() < 1e-5
    finally:
        if old is None:
            os.environ.pop("MFQ_GDN_COLUMN", None)
        else:
            os.environ["MFQ_GDN_COLUMN"] = old


def test_gdn_cuda_d128_shared_mem():
    q, k, v, g, beta = _inputs(1, 2, 16, 128)
    yc, sc = gdn_cuda(q, k, v, g, beta)
    yr, sr = gdn_ref(q, k, v, g, beta)
    assert ((yc - yr).norm() / yr.norm()).item() < 1e-5
    assert torch.isfinite(yc).all()


def test_gdn_cuda_state_carryover():
    q, k, v, g, beta = _inputs(1, 2, 12, 32)
    _, s_full = gdn_cuda(q, k, v, g, beta)
    _, s1 = gdn_cuda(q[:, :, :6], k[:, :, :6], v[:, :, :6], g[:, :, :6], beta[:, :, :6])
    _, s2 = gdn_cuda(q[:, :, 6:], k[:, :, 6:], v[:, :, 6:], g[:, :, 6:], beta[:, :, 6:], state=s1)
    assert (s2 - s_full).abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
# NINT INT-fused-GEMM
# ---------------------------------------------------------------------------
def _gpu_g(W_np, spec):
    nt = nint_quantize(W_np, spec, axis=0)
    return nt, to_gpu(nt, layout="experimental")


def _ref_out(nt, x):
    """fp32 反量化权重参考：y = x · Wq^T。"""
    Wq = torch.as_tensor(np.ascontiguousarray(nint_dequant(nt)), device=DEV)  # [out, K]
    return (x.to(torch.float32) @ Wq.T).to(torch.float16)


def test_nint_fused_matches_reference_gs24():
    torch.manual_seed(0); np.random.seed(0)
    out, K, M = 128, 256, 16
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt, g = _gpu_g(W, NintSpec(4, 24, 6))
    x = torch.randn(M, K, device=DEV) * 0.1
    y_fused = fused_matmul(g, x)
    y_ref = _ref_out(nt, x)
    rel = ((y_fused - y_ref).norm() / y_ref.norm()).item()
    assert rel < 2e-2, f"fused vs fp32-ref rel={rel}"
    y_tb = tb_matmul(g, x)
    rel2 = ((y_fused - y_tb).norm() / y_tb.norm()).item()
    assert rel2 < 2e-2, f"fused vs torch_backend rel={rel2}"


@pytest.mark.parametrize("gs", [16, 32, 48])
def test_nint_fused_other_profiles(gs):
    torch.manual_seed(1); np.random.seed(1)
    out, K, M = 64, 200, 8
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt, g = _gpu_g(W, NintSpec(4, gs, 6))
    x = torch.randn(M, K, device=DEV) * 0.1
    y_fused = fused_matmul(g, x)
    y_ref = _ref_out(nt, x)
    rel = ((y_fused - y_ref).norm() / y_ref.norm()).item()
    assert rel < 3e-2, f"gs={gs} rel={rel}"


def test_nint_fused_tail_group():
    """K 不整除 gs（尾组补零）。"""
    torch.manual_seed(2); np.random.seed(2)
    out, K, M = 48, 250, 12          # 250 % 24 != 0
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt, g = _gpu_g(W, NintSpec(4, 24, 6))
    x = torch.randn(M, K, device=DEV) * 0.1
    y_fused = fused_matmul(g, x)
    y_ref = _ref_out(nt, x)
    rel = ((y_fused - y_ref).norm() / y_ref.norm()).item()
    assert rel < 2e-2, f"tail rel={rel}"


def test_nint_fused_x_pad():
    """x 末维 < neuron_len（胶水补零）。"""
    torch.manual_seed(3); np.random.seed(3)
    out, K, M = 32, 256, 4
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt, g = _gpu_g(W, NintSpec(4, 24, 6))
    Kshort = 200
    x = torch.randn(M, Kshort, device=DEV) * 0.1
    y_fused = fused_matmul(g, x)
    # 参考：把 x 也补零到 K
    x_full = torch.nn.functional.pad(x, (0, K - Kshort))
    y_ref = _ref_out(nt, x_full)
    rel = ((y_fused - y_ref).norm() / y_ref.norm()).item()
    assert rel < 2e-2, f"x-pad rel={rel}"


def test_nint_fused_workspace_reuse_changes_input():
    """同一个 GPU dict 连续调用不同 x，workspace 复用不能残留旧激活。"""
    torch.manual_seed(33); np.random.seed(33)
    out, K, M = 64, 256, 8
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt, g = _gpu_g(W, NintSpec(4, 24, 6))
    x1 = torch.randn(M, K, device=DEV) * 0.1
    x2 = torch.randn(M, K, device=DEV) * 0.1
    _ = fused_matmul(g, x1)
    y2 = fused_matmul(g, x2)
    y_ref = _ref_out(nt, x2)
    rel = ((y2 - y_ref).norm() / y_ref.norm()).item()
    assert rel < 2e-2, f"workspace reuse rel={rel}"


def test_nint_fused_small_batch_default_matches_ref():
    """M2-M6 默认走 batched GEMV，M7 进入 MMQ。"""
    torch.manual_seed(37); np.random.seed(37)
    out, K = 96, 280
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt, g = _gpu_g(W, NintSpec(4, 24, 6))
    for M in range(2, 8):
        x = torch.randn(M, K, device=DEV) * 0.1
        y_fused = fused_matmul(g, x)
        y_ref = _ref_out(nt, x)
        rel = ((y_fused - y_ref).norm() / y_ref.norm()).item()
        assert rel < 2e-2, f"M{M} small-batch rel={rel}"


def test_nint_prefill_default_matches_ref():
    """M>64 默认 prefill 路径应保持 fp16 误差内一致。"""
    torch.manual_seed(41); np.random.seed(41)
    out, K, M = 96, 280, 128
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt, g = _gpu_g(W, NintSpec(4, 24, 6))
    x = torch.randn(M, K, device=DEV) * 0.1
    y_fused = fused_matmul(g, x)
    y_ref = _ref_out(nt, x)
    rel = ((y_fused - y_ref).norm() / y_ref.norm()).item()
    assert rel < 2e-2, f"prefill rel={rel}"


def test_nint_dequant_wq_packed_matches_torch():
    """CUDA packed Wq materialization 应与 torch_backend 的 Wq 一致。"""
    torch.manual_seed(42); np.random.seed(42)
    out, K = 48, 250
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    _, g = _gpu_g(W, NintSpec(4, 24, 6))
    wq_cuda = ext().nint_dequant_wq_packed_cuda(g["q_packed"], g["d_eff"], int(g["neuron_len"]), int(g["gs"]))
    wq_ref = (
        g["d_eff"][:, :, None] * g["q"].to(torch.float32)
    ).reshape(g["out"], int(g["ng"]) * int(g["gs"]))[:, : int(g["neuron_len"])].to(torch.float16)
    torch.testing.assert_close(wq_cuda, wq_ref, atol=0, rtol=0)


def test_nint_dequant_full_packed_matches_torch():
    """CUDA packed full W materialization 应与 torch_backend 的 dequant 权重一致。"""
    torch.manual_seed(43); np.random.seed(43)
    out, K = 48, 250
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    _, g = _gpu_g(W, NintSpec(4, 24, 6))
    w_cuda = ext().nint_dequant_full_packed_cuda(
        g["q_packed"], g["d_eff"], g["m_eff"], int(g["neuron_len"]), int(g["gs"])
    )
    w_ref = (
        g["d_eff"][:, :, None] * g["q"].to(torch.float32)
        - g["m_eff"][:, :, None]
    ).reshape(g["out"], int(g["ng"]) * int(g["gs"]))[:, : int(g["neuron_len"])].to(torch.float16)
    torch.testing.assert_close(w_cuda, w_ref, atol=0, rtol=0)


def test_nint_dequant_full_packed_compact_matches_torch():
    """Compact deploy metadata full dequant 应与展开 metadata 一致。"""
    torch.manual_seed(59); np.random.seed(59)
    out, K = 48, 250
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    _, g = _gpu_g(W, NintSpec(4, 24, 6))
    w_cuda = ext().nint_dequant_full_packed_compact_cuda(
        g["q_packed"],
        g["sub_scale"],
        g["sub_min"],
        g["neuron_scale"],
        g["neuron_min"],
        int(g["neuron_len"]),
        int(g["gs"]),
    )
    w_ref = (
        g["d_eff"][:, :, None] * g["q"].to(torch.float32)
        - g["m_eff"][:, :, None]
    ).reshape(g["out"], int(g["ng"]) * int(g["gs"]))[:, : int(g["neuron_len"])].to(torch.float16)
    torch.testing.assert_close(w_cuda, w_ref, atol=0, rtol=0)


def test_nint_dequant_full_packed_h2_matches_torch():
    """CUDA half2 full W materialization 应与 half execution metadata 一致。"""
    torch.manual_seed(44); np.random.seed(44)
    out, K = 48, 250
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    _, g = _gpu_g(W, NintSpec(4, 24, 6))
    w_cuda = ext().nint_dequant_full_packed_h2_cuda(
        g["q_packed"], g["eff_pair_h"], int(g["neuron_len"]), int(g["gs"])
    )
    w_ref = (
        g["d_eff_h"][:, :, None] * g["q"].to(torch.float16)
        - g["m_eff_h"][:, :, None]
    ).reshape(g["out"], int(g["ng"]) * int(g["gs"]))[:, : int(g["neuron_len"])]
    torch.testing.assert_close(w_cuda, w_ref, atol=1.3e-4, rtol=0)


def test_nint_dequant_full_packed_gs24_x2_matches_default():
    """gs24 x2 full dequant 应与默认 f32 metadata kernel 完全一致。"""
    torch.manual_seed(55); np.random.seed(55)
    out, K = 48, 250
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    _, g = _gpu_g(W, NintSpec(4, 24, 6))
    w_ref = ext().nint_dequant_full_packed_cuda(
        g["q_packed"], g["d_eff"], g["m_eff"], int(g["neuron_len"]), int(g["gs"])
    )
    w_x2 = ext().nint_dequant_full_packed_gs24_x2_cuda(
        g["q_packed"], g["d_eff"], g["m_eff"], int(g["neuron_len"])
    )
    torch.testing.assert_close(w_x2, w_ref, atol=0, rtol=0)


def test_nint_dequant_full_packed_gs24_x2h2_matches_half2_reference():
    """gs24 x2 half2 metadata 候选应与 half2 full dequant 误差一致。"""
    torch.manual_seed(56); np.random.seed(56)
    out, K = 48, 250
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    _, g = _gpu_g(W, NintSpec(4, 24, 6))
    w_ref = ext().nint_dequant_full_packed_h2_cuda(
        g["q_packed"], g["eff_pair_h"], int(g["neuron_len"]), int(g["gs"])
    )
    w_x2h2 = ext().nint_dequant_full_packed_gs24_x2h2_cuda(
        g["q_packed"], g["eff_pair_h"], int(g["neuron_len"])
    )
    torch.testing.assert_close(w_x2h2, w_ref, atol=0, rtol=0)


def test_nint_batched_gemv_matches_gemv():
    """MMVQ-style batched GEMV 应与逐 row GEMV 完全一致。"""
    torch.manual_seed(38); np.random.seed(38)
    out, K, M = 96, 280, 6
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    _, g = _gpu_g(W, NintSpec(4, 24, 6))
    x = torch.randn(M, K, device=DEV) * 0.1
    qx, xscale, xsum = _workspace(g, x)
    args = (g["q_packed"], g["sub_scale"], g["sub_min"], g["neuron_scale"], g["neuron_min"])
    xh = x.contiguous().to(torch.float16)
    y_gemv = ext().nint_gemv_packed_ws_cuda(*args, xh, int(g["gs"]), qx, xscale, xsum)
    y_batch = ext().nint_gemv_packed_batch_ws_cuda(*args, xh, int(g["gs"]), qx, xscale, xsum)
    torch.testing.assert_close(y_batch, y_gemv, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize(
    "spec",
    [
        NintSpec(2, 16, 5),
        NintSpec(3, 24, 5),
        NintSpec(5, 24, 6),
        NintSpec(6, 22, 6),
        NintSpec(8, 32, 6),
    ],
)
@pytest.mark.parametrize("M", [1, 3, 9])
def test_nint_packed_bits_matmul_matches_dequant(spec, M):
    torch.manual_seed(70 + spec.bits + M); np.random.seed(70 + spec.bits + M)
    out, K = 48, 88
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt = nint_quantize(W, spec, axis=0)
    g = to_gpu(nt, layout="deploy")
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    y = fused_matmul(g, x)
    w_ref = torch.as_tensor(np.ascontiguousarray(nint_dequant(nt)), device=DEV, dtype=torch.float16)
    ref = x @ w_ref.T
    torch.testing.assert_close(y, ref, atol=2e-3, rtol=3e-3)


@pytest.mark.parametrize("M", [2, 4, 8])
def test_nint6_gs26_special_gemv_matches_generic(monkeypatch, M):
    torch.manual_seed(92 + M); np.random.seed(92 + M)
    out, K = 64, 104
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt = nint_quantize(W, NintSpec(6, 26, 7), axis=0)
    g = to_gpu(nt, layout="deploy")
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    y_special = fused_matmul(g, x)
    monkeypatch.setenv("MFQ_NINT_BITS_GEMV_GENERIC", "1")
    y_generic = fused_matmul(g, x)
    torch.testing.assert_close(y_special, y_generic, atol=2e-3, rtol=3e-3)


@pytest.mark.parametrize("M", [2, 5, 8])
def test_nint5_gs28_special_gemv_matches_generic(monkeypatch, M):
    torch.manual_seed(104 + M); np.random.seed(104 + M)
    out, K = 64, 112
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt = nint_quantize(W, NintSpec(5, 28, 7), axis=0)
    g = to_gpu(nt, layout="deploy")
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    y_special = fused_matmul(g, x)
    monkeypatch.setenv("MFQ_NINT_BITS_GEMV_GENERIC", "1")
    y_generic = fused_matmul(g, x)
    torch.testing.assert_close(y_special, y_generic, atol=2e-3, rtol=3e-3)


@pytest.mark.parametrize("seed", [201, 202, 203])
def test_nint5_gs28_argmax_matches_materialized_logits(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    out, K = 257, 224
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt = nint_quantize(W, NintSpec(5, 28, 7), axis=0)
    g = to_gpu(nt, layout="deploy")
    x = (torch.randn(1, K, device=DEV) * 0.1).to(torch.float16)
    expected = fused_matmul(g, x).argmax(-1).to(torch.int64)
    actual = nint_argmax(g, x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_nint5_q5_exec_dequant_is_bit_exact():
    torch.manual_seed(211); np.random.seed(211)
    out, K = 64, 224
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt = nint_quantize(W, NintSpec(5, 28, 7), axis=0)
    g = to_gpu(nt, layout="deploy")
    q5 = nint5_q5_exec_repack(g)
    expected = ext().nint_dequant_full_packed_compact_bits_cuda(
        g["q_packed"], g["sub_scale"], g["sub_min"],
        g["neuron_scale"], g["neuron_min"], int(g["neuron_len"]), 28, 5,
    )
    actual = nint5_q5_exec_dequant(q5)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("M", [1, 2, 5, 8])
def test_nint5_q5_exec_gemv_matches_original(M):
    torch.manual_seed(220 + M); np.random.seed(220 + M)
    out, K = 128, 224
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt = nint_quantize(W, NintSpec(5, 28, 7), axis=0)
    g = to_gpu(nt, layout="deploy")
    q5 = nint5_q5_exec_repack(g)
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    actual = nint5_q5_exec_matmul(q5, x)
    expected = fused_matmul(g, x)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-3)


@pytest.mark.parametrize("seed", [231, 232, 233])
def test_nint5_q5_exec_argmax_matches_original(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    out, K = 257, 224
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt = nint_quantize(W, NintSpec(5, 28, 7), axis=0)
    g = to_gpu(nt, layout="deploy")
    q5 = nint5_q5_exec_repack(g)
    x = (torch.randn(1, K, device=DEV) * 0.1).to(torch.float16)
    actual = nint5_q5_exec_argmax(q5, x)
    expected = nint_argmax(g, x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_nint5_q5_exec_handles_partial_group_batch():
    torch.manual_seed(241); np.random.seed(241)
    out, K = 64, 252  # nine groups exercises the final partial 8-group warp batch.
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt = nint_quantize(W, NintSpec(5, 28, 7), axis=0)
    g = to_gpu(nt, layout="deploy")
    q5 = nint5_q5_exec_repack(g)
    x = (torch.randn(1, K, device=DEV) * 0.1).to(torch.float16)
    actual = nint5_q5_exec_matmul(q5, x)
    expected = fused_matmul(g, x)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-3)


def test_nint8_mmq_env_matches_dequant(monkeypatch):
    torch.manual_seed(83); np.random.seed(83)
    out, K, M = 48, 88, 9
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt = nint_quantize(W, NintSpec(8, 32, 6), axis=0)
    g = to_gpu(nt, layout="deploy")
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    monkeypatch.setenv("MFQ_NINT8_MMQ", "1")
    y = fused_matmul(g, x)
    w_ref = torch.as_tensor(np.ascontiguousarray(nint_dequant(nt)), device=DEV, dtype=torch.float16)
    ref = x @ w_ref.T
    torch.testing.assert_close(y, ref, atol=2e-3, rtol=3e-3)


@pytest.mark.parametrize("M", [2, 3])
def test_nint_batched_gemv_eff_metadata_matches_gemv(M):
    """预融合 fp16 execution metadata 的 MMVQ 数值应保持在 fp16 误差内。"""
    torch.manual_seed(39); np.random.seed(39)
    out, K = 96, 280
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    _, g = _gpu_g(W, NintSpec(4, 24, 6))
    x = torch.randn(M, K, device=DEV) * 0.1
    qx, xscale, xsum = _workspace(g, x)
    xh = x.contiguous().to(torch.float16)
    y_gemv = ext().nint_gemv_packed_batch_ws_cuda(
        g["q_packed"], g["sub_scale"], g["sub_min"], g["neuron_scale"], g["neuron_min"],
        xh, int(g["gs"]), qx, xscale, xsum,
    )
    y_eff = ext().nint_gemv_packed_batch_eff_ws_cuda(
        g["q_packed"], g["d_eff_h"], g["m_eff_h"], xh, int(g["gs"]), qx, xscale, xsum,
    )
    torch.testing.assert_close(y_eff, y_gemv, atol=1e-3, rtol=2e-3)


@pytest.mark.parametrize("M", [2, 4, 6, 7])
def test_nint_batched_gemv_eff2_metadata_matches_gemv(M):
    """half2 execution metadata 的 MMVQ 数值应保持在 fp16 误差内。"""
    torch.manual_seed(40); np.random.seed(40)
    out, K = 96, 280
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    _, g = _gpu_g(W, NintSpec(4, 24, 6))
    x = torch.randn(M, K, device=DEV) * 0.1
    qx, xscale, xsum = _workspace(g, x)
    xh = x.contiguous().to(torch.float16)
    y_gemv = ext().nint_gemv_packed_batch_ws_cuda(
        g["q_packed"], g["sub_scale"], g["sub_min"], g["neuron_scale"], g["neuron_min"],
        xh, int(g["gs"]), qx, xscale, xsum,
    )
    y_eff2 = ext().nint_gemv_packed_batch_eff2_ws_cuda(
        g["q_packed"], g["eff_pair_h"], xh, int(g["gs"]), qx, xscale, xsum,
    )
    torch.testing.assert_close(y_eff2, y_gemv, atol=1e-3, rtol=2e-3)


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_nint_input_mul_compact_decode_matches_materialized(activation):
    """实际 compact decode 路径应等价于先 materialize gated activation 再 GEMV。"""
    torch.manual_seed(60); np.random.seed(60)
    out, K, M = 96, 280, 1
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt = nint_quantize(W, NintSpec(4, 24, 6), axis=0)
    g = to_gpu(nt, layout="deploy")
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    gate = (torch.randn(M, K, device=DEV) * 0.5).to(torch.float16)
    if activation == "sigmoid":
        materialized = x * torch.sigmoid(gate)
    else:
        materialized = x * F.silu(gate)
    y_ref = fused_matmul(g, materialized)
    y_fused = nint_matmul_input_mul(g, x, gate, activation)
    torch.testing.assert_close(y_fused, y_ref, atol=2e-3, rtol=3e-3)


@pytest.mark.parametrize(
    "spec",
    [
        NintSpec(2, 16, 5),
        NintSpec(3, 24, 5),
        NintSpec(5, 28, 7),
        NintSpec(6, 24, 7),
        NintSpec(8, 48, 7),
    ],
)
@pytest.mark.parametrize("activation,mode", [("sigmoid", 1), ("silu", 2)])
@pytest.mark.parametrize("M", [1, 2, 7, 8])
def test_nint_packed_bits_input_mul_decode_matches_materialized(spec, activation, mode, M):
    """C++ decode uses this packed-bits gate path directly for NINT5/6/8."""
    torch.manual_seed(62 + spec.bits + M); np.random.seed(62 + spec.bits + M)
    out, K = 96, 280
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt = nint_quantize(W, spec, axis=0)
    g = to_gpu(nt, layout="deploy")
    x = (torch.randn(M, K, device=DEV) * 0.35).to(torch.float16)
    gate = (torch.randn(M, K, device=DEV) * 1.2).to(torch.float16)
    multiplier = torch.sigmoid(gate) if activation == "sigmoid" else F.silu(gate)
    y_ref = fused_matmul(g, x * multiplier)
    qx, xscale, xsum = _workspace(g, x)
    y_fused = ext().nint_gemv_packed_bits_gate_ws_cuda(
        g["q_packed"], g["sub_scale"], g["sub_min"],
        g["neuron_scale"], g["neuron_min"], x, gate,
        int(g["gs"]), int(g["bits"]), mode, qx, xscale, xsum,
    )
    diff = y_fused.float() - y_ref.float()
    assert (diff.norm() / y_ref.float().norm()).item() < 5e-3
    assert diff.abs().max().item() < 5e-3


def test_nint5_linear_out_norm_gate_decode_matches_materialized():
    """Validate the decode-only linear-attention output fusion used by the baseline recipe."""
    torch.manual_seed(70); np.random.seed(70)
    heads, head_dim, out = 8, 32, 96
    K = heads * head_dim
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt = nint_quantize(W, NintSpec(5, 28, 7), axis=0)
    g = to_gpu(nt, layout="deploy")
    y = (torch.randn(K, device=DEV) * 0.4).to(torch.float32)
    gate = (torch.randn(K, device=DEV) * 1.2).to(torch.float16)
    norm_weight = (torch.randn(head_dim, device=DEV) * 0.08 + 1.0).to(torch.float32)
    eps = 1e-6
    rows = y.reshape(heads, head_dim)
    normalized = rows * torch.rsqrt(rows.square().mean(-1, keepdim=True) + eps)
    materialized = (normalized * norm_weight).reshape(1, K) * F.silu(gate).reshape(1, K)
    y_ref = fused_matmul(g, materialized.to(torch.float16))
    qx, xscale, xsum = _workspace(g, materialized)
    rinv = torch.empty(heads, device=DEV, dtype=torch.float32)
    y_fused = ext().nint_gemv_packed_bits_linear_out_norm_gate_ws_cuda(
        g["q_packed"], g["sub_scale"], g["sub_min"],
        g["neuron_scale"], g["neuron_min"], y, gate, norm_weight,
        int(g["gs"]), int(g["bits"]), head_dim, eps,
        qx, xscale, xsum, rinv,
    )
    diff = y_fused.float() - y_ref.float()
    assert (diff.norm() / y_ref.float().norm()).item() < 3e-3
    assert diff.abs().max().item() < 5e-3


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
@pytest.mark.parametrize("M", [1, 2, 7])
def test_nint_input_mul_eff2_decode_matches_materialized(M, activation):
    """half2 metadata decode 路径应等价于先 materialize gated activation 再 GEMV。"""
    torch.manual_seed(61); np.random.seed(61)
    out, K = 96, 280
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt = nint_quantize(W, NintSpec(4, 24, 6), axis=0)
    g_exp = to_gpu(nt, layout="experimental")
    g_eff = {
        "q_packed": g_exp["q_packed"],
        "eff_pair_h": g_exp["eff_pair_h"],
        "out": g_exp["out"],
        "ng": g_exp["ng"],
        "gs": g_exp["gs"],
        "neuron_len": g_exp["neuron_len"],
    }
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    gate = (torch.randn(M, K, device=DEV) * 0.5).to(torch.float16)
    if activation == "sigmoid":
        materialized = x * torch.sigmoid(gate)
    else:
        materialized = x * F.silu(gate)
    y_ref = fused_matmul(g_exp, materialized)
    y_fused = nint_matmul_input_mul(g_eff, x, gate, activation)
    torch.testing.assert_close(y_fused, y_ref, atol=2e-3, rtol=3e-3)


@pytest.mark.parametrize("M", [16, 32, 64])
def test_nint_mmq_exec_weight_layout_matches_workspace(M):
    """MMQ 权重执行格式应与 row-major packed MMQ 一致。"""
    torch.manual_seed(36); np.random.seed(36)
    out, K = 96, 280
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    _, g = _gpu_g(W, NintSpec(4, 24, 6))
    x = torch.randn(M, K, device=DEV) * 0.1
    qx, xscale, xsum = _workspace(g, x)
    row_args = (g["q_packed"], g["sub_scale"], g["sub_min"], g["neuron_scale"], g["neuron_min"])
    exec_args = (
        g["q_mmq_packed"],
        g["sub_scale_mmq"],
        g["sub_min_mmq"],
        g["neuron_scale"],
        g["neuron_min"],
    )
    xh = x.contiguous().to(torch.float16)
    y_row = ext().nint_mmq_packed_ws_cuda(*row_args, xh, int(g["gs"]), qx, xscale, xsum)
    y_exec = ext().nint_mmq_packed_exec_ws_cuda(*exec_args, xh, int(g["ng"]), int(g["gs"]), qx, xscale, xsum)
    torch.testing.assert_close(y_exec, y_row, atol=0, rtol=0)


@pytest.mark.parametrize("M", [16, 32])
def test_nint4_gs24_group32_matches_dequant(M):
    torch.manual_seed(130 + M); np.random.seed(130 + M)
    out, K = 128, 280
    nt, g = _gpu_g((np.random.randn(out, K).astype(np.float32)) * 0.05, NintSpec(4, 24, 6))
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    ng = int(g["ng"])
    nchunks = (ng + 7) // 8
    qx_mmq = torch.empty(nchunks * M * 68, device=DEV, dtype=torch.int32)
    xscale = torch.empty((M, ng), device=DEV, dtype=torch.float32)
    xsum = torch.empty((M, ng), device=DEV, dtype=torch.int32)
    partial = torch.empty((2, M, out), device=DEV, dtype=torch.float32)
    y = ext().nint_mmq_gs24_group32_ws_cuda(
        g["q_packed"], g["sub_scale"], g["sub_min"], g["neuron_scale"],
        g["neuron_min"], x, qx_mmq, xscale, xsum, 2, partial,
    )
    ref = _ref_out(nt, x)
    rel = ((y - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"M{M} group32 rel={rel}"


@pytest.mark.parametrize("M", [9, 17, 65, 512])
def test_nint2_gs16_pair32_matches_dequant(M):
    torch.manual_seed(2160 + M); np.random.seed(2160 + M)
    out, K = 73, 1537
    nt, g = _gpu_g(
        (np.random.randn(out, K).astype(np.float32)) * 0.05,
        NintSpec(2, 16, 5),
    )
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    ng = int(g["ng"])
    m_pad = ((M + 15) // 16) * 16
    nchunks = (ng + 7) // 8
    qx_mmq = torch.empty(nchunks * m_pad * 36, device=DEV, dtype=torch.int32)
    xscale = torch.empty((M, ng), device=DEV, dtype=torch.float32)
    xsum = torch.empty((M, ng), device=DEV, dtype=torch.int32)
    partial = torch.empty((2, M, out), device=DEV, dtype=torch.float32)
    y = ext().nint_mmq_gs24_group32_ws_cuda(
        g["q_packed"], g["sub_scale"], g["sub_min"], g["neuron_scale"],
        g["neuron_min"], x, qx_mmq, xscale, xsum, 2, partial,
    )
    ref = _ref_out(nt, x)
    rel = ((y - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"NINT2 M{M} pair32 rel={rel}"


@pytest.mark.parametrize("M", [9, 32, 128])
def test_nint2_gs16_f16_mmq_matches_dequant(M):
    torch.manual_seed(2180 + M); np.random.seed(2180 + M)
    out, K = 128, 280
    nt, g = _gpu_g(
        (np.random.randn(out, K).astype(np.float32)) * 0.05,
        NintSpec(2, 16, 5),
    )
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    y = ext().nint_mmq_gs24_f16_nint3_cuda(
        g["q_packed"], g["sub_scale"], g["sub_min"],
        g["neuron_scale"], g["neuron_min"], x,
    )
    ref = _ref_out(nt, x)
    rel = ((y - ref).norm() / ref.norm()).item()
    assert rel < 3e-3, f"NINT2 M{M} fp16 MMQ rel={rel}"


def test_nint2_ffn_gate_up_swiglu_quant_matches_materialized():
    torch.manual_seed(2192); np.random.seed(2192)
    out, K = 64, 256
    nt, g = _gpu_g(
        (np.random.randn(out * 2, K).astype(np.float32)) * 0.05,
        NintSpec(2, 16, 5),
    )
    x = (torch.randn(1, K, device=DEV) * 0.1).to(torch.float16)
    gu_qx = torch.empty((1, int(g["ng"]) * 16), device=DEV, dtype=torch.int8)
    gu_xscale = torch.empty((1, int(g["ng"])), device=DEV, dtype=torch.float32)
    gu_xsum = torch.empty((1, int(g["ng"])), device=DEV, dtype=torch.int32)
    down_ng = (out + 15) // 16
    down_qx = torch.empty((1, down_ng * 16), device=DEV, dtype=torch.int8)
    down_xscale = torch.empty((1, down_ng), device=DEV, dtype=torch.float32)
    down_xsum = torch.empty((1, down_ng), device=DEV, dtype=torch.int32)
    ext().nint_ffn_gate_up_swiglu_quant_ws_cuda(
        g["q_packed"], g["sub_scale"], g["sub_min"],
        g["neuron_scale"], g["neuron_min"], x,
        16, 2, 16,
        gu_qx, gu_xscale, gu_xsum,
        down_qx, down_xscale, down_xsum,
    )
    gate_up = fused_matmul(g, x).float()
    expected = F.silu(gate_up[:, :out]) * gate_up[:, out:]
    actual = (
        down_qx[:, :out].float().reshape(1, down_ng, 16)
        * down_xscale[:, :, None]
    ).reshape(1, out)
    relative = ((actual - expected).norm() / expected.norm()).item()
    assert relative < 0.02, f"NINT2 fused FFN quantization relative={relative}"


@pytest.mark.parametrize("bits", [3, 4, 6])
@pytest.mark.parametrize("M", [16, 23, 32])
def test_nint_gs24_f16_mmq_matches_dequant(bits, M):
    torch.manual_seed(170 + bits + M); np.random.seed(170 + bits + M)
    out, K = 128, 280
    nt, g = _gpu_g((np.random.randn(out, K).astype(np.float32)) * 0.05, NintSpec(bits, 24, 6))
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    if bits == 3:
        y = ext().nint_mmq_gs24_f16_nint3_cuda(
            g["q_packed"], g["sub_scale"], g["sub_min"],
            g["neuron_scale"], g["neuron_min"], x,
        )
    elif bits == 4:
        y = ext().nint_mmq_gs24_f16_nint4_cuda(
            g["q_packed"], g["sub_scale"], g["sub_min"],
            g["neuron_scale"], g["neuron_min"], x,
        )
    else:
        partial = torch.empty((4, M, out), device=DEV, dtype=torch.float32)
        y = ext().nint_mmq_gs24_f16_nint6_split4_ws_cuda(
            g["q_packed"], g["sub_scale"], g["sub_min"],
            g["neuron_scale"], g["neuron_min"], x, partial,
        )
    ref = _ref_out(nt, x)
    rel = ((y - ref).norm() / ref.norm()).item()
    assert rel < 3e-3, f"NINT{bits} M{M} fp16 MMQ rel={rel}"


@pytest.mark.parametrize("bits", [3, 6])
@pytest.mark.parametrize("M", [9, 16, 32, 64, 257, 512])
def test_nint_gs24_group32_matches_dequant(bits, M):
    torch.manual_seed(260 + bits + M); np.random.seed(260 + bits + M)
    out, K = 128, 280
    nt, g = _gpu_g(
        (np.random.randn(out, K).astype(np.float32)) * 0.05,
        NintSpec(bits, 24, 7 if bits == 6 else 5),
    )
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    ng = int(g["ng"])
    m_pad = ((M + 15) // 16) * 16
    nchunks = (ng + 7) // 8
    qx_mmq = torch.empty(nchunks * m_pad * 68, device=DEV, dtype=torch.int32)
    xscale = torch.empty((M, ng), device=DEV, dtype=torch.float32)
    xsum = torch.empty((M, ng), device=DEV, dtype=torch.int32)
    partial = torch.empty((2, M, out), device=DEV, dtype=torch.float32)
    y = ext().nint_mmq_gs24_group32_ws_cuda(
        g["q_packed"], g["sub_scale"], g["sub_min"], g["neuron_scale"],
        g["neuron_min"], x, qx_mmq, xscale, xsum, 2, partial,
    )
    ref = _ref_out(nt, x)
    rel = ((y - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"NINT{bits} M{M} group32 rel={rel}"


@pytest.mark.parametrize("M", [9, 64, 257])
def test_nint3_production_dispatch_matches_dequant(M):
    torch.manual_seed(310 + M)
    np.random.seed(310 + M)
    out, K = 1024, 280
    nt, g = _gpu_g(
        (np.random.randn(out, K).astype(np.float32)) * 0.05,
        NintSpec(3, 24, 5),
    )
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    y = fused_matmul(g, x)
    ref = _ref_out(nt, x)
    rel = ((y - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"NINT3 M{M} production rel={rel}"


@pytest.mark.parametrize("M", [16, 23, 32, 64, 257, 512])
def test_nint6_production_dispatch_matches_dequant(M):
    torch.manual_seed(360 + M)
    np.random.seed(360 + M)
    out, K = 128, 280
    nt, g = _gpu_g(
        (np.random.randn(out, K).astype(np.float32)) * 0.05,
        NintSpec(6, 24, 7),
    )
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)
    y = fused_matmul(g, x)
    ref = _ref_out(nt, x)
    rel = ((y - ref).norm() / ref.norm()).item()
    assert rel < 3e-3, f"NINT6 M{M} production rel={rel}"


def test_nint6_int8_mmq_requires_explicit_opt_in(monkeypatch):
    torch.manual_seed(366)
    np.random.seed(366)
    out, K, M = 1024, 280, 16
    nt, g = _gpu_g(
        (np.random.randn(out, K).astype(np.float32)) * 0.05,
        NintSpec(6, 24, 7),
    )
    x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float16)

    monkeypatch.delenv("MFQ_NINT6_MMQ", raising=False)
    default = fused_matmul(g, x)
    assert "_group32_workspace" not in g

    monkeypatch.setenv("MFQ_NINT6_MMQ", "int8")
    explicit = fused_matmul(g, x)
    assert "_group32_workspace" in g

    ref = _ref_out(nt, x)
    default_rel = ((default - ref).norm() / ref.norm()).item()
    explicit_rel = ((explicit - ref).norm() / ref.norm()).item()
    assert default_rel < 3e-3
    assert explicit_rel < 1e-2


def test_nint_fused_large_shape():
    """较大形状（FFN 量级切片），检查多 block、无 NaN。"""
    torch.manual_seed(4); np.random.seed(4)
    out, K, M = 512, 768, 64
    W = (np.random.randn(out, K).astype(np.float32)) * 0.05
    nt, g = _gpu_g(W, NintSpec(4, 24, 6))
    x = torch.randn(M, K, device=DEV) * 0.1
    y_fused = fused_matmul(g, x)
    y_ref = _ref_out(nt, x)
    assert torch.isfinite(y_fused).all()
    rel = ((y_fused - y_ref).norm() / y_ref.norm()).item()
    assert rel < 2e-2, f"large rel={rel}"


# ---------------------------------------------------------------------------
# 标准算子 CUDA kernel（对照 torch 内建实现）
# ---------------------------------------------------------------------------
def test_rms_norm_cuda():
    torch.manual_seed(0)
    x = torch.randn(8, 128, device=DEV)
    w = torch.randn(128, device=DEV)
    eps = 1e-6
    var = x.pow(2).mean(-1, keepdim=True)
    y_ref = (x * torch.rsqrt(var + eps)) * w
    torch.testing.assert_close(rms_norm(x, w, eps), y_ref, atol=1e-5, rtol=1e-5)


def test_l2_norm_cuda():
    torch.manual_seed(1)
    x = torch.randn(4, 64, device=DEV)
    torch.testing.assert_close(l2_norm(x), F.normalize(x, dim=-1, eps=1e-5), atol=1e-5, rtol=1e-5)


def test_acc_cuda():
    torch.manual_seed(2)
    a = torch.randn(7, 32, device=DEV)
    b = torch.randn(7, 32, device=DEV)
    torch.testing.assert_close(acc(a, b), a + b, atol=1e-6, rtol=1e-6)


def test_acc_cuda_f16_preserves_dtype():
    torch.manual_seed(202)
    a = torch.randn(7, 32, device=DEV, dtype=torch.float16)
    b = torch.randn(7, 32, device=DEV, dtype=torch.float16)
    y = acc(a, b)
    assert y.dtype == torch.float16
    torch.testing.assert_close(y, a + b, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 16])
def test_gemma4_fused_pre_norms_match_materialized_path(rows):
    torch.manual_seed(203 + rows)
    width = 2816
    eps = 1e-6
    residual = torch.randn(rows, width, device=DEV, dtype=torch.float16)
    attn = torch.randn_like(residual)
    weights = [
        torch.randn(width, device=DEV, dtype=torch.float32)
        for _ in range(4)
    ]

    attn_post = ext().rms_norm_f16_cuda(attn, weights[0], eps, 0.0)
    residual_ref = ext().acc_cuda(residual, attn_post)
    dense_ref = ext().rms_norm_f16_cuda(residual_ref, weights[1], eps, 0.0)
    router_ref = ext().rms_norm_offset_cuda(
        residual_ref.float().contiguous(), weights[2], eps, 0.0
    )
    moe_ref = ext().rms_norm_f16_cuda(residual_ref, weights[3], eps, 0.0)

    actual = ext().gemma4_attn_residual_pre_norms_f16_cuda(
        residual, attn, *weights, eps
    )
    for got, expected in zip(actual, (residual_ref, dense_ref, router_ref, moe_ref)):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 16])
def test_gemma4_fused_ffn_merge_matches_materialized_path(rows):
    torch.manual_seed(223 + rows)
    width = 2816
    eps = 1e-6
    dense = torch.randn(rows, width, device=DEV, dtype=torch.float16)
    moe = torch.randn_like(dense)
    residual = torch.randn_like(dense)
    weights = [
        torch.randn(width, device=DEV, dtype=torch.float32)
        for _ in range(3)
    ]
    layer_scale = torch.tensor([0.9375], device=DEV, dtype=torch.float16)

    dense_post = ext().rms_norm_f16_cuda(dense, weights[0], eps, 0.0)
    moe_post = ext().rms_norm_f16_cuda(moe, weights[1], eps, 0.0)
    combined = (dense_post + moe_post).contiguous()
    post = ext().rms_norm_f16_cuda(combined, weights[2], eps, 0.0)
    expected = (ext().acc_cuda(residual, post) * layer_scale).contiguous()

    actual = ext().gemma4_ffn_merge_f16_cuda(
        dense, moe, residual, *weights, layer_scale, eps
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_silu_mul_cuda_f32():
    torch.manual_seed(21)
    gate = torch.randn(5, 37, device=DEV, dtype=torch.float32)
    up = torch.randn(5, 37, device=DEV, dtype=torch.float32)
    torch.testing.assert_close(silu_mul(gate, up), F.silu(gate) * up, atol=1e-6, rtol=1e-6)


def test_silu_mul_cuda_f16():
    torch.manual_seed(22)
    gate = torch.randn(6, 41, device=DEV, dtype=torch.float16)
    up = torch.randn(6, 41, device=DEV, dtype=torch.float16)
    ref = (F.silu(gate.float()) * up.float()).to(torch.float16)
    torch.testing.assert_close(silu_mul(gate, up), ref, atol=1e-3, rtol=1e-3)


def _ssm_conv_ref(conv_input, weight, n_tokens):
    if weight.dim() == 3:
        K = weight.shape[-1]
        w = weight[:, 0, :]
    elif weight.shape[0] == conv_input.shape[-1]:
        K = weight.shape[1]
        w = weight
    else:
        K = weight.shape[0]
        w = weight.t()
    windows = conv_input[:, : n_tokens + K - 1].unfold(1, K, 1)
    return F.silu((windows * w.view(1, 1, w.size(0), K)).sum(dim=-1))


def test_ssm_conv_silu_cuda_c1k_matches_torch():
    torch.manual_seed(29)
    B, T, C, K = 2, 7, 65, 4
    conv_input = torch.randn(B, T + K - 1, C, device=DEV)
    weight = torch.randn(C, 1, K, device=DEV)
    y = ssm_conv_silu(conv_input, weight, T)
    torch.testing.assert_close(y, _ssm_conv_ref(conv_input, weight, T), atol=1e-5, rtol=1e-5)


def test_ssm_conv_silu_cuda_kc_matches_torch():
    torch.manual_seed(30)
    B, T, C, K = 1, 9, 33, 5
    conv_input = torch.randn(B, T + K - 1, C, device=DEV)
    weight = torch.randn(K, C, device=DEV)
    y = ssm_conv_silu(conv_input, weight, T)
    torch.testing.assert_close(y, _ssm_conv_ref(conv_input, weight, T), atol=1e-5, rtol=1e-5)


def test_ssm_conv_silu_cuda_ck_matches_torch():
    torch.manual_seed(32)
    B, T, C, K = 1, 9, 33, 5
    conv_input = torch.randn(B, T + K - 1, C, device=DEV)
    weight = torch.randn(C, K, device=DEV)
    y = ssm_conv_silu(conv_input, weight, T)
    torch.testing.assert_close(y, _ssm_conv_ref(conv_input, weight, T), atol=1e-5, rtol=1e-5)


def test_ssm_conv_silu_cuda_bias_matches_torch():
    torch.manual_seed(31)
    B, T, C, K = 2, 6, 17, 4
    conv_input = torch.randn(B, T + K - 1, C, device=DEV)
    weight = torch.randn(C, 1, K, device=DEV)
    bias = torch.randn(C, device=DEV)
    windows = conv_input[:, : T + K - 1].unfold(1, K, 1)
    ref = F.silu((windows * weight[:, 0, :].view(1, 1, C, K)).sum(dim=-1) + bias.view(1, 1, C))
    y = ssm_conv_silu(conv_input, weight, T, bias)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


def test_kv_cache_write_matches_assignment():
    torch.manual_seed(23)
    B, H, T, D, max_seq = 2, 3, 4, 8, 11
    k = torch.randn(B, H, T, D, device=DEV)
    v = torch.randn(B, H, T, D, device=DEV)
    kc = torch.zeros(B, H, max_seq, D, device=DEV)
    vc = torch.zeros(B, H, max_seq, D, device=DEV)
    pos = torch.tensor([1, 3, 7, 9], device=DEV, dtype=torch.int64)
    kv_cache_write(kc, vc, k, v, pos)
    k_ref = torch.zeros_like(kc)
    v_ref = torch.zeros_like(vc)
    k_ref[:, :, pos.cpu().tolist(), :] = k
    v_ref[:, :, pos.cpu().tolist(), :] = v
    torch.testing.assert_close(kc, k_ref, atol=0, rtol=0)
    torch.testing.assert_close(vc, v_ref, atol=0, rtol=0)


def test_kv_cache_write_batch_positions():
    torch.manual_seed(24)
    B, H, T, D, max_seq = 2, 2, 3, 8, 9
    k = torch.randn(B, H, T, D, device=DEV)
    v = torch.randn(B, H, T, D, device=DEV)
    kc = torch.zeros(B, H, max_seq, D, device=DEV)
    vc = torch.zeros(B, H, max_seq, D, device=DEV)
    pos = torch.tensor([[0, 2, 4], [1, 3, 5]], device=DEV, dtype=torch.int64)
    kv_cache_write(kc, vc, k, v, pos)
    k_ref = torch.zeros_like(kc)
    v_ref = torch.zeros_like(vc)
    for b in range(B):
        for t in range(T):
            p = int(pos[b, t].item())
            k_ref[b, :, p, :] = k[b, :, t, :]
            v_ref[b, :, p, :] = v[b, :, t, :]
    torch.testing.assert_close(kc, k_ref, atol=0, rtol=0)
    torch.testing.assert_close(vc, v_ref, atol=0, rtol=0)


def test_kv_cache_append_attention_matches_sdpa():
    torch.manual_seed(25)
    B, Hq, Hkv, D = 1, 4, 2, 16
    cache = KVCache(B, Hkv, 8, D, device=DEV, dtype=torch.float32)
    k0 = torch.randn(B, Hkv, 3, D, device=DEV)
    v0 = torch.randn(B, Hkv, 3, D, device=DEV)
    k1 = torch.randn(B, Hkv, 2, D, device=DEV)
    v1 = torch.randn(B, Hkv, 2, D, device=DEV)
    cache.append(k0, v0)
    kc, vc = cache.append(k1, v1)
    q = torch.randn(B, Hq, 2, D, device=DEV)
    y = attention(q, kc, vc, causal=True)
    rep = Hq // Hkv
    kr = torch.cat([k0, k1], dim=2).repeat_interleave(rep, dim=1)
    vr = torch.cat([v0, v1], dim=2).repeat_interleave(rep, dim=1)
    att = torch.full((2, 5), float("-inf"), device=DEV)
    offset = 3
    for tq in range(2):
        for s in range(5):
            if s <= tq + offset:
                att[tq, s] = 0.0
    y_ref = F.scaled_dot_product_attention(q, kr, vr, attn_mask=att.view(1, 1, 2, 5))
    torch.testing.assert_close(y, y_ref, atol=2e-4, rtol=2e-4)


def test_kv_cache_default_f16_and_attention_matches_sdpa():
    torch.manual_seed(251)
    B, Hq, Hkv, D = 1, 4, 2, 16
    cache = KVCache(B, Hkv, 8, D, device=DEV)
    assert cache.k.dtype == torch.float16
    k = torch.randn(B, Hkv, 5, D, device=DEV, dtype=torch.float16)
    v = torch.randn(B, Hkv, 5, D, device=DEV, dtype=torch.float16)
    kc, vc = cache.append(k, v)
    assert kc.dtype == torch.float16
    q = torch.randn(B, Hq, 2, D, device=DEV, dtype=torch.float16)
    y = attention(q, kc, vc, causal=True)
    assert y.dtype == torch.float16
    rep = Hq // Hkv
    kr = k.repeat_interleave(rep, dim=1)
    vr = v.repeat_interleave(rep, dim=1)
    att = torch.full((2, 5), float("-inf"), device=DEV, dtype=torch.float16)
    offset = 3
    for tq in range(2):
        for s in range(5):
            if s <= tq + offset:
                att[tq, s] = 0.0
    y_ref = F.scaled_dot_product_attention(q, kr, vr, attn_mask=att.view(1, 1, 2, 5))
    torch.testing.assert_close(y, y_ref, atol=1e-3, rtol=1e-3)


def test_kv_cache_grows_without_full_prealloc():
    B, Hkv, D = 1, 2, 8
    cache = KVCache(B, Hkv, 1024, D, device=DEV, initial_capacity=2)
    assert cache.capacity == 2
    k = torch.randn(B, Hkv, 5, D, device=DEV)
    v = torch.randn(B, Hkv, 5, D, device=DEV)
    kc, vc = cache.append(k, v)
    assert cache.capacity == 8
    assert tuple(kc.shape) == (B, Hkv, 5, D)
    torch.testing.assert_close(kc, k.to(torch.float16), atol=0, rtol=0)
    torch.testing.assert_close(vc, v.to(torch.float16), atol=0, rtol=0)


def test_embedding_lookup_cuda_f16():
    torch.manual_seed(26)
    weight = torch.randn(17, 13, device=DEV, dtype=torch.float16)
    ids = torch.tensor([[0, 3, 16], [8, 2, 5]], device=DEV, dtype=torch.int64)
    torch.testing.assert_close(embedding(weight, ids), weight[ids], atol=0, rtol=0)


def test_embedding_lookup_cuda_f32():
    torch.manual_seed(27)
    weight = torch.randn(19, 11, device=DEV, dtype=torch.float32)
    ids = torch.tensor([18, 1, 7, 0], device=DEV, dtype=torch.int64)
    torch.testing.assert_close(embedding(weight, ids), weight[ids], atol=0, rtol=0)


def test_nint_embedding_lookup_matches_dequant_rows():
    torch.manual_seed(28); np.random.seed(28)
    vocab, D = 32, 70
    W = (np.random.randn(vocab, D).astype(np.float32)) * 0.05
    nt, g = _gpu_g(W, NintSpec(4, 24, 6))
    ids = torch.tensor([[0, 3, 31], [8, 2, 5]], device=DEV, dtype=torch.int64)
    y = nint_embedding(g, ids)
    w_ref = torch.as_tensor(np.ascontiguousarray(nint_dequant(nt)), device=DEV, dtype=torch.float16)
    torch.testing.assert_close(y, w_ref[ids], atol=0, rtol=0)


def test_nint_embedding_lookup_default_deploy_matches_dequant_rows():
    torch.manual_seed(60); np.random.seed(60)
    vocab, D = 32, 70
    W = (np.random.randn(vocab, D).astype(np.float32)) * 0.05
    nt = nint_quantize(W, NintSpec(4, 24, 6), axis=0)
    g = to_gpu(nt)
    assert "eff_pair_h" not in g
    ids = torch.tensor([[0, 3, 31], [8, 2, 5]], device=DEV, dtype=torch.int64)
    y = nint_embedding(g, ids)
    w_ref = torch.as_tensor(np.ascontiguousarray(nint_dequant(nt)), device=DEV, dtype=torch.float16)
    torch.testing.assert_close(y, w_ref[ids], atol=0, rtol=0)


@pytest.mark.parametrize(
    "spec",
    [
        NintSpec(2, 16, 5),
        NintSpec(3, 24, 5),
        NintSpec(5, 24, 6),
        NintSpec(6, 22, 6),
        NintSpec(8, 32, 6),
    ],
)
def test_nint_embedding_lookup_non4_bits_matches_dequant_rows(spec):
    torch.manual_seed(71 + spec.bits); np.random.seed(71 + spec.bits)
    vocab, D = 32, 70
    W = (np.random.randn(vocab, D).astype(np.float32)) * 0.05
    nt = nint_quantize(W, spec, axis=0)
    g = to_gpu(nt)
    ids = torch.tensor([[0, 3, 31], [8, 2, 5]], device=DEV, dtype=torch.int64)
    y = nint_embedding(g, ids)
    w_ref = torch.as_tensor(np.ascontiguousarray(nint_dequant(nt)), device=DEV, dtype=torch.float16)
    torch.testing.assert_close(y, w_ref[ids], atol=0, rtol=0)


def test_sample_greedy_cuda_f32_f16_bf16():
    logits = torch.tensor(
        [[1.0, 5.0, 5.0, -1.0], [0.0, -2.0, 3.0, 1.0]],
        device=DEV,
        dtype=torch.float32,
    )
    ref = torch.tensor([1, 2], device=DEV, dtype=torch.int64)
    torch.testing.assert_close(sample_greedy(logits), ref, atol=0, rtol=0)
    torch.testing.assert_close(sample_greedy(logits.to(torch.float16)), ref, atol=0, rtol=0)
    torch.testing.assert_close(sample_greedy(logits.to(torch.bfloat16)), ref, atol=0, rtol=0)


def test_sample_softmax_cuda_matches_reference():
    logits = torch.tensor(
        [[0.1, 0.4, -0.2, 1.0], [2.0, -1.0, 0.0, 0.5]],
        device=DEV,
        dtype=torch.float32,
    )
    rnd = torch.tensor([0.10, 0.95], device=DEV, dtype=torch.float32)
    y = sample(logits, temperature=0.7, random=rnd)
    probs = torch.softmax(logits / 0.7, dim=-1)
    ref = torch.searchsorted(torch.cumsum(probs, dim=-1), rnd[:, None]).squeeze(1).to(torch.int64)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


def test_sample_top_k_top_p_cuda_matches_reference():
    logits = torch.tensor(
        [[0.1, 2.0, 1.2, -0.5, 0.8], [1.0, 0.9, 0.1, 2.5, -1.0]],
        device=DEV,
        dtype=torch.float32,
    )
    rnd = torch.tensor([0.40, 0.80], device=DEV, dtype=torch.float32)
    y = sample(logits, temperature=1.0, top_k=3, top_p=0.75, random=rnd)
    refs = []
    vals, idx = torch.topk(logits, k=3, dim=-1)
    probs = torch.softmax(vals, dim=-1)
    for b in range(logits.size(0)):
        cutoff = 0.75
        c = torch.cumsum(probs[b], dim=0)
        keep = int((c >= cutoff).nonzero()[0].item()) + 1
        kept = probs[b, :keep]
        kept = kept / kept.sum()
        j = int(torch.searchsorted(torch.cumsum(kept, dim=0), rnd[b]).item())
        refs.append(int(idx[b, j].item()))
    ref = torch.tensor(refs, device=DEV, dtype=torch.int64)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


def test_sample_top_k_cuda_tie_breaks_by_lowest_token_id():
    logits = torch.ones((1, 8), device=DEV, dtype=torch.float32)
    rnd = torch.tensor([0.9], device=DEV, dtype=torch.float32)
    y = sample(logits, temperature=1.0, top_k=3, top_p=1.0, random=rnd)
    torch.testing.assert_close(y, torch.tensor([2], device=DEV), atol=0, rtol=0)


def test_sample_penalties_cuda_counts_and_updates_in_place():
    counts = torch.zeros(5, device=DEV, dtype=torch.int32)
    tokens = torch.tensor([1, 1, 3], device=DEV, dtype=torch.int64)
    ext().sample_token_counts_add_cuda(counts, tokens)
    torch.testing.assert_close(
        counts,
        torch.tensor([0, 2, 0, 1, 0], device=DEV, dtype=torch.int32),
        atol=0,
        rtol=0,
    )

    logits = torch.tensor([[2.0, -2.0, 1.0, 3.0, 4.0]], device=DEV, dtype=torch.float32)
    out = ext().sample_apply_penalties_cuda(logits, counts, 0.5, 0.25, 2.0)
    expected = torch.tensor([[2.0, -5.0, 1.0, 0.75, 4.0]], device=DEV, dtype=torch.float32)
    assert out.data_ptr() == logits.data_ptr()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def _rope_ref(x, pos, base=1e6, rotary_dim=None, sections=None):
    T, D = x.shape[-2], x.shape[-1]
    RD = D if rotary_dim is None else int(rotary_dim)
    half = RD // 2
    out = x.clone()
    freqs = base ** (-torch.arange(0, RD, 2, device=x.device, dtype=torch.float32) / RD)
    if sections is None:
        pos_used = pos.float().view(1, T).expand(half, T)
    else:
        axes = torch.empty(half, device=x.device, dtype=torch.long)
        s0, s1, s2 = sections
        axes[:s0] = 0
        axes[s0 : s0 + s1] = 1
        axes[s0 + s1 : s0 + s1 + s2] = 2
        pos2 = pos.float()
        if pos2.dim() == 1:
            pos2 = pos2.view(1, T).expand(3, T)
        pos_used = pos2[axes]
    angles = pos_used.t() * freqs.view(1, half)
    shape = (1,) * (x.dim() - 2) + (T, half)
    cos, sin = angles.cos().view(shape), angles.sin().view(shape)
    x0, x1 = x[..., :half], x[..., half:RD]
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:RD] = x1 * cos + x0 * sin
    return out


def test_rope_cuda_pos0_identity():
    torch.manual_seed(3)
    x = torch.randn(2, 3, 4, 8, device=DEV)
    torch.testing.assert_close(rope(x, torch.zeros(4, device=DEV)), x, atol=1e-5, rtol=1e-5)


def test_rope_cuda_matches_ref():
    torch.manual_seed(4)
    x = torch.randn(2, 3, 6, 16, device=DEV)
    pos = torch.arange(6.0, device=DEV)
    torch.testing.assert_close(rope(x, pos), _rope_ref(x, pos), atol=1e-5, rtol=1e-5)


def test_rope_cuda_table_matches_ref():
    torch.manual_seed(41)
    x = torch.randn(2, 3, 6, 16, device=DEV)
    pos = torch.arange(6, device=DEV, dtype=torch.int64) + 5
    torch.testing.assert_close(rope(x, pos, table_len=64), _rope_ref(x, pos), atol=1e-5, rtol=1e-5)


def test_rope_cuda_partial_matches_ref():
    torch.manual_seed(31)
    x = torch.randn(2, 3, 4, 10, device=DEV)
    pos = torch.arange(4.0, device=DEV) + 2
    y = rope(x, pos, rotary_dim=6)
    y_ref = _rope_ref(x, pos, rotary_dim=6)
    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y[..., 6:], x[..., 6:], atol=0, rtol=0)


def test_rope_cuda_mrope_sections_matches_ref():
    torch.manual_seed(32)
    x = torch.randn(1, 2, 5, 12, device=DEV)
    pos = torch.stack(
        [
            torch.arange(5, device=DEV),
            torch.arange(5, device=DEV) + 10,
            torch.arange(5, device=DEV) + 20,
        ],
        dim=0,
    )
    sections = (1, 2, 1)
    y = rope(x, pos, rotary_dim=8, sections=sections)
    y_ref = _rope_ref(x, pos, rotary_dim=8, sections=sections)
    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y[..., 8:], x[..., 8:], atol=0, rtol=0)


def test_attention_cuda_causal():
    torch.manual_seed(5)
    B, H, T, D = 2, 4, 8, 16
    q = torch.randn(B, H, T, D, device=DEV)
    k = torch.randn(B, H, T, D, device=DEV)
    v = torch.randn(B, H, T, D, device=DEV)
    y = attention(q, k, v, causal=True)
    y_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(y, y_ref, atol=2e-4, rtol=2e-4)


def test_attention_cuda_gqa():
    torch.manual_seed(6)
    B, Hq, Hk, T, D = 1, 8, 2, 6, 16
    q = torch.randn(B, Hq, T, D, device=DEV)
    k = torch.randn(B, Hk, T, D, device=DEV)
    v = torch.randn(B, Hk, T, D, device=DEV)
    rep = Hq // Hk
    kr = k.repeat_interleave(rep, dim=1)
    vr = v.repeat_interleave(rep, dim=1)
    y_ref = F.scaled_dot_product_attention(q, kr, vr, is_causal=False)
    torch.testing.assert_close(attention(q, k, v, causal=False), y_ref, atol=2e-4, rtol=2e-4)


def test_attention_cuda_decode_unequal():
    """Tq != Tk: validate the causal offset s <= tq + (Tk - T) against an explicit mask."""
    torch.manual_seed(7)
    B, Hq, Hk, Tq, Tk, D = 1, 4, 2, 2, 4, 16
    q = torch.randn(B, Hq, Tq, D, device=DEV)
    k = torch.randn(B, Hk, Tk, D, device=DEV)
    v = torch.randn(B, Hk, Tk, D, device=DEV)
    rep = Hq // Hk
    kr = k.repeat_interleave(rep, dim=1)
    vr = v.repeat_interleave(rep, dim=1)
    # explicit mask matching the kernel convention
    offset = Tk - Tq
    att = torch.full((Tq, Tk), float("-inf"), device=DEV)
    for tq in range(Tq):
        for s in range(Tk):
            if s <= tq + offset:
                att[tq, s] = 0.0
    y_ref = F.scaled_dot_product_attention(q, kr, vr, attn_mask=att.view(1, 1, Tq, Tk))
    torch.testing.assert_close(attention(q, k, v, causal=True), y_ref, atol=2e-4, rtol=2e-4)


def test_attention_cuda_splitk_decode_matches_sdpa():
    torch.manual_seed(71)
    B, Hq, Hk, Tq, Tk, D = 1, 8, 2, 1, 1024, 64
    q = torch.randn(B, Hq, Tq, D, device=DEV, dtype=torch.float16)
    k = torch.randn(B, Hk, Tk, D, device=DEV, dtype=torch.float16)
    v = torch.randn(B, Hk, Tk, D, device=DEV, dtype=torch.float16)
    rep = Hq // Hk
    kr = k.repeat_interleave(rep, dim=1)
    vr = v.repeat_interleave(rep, dim=1)
    y_ref = F.scaled_dot_product_attention(q, kr, vr, is_causal=False)
    y = attention(q, k, v, causal=True)
    torch.testing.assert_close(y, y_ref, atol=1e-3, rtol=1e-3)


def _swa_reference(q, k, v, window):
    rep = q.size(1) // k.size(1)
    kr = k.repeat_interleave(rep, dim=1)
    vr = v.repeat_interleave(rep, dim=1)
    tq, tk = q.size(2), k.size(2)
    offset = tk - tq
    mask = torch.full((tq, tk), float("-inf"), device=q.device, dtype=q.dtype)
    for row in range(tq):
        end = row + offset + 1
        start = max(0, end - window)
        mask[row, start:end] = 0
    return F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask.view(1, 1, tq, tk))


def test_attention_swa_gqa_unequal_lengths_matches_sdpa():
    torch.manual_seed(72)
    q = torch.randn(1, 8, 7, 64, device=DEV, dtype=torch.float16)
    k = torch.randn(1, 2, 19, 64, device=DEV, dtype=torch.float16)
    v = torch.randn_like(k)
    actual = sliding_window_attention(q, k, v, window=5)
    expected = _swa_reference(q, k, v, 5)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_attention_swa_circular_cache_wrap_matches_sdpa():
    torch.manual_seed(73)
    B, Hq, Hk, D, capacity, window = 1, 8, 2, 64, 8, 5
    k_all = torch.randn(B, Hk, 19, D, device=DEV, dtype=torch.float16)
    v_all = torch.randn_like(k_all)
    k_cache = torch.empty(B, Hk, capacity, D, device=DEV, dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    for start, end in ((0, 8), (8, 16), (16, 19)):
        kv_cache_write_ring(k_cache, v_cache, k_all[:, :, start:end], v_all[:, :, start:end], start)
    q = torch.randn(B, Hq, 3, D, device=DEV, dtype=torch.float16)
    seq_len = torch.tensor([19], device=DEV, dtype=torch.int64)
    actual = sliding_window_attention_cached(q, k_cache, v_cache, seq_len, window)
    expected = _swa_reference(q, k_all, v_all, window)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_attention_swa_circular_cache_splitk_matches_sdpa():
    torch.manual_seed(76)
    B, Hq, Hk, D, capacity, window, length = 1, 8, 2, 64, 768, 512, 900
    k_all = torch.randn(B, Hk, length, D, device=DEV, dtype=torch.float16)
    v_all = torch.randn_like(k_all)
    k_cache = torch.empty(B, Hk, capacity, D, device=DEV, dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    kv_cache_write_ring(k_cache, v_cache, k_all, v_all, 0)
    q = torch.randn(B, Hq, 3, D, device=DEV, dtype=torch.float16)
    seq_len = torch.tensor([length], device=DEV, dtype=torch.int64)
    actual = sliding_window_attention_cached(q, k_cache, v_cache, seq_len, window)
    expected = _swa_reference(q, k_all, v_all, window)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_sliding_window_cache_rejects_oversized_ubatch():
    cache = SlidingWindowKVCache(1, 2, 8, 64, ubatch_capacity=3)
    k = torch.randn(1, 2, 4, 64, device=DEV, dtype=torch.float16)
    with pytest.raises(ValueError, match="exceeding ubatch_capacity"):
        cache.append(k, k)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gelu_mul_matches_tanh_approximation(dtype):
    torch.manual_seed(74)
    gate = torch.randn(33, 71, device=DEV, dtype=dtype)
    up = torch.randn_like(gate)
    actual = gelu_mul(gate, up)
    expected = F.gelu(gate, approximate="tanh") * up
    atol = rtol = 1e-3 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_gelu_mul_saturates_f16_overflow_without_nonfinite_values():
    gate = torch.tensor(
        [[200.0, 200.0, -200.0]], device=DEV, dtype=torch.float16
    )
    up = torch.tensor(
        [[400.0, -400.0, 400.0]], device=DEV, dtype=torch.float16
    )
    actual = gelu_mul(gate, up)
    expected = torch.tensor(
        [[65504.0, -65504.0, -0.0]], device=DEV, dtype=torch.float16
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "spec",
    [NintSpec(3, 24, 5), NintSpec(4, 24, 6), NintSpec(6, 24, 6)],
)
def test_nint_fused_geglu_matches_materialized_quantized_projections(spec):
    torch.manual_seed(75 + spec.bits)
    np.random.seed(75 + spec.bits)
    width, kdim = 96, 193
    weights = np.random.randn(2 * width, kdim).astype(np.float32) * 0.04
    nt = nint_quantize(weights, spec, axis=0)
    g = to_gpu(nt)
    x = torch.randn(1, kdim, device=DEV, dtype=torch.float16)
    qx, xscale, xsum = _workspace(g, x)
    args = (
        g["q_packed"], g["sub_scale"], g["sub_min"],
        g["neuron_scale"], g["neuron_min"], x, int(g["gs"]),
    )
    if spec.bits == 4:
        pair = ext().nint_gemv_packed_ws_cuda(*args, qx, xscale, xsum)
        actual = ext().nint_gemv_packed_geglu_ws_cuda(*args, qx, xscale, xsum)
    else:
        pair = ext().nint_gemv_packed_bits_ws_cuda(
            *args, spec.bits, qx, xscale, xsum
        )
        actual = ext().nint_gemv_packed_bits_geglu_ws_cuda(
            *args, spec.bits, qx, xscale, xsum
        )
    gate, up = pair.chunk(2, dim=-1)
    expected = (F.gelu(gate.float(), approximate="tanh") * up.float()).half()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


_PACKED_BITS_GLU_SPECS = [
    NintSpec(3, 24, 5),
    *[
    NintSpec(bits, gs, 6 if bits < 8 else 7)
    for bits in (5, 6, 8)
    for gs in (16, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 48, 64)
    ],
]


@pytest.mark.parametrize("spec", _PACKED_BITS_GLU_SPECS)
def test_packed_bits_combined_glu_matches_two_warp_path(spec):
    seed = 91 + spec.bits * 100 + spec.groupsize
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden, width = spec.groupsize * 2 + 7, 64
    gate_up_np = np.random.randn(2 * width, hidden).astype(np.float32) * 0.04
    _, gate_up = _gpu_g(gate_up_np, spec)
    x = torch.randn(1, hidden, device=DEV, dtype=torch.float16)
    gu_qx, gu_xscale, gu_xsum = _workspace(gate_up, x)

    combined_key = "MFQ_NINT_GLU_COMBINED"
    previous_combined = os.environ.get(combined_key)
    try:
        os.environ[combined_key] = "0"
        refs = [
            fn(
                gate_up["q_packed"], gate_up["sub_scale"], gate_up["sub_min"],
                gate_up["neuron_scale"], gate_up["neuron_min"], x,
                int(gate_up["gs"]), int(gate_up["bits"]),
                gu_qx, gu_xscale, gu_xsum,
            )
            for fn in (
                ext().nint_gemv_packed_bits_swiglu_ws_cuda,
                ext().nint_gemv_packed_bits_geglu_ws_cuda,
            )
        ]
        os.environ[combined_key] = "1"
        actuals = [
            fn(
                gate_up["q_packed"], gate_up["sub_scale"], gate_up["sub_min"],
                gate_up["neuron_scale"], gate_up["neuron_min"], x,
                int(gate_up["gs"]), int(gate_up["bits"]),
                gu_qx, gu_xscale, gu_xsum,
            )
            for fn in (
                ext().nint_gemv_packed_bits_swiglu_ws_cuda,
                ext().nint_gemv_packed_bits_geglu_ws_cuda,
            )
        ]
    finally:
        if previous_combined is None:
            os.environ.pop(combined_key, None)
        else:
            os.environ[combined_key] = previous_combined
    for actual, expected in zip(actuals, refs):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("gs", [16, 24, 32, 48])
def test_nint4_glu_supports_silu_and_gelu_for_all_runtime_groupsizes(gs):
    seed = 700 + gs
    torch.manual_seed(seed)
    np.random.seed(seed)
    spec = NintSpec(4, gs, 6)
    hidden, width = gs * 2 + 7, 64
    weights = np.random.randn(2 * width, hidden).astype(np.float32) * 0.04
    nt = nint_quantize(weights, spec, axis=0)
    g = to_gpu(nt)
    x = torch.randn(1, hidden, device=DEV, dtype=torch.float16)
    qx, xscale, xsum = _workspace(g, x)
    args = (
        g["q_packed"], g["sub_scale"], g["sub_min"],
        g["neuron_scale"], g["neuron_min"], x, int(g["gs"]),
    )
    pair = ext().nint_gemv_packed_ws_cuda(*args, qx, xscale, xsum)
    gate, up = pair.chunk(2, dim=-1)
    expected_silu = (F.silu(gate.float()) * up.float()).half()
    expected_gelu = (F.gelu(gate.float(), approximate="tanh") * up.float()).half()
    actual_silu = ext().nint_gemv_packed_swiglu_ws_cuda(*args, qx, xscale, xsum)
    actual_gelu = ext().nint_gemv_packed_geglu_ws_cuda(*args, qx, xscale, xsum)
    torch.testing.assert_close(actual_silu, expected_silu, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual_gelu, expected_gelu, atol=1e-3, rtol=1e-3)
