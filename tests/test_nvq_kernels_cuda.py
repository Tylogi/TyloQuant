from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)
from mfq.formats.npq0_l import Npq0LTensor  # noqa: E402
from mfq.formats.npq0_s import Npq0STensor  # noqa: E402
from mfq.formats.nvq import (  # noqa: E402
    D4_256,
    D4_512,
    D4_1024,
    E8_256,
    E8_1024,
    E8_4096,
    NVQ2_E8,
    NVQ2_E8_1024,
    NVQ2_E8_4096,
    NVQ3_D4,
    NVQ3_D4_512,
    NVQ3_D4_1024,
    NvqJscTensor,
    NvqSpec,
    pack_jsc_metadata,
)
from mfq.formats.nvq1_l import IQ1S_TERNARY_2048, NVQ1_L_T8_S3  # noqa: E402
from mfq.formats.nvq1_s import NVQ1_S, NVQ1_S_SYNTHETIC_BANKS, Nvq1STensor  # noqa: E402
from mfq.kernels.cuda.nvq_matmul import (  # noqa: E402
    _select_matmul_path,
    nvq2_matmul_swiglu_vec4_ordered,
    nvq_dequantize,
    nvq_embedding,
    nvq_ffn_swiglu_down,
    nvq_gemm_f16,
    nvq_gemv_batch_vec8,
    nvq_gemv_m1_vec8,
    nvq_matmul,
    nvq_matmul_input_mul,
    nvq_matmul_multi2,
    nvq_matmul_swiglu,
    nvq_mmq,
    nvq_mmq_input_mul,
    to_gpu_nvq,
    to_gpu_nvq_exec,
)
from mfq.quantize.npq0_l import dequantize_npq0_l  # noqa: E402
from mfq.quantize.npq0_s import dequantize_npq0_s  # noqa: E402
from mfq.quantize.nvq1_l_quant import dequantize as dequantize_nvq1_l  # noqa: E402
from mfq.quantize.nvq1_l_quant import quantize as quantize_nvq1_l  # noqa: E402
from mfq.quantize.nvq1_s_quant import dequantize as dequantize_nvq1_s  # noqa: E402
from mfq.quantize.nvq_jsc import dequantize_nvq_jsc  # noqa: E402
from mfq.quantize.nvq_quant import dequantize as dequantize_nvq  # noqa: E402
from mfq.quantize.nvq_quant import quantize as quantize_nvq  # noqa: E402
from mfq.runtime.torch_linear import TorchLinearGroup, TorchSwiGLUFFN  # noqa: E402


def _quantized(format_id: int, *, custom: bool = False, index_parity: bool = False):
    if format_id == 8:
        return _nvq1_s_quantized(shifted=custom)
    if format_id == 9:
        return _npq0_s_quantized(shifted=custom)
    rng = np.random.default_rng(20260716 + format_id + 10 * custom + 20 * index_parity)
    weight = rng.normal(0.0, 0.05, size=(24, 80)).astype(np.float32)
    if format_id == 1:
        codebook = np.roll(IQ1S_TERNARY_2048, 7, axis=0).copy() if custom else None
        tensor = quantize_nvq1_l(
            weight,
            NVQ1_L_T8_S3,
            anchor_multipliers=(1.0,),
            refine_steps=1,
            group_chunk=64,
            codebook=codebook,
        )
        reference = dequantize_nvq1_l(tensor)
    else:
        base = NVQ2_E8 if format_id == 2 else NVQ3_D4
        spec = NvqSpec(
            base.codebook,
            groupsize=24,
            sub_bits=4,
            sign_mode="index_parity" if index_parity else "even",
        )
        builtin = E8_256 if format_id == 2 else D4_256
        codebook = np.roll(builtin, 11, axis=0).copy() if custom else None
        tensor = quantize_nvq(
            weight,
            spec,
            search_steps=3,
            group_chunk=128,
            codebook=codebook,
        )
        reference = dequantize_nvq(tensor)
    return tensor, np.ascontiguousarray(reference)


def _jsc_quantized(
    *,
    shifted: bool = False,
    spec: NvqSpec = NVQ2_E8,
    analytic_state: bool = False,
):
    rng = np.random.default_rng(20260731 + int(shifted))
    builtin = {
        "e8_256": E8_256,
        "e8_1024": E8_1024,
        "e8_4096": E8_4096,
        "d4_256": D4_256,
        "d4_512": D4_512,
        "d4_1024": D4_1024,
    }[spec.codebook]
    bank0 = (builtin.astype(np.int16) * 8).astype(np.int8)
    bank1 = np.clip(
        np.roll(bank0, 13 if shifted else 7, axis=0).astype(np.int16)
        + rng.integers(-2, 3, size=bank0.shape),
        1,
        127,
    ).astype(np.int8)
    scale_lut = np.arange(16, dtype=np.float32)
    if analytic_state:
        scale_lut = (np.arange(16, dtype=np.float32) // 2) + 1.0
    tensor = NvqJscTensor(
        shape=(24, 80),
        axis=0,
        neuron_len=80,
        neuron_scale=rng.uniform(2e-5, 8e-5, size=24).astype(np.float32),
        scale_lut=scale_lut,
        bank_for_state=(np.arange(16, dtype=np.uint8) & 1),
        state=rng.integers(0, 16, size=(24, 4), dtype=np.uint8),
        indices=rng.integers(
            0,
            spec.codebook_entries,
            size=(24, (80 + spec.vector_size - 1) // spec.vector_size),
            dtype=np.uint8 if spec.index_bits <= 8 else np.uint16,
        ),
        signs=rng.integers(0, 128, size=(24, 10), dtype=np.uint8),
        codebooks=np.stack((bank0, bank1)),
        base_spec=spec,
    )
    return tensor, np.ascontiguousarray(dequantize_nvq_jsc(tensor))


def _jsc_exec_gpu(tensor: NvqJscTensor):
    return to_gpu_nvq_exec(tensor)


def test_extended_jsc_cuda_dequant_gemv_and_mmq_match_reference():
    for spec, format_id in (
        (NVQ2_E8_1024, 13),
        (NVQ2_E8_4096, 14),
        (NVQ3_D4_1024, 15),
    ):
        tensor, reference = _jsc_quantized(spec=spec)
        gpu = to_gpu_nvq_exec(tensor)
        assert gpu["format"] == format_id
        expected_weight = torch.as_tensor(
            reference, device="cuda", dtype=torch.float16
        )
        torch.testing.assert_close(
            nvq_dequantize(gpu), expected_weight, atol=0, rtol=0
        )
        torch.manual_seed(20260730 + format_id)
        for m, kernel in (
            (1, nvq_gemv_m1_vec8),
            (4, nvq_gemv_batch_vec8),
            (32, nvq_mmq),
        ):
            x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
            actual = kernel(gpu, x)
            expected = x @ expected_weight.T
            relative = (
                (actual.float() - expected.float()).norm()
                / expected.float().norm()
            ).item()
            assert relative < 0.015


def _npq0_l_quantized(*, shifted: bool = False):
    rng = np.random.default_rng(20260722 + int(shifted))
    first = rng.integers(-48, 49, size=(8, 8, 4), dtype=np.int16).astype(np.int8)
    second = rng.integers(-48, 49, size=(8, 16, 4), dtype=np.int16).astype(np.int8)
    if shifted:
        first = np.roll(first, 1, axis=1).copy()
        second = np.roll(second, 3, axis=1).copy()
    tensor = Npq0LTensor(
        shape=(24, 80),
        axis=0,
        neuron_len=80,
        neuron_scale=rng.uniform(2e-4, 8e-4, size=24).astype(np.float16).astype(np.float32),
        scale_lut=np.linspace(0.125, 1.0, 8, dtype=np.float16).astype(np.float32),
        state=rng.integers(0, 8, size=(24, 4), dtype=np.uint8),
        indices=rng.integers(0, 128, size=(24, 10), dtype=np.uint8),
        first_codebooks=first,
        second_codebooks=second,
    )
    return tensor, np.ascontiguousarray(dequantize_npq0_l(tensor))


def _nvq1_s_quantized(
    *,
    shifted: bool = False,
    out: int = 24,
    neuron_len: int = 80,
):
    rng = np.random.default_rng(20260723 + int(shifted) + out + neuron_len)
    codebook = np.roll(NVQ1_S_SYNTHETIC_BANKS, 5 if shifted else 0, axis=1).copy()
    ng = (neuron_len + 23) // 24
    nvec = neuron_len // 8
    tensor = Nvq1STensor(
        spec=NVQ1_S,
        shape=(out, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=rng.uniform(2e-3, 8e-3, size=out).astype(np.float32),
        sub_scale=rng.integers(0, 16, size=(out, ng), dtype=np.uint8),
        indices=rng.integers(0, 512, size=(out, nvec), dtype=np.uint16),
        delta_sign=rng.integers(0, 2, size=(out, ng), dtype=np.uint8),
        codebook=codebook,
    )
    return tensor, np.ascontiguousarray(dequantize_nvq1_s(tensor))


def _npq0_s_quantized(
    *,
    shifted: bool = False,
    out: int = 24,
    neuron_len: int = 80,
):
    rng = np.random.default_rng(20260724 + int(shifted) + out + neuron_len)
    first_codebooks = rng.integers(
        -48,
        49,
        size=(4, 8, 4),
        dtype=np.int16,
    ).astype(np.int8)
    second_codebooks = rng.integers(
        -48,
        49,
        size=(4, 8, 4),
        dtype=np.int16,
    ).astype(np.int8)
    if shifted:
        first_codebooks = np.roll(first_codebooks, 3, axis=1).copy()
        second_codebooks = np.roll(second_codebooks, 5, axis=1).copy()
    ng = (neuron_len + 23) // 24
    nvec = neuron_len // 8
    tensor = Npq0STensor(
        shape=(out, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=rng.uniform(2e-4, 8e-4, size=out).astype(np.float16).astype(np.float32),
        scale_lut=np.linspace(0.25, 1.0, 4, dtype=np.float16).astype(np.float32),
        state=rng.integers(0, 4, size=(out, ng), dtype=np.uint8),
        indices=rng.integers(0, 64, size=(out, nvec), dtype=np.uint8),
        first_codebooks=first_codebooks,
        second_codebooks=second_codebooks,
    )
    return tensor, np.ascontiguousarray(dequantize_npq0_s(tensor))


def test_npq0_s_cuda_runtime_lut_is_the_cartesian_product():
    tensor, _ = _npq0_s_quantized()
    gpu = to_gpu_nvq(tensor)
    runtime = gpu["codebook"].cpu().numpy()
    assert runtime.shape == (64 + 4 * 64 * 8,)
    product = runtime[64:].reshape(4, 64, 8)
    for state in range(4):
        for second_index in range(8):
            for first_index in range(8):
                index = first_index | (second_index << 3)
                np.testing.assert_array_equal(
                    product[state, index, :4],
                    tensor.first_codebooks[state, first_index],
                )
                np.testing.assert_array_equal(
                    product[state, index, 4:],
                    tensor.second_codebooks[state, second_index],
                )


def test_npq0_l_cuda_dequant_gemv_mmq_and_fusion_match_reference():
    first, reference = _npq0_l_quantized()
    second, _ = _npq0_l_quantized(shifted=True)
    first_gpu = to_gpu_nvq(first)
    second_gpu = to_gpu_nvq(second)
    assert first_gpu["format"] == 7
    assert first_gpu["indices_packed"].numel() == (24 * 10 * 7 + 7) // 8
    assert first_gpu["aux_packed"].numel() == 0
    expected_weight = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(nvq_dequantize(first_gpu), expected_weight, atol=0, rtol=0)

    torch.manual_seed(20260722)
    for m in (1, 2, 8, 16):
        x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
        actual = nvq_matmul(first_gpu, x)
        expected = x @ expected_weight.T
        relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        assert relative < 0.012

    x16 = (torch.randn(16, 80, device="cuda") * 0.1).to(torch.float16)
    mmq = nvq_mmq(first_gpu, x16)
    mmq_expected = x16 @ expected_weight.T
    relative = ((mmq.float() - mmq_expected.float()).norm() / mmq_expected.float().norm()).item()
    assert relative < 0.012

    x1 = (torch.randn(1, 80, device="cuda") * 0.1).to(torch.float16)
    first_pair, second_pair = nvq_matmul_multi2(first_gpu, second_gpu, x1)
    assert torch.equal(first_pair, nvq_matmul(first_gpu, x1))
    assert torch.equal(second_pair, nvq_matmul(second_gpu, x1))
    fused = nvq_matmul_swiglu(first_gpu, second_gpu, x1)
    torch.testing.assert_close(
        fused,
        torch.nn.functional.silu(first_pair) * second_pair,
        atol=2e-3,
        rtol=3e-3,
    )


def test_nvq2j_cuda_dequant_gemv_mmq_and_fusion_match_reference():
    first, reference = _jsc_quantized()
    second, _ = _jsc_quantized(shifted=True)
    first_gpu = to_gpu_nvq(first)
    second_gpu = to_gpu_nvq(second)
    assert first_gpu["format"] == 5
    expected_weight = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(
        nvq_dequantize(first_gpu), expected_weight, atol=2e-4, rtol=0
    )

    torch.manual_seed(20260731)
    for m in (1, 4, 8):
        x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
        actual = nvq_matmul(first_gpu, x)
        expected = x @ expected_weight.T
        relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        assert relative < 0.012

    x16 = (torch.randn(16, 80, device="cuda") * 0.1).to(torch.float16)
    mmq = nvq_mmq(first_gpu, x16)
    mmq_expected = x16 @ expected_weight.T
    mmq_relative = ((mmq.float() - mmq_expected.float()).norm() / mmq_expected.float().norm()).item()
    assert mmq_relative < 0.012

    for m in (16, 128):
        x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
        f16_gemm = nvq_gemm_f16(first_gpu, x)
        expected = x @ expected_weight.T
        relative = ((f16_gemm.float() - expected.float()).norm() / expected.float().norm()).item()
        assert relative < 0.006

    x1 = (torch.randn(1, 80, device="cuda") * 0.1).to(torch.float16)
    first_pair, second_pair = nvq_matmul_multi2(first_gpu, second_gpu, x1)
    assert torch.equal(first_pair, nvq_matmul(first_gpu, x1))
    assert torch.equal(second_pair, nvq_matmul(second_gpu, x1))
    fused = nvq_matmul_swiglu(first_gpu, second_gpu, x1)
    expected_fused = torch.nn.functional.silu(first_pair) * second_pair
    torch.testing.assert_close(fused, expected_fused, atol=2e-3, rtol=3e-3)

    exec_gpu = _jsc_exec_gpu(first)
    torch.testing.assert_close(
        nvq_dequantize(exec_gpu), expected_weight, atol=2e-4, rtol=0
    )
    exec_actual = nvq_matmul(exec_gpu, x1)
    torch.testing.assert_close(exec_actual, first_pair, atol=0, rtol=0)
    x12 = (torch.randn(12, 80, device="cuda") * 0.1).to(torch.float16)
    torch.testing.assert_close(
        nvq_gemv_batch_vec8(exec_gpu, x12, 4),
        nvq_gemv_batch_vec8(first_gpu, x12, 4),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        nvq_mmq(exec_gpu, x16), nvq_mmq(first_gpu, x16), atol=0, rtol=0
    )
    torch.testing.assert_close(
        nvq_gemm_f16(exec_gpu, x16), nvq_gemm_f16(first_gpu, x16), atol=0, rtol=0
    )


def test_nvq3j_cuda_dequant_gemv_mmq_and_fusion_match_reference():
    first, reference = _jsc_quantized(spec=NVQ3_D4)
    second, _ = _jsc_quantized(shifted=True, spec=NVQ3_D4)
    first_gpu = to_gpu_nvq(first)
    second_gpu = to_gpu_nvq(second)
    assert first_gpu["format"] == 10
    expected_weight = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(nvq_dequantize(first_gpu), expected_weight, atol=2e-4, rtol=0)

    torch.manual_seed(20260801)
    for m in (1, 4, 8, 16):
        x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
        actual = nvq_matmul(first_gpu, x)
        expected = x @ expected_weight.T
        relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        assert relative < 0.012

    x16 = (torch.randn(16, 80, device="cuda") * 0.1).to(torch.float16)
    mmq = nvq_mmq(first_gpu, x16)
    mmq_expected = x16 @ expected_weight.T
    relative = ((mmq.float() - mmq_expected.float()).norm() / mmq_expected.float().norm()).item()
    assert relative < 0.012

    x1 = (torch.randn(1, 80, device="cuda") * 0.1).to(torch.float16)
    first_pair, second_pair = nvq_matmul_multi2(first_gpu, second_gpu, x1)
    torch.testing.assert_close(first_pair, nvq_matmul(first_gpu, x1), atol=0, rtol=0)
    torch.testing.assert_close(second_pair, nvq_matmul(second_gpu, x1), atol=0, rtol=0)


def test_nvq3j_512_cuda_dequant_gemv_mmq_and_fusion_match_reference():
    first, reference = _jsc_quantized(spec=NVQ3_D4_512)
    second, _ = _jsc_quantized(shifted=True, spec=NVQ3_D4_512)
    first_gpu = to_gpu_nvq(first)
    second_gpu = to_gpu_nvq(second)
    assert first_gpu["format"] == 12
    assert first_gpu["indices_packed"].numel() == (24 * 20 * 9 + 7) // 8
    expected_weight = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(
        nvq_dequantize(first_gpu), expected_weight, atol=2e-4, rtol=0
    )

    torch.manual_seed(20260804)
    for m in (1, 4, 8, 16):
        x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
        actual = nvq_matmul(first_gpu, x)
        expected = x @ expected_weight.T
        relative = (
            (actual.float() - expected.float()).norm()
            / expected.float().norm()
        ).item()
        assert relative < 0.012

    x16 = (torch.randn(16, 80, device="cuda") * 0.1).to(torch.float16)
    mmq = nvq_mmq(first_gpu, x16)
    mmq_expected = x16 @ expected_weight.T
    relative = (
        (mmq.float() - mmq_expected.float()).norm()
        / mmq_expected.float().norm()
    ).item()
    assert relative < 0.012

    x1 = (torch.randn(1, 80, device="cuda") * 0.1).to(torch.float16)
    first_pair, second_pair = nvq_matmul_multi2(first_gpu, second_gpu, x1)
    torch.testing.assert_close(
        first_pair, nvq_matmul(first_gpu, x1), atol=0, rtol=0
    )
    torch.testing.assert_close(
        second_pair, nvq_matmul(second_gpu, x1), atol=0, rtol=0
    )


def test_nvq3j_analytic_state_cuda_matches_reference():
    tensor, reference = _jsc_quantized(spec=NVQ3_D4, analytic_state=True)
    assert pack_jsc_metadata(tensor)[3] == 1
    gpu = to_gpu_nvq(tensor)
    assert gpu["format"] == 11
    expected_weight = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(nvq_dequantize(gpu), expected_weight, atol=2e-4, rtol=0)
    torch.manual_seed(20260802)
    for m in (1, 8, 16):
        x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
        actual = nvq_matmul(gpu, x)
        expected = x @ expected_weight.T
        relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        assert relative < 0.012


@pytest.mark.parametrize("format_id", [1, 2, 3, 8, 9])
def test_nvq_cuda_dequant_matches_packed_reference(format_id):
    tensor, reference = _quantized(format_id)
    gpu = to_gpu_nvq(tensor)
    actual = nvq_dequantize(gpu)
    expected = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)


@pytest.mark.parametrize("format_id", [1, 2, 3, 8, 9])
def test_nvq_cuda_embedding_matches_dequant_rows(format_id):
    tensor, reference = _quantized(format_id)
    gpu = to_gpu_nvq(tensor)
    ids = torch.tensor([[0, 5], [17, 23]], device="cuda", dtype=torch.int64)
    actual = nvq_embedding(gpu, ids)
    expected = torch.as_tensor(reference[[0, 5, 17, 23]], device="cuda", dtype=torch.float16).reshape(2, 2, 80)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)


@pytest.mark.parametrize("format_id", [1, 2, 3, 8, 9])
def test_nvq_cuda_direct_gemv_matches_dequant_matmul(format_id):
    tensor, reference = _quantized(format_id)
    gpu = to_gpu_nvq(tensor)
    weight = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.manual_seed(20260720 + format_id)
    for m in (1, 3, 8, 9, 12, 13, 16):
        x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
        actual = nvq_matmul(gpu, x)
        expected = x @ weight.T
        relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        assert relative < 0.012, f"NVQ{format_id} M={m} relative error {relative}"


def test_nvq_matmul_path_schedule_matches_measured_regions():
    small_e8 = {"format": 6, "storage_format": 5, "out": 1024, "neuron_len": 4096}
    medium_e8 = {"format": 6, "storage_format": 5, "out": 4096, "neuron_len": 4096}
    attn_q = {"format": 6, "storage_format": 5, "out": 8192, "neuron_len": 4096}
    ffn_up = {"format": 6, "storage_format": 5, "out": 12288, "neuron_len": 4096}
    ffn_down = {"format": 3, "storage_format": 3, "out": 4096, "neuron_len": 12288}
    extended_e8 = {"format": 14, "out": 8192, "neuron_len": 4096}
    extended_d4 = {"format": 15, "out": 4096, "neuron_len": 12288}
    npq0_l_qkv = {"format": 7, "out": 8192, "neuron_len": 4096}
    npq0_l_medium = {"format": 7, "out": 4096, "neuron_len": 4096}
    npq0_s_small = {"format": 9, "out": 1024, "neuron_len": 4096}
    npq0_s_square = {"format": 9, "out": 4096, "neuron_len": 4096}
    npq0_s_wide = {"format": 9, "out": 11008, "neuron_len": 4096}
    npq0_s_down = {"format": 9, "out": 4096, "neuron_len": 11008}
    nvq1_s_down = {"format": 8, "out": 4096, "neuron_len": 11008}

    assert _select_matmul_path(small_e8, 13) == "gemv"
    assert _select_matmul_path(small_e8, 14) == "dequant_gemm"
    assert _select_matmul_path(medium_e8, 14) == "gemv"
    assert _select_matmul_path(medium_e8, 16) == "gemv"
    assert _select_matmul_path(attn_q, 15) == "mmq"
    assert _select_matmul_path(attn_q, 16) == "mmq"
    assert _select_matmul_path(attn_q, 32) == "mmq"
    assert _select_matmul_path(attn_q, 24) == "dequant_gemm"
    assert _select_matmul_path(ffn_up, 17) == "online_f16"
    assert _select_matmul_path(ffn_up, 31) == "online_f16"
    assert _select_matmul_path(ffn_up, 32) == "mmq"
    assert _select_matmul_path(ffn_up, 33) == "online_f16"
    assert _select_matmul_path(ffn_up, 47) == "online_f16"
    assert _select_matmul_path(ffn_up, 48) == "mmq"
    assert _select_matmul_path(ffn_up, 49) == "dequant_gemm"
    assert _select_matmul_path(ffn_down, 16) == "mmq"
    assert _select_matmul_path(ffn_down, 32) == "dequant_gemm"
    assert _select_matmul_path(extended_e8, 15) == "mmq"
    assert _select_matmul_path(extended_e8, 32) == "mmq"
    assert _select_matmul_path(extended_d4, 16) == "mmq"
    assert _select_matmul_path(npq0_l_qkv, 64) == "mmq"
    assert _select_matmul_path(npq0_l_medium, 64) == "dequant_gemm"
    assert _select_matmul_path(npq0_s_small, 16) == "gemv"
    assert _select_matmul_path(npq0_s_small, 17) == "dequant_gemm"
    assert _select_matmul_path(npq0_s_square, 15) == "gemv"
    assert _select_matmul_path(npq0_s_square, 16) == "mmq"
    assert _select_matmul_path(npq0_s_square, 24) == "mmq"
    assert _select_matmul_path(npq0_s_square, 32) == "mmq"
    assert _select_matmul_path(npq0_s_square, 40) == "dequant_gemm"
    assert _select_matmul_path(npq0_s_wide, 13) == "mmq"
    assert _select_matmul_path(npq0_s_wide, 17) == "online_f16"
    assert _select_matmul_path(npq0_s_wide, 32) == "mmq"
    assert _select_matmul_path(npq0_s_wide, 40) == "online_f16"
    assert _select_matmul_path(npq0_s_wide, 48) == "mmq"
    assert _select_matmul_path(npq0_s_wide, 64) == "online_f16"
    assert _select_matmul_path(npq0_s_down, 14) == "mmq"
    assert _select_matmul_path(npq0_s_down, 24) == "dequant_gemm"
    assert _select_matmul_path(npq0_s_down, 32) == "mmq"
    assert _select_matmul_path(nvq1_s_down, 13) == "gemv"
    assert _select_matmul_path(nvq1_s_down, 14) == "dequant_gemm"
    assert _select_matmul_path(nvq1_s_down, 16) == "dequant_gemm"


@pytest.mark.parametrize("nwarps", [1, 2, 4, 8])
@pytest.mark.parametrize("format_id", [1, 2, 3, 8, 9])
def test_nvq_cuda_vec8_m1_matches_dequant_matmul(format_id, nwarps):
    tensor, reference = _quantized(format_id)
    gpu = to_gpu_nvq(tensor)
    weight = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.manual_seed(20261100 + 10 * format_id + nwarps)
    x = (torch.randn(1, 80, device="cuda") * 0.1).to(torch.float16)
    actual = nvq_gemv_m1_vec8(gpu, x, nwarps)
    expected = x @ weight.T
    relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative < 0.012, f"NVQ{format_id} nwarps={nwarps} relative error {relative}"


@pytest.mark.parametrize("nwarps", [2, 4, 8])
@pytest.mark.parametrize("m", [2, 4, 8, 9, 12, 16])
@pytest.mark.parametrize("format_id", [1, 2, 3, 8, 9])
def test_nvq_cuda_vec8_batch_matches_dequant_matmul(format_id, m, nwarps):
    tensor, reference = _quantized(format_id)
    gpu = to_gpu_nvq(tensor)
    weight = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.manual_seed(20261200 + 100 * format_id + 10 * m + nwarps)
    x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
    actual = nvq_gemv_batch_vec8(gpu, x, nwarps)
    expected = x @ weight.T
    relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative < 0.012, f"NVQ{format_id} M={m} nwarps={nwarps} relative error {relative}"


@pytest.mark.parametrize("format_id", [1, 2, 3, 8, 9])
@pytest.mark.parametrize("m", [1, 4])
def test_nvq_cuda_multi2_matches_independent_projections(format_id, m):
    first, first_reference = _quantized(format_id)
    second, second_reference = _quantized(format_id, custom=True)
    first_gpu = to_gpu_nvq(first)
    second_gpu = to_gpu_nvq(second)
    torch.manual_seed(20261300 + 10 * format_id + m)
    x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
    first_actual, second_actual = nvq_matmul_multi2(first_gpu, second_gpu, x)
    first_independent = nvq_matmul(first_gpu, x)
    second_independent = nvq_matmul(second_gpu, x)
    assert torch.equal(first_actual, first_independent)
    assert torch.equal(second_actual, second_independent)
    runtime_first, runtime_second = TorchLinearGroup((first, second))(x)
    assert torch.equal(runtime_first, first_independent)
    assert torch.equal(runtime_second, second_independent)
    first_expected = x @ torch.as_tensor(first_reference, device="cuda", dtype=torch.float16).T
    second_expected = x @ torch.as_tensor(second_reference, device="cuda", dtype=torch.float16).T
    first_relative = ((first_actual.float() - first_expected.float()).norm() / first_expected.float().norm()).item()
    second_relative = ((second_actual.float() - second_expected.float()).norm() / second_expected.float().norm()).item()
    assert first_relative < 0.012
    assert second_relative < 0.012


@pytest.mark.parametrize("format_id", [1, 2, 3, 8, 9])
def test_nvq_cuda_swiglu_pair_matches_materialized_path(format_id):
    gate, _ = _quantized(format_id)
    up, _ = _quantized(format_id, custom=True)
    gate_gpu = to_gpu_nvq(gate)
    up_gpu = to_gpu_nvq(up)
    torch.manual_seed(20261400 + format_id)
    x = (torch.randn(1, 80, device="cuda") * 0.1).to(torch.float16)
    actual = nvq_matmul_swiglu(gate_gpu, up_gpu, x)
    gate_value = nvq_matmul(gate_gpu, x)
    up_value = nvq_matmul(up_gpu, x)
    expected = torch.nn.functional.silu(gate_value) * up_value
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-3)


def test_nvq2_cuda_swiglu_vec4_ordered_matches_standard_pair():
    rng = np.random.default_rng(20261420)
    spec = NvqSpec("e8_256", groupsize=24, sub_bits=4, sign_mode="even")
    gate = quantize_nvq(
        rng.normal(0.0, 0.05, size=(48, 96)).astype(np.float32),
        spec, search_steps=3, group_chunk=128,
    )
    up = quantize_nvq(
        rng.normal(0.0, 0.05, size=(48, 96)).astype(np.float32),
        spec, search_steps=3, group_chunk=128,
    )
    gate_gpu = to_gpu_nvq(gate)
    up_gpu = to_gpu_nvq(up)
    torch.manual_seed(20261421)
    x = (torch.randn(1, 96, device="cuda") * 0.1).to(torch.float16)
    ordered = nvq2_matmul_swiglu_vec4_ordered(gate_gpu, up_gpu, x)
    expected = nvq_matmul_swiglu(gate_gpu, up_gpu, x)
    torch.testing.assert_close(ordered, expected, atol=0, rtol=0)


def test_nvq_cuda_fused_ffn_matches_existing_gated_down_path():
    rng = np.random.default_rng(20261500)
    spec = NvqSpec("e8_256", groupsize=24, sub_bits=4, sign_mode="even")
    gate = quantize_nvq(
        rng.normal(0.0, 0.05, size=(48, 96)).astype(np.float32),
        spec, search_steps=3, group_chunk=128,
    )
    up = quantize_nvq(
        rng.normal(0.0, 0.05, size=(48, 96)).astype(np.float32),
        spec, search_steps=3, group_chunk=128,
        codebook=np.roll(E8_256, 13, axis=0).copy(),
    )
    down = quantize_nvq(
        rng.normal(0.0, 0.05, size=(24, 48)).astype(np.float32),
        spec, search_steps=3, group_chunk=128,
    )
    gate_gpu = to_gpu_nvq(gate)
    up_gpu = to_gpu_nvq(up)
    down_gpu = to_gpu_nvq(down)
    torch.manual_seed(20261501)
    x = (torch.randn(1, 96, device="cuda") * 0.1).to(torch.float16)
    actual = nvq_ffn_swiglu_down(gate_gpu, up_gpu, down_gpu, x)
    gate_value = nvq_matmul(gate_gpu, x)
    up_value = nvq_matmul(up_gpu, x)
    expected = nvq_matmul_input_mul(down_gpu, up_value, gate_value, "silu")
    runtime_actual = TorchSwiGLUFFN.from_tensors(gate, up, down)(x)
    assert torch.equal(runtime_actual, actual)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-3)


@pytest.mark.parametrize("format_id", [8, 9])
def test_small_vq_cuda_fused_ffn_matches_existing_gated_down_path(format_id):
    factory = _nvq1_s_quantized if format_id == 8 else _npq0_s_quantized
    gate, _ = factory(out=48, neuron_len=96)
    up, _ = factory(shifted=True, out=48, neuron_len=96)
    down, _ = factory(out=24, neuron_len=48)
    gate_gpu = to_gpu_nvq(gate)
    up_gpu = to_gpu_nvq(up)
    down_gpu = to_gpu_nvq(down)
    torch.manual_seed(20260725 + format_id)
    x = (torch.randn(1, 96, device="cuda") * 0.1).to(torch.float16)
    actual = nvq_ffn_swiglu_down(gate_gpu, up_gpu, down_gpu, x)
    gate_value = nvq_matmul(gate_gpu, x)
    up_value = nvq_matmul(up_gpu, x)
    expected = nvq_matmul_input_mul(down_gpu, up_value, gate_value, "silu")
    runtime_actual = TorchSwiGLUFFN.from_tensors(gate, up, down)(x)
    assert torch.equal(runtime_actual, actual)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-3)


@pytest.mark.parametrize("format_id", [1, 2, 3, 8, 9])
@pytest.mark.parametrize("m", [4, 8, 16, 24, 32, 48, 64])
def test_nvq_cuda_mma_mmq_matches_dequant_matmul(format_id, m):
    tensor, reference = _quantized(format_id)
    gpu = to_gpu_nvq(tensor)
    weight = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.manual_seed(20260800 + 10 * format_id + m)
    x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
    actual = nvq_mmq(gpu, x)
    expected = x @ weight.T
    relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative < 0.012, f"NVQ{format_id} M={m} relative error {relative}"


@pytest.mark.parametrize("format_id", [1, 2, 3, 8, 9])
@pytest.mark.parametrize("m", [16, 24, 32, 48, 64, 80, 96, 112, 128, 192, 256])
def test_nvq_cuda_online_f16_gemm_matches_dequant_matmul(format_id, m):
    tensor, reference = _quantized(format_id)
    gpu = to_gpu_nvq(tensor)
    weight = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.manual_seed(20261600 + 100 * format_id + m)
    x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
    actual = nvq_gemm_f16(gpu, x)
    expected = x @ weight.T
    relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative < 0.006, f"NVQ{format_id} M={m} relative error {relative}"


def test_nvq2_cuda_online_f16_gemm_exact_tiles_match_dequant_matmul():
    rng = np.random.default_rng(20261680)
    weight = rng.normal(0.0, 0.05, size=(64, 96)).astype(np.float32)
    tensor = quantize_nvq(
        weight,
        NvqSpec("e8_256", groupsize=24, sub_bits=4, sign_mode="even"),
        search_steps=3,
        group_chunk=128,
    )
    gpu = to_gpu_nvq(tensor)
    expected_weight = torch.as_tensor(
        dequantize_nvq(tensor), device="cuda", dtype=torch.float16
    )
    torch.manual_seed(20261681)
    for m in (16, 32, 48, 64, 80, 96, 112, 128, 256):
        x = (torch.randn(m, 96, device="cuda") * 0.1).to(torch.float16)
        actual = nvq_gemm_f16(gpu, x)
        expected = x @ expected_weight.T
        relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        assert relative < 0.006, f"NVQ2 exact tile M={m} relative error {relative}"
        if m <= 64:
            mmq = nvq_mmq(gpu, x)
            mmq_relative = ((mmq.float() - expected.float()).norm() / expected.float().norm()).item()
            assert mmq_relative < 0.012, f"NVQ2 exact MMQ M={m} relative error {mmq_relative}"


@pytest.mark.parametrize("index_parity", [False, True])
def test_nvq2_cuda_exec_layout_all_m_kernels_match_compact_layout(index_parity):
    tensor, _ = _quantized(2, index_parity=index_parity)
    compact = to_gpu_nvq(tensor)
    execution = to_gpu_nvq_exec(tensor)
    assert execution["format"] == 4
    torch.testing.assert_close(
        nvq_dequantize(execution), nvq_dequantize(compact), atol=0, rtol=0
    )
    torch.manual_seed(20261690)
    for m in (2, 12, 16):
        x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
        actual = nvq_gemv_batch_vec8(execution, x, 4)
        expected = nvq_gemv_batch_vec8(compact, x, 4)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for m in (16, 48, 64):
        x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
        compact_mmq = nvq_mmq(compact, x)
        execution_mmq = nvq_mmq(execution, x)
        torch.testing.assert_close(execution_mmq, compact_mmq, atol=0, rtol=0)
        compact_f16 = nvq_gemm_f16(compact, x)
        execution_f16 = nvq_gemm_f16(execution, x)
        torch.testing.assert_close(execution_f16, compact_f16, atol=0, rtol=0)


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
@pytest.mark.parametrize("format_id", [1, 2, 3, 8, 9])
def test_nvq_cuda_gated_direct_matches_materialized_reference(format_id, activation):
    tensor, reference = _quantized(format_id)
    gpu = to_gpu_nvq(tensor)
    weight = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.manual_seed(20260900 + 10 * format_id + (activation == "silu"))
    x = (torch.randn(2, 80, device="cuda") * 0.1).to(torch.float16)
    gate = torch.randn(2, 80, device="cuda", dtype=torch.float16)
    actual = nvq_matmul_input_mul(gpu, x, gate, activation)
    multiplier = torch.sigmoid(gate) if activation == "sigmoid" else torch.nn.functional.silu(gate)
    expected = (x * multiplier) @ weight.T
    relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative < 0.015, f"NVQ{format_id} {activation} relative error {relative}"


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
@pytest.mark.parametrize("format_id", [1, 2, 3, 8, 9])
@pytest.mark.parametrize("m", [4, 16])
def test_nvq_cuda_gated_mma_matches_materialized_reference(format_id, activation, m):
    tensor, reference = _quantized(format_id)
    gpu = to_gpu_nvq(tensor)
    weight = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.manual_seed(20261000 + 100 * format_id + 10 * m + (activation == "silu"))
    x = (torch.randn(m, 80, device="cuda") * 0.1).to(torch.float16)
    gate = torch.randn(m, 80, device="cuda", dtype=torch.float16)
    actual = nvq_mmq_input_mul(gpu, x, gate, activation)
    multiplier = torch.sigmoid(gate) if activation == "sigmoid" else torch.nn.functional.silu(gate)
    expected = (x * multiplier) @ weight.T
    relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative < 0.015, f"NVQ{format_id} M={m} {activation} relative error {relative}"


@pytest.mark.parametrize("format_id", [1, 2, 3, 8, 9])
def test_nvq_cuda_custom_codebook_is_used(format_id):
    tensor, reference = _quantized(format_id, custom=True)
    gpu = to_gpu_nvq(tensor)
    actual = nvq_dequantize(gpu)
    expected = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)


def test_nvq2_cuda_index_parity_signs_match_reference():
    tensor, reference = _quantized(2, index_parity=True)
    gpu = to_gpu_nvq(tensor)
    actual = nvq_dequantize(gpu)
    expected = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)


@pytest.mark.parametrize("format_id", [1, 2, 3, 8, 9])
def test_nvq_cuda_prefill_dequant_cublas_matches_reference(format_id):
    tensor, reference = _quantized(format_id)
    gpu = to_gpu_nvq(tensor)
    torch.manual_seed(20260730 + format_id)
    x = (torch.randn(33, 80, device="cuda") * 0.1).to(torch.float16)
    actual = nvq_matmul(gpu, x)
    weight = torch.as_tensor(reference, device="cuda", dtype=torch.float16)
    expected = x @ weight.T
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-3)
