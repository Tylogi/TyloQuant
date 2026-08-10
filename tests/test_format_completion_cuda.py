from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from mfq.formats.moe import NintMoePool, NintMoeTensor  # noqa: E402
from mfq.formats.mx import MxTensor  # noqa: E402
from mfq.formats.nint8_zero import Nint8ZeroTensor  # noqa: E402
from mfq.formats.tpq import (  # noqa: E402
    TPQ_V,
    TPQ_VV,
    TPQ_W,
    TPQ_X,
    TpqPqSpec,
    TpqPqTensor,
)
from mfq.kernels.cuda._ext import ext  # noqa: E402
from mfq.kernels.cuda.moe import MoeRoutePlan, grouped_matmul, to_gpu  # noqa: E402
from mfq.kernels.cuda.mx_matmul import (  # noqa: E402
    mx_dequantize,
    mx_embedding,
    mx_matmul,
    to_gpu_mx,
)
from mfq.kernels.cuda.nint8_zero_matmul import (  # noqa: E402
    nint8_zero_dequantize,
    nint8_zero_matmul,
    to_gpu_nint8_zero,
)
from mfq.kernels.cuda.tpq_matmul import (  # noqa: E402
    to_gpu_tpq,
    tpq_dequantize,
    tpq_embedding,
    tpq_matmul,
)
from mfq.quantize.tpq import (  # noqa: E402
    dequantize_tpq_int4,
    dequantize_tpq_pq,
    quantize_tpq_int4,
)


def _dequant_mxfp4(tensor: MxTensor) -> np.ndarray:
    rows, width = tensor.shape
    packed = np.stack((tensor.values & 15, tensor.values >> 4), axis=-1).reshape(rows, width)
    magnitude = np.asarray(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=np.float32,
    )
    dense = magnitude[packed & 7]
    dense = np.where((packed & 8) == 0, dense, -dense)
    dense *= np.repeat(np.exp2(tensor.scales.astype(np.int16) - 127), 32, axis=1)
    return dense


def _mxfp4() -> tuple[MxTensor, np.ndarray]:
    rng = np.random.default_rng(604)
    rows, width = 11, 96
    tensor = MxTensor(
        "MXFP4",
        (rows, width),
        rng.integers(0, 256, (rows, width // 2), dtype=np.uint8),
        rng.integers(124, 130, (rows, width // 32), dtype=np.uint8),
    )
    return tensor, _dequant_mxfp4(tensor)


def _mxfp8() -> tuple[MxTensor, np.ndarray]:
    rng = np.random.default_rng(608)
    rows, width = 11, 128
    values = rng.integers(0, 255, (rows, width), dtype=np.uint8)
    values[(values & 127) == 127] = 126
    scales = rng.integers(124, 130, (1, 1), dtype=np.uint8)
    unsigned = values.astype(np.uint16)
    exponent = (unsigned >> 3) & 15
    mantissa = unsigned & 7
    subnormal = np.ldexp(mantissa.astype(np.float32) / 8.0, -6)
    normal = np.ldexp(
        1.0 + mantissa.astype(np.float32) / 8.0,
        exponent.astype(np.int16) - 7,
    )
    dense = np.where((unsigned & 128) == 0, normal, -normal)
    dense = np.where(exponent == 0, np.sign(dense) * subnormal, dense)
    dense *= np.exp2(scales.astype(np.int16)[0, 0] - 127)
    return MxTensor("MXFP8", (rows, width), values, scales), dense


def _tpq_pq_tensor(
    spec: TpqPqSpec,
    *,
    rows: int = 9,
    width: int = 128,
    seed: int = 613,
) -> TpqPqTensor:
    rng = np.random.default_rng(seed)
    index_dtype = np.uint8 if spec.index_bits == 8 else np.uint16
    return TpqPqTensor(
        spec,
        (rows, width),
        0,
        width,
        rng.integers(
            0,
            spec.codebook_entries,
            (rows, width // spec.vector_size),
            dtype=index_dtype,
        ),
        rng.normal(
            0.0,
            0.2,
            (spec.codebook_entries, spec.vector_size),
        ).astype(np.float32),
    )


_TPQ_FAMILIES = (
    pytest.param("int4", id="int4"),
    pytest.param(TPQ_X, id="x-p8"),
    pytest.param(TPQ_W, id="w-p16"),
    pytest.param(TPQ_V, id="v-p8"),
    pytest.param(TPQ_VV, id="vv-p16"),
    pytest.param(TpqPqSpec("p", 8, 300, 12), id="p-p12"),
    pytest.param(TpqPqSpec("p", 8, 300, 14), id="p-p14"),
)


@pytest.mark.parametrize("activation_rows", [1, 16, 65])
def test_mxfp4_native_matmul_matches_packed_reference(activation_rows: int):
    tensor, dense = _mxfp4()
    weight = to_gpu_mx(tensor)
    source = torch.randn(
        activation_rows,
        tensor.shape[1],
        device="cuda",
        dtype=torch.float16,
    )
    actual = mx_matmul(weight, source).float()
    expected = source.float() @ torch.as_tensor(dense, device="cuda").T
    torch.testing.assert_close(actual, expected, rtol=0.006, atol=0.02)
    torch.testing.assert_close(
        mx_dequantize(weight).float(),
        torch.as_tensor(dense, device="cuda"),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("activation_rows", [1, 16, 65])
def test_mxfp8_matmul_matches_packed_reference(activation_rows: int):
    tensor, dense = _mxfp8()
    weight = to_gpu_mx(tensor)
    source = torch.randn(
        activation_rows,
        tensor.shape[1],
        device="cuda",
        dtype=torch.float16,
    )
    actual = mx_matmul(weight, source).float()
    expected = source.float() @ torch.as_tensor(dense, device="cuda").T
    torch.testing.assert_close(actual, expected, rtol=0.006, atol=0.03)
    torch.testing.assert_close(
        mx_dequantize(weight).float(),
        torch.as_tensor(dense, device="cuda"),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("activation_rows", [16, 65])
def test_mxfp8_native_fp32_output_matches_packed_reference(activation_rows: int):
    tensor, dense = _mxfp8()
    weight = to_gpu_mx(tensor)
    source = torch.randn(
        activation_rows,
        tensor.shape[1],
        device="cuda",
        dtype=torch.float16,
    )
    actual = ext().mxfp8_gemm_f32_cuda(weight["values"], weight["scales"], source)
    expected = source.float() @ torch.as_tensor(dense, device="cuda").T
    torch.testing.assert_close(actual, expected, rtol=0.006, atol=0.03)


@pytest.mark.parametrize("activation_rows", [1, 16, 65])
def test_nint8_zero_native_matmul_matches_packed_reference(
    activation_rows: int,
):
    rng = np.random.default_rng(608)
    rows, width = 13, 96
    tensor = Nint8ZeroTensor(
        shape=(rows, width),
        axis=0,
        scale=rng.uniform(0.001, 0.02, (rows, width // 32)).astype(np.float16),
        q=rng.integers(-127, 128, (rows, width // 32, 32)).astype(np.int8),
        neuron_len=width,
    )
    weight = to_gpu_nint8_zero(tensor)
    source = torch.randn(activation_rows, width, device="cuda", dtype=torch.float16)
    dense = nint8_zero_dequantize(weight)
    actual = nint8_zero_matmul(weight, source).float()
    if activation_rows <= 64:
        grouped = source.float().reshape(activation_rows, width // 32, 32)
        activation_scale = grouped.abs().amax(dim=-1) / 127.0
        activation_scale = torch.where(
            activation_scale > 0,
            activation_scale,
            torch.ones_like(activation_scale),
        )
        quantized = torch.round(grouped / activation_scale[..., None]).clamp(-127, 127)
        expected = torch.einsum(
            "mgk,ngk,mg,ng->mn",
            quantized,
            torch.as_tensor(tensor.q, device="cuda").float(),
            activation_scale,
            torch.as_tensor(tensor.scale, device="cuda").float(),
        )
    else:
        expected = source.float() @ dense.float().T
    torch.testing.assert_close(actual, expected.half().float(), rtol=0.004, atol=0.02)


@pytest.mark.parametrize("activation_rows", [1, 16, 65])
@pytest.mark.parametrize("family", _TPQ_FAMILIES)
def test_tpq_native_matmul_matches_packed_reference(
    activation_rows: int,
    family: str | TpqPqSpec,
):
    rows, width = 9, 128
    if family == "int4":
        rng = np.random.default_rng(612)
        tensor = quantize_tpq_int4(rng.normal(0.0, 0.2, (rows, width)).astype(np.float32))
        dense = dequantize_tpq_int4(tensor)
    else:
        tensor = _tpq_pq_tensor(family)
        dense = dequantize_tpq_pq(tensor)
    weight = to_gpu_tpq(tensor)
    source = torch.randn(activation_rows, width, device="cuda", dtype=torch.float16)
    actual = tpq_matmul(weight, source).float()
    expected = source.float() @ torch.as_tensor(dense, device="cuda").float().T
    torch.testing.assert_close(actual, expected, rtol=0.006, atol=0.03)
    torch.testing.assert_close(
        tpq_dequantize(weight).float(),
        torch.as_tensor(dense, device="cuda").half().float(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("family", ["mxfp4", "mxfp8"])
def test_mx_embedding_decodes_only_selected_rows(family: str):
    tensor, dense = _mxfp4() if family == "mxfp4" else _mxfp8()
    weight = to_gpu_mx(tensor)
    ids = torch.tensor([[0, tensor.shape[0] - 1], [2, -1]], device="cuda")
    expected = np.zeros((*ids.shape, tensor.shape[1]), dtype=np.float16)
    expected[0, 0] = dense[0].astype(np.float16)
    expected[0, 1] = dense[-1].astype(np.float16)
    expected[1, 0] = dense[2].astype(np.float16)
    torch.testing.assert_close(
        mx_embedding(weight, ids),
        torch.as_tensor(expected, device="cuda"),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "family",
    (
        pytest.param("int4", id="int4"),
        pytest.param(TPQ_X, id="x-p8"),
        pytest.param(TPQ_W, id="w-p16"),
        pytest.param(TpqPqSpec("p", 8, 300, 12), id="p-p12"),
    ),
)
def test_tpq_embedding_decodes_only_selected_rows(family: str | TpqPqSpec):
    rows, width = 9, 128
    if family == "int4":
        rng = np.random.default_rng(620)
        tensor = quantize_tpq_int4(rng.normal(0.0, 0.2, (rows, width)).astype(np.float32))
        dense = dequantize_tpq_int4(tensor)
    else:
        tensor = _tpq_pq_tensor(family, seed=621)
        dense = dequantize_tpq_pq(tensor)
    weight = to_gpu_tpq(tensor)
    ids = torch.tensor([[0, rows - 1], [2, -1]], device="cuda")
    expected = np.zeros((*ids.shape, width), dtype=np.float16)
    expected[0, 0] = dense[0].astype(np.float16)
    expected[0, 1] = dense[-1].astype(np.float16)
    expected[1, 0] = dense[2].astype(np.float16)
    torch.testing.assert_close(
        tpq_embedding(weight, ids),
        torch.as_tensor(expected, device="cuda"),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("tokens", [2, 13])
def test_mixed_moe_routes_nint8_mxfp4_and_tpq(tokens: int):
    rng = np.random.default_rng(631)
    experts, out, width = 3, 8, 128
    q8 = Nint8ZeroTensor(
        shape=(out, width),
        axis=0,
        scale=rng.uniform(0.001, 0.02, (out, width // 32)).astype(np.float16),
        q=rng.integers(-127, 128, (out, width // 32, 32)).astype(np.int8),
        neuron_len=width,
    )
    mx_values = rng.integers(0, 256, (out, width // 2), dtype=np.uint8)
    mx_scales = rng.integers(124, 130, (out, width // 32), dtype=np.uint8)
    mx = MxTensor("MXFP4", (out, width), mx_values, mx_scales)
    mx_dense = _dequant_mxfp4(mx)
    tpq = TpqPqTensor(
        TPQ_X,
        (out, width),
        0,
        width,
        rng.integers(
            0,
            TPQ_X.codebook_entries,
            (out, width // TPQ_X.vector_size),
            dtype=np.uint8,
        ),
        rng.normal(
            0.0,
            0.2,
            (TPQ_X.codebook_entries, TPQ_X.vector_size),
        ).astype(np.float32),
    )
    tensor = NintMoeTensor(
        (experts, out, width),
        (
            NintMoePool(np.array([0], dtype=np.int32), q8),
            NintMoePool(np.array([1], dtype=np.int32), mx),
            NintMoePool(np.array([2], dtype=np.int32), tpq),
        ),
    )
    weight = to_gpu(tensor)
    ids = torch.arange(tokens * 2, device="cuda", dtype=torch.int32).reshape(tokens, 2)
    ids = (ids % experts).contiguous()
    route = MoeRoutePlan.build(ids, experts)
    source = torch.randn(tokens, width, device="cuda", dtype=torch.float16)
    actual = grouped_matmul(weight, source, route).float()

    q8_source = source.float().reshape(tokens, width // 32, 32)
    q8_source_scale = q8_source.abs().amax(dim=-1) / 127.0
    q8_source_scale = torch.where(
        q8_source_scale > 0,
        q8_source_scale,
        torch.ones_like(q8_source_scale),
    )
    q8_quantized = torch.round(q8_source / q8_source_scale[..., None]).clamp(-127, 127)
    q8_expected = torch.einsum(
        "tgk,ogk,tg,og->to",
        q8_quantized,
        torch.as_tensor(q8.q, device="cuda").float(),
        q8_source_scale,
        torch.as_tensor(q8.scale, device="cuda").float(),
    )
    dense = torch.stack(
        (
            torch.zeros_like(torch.as_tensor(mx_dense, device="cuda")),
            torch.as_tensor(mx_dense, device="cuda"),
            torch.as_tensor(dequantize_tpq_pq(tpq), device="cuda"),
        )
    ).float()
    expected = torch.empty_like(actual)
    for token in range(tokens):
        for route_index in range(2):
            expert = int(ids[token, route_index])
            expected[token, route_index] = (
                q8_expected[token] if expert == 0 else source[token].float() @ dense[expert].T
            )
    torch.testing.assert_close(actual, expected.half().float(), rtol=0.006, atol=0.03)
