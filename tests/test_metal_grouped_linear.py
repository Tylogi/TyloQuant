"""Heterogeneous ordinary grouped-linear Metal tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.formats.tpq import (  # noqa: E402
    CccpInt4Tensor,
    CccpPqSpec,
    CccpPqTensor,
)
from mfq.formats.nepq import NEPQ1_S, dequantize_nepq, rotation_signs  # noqa: E402
from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.formats.nint8_zero import (  # noqa: E402
    dequantize_nint8_zero,
    quantize_nint8_zero,
)
from mfq.formats.nvq import (  # noqa: E402
    NVQ2_E8,
    NVQ2_E8_1024,
    NVQ2_E8_4096,
    NVQ3_D4_1024,
    NvqTensor,
)
from mfq.quantize.nint_quant import dequantize, quantize  # noqa: E402
from mfq.quantize.nvq_quant import dequantize as dequantize_nvq  # noqa: E402
from mfq.runtime.mlx_linear import MlxLinearGroup  # noqa: E402
from tests.test_formats.test_nepq import _tensor as _nepq_tensor  # noqa: E402
from tests.test_metal_vq import _jsc  # noqa: E402


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _nvq(seed: int, out: int, width: int) -> tuple[NvqTensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    spec = NVQ2_E8
    tensor = NvqTensor(
        spec=spec,
        shape=(out, width),
        axis=0,
        neuron_len=width,
        neuron_scale=rng.uniform(0.001, 0.01, size=out).astype(np.float32),
        sub_scale=rng.integers(
            0,
            1 << spec.sub_bits,
            size=(out, math.ceil(width / spec.groupsize)),
            dtype=np.uint8,
        ),
        indices=rng.integers(
            0,
            spec.codebook_entries,
            size=(out, math.ceil(width / spec.vector_size)),
            dtype=np.uint8,
        ),
        signs=rng.integers(
            0,
            128,
            size=(out, math.ceil(width / 8)),
            dtype=np.uint8,
        ),
    )
    return tensor, dequantize_nvq(tensor)


def _fwht_reference(
    value: np.ndarray,
    block: int,
    signs: np.ndarray,
) -> np.ndarray:
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
    return blocks.reshape(value.shape)


@pytest.mark.parametrize("rows", [1, 4])
def test_grouped_linear_mixes_nint_q8_and_vq_with_variable_outputs(rows: int):
    rng = np.random.default_rng(20260801 + rows)
    width = 64
    nint = quantize(
        rng.normal(0, 0.1, size=(17, width)).astype(np.float32),
        NintSpec(4, 24, 6),
    )
    q8 = quantize_nint8_zero(rng.normal(0, 0.1, size=(11, width)).astype(np.float32))
    nvq, nvq_dense = _nvq(20260810 + rows, 13, width)
    group = MlxLinearGroup((nint, q8, nvq))
    assert group.uses_grouped_kernel
    assert group.grouped_weight is not None
    assert group.grouped_weight.output_widths == (17, 11, 13)

    source = rng.normal(0, 0.1, size=(rows, width)).astype(np.float16)
    actual = tuple(_array(item) for item in group(source))
    expected = (
        source.astype(np.float32) @ dequantize(nint).T,
        source.astype(np.float32) @ dequantize_nint8_zero(q8).T,
        source.astype(np.float32) @ nvq_dense.T,
    )
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            result,
            reference.astype(np.float16),
            rtol=4e-3,
            atol=4e-3,
        )


def test_grouped_linear_mixes_cccp_int4_and_p12():
    rng = np.random.default_rng(20260820)
    width = 64
    quantized = rng.integers(-8, 8, size=(9, width), dtype=np.int16)
    packed = ((quantized[:, 0::2] + 8) | ((quantized[:, 1::2] + 8) << 4)).astype(np.uint8)
    scales = rng.uniform(0.01, 0.1, size=(9, 1)).astype(np.float16)
    int4 = CccpInt4Tensor(
        shape=(9, width),
        axis=0,
        neuron_len=width,
        group_size=64,
        packed=packed,
        scales=scales,
    )
    int4_dense = quantized.astype(np.float32) * scales.astype(np.float32)

    spec = CccpPqSpec("w", 8, 300, storage_bits=12)
    indices = rng.integers(
        0,
        spec.codebook_entries,
        size=(7, width // spec.vector_size),
        dtype=np.uint16,
    )
    codebook = rng.normal(
        0,
        0.1,
        size=(spec.codebook_entries, spec.vector_size),
    ).astype(np.float32)
    pq = CccpPqTensor(
        spec=spec,
        shape=(7, width),
        axis=0,
        neuron_len=width,
        indices=indices,
        codebook=codebook,
    )
    pq_dense = codebook[indices].reshape(7, width)
    group = MlxLinearGroup((int4, pq))
    assert group.uses_grouped_kernel

    source = rng.normal(0, 0.1, size=(3, width)).astype(np.float16)
    actual_int4, actual_pq = (_array(item) for item in group(source))
    np.testing.assert_allclose(
        actual_int4,
        (source.astype(np.float32) @ int4_dense.T).astype(np.float16),
        rtol=4e-3,
        atol=4e-3,
    )
    np.testing.assert_allclose(
        actual_pq,
        (source.astype(np.float32) @ pq_dense.T).astype(np.float16),
        rtol=4e-3,
        atol=4e-3,
    )


def test_grouped_linear_mixes_every_nint_width():
    rng = np.random.default_rng(20260825)
    width = 96
    specs = (
        NintSpec(2, 16, 5),
        NintSpec(3, 24, 6),
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 7),
        NintSpec(6, 24, 7),
        NintSpec(8, 48, 7),
    )
    tensors = tuple(
        quantize(
            rng.normal(0, 0.1, size=(5 + index, width)).astype(np.float32),
            spec,
        )
        for index, spec in enumerate(specs)
    )
    group = MlxLinearGroup(tensors)
    assert group.uses_grouped_kernel
    source = rng.normal(0, 0.1, size=(2, width)).astype(np.float32)
    actual = tuple(_array(item) for item in group(source))
    for result, tensor in zip(actual, tensors, strict=True):
        expected = source @ dequantize(tensor).T
        np.testing.assert_allclose(result, expected, rtol=3e-5, atol=3e-5)


def test_grouped_linear_mixes_extended_jsc_index_widths():
    tensors_and_dense = (
        _jsc(NVQ2_E8_1024, 20260830),
        _jsc(NVQ2_E8_4096, 20260831),
        _jsc(NVQ3_D4_1024, 20260832),
    )
    tensors = tuple(item[0] for item in tensors_and_dense)
    group = MlxLinearGroup(tensors)
    assert group.uses_grouped_kernel
    source = (
        np.random.default_rng(20260833)
        .normal(
            0,
            0.1,
            size=(3, tensors[0].neuron_len),
        )
        .astype(np.float16)
    )
    actual = tuple(_array(item) for item in group(source))
    for result, (_, decoded) in zip(
        actual,
        tensors_and_dense,
        strict=True,
    ):
        expected = source.astype(np.float32) @ decoded.T
        np.testing.assert_allclose(
            result,
            expected.astype(np.float16),
            rtol=4e-3,
            atol=4e-3,
        )


def test_grouped_linear_preserves_rotated_nepq_output_shape():
    rng = np.random.default_rng(20260827)
    nepq = _nepq_tensor(NEPQ1_S)
    nint = quantize(
        rng.normal(0, 0.1, size=(5, nepq.neuron_len)).astype(np.float32),
        NintSpec(4, 24, 6),
    )
    group = MlxLinearGroup((nint, nepq))
    assert group.uses_grouped_kernel
    assert group.grouped_weight is not None
    assert group.grouped_weight.output_shapes == (
        (5,),
        (nepq.n_experts, nepq.out_per_expert),
    )

    source = rng.normal(
        0,
        0.1,
        size=(2, nepq.neuron_len),
    ).astype(np.float32)
    actual_nint, actual_nepq = (_array(item) for item in group(source))
    signs = rotation_signs(
        nepq.neuron_len,
        nepq.rotation_block,
        nepq.rotation_seed,
    )
    rotated = _fwht_reference(source, nepq.rotation_block, signs)
    decoded = dequantize_nepq(nepq).reshape(-1, nepq.neuron_len)
    expected_nepq = (rotated @ decoded.T).reshape(
        2,
        nepq.n_experts,
        nepq.out_per_expert,
    )
    np.testing.assert_allclose(
        actual_nint,
        source @ dequantize(nint).T,
        rtol=3e-5,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        actual_nepq,
        expected_nepq,
        rtol=3e-5,
        atol=3e-5,
    )


def test_grouped_linear_preserves_prefix_shape():
    rng = np.random.default_rng(20260830)
    width = 64
    first = quantize(
        rng.normal(0, 0.1, size=(5, width)).astype(np.float32),
        NintSpec(4, 24, 6),
    )
    second = quantize_nint8_zero(rng.normal(0, 0.1, size=(3, width)).astype(np.float32))
    source = rng.normal(0, 0.1, size=(2, 3, width)).astype(np.float16)
    outputs = MlxLinearGroup(
        (first, second),
        grouped_max_rows=None,
    )(source)
    assert outputs[0].shape == (2, 3, 5)
    assert outputs[1].shape == (2, 3, 3)


def test_grouped_linear_falls_back_for_dense_weights():
    first = np.ones((5, 8), dtype=np.float16)
    second = np.ones((3, 8), dtype=np.float16)
    group = MlxLinearGroup((first, second))
    assert not group.uses_grouped_kernel
    output = tuple(_array(item) for item in group(np.ones((1, 8), np.float16)))
    np.testing.assert_array_equal(output[0], np.full((1, 5), 8, np.float16))
    np.testing.assert_array_equal(output[1], np.full((1, 3), 8, np.float16))
