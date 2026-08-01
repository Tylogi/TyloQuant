"""Apple-silicon tests for TPQ2/CCCP packed Metal kernels."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.formats.tpq import (  # noqa: E402
    CCCP_W,
    CCCP_X,
    CccpInt4Tensor,
    CccpPqSpec,
    CccpPqTensor,
)
from mfq.formats.moe import NintMoePool, NintMoeTensor  # noqa: E402
from mfq.kernels.metal.tpq import (  # noqa: E402
    MetalCccpInt4Weight,
    MetalCccpMoeWeight,
    MetalCccpPqWeight,
    cccp_grouped_moe_matmul,
    cccp_int4_dequantize,
    cccp_int4_embedding,
    cccp_int4_grouped_row_matmul,
    cccp_int4_matmul,
    cccp_pq_dequantize,
    cccp_pq_matmul,
    cccp_pq_routed_matmul,
)


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _int4_tensor(seed: int, rows: int, columns: int) -> tuple[CccpInt4Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    quantized = rng.integers(-8, 8, size=(rows, columns), dtype=np.int16)
    packed = ((quantized[:, 0::2] + 8) | ((quantized[:, 1::2] + 8) << 4)).astype(np.uint8)
    scales = rng.uniform(0.01, 0.2, size=(rows, columns // 64)).astype(np.float16)
    dense = (
        quantized.reshape(rows, columns // 64, 64) * scales.astype(np.float32)[..., None]
    ).reshape(rows, columns)
    return (
        CccpInt4Tensor(
            shape=(rows, columns),
            axis=0,
            neuron_len=columns,
            group_size=64,
            packed=packed,
            scales=scales,
        ),
        dense,
    )


def _pq_tensor(
    seed: int,
    rows: int,
    columns: int,
    *,
    wide: bool,
) -> tuple[CccpPqTensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    spec = CCCP_W if wide else CCCP_X
    codebook = rng.normal(
        0.0,
        0.2,
        size=(spec.codebook_entries, spec.vector_size),
    ).astype(np.float32)
    index_dtype = np.uint16 if wide else np.uint8
    indices = rng.integers(
        0,
        spec.codebook_entries,
        size=(rows, columns // spec.vector_size),
        dtype=index_dtype,
    )
    dense = codebook[indices].reshape(rows, columns)
    return (
        CccpPqTensor(
            spec=spec,
            shape=(rows, columns),
            axis=0,
            neuron_len=columns,
            indices=indices,
            codebook=codebook,
        ),
        dense,
    )


def _packed_pq_tensor(
    seed: int,
    rows: int,
    columns: int,
    *,
    bits: int,
) -> tuple[CccpPqTensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    entries = 3000 if bits == 12 else 5000
    spec = CccpPqSpec(
        "w",
        8,
        entries,
        storage_bits=bits,
    )
    codebook = rng.normal(
        0.0,
        0.2,
        size=(entries, spec.vector_size),
    ).astype(np.float32)
    indices = rng.integers(
        0,
        entries,
        size=(rows, columns // spec.vector_size),
        dtype=np.uint16,
    )
    dense = codebook[indices].reshape(rows, columns)
    return (
        CccpPqTensor(
            spec=spec,
            shape=(rows, columns),
            axis=0,
            neuron_len=columns,
            indices=indices,
            codebook=codebook,
        ),
        dense,
    )


def test_cccp_int4_matmul_dequantize_and_embedding():
    tensor, dense = _int4_tensor(10, 11, 128)
    weight = MetalCccpInt4Weight.from_tensor(tensor)
    x = (
        np.random.default_rng(11)
        .normal(
            0.0,
            0.2,
            size=(3, 128),
        )
        .astype(np.float16)
    )
    actual = cccp_int4_matmul(weight, x)
    np.testing.assert_allclose(
        _array(actual),
        (x.astype(np.float32) @ dense.T).astype(np.float16),
        rtol=3e-3,
        atol=3e-3,
    )
    np.testing.assert_array_equal(
        _array(cccp_int4_dequantize(weight)),
        dense.astype(np.float16),
    )
    ids = np.array([[3, 7], [0, 10]], dtype=np.int32)
    np.testing.assert_array_equal(
        _array(cccp_int4_embedding(weight, ids)),
        dense[ids].astype(np.float16),
    )


def test_cccp_int4_grouped_row_matmul_selects_matching_rows():
    groups = 4
    rank = 3
    tensor, dense = _int4_tensor(14, groups * rank, 128)
    weight = MetalCccpInt4Weight.from_tensor(tensor)
    source = np.random.default_rng(15).normal(size=(2, groups, 128)).astype(np.float16)
    actual = _array(cccp_int4_grouped_row_matmul(weight, source, groups=groups))
    expected = np.einsum(
        "bgk,grk->bgr",
        source.astype(np.float32),
        dense.reshape(groups, rank, 128),
    ).astype(np.float16)
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("wide", [False, True])
def test_cccp_pq_matmul_and_dequantize(wide: bool):
    tensor, dense = _pq_tensor(20 + wide, 13, 64, wide=wide)
    weight = MetalCccpPqWeight.from_tensor(tensor)
    x = (
        np.random.default_rng(22)
        .normal(
            0.0,
            0.2,
            size=(4, 64),
        )
        .astype(np.float16)
    )
    np.testing.assert_allclose(
        _array(cccp_pq_matmul(weight, x)),
        (x.astype(np.float32) @ dense.T).astype(np.float16),
        rtol=4e-3,
        atol=3e-3,
    )
    np.testing.assert_array_equal(
        _array(cccp_pq_dequantize(weight)),
        dense.astype(np.float16),
    )


@pytest.mark.parametrize("bits", [12, 14])
def test_cccp_packed_pq_matmul_and_dequantize(bits: int):
    tensor, dense = _packed_pq_tensor(100 + bits, 13, 64, bits=bits)
    weight = MetalCccpPqWeight.from_tensor(tensor)
    assert weight.indices.dtype == mx.uint8
    assert weight.index_bits == bits
    assert int(weight.indices.nbytes) == (13 * weight.blocks * bits + 7) // 8
    x = (
        np.random.default_rng(120 + bits)
        .normal(
            0.0,
            0.2,
            size=(4, 64),
        )
        .astype(np.float16)
    )
    np.testing.assert_allclose(
        _array(cccp_pq_matmul(weight, x)),
        (x.astype(np.float32) @ dense.T).astype(np.float16),
        rtol=4e-3,
        atol=3e-3,
    )
    np.testing.assert_array_equal(
        _array(cccp_pq_dequantize(weight)),
        dense.astype(np.float16),
    )


def test_cccp_pq_routed_matmul_selects_only_pool_experts():
    experts, output, columns = 3, 7, 64
    tensor, dense = _pq_tensor(
        30,
        experts * output,
        columns,
        wide=False,
    )
    weight = MetalCccpPqWeight.from_tensor(tensor)
    pool_ids = np.array([1, 4, 6], dtype=np.int32)
    selected = np.array([[4, 2, 6], [1, 6, 0]], dtype=np.int32)
    x = (
        np.random.default_rng(31)
        .normal(
            0.0,
            0.2,
            size=(2, 3, columns),
        )
        .astype(np.float16)
    )
    expected = np.zeros((2, 3, output), dtype=np.float32)
    for token in range(2):
        for route in range(3):
            matches = np.flatnonzero(pool_ids == selected[token, route])
            if matches.size:
                start = int(matches[0]) * output
                expected[token, route] = (
                    x[token, route].astype(np.float32) @ dense[start : start + output].T
                )
    actual = cccp_pq_routed_matmul(
        weight,
        x,
        selected,
        pool_ids,
        out_per_expert=output,
    )
    np.testing.assert_allclose(
        _array(actual),
        expected.astype(np.float16),
        rtol=4e-3,
        atol=3e-3,
    )


def test_cccp_p14_routed_matmul_preserves_packed_indices():
    experts, output, columns = 3, 7, 64
    tensor, dense = _packed_pq_tensor(
        130,
        experts * output,
        columns,
        bits=14,
    )
    weight = MetalCccpPqWeight.from_tensor(tensor)
    pool_ids = np.array([1, 4, 6], dtype=np.int32)
    selected = np.array([[4, 2, 6], [1, 6, 0]], dtype=np.int32)
    x = (
        np.random.default_rng(131)
        .normal(
            0.0,
            0.2,
            size=(2, 3, columns),
        )
        .astype(np.float16)
    )
    expected = np.zeros((2, 3, output), dtype=np.float32)
    for token in range(2):
        for route in range(3):
            matches = np.flatnonzero(pool_ids == selected[token, route])
            if matches.size:
                start = int(matches[0]) * output
                expected[token, route] = (
                    x[token, route].astype(np.float32) @ dense[start : start + output].T
                )
    actual = cccp_pq_routed_matmul(
        weight,
        x,
        selected,
        pool_ids,
        out_per_expert=output,
    )
    np.testing.assert_allclose(
        _array(actual),
        expected.astype(np.float16),
        rtol=4e-3,
        atol=3e-3,
    )


def test_cccp_heterogeneous_moe_uses_one_u8_u16_dispatch():
    output, columns = 7, 64
    ids8 = np.array([0, 2], dtype=np.int32)
    ids16 = np.array([1, 3], dtype=np.int32)
    tensor8, dense8 = _pq_tensor(
        41,
        ids8.size * output,
        columns,
        wide=False,
    )
    tensor16, dense16 = _pq_tensor(
        42,
        ids16.size * output,
        columns,
        wide=True,
    )
    tensor = NintMoeTensor(
        shape=(4, output, columns),
        pools=(
            NintMoePool(expert_ids=ids8, tensor=tensor8),
            NintMoePool(expert_ids=ids16, tensor=tensor16),
        ),
    )
    weight = MetalCccpMoeWeight.from_tensor(tensor)
    selected = np.array([[3, 0], [2, 1]], dtype=np.int32)
    x = (
        np.random.default_rng(43)
        .normal(
            0.0,
            0.2,
            size=(2, columns),
        )
        .astype(np.float16)
    )
    dense_by_expert: dict[int, np.ndarray] = {}
    for ids, dense in ((ids8, dense8), (ids16, dense16)):
        for local, expert in enumerate(ids.tolist()):
            dense_by_expert[expert] = dense[local * output : (local + 1) * output]
    expected = np.stack(
        [
            np.stack(
                [
                    x[token].astype(np.float32) @ dense_by_expert[int(expert)].T
                    for expert in selected[token]
                ]
            )
            for token in range(2)
        ]
    )
    actual = cccp_grouped_moe_matmul(weight, x, selected)
    np.testing.assert_allclose(
        _array(actual),
        expected.astype(np.float16),
        rtol=4e-3,
        atol=3e-3,
    )


def test_cccp_heterogeneous_moe_keeps_p12_p14_streams_packed():
    output, columns = 7, 64
    ids12 = np.array([0, 2], dtype=np.int32)
    ids14 = np.array([1, 3], dtype=np.int32)
    tensor12, dense12 = _packed_pq_tensor(
        141,
        ids12.size * output,
        columns,
        bits=12,
    )
    tensor14, dense14 = _packed_pq_tensor(
        142,
        ids14.size * output,
        columns,
        bits=14,
    )
    tensor = NintMoeTensor(
        shape=(4, output, columns),
        pools=(
            NintMoePool(expert_ids=ids12, tensor=tensor12),
            NintMoePool(expert_ids=ids14, tensor=tensor14),
        ),
    )
    weight = MetalCccpMoeWeight.from_tensor(tensor)
    assert int(weight.indices_packed.nbytes) == (
        int(tensor12.indices.nbytes) + int(tensor14.indices.nbytes)
    )
    selected = np.array([[3, 0], [2, 1]], dtype=np.int32)
    x = (
        np.random.default_rng(143)
        .normal(
            0.0,
            0.2,
            size=(2, columns),
        )
        .astype(np.float16)
    )
    dense_by_expert: dict[int, np.ndarray] = {}
    for ids, dense in ((ids12, dense12), (ids14, dense14)):
        for local, expert in enumerate(ids.tolist()):
            dense_by_expert[expert] = dense[local * output : (local + 1) * output]
    expected = np.stack(
        [
            np.stack(
                [
                    x[token].astype(np.float32) @ dense_by_expert[int(expert)].T
                    for expert in selected[token]
                ]
            )
            for token in range(2)
        ]
    )
    actual = cccp_grouped_moe_matmul(weight, x, selected)
    np.testing.assert_allclose(
        _array(actual),
        expected.astype(np.float16),
        rtol=4e-3,
        atol=3e-3,
    )
