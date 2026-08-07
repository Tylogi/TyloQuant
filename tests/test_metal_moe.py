"""Apple-silicon routed NINTM and NEPQ tests."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.formats.moe import NintMoePool, NintMoeTensor  # noqa: E402
from mfq.formats.mx import MxTensor  # noqa: E402
from mfq.formats.nepq import NEPQ0_S, dequantize_nepq, rotation_signs  # noqa: E402
from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.formats.nint8_zero import (  # noqa: E402
    dequantize_nint8_zero,
    quantize_nint8_zero,
)
from mfq.formats.nvq import (  # noqa: E402
    NVQ2_E8,
    NVQ2_E8_4096,
    NVQ3_D4,
    NVQ3_D4_1024,
)
from mfq.kernels.metal.moe import grouped_moe_matmul  # noqa: E402
from mfq.quantize.nint_quant import dequantize, quantize  # noqa: E402
from mfq.runtime.mlx_moe import MlxRoutedLinear, MlxRoutedSwiGLUFFN  # noqa: E402
from tests.test_formats.test_nepq import _tensor as _nepq_tensor  # noqa: E402
from tests.test_metal_vq import (  # noqa: E402
    _fwht_reference,
    _jsc,
    _npq,
    _nvq,
)


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _nint_moe(
    dense: np.ndarray,
    cohorts: tuple[tuple[int, ...], ...],
    *,
    bits: tuple[int, ...] | None = None,
) -> NintMoeTensor:
    experts, out, width = dense.shape
    if bits is None:
        bits = (4,) * len(cohorts)
    profiles = {
        2: NintSpec(2, 16, 5),
        3: NintSpec(3, 24, 6),
        4: NintSpec(4, 24, 6),
        5: NintSpec(5, 28, 7),
        6: NintSpec(6, 24, 7),
        8: NintSpec(8, 48, 7),
    }
    pools = []
    for ids, profile in zip(cohorts, bits, strict=True):
        rows = dense[np.asarray(ids)].reshape(len(ids) * out, width)
        pools.append(
            NintMoePool(
                np.asarray(ids, dtype=np.int32),
                quantize(rows, profiles[profile]),
            )
        )
    return NintMoeTensor((experts, out, width), tuple(pools))


def _decode_nint_moe(tensor: NintMoeTensor) -> np.ndarray:
    result = np.empty(tensor.shape, dtype=np.float32)
    for pool in tensor.pools:
        decoded = dequantize(pool.tensor).reshape(
            len(pool.expert_ids),
            tensor.out_per_expert,
            tensor.neuron_len,
        )
        result[np.asarray(pool.expert_ids)] = decoded
    return result


def _mxfp4_rows(rows: int, width: int, seed: int) -> tuple[MxTensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 256, size=(rows, width // 2), dtype=np.uint8)
    scales = rng.integers(124, 130, size=(rows, width // 32), dtype=np.uint8)
    codes = np.stack((values & 15, values >> 4), axis=-1).reshape(rows, width)
    table = np.asarray(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=np.float32,
    )
    decoded = np.where((codes & 8) == 0, table[codes & 7], -table[codes & 7])
    decoded *= np.repeat(
        np.exp2(scales.astype(np.int16) - 127).astype(np.float32),
        32,
        axis=1,
    )
    return MxTensor("MXFP4", (rows, width), values, scales), decoded


@pytest.mark.parametrize("path", ["direct", "compact", "route_mma", "expert_mma"])
def test_routed_nintm_mxfp4_cohort_all_grouped_paths(path: str):
    rng = np.random.default_rng(948)
    experts, out, width = 4, 9, 96
    mx_ids = np.asarray([0, 2], dtype=np.int32)
    nint_ids = np.asarray([1, 3], dtype=np.int32)
    mx_tensor, mx_dense = _mxfp4_rows(mx_ids.size * out, width, 949)
    nint_source = rng.normal(0.0, 0.1, size=(nint_ids.size * out, width)).astype(
        np.float32
    )
    nint_tensor = quantize(nint_source, NintSpec(4, 24, 6))
    tensor = NintMoeTensor(
        (experts, out, width),
        (
            NintMoePool(mx_ids, mx_tensor),
            NintMoePool(nint_ids, nint_tensor),
        ),
    )
    dense = np.empty((experts, out, width), dtype=np.float32)
    dense[mx_ids] = mx_dense.reshape(mx_ids.size, out, width)
    dense[nint_ids] = dequantize(nint_tensor).reshape(nint_ids.size, out, width)
    source = rng.normal(0.0, 0.04, size=(9, width)).astype(np.float16)
    ids = np.tile(np.asarray([[0, 1]], dtype=np.int32), (9, 1))
    ids[1::2] = np.asarray([2, 3], dtype=np.int32)
    layer = MlxRoutedLinear(tensor)
    assert layer.uses_grouped_kernel
    options = {
        "direct": dict(compact_threshold=None),
        "compact": dict(compact_threshold=1, matrix_threshold=None),
        "route_mma": dict(
            compact_threshold=1,
            matrix_threshold=1,
            expert_matrix_threshold=None,
        ),
        "expert_mma": dict(
            compact_threshold=1,
            matrix_threshold=1,
            expert_matrix_threshold=1,
        ),
    }[path]
    actual = _array(grouped_moe_matmul(layer.grouped_weight, source, ids, **options))
    expected = np.stack(
        [
            np.stack([source[token].astype(np.float32) @ dense[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    tolerance = 4e-3 if "mma" in path else 8e-4
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_routed_nintm_mixed_precision_cohorts():
    rng = np.random.default_rng(10)
    dense = rng.normal(0, 0.1, size=(4, 7, 40)).astype(np.float32)
    tensor = _nint_moe(dense, ((0, 2), (1, 3)), bits=(4, 5))
    decoded = _decode_nint_moe(tensor)
    source = rng.normal(0, 0.1, size=(3, 40)).astype(np.float32)
    ids = np.asarray([[0, 3], [2, 1], [3, 0]], dtype=np.int32)
    layer = MlxRoutedLinear(tensor)
    assert layer.uses_grouped_kernel
    actual = _array(layer(source, ids))
    expected = np.stack(
        [
            np.stack([source[token] @ decoded[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize(
    ("bits", "width"),
    [(2, 48), (3, 48), (4, 48), (5, 56), (6, 48), (8, 48)],
)
def test_routed_grouped_nint_bit_widths(bits: int, width: int):
    rng = np.random.default_rng(100 + bits)
    dense = rng.normal(0, 0.1, size=(2, 7, width)).astype(np.float32)
    tensor = _nint_moe(dense, ((0, 1),), bits=(bits,))
    decoded = _decode_nint_moe(tensor)
    source = rng.normal(0, 0.1, size=(2, width)).astype(np.float32)
    ids = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
    layer = MlxRoutedLinear(tensor)
    assert layer.uses_grouped_kernel
    actual = _array(layer(source, ids))
    expected = np.stack(
        [
            np.stack([source[token] @ decoded[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("bits", [2, 3])
@pytest.mark.parametrize("expert_matrix_threshold", [None, 1])
def test_low_bit_nint_octet_loader_matches_dequant(
    bits: int,
    expert_matrix_threshold: int | None,
):
    rng = np.random.default_rng(180 + bits)
    width = 48
    dense = rng.normal(0, 0.1, size=(2, 13, width)).astype(np.float32)
    tensor = _nint_moe(dense, ((0, 1),), bits=(bits,))
    decoded = _decode_nint_moe(tensor)
    source = rng.normal(0, 0.1, size=(9, width)).astype(np.float16)
    ids = np.tile(np.asarray([[0, 1]], dtype=np.int32), (9, 1))
    layer = MlxRoutedLinear(tensor)
    actual = _array(
        grouped_moe_matmul(
            layer.grouped_weight,
            source,
            ids,
            compact_threshold=1,
            matrix_threshold=1,
            expert_matrix_threshold=expert_matrix_threshold,
        )
    )
    expected = np.stack(
        [
            np.stack([source[token] @ decoded[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("groupsize", [20, 64])
def test_nint_octet_loader_handles_group_boundaries(groupsize: int):
    rng = np.random.default_rng(280 + groupsize)
    width = 128
    dense = rng.normal(0, 0.1, size=(2, 13, width)).astype(np.float32)
    quantized = quantize(
        dense.reshape((-1, width)),
        NintSpec(4, groupsize, 6),
    )
    tensor = NintMoeTensor(
        dense.shape,
        (
            NintMoePool(
                np.asarray([0, 1], dtype=np.int32),
                quantized,
            ),
        ),
    )
    decoded = _decode_nint_moe(tensor)
    source = rng.normal(0, 0.1, size=(9, width)).astype(np.float16)
    ids = np.tile(np.asarray([[0, 1]], dtype=np.int32), (9, 1))
    layer = MlxRoutedLinear(tensor)
    actual = _array(
        grouped_moe_matmul(
            layer.grouped_weight,
            source,
            ids,
            compact_threshold=1,
            matrix_threshold=1,
            expert_matrix_threshold=None,
        )
    )
    expected = np.stack(
        [
            np.stack([source[token] @ decoded[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


def test_routed_nepq_selects_global_expert():
    tensor = _nepq_tensor(NEPQ0_S)
    tensor.rotation_block = 0
    tensor.rotation_seed = 0
    container = NintMoeTensor(
        tensor.shape,
        (NintMoePool(np.asarray([0, 1], dtype=np.int32), tensor),),
    )
    source = (
        np.random.default_rng(20).normal(0, 0.1, size=(3, tensor.neuron_len)).astype(np.float32)
    )
    ids = np.asarray([[0, 1], [1, 0], [1, 1]], dtype=np.int32)
    layer = MlxRoutedLinear(container)
    assert layer.uses_grouped_kernel
    actual = _array(layer(source, ids))
    decoded = dequantize_nepq(tensor)
    expected = np.stack(
        [
            np.stack([source[token] @ decoded[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)


def test_routed_mixed_nint_and_npq_single_dispatch():
    rng = np.random.default_rng(25)
    out, width = 13, 80
    dense = rng.normal(0, 0.1, size=(out, width)).astype(np.float32)
    nint = quantize(dense, NintSpec(4, 24, 6))
    npq, npq_decoded = _npq(True, 26)
    tensor = NintMoeTensor(
        (2, out, width),
        (
            NintMoePool(np.asarray([0], dtype=np.int32), nint),
            NintMoePool(np.asarray([1], dtype=np.int32), npq),
        ),
    )
    source = rng.normal(0, 0.1, size=(3, width)).astype(np.float32)
    ids = np.asarray([[0, 1], [1, 0], [1, 1]], dtype=np.int32)
    layer = MlxRoutedLinear(tensor)
    assert layer.uses_grouped_kernel
    actual = _array(layer(source, ids))
    decoded = (dequantize(nint), npq_decoded)
    expected = np.stack(
        [
            np.stack([source[token] @ decoded[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("compact", [False, True])
def test_routed_mixed_nint_and_nint8_zero_single_dispatch(compact: bool):
    rng = np.random.default_rng(251)
    out, width = 13, 64
    nint_dense = rng.normal(0, 0.1, size=(out, width)).astype(np.float32)
    q8_dense = rng.normal(0, 0.1, size=(out, width)).astype(np.float32)
    nint = quantize(nint_dense, NintSpec(4, 24, 6))
    q8 = quantize_nint8_zero(q8_dense)
    tensor = NintMoeTensor(
        (2, out, width),
        (
            NintMoePool(np.asarray([0], dtype=np.int32), nint),
            NintMoePool(np.asarray([1], dtype=np.int32), q8),
        ),
    )
    source = rng.normal(0, 0.1, size=(8, width)).astype(np.float16 if compact else np.float32)
    ids = np.tile(np.asarray([[0, 1]], dtype=np.int32), (8, 1))
    layer = MlxRoutedLinear(tensor)
    assert layer.uses_grouped_kernel
    actual = _array(
        grouped_moe_matmul(
            layer.grouped_weight,
            source,
            ids,
            compact_threshold=1 if compact else None,
            matrix_threshold=1 if compact else None,
        )
    )
    decoded = (dequantize(nint), dequantize_nint8_zero(q8))
    expected = np.stack(
        [
            np.stack([source[token] @ decoded[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    tolerance = 3e-3 if compact else 5e-5
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("expert_matrix_threshold", [None, 1])
def test_route_compacted_matrix_handles_expert_boundary_tile(
    expert_matrix_threshold: int | None,
):
    rng = np.random.default_rng(252)
    out, width = 13, 64
    first_dense = rng.normal(0, 0.1, size=(out, width)).astype(np.float32)
    second_dense = rng.normal(0, 0.1, size=(out, width)).astype(np.float32)
    first = quantize(first_dense, NintSpec(4, 24, 6))
    second = quantize_nint8_zero(second_dense)
    tensor = NintMoeTensor(
        (2, out, width),
        (
            NintMoePool(np.asarray([0], dtype=np.int32), first),
            NintMoePool(np.asarray([1], dtype=np.int32), second),
        ),
    )
    source = rng.normal(0, 0.1, size=(9, width)).astype(np.float16)
    ids = np.tile(np.asarray([[0, 1]], dtype=np.int32), (9, 1))
    layer = MlxRoutedLinear(tensor)
    actual = _array(
        grouped_moe_matmul(
            layer.grouped_weight,
            source,
            ids,
            compact_threshold=1,
            matrix_threshold=1,
            expert_matrix_threshold=expert_matrix_threshold,
        )
    )
    decoded = (dequantize(first), dequantize_nint8_zero(second))
    expected = np.stack(
        [
            np.stack([source[token] @ decoded[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


def test_routed_multiple_vq_pool_offsets():
    first, first_decoded = _nvq(NVQ2_E8, 27)
    second, second_decoded = _nvq(NVQ3_D4, 28)
    tensor = NintMoeTensor(
        (2, first.shape[0], first.neuron_len),
        (
            NintMoePool(np.asarray([0], dtype=np.int32), first),
            NintMoePool(np.asarray([1], dtype=np.int32), second),
        ),
    )
    rng = np.random.default_rng(29)
    source = rng.normal(0, 0.1, size=(2, first.neuron_len)).astype(np.float32)
    ids = np.asarray([[1, 0], [0, 1]], dtype=np.int32)
    layer = MlxRoutedLinear(tensor)
    assert layer.uses_grouped_kernel
    actual = _array(layer(source, ids))
    decoded = (first_decoded, second_decoded)
    expected = np.stack(
        [
            np.stack([source[token] @ decoded[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)


def test_routed_extended_jsc_index_widths():
    first, first_decoded = _jsc(NVQ2_E8_4096, 20260834)
    second, second_decoded = _jsc(NVQ3_D4_1024, 20260835)
    tensor = NintMoeTensor(
        (2, first.shape[0], first.neuron_len),
        (
            NintMoePool(np.asarray([0], dtype=np.int32), first),
            NintMoePool(np.asarray([1], dtype=np.int32), second),
        ),
    )
    source = (
        np.random.default_rng(20260836)
        .normal(
            0,
            0.1,
            size=(2, first.neuron_len),
        )
        .astype(np.float16)
    )
    ids = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
    actual = _array(MlxRoutedLinear(tensor)(source, ids))
    decoded = (first_decoded, second_decoded)
    expected = np.stack(
        [
            np.stack([source[token].astype(np.float32) @ decoded[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    np.testing.assert_allclose(
        actual,
        expected.astype(np.float16),
        rtol=4e-3,
        atol=4e-3,
    )


def test_route_compacted_mixed_nint_npq_prefill():
    rng = np.random.default_rng(30)
    out, width = 13, 80
    dense = rng.normal(0, 0.1, size=(out, width)).astype(np.float32)
    nint = quantize(dense, NintSpec(4, 24, 6))
    npq, npq_decoded = _npq(True, 31)
    tensor = NintMoeTensor(
        (2, out, width),
        (
            NintMoePool(np.asarray([0], dtype=np.int32), nint),
            NintMoePool(np.asarray([1], dtype=np.int32), npq),
        ),
    )
    source = rng.normal(0, 0.1, size=(8, width)).astype(np.float32)
    ids = np.tile(np.asarray([[0, 1]], dtype=np.int32), (8, 1))
    layer = MlxRoutedLinear(tensor)
    actual = _array(
        grouped_moe_matmul(
            layer.grouped_weight,
            source,
            ids,
            compact_threshold=1,
        )
    )
    decoded = (dequantize(nint), npq_decoded)
    expected = np.stack(
        [
            np.stack([source[token] @ decoded[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)


def test_route_compacted_matrix_mixed_nint_npq_prefill():
    rng = np.random.default_rng(301)
    out, width = 13, 80
    dense = rng.normal(0, 0.1, size=(out, width)).astype(np.float32)
    nint = quantize(dense, NintSpec(4, 24, 6))
    npq, npq_decoded = _npq(True, 302)
    tensor = NintMoeTensor(
        (2, out, width),
        (
            NintMoePool(np.asarray([0], dtype=np.int32), nint),
            NintMoePool(np.asarray([1], dtype=np.int32), npq),
        ),
    )
    source = rng.normal(0, 0.1, size=(8, width)).astype(np.float16)
    ids = np.tile(np.asarray([[0, 1]], dtype=np.int32), (8, 1))
    layer = MlxRoutedLinear(tensor)
    actual = _array(
        grouped_moe_matmul(
            layer.grouped_weight,
            source,
            ids,
            compact_threshold=1,
            matrix_threshold=1,
        )
    )
    decoded = (dequantize(nint), npq_decoded)
    expected = np.stack(
        [
            np.stack([source[token] @ decoded[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


def test_rotated_nepq_uses_grouped_dispatch():
    tensor = _nepq_tensor(NEPQ0_S)
    container = NintMoeTensor(
        tensor.shape,
        (NintMoePool(np.asarray([0, 1], dtype=np.int32), tensor),),
    )
    rng = np.random.default_rng(31)
    source = rng.normal(0, 0.1, size=(3, tensor.neuron_len)).astype(np.float32)
    ids = np.asarray([[0, 1], [1, 0], [1, 1]], dtype=np.int32)
    layer = MlxRoutedLinear(container)
    assert layer.uses_grouped_kernel
    actual = _array(layer(source, ids))
    signs = rotation_signs(
        tensor.neuron_len,
        tensor.rotation_block,
        tensor.rotation_seed,
    )
    rotated = _fwht_reference(source, tensor.rotation_block, signs)
    decoded = dequantize_nepq(tensor)
    expected = np.stack(
        [
            np.stack([rotated[token] @ decoded[expert].T for expert in row])
            for token, row in enumerate(ids)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)


def test_routed_nintm_swiglu_and_route_reduction():
    rng = np.random.default_rng(30)
    experts, hidden, intermediate = 3, 24, 32
    gate_dense = rng.normal(0, 0.1, size=(experts, intermediate, hidden)).astype(np.float32)
    up_dense = rng.normal(0, 0.1, size=(experts, intermediate, hidden)).astype(np.float32)
    down_dense = rng.normal(0, 0.1, size=(experts, hidden, intermediate)).astype(np.float32)
    cohorts = ((0, 2), (1,))
    gate = _nint_moe(gate_dense, cohorts, bits=(4, 5))
    up = _nint_moe(up_dense, cohorts, bits=(4, 5))
    down = _nint_moe(down_dense, cohorts, bits=(4, 5))
    source = rng.normal(0, 0.1, size=(2, hidden)).astype(np.float32)
    ids = np.asarray([[0, 2], [1, 0]], dtype=np.int32)
    weights = np.asarray([[0.7, 0.3], [0.6, 0.4]], dtype=np.float32)
    layer = MlxRoutedSwiGLUFFN(gate, up, down)
    assert layer.uses_grouped_gate_up
    assert layer.gate.uses_grouped_kernel
    assert layer.up.uses_grouped_kernel
    assert layer.down.uses_grouped_kernel
    actual = _array(layer(source, ids, weights))

    gate_w = _decode_nint_moe(gate)
    up_w = _decode_nint_moe(up)
    down_w = _decode_nint_moe(down)
    expected = np.zeros((2, hidden), dtype=np.float32)
    for token in range(2):
        for route in range(2):
            expert = ids[token, route]
            gate_value = source[token] @ gate_w[expert].T
            up_value = source[token] @ up_w[expert].T
            activated = gate_value / (1 + np.exp(-gate_value)) * up_value
            expected[token] += weights[token, route] * (activated @ down_w[expert].T)
    np.testing.assert_allclose(actual, expected, rtol=6e-5, atol=6e-5)
