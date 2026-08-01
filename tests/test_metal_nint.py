"""Apple-silicon packed NINT Metal kernel tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.formats import io  # noqa: E402
from mfq.formats.header import FileHeader  # noqa: E402
from mfq.formats.nint import NINT2_SPEC, NintSpec  # noqa: E402
from mfq.kernels.metal.nint import (  # noqa: E402
    MetalNintWeight,
    _can_use_nint4_gs24_decode,
    _can_use_nint5_gs28_decode,
    _can_use_nint6_gs24_decode,
    nint_dequantize,
    nint_dequantize_matmul,
    nint_embedding,
    nint_gemm,
    nint_gemv,
    nint_matmul,
    nint_mmq,
    nint_swiglu,
)
from mfq.quantize import nint_quant  # noqa: E402
from mfq.runtime.mlx_linear import MlxNintLinear, MlxNintModel, MlxSwiGLUFFN  # noqa: E402


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _random(seed: int, shape: tuple[int, ...], scale: float = 0.1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, size=shape).astype(np.float32)


def _nint_gs24_fixture(
    *,
    bits: int,
    out: int,
    width: int,
    seed: int,
) -> nint_quant.NintTensor:
    groups = (width + 23) // 24
    rng = np.random.default_rng(seed)
    return nint_quant.NintTensor(
        spec=NintSpec(bits, 24, 7),
        shape=(out, width),
        axis=0,
        q=rng.integers(
            0,
            1 << bits,
            size=(out, groups, 24),
            dtype=np.uint8,
        ),
        neuron_scale=rng.uniform(
            0.0004,
            0.0012,
            size=out,
        ).astype(np.float32),
        neuron_min=rng.uniform(
            0.0002,
            0.0007,
            size=out,
        ).astype(np.float32),
        sub_scale=rng.integers(
            1,
            9,
            size=(out, groups),
            dtype=np.uint8,
        ),
        sub_min=rng.integers(
            1,
            7,
            size=(out, groups),
            dtype=np.uint8,
        ),
        neuron_len=width,
    )


def _nint4_gs24_fixture(
    *,
    out: int,
    width: int,
    seed: int,
) -> nint_quant.NintTensor:
    return _nint_gs24_fixture(
        bits=4,
        out=out,
        width=width,
        seed=seed,
    )


def _nint6_gs24_fixture(
    *,
    out: int,
    width: int,
    seed: int,
) -> nint_quant.NintTensor:
    return _nint_gs24_fixture(
        bits=6,
        out=out,
        width=width,
        seed=seed,
    )


@pytest.mark.parametrize(
    "spec,width",
    [
        (NINT2_SPEC, 70),
        (NintSpec(3, 24, 5), 77),
        (NintSpec(4, 24, 6), 79),
        (NintSpec(5, 28, 7), 83),
        (NintSpec(6, 26, 7), 81),
        (NintSpec(8, 48, 7), 97),
    ],
)
@pytest.mark.parametrize("rows", [1, 3, 9])
def test_packed_nint_matmul_matches_numpy(spec: NintSpec, width: int, rows: int):
    weight = _random(1000 + spec.bits * 10 + rows, (23, width))
    source = _random(2000 + spec.bits * 10 + rows, (rows, width))
    tensor = nint_quant.quantize(weight, spec)
    packed = MetalNintWeight.from_tensor(tensor)

    actual = _array(nint_matmul(packed, mx.array(source)))
    expected = source @ nint_quant.dequantize(tensor).T

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "spec",
    [
        NINT2_SPEC,
        NintSpec(3, 24, 5),
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 7),
        NintSpec(6, 26, 7),
        NintSpec(8, 48, 7),
    ],
)
@pytest.mark.parametrize(
    "rows,operation",
    [
        (1, nint_gemv),
        (4, nint_mmq),
        (17, nint_gemm),
    ],
)
def test_explicit_nint_gemv_mmq_gemm_paths(spec: NintSpec, rows: int, operation):
    tensor = nint_quant.quantize(_random(300 + spec.bits, (19, 89)), spec)
    source = _random(400 + spec.bits + rows, (rows, 89))
    actual = _array(operation(MetalNintWeight.from_tensor(tensor), source))
    expected = source @ nint_quant.dequantize(tensor).T
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_packed_nint_matmul_preserves_prefix_shape():
    tensor = nint_quant.quantize(_random(11, (13, 65)), NintSpec(4, 24, 6))
    source = _random(12, (2, 3, 65))
    actual = _array(nint_matmul(MetalNintWeight.from_tensor(tensor), source))
    expected = source @ nint_quant.dequantize(tensor).T
    assert actual.shape == (2, 3, 13)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_packed_nint_matmul_supports_fp16_input():
    tensor = nint_quant.quantize(_random(15, (29, 91)), NintSpec(4, 24, 6))
    source = _random(16, (5, 91)).astype(np.float16)
    actual = _array(nint_matmul(MetalNintWeight.from_tensor(tensor), mx.array(source)))
    expected = source.astype(np.float32) @ nint_quant.dequantize(tensor).T
    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_nint5_gs28_decode_subsimd_matches_two_level_dequant():
    spec = NintSpec(5, 28, 7)
    tensor = nint_quant.quantize(_random(151, (37, 111)), spec)
    packed = MetalNintWeight.from_blob(io.pack_nint(tensor))
    source = _random(152, (1, 111)).astype(np.float16)
    source_mx = mx.array(source)

    assert _can_use_nint5_gs28_decode(packed, source_mx, 1)
    actual = _array(nint_gemv(packed, source_mx))
    expected = source.astype(np.float32) @ nint_quant.dequantize(tensor).T

    assert actual.dtype == np.float16
    assert actual.shape == (1, 37)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_nint5_gs28_decode_compatibility_keeps_fallbacks():
    gs28_tensor = nint_quant.quantize(
        _random(153, (19, 83)),
        NintSpec(5, 28, 7),
    )
    gs28 = MetalNintWeight.from_tensor(gs28_tensor)
    fp16 = mx.array(_random(154, (1, 83)).astype(np.float16))
    fp32_host = _random(155, (1, 83))
    fp32 = mx.array(fp32_host)
    assert _can_use_nint5_gs28_decode(gs28, fp16, 1)
    assert not _can_use_nint5_gs28_decode(gs28, fp32, 1)
    assert not _can_use_nint5_gs28_decode(gs28, fp16, 2)

    gs24_tensor = nint_quant.quantize(
        _random(156, (19, 83)),
        NintSpec(5, 24, 7),
    )
    other_group_size = MetalNintWeight.from_tensor(gs24_tensor)
    assert other_group_size.q5_exec
    assert not _can_use_nint5_gs28_decode(
        other_group_size,
        mx.array(_random(157, (1, 83)).astype(np.float16)),
        1,
    )

    # Exercise both compatibility fallbacks rather than only the predicate.
    fp32_actual = _array(nint_gemv(gs28, fp32))
    fp32_expected = fp32_host @ nint_quant.dequantize(gs28_tensor).T
    np.testing.assert_allclose(
        fp32_actual,
        fp32_expected,
        rtol=2e-5,
        atol=2e-5,
    )

    gs24_source = _random(158, (1, 83)).astype(np.float16)
    gs24_actual = _array(nint_gemv(other_group_size, gs24_source))
    gs24_expected = (
        gs24_source.astype(np.float32)
        @ nint_quant.dequantize(gs24_tensor).T
    )
    np.testing.assert_allclose(
        gs24_actual,
        gs24_expected,
        rtol=2e-3,
        atol=2e-3,
    )


def test_nint4_gs24_decode_tail_and_nonzero_metadata():
    tensor = _nint4_gs24_fixture(out=37, width=111, seed=169)
    packed = MetalNintWeight.from_tensor(tensor)
    source = _random(170, (1, 111)).astype(np.float16)
    source_mx = mx.array(source)

    assert 111 % 24 != 0
    assert 37 % 16 != 0
    assert np.all(tensor.sub_scale != 0)
    assert np.all(tensor.sub_min != 0)
    assert _can_use_nint4_gs24_decode(packed, source_mx, 1)

    actual = _array(nint_gemv(packed, source_mx))
    expected = (
        source.astype(np.float32)
        @ nint_quant.dequantize(tensor).T
    )
    assert actual.dtype == np.float16
    assert actual.shape == (1, 37)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_nint4_gs24_decode_compatibility_keeps_fallbacks():
    tensor = _nint4_gs24_fixture(out=19, width=83, seed=171)
    packed = MetalNintWeight.from_tensor(tensor)
    fp16 = mx.array(_random(172, (1, 83)).astype(np.float16))
    fp32_host = _random(173, (1, 83))
    fp32 = mx.array(fp32_host)

    assert _can_use_nint4_gs24_decode(packed, fp16, 1)
    assert not _can_use_nint4_gs24_decode(packed, fp32, 1)
    assert not _can_use_nint4_gs24_decode(packed, fp16, 2)

    other_group_tensor = nint_quant.quantize(
        _random(174, (19, 83)),
        NintSpec(4, 26, 7),
    )
    other_group = MetalNintWeight.from_tensor(other_group_tensor)
    assert not _can_use_nint4_gs24_decode(
        other_group,
        mx.array(_random(175, (1, 83)).astype(np.float16)),
        1,
    )

    # Exercise both dtype and row-count fallbacks.
    fp32_actual = _array(nint_gemv(packed, fp32))
    fp32_expected = fp32_host @ nint_quant.dequantize(tensor).T
    np.testing.assert_allclose(
        fp32_actual,
        fp32_expected,
        rtol=2e-5,
        atol=2e-5,
    )

    multirow = _random(176, (2, 83)).astype(np.float16)
    multirow_actual = _array(nint_mmq(packed, multirow))
    multirow_expected = (
        multirow.astype(np.float32)
        @ nint_quant.dequantize(tensor).T
    )
    np.testing.assert_allclose(
        multirow_actual,
        multirow_expected,
        rtol=2e-3,
        atol=2e-3,
    )


def test_nint4_gs24_decode_large_output_tail():
    tensor = _nint4_gs24_fixture(
        out=65_539,
        width=25,
        seed=177,
    )
    packed = MetalNintWeight.from_tensor(tensor)
    source = _random(178, (1, 25)).astype(np.float16)

    assert tensor.shape[0] % 16 != 0
    actual = _array(nint_gemv(packed, source))
    expected = (
        source.astype(np.float32)
        @ nint_quant.dequantize(tensor).T
    )
    assert actual.shape == (1, 65_539)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_nint6_gs24_decode_tail_and_nonzero_metadata():
    tensor = _nint6_gs24_fixture(out=37, width=111, seed=159)
    packed = MetalNintWeight.from_tensor(tensor)
    source = _random(160, (1, 111)).astype(np.float16)
    source_mx = mx.array(source)

    assert 111 % 24 != 0
    assert 37 % 16 != 0
    assert np.all(tensor.sub_scale != 0)
    assert np.all(tensor.sub_min != 0)
    assert _can_use_nint6_gs24_decode(packed, source_mx, 1)

    actual = _array(nint_gemv(packed, source_mx))
    expected = (
        source.astype(np.float32)
        @ nint_quant.dequantize(tensor).T
    )
    assert actual.dtype == np.float16
    assert actual.shape == (1, 37)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_nint6_gs24_decode_compatibility_keeps_fallbacks():
    tensor = _nint6_gs24_fixture(out=19, width=83, seed=161)
    packed = MetalNintWeight.from_tensor(tensor)
    fp16 = mx.array(_random(162, (1, 83)).astype(np.float16))
    fp32_host = _random(163, (1, 83))
    fp32 = mx.array(fp32_host)

    assert _can_use_nint6_gs24_decode(packed, fp16, 1)
    assert not _can_use_nint6_gs24_decode(packed, fp32, 1)
    assert not _can_use_nint6_gs24_decode(packed, fp16, 2)

    other_group_tensor = nint_quant.quantize(
        _random(164, (19, 83)),
        NintSpec(6, 26, 7),
    )
    other_group = MetalNintWeight.from_tensor(other_group_tensor)
    assert not _can_use_nint6_gs24_decode(
        other_group,
        mx.array(_random(165, (1, 83)).astype(np.float16)),
        1,
    )

    # Exercise both dtype and row-count fallbacks.
    fp32_actual = _array(nint_gemv(packed, fp32))
    fp32_expected = fp32_host @ nint_quant.dequantize(tensor).T
    np.testing.assert_allclose(
        fp32_actual,
        fp32_expected,
        rtol=2e-5,
        atol=2e-5,
    )

    multirow = _random(166, (2, 83)).astype(np.float16)
    multirow_actual = _array(nint_mmq(packed, multirow))
    multirow_expected = (
        multirow.astype(np.float32)
        @ nint_quant.dequantize(tensor).T
    )
    np.testing.assert_allclose(
        multirow_actual,
        multirow_expected,
        rtol=2e-3,
        atol=2e-3,
    )


def test_nint6_gs24_decode_large_output_tail():
    tensor = _nint6_gs24_fixture(
        out=65_539,
        width=25,
        seed=167,
    )
    packed = MetalNintWeight.from_tensor(tensor)
    source = _random(168, (1, 25)).astype(np.float16)

    assert tensor.shape[0] % 16 != 0
    actual = _array(nint_gemv(packed, source))
    expected = (
        source.astype(np.float32)
        @ nint_quant.dequantize(tensor).T
    )
    assert actual.shape == (1, 65_539)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    "spec",
    [
        NINT2_SPEC,
        NintSpec(3, 24, 5),
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 7),
        NintSpec(6, 26, 7),
        NintSpec(8, 48, 7),
    ],
)
def test_nint_fp16_simdgroup_matrix_gemm(spec: NintSpec):
    tensor = nint_quant.quantize(_random(160 + spec.bits, (35, 93)), spec)
    source = _random(170 + spec.bits, (33, 93)).astype(np.float16)
    actual = _array(nint_gemm(MetalNintWeight.from_tensor(tensor), source))
    expected = source.astype(np.float32) @ nint_quant.dequantize(tensor).T
    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


def test_nint_fp16_m64_tile_row_mapping():
    spec = NintSpec(4, 24, 6)
    tensor = nint_quant.quantize(_random(181, (67, 101)), spec)
    source = _random(182, (65, 101)).astype(np.float16)
    actual = _array(nint_gemm(MetalNintWeight.from_tensor(tensor), source))
    expected = source.astype(np.float32) @ nint_quant.dequantize(tensor).T
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize(
    "spec",
    [
        NINT2_SPEC,
        NintSpec(3, 24, 5),
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 7),
        NintSpec(6, 26, 7),
        NintSpec(8, 48, 7),
    ],
)
def test_nint_temporary_dequant_dense_gemm(spec: NintSpec):
    tensor = nint_quant.quantize(_random(190 + spec.bits, (35, 93)), spec)
    weight = MetalNintWeight.from_tensor(tensor)
    source = _random(200 + spec.bits, (67, 93)).astype(np.float16)
    decoded = nint_quant.dequantize(tensor)
    actual_weight = _array(nint_dequantize(weight))
    actual = _array(nint_dequantize_matmul(weight, source))
    selected = _array(nint_matmul(weight, source))
    np.testing.assert_allclose(actual_weight, decoded, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(
        actual,
        source.astype(np.float32) @ decoded.T,
        rtol=4e-3,
        atol=4e-3,
    )
    np.testing.assert_array_equal(selected, actual)


@pytest.mark.parametrize(
    "spec",
    [
        NINT2_SPEC,
        NintSpec(3, 24, 5),
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 7),
        NintSpec(6, 26, 7),
        NintSpec(8, 48, 7),
    ],
)
def test_packed_nint_blob_upload_matches_tensor_upload(spec: NintSpec):
    tensor = nint_quant.quantize(_random(17 + spec.bits, (21, 89)), spec)
    from_tensor = MetalNintWeight.from_tensor(tensor)
    from_blob = MetalNintWeight.from_blob(io.pack_nint(tensor))
    source = _random(18 + spec.bits, (3, 89))
    actual = _array(nint_matmul(from_blob, source))
    expected = _array(nint_matmul(from_tensor, source))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_packed_nint_embedding_decodes_selected_rows():
    tensor = nint_quant.quantize(_random(21, (31, 73)), NintSpec(5, 28, 7))
    ids = np.asarray([[0, 7, 30], [3, 9, 4]], dtype=np.int32)
    actual = _array(nint_embedding(MetalNintWeight.from_tensor(tensor), ids, dtype=mx.float32))
    expected = nint_quant.dequantize(tensor)[ids]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_mlx_linear_keeps_weight_packed():
    tensor = nint_quant.quantize(_random(31, (19, 96)), NintSpec(4, 24, 6))
    layer = MlxNintLinear(tensor)
    unpacked_nbytes = int(np.prod(tensor.shape)) * np.dtype(np.float16).itemsize
    assert layer.packed_nbytes < unpacked_nbytes

    source = _random(32, (4, 96))
    actual = _array(layer(source))
    expected = source @ nint_quant.dequantize(tensor).T
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_mlx_swiglu_ffn_matches_numpy():
    hidden, intermediate = 48, 64
    gate = nint_quant.quantize(_random(41, (intermediate, hidden)), NintSpec(4, 24, 6))
    up = nint_quant.quantize(_random(42, (intermediate, hidden)), NintSpec(4, 24, 6))
    down = nint_quant.quantize(_random(43, (hidden, intermediate)), NintSpec(4, 24, 6))
    source = _random(44, (2, hidden))

    actual = _array(MlxSwiGLUFFN(gate, up, down)(source))
    gate_value = source @ nint_quant.dequantize(gate).T
    up_value = source @ nint_quant.dequantize(up).T
    hidden_value = gate_value / (1.0 + np.exp(-gate_value)) * up_value
    expected = hidden_value @ nint_quant.dequantize(down).T
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=4e-5)


def test_mlx_important_neuron_ffn_sums_independent_branches():
    hidden = 8
    low_width = 6
    high_width = 2
    source = _random(45, (3, hidden))
    tensors = {
        "gate": _random(46, (low_width, hidden)),
        "up": _random(47, (low_width, hidden)),
        "down": _random(48, (hidden, low_width)),
        "gate.in_high": _random(49, (high_width, hidden)),
        "up.in_high": _random(50, (high_width, hidden)),
        "down.in_high": _random(51, (hidden, high_width)),
    }

    actual = _array(MlxNintModel(tensors).ffn("gate", "up", "down")(source))

    def branch(suffix: str) -> np.ndarray:
        gate = source @ tensors["gate" + suffix].T
        up = source @ tensors["up" + suffix].T
        return (
            gate / (1.0 + np.exp(-gate))
            * up
        ) @ tensors["down" + suffix].T

    expected = branch("") + branch(".in_high")
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=4e-5)


def test_mlx_important_neuron_ffn_rejects_partial_branch():
    tensors = {
        "gate": _random(52, (4, 8)),
        "up": _random(53, (4, 8)),
        "down": _random(54, (8, 4)),
        "gate.in_high": _random(55, (2, 8)),
    }
    with pytest.raises(ValueError, match="matching gate/up/down"):
        MlxNintModel(tensors).ffn("gate", "up", "down")


def test_nint5_fused_swiglu_matches_separate_projections():
    spec = NintSpec(5, 28, 7)
    gate = nint_quant.quantize(_random(45, (37, 85)), spec)
    up = nint_quant.quantize(_random(46, (37, 85)), spec)
    source = _random(47, (4, 85))
    actual = _array(
        nint_swiglu(
            MetalNintWeight.from_tensor(gate),
            MetalNintWeight.from_tensor(up),
            source,
        )
    )
    gate_value = source @ nint_quant.dequantize(gate).T
    up_value = source @ nint_quant.dequantize(up).T
    expected = gate_value / (1.0 + np.exp(-gate_value)) * up_value
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=4e-5)


def test_mlx_model_roundtrip(tmp_path: Path):
    tensor = nint_quant.quantize(_random(51, (17, 72)), NintSpec(4, 24, 6))
    path = tmp_path / "metal-test.mfq"
    io.save(
        path,
        FileHeader(model_arch="metal-test", num_tensors=1),
        {"model.embed_tokens.weight": tensor},
    )

    with MlxNintModel.from_mfq(path) as model:
        source = _random(52, (3, 72))
        actual_linear = _array(model.linear("model.embed_tokens.weight")(source))
        expected_linear = source @ nint_quant.dequantize(tensor).T
        np.testing.assert_allclose(actual_linear, expected_linear, rtol=2e-5, atol=2e-5)

        ids = np.asarray([0, 5, 16], dtype=np.int32)
        actual = _array(model.embedding("model.embed_tokens.weight")(ids))
        expected = nint_quant.dequantize(tensor)[ids].astype(np.float16)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
        assert isinstance(model.tensors, io.MMapTensorStore)
        assert not model.tensors._cache
