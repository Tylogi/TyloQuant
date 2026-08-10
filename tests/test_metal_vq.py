"""Apple-silicon packed NVQ/NPQ/NEPQ matrix-kernel tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.formats import io  # noqa: E402
from mfq.formats.header import FileHeader  # noqa: E402
from mfq.formats.nepq import (  # noqa: E402
    NEPQ0_A,
    NEPQ0_L,
    NEPQ0_S,
    NEPQ1_A,
    NEPQ1_L,
    NEPQ1_S,
    dequantize_nepq,
    rotation_signs,
)
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
    NvqTensor,
)
from mfq.formats.nvq1_l import IQ1S_TERNARY_2048, NVQ1_L_T8_S3, Nvq1LTensor  # noqa: E402
from mfq.formats.nvq1_s import NVQ1_S, NVQ1_S_SYNTHETIC_BANKS, Nvq1STensor  # noqa: E402
from mfq.kernels.metal.vq import (  # noqa: E402
    MetalVqWeight,
    signed_hadamard,
    vq_dequantize,
    vq_dequantize_matmul,
    vq_embedding,
    vq_gemm,
    vq_gemv,
    vq_matmul,
    vq_mmq,
    vq_swiglu,
    vq_swiglu_compatible,
)
from mfq.quantize.npq0_l import dequantize_npq0_l  # noqa: E402
from mfq.quantize.npq0_s import dequantize_npq0_s  # noqa: E402
from mfq.quantize.nvq1_l_quant import dequantize as dequantize_nvq1_l  # noqa: E402
from mfq.quantize.nvq1_s_quant import dequantize as dequantize_nvq1_s  # noqa: E402
from mfq.quantize.nvq_jsc import dequantize_nvq_jsc  # noqa: E402
from mfq.quantize.nvq_quant import dequantize as dequantize_nvq  # noqa: E402
from mfq.runtime import MlxNintModel, MlxVqEmbedding, MlxVqLinear  # noqa: E402
from tests.test_formats.test_nepq import _tensor as _nepq_tensor  # noqa: E402
from tests.test_formats.test_nepq_a import _a_tensor as _nepq_a_tensor  # noqa: E402


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _nvq(spec, seed: int) -> tuple[NvqTensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    out, width = 13, 80
    groups = math.ceil(width / spec.groupsize)
    vectors = math.ceil(width / spec.vector_size)
    signs = math.ceil(width / 8)
    tensor = NvqTensor(
        spec=spec,
        shape=(out, width),
        axis=0,
        neuron_len=width,
        neuron_scale=rng.uniform(0.001, 0.01, size=out).astype(np.float32),
        sub_scale=rng.integers(
            0,
            1 << spec.sub_bits,
            size=(out, groups),
            dtype=np.uint8,
        ),
        indices=rng.integers(
            0,
            spec.codebook_entries,
            size=(out, vectors),
            dtype=np.uint8 if spec.index_bits <= 8 else np.uint16,
        ),
        signs=rng.integers(0, 128, size=(out, signs), dtype=np.uint8),
    )
    return tensor, dequantize_nvq(tensor)


def _jsc(spec, seed: int) -> tuple[NvqJscTensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    out, width = 13, 80
    groups = math.ceil(width / spec.groupsize)
    vectors = math.ceil(width / spec.vector_size)
    signs = math.ceil(width / 8)
    builtin = {
        "e8_256": E8_256,
        "e8_1024": E8_1024,
        "e8_4096": E8_4096,
        "d4_256": D4_256,
        "d4_512": D4_512,
        "d4_1024": D4_1024,
    }[spec.codebook]
    bank0 = np.clip(builtin.astype(np.int16) * 8, 1, 127).astype(np.int8)
    bank1 = np.roll(bank0, 7, axis=0).copy()
    tensor = NvqJscTensor(
        shape=(out, width),
        axis=0,
        neuron_len=width,
        neuron_scale=rng.uniform(0.0001, 0.001, size=out).astype(np.float32),
        scale_lut=np.linspace(0.25, 4.0, 16, dtype=np.float32),
        bank_for_state=np.arange(16, dtype=np.uint8) & 1,
        state=rng.integers(0, 16, size=(out, groups), dtype=np.uint8),
        indices=rng.integers(
            0,
            spec.codebook_entries,
            size=(out, vectors),
            dtype=np.uint8 if spec.index_bits <= 8 else np.uint16,
        ),
        signs=rng.integers(0, 128, size=(out, signs), dtype=np.uint8),
        codebooks=np.stack((bank0, bank1)),
        base_spec=spec,
    )
    return tensor, dequantize_nvq_jsc(tensor)


def _nvq1_s(seed: int) -> tuple[Nvq1STensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    out, width = 13, 80
    groups = math.ceil(width / 24)
    tensor = Nvq1STensor(
        spec=NVQ1_S,
        shape=(out, width),
        axis=0,
        neuron_len=width,
        neuron_scale=rng.uniform(0.001, 0.01, size=out).astype(np.float32),
        sub_scale=rng.integers(0, 16, size=(out, groups), dtype=np.uint8),
        indices=rng.integers(0, 512, size=(out, width // 8), dtype=np.uint16),
        delta_sign=rng.integers(0, 2, size=(out, groups), dtype=np.uint8),
        codebook=np.roll(NVQ1_S_SYNTHETIC_BANKS, 3, axis=1).copy(),
    )
    return tensor, dequantize_nvq1_s(tensor)


def _nvq1_l(seed: int) -> tuple[Nvq1LTensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    out, width = 13, 80
    groups = math.ceil(width / 24)
    tensor = Nvq1LTensor(
        spec=NVQ1_L_T8_S3,
        shape=(out, width),
        axis=0,
        neuron_len=width,
        neuron_scale=rng.uniform(0.001, 0.01, size=out).astype(np.float32),
        sub_scale=rng.integers(0, 8, size=(out, groups), dtype=np.uint8),
        indices=rng.integers(0, 2048, size=(out, width // 8), dtype=np.uint16),
        delta_sign=rng.integers(0, 2, size=(out, groups), dtype=np.uint8),
        codebook=np.roll(IQ1S_TERNARY_2048, 5, axis=0).copy(),
    )
    return tensor, dequantize_nvq1_l(tensor)


def _npq(short: bool, seed: int):
    rng = np.random.default_rng(seed)
    out, width = 13, 80
    groups = math.ceil(width / 24)
    if short:
        tensor = Npq0STensor(
            shape=(out, width),
            axis=0,
            neuron_len=width,
            neuron_scale=rng.uniform(0.0001, 0.001, size=out).astype(np.float32),
            scale_lut=np.linspace(0.25, 1.0, 4, dtype=np.float32),
            state=rng.integers(0, 4, size=(out, groups), dtype=np.uint8),
            indices=rng.integers(0, 64, size=(out, width // 8), dtype=np.uint8),
            first_codebooks=rng.integers(-32, 33, size=(4, 8, 4), dtype=np.int16).astype(np.int8),
            second_codebooks=rng.integers(-32, 33, size=(4, 8, 4), dtype=np.int16).astype(np.int8),
        )
        return tensor, dequantize_npq0_s(tensor)
    tensor = Npq0LTensor(
        shape=(out, width),
        axis=0,
        neuron_len=width,
        neuron_scale=rng.uniform(0.0001, 0.001, size=out).astype(np.float32),
        scale_lut=np.linspace(0.125, 1.0, 8, dtype=np.float32),
        state=rng.integers(0, 8, size=(out, groups), dtype=np.uint8),
        indices=rng.integers(0, 128, size=(out, width // 8), dtype=np.uint8),
        first_codebooks=rng.integers(-32, 33, size=(8, 8, 4), dtype=np.int16).astype(np.int8),
        second_codebooks=rng.integers(-32, 33, size=(8, 16, 4), dtype=np.int16).astype(np.int8),
    )
    return tensor, dequantize_npq0_l(tensor)


_BASE_FACTORIES = [
    pytest.param(lambda: _nvq(NVQ2_E8, 101), id="nvq2"),
    pytest.param(
        lambda: _nvq(
            NvqSpec("e8_256", groupsize=24, sub_bits=4, sign_mode="index_parity"),
            102,
        ),
        id="nvq2-index-parity",
    ),
    pytest.param(lambda: _nvq(NVQ3_D4, 103), id="nvq3"),
    pytest.param(lambda: _nvq(NVQ3_D4_512, 104), id="nvq3-512"),
    pytest.param(lambda: _jsc(NVQ2_E8, 105), id="nvq2j"),
    pytest.param(lambda: _jsc(NVQ2_E8_1024, 112), id="nvq2j-l"),
    pytest.param(lambda: _jsc(NVQ2_E8_4096, 113), id="nvq2j-xl"),
    pytest.param(lambda: _jsc(NVQ3_D4, 106), id="nvq3j"),
    pytest.param(lambda: _jsc(NVQ3_D4_512, 107), id="nvq3j-512"),
    pytest.param(lambda: _jsc(NVQ3_D4_1024, 114), id="nvq3j-l"),
    pytest.param(lambda: _nvq1_s(108), id="nvq1-s"),
    pytest.param(lambda: _nvq1_l(109), id="nvq1-l"),
    pytest.param(lambda: _npq(True, 110), id="npq0-s"),
    pytest.param(lambda: _npq(False, 111), id="npq0-l"),
]


@pytest.mark.parametrize("factory", _BASE_FACTORIES)
@pytest.mark.parametrize(
    "rows,path",
    [
        (1, "gemv"),
        (4, "mmq"),
        (17, "gemm"),
    ],
)
def test_packed_vq_paths_match_dequantized_weight(factory, rows: int, path: str):
    tensor, decoded = factory()
    weight = MetalVqWeight.from_tensor(tensor)
    source = (
        np.random.default_rng(500 + rows)
        .normal(0.0, 0.1, size=(rows, tensor.neuron_len))
        .astype(np.float32)
    )
    operation = {"gemv": vq_gemv, "mmq": vq_mmq, "gemm": vq_gemm}[path]
    actual = _array(operation(weight, source))
    expected = source @ decoded.T
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("factory", _BASE_FACTORIES)
def test_vq_fp16_simdgroup_matrix_gemm(factory):
    tensor, decoded = factory()
    source = (
        np.random.default_rng(650).normal(0.0, 0.1, size=(33, tensor.neuron_len)).astype(np.float16)
    )
    actual = _array(vq_gemm(MetalVqWeight.from_tensor(tensor), source))
    expected = source.astype(np.float32) @ decoded.T
    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual, expected, rtol=4e-3, atol=4e-3)


def test_vq_fp16_m64_tile_row_mapping():
    tensor, decoded = _npq(True, 660)
    source = (
        np.random.default_rng(661).normal(0.0, 0.1, size=(65, tensor.neuron_len)).astype(np.float16)
    )
    actual = _array(vq_gemm(MetalVqWeight.from_tensor(tensor), source))
    expected = source.astype(np.float32) @ decoded.T
    np.testing.assert_allclose(actual, expected, rtol=4e-3, atol=4e-3)


@pytest.mark.parametrize("factory", _BASE_FACTORIES)
def test_vq_temporary_dequant_dense_gemm(factory):
    tensor, decoded = factory()
    weight = MetalVqWeight.from_tensor(tensor)
    source = (
        np.random.default_rng(680).normal(0.0, 0.1, size=(67, tensor.neuron_len)).astype(np.float16)
    )
    actual_weight = _array(vq_dequantize(weight))
    actual = _array(vq_dequantize_matmul(weight, source))
    selected = _array(vq_matmul(weight, source))
    np.testing.assert_allclose(actual_weight, decoded, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(
        actual,
        source.astype(np.float32) @ decoded.T,
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_array_equal(selected, actual)


@pytest.mark.parametrize("factory", _BASE_FACTORIES)
def test_vq_fused_swiglu(factory):
    gate_tensor, gate_decoded = factory()
    up_tensor, up_decoded = factory()
    source = (
        np.random.default_rng(690)
        .normal(0.0, 0.1, size=(4, gate_tensor.neuron_len))
        .astype(np.float32)
    )
    actual = _array(
        vq_swiglu(
            MetalVqWeight.from_tensor(gate_tensor),
            MetalVqWeight.from_tensor(up_tensor),
            source,
        )
    )
    gate_value = source @ gate_decoded.T
    up_value = source @ up_decoded.T
    expected = gate_value / (1.0 + np.exp(-gate_value)) * up_value
    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)


@pytest.mark.parametrize("factory", _BASE_FACTORIES)
def test_vq_selected_row_embedding(factory):
    tensor, decoded = factory()
    ids = np.asarray([[0, 5], [12, 2]], dtype=np.int32)
    actual = _array(
        vq_embedding(
            MetalVqWeight.from_tensor(tensor),
            ids,
            dtype=mx.float32,
        )
    )
    np.testing.assert_allclose(actual, decoded[ids], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("spec", [NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L])
@pytest.mark.parametrize(
    "rows,path",
    [
        (1, "gemv"),
        (4, "mmq"),
        (17, "gemm"),
    ],
)
def test_nepq_paths_use_bank_pool_and_output_expert_shape(spec, rows: int, path: str):
    tensor = _nepq_tensor(spec)
    tensor.rotation_block = 0
    tensor.rotation_seed = 0
    weight = MetalVqWeight.from_tensor(tensor)
    source = (
        np.random.default_rng(700 + spec.profile_id * 10 + rows)
        .normal(0.0, 0.1, size=(rows, tensor.neuron_len))
        .astype(np.float32)
    )
    operation = {"gemv": vq_gemv, "mmq": vq_mmq, "gemm": vq_gemm}[path]
    actual = _array(operation(weight, source))
    decoded = dequantize_nepq(tensor).reshape(-1, tensor.neuron_len)
    expected = (source @ decoded.T).reshape(rows, tensor.n_experts, tensor.out_per_expert)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("spec", [NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L])
def test_nepq_fused_swiglu_preserves_expert_shape(spec):
    gate = _nepq_tensor(spec)
    gate.rotation_block = 0
    gate.rotation_seed = 0
    up = _nepq_tensor(spec)
    up.rotation_block = 0
    up.rotation_seed = 0
    source = (
        np.random.default_rng(790 + spec.profile_id)
        .normal(0.0, 0.1, size=(3, gate.neuron_len))
        .astype(np.float32)
    )
    actual = _array(
        vq_swiglu(
            MetalVqWeight.from_tensor(gate),
            MetalVqWeight.from_tensor(up),
            source,
        )
    )
    gate_decoded = dequantize_nepq(gate)
    up_decoded = dequantize_nepq(up)
    gate_value = np.einsum("mk,eok->meo", source, gate_decoded)
    up_value = np.einsum("mk,eok->meo", source, up_decoded)
    expected = gate_value / (1.0 + np.exp(-gate_value)) * up_value
    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)


def _fwht_reference(value: np.ndarray, block: int, signs: np.ndarray) -> np.ndarray:
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


def test_nepq_rotation_runs_on_metal_before_matmul():
    tensor = _nepq_tensor(NEPQ0_S)
    weight = MetalVqWeight.from_tensor(tensor)
    source = (
        np.random.default_rng(801).normal(0.0, 0.1, size=(3, tensor.neuron_len)).astype(np.float32)
    )
    signs = rotation_signs(
        tensor.neuron_len,
        tensor.rotation_block,
        tensor.rotation_seed,
    )
    rotated = _fwht_reference(source, tensor.rotation_block, signs)
    actual_rotation = _array(
        signed_hadamard(
            mx.array(source),
            mx.array(signs),
            tensor.rotation_block,
        )
    )
    np.testing.assert_allclose(actual_rotation, rotated, rtol=2e-6, atol=2e-6)

    decoded = dequantize_nepq(tensor).reshape(-1, tensor.neuron_len)
    expected = (rotated @ decoded.T).reshape(3, tensor.n_experts, tensor.out_per_expert)
    actual = _array(vq_matmul(weight, source))
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
    actual_dequant = _array(vq_dequantize_matmul(weight, source.astype(np.float16)))
    np.testing.assert_allclose(actual_dequant, expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("spec", [NEPQ0_A, NEPQ1_A])
@pytest.mark.parametrize(
    "rows,operation",
    [(1, vq_gemv), (4, vq_mmq), (17, vq_gemm)],
)
def test_nepq_a_sparse_residual_paths(spec, rows: int, operation):
    tensor, _ = _nepq_a_tensor(spec)
    weight = MetalVqWeight.from_tensor(tensor)
    dense = dequantize_nepq(tensor).reshape(-1, tensor.neuron_len)
    actual_dense = _array(vq_dequantize(weight)).astype(np.float32)
    np.testing.assert_allclose(actual_dense, dense, rtol=0, atol=1.1e-3)

    source = (
        np.random.default_rng(20260840 + spec.profile_id * 100 + rows)
        .normal(0.0, 0.1, size=(rows, tensor.neuron_len))
        .astype(np.float16)
    )
    signs = rotation_signs(
        tensor.neuron_len,
        tensor.rotation_block,
        tensor.rotation_seed,
    )
    transformed = _fwht_reference(
        source,
        tensor.rotation_block,
        signs,
    ).astype(np.float32)
    expected = (transformed @ dense.T).reshape(
        rows,
        tensor.n_experts,
        tensor.out_per_expert,
    )
    actual = _array(operation(weight, source)).astype(np.float32)
    relative = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative < 1.5e-3


def test_nepq_a_rejects_residual_unaware_vq_swiglu_fusion():
    tensor, _ = _nepq_a_tensor(NEPQ1_A)
    weight = MetalVqWeight.from_tensor(tensor)
    assert not vq_swiglu_compatible(weight, weight)


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_vq_dispatcher_preserves_prefix_shape(rows: int):
    tensor, decoded = _npq(True, 901)
    weight = MetalVqWeight.from_tensor(tensor)
    source = (
        np.random.default_rng(902 + rows)
        .normal(0.0, 0.1, size=(2, rows, tensor.neuron_len))
        .astype(np.float32)
    )
    actual = _array(vq_matmul(weight, source))
    expected = source @ decoded.T
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


def test_mlx_vq_linear_and_mmap_blob_path(tmp_path):
    tensor, decoded = _npq(True, 1001)
    path = tmp_path / "metal-vq.mfq"
    io.save(
        path,
        FileHeader(model_arch="metal-vq", num_tensors=1),
        {"model.layers.0.mlp.gate_proj.weight": tensor},
    )
    source = (
        np.random.default_rng(1002).normal(0.0, 0.1, size=(5, tensor.neuron_len)).astype(np.float32)
    )

    with MlxNintModel.from_mfq(path) as model:
        layer = model.linear("model.layers.0.mlp.gate_proj.weight")
        assert isinstance(layer, MlxVqLinear)
        actual = _array(layer(source))
        expected = source @ decoded.T
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        embedding = model.embedding("model.layers.0.mlp.gate_proj.weight")
        assert isinstance(embedding, MlxVqEmbedding)
        ids = np.asarray([[0, 5], [12, 2]], dtype=np.int32)
        np.testing.assert_allclose(
            _array(embedding.forward(ids, dtype=mx.float32)),
            decoded[ids],
            rtol=5e-4,
            atol=4e-6,
        )
        assert isinstance(model.tensors, io.MMapTensorStore)
        assert not model.tensors._cache


@pytest.mark.parametrize(
    ("spec", "seed"),
    [
        (NVQ2_E8_1024, 1003),
        (NVQ2_E8_4096, 1004),
        (NVQ3_D4_1024, 1005),
    ],
)
def test_extended_jsc_mmap_uses_packed_metal_path(tmp_path, spec, seed: int):
    tensor, decoded = _jsc(spec, seed)
    name = "model.layers.0.self_attn.q_proj.weight"
    path = tmp_path / f"metal-{spec.codebook}.mfq"
    io.save(
        path,
        FileHeader(model_arch="metal-vq", num_tensors=1),
        {name: tensor},
    )
    source = (
        np.random.default_rng(seed + 100)
        .normal(
            0.0,
            0.1,
            size=(5, tensor.neuron_len),
        )
        .astype(np.float32)
    )
    with MlxNintModel.from_mfq(path) as model:
        layer = model.linear(name)
        assert isinstance(layer, MlxVqLinear)
        np.testing.assert_allclose(
            _array(layer(source)),
            source @ decoded.T,
            rtol=5e-4,
            atol=8e-5,
        )
        assert isinstance(model.tensors, io.MMapTensorStore)
        assert not model.tensors._cache


def test_nepq_mmap_runtime_preserves_expert_output_shape(tmp_path):
    tensor = _nepq_tensor(NEPQ1_S)
    tensor.rotation_block = 0
    tensor.rotation_seed = 0
    path = tmp_path / "metal-nepq.mfq"
    io.save(
        path,
        FileHeader(model_arch="metal-nepq", num_tensors=1),
        {"model.layers.0.mlp.experts.gate_up_proj.weight": tensor},
    )
    source = (
        np.random.default_rng(1011).normal(0.0, 0.1, size=(3, tensor.neuron_len)).astype(np.float32)
    )
    with MlxNintModel.from_mfq(path) as model:
        actual = _array(model.linear("model.layers.0.mlp.experts.gate_up_proj.weight")(source))
        canonical = io.unpack_tensor_payload(
            "NEPQ1-S",
            model.tensors.read_blob("model.layers.0.mlp.experts.gate_up_proj.weight"),
        )
        decoded = dequantize_nepq(canonical).reshape(-1, tensor.neuron_len)
        expected = (source @ decoded.T).reshape(3, tensor.n_experts, tensor.out_per_expert)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        assert not model.tensors._cache


@pytest.mark.parametrize("spec", [NEPQ0_A, NEPQ1_A])
def test_nepq_a_mmap_runtime_loads_packed_weight(spec, tmp_path):
    tensor, _ = _nepq_a_tensor(spec)
    name = "model.layers.0.mlp.experts.gate_up_proj.weight"
    path = tmp_path / f"metal-{spec.label}.mfq"
    io.save(
        path,
        FileHeader(model_arch="metal-nepq-a", num_tensors=1),
        {name: tensor},
    )
    source = (
        np.random.default_rng(20260891 + spec.profile_id)
        .normal(0.0, 0.1, size=(3, tensor.neuron_len))
        .astype(np.float16)
    )
    signs = rotation_signs(
        tensor.neuron_len,
        tensor.rotation_block,
        tensor.rotation_seed,
    )
    transformed = _fwht_reference(
        source,
        tensor.rotation_block,
        signs,
    ).astype(np.float32)
    dense = dequantize_nepq(tensor).reshape(-1, tensor.neuron_len)
    expected = (transformed @ dense.T).reshape(
        3,
        tensor.n_experts,
        tensor.out_per_expert,
    )
    with MlxNintModel.from_mfq(path) as model:
        layer = model.linear(name)
        assert isinstance(layer, MlxVqLinear)
        actual = _array(layer(source)).astype(np.float32)
        relative = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        assert relative < 1.5e-3
        assert not model.tensors._cache
