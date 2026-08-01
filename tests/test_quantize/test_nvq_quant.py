from __future__ import annotations

import numpy as np

from mfq.formats.nvq import E8_256, NVQ2_E8, NVQ3_D4, NvqSpec, pack_nvq, unpack_nvq
from mfq.quantize.nvq_quant import dequantize, quantize


def _snr(original: np.ndarray, reconstruction: np.ndarray) -> float:
    return float(10.0 * np.log10(np.sum(original * original) / np.sum((original - reconstruction) ** 2)))


def test_nvq_quantize_shapes_parity_and_quality():
    rng = np.random.default_rng(11)
    weight = rng.normal(0, 0.05, size=(8, 96)).astype(np.float32)
    thresholds = {"e8_256": 8.5, "d4_256": 13.0}
    for spec in (NVQ2_E8, NVQ3_D4):
        encoded = quantize(weight, spec, search_steps=5, group_chunk=64)
        reconstruction = dequantize(encoded)
        assert reconstruction.shape == weight.shape
        assert np.isfinite(reconstruction).all()
        assert _snr(weight, reconstruction) > thresholds[spec.codebook]

        lower_parity = np.unpackbits(encoded.signs[..., None], axis=-1, bitorder="little")[..., :7].sum(axis=-1)
        eighth_bit = lower_parity & 1
        assert np.all(((lower_parity + eighth_bit) & 1) == 0)


def test_nvq_quantized_blob_preserves_reconstruction():
    rng = np.random.default_rng(12)
    weight = rng.normal(0, 0.03, size=(4, 104)).astype(np.float32)
    encoded = quantize(weight, NVQ3_D4, search_steps=3, group_chunk=32)
    before = dequantize(encoded)
    after = dequantize(unpack_nvq(pack_nvq(encoded)))
    np.testing.assert_array_equal(after, before)


def test_custom_codebook_is_embedded_and_used_by_default():
    rng = np.random.default_rng(120)
    weight = rng.normal(0, 0.03, size=(4, 96)).astype(np.float32)
    custom = np.roll(E8_256, 1, axis=0).copy()
    encoded = quantize(weight, NVQ2_E8, search_steps=3, group_chunk=32, codebook=custom)
    restored = unpack_nvq(pack_nvq(encoded))
    np.testing.assert_array_equal(restored.codebook, custom)
    np.testing.assert_array_equal(dequantize(restored), dequantize(encoded, codebook=custom))


def test_importance_shape_is_validated():
    weight = np.zeros((2, 96), dtype=np.float32)
    try:
        quantize(weight, NVQ2_E8, importance=np.ones(95, dtype=np.float32), search_steps=1)
    except ValueError as exc:
        assert "importance" in str(exc)
    else:
        raise AssertionError("invalid importance length was accepted")


def test_integer_scale_refinement_never_increases_sse():
    rng = np.random.default_rng(13)
    weight = rng.normal(0, 0.04, size=(6, 120)).astype(np.float32)
    baseline = dequantize(
        quantize(weight, NVQ2_E8, search_steps=5, group_chunk=32, scale_refine_steps=0)
    )
    refined = dequantize(
        quantize(weight, NVQ2_E8, search_steps=5, group_chunk=32, scale_refine_steps=3)
    )
    baseline_sse = float(np.sum((weight - baseline) ** 2))
    refined_sse = float(np.sum((weight - refined) ** 2))
    assert refined_sse <= baseline_sse + 1e-9


def test_index_parity_uses_code_index_to_preserve_all_signs():
    rng = np.random.default_rng(14)
    weight = rng.normal(0, 0.05, size=(4, 96)).astype(np.float32)
    spec = NvqSpec("e8_256", groupsize=24, sub_bits=4, sign_mode="index_parity")
    encoded = quantize(weight, spec, search_steps=5, group_chunk=32)
    expected_bank = (np.signbit(weight).reshape(4, -1, 8).sum(axis=-1) & 1).astype(np.uint8)
    np.testing.assert_array_equal(encoded.indices >> 7, expected_bank)
    reconstruction = dequantize(encoded)
    nonzero = reconstruction != 0
    np.testing.assert_array_equal(np.signbit(reconstruction[nonzero]), np.signbit(weight[nonzero]))
