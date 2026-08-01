from __future__ import annotations

import numpy as np

from mfq.formats.nvq1_l import IQ1S_TERNARY_2048, NVQ1_L_T8_S3, pack_nvq1_l, unpack_nvq1_l
from mfq.formats.ternary import (
    NEURON_TERNARY_S3,
    pack_neuron_ternary,
    unpack_neuron_ternary,
)
from mfq.quantize.nvq1_l_quant import dequantize as dequantize_nvq1_l
from mfq.quantize.nvq1_l_quant import quantize as quantize_nvq1_l
from mfq.quantize.ternary_quant import dequantize as dequantize_ternary
from mfq.quantize.ternary_quant import quantize as quantize_ternary


def _snr(original: np.ndarray, reconstruction: np.ndarray) -> float:
    signal = np.sum(original * original)
    error = np.sum((original - reconstruction) ** 2)
    return float(10.0 * np.log10(signal / error))


def test_nvq1_l_quantization_quality_and_blob_reconstruction():
    weight = np.random.default_rng(32).normal(0, 0.05, size=(4, 96)).astype(np.float32)
    encoded = quantize_nvq1_l(weight, NVQ1_L_T8_S3, group_chunk=16)
    reconstruction = dequantize_nvq1_l(encoded)
    assert reconstruction.shape == weight.shape
    assert np.isfinite(reconstruction).all()
    assert _snr(weight, reconstruction) > 6.0
    restored = dequantize_nvq1_l(unpack_nvq1_l(pack_nvq1_l(encoded)))
    np.testing.assert_array_equal(restored, reconstruction)


def test_custom_nvq1_l_codebook_is_embedded_and_used_by_default():
    weight = np.random.default_rng(320).normal(0, 0.05, size=(3, 48)).astype(np.float32)
    custom = np.roll(IQ1S_TERNARY_2048, 1, axis=0).copy()
    encoded = quantize_nvq1_l(weight, NVQ1_L_T8_S3, group_chunk=16, codebook=custom)
    restored = unpack_nvq1_l(pack_nvq1_l(encoded))
    np.testing.assert_array_equal(restored.codebook, custom)
    np.testing.assert_array_equal(
        dequantize_nvq1_l(restored),
        dequantize_nvq1_l(encoded, codebook=custom),
    )


def test_neuron_ternary_quality_and_blob_reconstruction():
    weight = np.random.default_rng(33).normal(0, 0.05, size=(4, 101)).astype(np.float32)
    encoded = quantize_ternary(weight, NEURON_TERNARY_S3)
    reconstruction = dequantize_ternary(encoded)
    assert reconstruction.shape == weight.shape
    assert np.isfinite(reconstruction).all()
    assert _snr(weight, reconstruction) > 6.0
    restored = dequantize_ternary(
        unpack_neuron_ternary(pack_neuron_ternary(encoded))
    )
    np.testing.assert_array_equal(restored, reconstruction)


def test_nvq1_l_and_ternary_encode_zero_rows_exactly():
    weight = np.zeros((2, 48), dtype=np.float32)
    nvq1_l = quantize_nvq1_l(weight, NVQ1_L_T8_S3, group_chunk=8)
    ternary = quantize_ternary(weight, NEURON_TERNARY_S3)
    np.testing.assert_array_equal(dequantize_nvq1_l(nvq1_l), weight)
    np.testing.assert_array_equal(dequantize_ternary(ternary), weight)


def test_nvq1_l_importance_length_is_validated():
    weight = np.zeros((2, 96), dtype=np.float32)
    try:
        quantize_nvq1_l(
            weight,
            NVQ1_L_T8_S3,
            importance=np.ones(95, dtype=np.float32),
            anchor_multipliers=(1.0,),
            refine_steps=0,
        )
    except ValueError as exc:
        assert "importance" in str(exc)
    else:
        raise AssertionError("invalid NVQ1-L importance length was accepted")
