from __future__ import annotations

import numpy as np

from mfq.formats.nvq1_s import NVQ1_S, pack_nvq1_s, unpack_nvq1_s
from mfq.quantize.nvq1_s_codebook import (
    Nvq1SCodebookTrainingConfig,
    Nvq1STrainingMatrix,
    initialize_nvq1_s_banks_from_full,
    train_nvq1_s_codebook,
)
from mfq.quantize.nvq1_s_quant import dequantize, quantize


def _snr(original: np.ndarray, reconstruction: np.ndarray) -> float:
    signal = np.sum(original * original)
    error = np.sum((original - reconstruction) ** 2)
    return float(10.0 * np.log10(signal / error))


def test_nvq1_s_quantization_and_blob_reconstruction():
    weight = np.random.default_rng(41).normal(0, 0.05, size=(4, 104)).astype(np.float32)
    encoded = quantize(weight, anchor_multipliers=(1.0,), refine_steps=1, group_chunk=16)
    reconstruction = dequantize(encoded)
    assert encoded.payload_bpw == NVQ1_S.bpw(104, out=4)
    assert reconstruction.shape == weight.shape
    assert np.isfinite(reconstruction).all()
    assert _snr(weight, reconstruction) > 5.0
    restored = dequantize(unpack_nvq1_s(pack_nvq1_s(encoded)))
    np.testing.assert_array_equal(restored, reconstruction)


def test_nvq1_s_rejects_non_vector_aligned_k():
    weight = np.zeros((2, 82), dtype=np.float32)
    try:
        quantize(weight)
    except ValueError as exc:
        assert "divisible by 8" in str(exc)
    else:
        raise AssertionError("NVQ1-S accepted a non-vector-aligned K")


def test_nvq1_s_codebook_training_reduces_training_sse():
    weight = np.random.default_rng(42).normal(0, 0.05, size=(4, 96)).astype(np.float32)
    config = Nvq1SCodebookTrainingConfig(
        iterations=1,
        anchor_multipliers=(1.0,),
        refine_steps=0,
        group_chunk=16,
        projection_candidates=4,
        reseed_pool_size=64,
    )
    table, history = train_nvq1_s_codebook(
        [Nvq1STrainingMatrix("synthetic", weight)],
        config,
    )
    assert table.shape == (512, 8)
    assert np.unique(table, axis=0).shape[0] == 512
    assert history[-1]["sse"] <= history[0]["sse"]


def test_nvq1_s_full_table_initializer_produces_two_unique_banks():
    weight = np.random.default_rng(43).normal(0, 0.05, size=(2, 256)).astype(np.float32)
    table = initialize_nvq1_s_banks_from_full(
        [Nvq1STrainingMatrix("synthetic", weight)],
        anchor_multipliers=(1.0,),
        refine_steps=0,
        group_chunk=16,
    )
    assert table.shape == (2, 512, 8)
    assert np.unique(table[0], axis=0).shape[0] == 512
    assert np.unique(table[1], axis=0).shape[0] == 512
