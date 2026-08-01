from __future__ import annotations

import numpy as np
import torch

from mfq.formats.nvq1_l import IQ1S_TERNARY_2048
from mfq.quantize.nvq_lowbit_additive import (
    NvqLowBitAdditiveConfig,
    NvqLowBitAdditiveTables,
    combined_codebooks,
    dequantize_lowbit_additive,
    projected_nbytes,
    quantize_lowbit_additive_fixed,
    train_lowbit_additive,
)


def _config() -> NvqLowBitAdditiveConfig:
    return NvqLowBitAdditiveConfig(
        index_bits=11,
        first_bits=6,
        sub_bits=3,
        delta=0.125,
        banks=1,
        iterations=1,
        fixed_refine_steps=1,
        kmeans_iterations=2,
        kmeans_initialization_points=64,
        group_chunk=8,
        anchor_multipliers=(0.75, 1.0),
        seed=17,
    )


def test_combined_codebook_index_order() -> None:
    config = _config()
    first = np.zeros((1, 64, 8), dtype=np.int8)
    second = np.zeros((1, 32, 8), dtype=np.int8)
    first[0, 3] = 2
    second[0, 5] = -1
    table = combined_codebooks(
        NvqLowBitAdditiveTables(np.asarray([0.25], dtype=np.float32), first, second),
        config,
    )
    np.testing.assert_array_equal(table[0, 3 * 32 + 5], np.full(8, 0.25, dtype=np.float32))


def test_train_and_fixed_assignment_roundtrip() -> None:
    config = _config()
    rng = np.random.default_rng(23)
    weight = rng.normal(0.0, 0.04, size=(4, 48)).astype(np.float32)
    trained, history = train_lowbit_additive(
        torch.from_numpy(weight),
        IQ1S_TERNARY_2048[None, :, :],
        config=config,
        device="cpu",
    )
    assert len(history) == 2
    assert trained.first_indices.max() < 64
    assert trained.second_indices.max() < 32
    fixed = quantize_lowbit_additive_fixed(
        torch.from_numpy(weight),
        NvqLowBitAdditiveTables(
            trained.codebook_step,
            trained.first_codebooks,
            trained.second_codebooks,
        ),
        config=config,
        device="cpu",
    )
    reconstruction = dequantize_lowbit_additive(fixed, config)
    assert reconstruction.shape == weight.shape
    assert np.isfinite(reconstruction).all()
    assert np.mean(np.square(weight - reconstruction)) < np.mean(np.square(weight))


def test_projected_rate_only_adds_codebook_metadata() -> None:
    config = _config()
    baseline = projected_nbytes(4096, 4096, config, include_codebooks=False)
    additive = projected_nbytes(4096, 4096, config, include_codebooks=True)
    assert additive - baseline == (64 + 32) * 8 + 2
