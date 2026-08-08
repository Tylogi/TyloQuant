from __future__ import annotations

import numpy as np
import torch

from mfq.quantize.nvq_additive import (
    NvqAdditiveConfig,
    _maximum_additive_code,
    additive_tables_from_tensor,
    dequantize_nvq_additive,
    quantize_nvq_additive_fixed,
    train_nvq_additive,
)
from mfq.quantize.nvq_jsc import NvqJscConfig, train_nvq_jsc


def _configs() -> tuple[NvqJscConfig, NvqAdditiveConfig]:
    return (
        NvqJscConfig(
            banks=1,
            iterations=0,
            assignment_refine_steps=1,
            search_steps=3,
            group_chunk=16,
        ),
        NvqAdditiveConfig(
            banks=1,
            iterations=1,
            assignment_refine_steps=1,
            fixed_refine_steps=1,
            kmeans_iterations=1,
            kmeans_initialization_points=64,
            beam_size=2,
            pair_refine_steps=1,
            group_chunk=8,
            anchor_multipliers=(0.75, 1.0, 1.25),
            seed=31,
        ),
    )


def test_nvq_additive_maximum_uses_legal_paired_codes() -> None:
    first = np.zeros((1, 256, 8), dtype=np.int8)
    second = np.zeros((1, 128, 8), dtype=np.int8)
    first[0, 0, 0] = 10
    second[0, 0, 0] = -7
    second[0, 1, 0] = 2
    assert _maximum_additive_code(first, second) == 12


def test_nvq_additive_roundtrip_and_rate_on_cpu() -> None:
    generator = torch.Generator().manual_seed(31)
    weight = 0.04 * torch.randn((8, 48), generator=generator)
    importance = np.linspace(0.5, 1.5, 48, dtype=np.float32)
    jsc_config, additive_config = _configs()
    initial, _ = train_nvq_jsc(weight, importance=importance, config=jsc_config, device="cpu")
    additive, history = train_nvq_additive(
        weight,
        initial,
        importance=importance,
        config=additive_config,
        device="cpu",
    )
    reconstruction = dequantize_nvq_additive(additive)

    assert reconstruction.shape == tuple(weight.shape)
    assert np.isfinite(reconstruction).all()
    assert additive.first_codebooks.shape == (1, 256, 8)
    assert additive.second_codebooks.shape == (1, 128, 8)
    assert int(additive.second_indices.max()) < 128
    assert len(history) == 2
    assert min(item.weighted_sse for item in history) <= history[0].weighted_sse

    baseline_streams = initial.payload_nbytes - 64 - initial.codebooks.size
    additive_tables = 16 * 2 + 16 + additive.first_codebooks.size + additive.second_codebooks.size
    assert additive.payload_nbytes - additive_tables == baseline_streams
    assert additive.payload_nbytes > initial.payload_nbytes


def test_nvq_additive_fixed_tables_are_reused() -> None:
    generator = torch.Generator().manual_seed(37)
    train = 0.03 * torch.randn((8, 48), generator=generator)
    validation = 0.03 * torch.randn((4, 48), generator=generator)
    jsc_config, additive_config = _configs()
    initial, _ = train_nvq_jsc(train, config=jsc_config, device="cpu")
    trained, _ = train_nvq_additive(train, initial, config=additive_config, device="cpu")
    fixed = quantize_nvq_additive_fixed(
        validation,
        additive_tables_from_tensor(trained),
        config=additive_config,
        device="cpu",
    )
    reconstruction = dequantize_nvq_additive(fixed)

    assert fixed.shape == tuple(validation.shape)
    assert np.isfinite(reconstruction).all()
    np.testing.assert_array_equal(fixed.first_codebooks, trained.first_codebooks)
    np.testing.assert_array_equal(fixed.second_codebooks, trained.second_codebooks)
