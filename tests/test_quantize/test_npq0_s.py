from __future__ import annotations

import numpy as np
import pytest
import torch

from mfq.formats.npq0_s import pack_npq0_s, unpack_npq0_s
from mfq.quantize.npq0_s import (
    Npq0SConfig,
    dequantize_npq0_s,
    npq0_s_tables_from_tensor,
    quantize_npq0_s_fixed,
    train_npq0_s,
)


def _config() -> Npq0SConfig:
    return Npq0SConfig(
        iterations=1,
        assignment_refine_steps=1,
        fixed_refine_steps=1,
        kmeans_iterations=2,
        kmeans_initialization_points=128,
        group_chunk=16,
        anchor_multipliers=(0.8, 1.0, 1.25),
        seed=23,
    )


def test_npq0_s_cpu_training_and_fixed_table_assignment() -> None:
    generator = torch.Generator().manual_seed(24)
    train = 0.04 * torch.randn((12, 48), generator=generator)
    validation = 0.04 * torch.randn((4, 48), generator=generator)
    trained, history = train_npq0_s(train, config=_config(), device="cpu")
    fixed = quantize_npq0_s_fixed(
        validation,
        npq0_s_tables_from_tensor(trained),
        config=_config(),
        device="cpu",
    )
    reconstruction = dequantize_npq0_s(fixed)
    assert reconstruction.shape == tuple(validation.shape)
    assert np.isfinite(reconstruction).all()
    assert min(item.weighted_sse for item in history) <= history[0].weighted_sse
    signal = float(validation.square().sum())
    error = float(np.square(reconstruction - validation.numpy()).sum())
    assert 100.0 * error / signal < 65.0


def test_npq0_s_serialized_reconstruction_preserves_training_error() -> None:
    weight = torch.linspace(-0.07, 0.08, 8 * 40, dtype=torch.float32).reshape(8, 40)
    tensor, history = train_npq0_s(weight, config=_config(), device="cpu")
    restored = unpack_npq0_s(pack_npq0_s(tensor))
    reconstruction = dequantize_npq0_s(restored)
    error = float(np.square(reconstruction - weight.numpy()).sum())
    assert error == pytest.approx(min(item.weighted_sse for item in history), rel=2e-3, abs=1e-7)


def test_npq0_s_index_selects_a_cartesian_product_codeword() -> None:
    weight = torch.zeros((1, 24), dtype=torch.float32)
    tensor, _ = train_npq0_s(weight, config=_config(), device="cpu")
    tensor.neuron_scale[:] = 1.0
    tensor.scale_lut[:] = 1.0
    tensor.state[:] = 0
    tensor.first_codebooks.fill(0)
    tensor.second_codebooks.fill(0)
    for index in range(8):
        tensor.first_codebooks[0, index] = np.arange(4, dtype=np.int8) + 10 * index
        tensor.second_codebooks[0, index] = -np.arange(4, dtype=np.int8) - 10 * index

    first_indices = np.array([0, 3, 7], dtype=np.uint8)
    second_indices = np.array([2, 5, 7], dtype=np.uint8)
    tensor.indices[0] = first_indices | (second_indices << 3)
    expected = np.concatenate(
        (
            tensor.first_codebooks[0, first_indices],
            tensor.second_codebooks[0, second_indices],
        ),
        axis=1,
    ).reshape(1, 24)
    np.testing.assert_array_equal(dequantize_npq0_s(tensor), expected)


def test_npq0_s_rejects_non_vec8_width() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        train_npq0_s(torch.zeros((2, 34)), config=_config(), device="cpu")
