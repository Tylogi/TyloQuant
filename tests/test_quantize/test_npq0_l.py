from __future__ import annotations

import numpy as np
import pytest
import torch

from mfq.formats.npq0_l import pack_npq0_l, unpack_npq0_l
from mfq.quantize.npq0_l import (
    Npq0LConfig,
    dequantize_npq0_l,
    npq0_l_tables_from_tensor,
    quantize_npq0_l_fixed,
    train_npq0_l,
)


def _config() -> Npq0LConfig:
    return Npq0LConfig(
        iterations=1,
        assignment_refine_steps=1,
        fixed_refine_steps=1,
        kmeans_iterations=2,
        kmeans_initialization_points=128,
        group_chunk=16,
        anchor_multipliers=(0.8, 1.0, 1.25),
        seed=17,
    )


def test_npq0_l_cpu_training_and_fixed_table_assignment() -> None:
    generator = torch.Generator().manual_seed(18)
    train = 0.04 * torch.randn((8, 48), generator=generator)
    validation = 0.04 * torch.randn((4, 48), generator=generator)
    trained, history = train_npq0_l(train, config=_config(), device="cpu")
    fixed = quantize_npq0_l_fixed(
        validation,
        npq0_l_tables_from_tensor(trained),
        config=_config(),
        device="cpu",
    )
    reconstruction = dequantize_npq0_l(fixed)
    assert reconstruction.shape == tuple(validation.shape)
    assert np.isfinite(reconstruction).all()
    assert min(item.weighted_sse for item in history) <= history[0].weighted_sse
    signal = float(validation.square().sum())
    error = float(np.square(reconstruction - validation.numpy()).sum())
    assert 100.0 * error / signal < 45.0


def test_npq0_l_serialized_reconstruction_preserves_training_error() -> None:
    weight = torch.linspace(-0.07, 0.08, 6 * 40, dtype=torch.float32).reshape(6, 40)
    tensor, history = train_npq0_l(weight, config=_config(), device="cpu")
    restored = unpack_npq0_l(pack_npq0_l(tensor))
    reconstruction = dequantize_npq0_l(restored)
    error = float(np.square(reconstruction - weight.numpy()).sum())
    assert error == pytest.approx(min(item.weighted_sse for item in history), rel=2e-3, abs=1e-7)


def test_npq0_l_rejects_non_vec8_width() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        train_npq0_l(torch.zeros((2, 34)), config=_config(), device="cpu")
