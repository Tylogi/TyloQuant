from __future__ import annotations

import numpy as np
import pytest
import torch

from mfq.formats.nvq import NVQ2_E8
from mfq.quantize.nvq_quant import dequantize as dequantize_nvq
from mfq.quantize.nvq_quant_torch import quantize_axis0
from mfq.quantize.nvq_jsc import (
    NvqJscConfig,
    dequantize_nvq_jsc,
    initial_jsc_tables,
    quantize_nvq_jsc_fixed,
)
from mfq.quantize.second_order import (
    activation_regressed_gain,
    apply_neuron_gain,
    diagonal_regressed_gain,
    linear_output_metrics,
    refine_nvq2_block24,
    refine_nvq2j_block24,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _block_independent_activations(tokens: int, width: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260718)
    value = torch.zeros((tokens, width), dtype=torch.float32)
    midpoint = tokens // 2
    value[:midpoint, :24] = torch.randn((midpoint, 24), generator=generator)
    value[midpoint:, 24:48] = torch.randn((tokens - midpoint, 24), generator=generator)
    return value


def test_activation_regressed_gain_reduces_training_output_error() -> None:
    generator = torch.Generator().manual_seed(17)
    weight = 0.05 * torch.randn((8, 48), generator=generator)
    activations = _block_independent_activations(128, 48)
    importance = activations.square().mean(0).numpy()
    encoded = quantize_axis0(weight, NVQ2_E8, importance=importance, search_steps=5)
    reconstruction = dequantize_nvq(encoded)
    before = linear_output_metrics(weight, reconstruction, activations)
    gain = activation_regressed_gain(weight, reconstruction, activations)
    calibrated = apply_neuron_gain(encoded, gain)
    after = linear_output_metrics(weight, dequantize_nvq(calibrated), activations)
    assert after.nmse_percent <= before.nmse_percent
    assert np.isfinite(gain).all()
    assert np.all(gain >= 0)
    assert calibrated.neuron_scale.dtype == np.float32


def test_diagonal_regressed_gain_is_rowwise_weighted_least_squares() -> None:
    reference = np.asarray([[1.0, 2.0], [3.0, -1.0]], dtype=np.float32)
    quantized = np.asarray([[0.5, 1.0], [2.0, -2.0]], dtype=np.float32)
    importance = np.asarray([1.0, 4.0], dtype=np.float32)
    gain = diagonal_regressed_gain(reference, quantized, importance)
    expected = np.asarray(
        [
            (1.0 * 1.0 * 0.5 + 4.0 * 2.0 * 1.0) / (1.0 * 0.25 + 4.0),
            (1.0 * 3.0 * 2.0 + 4.0 * -1.0 * -2.0) / (1.0 * 4.0 + 4.0 * 4.0),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(gain, expected)


def test_block24_refinement_reduces_block_diagonal_output_error() -> None:
    generator = torch.Generator().manual_seed(23)
    weight = 0.05 * torch.randn((6, 48), generator=generator)
    activations = _block_independent_activations(128, 48)
    importance = activations.square().mean(0).numpy()
    encoded = quantize_axis0(weight, NVQ2_E8, importance=importance, search_steps=5)
    before = linear_output_metrics(weight, dequantize_nvq(encoded), activations)
    refined, history = refine_nvq2_block24(
        weight,
        encoded,
        activations,
        outer_iterations=1,
        coordinate_sweeps=1,
        group_chunk=32,
    )
    after = linear_output_metrics(weight, dequantize_nvq(refined), activations)
    assert len(history) == 1
    assert after.nmse_percent <= before.nmse_percent * (1.0 + 1e-5)
    assert history[0].train_output_nmse_percent == pytest.approx(
        after.nmse_percent,
        rel=2e-5,
    )


def test_jsc_block24_refinement_reduces_training_output_error() -> None:
    generator = torch.Generator().manual_seed(29)
    weight = 0.05 * torch.randn((4, 48), generator=generator)
    activations = _block_independent_activations(96, 48)
    tensor = quantize_nvq_jsc_fixed(
        weight,
        initial_jsc_tables(NvqJscConfig(banks=2)),
        assignment_refine_steps=1,
        search_steps=3,
        group_chunk=32,
        device="cuda",
    )
    before = linear_output_metrics(weight, dequantize_nvq_jsc(tensor), activations)
    refined, history = refine_nvq2j_block24(
        weight,
        tensor,
        activations,
        outer_iterations=1,
        coordinate_sweeps=1,
        group_chunk=16,
    )
    after = linear_output_metrics(weight, dequantize_nvq_jsc(refined), activations)
    assert len(history) == 1
    assert after.nmse_percent <= before.nmse_percent * (1.0 + 2e-4)
    assert history[0].train_output_nmse_percent == pytest.approx(
        after.nmse_percent,
        rel=3e-5,
    )
