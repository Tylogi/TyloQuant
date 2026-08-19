"""Regression tests for quantization boundary conditions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mfq.formats.nvq import NvqSpec
from mfq.calibration.inint import _solve_milp
from mfq.quantize.nvq_quant import _indices_at_scale, quantize
from mfq.quantize.nvq_quant_torch import quantize_axis0
from mfq.quantize.second_order import refine_nvq2_block24


def test_extended_codebook_search_preserves_indices_above_uint8() -> None:
    cross = np.zeros((1, 512), dtype=np.float32)
    cross[0, 300] = 1.0
    quad = np.zeros_like(cross)
    indices = _indices_at_scale(
        cross,
        quad,
        np.ones(1, dtype=np.float32),
    )
    assert indices.dtype == np.uint16
    assert int(indices[0]) == 300


def test_block24_refinement_rejects_index_parity() -> None:
    rng = np.random.default_rng(15)
    weight = rng.normal(0, 0.05, size=(2, 24)).astype(np.float32)
    spec = NvqSpec(
        "e8_256",
        groupsize=24,
        sub_bits=4,
        sign_mode="index_parity",
    )
    encoded = quantize(
        weight,
        spec,
        search_steps=1,
        group_chunk=8,
    )
    with pytest.raises(ValueError, match="index-parity"):
        refine_nvq2_block24(
            weight,
            encoded,
            np.zeros((2, 24), dtype=np.float32),
            device="cpu",
        )


def test_torch_quantizer_rejects_unsupported_index_parity() -> None:
    spec = NvqSpec(
        "e8_256",
        groupsize=24,
        sub_bits=4,
        sign_mode="index_parity",
    )
    with pytest.raises(ValueError, match="index-parity"):
        quantize_axis0(
            torch.zeros((1, 24), dtype=torch.float32),
            spec,
            device="cpu",
        )


@pytest.mark.parametrize("shape", [(0, 24), (1, 0)])
def test_torch_quantizer_rejects_empty_dimensions(shape: tuple[int, int]) -> None:
    spec = NvqSpec("e8_256", groupsize=24, sub_bits=4)
    with pytest.raises(ValueError, match="dimensions must be positive"):
        quantize_axis0(torch.zeros(shape), spec, device="cpu")


def test_inint_milp_tightens_constraint_after_rounding_overshoot(monkeypatch) -> None:
    from scipy import optimize

    upper_bounds: list[float] = []

    class _Result:
        success = True
        message = ""

        def __init__(self, x: np.ndarray) -> None:
            self.x = x

    def fake_milp(*, constraints, **_kwargs):
        upper_bounds.append(float(constraints.ub[0]))
        if len(upper_bounds) == 1:
            return _Result(np.asarray([1.0, 1.0]))
        return _Result(np.asarray([1.0, 0.0]))

    monkeypatch.setattr(optimize, "milp", fake_milp)
    selected = _solve_milp(
        np.asarray([2.0, 1.0]),
        np.asarray([6, 5], dtype=np.int64),
        10,
    )
    np.testing.assert_array_equal(selected, np.asarray([True, False]))
    assert upper_bounds[1] < upper_bounds[0]
