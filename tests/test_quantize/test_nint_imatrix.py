from __future__ import annotations

import numpy as np
import pytest

from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import (
    _imatrix_element_weights,
    _make_qp,
    dequantize,
    quantize,
)


@pytest.mark.parametrize(
    "spec",
    (
        NintSpec(3, 24, 6),
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 7),
        NintSpec(6, 26, 7),
    ),
)
def test_no_imatrix_path_is_exactly_unchanged(spec: NintSpec):
    weight = np.random.default_rng(7).normal(0, 0.05, size=(13, 113)).astype(
        np.float32
    )

    original = quantize(weight, spec, axis=0)
    explicit_none = quantize(weight, spec, axis=0, importance=None)

    for field in (
        "q",
        "neuron_scale",
        "neuron_min",
        "sub_scale",
        "sub_min",
    ):
        np.testing.assert_array_equal(
            getattr(original, field), getattr(explicit_none, field)
        )


@pytest.mark.parametrize(
    "spec",
    (
        NintSpec(3, 24, 6),
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 7),
        NintSpec(6, 26, 7),
    ),
)
def test_imatrix_reduces_weighted_error(spec: NintSpec):
    weight = np.random.default_rng(123).normal(
        0, 0.08, size=(16, 113)
    ).astype(np.float32)
    importance = np.geomspace(0.001, 1000.0, weight.shape[1]).astype(np.float32)

    plain = dequantize(quantize(weight, spec, axis=0))
    weighted = dequantize(
        quantize(weight, spec, axis=0, importance=importance)
    )

    plain_error = float(np.sum(importance * (plain - weight) ** 2))
    weighted_error = float(np.sum(importance * (weighted - weight) ** 2))
    assert weighted_error < plain_error


def test_row_specific_imatrix_and_axis_are_supported():
    weight = np.random.default_rng(19).normal(0, 0.05, size=(11, 7)).astype(
        np.float32
    )
    importance = np.geomspace(0.01, 100.0, weight.size).reshape(weight.shape)
    encoded = quantize(
        weight,
        NintSpec(4, 24, 6),
        axis=1,
        importance=importance,
    )
    assert dequantize(encoded).shape == weight.shape


@pytest.mark.parametrize(
    "importance",
    (
        np.ones(12, dtype=np.float32),
        np.asarray([1.0, -1.0, 1.0], dtype=np.float32),
        np.asarray([1.0, np.nan, 1.0], dtype=np.float32),
    ),
)
def test_invalid_imatrix_is_rejected(importance: np.ndarray):
    with pytest.raises(ValueError, match="importance"):
        quantize(
            np.ones((3, 13), dtype=np.float32),
            NintSpec(4, 24, 6),
            importance=importance,
        )


def test_imatrix_sigma2_uses_256_value_superblocks():
    neuron_len = 300
    rows = np.zeros((1, 312), dtype=np.float32)
    rows[:, :256] = 1.0
    rows[:, 256:neuron_len] = 2.0
    importance = np.ones((1, neuron_len), dtype=np.float32)

    weights = _imatrix_element_weights(rows, importance, neuron_len)

    np.testing.assert_allclose(weights[0, 0], np.sqrt(3.0), rtol=1e-6)
    np.testing.assert_allclose(weights[0, 255], np.sqrt(3.0), rtol=1e-6)
    np.testing.assert_allclose(weights[0, 256], np.sqrt(12.0), rtol=1e-6)
    np.testing.assert_array_equal(
        weights[0, neuron_len:], np.zeros(rows.shape[1] - neuron_len)
    )


def test_weighted_neuron_scale_search_protects_important_groups():
    scales = np.asarray([[1.0, 2.0, 10.0]], dtype=np.float32)
    group_weights = np.asarray([[1000.0, 1000.0, 0.001]], dtype=np.float32)
    nmax = 7

    fitted_scale, levels = _make_qp(scales, group_weights, nmax=nmax)
    fitted = fitted_scale[:, None] * levels
    maximum = scales.max(axis=-1)
    baseline_scale = maximum / nmax
    baseline_levels = np.rint(
        nmax * scales / maximum[:, None]
    )
    baseline = baseline_scale[:, None] * baseline_levels

    fitted_error = np.sum(group_weights * (scales - fitted) ** 2)
    baseline_error = np.sum(group_weights * (scales - baseline) ** 2)
    assert fitted_error < baseline_error * 0.001
