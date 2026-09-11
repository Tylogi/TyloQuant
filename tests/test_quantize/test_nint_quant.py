"""Tensor-level quantization tests for nint_quant."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mfq.formats import io
from mfq.formats.nint import NintSpec
from mfq.quantize import nint_quant


def test_quantize_preserves_shape():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.05, size=(64, 480)).astype(np.float32)
    t = nint_quant.quantize(W, NintSpec(4, 24, 6), axis=0)
    assert t.shape == W.shape
    r = nint_quant.dequantize(t)
    assert r.shape == W.shape


def test_axis_1():
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.05, size=(480, 64)).astype(np.float32)
    t = nint_quant.quantize(W, NintSpec(4, 16, 6), axis=1)
    assert t.shape == W.shape and t.axis == 1
    r = nint_quant.dequantize(t)
    assert r.shape == W.shape
    assert np.isfinite(r).all()


def test_gaussian_snr():
    rng = np.random.default_rng(2)
    W = rng.normal(0, 0.05, size=(128, 528)).astype(np.float32)
    t = nint_quant.quantize(W, NintSpec(4, 24, 6), axis=0)
    r = nint_quant.dequantize(t)
    err = W - r
    snr = 10 * np.log10((W ** 2).sum() / (err ** 2).sum())
    assert snr > 20.0


def test_rejects_1d():
    with pytest.raises(ValueError):
        nint_quant.quantize(np.zeros(64, dtype=np.float32), NintSpec())


def test_rejects_nonfinite_weights_and_fp16_metadata_overflow() -> None:
    weight = np.zeros((2, 48), dtype=np.float32)
    weight[0, 0] = np.nan
    with pytest.raises(ValueError, match="weights must be finite"):
        nint_quant.quantize(weight, NintSpec(4, 24, 6))

    tensor = nint_quant.quantize(np.zeros((2, 48), dtype=np.float32), NintSpec(4, 24, 6))
    tensor.neuron_scale[0] = 1e10
    with pytest.raises(ValueError, match="FP16 storage"):
        io.pack_nint(tensor)


def test_allzero_row_stays_finite():
    W = np.zeros((4, 48), dtype=np.float32)
    W[0] = np.random.default_rng(3).normal(0, 0.05, 48)
    t = nint_quant.quantize(W, NintSpec(4, 24, 6), axis=0)
    r = nint_quant.dequantize(t)
    assert np.isfinite(r).all()


def test_naq_sub_bits_allocator_preserves_budget_and_transfers_to_important_rows():
    allocated = nint_quant.allocate_row_sub_bits(
        np.asarray([1.0, 2.0, 3.0, 100.0], dtype=np.float32), 6
    )

    np.testing.assert_array_equal(allocated, [5, 5, 6, 8])
    assert float(allocated.mean()) == 6.0

    np.testing.assert_array_equal(
        nint_quant.allocate_row_sub_bits(np.zeros(5, dtype=np.float32), 6),
        np.full(5, 6, dtype=np.uint8),
    )
    capped = nint_quant.allocate_row_sub_bits(
        np.asarray([1.0, 2.0, 100.0], dtype=np.float32), 7
    )
    assert int(capped.min()) >= 6
    assert int(capped.max()) <= 8
    assert float(capped.mean()) == 7.0
    with pytest.raises(ValueError, match="greater than one"):
        nint_quant.allocate_row_sub_bits(
            np.ones(4, dtype=np.float32), 6, min_gain_ratio=1.0
        )


def test_mixed_sub_bits_quantization_round_trips_cpu_and_torch():
    rng = np.random.default_rng(41)
    weight = rng.normal(0, 0.05, size=(4, 73)).astype(np.float32)
    importance = np.geomspace(0.1, 10.0, weight.shape[1]).astype(np.float32)
    row_sub_bits = np.asarray([5, 6, 7, 6], dtype=np.uint8)
    spec = NintSpec(4, 24, 6)

    cpu = nint_quant.quantize(
        weight,
        spec,
        importance=importance,
        row_sub_bits=row_sub_bits,
    )
    from mfq.quantize.nint_quant_torch import quantize_axis0

    torch_encoded, row_sse = quantize_axis0(
        torch.from_numpy(weight),
        spec,
        device="cpu",
        importance=importance,
        use_priority_group_refinement=False,
        row_sub_bits=row_sub_bits,
        return_row_sse=True,
    )

    np.testing.assert_array_equal(cpu.row_sub_bits, row_sub_bits)
    np.testing.assert_array_equal(torch_encoded.row_sub_bits, row_sub_bits)
    np.testing.assert_allclose(
        nint_quant.dequantize(io.unpack_nint(io.pack_nint(cpu))),
        nint_quant.dequantize(cpu),
        rtol=0,
        atol=0,
    )
    assert np.isfinite(nint_quant.dequantize(torch_encoded)).all()
    assert tuple(row_sse.shape) == (weight.shape[0],)
    assert torch.isfinite(row_sse).all()


def test_mixed_q_and_sub_bits_quantization_round_trips_cpu_and_torch():
    rng = np.random.default_rng(43)
    weight = rng.normal(0, 0.05, size=(4, 73)).astype(np.float32)
    row_q_bits = np.asarray([2, 3, 5, 6], dtype=np.uint8)
    row_sub_bits = np.asarray([5, 6, 7, 6], dtype=np.uint8)
    spec = NintSpec(4, 24, 6)

    cpu = nint_quant.quantize(
        weight,
        spec,
        row_q_bits=row_q_bits,
        row_sub_bits=row_sub_bits,
    )
    from mfq.quantize.nint_quant_torch import quantize_axis0

    torch_encoded, row_sse = quantize_axis0(
        torch.from_numpy(weight),
        spec,
        device="cpu",
        use_priority_group_refinement=False,
        row_q_bits=row_q_bits,
        row_sub_bits=row_sub_bits,
        return_row_sse=True,
    )

    for encoded in (cpu, torch_encoded):
        np.testing.assert_array_equal(encoded.row_q_bits, row_q_bits)
        np.testing.assert_array_equal(encoded.row_sub_bits, row_sub_bits)
        assert np.isfinite(nint_quant.dequantize(encoded)).all()
    np.testing.assert_allclose(
        nint_quant.dequantize(io.unpack_nint(io.pack_nint(cpu))),
        nint_quant.dequantize(cpu),
        rtol=0,
        atol=0,
    )
    assert tuple(row_sse.shape) == (weight.shape[0],)
    assert torch.isfinite(row_sse).all()


@pytest.mark.parametrize(
    "row_q_bits",
    (
        np.asarray([0, 4], dtype=np.uint8),
        np.asarray([4, 9], dtype=np.uint8),
        np.asarray([4.0, 4.0], dtype=np.float32),
        np.asarray([4], dtype=np.uint8),
    ),
)
def test_mixed_q_bits_reject_invalid_descriptors(row_q_bits):
    with pytest.raises(ValueError, match="row_q_bits"):
        nint_quant.quantize(
            np.zeros((2, 48), dtype=np.float32),
            NintSpec(4, 24, 6),
            row_q_bits=row_q_bits,
        )


def test_naq_sub_bits_redistribution_improves_fixed_budget_weighted_sse():
    rng = np.random.default_rng(42)
    weight = rng.normal(0, 0.05, size=(64, 528)).astype(np.float32)
    neuron_importance = np.exp(rng.normal(0, 1.5, size=64)).astype(np.float32)
    neuron_importance /= neuron_importance.mean()
    spec = NintSpec(4, 24, 6)
    row_sub_bits = nint_quant.allocate_row_sub_bits(neuron_importance, spec.sub_bits)

    uniform = nint_quant.quantize(weight, spec)
    adaptive = nint_quant.quantize(
        weight,
        spec,
        row_sub_bits=row_sub_bits,
    )
    uniform_sse = float(
        (neuron_importance[:, None] * (nint_quant.dequantize(uniform) - weight) ** 2).sum()
    )
    adaptive_sse = float(
        (neuron_importance[:, None] * (nint_quant.dequantize(adaptive) - weight) ** 2).sum()
    )

    assert row_sub_bits.mean() == spec.sub_bits
    assert adaptive_sse < uniform_sse
