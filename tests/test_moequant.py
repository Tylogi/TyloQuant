from pathlib import Path

import numpy as np

from mfq.formats.nint import NintSpec
from mfq.quantize.moequant import (
    ExpertAffinityAccumulator,
    diagonal_second_moment,
)
from mfq.quantize.nint_quant import dequantize, quantize


def test_diagonal_second_moment_matches_moequant_objective() -> None:
    inputs = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    affinity = np.asarray([0.25, 2.0], dtype=np.float32)

    count = diagonal_second_moment(inputs, normalize=False)
    agq = diagonal_second_moment(inputs, affinity, normalize=False)

    np.testing.assert_array_equal(count, [10.0, 20.0])
    np.testing.assert_array_equal(agq, [18.25, 33.0])


def test_accumulator_streaming_merge_and_roundtrip(tmp_path: Path) -> None:
    inputs = np.asarray(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        dtype=np.float32,
    )
    ids = np.asarray([0, 1, 0, 1], dtype=np.int64)
    affinity = np.asarray([0.5, 0.25, 1.5, 2.0], dtype=np.float32)
    first = ExpertAffinityAccumulator.create(2, 2)
    second = ExpertAffinityAccumulator.create(2, 2)
    first.update(inputs[:2], ids[:2], affinity[:2])
    second.update(inputs[2:], ids[2:], affinity[2:])
    first.merge(second)

    np.testing.assert_array_equal(first.route_counts, [2, 2])
    np.testing.assert_allclose(first.affinity_sums, [2.0, 2.25])
    np.testing.assert_allclose(first.input_sum2, [[26.0, 40.0], [58.0, 80.0]])
    np.testing.assert_allclose(
        first.affinity_input_sum2,
        [[38.0, 56.0], [100.25, 132.0]],
    )

    path = tmp_path / "layer0-gate-up-agq.npz"
    first.save(path, metadata={"layer": 0, "projection": "gate_up"})
    restored, metadata = ExpertAffinityAccumulator.load(path)
    np.testing.assert_array_equal(restored.route_counts, first.route_counts)
    np.testing.assert_array_equal(restored.input_sum2, first.input_sum2)
    np.testing.assert_array_equal(
        restored.affinity_input_sum2, first.affinity_input_sum2
    )
    assert metadata == {"layer": 0, "projection": "gate_up"}


def test_agq_q4_improves_heldout_affinity_weighted_error() -> None:
    rng = np.random.default_rng(731)
    weight = rng.normal(0.0, 0.4, size=(6, 32)).astype(np.float32)
    weight[:, :8] *= 4.0
    train = rng.normal(size=(96, 32)).astype(np.float32)
    heldout = rng.normal(size=(64, 32)).astype(np.float32)
    affinity = np.full(96, 0.02, dtype=np.float32)
    affinity[:48] = 1.0
    train[:48, :8] *= 0.08
    train[:48, 8:] *= 2.0
    heldout[:, :8] *= 0.08
    heldout[:, 8:] *= 2.0
    count = diagonal_second_moment(train)
    agq = diagonal_second_moment(train, affinity)
    spec = NintSpec(bits=4, groupsize=16, sub_bits=5)

    count_weight = dequantize(quantize(weight, spec, importance=count))
    agq_weight = dequantize(quantize(weight, spec, importance=agq))
    reference = heldout @ weight.T
    count_error = np.square(reference - heldout @ count_weight.T).sum()
    agq_error = np.square(reference - heldout @ agq_weight.T).sum()

    assert agq_error < count_error
