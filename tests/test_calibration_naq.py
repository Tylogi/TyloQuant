from pathlib import Path

import numpy as np
import pytest

from mfq.calibration.naq import (
    NaqIBook,
    NaqIMap,
    NaqImatrixCalibrator,
    NaqTensorBinding,
    TensorRateBudget,
)
from mfq.quantize.imatrix import ImportanceEntry, ImportanceMatrix


def test_naq_imap_keeps_input_and_neuron_factors_separate_and_compact():
    entry = ImportanceEntry(
        values=np.asarray([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=np.float32),
        counts=np.asarray([11, 13], dtype=np.int64),
        row_importance=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
    imap = NaqIMap.from_entry(
        entry,
        original_shape=(2, 2, 3),
        storage_shape=(4, 3),
        provenance={"entry_name": "experts.up.weight"},
    )

    assert imap.shape == (4, 3)
    assert imap.input_moments.shape == (2, 3)
    np.testing.assert_array_equal(imap.row_to_input, [0, 0, 1, 1])
    np.testing.assert_array_equal(
        imap.channel_importance(np.asarray([1, 2])),
        [[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]],
    )
    np.testing.assert_array_equal(imap.neuron_importance(slice(1, 3)), [2.0, 3.0])
    np.testing.assert_array_equal(
        imap.element_importance(np.asarray([1, 2])),
        [[2.0, 4.0, 8.0], [9.0, 15.0, 21.0]],
    )
    error = np.asarray([[1.0, -2.0, 0.5], [2.0, 1.0, -1.0]], dtype=np.float32)
    expected = (imap.element_importance([1, 2]) * np.square(error)).sum(axis=1)
    np.testing.assert_allclose(imap.weighted_row_loss(error, [1, 2]), expected)
    with pytest.raises(ValueError):
        imap.input_moments[0, 0] = 0


def test_naq_calibrator_builds_canonical_budget_aware_book():
    matrix = ImportanceMatrix(
        path=Path("unused.npz"),
        entries={
            "legacy.layer.weight": ImportanceEntry(
                values=np.asarray([[1.0, 3.0]], dtype=np.float32),
                counts=np.asarray([8], dtype=np.int64),
                row_importance=np.asarray([0.5, 1.5], dtype=np.float32),
            )
        },
        datasets=("calibration-a",),
        chunk_count=4,
        chunk_size=128,
        legacy=False,
        metadata={"format": "mfq.imatrix.v1"},
    )
    binding = NaqTensorBinding(
        tensor_key="model.layers.0.mlp.gate.weight",
        entry_names=("canonical.missing", "legacy.layer.weight"),
        original_shape=(2, 2),
        storage_shape=(2, 2),
    )
    budget = TensorRateBudget(target_bits=20, value_count=4)
    book = NaqImatrixCalibrator("end-to-end-kl:naq-v1").calibrate(
        matrix, (binding,), {binding.tensor_key: budget}
    )

    assert isinstance(book, NaqIBook)
    assert book.budget(binding.tensor_key).bpw == 5.0
    assert book.imap(binding.tensor_key).provenance["entry_name"] == "legacy.layer.weight"
    assert book.imap(binding.tensor_key).objective_fingerprint == "end-to-end-kl:naq-v1"
    book.validate_inventory([binding.tensor_key])
    assert book.fingerprint() == book.fingerprint()
    with pytest.raises(ValueError, match="inventory mismatch"):
        book.validate_inventory([binding.tensor_key, "model.layers.1.mlp.gate.weight"])


def test_naq_ibook_rejects_budget_shape_mismatch():
    imap = NaqIMap(
        tensor_shape=(2, 3),
        input_moments=np.ones((1, 3), dtype=np.float32),
        neuron_factors=np.ones(2, dtype=np.float32),
        row_to_input=np.zeros(2, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="value count"):
        NaqIBook(
            {"model.weight": imap},
            {"model.weight": TensorRateBudget(target_bits=24, value_count=5)},
        )
