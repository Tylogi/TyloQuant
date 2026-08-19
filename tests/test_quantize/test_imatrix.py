from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from mfq.quantize.imatrix import (
    ImportanceEntry,
    ImportanceMatrix,
    _load_gguf_reader,
    load_importance_matrix,
    save_importance_matrix,
)


def test_native_imatrix_round_trip_preserves_experts_and_metadata(tmp_path: Path):
    path = tmp_path / "calibration.imatrix"
    saved = save_importance_matrix(
        path,
        {
            "model.layers.0.experts.gate_up_proj": ImportanceEntry(
                values=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                counts=np.asarray([11, 7], dtype=np.int64),
            )
        },
        datasets=("balanced-corpus",),
        chunk_count=3,
        chunk_size=128,
        metadata={"backend": "metal", "manifest_sha256": "abc"},
    )

    loaded = load_importance_matrix(path)

    assert saved.metadata == loaded.metadata == {
        "backend": "metal",
        "manifest_sha256": "abc",
    }
    assert loaded.datasets == ("balanced-corpus",)
    assert loaded.chunk_count == 3
    assert loaded.chunk_size == 128
    np.testing.assert_array_equal(
        loaded.entries["model.layers.0.experts.gate_up_proj"].values,
        [[1.0, 2.0], [3.0, 4.0]],
    )
    np.testing.assert_array_equal(
        loaded.entries["model.layers.0.experts.gate_up_proj"].counts,
        [11, 7],
    )


def test_load_legacy_imatrix_normalizes_values(tmp_path: Path):
    path = tmp_path / "imatrix.dat"
    name = b"blk.0.attn_q.weight"
    dataset = b"calibration.txt"
    path.write_bytes(
        struct.pack("<i", 1)
        + struct.pack("<i", len(name))
        + name
        + struct.pack("<ii", 4, 3)
        + np.asarray([4.0, 8.0, 12.0], dtype="<f4").tobytes()
        + struct.pack("<ii", 7, len(dataset))
        + dataset
    )

    loaded = load_importance_matrix(path)

    assert loaded.legacy
    assert loaded.datasets == ("calibration.txt",)
    assert loaded.chunk_count == 7
    np.testing.assert_array_equal(
        loaded.entries["blk.0.attn_q.weight"].values,
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
    )


def test_load_gguf_imatrix_normalizes_each_matrix(tmp_path: Path):
    _load_gguf_reader()
    from gguf import GGUFWriter  # type: ignore

    path = tmp_path / "imatrix.gguf"
    writer = GGUFWriter(path, "imatrix")
    writer.add_string("general.type", "imatrix")
    writer.add_array("imatrix.datasets", ["calibration.txt"])
    writer.add_uint32("imatrix.chunk_count", 8)
    writer.add_uint32("imatrix.chunk_size", 512)
    writer.add_tensor(
        "blk.0.ffn_gate_exps.weight.in_sum2",
        np.asarray([[2.0, 4.0, 6.0], [12.0, 15.0, 18.0]], dtype=np.float32),
    )
    writer.add_tensor(
        "blk.0.ffn_gate_exps.weight.counts",
        np.asarray([[2.0], [3.0]], dtype=np.float32),
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    loaded = load_importance_matrix(path)

    assert not loaded.legacy
    assert loaded.datasets == ("calibration.txt",)
    assert loaded.chunk_count == 8
    assert loaded.chunk_size == 512
    entry = loaded.entries["blk.0.ffn_gate_exps.weight"]
    np.testing.assert_array_equal(entry.counts, np.asarray([2, 3]))
    np.testing.assert_array_equal(
        entry.values,
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    )


def test_expert_imatrix_selects_weights_by_flattened_row(tmp_path: Path):
    matrix = ImportanceMatrix(
        path=tmp_path / "unused.gguf",
        entries={
            "blk.0.ffn_gate_exps.weight": ImportanceEntry(
                values=np.asarray(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
                ),
                counts=np.asarray([10, 20], dtype=np.int64),
            )
        },
        datasets=(),
        chunk_count=0,
        chunk_size=0,
        legacy=False,
    )

    name, selected = matrix.for_rows(
        ("blk.0.ffn_gate_exps.weight",),
        (2, 3, 3),
        (6, 3),
        np.asarray([0, 2, 3, 5]),
    )

    assert name == "blk.0.ffn_gate_exps.weight"
    np.testing.assert_array_equal(
        selected,
        np.asarray(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
            dtype=np.float32,
        ),
    )


def test_gguf_imatrix_requires_llama_metadata(tmp_path: Path):
    _load_gguf_reader()
    from gguf import GGUFWriter  # type: ignore

    path = tmp_path / "missing-metadata.gguf"
    writer = GGUFWriter(path, "imatrix")
    writer.add_string("general.type", "imatrix")
    writer.add_tensor("test.weight.in_sum2", np.ones((1, 3), dtype=np.float32))
    writer.add_tensor("test.weight.counts", np.ones((1, 1), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    with pytest.raises(ValueError, match="missing metadata"):
        load_importance_matrix(path)
