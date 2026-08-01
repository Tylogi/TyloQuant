from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mfq.quantize.row_importance import load_row_importance, save_row_importance
from mfq.tools.collect_row_fisher import Target, _reorder_v_rows, _transform_rows


def test_row_importance_roundtrip(tmp_path: Path):
    path = tmp_path / "rows.npz"
    entries = {
        "blk.0.ffn_gate.weight": np.asarray([0.5, 1.5], dtype=np.float32),
        "blk.0.attn_qkv.weight": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    }

    save_row_importance(path, entries, {"objective": "test"})
    loaded = load_row_importance(path)

    assert loaded.metadata["objective"] == "test"
    np.testing.assert_array_equal(
        loaded.require("blk.0.ffn_gate.weight", 2), entries["blk.0.ffn_gate.weight"]
    )
    with pytest.raises(ValueError, match="expected"):
        loaded.require("blk.0.ffn_gate.weight", 3)


def test_row_importance_refuses_overwrite(tmp_path: Path):
    path = tmp_path / "rows.npz"
    save_row_importance(path, {"tensor": np.ones(2, dtype=np.float32)}, {})
    with pytest.raises(FileExistsError):
        save_row_importance(path, {"tensor": np.ones(2, dtype=np.float32)}, {})


def test_qwen35_linear_v_row_reorder_matches_gguf_layout():
    value = np.arange(16, dtype=np.float32)
    got = _reorder_v_rows(value, num_k_heads=2, num_v_heads=4, head_dim=4)
    expected = value.reshape(2, 2, 4).transpose(1, 0, 2).reshape(-1)
    np.testing.assert_array_equal(got, expected)


def test_qwen35_linear_qkv_transform_only_reorders_v_rows():
    class Config:
        linear_num_key_heads = 2
        linear_num_value_heads = 4
        linear_key_head_dim = 4
        linear_value_head_dim = 4

    value = np.arange(32, dtype=np.float32)
    target = Target("blk.0.attn_qkv.weight", "unused", 32, "linear_qkv")
    got = _transform_rows(value, target, Config())

    np.testing.assert_array_equal(got[:16], value[:16])
    np.testing.assert_array_equal(
        got[16:], value[16:].reshape(2, 2, 4).transpose(1, 0, 2).reshape(-1)
    )
