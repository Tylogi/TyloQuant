from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mfq.tools.quantize_important_neurons import (
    SelectedMatrixSource,
    _choice_groups,
    _indices_asset,
)
from mfq.tools.quantize_tensor_upgrade_control import _candidate_units


class _Rows:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.rows = value.size(0)
        self.neuron_len = value.size(1)

    def read_rows(self, start_or_indices, end=None, *, device=None):
        if end is None:
            index = torch.as_tensor(start_or_indices, dtype=torch.int64)
            result = self.value.index_select(0, index)
        else:
            result = self.value[int(start_or_indices) : int(end)]
        return result.to("cpu" if device is None else device)


def test_selected_matrix_rows_partition_original_matrix() -> None:
    original = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    source = _Rows(original)
    hot = np.asarray([1, 4, 6], dtype=np.int64)
    cold = np.asarray([0, 2, 3, 5], dtype=np.int64)

    high = SelectedMatrixSource(source, row_indices=hot)
    low = SelectedMatrixSource(source, row_indices=cold)

    assert torch.equal(high[:], original[hot])
    assert torch.equal(low[:], original[cold])
    restored = torch.empty_like(original)
    restored[hot] = high[:]
    restored[cold] = low[:]
    assert torch.equal(restored, original)


def test_selected_matrix_columns_partition_original_matrix() -> None:
    original = torch.arange(42, dtype=torch.float32).reshape(6, 7)
    source = _Rows(original)
    hot = np.asarray([0, 3, 5], dtype=np.int64)
    cold = np.asarray([1, 2, 4, 6], dtype=np.int64)

    high = SelectedMatrixSource(source, column_indices=hot)
    low = SelectedMatrixSource(source, column_indices=cold)

    assert torch.equal(high[:], original[:, hot])
    assert torch.equal(low[:], original[:, cold])
    restored = torch.empty_like(original)
    restored[:, hot] = high[:]
    restored[:, cold] = low[:]
    assert torch.equal(restored, original)


def test_split_swiglu_branches_sum_to_original_ffn() -> None:
    generator = torch.Generator().manual_seed(20260731)
    batch, hidden, intermediate = 5, 7, 11
    x = torch.randn(batch, hidden, generator=generator, dtype=torch.float64)
    gate = torch.randn(
        intermediate, hidden, generator=generator, dtype=torch.float64
    )
    up = torch.randn(
        intermediate, hidden, generator=generator, dtype=torch.float64
    )
    down = torch.randn(
        hidden, intermediate, generator=generator, dtype=torch.float64
    )
    hot = torch.tensor([1, 4, 6, 9], dtype=torch.int64)
    mask = torch.ones(intermediate, dtype=torch.bool)
    mask[hot] = False
    cold = torch.arange(intermediate, dtype=torch.int64)[mask]

    def branch(indices: torch.Tensor) -> torch.Tensor:
        gate_value = x @ gate.index_select(0, indices).T
        up_value = x @ up.index_select(0, indices).T
        activated = torch.nn.functional.silu(gate_value) * up_value
        return activated @ down.index_select(1, indices).T

    full = (
        torch.nn.functional.silu(x @ gate.T)
        * (x @ up.T)
    ) @ down.T
    split = branch(cold) + branch(hot)
    torch.testing.assert_close(split, full, rtol=1e-12, atol=1e-12)


def test_precision_planner_keeps_gate_and_up_together() -> None:
    matrices = [
        SimpleNamespace(
            layer=layer,
            projection=projection,
            low_dtype="NINT4",
            high_options=("NINT5", "NINT6"),
        )
        for layer in (0, 3)
        for projection in ("gate", "up", "down")
    ]
    assert _choice_groups(matrices) == [
        (0, 1),
        (2,),
        (3, 4),
        (5,),
    ]


def test_control_planner_keeps_runtime_fused_pairs_together() -> None:
    names = (
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight",
        "blk.3.attn_v.weight",
        "blk.7.ffn_gate.weight",
        "blk.7.ffn_up.weight",
        "blk.7.ffn_down.weight",
    )
    candidates = [
        SimpleNamespace(
            item=SimpleNamespace(name=name),
            low_dtype="NINT3",
            high_dtype="NINT4",
        )
        for name in names
    ]
    units = _candidate_units(candidates)
    unit_names = [
        tuple(candidate.item.name for candidate in unit)
        for unit in units
    ]
    assert unit_names == [
        ("blk.3.attn_k.weight", "blk.3.attn_q.weight"),
        ("blk.3.attn_v.weight",),
        ("blk.7.ffn_gate.weight", "blk.7.ffn_up.weight"),
        ("blk.7.ffn_down.weight",),
    ]


def test_important_neuron_asset_is_deterministic() -> None:
    indices = {
        0: np.asarray([2, 7, 9], dtype=np.int64),
        3: np.asarray([1, 4, 8], dtype=np.int64),
    }
    first = _indices_asset(indices, 3)
    second = _indices_asset(indices, 3)
    assert first == second
    assert first[:4] == b"IN01"


def test_runtime_parallelizes_in_branches_only_for_decode() -> None:
    source = (
        Path(__file__).parents[1]
        / "cpp_runtime"
        / "mfq_decode.cpp"
    ).read_text(encoding="utf-8")
    branch = source[
        source.index(
            "if (allow_important_neurons && important_neurons)"
        ) :
    ]
    branch = branch[: branch.index("if (tensor_parallel_dense_compatible())")]
    assert "decode_branch_parallel_enabled(rows) &&" in branch
