"""Build dense Important-Neuron (IN) MFQ models from BF16 weights.

For every dense SwiGLU layer, the imatrix Top-K intermediate neurons are
removed from the ordinary FFN matrices and stored as three additional
``.in_high`` records.  The low and high branches therefore form an exact
partition of the original graph:

    down_low(silu(gate_low(x)) * up_low(x))
  + down_high(silu(gate_high(x)) * up_high(x))

Unchanged records are copied byte-for-byte from the baseline MFQ.  Split
weights are always requantized directly from the BF16 GGUF.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from mfq.formats.assets import ASSET_DTYPE, ASSET_PREFIX
from mfq.formats.header import FileHeader
from mfq.formats.io import MMapTensorRecord, open_mmap
from mfq.formats.shards import write_blob_record_shards
from mfq.quantize.imatrix import ImportanceMatrix, load_importance_matrix
from mfq.quantize.nvq_jsc import NvqJscConfig
from mfq.tools.quantize_gguf_to_mfq import (
    GgufRowSource,
    GgufTensorPlan,
    ImatrixBinding,
    _JSC_DTYPES,
    _NINT_SPECS,
    _NVQ_SPECS,
    _build_plan,
    _estimate_blob_bytes,
    _load_gguf,
    _train_or_load_jsc_tables,
    _write_nint8_zero_axis0_blob,
    _write_nvq_blob,
)
from mfq.tools.quantize_hf_to_mfq import (
    BlobRecord,
    _write_nint_axis0_blob,
)


IN_ASSET_NAME = ASSET_PREFIX + "important_neurons.v1"
IN_PADDING_ASSET_NAME = ASSET_PREFIX + "important_neurons.padding"
_FFN_NAME = "blk.{layer}.ffn_{projection}.weight"
_HIGH_SUFFIX = ".in_high"
_SUPPORTED_LOW_DTYPES = {
    "NINT3",
    "NINT4",
    "NINT5",
    "NINT6",
    "NINT8-0",
    "NVQ3J",
}
_QUALITY_BITS = {
    "NINT3": 3.0,
    "NVQ3J": 3.35,
    "NINT4": 4.0,
    "NINT5": 5.0,
    "NINT6": 6.0,
    "NINT8-0": 8.0,
}
_HIGH_CHOICES = {
    "NINT3": ("NINT4", "NINT5", "NINT6", "NINT8-0"),
    "NVQ3J": ("NINT4", "NINT5", "NINT6", "NINT8-0"),
    "NINT4": ("NINT5", "NINT6", "NINT8-0"),
    "NINT5": ("NINT6", "NINT8-0"),
    "NINT6": ("NINT8-0",),
}


@dataclass(frozen=True)
class FileSpanRecord:
    name: str
    dtype: str
    nbytes: int
    path: Path
    offset: int


@dataclass(frozen=True)
class SplitMatrix:
    layer: int
    projection: str
    item: GgufTensorPlan
    low_dtype: str
    low_shape: tuple[int, int]
    high_shape: tuple[int, int]
    cold_indices: np.ndarray
    hot_indices: np.ndarray
    importance_mass: float
    high_options: tuple[str, ...]


@dataclass(frozen=True)
class PlannedSplit:
    matrix: SplitMatrix
    high_dtype: str
    low_nbytes: int
    high_nbytes: int


class SelectedMatrixSource:
    """Expose an arbitrary row or column subset as a dense row source."""

    def __init__(
        self,
        source: GgufRowSource,
        *,
        row_indices: np.ndarray | None = None,
        column_indices: np.ndarray | None = None,
    ) -> None:
        if row_indices is not None and column_indices is not None:
            raise ValueError("a selected matrix source may select rows or columns, not both")
        self.source = source
        self.row_indices = (
            None
            if row_indices is None
            else np.ascontiguousarray(row_indices, dtype=np.int64)
        )
        self.column_indices = (
            None
            if column_indices is None
            else np.ascontiguousarray(column_indices, dtype=np.int64)
        )
        self.rows = (
            source.rows if self.row_indices is None else int(self.row_indices.size)
        )
        self.neuron_len = (
            source.neuron_len
            if self.column_indices is None
            else int(self.column_indices.size)
        )

    def _output_rows(
        self,
        rows: np.ndarray,
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
        source_rows = rows if self.row_indices is None else self.row_indices[rows]
        value = self.source.read_rows(source_rows, device=device)
        if self.column_indices is not None:
            columns = torch.as_tensor(
                self.column_indices,
                dtype=torch.int64,
                device=value.device,
            )
            value = value.index_select(1, columns)
        return value.contiguous()

    def read_rows(
        self,
        start_or_indices: int | np.ndarray,
        end: int | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        target_device = "cpu" if device is None else device
        if end is None:
            rows = np.asarray(start_or_indices, dtype=np.int64).reshape(-1)
        else:
            rows = np.arange(
                int(start_or_indices), int(end), dtype=np.int64
            )
        if rows.size and (
            int(rows.min()) < 0 or int(rows.max()) >= self.rows
        ):
            raise IndexError(
                f"selected row index is outside [0, {self.rows})"
            )
        return self._output_rows(rows, device=target_device)

    def __getitem__(self, key: slice) -> torch.Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("selected matrix source accepts contiguous slices only")
        start = 0 if key.start is None else int(key.start)
        end = self.rows if key.stop is None else int(key.stop)
        return self.read_rows(start, end, device="cpu")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _record_span(store, record: MMapTensorRecord) -> FileSpanRecord:
    return FileSpanRecord(
        record.name,
        record.dtype,
        record.nbytes,
        Path(store.file_for(record).name).resolve(),
        record.offset,
    )


def _importance_vector(
    imatrix: ImportanceMatrix,
    item: GgufTensorPlan,
    columns: np.ndarray | None = None,
) -> tuple[str, np.ndarray]:
    match = imatrix.find((item.name, item.source_name))
    if match is None:
        raise KeyError(f"imatrix has no entry for {item.name}")
    entry_name, entry = match
    if entry.matrices != 1:
        raise ValueError(
            f"IN currently requires a dense imatrix entry: {entry_name}"
        )
    values = np.asarray(entry.values[0], dtype=np.float32)
    if values.size != int(item.storage_shape[1]):
        raise ValueError(
            f"imatrix width mismatch for {item.name}: "
            f"{values.size} != {item.storage_shape[1]}"
        )
    if columns is not None:
        values = values[np.asarray(columns, dtype=np.int64)]
    return entry_name, np.ascontiguousarray(values, dtype=np.float32)


def _binding(entry_name: str, values: np.ndarray) -> ImatrixBinding:
    def rows(_start: int, _end: int) -> np.ndarray:
        return values

    def selected(_row_ids: np.ndarray) -> np.ndarray:
        return values

    return ImatrixBinding(entry_name, rows, selected)


def _estimated_nbytes(
    item: GgufTensorPlan,
    dtype: str,
    shape: tuple[int, int],
) -> int:
    candidate = replace(
        item,
        target_dtype=dtype,
        original_shape=shape,
        storage_shape=shape,
        split=None,
        expert_shape=None,
        expert_precisions=None,
    )
    return _estimate_blob_bytes(
        candidate,
        custom_codebook=dtype.startswith("NVQ"),
        jsc_banks=2 if dtype == "NVQ3J" else 4,
    )


def _container_overhead(
    header: FileHeader,
    record_meta: list[tuple[str, str, int]],
) -> int:
    total = 4 + 4
    architecture = header.model_arch.encode("utf-8")
    total += 4 + len(architecture)
    if int(header.version) >= 2:
        total += 4
        for key, value in header.extra.items():
            key_bytes = str(key).encode("utf-8")
            value_bytes = json.dumps(value).encode("utf-8")
            total += 4 + len(key_bytes) + 4 + len(value_bytes)
    total += 4
    for name, dtype, _nbytes in record_meta:
        total += (
            4
            + len(name.encode("utf-8"))
            + 4
            + len(dtype.encode("utf-8"))
            + 8
        )
    return total


def _record_table_overhead(name: str, dtype: str) -> int:
    return (
        4
        + len(name.encode("utf-8"))
        + 4
        + len(dtype.encode("utf-8"))
        + 8
    )


def _indices_asset(
    layer_indices: dict[int, np.ndarray],
    top_k: int,
) -> bytes:
    layers = sorted(layer_indices)
    parts = [
        struct.pack("<4sIII", b"IN01", 1, len(layers), top_k)
    ]
    for layer in layers:
        values = np.ascontiguousarray(
            layer_indices[layer], dtype=np.uint32
        )
        if values.size != top_k:
            raise ValueError("IN index asset has an inconsistent Top-K")
        parts.append(struct.pack("<I", layer))
        parts.append(values.tobytes())
    return b"".join(parts)


def _build_split_matrices(
    plan_by_name: dict[str, GgufTensorPlan],
    store,
    imatrix: ImportanceMatrix,
    layer_ids: list[int],
    top_k: int,
) -> tuple[list[SplitMatrix], dict[int, np.ndarray]]:
    matrices: list[SplitMatrix] = []
    layer_indices: dict[int, np.ndarray] = {}
    for layer in layer_ids:
        names = {
            projection: _FFN_NAME.format(
                layer=layer, projection=projection
            )
            for projection in ("gate", "up", "down")
        }
        missing = [
            name
            for name in names.values()
            if name not in plan_by_name or name not in store.records
        ]
        if missing:
            raise KeyError(
                f"dense FFN layer {layer} is incomplete: {missing}"
            )
        down_item = plan_by_name[names["down"]]
        _entry, down_importance = _importance_vector(
            imatrix, down_item
        )
        intermediate = int(down_item.storage_shape[1])
        if not 0 < top_k < intermediate:
            raise ValueError(
                f"Top-K must be in (0, {intermediate}), got {top_k}"
            )
        order = np.argsort(
            -down_importance, kind="stable"
        ).astype(np.int64)
        hot = np.sort(order[:top_k])
        mask = np.ones(intermediate, dtype=bool)
        mask[hot] = False
        cold = np.flatnonzero(mask).astype(np.int64)
        layer_indices[layer] = hot
        mass = float(np.sum(down_importance[hot], dtype=np.float64))

        for projection in ("gate", "up", "down"):
            item = plan_by_name[names[projection]]
            dtype = store.records[item.name].dtype
            if dtype not in _SUPPORTED_LOW_DTYPES:
                raise ValueError(
                    f"unsupported IN baseline dtype {dtype}: {item.name}"
                )
            if dtype not in _HIGH_CHOICES:
                raise ValueError(
                    f"baseline dtype has no higher IN precision: "
                    f"{dtype}: {item.name}"
                )
            if projection == "down":
                low_shape = (int(item.storage_shape[0]), cold.size)
                high_shape = (int(item.storage_shape[0]), hot.size)
            else:
                low_shape = (cold.size, int(item.storage_shape[1]))
                high_shape = (hot.size, int(item.storage_shape[1]))
            matrices.append(
                SplitMatrix(
                    layer=layer,
                    projection=projection,
                    item=item,
                    low_dtype=dtype,
                    low_shape=tuple(map(int, low_shape)),
                    high_shape=tuple(map(int, high_shape)),
                    cold_indices=cold,
                    hot_indices=hot,
                    importance_mass=mass,
                    high_options=_HIGH_CHOICES[dtype],
                )
            )
    return matrices, layer_indices


def _choice_utility(matrix: SplitMatrix, dtype: str) -> float:
    baseline_error = 2.0 ** (
        -2.0 * _QUALITY_BITS[matrix.low_dtype]
    )
    candidate_error = 2.0 ** (-2.0 * _QUALITY_BITS[dtype])
    return matrix.importance_mass * (
        baseline_error - candidate_error
    )


def _choice_groups(
    matrices: list[SplitMatrix],
) -> list[tuple[int, ...]]:
    by_key = {
        (matrix.layer, matrix.projection): index
        for index, matrix in enumerate(matrices)
    }
    layers = sorted({matrix.layer for matrix in matrices})
    groups: list[tuple[int, ...]] = []
    for layer in layers:
        try:
            gate_index = by_key[(layer, "gate")]
            up_index = by_key[(layer, "up")]
            down_index = by_key[(layer, "down")]
        except KeyError as exc:
            raise ValueError(
                f"IN precision planning requires complete layer {layer}"
            ) from exc
        gate = matrices[gate_index]
        up = matrices[up_index]
        if (
            gate.low_dtype != up.low_dtype
            or gate.high_options != up.high_options
        ):
            raise ValueError(
                f"IN gate/up precision options differ in layer {layer}"
            )
        groups.extend(((gate_index, up_index), (down_index,)))
    covered = sorted(index for group in groups for index in group)
    if covered != list(range(len(matrices))):
        raise ValueError("IN precision groups do not cover every split matrix")
    return groups


def _knapsack_choices(
    matrices: list[SplitMatrix],
    high_budget: int,
    *,
    unit: int = 16 * 1024,
) -> list[str]:
    groups = _choice_groups(matrices)
    minimum_bytes = [
        _estimated_nbytes(
            matrix.item,
            matrix.high_options[0],
            matrix.high_shape,
        )
        for matrix in matrices
    ]
    minimum_total = sum(minimum_bytes)
    if minimum_total > high_budget:
        raise ValueError(
            f"UD size budget cannot give every IN branch higher precision: "
            f"minimum={minimum_total}, budget={high_budget}"
        )
    capacity = (high_budget - minimum_total) // unit
    negative = -np.inf
    dp = np.full(capacity + 1, negative, dtype=np.float64)
    dp[0] = 0.0
    parent_cost = np.full(
        (len(groups), capacity + 1), -1, dtype=np.int32
    )
    parent_choice = np.full(
        (len(groups), capacity + 1), -1, dtype=np.int8
    )

    for group_index, group in enumerate(groups):
        next_dp = np.full_like(dp, negative)
        options = matrices[group[0]].high_options
        if any(matrices[index].high_options != options for index in group):
            raise ValueError("IN precision group has inconsistent options")
        base_bytes = sum(minimum_bytes[index] for index in group)
        for option_index, dtype in enumerate(options):
            option_bytes = sum(
                _estimated_nbytes(
                    matrices[index].item,
                    dtype,
                    matrices[index].high_shape,
                )
                for index in group
            )
            delta_units = math.ceil(
                max(0, option_bytes - base_bytes) / unit
            )
            utility = sum(
                _choice_utility(matrices[index], dtype)
                for index in group
            )
            if delta_units > capacity:
                continue
            source = dp[: capacity + 1 - delta_units]
            candidate = source + utility
            target = next_dp[delta_units:]
            better = candidate > target
            if not np.any(better):
                continue
            positions = np.flatnonzero(better) + delta_units
            next_dp[positions] = candidate[better]
            parent_cost[group_index, positions] = (
                positions - delta_units
            )
            parent_choice[group_index, positions] = option_index
        dp = next_dp

    cost = int(np.nanargmax(dp))
    choices = [""] * len(matrices)
    for group_index in range(len(groups) - 1, -1, -1):
        option_index = int(parent_choice[group_index, cost])
        previous = int(parent_cost[group_index, cost])
        if option_index < 0 or previous < 0:
            raise RuntimeError("IN precision knapsack has no recoverable path")
        group = groups[group_index]
        dtype = matrices[group[0]].high_options[option_index]
        for index in group:
            choices[index] = dtype
        cost = previous
    return choices


def _header(
    baseline: FileHeader,
    *,
    source: Path,
    recipe: Path,
    imatrix: Path,
    baseline_model: Path,
    top_k: int,
    target_bytes: int,
    choices: list[str],
    matrices: list[SplitMatrix],
) -> FileHeader:
    extra = dict(baseline.extra)
    extra["important_neurons"] = {
        "version": 1,
        "method": "imatrix_topk_dense_ffn_partition",
        "top_k": top_k,
        "index_asset": IN_ASSET_NAME,
        "source_bf16": str(source),
        "recipe": str(recipe),
        "imatrix": str(imatrix),
        "baseline_mfq": str(baseline_model),
        "target_bytes": target_bytes,
        "high_precision_counts": {
            dtype: choices.count(dtype)
            for dtype in sorted(set(choices))
        },
        "branch_record_suffix": _HIGH_SUFFIX,
        "activation": "independent_swiglu_then_output_sum",
        "allocation": [
            {
                "layer": matrix.layer,
                "projection": matrix.projection,
                "low_dtype": matrix.low_dtype,
                "high_dtype": dtype,
            }
            for matrix, dtype in zip(matrices, choices, strict=True)
        ],
    }
    extra.pop("tensor_codebook_results", None)
    extra.pop("tensor_gain_results", None)
    return FileHeader(
        version=max(2, int(baseline.version)),
        model_arch=baseline.model_arch,
        num_tensors=0,
        extra=extra,
    )


def _record_metadata(
    store,
    split_by_name: dict[str, PlannedSplit],
    asset_nbytes: int,
) -> list[tuple[str, str, int]]:
    result: list[tuple[str, str, int]] = []
    for record in store.records.values():
        split = split_by_name.get(record.name)
        if split is None:
            result.append(
                (record.name, record.dtype, record.nbytes)
            )
            continue
        result.append(
            (
                record.name,
                split.matrix.low_dtype,
                split.low_nbytes,
            )
        )
        result.append(
            (
                record.name + _HIGH_SUFFIX,
                split.high_dtype,
                split.high_nbytes,
            )
        )
    result.append((IN_ASSET_NAME, ASSET_DTYPE, asset_nbytes))
    return result


def _planned_splits(
    matrices: list[SplitMatrix],
    choices: list[str],
) -> list[PlannedSplit]:
    return [
        PlannedSplit(
            matrix=matrix,
            high_dtype=dtype,
            low_nbytes=_estimated_nbytes(
                matrix.item,
                matrix.low_dtype,
                matrix.low_shape,
            ),
            high_nbytes=_estimated_nbytes(
                matrix.item,
                dtype,
                matrix.high_shape,
            ),
        )
        for matrix, dtype in zip(matrices, choices, strict=True)
    ]


def _plan_precisions(
    store,
    baseline_header: FileHeader,
    matrices: list[SplitMatrix],
    asset: bytes,
    target_bytes: int,
    header_factory: Callable[[list[str]], FileHeader],
) -> tuple[list[PlannedSplit], FileHeader, int, int]:
    groups = _choice_groups(matrices)
    minimum = [matrix.high_options[0] for matrix in matrices]
    minimum_splits = _planned_splits(matrices, minimum)
    minimum_map = {
        split.matrix.item.name: split
        for split in minimum_splits
    }
    minimum_header = header_factory(minimum)
    minimum_meta = _record_metadata(
        store, minimum_map, len(asset)
    )
    fixed_payload = sum(nbytes for _, _, nbytes in minimum_meta)
    minimum_total = (
        fixed_payload
        + _container_overhead(minimum_header, minimum_meta)
    )
    minimum_high = sum(
        split.high_nbytes for split in minimum_splits
    )
    high_budget = minimum_high + (target_bytes - minimum_total)
    choices = _knapsack_choices(matrices, high_budget)

    def materialized_total(values: list[str]) -> tuple[
        int, list[PlannedSplit], FileHeader
    ]:
        splits = _planned_splits(matrices, values)
        split_map = {
            split.matrix.item.name: split for split in splits
        }
        header = header_factory(values)
        meta = _record_metadata(store, split_map, len(asset))
        return (
            sum(nbytes for _, _, nbytes in meta)
            + _container_overhead(header, meta),
            splits,
            header,
        )

    total, splits, header = materialized_total(choices)
    while total > target_bytes:
        candidates: list[tuple[float, tuple[int, ...], str]] = []
        for group in groups:
            matrix = matrices[group[0]]
            dtype = choices[group[0]]
            if any(choices[index] != dtype for index in group):
                raise RuntimeError("IN precision group was split")
            option = matrix.high_options.index(dtype)
            if option == 0:
                continue
            lower = matrix.high_options[option - 1]
            current_bytes = sum(
                _estimated_nbytes(
                    matrices[index].item,
                    dtype,
                    matrices[index].high_shape,
                )
                for index in group
            )
            lower_bytes = sum(
                _estimated_nbytes(
                    matrices[index].item,
                    lower,
                    matrices[index].high_shape,
                )
                for index in group
            )
            loss = sum(
                _choice_utility(matrices[index], dtype)
                - _choice_utility(matrices[index], lower)
                for index in group
            )
            candidates.append(
                (
                    loss / max(1, current_bytes - lower_bytes),
                    group,
                    lower,
                )
            )
        if not candidates:
            raise ValueError("minimum IN model exceeds target bytes")
        _ratio, group, lower = min(candidates)
        for index in group:
            choices[index] = lower
        total, splits, header = materialized_total(choices)

    while True:
        remaining = target_bytes - total
        candidates = []
        for group in groups:
            matrix = matrices[group[0]]
            dtype = choices[group[0]]
            if any(choices[index] != dtype for index in group):
                raise RuntimeError("IN precision group was split")
            option = matrix.high_options.index(dtype)
            if option + 1 >= len(matrix.high_options):
                continue
            higher = matrix.high_options[option + 1]
            current_bytes = sum(
                _estimated_nbytes(
                    matrices[index].item,
                    dtype,
                    matrices[index].high_shape,
                )
                for index in group
            )
            higher_bytes = sum(
                _estimated_nbytes(
                    matrices[index].item,
                    higher,
                    matrices[index].high_shape,
                )
                for index in group
            )
            cost = higher_bytes - current_bytes
            if cost <= remaining:
                gain = sum(
                    _choice_utility(matrices[index], higher)
                    - _choice_utility(matrices[index], dtype)
                    for index in group
                )
                candidates.append(
                    (-(gain / max(1, cost)), group, higher)
                )
        if not candidates:
            break
        _ratio, group, higher = min(candidates)
        for index in group:
            choices[index] = higher
        total, splits, header = materialized_total(choices)

    remaining = target_bytes - total
    padding_overhead = _record_table_overhead(
        IN_PADDING_ASSET_NAME, ASSET_DTYPE
    )
    padding_nbytes = (
        remaining - padding_overhead
        if remaining >= padding_overhead
        else 0
    )
    exact_total = (
        total + padding_overhead + padding_nbytes
        if padding_nbytes
        else total
    )
    return splits, header, exact_total, padding_nbytes


def _selected_source(
    source: GgufRowSource,
    split: SplitMatrix,
    high: bool,
) -> SelectedMatrixSource:
    indices = split.hot_indices if high else split.cold_indices
    if split.projection == "down":
        return SelectedMatrixSource(
            source, column_indices=indices
        )
    return SelectedMatrixSource(source, row_indices=indices)


def _selected_binding(
    imatrix: ImportanceMatrix,
    split: SplitMatrix,
    high: bool,
) -> ImatrixBinding:
    columns = (
        split.hot_indices if high else split.cold_indices
    ) if split.projection == "down" else None
    entry_name, values = _importance_vector(
        imatrix, split.item, columns
    )
    return _binding(entry_name, values)


def _quantize_blob(
    source: SelectedMatrixSource,
    split: SplitMatrix,
    *,
    high: bool,
    dtype: str,
    blob_path: Path,
    imatrix: ImportanceMatrix,
    source_path: Path,
    recipe_path: Path,
    artifact_root: Path,
    row_chunk: int,
    device: str,
) -> tuple[int, dict[str, Any] | None]:
    shape = split.high_shape if high else split.low_shape
    binding = _selected_binding(imatrix, split, high)
    if dtype == "NINT8-0":
        return (
            _write_nint8_zero_axis0_blob(
                source, shape, blob_path, row_chunk
            ),
            None,
        )
    if dtype.startswith("NINT"):
        return (
            _write_nint_axis0_blob(
                source,
                shape,
                _NINT_SPECS[dtype],
                blob_path,
                row_chunk,
                "cuda",
                device,
                importance_rows=binding.rows,
            ),
            None,
        )
    if dtype != "NVQ3J":
        raise ValueError(f"unsupported IN blob dtype: {dtype}")

    item = replace(
        split.item,
        name=split.item.name + (".in_high_plan" if high else ".in_low_plan"),
        original_shape=shape,
        storage_shape=shape,
        target_dtype=dtype,
        split=None,
    )
    config = NvqJscConfig(
        spec=_NVQ_SPECS[dtype],
        banks=2,
        iterations=4,
        assignment_refine_steps=2,
        search_steps=19,
        raw_multiplier=8,
        learned_scale_lut=False,
        codebook_storage="int8",
        group_chunk=32768,
        seed=20260716,
    )
    tables, metrics = _train_or_load_jsc_tables(
        source,
        item,
        source_path,
        recipe_path,
        artifact_root,
        config,
        2048,
        512,
        20260716,
        device,
        imatrix,
        binding,
    )
    result = _write_nvq_blob(
        source=source,
        shape=shape,
        target_dtype=dtype,
        blob_path=blob_path,
        row_chunk=row_chunk,
        quant_backend="cuda",
        device=device,
        group_chunk=32768,
        nvq1_l_candidates=0,
        nvq1_l_anchor_multipliers=(0.75,),
        nvq1_l_refine_steps=2,
        importance_rows=binding.rows,
        codebook=None,
        search_steps=19,
        nvq_native_assignment=True,
        nvq1_l_native_assignment=True,
        jsc_tables=tables,
        jsc_assignment_refine_steps=2,
        npq0_l_tables=None,
        npq0_l_config=None,
        calibration_mode="gain",
    )
    return result.nbytes, metrics


def convert(args: argparse.Namespace) -> None:
    source_path = Path(args.input_bf16_gguf).resolve()
    recipe_path = Path(args.recipe_gguf).resolve()
    baseline_path = Path(args.baseline_mfq).resolve()
    imatrix_path = Path(args.imatrix).resolve()
    output = Path(args.output).resolve()
    target_bytes = (
        int(args.target_bytes)
        if args.target_bytes
        else recipe_path.stat().st_size
    )
    if output.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    GGUFReader, dequantize = _load_gguf()
    source_reader = GGUFReader(str(source_path), "r")
    recipe_reader = GGUFReader(str(recipe_path), "r")
    source_tensors = {
        str(tensor.name): tensor
        for tensor in source_reader.tensors
    }
    raw_plan = _build_plan(
        source_reader, recipe_reader, exclude_mtp=True
    )
    plan_by_name = {item.name: item for item in raw_plan}
    imatrix = load_importance_matrix(imatrix_path)
    layer_ids = (
        list(range(args.layers))
        if not args.layer_indices
        else [
            int(value)
            for value in args.layer_indices.split(",")
            if value.strip()
        ]
    )
    if (
        not layer_ids
        or len(set(layer_ids)) != len(layer_ids)
        or min(layer_ids) < 0
        or max(layer_ids) >= args.layers
    ):
        raise ValueError(
            f"invalid --layer-indices for {args.layers} layers: "
            f"{args.layer_indices!r}"
        )
    layer_ids = sorted(layer_ids)

    with open_mmap(baseline_path) as store:
        matrices, layer_indices = _build_split_matrices(
            plan_by_name,
            store,
            imatrix,
            layer_ids,
            args.top_k,
        )
        asset = _indices_asset(layer_indices, args.top_k)
        header_factory = lambda choices: _header(
            store.header,
            source=source_path,
            recipe=recipe_path,
            imatrix=imatrix_path,
            baseline_model=baseline_path,
            top_k=args.top_k,
            target_bytes=target_bytes,
            choices=choices,
            matrices=matrices,
        )
        splits, header, estimated_total, padding_nbytes = _plan_precisions(
            store,
            store.header,
            matrices,
            asset,
            target_bytes,
            header_factory,
        )
        contract = {
            "format": "mfq.important-neurons.run.v1",
            "input_bf16_gguf": _file_identity(source_path),
            "recipe_gguf": _file_identity(recipe_path),
            "baseline_mfq": _file_identity(baseline_path),
            "imatrix": _file_identity(imatrix_path),
            "output": str(output),
            "model_layers": args.layers,
            "in_layers": layer_ids,
            "top_k": args.top_k,
            "split_tensors": len(splits),
            "target_bytes": target_bytes,
            "estimated_output_bytes": estimated_total,
            "estimated_gap_bytes": target_bytes - estimated_total,
            "padding_bytes": padding_nbytes,
            "high_precision_counts": {
                dtype: sum(
                    split.high_dtype == dtype for split in splits
                )
                for dtype in sorted(
                    {split.high_dtype for split in splits}
                )
            },
            "device": args.device,
            "row_chunk": args.row_chunk,
        }
        print(json.dumps(contract, ensure_ascii=False), flush=True)
        if args.dry_run:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("IN quantization requires CUDA")

        temporary_root = (
            output.parent / f".{output.name}.important-neurons"
        )
        blob_root = temporary_root / "blobs"
        artifact_root = temporary_root / "codebooks"
        if temporary_root.exists() and not args.resume:
            raise FileExistsError(
                f"IN temporary directory exists; use --resume: {temporary_root}"
            )
        blob_root.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(temporary_root / "contract.json", contract)

        split_by_name = {
            split.matrix.item.name: split for split in splits
        }
        generated: dict[tuple[str, bool], BlobRecord] = {}
        metrics: dict[str, Any] = {}
        completed = 0
        total_blobs = len(splits) * 2
        started = time.time()
        for index, split in enumerate(splits):
            source_tensor = source_tensors[
                split.matrix.item.source_name
            ]
            full_source = GgufRowSource(
                source_tensor, split.matrix.item, dequantize
            )
            for high, dtype, expected in (
                (False, split.matrix.low_dtype, split.low_nbytes),
                (True, split.high_dtype, split.high_nbytes),
            ):
                tag = "high" if high else "low"
                blob_path = blob_root / (
                    f"{index:03d}-{split.matrix.layer:03d}-"
                    f"{split.matrix.projection}-{tag}.blob"
                )
                metric_key = (
                    split.matrix.item.name
                    + (_HIGH_SUFFIX if high else "")
                )
                t0 = time.time()
                if blob_path.is_file():
                    nbytes = blob_path.stat().st_size
                    if nbytes != expected:
                        raise ValueError(
                            f"resume blob size mismatch for {metric_key}: "
                            f"{nbytes} != {expected}"
                        )
                else:
                    selected = _selected_source(
                        full_source, split.matrix, high
                    )
                    nbytes, metric = _quantize_blob(
                        selected,
                        split.matrix,
                        high=high,
                        dtype=dtype,
                        blob_path=blob_path,
                        imatrix=imatrix,
                        source_path=source_path,
                        recipe_path=recipe_path,
                        artifact_root=artifact_root,
                        row_chunk=args.row_chunk,
                        device=args.device,
                    )
                    if nbytes != expected:
                        raise ValueError(
                            f"IN blob estimate mismatch for {metric_key}: "
                            f"{nbytes} != {expected}"
                        )
                    if metric is not None:
                        metrics[metric_key] = metric
                generated[(split.matrix.item.name, high)] = BlobRecord(
                    metric_key, dtype, nbytes, blob_path
                )
                completed += 1
                state = {
                    "status": "quantizing",
                    "completed": completed,
                    "blobs": total_blobs,
                    "last_tensor": metric_key,
                    "last_dtype": dtype,
                    "last_seconds": time.time() - t0,
                    "elapsed_seconds": time.time() - started,
                }
                _atomic_json(
                    temporary_root / "state.json", state
                )
                print(
                    json.dumps(state, ensure_ascii=False),
                    flush=True,
                )

        asset_path = temporary_root / "important-neurons.v1"
        asset_path.write_bytes(asset)
        records: list[FileSpanRecord | BlobRecord] = []
        for record in store.records.values():
            split = split_by_name.get(record.name)
            if split is None:
                records.append(_record_span(store, record))
                continue
            low = generated[(record.name, False)]
            high = generated[(record.name, True)]
            records.extend((low, high))
        records.append(
            BlobRecord(
                IN_ASSET_NAME,
                ASSET_DTYPE,
                len(asset),
                asset_path,
            )
        )
        if padding_nbytes:
            padding_path = temporary_root / "padding.bin"
            with padding_path.open("wb") as handle:
                handle.truncate(padding_nbytes)
            records.append(
                BlobRecord(
                    IN_PADDING_ASSET_NAME,
                    ASSET_DTYPE,
                    padding_nbytes,
                    padding_path,
                )
            )
        if metrics:
            _atomic_json(
                temporary_root / "tensor-codebook-results.json",
                metrics,
            )
        outputs = write_blob_record_shards(
            output,
            header,
            records,
            overwrite=args.overwrite,
        )
        actual_size = sum(path.stat().st_size for path in outputs)
        if actual_size != estimated_total:
            raise ValueError(
                f"IN output size differs from the plan: "
                f"{actual_size} != {estimated_total}"
            )
        final_state = {
            "status": "complete",
            "output": str(outputs[0]),
            "outputs": [str(path) for path in outputs],
            "output_bytes": actual_size,
            "target_bytes": target_bytes,
            "gap_bytes": target_bytes - actual_size,
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(temporary_root / "state.json", final_state)
        print(json.dumps(final_state, ensure_ascii=False), flush=True)
    if not args.keep_temp:
        shutil.rmtree(temporary_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bf16-gguf", required=True)
    parser.add_argument("--recipe-gguf", required=True)
    parser.add_argument("--baseline-mfq", required=True)
    parser.add_argument("--imatrix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument(
        "--layer-indices",
        default="",
        help="optional comma-separated subset used for production-path probes",
    )
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=0,
        help="defaults to the recipe GGUF file size",
    )
    parser.add_argument("--row-chunk", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    return parser


def main() -> None:
    convert(build_parser().parse_args())


if __name__ == "__main__":
    main()
