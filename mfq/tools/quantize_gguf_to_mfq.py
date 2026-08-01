"""Stream a BF16 GGUF checkpoint into an MFQ mixed-NINT model.

The recipe GGUF supplies only tensor names and precision choices. Weight values
always come from the BF16 GGUF, so this converter never requantizes IQ/K-quant
data. MoE expert matrices are stored as ``[expert * out, in]`` NINT rows: every
expert output neuron retains its own top-level scale and minimum.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from mfq.calibration.artifact import (
    CalibrationScheme,
    ExpertPrecision,
    load_scheme,
)
from mfq.formats.header import FileHeader
from mfq.formats.assets import (
    ASSET_DTYPE,
    ASSET_MANIFEST_KEY,
    discover_model_config,
    gguf_metadata_asset,
    is_asset_record,
    model_config_asset,
    runtime_asset_manifest,
)
from mfq.formats.io import _NINT_HDR, open_mmap
from mfq.formats.shards import (
    matching_shard_paths,
    parse_size,
    validate_split_limits,
    write_blob_record_shards,
)
from mfq.formats.nint import NintSpec
from mfq.formats.nint8_zero import (
    pack_nint8_zero_blocks,
    pack_nint8_zero_header,
    payload_nbytes as nint8_zero_payload_nbytes,
    quantize_nint8_zero,
)
from mfq.formats.npq0_l import (
    _HEADER as _NPQ0_L_HEADER,
)
from mfq.formats.npq0_l import (
    _MAGIC as _NPQ0_L_MAGIC,
)
from mfq.formats.npq0_l import (
    _VERSION as _NPQ0_L_VERSION,
)
from mfq.formats.npq0_l import (
    NPQ0_L,
)
from mfq.formats.npq0_l import (
    pack_npq0_l_tables as _pack_npq0_l_tables,
)
from mfq.formats.npq0_l import (
    unpack_npq0_l_tables as _unpack_npq0_l_tables,
)
from mfq.formats.nvq import (
    _CODEBOOK_ID,
    _CUSTOM_CODEBOOK_FLAG,
    _INDEX_PARITY_FLAG,
    _JSC_FLAG,
    NVQ2_E8,
    NVQ2_E8_1024,
    NVQ2_E8_4096,
    NVQ3_D4,
    NVQ3_D4_512,
    NVQ3_D4_1024,
)
from mfq.formats.nvq import (
    _HEADER as _NVQ_HEADER,
)
from mfq.formats.nvq import (
    _MAGIC as _NVQ_MAGIC,
)
from mfq.formats.nvq import (
    _pack_bits as _pack_nvq_bits,
)
from mfq.formats.nvq import (
    pack_codebook as _pack_nvq_codebook,
)
from mfq.formats.nvq import (
    pack_jsc_tables as _pack_jsc_tables,
)
from mfq.formats.nvq import (
    unpack_codebook as _unpack_nvq_codebook,
)
from mfq.formats.nvq import (
    unpack_jsc_metadata as _unpack_jsc_metadata,
)
from mfq.formats.nvq1_l import (
    _HEADER as _NVQ1_L_HEADER,
)
from mfq.formats.nvq1_l import (
    _MAGIC as _NVQ1_L_MAGIC,
)
from mfq.formats.nvq1_l import (
    _PROFILE_CUSTOM_TERNARY,
    _PROFILE_IQ1S_GRID,
    NVQ1_L_T8_S3,
)
from mfq.formats.nvq1_l import (
    _pack_bits as _pack_nvq1_l_bits,
)
from mfq.formats.nvq1_l import (
    pack_ternary_codebook as _pack_nvq1_l_codebook,
)
from mfq.formats.nvq1_l import (
    unpack_ternary_codebook as _unpack_nvq1_l_codebook,
)
from mfq.quantize.imatrix import ImportanceMatrix, load_importance_matrix
from mfq.quantize.npq0_l import (
    Npq0LConfig,
    Npq0LTables,
    dequantize_npq0_l,
    npq0_l_tables_from_tensor,
    quantize_npq0_l_fixed,
    train_npq0_l,
)
from mfq.quantize.nvq1_l_quant import (
    dequantize as dequantize_nvq1_l,
)
from mfq.quantize.nvq1_l_quant import (
    quantize as nvq1_l_quantize,
)
from mfq.quantize.nvq_jsc import (
    NvqJscConfig,
    NvqJscTables,
    dequantize_nvq_jsc,
    initial_jsc_tables,
    jsc_tables_from_tensor,
    quantize_nvq_jsc_fixed,
    train_nvq_jsc,
)
from mfq.quantize.nvq_quant import (
    dequantize as dequantize_nvq,
)
from mfq.quantize.nvq_quant import (
    quantize as nvq_quantize,
)
from mfq.quantize.nvq_tensor_codebook import (
    TensorCodebookTrainingConfig,
    train_tensor_codebook,
)
from mfq.quantize.row_importance import RowImportance, load_row_importance
from mfq.quantize.second_order import diagonal_regressed_gain
from mfq.tools.quantize_hf_to_mfq import (
    _RECIPE_SPECS,
    BlobRecord,
    _dense_blob_from_tensor,
    _hf_to_gguf_name,
    _mixed_moe_blob_nbytes,
    _write_mfq,
    _write_nint_axis0_blob,
    _write_mixed_moe_axis0_blob,
)


def _trim_windows_working_set() -> bool:
    """Release clean GGUF-mapped pages after a large expert tensor."""
    if sys.platform != "win32":
        return False
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    empty_working_set = ctypes.windll.psapi.EmptyWorkingSet
    empty_working_set.argtypes = [ctypes.c_void_p]
    empty_working_set.restype = ctypes.c_int
    return bool(empty_working_set(get_current_process()))


def _hf_output_name_map(
    gguf_names: list[str],
    hf_names: list[str],
) -> dict[str, str]:
    inverse: dict[str, str] = {}
    for hf_name in hf_names:
        if is_asset_record(hf_name):
            continue
        gguf_name = _hf_to_gguf_name(hf_name)
        if gguf_name is None:
            raise ValueError(f"HF template tensor has no GGUF mapping: {hf_name}")
        if gguf_name in inverse:
            raise ValueError(f"duplicate GGUF mapping in HF template: {gguf_name}")
        inverse[gguf_name] = hf_name

    mapped = {
        gguf_name: inverse[gguf_name]
        for gguf_name in gguf_names
        if gguf_name in inverse
    }
    unmapped = [
        gguf_name for gguf_name in gguf_names if gguf_name not in inverse
    ]
    if unmapped != ["rope_freqs.weight"]:
        raise ValueError(f"unexpected unmapped GGUF tensors: {unmapped}")
    if set(mapped.values()) != set(hf_names):
        raise ValueError("GGUF tensors do not exactly cover the HF template")
    return mapped


_RECIPE_TARGETS = {
    "IQ1_M": "NVQ1-L",
    "IQ2_S": "NVQ2J-XL",
    "IQ2_XS": "NVQ2J-L",
    "IQ2_XXS": "NVQ2J",
    "IQ3_S": "NVQ3J-L",
    "IQ3_XXS": "NVQ3",
    "IQ4_NL": "NINT4",
    "IQ4_XS": "NINT4",
    "Q2_K": "NINT2",
    "Q3_K": "NINT3",
    "Q4_K": "NINT4",
    "Q5_0": "NINT5",
    "Q5_1": "NINT5",
    "Q5_K": "NINT5",
    "Q6_K": "NINT6",
    "Q8_0": "NINT8",
    "F32": "F32",
    "F16": "F16",
    "BF16": "F16",
}

_NINT_SPECS = {
    "NINT2": _RECIPE_SPECS["Q2_K"],
    "NINT3": _RECIPE_SPECS["Q3_K"],
    "NINT4": _RECIPE_SPECS["Q4_K"],
    "NINT5": _RECIPE_SPECS["Q5_K"],
    "NINT6": _RECIPE_SPECS["Q6_K"],
    "NINT8": _RECIPE_SPECS["Q8_0"],
}

_NVQ_SPECS = {
    "NVQ1-L": NVQ1_L_T8_S3,
    "NVQ2": NVQ2_E8,
    "NVQ2J": NVQ2_E8,
    "NVQ2J-L": NVQ2_E8_1024,
    "NVQ2J-XL": NVQ2_E8_4096,
    "NVQ3": NVQ3_D4,
    "NVQ3J": NVQ3_D4,
    "NVQ3J-512": NVQ3_D4_512,
    "NVQ3J-L": NVQ3_D4_1024,
}

_JSC_DTYPES = {
    "NVQ2J",
    "NVQ2J-L",
    "NVQ2J-XL",
    "NVQ3J",
    "NVQ3J-512",
    "NVQ3J-L",
}

_SPLIT_EXPERT_RE = re.compile(r"^(blk\.\d+)\.ffn_(gate|up)_exps\.weight$")
_BLOCK_TENSOR_RE = re.compile(r"^blk\.(\d+)\.")
_IMATRIX_OPTIONAL_TENSORS = {"token_embd.weight", "output.weight"}
_IMATRIX_NINT_DTYPES = {"NINT2", "NINT3", "NINT4", "NINT5", "NINT6"}
_TENSOR_OVERRIDE_DTYPES = {
    "F16",
    "F32",
    "NINT2",
    "NINT3",
    "NINT4",
    "NINT5",
    "NINT6",
    "NINT8",
    "NINT8-0",
    "NVQ1-L",
    "NVQ2",
    "NVQ2J",
    "NVQ2J-L",
    "NVQ2J-XL",
    "NVQ3",
    "NVQ3J",
    "NVQ3J-512",
    "NVQ3J-L",
}

ImportanceRows = Callable[[int, int], np.ndarray | None]
ImportanceSelection = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class GgufTensorPlan:
    name: str
    source_name: str
    source_shape: tuple[int, ...]
    original_shape: tuple[int, ...]
    storage_shape: tuple[int, ...]
    source_type: str
    recipe_type: str
    target_dtype: str
    split: str | None = None
    expert_shape: tuple[int, int, int] | None = None
    expert_precisions: tuple[ExpertPrecision, ...] | None = None

    @property
    def expert_specs(self) -> tuple[NintSpec, ...] | None:
        if self.expert_precisions is None:
            return None
        if any(value.nint_spec is None for value in self.expert_precisions):
            return None
        return tuple(
            value.nint_spec
            for value in self.expert_precisions
            if value.nint_spec is not None
        )


@dataclass(frozen=True)
class ImatrixBinding:
    entry_name: str
    rows: ImportanceRows
    selected: ImportanceSelection


@dataclass(frozen=True)
class NvqBlobWriteResult:
    nbytes: int
    gain_calibration: dict[str, Any] | None


def _bind_imatrix(
    imatrix: ImportanceMatrix,
    plan: list[GgufTensorPlan],
) -> dict[str, ImatrixBinding]:
    bindings: dict[str, ImatrixBinding] = {}
    missing: list[str] = []
    for item in plan:
        supports_imatrix = (
            item.target_dtype.startswith("NVQ")
            or item.target_dtype in _IMATRIX_NINT_DTYPES
            or (
                item.target_dtype == "NINTM"
                and item.expert_precisions is not None
                and any(
                    precision.family in _IMATRIX_NINT_DTYPES
                    or precision.family.startswith("NVQ")
                    or precision.family.startswith("NEPQ")
                    for precision in item.expert_precisions
                )
            )
        )
        if not supports_imatrix:
            continue
        names = (item.name, item.source_name)
        match = imatrix.for_rows(
            names,
            item.original_shape,
            item.storage_shape,
            slice(0, min(1, item.storage_shape[0])),
        )
        if match is None:
            if item.name not in _IMATRIX_OPTIONAL_TENSORS:
                missing.append(item.name)
            continue
        entry_name, _ = match

        def rows(start: int, end: int, *, _item=item, _names=names) -> np.ndarray:
            resolved = imatrix.for_rows(
                _names,
                _item.original_shape,
                _item.storage_shape,
                slice(start, end),
            )
            if resolved is None:
                raise RuntimeError(f"imatrix binding disappeared for {_item.name}")
            return resolved[1]

        def selected(
            row_ids: np.ndarray,
            *,
            _item=item,
            _names=names,
        ) -> np.ndarray:
            resolved = imatrix.for_rows(
                _names,
                _item.original_shape,
                _item.storage_shape,
                np.asarray(row_ids, dtype=np.int64),
            )
            if resolved is None:
                raise RuntimeError(f"imatrix binding disappeared for {_item.name}")
            return resolved[1]

        bindings[item.name] = ImatrixBinding(entry_name, rows, selected)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f" ... ({len(missing)} total)"
        raise ValueError(
            f"imatrix is missing required NINT/NVQ tensors: {preview}{suffix}"
        )
    return bindings


def _load_gguf():
    try:
        from gguf import GGUFReader  # type: ignore
        from gguf.quants import dequantize  # type: ignore
    except ModuleNotFoundError:
        import sys

        gguf_py = Path(__file__).resolve().parents[2] / "references" / "llamacpp" / "gguf-py"
        if not gguf_py.exists():
            raise
        sys.path.insert(0, str(gguf_py))
        from gguf import GGUFReader  # type: ignore
        from gguf.quants import dequantize  # type: ignore
    return GGUFReader, dequantize


def _logical_shape(tensor: Any) -> tuple[int, ...]:
    """Return row-major logical shape from GGML's ne0-first dimensions."""

    return tuple(int(v) for v in reversed(tensor.shape))


def _target_dtype(recipe_type: str) -> str:
    try:
        return _RECIPE_TARGETS[recipe_type]
    except KeyError as exc:
        raise ValueError(f"unsupported recipe tensor type: {recipe_type}") from exc


def _source_binding(recipe_name: str, source_names: set[str]) -> tuple[str, str | None]:
    if recipe_name in source_names:
        return recipe_name, None
    match = _SPLIT_EXPERT_RE.match(recipe_name)
    if match is not None:
        source_name = f"{match.group(1)}.ffn_gate_up_exps.weight"
        if source_name in source_names:
            return source_name, match.group(2)
    raise KeyError(f"recipe tensor has no BF16 source: {recipe_name}")


def _build_plan(
    source_reader: Any,
    recipe_reader: Any,
    *,
    exclude_mtp: bool = False,
    excluded_recipe_tensors: list[str] | None = None,
) -> list[GgufTensorPlan]:
    source = {str(t.name): t for t in source_reader.tensors}
    source_names = set(source)
    source_block_ids = [
        int(match.group(1))
        for name in source_names
        if (match := _BLOCK_TENSOR_RE.match(name)) is not None
    ]
    source_block_count = max(source_block_ids) + 1 if source_block_ids else None
    if exclude_mtp and hasattr(source_reader, "fields"):
        architecture = str(
            _field_value(source_reader, "general.architecture", "")
        )
        advertised_block_count = _field_value(
            source_reader,
            f"{architecture}.block_count",
            None,
        )
        if (
            isinstance(advertised_block_count, (int, np.integer))
            and int(advertised_block_count) > 0
        ):
            source_block_count = int(advertised_block_count)
            nextn_predict_layers = _field_value(
                source_reader,
                f"{architecture}.nextn_predict_layers",
                0,
            )
            if (
                isinstance(nextn_predict_layers, (int, np.integer))
                and 0 < int(nextn_predict_layers) < source_block_count
            ):
                source_block_count -= int(nextn_predict_layers)
    plan: list[GgufTensorPlan] = []

    for recipe_tensor in recipe_reader.tensors:
        name = str(recipe_tensor.name)
        block_match = _BLOCK_TENSOR_RE.match(name)
        if (
            exclude_mtp
            and source_block_count is not None
            and block_match is not None
            and int(block_match.group(1)) >= source_block_count
        ):
            if excluded_recipe_tensors is not None:
                excluded_recipe_tensors.append(name)
            continue
        recipe_type = str(recipe_tensor.tensor_type.name)
        target_dtype = _target_dtype(recipe_type)
        try:
            source_name, split = _source_binding(name, source_names)
        except KeyError:
            raise
        source_tensor = source[source_name]
        original_shape = _logical_shape(recipe_tensor)
        source_shape = _logical_shape(source_tensor)

        if split is None:
            singleton_reshape = (
                not target_dtype.startswith(("NINT", "NVQ"))
                and math.prod(source_shape) == math.prod(original_shape)
                and tuple(v for v in source_shape if v != 1) == original_shape
            )
            if source_shape != original_shape and not singleton_reshape:
                raise ValueError(
                    f"source/recipe shape mismatch for {name}: {source_shape} != {original_shape}"
                )
        else:
            if len(original_shape) != 3 or len(source_shape) != 3:
                raise ValueError(f"split expert tensor must be 3D: {name}")
            experts, out, neuron_len = original_shape
            expected = (experts, out * 2, neuron_len)
            if source_shape != expected:
                raise ValueError(
                    f"merged expert shape mismatch for {name}: {source_shape} != {expected}"
                )

        if target_dtype.startswith(("NINT", "NVQ")):
            if len(original_shape) < 2:
                raise ValueError(f"recipe maps a non-matrix tensor to {target_dtype}: {name}")
            storage_shape = (math.prod(original_shape[:-1]), original_shape[-1])
        else:
            storage_shape = original_shape

        plan.append(
            GgufTensorPlan(
                name=name,
                source_name=source_name,
                source_shape=source_shape,
                original_shape=original_shape,
                storage_shape=storage_shape,
                source_type=str(source_tensor.tensor_type.name),
                recipe_type=recipe_type,
                target_dtype=target_dtype,
                split=split,
            )
        )
    return plan


def _apply_expert_scheme(
    plan: list[GgufTensorPlan],
    scheme: CalibrationScheme | None,
) -> list[GgufTensorPlan]:
    if scheme is None or not scheme.expert_selections:
        return plan
    result: list[GgufTensorPlan] = []
    used: set[str] = set()
    for item in plan:
        selection = scheme.expert_selections.get(item.name)
        if selection is None:
            result.append(item)
            continue
        expected = (
            int(selection.n_experts),
            int(selection.rows_per_expert),
            int(selection.columns),
        )
        if item.original_shape != expected:
            raise ValueError(
                f"expert calibration shape mismatch for {item.name}: "
                f"GGUF={item.original_shape}, scheme={expected}"
            )
        result.append(
            replace(
                item,
                target_dtype="NINTM",
                expert_shape=expected,
                expert_precisions=selection.precisions,
            )
        )
        used.add(item.name)
    missing = sorted(set(scheme.expert_selections) - used)
    if missing:
        raise ValueError(
            f"expert calibration tensors are absent from the GGUF recipe: {missing[:8]}"
        )
    return result


def _load_tensor_precision_overrides(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("tensor precision override document must be an object")
    overrides = document.get("overrides", document)
    if not isinstance(overrides, dict):
        raise TypeError("tensor precision overrides must be an object")
    result: dict[str, str] = {}
    for raw_name, raw_dtype in overrides.items():
        name = str(raw_name)
        dtype = str(raw_dtype).upper()
        if not name:
            raise ValueError("tensor precision override contains an empty name")
        if dtype not in _TENSOR_OVERRIDE_DTYPES:
            raise ValueError(
                f"unsupported tensor precision override for {name}: {dtype}"
            )
        result[name] = dtype
    if not result:
        raise ValueError("tensor precision override document is empty")
    return result


def _apply_tensor_precision_overrides(
    plan: list[GgufTensorPlan],
    overrides: dict[str, str],
) -> list[GgufTensorPlan]:
    if not overrides:
        return plan
    result: list[GgufTensorPlan] = []
    used: set[str] = set()
    for item in plan:
        target_dtype = overrides.get(item.name)
        if target_dtype is None:
            result.append(item)
            continue
        if target_dtype.startswith(("NINT", "NVQ")) and len(item.original_shape) < 2:
            raise ValueError(
                f"tensor precision override maps a non-matrix tensor to "
                f"{target_dtype}: {item.name}"
            )
        storage_shape = (
            (math.prod(item.original_shape[:-1]), item.original_shape[-1])
            if target_dtype.startswith(("NINT", "NVQ"))
            else item.original_shape
        )
        result.append(
            replace(
                item,
                target_dtype=target_dtype,
                storage_shape=storage_shape,
                expert_shape=None,
                expert_precisions=None,
            )
        )
        used.add(item.name)
    missing = sorted(set(overrides) - used)
    if missing:
        raise ValueError(
            "tensor precision overrides are absent from the GGUF recipe: "
            f"{missing[:8]}"
        )
    return result


def _apply_npq0_l_mapping(
    plan: list[GgufTensorPlan],
    enabled: bool,
) -> list[GgufTensorPlan]:
    if not enabled:
        return plan
    return [
        replace(item, target_dtype="NPQ0-L")
        if item.target_dtype == "NVQ1-L"
        else item
        for item in plan
    ]


def _apply_nvq3_jsc_mapping(
    plan: list[GgufTensorPlan],
    enabled: bool,
    *,
    target_dtype: str = "NVQ3J",
) -> list[GgufTensorPlan]:
    if not enabled:
        return plan
    return [
        replace(item, target_dtype=target_dtype)
        if item.target_dtype == "NVQ3"
        else item
        for item in plan
    ]


def _apply_nvq3_to_nint3_mapping(
    plan: list[GgufTensorPlan],
    enabled: bool,
) -> list[GgufTensorPlan]:
    if not enabled:
        return plan
    return [
        replace(item, target_dtype="NINT3")
        if item.target_dtype == "NVQ3"
        else item
        for item in plan
    ]


def _apply_iq2_s_to_nint2_mapping(
    plan: list[GgufTensorPlan],
    enabled: bool,
) -> list[GgufTensorPlan]:
    if not enabled:
        return plan
    return [
        replace(item, target_dtype="NINT2")
        if item.recipe_type == "IQ2_S" and item.target_dtype.startswith("NVQ2J")
        else item
        for item in plan
    ]


def _apply_q8_to_nint8_zero_mapping(
    plan: list[GgufTensorPlan],
    enabled: bool,
) -> list[GgufTensorPlan]:
    if not enabled:
        return plan
    return [
        replace(item, target_dtype="NINT8-0")
        if item.recipe_type == "Q8_0" and item.target_dtype == "NINT8"
        else item
        for item in plan
    ]


class GgufRowSource:
    """Slice BF16/F16/F32 GGUF tensors as float32 ``[rows, in]`` tensors."""

    def __init__(self, tensor: Any, plan: GgufTensorPlan, dequantize) -> None:
        self.tensor = tensor
        self.plan = plan
        self.dequantize = dequantize
        self.rows = int(plan.storage_shape[0])
        self.neuron_len = int(plan.storage_shape[1])
        self._raw_rows = tensor.data.reshape(-1, tensor.data.shape[-1])

    def _source_indices(self, start: int, end: int) -> slice | np.ndarray:
        if self.plan.split is None:
            return slice(start, end)
        _experts, out, _neuron_len = self.plan.original_shape
        target = np.arange(start, end, dtype=np.int64)
        expert = target // out
        local = target - expert * out
        offset = 0 if self.plan.split == "gate" else out
        return expert * (2 * out) + local + offset

    def __getitem__(self, key: slice) -> torch.Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("GGUF row source accepts contiguous slices only")
        start = 0 if key.start is None else int(key.start)
        end = self.rows if key.stop is None else int(key.stop)
        if start < 0 or end < start or end > self.rows:
            raise IndexError(f"invalid GGUF row slice: {start}:{end} of {self.rows}")
        raw = self._raw_rows[self._source_indices(start, end)]
        values = self.dequantize(raw, self.tensor.tensor_type)
        values = np.ascontiguousarray(values, dtype=np.float32).reshape(
            end - start,
            self.neuron_len,
        )
        return torch.from_numpy(values)

    def read_rows(
        self,
        start_or_indices: int | np.ndarray,
        end: int | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        if end is not None:
            values = self[slice(int(start_or_indices), int(end))]
            return values if device is None else values.to(device=device, non_blocking=True)

        rows = np.asarray(start_or_indices, dtype=np.int64).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= self.rows):
            raise IndexError(f"GGUF row indices fall outside [0, {self.rows})")
        if self.plan.split is None:
            source_rows = rows
        else:
            _experts, out, _neuron_len = self.plan.original_shape
            expert = rows // out
            local = rows - expert * out
            offset = 0 if self.plan.split == "gate" else out
            source_rows = expert * (2 * out) + local + offset
        raw = self._raw_rows[source_rows]
        values = self.dequantize(raw, self.tensor.tensor_type)
        values = np.ascontiguousarray(values, dtype=np.float32).reshape(
            rows.size, self.neuron_len
        )
        result = torch.from_numpy(values)
        return result if device is None else result.to(device=device, non_blocking=True)


def _balanced_allocation(count: int, capacity: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = np.zeros_like(capacity, dtype=np.int64)
    remaining = int(count)
    order = rng.permutation(capacity.size)
    while remaining:
        progressed = False
        for bucket in order:
            if result[bucket] < capacity[bucket]:
                result[bucket] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError("sample count exceeds row capacity")
    return result


def _sample_codebook_rows(
    item: GgufTensorPlan,
    train_rows: int,
    validation_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    total_rows = int(item.storage_shape[0])
    requested_train = max(int(train_rows), 0)
    requested_validation = max(int(validation_rows), 0)
    requested_total = requested_train + requested_validation
    if requested_train and requested_validation and total_rows < requested_total:
        if total_rows < 2:
            raise ValueError(f"tensor has too few rows for a held-out split: {item.name}")
        validation_count = round(total_rows * requested_validation / requested_total)
        validation_count = min(max(validation_count, 1), total_rows - 1)
        train_count = total_rows - validation_count
    else:
        train_count = min(requested_train, total_rows)
        validation_count = min(requested_validation, total_rows - train_count)
    digest = hashlib.blake2b(
        f"{seed}:{item.name}".encode("utf-8"), digest_size=8
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))

    if len(item.original_shape) == 3:
        experts, rows_per_expert, _neuron_len = item.original_shape
        if experts * rows_per_expert != total_rows:
            raise ValueError(f"invalid flattened expert shape for {item.name}")
        capacity = np.full(experts, rows_per_expert, dtype=np.int64)
        train_per_expert = _balanced_allocation(train_count, capacity, rng)
        validation_per_expert = _balanced_allocation(
            validation_count,
            capacity - train_per_expert,
            rng,
        )
        train: list[np.ndarray] = []
        validation: list[np.ndarray] = []
        for expert in range(experts):
            order = rng.permutation(rows_per_expert)
            train_end = int(train_per_expert[expert])
            validation_end = train_end + int(validation_per_expert[expert])
            base = expert * rows_per_expert
            train.append(base + order[:train_end])
            validation.append(base + order[train_end:validation_end])
        train_index = np.concatenate(train) if train else np.empty(0, dtype=np.int64)
        validation_index = (
            np.concatenate(validation) if validation else np.empty(0, dtype=np.int64)
        )
    else:
        selected = rng.choice(
            total_rows,
            size=train_count + validation_count,
            replace=False,
        )
        train_index = selected[:train_count]
        validation_index = selected[train_count:]
    return np.sort(train_index), np.sort(validation_index)


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _codebook_config_dict(config: TensorCodebookTrainingConfig) -> dict[str, Any]:
    return {
        "iterations": config.iterations,
        "projection_candidates": config.projection_candidates,
        "quant_backend": config.quant_backend,
        "device": config.device,
        "group_chunk": config.group_chunk,
        "row_chunk": config.row_chunk,
        "search_steps": config.search_steps,
        "nvq1_l_anchor_multipliers": list(config.nvq1_l_anchor_multipliers),
        "nvq1_l_refine_steps": config.nvq1_l_refine_steps,
        "nvq_native_assignment": config.nvq_native_assignment,
        "nvq1_l_native_assignment": config.nvq1_l_native_assignment,
        "min_validation_improvement": config.min_validation_improvement,
        "initializations": list(config.initializations),
    }


def _jsc_config_dict(config: NvqJscConfig) -> dict[str, Any]:
    result = {
        "banks": config.banks,
        "iterations": config.iterations,
        "assignment_refine_steps": config.assignment_refine_steps,
        "search_steps": config.search_steps,
        "raw_multiplier": config.raw_multiplier,
        "learned_scale_lut": config.learned_scale_lut,
        "codebook_storage": config.codebook_storage,
        "group_chunk": config.group_chunk,
        "seed": config.seed,
    }
    if config.spec != NVQ2_E8:
        result["base"] = config.spec.codebook
    return result


def _npq0_l_config_dict(config: Npq0LConfig) -> dict[str, Any]:
    return {
        "iterations": config.iterations,
        "assignment_refine_steps": config.assignment_refine_steps,
        "fixed_refine_steps": config.fixed_refine_steps,
        "kmeans_iterations": config.kmeans_iterations,
        "kmeans_initialization_points": config.kmeans_initialization_points,
        "group_chunk": config.group_chunk,
        "anchor_multipliers": list(config.anchor_multipliers),
        "seed": config.seed,
    }


def _codebook_artifact_path(root: Path, item: GgufTensorPlan) -> Path:
    digest = hashlib.sha256(item.name.encode("utf-8")).hexdigest()[:16]
    return root / f"{digest}-{item.target_dtype.lower()}.json"


def _existing_codebook_artifact_path(path: Path) -> Path:
    if path.exists():
        return path
    legacy = path.with_name(path.name.replace("-nvq", "-niq", 1))
    return legacy if legacy.exists() else path


def _canonical_artifact_signature(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    target_dtype = result.get("target_dtype")
    if isinstance(target_dtype, str) and target_dtype.startswith("NIQ"):
        result["target_dtype"] = "NVQ" + target_dtype[3:]
    config = result.get("config")
    if isinstance(config, dict):
        config = dict(config)
        for legacy, canonical in (
            ("niq1_anchor_multipliers", "nvq1_l_anchor_multipliers"),
            ("niq1_refine_steps", "nvq1_l_refine_steps"),
            ("niq_native_assignment", "nvq_native_assignment"),
            ("niq1_native_assignment", "nvq1_l_native_assignment"),
        ):
            if legacy in config and canonical not in config:
                config[canonical] = config.pop(legacy)
        result["config"] = config
    return result


def _factorized_jsc_importance(
    item: GgufTensorPlan,
    row_ids: np.ndarray,
    imatrix_binding: ImatrixBinding,
    row_importance: RowImportance,
) -> np.ndarray:
    indices = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    if not indices.size:
        raise ValueError(f"empty JSC importance selection for {item.name}")
    rows, width = map(int, item.storage_shape)
    if int(indices.min()) < 0 or int(indices.max()) >= rows:
        raise IndexError(f"JSC importance selection is outside {rows} rows")
    row_weight = row_importance.require(item.name, rows)[indices]
    column_weight = np.asarray(imatrix_binding.selected(indices), dtype=np.float32)
    if column_weight.ndim == 1:
        if column_weight.shape != (width,):
            raise ValueError(
                f"imatrix for {item.name} has shape {column_weight.shape}, expected {(width,)}"
            )
        column_weight = np.broadcast_to(column_weight, (indices.size, width))
    elif column_weight.shape != (indices.size, width):
        raise ValueError(
            f"imatrix for {item.name} has shape {column_weight.shape}, "
            f"expected {(indices.size, width)}"
        )
    result = np.ascontiguousarray(
        row_weight[:, None].astype(np.float32, copy=False) * column_weight,
        dtype=np.float32,
    )
    if not np.isfinite(result).all() or np.any(result < 0) or not np.any(result > 0):
        raise ValueError(f"invalid factorized JSC importance for {item.name}")
    return result


def _train_or_load_jsc_tables(
    source: GgufRowSource,
    item: GgufTensorPlan,
    source_path: Path,
    recipe_path: Path,
    artifact_root: Path,
    config: NvqJscConfig,
    train_rows: int,
    validation_rows: int,
    seed: int,
    device: str,
    imatrix: ImportanceMatrix | None = None,
    imatrix_binding: ImatrixBinding | None = None,
    row_importance: RowImportance | None = None,
) -> tuple[NvqJscTables, dict[str, Any]]:
    """Fit/reuse tensor-wise JSC tables on deterministic rows."""

    if imatrix_binding is not None and imatrix is None:
        raise ValueError("an imatrix binding requires its source imatrix")
    if row_importance is not None and (imatrix is None or imatrix_binding is None):
        raise ValueError("row-Fisher JSC training requires a bound imatrix")

    train_index, validation_index = _sample_codebook_rows(
        item, train_rows, validation_rows, seed
    )
    if not train_index.size or not validation_index.size:
        raise ValueError(f"tensor-wise JSC training needs non-empty splits: {item.name}")
    objective = (
        "factorized_row_fisher_x_imatrix_sse"
        if row_importance is not None
        else (
            "imatrix_weighted_sse"
            if imatrix_binding is not None
            else "unweighted_weight_sse"
        )
    )
    signature = {
        "source": _file_identity(source_path),
        "recipe": _file_identity(recipe_path),
        "tensor_name": item.name,
        "source_name": item.source_name,
        "target_dtype": item.target_dtype,
        "storage_shape": list(item.storage_shape),
        "train_rows": train_index.tolist(),
        "validation_rows": validation_index.tolist(),
        "seed": seed,
        "config": _jsc_config_dict(config),
        "objective": objective,
        "imatrix": (
            None
            if imatrix_binding is None
            else {
                "file": _file_identity(imatrix.path),
                "entry": imatrix_binding.entry_name,
            }
        ),
    }
    if row_importance is not None:
        signature.update(
            {
                "row_importance": {
                    "file": _file_identity(row_importance.path),
                    "entry": item.name,
                    "metadata": row_importance.metadata,
                },
            }
        )
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = _existing_codebook_artifact_path(
        _codebook_artifact_path(artifact_root, item)
    )
    if artifact_path.exists():
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
        if document.get("format") not in {
            "mfq.tensorwise-nvq-jsc.v1",
            "mfq.tensorwise-niq-jsc.v1",
        }:
            raise ValueError(f"unsupported JSC artifact: {artifact_path}")
        if _canonical_artifact_signature(document.get("signature", {})) != signature:
            raise ValueError(f"stale JSC artifact signature: {artifact_path}")
        payload = base64.b64decode(document["tables_b64"], validate=True)
        if hashlib.sha256(payload).hexdigest() != document.get("tables_sha256"):
            raise ValueError(f"JSC artifact checksum mismatch: {artifact_path}")
        scale_lut, bank_for_state, codebooks, consumed = _unpack_jsc_metadata(
            payload,
            vector_size=config.spec.vector_size,
            codebook_entries=config.spec.codebook_entries,
        )
        if consumed != len(payload):
            raise ValueError(f"JSC artifact has an invalid table tail: {artifact_path}")
        metrics = dict(document["metrics"])
        metrics.update({"artifact": str(artifact_path), "loaded": True})
        return NvqJscTables(scale_lut, bank_for_state, codebooks, config.spec), metrics

    train_weight = source.read_rows(train_index)
    train_importance = (
        (
            None
            if imatrix_binding is None
            else imatrix_binding.selected(train_index)
        )
        if row_importance is None
        else _factorized_jsc_importance(
            item, train_index, imatrix_binding, row_importance
        )
    )
    trained, history = train_nvq_jsc(
        train_weight,
        importance=train_importance,
        config=config,
        device=device,
    )
    tables = jsc_tables_from_tensor(trained)
    validation_weight = source.read_rows(validation_index)
    validation_importance = (
        (
            None
            if imatrix_binding is None
            else imatrix_binding.selected(validation_index)
        )
        if row_importance is None
        else _factorized_jsc_importance(
            item, validation_index, imatrix_binding, row_importance
        )
    )
    validation = quantize_nvq_jsc_fixed(
        validation_weight,
        tables,
        importance=validation_importance,
        assignment_refine_steps=config.assignment_refine_steps,
        search_steps=config.search_steps,
        group_chunk=config.group_chunk,
        device=device,
    )
    validation_reference = validation_weight.numpy().astype(np.float32, copy=False)
    validation_reconstruction = dequantize_nvq_jsc(validation)
    validation_error_elementwise = np.square(
        validation_reconstruction - validation_reference, dtype=np.float64
    )
    validation_signal_elementwise = np.square(validation_reference, dtype=np.float64)
    if validation_importance is not None:
        validation_error_elementwise *= validation_importance
        validation_signal_elementwise *= validation_importance
    validation_error = float(validation_error_elementwise.sum())
    validation_signal = float(validation_signal_elementwise.sum())
    metrics = {
        "objective": objective,
        "train_best_nmse_percent": min(item.weighted_nmse_percent for item in history),
        "validation_nmse_percent": (
            100.0 * validation_error / validation_signal if validation_signal else 0.0
        ),
        "validation_snr_db": (
            10.0 * math.log10(validation_signal / validation_error)
            if validation_error
            else math.inf
        ),
        "banks": config.banks,
        "used_states": min(history, key=lambda item: item.weighted_sse).used_states,
    }
    if imatrix_binding is not None:
        metrics["imatrix_entry"] = imatrix_binding.entry_name
    if row_importance is not None:
        metrics.update(
            {
                "row_importance_entry": item.name,
            }
        )
    payload = _pack_jsc_tables(
        tables.scale_lut, tables.bank_for_state, tables.codebooks
    )
    document = {
        "format": "mfq.tensorwise-nvq-jsc.v1",
        "signature": signature,
        "metrics": metrics,
        "history": [
            {
                "iteration": item.iteration,
                "weighted_sse": item.weighted_sse,
                "weighted_nmse_percent": item.weighted_nmse_percent,
                "used_states": item.used_states,
                "used_banks": item.used_banks,
                "used_codes": list(item.used_codes),
            }
            for item in history
        ],
        "tables_b64": base64.b64encode(payload).decode("ascii"),
        "tables_sha256": hashlib.sha256(payload).hexdigest(),
    }
    artifact_tmp = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    artifact_tmp.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(artifact_tmp, artifact_path)
    metrics.update({"artifact": str(artifact_path), "loaded": False})
    return tables, metrics


def _train_or_load_npq0_l_tables(
    source: GgufRowSource,
    item: GgufTensorPlan,
    source_path: Path,
    recipe_path: Path,
    artifact_root: Path,
    config: Npq0LConfig,
    train_rows: int,
    validation_rows: int,
    seed: int,
    device: str,
    imatrix: ImportanceMatrix | None = None,
    imatrix_binding: ImatrixBinding | None = None,
) -> tuple[Npq0LTables, dict[str, Any]]:
    """Fit or reuse tensor-wise NPQ0-L product tables on disjoint rows."""

    if imatrix_binding is not None and imatrix is None:
        raise ValueError("an imatrix binding requires its source imatrix")
    train_index, validation_index = _sample_codebook_rows(
        item,
        train_rows,
        validation_rows,
        seed,
    )
    if not train_index.size or not validation_index.size:
        raise ValueError(f"tensor-wise NPQ0-L training needs non-empty splits: {item.name}")
    signature = {
        "source": _file_identity(source_path),
        "recipe": _file_identity(recipe_path),
        "tensor_name": item.name,
        "source_name": item.source_name,
        "target_dtype": item.target_dtype,
        "storage_shape": list(item.storage_shape),
        "train_rows": train_index.tolist(),
        "validation_rows": validation_index.tolist(),
        "seed": seed,
        "config": _npq0_l_config_dict(config),
        "objective": (
            "imatrix_weighted_sse"
            if imatrix_binding is not None
            else "unweighted_weight_sse"
        ),
        "imatrix": (
            None
            if imatrix_binding is None
            else {
                "file": _file_identity(imatrix.path),
                "entry": imatrix_binding.entry_name,
            }
        ),
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = _codebook_artifact_path(artifact_root, item)
    if artifact_path.exists():
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
        if document.get("format") != "mfq.tensorwise-npq0-l-pq7.v1":
            raise ValueError(f"unsupported NPQ0-L artifact: {artifact_path}")
        if document.get("signature") != signature:
            raise ValueError(f"stale NPQ0-L artifact signature: {artifact_path}")
        payload = base64.b64decode(document["tables_b64"], validate=True)
        if hashlib.sha256(payload).hexdigest() != document.get("tables_sha256"):
            raise ValueError(f"NPQ0-L artifact checksum mismatch: {artifact_path}")
        scale, first, second, consumed = _unpack_npq0_l_tables(payload)
        if consumed != len(payload):
            raise ValueError(f"NPQ0-L artifact has an invalid table tail: {artifact_path}")
        metrics = dict(document["metrics"])
        metrics.update({"artifact": str(artifact_path), "loaded": True})
        return Npq0LTables(scale, first, second), metrics

    train_weight = source.read_rows(train_index)
    train_importance = (
        None if imatrix_binding is None else imatrix_binding.selected(train_index)
    )
    trained, history = train_npq0_l(
        train_weight,
        importance=train_importance,
        config=config,
        device=device,
    )
    tables = npq0_l_tables_from_tensor(trained)
    validation_weight = source.read_rows(validation_index)
    validation_importance = (
        None
        if imatrix_binding is None
        else imatrix_binding.selected(validation_index)
    )
    validation = quantize_npq0_l_fixed(
        validation_weight,
        tables,
        importance=validation_importance,
        config=config,
        device=device,
    )
    reference = validation_weight.numpy().astype(np.float32, copy=False)
    reconstruction = dequantize_npq0_l(validation)
    error_elementwise = np.square(reconstruction - reference, dtype=np.float64)
    signal_elementwise = np.square(reference, dtype=np.float64)
    if validation_importance is not None:
        error_elementwise *= validation_importance
        signal_elementwise *= validation_importance
    error = float(error_elementwise.sum())
    signal = float(signal_elementwise.sum())
    best = min(history, key=lambda result: result.weighted_sse)
    metrics = {
        "objective": signature["objective"],
        "train_best_nmse_percent": best.weighted_nmse_percent,
        "validation_nmse_percent": 100.0 * error / signal if signal else 0.0,
        "validation_snr_db": (
            10.0 * math.log10(signal / error) if error else math.inf
        ),
        "used_states": best.used_states,
        "used_first_codes": list(best.used_first_codes),
        "used_second_codes": list(best.used_second_codes),
        "imatrix_entry": (
            None if imatrix_binding is None else imatrix_binding.entry_name
        ),
    }
    payload = _pack_npq0_l_tables(
        tables.scale_lut,
        tables.first_codebooks,
        tables.second_codebooks,
    )
    document = {
        "format": "mfq.tensorwise-npq0-l-pq7.v1",
        "signature": signature,
        "metrics": metrics,
        "history": [
            {
                "iteration": result.iteration,
                "weighted_sse": result.weighted_sse,
                "weighted_nmse_percent": result.weighted_nmse_percent,
                "used_states": result.used_states,
                "used_first_codes": list(result.used_first_codes),
                "used_second_codes": list(result.used_second_codes),
            }
            for result in history
        ],
        "tables_b64": base64.b64encode(payload).decode("ascii"),
        "tables_sha256": hashlib.sha256(payload).hexdigest(),
    }
    artifact_tmp = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    artifact_tmp.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(artifact_tmp, artifact_path)
    metrics.update({"artifact": str(artifact_path), "loaded": False})
    return tables, metrics


def _train_or_load_tensor_codebook(
    source: GgufRowSource,
    item: GgufTensorPlan,
    source_path: Path,
    recipe_path: Path,
    artifact_root: Path,
    config: TensorCodebookTrainingConfig,
    train_rows: int,
    validation_rows: int,
    seed: int,
    imatrix: ImportanceMatrix | None = None,
    imatrix_binding: ImatrixBinding | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if imatrix_binding is not None and imatrix is None:
        raise ValueError("an imatrix binding requires its source imatrix")
    train_index, validation_index = _sample_codebook_rows(
        item,
        train_rows,
        validation_rows,
        seed,
    )
    if not train_index.size or not validation_index.size:
        raise ValueError(f"tensor-wise codebook training needs non-empty splits: {item.name}")
    signature = {
        "source": _file_identity(source_path),
        "recipe": _file_identity(recipe_path),
        "tensor_name": item.name,
        "source_name": item.source_name,
        "target_dtype": item.target_dtype,
        "storage_shape": list(item.storage_shape),
        "train_rows": train_index.tolist(),
        "validation_rows": validation_index.tolist(),
        "seed": seed,
        "config": _codebook_config_dict(config),
        "imatrix": (
            None
            if imatrix_binding is None
            else {
                "file": _file_identity(imatrix.path) if imatrix is not None else None,
                "entry": imatrix_binding.entry_name,
            }
        ),
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = _existing_codebook_artifact_path(
        _codebook_artifact_path(artifact_root, item)
    )
    spec = _NVQ_SPECS[item.target_dtype]

    if artifact_path.exists():
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
        if document.get("format") not in {
            "mfq.tensorwise-nvq-codebook.v1",
            "mfq.tensorwise-niq-codebook.v1",
        }:
            raise ValueError(f"unsupported codebook artifact: {artifact_path}")
        if _canonical_artifact_signature(document.get("signature", {})) != signature:
            raise ValueError(f"stale codebook artifact signature: {artifact_path}")
        payload = base64.b64decode(document["trained_codebook_b64"], validate=True)
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != document.get("trained_codebook_sha256"):
            raise ValueError(f"codebook artifact checksum mismatch: {artifact_path}")
        trained = (
            _unpack_nvq1_l_codebook(payload)
            if item.target_dtype == "NVQ1-L"
            else _unpack_nvq_codebook(spec, payload)
        )
        selected = trained if document["selected_custom"] else None
        metrics = dict(document["metrics"])
        metrics.update(
            {
                "artifact": str(artifact_path),
                "loaded": True,
                "selected_custom": bool(document["selected_custom"]),
            }
        )
        return selected, metrics

    train_weight = source.read_rows(train_index).numpy()
    validation_weight = source.read_rows(validation_index).numpy()
    train_importance = (
        None if imatrix_binding is None else imatrix_binding.selected(train_index)
    )
    validation_importance = (
        None
        if imatrix_binding is None
        else imatrix_binding.selected(validation_index)
    )
    result = train_tensor_codebook(
        item.name,
        train_weight,
        validation_weight,
        spec,
        config,
        train_importance=train_importance,
        validation_importance=validation_importance,
    )
    payload = (
        _pack_nvq1_l_codebook(result.trained_codebook)
        if item.target_dtype == "NVQ1-L"
        else _pack_nvq_codebook(spec, result.trained_codebook)
    )
    metrics = {
        "fixed_validation_sse": result.fixed_validation_sse,
        "trained_validation_sse": result.trained_validation_sse,
        "fixed_validation_snr_db": result.fixed_validation_snr_db,
        "trained_validation_snr_db": result.trained_validation_snr_db,
        "validation_sse_improvement_percent": (
            result.validation_sse_improvement_percent
        ),
        "objective": (
            "imatrix_weighted_sse" if imatrix_binding is not None else "unweighted_sse"
        ),
        "imatrix_entry": (
            None if imatrix_binding is None else imatrix_binding.entry_name
        ),
    }
    document = {
        "format": "mfq.tensorwise-nvq-codebook.v1",
        "signature": signature,
        "selected_custom": result.selected_custom,
        "metrics": metrics,
        "history": list(result.history),
        "trained_codebook_b64": base64.b64encode(payload).decode("ascii"),
        "trained_codebook_sha256": hashlib.sha256(payload).hexdigest(),
    }
    artifact_tmp = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    artifact_tmp.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(artifact_tmp, artifact_path)
    metrics.update(
        {
            "artifact": str(artifact_path),
            "loaded": False,
            "selected_custom": result.selected_custom,
        }
    )
    return result.selected_codebook, metrics


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    return value


def _field_value(reader: Any, key: str, default: Any = None) -> Any:
    field = reader.fields.get(key)
    if field is None:
        return default
    return _json_value(field.contents())


def _architecture_metadata(reader: Any) -> tuple[str, dict[str, Any]]:
    architecture = str(_field_value(reader, "general.architecture", "unknown"))
    prefix = architecture + "."
    metadata: dict[str, Any] = {}
    for key, field in reader.fields.items():
        if key.startswith(prefix):
            metadata[key] = _json_value(field.contents())
    return architecture, metadata


def _recipe_metadata(reader: Any) -> dict[str, Any]:
    keep = {"general.name", "general.quantized_by", "general.quantization_version"}
    result: dict[str, Any] = {}
    for key, field in reader.fields.items():
        if key in keep or key.startswith("quantize."):
            result[key] = _json_value(field.contents())
    return result


def _estimate_blob_bytes(
    item: GgufTensorPlan,
    *,
    custom_codebook: bool = False,
    jsc_banks: int = 4,
    expert_artifact_root: str | Path | None = None,
) -> int:
    n = math.prod(item.storage_shape)
    if item.target_dtype == "NINTM":
        if item.expert_shape is None or item.expert_precisions is None:
            raise ValueError(f"NINTM plan lacks expert metadata: {item.name}")
        return _mixed_moe_blob_nbytes(
            item.expert_shape,
            item.expert_precisions,
            expert_artifact_root,
        )
    if item.target_dtype == "NINT8-0":
        out, neuron_len = item.storage_shape
        return nint8_zero_payload_nbytes(
            (out, neuron_len),
            0,
            neuron_len,
        )
    if item.target_dtype.startswith("NINT"):
        spec = _NINT_SPECS[item.target_dtype]
        out, neuron_len = item.storage_shape
        ng = (neuron_len + spec.groupsize - 1) // spec.groupsize
        header = _NINT_HDR.size + 4 + 8 * len(item.storage_shape) + 8
        scales = out * 4
        sub = ((out * ng * spec.sub_bits + 7) // 8) * 2
        q = (out * ng * spec.groupsize * spec.bits + 7) // 8
        return header + scales + sub + q
    if item.target_dtype == "NPQ0-L":
        out, neuron_len = item.storage_shape
        header = _NPQ0_L_HEADER.size + 8 * len(item.storage_shape) + 4
        return header + NPQ0_L.payload_nbytes(out, neuron_len)
    if item.target_dtype == "NVQ1-L":
        spec = NVQ1_L_T8_S3
        out, neuron_len = item.storage_shape
        header = _NVQ1_L_HEADER.size + 8 * len(item.storage_shape) + 4
        return header + (4096 if custom_codebook else 0) + spec.payload_nbytes(out, neuron_len)
    if item.target_dtype in _JSC_DTYPES:
        spec = _NVQ_SPECS[item.target_dtype]
        out, neuron_len = item.storage_shape
        header = _NVQ_HEADER.size + 8 * len(item.storage_shape) + 4
        return (
            header
            + 64
            + jsc_banks * spec.codebook_entries * spec.vector_size
            + spec.payload_nbytes(out, neuron_len)
        )
    if item.target_dtype in {"NVQ2", "NVQ3"}:
        spec = _NVQ_SPECS[item.target_dtype]
        out, neuron_len = item.storage_shape
        header = _NVQ_HEADER.size + 8 * len(item.storage_shape) + 4
        return header + (512 if custom_codebook else 0) + spec.payload_nbytes(out, neuron_len)
    item_size = 4 if item.target_dtype == "F32" else 2
    return 4 + 8 * len(item.storage_shape) + n * item_size


def _read_dense_tensor(tensor: Any, shape: tuple[int, ...], dequantize) -> torch.Tensor:
    values = dequantize(tensor.data, tensor.tensor_type)
    values = np.ascontiguousarray(values, dtype=np.float32).reshape(shape)
    return torch.from_numpy(values)


def _write_nint8_zero_axis0_blob(
    source: GgufRowSource,
    shape: tuple[int, ...],
    blob_path: Path,
    row_chunk: int,
) -> int:
    if len(shape) != 2:
        raise ValueError(
            f"NINT8-0 stream writer only supports 2D tensors, got {shape}"
        )
    out, neuron_len = shape
    if neuron_len % 32:
        raise ValueError(
            f"NINT8-0 input width must be divisible by 32, got {neuron_len}"
        )
    with blob_path.open("wb") as handle:
        handle.write(pack_nint8_zero_header(shape, 0, neuron_len))
        for start in range(0, out, row_chunk):
            end = min(start + row_chunk, out)
            chunk = source[start:end].float().cpu().numpy()
            quantized = quantize_nint8_zero(chunk, axis=0)
            handle.write(
                pack_nint8_zero_blocks(quantized.scale, quantized.q)
            )
    actual = blob_path.stat().st_size
    expected = nint8_zero_payload_nbytes(shape, 0, neuron_len)
    if actual != expected:
        raise RuntimeError(
            f"NINT8-0 payload size mismatch: {actual} != {expected}"
        )
    return actual


def _check_row_stream_alignment(row_chunk: int, bits_per_row: list[int]) -> None:
    for bits in bits_per_row:
        if row_chunk * bits % 8:
            raise ValueError(
                f"row_chunk={row_chunk} does not byte-align a {bits}-bit row stream"
            )


def _quantize_nvq_chunk(
    weight: torch.Tensor,
    target_dtype: str,
    quant_backend: str,
    device: str,
    group_chunk: int,
    nvq1_l_candidates: int,
    nvq1_l_anchor_multipliers: tuple[float, ...],
    nvq1_l_refine_steps: int,
    importance: np.ndarray | None,
    codebook: np.ndarray | None,
    search_steps: int,
    nvq_native_assignment: bool,
    nvq1_l_native_assignment: bool,
    jsc_assignment_refine_steps: int = 2,
    jsc_tables: NvqJscTables | None = None,
    npq0_l_tables: Npq0LTables | None = None,
    npq0_l_config: Npq0LConfig | None = None,
):
    if target_dtype == "NPQ0-L":
        if npq0_l_tables is None or npq0_l_config is None:
            raise ValueError("NPQ0-L quantization requires fixed tensor-wise product tables")
        return quantize_npq0_l_fixed(
            weight,
            npq0_l_tables,
            importance=importance,
            config=npq0_l_config,
            device=device if quant_backend == "cuda" else "cpu",
        )
    if target_dtype in _JSC_DTYPES:
        if jsc_tables is None:
            raise ValueError(f"{target_dtype} quantization requires fixed tensor-wise tables")
        return quantize_nvq_jsc_fixed(
            weight,
            jsc_tables,
            importance=importance,
            assignment_refine_steps=jsc_assignment_refine_steps,
            search_steps=search_steps,
            group_chunk=group_chunk,
            device=device if quant_backend == "cuda" else "cpu",
        )
    if quant_backend == "cuda":
        from mfq.quantize.nvq_quant_torch import quantize_axis0

        return quantize_axis0(
            weight,
            _NVQ_SPECS[target_dtype],
            device=device,
            importance=importance,
            group_chunk=group_chunk,
            nvq1_l_candidates=nvq1_l_candidates,
            anchor_multipliers=nvq1_l_anchor_multipliers,
            refine_steps=nvq1_l_refine_steps,
            codebook=codebook,
            search_steps=search_steps,
            nvq_native_assignment=nvq_native_assignment,
            nvq1_l_native_assignment=nvq1_l_native_assignment,
        )
    array = weight.numpy()
    if target_dtype == "NVQ1-L":
        return nvq1_l_quantize(
            array,
            NVQ1_L_T8_S3,
            axis=0,
            importance=importance,
            group_chunk=group_chunk,
            anchor_multipliers=nvq1_l_anchor_multipliers,
            refine_steps=nvq1_l_refine_steps,
            codebook=codebook,
        )
    return nvq_quantize(
        array,
        _NVQ_SPECS[target_dtype],
        axis=0,
        importance=importance,
        group_chunk=group_chunk,
        codebook=codebook,
        search_steps=search_steps,
    )


def _dequantize_nvq_chunk(
    tensor: Any,
    target_dtype: str,
) -> np.ndarray:
    if target_dtype == "NPQ0-L":
        return dequantize_npq0_l(tensor)
    if target_dtype == "NVQ1-L":
        return dequantize_nvq1_l(tensor)
    if target_dtype in _JSC_DTYPES:
        return dequantize_nvq_jsc(tensor)
    return dequantize_nvq(tensor)


def _nvq_per_neuron_gain(
    weight: torch.Tensor,
    tensor: Any,
    target_dtype: str,
    importance: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(weight, dtype=np.float32).reshape(weight.shape[0], -1)
    reconstructed = np.asarray(
        _dequantize_nvq_chunk(tensor, target_dtype), dtype=np.float32
    ).reshape(reference.shape)
    return diagonal_regressed_gain(reference, reconstructed, importance)


def _write_nvq_blob(
    source: GgufRowSource,
    shape: tuple[int, int],
    target_dtype: str,
    blob_path: Path,
    row_chunk: int,
    quant_backend: str,
    device: str,
    group_chunk: int,
    nvq1_l_candidates: int,
    nvq1_l_anchor_multipliers: tuple[float, ...],
    nvq1_l_refine_steps: int,
    importance_rows: ImportanceRows | None = None,
    codebook: np.ndarray | None = None,
    search_steps: int = 19,
    nvq_native_assignment: bool = True,
    nvq1_l_native_assignment: bool = True,
    jsc_tables: NvqJscTables | None = None,
    jsc_assignment_refine_steps: int = 2,
    npq0_l_tables: Npq0LTables | None = None,
    npq0_l_config: Npq0LConfig | None = None,
    calibration_mode: str = "none",
) -> NvqBlobWriteResult:
    if calibration_mode not in {"none", "gain", "group24"}:
        raise ValueError(f"unsupported NVQ calibration mode: {calibration_mode}")
    if calibration_mode != "none" and importance_rows is None:
        raise ValueError(f"NVQ calibration mode {calibration_mode} requires an imatrix")
    if jsc_assignment_refine_steps < 0:
        raise ValueError("JSC assignment refine steps must be non-negative")
    out, neuron_len = shape
    if target_dtype == "NPQ0-L":
        if npq0_l_tables is None or npq0_l_config is None:
            raise ValueError("NPQ0-L blob writing requires fixed tensor-wise product tables")
        spec = NPQ0_L
        ng = math.ceil(neuron_len / spec.groupsize)
        nvec = neuron_len // spec.vector_size
        _check_row_stream_alignment(
            row_chunk,
            [ng * spec.state_bits, nvec * spec.index_bits],
        )
        header = _NPQ0_L_HEADER.pack(
            _NPQ0_L_MAGIC,
            _NPQ0_L_VERSION,
            spec.state_bits,
            spec.groupsize,
            0,
            neuron_len,
            len(shape),
        )
        stream_bits = [ng * spec.state_bits, nvec * spec.index_bits]
        codebook_payload = _pack_npq0_l_tables(
            npq0_l_tables.scale_lut,
            npq0_l_tables.first_codebooks,
            npq0_l_tables.second_codebooks,
        )
    elif target_dtype == "NVQ1-L":
        spec = NVQ1_L_T8_S3
        ng = math.ceil(neuron_len / spec.groupsize)
        nvec = math.ceil(neuron_len / spec.vector_size)
        _check_row_stream_alignment(
            row_chunk,
            [ng * spec.sub_bits, nvec * spec.index_bits, ng],
        )
        profile = _PROFILE_CUSTOM_TERNARY if codebook is not None else _PROFILE_IQ1S_GRID
        header = _NVQ1_L_HEADER.pack(
            _NVQ1_L_MAGIC,
            profile,
            spec.sub_bits,
            spec.groupsize,
            0,
            neuron_len,
            len(shape),
        )
        stream_bits = [ng * spec.sub_bits, nvec * spec.index_bits, ng]
        codebook_payload = b"" if codebook is None else _pack_nvq1_l_codebook(codebook)
    elif target_dtype in _JSC_DTYPES:
        if jsc_tables is None:
            raise ValueError(f"{target_dtype} blob writing requires fixed tensor-wise tables")
        spec = _NVQ_SPECS[target_dtype]
        if jsc_tables.spec != spec:
            raise ValueError(
                f"{target_dtype} table base mismatch: {jsc_tables.spec.codebook}"
            )
        ng = math.ceil(neuron_len / spec.groupsize)
        nvec = math.ceil(neuron_len / spec.vector_size)
        nsign = math.ceil(neuron_len / 8)
        _check_row_stream_alignment(
            row_chunk,
            [ng * 4, nvec * spec.index_bits, nsign * 7],
        )
        header = _NVQ_HEADER.pack(
            _NVQ_MAGIC,
            _CODEBOOK_ID[spec.codebook] | _JSC_FLAG,
            4,
            24,
            0,
            neuron_len,
            len(shape),
        )
        stream_bits = [ng * 4, nvec * spec.index_bits, nsign * 7]
        codebook_payload = _pack_jsc_tables(
            jsc_tables.scale_lut,
            jsc_tables.bank_for_state,
            jsc_tables.codebooks,
        )
    else:
        spec = _NVQ_SPECS[target_dtype]
        ng = math.ceil(neuron_len / spec.groupsize)
        nvec = math.ceil(neuron_len / spec.vector_size)
        nsign = math.ceil(neuron_len / 8)
        _check_row_stream_alignment(
            row_chunk,
            [ng * spec.sub_bits, nvec * spec.index_bits, nsign * 7],
        )
        encoded_codebook = _CODEBOOK_ID[spec.codebook]
        if spec.sign_mode == "index_parity":
            encoded_codebook |= _INDEX_PARITY_FLAG
        if codebook is not None:
            encoded_codebook |= _CUSTOM_CODEBOOK_FLAG
        header = _NVQ_HEADER.pack(
            _NVQ_MAGIC,
            encoded_codebook,
            spec.sub_bits,
            spec.groupsize,
            0,
            neuron_len,
            len(shape),
        )
        stream_bits = [ng * spec.sub_bits, nvec * spec.index_bits, nsign * 7]
        codebook_payload = b"" if codebook is None else _pack_nvq_codebook(spec, codebook)

    gain_values: list[np.ndarray] = []
    with blob_path.open("wb+") as f:
        f.write(header)
        f.write(struct.pack(f"<{len(shape)}q", *shape))
        f.write(struct.pack("<I", out))
        f.write(codebook_payload)
        anchor_off = f.tell()
        stream_offsets: list[int] = []
        offset = anchor_off + out * 2
        for bits in stream_bits:
            stream_offsets.append(offset)
            offset += (out * bits + 7) // 8
        f.truncate(offset)

        for start in range(0, out, row_chunk):
            end = min(start + row_chunk, out)
            importance = None if importance_rows is None else importance_rows(start, end)
            quant_importance = importance
            weight = source[start:end]
            tensor = _quantize_nvq_chunk(
                weight,
                target_dtype,
                quant_backend,
                device,
                group_chunk,
                nvq1_l_candidates,
                nvq1_l_anchor_multipliers,
                nvq1_l_refine_steps,
                quant_importance,
                codebook,
                search_steps,
                nvq_native_assignment,
                nvq1_l_native_assignment,
                jsc_assignment_refine_steps=jsc_assignment_refine_steps,
                jsc_tables=jsc_tables,
                npq0_l_tables=npq0_l_tables,
                npq0_l_config=npq0_l_config,
            )
            anchors = np.asarray(tensor.neuron_scale, dtype=np.float32)
            if calibration_mode != "none":
                gain = _nvq_per_neuron_gain(
                    weight,
                    tensor,
                    target_dtype,
                    np.asarray(importance, dtype=np.float32),
                )
                gain_values.append(gain)
                anchors = anchors * gain
            f.seek(anchor_off + start * 2)
            f.write(np.ascontiguousarray(anchors, dtype=np.float16).tobytes())
            f.seek(stream_offsets[0] + (start * stream_bits[0]) // 8)
            if target_dtype == "NPQ0-L":
                f.write(_pack_nvq_bits(tensor.state, spec.state_bits))
                f.seek(stream_offsets[1] + (start * stream_bits[1]) // 8)
                f.write(_pack_nvq_bits(tensor.indices, spec.index_bits))
            elif target_dtype == "NVQ1-L":
                f.write(_pack_nvq1_l_bits(tensor.sub_scale, spec.sub_bits))
                f.seek(stream_offsets[1] + (start * stream_bits[1]) // 8)
                f.write(_pack_nvq1_l_bits(tensor.indices, spec.index_bits))
                f.seek(stream_offsets[2] + (start * stream_bits[2]) // 8)
                f.write(_pack_nvq1_l_bits(tensor.delta_sign, 1))
            else:
                f.write(_pack_nvq_bits(tensor.sub_scale, spec.sub_bits))
                f.seek(stream_offsets[1] + (start * stream_bits[1]) // 8)
                f.write(_pack_nvq_bits(tensor.indices, spec.index_bits))
                f.seek(stream_offsets[2] + (start * stream_bits[2]) // 8)
                f.write(_pack_nvq_bits(tensor.signs, 7))
            del tensor

        gain_calibration = None
        if gain_values:
            gains = np.concatenate(gain_values)
            gain_calibration = {
                "mode": "per_neuron_diagonal_regression",
                "assignment": (
                    "imatrix_group24" if calibration_mode == "group24" else "weight_only"
                ),
                "rows": int(gains.size),
                "gain_p01": float(np.quantile(gains, 0.01)),
                "gain_p50": float(np.quantile(gains, 0.50)),
                "gain_p99": float(np.quantile(gains, 0.99)),
            }
    return NvqBlobWriteResult(blob_path.stat().st_size, gain_calibration)


def convert(args: argparse.Namespace) -> None:
    source_path = Path(args.input_bf16_gguf).resolve()
    recipe_path = Path(args.recipe_gguf).resolve()
    output = Path(args.output).resolve()
    split_max_size = int(getattr(args, "split_max_size", 0))
    split_max_tensors = int(getattr(args, "split_max_tensors", 0))
    validate_split_limits(split_max_size, split_max_tensors)
    if (
        output.exists() or matching_shard_paths(output)
    ) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    GGUFReader, dequantize = _load_gguf()
    source_reader = GGUFReader(str(source_path), "r")
    recipe_reader = GGUFReader(str(recipe_path), "r")
    excluded_recipe_tensors: list[str] = []
    plan = _build_plan(
        source_reader,
        recipe_reader,
        exclude_mtp=bool(getattr(args, "exclude_mtp", False)),
        excluded_recipe_tensors=excluded_recipe_tensors,
    )
    calibration_scheme_arg = getattr(args, "calibration_scheme", "")
    calibration_scheme = (
        load_scheme(Path(calibration_scheme_arg).resolve())
        if calibration_scheme_arg
        else None
    )
    expert_artifact_root = (
        calibration_scheme.path.parent
        if calibration_scheme is not None and calibration_scheme.path is not None
        else None
    )
    plan = _apply_expert_scheme(plan, calibration_scheme)
    npq0_l_enabled = bool(getattr(args, "npq0_l", False))
    plan = _apply_npq0_l_mapping(plan, npq0_l_enabled)
    nvq3_jsc_enabled = bool(getattr(args, "nvq3_jsc", False))
    nvq3_jsc_512_enabled = bool(getattr(args, "nvq3_jsc_512", False))
    nvq3_to_nint3_enabled = bool(
        getattr(args, "nvq3_to_nint3", False)
    )
    if sum(
        (
            nvq3_jsc_enabled,
            nvq3_jsc_512_enabled,
            nvq3_to_nint3_enabled,
        )
    ) > 1:
        raise ValueError(
            "--nvq3-jsc, --nvq3-jsc-512, and --nvq3-to-nint3 "
            "are mutually exclusive"
        )
    plan = _apply_nvq3_jsc_mapping(
        plan,
        nvq3_jsc_enabled or nvq3_jsc_512_enabled,
        target_dtype="NVQ3J-512" if nvq3_jsc_512_enabled else "NVQ3J",
    )
    plan = _apply_nvq3_to_nint3_mapping(
        plan, nvq3_to_nint3_enabled
    )
    iq2_s_to_nint2_enabled = bool(
        getattr(args, "iq2_s_to_nint2", False)
    )
    plan = _apply_iq2_s_to_nint2_mapping(
        plan,
        iq2_s_to_nint2_enabled,
    )
    q8_to_nint8_zero_enabled = bool(
        getattr(args, "q8_to_nint8_zero", False)
    )
    plan = _apply_q8_to_nint8_zero_mapping(
        plan,
        q8_to_nint8_zero_enabled,
    )
    tensor_precision_overrides_arg = getattr(
        args, "tensor_precision_overrides", ""
    )
    tensor_precision_overrides = (
        _load_tensor_precision_overrides(tensor_precision_overrides_arg)
        if tensor_precision_overrides_arg
        else {}
    )
    plan = _apply_tensor_precision_overrides(
        plan,
        tensor_precision_overrides,
    )
    hf_name_template_arg = getattr(args, "hf_name_template", "")
    hf_name_template = (
        Path(hf_name_template_arg).resolve()
        if hf_name_template_arg
        else None
    )
    template_config: dict[str, Any] | None = None
    if hf_name_template is None:
        output_name_by_gguf = {item.name: item.name for item in plan}
    else:
        with open_mmap(hf_name_template) as template:
            output_name_by_gguf = _hf_output_name_map(
                [item.name for item in plan],
                list(template.records),
            )
            raw_template_config = template.header.extra.get("hf_config")
            if isinstance(raw_template_config, dict):
                template_config = raw_template_config
        plan = [
            item for item in plan if item.name in output_name_by_gguf
        ]
    if npq0_l_enabled and args.nvq_codebook_scope != "tensor":
        raise ValueError("--npq0-l requires --nvq-codebook-scope tensor")
    if args.limit_tensors:
        plan = plan[: args.limit_tensors]

    imatrix_path_arg = getattr(args, "imatrix", "")
    requested_calibration = getattr(args, "nvq_calibration", "auto")
    calibration_mode = (
        "gain" if requested_calibration == "auto" and imatrix_path_arg
        else "none" if requested_calibration == "auto"
        else requested_calibration
    )
    if calibration_mode != "none" and not imatrix_path_arg:
        raise ValueError(f"NVQ calibration mode {calibration_mode} requires --imatrix")
    imatrix = (
        load_importance_matrix(Path(imatrix_path_arg)) if imatrix_path_arg else None
    )
    imatrix_bindings = {} if imatrix is None else _bind_imatrix(imatrix, plan)
    row_importance_path_arg = getattr(args, "nvq_jsc_row_importance", "")
    row_importance = (
        load_row_importance(Path(row_importance_path_arg))
        if row_importance_path_arg
        else None
    )
    if row_importance is not None:
        if imatrix is None or calibration_mode != "group24":
            raise ValueError(
                "--nvq-jsc-row-importance requires --imatrix and "
                "--nvq-calibration group24"
            )
        if args.nvq_codebook_scope != "tensor":
            raise ValueError(
                "--nvq-jsc-row-importance requires --nvq-codebook-scope tensor"
            )
        for item in plan:
            if item.target_dtype in _JSC_DTYPES:
                row_importance.require(item.name, int(item.storage_shape[0]))

    quant_backend = args.quant_backend
    if quant_backend == "auto":
        quant_backend = "cuda" if torch.cuda.is_available() else "cpu"
    if quant_backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA quant backend requested but torch.cuda.is_available() is false")
    codebook_artifact_root = (
        Path(args.nvq_codebook_artifact_dir).resolve()
        if args.nvq_codebook_artifact_dir
        else Path(str(output) + ".codebooks")
    )
    codebook_config = TensorCodebookTrainingConfig(
        iterations=args.nvq_codebook_iterations,
        projection_candidates=args.nvq_codebook_projection_candidates,
        quant_backend=quant_backend,
        device=args.device if quant_backend == "cuda" else "cpu",
        group_chunk=args.nvq_group_chunk,
        row_chunk=args.nvq_codebook_row_chunk,
        search_steps=args.nvq_search_steps,
        nvq1_l_anchor_multipliers=tuple(args.nvq1_l_anchor_multipliers),
        nvq1_l_refine_steps=args.nvq1_l_refine_steps,
        nvq_native_assignment=args.nvq_assignment == "native",
        nvq1_l_native_assignment=args.nvq1_l_assignment == "native",
        min_validation_improvement=args.nvq_codebook_min_improvement,
    )
    jsc_config = NvqJscConfig(
        banks=getattr(args, "nvq_jsc_banks", 4),
        iterations=getattr(args, "nvq_jsc_iterations", 4),
        assignment_refine_steps=getattr(args, "nvq_jsc_assignment_refine_steps", 2),
        search_steps=args.nvq_search_steps,
        raw_multiplier=getattr(args, "nvq_jsc_raw_multiplier", 8),
        learned_scale_lut=True,
        codebook_storage="int8",
        group_chunk=args.nvq_group_chunk,
        seed=args.nvq_codebook_seed,
    )
    npq0_l_config = Npq0LConfig(
        iterations=getattr(args, "npq0_l_iterations", 4),
        assignment_refine_steps=getattr(args, "npq0_l_assignment_refine_steps", 2),
        fixed_refine_steps=getattr(args, "npq0_l_fixed_refine_steps", 3),
        kmeans_iterations=getattr(args, "npq0_l_kmeans_iterations", 8),
        group_chunk=getattr(args, "npq0_l_group_chunk", 512),
        seed=args.nvq_codebook_seed,
    )

    source_by_name = {str(t.name): t for t in source_reader.tensors}
    target_counts: dict[str, int] = {}
    target_bytes: dict[str, int] = {}
    recipe_counts: dict[str, int] = {}
    estimated_bytes = 0
    for item in plan:
        nbytes = _estimate_blob_bytes(
            item,
            custom_codebook=(
                args.nvq_codebook_scope == "tensor"
                and item.target_dtype.startswith("NVQ")
                and item.target_dtype not in _JSC_DTYPES
            ),
            jsc_banks=(
                getattr(args, "nvq3_jsc_banks", 2)
                if item.target_dtype in {"NVQ3J", "NVQ3J-512", "NVQ3J-L"}
                else jsc_config.banks
            ),
            expert_artifact_root=expert_artifact_root,
        )
        estimated_bytes += nbytes
        target_counts[item.target_dtype] = target_counts.get(item.target_dtype, 0) + 1
        target_bytes[item.target_dtype] = target_bytes.get(item.target_dtype, 0) + nbytes
        recipe_counts[item.recipe_type] = recipe_counts.get(item.recipe_type, 0) + 1

    free_bytes = shutil.disk_usage(output.parent).free
    required_peak = 2 * estimated_bytes + 1024**3
    contract = {
        "input_bf16_gguf": str(source_path),
        "recipe_gguf": str(recipe_path),
        "hf_name_template": (
            None if hf_name_template is None else str(hf_name_template)
        ),
        "tensor_name_namespace": (
            "gguf" if hf_name_template is None else "huggingface"
        ),
        "excluded_recipe_tensors": excluded_recipe_tensors,
        "output": str(output),
        "tensors": len(plan),
        "recipe_counts": dict(sorted(recipe_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "target_estimated_gb": {
            key: round(value / 1e9, 3) for key, value in sorted(target_bytes.items())
        },
        "estimated_mfq_gb": round(estimated_bytes / 1e9, 3),
        "required_peak_gb": round(required_peak / 1e9, 3),
        "output_free_gb": round(free_bytes / 1e9, 3),
        "quant_backend": quant_backend,
        "device": args.device if quant_backend == "cuda" else "cpu",
        "row_chunk": args.row_chunk,
        "nvq_group_chunk": args.nvq_group_chunk,
        "nvq1_l_anchor_multipliers": list(args.nvq1_l_anchor_multipliers),
        "nvq1_l_refine_steps": args.nvq1_l_refine_steps,
        "nvq1_l_candidates": args.nvq1_l_candidates,
        "nvq1_l_assignment": args.nvq1_l_assignment,
        "nvq_search_steps": args.nvq_search_steps,
        "nvq_assignment": args.nvq_assignment,
        "nvq_calibration": calibration_mode,
        "nvq_jsc": _jsc_config_dict(jsc_config),
        "nvq3_jsc": nvq3_jsc_enabled,
        "nvq3_jsc_512": nvq3_jsc_512_enabled,
        "iq2_s_to_nint2": iq2_s_to_nint2_enabled,
        "q8_to_nint8_zero": q8_to_nint8_zero_enabled,
        "tensor_precision_overrides": {
            "path": (
                str(Path(tensor_precision_overrides_arg).resolve())
                if tensor_precision_overrides_arg
                else None
            ),
            "count": len(tensor_precision_overrides),
            "values": dict(sorted(tensor_precision_overrides.items())),
        },
        "nvq3_jsc_profile": {
            "banks": getattr(args, "nvq3_jsc_banks", 2),
            "learned_scale_lut": bool(
                getattr(args, "nvq3_jsc_learned_scale", False)
            ),
        },
        "npq0_l": {
            "enabled": npq0_l_enabled,
            "mapping": "NVQ1-L recipe tensors -> NPQ0-L",
            "config": _npq0_l_config_dict(npq0_l_config),
        },
        "nvq_codebook_scope": args.nvq_codebook_scope,
        "nvq_codebook_artifact_dir": str(codebook_artifact_root),
        "nvq_codebook_train_rows": args.nvq_codebook_train_rows,
        "nvq_codebook_validation_rows": args.nvq_codebook_validation_rows,
        "nvq_codebook_seed": args.nvq_codebook_seed,
        "nvq_codebook_config": _codebook_config_dict(codebook_config),
        "mapping": dict(sorted(_RECIPE_TARGETS.items())),
        "expert_calibration_scheme": (
            str(Path(calibration_scheme_arg).resolve()) if calibration_scheme_arg else None
        ),
        "imatrix": None if imatrix is None else {
            "path": str(imatrix.path),
            "entries": len(imatrix.entries),
            "bound_tensors": len(imatrix_bindings),
            "bound_nint_tensors": sum(
                item.target_dtype in _IMATRIX_NINT_DTYPES
                and item.name in imatrix_bindings
                for item in plan
            ),
            "bound_nvq_tensors": sum(
                item.target_dtype.startswith("NVQ")
                and item.name in imatrix_bindings
                for item in plan
            ),
            "objective": "activation_second_moment_weighted_sse",
            "datasets": list(imatrix.datasets),
            "chunk_count": imatrix.chunk_count,
            "chunk_size": imatrix.chunk_size,
            "legacy": imatrix.legacy,
        },
        "nvq_jsc_row_importance": None if row_importance is None else {
            "path": str(row_importance.path),
            "file": _file_identity(row_importance.path),
            "entries": len(row_importance.entries),
            "objective": "factorized_row_fisher_x_imatrix_sse",
            "metadata": row_importance.metadata,
        },
        "weight_only_default": not bool(imatrix_path_arg),
    }
    print(json.dumps(contract, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    if free_bytes < required_peak:
        raise OSError(
            f"insufficient free space under {output.parent}: "
            f"need about {required_peak / 1e9:.1f} GB, have {free_bytes / 1e9:.1f} GB"
        )

    tmp_root = output.parent / f".{output.name}.tmp_blobs"
    resume_completed = int(getattr(args, "resume_completed", 0))
    if resume_completed < 0 or resume_completed > len(plan):
        raise ValueError(
            f"--resume-completed must be in [0, {len(plan)}], got {resume_completed}"
        )
    if resume_completed:
        if not tmp_root.is_dir():
            raise FileNotFoundError(
                f"--resume-completed requires the existing temporary directory: {tmp_root}"
            )
    else:
        if tmp_root.exists():
            raise FileExistsError(f"temporary directory exists: {tmp_root}")
        tmp_root.mkdir()
    records: list[BlobRecord] = []
    codebook_results: dict[str, dict[str, Any]] = {}
    gain_results: dict[str, dict[str, Any]] = {}
    started = time.time()

    try:
        for done, item in enumerate(plan, start=1):
            blob_path = tmp_root / f"{done:05d}.blob"
            t0 = time.time()
            if done <= resume_completed:
                if not blob_path.is_file() or blob_path.stat().st_size <= 0:
                    raise FileNotFoundError(
                        f"completed resume blob is absent or empty: {blob_path}"
                    )
                nbytes = blob_path.stat().st_size
                records.append(
                    BlobRecord(
                        output_name_by_gguf[item.name],
                        item.target_dtype,
                        nbytes,
                        blob_path,
                    )
                )
                print(
                    json.dumps(
                        {
                            "event": "tensor_resumed",
                            "done": done,
                            "total": len(plan),
                            "name": item.name,
                            "dtype": item.target_dtype,
                            "blob_mb": round(nbytes / 1e6, 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            source_tensor = source_by_name[item.source_name]
            if item.target_dtype == "NINTM":
                if item.expert_shape is None or item.expert_precisions is None:
                    raise ValueError(f"NINTM plan lacks expert metadata: {item.name}")
                row_source = GgufRowSource(source_tensor, item, dequantize)
                imatrix_binding = imatrix_bindings.get(item.name)
                expert_importance = None
                if imatrix_binding is not None:
                    n_experts, rows_per_expert, _ = item.expert_shape
                    expert_importance = imatrix_binding.selected(
                        np.arange(n_experts, dtype=np.int64) * rows_per_expert
                    )
                nbytes = _write_mixed_moe_axis0_blob(
                    row_source,
                    item.storage_shape,
                    item.expert_shape,
                    item.expert_precisions,
                    blob_path,
                    args.row_chunk,
                    quant_backend,
                    args.device,
                    expert_artifact_root,
                    importance=expert_importance,
                )
            elif item.target_dtype == "NINT8-0":
                row_source = GgufRowSource(source_tensor, item, dequantize)
                nbytes = _write_nint8_zero_axis0_blob(
                    row_source,
                    item.storage_shape,
                    blob_path,
                    args.row_chunk,
                )
            elif item.target_dtype.startswith("NINT"):
                row_source = GgufRowSource(source_tensor, item, dequantize)
                imatrix_binding = imatrix_bindings.get(item.name)
                nbytes = _write_nint_axis0_blob(
                    row_source,
                    item.storage_shape,
                    _NINT_SPECS[item.target_dtype],
                    blob_path,
                    args.row_chunk,
                    quant_backend,
                    args.device,
                    importance_rows=(
                        None if imatrix_binding is None else imatrix_binding.rows
                    ),
                )
            elif item.target_dtype.startswith("NVQ"):
                row_source = GgufRowSource(source_tensor, item, dequantize)
                imatrix_binding = imatrix_bindings.get(item.name)
                codebook = None
                jsc_tables = None
                item_jsc_config = (
                    replace(
                        jsc_config,
                        spec=_NVQ_SPECS[item.target_dtype],
                        banks=(
                            getattr(args, "nvq3_jsc_banks", 2)
                            if item.target_dtype in {"NVQ3J", "NVQ3J-512", "NVQ3J-L"}
                            else jsc_config.banks
                        ),
                        learned_scale_lut=(
                            bool(getattr(args, "nvq3_jsc_learned_scale", False))
                            if item.target_dtype in {"NVQ3J", "NVQ3J-512", "NVQ3J-L"}
                            else jsc_config.learned_scale_lut
                        ),
                    )
                    if item.target_dtype in _JSC_DTYPES
                    else jsc_config
                )
                npq0_l_tables = None
                if item.target_dtype == "NPQ0-L":
                    npq0_l_tables, codebook_metrics = _train_or_load_npq0_l_tables(
                        row_source,
                        item,
                        source_path,
                        recipe_path,
                        codebook_artifact_root,
                        npq0_l_config,
                        args.nvq_codebook_train_rows,
                        args.nvq_codebook_validation_rows,
                        args.nvq_codebook_seed,
                        args.device if quant_backend == "cuda" else "cpu",
                        imatrix,
                        imatrix_binding,
                    )
                    codebook_results[item.name] = codebook_metrics
                    print(
                        json.dumps(
                            {
                                "event": "tensor_npq0_l_tables",
                                "name": item.name,
                                "dtype": item.target_dtype,
                                **codebook_metrics,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                elif item.target_dtype in _JSC_DTYPES and args.nvq_codebook_scope == "tensor":
                    jsc_tables, codebook_metrics = _train_or_load_jsc_tables(
                        row_source,
                        item,
                        source_path,
                        recipe_path,
                        codebook_artifact_root,
                        item_jsc_config,
                        args.nvq_codebook_train_rows,
                        args.nvq_codebook_validation_rows,
                        args.nvq_codebook_seed,
                        args.device if quant_backend == "cuda" else "cpu",
                        imatrix,
                        imatrix_binding,
                        row_importance,
                    )
                    codebook_results[item.name] = codebook_metrics
                    print(
                        json.dumps(
                            {
                                "event": "tensor_jsc_tables",
                                "name": item.name,
                                "dtype": item.target_dtype,
                                **codebook_metrics,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                elif item.target_dtype in _JSC_DTYPES:
                    jsc_tables = initial_jsc_tables(item_jsc_config)
                elif args.nvq_codebook_scope == "tensor":
                    codebook, codebook_metrics = _train_or_load_tensor_codebook(
                        row_source,
                        item,
                        source_path,
                        recipe_path,
                        codebook_artifact_root,
                        codebook_config,
                        args.nvq_codebook_train_rows,
                        args.nvq_codebook_validation_rows,
                        args.nvq_codebook_seed,
                        imatrix,
                        imatrix_binding,
                    )
                    codebook_results[item.name] = codebook_metrics
                    print(
                        json.dumps(
                            {
                                "event": "tensor_codebook",
                                "name": item.name,
                                "dtype": item.target_dtype,
                                **codebook_metrics,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                nvq_result = _write_nvq_blob(
                    source=row_source,
                    shape=item.storage_shape,
                    target_dtype=item.target_dtype,
                    blob_path=blob_path,
                    row_chunk=args.row_chunk,
                    quant_backend=quant_backend,
                    device=args.device,
                    group_chunk=args.nvq_group_chunk,
                    nvq1_l_candidates=args.nvq1_l_candidates,
                    nvq1_l_anchor_multipliers=tuple(args.nvq1_l_anchor_multipliers),
                    nvq1_l_refine_steps=args.nvq1_l_refine_steps,
                    importance_rows=(
                        None if imatrix_binding is None else imatrix_binding.rows
                    ),
                    codebook=codebook,
                    search_steps=args.nvq_search_steps,
                    nvq_native_assignment=args.nvq_assignment == "native",
                    nvq1_l_native_assignment=args.nvq1_l_assignment == "native",
                    jsc_tables=jsc_tables,
                    jsc_assignment_refine_steps=item_jsc_config.assignment_refine_steps,
                    npq0_l_tables=npq0_l_tables,
                    npq0_l_config=npq0_l_config,
                    calibration_mode=(
                        calibration_mode if imatrix_binding is not None else "none"
                    ),
                )
                nbytes = nvq_result.nbytes
                if nvq_result.gain_calibration is not None:
                    gain_results[item.name] = nvq_result.gain_calibration
            else:
                dense = _read_dense_tensor(source_tensor, item.original_shape, dequantize)
                nbytes = _dense_blob_from_tensor(dense, blob_path, item.target_dtype)
                del dense
            records.append(
                BlobRecord(
                    output_name_by_gguf[item.name],
                    item.target_dtype,
                    nbytes,
                    blob_path,
                )
            )
            print(
                json.dumps(
                    {
                        "done": done,
                        "total": len(plan),
                        "name": item.name,
                        "source": item.source_name,
                        "original_shape": item.original_shape,
                        "storage_shape": item.storage_shape,
                        "recipe_type": item.recipe_type,
                        "dtype": item.target_dtype,
                        "gain_calibration": gain_results.get(item.name),
                        "blob_mb": round(nbytes / 1e6, 2),
                        "sec": round(time.time() - t0, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if item.target_dtype == "NINTM":
                _trim_windows_working_set()

        architecture, architecture_metadata = _architecture_metadata(source_reader)
        runtime_assets = [
            gguf_metadata_asset(
                GGUFReader(
                    str(Path(args.tokenizer_gguf).resolve()),
                    "r",
                )
                if getattr(args, "tokenizer_gguf", "")
                else source_reader
            )
        ]
        config_path = discover_model_config(
            source_path,
            getattr(args, "model_config", "") or None,
        )
        if config_path is not None:
            runtime_assets.append(model_config_asset(config_path.read_bytes()))
        elif template_config is not None:
            runtime_assets.append(model_config_asset(template_config))
        else:
            print(
                json.dumps(
                    {
                        "warning": "output MFQ has no embedded model config",
                        "hint": "pass --model-config",
                    }
                ),
                flush=True,
            )
        for index, asset in enumerate(runtime_assets):
            asset_path = tmp_root / f"runtime-asset-{index:02d}.blob"
            asset_path.write_bytes(asset.data)
            records.append(
                BlobRecord(asset.name, ASSET_DTYPE, len(asset.data), asset_path)
            )
        flattened_shapes = {
            output_name_by_gguf[item.name]: list(item.original_shape)
            for item in plan
            if item.original_shape != item.storage_shape
        }
        header = FileHeader(
            version=2,
            model_arch=f"{architecture}-gguf-mfq-nint-recipe",
            num_tensors=len(records),
            extra={
                "source": str(source_path),
                "recipe": str(recipe_path),
                "tensor_name_namespace": (
                    "gguf" if hf_name_template is None else "huggingface"
                ),
                "hf_name_template": (
                    None if hf_name_template is None else str(hf_name_template)
                ),
                "excluded_recipe_tensors": excluded_recipe_tensors,
                "expert_calibration_scheme": (
                    str(Path(calibration_scheme_arg).resolve())
                    if calibration_scheme_arg
                    else None
                ),
                "policy": "gguf-recipe-iq2-to-nvq2j-kquant-to-nint",
                "recipe_type_mapping": dict(sorted(_RECIPE_TARGETS.items())),
                "recipe_type_overrides": {
                    "IQ2_S_to_NINT2": iq2_s_to_nint2_enabled,
                    "NVQ3_to_NINT3": nvq3_to_nint3_enabled,
                    "Q8_0_to_NINT8-0": q8_to_nint8_zero_enabled,
                    "tensor_precision_overrides": dict(
                        sorted(tensor_precision_overrides.items())
                    ),
                },
                "recipe_specs": {
                    key: {
                        "bits": value.bits,
                        "groupsize": value.groupsize,
                        "sub_bits": value.sub_bits,
                    }
                    for key, value in sorted(_NINT_SPECS.items())
                },
                "nvq_specs": {
                    key: value.label for key, value in sorted(_NVQ_SPECS.items())
                },
                "tensor_layout": {
                    "matrices": "row-major-[out,in]",
                    "experts": "flattened-[expert*out,in]",
                    "merged_gate_up_source": "split-per-expert-into-gate-and-up",
                },
                "original_shapes": flattened_shapes,
                "gguf_architecture": architecture,
                "gguf_architecture_metadata": architecture_metadata,
                "gguf_recipe_metadata": _recipe_metadata(recipe_reader),
                ASSET_MANIFEST_KEY: runtime_asset_manifest(runtime_assets),
                "target_counts": target_counts,
                "quant_backend": quant_backend,
                "device": args.device if quant_backend == "cuda" else "cpu",
                "nvq_quantizer": {
                    "group_chunk": args.nvq_group_chunk,
                    "nvq1_l_anchor_multipliers": list(args.nvq1_l_anchor_multipliers),
                    "nvq1_l_refine_steps": args.nvq1_l_refine_steps,
                    "nvq1_l_candidates": args.nvq1_l_candidates,
                    "nvq1_l_assignment": args.nvq1_l_assignment,
                    "search_steps": args.nvq_search_steps,
                    "nvq_assignment": args.nvq_assignment,
                    "jsc": _jsc_config_dict(jsc_config),
                    "calibration": {
                        "requested": requested_calibration,
                        "resolved": calibration_mode,
                        "default_with_imatrix": "per_neuron_diagonal_regression",
                        "group24": "imatrix_weighted_state_and_index_assignment_plus_gain",
                    },
                    "codebook_scope": args.nvq_codebook_scope,
                    "codebook_artifact_dir": str(codebook_artifact_root),
                    "codebook_train_rows": args.nvq_codebook_train_rows,
                    "codebook_validation_rows": args.nvq_codebook_validation_rows,
                    "codebook_seed": args.nvq_codebook_seed,
                    "codebook_config": _codebook_config_dict(codebook_config),
                    "jsc_row_importance": None if row_importance is None else {
                        "path": str(row_importance.path),
                        "file": _file_identity(row_importance.path),
                        "entries": len(row_importance.entries),
                        "objective": "factorized_row_fisher_x_imatrix_sse",
                        "metadata": row_importance.metadata,
                    },
                    "imatrix": None if imatrix is None else {
                        "path": str(imatrix.path),
                        "file": _file_identity(imatrix.path),
                        "entries": len(imatrix.entries),
                        "objective": "activation_second_moment_weighted_sse",
                        "bound_tensors": {
                            name: binding.entry_name
                            for name, binding in sorted(imatrix_bindings.items())
                        },
                        "datasets": list(imatrix.datasets),
                        "chunk_count": imatrix.chunk_count,
                        "chunk_size": imatrix.chunk_size,
                        "legacy": imatrix.legacy,
                    },
                },
                "tensor_codebook_results": {
                    output_name_by_gguf[name]: value
                    for name, value in codebook_results.items()
                },
                "tensor_gain_results": {
                    output_name_by_gguf[name]: value
                    for name, value in gain_results.items()
                },
            },
        )
        outputs = write_blob_record_shards(
            output,
            header,
            records,
            split_max_size=split_max_size,
            split_max_tensors=split_max_tensors,
            overwrite=bool(args.overwrite),
        )
        if len(outputs) > 1 and output.exists() and args.overwrite:
            output.unlink()
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output": str(outputs[0]),
                    "outputs": [str(path) for path in outputs],
                    "shard_count": len(outputs),
                    "output_gb": round(
                        sum(path.stat().st_size for path in outputs) / 1e9, 3
                    ),
                    "elapsed_sec": round(time.time() - started, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        if not args.keep_temp and tmp_root.exists():
            shutil.rmtree(tmp_root)


def _canonical_cli_args(argv: list[str]) -> list[str]:
    return [
        "--nvq" + argument[len("--niq") :] if argument.startswith("--niq") else argument
        for argument in argv
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bf16-gguf", required=True)
    parser.add_argument("--recipe-gguf", required=True)
    parser.add_argument(
        "--tokenizer-gguf",
        default="",
        help="override the source GGUF used for embedded tokenizer metadata",
    )
    parser.add_argument(
        "--model-config",
        default="",
        help="model config JSON to embed; defaults beside the BF16 GGUF",
    )
    parser.add_argument(
        "--hf-name-template",
        default="",
        help=(
            "optional MFQ model whose HF tensor names define the output "
            "namespace; rope_freqs.weight is omitted"
        ),
    )
    parser.add_argument(
        "--calibration-scheme",
        default="",
        help="optional v2 scheme with per-expert profiles keyed by GGUF tensor name",
    )
    parser.add_argument(
        "--tensor-precision-overrides",
        default="",
        help=(
            "optional JSON mapping exact GGUF tensor names to final MFQ dtypes; "
            "applied after recipe-wide mappings"
        ),
    )
    parser.add_argument(
        "--imatrix",
        default="",
        help="optional llama.cpp GGUF or legacy importance matrix for NINT/NVQ calibration",
    )
    parser.add_argument(
        "--nvq-calibration",
        choices=("auto", "none", "gain", "group24"),
        default="auto",
        help="auto selects per-neuron gain when --imatrix is present; weight-only otherwise",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--exclude-mtp",
        action="store_true",
        help="exclude recipe block tensors beyond the BF16 source's final main layer",
    )
    parser.add_argument("--row-chunk", type=int, default=1024)
    parser.add_argument("--nvq-group-chunk", type=int, default=32768)
    parser.add_argument("--nvq-search-steps", type=int, default=19)
    parser.add_argument(
        "--nvq-assignment",
        choices=("native", "torch"),
        default="native",
    )
    parser.add_argument("--nvq-jsc-banks", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--nvq-jsc-iterations", type=int, default=4)
    parser.add_argument("--nvq-jsc-assignment-refine-steps", type=int, default=2)
    parser.add_argument("--nvq-jsc-raw-multiplier", type=int, default=8)
    parser.add_argument(
        "--nvq3-jsc",
        action="store_true",
        help="map recipe tensors that would use NVQ3 to 256-entry tensor-wise NVQ3J",
    )
    parser.add_argument(
        "--nvq3-jsc-512",
        action="store_true",
        help="map recipe tensors that would use NVQ3 to 512-entry/9-bit NVQ3J",
    )
    parser.add_argument(
        "--nvq3-to-nint3",
        action="store_true",
        help="map IQ3 recipe tensors that would use NVQ3 to NINT3",
    )
    parser.add_argument(
        "--iq2-s-to-nint2",
        action="store_true",
        help="map recipe IQ2_S tensors from NVQ2J to NINT2",
    )
    parser.add_argument(
        "--q8-to-nint8-zero",
        action="store_true",
        help="directly quantize BF16 source tensors selected as Q8_0 into NINT8-0",
    )
    parser.add_argument("--nvq3-jsc-banks", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument(
        "--nvq3-jsc-learned-scale",
        action="store_true",
        help="learn the NVQ3J scale LUT instead of using the analytic state layout",
    )
    parser.add_argument(
        "--nvq-jsc-row-importance",
        default="",
        help="optional per-output-row Fisher artifact for tensor-wise JSC training",
    )
    parser.add_argument(
        "--npq0-l",
        action="store_true",
        help="map recipe tensors that would use NVQ1-L to tensor-wise NPQ0-L",
    )
    parser.add_argument("--npq0-l-iterations", type=int, default=4)
    parser.add_argument("--npq0-l-assignment-refine-steps", type=int, default=2)
    parser.add_argument("--npq0-l-fixed-refine-steps", type=int, default=3)
    parser.add_argument("--npq0-l-kmeans-iterations", type=int, default=8)
    parser.add_argument("--npq0-l-group-chunk", type=int, default=512)
    parser.add_argument("--nvq1-l-candidates", type=int, default=0)
    parser.add_argument(
        "--nvq1-l-anchor-multipliers",
        type=float,
        nargs="+",
        default=(0.75,),
    )
    parser.add_argument("--nvq1-l-refine-steps", type=int, default=2)
    parser.add_argument(
        "--nvq1-l-assignment",
        choices=("native", "torch"),
        default="native",
    )
    parser.add_argument(
        "--nvq-codebook-scope",
        choices=("fixed", "tensor"),
        default="tensor",
    )
    parser.add_argument("--nvq-codebook-artifact-dir", default="")
    parser.add_argument("--nvq-codebook-train-rows", type=int, default=2048)
    parser.add_argument("--nvq-codebook-validation-rows", type=int, default=512)
    parser.add_argument("--nvq-codebook-row-chunk", type=int, default=512)
    parser.add_argument("--nvq-codebook-iterations", type=int, default=4)
    parser.add_argument("--nvq-codebook-projection-candidates", type=int, default=48)
    parser.add_argument("--nvq-codebook-min-improvement", type=float, default=0.0)
    parser.add_argument("--nvq-codebook-seed", type=int, default=20260716)
    parser.add_argument("--quant-backend", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-tensors", type=int, default=0)
    parser.add_argument(
        "--resume-completed",
        type=int,
        default=0,
        help="reuse this many contiguous completed blobs from the existing temp directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    split = parser.add_mutually_exclusive_group()
    split.add_argument(
        "--split-max-size",
        type=parse_size,
        default=0,
        metavar="N[M|G]",
        help="write numbered MFQ shards with at most this tensor payload per shard",
    )
    split.add_argument(
        "--split-max-tensors",
        type=int,
        default=0,
        help="write numbered MFQ shards with at most this many tensors per shard",
    )
    return parser


def main() -> None:
    convert(build_parser().parse_args(_canonical_cli_args(sys.argv[1:])))


if __name__ == "__main__":
    main()
