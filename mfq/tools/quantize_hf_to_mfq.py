"""Convert a HF safetensors checkpoint to MFQ.

Default policy for the first Qwen3.5/3.6 smoke:
- all 2D tensors -> NINT4, axis=0
- all other tensors -> dense F16
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from mfq.calibration.artifact import (
    CalibrationScheme,
    ExpertPrecision,
    ExpertTensorSelection,
    load_scheme,
    nint_expert_precision,
)
from mfq.formats.assets import (
    ASSET_DTYPE,
    ASSET_MANIFEST_KEY,
    MINICPMO45_RESAMPLER_POS_EMBED_ASSET,
    MODEL_CONFIG_ASSET,
    TOKENIZER_GGUF_ASSET,
    RuntimeAsset,
    gguf_metadata_asset,
    is_asset_record,
    minicpmo45_resampler_pos_embed_asset,
    model_config_asset,
    runtime_asset_manifest,
)
from mfq.formats.header import MFQ_MAGIC, FileHeader
from mfq.formats.io import (
    _DENSE_NAMES,
    _NINT_HDR,
    _NINT_MOE_HDR,
    _NINT_MOE_MAGIC_V2,
    _NINT_MOE_POOL_V2_HDR,
    _NINT_MOE_ROTATION_HDR,
    _pack_nint_moe_runtime,
    _u32,
    pack_bits,
)
from mfq.formats.mx import mx_header_bytes
from mfq.formats.nepq import (
    _FLAG_ROTATED as _NEPQ_FLAG_ROTATED,
)
from mfq.formats.nepq import (
    _HEADER as _NEPQ_HEADER,
)
from mfq.formats.nepq import (
    _MAGIC as _NEPQ_MAGIC,
)
from mfq.formats.nepq import (
    _RESIDUAL_DICTIONARY_ENTRIES,
    _RESIDUAL_FLAG_SECOND,
    _RESIDUAL_HEADER,
    _RESIDUAL_MAGIC,
    _RESIDUAL_VERSION,
    NEPQ0_A,
    NEPQ0_L,
    NEPQ0_S,
    NEPQ1_A,
    NEPQ1_L,
    NEPQ1_S,
    validate_nepq,
)
from mfq.formats.nepq import (
    _VERSION as _NEPQ_VERSION,
)
from mfq.formats.nepq import (
    _pack_bits as _pack_nepq_bits,
)
from mfq.formats.nint import NINT2_SPEC, NintSpec
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
    pack_npq0_l_tables,
)
from mfq.formats.npq0_l import (
    _pack_bits as _pack_npq0_l_bits,
)
from mfq.formats.npq0_s import (
    _HEADER as _NPQ0_S_HEADER,
)
from mfq.formats.npq0_s import (
    _MAGIC as _NPQ0_S_MAGIC,
)
from mfq.formats.npq0_s import (
    _VERSION as _NPQ0_S_VERSION,
)
from mfq.formats.npq0_s import (
    NPQ0_S,
    pack_npq0_s_tables,
)
from mfq.formats.npq0_s import (
    _pack_bits as _pack_npq0_s_bits,
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
    jsc_payload_nbytes,
    pack_codebook,
    pack_jsc_group64,
    pack_jsc_tables,
    resolve_jsc_storage_layout,
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
    pack_ternary_codebook,
)
from mfq.formats.nvq1_l import (
    _pack_bits as _pack_nvq1_l_bits,
)
from mfq.formats.nvq1_s import (
    _HEADER as _NVQ1_S_HEADER,
)
from mfq.formats.nvq1_s import (
    _MAGIC as _NVQ1_S_MAGIC,
)
from mfq.formats.nvq1_s import (
    _VERSION as _NVQ1_S_VERSION,
)
from mfq.formats.nvq1_s import (
    NVQ1_S,
    NVQ1_S_SYNTHETIC_BANKS,
    pack_nvq1_s_banked_codebook,
)
from mfq.formats.runtime_profile import (
    RUNTIME_SAMPLING_METADATA_KEY,
    profile_for_new_mfq,
)
from mfq.formats.shards import (
    matching_shard_paths,
    parse_size,
    validate_split_limits,
    write_blob_record_shards,
)
from mfq.formats.tpq import (
    TPQ_PQ_SPECS_BY_LABEL,
    pack_tpq_indices,
    pack_tpq_pq_prefix,
    tpq_pq_payload_nbytes,
)
from mfq.quantize.backend import (
    ACCELERATOR_BACKENDS,
    QUANT_BACKENDS,
    resolve_quant_backend,
    resolve_row_chunk,
)
from mfq.quantize.expert_nint import (
    quantize_flat_cohort,
    resolve_precision_artifact,
)
from mfq.quantize.imatrix import ImportanceMatrix, load_importance_matrix
from mfq.quantize.nepq import NepqQuantConfig, quantize_nepq_fixed
from mfq.quantize.nepq_a import (
    NepqAArtifact,
    NepqAQuantConfig,
    quantize_nepq_a_fixed,
)
from mfq.quantize.nint_quant import quantize as nint_quantize
from mfq.quantize.nint_quant_torch import quantize_axis0 as nint_quantize_axis0_torch
from mfq.quantize.npq0_l import Npq0LConfig, Npq0LTables, quantize_npq0_l_fixed
from mfq.quantize.npq0_s import Npq0SConfig, Npq0STables, quantize_npq0_s_fixed
from mfq.quantize.nvq_jsc import NvqJscConfig, NvqJscTables, initial_jsc_tables
from mfq.quantize.tpq import quantize_tpq_pq_fixed


@dataclass(frozen=True)
class TensorPlan:
    name: str
    shard: str
    shape: tuple[int, ...]
    source_dtype: str
    target_dtype: str
    gguf_name: str | None = None
    gguf_type: str | None = None
    source_name: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    target_spec: NintSpec | None = None
    expert_shape: tuple[int, int, int] | None = None
    expert_precisions: tuple[ExpertPrecision, ...] | None = None
    transform: str | None = None
    expert_source_names: tuple[tuple[str, ...], ...] | None = None
    expert_source_shards: tuple[tuple[str, ...], ...] | None = None

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
class BlobRecord:
    name: str
    dtype: str
    nbytes: int
    path: Path


@dataclass(frozen=True)
class SourceTensorMetadata:
    """Container-neutral tensor metadata used by the shared planner."""

    name: str
    shard: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int = 0


_RAW_SAFETENSOR_HEADERS: dict[Path, tuple[int, dict[str, object]]] = {}


def _raw_safetensor_header(path: Path) -> tuple[int, dict[str, object]]:
    resolved = path.resolve()
    cached = _RAW_SAFETENSOR_HEADERS.get(resolved)
    if cached is not None:
        return cached
    with resolved.open("rb") as handle:
        header_nbytes_raw = handle.read(8)
        if len(header_nbytes_raw) != 8:
            raise ValueError(f"truncated safetensors header length: {resolved}")
        header_nbytes = struct.unpack("<Q", header_nbytes_raw)[0]
        header_raw = handle.read(header_nbytes)
        if len(header_raw) != header_nbytes:
            raise ValueError(f"truncated safetensors header: {resolved}")
    header = json.loads(header_raw)
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header: {resolved}")
    value = (8 + header_nbytes, header)
    _RAW_SAFETENSOR_HEADERS[resolved] = value
    return value


class _RawSafeTensorSlice:
    """Read contiguous tensor rows without a whole-shard memory mapping."""

    _NUMPY_DTYPES = {
        "BOOL": np.dtype("?"),
        "U8": np.dtype("u1"),
        "I8": np.dtype("i1"),
        "I16": np.dtype("<i2"),
        "U16": np.dtype("<u2"),
        "I32": np.dtype("<i4"),
        "U32": np.dtype("<u4"),
        "I64": np.dtype("<i8"),
        "U64": np.dtype("<u8"),
        "F16": np.dtype("<f2"),
        "BF16": np.dtype("<u2"),
        "F32": np.dtype("<f4"),
        "F64": np.dtype("<f8"),
    }

    def __init__(self, path: str | Path, name: str) -> None:
        self.path = Path(path).resolve()
        self.name = str(name)
        data_start, header = _raw_safetensor_header(self.path)
        entry = header.get(self.name)
        if not isinstance(entry, dict):
            raise KeyError(f"safetensors tensor is absent: {self.name}")
        dtype_name = str(entry.get("dtype", ""))
        if dtype_name not in self._NUMPY_DTYPES:
            raise ValueError(
                f"unsupported raw safetensors dtype {dtype_name}: {self.name}"
            )
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not shape
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise ValueError(f"invalid safetensors entry: {self.name}")
        self.dtype_name = dtype_name
        self.numpy_dtype = self._NUMPY_DTYPES[dtype_name]
        self.shape = tuple(int(value) for value in shape)
        self.data_offset = data_start + int(offsets[0])
        self.data_nbytes = int(offsets[1]) - int(offsets[0])
        expected_nbytes = int(np.prod(self.shape)) * self.numpy_dtype.itemsize
        if self.data_nbytes != expected_nbytes:
            raise ValueError(
                f"safetensors byte size differs for {self.name}: "
                f"{self.data_nbytes} != {expected_nbytes}"
            )

    @property
    def columns(self) -> int:
        return int(self.shape[-1])

    @property
    def rows(self) -> int:
        return int(np.prod(self.shape[:-1])) if len(self.shape) > 1 else 1

    def _decode(self, raw: bytes, shape: tuple[int, ...]) -> torch.Tensor:
        values = np.frombuffer(raw, dtype=self.numpy_dtype).copy()
        tensor = torch.from_numpy(values)
        if self.dtype_name == "BF16":
            tensor = tensor.view(torch.bfloat16)
        return tensor.reshape(shape)

    def read_rows(
        self,
        start: int | np.ndarray,
        end: int | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        if end is None:
            indices = np.asarray(start, dtype=np.int64).reshape(-1)
            if indices.size and (
                int(indices.min()) < 0 or int(indices.max()) >= self.rows
            ):
                raise IndexError(
                    f"safetensors row indices fall outside [0, {self.rows})"
                )
            mapped = np.memmap(
                self.path,
                mode="r",
                dtype=self.numpy_dtype,
                offset=self.data_offset,
                shape=(self.rows, self.columns),
            )
            try:
                values = np.ascontiguousarray(mapped[indices])
            finally:
                del mapped
            tensor = torch.from_numpy(values)
            if self.dtype_name == "BF16":
                tensor = tensor.view(torch.bfloat16)
            return tensor.to(device=device)
        start = int(start)
        if start < 0 or end < start or end > self.rows:
            raise IndexError(
                f"invalid safetensors row slice {start}:{end} of {self.rows}"
            )
        row_nbytes = self.columns * self.numpy_dtype.itemsize
        nbytes = (end - start) * row_nbytes
        with self.path.open("rb", buffering=0) as handle:
            handle.seek(self.data_offset + start * row_nbytes)
            raw = handle.read(nbytes)
        if len(raw) != nbytes:
            raise EOFError(
                f"short safetensors row read for {self.name}: "
                f"{len(raw)} != {nbytes}"
            )
        return self._decode(raw, (end - start, self.columns)).to(
            device=device
        )

    def read_expert_rows(
        self,
        expert: int,
        start: int,
        end: int,
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
        if len(self.shape) != 3:
            raise ValueError(
                f"expert row reads require rank 3, got {self.shape}"
            )
        n_experts, rows_per_expert, _ = self.shape
        if (
            expert < 0
            or expert >= n_experts
            or start < 0
            or end < start
            or end > rows_per_expert
        ):
            raise IndexError(
                f"invalid expert slice expert={expert} rows={start}:{end}"
            )
        row0 = expert * rows_per_expert + start
        row1 = expert * rows_per_expert + end
        return self.read_rows(row0, row1, device=device)

    def tensor(self) -> torch.Tensor:
        with self.path.open("rb", buffering=0) as handle:
            handle.seek(self.data_offset)
            raw = handle.read(self.data_nbytes)
        if len(raw) != self.data_nbytes:
            raise EOFError(
                f"short safetensors tensor read for {self.name}: "
                f"{len(raw)} != {self.data_nbytes}"
            )
        return self._decode(raw, self.shape)

    def __getitem__(self, key: slice) -> torch.Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("raw safetensors source accepts contiguous slices")
        start = 0 if key.start is None else int(key.start)
        end = self.rows if key.stop is None else int(key.stop)
        return self.read_rows(start, end, device="cpu")


class _HfPlanRowSource:
    """Expose one HF plan's flattened row range to shared quantizers."""

    def __init__(
        self,
        source: _RawSafeTensorSlice,
        item: TensorPlan,
    ) -> None:
        if len(item.shape) != 2:
            raise ValueError(
                f"flat HF row source requires rank 2, got {item.shape}"
            )
        self.source = source
        self.offset = int(item.row_start or 0)
        self.rows = int(item.shape[0])
        self.columns = int(item.shape[1])
        if self.offset + self.rows > source.rows:
            raise ValueError(
                f"HF row range exceeds source tensor for {item.name}"
            )

    def read_rows(
        self,
        start: int | np.ndarray,
        end: int | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        if end is None:
            indices = np.asarray(start, dtype=np.int64).reshape(-1)
            return self.source.read_rows(
                indices + self.offset,
                device=device,
            ).to(torch.float32)
        return self.source.read_rows(
            self.offset + int(start),
            self.offset + int(end),
            device=device,
        ).to(torch.float32)

    def __getitem__(self, key: slice) -> torch.Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("HF plan row source accepts contiguous slices")
        start = 0 if key.start is None else int(key.start)
        end = self.rows if key.stop is None else int(key.stop)
        return self.read_rows(start, end)


def _read_index(root: Path) -> dict[str, str]:
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            return dict(json.load(f)["weight_map"])
    shards = sorted(root.glob("*.safetensors"))
    if len(shards) != 1:
        raise FileNotFoundError(f"missing model.safetensors.index.json under {root}")
    with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
        return {name: shards[0].name for name in f.keys()}  # noqa: SIM118


def _hf_source_inventory(root: Path) -> dict[str, SourceTensorMetadata]:
    weight_map = _read_index(root)
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    inventory: dict[str, SourceTensorMetadata] = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(str(root / shard), framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_slice(name)
                shape = tuple(int(value) for value in tensor.get_shape())
                inventory[name] = SourceTensorMetadata(
                    name=name,
                    shard=shard,
                    shape=shape,
                    dtype=str(tensor.get_dtype()),
                )
    return inventory


_RECIPE_SPECS = {
    "Q2_K": NINT2_SPEC,
    "Q3_K": NintSpec(3, 24, 5),
    "Q4_0": NintSpec(4, 24, 6),
    "Q4_1": NintSpec(4, 24, 6),
    "Q4_K": NintSpec(4, 24, 6),
    "Q5_0": NintSpec(5, 28, 7),
    "Q5_1": NintSpec(5, 28, 7),
    "Q5_K": NintSpec(5, 28, 7),
    "Q6_K": NintSpec(6, 24, 7),
    "Q8_0": NintSpec(8, 48, 7),
}

# Keep HF-source recipe interpretation identical to the mature GGUF-source
# converter.  The source container changes; the requested MFQ dtype does not.
_RECIPE_TARGETS = {
    "IQ1_M": "NVQ1-L",
    "IQ2_S": "NVQ2J-XL",
    "IQ2_XS": "NVQ2J-L",
    "IQ2_XXS": "NVQ2J",
    "IQ3_S": "NVQ3J-L",
    "IQ3_XXS": "NVQ3J",
    "IQ4_NL": "NINT4",
    "IQ4_XS": "NINT4",
    "Q2_K": "NINT2",
    "Q3_K": "NINT3",
    "Q4_0": "NINT4",
    "Q4_1": "NINT4",
    "Q4_K": "NINT4",
    "Q5_0": "NINT5",
    "Q5_1": "NINT5",
    "Q5_K": "NINT5",
    "Q6_K": "NINT6",
    "Q8_0": "NINT8",
    "F32": "F32",
    "F16": "F16",
    "BF16": "BF16",
}

_NVQ_SPECS = {
    "NVQ2": NVQ2_E8,
    "NVQ2J": NVQ2_E8,
    "NVQ2J-L": NVQ2_E8_1024,
    "NVQ2J-XL": NVQ2_E8_4096,
    "NVQ3": NVQ3_D4,
    "NVQ3J": NVQ3_D4,
    "NVQ3J-512": NVQ3_D4_512,
    "NVQ3J-L": NVQ3_D4_1024,
}
_IMATRIX_NINT_DTYPES = {"NINT2", "NINT3", "NINT4", "NINT5", "NINT6"}
_IMATRIX_OPTIONAL_TENSORS = {"token_embd.weight", "output.weight"}
_NEPQ_SPECS = {
    "NEPQ0-A": NEPQ0_A,
    "NEPQ0-S": NEPQ0_S,
    "NEPQ0-L": NEPQ0_L,
    "NEPQ1-S": NEPQ1_S,
    "NEPQ1-L": NEPQ1_L,
    "NEPQ1-A": NEPQ1_A,
}

_JSC_DTYPES = frozenset(
    {
        "NVQ2J",
        "NVQ2J-L",
        "NVQ2J-XL",
        "NVQ3J",
        "NVQ3J-512",
        "NVQ3J-L",
    }
)
_TENSOR_OVERRIDE_DTYPES = frozenset(
    {
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
        "NPQ0-L",
    }
)


def _is_compact_dtype(dtype: str) -> bool:
    return dtype.startswith(("NINT", "NVQ")) or dtype == "NPQ0-L"


_NATIVE_SOURCE_BITS = {"MXFP4": 4.0, "MXFP8": 8.0}
_COMPACT_FAMILY_BITS = {
    "NVQ1-L": 1.0,
    "NVQ1-S": 1.0,
    "NPQ0-L": 1.0,
    "NPQ0-S": 1.0,
    "NVQ2": 2.0,
    "NVQ2J": 2.0,
    "NVQ2J-L": 2.0,
    "NVQ2J-XL": 2.0,
    "NVQ3": 3.0,
    "NVQ3J": 3.0,
    "NVQ3J-512": 3.0,
    "NVQ3J-L": 3.0,
    "NEPQ0-S": 1.0,
    "NEPQ0-L": 1.0,
    "NEPQ0-A": 1.0,
    "NEPQ1-S": 2.0,
    "NEPQ1-L": 2.0,
    "NEPQ1-A": 1.5625,
    "TPQ-X": 1.0,
    "TPQ-W": 1.5,
    "TPQ-V": 2.0,
    "TPQ-VV": 3.0,
}


def _compact_family_bits(dtype: str) -> float | None:
    match = re.fullmatch(r"NINT([0-9]+)(?:-0)?", dtype)
    if match is not None:
        return float(match.group(1))
    return _COMPACT_FAMILY_BITS.get(dtype)


def _validate_native_source_precision(plan: Sequence[TensorPlan]) -> None:
    """Forbid a native low-bit source from being promoted or requantized equally."""

    for item in plan:
        source_bits = _NATIVE_SOURCE_BITS.get(item.source_dtype)
        if source_bits is None:
            continue
        families = (
            [value.family for value in item.expert_precisions]
            if item.target_dtype == "NINTM" and item.expert_precisions is not None
            else [item.target_dtype]
        )
        for family in families:
            target_bits = _compact_family_bits(family)
            if target_bits is None or target_bits >= source_bits:
                raise ValueError(
                    f"native {item.source_dtype} source {item.name} cannot be "
                    f"converted to {family}; target precision must be strictly "
                    f"below {source_bits:g} bits"
                )


def _gguf_reader(path: Path):
    try:
        from gguf import GGUFReader  # type: ignore
    except ModuleNotFoundError:
        import sys

        gguf_py = Path(__file__).resolve().parents[2] / "references" / "llamacpp" / "gguf-py"
        if gguf_py.exists():
            sys.path.insert(0, str(gguf_py))
        from gguf import GGUFReader  # type: ignore

    return GGUFReader(str(path), "r")


def _load_gguf_recipe(path: Path) -> dict[str, str]:
    reader = _gguf_reader(path)
    return {str(t.name): str(t.tensor_type.name) for t in reader.tensors}


def _artifact_provenance_name(value: str) -> str | None:
    return Path(value).name if value else None


_LAYER_NAME_MAP = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.experts.down_proj": "ffn_down_exps.weight",
    "mlp.experts.gate_up_proj": "ffn_gate_up_exps.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
    "experts.down_proj": "ffn_down_exps.weight",
    "experts.gate_up_proj": "ffn_gate_up_exps.weight",
    "router.proj.weight": "ffn_gate_inp.weight",
    "router.scale": "ffn_gate_inp.scale",
    "router.per_expert_scale": "ffn_down_exps.scale",
    "layer_scalar": "layer_output_scale.weight",
    "pre_feedforward_layernorm.weight": "ffn_norm.weight",
    "pre_feedforward_layernorm_2.weight": "pre_ffw_norm_2.weight",
    "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
    "post_feedforward_layernorm_1.weight": "post_ffw_norm_1.weight",
    "post_feedforward_layernorm_2.weight": "post_ffw_norm_2.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.dt_bias": "ssm_dt.bias",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "linear_attn.A_log": "ssm_a",
}


def _hf_to_gguf_name(
    name: str,
    *,
    mtp_layer_index: int = 40,
) -> str | None:
    if name == "llm.lm_head.weight":
        return "output.weight"
    if name == "llm.model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "llm.model.norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"
    if name == "model.language_model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.language_model.norm.weight":
        return "output_norm.weight"
    if name == "mtp.fc.weight":
        return f"blk.{mtp_layer_index}.nextn.eh_proj.weight"
    if name == "mtp.pre_fc_norm_embedding.weight":
        return f"blk.{mtp_layer_index}.nextn.enorm.weight"
    if name == "mtp.pre_fc_norm_hidden.weight":
        return f"blk.{mtp_layer_index}.nextn.hnorm.weight"
    if name == "mtp.norm.weight":
        return f"blk.{mtp_layer_index}.nextn.shared_head_norm.weight"

    prefix = "model.language_model.layers."
    if name.startswith(prefix):
        rest = name[len(prefix):]
        layer, _, suffix = rest.partition(".")
        mapped = _LAYER_NAME_MAP.get(suffix)
        if mapped is None and suffix.endswith(".weight"):
            mapped = _LAYER_NAME_MAP.get(suffix[: -len(".weight")])
        return f"blk.{layer}.{mapped}" if mapped is not None else None

    minicpmo_prefix = "llm.model.layers."
    if name.startswith(minicpmo_prefix):
        rest = name[len(minicpmo_prefix) :]
        layer, _, suffix = rest.partition(".")
        mapped = (
            "ffn_norm.weight"
            if suffix == "post_attention_layernorm.weight"
            else _LAYER_NAME_MAP.get(suffix)
        )
        return f"blk.{layer}.{mapped}" if mapped is not None else None

    mtp_match = re.match(r"^mtp\.layers\.(\d+)\.(.+)$", name)
    if mtp_match is not None:
        layer = mtp_layer_index + int(mtp_match.group(1))
        suffix = mtp_match.group(2)
        mapped = _LAYER_NAME_MAP.get(suffix)
        return (
            f"blk.{layer}.{mapped}"
            if mapped is not None
            else None
        )
    return None


_MTP_PROTECTED_TENSORS = {
    "mtp.fc.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
}


def _mtp_inventory_status(
    inventory: dict[str, SourceTensorMetadata],
    model_config: dict[str, object],
) -> tuple[bool, int]:
    """Validate that a Qwen MTP head is either complete or absent."""

    count = int(model_config.get("mtp_num_hidden_layers", 0) or 0)
    names = {name for name in inventory if name.startswith("mtp.")}
    if count <= 0:
        if names:
            raise ValueError(
                "checkpoint contains mtp.* tensors but "
                "mtp_num_hidden_layers is not positive"
            )
        return False, 0
    if not names:
        return False, count

    required = set(_MTP_PROTECTED_TENSORS)
    for index in range(count):
        prefix = f"mtp.layers.{index}."
        required.update(
            {
                prefix + "input_layernorm.weight",
                prefix + "post_attention_layernorm.weight",
                prefix + "self_attn.q_proj.weight",
                prefix + "self_attn.k_proj.weight",
                prefix + "self_attn.v_proj.weight",
                prefix + "self_attn.o_proj.weight",
                prefix + "self_attn.q_norm.weight",
                prefix + "self_attn.k_norm.weight",
                prefix + "mlp.gate_proj.weight",
                prefix + "mlp.up_proj.weight",
                prefix + "mlp.down_proj.weight",
            }
        )
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            "incomplete Qwen MTP head; missing tensors: "
            + ", ".join(missing[:8])
        )
    return True, count


ImportanceRows = Callable[[int, int], np.ndarray | None]
ImportanceSelection = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class HfImatrixBinding:
    entry_name: str
    rows: ImportanceRows
    selected: ImportanceSelection


def _hf_imatrix_names(item: TensorPlan) -> tuple[str, ...]:
    """Return canonical and compatibility names for one HF tensor plan."""

    candidates: list[str] = []

    def append(name: str | None) -> None:
        if name and name not in candidates:
            candidates.append(name)

    # Prefer the tensor's own canonical name over a recipe-group anchor.  Q/K
    # and gate/up may share one recipe precision, but an imatrix can still
    # carry distinct entries for those projections.
    append(_hf_to_gguf_name(item.name))
    if item.source_name is not None and item.transform is None:
        append(_hf_to_gguf_name(item.source_name))
    append(item.gguf_name)
    # Accept imatrix artifacts produced in an HF namespace as well as the
    # canonical llama.cpp GGUF namespace.
    append(item.name)
    if item.source_name is not None and item.transform is None:
        append(item.source_name)
    return tuple(candidates)


def _hf_imatrix_shapes(
    item: TensorPlan,
) -> tuple[tuple[int, ...], tuple[int, int]]:
    original_shape = item.expert_shape or item.shape
    if len(original_shape) < 2:
        raise ValueError(
            f"imatrix binding requires a matrix tensor: {item.name} {original_shape}"
        )
    storage_shape = (
        int(np.prod(original_shape[:-1])),
        int(original_shape[-1]),
    )
    return original_shape, storage_shape


def _hf_plan_supports_imatrix(item: TensorPlan) -> bool:
    if (
        item.target_dtype.startswith("NVQ")
        or item.target_dtype in _IMATRIX_NINT_DTYPES
    ):
        return True
    return bool(
        item.target_dtype == "NINTM"
        and item.expert_precisions is not None
        and any(
            precision.family in _IMATRIX_NINT_DTYPES
            or precision.family.startswith("NVQ")
            or precision.family.startswith("NEPQ")
            for precision in item.expert_precisions
        )
    )


def _bind_hf_imatrix(
    imatrix: ImportanceMatrix,
    plan: list[TensorPlan],
) -> dict[str, HfImatrixBinding]:
    """Bind a llama.cpp or HF-namespaced imatrix to an HF conversion plan."""

    bindings: dict[str, HfImatrixBinding] = {}
    missing: list[str] = []
    for item in plan:
        if not _hf_plan_supports_imatrix(item):
            continue
        names = _hf_imatrix_names(item)
        original_shape, storage_shape = _hf_imatrix_shapes(item)
        match = imatrix.for_rows(
            names,
            original_shape,
            storage_shape,
            slice(0, min(1, storage_shape[0])),
        )
        if match is None:
            if not any(name in _IMATRIX_OPTIONAL_TENSORS for name in names):
                missing.append(names[0] if names else item.name)
            continue
        entry_name, _ = match

        def rows(
            start: int,
            end: int,
            *,
            _names=names,
            _original_shape=original_shape,
            _storage_shape=storage_shape,
            _item=item,
        ) -> np.ndarray:
            resolved = imatrix.for_rows(
                _names,
                _original_shape,
                _storage_shape,
                slice(start, end),
            )
            if resolved is None:
                raise RuntimeError(
                    f"imatrix binding disappeared for {_item.name}"
                )
            return resolved[1]

        def selected(
            row_ids: np.ndarray,
            *,
            _names=names,
            _original_shape=original_shape,
            _storage_shape=storage_shape,
            _item=item,
        ) -> np.ndarray:
            resolved = imatrix.for_rows(
                _names,
                _original_shape,
                _storage_shape,
                np.asarray(row_ids, dtype=np.int64),
            )
            if resolved is None:
                raise RuntimeError(
                    f"imatrix binding disappeared for {_item.name}"
                )
            return resolved[1]

        bindings[item.name] = HfImatrixBinding(entry_name, rows, selected)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f" ... ({len(missing)} total)"
        raise ValueError(
            f"imatrix is missing required HF NINT/NVQ tensors: {preview}{suffix}"
        )
    return bindings


def _hf_expert_importance(
    item: TensorPlan,
    binding: HfImatrixBinding | None,
) -> np.ndarray | None:
    if binding is None:
        return None
    if item.expert_shape is None:
        raise ValueError(f"NINTM plan lacks expert shape: {item.name}")
    n_experts, rows_per_expert, _ = item.expert_shape
    return binding.selected(
        np.arange(n_experts, dtype=np.int64) * rows_per_expert
    )


def _dtype_for_recipe_type(gguf_type: str, dense_dtype: str) -> str:
    del dense_dtype  # Recipe storage types are authoritative, as in GGUF->MFQ.
    try:
        return _RECIPE_TARGETS[gguf_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported GGUF recipe tensor type: {gguf_type}"
        ) from exc


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
    plan: list[TensorPlan],
    overrides: dict[str, str],
) -> list[TensorPlan]:
    """Apply GGUF-name overrides to an HF plan without changing its names."""

    if not overrides:
        return plan
    result: list[TensorPlan] = []
    used: set[str] = set()
    for item in plan:
        candidates = tuple(
            value
            for value in (item.gguf_name, _hf_to_gguf_name(item.name), item.name)
            if value
        )
        matched = [name for name in candidates if name in overrides]
        if not matched:
            result.append(item)
            continue
        dtypes = {overrides[name] for name in matched}
        if len(dtypes) != 1:
            raise ValueError(
                f"conflicting tensor precision overrides for {item.name}: "
                f"{[(name, overrides[name]) for name in matched]}"
            )
        target = dtypes.pop()
        used.update(matched)
        if _is_compact_dtype(target) and len(item.shape) not in (2, 3):
            raise ValueError(
                f"tensor precision override maps a non-matrix tensor to "
                f"{target}: {item.name}"
            )
        if len(item.shape) == 3 and _is_compact_dtype(target):
            precision = (
                nint_expert_precision(
                    _spec_for_target(target, NintSpec())
                )
                if target.startswith("NINT")
                else ExpertPrecision(family=target)
            )
            result.append(
                replace(
                    item,
                    target_dtype="NINTM",
                    target_spec=None,
                    expert_shape=tuple(int(value) for value in item.shape),
                    expert_precisions=(precision,) * int(item.shape[0]),
                )
            )
        else:
            result.append(
                replace(
                    item,
                    target_dtype=target,
                    target_spec=(
                        _spec_for_target(target, NintSpec())
                        if target.startswith("NINT") and target != "NINT8-0"
                        else None
                    ),
                    expert_shape=None,
                    expert_precisions=None,
                )
            )
    missing = sorted(set(overrides) - used)
    if missing:
        raise ValueError(
            "tensor precision overrides are absent from the HF conversion "
            f"plan: {missing[:8]}"
        )
    return result


def _apply_recipe_family_mappings(
    plan: list[TensorPlan],
    *,
    npq0_l: bool,
    nvq3_jsc: bool,
    nvq3_jsc_512: bool,
    nvq3_to_nint3: bool,
    iq2_s_to_nint2: bool,
    q8_to_nint8_zero: bool,
) -> list[TensorPlan]:
    if sum((nvq3_jsc, nvq3_jsc_512, nvq3_to_nint3)) > 1:
        raise ValueError(
            "--nvq3-jsc, --nvq3-jsc-512, and --nvq3-to-nint3 "
            "are mutually exclusive"
        )
    result: list[TensorPlan] = []
    for item in plan:
        target = item.target_dtype
        if npq0_l and target == "NVQ1-L":
            target = "NPQ0-L"
        if item.gguf_type == "IQ3_XXS" and target in {"NVQ3", "NVQ3J"}:
            if nvq3_jsc or nvq3_jsc_512:
                target = "NVQ3J-512" if nvq3_jsc_512 else "NVQ3J"
            elif nvq3_to_nint3:
                target = "NINT3"
        if (
            iq2_s_to_nint2
            and item.gguf_type == "IQ2_S"
            and target.startswith("NVQ2J")
        ):
            target = "NINT2"
        if (
            q8_to_nint8_zero
            and item.gguf_type == "Q8_0"
            and target == "NINT8"
        ):
            target = "NINT8-0"
        result.append(
            item
            if target == item.target_dtype
            else replace(
                item,
                target_dtype=target,
                target_spec=(
                    _spec_for_target(target, NintSpec())
                    if target.startswith("NINT") and target != "NINT8-0"
                    else None
                ),
            )
        )
    return result


def _normalize_hf_expert_storage(
    plan: list[TensorPlan],
) -> list[TensorPlan]:
    """Store homogeneous rank-3 compact recipes in runtime NINTM layout."""

    result: list[TensorPlan] = []
    for item in plan:
        if item.target_dtype == "NINTM":
            if (
                len(item.shape) != 3
                or item.expert_shape != tuple(item.shape)
                or item.expert_precisions is None
                or len(item.expert_precisions) != item.shape[0]
            ):
                raise ValueError(
                    f"invalid mixed-expert conversion plan for {item.name}: "
                    f"shape={item.shape}, expert_shape={item.expert_shape}, "
                    f"precisions={0 if item.expert_precisions is None else len(item.expert_precisions)}"
                )
            result.append(item)
            continue
        if len(item.shape) != 3 or not _is_compact_dtype(item.target_dtype):
            result.append(item)
            continue
        precision = (
            nint_expert_precision(
                _spec_for_target(item.target_dtype, NintSpec())
            )
            if item.target_dtype.startswith("NINT")
            else ExpertPrecision(family=item.target_dtype)
        )
        expert_shape = tuple(int(value) for value in item.shape)
        result.append(
            replace(
                item,
                target_dtype="NINTM",
                target_spec=None,
                expert_shape=expert_shape,
                expert_precisions=(precision,) * expert_shape[0],
            )
        )
    return result


_HF_FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})
_HF_INTEGER_DTYPES = frozenset({"I32", "I64"})
_MOSTLY_BF16_F32_WEIGHT_MARKERS = (
    # Match llama.cpp's tensors which are deliberately excluded from ordinary
    # weight quantization / 16-bit storage.  These are small or numerically
    # sensitive, so the BF16 container keeps them in F32.
    "ffn_gate_inp",
    ".mlp.gate.weight",
    "shared_expert_gate.weight",
    "router.proj.weight",
    "ssm_conv1d",
    "linear_attn.conv1d",
    "shortconv.conv",
    "time_mix",
    "indexer",
    "pos_embd",
    "position_embeddings",
    "token_type_embeddings",
    "altup_correct_coef",
    "altup_predict_coef",
)


def _mostly_bf16_target(
    name: str,
    shape: tuple[int, ...],
    source_dtype: str,
) -> str:
    """Apply llama.cpp's MOSTLY_BF16 storage policy to one HF tensor.

    Ordinary matrix weights use BF16.  One-dimensional values, norms,
    non-weight tensors, routers and other special small/sensitive weights use
    F32.  Integer tensors remain integers.  This intentionally differs from
    an "all floating tensors are BF16" conversion.
    """

    if source_dtype in _HF_INTEGER_DTYPES:
        return source_dtype
    if source_dtype not in _HF_FLOAT_DTYPES:
        raise ValueError(
            f"unsupported HF dtype for dense BF16 conversion: "
            f"{name} {source_dtype}"
        )

    gguf_name = _hf_to_gguf_name(name)
    logical_name = gguf_name or name
    if (
        len(shape) <= 1
        or "norm" in logical_name.lower()
        or not logical_name.endswith((".weight", ".lora_a", ".lora_b"))
        or any(marker in logical_name for marker in _MOSTLY_BF16_F32_WEIGHT_MARKERS)
    ):
        return "F32"
    return "BF16"


def _layer_prefix_suffix(name: str) -> tuple[str, str] | None:
    for prefix in ("model.language_model.layers.", "model.layers."):
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        layer, sep, suffix = rest.partition(".")
        if not sep:
            return None
        return f"{prefix}{layer}.", suffix
    return None


def _recipe_group_anchor(name: str) -> str | None:
    """Return the HF tensor whose recipe should be shared inside a fused group."""

    parsed = _layer_prefix_suffix(name)
    if parsed is None:
        return None
    lp, suffix = parsed
    if suffix in {"self_attn.q_proj.weight", "self_attn.k_proj.weight"}:
        return lp + "self_attn.q_proj.weight"
    if suffix in {"mlp.gate_proj.weight", "mlp.up_proj.weight"}:
        return lp + "mlp.gate_proj.weight"
    if suffix in {
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
    }:
        return lp + "mlp.shared_expert.gate_proj.weight"
    return None


def _target_for_recipe_name(
    name: str,
    gguf_type: str | None,
    recipe_types: dict[str, str] | None,
    dense_dtype: str,
    *,
    mtp_layer_index: int = 40,
) -> tuple[str, str | None, str | None]:
    anchor = _recipe_group_anchor(name) if recipe_types is not None else None
    anchor_gguf_name = (
        _hf_to_gguf_name(anchor, mtp_layer_index=mtp_layer_index)
        if anchor is not None
        else None
    )
    if anchor_gguf_name is not None and anchor_gguf_name in recipe_types:
        anchor_type = recipe_types[anchor_gguf_name]
        return _dtype_for_recipe_type(anchor_type, dense_dtype), anchor_gguf_name, anchor_type
    if gguf_type is not None:
        return _dtype_for_recipe_type(gguf_type, dense_dtype), None, gguf_type
    return dense_dtype, None, None


def _linear_attn_qkv_split_config(cfg: dict[str, object]) -> tuple[int, int] | None:
    text = cfg.get("text_config", cfg)
    if not isinstance(text, dict):
        return None
    nk = int(text.get("linear_num_key_heads", text.get("num_key_value_heads", 0)))
    nv = int(text.get("linear_num_value_heads", text.get("num_attention_heads", 0)))
    dk = int(text.get("linear_key_head_dim", text.get("head_dim", 0)))
    dv = int(text.get("linear_value_head_dim", text.get("head_dim", 0)))
    if nk <= 0 or nv <= 0 or dk <= 0 or dv <= 0:
        return None
    ksz = nk * dk
    vsz = nv * dv
    return 2 * ksz, 2 * ksz + vsz


def _linear_attn_qkv_split(root: Path) -> tuple[int, int] | None:
    cfg_path = root / "config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return _linear_attn_qkv_split_config(cfg)


def _spec_for_target(target_dtype: str, default_spec: NintSpec) -> NintSpec:
    if target_dtype == "NINT2":
        return _RECIPE_SPECS["Q2_K"]
    if target_dtype == "NINT3":
        return _RECIPE_SPECS["Q3_K"]
    if target_dtype == "NINT4":
        return _RECIPE_SPECS["Q4_K"]
    if target_dtype == "NINT5":
        return _RECIPE_SPECS["Q5_K"]
    if target_dtype == "NINT6":
        return _RECIPE_SPECS["Q6_K"]
    if target_dtype == "NINT8":
        return _RECIPE_SPECS["Q8_0"]
    return default_spec


def _spec_for_plan(item: TensorPlan, default_spec: NintSpec) -> NintSpec:
    return item.target_spec or _spec_for_target(item.target_dtype, default_spec)


def _runtime_precision_signature(item: TensorPlan) -> tuple[object, ...]:
    if item.target_dtype == "NINTM":
        return ("NINTM", item.expert_precisions)
    if item.target_dtype.startswith("NINT") and item.target_dtype != "NINTM":
        spec = _spec_for_plan(item, NintSpec())
        return ("NINT", spec.bits, spec.groupsize, spec.sub_bits)
    return (item.target_dtype,)


def _validate_runtime_fused_pairs(plan: list[TensorPlan]) -> None:
    """Reject plans that would silently split mandatory decode projection pairs."""

    by_name = {item.name: item for item in plan}
    pairs = (
        ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "Q/K"),
        ("mlp.gate_proj.weight", "mlp.up_proj.weight", "gate/up"),
        (
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "shared-expert gate/up",
        ),
        (
            "mlp.shared_experts.gate_proj.weight",
            "mlp.shared_experts.up_proj.weight",
            "GLM shared-expert gate/up",
        ),
    )
    for left_suffix, right_suffix, label in pairs:
        for name, left in by_name.items():
            if not name.endswith(left_suffix):
                continue
            right_name = name[: -len(left_suffix)] + right_suffix
            right = by_name.get(right_name)
            if right is None:
                continue
            left_sig = _runtime_precision_signature(left)
            right_sig = _runtime_precision_signature(right)
            if left_sig != right_sig:
                raise ValueError(
                    f"runtime-fused {label} tensors must share one precision layout: "
                    f"{name}={left_sig}, {right_name}={right_sig}"
                )


def _expert_plan_shape(
    source_shape: tuple[int, ...],
    selection: ExpertTensorSelection,
) -> tuple[int, int, int]:
    logical = (
        int(selection.n_experts),
        int(selection.rows_per_expert),
        int(selection.columns),
    )
    flattened = (logical[0] * logical[1], logical[2])
    if source_shape not in {logical, flattened}:
        raise ValueError(
            f"calibration scheme shape mismatch for {selection.name}: "
            f"checkpoint={source_shape}, scheme={logical} (or flattened {flattened})"
        )
    return logical


def _glm_layer_index(name: str) -> int | None:
    match = re.match(r"model\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match is not None else None


_MINICPMO45_DENSE_MATRICES = frozenset(
    {
        "apm.embed_positions.weight",
        "resampler.attn.in_proj_weight",
        "resampler.attn.out_proj.weight",
        "resampler.proj",
        "resampler.query",
        "tts.head_code.0.parametrizations.weight.original0",
        "tts.head_code.0.parametrizations.weight.original1",
        "vpm.embeddings.position_embedding.weight",
    }
)


def _is_minicpmo45_config(config: dict[str, object]) -> bool:
    return (
        str(config.get("model_type", "")).lower() == "minicpmo"
        and str(config.get("version", "")) == "4.5"
    )


def _minicpmo45_quantizable_matrix(name: str, shape: tuple[int, ...]) -> bool:
    """Return whether the official graph invokes this matrix as a module.

    The excluded tensors are consumed through raw parameter access, positional
    lookup, packed ``nn.MultiheadAttention`` projection, or PyTorch weight
    parametrization. Replacing them with an ordinary linear/embedding module
    would change the official graph.
    """

    return (
        len(shape) == 2
        and name.endswith(".weight")
        and name not in _MINICPMO45_DENSE_MATRICES
        and ".parametrizations.weight." not in name
    )


def _glm_expert_precisions(
    name: str,
    shape: tuple[int, int, int],
    calibration_scheme: CalibrationScheme | None,
) -> tuple[ExpertPrecision, ...]:
    default = NintSpec(4, 24, 6)
    if calibration_scheme is None:
        return (nint_expert_precision(default),) * shape[0]
    uniform = calibration_scheme.selections.get(name)
    expert = calibration_scheme.expert_selections.get(name)
    if uniform is not None and expert is not None:
        raise ValueError(f"tensor {name} has both uniform and expert-wise selections")
    if expert is not None:
        _expert_plan_shape(shape, expert)
        return expert.precisions
    if uniform is not None:
        if (uniform.rows, uniform.columns) != (shape[0] * shape[1], shape[2]):
            raise ValueError(
                f"calibration scheme shape mismatch for {name}: "
                f"target={(shape[0] * shape[1], shape[2])}, "
                f"scheme={(uniform.rows, uniform.columns)}"
            )
        return (nint_expert_precision(uniform.spec),) * shape[0]
    return (nint_expert_precision(default),) * shape[0]


def _glm_derived_plans(
    config: dict,
    weight_map: dict[str, str],
    source_shapes: dict[str, tuple[int, ...]],
    source_dtypes: dict[str, str],
    calibration_scheme: CalibrationScheme | None,
) -> list[TensorPlan]:
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    hidden = int(config["hidden_size"])
    kv_rank = int(config["kv_lora_rank"])
    nope = int(config["qk_nope_head_dim"])
    value = int(config["v_head_dim"])
    experts = int(config["n_routed_experts"])
    expert_hidden = int(config["moe_intermediate_size"])
    first_sparse = int(config.get("first_k_dense_replace", 0))
    moe_freq = max(1, int(config.get("moe_layer_freq", 1)))
    mlp_types = list(config.get("mlp_layer_types") or [])
    if mlp_types and len(mlp_types) != layers:
        raise ValueError(
            "GLM mlp_layer_types length does not match num_hidden_layers"
        )
    invalid_mlp_types = sorted(set(mlp_types) - {"dense", "sparse"})
    if invalid_mlp_types:
        raise ValueError(f"unsupported GLM MLP layer types: {invalid_mlp_types}")
    plans: list[TensorPlan] = []

    def require_shape(name: str, expected: tuple[int, ...]) -> None:
        actual = source_shapes.get(name)
        if actual != expected:
            raise ValueError(
                f"GLM source tensor shape mismatch for {name}: "
                f"checkpoint={actual}, expected={expected}"
            )

    for layer in range(layers):
        ap = f"model.layers.{layer}.self_attn."
        kv_source = ap + "kv_b_proj.weight"
        require_shape(kv_source, (heads * (nope + value), kv_rank))
        for target_suffix, transform, shape in (
            ("embed_q", "glm_kv_b_embed_q", (heads, kv_rank, nope)),
            ("unembed_out", "glm_kv_b_unembed_out", (heads, value, kv_rank)),
        ):
            target = ap + target_suffix
            plans.append(
                TensorPlan(
                    name=target,
                    shard=weight_map[kv_source],
                    shape=shape,
                    source_dtype=source_dtypes[kv_source],
                    target_dtype="NINTM",
                    source_name=kv_source,
                    expert_shape=shape,
                    expert_precisions=_glm_expert_precisions(
                        target, shape, calibration_scheme
                    ),
                    transform=transform,
                )
            )

        sparse = (
            mlp_types[layer] == "sparse"
            if len(mlp_types) == layers
            else layer >= first_sparse and layer % moe_freq == 0
        )
        if not sparse:
            continue
        mp = f"model.layers.{layer}.mlp."
        gate_up_target = mp + "experts.gate_up_proj"
        down_target = mp + "experts.down_proj"
        gate_up_shape = (experts, 2 * expert_hidden, hidden)
        down_shape = (experts, hidden, expert_hidden)
        gate_up_names: list[tuple[str, ...]] = []
        gate_up_shards: list[tuple[str, ...]] = []
        down_names: list[tuple[str, ...]] = []
        down_shards: list[tuple[str, ...]] = []
        for expert in range(experts):
            ep = mp + f"experts.{expert}."
            gate = ep + "gate_proj.weight"
            up = ep + "up_proj.weight"
            down = ep + "down_proj.weight"
            require_shape(gate, (expert_hidden, hidden))
            require_shape(up, (expert_hidden, hidden))
            require_shape(down, (hidden, expert_hidden))
            gate_up_names.append((gate, up))
            gate_up_shards.append((weight_map[gate], weight_map[up]))
            down_names.append((down,))
            down_shards.append((weight_map[down],))
        plans.append(
            TensorPlan(
                name=gate_up_target,
                shard=gate_up_shards[0][0],
                shape=gate_up_shape,
                source_dtype=source_dtypes[gate_up_names[0][0]],
                target_dtype="NINTM",
                expert_shape=gate_up_shape,
                expert_precisions=_glm_expert_precisions(
                    gate_up_target, gate_up_shape, calibration_scheme
                ),
                transform="glm_expert_gate_up",
                expert_source_names=tuple(gate_up_names),
                expert_source_shards=tuple(gate_up_shards),
            )
        )
        plans.append(
            TensorPlan(
                name=down_target,
                shard=down_shards[0][0],
                shape=down_shape,
                source_dtype=source_dtypes[down_names[0][0]],
                target_dtype="NINTM",
                expert_shape=down_shape,
                expert_precisions=_glm_expert_precisions(
                    down_target, down_shape, calibration_scheme
                ),
                transform="glm_expert_down",
                expert_source_names=tuple(down_names),
                expert_source_shards=tuple(down_shards),
            )
        )
    return plans


def _plan(
    root: Path,
    text_only: bool,
    recipe_types: dict[str, str] | None,
    dense_dtype: str,
    calibration_scheme: CalibrationScheme | None = None,
    mostly_bf16: bool = False,
    *,
    source_inventory: dict[str, SourceTensorMetadata] | None = None,
    source_config: dict[str, object] | None = None,
    default_nint_dtype: str = "NINT4",
) -> list[TensorPlan]:
    inventory = source_inventory or _hf_source_inventory(root)
    weight_map = {name: item.shard for name, item in inventory.items()}
    if source_config is None:
        config_path = root / "config.json"
        raw_config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
    else:
        raw_config = source_config
    model_config = raw_config.get("text_config", raw_config)
    if not isinstance(model_config, dict):
        raise ValueError("HF text_config must be an object")
    mtp_included, _mtp_layers = _mtp_inventory_status(
        inventory,
        model_config,
    )
    mtp_layer_index = int(model_config.get("num_hidden_layers", 0) or 0)
    is_glm_dsa = model_config.get("model_type") == "glm_moe_dsa"
    is_minicpmo45 = _is_minicpmo45_config(raw_config)
    if is_glm_dsa and recipe_types is not None:
        raise ValueError(
            "GLM DSA GGUF recipe mapping is not implemented; use a calibration scheme"
        )
    glm_layers = int(model_config.get("num_hidden_layers", 0)) if is_glm_dsa else 0
    linear_qkv_split = (
        _linear_attn_qkv_split_config(raw_config)
        if recipe_types is not None or calibration_scheme is not None
        else None
    )
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if text_only and not (
            name.startswith("model.language_model.")
            or (mtp_included and name.startswith("mtp."))
            or (is_glm_dsa and name.startswith("model."))
            or name == "lm_head.weight"
            or (is_minicpmo45 and name.startswith("llm."))
        ):
            continue
        if is_glm_dsa:
            layer_index = _glm_layer_index(name)
            if layer_index is not None and layer_index >= glm_layers:
                continue
        if recipe_types is not None:
            gguf_name = _hf_to_gguf_name(
                name,
                mtp_layer_index=mtp_layer_index,
            )
            if is_minicpmo45 and not name.startswith("llm."):
                # The official MiniCPM-o 4.5 Q4_K_M GGUF contains only the
                # language model.  Keep every other official component at its
                # source precision so the resulting MFQ still represents the
                # complete vision/audio/TTS graph.
                pass
            elif name.startswith("mtp."):
                # MTP is an all-or-nothing model capability.  Preserve the
                # complete head even when an older GGUF recipe omitted it;
                # protected fusion/norm tensors remain at source precision.
                pass
            elif gguf_name is None or gguf_name not in recipe_types:
                if is_minicpmo45:
                    raise ValueError(
                        "MiniCPM-o 4.5 language tensor is absent from the "
                        f"GGUF recipe: {name} (GGUF name: {gguf_name!r})"
                    )
                continue
        by_shard.setdefault(shard, []).append(name)

    out: list[TensorPlan] = []
    source_shapes: dict[str, tuple[int, ...]] = {}
    source_dtypes: dict[str, str] = {}
    for shard in sorted(by_shard):
        for name in sorted(by_shard[shard]):
                metadata = inventory[name]
                shape = metadata.shape
                source_dtype = metadata.dtype
                if is_glm_dsa and not mostly_bf16 and (
                    name.endswith(".self_attn.kv_b_proj.weight")
                    or re.match(
                        r"model\.layers\.\d+\.mlp\.experts\.\d+\."
                        r"(?:gate_proj|up_proj|down_proj)\.weight$",
                        name,
                    )
                ):
                    source_shapes[name] = shape
                    source_dtypes[name] = source_dtype
                    continue
                gguf_name = (
                    _hf_to_gguf_name(
                        name,
                        mtp_layer_index=mtp_layer_index,
                    )
                    if recipe_types is not None
                    else None
                )
                gguf_type = (
                    recipe_types.get(gguf_name)
                    if gguf_name is not None and recipe_types is not None
                    else None
                )
                if name in _MTP_PROTECTED_TENSORS:
                    target = (
                        source_dtype
                        if source_dtype in {"BF16", "F16", "F32"}
                        else dense_dtype
                    )
                elif mostly_bf16:
                    target = _mostly_bf16_target(name, shape, source_dtype)
                elif recipe_types is not None:
                    if is_minicpmo45 and not name.startswith("llm."):
                        target = (
                            source_dtype
                            if source_dtype
                            in {"BF16", "F16", "F32", "I32", "I64"}
                            else dense_dtype
                        )
                    elif name.startswith("mtp.") and gguf_type is None:
                        target = (
                            default_nint_dtype
                            if len(shape) in (2, 3)
                            else dense_dtype
                        )
                    else:
                        target, anchor_gguf_name, anchor_type = (
                            _target_for_recipe_name(
                                name,
                                gguf_type,
                                recipe_types,
                                dense_dtype,
                                mtp_layer_index=mtp_layer_index,
                            )
                        )
                        if anchor_gguf_name is not None:
                            gguf_name = anchor_gguf_name
                            gguf_type = anchor_type
                elif is_minicpmo45:
                    target = (
                        default_nint_dtype
                        if _minicpmo45_quantizable_matrix(name, shape)
                        else (
                            source_dtype
                            if source_dtype in {"BF16", "F16", "F32", "I32", "I64"}
                            else dense_dtype
                        )
                    )
                else:
                    target = default_nint_dtype if len(shape) == 2 else dense_dtype
                    if is_glm_dsa and name.endswith(".mlp.gate.weight"):
                        target = "F16"
                if _is_compact_dtype(target) and len(shape) not in (2, 3):
                    raise ValueError(f"recipe maps non-matrix tensor to {target}: {name} {shape}")
                parsed = _layer_prefix_suffix(name)
                if (
                    (recipe_types is not None or calibration_scheme is not None)
                    and parsed is not None
                    and parsed[1] == "linear_attn.in_proj_qkv.weight"
                    and linear_qkv_split is not None
                ):
                    qk_end, qkv_end = linear_qkv_split
                    if len(shape) != 2 or shape[0] < qkv_end:
                        raise ValueError(f"cannot split linear attention qkv tensor: {name} {shape}")
                    lp, _suffix = parsed
                    qk_name = lp + "linear_attn.in_proj_qk.weight"
                    v_name = lp + "linear_attn.in_proj_v.weight"
                    qk_selection = (
                        calibration_scheme.selections.get(qk_name)
                        if calibration_scheme is not None
                        else None
                    )
                    v_selection = (
                        calibration_scheme.selections.get(v_name)
                        if calibration_scheme is not None
                        else None
                    )
                    if qk_selection is not None and (qk_selection.rows, qk_selection.columns) != (
                        qk_end,
                        shape[1],
                    ):
                        raise ValueError(
                            f"calibration scheme shape mismatch for {qk_name}: "
                            f"checkpoint={(qk_end, shape[1])}, "
                            f"scheme={(qk_selection.rows, qk_selection.columns)}"
                        )
                    if v_selection is not None and (v_selection.rows, v_selection.columns) != (
                        qkv_end - qk_end,
                        shape[1],
                    ):
                        raise ValueError(
                            f"calibration scheme shape mismatch for {v_name}: "
                            f"checkpoint={(qkv_end - qk_end, shape[1])}, "
                            f"scheme={(v_selection.rows, v_selection.columns)}"
                        )
                    out.append(
                        TensorPlan(
                            qk_name,
                            shard,
                            (qk_end, shape[1]),
                            source_dtype,
                            f"NINT{qk_selection.spec.bits}" if qk_selection else target,
                            gguf_name,
                            gguf_type,
                            source_name=name,
                            row_start=0,
                            row_end=qk_end,
                            target_spec=qk_selection.spec if qk_selection else None,
                        )
                    )
                    out.append(
                        TensorPlan(
                            v_name,
                            shard,
                            (qkv_end - qk_end, shape[1]),
                            source_dtype,
                            f"NINT{v_selection.spec.bits}" if v_selection else target,
                            gguf_name,
                            gguf_type,
                            source_name=name,
                            row_start=qk_end,
                            row_end=qkv_end,
                            target_spec=v_selection.spec if v_selection else None,
                        )
                    )
                    continue
                selection = (
                    calibration_scheme.selections.get(name)
                    if calibration_scheme is not None
                    else None
                )
                expert_selection = (
                    calibration_scheme.expert_selections.get(name)
                    if calibration_scheme is not None
                    else None
                )
                if selection is not None and expert_selection is not None:
                    raise ValueError(f"tensor {name} has both uniform and expert-wise selections")
                expert_shape: tuple[int, int, int] | None = None
                expert_precisions: tuple[ExpertPrecision, ...] | None = None
                if selection is not None:
                    if tuple(shape) != (selection.rows, selection.columns):
                        raise ValueError(
                            f"calibration scheme shape mismatch for {name}: "
                            f"checkpoint={shape}, scheme={(selection.rows, selection.columns)}"
                        )
                    target = f"NINT{selection.spec.bits}"
                elif expert_selection is not None:
                    expert_shape = _expert_plan_shape(shape, expert_selection)
                    expert_precisions = expert_selection.precisions
                    target = "NINTM"
                out.append(
                    TensorPlan(
                        name,
                        shard,
                        shape,
                        source_dtype,
                        target,
                        gguf_name,
                        gguf_type,
                        target_spec=selection.spec if selection else None,
                        expert_shape=expert_shape,
                        expert_precisions=expert_precisions,
                    )
                )
    if is_glm_dsa and not mostly_bf16:
        out.extend(
            _glm_derived_plans(
                model_config,
                weight_map,
                source_shapes,
                source_dtypes,
                calibration_scheme,
            )
        )
    if calibration_scheme is not None:
        planned = {item.name for item in out}
        selected_names = set(calibration_scheme.selections) | set(
            calibration_scheme.expert_selections
        )
        missing = sorted(selected_names - planned)
        if missing:
            raise ValueError(
                f"calibration scheme tensors are absent from the conversion plan: {missing[:8]}"
            )
    _validate_runtime_fused_pairs(out)
    return out


def _dense_blob_from_tensor(t: torch.Tensor, blob_path: Path, dtype: str) -> int:
    if dtype == "F32":
        arr = t.to(torch.float32).contiguous().cpu().numpy()
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    elif dtype == "F16":
        arr = t.to(torch.float16).contiguous().cpu().numpy()
        arr = np.ascontiguousarray(arr, dtype=np.float16)
    elif dtype == "BF16":
        arr = (
            t.to(torch.bfloat16)
            .contiguous()
            .cpu()
            .view(torch.uint16)
            .numpy()
        )
        arr = np.ascontiguousarray(arr, dtype="<u2")
    elif dtype == "I32":
        arr = t.to(torch.int32).contiguous().cpu().numpy()
        arr = np.ascontiguousarray(arr, dtype=np.int32)
    elif dtype == "I64":
        arr = t.to(torch.int64).contiguous().cpu().numpy()
        arr = np.ascontiguousarray(arr, dtype=np.int64)
    else:
        raise ValueError(f"unsupported dense target dtype: {dtype}")
    produced_dtype = "BF16" if dtype == "BF16" else _DENSE_NAMES.get(arr.dtype)
    if produced_dtype is None:
        raise ValueError(f"dense conversion produced unsupported dtype: {arr.dtype}")
    with blob_path.open("wb") as f:
        f.write(struct.pack("<I", arr.ndim))
        f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
        f.write(arr.tobytes())
    return blob_path.stat().st_size


def _write_nint_axis0_blob(
    sl,
    shape: tuple[int, ...],
    spec: NintSpec,
    blob_path: Path,
    row_chunk: int,
    quant_backend: str,
    device: str,
    importance_rows=None,
) -> int:
    if len(shape) != 2:
        raise ValueError(f"NINT stream writer only supports 2D tensors, got {shape}")
    out, neuron_len = shape
    gs = int(spec.groupsize)
    ng = (neuron_len + gs - 1) // gs
    scale_nbytes = out * np.dtype(np.float16).itemsize
    sub_nbytes = (out * ng * spec.sub_bits + 7) // 8
    q_nbytes = (out * ng * gs * spec.bits + 7) // 8
    if (row_chunk * ng * spec.sub_bits) % 8 != 0:
        raise ValueError(
            f"row_chunk={row_chunk} does not align sub_bits={spec.sub_bits}, ng={ng} to byte boundary"
        )
    if (row_chunk * ng * gs * spec.bits) % 8 != 0:
        raise ValueError(
            f"row_chunk={row_chunk} does not align bits={spec.bits}, ng={ng}, gs={gs} to byte boundary"
        )

    with blob_path.open("wb+") as f:
        f.write(_NINT_HDR.pack(spec.bits, spec.sub_bits, spec.groupsize, 0, neuron_len))
        f.write(struct.pack("<I", len(shape)))
        f.write(struct.pack(f"<{len(shape)}q", *shape))
        f.write(struct.pack("<II", out, ng))
        scale_off = f.tell()
        min_off = scale_off + scale_nbytes
        sub_scale_off = min_off + scale_nbytes
        sub_min_off = sub_scale_off + sub_nbytes
        q_off = sub_min_off + sub_nbytes
        f.truncate(q_off + q_nbytes)

        for start in range(0, out, row_chunk):
            end = min(start + row_chunk, out)
            importance = (
                None if importance_rows is None else importance_rows(start, end)
            )
            if quant_backend in ACCELERATOR_BACKENDS and hasattr(sl, "read_rows"):
                chunk = sl.read_rows(start, end, device=device)
            else:
                chunk = sl[start:end]
            if quant_backend in ACCELERATOR_BACKENDS:
                nt = nint_quantize_axis0_torch(
                    chunk, spec, device=device, importance=importance
                )
            elif quant_backend == "cpu":
                nt = nint_quantize(
                    chunk.float().cpu().numpy(),
                    spec,
                    axis=0,
                    importance=importance,
                )
            else:
                raise ValueError(f"unsupported quant backend: {quant_backend}")
            rows = end - start
            if nt.neuron_len != neuron_len or nt.q.shape != (rows, ng, gs):
                raise ValueError(f"unexpected NINT chunk shape: {nt.q.shape}, neuron_len={nt.neuron_len}")
            f.seek(scale_off + start * 2)
            f.write(np.ascontiguousarray(nt.neuron_scale, dtype=np.float16).tobytes())
            f.seek(min_off + start * 2)
            f.write(np.ascontiguousarray(nt.neuron_min, dtype=np.float16).tobytes())
            f.seek(sub_scale_off + (start * ng * spec.sub_bits) // 8)
            f.write(pack_bits(nt.sub_scale, spec.sub_bits))
            f.seek(sub_min_off + (start * ng * spec.sub_bits) // 8)
            f.write(pack_bits(nt.sub_min, spec.sub_bits))
            f.seek(q_off + (start * ng * gs * spec.bits) // 8)
            f.write(pack_bits(nt.q, spec.bits))
            del chunk, nt, importance
    return blob_path.stat().st_size


class _ExpertPoolRowSource:
    """Present selected experts as contiguous flattened rows to the NINT writer."""

    def __init__(
        self,
        source,
        source_shape: tuple[int, ...],
        expert_shape: tuple[int, int, int],
        expert_ids: tuple[int, ...],
    ) -> None:
        n_experts, rows_per_expert, columns = expert_shape
        flattened = (n_experts * rows_per_expert, columns)
        if source_shape not in {expert_shape, flattened}:
            raise ValueError(
                f"expert source shape {source_shape} must be {expert_shape} or {flattened}"
            )
        self.source = source
        self.source_shape = source_shape
        self.rows_per_expert = rows_per_expert
        self.columns = columns
        self.expert_ids = expert_ids
        self.shape = (len(expert_ids), rows_per_expert, columns)
        if any(expert < 0 or expert >= n_experts for expert in expert_ids):
            raise ValueError("expert row source contains an invalid expert id")

    def reshape(self, *shape: int) -> _ExpertPoolRowSource:
        requested = tuple(int(value) for value in shape)
        if requested not in {
            self.shape,
            (-1, self.columns),
            (len(self.expert_ids) * self.rows_per_expert, self.columns),
        }:
            raise ValueError(f"unsupported expert row source reshape: {requested}")
        return self

    def _read_expert_rows(self, expert: int, start: int, end: int) -> torch.Tensor:
        if hasattr(self.source, "read_expert_rows"):
            return self.source.read_expert_rows(
                expert, start, end, device="cpu"
            )
        if len(self.source_shape) == 2:
            row0 = expert * self.rows_per_expert + start
            row1 = expert * self.rows_per_expert + end
            value = self.source[row0:row1]
        else:
            block = self.source[expert : expert + 1]
            value = block.reshape(self.rows_per_expert, self.columns)[start:end]
        return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)

    def read_rows(
        self,
        start: int,
        end: int,
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
        total_rows = len(self.expert_ids) * self.rows_per_expert
        if start < 0 or end < start or end > total_rows:
            raise IndexError(f"invalid expert row slice {start}:{end} of {total_rows}")
        pieces: list[torch.Tensor] = []
        cursor = start
        while cursor < end:
            local_expert = cursor // self.rows_per_expert
            local_row = cursor % self.rows_per_expert
            take = min(end - cursor, self.rows_per_expert - local_row)
            expert = self.expert_ids[local_expert]
            if hasattr(self.source, "read_expert_rows"):
                value = self.source.read_expert_rows(
                    expert,
                    local_row,
                    local_row + take,
                    device=device,
                )
            else:
                value = self._read_expert_rows(
                    expert, local_row, local_row + take
                ).to(device=device)
            pieces.append(value)
            cursor += take
        if not pieces:
            return torch.empty(
                (0, self.columns), device=device, dtype=torch.float32
            )
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    def __getitem__(self, key: slice) -> torch.Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("expert row source accepts contiguous slices only")
        total_rows = len(self.expert_ids) * self.rows_per_expert
        start = 0 if key.start is None else int(key.start)
        end = total_rows if key.stop is None else int(key.stop)
        if start < 0 or end < start or end > total_rows:
            raise IndexError(f"invalid expert row slice {start}:{end} of {total_rows}")
        pieces: list[torch.Tensor] = []
        cursor = start
        while cursor < end:
            local_expert = cursor // self.rows_per_expert
            local_row = cursor % self.rows_per_expert
            take = min(end - cursor, self.rows_per_expert - local_row)
            pieces.append(
                self._read_expert_rows(
                    self.expert_ids[local_expert], local_row, local_row + take
                )
            )
            cursor += take
        if not pieces:
            return torch.empty((0, self.columns), dtype=torch.float32)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


class _GlmExpertRowSource:
    """Read one GLM expert slice at a time without stacking the expert layer."""

    def __init__(
        self,
        root: Path,
        expert_shape: tuple[int, int, int],
        source_names: tuple[tuple[str, ...], ...],
        source_shards: tuple[tuple[str, ...], ...],
    ) -> None:
        self.root = root
        self.n_experts, self.rows_per_expert, self.columns = expert_shape
        self.source_names = source_names
        self.source_shards = source_shards
        if len(source_names) != self.n_experts or len(source_shards) != self.n_experts:
            raise ValueError("GLM expert source count does not match expert shape")
        self._context = None
        self._handle = None
        self._shard: str | None = None

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
        self._context = None
        self._handle = None
        self._shard = None

    def _open(self, shard: str):
        if self._shard == shard and self._handle is not None:
            return self._handle
        self.close()
        self._context = safe_open(
            str(self.root / shard), framework="pt", device="cpu"
        )
        self._handle = self._context.__enter__()
        self._shard = shard
        return self._handle

    def _read(self, expert: int, start: int, end: int) -> torch.Tensor:
        names = self.source_names[expert]
        shards = self.source_shards[expert]
        if len(names) == 0 or len(names) != len(shards):
            raise ValueError("invalid GLM expert source tuple")
        if self.rows_per_expert % len(names) != 0:
            raise ValueError("GLM fused expert rows do not split evenly")
        rows_per_source = self.rows_per_expert // len(names)
        pieces: list[torch.Tensor] = []
        cursor = start
        while cursor < end:
            source_index = cursor // rows_per_source
            local_row = cursor % rows_per_source
            take = min(end - cursor, rows_per_source - local_row)
            handle = self._open(shards[source_index])
            pieces.append(
                handle.get_slice(names[source_index])[
                    local_row : local_row + take
                ]
            )
            cursor += take
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    def __getitem__(self, key: slice) -> torch.Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("GLM expert source accepts contiguous slices only")
        total_rows = self.n_experts * self.rows_per_expert
        start = 0 if key.start is None else int(key.start)
        end = total_rows if key.stop is None else int(key.stop)
        if start < 0 or end < start or end > total_rows:
            raise IndexError(f"invalid GLM expert row slice {start}:{end}")
        pieces: list[torch.Tensor] = []
        cursor = start
        while cursor < end:
            expert = cursor // self.rows_per_expert
            local_row = cursor % self.rows_per_expert
            take = min(end - cursor, self.rows_per_expert - local_row)
            pieces.append(self._read(expert, local_row, local_row + take))
            cursor += take
        if not pieces:
            return torch.empty((0, self.columns), dtype=torch.float32)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


class _MfqGlmExpertRowSource:
    """Container-equivalent GLM expert merger for full-precision MFQ input."""

    def __init__(
        self,
        checkpoint,
        expert_shape: tuple[int, int, int],
        source_names: tuple[tuple[str, ...], ...],
    ) -> None:
        self.checkpoint = checkpoint
        self.n_experts, self.rows_per_expert, self.columns = expert_shape
        self.source_names = source_names
        if len(source_names) != self.n_experts:
            raise ValueError("GLM expert source count does not match expert shape")

    def close(self) -> None:
        return None

    def _read(
        self,
        expert: int,
        start: int,
        end: int,
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
        names = self.source_names[expert]
        if not names or self.rows_per_expert % len(names):
            raise ValueError("invalid GLM expert source tuple")
        rows_per_source = self.rows_per_expert // len(names)
        pieces: list[torch.Tensor] = []
        cursor = start
        while cursor < end:
            source_index = cursor // rows_per_source
            local_row = cursor % rows_per_source
            take = min(end - cursor, rows_per_source - local_row)
            pieces.append(
                self.checkpoint.tensor_source(names[source_index]).read_rows(
                    local_row,
                    local_row + take,
                    device=device,
                )
            )
            cursor += take
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    def read_rows(
        self,
        start: int,
        end: int,
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
        total_rows = self.n_experts * self.rows_per_expert
        if start < 0 or end < start or end > total_rows:
            raise IndexError(f"invalid GLM expert row slice {start}:{end}")
        pieces: list[torch.Tensor] = []
        cursor = start
        while cursor < end:
            expert = cursor // self.rows_per_expert
            local_row = cursor % self.rows_per_expert
            take = min(end - cursor, self.rows_per_expert - local_row)
            pieces.append(
                self._read(
                    expert,
                    local_row,
                    local_row + take,
                    device=device,
                )
            )
            cursor += take
        if not pieces:
            return torch.empty((0, self.columns), device=device)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    def __getitem__(self, key: slice) -> torch.Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("GLM expert source accepts contiguous slices only")
        total_rows = self.n_experts * self.rows_per_expert
        start = 0 if key.start is None else int(key.start)
        end = total_rows if key.stop is None else int(key.stop)
        return self.read_rows(start, end, device="cpu")


def _transform_glm_kv_b(source: torch.Tensor, item: TensorPlan) -> torch.Tensor:
    if item.expert_shape is None or item.transform is None:
        raise ValueError("GLM kv_b transform lacks target metadata")
    heads, rows, columns = item.expert_shape
    if source.ndim != 2 or source.shape[0] % heads != 0:
        raise ValueError(
            f"GLM kv_b source shape mismatch: {tuple(source.shape)}"
        )
    per_head = source.shape[0] // heads
    if item.transform == "glm_kv_b_embed_q":
        if source.shape[1] != rows or columns > per_head:
            raise ValueError("invalid GLM embed_q transform shape")
        reshaped = source.reshape(heads, per_head, rows)
        return reshaped[:, :columns, :].transpose(1, 2).contiguous()
    if item.transform == "glm_kv_b_unembed_out":
        if source.shape[1] != columns or rows > per_head:
            raise ValueError("invalid GLM unembed_out transform shape")
        reshaped = source.reshape(heads, per_head, columns)
        return reshaped[:, per_head - rows :, :].contiguous()
    raise ValueError(f"unknown GLM kv_b transform: {item.transform}")


def _write_nint_moe_axis0_blob(
    source,
    source_shape: tuple[int, ...],
    expert_shape: tuple[int, int, int],
    expert_specs: tuple[NintSpec, ...],
    blob_path: Path,
    row_chunk: int,
    quant_backend: str,
    device: str,
    importance: np.ndarray | torch.Tensor | None = None,
) -> int:
    """Stream a mixed-profile expert tensor into the ``NINTM`` container."""

    n_experts, rows_per_expert, columns = expert_shape
    if len(expert_specs) != n_experts:
        raise ValueError(
            f"received {len(expert_specs)} expert specs for {n_experts} experts"
        )
    cohorts: dict[NintSpec, list[int]] = {}
    for expert, profile in enumerate(expert_specs):
        cohorts.setdefault(profile, []).append(expert)

    pool_paths: list[Path] = []
    try:
        with blob_path.open("wb") as output:
            output.write(
                _NINT_MOE_HDR.pack(
                    _NINT_MOE_MAGIC_V2,
                    n_experts,
                    rows_per_expert,
                    columns,
                    len(cohorts),
                )
            )
            for pool_index, (profile, expert_list) in enumerate(cohorts.items()):
                expert_ids = tuple(int(value) for value in expert_list)
                if importance is None:
                    pool_importance = None
                else:
                    importance_array = (
                        importance.detach().to("cpu", torch.float32).numpy()
                        if isinstance(importance, torch.Tensor)
                        else np.asarray(importance, dtype=np.float32)
                    )
                    if importance_array.shape == (columns,):
                        pool_importance = importance_array
                    else:
                        pool_importance = np.ascontiguousarray(
                            importance_array[np.asarray(expert_ids, dtype=np.int64)]
                        )

                def importance_rows(
                    start: int,
                    end: int,
                    *,
                    _importance=pool_importance,
                ) -> np.ndarray | None:
                    if _importance is None or _importance.ndim == 1:
                        return _importance
                    expert_rows = (
                        np.arange(start, end, dtype=np.int64) // rows_per_expert
                    )
                    return np.ascontiguousarray(_importance[expert_rows])

                pool_source = _ExpertPoolRowSource(
                    source, source_shape, expert_shape, expert_ids
                )
                pool_path = blob_path.with_name(f"{blob_path.name}.pool{pool_index}.tmp")
                pool_paths.append(pool_path)
                pool_shape = (len(expert_ids) * rows_per_expert, columns)
                pool_nbytes = _write_nint_axis0_blob(
                    pool_source,
                    pool_shape,
                    profile,
                    pool_path,
                    row_chunk,
                    quant_backend,
                    device,
                    importance_rows=(
                        importance_rows if profile.bits in {2, 3, 4, 5, 6} else None
                    ),
                )
                dtype = f"NINT{profile.bits}".encode("ascii")
                output.write(
                    _NINT_MOE_POOL_V2_HDR.pack(
                        len(expert_ids),
                        len(dtype),
                        pool_nbytes,
                        0,
                    )
                )
                output.write(np.asarray(expert_ids, dtype=np.int32).tobytes())
                output.write(dtype)
                with pool_path.open("rb") as pool_file:
                    shutil.copyfileobj(pool_file, output, length=32 * 1024 * 1024)
        return blob_path.stat().st_size
    finally:
        for pool_path in pool_paths:
            pool_path.unlink(missing_ok=True)


def _check_row_stream_alignment(row_chunk: int, bits_per_row: Sequence[int]) -> None:
    for bits in bits_per_row:
        if row_chunk * int(bits) % 8:
            raise ValueError(
                f"row_chunk={row_chunk} does not byte-align a {bits}-bit row stream"
            )


def _quantize_flat_stream_chunk(
    weight: torch.Tensor | np.ndarray,
    precision: ExpertPrecision,
    artifact: object | None,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    quant_backend: str,
    device: str,
):
    family = precision.family
    weight_tensor = (
        weight
        if isinstance(weight, torch.Tensor)
        else torch.as_tensor(weight, dtype=torch.float32)
    )
    if quant_backend in ACCELERATOR_BACKENDS and family in {"NVQ1-L", "NVQ2", "NVQ3"}:
        from mfq.quantize.nvq_quant_torch import quantize_axis0

        spec = NVQ1_L_T8_S3 if family == "NVQ1-L" else _NVQ_SPECS[family]
        return quantize_axis0(
            weight_tensor,
            spec,
            device=device,
            importance=importance,
            search_steps=int(precision.option("search_steps", 19)),
            refine_steps=int(precision.option("refine_steps", 2)),
            group_chunk=int(precision.option("group_chunk", 1024)),
            codebook=None if artifact is None else np.asarray(artifact),
        )
    if quant_backend == "metal" and family == "NVQ1-S":
        from mfq.quantize.nvq1_s_quant_torch import quantize_axis0

        return quantize_axis0(
            weight_tensor,
            device=device,
            importance=importance,
            codebook=(
                NVQ1_S_SYNTHETIC_BANKS
                if artifact is None
                else np.asarray(artifact)
            ),
            anchor_multipliers=tuple(
                float(value)
                for value in str(
                    precision.option(
                        "anchor_multipliers", "0.75,1.0,1.25"
                    )
                ).split(",")
            ),
            refine_steps=int(precision.option("refine_steps", 2)),
        )
    if quant_backend in ACCELERATOR_BACKENDS and family in {
        "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
        "NVQ3J", "NVQ3J-512", "NVQ3J-L",
    }:
        from mfq.quantize.nvq_jsc import quantize_nvq_jsc_fixed

        if not isinstance(artifact, NvqJscTables):
            raise TypeError(f"{family} streaming requires fixed NvqJscTables")
        return quantize_nvq_jsc_fixed(
            weight_tensor,
            artifact,
            importance=importance,
            assignment_refine_steps=int(
                precision.option("assignment_refine_steps", 2)
            ),
            search_steps=int(precision.option("search_steps", 19)),
            group_chunk=int(precision.option("group_chunk", 1024)),
            device=device,
        )
    if quant_backend in ACCELERATOR_BACKENDS and family in TPQ_PQ_SPECS_BY_LABEL:
        if artifact is None:
            raise ValueError(f"streaming {family} requires a frozen codebook")
        return quantize_tpq_pq_fixed(
            weight_tensor,
            TPQ_PQ_SPECS_BY_LABEL[family],
            np.asarray(artifact, dtype=np.float32),
            device=device,
            distance_bytes=int(
                precision.option("distance_bytes", 1 << 30)
            ),
        )
    if quant_backend in ACCELERATOR_BACKENDS and family in {"NPQ0-L", "NPQ0-S"}:
        if family == "NPQ0-L":
            if not isinstance(artifact, Npq0LTables):
                raise TypeError("NPQ0-L streaming requires fixed Npq0LTables")
            return quantize_npq0_l_fixed(
                weight_tensor,
                artifact,
                importance=importance,
                config=Npq0LConfig(
                    fixed_refine_steps=int(
                        precision.option("fixed_refine_steps", 3)
                    ),
                    group_chunk=int(precision.option("group_chunk", 512)),
                ),
                device=device,
            )
        if not isinstance(artifact, Npq0STables):
            raise TypeError("NPQ0-S streaming requires fixed Npq0STables")
        return quantize_npq0_s_fixed(
            weight_tensor,
            artifact,
            importance=importance,
            config=Npq0SConfig(
                fixed_refine_steps=int(
                    precision.option("fixed_refine_steps", 3)
                ),
                group_chunk=int(precision.option("group_chunk", 256)),
            ),
            device=device,
        )
    array = weight_tensor.to(torch.float32).cpu().numpy()
    return quantize_flat_cohort(
        array,
        precision,
        artifact=artifact,
        importance=importance,
        device=device if quant_backend in ACCELERATOR_BACKENDS else "cpu",
    )


def _write_flat_family_axis0_blob(
    source,
    shape: tuple[int, int],
    precision: ExpertPrecision,
    blob_path: Path,
    row_chunk: int,
    quant_backend: str,
    device: str,
    artifact_root: str | Path | None,
    importance: np.ndarray | torch.Tensor | None = None,
    importance_rows_per_entry: int | None = None,
) -> int:
    family = precision.family
    if family.startswith("NINT") or family.startswith("NEPQ"):
        raise ValueError(f"{family} is not a flat compact-family stream")
    out, neuron_len = (int(value) for value in shape)
    artifact = resolve_precision_artifact(
        precision, artifact_root=artifact_root
    )
    if artifact is None and family in {
        "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
        "NVQ3J", "NVQ3J-512", "NVQ3J-L",
    }:
        artifact = initial_jsc_tables(
            NvqJscConfig(
                banks=int(precision.option("banks", 4)),
                iterations=int(precision.option("iterations", 4)),
                assignment_refine_steps=int(
                    precision.option("assignment_refine_steps", 2)
                ),
                search_steps=int(precision.option("search_steps", 19)),
                group_chunk=int(precision.option("group_chunk", 1024)),
                spec=_NVQ_SPECS[family],
            )
        )
    if (
        family
        in {
            "NVQ2J",
            "NVQ2J-L",
            "NVQ2J-XL",
            "NVQ3J",
            "NVQ3J-512",
            "NVQ3J-L",
            "NPQ0-L",
            "NPQ0-S",
            *TPQ_PQ_SPECS_BY_LABEL,
        }
        and artifact is None
    ):
        raise ValueError(
            f"streaming {family} requires a fixed table artifact; "
            "per-chunk table training would change the format graph"
        )
    if quant_backend == "cuda" and family == "NVQ1-S":
        warnings.warn(
            "NVQ1-S has no CUDA offline assignment kernel; this cohort uses CPU quantization",
            RuntimeWarning,
            stacklevel=2,
        )

    anchor_bytes_per_row = 2
    if family in TPQ_PQ_SPECS_BY_LABEL:
        spec = TPQ_PQ_SPECS_BY_LABEL[family]
        codebook = np.ascontiguousarray(artifact, dtype=np.float32)
        header = pack_tpq_pq_prefix(spec, shape, codebook)
        table_payload = b""
        stream_bits = (
            (neuron_len // spec.vector_size) * spec.index_bits,
        )
        packer = pack_tpq_indices
        fields = ("indices",)
        anchor_bytes_per_row = 0
    elif family == "NVQ1-L":
        spec = NVQ1_L_T8_S3
        ng = (neuron_len + spec.groupsize - 1) // spec.groupsize
        nvec = (neuron_len + spec.vector_size - 1) // spec.vector_size
        profile = (
            _PROFILE_IQ1S_GRID
            if artifact is None
            else _PROFILE_CUSTOM_TERNARY
        )
        header = _NVQ1_L_HEADER.pack(
            _NVQ1_L_MAGIC,
            profile,
            spec.sub_bits,
            spec.groupsize,
            0,
            neuron_len,
            len(shape),
        )
        table_payload = (
            b"" if artifact is None else pack_ternary_codebook(np.asarray(artifact))
        )
        stream_bits = (ng * spec.sub_bits, nvec * spec.index_bits, ng)
        packer = _pack_nvq1_l_bits
        fields = ("sub_scale", "indices", "delta_sign")
    elif family == "NVQ1-S":
        spec = NVQ1_S
        ng = (neuron_len + spec.groupsize - 1) // spec.groupsize
        nvec = neuron_len // spec.vector_size
        header = _NVQ1_S_HEADER.pack(
            _NVQ1_S_MAGIC,
            _NVQ1_S_VERSION,
            spec.sub_bits,
            spec.groupsize,
            0,
            neuron_len,
            len(shape),
        )
        codebook = NVQ1_S_SYNTHETIC_BANKS if artifact is None else np.asarray(artifact)
        table_payload = pack_nvq1_s_banked_codebook(codebook)
        stream_bits = (ng * spec.sub_bits, nvec * spec.index_bits, ng)
        packer = _pack_nvq1_l_bits
        fields = ("sub_scale", "indices", "delta_sign")
    elif family in {"NPQ0-L", "NPQ0-S"}:
        if family == "NPQ0-L":
            spec = NPQ0_L
            if not isinstance(artifact, Npq0LTables):
                raise TypeError("NPQ0-L stream artifact must contain Npq0LTables")
            header = _NPQ0_L_HEADER.pack(
                _NPQ0_L_MAGIC,
                _NPQ0_L_VERSION,
                spec.state_bits,
                spec.groupsize,
                0,
                neuron_len,
                len(shape),
            )
            table_payload = pack_npq0_l_tables(
                artifact.scale_lut,
                artifact.first_codebooks,
                artifact.second_codebooks,
            )
            packer = _pack_npq0_l_bits
        else:
            spec = NPQ0_S
            if not isinstance(artifact, Npq0STables):
                raise TypeError("NPQ0-S stream artifact must contain Npq0STables")
            header = _NPQ0_S_HEADER.pack(
                _NPQ0_S_MAGIC,
                _NPQ0_S_VERSION,
                spec.state_bits,
                spec.groupsize,
                0,
                neuron_len,
                len(shape),
            )
            table_payload = pack_npq0_s_tables(
                artifact.scale_lut,
                artifact.first_codebooks,
                artifact.second_codebooks,
            )
            packer = _pack_npq0_s_bits
        ng = (neuron_len + spec.groupsize - 1) // spec.groupsize
        nvec = neuron_len // spec.vector_size
        stream_bits = (ng * spec.state_bits, nvec * spec.index_bits)
        fields = ("state", "indices")
    elif family in _NVQ_SPECS:
        spec = _NVQ_SPECS[family]
        ng = (neuron_len + spec.groupsize - 1) // spec.groupsize
        nvec = (neuron_len + spec.vector_size - 1) // spec.vector_size
        nsign = (neuron_len + 7) // 8
        if family in {
            "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
            "NVQ3J", "NVQ3J-512", "NVQ3J-L",
        }:
            if not isinstance(artifact, NvqJscTables):
                raise TypeError(f"{family} stream artifact must contain NvqJscTables")
            encoded_codebook = _CODEBOOK_ID[spec.codebook] | _JSC_FLAG
            storage_layout = resolve_jsc_storage_layout(spec)
            table_payload = pack_jsc_tables(
                artifact.scale_lut,
                artifact.bank_for_state,
                artifact.codebooks,
                storage_layout=storage_layout,
            )
        else:
            encoded_codebook = _CODEBOOK_ID[spec.codebook]
            if spec.sign_mode == "index_parity":
                encoded_codebook |= _INDEX_PARITY_FLAG
            if artifact is not None:
                encoded_codebook |= _CUSTOM_CODEBOOK_FLAG
            table_payload = (
                b"" if artifact is None else pack_codebook(spec, np.asarray(artifact))
            )
        header = _NVQ_HEADER.pack(
            _NVQ_MAGIC,
            encoded_codebook,
            spec.sub_bits,
            spec.groupsize,
            0,
            neuron_len,
            len(shape),
        )
        if family in {
            "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
            "NVQ3J", "NVQ3J-512", "NVQ3J-L",
        } and storage_layout == "group64":
            stream_bits = (ng * 64,)
            packer = None
            fields = ("group64",)
        else:
            stream_bits = (ng * spec.sub_bits, nvec * spec.index_bits, nsign * 7)
            packer = _pack_nvq_bits
            fields = ("sub_scale", "indices", "signs")
    else:
        raise ValueError(f"unsupported flat expert precision: {family}")

    _check_row_stream_alignment(row_chunk, stream_bits)
    device_importance = None
    if (
        importance is not None
        and importance_rows_per_entry is not None
        and quant_backend in ACCELERATOR_BACKENDS
    ):
        device_importance = torch.as_tensor(
            importance, device=device, dtype=torch.float32
        ).contiguous()
    with blob_path.open("wb+") as output:
        if family in TPQ_PQ_SPECS_BY_LABEL:
            output.write(header)
        else:
            output.write(header)
            output.write(struct.pack(f"<{len(shape)}q", *shape))
            output.write(struct.pack("<I", out))
            output.write(table_payload)
        anchor_offset = output.tell()
        stream_offsets: list[int] = []
        offset = anchor_offset + out * anchor_bytes_per_row
        for bits in stream_bits:
            stream_offsets.append(offset)
            offset += (out * bits + 7) // 8
        output.truncate(offset)

        for start in range(0, out, row_chunk):
            end = min(start + row_chunk, out)
            chunk = (
                source.read_rows(start, end, device=device)
                if quant_backend in ACCELERATOR_BACKENDS and hasattr(source, "read_rows")
                else source[start:end]
            )
            chunk_importance = None
            if importance is not None:
                importance_shape = tuple(int(value) for value in importance.shape)
                if importance_rows_per_entry is not None:
                    if importance_rows_per_entry <= 0:
                        raise ValueError(
                            "importance_rows_per_entry must be positive"
                        )
                    expected_entries = (
                        out + importance_rows_per_entry - 1
                    ) // importance_rows_per_entry
                    if importance_shape != (expected_entries, neuron_len):
                        raise ValueError(
                            "streaming expert importance has shape "
                            f"{importance_shape}, expected "
                            f"{(expected_entries, neuron_len)}"
                        )
                    entry_ids = torch.arange(
                        start, end, device=device, dtype=torch.int64
                    ).div(importance_rows_per_entry, rounding_mode="floor")
                    objective = (
                        device_importance
                        if device_importance is not None
                        else torch.as_tensor(
                            importance, device=device, dtype=torch.float32
                        )
                    )
                    chunk_importance = objective.index_select(0, entry_ids)
                elif importance_shape == (neuron_len,):
                    chunk_importance = importance
                elif importance_shape == (out, neuron_len):
                    chunk_importance = importance[start:end]
                else:
                    raise ValueError(
                        f"streaming importance has unsupported shape "
                        f"{importance_shape}"
                    )
            tensor = _quantize_flat_stream_chunk(
                chunk,
                precision,
                artifact,
                importance=chunk_importance,
                quant_backend=quant_backend,
                device=device,
            )
            if anchor_bytes_per_row:
                output.seek(anchor_offset + start * anchor_bytes_per_row)
                output.write(
                    np.ascontiguousarray(
                        tensor.neuron_scale, dtype="<f2"
                    ).tobytes()
                )
            for stream_offset, bits, field in zip(
                stream_offsets, stream_bits, fields, strict=True
            ):
                output.seek(stream_offset + (start * bits) // 8)
                if field == "group64":
                    output.write(
                        pack_jsc_group64(
                            tensor.state,
                            tensor.indices,
                            tensor.signs,
                            neuron_len=neuron_len,
                        )
                    )
                    continue
                if field in {"delta_sign"}:
                    packed_bits = 1
                elif field == "signs":
                    packed_bits = 7
                elif field == "indices":
                    packed_bits = spec.index_bits
                elif field == "state":
                    packed_bits = spec.state_bits
                else:
                    packed_bits = spec.sub_bits
                if packer is None:
                    raise AssertionError("missing compact stream packer")
                output.write(packer(np.asarray(getattr(tensor, field)), packed_bits))
            del tensor
    return blob_path.stat().st_size


def _write_nepq_cohort_blob(
    source: _ExpertPoolRowSource,
    precision: ExpertPrecision,
    blob_path: Path,
    device: str,
    artifact_root: str | Path | None,
    importance: np.ndarray | torch.Tensor | None = None,
) -> tuple[int, bytes]:
    tables = resolve_precision_artifact(
        precision, artifact_root=artifact_root
    )
    if tables is None:
        raise ValueError(
            f"{precision.family} requires a frozen cross-expert table artifact"
        )
    anchor_multipliers = tuple(
        float(value)
        for value in str(
            precision.option("anchor_multipliers", "0.75,1.0,1.25")
        ).split(",")
    )
    base_config = NepqQuantConfig(
        anchor_multipliers=anchor_multipliers,
        refine_steps=int(precision.option("refine_steps", 2)),
        row_chunk=int(precision.option("row_chunk", 8)),
        bank_chunk=int(precision.option("bank_chunk", 8)),
        admm_iterations=int(precision.option("admm_iterations", 0)),
        admm_rho=float(precision.option("admm_rho", 1.0)),
    )
    if precision.family.endswith("-A"):
        if not isinstance(tables, NepqAArtifact):
            raise TypeError(
                f"{precision.family} artifact must contain NepqAArtifact"
            )
        tensor = quantize_nepq_a_fixed(
            source,
            precision.family,
            tables,
            importance=importance,
            rotation_block=int(precision.option("rotation_block", 2048)),
            rotation_seed=int(precision.option("rotation_seed", 0)),
            second_records=(
                None
                if precision.option("second_records") is None
                else int(precision.option("second_records"))
            ),
            target_nbytes=(
                None
                if precision.option("target_nbytes") is None
                else int(precision.option("target_nbytes"))
            ),
            target_bpw=(
                None
                if precision.option("target_bpw") is None
                else float(precision.option("target_bpw"))
            ),
            config=NepqAQuantConfig(
                base=base_config,
                residual_row_chunk=int(
                    precision.option("residual_row_chunk", 128)
                ),
                residual_block_chunk=int(
                    precision.option("residual_block_chunk", 1024)
                ),
            ),
            device=device,
        )
    else:
        tensor = quantize_nepq_fixed(
            source,
            precision.family,
            tables,
            importance=importance,
            rotation_block=int(precision.option("rotation_block", 0)),
            rotation_seed=int(precision.option("rotation_seed", 0)),
            config=base_config,
            device=device,
        )
    ng, nvec, nsuper, bank_count, rows = validate_nepq(tensor)
    spec = tensor.spec
    n_experts, out_per_expert, neuron_len = tensor.shape
    aux = (
        np.empty(0, dtype=np.uint8)
        if tensor.aux is None
        else np.asarray(tensor.aux)
    )
    with blob_path.open("wb") as output:
        output.write(
            _NEPQ_HEADER.pack(
                _NEPQ_MAGIC,
                _NEPQ_VERSION,
                spec.profile_id,
                spec.groups_per_supergroup,
                _NEPQ_FLAG_ROTATED if tensor.rotation_block else 0,
                n_experts,
                out_per_expert,
                neuron_len,
                bank_count,
                int(tensor.rotation_block),
                int(tensor.rotation_seed),
            )
        )
        output.write(
            np.ascontiguousarray(tensor.table_payloads, dtype=np.uint8).tobytes()
        )
        output.write(
            np.ascontiguousarray(tensor.neuron_scale, dtype="<f2").tobytes()
        )
        for values, columns, bits in (
            (tensor.state, ng, spec.state_bits),
            (tensor.indices, nvec, spec.index_bits),
            (aux, ng, spec.aux_bits),
        ):
            if not bits:
                continue
            flat = np.asarray(values).reshape(rows, columns)
            for start in range(0, rows, 256):
                output.write(_pack_nepq_bits(flat[start : start + 256], bits))
        output.write(
            np.ascontiguousarray(
                np.asarray(tensor.bank_ids).reshape(rows, nsuper),
                dtype=np.uint8,
            ).tobytes()
        )
        if spec.is_residual:
            second = np.asarray(
                tensor.residual_second_records
                if tensor.residual_second_records is not None
                else np.empty(0, dtype=np.uint16)
            )
            output.write(
                _RESIDUAL_HEADER.pack(
                    _RESIDUAL_MAGIC,
                    _RESIDUAL_VERSION,
                    spec.residual_record_bits,
                    spec.residual_position_bits,
                    _RESIDUAL_FLAG_SECOND if spec.residual_second else 0,
                    _RESIDUAL_DICTIONARY_ENTRIES,
                    tensor.residual_block_count,
                    second.size,
                    int(tensor.residual_padding_nbytes),
                    0,
                )
            )
            output.write(
                np.ascontiguousarray(
                    tensor.residual_codebook, dtype="<f2"
                ).tobytes()
            )
            output.write(
                _pack_nepq_bits(
                    np.asarray(tensor.residual_first),
                    spec.residual_record_bits,
                )
            )
            if spec.residual_second:
                output.write(
                    _pack_nepq_bits(
                        np.asarray(tensor.residual_second_mask), 1
                    )
                )
                output.write(
                    _pack_nepq_bits(second, spec.residual_record_bits)
                )
            if tensor.residual_padding_nbytes:
                output.write(bytes(int(tensor.residual_padding_nbytes)))
    payload_nbytes = blob_path.stat().st_size
    expected_nbytes = _NEPQ_HEADER.size + tensor.payload_nbytes
    if payload_nbytes != expected_nbytes:
        raise RuntimeError(
            f"{precision.family} stream size mismatch: "
            f"{payload_nbytes} != {expected_nbytes}"
        )
    runtime_payload = _pack_nint_moe_runtime(tensor)
    return payload_nbytes, runtime_payload


def _write_mixed_moe_axis0_blob(
    source,
    source_shape: tuple[int, ...],
    expert_shape: tuple[int, int, int],
    expert_precisions: tuple[ExpertPrecision, ...],
    blob_path: Path,
    row_chunk: int,
    quant_backend: str,
    device: str,
    artifact_root: str | Path | None,
    importance: np.ndarray | torch.Tensor | None = None,
) -> int:
    """Stream all supported precision families into one NIM2 container."""

    n_experts, rows_per_expert, columns = expert_shape
    if len(expert_precisions) != n_experts:
        raise ValueError(
            f"received {len(expert_precisions)} expert precisions for {n_experts} experts"
        )
    importance_shape = (
        None
        if importance is None
        else tuple(int(value) for value in importance.shape)
    )
    if importance_shape not in {
        None,
        (columns,),
        (n_experts, columns),
    }:
        raise ValueError(
            "mixed MoE importance must have shape "
            f"[{columns}] or [{n_experts},{columns}], got {importance_shape}"
        )
    if all(value.nint_spec is not None for value in expert_precisions):
        specs = tuple(
            value.nint_spec
            for value in expert_precisions
            if value.nint_spec is not None
        )
        return _write_nint_moe_axis0_blob(
            source,
            source_shape,
            expert_shape,
            specs,
            blob_path,
            row_chunk,
            quant_backend,
            device,
            importance=importance,
        )

    cohorts: dict[ExpertPrecision, list[int]] = {}
    for expert, precision in enumerate(expert_precisions):
        cohorts.setdefault(precision, []).append(expert)
    pool_paths: list[Path] = []
    try:
        with blob_path.open("wb") as output:
            output.write(
                _NINT_MOE_HDR.pack(
                    _NINT_MOE_MAGIC_V2,
                    n_experts,
                    rows_per_expert,
                    columns,
                    len(cohorts),
                )
            )
            for pool_index, (precision, expert_list) in enumerate(cohorts.items()):
                expert_ids = tuple(int(value) for value in expert_list)
                if importance is None or importance_shape == (columns,):
                    pool_importance = importance
                elif isinstance(importance, torch.Tensor):
                    pool_importance = importance.index_select(
                        0,
                        torch.as_tensor(
                            expert_ids,
                            device=importance.device,
                            dtype=torch.int64,
                        ),
                    )
                else:
                    pool_importance = np.ascontiguousarray(
                        np.asarray(importance)[np.asarray(expert_ids)]
                    )
                pool_source = _ExpertPoolRowSource(
                    source, source_shape, expert_shape, expert_ids
                )
                pool_path = blob_path.with_name(f"{blob_path.name}.pool{pool_index}.tmp")
                pool_paths.append(pool_path)
                if precision.nint_spec is not None:
                    def importance_rows(
                        start: int,
                        end: int,
                        *,
                        _importance=pool_importance,
                    ) -> np.ndarray | torch.Tensor | None:
                        if _importance is None or len(_importance.shape) == 1:
                            return _importance
                        row_ids = np.arange(start, end, dtype=np.int64)
                        expert_rows = row_ids // rows_per_expert
                        if isinstance(_importance, torch.Tensor):
                            return _importance.index_select(
                                0,
                                torch.as_tensor(
                                    expert_rows,
                                    device=_importance.device,
                                    dtype=torch.int64,
                                ),
                            )
                        return np.ascontiguousarray(_importance[expert_rows])

                    pool_nbytes = _write_nint_axis0_blob(
                        pool_source,
                        (len(expert_ids) * rows_per_expert, columns),
                        precision.nint_spec,
                        pool_path,
                        row_chunk,
                        quant_backend,
                        device,
                        importance_rows=(
                            importance_rows
                            if precision.nint_spec.bits in {2, 3, 4, 5, 6}
                            else None
                        ),
                    )
                    runtime_payload = b""
                elif precision.family == "MXFP4":
                    exact_writer = getattr(source, "write_mxfp4_expert_pool", None)
                    if exact_writer is None:
                        raise TypeError(
                            "MXFP4 expert preservation requires an exact native "
                            "MXFP4 source"
                        )
                    pool_nbytes = exact_writer(expert_ids, pool_path)
                    runtime_payload = b""
                elif precision.family.startswith("NEPQ"):
                    pool_nbytes, runtime_payload = _write_nepq_cohort_blob(
                        pool_source,
                        precision,
                        pool_path,
                        device,
                        artifact_root,
                        pool_importance,
                    )
                else:
                    pool_nbytes = _write_flat_family_axis0_blob(
                        pool_source,
                        (len(expert_ids) * rows_per_expert, columns),
                        precision,
                        pool_path,
                        row_chunk,
                        quant_backend,
                        device,
                        artifact_root,
                        importance=pool_importance,
                        importance_rows_per_entry=(
                            rows_per_expert
                            if importance_shape == (n_experts, columns)
                            else None
                        ),
                    )
                    runtime_payload = b""
                dtype = precision.family.encode("ascii")
                output.write(
                    _NINT_MOE_POOL_V2_HDR.pack(
                        len(expert_ids),
                        len(dtype),
                        pool_nbytes,
                        len(runtime_payload),
                    )
                )
                output.write(np.asarray(expert_ids, dtype=np.int32).tobytes())
                output.write(dtype)
                output.write(runtime_payload)
                with pool_path.open("rb") as pool_file:
                    shutil.copyfileobj(pool_file, output, length=32 * 1024 * 1024)
        return blob_path.stat().st_size
    finally:
        for pool_path in pool_paths:
            pool_path.unlink(missing_ok=True)


def _write_mfq(
    output_tmp: Path,
    header: FileHeader,
    records: list[BlobRecord],
    *,
    consume_blobs: bool = False,
) -> None:
    version = int(header.version)
    extra = dict(header.extra)
    if extra and version < 2:
        version = 2

    with output_tmp.open("wb") as out:
        out.write(MFQ_MAGIC)
        out.write(_u32(version))
        arch_b = header.model_arch.encode("utf-8")
        out.write(_u32(len(arch_b)))
        out.write(arch_b)
        if version >= 2:
            out.write(_u32(len(extra)))
            for k, v in extra.items():
                kb = str(k).encode("utf-8")
                vb = json.dumps(v).encode("utf-8")
                out.write(_u32(len(kb)))
                out.write(kb)
                out.write(_u32(len(vb)))
                out.write(vb)
        out.write(_u32(len(records)))
        for rec in records:
            name_b = rec.name.encode("utf-8")
            dtype_b = rec.dtype.encode("utf-8")
            out.write(_u32(len(name_b)))
            out.write(name_b)
            out.write(_u32(len(dtype_b)))
            out.write(dtype_b)
            out.write(struct.pack("<Q", rec.nbytes))
        for rec in records:
            try:
                with rec.path.open("rb") as src:
                    shutil.copyfileobj(src, out, length=32 * 1024 * 1024)
            finally:
                if consume_blobs:
                    rec.path.unlink(missing_ok=True)


def _nint_blob_nbytes(rows: int, columns: int, spec: NintSpec) -> int:
    groups = (columns + spec.groupsize - 1) // spec.groupsize
    header = _NINT_HDR.size + 4 + 2 * 8 + 8
    sub = (rows * groups * spec.sub_bits + 7) // 8
    q = (rows * groups * spec.groupsize * spec.bits + 7) // 8
    return int(header + rows * 4 + 2 * sub + q)


def _nint_moe_blob_nbytes(
    expert_shape: tuple[int, int, int],
    expert_specs: tuple[NintSpec, ...],
) -> int:
    n_experts, rows_per_expert, columns = expert_shape
    if len(expert_specs) != n_experts:
        raise ValueError("expert spec count does not match expert tensor shape")
    cohorts: dict[NintSpec, int] = {}
    for profile in expert_specs:
        cohorts[profile] = cohorts.get(profile, 0) + 1
    total = _NINT_MOE_HDR.size
    for profile, count in cohorts.items():
        dtype_nbytes = len(f"NINT{profile.bits}".encode("ascii"))
        total += (
            _NINT_MOE_POOL_V2_HDR.size
            + count * np.dtype(np.int32).itemsize
            + dtype_nbytes
        )
        total += _nint_blob_nbytes(count * rows_per_expert, columns, profile)
    return int(total)


def _mixed_moe_blob_nbytes(
    expert_shape: tuple[int, int, int],
    expert_precisions: tuple[ExpertPrecision, ...],
    artifact_root: str | Path | None,
) -> int:
    n_experts, rows_per_expert, columns = expert_shape
    if len(expert_precisions) != n_experts:
        raise ValueError("expert precision count does not match expert tensor shape")
    if all(value.nint_spec is not None for value in expert_precisions):
        specs = tuple(
            value.nint_spec
            for value in expert_precisions
            if value.nint_spec is not None
        )
        return _nint_moe_blob_nbytes(expert_shape, specs)

    cohorts: dict[ExpertPrecision, int] = {}
    for precision in expert_precisions:
        cohorts[precision] = cohorts.get(precision, 0) + 1
    total = _NINT_MOE_HDR.size
    flat_shape_header = 2 * 8 + 4
    for precision, expert_count in cohorts.items():
        rows = expert_count * rows_per_expert
        artifact = resolve_precision_artifact(
            precision, artifact_root=artifact_root
        )
        runtime_nbytes = 0
        if precision.nint_spec is not None:
            payload_nbytes = _nint_blob_nbytes(
                rows, columns, precision.nint_spec
            )
        elif precision.family == "MXFP4":
            payload_nbytes = len(
                mx_header_bytes(
                    "MXFP4",
                    (rows, columns),
                    (rows, columns // 2),
                    (rows, columns // 32),
                )
            ) + rows * (columns // 2 + columns // 32)
        elif precision.family == "NVQ1-L":
            payload_nbytes = (
                _NVQ1_L_HEADER.size
                + flat_shape_header
                + (4096 if artifact is not None else 0)
                + NVQ1_L_T8_S3.payload_nbytes(rows, columns)
            )
        elif precision.family == "NVQ1-S":
            payload_nbytes = (
                _NVQ1_S_HEADER.size
                + flat_shape_header
                + NVQ1_S.payload_nbytes(rows, columns)
            )
        elif precision.family == "NPQ0-L":
            if not isinstance(artifact, Npq0LTables):
                raise ValueError("NPQ0-L size estimation requires its table artifact")
            payload_nbytes = (
                _NPQ0_L_HEADER.size
                + flat_shape_header
                + NPQ0_L.payload_nbytes(rows, columns)
            )
        elif precision.family == "NPQ0-S":
            if not isinstance(artifact, Npq0STables):
                raise ValueError("NPQ0-S size estimation requires its table artifact")
            payload_nbytes = (
                _NPQ0_S_HEADER.size
                + flat_shape_header
                + NPQ0_S.payload_nbytes(rows, columns)
            )
        elif precision.family in TPQ_PQ_SPECS_BY_LABEL:
            if artifact is None:
                raise ValueError(
                    f"{precision.family} size estimation requires its codebook artifact"
                )
            payload_nbytes = tpq_pq_payload_nbytes(
                (rows, columns),
                TPQ_PQ_SPECS_BY_LABEL[precision.family],
            )
        elif precision.family in _NVQ_SPECS:
            spec = _NVQ_SPECS[precision.family]
            table_nbytes = 0
            if precision.family in {
                "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
                "NVQ3J", "NVQ3J-512", "NVQ3J-L",
            }:
                if artifact is None:
                    artifact = initial_jsc_tables(
                        NvqJscConfig(
                            banks=int(precision.option("banks", 4)),
                            iterations=int(precision.option("iterations", 4)),
                            assignment_refine_steps=int(
                                precision.option("assignment_refine_steps", 2)
                            ),
                            search_steps=int(precision.option("search_steps", 19)),
                            group_chunk=int(precision.option("group_chunk", 1024)),
                            spec=spec,
                        )
                    )
                if not isinstance(artifact, NvqJscTables):
                    raise ValueError(
                        f"{precision.family} size estimation requires its table artifact"
                    )
                table_nbytes = 64 + int(np.asarray(artifact.codebooks).size)
            elif artifact is not None:
                table_nbytes = spec.codebook_entries * 2
            payload_nbytes = (
                _NVQ_HEADER.size
                + flat_shape_header
                + table_nbytes
                + (
                    jsc_payload_nbytes(spec, rows, columns)
                    if precision.family in {
                        "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
                        "NVQ3J", "NVQ3J-512", "NVQ3J-L",
                    }
                    else spec.payload_nbytes(rows, columns)
                )
            )
        elif precision.family in _NEPQ_SPECS:
            if artifact is None:
                raise ValueError(
                    f"{precision.family} size estimation requires its table artifact"
                )
            tables = np.asarray(
                artifact.table_payloads
                if isinstance(artifact, NepqAArtifact)
                else artifact
            )
            if tables.ndim != 2:
                raise ValueError(
                    f"{precision.family} table artifact must be a 2D byte pool"
                )
            nepq_spec = _NEPQ_SPECS[precision.family]
            payload_nbytes = (
                nepq_spec.payload_nbytes(
                    expert_count,
                    rows_per_expert,
                    columns,
                    bank_count=int(tables.shape[0]),
                )
                + _NEPQ_HEADER.size
            )
            if nepq_spec.is_residual:
                block_count = expert_count * rows_per_expert * math.ceil(
                    (columns // nepq_spec.vector_size)
                    / nepq_spec.residual_block_vectors
                )
                payload_nbytes += (
                    _RESIDUAL_HEADER.size
                    + _RESIDUAL_DICTIONARY_ENTRIES * nepq_spec.vector_size * 2
                    + math.ceil(
                        block_count * nepq_spec.residual_record_bits / 8
                    )
                )
                if nepq_spec.residual_second:
                    weights = expert_count * rows_per_expert * columns
                    requested_nbytes = precision.option("target_nbytes")
                    requested_bpw = precision.option("target_bpw")
                    requested_records = precision.option("second_records")
                    supplied = sum(
                        value is not None
                        for value in (
                            requested_nbytes,
                            requested_bpw,
                            requested_records,
                        )
                    )
                    if supplied > 1:
                        raise ValueError(
                            "NEPQ1-A accepts one residual budget option"
                        )
                    if requested_nbytes is not None:
                        payload_nbytes = int(requested_nbytes)
                    elif requested_bpw is not None or supplied == 0:
                        bpw = 1.5625 if requested_bpw is None else float(requested_bpw)
                        payload_nbytes = int(round(bpw * weights / 8.0))
                    else:
                        payload_nbytes += math.ceil(block_count / 8) + math.ceil(
                            int(requested_records)
                            * nepq_spec.residual_record_bits
                            / 8
                        )
            if int(
                precision.option(
                    "rotation_block", 2048 if nepq_spec.is_residual else 0
                )
            ):
                runtime_nbytes = _NINT_MOE_ROTATION_HDR.size + columns
        else:
            raise ValueError(f"unsupported expert precision: {precision.family}")
        dtype_nbytes = len(precision.family.encode("ascii"))
        total += (
            _NINT_MOE_POOL_V2_HDR.size
            + expert_count * np.dtype(np.int32).itemsize
            + dtype_nbytes
            + runtime_nbytes
            + payload_nbytes
        )
    return int(total)


def _estimate_bytes(
    plan: list[TensorPlan],
    spec: NintSpec,
    artifact_root: str | Path | None = None,
    *,
    custom_codebook: bool = True,
    nvq3_jsc_banks: int = 2,
    nvq_jsc_banks: int = 4,
) -> tuple[int, int]:
    nint_total = 0
    dense_total = 0
    for item in plan:
        n = int(np.prod(item.shape))
        if item.target_dtype == "NINTM":
            if item.expert_shape is None or item.expert_precisions is None:
                raise ValueError(f"NINTM plan lacks expert metadata: {item.name}")
            nint_total += _mixed_moe_blob_nbytes(
                item.expert_shape,
                item.expert_precisions,
                artifact_root,
            )
        elif item.target_dtype == "NINT8-0":
            nint_total += _plan_blob_nbytes(item, spec, artifact_root)
        elif item.target_dtype.startswith("NINT"):
            item_spec = _spec_for_plan(item, spec)
            nint_total += _nint_blob_nbytes(item.shape[0], item.shape[1], item_spec)
        elif item.target_dtype.startswith("NVQ") or item.target_dtype == "NPQ0-L":
            nint_total += _plan_blob_nbytes(
                item,
                spec,
                artifact_root,
                custom_codebook=custom_codebook,
                nvq3_jsc_banks=nvq3_jsc_banks,
                nvq_jsc_banks=nvq_jsc_banks,
            )
        else:
            item_size = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8}[
                item.target_dtype
            ]
            dense_total += 4 + 8 * len(item.shape) + n * item_size
    return nint_total, dense_total


def _plan_blob_nbytes(
    item: TensorPlan,
    spec: NintSpec,
    artifact_root: str | Path | None = None,
    *,
    custom_codebook: bool = True,
    nvq3_jsc_banks: int = 2,
    nvq_jsc_banks: int = 4,
) -> int:
    n = int(np.prod(item.shape))
    if item.target_dtype == "NINTM":
        if item.expert_shape is None or item.expert_precisions is None:
            raise ValueError(f"NINTM plan lacks expert metadata: {item.name}")
        return _mixed_moe_blob_nbytes(
            item.expert_shape,
            item.expert_precisions,
            artifact_root,
        )
    if item.target_dtype == "NINT8-0":
        from mfq.tools import quantize_gguf_to_mfq as gguf_quantizer

        shared_plan = gguf_quantizer.GgufTensorPlan(
            name=item.name,
            source_name=item.source_name or item.name,
            source_shape=item.shape,
            original_shape=item.shape,
            storage_shape=(int(item.shape[0]), int(item.shape[1])),
            source_type=item.source_dtype,
            recipe_type=item.gguf_type or "Q8_0",
            target_dtype=item.target_dtype,
        )
        return gguf_quantizer._estimate_blob_bytes(shared_plan)
    if item.target_dtype.startswith("NINT"):
        item_spec = _spec_for_plan(item, spec)
        return _nint_blob_nbytes(item.shape[0], item.shape[1], item_spec)
    if item.target_dtype.startswith("NVQ") or item.target_dtype == "NPQ0-L":
        from mfq.tools import quantize_gguf_to_mfq as gguf_quantizer

        shared_plan = gguf_quantizer.GgufTensorPlan(
            name=item.name,
            source_name=item.source_name or item.name,
            source_shape=item.shape,
            original_shape=item.shape,
            storage_shape=(
                int(np.prod(item.shape[:-1])),
                int(item.shape[-1]),
            ),
            source_type=item.source_dtype,
            recipe_type=item.gguf_type or item.target_dtype,
            target_dtype=item.target_dtype,
        )
        jsc_banks = (
            nvq3_jsc_banks
            if item.target_dtype in {"NVQ3J", "NVQ3J-512", "NVQ3J-L"}
            else nvq_jsc_banks
        )
        return gguf_quantizer._estimate_blob_bytes(
            shared_plan,
            custom_codebook=custom_codebook,
            jsc_banks=jsc_banks,
            expert_artifact_root=artifact_root,
        )
    item_size = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8}[
        item.target_dtype
    ]
    return 4 + 8 * len(item.shape) + n * item_size


def convert(args: argparse.Namespace) -> None:
    input_mfq_arg = getattr(args, "input_mfq", "")
    root = Path(input_mfq_arg or args.input).resolve()
    mfq_checkpoint = None
    source_inventory = None
    source_config = None
    if input_mfq_arg:
        from mfq.quantize.mfq_source import FullPrecisionMfqCheckpoint

        mfq_checkpoint = FullPrecisionMfqCheckpoint(root)
        source_config = mfq_checkpoint.model_config()
        source_inventory = {
            name: SourceTensorMetadata(
                name=name,
                shard=f"mfq-{info.source_index:05d}",
                shape=info.shape,
                dtype=info.dtype,
                nbytes=info.nbytes,
            )
            for name, info in mfq_checkpoint.infos.items()
        }
    output = Path(args.output).resolve()
    split_max_size = int(getattr(args, "split_max_size", 0))
    split_max_tensors = int(getattr(args, "split_max_tensors", 0))
    validate_split_limits(split_max_size, split_max_tensors)
    if (output.exists() or matching_shard_paths(output)) and not args.overwrite:
        raise FileExistsError(f"output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    spec = NintSpec(bits=args.bits, groupsize=args.groupsize, sub_bits=args.sub_bits)
    mostly_bf16 = bool(getattr(args, "bf16", False))
    if mfq_checkpoint is not None and mostly_bf16:
        mfq_checkpoint.close()
        raise ValueError("--bf16/full-precision output requires an HF source")
    selected_backend = resolve_quant_backend(
        "cpu" if mostly_bf16 else args.quant_backend,
        args.device,
    )
    quant_backend = selected_backend.name
    quant_device = selected_backend.device
    row_chunk = resolve_row_chunk(args.row_chunk, quant_backend)
    recipe_gguf = getattr(args, "recipe_gguf", "")
    recipe_types = _load_gguf_recipe(Path(recipe_gguf).resolve()) if recipe_gguf else None
    calibration_scheme_path = getattr(args, "calibration_scheme", "")
    calibration_scheme = (
        load_scheme(Path(calibration_scheme_path).resolve()) if calibration_scheme_path else None
    )
    artifact_root = (
        calibration_scheme.path.parent
        if calibration_scheme is not None and calibration_scheme.path is not None
        else None
    )
    if calibration_scheme is not None and calibration_scheme.inint_selector is not None:
        raise ValueError(
            "this converter does not yet write mixed-row ININT tensors; "
            "use a uniform/mixed-tensor calibration scheme"
        )
    dense_dtype = getattr(args, "dense_dtype", "f16").upper()
    if mostly_bf16 and (recipe_types is not None or calibration_scheme is not None):
        raise ValueError("--bf16 cannot be combined with a recipe or calibration scheme")
    plan = _plan(
        root,
        text_only=args.text_only,
        recipe_types=recipe_types,
        dense_dtype=dense_dtype,
        calibration_scheme=calibration_scheme,
        mostly_bf16=mostly_bf16,
        source_inventory=source_inventory,
        source_config=source_config,
        default_nint_dtype=f"NINT{spec.bits}",
    )
    plan = _apply_recipe_family_mappings(
        plan,
        npq0_l=bool(getattr(args, "npq0_l", False)),
        nvq3_jsc=bool(getattr(args, "nvq3_jsc", False)),
        nvq3_jsc_512=bool(getattr(args, "nvq3_jsc_512", False)),
        nvq3_to_nint3=bool(getattr(args, "nvq3_to_nint3", False)),
        iq2_s_to_nint2=bool(getattr(args, "iq2_s_to_nint2", False)),
        q8_to_nint8_zero=bool(getattr(args, "q8_to_nint8_zero", False)),
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
    plan = _normalize_hf_expert_storage(plan)
    _validate_native_source_precision(plan)
    _validate_runtime_fused_pairs(plan)
    if args.limit_tensors:
        plan = plan[: args.limit_tensors]
    imatrix_path_arg = getattr(args, "imatrix", "")
    if mostly_bf16 and imatrix_path_arg:
        raise ValueError("--bf16 cannot be combined with an imatrix")
    imatrix = (
        load_importance_matrix(Path(imatrix_path_arg).resolve())
        if imatrix_path_arg
        else None
    )
    imatrix_bindings = (
        {} if imatrix is None else _bind_hf_imatrix(imatrix, plan)
    )
    requested_calibration = getattr(args, "nvq_calibration", "auto")
    calibration_mode = (
        "gain"
        if requested_calibration == "auto" and imatrix_path_arg
        else "none"
        if requested_calibration == "auto"
        else requested_calibration
    )
    if calibration_mode != "none" and imatrix is None:
        raise ValueError(
            f"NVQ calibration mode {calibration_mode} requires --imatrix"
        )
    nvq_codebook_scope = getattr(args, "nvq_codebook_scope", "tensor")
    if bool(getattr(args, "npq0_l", False)) and nvq_codebook_scope != "tensor":
        raise ValueError("--npq0-l requires --nvq-codebook-scope tensor")
    nint_est, dense_est = _estimate_bytes(
        plan,
        spec,
        artifact_root,
        custom_codebook=nvq_codebook_scope == "tensor",
        nvq3_jsc_banks=int(getattr(args, "nvq3_jsc_banks", 2)),
        nvq_jsc_banks=int(getattr(args, "nvq_jsc_banks", 4)),
    )
    total_src = 0
    for item in plan:
        metadata = (
            None
            if source_inventory is None
            else source_inventory.get(item.source_name or item.name)
        )
        total_src += (
            metadata.nbytes
            if metadata is not None
            else int(np.prod(item.shape)) * 2
        )
    target_counts: dict[str, int] = {}
    target_bytes_est: dict[str, int] = {}
    for p in plan:
        target_counts[p.target_dtype] = target_counts.get(p.target_dtype, 0) + 1
        nb = _plan_blob_nbytes(
            p,
            spec,
            artifact_root,
            custom_codebook=nvq_codebook_scope == "tensor",
            nvq3_jsc_banks=int(getattr(args, "nvq3_jsc_banks", 2)),
            nvq_jsc_banks=int(getattr(args, "nvq_jsc_banks", 4)),
        )
        target_bytes_est[p.target_dtype] = target_bytes_est.get(p.target_dtype, 0) + nb
    print(
        json.dumps(
            {
                "input": str(root),
                "output": str(output),
                "tensors": len(plan),
                "target_counts": target_counts,
                "target_estimated_gb": {k: round(v / 1e9, 3) for k, v in sorted(target_bytes_est.items())},
                "source_full_precision_gb": round(total_src / 1e9, 3),
                "estimated_mfq_gb": round((nint_est + dense_est) / 1e9, 3),
                "default_spec": {"bits": spec.bits, "groupsize": spec.groupsize, "sub_bits": spec.sub_bits},
                "recipe": _artifact_provenance_name(recipe_gguf),
                "calibration_scheme": (
                    _artifact_provenance_name(calibration_scheme_path)
                    if calibration_scheme_path
                    else None
                ),
                "imatrix": (
                    None
                    if imatrix is None
                    else {
                        "path": str(imatrix.path),
                        "entries": len(imatrix.entries),
                        "bound_tensors": len(imatrix_bindings),
                        "datasets": list(imatrix.datasets),
                        "chunk_count": imatrix.chunk_count,
                        "chunk_size": imatrix.chunk_size,
                        "legacy": imatrix.legacy,
                    }
                ),
                "quant_backend": quant_backend,
                "device": quant_device,
                "row_chunk": row_chunk,
                "text_only": args.text_only,
                "dense_dtype": "MOSTLY_BF16" if mostly_bf16 else dense_dtype,
                "mostly_bf16": mostly_bf16,
                "mapping": dict(sorted(_RECIPE_TARGETS.items())),
                "nvq_calibration": calibration_mode,
                "nvq_codebook_scope": nvq_codebook_scope,
                "tensor_precision_overrides": dict(
                    sorted(tensor_precision_overrides.items())
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if getattr(args, "dry_run", False):
        if mfq_checkpoint is not None:
            mfq_checkpoint.close()
        return

    # Reuse the mature GGUF-source tensor-wise VQ trainers and blob writer.
    # Only the row source differs; format selection and on-disk layout remain
    # one implementation.
    from mfq.tools import quantize_gguf_to_mfq as gguf_quantizer

    codebook_artifact_root = (
        Path(getattr(args, "nvq_codebook_artifact_dir", "")).resolve()
        if getattr(args, "nvq_codebook_artifact_dir", "")
        else Path(str(output) + ".codebooks")
    )
    codebook_config = gguf_quantizer.TensorCodebookTrainingConfig(
        iterations=int(getattr(args, "nvq_codebook_iterations", 4)),
        projection_candidates=int(
            getattr(args, "nvq_codebook_projection_candidates", 48)
        ),
        quant_backend=quant_backend,
        device=quant_device,
        group_chunk=int(getattr(args, "nvq_group_chunk", 32768)),
        row_chunk=int(getattr(args, "nvq_codebook_row_chunk", 512)),
        search_steps=int(getattr(args, "nvq_search_steps", 19)),
        nvq1_l_anchor_multipliers=tuple(
            getattr(args, "nvq1_l_anchor_multipliers", (0.75,))
        ),
        nvq1_l_refine_steps=int(getattr(args, "nvq1_l_refine_steps", 2)),
        nvq_native_assignment=(
            quant_backend in {"cuda", "metal"}
            and getattr(args, "nvq_assignment", "native") == "native"
        ),
        nvq1_l_native_assignment=(
            quant_backend in {"cuda", "metal"}
            and getattr(args, "nvq1_l_assignment", "native") == "native"
        ),
        min_validation_improvement=float(
            getattr(args, "nvq_codebook_min_improvement", 0.0)
        ),
    )
    jsc_config = gguf_quantizer.NvqJscConfig(
        banks=int(getattr(args, "nvq_jsc_banks", 4)),
        iterations=int(getattr(args, "nvq_jsc_iterations", 4)),
        assignment_refine_steps=int(
            getattr(args, "nvq_jsc_assignment_refine_steps", 2)
        ),
        search_steps=int(getattr(args, "nvq_search_steps", 19)),
        raw_multiplier=int(getattr(args, "nvq_jsc_raw_multiplier", 8)),
        learned_scale_lut=True,
        codebook_storage="int8",
        group_chunk=int(getattr(args, "nvq_group_chunk", 32768)),
        seed=int(getattr(args, "nvq_codebook_seed", 20260716)),
    )
    npq0_l_config = gguf_quantizer.Npq0LConfig(
        iterations=int(getattr(args, "npq0_l_iterations", 4)),
        assignment_refine_steps=int(
            getattr(args, "npq0_l_assignment_refine_steps", 2)
        ),
        fixed_refine_steps=int(getattr(args, "npq0_l_fixed_refine_steps", 3)),
        kmeans_iterations=int(getattr(args, "npq0_l_kmeans_iterations", 8)),
        group_chunk=int(getattr(args, "npq0_l_group_chunk", 512)),
        seed=int(getattr(args, "nvq_codebook_seed", 20260716)),
    )
    row_importance_path = getattr(args, "nvq_jsc_row_importance", "")
    row_importance = (
        gguf_quantizer.load_row_importance(Path(row_importance_path))
        if row_importance_path
        else None
    )
    if row_importance is not None:
        if imatrix is None or calibration_mode != "group24":
            raise ValueError(
                "--nvq-jsc-row-importance requires --imatrix and "
                "--nvq-calibration group24"
            )
        if nvq_codebook_scope != "tensor":
            raise ValueError(
                "--nvq-jsc-row-importance requires --nvq-codebook-scope tensor"
            )
        for item in plan:
            if item.target_dtype in _JSC_DTYPES:
                row_importance.require(item.name, int(item.shape[0]))

    temp_dir_arg = getattr(args, "temp_dir", "")
    tmp_root = (
        Path(temp_dir_arg).resolve()
        if temp_dir_arg
        else output.parent / f".{output.name}.tmp_blobs"
    )
    resume_temp = bool(getattr(args, "resume_temp", False))
    if tmp_root.exists() and not resume_temp:
        raise FileExistsError(f"temporary directory exists: {tmp_root}")
    tmp_root.mkdir(parents=True, exist_ok=resume_temp)
    records: list[BlobRecord] = []
    codebook_results: dict[str, dict[str, object]] = {}
    gain_results: dict[str, dict[str, object]] = {}
    start_time = time.time()
    completed = False

    try:
        by_shard: dict[str, list[TensorPlan]] = {}
        for item in plan:
            by_shard.setdefault(item.shard, []).append(item)

        done = 0
        for shard in sorted(by_shard):
            for item in by_shard[shard]:
                done += 1
                blob_path = tmp_root / f"{done:05d}.blob"
                expected_nbytes = _plan_blob_nbytes(
                    item,
                    spec,
                    artifact_root,
                    custom_codebook=nvq_codebook_scope == "tensor",
                    nvq3_jsc_banks=int(getattr(args, "nvq3_jsc_banks", 2)),
                    nvq_jsc_banks=int(getattr(args, "nvq_jsc_banks", 4)),
                )
                variable_codebook_size = (
                    nvq_codebook_scope == "tensor"
                    and item.target_dtype in {"NVQ1-L", "NVQ2", "NVQ3"}
                )
                if (
                    resume_temp
                    and blob_path.is_file()
                    and blob_path.stat().st_size > 0
                    and (
                        variable_codebook_size
                        or blob_path.stat().st_size == expected_nbytes
                    )
                ):
                    reused_nbytes = blob_path.stat().st_size
                    records.append(
                        BlobRecord(item.name, item.target_dtype, reused_nbytes, blob_path)
                    )
                    print(
                        json.dumps(
                            {
                                "done": done,
                                "total": len(plan),
                                "name": item.name,
                                "dtype": item.target_dtype,
                                "blob_mb": round(reused_nbytes / 1e6, 2),
                                "status": "reused",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    continue
                if blob_path.exists():
                    blob_path.unlink()
                for stale_pool in tmp_root.glob(f"{blob_path.name}.pool*.tmp"):
                    stale_pool.unlink()

                t0 = time.time()
                source_name = item.source_name or item.name
                # Keep only one safetensors mmap alive at a time.  Holding a
                # multi-gigabyte shard open across all tensors lets touched
                # pages accumulate in the process working set on Windows.
                if item.target_dtype == "NINTM" and item.expert_source_names is not None:
                    if item.expert_shape is None or item.expert_precisions is None or \
                            item.expert_source_shards is None:
                        raise ValueError(f"NINTM plan lacks GLM expert metadata: {item.name}")
                    source = (
                        _MfqGlmExpertRowSource(
                            mfq_checkpoint,
                            item.expert_shape,
                            item.expert_source_names,
                        )
                        if mfq_checkpoint is not None
                        else _GlmExpertRowSource(
                            root,
                            item.expert_shape,
                            item.expert_source_names,
                            item.expert_source_shards,
                        )
                    )
                    try:
                        expert_importance = _hf_expert_importance(
                            item, imatrix_bindings.get(item.name)
                        )
                        flattened_shape = (
                            item.expert_shape[0] * item.expert_shape[1],
                            item.expert_shape[2],
                        )
                        nbytes = _write_mixed_moe_axis0_blob(
                            source,
                            flattened_shape,
                            item.expert_shape,
                            item.expert_precisions,
                            blob_path,
                            row_chunk,
                            quant_backend,
                            quant_device,
                            artifact_root,
                            importance=expert_importance,
                        )
                    finally:
                        source.close()
                    del source
                else:
                    source_name = item.source_name or item.name
                    raw_source = (
                        mfq_checkpoint.tensor_source(source_name)
                        if mfq_checkpoint is not None
                        else _RawSafeTensorSlice(root / shard, source_name)
                    )
                    if item.target_dtype == "NINTM":
                        if item.expert_shape is None or item.expert_precisions is None:
                            raise ValueError(f"NINTM plan lacks expert metadata: {item.name}")
                        if item.transform is not None:
                            source = _transform_glm_kv_b(
                                raw_source.tensor(), item
                            )
                        else:
                            source = raw_source
                        expert_importance = _hf_expert_importance(
                            item, imatrix_bindings.get(item.name)
                        )
                        nbytes = _write_mixed_moe_axis0_blob(
                            source,
                            item.shape,
                            item.expert_shape,
                            item.expert_precisions,
                            blob_path,
                            row_chunk,
                            quant_backend,
                            quant_device,
                            artifact_root,
                            importance=expert_importance,
                        )
                    elif item.target_dtype == "NINT8-0":
                        source = _HfPlanRowSource(raw_source, item)
                        nbytes = gguf_quantizer._write_nint8_zero_axis0_blob(
                            source,
                            item.shape,
                            blob_path,
                            row_chunk,
                            quant_backend,
                            quant_device,
                        )
                    elif item.target_dtype.startswith("NINT"):
                        item_spec = _spec_for_plan(item, spec)
                        source = _HfPlanRowSource(raw_source, item)
                        nbytes = _write_nint_axis0_blob(
                            source,
                            item.shape,
                            item_spec,
                            blob_path,
                            row_chunk,
                            quant_backend,
                            quant_device,
                            importance_rows=(
                                None
                                if item.name not in imatrix_bindings
                                else imatrix_bindings[item.name].rows
                            ),
                        )
                    elif (
                        item.target_dtype.startswith("NVQ")
                        or item.target_dtype == "NPQ0-L"
                    ):
                        source = _HfPlanRowSource(raw_source, item)
                        imatrix_binding = imatrix_bindings.get(item.name)
                        shared_plan = gguf_quantizer.GgufTensorPlan(
                            name=item.name,
                            source_name=source_name,
                            source_shape=item.shape,
                            original_shape=item.shape,
                            storage_shape=item.shape,
                            source_type=item.source_dtype,
                            recipe_type=item.gguf_type or item.target_dtype,
                            target_dtype=item.target_dtype,
                        )
                        source_identity = (
                            root
                            if mfq_checkpoint is not None
                            else root / shard
                        )
                        recipe_identity = (
                            Path(recipe_gguf).resolve()
                            if recipe_gguf
                            else (
                                root / "config.json"
                                if root.is_dir() and (root / "config.json").is_file()
                                else source_identity
                            )
                        )
                        codebook = None
                        jsc_tables = None
                        npq0_l_tables = None
                        item_jsc_config = (
                            replace(
                                jsc_config,
                                spec=_NVQ_SPECS[item.target_dtype],
                                banks=(
                                    int(getattr(args, "nvq3_jsc_banks", 2))
                                    if item.target_dtype
                                    in {"NVQ3J", "NVQ3J-512", "NVQ3J-L"}
                                    else jsc_config.banks
                                ),
                                learned_scale_lut=(
                                    bool(getattr(args, "nvq3_jsc_learned_scale", False))
                                    if item.target_dtype
                                    in {"NVQ3J", "NVQ3J-512", "NVQ3J-L"}
                                    else jsc_config.learned_scale_lut
                                ),
                            )
                            if item.target_dtype in _JSC_DTYPES
                            else jsc_config
                        )
                        if item.target_dtype == "NPQ0-L":
                            npq0_l_tables, metrics = (
                                gguf_quantizer._train_or_load_npq0_l_tables(
                                    source,
                                    shared_plan,
                                    source_identity,
                                    recipe_identity,
                                    codebook_artifact_root,
                                    npq0_l_config,
                                    int(getattr(args, "nvq_codebook_train_rows", 2048)),
                                    int(getattr(args, "nvq_codebook_validation_rows", 512)),
                                    int(getattr(args, "nvq_codebook_seed", 20260716)),
                                    quant_device,
                                    imatrix,
                                    imatrix_binding,
                                )
                            )
                            codebook_results[item.name] = metrics
                        elif (
                            item.target_dtype in _JSC_DTYPES
                            and nvq_codebook_scope == "tensor"
                        ):
                            jsc_tables, metrics = (
                                gguf_quantizer._train_or_load_jsc_tables(
                                    source,
                                    shared_plan,
                                    source_identity,
                                    recipe_identity,
                                    codebook_artifact_root,
                                    item_jsc_config,
                                    int(getattr(args, "nvq_codebook_train_rows", 2048)),
                                    int(getattr(args, "nvq_codebook_validation_rows", 512)),
                                    int(getattr(args, "nvq_codebook_seed", 20260716)),
                                    quant_device,
                                    imatrix,
                                    imatrix_binding,
                                    row_importance,
                                )
                            )
                            codebook_results[item.name] = metrics
                        elif item.target_dtype in _JSC_DTYPES:
                            jsc_tables = gguf_quantizer.initial_jsc_tables(
                                item_jsc_config
                            )
                        elif nvq_codebook_scope == "tensor":
                            codebook, metrics = (
                                gguf_quantizer._train_or_load_tensor_codebook(
                                    source,
                                    shared_plan,
                                    source_identity,
                                    recipe_identity,
                                    codebook_artifact_root,
                                    codebook_config,
                                    int(getattr(args, "nvq_codebook_train_rows", 2048)),
                                    int(getattr(args, "nvq_codebook_validation_rows", 512)),
                                    int(getattr(args, "nvq_codebook_seed", 20260716)),
                                    imatrix,
                                    imatrix_binding,
                                )
                            )
                            codebook_results[item.name] = metrics
                        nvq_result = gguf_quantizer._write_nvq_blob(
                            source=source,
                            shape=item.shape,
                            target_dtype=item.target_dtype,
                            blob_path=blob_path,
                            row_chunk=row_chunk,
                            quant_backend=quant_backend,
                            device=quant_device,
                            group_chunk=int(getattr(args, "nvq_group_chunk", 32768)),
                            nvq1_l_candidates=int(getattr(args, "nvq1_l_candidates", 0)),
                            nvq1_l_anchor_multipliers=tuple(
                                getattr(args, "nvq1_l_anchor_multipliers", (0.75,))
                            ),
                            nvq1_l_refine_steps=int(
                                getattr(args, "nvq1_l_refine_steps", 2)
                            ),
                            importance_rows=(
                                None if imatrix_binding is None else imatrix_binding.rows
                            ),
                            codebook=codebook,
                            search_steps=int(getattr(args, "nvq_search_steps", 19)),
                            nvq_native_assignment=(
                                quant_backend in {"cuda", "metal"}
                                and getattr(args, "nvq_assignment", "native") == "native"
                            ),
                            nvq1_l_native_assignment=(
                                quant_backend in {"cuda", "metal"}
                                and getattr(args, "nvq1_l_assignment", "native") == "native"
                            ),
                            jsc_tables=jsc_tables,
                            jsc_assignment_refine_steps=(
                                item_jsc_config.assignment_refine_steps
                            ),
                            npq0_l_tables=npq0_l_tables,
                            npq0_l_config=npq0_l_config,
                            calibration_mode=(
                                calibration_mode
                                if imatrix_binding is not None
                                else "none"
                            ),
                        )
                        nbytes = nvq_result.nbytes
                        if nvq_result.gain_calibration is not None:
                            gain_results[item.name] = nvq_result.gain_calibration
                    else:
                        source = raw_source.tensor()
                        if item.row_start is not None or item.row_end is not None:
                            source = source[item.row_start:item.row_end]
                        nbytes = _dense_blob_from_tensor(source, blob_path, item.target_dtype)
                    del source, raw_source
                if not variable_codebook_size and nbytes != expected_nbytes:
                    raise RuntimeError(
                        f"blob size mismatch for {item.name}: {nbytes} != {expected_nbytes}"
                    )
                records.append(BlobRecord(item.name, item.target_dtype, nbytes, blob_path))
                elapsed = time.time() - t0
                print(
                    json.dumps(
                        {
                            "done": done,
                            "total": len(plan),
                            "name": item.name,
                            "shape": item.shape,
                            "dtype": item.target_dtype,
                            "gguf_name": item.gguf_name,
                            "gguf_type": item.gguf_type,
                            "blob_mb": round(nbytes / 1e6, 2),
                            "sec": round(elapsed, 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        config_path_arg = getattr(args, "model_config", "")
        if config_path_arg:
            config_path = Path(config_path_arg).resolve()
            config = json.loads(config_path.read_text(encoding="utf-8"))
        elif mfq_checkpoint is not None:
            config = mfq_checkpoint.model_config()
        else:
            config_path = root / "config.json"
            config = (
                json.loads(config_path.read_text(encoding="utf-8"))
                if config_path.exists()
                else {}
            )
        config_text = config.get("text_config", config)
        mtp_plan_names = [item.name for item in plan if item.name.startswith("mtp.")]
        mtp_config_layers = (
            int(config_text.get("mtp_num_hidden_layers", 0) or 0)
            if isinstance(config_text, dict)
            else 0
        )
        mtp_included = bool(mtp_plan_names)
        if mtp_config_layers > 0 and not mtp_included:
            # A stripped head must not leave a capability bit behind.  Copy
            # through JSON so callers' config dictionaries remain untouched.
            config = json.loads(json.dumps(config))
            config_text = config.get("text_config", config)
            if isinstance(config_text, dict):
                config_text["mtp_num_hidden_layers"] = 0
                config_text["mtp_use_dedicated_embeddings"] = False
        model_type = str(config.get("model_type", "unknown"))
        runtime_assets: list[RuntimeAsset] = []
        if mfq_checkpoint is not None:
            manifest = mfq_checkpoint.header.extra.get(ASSET_MANIFEST_KEY, {})
            manifest_assets = (
                manifest.get("assets", {}) if isinstance(manifest, dict) else {}
            )
            media_by_record = {
                str(value.get("record")): str(
                    value.get("media_type", "application/octet-stream")
                )
                for value in manifest_assets.values()
                if isinstance(value, dict) and value.get("record")
            }
            runtime_assets.extend(
                RuntimeAsset(
                    name,
                    media_by_record.get(name, "application/octet-stream"),
                    mfq_checkpoint.store.read_blob(name),
                )
                for name, record in mfq_checkpoint.store.records.items()
                if is_asset_record(name) and record.dtype == ASSET_DTYPE
            )
        assets_by_name = {asset.name: asset for asset in runtime_assets}
        if config and (
            mfq_checkpoint is None
            or config_path_arg
            or MODEL_CONFIG_ASSET not in assets_by_name
        ):
            assets_by_name[MODEL_CONFIG_ASSET] = model_config_asset(config)
        if (
            _is_minicpmo45_config(config)
            and MINICPMO45_RESAMPLER_POS_EMBED_ASSET not in assets_by_name
        ):
            position_asset = minicpmo45_resampler_pos_embed_asset()
            assets_by_name[position_asset.name] = position_asset
        tokenizer_gguf_arg = getattr(args, "tokenizer_gguf", "")
        tokenizer_gguf = (
            Path(tokenizer_gguf_arg).resolve()
            if tokenizer_gguf_arg
            else (
                Path(recipe_gguf).resolve()
                if recipe_gguf and TOKENIZER_GGUF_ASSET not in assets_by_name
                else None
            )
        )
        if tokenizer_gguf is not None:
            tokenizer_asset = gguf_metadata_asset(_gguf_reader(tokenizer_gguf))
            assets_by_name[tokenizer_asset.name] = tokenizer_asset
        if TOKENIZER_GGUF_ASSET not in assets_by_name:
            warnings.warn(
                "output MFQ has no embedded tokenizer; pass --tokenizer-gguf",
                stacklevel=2,
            )
        runtime_assets = list(assets_by_name.values())
        for index, asset in enumerate(runtime_assets):
            asset_path = tmp_root / f"runtime-asset-{index:02d}.blob"
            asset_path.write_bytes(asset.data)
            records.append(
                BlobRecord(asset.name, ASSET_DTYPE, len(asset.data), asset_path)
            )
        inherited_profile = (
            mfq_checkpoint.header.extra.get(RUNTIME_SAMPLING_METADATA_KEY)
            if mfq_checkpoint is not None
            else None
        )
        runtime_profile = profile_for_new_mfq(
            root if mfq_checkpoint is None else None,
            config,
            explicit_profile=getattr(args, "sampling_profile", "") or None,
            inherited_profile=(
                inherited_profile if isinstance(inherited_profile, dict) else None
            ),
        )
        header_extra = {
                "source": root.name,
                "source_format": "mfq" if mfq_checkpoint is not None else "hf",
                "policy": (
                    "mostly-BF16;1d-and-special=F32"
                    if mostly_bf16
                    else (
                        "gguf-recipe-split-qkv"
                        if recipe_gguf
                        else (
                            "minicpmo45-module-matrices=NINT-axis0,raw-parameters=source-dtype"
                            if _is_minicpmo45_config(config)
                            else "2d=NINT-axis0,other=dense"
                        )
                    )
                ),
        }
        if runtime_profile is not None:
            header_extra[RUNTIME_SAMPLING_METADATA_KEY] = runtime_profile
        header = FileHeader(
            version=2,
            model_arch=(
                f"{model_type}-hf-mfq-bf16"
                if mostly_bf16
                else (
                    f"{model_type}-full-mfq-nint-recipe"
                    if mfq_checkpoint is not None
                    else f"{model_type}-hf-mfq-nint-recipe"
                )
            ),
            num_tensors=len(records),
            extra={
                **header_extra,
                "calibration_scheme": (
                    _artifact_provenance_name(calibration_scheme_path)
                    if calibration_scheme_path
                    else None
                ),
                "imatrix": (
                    None
                    if imatrix is None
                    else {
                        "file": imatrix.path.name,
                        "entries": len(imatrix.entries),
                        "bindings": {
                            name: binding.entry_name
                            for name, binding in sorted(imatrix_bindings.items())
                        },
                        "datasets": list(imatrix.datasets),
                        "chunk_count": imatrix.chunk_count,
                        "chunk_size": imatrix.chunk_size,
                        "legacy": imatrix.legacy,
                    }
                ),
                "fused_layout": {
                    "full_attention": "qk_group,v_separate",
                    "linear_attention": "qk_group,v_separate,z_separate,ab_recipe_group",
                    "ffn": "gate_up_group,down_separate",
                } if recipe_gguf else None,
                "default_spec": {"bits": spec.bits, "groupsize": spec.groupsize, "sub_bits": spec.sub_bits},
                "recipe_specs": {
                    "Q4_0": {"bits": 4, "groupsize": 24, "sub_bits": 6},
                    "Q4_1": {"bits": 4, "groupsize": 24, "sub_bits": 6},
                    "Q4_K": {"bits": 4, "groupsize": 24, "sub_bits": 6},
                    "Q5_0": {"bits": 5, "groupsize": 28, "sub_bits": 7},
                    "Q5_1": {"bits": 5, "groupsize": 28, "sub_bits": 7},
                    "Q5_K": {"bits": 5, "groupsize": 28, "sub_bits": 7},
                    "Q6_K": {"bits": 6, "groupsize": 24, "sub_bits": 7},
                    "Q8_0": {"bits": 8, "groupsize": 48, "sub_bits": 7},
                },
                "dense_dtype": "MOSTLY_BF16" if mostly_bf16 else dense_dtype,
                "mostly_bf16": mostly_bf16,
                "text_only": bool(args.text_only),
                "mtp": {
                    "included": mtp_included,
                    "hidden_layers": mtp_config_layers if mtp_included else 0,
                    "tensor_count": len(mtp_plan_names),
                    "protected_full_precision": sorted(
                        name
                        for name in mtp_plan_names
                        if name in _MTP_PROTECTED_TENSORS
                    ),
                },
                "hf_config": config,
                ASSET_MANIFEST_KEY: runtime_asset_manifest(runtime_assets),
                "target_counts": target_counts,
                "recipe_mapping": dict(sorted(_RECIPE_TARGETS.items())),
                "tensor_precision_overrides": dict(
                    sorted(tensor_precision_overrides.items())
                ),
                "nvq_calibration": calibration_mode,
                "nvq_codebook_scope": nvq_codebook_scope,
                "nvq_codebooks": codebook_results,
                "nvq_gain_calibration": gain_results,
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
        completed = True
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
                    "elapsed_sec": round(time.time() - start_time, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        if not args.keep_temp and tmp_root.exists() and (completed or not resume_temp):
            shutil.rmtree(tmp_root)
        if mfq_checkpoint is not None:
            mfq_checkpoint.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        help="HF checkpoint directory with safetensors",
    )
    source.add_argument(
        "--input-mfq",
        default="",
        help="full-precision MFQ containing no NINT/NVQ/NPQ/NEPQ/TPQ tensors",
    )
    parser.add_argument("--output", required=True, help="output .mfq path")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--groupsize", type=int, default=24)
    parser.add_argument("--sub-bits", type=int, default=6)
    parser.add_argument("--row-chunk", type=int, default=0)
    parser.add_argument("--quant-backend", choices=QUANT_BACKENDS, default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-only", action="store_true", help="only convert language_model + lm_head tensors")
    parser.add_argument("--recipe-gguf", default="", help="GGUF file whose tensor types define the MFQ mixed-NINT recipe")
    parser.add_argument(
        "--tokenizer-gguf",
        default="",
        help="GGUF whose tokenizer, chat template, special tokens, and model metadata are embedded",
    )
    parser.add_argument(
        "--model-config",
        default="",
        help="model config JSON to embed; defaults to INPUT/config.json",
    )
    parser.add_argument("--sampling-profile", default="")
    parser.add_argument(
        "--calibration-scheme",
        default="",
        help="calibration scheme whose per-tensor NINT specs override the recipe",
    )
    parser.add_argument(
        "--imatrix",
        default="",
        help=(
            "optional llama.cpp GGUF or legacy importance matrix for "
            "HF NINT/NINTM calibration"
        ),
    )
    parser.add_argument(
        "--tensor-precision-overrides",
        default="",
        help=(
            "JSON mapping exact GGUF or HF tensor names to final MFQ dtypes"
        ),
    )
    parser.add_argument(
        "--nvq-calibration",
        choices=("auto", "none", "gain", "group24"),
        default="auto",
    )
    parser.add_argument("--nvq-group-chunk", type=int, default=32768)
    parser.add_argument("--nvq-search-steps", type=int, default=19)
    parser.add_argument(
        "--nvq-assignment", choices=("native", "torch"), default="native"
    )
    parser.add_argument("--nvq-jsc-banks", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--nvq-jsc-iterations", type=int, default=4)
    parser.add_argument("--nvq-jsc-assignment-refine-steps", type=int, default=2)
    parser.add_argument("--nvq-jsc-raw-multiplier", type=int, default=8)
    parser.add_argument("--nvq3-jsc", action="store_true")
    parser.add_argument("--nvq3-jsc-512", action="store_true")
    parser.add_argument("--nvq3-to-nint3", action="store_true")
    parser.add_argument("--iq2-s-to-nint2", action="store_true")
    parser.add_argument("--q8-to-nint8-zero", action="store_true")
    parser.add_argument("--nvq3-jsc-banks", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--nvq3-jsc-learned-scale", action="store_true")
    parser.add_argument("--nvq-jsc-row-importance", default="")
    parser.add_argument("--npq0-l", action="store_true")
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
        "--nvq1-l-assignment", choices=("native", "torch"), default="native"
    )
    parser.add_argument(
        "--nvq-codebook-scope", choices=("fixed", "tensor"), default="tensor"
    )
    parser.add_argument("--nvq-codebook-artifact-dir", default="")
    parser.add_argument("--nvq-codebook-train-rows", type=int, default=2048)
    parser.add_argument("--nvq-codebook-validation-rows", type=int, default=512)
    parser.add_argument("--nvq-codebook-row-chunk", type=int, default=512)
    parser.add_argument("--nvq-codebook-iterations", type=int, default=4)
    parser.add_argument("--nvq-codebook-projection-candidates", type=int, default=48)
    parser.add_argument("--nvq-codebook-min-improvement", type=float, default=0.0)
    parser.add_argument("--nvq-codebook-seed", type=int, default=20260716)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help=(
            "store ordinary HF matrix weights as BF16 while preserving "
            "1D/norm/special tensors as F32, matching llama.cpp MOSTLY_BF16"
        ),
    )
    parser.add_argument("--dense-dtype", choices=("f16", "f32"), default="f32", help="dense dtype for non-quantized recipe tensors")
    parser.add_argument("--limit-tensors", type=int, default=0, help="debug/smoke: convert first N planned tensors")
    parser.add_argument("--dry-run", action="store_true", help="print plan and estimated size without writing output")
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
    parser.add_argument(
        "--resume-temp",
        action="store_true",
        help="reuse complete, size-validated blobs from an interrupted temp directory",
    )
    parser.add_argument(
        "--temp-dir",
        default="",
        help="exact temporary blob directory; defaults beside the output file",
    )
    return parser


def main() -> None:
    convert(build_parser().parse_args())


if __name__ == "__main__":
    main()
