"""Convert a HF safetensors checkpoint to MFQ.

Default policy for the first Qwen3.5/3.6 smoke:
- all 2D tensors -> NINT4, axis=0
- all other tensors -> dense F16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
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
from mfq.formats.tpq import (
    cccp_pq_payload_nbytes,
    pack_cccp_indices,
    pack_cccp_pq_prefix,
)
from mfq.formats.tpq import TPQ_PQ_SPECS_BY_LABEL
from mfq.formats.assets import (
    ASSET_DTYPE,
    ASSET_MANIFEST_KEY,
    gguf_metadata_asset,
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
from mfq.formats.nepq import (
    NEPQ0_L,
    NEPQ0_S,
    NEPQ1_L,
    NEPQ1_S,
    _FLAG_ROTATED as _NEPQ_FLAG_ROTATED,
    _HEADER as _NEPQ_HEADER,
    _MAGIC as _NEPQ_MAGIC,
    _VERSION as _NEPQ_VERSION,
    _pack_bits as _pack_nepq_bits,
    validate_nepq,
)
from mfq.formats.nint import NINT2_SPEC, NintSpec
from mfq.formats.npq0_l import (
    NPQ0_L,
    _HEADER as _NPQ0_L_HEADER,
    _MAGIC as _NPQ0_L_MAGIC,
    _VERSION as _NPQ0_L_VERSION,
    _pack_bits as _pack_npq0_l_bits,
    pack_npq0_l_tables,
)
from mfq.formats.npq0_s import (
    NPQ0_S,
    _HEADER as _NPQ0_S_HEADER,
    _MAGIC as _NPQ0_S_MAGIC,
    _VERSION as _NPQ0_S_VERSION,
    _pack_bits as _pack_npq0_s_bits,
    pack_npq0_s_tables,
)
from mfq.formats.nvq import (
    NVQ2_E8,
    NVQ2_E8_1024,
    NVQ2_E8_4096,
    NVQ3_D4,
    NVQ3_D4_512,
    NVQ3_D4_1024,
    _CODEBOOK_ID,
    _CUSTOM_CODEBOOK_FLAG,
    _HEADER as _NVQ_HEADER,
    _INDEX_PARITY_FLAG,
    _JSC_FLAG,
    _MAGIC as _NVQ_MAGIC,
    _pack_bits as _pack_nvq_bits,
    pack_codebook,
    pack_jsc_tables,
)
from mfq.formats.nvq1_l import (
    NVQ1_L_T8_S3,
    _HEADER as _NVQ1_L_HEADER,
    _MAGIC as _NVQ1_L_MAGIC,
    _PROFILE_CUSTOM_TERNARY,
    _PROFILE_IQ1S_GRID,
    _pack_bits as _pack_nvq1_l_bits,
    pack_ternary_codebook,
)
from mfq.formats.nvq1_s import (
    NVQ1_S,
    NVQ1_S_SYNTHETIC_BANKS,
    _HEADER as _NVQ1_S_HEADER,
    _MAGIC as _NVQ1_S_MAGIC,
    _VERSION as _NVQ1_S_VERSION,
    pack_nvq1_s_banked_codebook,
)
from mfq.formats.shards import (
    matching_shard_paths,
    parse_size,
    validate_split_limits,
    write_blob_record_shards,
)
from mfq.quantize.expert_nint import (
    quantize_flat_cohort,
    resolve_precision_artifact,
)
from mfq.quantize.tpq import quantize_tpq_pq_fixed
from mfq.quantize.nepq import NepqQuantConfig, quantize_nepq_fixed
from mfq.quantize.nint_quant import quantize as nint_quantize
from mfq.quantize.nint_quant_torch import quantize_axis0 as nint_quantize_axis0_torch
from mfq.quantize.npq0_l import Npq0LTables
from mfq.quantize.npq0_s import Npq0STables
from mfq.quantize.nvq_jsc import NvqJscTables


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
        start: int,
        end: int,
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
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


_RECIPE_SPECS = {
    "Q2_K": NINT2_SPEC,
    "Q3_K": NintSpec(3, 24, 5),
    "Q4_K": NintSpec(4, 24, 6),
    "Q5_0": NintSpec(5, 28, 7),
    "Q5_1": NintSpec(5, 28, 7),
    "Q5_K": NintSpec(5, 28, 7),
    "Q6_K": NintSpec(6, 24, 7),
    "Q8_0": NintSpec(8, 48, 7),
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
_NEPQ_SPECS = {
    "NEPQ0-S": NEPQ0_S,
    "NEPQ0-L": NEPQ0_L,
    "NEPQ1-S": NEPQ1_S,
    "NEPQ1-L": NEPQ1_L,
}


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


def _hf_to_gguf_name(name: str) -> str | None:
    if name == "lm_head.weight":
        return "output.weight"
    if name == "model.language_model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.language_model.norm.weight":
        return "output_norm.weight"
    if name == "mtp.fc.weight":
        return "blk.40.nextn.eh_proj.weight"
    if name == "mtp.pre_fc_norm_embedding.weight":
        return "blk.40.nextn.enorm.weight"
    if name == "mtp.pre_fc_norm_hidden.weight":
        return "blk.40.nextn.hnorm.weight"
    if name == "mtp.norm.weight":
        return "blk.40.nextn.shared_head_norm.weight"

    prefix = "model.language_model.layers."
    if name.startswith(prefix):
        rest = name[len(prefix):]
        layer, _, suffix = rest.partition(".")
        mapped = _LAYER_NAME_MAP.get(suffix)
        return f"blk.{layer}.{mapped}" if mapped is not None else None

    mtp_prefix = "mtp.layers.0."
    if name.startswith(mtp_prefix):
        suffix = name[len(mtp_prefix):]
        mapped = _LAYER_NAME_MAP.get(suffix)
        return f"blk.40.{mapped}" if mapped is not None else None
    return None


def _dtype_for_recipe_type(gguf_type: str, dense_dtype: str) -> str:
    if gguf_type in _RECIPE_SPECS:
        return f"NINT{_RECIPE_SPECS[gguf_type].bits}"
    if gguf_type == "F32":
        return dense_dtype
    if gguf_type in {"F16", "BF16"}:
        return "F16"
    raise ValueError(f"unsupported GGUF recipe tensor type: {gguf_type}")


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
) -> tuple[str, str | None, str | None]:
    anchor = _recipe_group_anchor(name) if recipe_types is not None else None
    anchor_gguf_name = _hf_to_gguf_name(anchor) if anchor is not None else None
    if anchor_gguf_name is not None and anchor_gguf_name in recipe_types:
        anchor_type = recipe_types[anchor_gguf_name]
        return _dtype_for_recipe_type(anchor_type, dense_dtype), anchor_gguf_name, anchor_type
    if gguf_type is not None:
        return _dtype_for_recipe_type(gguf_type, dense_dtype), None, gguf_type
    return dense_dtype, None, None


def _linear_attn_qkv_split(root: Path) -> tuple[int, int] | None:
    cfg_path = root / "config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    text = cfg.get("text_config", cfg)
    nk = int(text.get("linear_num_key_heads", text.get("num_key_value_heads", 0)))
    nv = int(text.get("linear_num_value_heads", text.get("num_attention_heads", 0)))
    dk = int(text.get("linear_key_head_dim", text.get("head_dim", 0)))
    dv = int(text.get("linear_value_head_dim", text.get("head_dim", 0)))
    if nk <= 0 or nv <= 0 or dk <= 0 or dv <= 0:
        return None
    ksz = nk * dk
    vsz = nv * dv
    return 2 * ksz, 2 * ksz + vsz


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
) -> list[TensorPlan]:
    weight_map = _read_index(root)
    config_path = root / "config.json"
    raw_config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    model_config = raw_config.get("text_config", raw_config)
    is_glm_dsa = model_config.get("model_type") == "glm_moe_dsa"
    if is_glm_dsa and recipe_types is not None:
        raise ValueError(
            "GLM DSA GGUF recipe mapping is not implemented; use a calibration scheme"
        )
    glm_layers = int(model_config.get("num_hidden_layers", 0)) if is_glm_dsa else 0
    linear_qkv_split = (
        _linear_attn_qkv_split(root)
        if recipe_types is not None or calibration_scheme is not None
        else None
    )
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if text_only and not (
            name.startswith("model.language_model.")
            or (is_glm_dsa and name.startswith("model."))
            or name == "lm_head.weight"
        ):
            continue
        if is_glm_dsa:
            layer_index = _glm_layer_index(name)
            if layer_index is not None and layer_index >= glm_layers:
                continue
        if recipe_types is not None:
            gguf_name = _hf_to_gguf_name(name)
            if gguf_name is None or gguf_name not in recipe_types:
                continue
        by_shard.setdefault(shard, []).append(name)

    out: list[TensorPlan] = []
    source_shapes: dict[str, tuple[int, ...]] = {}
    source_dtypes: dict[str, str] = {}
    for shard in sorted(by_shard):
        with safe_open(str(root / shard), framework="pt", device="cpu") as f:
            for name in sorted(by_shard[shard]):
                sl = f.get_slice(name)
                shape = tuple(int(v) for v in sl.get_shape())
                if is_glm_dsa and (
                    name.endswith(".self_attn.kv_b_proj.weight")
                    or re.match(
                        r"model\.layers\.\d+\.mlp\.experts\.\d+\."
                        r"(?:gate_proj|up_proj|down_proj)\.weight$",
                        name,
                    )
                ):
                    source_shapes[name] = shape
                    source_dtypes[name] = str(sl.get_dtype())
                    continue
                gguf_name = _hf_to_gguf_name(name) if recipe_types is not None else None
                gguf_type = recipe_types[gguf_name] if gguf_name is not None and recipe_types is not None else None
                if recipe_types is not None:
                    target, anchor_gguf_name, anchor_type = _target_for_recipe_name(name, gguf_type, recipe_types, dense_dtype)
                    if anchor_gguf_name is not None:
                        gguf_name = anchor_gguf_name
                        gguf_type = anchor_type
                else:
                    target = "NINT4" if len(shape) == 2 else dense_dtype
                    if is_glm_dsa and name.endswith(".mlp.gate.weight"):
                        target = "F16"
                if target.startswith("NINT") and len(shape) not in (2, 3):
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
                            str(sl.get_dtype()),
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
                            str(sl.get_dtype()),
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
                elif target.startswith("NINT") and len(shape) == 3:
                    expert_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
                    homogeneous_spec = _spec_for_target(target, NintSpec())
                    expert_precisions = (
                        nint_expert_precision(homogeneous_spec),
                    ) * expert_shape[0]
                    target = "NINTM"
                out.append(
                    TensorPlan(
                        name,
                        shard,
                        shape,
                        str(sl.get_dtype()),
                        target,
                        gguf_name,
                        gguf_type,
                        target_spec=selection.spec if selection else None,
                        expert_shape=expert_shape,
                        expert_precisions=expert_precisions,
                    )
                )
    if is_glm_dsa:
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
    elif dtype == "I32":
        arr = t.to(torch.int32).contiguous().cpu().numpy()
        arr = np.ascontiguousarray(arr, dtype=np.int32)
    elif dtype == "I64":
        arr = t.to(torch.int64).contiguous().cpu().numpy()
        arr = np.ascontiguousarray(arr, dtype=np.int64)
    else:
        raise ValueError(f"unsupported dense target dtype: {dtype}")
    dtype = _DENSE_NAMES.get(arr.dtype)
    if dtype is None:
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
            if quant_backend == "cuda" and hasattr(sl, "read_rows"):
                chunk = sl.read_rows(start, end, device=device)
            else:
                chunk = sl[start:end]
            if quant_backend == "cuda":
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
    if quant_backend == "cuda" and family in {"NVQ1-L", "NVQ2", "NVQ3"}:
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
    if quant_backend == "cuda" and family in {
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
    if quant_backend == "cuda" and family in TPQ_PQ_SPECS_BY_LABEL:
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
    array = weight_tensor.to(torch.float32).cpu().numpy()
    return quantize_flat_cohort(
        array,
        precision,
        artifact=artifact,
        importance=importance,
        device=device if quant_backend == "cuda" else "cpu",
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
        header = pack_cccp_pq_prefix(spec, shape, codebook)
        table_payload = b""
        stream_bits = (
            (neuron_len // spec.vector_size) * spec.index_bits,
        )
        packer = pack_cccp_indices
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
            table_payload = pack_jsc_tables(
                artifact.scale_lut,
                artifact.bank_for_state,
                artifact.codebooks,
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
        and quant_backend == "cuda"
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
                if quant_backend == "cuda" and hasattr(source, "read_rows")
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
    tensor = quantize_nepq_fixed(
        source,
        precision.family,
        tables,
        importance=importance,
        rotation_block=int(precision.option("rotation_block", 0)),
        rotation_seed=int(precision.option("rotation_seed", 0)),
        config=NepqQuantConfig(
            anchor_multipliers=anchor_multipliers,
            refine_steps=int(precision.option("refine_steps", 2)),
            row_chunk=int(precision.option("row_chunk", 8)),
            bank_chunk=int(precision.option("bank_chunk", 8)),
            admm_iterations=int(precision.option("admm_iterations", 0)),
            admm_rho=float(precision.option("admm_rho", 1.0)),
        ),
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
            payload_nbytes = cccp_pq_payload_nbytes(
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
                + spec.payload_nbytes(rows, columns)
            )
        elif precision.family in _NEPQ_SPECS:
            if artifact is None:
                raise ValueError(
                    f"{precision.family} size estimation requires its table artifact"
                )
            tables = np.asarray(artifact)
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
            if int(precision.option("rotation_block", 0)):
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
        elif item.target_dtype.startswith("NINT"):
            item_spec = _spec_for_plan(item, spec)
            nint_total += _nint_blob_nbytes(item.shape[0], item.shape[1], item_spec)
        else:
            item_size = {"F16": 2, "F32": 4, "I32": 4, "I64": 8}[
                item.target_dtype
            ]
            dense_total += 4 + 8 * len(item.shape) + n * item_size
    return nint_total, dense_total


def _plan_blob_nbytes(
    item: TensorPlan,
    spec: NintSpec,
    artifact_root: str | Path | None = None,
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
    if item.target_dtype.startswith("NINT"):
        item_spec = _spec_for_plan(item, spec)
        return _nint_blob_nbytes(item.shape[0], item.shape[1], item_spec)
    item_size = {"F16": 2, "F32": 4, "I32": 4, "I64": 8}[
        item.target_dtype
    ]
    return 4 + 8 * len(item.shape) + n * item_size


def convert(args: argparse.Namespace) -> None:
    root = Path(args.input).resolve()
    output = Path(args.output).resolve()
    split_max_size = int(getattr(args, "split_max_size", 0))
    split_max_tensors = int(getattr(args, "split_max_tensors", 0))
    validate_split_limits(split_max_size, split_max_tensors)
    if (output.exists() or matching_shard_paths(output)) and not args.overwrite:
        raise FileExistsError(f"output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    spec = NintSpec(bits=args.bits, groupsize=args.groupsize, sub_bits=args.sub_bits)
    quant_backend = args.quant_backend
    if quant_backend == "auto":
        quant_backend = "cuda" if torch.cuda.is_available() else "cpu"
    if quant_backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA quant backend requested but torch.cuda.is_available() is false")
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
    plan = _plan(
        root,
        text_only=args.text_only,
        recipe_types=recipe_types,
        dense_dtype=dense_dtype,
        calibration_scheme=calibration_scheme,
    )
    if args.limit_tensors:
        plan = plan[: args.limit_tensors]
    nint_est, dense_est = _estimate_bytes(plan, spec, artifact_root)
    total_src = sum(int(np.prod(p.shape)) * 2 for p in plan)
    target_counts: dict[str, int] = {}
    target_bytes_est: dict[str, int] = {}
    for p in plan:
        target_counts[p.target_dtype] = target_counts.get(p.target_dtype, 0) + 1
        nb = _plan_blob_nbytes(p, spec, artifact_root)
        target_bytes_est[p.target_dtype] = target_bytes_est.get(p.target_dtype, 0) + nb
    print(
        json.dumps(
            {
                "input": str(root),
                "output": str(output),
                "tensors": len(plan),
                "target_counts": target_counts,
                "target_estimated_gb": {k: round(v / 1e9, 3) for k, v in sorted(target_bytes_est.items())},
                "source_bf16_gb": round(total_src / 1e9, 3),
                "estimated_mfq_gb": round((nint_est + dense_est) / 1e9, 3),
                "default_spec": {"bits": spec.bits, "groupsize": spec.groupsize, "sub_bits": spec.sub_bits},
                "recipe": str(Path(recipe_gguf).resolve()) if recipe_gguf else None,
                "calibration_scheme": (
                    str(Path(calibration_scheme_path).resolve()) if calibration_scheme_path else None
                ),
                "quant_backend": quant_backend,
                "device": args.device if quant_backend == "cuda" else "cpu",
                "row_chunk": args.row_chunk,
                "text_only": args.text_only,
                "dense_dtype": dense_dtype,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if getattr(args, "dry_run", False):
        return

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
                expected_nbytes = _plan_blob_nbytes(item, spec, artifact_root)
                if resume_temp and blob_path.is_file() and blob_path.stat().st_size == expected_nbytes:
                    records.append(
                        BlobRecord(item.name, item.target_dtype, expected_nbytes, blob_path)
                    )
                    print(
                        json.dumps(
                            {
                                "done": done,
                                "total": len(plan),
                                "name": item.name,
                                "dtype": item.target_dtype,
                                "blob_mb": round(expected_nbytes / 1e6, 2),
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
                    source = _GlmExpertRowSource(
                        root,
                        item.expert_shape,
                        item.expert_source_names,
                        item.expert_source_shards,
                    )
                    try:
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
                            args.row_chunk,
                            quant_backend,
                            args.device,
                            artifact_root,
                        )
                    finally:
                        source.close()
                    del source
                else:
                    source_name = item.source_name or item.name
                    raw_source = _RawSafeTensorSlice(root / shard, source_name)
                    if item.target_dtype == "NINTM":
                        if item.expert_shape is None or item.expert_precisions is None:
                            raise ValueError(f"NINTM plan lacks expert metadata: {item.name}")
                        if item.transform is not None:
                            source = _transform_glm_kv_b(
                                raw_source.tensor(), item
                            )
                        else:
                            source = raw_source
                        nbytes = _write_mixed_moe_axis0_blob(
                            source,
                            item.shape,
                            item.expert_shape,
                            item.expert_precisions,
                            blob_path,
                            args.row_chunk,
                            quant_backend,
                            args.device,
                            artifact_root,
                        )
                    elif item.target_dtype.startswith("NINT"):
                        item_spec = _spec_for_plan(item, spec)
                        source = raw_source
                        if item.row_start is not None or item.row_end is not None:
                            start = 0 if item.row_start is None else item.row_start
                            end = item.shape[0] if item.row_end is None else item.row_end
                            source = source[start:end]
                        nbytes = _write_nint_axis0_blob(
                            source,
                            item.shape,
                            item_spec,
                            blob_path,
                            args.row_chunk,
                            quant_backend,
                            args.device,
                        )
                    else:
                        source = raw_source.tensor()
                        if item.row_start is not None or item.row_end is not None:
                            source = source[item.row_start:item.row_end]
                        nbytes = _dense_blob_from_tensor(source, blob_path, item.target_dtype)
                    del source, raw_source
                if nbytes != expected_nbytes:
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
        config_path = (
            Path(config_path_arg).resolve()
            if config_path_arg
            else root / "config.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        model_type = str(config.get("model_type", "unknown"))
        runtime_assets = []
        if config:
            runtime_assets.append(model_config_asset(config))
        tokenizer_gguf_arg = getattr(args, "tokenizer_gguf", "")
        tokenizer_gguf = (
            Path(tokenizer_gguf_arg).resolve()
            if tokenizer_gguf_arg
            else (Path(recipe_gguf).resolve() if recipe_gguf else None)
        )
        if tokenizer_gguf is not None:
            runtime_assets.append(
                gguf_metadata_asset(_gguf_reader(tokenizer_gguf))
            )
        else:
            warnings.warn(
                "output MFQ has no embedded tokenizer; pass --tokenizer-gguf",
                stacklevel=2,
            )
        for index, asset in enumerate(runtime_assets):
            asset_path = tmp_root / f"runtime-asset-{index:02d}.blob"
            asset_path.write_bytes(asset.data)
            records.append(
                BlobRecord(asset.name, ASSET_DTYPE, len(asset.data), asset_path)
            )
        header = FileHeader(
            version=2,
            model_arch=f"{model_type}-hf-mfq-nint-recipe",
            num_tensors=len(records),
            extra={
                "source": str(root),
                "policy": "gguf-recipe-split-qkv" if recipe_gguf else "2d=NINT-axis0,other=dense",
                "recipe": str(Path(recipe_gguf).resolve()) if recipe_gguf else None,
                "calibration_scheme": (
                    str(Path(calibration_scheme_path).resolve()) if calibration_scheme_path else None
                ),
                "fused_layout": {
                    "full_attention": "qk_group,v_separate",
                    "linear_attention": "qk_group,v_separate,z_separate,ab_recipe_group",
                    "ffn": "gate_up_group,down_separate",
                } if recipe_gguf else None,
                "default_spec": {"bits": spec.bits, "groupsize": spec.groupsize, "sub_bits": spec.sub_bits},
                "recipe_specs": {
                    "Q4_K": {"bits": 4, "groupsize": 24, "sub_bits": 6},
                    "Q5_0": {"bits": 5, "groupsize": 28, "sub_bits": 7},
                    "Q5_1": {"bits": 5, "groupsize": 28, "sub_bits": 7},
                    "Q5_K": {"bits": 5, "groupsize": 28, "sub_bits": 7},
                    "Q6_K": {"bits": 6, "groupsize": 24, "sub_bits": 7},
                    "Q8_0": {"bits": 8, "groupsize": 48, "sub_bits": 7},
                },
                "dense_dtype": dense_dtype,
                "quant_backend": quant_backend,
                "device": args.device if quant_backend == "cuda" else "cpu",
                "hf_config": config,
                ASSET_MANIFEST_KEY: runtime_asset_manifest(runtime_assets),
                "target_counts": target_counts,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HF checkpoint directory with safetensors")
    parser.add_argument("--output", required=True, help="output .mfq path")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--groupsize", type=int, default=24)
    parser.add_argument("--sub-bits", type=int, default=6)
    parser.add_argument("--row-chunk", type=int, default=1024)
    parser.add_argument("--quant-backend", choices=("auto", "cuda", "cpu"), default="cuda")
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
    parser.add_argument(
        "--calibration-scheme",
        default="",
        help="calibration scheme whose per-tensor NINT specs override the recipe",
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
