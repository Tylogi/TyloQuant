"""Qwen3.5 tensor bindings used by calibration and layerwise replay."""

from __future__ import annotations

import json
import os
import struct
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class Qwen35LinearTarget:
    name: str
    source_name: str
    module_name: str
    rows: int
    columns: int
    row_start: int
    row_end: int
    group: str
    role: str
    gguf_name: str

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError(f"invalid target shape for {self.name}: {(self.rows, self.columns)}")
        if self.row_start < 0 or self.row_end - self.row_start != self.rows:
            raise ValueError(
                f"invalid row binding for {self.name}: {self.row_start}:{self.row_end}"
            )


class HfSafetensorIndex:
    """Read individual HF tensors without materializing the full checkpoint."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_file():
            document = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = {
                str(key): str(value) for key, value in document["weight_map"].items()
            }
        else:
            shards = sorted(self.root.glob("*.safetensors"))
            if len(shards) != 1:
                raise FileNotFoundError(f"missing safetensors index under {self.root}")
            with safe_open(str(shards[0]), framework="pt", device="cpu") as reader:
                self.weight_map = {
                    str(name): shards[0].name for name in reader.keys()  # noqa: SIM118
                }
        self._shapes: dict[str, tuple[int, ...]] = {}
        self.direct_io = os.environ.get("MFQ_SAFETENSORS_DIRECT_IO", "") == "1"
        self._direct_headers: dict[str, tuple[int, dict[str, Any]]] = {}

    def _direct_header(self, shard: str) -> tuple[int, dict[str, Any]]:
        try:
            return self._direct_headers[shard]
        except KeyError:
            pass
        path = self.root / shard
        with path.open("rb", buffering=0) as stream:
            raw_size = stream.read(8)
            if len(raw_size) != 8:
                raise ValueError(f"truncated safetensors header in {path}")
            header_size = struct.unpack("<Q", raw_size)[0]
            header = json.loads(stream.read(header_size))
        result = 8 + int(header_size), header
        self._direct_headers[shard] = result
        return result

    def _direct_tensor(
        self,
        name: str,
        *,
        row_start: int | None = None,
        row_end: int | None = None,
    ) -> torch.Tensor:
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"HF checkpoint has no tensor {name!r}") from exc
        data_start, header = self._direct_header(shard)
        try:
            entry = header[name]
        except KeyError as exc:
            raise KeyError(f"safetensors shard {shard!r} has no tensor {name!r}") from exc
        dtype_map = {
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "F32": torch.float32,
            "F64": torch.float64,
            "I8": torch.int8,
            "I16": torch.int16,
            "I32": torch.int32,
            "I64": torch.int64,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        try:
            dtype = dtype_map[str(entry["dtype"])]
        except KeyError as exc:
            raise ValueError(
                f"unsupported direct safetensors dtype {entry.get('dtype')!r}"
            ) from exc
        shape = tuple(int(value) for value in entry["shape"])
        if not shape:
            start = 0
            end = 1
            output_shape: tuple[int, ...] = ()
            row_elements = 1
        else:
            start = 0 if row_start is None else int(row_start)
            end = shape[0] if row_end is None else int(row_end)
            if start < 0 or end < start or end > shape[0]:
                raise IndexError(f"invalid rows {start}:{end} for {name} with shape {shape}")
            output_shape = (end - start, *shape[1:])
            row_elements = 1
            for value in shape[1:]:
                row_elements *= value
        item_size = torch.empty((), dtype=dtype).element_size()
        element_count = (end - start) * row_elements
        byte_count = element_count * item_size
        offsets = tuple(int(value) for value in entry["data_offsets"])
        expected_bytes = item_size
        for value in shape:
            expected_bytes *= value
        if len(offsets) != 2 or offsets[1] - offsets[0] != expected_bytes:
            raise ValueError(f"invalid safetensors offsets for {name}")
        file_offset = data_start + offsets[0] + start * row_elements * item_size
        value = torch.empty(element_count, dtype=dtype, device="cpu")
        byte_view = value.view(torch.uint8).numpy().reshape(-1)
        path = self.root / shard
        with path.open("rb", buffering=0) as stream:
            stream.seek(file_offset)
            filled = 0
            while filled < byte_count:
                count = stream.readinto(byte_view[filled:])
                if not count:
                    raise EOFError(
                        f"truncated safetensors data for {name}: "
                        f"{filled} of {byte_count} bytes"
                    )
                filled += count
        return value.reshape(output_shape)

    def names(self) -> tuple[str, ...]:
        return tuple(self.weight_map)

    def shape(self, name: str) -> tuple[int, ...]:
        try:
            return self._shapes[name]
        except KeyError:
            pass
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"HF checkpoint has no tensor {name!r}") from exc
        if self.direct_io:
            _data_start, header = self._direct_header(shard)
            shape = tuple(int(value) for value in header[name]["shape"])
        else:
            with safe_open(str(self.root / shard), framework="pt", device="cpu") as reader:
                shape = tuple(int(value) for value in reader.get_slice(name).get_shape())
        self._shapes[name] = shape
        return shape

    def tensor(
        self,
        name: str,
        *,
        row_start: int | None = None,
        row_end: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"HF checkpoint has no tensor {name!r}") from exc
        if self.direct_io:
            value = self._direct_tensor(
                name,
                row_start=row_start,
                row_end=row_end,
            )
        else:
            with safe_open(str(self.root / shard), framework="pt", device="cpu") as reader:
                if row_start is None and row_end is None:
                    value = reader.get_tensor(name)
                else:
                    start = 0 if row_start is None else int(row_start)
                    end = self.shape(name)[0] if row_end is None else int(row_end)
                    value = reader.get_slice(name)[start:end]
        if dtype is not None or str(device) != "cpu":
            value = value.to(device=device, dtype=dtype or value.dtype)
        return value.contiguous()

    def layer_state(
        self,
        layer_index: int,
        *,
        exclude: set[str] | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        prefix = f"model.language_model.layers.{layer_index}."
        excluded = exclude or set()
        by_shard: dict[str, list[str]] = defaultdict(list)
        for name, shard in self.weight_map.items():
            if name.startswith(prefix) and name not in excluded:
                by_shard[shard].append(name)
        for shard, names in sorted(by_shard.items()):
            if self.direct_io:
                for name in sorted(names):
                    yield name[len(prefix) :], self._direct_tensor(name)
            else:
                with safe_open(str(self.root / shard), framework="pt", device="cpu") as reader:
                    for name in sorted(names):
                        yield name[len(prefix) :], reader.get_tensor(name)


def qwen35_head_weight_name(index: HfSafetensorIndex) -> str:
    """Resolve an explicit or tied Qwen3.5 output projection."""

    if "lm_head.weight" in index.weight_map:
        return "lm_head.weight"
    tied = "model.language_model.embed_tokens.weight"
    if tied in index.weight_map:
        return tied
    raise KeyError("Qwen3.5 checkpoint has neither lm_head nor tied embedding weights")


def _text_config(model_root: Path) -> dict[str, Any]:
    document = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    return dict(document.get("text_config", document))


def _module_name(source_name: str) -> str:
    prefix = "model.language_model."
    if not source_name.startswith(prefix) or not source_name.endswith(".weight"):
        raise ValueError(f"cannot map HF source tensor to a module: {source_name}")
    return "model." + source_name[len(prefix) : -len(".weight")]


def _gguf_name(layer: int, role: str) -> str:
    suffixes = {
        "attn_q": "attn_q.weight",
        "attn_k": "attn_k.weight",
        "attn_v": "attn_v.weight",
        "attn_out": "attn_output.weight",
        "linear_qk": "attn_qk.weight",
        "linear_v": "attn_v.weight",
        "linear_z": "attn_gate.weight",
        "linear_out": "ssm_out.weight",
        "ffn_gate": "ffn_gate.weight",
        "ffn_up": "ffn_up.weight",
        "ffn_down": "ffn_down.weight",
    }
    return f"blk.{layer}.{suffixes[role]}"


def qwen35_linear_targets(model_root: str | Path) -> tuple[Qwen35LinearTarget, ...]:
    root = Path(model_root).resolve()
    config = _text_config(root)
    index = HfSafetensorIndex(root)
    layer_types = tuple(str(value) for value in config["layer_types"])
    targets: list[Qwen35LinearTarget] = []

    def add(
        layer: int,
        suffix: str,
        group_suffix: str,
        role: str,
        *,
        target_suffix: str | None = None,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> None:
        source = f"model.language_model.layers.{layer}.{suffix}.weight"
        shape = index.shape(source)
        if len(shape) != 2:
            raise ValueError(f"calibration target is not a matrix: {source} {shape}")
        end = shape[0] if row_end is None else int(row_end)
        if end > shape[0] or row_start >= end:
            raise ValueError(f"invalid target rows for {source}: {row_start}:{end} of {shape[0]}")
        name = (
            source
            if target_suffix is None
            else f"model.language_model.layers.{layer}.{target_suffix}.weight"
        )
        targets.append(
            Qwen35LinearTarget(
                name=name,
                source_name=source,
                module_name=_module_name(source),
                rows=end - row_start,
                columns=shape[1],
                row_start=row_start,
                row_end=end,
                group=f"layer.{layer}.{group_suffix}",
                role=role,
                gguf_name=_gguf_name(layer, role),
            )
        )

    for layer, layer_type in enumerate(layer_types):
        if layer_type == "full_attention":
            add(layer, "self_attn.q_proj", "attn_qk", "attn_q")
            add(layer, "self_attn.k_proj", "attn_qk", "attn_k")
            add(layer, "self_attn.v_proj", "attn_v", "attn_v")
            add(layer, "self_attn.o_proj", "attn_out", "attn_out")
        elif layer_type == "linear_attention":
            qkv = f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight"
            qkv_shape = index.shape(qkv)
            key_rows = int(config["linear_num_key_heads"]) * int(config["linear_key_head_dim"])
            value_rows = int(config["linear_num_value_heads"]) * int(
                config["linear_value_head_dim"]
            )
            qk_end = 2 * key_rows
            v_end = qk_end + value_rows
            if qkv_shape[0] != v_end:
                raise ValueError(
                    f"linear qkv shape disagrees with config at layer {layer}: "
                    f"{qkv_shape[0]} != {v_end}"
                )
            add(
                layer,
                "linear_attn.in_proj_qkv",
                "linear_qk",
                "linear_qk",
                target_suffix="linear_attn.in_proj_qk",
                row_start=0,
                row_end=qk_end,
            )
            add(
                layer,
                "linear_attn.in_proj_qkv",
                "linear_v",
                "linear_v",
                target_suffix="linear_attn.in_proj_v",
                row_start=qk_end,
                row_end=v_end,
            )
            add(layer, "linear_attn.in_proj_z", "linear_z", "linear_z")
            add(layer, "linear_attn.out_proj", "linear_out", "linear_out")
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type: {layer_type!r}")

        add(layer, "mlp.gate_proj", "ffn_gate_up", "ffn_gate")
        add(layer, "mlp.up_proj", "ffn_gate_up", "ffn_up")
        add(layer, "mlp.down_proj", "ffn_down", "ffn_down")

    if len({target.name for target in targets}) != len(targets):
        raise RuntimeError("Qwen3.5 calibration target names are not unique")
    return tuple(targets)


__all__ = [
    "HfSafetensorIndex",
    "Qwen35LinearTarget",
    "qwen35_head_weight_name",
    "qwen35_linear_targets",
]
