"""Stream a TPQ model directory into one native MFQ file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from safetensors import safe_open

from mfq.formats.tpq import (
    TpqPqSpec,
    tpq_int4_payload_nbytes,
    tpq_pq_payload_nbytes,
    pack_tpq_indices,
    pack_tpq_int4_prefix,
    pack_tpq_pq_prefix,
)
from mfq.formats.header import MFQ_MAGIC, FileHeader
from mfq.formats.io import (
    _NINT_MOE_HDR,
    _NINT_MOE_MAGIC_V2,
    _NINT_MOE_POOL_V2_HDR,
    _u32,
)
from mfq.formats.mx import MXFP8_DTYPE, mx_header_bytes

_TIER_ORDER = ("vv", "v", "w", "x")
_DROP_TIER = "drop"
_POOL_ORDER = (*_TIER_ORDER, _DROP_TIER)
_DENSE_ITEMSIZE = {
    "F16": 2,
    "F32": 4,
    "I32": 4,
    "I64": 8,
}


def _dense_paths(root: Path, manifest: dict) -> tuple[Path, ...]:
    dense_files = manifest.get("dense_files")
    if dense_files is None:
        return (root / str(manifest.get("dense_file", "")),)
    dense_name = str(
        (manifest.get("nonexpert") or {}).get("path", "dense")
    ).strip("/\\")
    normalized = [str(name).replace("\\", "/") for name in dense_files]
    prefixed = bool(dense_name) and all(
        name.startswith(dense_name.replace("\\", "/") + "/")
        for name in normalized
    )
    base = root if prefixed else root / dense_name
    return tuple(base / name for name in normalized)


@dataclass(frozen=True)
class _StreamRecord:
    name: str
    dtype: str
    nbytes: int
    write: Callable[[BinaryIO], None]


def _manifest_path(root: Path) -> Path:
    return root / "tpq.json"


def _manifest(root: Path) -> dict:
    path = _manifest_path(root)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != "tpq-1":
        raise ValueError(
            f"unsupported TPQ artifact format: {document.get('format')!r}"
        )
    config = document.get("config")
    quant = document.get("quant")
    expert_files = document.get("expert_files")
    routed_layers = (
        (document.get("routed_experts") or {}).get("layer_files") or {}
    )
    if expert_files is None and routed_layers:
        expert_files = {
            str(layer): str(item["path"])
            for layer, item in routed_layers.items()
        }
        document["expert_files"] = expert_files
    if not isinstance(config, dict) or not isinstance(quant, dict):
        raise ValueError("TPQ manifest lacks config or quant metadata")
    if not isinstance(expert_files, dict) or not expert_files:
        raise ValueError("TPQ manifest has no expert shards")
    files = [
        *_dense_paths(root, document),
        *(root / str(name) for name in expert_files.values()),
    ]
    missing = [str(path) for path in files if not path.name or not path.is_file()]
    if missing:
        raise FileNotFoundError(f"TPQ artifact is incomplete: {missing[:8]}")
    return document


def _manifest_sha256(root: Path) -> str:
    return hashlib.sha256(_manifest_path(root).read_bytes()).hexdigest()


def _projection_metadata(
    manifest: dict,
) -> tuple[
    dict[int, dict[str, str | tuple[str, ...]]],
    dict[str, dict],
] | None:
    quant = manifest["quant"]
    if quant.get("method") != "projection-vq":
        return None
    routed_layers = (
        (manifest.get("routed_experts") or {}).get("layer_files") or {}
    )
    heterogeneous = quant.get("heterogeneous_expert_tiering") or {}
    precision_levels = heterogeneous.get("precision_levels") or {}
    layer_levels = heterogeneous.get("layer_expert_levels") or {}
    if quant.get("layouts") and precision_levels and layer_levels:
        n_experts = int(manifest["config"]["n_experts"])
        assignments = {}
        for raw_layer, raw_levels in layer_levels.items():
            levels = tuple(str(value) for value in raw_levels)
            if len(levels) != n_experts:
                raise ValueError(
                    f"TPQ projection layout L{raw_layer} has "
                    f"{len(levels)} expert levels, expected {n_experts}"
                )
            unknown_levels = sorted(set(levels).difference(precision_levels))
            if unknown_levels:
                raise ValueError(
                    f"TPQ projection layout L{raw_layer} uses unknown "
                    f"precision levels: {unknown_levels[:8]}"
                )
            assignments[int(raw_layer)] = {
                projection: tuple(
                    str(precision_levels[level][projection])
                    for level in levels
                )
                for projection in ("gate", "up", "down")
            }
        specs = quant.get("layouts") or {}
    elif routed_layers:
        assignments = {
            int(layer): {
                str(projection): str(layout)
                for projection, layout in item["projection_layout"].items()
            }
            for layer, item in routed_layers.items()
        }
        specs = quant.get("projection_layouts") or {}
    else:
        raw_assignments = quant.get("projection_layouts") or {}
        if raw_assignments:
            assignments = {
                int(layer): {
                    str(projection): str(layout)
                    for projection, layout in value.items()
                }
                for layer, value in raw_assignments.items()
            }
        else:
            assignments = {}
        specs = quant.get("layouts") or {}
    if not assignments or not specs:
        raise ValueError("TPQ projection-VQ manifest lacks layouts")
    required = {"gate", "up", "down"}
    n_experts = int(manifest["config"]["n_experts"])
    expected_layers = {int(layer) for layer in manifest["expert_files"]}
    if set(assignments) != expected_layers:
        raise ValueError(
            "TPQ projection layout layers differ from expert shards: "
            f"{sorted(assignments)} != {sorted(expected_layers)}"
        )
    for layer, value in assignments.items():
        if set(value) != required:
            raise ValueError(
                f"TPQ projection layout L{layer} must define gate/up/down"
            )
        missing = required.difference(value)
        if missing:
            raise ValueError(f"TPQ projection layout L{layer} lacks {missing}")
        used_layouts: set[str] = set()
        for projection, layout in value.items():
            if isinstance(layout, str):
                used_layouts.add(layout)
                continue
            if len(layout) != n_experts:
                raise ValueError(
                    f"TPQ projection layout L{layer}/{projection} has "
                    f"{len(layout)} experts, expected {n_experts}"
                )
            used_layouts.update(layout)
        unknown = sorted(used_layouts.difference(specs))
        if unknown:
            raise ValueError(f"TPQ projection layout L{layer} is unknown: {unknown}")
    return assignments, {str(name): dict(value) for name, value in specs.items()}


def _manifest_specs(manifest: dict) -> dict[str, TpqPqSpec]:
    raw = manifest["quant"].get("vq")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("TPQ manifest has no VQ tier definitions")
    result: dict[str, TpqPqSpec] = {}
    for tier, values in raw.items():
        if tier not in _TIER_ORDER:
            continue
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 2
        ):
            raise ValueError(f"invalid TPQ VQ definition for {tier}: {values}")
        result[tier] = TpqPqSpec(
            tier=tier,
            vector_size=int(values[0]),
            codebook_entries=int(values[1]),
        )
    if not result:
        raise ValueError("TPQ manifest defines no supported VQ tiers")
    return result


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy()


def _slice_shape(handle, name: str) -> tuple[int, ...]:
    return tuple(int(value) for value in handle.get_slice(name).get_shape())


def _slice_dtype(handle, name: str) -> str:
    return str(handle.get_slice(name).get_dtype())


def _stream_slice(
    handle,
    name: str,
    output: BinaryIO,
    *,
    target_dtype: np.dtype,
    row_chunk: int,
) -> None:
    source = handle.get_slice(name)
    shape = tuple(int(value) for value in source.get_shape())
    if not shape:
        value = _numpy(handle.get_tensor(name)).astype(target_dtype, copy=False)
        output.write(value.tobytes())
        return
    rows = shape[0]
    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)
        value = _numpy(source[start:end]).astype(target_dtype, copy=False)
        output.write(np.ascontiguousarray(value).tobytes())


def _stream_byte_slice(
    handle,
    name: str,
    output: BinaryIO,
    *,
    row_chunk: int,
) -> None:
    source = handle.get_slice(name)
    shape = tuple(int(value) for value in source.get_shape())
    if not shape:
        value = source[:].detach().cpu().contiguous().view(torch.uint8)
        output.write(value.numpy().tobytes())
        return
    for start in range(0, shape[0], row_chunk):
        end = min(start + row_chunk, shape[0])
        value = source[start:end].detach().cpu().contiguous().view(torch.uint8)
        output.write(value.numpy().tobytes())


def _dense_target_dtype(source_dtype: str) -> tuple[str, np.dtype]:
    mapping = {
        "BF16": ("F16", np.dtype("<f2")),
        "F16": ("F16", np.dtype("<f2")),
        "F32": ("F32", np.dtype("<f4")),
        "I32": ("I32", np.dtype("<i4")),
        "I64": ("I64", np.dtype("<i8")),
    }
    try:
        return mapping[source_dtype]
    except KeyError as exc:
        raise ValueError(
            f"TPQ dense tensor uses unsupported dtype {source_dtype!r}"
        ) from exc


def _dense_file_records(
    dense_path: Path,
    group_size: int,
    *,
    row_chunk: int,
) -> list[_StreamRecord]:
    records: list[_StreamRecord] = []
    with safe_open(str(dense_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        mx_pairs: dict[str, str] = {}
        consumed_scales: set[str] = set()
        for name in sorted(keys):
            if _slice_dtype(handle, name) not in {"F8_E4M3", "F8_E4M3FN"}:
                continue
            if not name.endswith(".weight"):
                raise ValueError(f"TPQ MXFP8 tensor is not named *.weight: {name}")
            scale_name = name.removesuffix(".weight") + ".scale"
            if scale_name not in keys:
                raise ValueError(f"TPQ MXFP8 tensor has no E8M0 scale: {name}")
            if _slice_dtype(handle, scale_name) != "F8_E8M0":
                raise ValueError(
                    f"TPQ MXFP8 scale {scale_name} is not F8_E8M0"
                )
            mx_header_bytes(
                MXFP8_DTYPE,
                _slice_shape(handle, name),
                _slice_shape(handle, name),
                _slice_shape(handle, scale_name),
            )
            mx_pairs[name] = scale_name
            consumed_scales.add(scale_name)
        for name in sorted(keys):
            if name in consumed_scales:
                continue
            if name.endswith(".qs"):
                continue
            shape = _slice_shape(handle, name)
            if name in mx_pairs:
                scale_name = mx_pairs[name]
                scale_shape = _slice_shape(handle, scale_name)
                prefix = mx_header_bytes(
                    MXFP8_DTYPE,
                    shape,
                    shape,
                    scale_shape,
                )
                nbytes = (
                    len(prefix)
                    + int(np.prod(shape, dtype=np.int64))
                    + int(np.prod(scale_shape, dtype=np.int64))
                )

                def write_mxfp8(
                    output: BinaryIO,
                    *,
                    _path=dense_path,
                    _name=name,
                    _scale_name=scale_name,
                    _prefix=prefix,
                    _chunk=row_chunk,
                ) -> None:
                    with safe_open(
                        str(_path), framework="pt", device="cpu"
                    ) as source:
                        output.write(_prefix)
                        _stream_byte_slice(
                            source,
                            _name,
                            output,
                            row_chunk=_chunk,
                        )
                        _stream_byte_slice(
                            source,
                            _scale_name,
                            output,
                            row_chunk=_chunk,
                        )

                records.append(
                    _StreamRecord(name, MXFP8_DTYPE, nbytes, write_mxfp8)
                )
                continue
            if _slice_dtype(handle, name) == "F8_E8M0":
                raise ValueError(f"orphan TPQ E8M0 scale tensor: {name}")
            scale_name = name + ".qs"
            if scale_name in keys:
                packed_shape = shape
                if len(packed_shape) != 2:
                    raise ValueError(f"TPQ int4 tensor is not rank two: {name}")
                logical_shape = (
                    int(packed_shape[0]),
                    int(packed_shape[1]) * 2,
                )
                scale_shape = _slice_shape(handle, scale_name)
                expected_scale = (
                    logical_shape[0],
                    logical_shape[1] // group_size,
                )
                if scale_shape != expected_scale:
                    raise ValueError(
                        f"TPQ int4 scale shape mismatch for {name}: "
                        f"{scale_shape} != {expected_scale}"
                    )
                nbytes = tpq_int4_payload_nbytes(
                    logical_shape, group_size
                )

                def write_int4(
                    output: BinaryIO,
                    *,
                    _path=dense_path,
                    _name=name,
                    _scale_name=scale_name,
                    _shape=logical_shape,
                    _group=group_size,
                    _chunk=row_chunk,
                ) -> None:
                    with safe_open(
                        str(_path), framework="pt", device="cpu"
                    ) as source:
                        output.write(pack_tpq_int4_prefix(_shape, _group))
                        _stream_slice(
                            source,
                            _name,
                            output,
                            target_dtype=np.dtype(np.uint8),
                            row_chunk=_chunk,
                        )
                        _stream_slice(
                            source,
                            _scale_name,
                            output,
                            target_dtype=np.dtype("<f2"),
                            row_chunk=_chunk,
                        )

                records.append(
                    _StreamRecord(name, "TPQ-I4G64", nbytes, write_int4)
                )
                continue

            dtype, numpy_dtype = _dense_target_dtype(
                _slice_dtype(handle, name)
            )
            element_count = int(np.prod(shape, dtype=np.int64))
            nbytes = 4 + 8 * len(shape) + element_count * _DENSE_ITEMSIZE[dtype]

            def write_dense(
                output: BinaryIO,
                *,
                _path=dense_path,
                _name=name,
                _shape=shape,
                _dtype=numpy_dtype,
                _chunk=row_chunk,
            ) -> None:
                with safe_open(
                    str(_path), framework="pt", device="cpu"
                ) as source:
                    output.write(struct.pack("<I", len(_shape)))
                    output.write(struct.pack(f"<{len(_shape)}q", *_shape))
                    _stream_slice(
                        source,
                        _name,
                        output,
                        target_dtype=_dtype,
                        row_chunk=_chunk,
                    )

            records.append(_StreamRecord(name, dtype, nbytes, write_dense))
    return records


def _dense_records(
    root: Path,
    manifest: dict,
    *,
    row_chunk: int,
) -> list[_StreamRecord]:
    group_size = int(manifest["quant"].get("int4_group", 64))
    if group_size != 64:
        raise ValueError(
            f"native TPQ-I4G64 import requires int4_group=64, got {group_size}"
        )
    records: list[_StreamRecord] = []
    names: set[str] = set()
    for path in _dense_paths(root, manifest):
        shard_records = _dense_file_records(
            path,
            group_size,
            row_chunk=row_chunk,
        )
        duplicates = names.intersection(record.name for record in shard_records)
        if duplicates:
            raise ValueError(
                f"TPQ dense shards repeat tensors: {sorted(duplicates)[:8]}"
            )
        names.update(record.name for record in shard_records)
        records.extend(shard_records)
    return records


def _expert_kind(keys: set[str], expert: int, tag: str) -> str:
    found = [
        tier
        for tier in _TIER_ORDER
        if f"e{expert}.{tag}{tier}" in keys
        or f"e{expert}.{tag}{tier}z" in keys
    ]
    if len(found) != 1:
        raise ValueError(
            f"TPQ expert {expert} projection {tag} has tiers {found}"
        )
    return found[0]


def _dropped_experts(
    manifest: dict,
    *,
    layer: int,
    n_experts: int,
) -> tuple[int, ...]:
    raw = manifest.get("tiers_per_layer", {}).get(str(layer))
    if raw is None:
        raw = manifest.get("tiers_per_layer", {}).get(layer)
    if raw is None:
        return ()
    assignments = str(raw)
    if len(assignments) != n_experts:
        raise ValueError(
            f"TPQ layer {layer} tier count {len(assignments)} "
            f"does not match n_experts={n_experts}"
        )
    invalid = sorted(set(assignments).difference("xwvVd"))
    if invalid:
        raise ValueError(
            f"TPQ layer {layer} has invalid tier characters {invalid}"
        )
    return tuple(
        expert for expert, tier in enumerate(assignments) if tier == "d"
    )


def _read_expert_indices(
    handle,
    keys: set[str],
    *,
    expert: int,
    tag: str,
    spec: TpqPqSpec,
    shape: tuple[int, int],
) -> np.ndarray:
    raw_name = f"e{expert}.{tag}{spec.tier}"
    compressed_name = raw_name + "z"
    dtype = np.uint8 if spec.index_bits <= 8 else np.uint16
    if compressed_name in keys:
        compressed = _numpy(handle.get_tensor(compressed_name))
        raw = zlib.decompress(np.asarray(compressed, dtype=np.uint8).tobytes())
        values = np.frombuffer(raw, dtype=dtype).copy()
    elif raw_name in keys:
        values = _numpy(handle.get_tensor(raw_name)).astype(dtype, copy=False)
    else:
        raise KeyError(f"TPQ expert index tensor is absent: {raw_name}")
    expected = (shape[0], shape[1] // spec.vector_size)
    if values.size != expected[0] * expected[1]:
        raise ValueError(
            f"TPQ expert index size mismatch for {raw_name}: "
            f"{values.size} != {expected}"
        )
    values = np.ascontiguousarray(values.reshape(expected))
    if values.size and int(values.max()) >= spec.codebook_entries:
        raise ValueError(f"TPQ expert index exceeds {spec.codebook_entries}")
    return values


def _expert_projection_layout(
    handle,
    *,
    n_experts: int,
    tag: str,
    dropped_experts: tuple[int, ...] = (),
) -> tuple[dict[str, tuple[int, ...]], set[str]]:
    keys = set(handle.keys())
    dropped = set(dropped_experts)
    by_tier: dict[str, list[int]] = {}
    for expert in range(n_experts):
        tier = (
            _DROP_TIER
            if expert in dropped
            else _expert_kind(keys, expert, tag)
        )
        by_tier.setdefault(tier, []).append(expert)
    return {
        tier: tuple(experts)
        for tier, experts in by_tier.items()
    }, keys


def _drop_placeholder_spec(
    specs: dict[str, TpqPqSpec],
    *,
    columns: int,
) -> TpqPqSpec:
    for tier in reversed(_TIER_ORDER):
        spec = specs.get(tier)
        if spec is not None and columns % spec.vector_size == 0:
            return spec
    raise ValueError(
        f"TPQ dropped-expert placeholder has no VQ tier compatible "
        f"with {columns} columns"
    )


def _expert_record_nbytes(
    layout: dict[str, tuple[int, ...]],
    *,
    rows_per_expert: int,
    columns: int,
    specs: dict[str, TpqPqSpec],
) -> int:
    total = _NINT_MOE_HDR.size
    for tier in _POOL_ORDER:
        expert_ids = layout.get(tier)
        if not expert_ids:
            continue
        if tier == _DROP_TIER:
            spec = _drop_placeholder_spec(specs, columns=columns)
        else:
            try:
                spec = specs[tier]
            except KeyError as exc:
                raise ValueError(
                    f"TPQ expert uses undefined tier {tier!r}"
                ) from exc
        dtype = spec.label.encode("ascii")
        payload = tpq_pq_payload_nbytes(
            (len(expert_ids) * rows_per_expert, columns),
            spec,
        )
        total += (
            _NINT_MOE_POOL_V2_HDR.size
            + len(expert_ids) * 4
            + len(dtype)
            + payload
        )
    return total


def _write_expert_projection(
    output: BinaryIO,
    *,
    shard: Path,
    n_experts: int,
    rows_per_expert: int,
    columns: int,
    tag: str,
    specs: dict[str, TpqPqSpec],
    workers: int,
    dropped_experts: tuple[int, ...] = (),
) -> None:
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        layout, keys = _expert_projection_layout(
            handle,
            n_experts=n_experts,
            tag=tag,
            dropped_experts=dropped_experts,
        )
        output.write(
            _NINT_MOE_HDR.pack(
                _NINT_MOE_MAGIC_V2,
                n_experts,
                rows_per_expert,
                columns,
                len(layout),
            )
        )
        for tier in _POOL_ORDER:
            expert_ids = layout.get(tier)
            if not expert_ids:
                continue
            if tier == _DROP_TIER:
                spec = _drop_placeholder_spec(specs, columns=columns)
            else:
                try:
                    spec = specs[tier]
                except KeyError as exc:
                    raise ValueError(
                        f"TPQ expert uses undefined tier {tier!r}"
                    ) from exc
            dtype = spec.label.encode("ascii")
            payload_nbytes = tpq_pq_payload_nbytes(
                (len(expert_ids) * rows_per_expert, columns),
                spec,
            )
            output.write(
                _NINT_MOE_POOL_V2_HDR.pack(
                    len(expert_ids),
                    len(dtype),
                    payload_nbytes,
                    0,
                )
            )
            output.write(np.asarray(expert_ids, dtype="<i4").tobytes())
            output.write(dtype)
            codebook = (
                np.zeros(
                    (spec.codebook_entries, spec.vector_size),
                    dtype=np.float32,
                )
                if tier == _DROP_TIER
                else _numpy(
                    handle.get_tensor(f"cb.{tag}.{tier}")
                ).astype(np.float32, copy=False)
            )
            output.write(
                pack_tpq_pq_prefix(
                    spec,
                    (len(expert_ids) * rows_per_expert, columns),
                    codebook,
                )
            )
            expert_shape = (rows_per_expert, columns)

            def encode_batch(
                batch: tuple[int, ...],
                *,
                _spec=spec,
                _shape=expert_shape,
                _is_drop=tier == _DROP_TIER,
            ) -> list[bytes]:
                if _is_drop:
                    packed = pack_tpq_indices(
                        np.zeros(
                            (
                                _shape[0],
                                _shape[1] // _spec.vector_size,
                            ),
                            dtype=np.uint8
                            if _spec.index_bits <= 8
                            else np.uint16,
                        ),
                        _spec.index_bits,
                    )
                    return [packed] * len(batch)
                with safe_open(
                    str(shard), framework="pt", device="cpu"
                ) as worker_handle:
                    return [
                        pack_tpq_indices(
                            _read_expert_indices(
                                worker_handle,
                                keys,
                                expert=expert,
                                tag=tag,
                                spec=_spec,
                                shape=_shape,
                            ),
                            _spec.index_bits,
                        )
                        for expert in batch
                    ]

            batch_size = 2
            batches = [
                tuple(expert_ids[start : start + batch_size])
                for start in range(0, len(expert_ids), batch_size)
            ]
            if workers == 1 or len(batches) == 1:
                encoded_batches = map(encode_batch, batches)
                for encoded in encoded_batches:
                    for payload in encoded:
                        output.write(payload)
            else:
                with ThreadPoolExecutor(
                    max_workers=min(workers, len(batches))
                ) as executor:
                    for encoded in executor.map(encode_batch, batches):
                        for payload in encoded:
                            output.write(payload)


def _projection_spec(manifest: dict, layout: str, raw: dict) -> TpqPqSpec:
    dim = int(raw["dim"])
    entries = int(raw["size"])
    packing = str(
        manifest["quant"].get("index_packing", {}).get(layout, "")
    )
    if packing in {"u8", "u16"}:
        bits = 8 if packing == "u8" else 16
    elif packing.startswith("packed-u"):
        bits = int(packing.removeprefix("packed-u"))
    else:
        bits = entries.bit_length() - 1
        if entries <= 0 or 1 << bits != entries:
            raise ValueError(
                f"TPQ layout {layout} cannot infer index width from {entries}"
            )
    return TpqPqSpec("p", dim, entries, bits)


def _projection_codebook_key(
    projection: str,
    layout: str,
    spec: dict,
    expert: int,
) -> str:
    key = f"cb.{projection}.{layout}"
    group_size = spec.get("group_size")
    if group_size is None:
        return key
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError(f"TPQ layout {layout} has invalid group_size")
    group = int(expert) // group_size
    groups = spec.get("groups")
    if groups is not None and group >= int(groups):
        raise ValueError(f"TPQ layout {layout} codebook group is out of range")
    return f"{key}.g{group:03d}"


def _projection_pools(
    manifest: dict,
    *,
    layer: int,
    projection: str,
    n_experts: int,
    assignments: dict[int, dict[str, str | tuple[str, ...]]],
    layouts: dict[str, dict],
) -> tuple[tuple[TpqPqSpec, str, tuple[int, ...]], ...]:
    assignment = assignments[layer][projection]
    per_expert = (
        (assignment,) * n_experts
        if isinstance(assignment, str)
        else assignment
    )
    if len(per_expert) != n_experts:
        raise ValueError(
            f"TPQ projection layout L{layer}/{projection} has "
            f"{len(per_expert)} experts, expected {n_experts}"
        )
    grouped: dict[tuple[str, str], list[int]] = {}
    for expert, layout in enumerate(per_expert):
        raw_spec = layouts[layout]
        key = _projection_codebook_key(
            projection, layout, raw_spec, expert
        )
        grouped.setdefault((layout, key), []).append(expert)
    return tuple(
        (
            _projection_spec(manifest, layout, layouts[layout]),
            key,
            tuple(experts),
        )
        for (layout, key), experts in grouped.items()
    )


def _projection_raw_bytes(handle, key: str) -> bytes:
    tensor = handle.get_tensor(key).detach().cpu().contiguous()
    return tensor.view(torch.uint8).reshape(-1).numpy().tobytes()


def _projection_record_nbytes(
    pools: tuple[tuple[TpqPqSpec, str, tuple[int, ...]], ...],
    *,
    rows_per_expert: int,
    columns: int,
) -> int:
    total = _NINT_MOE_HDR.size
    for spec, _codebook_key, experts in pools:
        dtype = spec.label.encode("ascii")
        payload = tpq_pq_payload_nbytes(
            (len(experts) * rows_per_expert, columns), spec
        )
        total += (
            _NINT_MOE_POOL_V2_HDR.size
            + len(experts) * 4
            + len(dtype)
            + payload
        )
    return total


def _write_projection_vq_record(
    output: BinaryIO,
    *,
    shard: Path,
    projection: str,
    layouts: tuple[str, ...],
    n_experts: int,
    rows_per_expert: int,
    columns: int,
    pools: tuple[tuple[TpqPqSpec, str, tuple[int, ...]], ...],
) -> None:
    output.write(
        _NINT_MOE_HDR.pack(
            _NINT_MOE_MAGIC_V2,
            n_experts,
            rows_per_expert,
            columns,
            len(pools),
        )
    )
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for spec, codebook_key, expert_ids in pools:
            dtype = spec.label.encode("ascii")
            payload_nbytes = tpq_pq_payload_nbytes(
                (len(expert_ids) * rows_per_expert, columns), spec
            )
            output.write(
                _NINT_MOE_POOL_V2_HDR.pack(
                    len(expert_ids), len(dtype), payload_nbytes, 0
                )
            )
            output.write(np.asarray(expert_ids, dtype="<i4").tobytes())
            output.write(dtype)
            if codebook_key not in keys:
                raise KeyError(f"TPQ codebook tensor is absent: {codebook_key}")
            codebook = _numpy(handle.get_tensor(codebook_key)).astype(
                np.float32, copy=False
            )
            output.write(
                pack_tpq_pq_prefix(
                    spec,
                    (len(expert_ids) * rows_per_expert, columns),
                    codebook,
                )
            )
            blocks = columns // spec.vector_size
            expected_bits = rows_per_expert * blocks * spec.index_bits
            if expected_bits % 8:
                raise ValueError(
                    f"TPQ {projection} expert payload is not byte aligned"
                )
            expected_nbytes = expected_bits // 8
            for expert in expert_ids:
                layout = layouts[int(expert)]
                key = f"e{expert}.{projection}.{layout}"
                if key not in keys:
                    raise KeyError(f"TPQ expert tensor is absent: {key}")
                raw = _projection_raw_bytes(handle, key)
                if len(raw) != expected_nbytes:
                    raise ValueError(
                        f"TPQ expert payload size mismatch for {key}: "
                        f"{len(raw)} != {expected_nbytes}"
                    )
                output.write(raw)


def _projection_expert_records(
    root: Path,
    manifest: dict,
) -> list[_StreamRecord]:
    metadata = _projection_metadata(manifest)
    if metadata is None:
        raise ValueError("TPQ manifest is not projection-VQ")
    assignments, layouts = metadata
    config = manifest["config"]
    n_experts = int(config["n_experts"])
    hidden = int(config.get("routed_hidden", config["hidden"]))
    intermediate = int(config["moe_inter"])
    records: list[_StreamRecord] = []
    shapes = {
        "gate": (intermediate, hidden),
        "up": (intermediate, hidden),
        "down": (hidden, intermediate),
    }
    for layer, layout_by_projection in sorted(assignments.items()):
        try:
            shard = root / str(manifest["expert_files"][str(layer)])
        except KeyError:
            shard = root / str(manifest["expert_files"][layer])
        for projection in ("gate", "up", "down"):
            rows, columns = shapes[projection]
            layout = layout_by_projection[projection]
            expert_layouts = (
                (layout,) * n_experts
                if isinstance(layout, str)
                else tuple(str(value) for value in layout)
            )
            if len(expert_layouts) != n_experts:
                raise ValueError(
                    f"TPQ projection layout L{layer}/{projection} has "
                    f"{len(expert_layouts)} experts, expected {n_experts}"
                )
            pools = _projection_pools(
                manifest,
                layer=layer,
                projection=projection,
                n_experts=n_experts,
                assignments=assignments,
                layouts=layouts,
            )
            nbytes = _projection_record_nbytes(
                pools,
                rows_per_expert=rows,
                columns=columns,
            )
            name = f"layers.{layer}.ffn.experts.{projection}.weight"

            def write_projection(
                output: BinaryIO,
                *,
                _shard=shard,
                _projection=projection,
                _layouts=expert_layouts,
                _n_experts=n_experts,
                _rows=rows,
                _columns=columns,
                _pools=pools,
            ) -> None:
                _write_projection_vq_record(
                    output,
                    shard=_shard,
                    projection=_projection,
                    layouts=_layouts,
                    n_experts=_n_experts,
                    rows_per_expert=_rows,
                    columns=_columns,
                    pools=_pools,
                )

            records.append(_StreamRecord(name, "NINTM", nbytes, write_projection))
    return records


def _expert_records(
    root: Path,
    manifest: dict,
    *,
    workers: int = 8,
) -> list[_StreamRecord]:
    if _projection_metadata(manifest) is not None:
        return _projection_expert_records(root, manifest)
    config = manifest["config"]
    n_experts = int(config["n_experts"])
    hidden = int(config.get("routed_hidden", config["hidden"]))
    intermediate = int(config["moe_inter"])
    specs = _manifest_specs(manifest)
    records: list[_StreamRecord] = []
    for raw_layer, filename in sorted(
        manifest["expert_files"].items(),
        key=lambda item: int(item[0]),
    ):
        layer = int(raw_layer)
        shard = root / str(filename)
        dropped_experts = _dropped_experts(
            manifest,
            layer=layer,
            n_experts=n_experts,
        )
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            gate_layout, _ = _expert_projection_layout(
                handle,
                n_experts=n_experts,
                tag="gu",
                dropped_experts=dropped_experts,
            )
            down_layout, _ = _expert_projection_layout(
                handle,
                n_experts=n_experts,
                tag="dn",
                dropped_experts=dropped_experts,
            )
        projections = (
            (
                f"layers.{layer}.ffn.experts.gate_up.weight",
                "gu",
                2 * intermediate,
                hidden,
                gate_layout,
            ),
            (
                f"layers.{layer}.ffn.experts.down.weight",
                "dn",
                hidden,
                intermediate,
                down_layout,
            ),
        )
        for name, tag, rows, columns, layout in projections:
            nbytes = _expert_record_nbytes(
                layout,
                rows_per_expert=rows,
                columns=columns,
                specs=specs,
            )

            def write_projection(
                output: BinaryIO,
                *,
                _shard=shard,
                _n_experts=n_experts,
                _rows=rows,
                _columns=columns,
                _tag=tag,
                _specs=specs,
                _workers=workers,
                _dropped_experts=dropped_experts,
            ) -> None:
                _write_expert_projection(
                    output,
                    shard=_shard,
                    n_experts=_n_experts,
                    rows_per_expert=_rows,
                    columns=_columns,
                    tag=_tag,
                    specs=_specs,
                    workers=_workers,
                    dropped_experts=_dropped_experts,
                )

            records.append(_StreamRecord(name, "NINTM", nbytes, write_projection))
    return records


def _write_mfq(
    path: Path,
    header: FileHeader,
    records: list[_StreamRecord],
) -> None:
    version = max(2, int(header.version))
    with path.open("wb") as output:
        output.write(MFQ_MAGIC)
        output.write(_u32(version))
        architecture = header.model_arch.encode("utf-8")
        output.write(_u32(len(architecture)))
        output.write(architecture)
        output.write(_u32(len(header.extra)))
        for key, value in header.extra.items():
            encoded_key = str(key).encode("utf-8")
            encoded_value = json.dumps(
                value, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            output.write(_u32(len(encoded_key)))
            output.write(encoded_key)
            output.write(_u32(len(encoded_value)))
            output.write(encoded_value)
        output.write(_u32(len(records)))
        for record in records:
            encoded_name = record.name.encode("utf-8")
            encoded_dtype = record.dtype.encode("ascii")
            output.write(_u32(len(encoded_name)))
            output.write(encoded_name)
            output.write(_u32(len(encoded_dtype)))
            output.write(encoded_dtype)
            output.write(struct.pack("<Q", record.nbytes))
        for record in records:
            start = output.tell()
            record.write(output)
            actual = output.tell() - start
            if actual != record.nbytes:
                raise RuntimeError(
                    f"TPQ import size mismatch for {record.name}: "
                    f"{actual} != {record.nbytes}"
                )


def convert(
    input_root: str | Path,
    output_path: str | Path,
    *,
    row_chunk: int = 4096,
    workers: int = 8,
    overwrite: bool = False,
) -> Path:
    """Convert a complete tpq-1 directory into one native MFQ file."""

    if row_chunk <= 0:
        raise ValueError("TPQ import row chunk must be positive")
    if workers <= 0:
        raise ValueError("TPQ import workers must be positive")
    root = Path(input_root).resolve()
    output = Path(output_path).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"MFQ output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        if not overwrite:
            raise FileExistsError(f"partial MFQ output already exists: {partial}")
        partial.unlink()
    manifest = _manifest(root)
    projection = _projection_metadata(manifest)
    specs = (
        _manifest_specs(manifest)
        if projection is None
        else {
            name: _projection_spec(manifest, name, raw)
            for name, raw in projection[1].items()
        }
    )
    records = [
        *_dense_records(root, manifest, row_chunk=row_chunk),
        *_expert_records(root, manifest, workers=workers),
    ]
    header = FileHeader(
        version=2,
        model_arch=(
            "deepseek_v4-tpq-mfq"
            if "hc_mult" in manifest["config"]
            else (
                "kimi_k3-tpq-mfq"
                if (
                    manifest.get("model_family") == "kimi_k3"
                    or (
                        "kda_layers" in manifest["config"]
                        and "routed_hidden" in manifest["config"]
                    )
                )
                else "tpq-mfq"
            )
        ),
        num_tensors=len(records),
        extra={
            "source_format": str(manifest["format"]),
            "source_manifest_sha256": _manifest_sha256(root),
            "tpq_manifest": manifest,
            "tpq_index_storage": {
                tier: spec.index_bits
                for tier, spec in specs.items()
            },
        },
    )
    try:
        _write_mfq(partial, header, records)
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--row-chunk", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = convert(
        args.input,
        args.output,
        row_chunk=args.row_chunk,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "gb": output.stat().st_size / 1e9,
            }
        )
    )


if __name__ == "__main__":
    main()
