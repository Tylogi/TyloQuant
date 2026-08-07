"""Materialize an MFQ expert overlay into one standalone MFQ file.

The merger is structural: unchanged tensor blobs are copied byte-for-byte,
overlay pools are copied byte-for-byte, and superseded experts are removed
from base NINT/NVQ-JSC pools by slicing their serialized row streams. No
weights are dequantized or requantized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_NINT_MOE_HDR = struct.Struct("<4sIIII")
_NINT_MOE_POOL_HDR = struct.Struct("<IIQQ")
_NINT_HDR = struct.Struct("<BBiii")
_NVQ_HDR = struct.Struct("<4sBBHiiI")
_NEPQ_HDR = struct.Struct("<4sBBBBIIIIIQ")
_MX_HDR = struct.Struct("<4sBBHQQQQQQ")

_NIM2 = b"NIM2"
_NID2 = b"NID2"
_NVQ_MAGIC = {b"NVQ1", b"NIQ1"}
_NVQ_JSC_FLAG = 0x20
_NVQ_FLAG_MASK = 0xE0
_NEPQ_MAGIC = b"NEP1"
_NEPQ_VERSION = 1
_NEPQ_ROTATED_FLAG = 1
_MX_MAGIC = b"MXT1"
_MX_VERSION = 1
_MXFP4_KIND = 4
_NEPQ_PROFILES = {
    0: ("NEPQ0-S", 2, 6, 0, 320),
    1: ("NEPQ0-L", 3, 7, 0, 832),
    2: ("NEPQ1-S", 4, 9, 1, 2048),
    3: ("NEPQ1-L", 3, 11, 1, 4096),
}


@dataclass(frozen=True)
class MfqRecord:
    name: str
    dtype: str
    offset: int
    nbytes: int


@dataclass(frozen=True)
class MfqIndex:
    path: Path
    version: int
    model_arch: str
    extra: dict
    records: tuple[MfqRecord, ...]
    file_size: int

    @property
    def by_name(self) -> dict[str, MfqRecord]:
        return {record.name: record for record in self.records}


@dataclass(frozen=True)
class MoePool:
    source: str
    serial_offset: int
    serial_nbytes: int
    expert_ids: tuple[int, ...]
    dtype: str
    runtime_offset: int
    runtime_nbytes: int
    payload_offset: int
    payload_nbytes: int


@dataclass(frozen=True)
class MoeContainer:
    magic: bytes
    n_experts: int
    out_per_expert: int
    neuron_len: int
    pools: tuple[MoePool, ...]


@dataclass(frozen=True)
class LiteralSegment:
    data: bytes

    @property
    def nbytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class SourceSegment:
    source: str
    offset: int
    nbytes: int


Segment = LiteralSegment | SourceSegment


@dataclass(frozen=True)
class RecordPlan:
    name: str
    dtype: str
    nbytes: int
    segments: tuple[Segment, ...]
    changed_experts: int = 0


@dataclass(frozen=True)
class MaterializationPlan:
    segments: tuple[Segment, ...]
    total_bytes: int
    tensor_payload_bytes: int
    changed_records: int
    changed_experts: int
    base_records: int
    final_extra: dict
    family_expert_counts: dict[str, int]


def validate_materialized_mfq(
    path: str | Path,
    *,
    expected_bytes: int | None = None,
    expected_allocation_sha256: str | None = None,
    expected_overlay_sha256: str | None = None,
    expected_family_expert_counts: dict[str, int] | None = None,
) -> dict:
    index = read_mfq_index(path)
    if expected_bytes is not None and index.file_size != expected_bytes:
        raise ValueError(f"materialized file size mismatch: {index.file_size} != {expected_bytes}")
    if (
        expected_allocation_sha256 is not None
        and index.extra.get("allocation_sha256") != expected_allocation_sha256
    ):
        raise ValueError("materialized allocation SHA256 mismatch")
    if (
        expected_overlay_sha256 is not None
        and index.extra.get("materialized_overlay_sha256") != expected_overlay_sha256
    ):
        raise ValueError("materialized overlay SHA256 mismatch")

    family_counts: dict[str, int] = {}
    moe_records = 0
    pools = 0
    with index.path.open("rb") as handle:
        handles = {"standalone": handle}
        for record in index.records:
            if record.dtype == "NINTMD":
                raise ValueError(f"standalone file still contains an overlay record: {record.name}")
            if record.dtype != "NINTM":
                continue
            container = _parse_moe_container(
                handle,
                record,
                source="standalone",
                expected_magic=_NIM2,
                require_full_coverage=True,
            )
            moe_records += 1
            for pool in container.pools:
                selected = tuple(range(len(pool.expert_ids)))
                segments, nbytes = _subset_pool(
                    handles,
                    pool,
                    selected,
                    rows_per_expert=container.out_per_expert,
                    neuron_len=container.neuron_len,
                )
                if nbytes != pool.serial_nbytes or _sum_segments(segments) != nbytes:
                    raise ValueError(f"{record.name}: full pool rewrite changes serialized size")
                family_counts[pool.dtype] = family_counts.get(pool.dtype, 0) + len(pool.expert_ids)
                pools += 1
    family_counts = dict(sorted(family_counts.items()))
    if expected_family_expert_counts is not None and family_counts != dict(
        sorted(expected_family_expert_counts.items())
    ):
        raise ValueError(
            f"materialized family counts differ: {family_counts} != "
            f"{expected_family_expert_counts}"
        )
    return {
        "format": "mfq.materialized-validation.v1",
        "path": str(index.path),
        "file_bytes": index.file_size,
        "model_arch": index.model_arch,
        "tensor_records": len(index.records),
        "moe_records": moe_records,
        "moe_pools": pools,
        "family_expert_counts": family_counts,
        "allocation_sha256": index.extra.get("allocation_sha256"),
        "overlay_sha256": index.extra.get("materialized_overlay_sha256"),
        "status": "passed",
    }


def _read_exact(handle: BinaryIO, offset: int, nbytes: int) -> bytes:
    handle.seek(offset)
    value = handle.read(nbytes)
    if len(value) != nbytes:
        raise ValueError(f"truncated file read at {offset}: requested {nbytes}, got {len(value)}")
    return value


def _read_u32(handle: BinaryIO) -> int:
    value = handle.read(_U32.size)
    if len(value) != _U32.size:
        raise ValueError("truncated MFQ uint32")
    return int(_U32.unpack(value)[0])


def _read_string(handle: BinaryIO) -> str:
    nbytes = _read_u32(handle)
    value = handle.read(nbytes)
    if len(value) != nbytes:
        raise ValueError("truncated MFQ string")
    return value.decode("utf-8")


def read_mfq_index(path: str | Path) -> MfqIndex:
    resolved = Path(path).resolve()
    file_size = resolved.stat().st_size
    with resolved.open("rb") as handle:
        if handle.read(4) != b"MFQ1":
            raise ValueError(f"invalid MFQ magic: {resolved}")
        version = _read_u32(handle)
        model_arch = _read_string(handle)
        extra = {}
        if version >= 2:
            for _ in range(_read_u32(handle)):
                key = _read_string(handle)
                value = _read_string(handle)
                extra[key] = json.loads(value)
        raw_records = []
        for _ in range(_read_u32(handle)):
            name = _read_string(handle)
            dtype = _read_string(handle)
            nbytes_raw = handle.read(_U64.size)
            if len(nbytes_raw) != _U64.size:
                raise ValueError("truncated MFQ tensor table")
            raw_records.append((name, dtype, int(_U64.unpack(nbytes_raw)[0])))
        blob_offset = handle.tell()

    records = []
    names = set()
    for name, dtype, nbytes in raw_records:
        if name in names:
            raise ValueError(f"duplicate MFQ tensor name: {name}")
        names.add(name)
        records.append(MfqRecord(name, dtype, blob_offset, nbytes))
        blob_offset += nbytes
    if blob_offset != file_size:
        raise ValueError(
            f"MFQ size mismatch for {resolved}: table ends at {blob_offset}, "
            f"file has {file_size}"
        )
    return MfqIndex(
        path=resolved,
        version=version,
        model_arch=model_arch,
        extra=extra,
        records=tuple(records),
        file_size=file_size,
    )


def _parse_moe_container(
    handle: BinaryIO,
    record: MfqRecord,
    *,
    source: str,
    expected_magic: bytes,
    require_full_coverage: bool,
) -> MoeContainer:
    header = _read_exact(handle, record.offset, _NINT_MOE_HDR.size)
    magic, n_experts, out_per_expert, neuron_len, pool_count = _NINT_MOE_HDR.unpack(header)
    if magic != expected_magic:
        raise ValueError(f"{record.name}: expected {expected_magic!r}, found {magic!r}")
    if (
        n_experts <= 0
        or out_per_expert <= 0
        or neuron_len <= 0
        or pool_count <= 0
        or pool_count > n_experts
    ):
        raise ValueError(f"{record.name}: invalid NINTM dimensions")

    offset = record.offset + _NINT_MOE_HDR.size
    record_end = record.offset + record.nbytes
    owners = set()
    pools = []
    for _ in range(pool_count):
        serial_offset = offset
        raw = _read_exact(handle, offset, _NINT_MOE_POOL_HDR.size)
        expert_count, dtype_nbytes, payload_nbytes, runtime_nbytes = _NINT_MOE_POOL_HDR.unpack(raw)
        offset += _NINT_MOE_POOL_HDR.size
        if expert_count <= 0 or dtype_nbytes <= 0 or dtype_nbytes > 32:
            raise ValueError(f"{record.name}: invalid NINTM pool header")
        ids_raw = _read_exact(handle, offset, int(expert_count) * 4)
        expert_ids = tuple(int(value) for value in struct.unpack(f"<{int(expert_count)}i", ids_raw))
        offset += int(expert_count) * 4
        if (
            len(set(expert_ids)) != len(expert_ids)
            or any(value < 0 or value >= n_experts for value in expert_ids)
            or owners.intersection(expert_ids)
        ):
            raise ValueError(f"{record.name}: duplicate or invalid expert ID")
        owners.update(expert_ids)
        dtype_raw = _read_exact(handle, offset, int(dtype_nbytes))
        try:
            dtype = dtype_raw.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{record.name}: non-ASCII pool dtype") from exc
        offset += int(dtype_nbytes)
        runtime_offset = offset
        offset += int(runtime_nbytes)
        payload_offset = offset
        offset += int(payload_nbytes)
        if offset > record_end:
            raise ValueError(f"{record.name}: truncated NINTM pool")
        pools.append(
            MoePool(
                source=source,
                serial_offset=serial_offset,
                serial_nbytes=offset - serial_offset,
                expert_ids=expert_ids,
                dtype=dtype,
                runtime_offset=runtime_offset,
                runtime_nbytes=int(runtime_nbytes),
                payload_offset=payload_offset,
                payload_nbytes=int(payload_nbytes),
            )
        )
    if offset != record_end:
        raise ValueError(f"{record.name}: NINTM tail mismatch")
    if require_full_coverage and owners != set(range(int(n_experts))):
        missing = sorted(set(range(int(n_experts))) - owners)
        raise ValueError(f"{record.name}: base NINTM misses experts {missing[:16]}")
    return MoeContainer(
        magic=magic,
        n_experts=int(n_experts),
        out_per_expert=int(out_per_expert),
        neuron_len=int(neuron_len),
        pools=tuple(pools),
    )


def _literal(data: bytes) -> LiteralSegment:
    return LiteralSegment(bytes(data))


def _source(source: str, offset: int, nbytes: int) -> SourceSegment:
    if nbytes < 0:
        raise ValueError("negative source segment")
    return SourceSegment(source, int(offset), int(nbytes))


def _sum_segments(segments: list[Segment] | tuple[Segment, ...]) -> int:
    return sum(segment.nbytes for segment in segments)


def _selected_stream_segments(
    *,
    source: str,
    stream_offset: int,
    bytes_per_expert: int,
    selected_positions: tuple[int, ...],
) -> list[Segment]:
    if bytes_per_expert <= 0:
        raise ValueError("invalid expert stream stride")
    return [
        _source(
            source,
            stream_offset + position * bytes_per_expert,
            bytes_per_expert,
        )
        for position in selected_positions
    ]


def _subset_nint_payload(
    handle: BinaryIO,
    pool: MoePool,
    selected_positions: tuple[int, ...],
    *,
    rows_per_expert: int,
    neuron_len: int,
) -> tuple[tuple[Segment, ...], int]:
    prefix = _read_exact(handle, pool.payload_offset, _NINT_HDR.size + 4)
    bits, sub_bits, groupsize, axis, payload_neuron_len = _NINT_HDR.unpack_from(prefix, 0)
    ndim = _U32.unpack_from(prefix, _NINT_HDR.size)[0]
    header_nbytes = _NINT_HDR.size + 4 + int(ndim) * 8 + 8
    header = _read_exact(handle, pool.payload_offset, header_nbytes)
    shape_offset = _NINT_HDR.size + 4
    shape = list(struct.unpack_from(f"<{int(ndim)}q", header, shape_offset))
    out, groups = struct.unpack_from("<II", header, shape_offset + int(ndim) * 8)
    expected_out = len(pool.expert_ids) * rows_per_expert
    if (
        axis != 0
        or ndim != 2
        or shape != [expected_out, neuron_len]
        or out != expected_out
        or payload_neuron_len != neuron_len
        or groupsize <= 0
        or groups <= 0
        or not (1 <= bits <= 8 and 1 <= sub_bits <= 8)
    ):
        raise ValueError(f"unsupported sliced NINT pool layout: {pool.dtype}")

    anchors_per_expert = rows_per_expert * 2
    sub_bits_per_expert = rows_per_expert * groups * sub_bits
    q_bits_per_expert = rows_per_expert * groups * groupsize * bits
    if sub_bits_per_expert % 8 or q_bits_per_expert % 8:
        raise ValueError(f"{pool.dtype} expert streams are not byte-aligned for structural slicing")
    sub_per_expert = sub_bits_per_expert // 8
    q_per_expert = q_bits_per_expert // 8
    stream_offset = pool.payload_offset + header_nbytes
    anchors_offset = stream_offset
    minimum_offset = anchors_offset + int(out) * 2
    sub_scale_offset = minimum_offset + int(out) * 2
    sub_min_offset = sub_scale_offset + (int(out) * int(groups) * sub_bits + 7) // 8
    q_offset = sub_min_offset + (int(out) * int(groups) * sub_bits + 7) // 8
    payload_end = q_offset + (int(out) * int(groups) * int(groupsize) * bits + 7) // 8
    if payload_end != pool.payload_offset + pool.payload_nbytes:
        raise ValueError(f"{pool.dtype} payload size does not match its metadata")

    selected_count = len(selected_positions)
    new_out = selected_count * rows_per_expert
    new_shape = [new_out, neuron_len]
    new_header = b"".join(
        [
            _NINT_HDR.pack(bits, sub_bits, groupsize, axis, payload_neuron_len),
            _U32.pack(int(ndim)),
            struct.pack(f"<{int(ndim)}q", *new_shape),
            struct.pack("<II", new_out, groups),
        ]
    )
    segments: list[Segment] = [_literal(new_header)]
    for offset, stride in (
        (anchors_offset, anchors_per_expert),
        (minimum_offset, anchors_per_expert),
        (sub_scale_offset, sub_per_expert),
        (sub_min_offset, sub_per_expert),
        (q_offset, q_per_expert),
    ):
        segments.extend(
            _selected_stream_segments(
                source=pool.source,
                stream_offset=offset,
                bytes_per_expert=stride,
                selected_positions=selected_positions,
            )
        )
    return tuple(segments), _sum_segments(segments)


def _subset_nvq_jsc_payload(
    handle: BinaryIO,
    pool: MoePool,
    selected_positions: tuple[int, ...],
    *,
    rows_per_expert: int,
    neuron_len: int,
) -> tuple[tuple[Segment, ...], int]:
    prefix = _read_exact(handle, pool.payload_offset, _NVQ_HDR.size)
    (
        magic,
        encoded_codebook,
        sub_bits,
        groupsize,
        axis,
        payload_neuron_len,
        ndim,
    ) = _NVQ_HDR.unpack(prefix)
    header_nbytes = _NVQ_HDR.size + int(ndim) * 8 + 4
    header = _read_exact(handle, pool.payload_offset, header_nbytes)
    shape = list(struct.unpack_from(f"<{int(ndim)}q", header, _NVQ_HDR.size))
    out = _U32.unpack_from(header, _NVQ_HDR.size + int(ndim) * 8)[0]
    expected_out = len(pool.expert_ids) * rows_per_expert
    codebook_id = int(encoded_codebook) & ~_NVQ_FLAG_MASK
    vector_size = {1: 8, 2: 4, 3: 4, 4: 8, 5: 8, 6: 4}.get(
        codebook_id
    )
    codebook_entries = {
        1: 256,
        2: 256,
        3: 512,
        4: 1024,
        5: 4096,
        6: 1024,
    }.get(codebook_id)
    index_bits = {
        1: 8,
        2: 8,
        3: 9,
        4: 10,
        5: 12,
        6: 10,
    }.get(codebook_id)
    if (
        magic not in _NVQ_MAGIC
        or not (encoded_codebook & _NVQ_JSC_FLAG)
        or vector_size is None
        or axis != 0
        or ndim != 2
        or shape != [expected_out, neuron_len]
        or out != expected_out
        or payload_neuron_len != neuron_len
        or groupsize <= 0
        or not (1 <= sub_bits <= 8)
    ):
        raise ValueError(f"unsupported sliced NVQ-JSC pool layout: {pool.dtype}")

    metadata_offset = pool.payload_offset + header_nbytes
    metadata_header = _read_exact(handle, metadata_offset, 64)
    banks = int(metadata_header[1])
    state_count = int(metadata_header[2])
    if banks not in {1, 2, 4} or state_count != 16:
        raise ValueError(f"{pool.dtype} has invalid JSC metadata")
    metadata_nbytes = 64 + banks * codebook_entries * vector_size
    groups = (neuron_len + groupsize - 1) // groupsize
    vectors = (neuron_len + vector_size - 1) // vector_size
    signs = (neuron_len + 7) // 8
    anchors_per_expert = rows_per_expert * 2
    state_bits_per_expert = rows_per_expert * groups * sub_bits
    index_bits_per_expert = rows_per_expert * vectors * index_bits
    sign_bits_per_expert = rows_per_expert * signs * 7
    if state_bits_per_expert % 8 or index_bits_per_expert % 8 or sign_bits_per_expert % 8:
        raise ValueError(f"{pool.dtype} expert streams are not byte-aligned for structural slicing")
    state_per_expert = state_bits_per_expert // 8
    index_per_expert = index_bits_per_expert // 8
    sign_per_expert = sign_bits_per_expert // 8

    anchors_offset = metadata_offset + metadata_nbytes
    state_offset = anchors_offset + int(out) * 2
    indices_offset = state_offset + (int(out) * groups * sub_bits + 7) // 8
    signs_offset = indices_offset + (int(out) * vectors * index_bits + 7) // 8
    payload_end = signs_offset + (int(out) * signs * 7 + 7) // 8
    if payload_end != pool.payload_offset + pool.payload_nbytes:
        raise ValueError(f"{pool.dtype} payload size does not match its metadata")

    selected_count = len(selected_positions)
    new_out = selected_count * rows_per_expert
    new_shape = [new_out, neuron_len]
    new_header = b"".join(
        [
            _NVQ_HDR.pack(
                magic,
                encoded_codebook,
                sub_bits,
                groupsize,
                axis,
                payload_neuron_len,
                int(ndim),
            ),
            struct.pack(f"<{int(ndim)}q", *new_shape),
            _U32.pack(new_out),
        ]
    )
    segments: list[Segment] = [
        _literal(new_header),
        _source(pool.source, metadata_offset, metadata_nbytes),
    ]
    for offset, stride in (
        (anchors_offset, anchors_per_expert),
        (state_offset, state_per_expert),
        (indices_offset, index_per_expert),
        (signs_offset, sign_per_expert),
    ):
        segments.extend(
            _selected_stream_segments(
                source=pool.source,
                stream_offset=offset,
                bytes_per_expert=stride,
                selected_positions=selected_positions,
            )
        )
    return tuple(segments), _sum_segments(segments)


def _subset_nepq_payload(
    handle: BinaryIO,
    pool: MoePool,
    selected_positions: tuple[int, ...],
    *,
    rows_per_expert: int,
    neuron_len: int,
) -> tuple[tuple[Segment, ...], int]:
    header = _read_exact(handle, pool.payload_offset, _NEPQ_HDR.size)
    (
        magic,
        version,
        profile_id,
        groups_per_supergroup,
        flags,
        n_experts,
        payload_rows_per_expert,
        payload_neuron_len,
        bank_count,
        rotation_block,
        rotation_seed,
    ) = _NEPQ_HDR.unpack(header)
    profile = _NEPQ_PROFILES.get(int(profile_id))
    if profile is None:
        raise ValueError(f"{pool.dtype} has an unknown NEPQ profile")
    expected_dtype, state_bits, index_bits, aux_bits, table_bytes = profile
    if (
        magic != _NEPQ_MAGIC
        or version != _NEPQ_VERSION
        or pool.dtype != expected_dtype
        or groups_per_supergroup != 4
        or flags & ~_NEPQ_ROTATED_FLAG
        or bool(flags & _NEPQ_ROTATED_FLAG) != bool(rotation_block)
        or (not rotation_block and rotation_seed)
        or n_experts != len(pool.expert_ids)
        or payload_rows_per_expert != rows_per_expert
        or payload_neuron_len != neuron_len
        or neuron_len % 8
        or not 1 <= bank_count <= 256
    ):
        raise ValueError(f"unsupported sliced NEPQ pool layout: {pool.dtype}")

    groups = math.ceil(neuron_len / 24)
    vectors = neuron_len // 8
    supergroups = math.ceil(groups / int(groups_per_supergroup))
    rows = int(n_experts) * rows_per_expert
    table_nbytes = int(bank_count) * table_bytes
    anchors_nbytes = rows * 2
    states_nbytes = (rows * groups * state_bits + 7) // 8
    indices_nbytes = (rows * vectors * index_bits + 7) // 8
    aux_nbytes = (rows * groups * aux_bits + 7) // 8
    selectors_nbytes = rows * supergroups

    tables_offset = pool.payload_offset + _NEPQ_HDR.size
    anchors_offset = tables_offset + table_nbytes
    states_offset = anchors_offset + anchors_nbytes
    indices_offset = states_offset + states_nbytes
    aux_offset = indices_offset + indices_nbytes
    selectors_offset = aux_offset + aux_nbytes
    payload_end = selectors_offset + selectors_nbytes
    if payload_end != pool.payload_offset + pool.payload_nbytes:
        raise ValueError(f"{pool.dtype} payload size does not match its metadata")

    stream_layout = (
        (anchors_offset, rows_per_expert * 2),
        (states_offset, rows_per_expert * groups * state_bits // 8),
        (indices_offset, rows_per_expert * vectors * index_bits // 8),
        (aux_offset, rows_per_expert * groups * aux_bits // 8),
        (selectors_offset, rows_per_expert * supergroups),
    )
    for bits_per_expert in (
        rows_per_expert * groups * state_bits,
        rows_per_expert * vectors * index_bits,
        rows_per_expert * groups * aux_bits,
    ):
        if bits_per_expert % 8:
            raise ValueError(
                f"{pool.dtype} expert streams are not byte-aligned for structural slicing"
            )

    new_header = _NEPQ_HDR.pack(
        magic,
        version,
        profile_id,
        groups_per_supergroup,
        flags,
        len(selected_positions),
        payload_rows_per_expert,
        payload_neuron_len,
        bank_count,
        rotation_block,
        rotation_seed,
    )
    segments: list[Segment] = [
        _literal(new_header),
        _source(pool.source, tables_offset, table_nbytes),
    ]
    for offset, stride in stream_layout:
        if stride:
            segments.extend(
                _selected_stream_segments(
                    source=pool.source,
                    stream_offset=offset,
                    bytes_per_expert=stride,
                    selected_positions=selected_positions,
                )
            )
    return tuple(segments), _sum_segments(segments)


def _subset_mxfp4_payload(
    handle: BinaryIO,
    pool: MoePool,
    selected_positions: tuple[int, ...],
    *,
    rows_per_expert: int,
    neuron_len: int,
) -> tuple[tuple[Segment, ...], int]:
    header = _read_exact(handle, pool.payload_offset, _MX_HDR.size)
    (
        magic,
        version,
        kind,
        reserved,
        rows,
        columns,
        storage_rows,
        storage_columns,
        scale_rows,
        scale_columns,
    ) = _MX_HDR.unpack(header)
    expected_rows = len(pool.expert_ids) * rows_per_expert
    if (
        magic != _MX_MAGIC
        or version != _MX_VERSION
        or kind != _MXFP4_KIND
        or reserved
        or rows != expected_rows
        or columns != neuron_len
        or storage_rows != expected_rows
        or storage_columns != neuron_len // 2
        or scale_rows != expected_rows
        or scale_columns != neuron_len // 32
        or neuron_len % 32
    ):
        raise ValueError("unsupported sliced MXFP4 pool layout")
    values_per_expert = rows_per_expert * int(storage_columns)
    scales_per_expert = rows_per_expert * int(scale_columns)
    values_offset = pool.payload_offset + _MX_HDR.size
    scales_offset = values_offset + int(storage_rows * storage_columns)
    payload_end = scales_offset + int(scale_rows * scale_columns)
    if payload_end != pool.payload_offset + pool.payload_nbytes:
        raise ValueError("MXFP4 payload size does not match its metadata")
    selected_rows = len(selected_positions) * rows_per_expert
    new_header = _MX_HDR.pack(
        magic,
        version,
        kind,
        reserved,
        selected_rows,
        neuron_len,
        selected_rows,
        neuron_len // 2,
        selected_rows,
        neuron_len // 32,
    )
    segments: list[Segment] = [_literal(new_header)]
    segments.extend(
        _selected_stream_segments(
            source=pool.source,
            stream_offset=values_offset,
            bytes_per_expert=values_per_expert,
            selected_positions=selected_positions,
        )
    )
    segments.extend(
        _selected_stream_segments(
            source=pool.source,
            stream_offset=scales_offset,
            bytes_per_expert=scales_per_expert,
            selected_positions=selected_positions,
        )
    )
    return tuple(segments), _sum_segments(segments)


def _subset_pool(
    handles: dict[str, BinaryIO],
    pool: MoePool,
    selected_positions: tuple[int, ...],
    *,
    rows_per_expert: int,
    neuron_len: int,
) -> tuple[tuple[Segment, ...], int]:
    handle = handles[pool.source]
    if pool.dtype.startswith("NINT") and pool.dtype != "NINTM":
        payload_segments, payload_nbytes = _subset_nint_payload(
            handle,
            pool,
            selected_positions,
            rows_per_expert=rows_per_expert,
            neuron_len=neuron_len,
        )
    elif pool.dtype in {
        "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
        "NVQ3J", "NVQ3J-512", "NVQ3J-L",
    }:
        payload_segments, payload_nbytes = _subset_nvq_jsc_payload(
            handle,
            pool,
            selected_positions,
            rows_per_expert=rows_per_expert,
            neuron_len=neuron_len,
        )
    elif pool.dtype.startswith("NEPQ"):
        payload_segments, payload_nbytes = _subset_nepq_payload(
            handle,
            pool,
            selected_positions,
            rows_per_expert=rows_per_expert,
            neuron_len=neuron_len,
        )
    elif pool.dtype == "MXFP4":
        if pool.runtime_nbytes:
            raise ValueError("MXFP4 pool cannot carry runtime metadata")
        payload_segments, payload_nbytes = _subset_mxfp4_payload(
            handle,
            pool,
            selected_positions,
            rows_per_expert=rows_per_expert,
            neuron_len=neuron_len,
        )
    else:
        raise ValueError(f"cannot structurally slice base pool dtype {pool.dtype!r}")

    selected_ids = tuple(pool.expert_ids[index] for index in selected_positions)
    dtype_bytes = pool.dtype.encode("ascii")
    segments: list[Segment] = [
        _literal(
            _NINT_MOE_POOL_HDR.pack(
                len(selected_ids),
                len(dtype_bytes),
                payload_nbytes,
                pool.runtime_nbytes,
            )
        ),
        _literal(struct.pack(f"<{len(selected_ids)}i", *selected_ids)),
        _literal(dtype_bytes),
    ]
    if pool.runtime_nbytes:
        segments.append(_source(pool.source, pool.runtime_offset, pool.runtime_nbytes))
    segments.extend(payload_segments)
    return tuple(segments), _sum_segments(segments)


def _merge_moe_record(
    handles: dict[str, BinaryIO],
    base_record: MfqRecord,
    overlay_record: MfqRecord,
) -> tuple[tuple[Segment, ...], int, int, dict[str, int]]:
    base = _parse_moe_container(
        handles["base"],
        base_record,
        source="base",
        expected_magic=_NIM2,
        require_full_coverage=True,
    )
    delta = _parse_moe_container(
        handles["overlay"],
        overlay_record,
        source="overlay",
        expected_magic=_NID2,
        require_full_coverage=False,
    )
    if (
        base.n_experts,
        base.out_per_expert,
        base.neuron_len,
    ) != (
        delta.n_experts,
        delta.out_per_expert,
        delta.neuron_len,
    ):
        raise ValueError(f"{base_record.name}: overlay shape mismatch")

    changed_ids = {expert_id for pool in delta.pools for expert_id in pool.expert_ids}
    if not changed_ids:
        raise ValueError(f"{base_record.name}: empty expert overlay")
    pool_segments: list[tuple[Segment, ...]] = []
    final_owners = set()
    family_counts: dict[str, int] = {}
    for pool in base.pools:
        selected = tuple(
            index for index, expert_id in enumerate(pool.expert_ids) if expert_id not in changed_ids
        )
        if not selected:
            continue
        final_ids = tuple(pool.expert_ids[index] for index in selected)
        if len(selected) == len(pool.expert_ids):
            segments = (_source(pool.source, pool.serial_offset, pool.serial_nbytes),)
        else:
            segments, _ = _subset_pool(
                handles,
                pool,
                selected,
                rows_per_expert=base.out_per_expert,
                neuron_len=base.neuron_len,
            )
        pool_segments.append(segments)
        final_owners.update(final_ids)
        family_counts[pool.dtype] = family_counts.get(pool.dtype, 0) + len(final_ids)
    for pool in delta.pools:
        if final_owners.intersection(pool.expert_ids):
            raise ValueError(f"{base_record.name}: duplicate final expert owner")
        final_owners.update(pool.expert_ids)
        pool_segments.append((_source(pool.source, pool.serial_offset, pool.serial_nbytes),))
        family_counts[pool.dtype] = family_counts.get(pool.dtype, 0) + len(pool.expert_ids)
    if final_owners != set(range(base.n_experts)):
        missing = sorted(set(range(base.n_experts)) - final_owners)
        raise ValueError(f"{base_record.name}: merged NINTM misses experts {missing[:16]}")

    segments: list[Segment] = [
        _literal(
            _NINT_MOE_HDR.pack(
                _NIM2,
                base.n_experts,
                base.out_per_expert,
                base.neuron_len,
                len(pool_segments),
            )
        )
    ]
    for values in pool_segments:
        segments.extend(values)
    return (
        tuple(segments),
        _sum_segments(segments),
        len(changed_ids),
        family_counts,
    )


def _encode_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return _U32.pack(len(data)) + data


def _encode_file_table(
    *,
    version: int,
    model_arch: str,
    extra: dict,
    records: tuple[RecordPlan, ...],
) -> bytes:
    if extra and version < 2:
        version = 2
    parts = [b"MFQ1", _U32.pack(version), _encode_string(model_arch)]
    if version >= 2:
        parts.append(_U32.pack(len(extra)))
        for key, value in extra.items():
            parts.append(_encode_string(str(key)))
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            parts.append(_encode_string(encoded))
    parts.append(_U32.pack(len(records)))
    for record in records:
        parts.append(_encode_string(record.name))
        parts.append(_encode_string(record.dtype))
        parts.append(_U64.pack(record.nbytes))
    return b"".join(parts)


def _sha256_file(path: str | Path, *, chunk_bytes: int = 32 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            value = handle.read(chunk_bytes)
            if not value:
                break
            digest.update(value)
    return digest.hexdigest()


def _load_allocation(
    path: str | Path | None,
    overlay_extra: dict,
) -> tuple[dict | None, str | None]:
    if path is None:
        return None, None
    allocation_path = Path(path).resolve()
    raw = allocation_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = overlay_extra.get("plan_sha256")
    if expected and digest != expected:
        raise ValueError(f"allocation SHA256 mismatch: {digest} != overlay {expected}")
    return json.loads(raw), digest


def build_materialization_plan(
    base_path: str | Path,
    overlay_path: str | Path,
    *,
    allocation_path: str | Path | None = None,
    overlay_sha256: str | None = None,
) -> MaterializationPlan:
    base = read_mfq_index(base_path)
    overlay = read_mfq_index(overlay_path)
    if base.version < 2 or overlay.version < 2:
        raise ValueError("overlay materialization requires MFQ file version 2")
    overlay_records = overlay.by_name
    if not overlay_records:
        raise ValueError("overlay has no tensor records")
    unknown = sorted(set(overlay_records) - set(base.by_name))
    if unknown:
        raise ValueError(f"overlay contains unknown tensors: {unknown[:8]}")
    if any(record.dtype != "NINTMD" for record in overlay.records):
        raise ValueError("overlay may contain only NINTMD records")
    for name in overlay_records:
        if base.by_name[name].dtype != "NINTM":
            raise ValueError(f"overlay target is not NINTM: {name}")

    base_allocation = overlay.extra.get("base_allocation_sha256")
    if base_allocation and base.extra.get("allocation_sha256") != base_allocation:
        raise ValueError("overlay was built for another base allocation")
    source_sha = overlay.extra.get("source_index_sha256")
    if source_sha and base.extra.get("source_index_sha256") != source_sha:
        raise ValueError("overlay was built from another source checkpoint")
    allocation, allocation_sha = _load_allocation(allocation_path, overlay.extra)

    records: list[RecordPlan] = []
    changed_records = 0
    changed_experts = 0
    family_counts: dict[str, int] = {}
    with base.path.open("rb") as base_handle, overlay.path.open("rb") as overlay_handle:
        handles = {"base": base_handle, "overlay": overlay_handle}
        for record in base.records:
            delta_record = overlay_records.get(record.name)
            if delta_record is None:
                if record.dtype == "NINTM":
                    unchanged = _parse_moe_container(
                        base_handle,
                        record,
                        source="base",
                        expected_magic=_NIM2,
                        require_full_coverage=True,
                    )
                    for pool in unchanged.pools:
                        family_counts[pool.dtype] = family_counts.get(pool.dtype, 0) + len(
                            pool.expert_ids
                        )
                records.append(
                    RecordPlan(
                        record.name,
                        record.dtype,
                        record.nbytes,
                        (_source("base", record.offset, record.nbytes),),
                    )
                )
                continue
            segments, nbytes, count, record_families = _merge_moe_record(
                handles, record, delta_record
            )
            records.append(
                RecordPlan(
                    record.name,
                    record.dtype,
                    nbytes,
                    segments,
                    changed_experts=count,
                )
            )
            changed_records += 1
            changed_experts += count
            for family, family_count in record_families.items():
                family_counts[family] = family_counts.get(family, 0) + family_count

    if changed_records != len(overlay.records):
        raise ValueError("not every overlay record was materialized")
    tensor_payload_bytes = sum(record.nbytes for record in records)
    extra = dict(base.extra)
    for stale_key in (
        "expert_base_family",
        "expert_upgrade_family",
        "imatrix",
        "imatrix_datasets",
        "scheme_sha256",
        "source",
    ):
        extra.pop(stale_key, None)
    if allocation_sha is not None:
        extra["base_allocation_sha256"] = base.extra.get("allocation_sha256")
        extra["allocation_sha256"] = allocation_sha
    extra["estimated_blob_bytes"] = tensor_payload_bytes
    extra["materialized_overlay_plan_sha256"] = overlay.extra.get("plan_sha256")
    if overlay_sha256 is not None:
        extra["materialized_overlay_sha256"] = overlay_sha256
    extra["expert_precision_families"] = sorted(family_counts)
    if allocation is not None:
        for source_key, output_key in (
            ("reap_sha256", "reap_sha256"),
            ("sensitivity_map_sha256", "expert_sensitivity_map_sha256"),
        ):
            if allocation.get(source_key):
                extra[output_key] = allocation[source_key]
        method = allocation.get("method")
        if isinstance(method, dict):
            if method.get("solver"):
                extra["expert_allocation_method"] = method["solver"]
            if method.get("objective"):
                extra["expert_allocation_objective"] = method["objective"]
            constraints = {key: method[key] for key in ("V", "v", "w") if method.get(key)}
            if constraints:
                extra["expert_precision_constraints"] = constraints
        elif method:
            extra["expert_allocation_method"] = method
        surrogate = allocation.get("surrogate")
        if isinstance(surrogate, dict) and surrogate.get("family_snr_db"):
            extra["expert_family_snr_db"] = surrogate["family_snr_db"]

    records_tuple = tuple(records)
    table = _encode_file_table(
        version=base.version,
        model_arch=base.model_arch,
        extra=extra,
        records=records_tuple,
    )
    segments: list[Segment] = [_literal(table)]
    for record in records_tuple:
        segments.extend(record.segments)
    total_bytes = _sum_segments(segments)
    return MaterializationPlan(
        segments=tuple(segments),
        total_bytes=total_bytes,
        tensor_payload_bytes=tensor_payload_bytes,
        changed_records=changed_records,
        changed_experts=changed_experts,
        base_records=len(base.records),
        final_extra=extra,
        family_expert_counts=dict(sorted(family_counts.items())),
    )


def plan_manifest(
    plan: MaterializationPlan,
    *,
    base_path: str | Path,
    overlay_path: str | Path,
    allocation_path: str | Path | None,
    overlay_sha256: str | None,
) -> dict:
    return {
        "format": "mfq.materialized-overlay.v1",
        "base": str(Path(base_path).resolve()),
        "base_bytes": Path(base_path).stat().st_size,
        "overlay": str(Path(overlay_path).resolve()),
        "overlay_bytes": Path(overlay_path).stat().st_size,
        "overlay_sha256": overlay_sha256,
        "allocation": (
            str(Path(allocation_path).resolve()) if allocation_path is not None else None
        ),
        "output_bytes": plan.total_bytes,
        "tensor_payload_bytes": plan.tensor_payload_bytes,
        "tensor_records": plan.base_records,
        "changed_records": plan.changed_records,
        "changed_experts": plan.changed_experts,
        "family_expert_counts": plan.family_expert_counts,
        "segment_count": len(plan.segments),
        "header_extra": plan.final_extra,
    }


def _stream_plan(
    plan: MaterializationPlan,
    sources: dict[str, BinaryIO],
    output: BinaryIO,
    *,
    start_offset: int,
    length: int | None,
    chunk_bytes: int,
    progress_bytes: int,
) -> int:
    if start_offset < 0 or start_offset > plan.total_bytes:
        raise ValueError(f"start offset {start_offset} outside [0,{plan.total_bytes}]")
    available = plan.total_bytes - start_offset
    target = available if length is None else min(available, max(0, length))
    end_offset = start_offset + target
    logical_offset = 0
    written = 0
    next_progress = progress_bytes
    started = time.monotonic()
    for segment in plan.segments:
        segment_end = logical_offset + segment.nbytes
        if segment_end <= start_offset:
            logical_offset = segment_end
            continue
        if logical_offset >= end_offset:
            break
        relative_start = max(0, start_offset - logical_offset)
        relative_end = min(segment.nbytes, end_offset - logical_offset)
        nbytes = relative_end - relative_start
        if nbytes <= 0:
            logical_offset = segment_end
            continue
        if isinstance(segment, LiteralSegment):
            output.write(segment.data[relative_start:relative_end])
            written += nbytes
        else:
            source = sources[segment.source]
            source.seek(segment.offset + relative_start)
            remaining = nbytes
            while remaining:
                value = source.read(min(chunk_bytes, remaining))
                if not value:
                    raise ValueError(
                        f"truncated {segment.source} source at "
                        f"{segment.offset + relative_start + nbytes - remaining}"
                    )
                output.write(value)
                remaining -= len(value)
                written += len(value)
                if progress_bytes and written >= next_progress:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = written / elapsed / (1 << 20)
                    print(
                        f"materialize {written}/{target} bytes "
                        f"({100.0 * written / max(target, 1):.2f}%) "
                        f"{rate:.1f} MiB/s",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_progress += progress_bytes
        logical_offset = segment_end
    if written != target:
        raise RuntimeError(f"materializer wrote {written} bytes, expected {target}")
    output.flush()
    return written


def _write_json(path: str | Path, value: dict) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _run(args: argparse.Namespace) -> None:
    overlay_sha256 = args.expected_overlay_sha256
    if args.verify_overlay_sha256:
        actual = _sha256_file(args.overlay)
        if overlay_sha256 is not None and actual != overlay_sha256:
            raise ValueError(f"overlay SHA256 mismatch: {actual} != {overlay_sha256}")
        overlay_sha256 = actual
    plan = build_materialization_plan(
        args.base,
        args.overlay,
        allocation_path=args.allocation,
        overlay_sha256=overlay_sha256,
    )
    manifest = plan_manifest(
        plan,
        base_path=args.base,
        overlay_path=args.overlay,
        allocation_path=args.allocation,
        overlay_sha256=overlay_sha256,
    )
    if args.manifest:
        _write_json(args.manifest, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    start_offset = int(args.start_offset)
    output_path = None if args.output == "-" else Path(args.output).resolve()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume:
            start_offset = output_path.stat().st_size if output_path.exists() else 0
        mode = "ab" if start_offset else "wb"
        if start_offset and (
            not output_path.exists() or output_path.stat().st_size != start_offset
        ):
            raise ValueError("output size does not match the requested resume offset")
        output_handle = output_path.open(mode)
    else:
        output_handle = sys.stdout.buffer

    try:
        with (
            Path(args.base).resolve().open("rb") as base_handle,
            Path(args.overlay).resolve().open("rb") as overlay_handle,
        ):
            written = _stream_plan(
                plan,
                {"base": base_handle, "overlay": overlay_handle},
                output_handle,
                start_offset=start_offset,
                length=args.length,
                chunk_bytes=int(args.chunk_mib) << 20,
                progress_bytes=int(args.progress_mib) << 20,
            )
    finally:
        if output_path is not None:
            output_handle.close()
    print(
        json.dumps(
            {
                "status": "ok",
                "start_offset": start_offset,
                "written_bytes": written,
                "output_bytes": plan.total_bytes,
                "complete": start_offset + written == plan.total_bytes,
            },
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--allocation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--expected-overlay-sha256")
    parser.add_argument("--verify-overlay-sha256", action="store_true")
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--length", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--progress-mib", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    _run(build_parser().parse_args())


if __name__ == "__main__":
    main()
