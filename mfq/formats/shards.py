"""MFQ shard naming, planning, and streaming writers."""

from __future__ import annotations

import json
import os
import re
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from mfq.formats.assets import is_asset_record
from mfq.formats.header import MFQ_MAGIC, FileHeader

SPLIT_NO_KEY = "split.no"
SPLIT_COUNT_KEY = "split.count"
SPLIT_TENSORS_COUNT_KEY = "split.tensors.count"
SPLIT_RECORDS_COUNT_KEY = "split.records.count"
SPLIT_KEYS = frozenset(
    {
        SPLIT_NO_KEY,
        SPLIT_COUNT_KEY,
        SPLIT_TENSORS_COUNT_KEY,
        SPLIT_RECORDS_COUNT_KEY,
    }
)

_SHARD_RE = re.compile(
    r"^(?P<stem>.+)-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.mfq$"
)


class BlobRecordLike(Protocol):
    name: str
    dtype: str
    nbytes: int
    path: Path


def _u32(value: int) -> bytes:
    return struct.pack("<I", int(value))


def format_shard_path(path: str | Path, index: int, count: int) -> Path:
    """Return the 1-based ``xxxxx-00001-of-00004.mfq`` shard path."""

    source = Path(path)
    if not 1 <= index <= count <= 99999:
        raise ValueError(f"invalid MFQ shard index/count: {index}/{count}")
    stem = source.name[:-4] if source.name.lower().endswith(".mfq") else source.name
    return source.with_name(f"{stem}-{index:05d}-of-{count:05d}.mfq")


def parse_shard_path(path: str | Path) -> tuple[Path, int, int] | None:
    """Parse a shard name into ``(base_path, 1-based index, count)``."""

    source = Path(path)
    match = _SHARD_RE.fullmatch(source.name)
    if match is None:
        return None
    count = int(match.group("count"))
    index = int(match.group("index"))
    if count < 1 or index < 1 or index > count:
        raise ValueError(f"invalid MFQ shard file name: {source}")
    base = source.with_name(f"{match.group('stem')}.mfq")
    return base, index, count


def shard_paths_from_any(path: str | Path, count: int) -> list[Path]:
    """Resolve every shard path when ``path`` names any shard in the set."""

    parsed = parse_shard_path(path)
    if parsed is None:
        raise ValueError(f"sharded MFQ path lacks -00001-of-00000 suffix: {path}")
    base, _, filename_count = parsed
    if filename_count != count:
        raise ValueError(
            f"MFQ shard count mismatch in filename/metadata: "
            f"{filename_count} != {count}: {path}"
        )
    return [format_shard_path(base, index, count) for index in range(1, count + 1)]


def matching_shard_paths(path: str | Path) -> list[Path]:
    """List shard files whose decoded base path equals ``path``."""

    base = Path(path)
    result: list[Path] = []
    if not base.parent.is_dir():
        return result
    for candidate in base.parent.iterdir():
        parsed = parse_shard_path(candidate)
        if parsed is not None and parsed[0].name == base.name:
            result.append(candidate)
    return sorted(result)


def split_values(extra: dict[str, object]) -> tuple[int, int, int | None, int | None]:
    """Read and validate the split metadata from a decoded MFQ header."""

    split_no = int(extra.get(SPLIT_NO_KEY, 0))
    split_count = int(extra.get(SPLIT_COUNT_KEY, 1))
    tensor_count = extra.get(SPLIT_TENSORS_COUNT_KEY)
    record_count = extra.get(SPLIT_RECORDS_COUNT_KEY)
    if split_count < 1 or split_count > 99999:
        raise ValueError(f"invalid {SPLIT_COUNT_KEY}: {split_count}")
    if split_no < 0 or split_no >= split_count:
        raise ValueError(f"invalid {SPLIT_NO_KEY}: {split_no}")
    return (
        split_no,
        split_count,
        None if tensor_count is None else int(tensor_count),
        None if record_count is None else int(record_count),
    )


def validate_split_limits(
    split_max_size: int,
    split_max_tensors: int,
    *,
    required: bool = False,
) -> None:
    if split_max_size < 0 or split_max_tensors < 0:
        raise ValueError("MFQ shard limits must be non-negative")
    if split_max_size and split_max_tensors:
        raise ValueError("--split-max-size and --split-max-tensors are mutually exclusive")
    if required and not (split_max_size or split_max_tensors):
        raise ValueError("one positive MFQ shard limit is required")


def plan_record_shards(
    records: Sequence[BlobRecordLike],
    *,
    split_max_size: int = 0,
    split_max_tensors: int = 0,
) -> list[list[BlobRecordLike]]:
    """Assign records to shards using llama.cpp's pre-add boundary rule."""

    validate_split_limits(split_max_size, split_max_tensors)
    if not split_max_size and not split_max_tensors:
        return [list(records)]

    assets = [record for record in records if is_asset_record(record.name)]
    tensors = [record for record in records if not is_asset_record(record.name)]
    shards: list[list[BlobRecordLike]] = [[]]
    payload_sizes = [0]
    tensor_counts = [0]

    for record in tensors:
        current = len(shards) - 1
        exceeds_size = bool(
            split_max_size
            and tensor_counts[current] > 0
            and payload_sizes[current] + int(record.nbytes) > split_max_size
        )
        exceeds_count = bool(
            split_max_tensors
            and tensor_counts[current] >= split_max_tensors
        )
        if exceeds_size or exceeds_count:
            shards.append([])
            payload_sizes.append(0)
            tensor_counts.append(0)
            current += 1
        shards[current].append(record)
        payload_sizes[current] += int(record.nbytes)
        tensor_counts[current] += 1

    # Runtime assets are container metadata and live only in the first shard.
    shards[0][0:0] = assets
    return shards


def shard_header(
    header: FileHeader,
    *,
    split_no: int,
    split_count: int,
    tensor_count: int,
    record_count: int,
) -> FileHeader:
    """Build one shard header; model metadata is retained only in shard zero."""

    if split_no == 0:
        extra = {key: value for key, value in header.extra.items() if key not in SPLIT_KEYS}
    else:
        extra = {}
    extra.update(
        {
            SPLIT_NO_KEY: split_no,
            SPLIT_COUNT_KEY: split_count,
            SPLIT_TENSORS_COUNT_KEY: tensor_count,
            SPLIT_RECORDS_COUNT_KEY: record_count,
        }
    )
    return FileHeader(
        version=max(2, int(header.version)),
        model_arch=header.model_arch,
        num_tensors=0,
        extra=extra,
    )


def _write_header_and_table(
    output,
    header: FileHeader,
    records: Sequence[BlobRecordLike],
) -> None:
    version = int(header.version)
    extra = dict(header.extra)
    if extra and version < 2:
        version = 2
    output.write(MFQ_MAGIC)
    output.write(_u32(version))
    arch = header.model_arch.encode("utf-8")
    output.write(_u32(len(arch)))
    output.write(arch)
    if version >= 2:
        output.write(_u32(len(extra)))
        for key, value in extra.items():
            key_bytes = str(key).encode("utf-8")
            value_bytes = json.dumps(value).encode("utf-8")
            output.write(_u32(len(key_bytes)))
            output.write(key_bytes)
            output.write(_u32(len(value_bytes)))
            output.write(value_bytes)
    output.write(_u32(len(records)))
    for record in records:
        name = record.name.encode("utf-8")
        dtype = record.dtype.encode("utf-8")
        output.write(_u32(len(name)))
        output.write(name)
        output.write(_u32(len(dtype)))
        output.write(dtype)
        output.write(struct.pack("<Q", int(record.nbytes)))


def _copy_record(record: BlobRecordLike, target) -> None:
    remaining = int(record.nbytes)
    with Path(record.path).open("rb") as source:
        source.seek(int(getattr(record, "offset", 0)))
        while remaining:
            chunk = source.read(min(32 * 1024 * 1024, remaining))
            if not chunk:
                raise EOFError(
                    f"truncated MFQ blob source for {record.name}: "
                    f"{remaining} bytes missing"
                )
            target.write(chunk)
            remaining -= len(chunk)


def write_blob_record_shards(
    output: str | Path,
    header: FileHeader,
    records: Sequence[BlobRecordLike],
    *,
    split_max_size: int = 0,
    split_max_tensors: int = 0,
    overwrite: bool = False,
    consume_blobs: bool = False,
) -> list[Path]:
    """Stage all output files, then publish an MFQ file or shard set."""

    planned = plan_record_shards(
        records,
        split_max_size=split_max_size,
        split_max_tensors=split_max_tensors,
    )
    sharded_output = bool(split_max_size or split_max_tensors)
    if len(planned) == 1 and not sharded_output:
        destinations = [Path(output)]
    else:
        destinations = [
            format_shard_path(output, index, len(planned))
            for index in range(1, len(planned) + 1)
        ]
    stale_outputs = set(matching_shard_paths(output))
    if not sharded_output and Path(output).exists():
        stale_outputs.add(Path(output))
    existing = sorted(stale_outputs)
    if existing and not overwrite:
        raise FileExistsError(f"MFQ output exists: {existing[0]}")
    temporary = [path.with_suffix(path.suffix + ".tmp") for path in destinations]
    stale_tmp = [path for path in temporary if path.exists()]
    if stale_tmp:
        raise FileExistsError(f"temporary MFQ output exists: {stale_tmp[0]}")

    tensor_count = sum(not is_asset_record(record.name) for record in records)
    record_count = len(records)
    try:
        for split_no, (destination, tmp, shard_records) in enumerate(
            zip(destinations, temporary, planned, strict=True)
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            part_header = (
                shard_header(
                    header,
                    split_no=split_no,
                    split_count=len(destinations),
                    tensor_count=tensor_count,
                    record_count=record_count,
                )
                if sharded_output
                else header
            )
            with tmp.open("xb") as target:
                _write_header_and_table(target, part_header, shard_records)
                for record in shard_records:
                    _copy_record(record, target)
            if tmp.stat().st_size <= 0:
                raise RuntimeError(f"empty MFQ shard output: {tmp}")

        for tmp, destination in zip(temporary, destinations, strict=True):
            os.replace(tmp, destination)
        for stale in stale_outputs.difference(destinations):
            stale.unlink()
        if consume_blobs:
            for record in records:
                Path(record.path).unlink(missing_ok=True)
    except Exception:
        for tmp in temporary:
            tmp.unlink(missing_ok=True)
        raise
    return destinations


def parse_size(value: str) -> int:
    """Parse an integer byte count or llama.cpp-style decimal K/M/G suffix."""

    match = re.fullmatch(r"\s*([0-9]+)\s*([KMGkmg]?)\s*", value)
    if match is None:
        raise ValueError(f"invalid split size: {value!r}")
    scale = {"": 1, "k": 1000, "m": 1000**2, "g": 1000**3}
    result = int(match.group(1)) * scale[match.group(2).lower()]
    if result <= 0:
        raise ValueError("split size must be positive")
    return result
