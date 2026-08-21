"""Rewrite compatible MFQ tensors into native execution storage layouts."""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, replace
from pathlib import Path

from mfq.formats.header import FileHeader
from mfq.formats.io import MMapTensorRecord, open_mmap
from mfq.formats.nvq import NvqJscTensor, pack_nvq, unpack_nvq
from mfq.formats.shards import SPLIT_KEYS, _write_header_and_table


@dataclass(frozen=True)
class LayoutRecord:
    name: str
    dtype: str
    nbytes: int
    source: MMapTensorRecord
    rewrite: bool = False


def _nvq2j_xl_layout(record: MMapTensorRecord, blob: memoryview) -> tuple[str, int]:
    if record.dtype != "NVQ2J-XL" or len(blob) < 40:
        return "unchanged", record.nbytes
    ndim = struct.unpack_from("<I", blob, 16)[0]
    if ndim == 0 or ndim > 8:
        raise ValueError(f"{record.name}: invalid NVQ ndim")
    metadata = 24 + ndim * 8
    if metadata + 64 > len(blob):
        raise ValueError(f"{record.name}: truncated NVQ-JSC metadata")
    version = int(blob[metadata])
    layout = int(blob[metadata + 52])
    if version == 2 and layout == 1:
        return "group64", record.nbytes
    if version != 1 or layout != 0:
        raise ValueError(
            f"{record.name}: unsupported NVQ-JSC metadata version/layout "
            f"{version}/{layout}"
        )
    neuron_len = struct.unpack_from("<i", blob, 12)[0]
    out = struct.unpack_from("<I", blob, 20 + ndim * 8)[0]
    groups = (neuron_len + 23) // 24
    vectors = (neuron_len + 7) // 8
    old_streams = (
        (out * groups * 4 + 7) // 8
        + (out * vectors * 12 + 7) // 8
        + (out * vectors * 7 + 7) // 8
    )
    new_streams = out * groups * 8
    return "streams", record.nbytes - old_streams + new_streams


def _copy_blob(store, record: MMapTensorRecord, target) -> None:
    source = store.mmap_for(record)
    remaining = record.nbytes
    offset = record.offset
    while remaining:
        count = min(32 * 1024 * 1024, remaining)
        target.write(source[offset : offset + count])
        offset += count
        remaining -= count


def optimize_layouts(
    source: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, int | str]:
    """Write a model whose NVQ2J-XL tensors use direct group64 storage."""

    source_path = Path(source)
    output_path = Path(output)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("layout optimization requires a distinct output path")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"MFQ output exists: {output_path}")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary MFQ output exists: {temporary}")

    with open_mmap(source_path) as store:
        records: list[LayoutRecord] = []
        input_bytes = 0
        output_bytes = 0
        changed = 0
        for record in store.records.values():
            input_bytes += record.nbytes
            view = store.blob_view(record)
            try:
                layout, nbytes = _nvq2j_xl_layout(record, view)
            finally:
                view.release()
            rewrite = layout == "streams"
            changed += int(rewrite)
            output_bytes += nbytes
            records.append(
                LayoutRecord(
                    record.name,
                    record.dtype,
                    nbytes,
                    record,
                    rewrite,
                )
            )

        extra = {
            key: value
            for key, value in store.header.extra.items()
            if key not in SPLIT_KEYS
        }
        if changed:
            extra["nvq2j_xl.storage_layout"] = "group64-v1"
        header = FileHeader(
            version=max(2, store.header.version),
            model_arch=store.header.model_arch,
            num_tensors=store.header.num_tensors,
            extra=extra,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("xb") as target:
                _write_header_and_table(target, header, records)
                for record in records:
                    if not record.rewrite:
                        _copy_blob(store, record.source, target)
                        continue
                    view = store.blob_view(record.source)
                    try:
                        tensor = unpack_nvq(view)
                    finally:
                        view.release()
                    if not isinstance(tensor, NvqJscTensor):
                        raise TypeError(f"{record.name}: expected NVQ-JSC tensor")
                    payload = pack_nvq(
                        replace(tensor, storage_layout="group64")
                    )
                    if len(payload) != record.nbytes:
                        raise RuntimeError(
                            f"{record.name}: planned {record.nbytes} bytes, "
                            f"packed {len(payload)}"
                        )
                    target.write(payload)
            os.replace(temporary, output_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    return {
        "input": str(source_path),
        "output": str(output_path),
        "changed_tensors": changed,
        "input_payload_bytes": input_bytes,
        "output_payload_bytes": output_bytes,
    }


def run(args) -> int:
    result = optimize_layouts(
        args.input,
        args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0
