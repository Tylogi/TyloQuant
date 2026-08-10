"""Stream an unquantized/native-precision HF checkpoint into full-precision MFQ.

BF16/F16/F32/integer tensors retain their exact bytes.  Native block-scaled
FP8 and MXFP4 weight/scale pairs are fused into one self-contained MFQ record.
No numerical conversion or MFQ quantization is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path

from mfq.formats.assets import (
    ASSET_DTYPE,
    ASSET_MANIFEST_KEY,
    gguf_metadata_asset,
    model_config_asset,
    runtime_asset_manifest,
)
from mfq.formats.header import FileHeader
from mfq.formats.mx import MXFP4_DTYPE, MXFP8_DTYPE, mx_header_bytes
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

_DENSE_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8}
_FP8_DTYPES = frozenset({"F8_E4M3", "F8_E4M3FN"})


@dataclass(frozen=True)
class SafeTensorEntry:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    data_offset: int
    nbytes: int


@dataclass(frozen=True)
class FullPrecisionPlan:
    name: str
    dtype: str
    shape: tuple[int, ...]
    values: SafeTensorEntry
    scales: SafeTensorEntry | None = None


@dataclass(frozen=True)
class BlobRecord:
    name: str
    dtype: str
    nbytes: int
    path: Path


def _read_header(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors header length: {path}")
        header_nbytes = struct.unpack("<Q", raw)[0]
        header_raw = handle.read(header_nbytes)
    if len(header_raw) != header_nbytes:
        raise ValueError(f"truncated safetensors header: {path}")
    value = json.loads(header_raw)
    if not isinstance(value, dict):
        raise ValueError(f"invalid safetensors header: {path}")
    value["__mfq_data_offset__"] = 8 + header_nbytes
    return value


def _discover_entries(root: Path) -> tuple[dict[str, SafeTensorEntry], bool]:
    index_path = root / "model.safetensors.index.json"
    indexed = index_path.is_file()
    if indexed:
        document = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = document.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors index: {index_path}")
        shard_paths = sorted({root / str(value) for value in weight_map.values()})
        missing = [path for path in shard_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"indexed safetensors shard is missing: {missing[0]}")
    else:
        shard_paths = sorted(root.glob("*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(f"no complete safetensors shards under {root}")
        warnings.warn(
            "checkpoint has no model.safetensors.index.json; converting the "
            "currently complete shards as an explicitly partial full-precision MFQ",
            stacklevel=2,
        )

    entries: dict[str, SafeTensorEntry] = {}
    for path in shard_paths:
        header = _read_header(path)
        data_offset = int(header.pop("__mfq_data_offset__"))
        for name, raw in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"invalid tensor entry {name}: {path}")
            dtype = str(raw.get("dtype", ""))
            shape = tuple(int(value) for value in raw.get("shape", ()))
            offsets = raw.get("data_offsets")
            if not shape or not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"invalid safetensors tensor {name}: {path}")
            begin, end = (int(value) for value in offsets)
            if begin < 0 or end <= begin:
                raise ValueError(f"invalid safetensors offsets for {name}: {path}")
            if name in entries:
                raise ValueError(f"duplicate safetensors tensor: {name}")
            entries[name] = SafeTensorEntry(
                name=name,
                path=path,
                dtype=dtype,
                shape=shape,
                data_offset=data_offset + begin,
                nbytes=end - begin,
            )
    return entries, not indexed


def _scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"native MX tensor is not named *.weight: {weight_name}")
    return weight_name.removesuffix(".weight") + ".scale"


def _build_plan(entries: dict[str, SafeTensorEntry]) -> list[FullPrecisionPlan]:
    consumed_scales: set[str] = set()
    plans: list[FullPrecisionPlan] = []
    for name, entry in entries.items():
        if entry.dtype == "F8_E8M0":
            continue
        if entry.dtype in _DENSE_ITEMSIZE:
            expected = _DENSE_ITEMSIZE[entry.dtype]
            for dimension in entry.shape:
                expected *= dimension
            if entry.nbytes != expected:
                raise ValueError(f"dense byte size mismatch for {name}")
            plans.append(FullPrecisionPlan(name, entry.dtype, entry.shape, entry))
            continue
        if entry.dtype in _FP8_DTYPES or entry.dtype == "I8":
            scale_name = _scale_name(name)
            try:
                scales = entries[scale_name]
            except KeyError as exc:
                raise ValueError(f"native MX tensor has no E8M0 scale: {name}") from exc
            if scales.dtype != "F8_E8M0":
                raise ValueError(
                    f"native MX scale {scale_name} is {scales.dtype}, expected F8_E8M0"
                )
            if entry.path != scales.path:
                raise ValueError(f"native MX weight and scale are split: {name}")
            if len(entry.shape) != 2 or len(scales.shape) != 2:
                raise ValueError(f"native MX tensor must be rank 2: {name}")
            if entry.dtype == "I8":
                dtype = MXFP4_DTYPE
                shape = (entry.shape[0], entry.shape[1] * 2)
            else:
                dtype = MXFP8_DTYPE
                shape = entry.shape
            mx_header_bytes(dtype, shape, entry.shape, scales.shape)
            plans.append(FullPrecisionPlan(name, dtype, shape, entry, scales))
            consumed_scales.add(scale_name)
            continue
        raise ValueError(f"unsupported full-precision HF dtype {entry.dtype}: {name}")
    orphan_scales = sorted(
        name
        for name, entry in entries.items()
        if entry.dtype == "F8_E8M0" and name not in consumed_scales
    )
    if orphan_scales:
        raise ValueError(f"orphan E8M0 scale tensor: {orphan_scales[0]}")
    plans.sort(key=lambda item: (str(item.values.path), item.values.data_offset))
    return plans


def _copy_range(source: SafeTensorEntry, target) -> None:
    remaining = source.nbytes
    with source.path.open("rb", buffering=0) as handle:
        handle.seek(source.data_offset)
        while remaining:
            chunk = handle.read(min(32 * 1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"short safetensors read for {source.name}")
            target.write(chunk)
            remaining -= len(chunk)


def _write_plan_blob(item: FullPrecisionPlan, path: Path) -> int:
    with path.open("xb") as target:
        if item.dtype in {MXFP4_DTYPE, MXFP8_DTYPE}:
            assert item.scales is not None
            target.write(
                mx_header_bytes(
                    item.dtype,
                    (int(item.shape[0]), int(item.shape[1])),
                    (int(item.values.shape[0]), int(item.values.shape[1])),
                    (int(item.scales.shape[0]), int(item.scales.shape[1])),
                )
            )
            _copy_range(item.values, target)
            _copy_range(item.scales, target)
        else:
            target.write(struct.pack("<I", len(item.shape)))
            target.write(struct.pack(f"<{len(item.shape)}q", *item.shape))
            _copy_range(item.values, target)
    return path.stat().st_size


def _plan_blob_nbytes(item: FullPrecisionPlan) -> int:
    if item.dtype in {MXFP4_DTYPE, MXFP8_DTYPE}:
        assert item.scales is not None
        header = mx_header_bytes(
            item.dtype,
            (int(item.shape[0]), int(item.shape[1])),
            (int(item.values.shape[0]), int(item.values.shape[1])),
            (int(item.scales.shape[0]), int(item.scales.shape[1])),
        )
        return len(header) + item.values.nbytes + item.scales.nbytes
    return 4 + 8 * len(item.shape) + item.values.nbytes


def _gguf_reader(path: Path):
    from mfq.tools.quantize_hf_to_mfq import _gguf_reader as reader

    return reader(path)


def _selection(plans: list[FullPrecisionPlan], args) -> list[FullPrecisionPlan]:
    patterns = [re.compile(value) for value in getattr(args, "tensor_pattern", ())]
    selected = (
        [item for item in plans if any(pattern.search(item.name) for pattern in patterns)]
        if patterns
        else plans
    )
    limit = int(getattr(args, "limit_tensors", 0))
    return selected[:limit] if limit else selected


def convert(args: argparse.Namespace) -> list[Path]:
    root = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"HF source is not a directory: {root}")
    split_max_size = int(getattr(args, "split_max_size", 0))
    split_max_tensors = int(getattr(args, "split_max_tensors", 0))
    validate_split_limits(split_max_size, split_max_tensors)
    if (output.exists() or matching_shard_paths(output)) and not args.overwrite:
        raise FileExistsError(f"output exists: {output}")
    entries, partial_source = _discover_entries(root)
    plans = _selection(_build_plan(entries), args)
    if not plans:
        raise ValueError("full-precision conversion selected no tensors")

    counts: dict[str, int] = {}
    for item in plans:
        counts[item.dtype] = counts.get(item.dtype, 0) + 1
    print(
        json.dumps(
            {
                "event": "full_precision_mfq_plan",
                "input": str(root),
                "output": str(output),
                "partial_source": partial_source,
                "tensors": len(plans),
                "dtypes": counts,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if getattr(args, "dry_run", False):
        return []

    temp_arg = getattr(args, "temp_dir", "")
    temp_root = (
        Path(temp_arg).resolve()
        if temp_arg
        else output.parent / f".{output.name}.full-precision-blobs"
    )
    resume = bool(getattr(args, "resume", False))
    if temp_root.exists() and not resume:
        raise FileExistsError(f"temporary directory exists: {temp_root}")
    temp_root.mkdir(parents=True, exist_ok=resume)
    records: list[BlobRecord] = []
    completed = False
    try:
        for index, item in enumerate(plans):
            blob_path = temp_root / f"{index:06d}.blob"
            expected_nbytes = _plan_blob_nbytes(item)
            if (
                blob_path.is_file()
                and resume
                and blob_path.stat().st_size == expected_nbytes
            ):
                nbytes = blob_path.stat().st_size
                status = "reused"
            else:
                if blob_path.exists():
                    blob_path.unlink()
                nbytes = _write_plan_blob(item, blob_path)
                status = "written"
            if nbytes != expected_nbytes:
                raise RuntimeError(
                    f"full-precision blob size mismatch for {item.name}: "
                    f"{nbytes} != {expected_nbytes}"
                )
            records.append(BlobRecord(item.name, item.dtype, nbytes, blob_path))
            print(
                json.dumps(
                    {
                        "done": index + 1,
                        "total": len(plans),
                        "name": item.name,
                        "dtype": item.dtype,
                        "shape": item.shape,
                        "status": status,
                    }
                ),
                flush=True,
            )

        config_path = Path(getattr(args, "model_config", "") or root / "config.json")
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        assets = [model_config_asset(config)] if config else []
        tokenizer_arg = getattr(args, "tokenizer_gguf", "")
        if tokenizer_arg:
            assets.append(gguf_metadata_asset(_gguf_reader(Path(tokenizer_arg).resolve())))
        for index, asset in enumerate(assets):
            path = temp_root / f"asset-{index:02d}.blob"
            path.write_bytes(asset.data)
            records.append(BlobRecord(asset.name, ASSET_DTYPE, len(asset.data), path))

        fingerprint = hashlib.sha256()
        for item in plans:
            fingerprint.update(item.name.encode("utf-8"))
            fingerprint.update(item.dtype.encode("ascii"))
            fingerprint.update(str(item.shape).encode("ascii"))
        model_type = str(config.get("model_type", "unknown"))
        extra = {
            "full_precision_mfq": True,
            "source_format": "hf-safetensors",
            "source_name": root.name,
            "partial_source": partial_source,
            "tensor_manifest_sha256": fingerprint.hexdigest(),
            "native_dtypes": counts,
            "hf_config": config,
            ASSET_MANIFEST_KEY: runtime_asset_manifest(assets),
        }
        runtime_profile = profile_for_new_mfq(
            root,
            config,
            explicit_profile=getattr(args, "sampling_profile", "") or None,
        )
        if runtime_profile is not None:
            extra[RUNTIME_SAMPLING_METADATA_KEY] = runtime_profile
        header = FileHeader(
            version=2,
            model_arch=f"{model_type}-hf-full-mfq",
            num_tensors=len(records),
            extra=extra,
        )
        outputs = write_blob_record_shards(
            output,
            header,
            records,
            split_max_size=split_max_size,
            split_max_tensors=split_max_tensors,
            overwrite=bool(args.overwrite),
        )
        completed = True
        return outputs
    finally:
        if not getattr(args, "keep_temp", False) and temp_root.exists() and completed:
            shutil.rmtree(temp_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-config", default="")
    parser.add_argument("--sampling-profile", default="")
    parser.add_argument("--tokenizer-gguf", default="")
    parser.add_argument("--tensor-pattern", action="append", default=[])
    parser.add_argument("--limit-tensors", type=int, default=0)
    parser.add_argument("--temp-dir", default="")
    parser.add_argument("--resume", action="store_true")
    split = parser.add_mutually_exclusive_group()
    split.add_argument("--split-max-size", type=parse_size, default=0)
    split.add_argument("--split-max-tensors", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    convert(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
