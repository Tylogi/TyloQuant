"""Embed model configuration and GGUF tokenizer metadata into an MFQ file."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

from mfq.formats.assets import (
    ASSET_DTYPE,
    ASSET_MANIFEST_KEY,
    MODEL_CONFIG_ASSET,
    TOKENIZER_GGUF_ASSET,
    gguf_metadata_asset,
    is_asset_record,
    model_config_asset,
    runtime_asset_manifest,
)
from mfq.formats.io import MMapTensorRecord, open_mmap

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")


def _load_gguf_reader():
    try:
        from gguf import GGUFReader  # type: ignore
        return GGUFReader
    except ImportError:
        gguf_py = (
            Path(__file__).resolve().parents[2]
            / "references"
            / "llamacpp"
            / "gguf-py"
        )
        if not gguf_py.exists():
            raise
        sys.path.insert(0, str(gguf_py))
        from gguf import GGUFReader  # type: ignore
        return GGUFReader


def _encoded_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return _U32.pack(len(data)) + data


def _file_table(
    *,
    version: int,
    model_arch: str,
    extra: dict,
    records: list[tuple[str, str, int]],
) -> bytes:
    version = max(2, int(version))
    parts = [b"MFQ1", _U32.pack(version), _encoded_string(model_arch)]
    parts.append(_U32.pack(len(extra)))
    for key, value in extra.items():
        parts.append(_encoded_string(str(key)))
        parts.append(
            _encoded_string(
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            )
        )
    parts.append(_U32.pack(len(records)))
    for name, dtype, nbytes in records:
        parts.extend(
            (
                _encoded_string(name),
                _encoded_string(dtype),
                _U64.pack(nbytes),
            )
        )
    return b"".join(parts)


def _copy_span(
    source,
    destination,
    record: MMapTensorRecord,
    *,
    chunk_bytes: int = 32 << 20,
) -> None:
    source.seek(record.offset)
    remaining = record.nbytes
    while remaining:
        data = source.read(min(remaining, chunk_bytes))
        if not data:
            raise EOFError(f"truncated MFQ record: {record.name}")
        destination.write(data)
        remaining -= len(data)


def pack_runtime_assets(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path,
    tokenizer_gguf: str | Path,
    overwrite: bool = False,
) -> Path:
    source_path = Path(input_path).resolve()
    output = Path(output_path).resolve()
    config = model_config_asset(Path(config_path).read_bytes())
    GGUFReader = _load_gguf_reader()
    reader = GGUFReader(Path(tokenizer_gguf).resolve())
    tokenizer = gguf_metadata_asset(reader)
    del reader
    assets = (config, tokenizer)

    same_path = source_path == output
    if output.exists() and not same_path and not overwrite:
        raise FileExistsError(f"output MFQ already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".assets.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")

    try:
        with open_mmap(source_path) as store:
            if len(store.paths) != 1:
                raise ValueError(
                    "pack_runtime_assets requires a single-file MFQ input; "
                    "pack assets before splitting"
                )
            kept = [
                record
                for record in store.records.values()
                if not is_asset_record(record.name)
            ]
            extra = dict(store.header.extra)
            extra[ASSET_MANIFEST_KEY] = runtime_asset_manifest(assets)
            records = [
                (record.name, record.dtype, record.nbytes)
                for record in kept
            ]
            records.extend(
                (asset.name, ASSET_DTYPE, len(asset.data))
                for asset in assets
            )
            with source_path.open("rb") as source, temporary.open("wb") as dest:
                dest.write(
                    _file_table(
                        version=store.header.version,
                        model_arch=store.header.model_arch,
                        extra=extra,
                        records=records,
                    )
                )
                for record in kept:
                    _copy_span(source, dest, record)
                for asset in assets:
                    dest.write(asset.data)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-gguf", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = pack_runtime_assets(
        args.input,
        args.output,
        config_path=args.config,
        tokenizer_gguf=args.tokenizer_gguf,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {"output": str(output), "bytes": output.stat().st_size},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
