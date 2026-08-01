"""Split one complete MFQ file into llama.cpp-style numbered shards."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from mfq.formats.io import open_mmap
from mfq.formats.shards import (
    format_shard_path,
    parse_size,
    plan_record_shards,
    validate_split_limits,
    write_blob_record_shards,
)


@dataclass(frozen=True)
class FileSpanRecord:
    name: str
    dtype: str
    nbytes: int
    path: Path
    offset: int


def split_mfq(
    input_path: str | Path,
    output_path: str | Path,
    *,
    split_max_size: int = 0,
    split_max_tensors: int = 0,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    validate_split_limits(
        split_max_size,
        split_max_tensors,
        required=True,
    )
    with open_mmap(source) as store:
        if len(store.paths) != 1:
            raise ValueError("split_mfq input must be one complete MFQ file")
        records = [
            FileSpanRecord(
                name=record.name,
                dtype=record.dtype,
                nbytes=record.nbytes,
                path=store.paths[record.source_index],
                offset=record.offset,
            )
            for record in store.records.values()
        ]
        planned = plan_record_shards(
            records,
            split_max_size=split_max_size,
            split_max_tensors=split_max_tensors,
        )
        if dry_run:
            return [
                format_shard_path(output, index, len(planned))
                for index in range(1, len(planned) + 1)
            ]
        return write_blob_record_shards(
            output,
            store.header,
            records,
            split_max_size=split_max_size,
            split_max_tensors=split_max_tensors,
            overwrite=overwrite,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="complete single-file MFQ")
    parser.add_argument("--output", required=True, help="output MFQ base path")
    limits = parser.add_mutually_exclusive_group(required=True)
    limits.add_argument(
        "--split-max-size",
        type=parse_size,
        default=0,
        metavar="N[M|G]",
        help="maximum tensor payload per shard, using decimal suffixes",
    )
    limits.add_argument(
        "--split-max-tensors",
        type=int,
        default=0,
        help="maximum non-asset tensor records per shard",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = split_mfq(
        args.input,
        args.output,
        split_max_size=args.split_max_size,
        split_max_tensors=args.split_max_tensors,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "status": "dry-run" if args.dry_run else "ok",
                "shard_count": len(paths),
                "outputs": [str(path) for path in paths],
                "bytes": (
                    None
                    if args.dry_run
                    else sum(path.stat().st_size for path in paths)
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
