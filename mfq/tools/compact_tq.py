"""Remove encoded TQ tail padding from existing model shards without fitting.

Run ``python -m mfq.tools.compact_tq source.mfq destination.tq``. Process
shards independently; only one ordinary tensor or expert is copied into RAM.
Row choices, decoded weights, scales, tensor order and calibration stay fixed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import struct
import time

import numpy as np

from mfq.formats.io import _open_single_mmap
from mfq.formats.shards import _write_header_and_table
from mfq.formats.tq import TQTensor
from mfq.formats.tq_rows import TQRows


@dataclass(frozen=True)
class Plan:
    offset: int
    source_bytes: int
    nbytes: int
    width: int = 0
    children: tuple[Plan, ...] = ()


def _plan(mapping, offset: int, size: int, logical: int = 0) -> Plan:
    if offset < 0 or size < 64 or offset > len(mapping) or size > len(mapping) - offset:
        raise ValueError('TQ payload exceeds its shard')
    magic, version, rows, width, columns, gs, packed = struct.unpack_from('<8s6I', mapping, offset)
    if magic == b'RTQEXP01':
        count, outputs, width, stride = rows, width, columns, gs
        if (version != 1 or not count or not outputs or not width or stride != 16 or packed
                or count > (size - 64) // 16 or any(mapping[offset + 32:offset + 64])
                or (logical and logical != width)):
            raise ValueError('Invalid TQ expert container')
        children = []
        end = 64 + count * 16
        for index in range(count):
            begin, length = struct.unpack_from('<QQ', mapping, offset + 64 + index * 16)
            if begin != end or length < 64 or begin > size or length > size - begin:
                raise ValueError('Invalid TQ expert span')
            if mapping[offset + begin:offset + begin + 8] not in (b'RTRELL05', b'RTRELL06'):
                raise ValueError('Expected ordinary TQ expert rows')
            if struct.unpack_from('<I', mapping, offset + begin + 12)[0] != outputs:
                raise ValueError('TQ expert row count differs')
            children.append(_plan(mapping, offset + begin, length, width))
            end = begin + length
        if end != size:
            raise ValueError('TQ expert payload length differs')
        total = 64 + count * 16 + sum(child.nbytes for child in children)
        return Plan(offset, size, total, width, tuple(children))
    rows_view = TQRows(size, lambda begin, length: mapping[offset + begin:offset + begin + length])
    logical = logical or rows_view.width
    if not 0 < logical <= rows_view.width:
        raise ValueError('Invalid TQ logical width')
    if logical == rows_view.width:
        return Plan(offset, size, size)
    if (not rows_view.columns or logical % 32 or not logical % 256
            or rows_view.width != (logical + 255) // 256 * 256):
        raise ValueError('Unsupported TQ compact tail geometry')
    descriptors = np.frombuffer(mapping, '<u8', rows_view.rows, offset + 64)
    bits = ((descriptors >> 32) & 7) + 1
    words = logical * bits // 32
    if np.any(words & 1) or int(words.sum()) >= 2**31:
        raise ValueError('Compact TQ rows require even word offsets below 2**31')
    total = 64 + rows_view.rows * 8 + logical * 2 + int(words.sum()) * 4
    return Plan(offset, size, total, logical)


def _copy(mapping, output, offset: int, size: int) -> None:
    with memoryview(mapping) as view:
        for begin in range(offset, offset + size, 8 * 1024 * 1024):
            output.write(view[begin:min(begin + 8 * 1024 * 1024, offset + size)])


def _write(mapping, output, plan: Plan) -> int:
    start = output.tell()
    changed = 0
    if plan.nbytes == plan.source_bytes:
        _copy(mapping, output, plan.offset, plan.source_bytes)
    elif plan.children:
        output.write(mapping[plan.offset:plan.offset + 64])
        position = 64 + len(plan.children) * 16
        for child in plan.children:
            output.write(struct.pack('<QQ', position, child.nbytes))
            position += child.nbytes
        for child in plan.children:
            changed += _write(mapping, output, child)
    else:
        with memoryview(mapping)[plan.offset:plan.offset + plan.source_bytes] as source:
            tensor = TQTensor.from_bytes(source).compact_tail(plan.width)
        output.write(tensor.to_bytes())
        changed = 1
    if output.tell() - start != plan.nbytes:
        raise ValueError('Compacted TQ payload length differs from its plan')
    return changed


def repack_file(source: str | Path, destination: str | Path) -> dict:
    source, destination = Path(source), Path(destination)
    if destination.suffix.lower() != '.tq':
        raise ValueError('Compacted model shards use the .tq extension')
    if source.resolve() == destination.resolve() or destination.exists():
        raise FileExistsError(destination)
    started = time.monotonic()
    header, records, handle, mapping = _open_single_mmap(source)
    try:
        shapes = header.extra.get('tq_shapes', {})
        if isinstance(shapes, str):
            shapes = json.loads(shapes)
        plans = []
        target_records = []
        for record in records.values():
            if record.dtype == 'TQ':
                shape = shapes.get(record.name)
                plan = _plan(mapping, record.offset, record.nbytes, int(shape[-1]) if shape else 0)
            else:
                plan = Plan(record.offset, record.nbytes, record.nbytes)
            plans.append(plan)
            target_records.append(replace(record, nbytes=plan.nbytes))
        if any(plan.nbytes != plan.source_bytes for plan in plans):
            key = 'routed_expert_layout'
            if key in header.extra:
                header.extra[key] = str(header.extra[key]).replace('RTRELL05', 'RTRELL05/RTRELL06')
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + '.partial')
        with partial.open('xb') as output:
            _write_header_and_table(output, header, target_records)
            expected = output.tell() + sum(plan.nbytes for plan in plans)
            changed = sum(_write(mapping, output, plan) for plan in plans)
            if output.tell() != expected:
                raise ValueError('Compacted shard size differs')
        partial.replace(destination)
        seconds = time.monotonic() - started
        return dict(source=str(source), destination=str(destination), compacted_tensors=changed,
                    source_bytes=source.stat().st_size, output_bytes=expected, seconds=seconds,
                    source_MBps=source.stat().st_size / max(seconds, 1e-9) / 1e6)
    finally:
        mapping.close()
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(repack_file(args.source, args.destination)))


if __name__ == '__main__':
    main()
