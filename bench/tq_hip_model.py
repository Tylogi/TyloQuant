"""Assemble real fitted TQ rows for a native speed comparison at fixed budgets.

This deterministic row assignment measures execution speed, not allocation
quality. Other tensor formats and model assets come from the supplied base.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import numpy as np

from mfq.architectures.tensor_schema import map_source_tensor_name
from mfq.formats.io import MFQ_MAGIC, load_mmap
from mfq.formats.tq import TQTensor


def row_bits(rows, total, seed):
    mean = total / rows
    levels = np.arange(1, 9, dtype=np.float64)
    lo, hi = -40., 40.
    for _ in range(64):
        slope = (lo + hi) / 2
        p = np.exp(slope * levels - (slope * levels).max())
        p /= p.sum()
        if float(p @ levels) < mean:
            lo = slope
        else:
            hi = slope
    rng = np.random.default_rng(seed)
    bits = rng.choice(np.arange(1, 9, dtype=np.uint8), rows, p=p)
    delta = total - int(bits.sum(dtype=np.int64))
    while delta:
        available = np.flatnonzero(bits < 8 if delta > 0 else bits > 1)
        chosen = rng.choice(available, min(abs(delta), len(available)), replace=False)
        if delta > 0:
            bits[chosen] += 1
            delta -= len(chosen)
        else:
            bits[chosen] -= 1
            delta += len(chosen)
    return bits


def select_rows(bank, filename, budget, seed):
    profiles = [TQTensor.load(bank / f'Q{k}' / filename) for k in range(1, 9)]
    first = profiles[0]
    assert first.columns
    for tensor in profiles[1:]:
        assert (tensor.rows, tensor.width, tensor.gs, tensor.columns) == (
            first.rows, first.width, first.gs, first.columns)
        np.testing.assert_array_equal(tensor.scales, first.scales)
    metadata = 64 + first.rows * 8 + first.width * 2
    total = int(np.clip(round((budget - metadata) * 8 / first.width), first.rows, first.rows * 8))
    bits = row_bits(first.rows, total, seed)
    descriptors = np.empty(first.rows, dtype=np.uint64)
    payload = np.empty(first.width * total // 8, dtype=np.uint8)
    offset = 0
    for row, k in enumerate(bits):
        tensor = profiles[int(k) - 1]
        descriptor = int(tensor.descriptors[row])
        begin = (descriptor & 0xffffffff) * 4
        count = first.width * int(k) // 8
        descriptors[row] = (descriptor & ~0xffffffff) | (offset // 4)
        payload[offset:offset + count] = tensor.packed[begin:begin + count]
        offset += count
    return TQTensor(first.rows, first.width, first.gs, True, descriptors, first.scales, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    report = json.loads((args.bank / 'report.json').read_text())
    assert report['verified'] and not report['failures']
    jobs = {e['source']: e for e in report['entries'] if e['bits'] == 4}
    header, store = load_mmap(args.base)
    replacements, shapes, records = {}, {}, []
    for index, (name, entry) in enumerate(sorted(jobs.items())):
        mapping = map_source_tensor_name(name, header.extra['hf_config'], require_registered=True)
        target = mapping.canonical_name
        if target not in store.records:
            raise ValueError('Base has no projection ' + target)
        base = store.records[target]
        tensor = select_rows(args.bank, name + '.tq', base.nbytes, 1000 + index)
        payload = tensor.to_bytes()
        replacements[target] = payload
        shapes[target] = entry['shape']
        records.append(dict(source=name, name=target, shape=entry['shape'], base_bytes=base.nbytes,
                            tq_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                            bits_histogram={str(k):int((tensor.bits == k).sum()) for k in range(1, 9)}))
        if index % 32 == 31:
            print(json.dumps(dict(assembled=index + 1, total=len(jobs), seconds=time.perf_counter()-start)), flush=True)
    extra = dict(header.extra, tq_shapes=shapes,
                 tq_speed_benchmark_only=True,
                 tq_precision_policy='Deterministic real Q1..Q8 rows matching each base projection byte budget; allocation quality unevaluated')
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    u32 = lambda n: struct.pack('<I', n)
    def string(value):
        value = value.encode()
        return u32(len(value)) + value
    with temporary.open('wb') as f:
        f.write(MFQ_MAGIC + u32(max(2, header.version)) + string(header.model_arch))
        f.write(u32(len(extra)))
        for key, value in extra.items():
            f.write(string(str(key)) + string(json.dumps(value)))
        f.write(u32(len(store.records)))
        for name, record in store.records.items():
            dtype = 'TQ' if name in replacements else record.dtype
            nbytes = len(replacements[name]) if name in replacements else record.nbytes
            f.write(string(name) + string(dtype) + struct.pack('<Q', nbytes))
        for name, record in store.records.items():
            if name in replacements:
                f.write(replacements.pop(name))
            else:
                data = store._mmaps[record.source_index]
                for offset in range(0, record.nbytes, 8 << 20):
                    f.write(data[record.offset + offset:record.offset + min(offset + (8 << 20), record.nbytes)])
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, output)
    result = dict(model=str(output), seconds=time.perf_counter()-start,
                  file_bytes=output.stat().st_size, base_file_bytes=args.base.stat().st_size,
                  projections=len(records), parameters=sum(np.prod(r['shape']).item() for r in records),
                  base_projection_bytes=sum(r['base_bytes'] for r in records),
                  tq_projection_bytes=sum(r['tq_bytes'] for r in records),
                  allocation_quality_evaluated=False, entries=records)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'entries'}), flush=True)
    store.close()


if __name__ == '__main__':
    main()
