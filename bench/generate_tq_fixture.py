"""Create a thin, explicitly synthetic TQ model from an existing MFQ model.

No old-format arithmetic is used: original file bytes only back the arbitrary
Trellis state stream. Precision and scale metadata are newly generated. The
manifest preserves full model shapes without duplicating hundreds of GiB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mfq.formats.io import open_mmap


def shape_from_prefix(dtype: str, prefix: bytes) -> list[int]:
    if dtype == 'MFE':
        return list(struct.unpack_from('<III', prefix, 4))
    if dtype == 'NINT':
        offset = 14
    elif dtype == 'NINT8-0':
        offset = 12
    elif dtype in ('F32', 'F16', 'BF16', 'I64', 'I32', 'F8_E4M3'):
        offset = 0
    elif dtype in ('MXFP8', 'MXFP4'):
        return list(struct.unpack_from('<QQ', prefix, 8))
    else:
        raise ValueError(f'Unsupported reference format: {dtype}')
    rank = struct.unpack_from('<I', prefix, offset)[0]
    if not 1 <= rank <= 5:
        raise ValueError(f'Invalid reference shape rank: {rank}')
    return list(struct.unpack_from(f'<{rank}q', prefix, offset+4))


def histogram(rows: int, bit_sum: int) -> list[int]:
    """Maximum-entropy histogram, with exact row count and bit sum."""
    levels = np.arange(1, 9, dtype=np.float64)
    if bit_sum in (rows, rows*8):
        result = [0]*8
        result[bit_sum//rows-1] = rows
        return result
    lo, hi = -40., 40.
    for _ in range(64):
        slope = (lo+hi)/2
        probabilities = np.exp(slope*levels-max(slope*levels))
        probabilities /= probabilities.sum()
        if probabilities@levels < bit_sum/rows:
            lo = slope
        else:
            hi = slope
    expected = probabilities*rows
    counts = np.floor(expected).astype(np.int64)
    order = np.argsort(-(expected-counts), kind='stable')
    counts[order[:rows-int(counts.sum())]] += 1
    delta = bit_sum-int(counts@levels)
    while delta:
        candidates = np.flatnonzero(counts[:-1] if delta > 0 else counts[1:])
        index = int(candidates[len(candidates)//2]) + (0 if delta > 0 else 1)
        counts[index] -= 1
        counts[index+(1 if delta > 0 else -1)] += 1
        delta += -1 if delta > 0 else 1
    return counts.tolist()


def geometry(shape: list[int], gs: int) -> tuple[int, int, int, int]:
    rows, columns = math.prod(shape[:-1]), shape[-1]
    width = (columns+255)//256*256
    # Every segment is an ordinary RTRELL05 tensor, including at 8 bits/row.
    per_segment = 0xffffffff//width
    segments = (rows+per_segment-1)//per_segment
    metadata = rows*8+(rows*((width+gs-1)//gs)+1)//2+segments*64
    return rows, width, per_segment, metadata


def generate(reference: Path, destination: Path, *, seed: int = 1, groupsize: int = 512) -> dict:
    if groupsize < 16 or groupsize > 2**20 or groupsize & (groupsize-1):
        raise ValueError('Invalid TQ subgroup size')
    with open_mmap(reference) as source:
        entries = {}
        removed = []
        for name, record in source.records.items():
            if record.dtype == 'BLOB':
                continue
            prefix = source.mmap_for(record)[record.offset:record.offset+min(96,record.nbytes)]
            shape = shape_from_prefix(record.dtype, prefix)
            quantized = record.dtype not in ('BF16','F16','F32','I64','I32','F8_E4M3')
            # Learned position tables are read as floating parameters for
            # interpolation/addition. They do not execute a projection.
            matrix = (name.endswith('.weight') and not name.endswith('.position_embedding.weight')
                      and (len(shape)==2 or '.experts.' in name))
            if not quantized and not matrix:
                continue
            projections = [(name, shape, record.nbytes)]
            if name.endswith('.experts.gate_up.weight'):
                if len(shape) != 3 or shape[-2] % 2:
                    raise ValueError(f'Invalid combined expert shape: {name}')
                separated_shape = [shape[0], shape[1]//2, shape[2]]
                projections = [
                    (name.replace('.gate_up.weight', '.gate.weight'), separated_shape, record.nbytes//2),
                    (name.replace('.gate_up.weight', '.up.weight'), separated_shape, record.nbytes-record.nbytes//2)]
                if any(key in source.records for key, _, _ in projections):
                    raise ValueError(f'Ambiguous combined and separate expert banks: {name}')
                removed.append(name)
            for key, projection_shape, original_bytes in projections:
                rows, width, max_rows, metadata = geometry(projection_shape, groupsize)
                minimum, maximum = metadata+rows*width//8, metadata+rows*width
                entries[key] = dict(shape=projection_shape, rows=rows, width=width, max_rows=max_rows,
                    metadata=metadata, original_bytes=original_bytes, minimum=minimum, maximum=maximum,
                    target=min(max(original_bytes,minimum),maximum))
        target_total = sum(v['original_bytes'] for v in entries.values())
        if not sum(v['minimum'] for v in entries.values()) <= target_total <= sum(v['maximum'] for v in entries.values()):
            raise ValueError('Reference average bpw cannot be represented by TQ 1..8-bit rows')
        delta = target_total-sum(v['target'] for v in entries.values())
        capacities = {name: (v['maximum']-v['target'] if delta>0 else v['target']-v['minimum'])
                      for name,v in entries.items()}
        remaining_capacity = sum(capacities.values())
        remaining = abs(delta)
        for name, capacity in capacities.items():
            amount = round(remaining*capacity/remaining_capacity) if remaining_capacity else 0
            entries[name]['target'] += amount if delta>0 else -amount
            remaining -= amount
            remaining_capacity -= capacity
        if remaining:
            raise AssertionError('Byte budget was not assigned')
        paths = source.paths
        sizes = [p.stat().st_size for p in paths]
        # Packed random-state bytes consume disjoint ranges from original files.
        file_index, file_offset = 0, 0
        def backing(size):
            nonlocal file_index, file_offset
            pieces = []
            while size:
                if file_index >= len(paths):
                    raise ValueError('Reference files cannot back the requested TQ state stream')
                available = sizes[file_index]-file_offset
                take = min(size, available)
                if take:
                    pieces.append(dict(file=file_index, offset=file_offset, nbytes=take))
                    file_offset += take
                    size -= take
                if file_offset == sizes[file_index]:
                    file_index += 1
                    file_offset = 0
            return pieces
        tensors = {}
        for name, item in entries.items():
            rows, width = item['rows'], item['width']
            total_bits = min(8*rows,max(rows,round((item['target']-item['metadata'])*8/width)))
            segments = []
            for first in range(0, rows, item['max_rows']):
                count = min(item['max_rows'], rows-first)
                bits = total_bits*(first+count)//rows-total_bits*first//rows
                name_hash = hashlib.sha256(f'{seed}:{name}:{first}'.encode()).digest()
                segment_seed = int.from_bytes(name_hash[:8],'little')
                histograms = {}
                # A proportional bit budget gives at most two full-block
                # histograms plus one short final block. No giant row index.
                full_blocks = count//4096
                base, remainder = divmod(bits*4096,count)
                if full_blocks:
                    histograms[f'4096:{base}'] = histogram(4096,base)
                    if remainder:
                        histograms[f'4096:{base+1}'] = histogram(4096,base+1)
                tail = count%4096
                if tail:
                    tail_bits = bits-bits*(count-tail)//count
                    histograms[f'{tail}:{tail_bits}'] = histogram(tail,tail_bits)
                packed_bytes = width*bits//8
                metadata_bytes = 64+count*8+(count*((width+groupsize-1)//groupsize)+1)//2
                segments.append(dict(rows=count,width=width,gs=groupsize,bit_sum=bits,seed=segment_seed,
                    anchor_bits=int(np.float16(.8/math.sqrt(item['shape'][-1])).view(np.uint16)),
                    histograms=histograms,packed=backing(packed_bytes),nbytes=metadata_bytes+packed_bytes))
            tensors[name] = dict(shape=item['shape'], original_bytes=item['original_bytes'],
                                 segments=segments, nbytes=sum(s['nbytes'] for s in segments))
        parameters = sum(math.prod(v['shape']) for v in tensors.values())
        result = dict(format='mfq-tq-fixture-v1',synthetic=True,reference=str(reference.resolve()),
            source_files=[str(p.resolve()) for p in paths],tensors=tensors,remove_tensors=removed,
            original_bytes=target_total,tq_bytes=sum(v['nbytes'] for v in tensors.values()),
            parameters=parameters,original_bpw=target_total*8/parameters)
        result['tq_bpw'] = result['tq_bytes']*8/parameters
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--groupsize',type=int,default=512)
    args = parser.parse_args()
    result = generate(args.reference,args.output,seed=args.seed,groupsize=args.groupsize)
    print(json.dumps({k:result[k] for k in ('original_bytes','tq_bytes','original_bpw','tq_bpw')}))
