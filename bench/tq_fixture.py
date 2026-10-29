"""Reader for the explicitly synthetic manifests from generate_tq_fixture."""
from __future__ import annotations

import json
import mmap
from pathlib import Path
import struct

import numpy as np

from mfq.formats.tq import TQTensor
from mfq.formats.tq_rows import TQRows

MASK = (1 << 64)-1
GAMMA = 0x9e3779b97f4a7c15


def mix(value: int) -> int:
    value = ((value ^ (value >> 30))*0xbf58476d1ce4e5b9) & MASK
    value = ((value ^ (value >> 27))*0x94d049bb133111eb) & MASK
    return value ^ (value >> 31)


class Segment:
    def __init__(self, spec: dict, read_file):
        self.spec = spec
        self.rows, self.width, self.gs = (spec[k] for k in ('rows','width','gs'))
        self.groups = (self.width+self.gs-1)//self.gs
        self.scale_offset = 64+self.rows*8
        self.packed_offset = self.scale_offset+(self.rows*self.groups+1)//2
        self.header = struct.pack('<8s6I',b'RTRELL05',5,self.rows,self.width,0,self.gs,
                                  self.width*spec['bit_sum']//8)+bytes(32)
        self.read_file = read_file

    def bits(self, block: int) -> np.ndarray:
        start = block*4096
        count = min(4096,self.rows-start)
        total = self.spec['bit_sum']
        budget = total*(start+count)//self.rows-total*start//self.rows
        counts = self.spec['histograms'][f'{count}:{budget}']
        result = np.repeat(np.arange(1,9,dtype=np.uint8),counts)
        state = self.spec['seed'] ^ block
        for index in range(count-1,0,-1):
            state = (state+GAMMA) & MASK
            other = mix(state) % (index+1)
            result[index],result[other] = result[other],result[index]
        return result

    def descriptors(self, begin: int, end: int) -> bytes:
        result = np.empty(end-begin,dtype='<u8')
        for block in range(begin//4096,(end+4095)//4096):
            start = block*4096
            bits = self.bits(block).astype(np.uint64)
            prefix = np.r_[np.uint64(0),np.cumsum(bits[:-1],dtype=np.uint64)]
            prefix += np.uint64(self.spec['bit_sum']*start//self.rows)
            first,last = max(begin,start),min(end,start+len(bits))
            values = (prefix*self.width//32) | ((bits-1)<<np.uint64(32))
            values |= np.uint64(self.spec['anchor_bits'])<<np.uint64(40)
            result[first-begin:last-begin] = values[first-start:last-start]
        return result.tobytes()

    def scales(self, begin: int, end: int) -> bytes:
        # Counter-based bytes are reproducible without retaining the table.
        index = np.arange(begin,end,dtype=np.uint64)
        with np.errstate(over='ignore'):
            value = np.uint64(self.spec['seed'])+(index+np.uint64(1))*np.uint64(GAMMA)
            value = (value ^ (value >> np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
            value = (value ^ (value >> np.uint64(27)))*np.uint64(0x94d049bb133111eb)
            value ^= value >> np.uint64(31)
        result = value.astype(np.uint8)
        scale_bytes = self.packed_offset-self.scale_offset
        if self.rows*self.groups & 1 and begin < scale_bytes <= end:
            result[scale_bytes-1-begin] &= 15
        return result.tobytes()

    def packed(self, begin: int, end: int) -> bytes:
        result = []
        cursor = 0
        for piece in self.spec['packed']:
            finish = cursor+piece['nbytes']
            first,last = max(begin,cursor),min(end,finish)
            if first < last:
                result.append(self.read_file(piece['file'],piece['offset']+first-cursor,last-first))
            cursor = finish
        value = b''.join(result)
        if len(value) != end-begin:
            raise ValueError('Truncated synthetic TQ backing stream')
        return value

    def read(self, offset: int, size: int) -> bytes:
        end = offset+size
        if offset < 0 or size < 0 or end > self.spec['nbytes']:
            raise ValueError('Synthetic TQ byte range is out of bounds')
        result = []
        first,last = max(offset,0),min(end,64)
        if first < last:
            result.append(self.header[first:last])
        first,last = max(offset,64),min(end,self.scale_offset)
        if first < last:
            row_begin,row_end = (first-64)//8,(last-64+7)//8
            value = self.descriptors(row_begin,row_end)
            result.append(value[(first-64)%8:(first-64)%8+last-first])
        first,last = max(offset,self.scale_offset),min(end,self.packed_offset)
        if first < last:
            result.append(self.scales(first-self.scale_offset,last-self.scale_offset))
        first,last = max(offset,self.packed_offset),min(end,self.spec['nbytes'])
        if first < last:
            result.append(self.packed(first-self.packed_offset,last-self.packed_offset))
        return b''.join(result)


class TQFixture:
    def __init__(self, path: str | Path):
        self.spec = json.loads(Path(path).read_text(encoding='utf-8'))
        if self.spec.get('format') != 'mfq-tq-fixture-v1' or self.spec.get('synthetic') is not True:
            raise ValueError('Expected an explicit synthetic TQ fixture')
        self.files = {}
        self.mappings = {}

    def read_file(self, index: int, offset: int, size: int) -> bytes:
        if index not in self.mappings:
            file = open(self.spec['source_files'][index],'rb')
            self.files[index] = file
            self.mappings[index] = mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)
        mapping = self.mappings[index]
        if offset < 0 or size < 0 or offset+size > len(mapping):
            raise ValueError('Synthetic TQ backing range is out of bounds')
        return mapping[offset:offset+size]

    def rows(self, name: str, begin: int, end: int) -> TQTensor:
        tensor = self.spec['tensors'][name]
        total_rows = int(np.prod(tensor['shape'][:-1]))
        if not 0 <= begin < end <= total_rows:
            raise ValueError('Invalid synthetic TQ row range')
        chunks = []
        start = 0
        for spec in tensor['segments']:
            first,last = max(begin,start),min(end,start+spec['rows'])
            if first < last:
                segment = Segment(spec,self.read_file)
                chunks.append(TQRows(spec['nbytes'],segment.read).slice_rows(first-start,last-start))
            start += spec['rows']
        if len(chunks)==1:
            return chunks[0]
        descriptors, packed, selectors = [], [], []
        words = 0
        for chunk in chunks:
            descriptor = chunk.descriptors.copy()
            # Segment offsets occupy only the low 32 bits; adding the preceding
            # stream length preserves the precision/anchor fields.
            descriptor += np.uint64(words)
            descriptors.append(descriptor)
            words += len(chunk.packed)//4
            packed.append(chunk.packed)
            values = np.stack((chunk.scales & 15,chunk.scales >> 4),axis=1).reshape(-1)
            selectors.append(values[:chunk.rows*chunk.groups])
        values = np.concatenate(selectors)
        scales = np.zeros((len(values)+1)//2,dtype=np.uint8)
        scales[:] = values[::2]
        scales[:len(values)//2] |= values[1::2]<<4
        return TQTensor(end-begin,chunks[0].width,chunks[0].gs,False,
                         np.concatenate(descriptors),scales,np.concatenate(packed))

    def close(self):
        for mapping in self.mappings.values():
            mapping.close()
        for file in self.files.values():
            file.close()
        self.files.clear()
        self.mappings.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
