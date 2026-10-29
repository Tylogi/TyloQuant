"""Bounded row reads from ordinary RTRELL05 tensors.

Embedding tables and routed projection banks share this reader. It does not
own a cache or read the full packed stream when only a row range is needed.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import struct
import numpy as np

from mfq.formats.tq import TQTensor


class TQRows:
    def __init__(self, nbytes: int, read: Callable[[int, int], bytes]):
        self._read = read
        self.nbytes = nbytes
        header = self._range(0, 64)
        magic, version, self.rows, self.width, columns, self.gs, packed_bytes = (
            struct.unpack_from('<8s6I', header)
        )
        if magic != b'RTRELL05' or version != 5 or columns not in (0, 1):
            raise ValueError('Invalid RTRELL05 header')
        if not self.rows or not self.width or self.width % 256 or any(header[32:64]):
            raise ValueError('Invalid RTRELL05 geometry or reserved bytes')
        self.columns = bool(columns)
        if not self.columns and (not 16 <= self.gs <= 2**20 or self.gs & (self.gs-1)):
            raise ValueError('Invalid TQ subgroup size')
        self.groups = 0 if self.columns else (self.width+self.gs-1)//self.gs
        self._scale_offset = 64+self.rows*8
        self._scale_bytes = self.width*2 if self.columns else (self.rows*self.groups+1)//2
        self._packed_offset = self._scale_offset+self._scale_bytes
        self._packed_bytes = packed_bytes
        if self._packed_offset+packed_bytes != nbytes:
            raise ValueError('TQ tensor length differs from header')

    def _range(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset > self.nbytes or size > self.nbytes-offset:
            raise ValueError('TQ byte range is out of bounds')
        result = self._read(offset, size)
        if len(result) != size:
            raise ValueError('Truncated TQ range read')
        return result

    def slice_rows(self, begin: int, end: int) -> TQTensor:
        """Read [begin,end), preserving precision/order and rebasing offsets."""
        if not 0 <= begin < end <= self.rows:
            raise ValueError('Invalid TQ row range')
        count = end-begin
        descriptors = np.frombuffer(self._range(64+begin*8, count*8), '<u8').copy()
        offsets = descriptors & np.uint64(0xffffffff)
        bits = ((descriptors >> np.uint64(32)) & np.uint64(7))+1
        first = int(offsets[0])*4
        finish = int(offsets[-1])*4+self.width*int(bits[-1])//8
        if finish > self._packed_bytes or first > finish:
            raise ValueError('TQ row payload is out of bounds')
        packed = np.frombuffer(self._range(self._packed_offset+first, finish-first), 'u1')
        descriptors = ((descriptors & np.uint64(0xffffffff00000000)) |
                       (offsets-offsets[0]))
        if self.columns:
            scales = np.frombuffer(self._range(self._scale_offset, self._scale_bytes), '<f2')
        else:
            first_selector = begin*self.groups
            selector_count = count*self.groups
            start_byte = first_selector//2
            end_byte = (first_selector+selector_count+1)//2
            source = np.frombuffer(self._range(self._scale_offset+start_byte,
                                               end_byte-start_byte), 'u1')
            # A selected row can start at the high nibble. Repack to a canonical
            # low-nibble start; copying bytes directly would shift all scales.
            indices = np.arange(selector_count)+(first_selector & 1)
            selectors = ((source[indices//2] >> ((indices & 1)*4)) & 15).astype(np.uint8)
            scales = np.zeros((selector_count+1)//2, dtype=np.uint8)
            scales[:] = selectors[::2]
            scales[:selector_count//2] |= selectors[1::2] << 4
        return TQTensor(count, self.width, self.gs, self.columns, descriptors, scales, packed)

    def gather_rows(self, rows: Sequence[int]) -> TQTensor:
        """Return requested rows, including duplicates, in their original order."""
        requested = np.asarray(rows, dtype=np.int64)
        if requested.ndim != 1 or not len(requested) or np.any(requested < 0) or np.any(requested >= self.rows):
            raise ValueError('Invalid TQ row indices')
        unique = np.unique(requested)
        runs = np.split(unique, np.flatnonzero(np.diff(unique) != 1)+1)
        selections = {}
        for run in runs:
            chunk = self.slice_rows(int(run[0]), int(run[-1])+1)
            offsets = chunk.descriptors & np.uint64(0xffffffff)
            if not self.columns:
                selectors = np.stack((chunk.scales & 15, chunk.scales >> 4), axis=1).reshape(-1)
                selectors = selectors[:chunk.rows*self.groups].reshape(chunk.rows, self.groups)
            for index, row in enumerate(run):
                start = int(offsets[index])*4
                size = chunk.width*int(chunk.bits[index])//8
                selections[int(row)] = (chunk.descriptors[index], chunk.packed[start:start+size],
                                        chunk.scales if self.columns else selectors[index])
        descriptors, packed, selectors = [], [], []
        words = 0
        for row in requested:
            descriptor, stream, scale = selections[int(row)]
            descriptors.append((int(descriptor) & 0xffffffff00000000) | words)
            words += len(stream)//4
            packed.append(stream)
            selectors.append(scale)
        if self.columns:
            scales = selectors[0]
        else:
            values = np.concatenate(selectors)
            scales = np.zeros((len(values)+1)//2, dtype=np.uint8)
            scales[:] = values[::2]
            scales[:len(values)//2] |= values[1::2] << 4
        return TQTensor(len(requested), self.width, self.gs, self.columns,
                        np.asarray(descriptors, dtype='<u8'), scales, np.concatenate(packed))
