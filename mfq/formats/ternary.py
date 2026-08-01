"""Neuron-anchored scalar ternary reference format.

Five ternary symbols are packed into one byte. Each output neuron has one
FP16 anchor and each small group has one packed integer relative scale.

This is an offline format definition. No CUDA runtime path is implied.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeuronTernarySpec:
    groupsize: int = 24
    sub_bits: int = 3

    def __post_init__(self) -> None:
        if self.groupsize <= 0:
            raise ValueError("ternary groupsize must be positive")
        if not 1 <= self.sub_bits <= 8:
            raise ValueError("ternary sub_bits must be in [1, 8]")

    @property
    def label(self) -> str:
        return f"Neuron-Ternary-S{self.sub_bits}"

    def payload_nbytes(self, out: int, neuron_len: int) -> int:
        if out <= 0 or neuron_len <= 0:
            raise ValueError("ternary tensor dimensions must be positive")
        ng = math.ceil(neuron_len / self.groupsize)
        anchors = out * 2
        scales = (out * ng * self.sub_bits + 7) // 8
        trits = (out * neuron_len + 4) // 5
        return anchors + scales + trits

    def bpw(self, neuron_len: int, *, out: int = 1) -> float:
        return 8.0 * self.payload_nbytes(out, neuron_len) / (out * neuron_len)


NEURON_TERNARY_S3 = NeuronTernarySpec(groupsize=24, sub_bits=3)
NEURON_TERNARY_S4 = NeuronTernarySpec(groupsize=24, sub_bits=4)


@dataclass
class NeuronTernaryTensor:
    spec: NeuronTernarySpec
    shape: tuple[int, ...]
    axis: int
    neuron_len: int
    neuron_scale: np.ndarray
    sub_scale: np.ndarray
    trits: np.ndarray

    @property
    def payload_nbytes(self) -> int:
        return self.spec.payload_nbytes(self.neuron_scale.size, self.neuron_len)

    @property
    def payload_bpw(self) -> float:
        return 8.0 * self.payload_nbytes / int(np.prod(self.shape))


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    values_u16 = np.ascontiguousarray(values, dtype=np.uint16).reshape(-1)
    if values_u16.size and np.any(values_u16 >= (1 << bits)):
        raise ValueError(f"value does not fit in {bits} bits")
    shifts = np.arange(bits, dtype=np.uint16)
    rows = ((values_u16[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return np.packbits(rows.reshape(-1), bitorder="little").tobytes()


def _unpack_bits(blob: bytes, off: int, count: int, bits: int) -> tuple[np.ndarray, int]:
    nbytes = (count * bits + 7) // 8
    end = off + nbytes
    if end > len(blob):
        raise ValueError("truncated ternary scale stream")
    packed = np.frombuffer(blob, dtype=np.uint8, count=nbytes, offset=off)
    stream = np.unpackbits(packed, bitorder="little")[: count * bits].reshape(count, bits)
    shifts = 1 << np.arange(bits, dtype=np.uint16)
    values = (stream.astype(np.uint16) * shifts).sum(axis=1)
    return values.astype(np.uint8), end


def pack_trits(trits: np.ndarray) -> bytes:
    """Pack symbols in [0, 2] at five symbols per byte."""

    values = np.ascontiguousarray(trits, dtype=np.uint8).reshape(-1)
    if values.size and np.any(values > 2):
        raise ValueError("ternary stream contains a symbol outside [0, 2]")
    padded_count = (-values.size) % 5
    if padded_count:
        values = np.pad(values, (0, padded_count))
    groups = values.reshape(-1, 5).astype(np.uint16)
    powers = np.asarray([1, 3, 9, 27, 81], dtype=np.uint16)
    return (groups * powers).sum(axis=1).astype(np.uint8).tobytes()


def unpack_trits(blob: bytes, off: int, count: int) -> tuple[np.ndarray, int]:
    nbytes = (count + 4) // 5
    end = off + nbytes
    if end > len(blob):
        raise ValueError("truncated ternary symbol stream")
    packed = np.frombuffer(blob, dtype=np.uint8, count=nbytes, offset=off)
    if packed.size and np.any(packed > 242):
        raise ValueError("invalid base-3 packed byte")
    work = packed.astype(np.uint16)
    decoded = np.empty((nbytes, 5), dtype=np.uint8)
    for column in range(5):
        decoded[:, column] = work % 3
        work //= 3
    return decoded.reshape(-1)[:count].copy(), end


_MAGIC = b"NTQ\x01"
_HEADER = struct.Struct("<4sBHiiI")


def pack_neuron_ternary(tensor: NeuronTernaryTensor) -> bytes:
    spec = tensor.spec
    shape = tuple(int(value) for value in tensor.shape)
    if not shape or not 0 <= tensor.axis < len(shape):
        raise ValueError(f"invalid ternary shape/axis: {shape}, axis={tensor.axis}")
    out = int(tensor.neuron_scale.size)
    if int(np.prod(shape)) != out * tensor.neuron_len:
        raise ValueError("ternary shape does not match neuron dimensions")
    ng = math.ceil(tensor.neuron_len / spec.groupsize)
    if np.asarray(tensor.sub_scale).shape != (out, ng):
        raise ValueError(
            f"bad sub_scale shape: {np.asarray(tensor.sub_scale).shape}, expected {(out, ng)}"
        )
    if np.asarray(tensor.trits).shape != (out, tensor.neuron_len):
        raise ValueError(
            f"bad trits shape: {np.asarray(tensor.trits).shape}, "
            f"expected {(out, tensor.neuron_len)}"
        )

    parts = [
        _HEADER.pack(
            _MAGIC,
            spec.sub_bits,
            spec.groupsize,
            tensor.axis,
            tensor.neuron_len,
            len(shape),
        ),
        struct.pack(f"<{len(shape)}q", *shape),
        struct.pack("<I", out),
        np.ascontiguousarray(tensor.neuron_scale, dtype=np.float16).tobytes(),
        _pack_bits(tensor.sub_scale, spec.sub_bits),
        pack_trits(tensor.trits),
    ]
    return b"".join(parts)


def unpack_neuron_ternary(blob: bytes) -> NeuronTernaryTensor:
    if len(blob) < _HEADER.size:
        raise ValueError("truncated ternary header")
    magic, sub_bits, groupsize, axis, neuron_len, ndim = _HEADER.unpack_from(blob)
    if magic != _MAGIC:
        raise ValueError(f"invalid ternary magic: {magic!r}")
    if ndim <= 0:
        raise ValueError(f"invalid ternary ndim: {ndim}")

    off = _HEADER.size
    shape_bytes = 8 * ndim
    if off + shape_bytes + 4 > len(blob):
        raise ValueError("truncated ternary shape")
    shape = tuple(struct.unpack_from(f"<{ndim}q", blob, off))
    off += shape_bytes
    out = struct.unpack_from("<I", blob, off)[0]
    off += 4

    spec = NeuronTernarySpec(groupsize=groupsize, sub_bits=sub_bits)
    ng = math.ceil(neuron_len / spec.groupsize)
    anchor_bytes = out * 2
    if off + anchor_bytes > len(blob):
        raise ValueError("truncated ternary neuron anchors")
    neuron_scale = np.frombuffer(
        blob,
        dtype=np.float16,
        count=out,
        offset=off,
    ).astype(np.float32)
    off += anchor_bytes
    sub_scale, off = _unpack_bits(blob, off, out * ng, sub_bits)
    trits, off = unpack_trits(blob, off, out * neuron_len)
    if off != len(blob):
        raise ValueError(f"invalid ternary blob tail: consumed={off}, size={len(blob)}")

    return NeuronTernaryTensor(
        spec=spec,
        shape=shape,
        axis=axis,
        neuron_len=neuron_len,
        neuron_scale=neuron_scale,
        sub_scale=sub_scale.reshape(out, ng),
        trits=trits.reshape(out, neuron_len),
    )

