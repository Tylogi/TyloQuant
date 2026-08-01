"""NINT8-0: the GGML Q8_0 block layout under an MFQ dtype label.

Each block contains one FP16 scale followed by 32 signed INT8 values:

    block = scale(f16) || q[32](i8)
    weight = scale * q

The serialized payload keeps the 34-byte blocks byte-for-byte compatible with
GGML Q8_0.  The small MFQ header records the logical tensor shape and axis.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


QK8_0 = 32
Q8_0_BLOCK_BYTES = 34
_MAGIC = b"NI80"
_HEADER = struct.Struct("<4siiI")


@dataclass
class Nint8ZeroTensor:
    """One matrix stored as exact GGML Q8_0 blocks."""

    shape: tuple[int, ...]
    axis: int
    scale: np.ndarray
    q: np.ndarray
    neuron_len: int

    def __post_init__(self) -> None:
        self.shape = tuple(int(value) for value in self.shape)
        self.axis = int(self.axis)
        self.neuron_len = int(self.neuron_len)
        if not self.shape or not 0 <= self.axis < len(self.shape):
            raise ValueError("NINT8-0 axis is outside the tensor rank")
        if self.neuron_len <= 0 or self.neuron_len % QK8_0:
            raise ValueError("NINT8-0 neuron_len must be a positive multiple of 32")
        out = int(self.shape[self.axis])
        expected_len = int(np.prod(self.shape, dtype=np.int64)) // out
        if expected_len != self.neuron_len:
            raise ValueError(
                f"NINT8-0 neuron_len mismatch: shape implies {expected_len}, "
                f"got {self.neuron_len}"
            )
        ng = self.neuron_len // QK8_0
        scale = np.ascontiguousarray(self.scale, dtype=np.float16)
        q = np.ascontiguousarray(self.q, dtype=np.int8)
        if scale.shape != (out, ng):
            raise ValueError(
                f"NINT8-0 scale shape must be {(out, ng)}, got {scale.shape}"
            )
        if q.shape != (out, ng, QK8_0):
            raise ValueError(
                f"NINT8-0 q shape must be {(out, ng, QK8_0)}, got {q.shape}"
            )
        self.scale = scale
        self.q = q

    @property
    def out(self) -> int:
        return int(self.shape[self.axis])

    @property
    def ng(self) -> int:
        return self.neuron_len // QK8_0


def payload_nbytes(shape: tuple[int, ...], axis: int, neuron_len: int) -> int:
    shape = tuple(int(value) for value in shape)
    out = shape[int(axis)]
    ng = int(neuron_len) // QK8_0
    return _HEADER.size + 8 * len(shape) + 8 + out * ng * Q8_0_BLOCK_BYTES


def pack_nint8_zero_header(
    shape: tuple[int, ...],
    axis: int,
    neuron_len: int,
) -> bytes:
    """Serialize the fixed-size portion preceding the Q8_0-compatible blocks."""

    shape = tuple(int(value) for value in shape)
    axis = int(axis)
    neuron_len = int(neuron_len)
    if not shape or not 0 <= axis < len(shape):
        raise ValueError("NINT8-0 axis is outside the tensor rank")
    if neuron_len <= 0 or neuron_len % QK8_0:
        raise ValueError("NINT8-0 neuron_len must be a positive multiple of 32")
    out = int(shape[axis])
    expected_len = int(np.prod(shape, dtype=np.int64)) // out
    if expected_len != neuron_len:
        raise ValueError(
            f"NINT8-0 neuron_len mismatch: shape implies {expected_len}, "
            f"got {neuron_len}"
        )
    return b"".join(
        (
            _HEADER.pack(_MAGIC, axis, neuron_len, len(shape)),
            struct.pack(f"<{len(shape)}q", *shape),
            struct.pack("<II", out, neuron_len // QK8_0),
        )
    )


def pack_nint8_zero_blocks(scale: np.ndarray, q: np.ndarray) -> bytes:
    """Interleave FP16 scales and signed INT8 values into 34-byte blocks."""

    scale = np.ascontiguousarray(scale, dtype=np.float16)
    q = np.ascontiguousarray(q, dtype=np.int8)
    if scale.ndim != 2 or q.shape != (*scale.shape, QK8_0):
        raise ValueError(
            f"NINT8-0 block shape mismatch: scale={scale.shape}, q={q.shape}"
        )
    blocks = np.empty((*scale.shape, Q8_0_BLOCK_BYTES), dtype=np.uint8)
    blocks[..., :2] = scale.view(np.uint8).reshape(*scale.shape, 2)
    blocks[..., 2:] = q.view(np.uint8)
    return blocks.tobytes()


def quantize_nint8_zero(
    weight: np.ndarray,
    *,
    axis: int = 0,
) -> Nint8ZeroTensor:
    """Directly quantize float weights with the deterministic GGML Q8_0 rule."""

    values = np.asarray(weight, dtype=np.float32)
    if not values.ndim or not 0 <= axis < values.ndim:
        raise ValueError("NINT8-0 axis is outside the tensor rank")
    moved = np.moveaxis(values, axis, 0)
    out = moved.shape[0]
    neuron_len = moved.size // out
    if neuron_len <= 0 or neuron_len % QK8_0:
        raise ValueError("NINT8-0 neuron_len must be a positive multiple of 32")
    blocks = np.ascontiguousarray(moved).reshape(out, neuron_len // QK8_0, QK8_0)
    scale_f32 = np.max(np.abs(blocks), axis=-1, keepdims=True) / np.float32(127.0)
    inverse = np.zeros_like(scale_f32, dtype=np.float32)
    np.divide(np.float32(1.0), scale_f32, out=inverse, where=scale_f32 != 0)
    normalized = blocks * inverse
    rounded = np.sign(normalized) * np.floor(np.abs(normalized) + np.float32(0.5))
    q = np.clip(rounded, -127, 127).astype(np.int8)
    return Nint8ZeroTensor(
        shape=tuple(int(value) for value in values.shape),
        axis=axis,
        scale=scale_f32[..., 0].astype(np.float16),
        q=q,
        neuron_len=neuron_len,
    )


def pack_nint8_zero(tensor: Nint8ZeroTensor) -> bytes:
    """Serialize to the MFQ NINT8-0 payload."""

    return pack_nint8_zero_header(
        tensor.shape,
        tensor.axis,
        tensor.neuron_len,
    ) + pack_nint8_zero_blocks(tensor.scale, tensor.q)


def unpack_nint8_zero(blob: bytes | memoryview) -> Nint8ZeroTensor:
    """Deserialize and validate an MFQ NINT8-0 payload."""

    if len(blob) < _HEADER.size:
        raise ValueError("truncated NINT8-0 header")
    magic, axis, neuron_len, ndim = _HEADER.unpack_from(blob, 0)
    if magic != _MAGIC:
        raise ValueError(f"invalid NINT8-0 magic: {magic!r}")
    if ndim == 0:
        raise ValueError("NINT8-0 tensor rank cannot be zero")
    off = _HEADER.size
    shape_end = off + 8 * int(ndim)
    if shape_end + 8 > len(blob):
        raise ValueError("truncated NINT8-0 shape")
    shape = struct.unpack_from(f"<{ndim}q", blob, off)
    off = shape_end
    out, ng = struct.unpack_from("<II", blob, off)
    off += 8
    expected = int(out) * int(ng) * Q8_0_BLOCK_BYTES
    if len(blob) - off != expected:
        raise ValueError(
            f"invalid NINT8-0 block bytes: got {len(blob) - off}, expected {expected}"
        )
    blocks = np.frombuffer(
        blob, dtype=np.uint8, count=expected, offset=off
    ).reshape(int(out), int(ng), Q8_0_BLOCK_BYTES)
    scale = np.ascontiguousarray(blocks[..., :2]).view(np.float16).reshape(
        int(out), int(ng)
    )
    q = np.ascontiguousarray(blocks[..., 2:]).view(np.int8)
    tensor = Nint8ZeroTensor(
        shape=tuple(int(value) for value in shape),
        axis=int(axis),
        scale=scale,
        q=q,
        neuron_len=int(neuron_len),
    )
    if tensor.out != int(out) or tensor.ng != int(ng):
        raise ValueError("NINT8-0 dimensions disagree with the logical shape")
    return tensor


def dequantize_nint8_zero(tensor: Nint8ZeroTensor) -> np.ndarray:
    """Return the exact Q8_0 reconstruction as float32."""

    rows = (
        tensor.scale.astype(np.float32)[..., None]
        * tensor.q.astype(np.float32)
    ).reshape(tensor.out, tensor.neuron_len)
    moved_shape = (
        (tensor.shape[tensor.axis],)
        + tensor.shape[: tensor.axis]
        + tensor.shape[tensor.axis + 1 :]
    )
    return np.moveaxis(rows.reshape(moved_shape), 0, tensor.axis)
