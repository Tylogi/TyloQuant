"""TQ: fixed-book, row-dynamic Trellis quantization (RTRELL05 payload).

Each row selects 1..8 bits per weight. Independent 256-element tail-biting
rings address a procedural 65536-state Gaussian book. A row has an FP16
anchor and either 4-bit subgroup selectors or shared FP16 column scales.
No rotation is applied. Model containers store this payload as ``TQ``;
their ``tq_shapes`` metadata retains logical 2-D or expert-bank dimensions.

The payload stores the padded width; callers retain the logical input width
in their model metadata and pass it to ``Projection(input_columns=...)``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RELATIVE = np.exp2(np.linspace(-2, 2, 16, dtype=np.float32)).astype(np.float16)
_HEADER = struct.Struct("<8s6I")


def fixed_codebook() -> np.ndarray:
    """Encoder/reference table; CUDA inference generates pairs in registers."""
    product = np.arange(65536, dtype=np.uint32) * np.uint32(0x83DCD12D)
    total = sum((product >> shift) & 255 for shift in (0, 8, 16, 24))
    inv, bias = np.array([0x1EEE, 0xC931], dtype=np.uint16).view(np.float16)
    return ((total.astype(np.float32) + 1024) * np.float32(inv) + np.float32(bias)).astype(
        np.float16
    )


def pack_states(indices: np.ndarray, bits: int) -> np.ndarray:
    """Pack low transition bits in reverse ring order, little-endian."""
    indices = np.asarray(indices)
    if bits not in range(1, 9) or indices.ndim != 2 or indices.shape[1] != 256:
        raise ValueError("Expected [rings, 256] states and 1..8 bits")
    symbols = indices.astype(np.uint16)[:, ::-1] & ((1 << bits) - 1)
    stream = (symbols[:, :, None] >> np.arange(bits, dtype=np.uint16)) & 1
    return np.packbits(stream.reshape(len(symbols), -1), axis=-1, bitorder="little")


def unpack_states(packed: np.ndarray, bits: int) -> np.ndarray:
    """Reconstruct 16-bit states, including the wrap across each ring tail."""
    packed = np.asarray(packed, dtype=np.uint8)
    if bits not in range(1, 9) or packed.ndim != 2 or packed.shape[1] != 32 * bits:
        raise ValueError("Invalid ring payload or bit width")
    wrapped = np.concatenate((packed, packed[:, :2]), axis=1).astype(np.uint32)
    pos = (255 - np.arange(256)) * bits
    byte, shift = pos // 8, pos % 8
    word = wrapped[:, byte] | (wrapped[:, byte + 1] << 8) | (wrapped[:, byte + 2] << 16)
    return ((word >> shift) & 65535).astype(np.uint16)


@dataclass(frozen=True)
class TQTensor:
    """Validated RTRELL05 components in host memory, independent of PyTorch."""

    rows: int
    width: int
    gs: int
    columns: bool
    descriptors: np.ndarray
    scales: np.ndarray
    packed: np.ndarray

    def __post_init__(self):
        if self.rows <= 0 or self.width <= 0 or self.width % 256:
            raise ValueError("Expected positive rows and a width divisible by 256")
        if not self.columns and (
            self.gs < 16 or self.gs > 2**20 or self.gs & (self.gs - 1)
        ):
            raise ValueError("Subgroup size must be a power of two in [16, 2**20]")
        for name, dtype in (
            ("descriptors", "<u8"),
            ("scales", "<f2" if self.columns else "u1"),
            ("packed", "u1"),
        ):
            value = np.array(getattr(self, name), dtype=dtype, copy=True).reshape(-1)
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if len(self.descriptors) != self.rows:
            raise ValueError("Row descriptor count differs")
        # Book IDs are reserved; the fixed book is shared by all precisions.
        allowed = np.uint64(0x00FFFF07FFFFFFFF)
        if np.any(self.descriptors & ~allowed):
            raise ValueError("Reserved row descriptor bits must be zero")
        bits = self.bits.astype(np.uint64)
        lengths = self.width * bits // 32
        offsets = self.descriptors & np.uint64(0xFFFFFFFF)
        expected = np.r_[np.uint64(0), np.cumsum(lengths[:-1], dtype=np.uint64)]
        if not np.array_equal(offsets, expected) or int(lengths.sum()) * 4 != len(self.packed):
            raise ValueError("Row offsets or packed payload length differ")
        if int(lengths.sum()) > 2**32:
            raise ValueError("Packed tensor exceeds 32-bit word offsets")
        nscale = self.width if self.columns else (self.rows * self.groups + 1) // 2
        if len(self.scales) != nscale:
            raise ValueError("Scale payload length differs")
        if not np.isfinite(self.anchors).all() or np.any(self.anchors < 0):
            raise ValueError("Row anchors must be finite and nonnegative")
        if self.columns and (not np.isfinite(self.scales).all() or np.any(self.scales <= 0)):
            raise ValueError("Column scales must be finite and positive")

    @property
    def bits(self) -> np.ndarray:
        return (((self.descriptors >> 32) & 7) + 1).astype(np.uint8)

    @property
    def anchors(self) -> np.ndarray:
        return (self.descriptors >> 40).astype(np.uint16).view(np.float16)

    @property
    def groups(self) -> int:
        return 0 if self.columns else (self.width + self.gs - 1) // self.gs

    @property
    def nbytes(self) -> int:
        """Serialized bytes, including header, row descriptors and scales."""
        return 64 + self.descriptors.nbytes + self.scales.nbytes + self.packed.nbytes

    @property
    def bpw(self) -> float:
        return self.nbytes * 8 / (self.rows * self.width)

    def components(self) -> dict:
        return dict(
            rows=self.rows, width=self.width, gs=self.gs, columns=self.columns,
            descriptors=self.descriptors, scales=self.scales, packed=self.packed,
        )

    def to_bytes(self) -> bytes:
        header = _HEADER.pack(
            b"RTRELL05", 5, self.rows, self.width, int(self.columns), self.gs, len(self.packed)
        )
        # Reserved polynomial coefficients are zero for the fixed-book format.
        return b"".join((header, bytes(32), self.descriptors.tobytes(),
                         self.scales.tobytes(), self.packed.tobytes()))

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def from_bytes(cls, payload: bytes) -> TQTensor:
        if len(payload) < 64:
            raise ValueError("Truncated Trellis header")
        magic, version, rows, width, columns, gs, nbytes = _HEADER.unpack_from(payload)
        if magic != b"RTRELL05" or version != 5 or columns not in (0, 1):
            raise ValueError("Invalid RTRELL05 header")
        if rows <= 0 or width <= 0 or width % 256 or (not columns and gs <= 0):
            raise ValueError("Invalid Trellis geometry")
        if any(payload[32:64]):
            raise ValueError("Fixed-book tensors require zero polynomial coefficients")
        scale_bytes = width * 2 if columns else (rows * ((width + gs - 1) // gs) + 1) // 2
        end = 64 + rows * 8
        if len(payload) != end + scale_bytes + nbytes:
            raise ValueError("Trellis payload length differs")
        return cls(
            rows, width, gs, bool(columns),
            np.frombuffer(payload, "<u8", rows, 64),
            np.frombuffer(payload, "<f2" if columns else "u1",
                          width if columns else scale_bytes, end),
            np.frombuffer(payload, "u1", nbytes, end + scale_bytes),
        )

    @classmethod
    def load(cls, path: str | Path) -> TQTensor:
        return cls.from_bytes(Path(path).read_bytes())

    def dequantize(self, input_columns: int | None = None) -> np.ndarray:
        """Slow CPU reference, reconstructing weights with one FP16 rounding."""
        logical = self.width if input_columns is None else input_columns
        if not 0 < logical <= self.width:
            raise ValueError("Invalid logical input width")
        lut = fixed_codebook()
        output = np.empty((self.rows, self.width), dtype=np.float16)
        if not self.columns:
            selectors = np.stack((self.scales & 15, self.scales >> 4), axis=1).reshape(-1)
            selectors = selectors[:self.rows * self.groups].reshape(self.rows, self.groups)
        for row, bits in enumerate(self.bits):
            start = int(self.descriptors[row] & np.uint64(0xFFFFFFFF)) * 4
            size = self.width * int(bits) // 8
            states = unpack_states(self.packed[start:start + size].reshape(-1, 32 * int(bits)), int(bits))
            relative = self.scales if self.columns else np.repeat(RELATIVE[selectors[row]], self.gs)[:self.width]
            scale = np.float32(self.anchors[row]) * relative.astype(np.float32)
            output[row] = (lut[states.reshape(-1)].astype(np.float32) * scale).astype(np.float16)
        return output[:, :logical].copy()
