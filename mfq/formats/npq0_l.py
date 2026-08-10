"""NPQ0-L: one-bit neuron-anchored product vector quantization.

Each eight-weight vector uses a 3-bit and a 4-bit sub-codebook index.  One
3-bit state is shared by the three vectors in every gs24 group.  The state
selects a product-codebook pair and a relative scale; one fp16 anchor is kept
per output neuron.

The index and state streams are packed independently across the whole tensor,
so no K padding or per-row byte padding is stored.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

_GROUP_SIZE = 24
_VECTOR_SIZE = 8
_SUBVECTOR_SIZE = 4
_STATE_COUNT = 8
_FIRST_BITS = 3
_SECOND_BITS = 4
_INDEX_BITS = _FIRST_BITS + _SECOND_BITS
_FIRST_ENTRIES = 1 << _FIRST_BITS
_SECOND_ENTRIES = 1 << _SECOND_BITS

_TABLE_VERSION = 1
_TABLE_HEADER_BYTES = 64
_SCALE_LUT_OFFSET = 8
_FIRST_CODEBOOK_BYTES = _STATE_COUNT * _FIRST_ENTRIES * _SUBVECTOR_SIZE
_SECOND_CODEBOOK_BYTES = _STATE_COUNT * _SECOND_ENTRIES * _SUBVECTOR_SIZE
NPQ0_L_TABLE_BYTES = _TABLE_HEADER_BYTES + _FIRST_CODEBOOK_BYTES + _SECOND_CODEBOOK_BYTES


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    value = np.ascontiguousarray(values).reshape(-1)
    if not 1 <= bits <= 8:
        raise ValueError(f"packed width must be in [1,8], got {bits}")
    if value.size == 0:
        return b""
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError("packed values must be integers")
    if np.any(value < 0) or np.any(value >= (1 << bits)):
        raise ValueError(f"packed value exceeds {bits} bits")
    unpacked = np.unpackbits(
        value.astype(np.uint8, copy=False)[:, None],
        axis=1,
        bitorder="little",
    )[:, :bits]
    return np.packbits(unpacked.reshape(-1), bitorder="little").tobytes()


def _unpack_bits(
    blob: bytes | memoryview,
    offset: int,
    count: int,
    bits: int,
) -> tuple[np.ndarray, int]:
    nbytes = (count * bits + 7) // 8
    end = offset + nbytes
    if end > len(blob):
        raise ValueError("truncated NPQ0-L bit stream")
    if count == 0:
        return np.empty(0, dtype=np.uint8), end
    packed = np.frombuffer(blob, dtype=np.uint8, count=nbytes, offset=offset)
    stream = np.unpackbits(packed, bitorder="little")[: count * bits].reshape(count, bits)
    shifts = 1 << np.arange(bits, dtype=np.uint16)
    return (stream.astype(np.uint16) * shifts).sum(1).astype(np.uint8), end


def validate_npq0_l_tables(
    scale_lut: np.ndarray,
    first_codebooks: np.ndarray,
    second_codebooks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.asarray(scale_lut, dtype=np.float32).reshape(-1)
    first = np.asarray(first_codebooks)
    second = np.asarray(second_codebooks)
    if scale.shape != (_STATE_COUNT,):
        raise ValueError(f"NPQ0-L scale LUT must have shape ({_STATE_COUNT},)")
    if not np.isfinite(scale).all() or np.any(scale < 0) or float(scale.max()) <= 0:
        raise ValueError("NPQ0-L scale LUT must be finite, non-negative, and nonzero")
    if first.shape != (_STATE_COUNT, _FIRST_ENTRIES, _SUBVECTOR_SIZE):
        raise ValueError(
            "NPQ0-L first codebooks must have shape "
            f"{(_STATE_COUNT, _FIRST_ENTRIES, _SUBVECTOR_SIZE)}"
        )
    if second.shape != (_STATE_COUNT, _SECOND_ENTRIES, _SUBVECTOR_SIZE):
        raise ValueError(
            "NPQ0-L second codebooks must have shape "
            f"{(_STATE_COUNT, _SECOND_ENTRIES, _SUBVECTOR_SIZE)}"
        )
    for name, value in (("first", first), ("second", second)):
        rounded = np.rint(value)
        if (
            not np.isfinite(value).all()
            or not np.array_equal(value, rounded)
            or np.any(rounded < -127)
            or np.any(rounded > 127)
        ):
            raise ValueError(f"NPQ0-L {name} codebooks must be int8-valued")
    return (
        np.ascontiguousarray(scale, dtype=np.float32),
        np.ascontiguousarray(first, dtype=np.int8),
        np.ascontiguousarray(second, dtype=np.int8),
    )


def pack_npq0_l_tables(
    scale_lut: np.ndarray,
    first_codebooks: np.ndarray,
    second_codebooks: np.ndarray,
) -> bytes:
    scale, first, second = validate_npq0_l_tables(
        scale_lut,
        first_codebooks,
        second_codebooks,
    )
    header = bytearray(_TABLE_HEADER_BYTES)
    header[0] = _TABLE_VERSION
    header[1] = _STATE_COUNT
    header[2] = _FIRST_BITS
    header[3] = _SECOND_BITS
    header[4] = _GROUP_SIZE
    header[5] = _VECTOR_SIZE
    header[_SCALE_LUT_OFFSET : _SCALE_LUT_OFFSET + 2 * _STATE_COUNT] = np.ascontiguousarray(
        scale,
        dtype="<f2",
    ).tobytes()
    return bytes(header) + first.tobytes() + second.tobytes()


def unpack_npq0_l_tables(
    payload: bytes | memoryview,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if len(payload) < NPQ0_L_TABLE_BYTES:
        raise ValueError("truncated NPQ0-L table payload")
    header = memoryview(payload)[:_TABLE_HEADER_BYTES]
    expected = (
        _TABLE_VERSION,
        _STATE_COUNT,
        _FIRST_BITS,
        _SECOND_BITS,
        _GROUP_SIZE,
        _VECTOR_SIZE,
    )
    if tuple(int(header[i]) for i in range(6)) != expected:
        raise ValueError("unsupported NPQ0-L table profile")
    if any(header[6:8]) or any(header[24:_TABLE_HEADER_BYTES]):
        raise ValueError("NPQ0-L reserved table bytes must be zero")
    scale = np.frombuffer(
        header,
        dtype="<f2",
        count=_STATE_COUNT,
        offset=_SCALE_LUT_OFFSET,
    ).astype(np.float32)
    offset = _TABLE_HEADER_BYTES
    first = (
        np.frombuffer(
            payload,
            dtype=np.int8,
            count=_FIRST_CODEBOOK_BYTES,
            offset=offset,
        )
        .copy()
        .reshape(_STATE_COUNT, _FIRST_ENTRIES, _SUBVECTOR_SIZE)
    )
    offset += _FIRST_CODEBOOK_BYTES
    second = (
        np.frombuffer(
            payload,
            dtype=np.int8,
            count=_SECOND_CODEBOOK_BYTES,
            offset=offset,
        )
        .copy()
        .reshape(_STATE_COUNT, _SECOND_ENTRIES, _SUBVECTOR_SIZE)
    )
    offset += _SECOND_CODEBOOK_BYTES
    scale, first, second = validate_npq0_l_tables(scale, first, second)
    return scale, first, second, offset


@dataclass(frozen=True)
class Npq0LSpec:
    groupsize: int = _GROUP_SIZE
    state_bits: int = 3
    first_bits: int = _FIRST_BITS
    second_bits: int = _SECOND_BITS

    def __post_init__(self) -> None:
        if (
            self.groupsize != _GROUP_SIZE
            or self.state_bits != 3
            or self.first_bits != _FIRST_BITS
            or self.second_bits != _SECOND_BITS
        ):
            raise ValueError("NPQ0-L has a fixed gs24, S3, PQ3+4 profile")

    @property
    def vector_size(self) -> int:
        return _VECTOR_SIZE

    @property
    def index_bits(self) -> int:
        return _INDEX_BITS

    @property
    def sub_bits(self) -> int:
        return self.state_bits

    @property
    def label(self) -> str:
        return "NPQ0-L"

    def stream_nbytes(self, out: int, neuron_len: int) -> int:
        if out <= 0 or neuron_len <= 0 or neuron_len % _VECTOR_SIZE:
            raise ValueError("NPQ0-L dimensions must be positive and K divisible by 8")
        ng = math.ceil(neuron_len / _GROUP_SIZE)
        nvec = neuron_len // _VECTOR_SIZE
        anchors = 2 * out
        states = (out * ng * self.state_bits + 7) // 8
        indices = (out * nvec * self.index_bits + 7) // 8
        return anchors + states + indices

    def payload_nbytes(
        self,
        out: int,
        neuron_len: int,
        *,
        include_tables: bool = True,
    ) -> int:
        tables = NPQ0_L_TABLE_BYTES if include_tables else 0
        return tables + self.stream_nbytes(out, neuron_len)

    def bpw(
        self,
        neuron_len: int,
        *,
        out: int = 1,
        include_tables: bool = True,
    ) -> float:
        return (
            8.0
            * self.payload_nbytes(
                out,
                neuron_len,
                include_tables=include_tables,
            )
            / (out * neuron_len)
        )


NPQ0_L = Npq0LSpec()


@dataclass
class Npq0LTensor:
    shape: tuple[int, ...]
    axis: int
    neuron_len: int
    neuron_scale: np.ndarray
    scale_lut: np.ndarray
    state: np.ndarray
    indices: np.ndarray
    first_codebooks: np.ndarray
    second_codebooks: np.ndarray

    @property
    def spec(self) -> Npq0LSpec:
        return NPQ0_L

    @property
    def payload_nbytes(self) -> int:
        return NPQ0_L.payload_nbytes(int(np.asarray(self.neuron_scale).size), self.neuron_len)

    @property
    def payload_bpw(self) -> float:
        return 8.0 * self.payload_nbytes / int(np.prod(self.shape))


_MAGIC = b"NPQL"
_VERSION = 1
_HEADER = struct.Struct("<4sBBHiiI")


def _validate_tensor(tensor: Npq0LTensor) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in tensor.shape)
    if not shape or not 0 <= tensor.axis < len(shape):
        raise ValueError(f"invalid NPQ0-L shape/axis: {shape}, axis={tensor.axis}")
    out = int(np.asarray(tensor.neuron_scale).size)
    if tensor.neuron_len <= 0 or tensor.neuron_len % _VECTOR_SIZE:
        raise ValueError("NPQ0-L neuron length must be a positive multiple of 8")
    if int(np.prod(shape)) != out * tensor.neuron_len:
        raise ValueError("NPQ0-L shape does not match neuron dimensions")
    ng = math.ceil(tensor.neuron_len / _GROUP_SIZE)
    nvec = tensor.neuron_len // _VECTOR_SIZE
    with np.errstate(over="ignore", invalid="ignore"):
        anchors = np.asarray(tensor.neuron_scale, dtype=np.float32).astype(np.float16)
    if not np.isfinite(anchors).all() or np.signbit(anchors).any():
        raise ValueError("NPQ0-L neuron anchors must be finite and non-negative in FP16")
    state = np.asarray(tensor.state)
    indices = np.asarray(tensor.indices)
    if state.shape != (out, ng):
        raise ValueError(f"bad NPQ0-L state shape: {state.shape}, expected {(out, ng)}")
    if indices.shape != (out, nvec):
        raise ValueError(f"bad NPQ0-L index shape: {indices.shape}, expected {(out, nvec)}")
    if not np.issubdtype(state.dtype, np.integer) or np.any(state < 0) or np.any(state >= 8):
        raise ValueError("NPQ0-L states must be integers in [0,7]")
    if (
        not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= 128)
    ):
        raise ValueError("NPQ0-L indices must be integers in [0,127]")
    validate_npq0_l_tables(
        tensor.scale_lut,
        tensor.first_codebooks,
        tensor.second_codebooks,
    )
    return out, ng, nvec


def pack_npq0_l(tensor: Npq0LTensor) -> bytes:
    out, _, _ = _validate_tensor(tensor)
    shape = tuple(int(value) for value in tensor.shape)
    return b"".join(
        [
            _HEADER.pack(
                _MAGIC,
                _VERSION,
                NPQ0_L.state_bits,
                NPQ0_L.groupsize,
                tensor.axis,
                tensor.neuron_len,
                len(shape),
            ),
            struct.pack(f"<{len(shape)}q", *shape),
            struct.pack("<I", out),
            pack_npq0_l_tables(
                tensor.scale_lut,
                tensor.first_codebooks,
                tensor.second_codebooks,
            ),
            np.ascontiguousarray(tensor.neuron_scale, dtype="<f2").tobytes(),
            _pack_bits(np.asarray(tensor.state), NPQ0_L.state_bits),
            _pack_bits(np.asarray(tensor.indices), NPQ0_L.index_bits),
        ]
    )


def unpack_npq0_l(blob: bytes | memoryview) -> Npq0LTensor:
    if len(blob) < _HEADER.size:
        raise ValueError("truncated NPQ0-L header")
    magic, version, state_bits, groupsize, axis, neuron_len, ndim = _HEADER.unpack_from(blob)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("invalid or unsupported NPQ0-L header")
    if state_bits != NPQ0_L.state_bits or groupsize != NPQ0_L.groupsize:
        raise ValueError("unsupported NPQ0-L stream profile")
    if ndim <= 0 or neuron_len <= 0 or neuron_len % _VECTOR_SIZE:
        raise ValueError("invalid NPQ0-L dimensions")
    offset = _HEADER.size
    shape_bytes = 8 * ndim
    if offset + shape_bytes + 4 > len(blob):
        raise ValueError("truncated NPQ0-L shape")
    shape = tuple(struct.unpack_from(f"<{ndim}q", blob, offset))
    offset += shape_bytes
    out = struct.unpack_from("<I", blob, offset)[0]
    offset += 4
    scale_lut, first, second, consumed = unpack_npq0_l_tables(memoryview(blob)[offset:])
    offset += consumed
    anchor_bytes = 2 * out
    if offset + anchor_bytes > len(blob):
        raise ValueError("truncated NPQ0-L neuron anchors")
    neuron_scale = np.frombuffer(
        blob,
        dtype="<f2",
        count=out,
        offset=offset,
    ).astype(np.float32)
    offset += anchor_bytes
    ng = math.ceil(neuron_len / _GROUP_SIZE)
    nvec = neuron_len // _VECTOR_SIZE
    state, offset = _unpack_bits(blob, offset, out * ng, NPQ0_L.state_bits)
    indices, offset = _unpack_bits(blob, offset, out * nvec, NPQ0_L.index_bits)
    if offset != len(blob):
        raise ValueError(f"invalid NPQ0-L blob tail: consumed={offset}, size={len(blob)}")
    tensor = Npq0LTensor(
        shape=shape,
        axis=axis,
        neuron_len=neuron_len,
        neuron_scale=neuron_scale,
        scale_lut=scale_lut,
        state=state.reshape(out, ng),
        indices=indices.reshape(out, nvec),
        first_codebooks=first,
        second_codebooks=second,
    )
    _validate_tensor(tensor)
    return tensor


__all__ = [
    "NPQ0_L",
    "NPQ0_L_TABLE_BYTES",
    "Npq0LSpec",
    "Npq0LTensor",
    "pack_npq0_l",
    "pack_npq0_l_tables",
    "unpack_npq0_l",
    "unpack_npq0_l_tables",
    "validate_npq0_l_tables",
]
