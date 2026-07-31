"""Native MFQ payloads for TPQ learned product-VQ and dense int4 weights.

``CCCP-*`` was the experimental name used by the first model artifacts.  The
payload bytes are unchanged; new MFQ files use the canonical ``TPQ-*`` dtype
labels while readers accept both spellings.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CccpPqSpec:
    """One TPQ product-vector tier."""

    tier: str
    vector_size: int
    codebook_entries: int
    storage_bits: int | None = None

    def __post_init__(self) -> None:
        if self.tier not in {"x", "w", "v", "vv"}:
            raise ValueError(f"unsupported CCCP tier: {self.tier!r}")
        if self.vector_size <= 0 or self.codebook_entries <= 1:
            raise ValueError("CCCP vector size and codebook size must be positive")
        bits = self.storage_bits
        if bits is not None and (
            bits not in {8, 12, 14, 16}
            or self.codebook_entries > 1 << bits
        ):
            raise ValueError(
                "CCCP storage width must be 8/12/14/16 bits and cover "
                "the complete codebook"
            )

    @property
    def index_bits(self) -> int:
        # Legacy CCCP stores a physical byte/word dtype, while TPQ2 Kimi
        # archives may preserve row-aligned p12/p14 streams.
        if self.storage_bits is not None:
            return int(self.storage_bits)
        return 8 if self.codebook_entries <= 256 else 16

    @property
    def label(self) -> str:
        return f"TPQ-{self.tier.upper()}"

    @property
    def bpw(self) -> float:
        return self.index_bits / self.vector_size


TPQ_X = CccpPqSpec("x", 8, 256)
TPQ_W = CccpPqSpec("w", 8, 4096)
TPQ_V = CccpPqSpec("v", 4, 256)
TPQ_VV = CccpPqSpec("vv", 4, 4096)

TPQ_PQ_SPECS = {
    spec.label: spec
    for spec in (TPQ_X, TPQ_W, TPQ_V, TPQ_VV)
}
CCCP_PQ_SPECS = {
    f"CCCP-{spec.tier.upper()}": spec
    for spec in TPQ_PQ_SPECS.values()
}
TPQ_PQ_SPECS_BY_LABEL = {
    **TPQ_PQ_SPECS,
    **CCCP_PQ_SPECS,
}

# Source-compatible aliases.  Their labels intentionally resolve to TPQ so
# old Python callers write the canonical dtype when creating a new file.
CCCP_X = TPQ_X
CCCP_W = TPQ_W
CCCP_V = TPQ_V
CCCP_VV = TPQ_VV

_SPEC_BY_TIER = {spec.tier: spec for spec in TPQ_PQ_SPECS.values()}
_TIER_ID = {"x": 1, "w": 2, "v": 3, "vv": 4}
_TIER_FROM_ID = {value: key for key, value in _TIER_ID.items()}


def normalize_tpq_dtype(dtype: str) -> str:
    """Return the canonical TPQ spelling for a public MFQ dtype."""

    value = str(dtype)
    if value == "CCCP-I4G64":
        return "TPQ-I4G64"
    if value.startswith("CCCP-"):
        candidate = "TPQ-" + value[len("CCCP-") :]
        if candidate in TPQ_PQ_SPECS:
            return candidate
    return value


def legacy_cccp_dtype(dtype: str) -> str:
    """Return the historical CCCP spelling for a TPQ dtype."""

    value = normalize_tpq_dtype(dtype)
    if value == "TPQ-I4G64":
        return "CCCP-I4G64"
    if value in TPQ_PQ_SPECS:
        return "CCCP-" + value[len("TPQ-") :]
    return value


@dataclass(frozen=True)
class CccpPqTensor:
    """A row-major matrix represented by learned vector indices."""

    spec: CccpPqSpec
    shape: tuple[int, ...]
    axis: int
    neuron_len: int
    indices: np.ndarray
    codebook: np.ndarray

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("CCCP-PQ currently requires a positive 2-D matrix")
        if self.axis != 0 or self.neuron_len != shape[1]:
            raise ValueError("CCCP-PQ requires axis=0 and neuron_len=shape[1]")
        if self.neuron_len % self.spec.vector_size:
            raise ValueError(
                f"CCCP-PQ width {self.neuron_len} is not divisible by "
                f"vector size {self.spec.vector_size}"
            )
        expected_indices = (
            shape[0],
            self.neuron_len // self.spec.vector_size,
        )
        indices = np.ascontiguousarray(self.indices)
        bits = self.spec.index_bits
        if bits in {12, 14}:
            packed_nbytes = (
                expected_indices[0] * expected_indices[1] * bits + 7
            ) // 8
            if tuple(indices.shape) == expected_indices:
                if (
                    indices.size
                    and int(indices.max()) >= self.spec.codebook_entries
                ):
                    raise ValueError(
                        "CCCP-PQ index references a missing codeword"
                    )
                indices = np.frombuffer(
                    pack_cccp_indices(indices, bits),
                    dtype=np.uint8,
                ).copy()
            else:
                indices = np.ascontiguousarray(
                    indices, dtype=np.uint8
                ).reshape(-1)
                if indices.size != packed_nbytes:
                    raise ValueError(
                        f"CCCP-PQ packed indices have {indices.size} bytes; "
                        f"expected {packed_nbytes}"
                    )
        else:
            if tuple(indices.shape) != expected_indices:
                raise ValueError(
                    f"CCCP-PQ indices have shape {indices.shape}; "
                    f"expected {expected_indices}"
                )
            expected_dtype = (
                np.dtype(np.uint8)
                if bits == 8
                else np.dtype(np.uint16)
            )
            if indices.dtype != expected_dtype:
                indices = indices.astype(expected_dtype)
            if indices.size and int(indices.max()) >= self.spec.codebook_entries:
                raise ValueError(
                    "CCCP-PQ index references a missing codeword"
                )
        codebook = np.ascontiguousarray(self.codebook, dtype=np.float32)
        expected_codebook = (
            self.spec.codebook_entries,
            self.spec.vector_size,
        )
        if tuple(codebook.shape) != expected_codebook:
            raise ValueError(
                f"CCCP-PQ codebook has shape {codebook.shape}; "
                f"expected {expected_codebook}"
            )
        if not np.isfinite(codebook).all():
            raise ValueError("CCCP-PQ codebook must be finite")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "codebook", codebook)

    @property
    def payload_nbytes(self) -> int:
        return cccp_pq_payload_nbytes(self.shape, self.spec)


@dataclass(frozen=True)
class CccpInt4Tensor:
    """CCCP symmetric int4-g64 dense matrix."""

    shape: tuple[int, ...]
    axis: int
    neuron_len: int
    group_size: int
    packed: np.ndarray
    scales: np.ndarray

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("CCCP-I4 currently requires a positive 2-D matrix")
        rows, columns = shape
        if self.axis != 0 or self.neuron_len != columns:
            raise ValueError("CCCP-I4 requires axis=0 and neuron_len=shape[1]")
        if self.group_size <= 0 or columns % self.group_size or columns % 2:
            raise ValueError("CCCP-I4 width must be divisible by group size and two")
        packed = np.ascontiguousarray(self.packed, dtype=np.uint8)
        scales = np.ascontiguousarray(self.scales, dtype=np.float16)
        if tuple(packed.shape) != (rows, columns // 2):
            raise ValueError("CCCP-I4 packed values have the wrong shape")
        if tuple(scales.shape) != (rows, columns // self.group_size):
            raise ValueError("CCCP-I4 scales have the wrong shape")
        if not np.isfinite(scales).all() or np.any(scales < 0):
            raise ValueError("CCCP-I4 scales must be finite and non-negative")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "packed", packed)
        object.__setattr__(self, "scales", scales)

    @property
    def payload_nbytes(self) -> int:
        return cccp_int4_payload_nbytes(self.shape, self.group_size)


_PQ_MAGIC = b"CPQ1"
_PQ_VERSION = 1
_PQ_HEADER = struct.Struct("<4sBBBBiiII")
_INT4_MAGIC = b"CI41"
_INT4_VERSION = 1
_INT4_HEADER = struct.Struct("<4sB3xIiiI")
_MATRIX_TAIL = struct.Struct("<II")


def pack_cccp_indices(values: np.ndarray, bits: int) -> bytes:
    """Pack unsigned indices as a little-endian bitstream."""

    array = np.ascontiguousarray(values).reshape(-1)
    if array.size and int(array.max()) >= 1 << bits:
        raise ValueError(f"an index does not fit in {bits} bits")
    if bits == 8:
        return array.astype(np.uint8, copy=False).tobytes()
    if bits == 16:
        return array.astype("<u2", copy=False).tobytes()
    if bits not in {12, 14}:
        raise ValueError(
            f"CCCP indices require 8/12/14/16-bit storage, got {bits}"
        )
    values16 = array.astype(np.uint16, copy=False)
    shifts = np.arange(bits, dtype=np.uint16)
    stream = (
        (values16[:, None] >> shifts[None, :]) & 1
    ).astype(np.uint8)
    return np.packbits(stream.reshape(-1), bitorder="little").tobytes()


def unpack_cccp_indices(
    blob: bytes | memoryview,
    offset: int,
    count: int,
    bits: int,
) -> tuple[np.ndarray, int]:
    """Decode one little-endian CCCP index stream."""

    nbytes = (count * bits + 7) // 8
    end = offset + nbytes
    if end > len(blob):
        raise ValueError("truncated CCCP-PQ index stream")
    if bits == 8:
        return (
            np.frombuffer(blob, dtype=np.uint8, count=count, offset=offset).copy(),
            end,
        )
    if bits == 16:
        return (
            np.frombuffer(blob, dtype="<u2", count=count, offset=offset).copy(),
            end,
        )
    if bits not in {12, 14}:
        raise ValueError(
            f"CCCP indices require 8/12/14/16-bit storage, got {bits}"
        )
    packed = np.frombuffer(
        blob,
        dtype=np.uint8,
        count=nbytes,
        offset=offset,
    )
    stream = np.unpackbits(packed, bitorder="little")[: count * bits]
    stream = stream.reshape(count, bits)
    shifts = 1 << np.arange(bits, dtype=np.uint16)
    values = (stream.astype(np.uint16) * shifts).sum(axis=1)
    return values.astype(np.uint16), end


def cccp_pq_payload_nbytes(
    shape: tuple[int, ...],
    spec: CccpPqSpec,
) -> int:
    rows, columns = (int(value) for value in shape)
    if columns % spec.vector_size:
        raise ValueError("CCCP-PQ width must be divisible by its vector size")
    index_count = rows * (columns // spec.vector_size)
    return (
        _PQ_HEADER.size
        + 8 * len(shape)
        + 4
        + spec.codebook_entries * spec.vector_size * 4
        + (index_count * spec.index_bits + 7) // 8
    )


def pack_cccp_pq_prefix(
    spec: CccpPqSpec,
    shape: tuple[int, int],
    codebook: np.ndarray,
) -> bytes:
    """Pack the fixed metadata preceding a streamed CCCP index payload."""

    rows, columns = (int(value) for value in shape)
    table = np.ascontiguousarray(codebook, dtype=np.float32)
    if tuple(table.shape) != (spec.codebook_entries, spec.vector_size):
        raise ValueError("CCCP-PQ streamed codebook has the wrong shape")
    if columns % spec.vector_size:
        raise ValueError("CCCP-PQ streamed width is not vector aligned")
    return b"".join(
        (
            _PQ_HEADER.pack(
                _PQ_MAGIC,
                _PQ_VERSION,
                _TIER_ID[spec.tier],
                spec.vector_size,
                spec.index_bits,
                0,
                columns,
                2,
                spec.codebook_entries,
            ),
            struct.pack("<2q", rows, columns),
            struct.pack("<I", rows),
            table.astype("<f4", copy=False).tobytes(),
        )
    )


def pack_cccp_pq(tensor: CccpPqTensor) -> bytes:
    """Serialize a CCCP product-VQ matrix."""

    parts = [
        pack_cccp_pq_prefix(
            tensor.spec,
            (int(tensor.shape[0]), int(tensor.shape[1])),
            tensor.codebook,
        ),
        (
            np.ascontiguousarray(tensor.indices, dtype=np.uint8).tobytes()
            if tensor.spec.index_bits in {12, 14}
            else pack_cccp_indices(
                tensor.indices,
                tensor.spec.index_bits,
            )
        ),
    ]
    payload = b"".join(parts)
    if len(payload) != tensor.payload_nbytes:
        raise RuntimeError(
            f"CCCP-PQ payload size mismatch: {len(payload)} != "
            f"{tensor.payload_nbytes}"
        )
    return payload


def unpack_cccp_pq(blob: bytes | memoryview) -> CccpPqTensor:
    """Deserialize a CCCP product-VQ matrix."""

    if len(blob) < _PQ_HEADER.size:
        raise ValueError("truncated CCCP-PQ header")
    (
        magic,
        version,
        tier_id,
        vector_size,
        index_bits,
        axis,
        neuron_len,
        ndim,
        codebook_entries,
    ) = _PQ_HEADER.unpack_from(blob)
    if magic != _PQ_MAGIC or version != _PQ_VERSION:
        raise ValueError("invalid CCCP-PQ header")
    try:
        tier = _TIER_FROM_ID[int(tier_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported CCCP-PQ tier id: {tier_id}") from exc
    spec = CccpPqSpec(
        tier=tier,
        vector_size=int(vector_size),
        codebook_entries=int(codebook_entries),
        storage_bits=(
            int(index_bits)
            if int(index_bits) in {12, 14}
            else None
        ),
    )
    if index_bits != spec.index_bits:
        raise ValueError("CCCP-PQ tier metadata is inconsistent")
    if ndim != 2:
        raise ValueError("CCCP-PQ payload must contain a 2-D matrix")
    offset = _PQ_HEADER.size
    shape = tuple(int(value) for value in struct.unpack_from("<2q", blob, offset))
    offset += 16
    rows = struct.unpack_from("<I", blob, offset)[0]
    offset += 4
    if rows != shape[0]:
        raise ValueError("CCCP-PQ row count does not match its shape")
    codebook_count = spec.codebook_entries * spec.vector_size
    codebook_nbytes = codebook_count * 4
    if offset + codebook_nbytes > len(blob):
        raise ValueError("truncated CCCP-PQ codebook")
    codebook = np.frombuffer(
        blob, dtype="<f4", count=codebook_count, offset=offset
    ).astype(np.float32, copy=True)
    codebook = codebook.reshape(spec.codebook_entries, spec.vector_size)
    offset += codebook_nbytes
    index_count = rows * (shape[1] // spec.vector_size)
    if spec.index_bits in {12, 14}:
        index_nbytes = (index_count * spec.index_bits + 7) // 8
        indices = np.frombuffer(
            blob,
            dtype=np.uint8,
            count=index_nbytes,
            offset=offset,
        )
        offset += index_nbytes
    else:
        indices, offset = unpack_cccp_indices(
            blob, offset, index_count, spec.index_bits
        )
    if offset != len(blob):
        raise ValueError(f"invalid CCCP-PQ tail: {len(blob) - offset} bytes")
    return CccpPqTensor(
        spec=spec,
        shape=shape,
        axis=int(axis),
        neuron_len=int(neuron_len),
        indices=(
            indices
            if spec.index_bits in {12, 14}
            else indices.reshape(rows, shape[1] // spec.vector_size)
        ),
        codebook=codebook,
    )


def pack_cccp_int4(tensor: CccpInt4Tensor) -> bytes:
    """Serialize a CCCP symmetric int4 matrix."""

    return b"".join(
        (
            pack_cccp_int4_prefix(tensor.shape, tensor.group_size),
            tensor.packed.tobytes(),
            tensor.scales.astype("<f2", copy=False).tobytes(),
        )
    )


def cccp_int4_payload_nbytes(
    shape: tuple[int, ...],
    group_size: int = 64,
) -> int:
    """Return the exact native payload size for one CCCP int4 matrix."""

    if len(shape) != 2:
        raise ValueError("CCCP-I4 payload size requires a 2-D shape")
    rows, columns = (int(value) for value in shape)
    if rows <= 0 or columns <= 0 or columns % group_size or columns % 2:
        raise ValueError("CCCP-I4 payload shape is not group aligned")
    return (
        _INT4_HEADER.size
        + 16
        + _MATRIX_TAIL.size
        + rows * (columns // 2)
        + rows * (columns // group_size) * 2
    )


def pack_cccp_int4_prefix(
    shape: tuple[int, ...],
    group_size: int = 64,
) -> bytes:
    """Pack fixed metadata preceding streamed CCCP int4 values."""

    if len(shape) != 2:
        raise ValueError("CCCP-I4 prefix requires a 2-D shape")
    rows, columns = (int(value) for value in shape)
    cccp_int4_payload_nbytes((rows, columns), group_size)
    return b"".join(
        (
            _INT4_HEADER.pack(
                _INT4_MAGIC,
                _INT4_VERSION,
                group_size,
                0,
                columns,
                2,
            ),
            struct.pack("<2q", rows, columns),
            _MATRIX_TAIL.pack(rows, columns // group_size),
        )
    )


def unpack_cccp_int4(blob: bytes | memoryview) -> CccpInt4Tensor:
    """Deserialize a CCCP symmetric int4 matrix."""

    if len(blob) < _INT4_HEADER.size:
        raise ValueError("truncated CCCP-I4 header")
    (
        magic,
        version,
        group_size,
        axis,
        neuron_len,
        ndim,
    ) = _INT4_HEADER.unpack_from(blob)
    if magic != _INT4_MAGIC or version != _INT4_VERSION or ndim != 2:
        raise ValueError("invalid CCCP-I4 header")
    offset = _INT4_HEADER.size
    shape = tuple(int(value) for value in struct.unpack_from("<2q", blob, offset))
    offset += 16
    rows, groups = _MATRIX_TAIL.unpack_from(blob, offset)
    offset += _MATRIX_TAIL.size
    if rows != shape[0] or neuron_len != shape[1]:
        raise ValueError("CCCP-I4 dimensions are inconsistent")
    packed_count = rows * (neuron_len // 2)
    scale_count = rows * groups
    expected_end = offset + packed_count + scale_count * 2
    if expected_end != len(blob):
        raise ValueError(
            f"invalid CCCP-I4 payload size: {len(blob)} != {expected_end}"
        )
    packed = np.frombuffer(
        blob, dtype=np.uint8, count=packed_count, offset=offset
    ).copy()
    offset += packed_count
    scales = np.frombuffer(
        blob, dtype="<f2", count=scale_count, offset=offset
    ).astype(np.float16, copy=True)
    return CccpInt4Tensor(
        shape=shape,
        axis=int(axis),
        neuron_len=int(neuron_len),
        group_size=int(group_size),
        packed=packed.reshape(rows, neuron_len // 2),
        scales=scales.reshape(rows, groups),
    )


# Canonical public API.  The original names stay available for source
# compatibility and refer to the exact same classes and payload functions.
TpqPqSpec = CccpPqSpec
TpqPqTensor = CccpPqTensor
TpqInt4Tensor = CccpInt4Tensor
tpq_int4_payload_nbytes = cccp_int4_payload_nbytes
tpq_pq_payload_nbytes = cccp_pq_payload_nbytes
pack_tpq_indices = pack_cccp_indices
pack_tpq_int4 = pack_cccp_int4
pack_tpq_int4_prefix = pack_cccp_int4_prefix
pack_tpq_pq = pack_cccp_pq
pack_tpq_pq_prefix = pack_cccp_pq_prefix
unpack_tpq_indices = unpack_cccp_indices
unpack_tpq_int4 = unpack_cccp_int4
unpack_tpq_pq = unpack_cccp_pq


__all__ = [
    "CCCP_PQ_SPECS",
    "CCCP_V",
    "CCCP_VV",
    "CCCP_W",
    "CCCP_X",
    "TPQ_PQ_SPECS",
    "TPQ_PQ_SPECS_BY_LABEL",
    "TPQ_V",
    "TPQ_VV",
    "TPQ_W",
    "TPQ_X",
    "CccpInt4Tensor",
    "CccpPqSpec",
    "CccpPqTensor",
    "TpqInt4Tensor",
    "TpqPqSpec",
    "TpqPqTensor",
    "cccp_int4_payload_nbytes",
    "cccp_pq_payload_nbytes",
    "legacy_cccp_dtype",
    "normalize_tpq_dtype",
    "pack_cccp_indices",
    "pack_cccp_int4",
    "pack_cccp_int4_prefix",
    "pack_cccp_pq",
    "pack_cccp_pq_prefix",
    "pack_tpq_indices",
    "pack_tpq_int4",
    "pack_tpq_int4_prefix",
    "pack_tpq_pq",
    "pack_tpq_pq_prefix",
    "tpq_int4_payload_nbytes",
    "tpq_pq_payload_nbytes",
    "unpack_cccp_indices",
    "unpack_cccp_int4",
    "unpack_cccp_pq",
    "unpack_tpq_indices",
    "unpack_tpq_int4",
    "unpack_tpq_pq",
]
