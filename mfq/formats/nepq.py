"""Cross-expert neuron-anchored VQ/PQ formats.

NEPQ stores one codebook-table pool for a logical ``[experts, out, K]``
projection.  Every four consecutive gs24 groups share one uint8 table-bank
selector.  Per-group state/scale and auxiliary streams, per-vector indices,
and per-neuron FP16 anchors keep the semantics of the corresponding NPQ/NVQ
base format.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np

from mfq.formats.npq0_l import NPQ0_L_TABLE_BYTES, unpack_npq0_l_tables
from mfq.formats.npq0_s import NPQ0_S_TABLE_BYTES, unpack_npq0_s_tables
from mfq.formats.nvq1_l import unpack_ternary_codebook
from mfq.formats.nvq1_s import (
    NVQ1_S_TABLE_BYTES,
    unpack_nvq1_s_banked_codebook,
)


_GROUP_SIZE = 24
_VECTOR_SIZE = 8
_GROUPS_PER_SUPERGROUP = 4
_NVQ1_L_TABLE_BYTES = 4096


@dataclass(frozen=True)
class NepqSpec:
    label: str
    profile_id: int
    state_bits: int
    index_bits: int
    aux_bits: int
    table_bytes: int
    runtime_table_bytes: int
    base_format_id: int
    base_profile_id: int | None = None
    residual_block_vectors: int = 0
    residual_second: bool = False

    @property
    def groupsize(self) -> int:
        return _GROUP_SIZE

    @property
    def vector_size(self) -> int:
        return _VECTOR_SIZE

    @property
    def groups_per_supergroup(self) -> int:
        return _GROUPS_PER_SUPERGROUP

    @property
    def is_residual(self) -> bool:
        return self.base_profile_id is not None

    @property
    def residual_position_bits(self) -> int:
        if not self.is_residual:
            return 0
        return int(math.ceil(math.log2(self.residual_block_vectors)))

    @property
    def residual_record_bits(self) -> int:
        return self.residual_position_bits + 10 if self.is_residual else 0

    def payload_nbytes(
        self,
        n_experts: int,
        out_per_expert: int,
        neuron_len: int,
        *,
        bank_count: int,
    ) -> int:
        _validate_dimensions(n_experts, out_per_expert, neuron_len, bank_count)
        rows = n_experts * out_per_expert
        ng = math.ceil(neuron_len / self.groupsize)
        nvec = neuron_len // self.vector_size
        nsuper = math.ceil(ng / self.groups_per_supergroup)
        anchors = 2 * rows
        states = _packed_nbytes(rows * ng, self.state_bits)
        indices = _packed_nbytes(rows * nvec, self.index_bits)
        aux = _packed_nbytes(rows * ng, self.aux_bits)
        selectors = rows * nsuper
        tables = bank_count * self.table_bytes
        return tables + anchors + states + indices + aux + selectors

    def bpw(
        self,
        neuron_len: int,
        *,
        n_experts: int = 1,
        out_per_expert: int = 1,
        bank_count: int = 1,
    ) -> float:
        weights = n_experts * out_per_expert * neuron_len
        return 8.0 * self.payload_nbytes(
            n_experts,
            out_per_expert,
            neuron_len,
            bank_count=bank_count,
        ) / weights


NEPQ0_S = NepqSpec(
    label="NEPQ0-S",
    profile_id=0,
    state_bits=2,
    index_bits=6,
    aux_bits=0,
    table_bytes=NPQ0_S_TABLE_BYTES,
    runtime_table_bytes=NPQ0_S_TABLE_BYTES,
    base_format_id=9,
)
NEPQ0_L = NepqSpec(
    label="NEPQ0-L",
    profile_id=1,
    state_bits=3,
    index_bits=7,
    aux_bits=0,
    table_bytes=NPQ0_L_TABLE_BYTES,
    runtime_table_bytes=NPQ0_L_TABLE_BYTES,
    base_format_id=7,
)
NEPQ1_S = NepqSpec(
    label="NEPQ1-S",
    profile_id=2,
    state_bits=4,
    index_bits=9,
    aux_bits=1,
    table_bytes=NVQ1_S_TABLE_BYTES,
    runtime_table_bytes=1024 * 8,
    base_format_id=8,
)
NEPQ1_L = NepqSpec(
    label="NEPQ1-L",
    profile_id=3,
    state_bits=3,
    index_bits=11,
    aux_bits=1,
    table_bytes=_NVQ1_L_TABLE_BYTES,
    runtime_table_bytes=2048 * 8,
    base_format_id=1,
)
NEPQ0_A = NepqSpec(
    label="NEPQ0-A",
    profile_id=4,
    state_bits=NEPQ0_S.state_bits,
    index_bits=NEPQ0_S.index_bits,
    aux_bits=NEPQ0_S.aux_bits,
    table_bytes=NEPQ0_S.table_bytes,
    runtime_table_bytes=NEPQ0_S.runtime_table_bytes,
    base_format_id=NEPQ0_S.base_format_id,
    base_profile_id=NEPQ0_S.profile_id,
    residual_block_vectors=24,
)
NEPQ1_A = NepqSpec(
    label="NEPQ1-A",
    profile_id=5,
    state_bits=NEPQ1_S.state_bits,
    index_bits=NEPQ1_S.index_bits,
    aux_bits=NEPQ1_S.aux_bits,
    table_bytes=NEPQ1_S.table_bytes,
    runtime_table_bytes=NEPQ1_S.runtime_table_bytes,
    base_format_id=NEPQ1_S.base_format_id,
    base_profile_id=NEPQ1_S.profile_id,
    residual_block_vectors=16,
    residual_second=True,
)
NEPQ_SPECS = (
    NEPQ0_S,
    NEPQ0_L,
    NEPQ1_S,
    NEPQ1_L,
    NEPQ0_A,
    NEPQ1_A,
)
_SPEC_BY_ID = {spec.profile_id: spec for spec in NEPQ_SPECS}
_SPEC_BY_LABEL = {spec.label: spec for spec in NEPQ_SPECS}


def rotation_signs(width: int, block: int, seed: int) -> np.ndarray:
    """Build the deterministic signed-Hadamard diagonal for one tensor."""

    family = f"h{int(block)}"
    digest = hashlib.blake2b(
        f"{family}:{int(seed)}:{int(width)}".encode(), digest_size=16
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    return np.where(rng.integers(0, 2, size=int(width)), 1, -1).astype(np.int8)


def nepq_spec(value: str | int | NepqSpec) -> NepqSpec:
    if isinstance(value, NepqSpec):
        return value
    if isinstance(value, str):
        try:
            return _SPEC_BY_LABEL[value]
        except KeyError as exc:
            raise ValueError(f"unknown NEPQ profile: {value}") from exc
    try:
        return _SPEC_BY_ID[int(value)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unknown NEPQ profile id: {value}") from exc


def nepq_base_spec(value: str | int | NepqSpec) -> NepqSpec:
    """Return the base profile used by a plain or residual NEPQ tensor."""

    spec = nepq_spec(value)
    return spec if spec.base_profile_id is None else nepq_spec(spec.base_profile_id)


@dataclass
class NepqTensor:
    spec: NepqSpec
    shape: tuple[int, int, int]
    neuron_scale: np.ndarray
    state: np.ndarray
    indices: np.ndarray
    aux: np.ndarray | None
    bank_ids: np.ndarray
    table_payloads: np.ndarray
    rotation_block: int = 0
    rotation_seed: int = 0
    residual_codebook: np.ndarray | None = None
    residual_first: np.ndarray | None = None
    residual_second_mask: np.ndarray | None = None
    residual_second_records: np.ndarray | None = None
    residual_padding_nbytes: int = 0

    @property
    def n_experts(self) -> int:
        return int(self.shape[0])

    @property
    def out_per_expert(self) -> int:
        return int(self.shape[1])

    @property
    def neuron_len(self) -> int:
        return int(self.shape[2])

    @property
    def bank_count(self) -> int:
        value = np.asarray(self.table_payloads)
        return int(value.shape[0]) if value.ndim == 2 else 0

    @property
    def payload_nbytes(self) -> int:
        size = self.spec.payload_nbytes(
            self.n_experts,
            self.out_per_expert,
            self.neuron_len,
            bank_count=self.bank_count,
        )
        if self.spec.is_residual:
            blocks = self.residual_block_count
            second = np.asarray(
                self.residual_second_records
                if self.residual_second_records is not None
                else np.empty(0, dtype=np.uint16)
            ).size
            size += (
                _RESIDUAL_HEADER.size
                + _RESIDUAL_DICTIONARY_ENTRIES * _VECTOR_SIZE * 2
                + _packed_nbytes(blocks, self.spec.residual_record_bits)
                + (
                    _packed_nbytes(blocks, 1)
                    + _packed_nbytes(second, self.spec.residual_record_bits)
                    if self.spec.residual_second
                    else 0
                )
                + int(self.residual_padding_nbytes)
            )
        return size

    @property
    def payload_bpw(self) -> float:
        return 8.0 * self.payload_nbytes / int(np.prod(self.shape))

    @property
    def residual_blocks_per_row(self) -> int:
        if not self.spec.is_residual:
            return 0
        vectors = self.neuron_len // self.spec.vector_size
        return math.ceil(vectors / self.spec.residual_block_vectors)

    @property
    def residual_block_count(self) -> int:
        return self.n_experts * self.out_per_expert * self.residual_blocks_per_row


_MAGIC = b"NEP1"
_VERSION = 1
_FLAG_ROTATED = 1
_HEADER = struct.Struct("<4sBBBBIIIIIQ")
_RESIDUAL_MAGIC = b"NRA1"
_RESIDUAL_VERSION = 1
_RESIDUAL_FLAG_SECOND = 1
_RESIDUAL_DICTIONARY_ENTRIES = 1024
_RESIDUAL_HEADER = struct.Struct("<4sBBBBIIIIQ32x")


def _packed_nbytes(count: int, bits: int) -> int:
    return (count * bits + 7) // 8 if bits else 0


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    if bits == 0:
        if np.asarray(values).size:
            raise ValueError("zero-bit stream must be empty")
        return b""
    value = np.ascontiguousarray(values).reshape(-1)
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError("packed NEPQ stream must contain integers")
    if np.any(value < 0) or np.any(value >= (1 << bits)):
        raise ValueError(f"NEPQ stream value exceeds {bits} bits")
    if bits == 8:
        return value.astype(np.uint8, copy=False).tobytes()
    bit_rows = np.unpackbits(
        value.astype(np.uint16, copy=False).view(np.uint8).reshape(-1, 2),
        axis=1,
        bitorder="little",
    )[:, :bits]
    return np.packbits(bit_rows.reshape(-1), bitorder="little").tobytes()


def _unpack_bits(
    blob: bytes | memoryview,
    offset: int,
    count: int,
    bits: int,
) -> tuple[np.ndarray, int]:
    nbytes = _packed_nbytes(count, bits)
    end = offset + nbytes
    if end > len(blob):
        raise ValueError("truncated NEPQ bit stream")
    if bits == 0 or count == 0:
        return np.empty(0, dtype=np.uint8), end
    packed = np.frombuffer(blob, dtype=np.uint8, count=nbytes, offset=offset)
    rows = np.unpackbits(packed, bitorder="little")[: count * bits].reshape(count, bits)
    shifts = 1 << np.arange(bits, dtype=np.uint16)
    dtype = np.uint16 if bits > 8 else np.uint8
    return (rows.astype(np.uint16) * shifts).sum(1).astype(dtype), end


def _validate_dimensions(
    n_experts: int,
    out_per_expert: int,
    neuron_len: int,
    bank_count: int,
) -> None:
    if n_experts <= 0 or out_per_expert <= 0 or neuron_len <= 0:
        raise ValueError("NEPQ shape dimensions must be positive")
    if neuron_len % _VECTOR_SIZE:
        raise ValueError("NEPQ K must be divisible by 8")
    if not 1 <= bank_count <= 256:
        raise ValueError("NEPQ bank count must be in [1,256]")


def _validate_table_payload(spec: NepqSpec, payload: bytes) -> None:
    base = nepq_base_spec(spec)
    if len(payload) != spec.table_bytes:
        raise ValueError(
            f"{spec.label} bank must contain {spec.table_bytes} bytes"
        )
    if base is NEPQ0_S:
        _, _, _, consumed = unpack_npq0_s_tables(payload)
    elif base is NEPQ0_L:
        _, _, _, consumed = unpack_npq0_l_tables(payload)
    elif base is NEPQ1_S:
        unpack_nvq1_s_banked_codebook(payload)
        consumed = len(payload)
    elif base is NEPQ1_L:
        unpack_ternary_codebook(payload)
        consumed = len(payload)
    else:
        raise ValueError(f"unsupported NEPQ spec: {spec.label}")
    if consumed != len(payload):
        raise ValueError(f"{spec.label} bank has an invalid tail")


def validate_nepq(tensor: NepqTensor) -> tuple[int, int, int, int, int]:
    spec = nepq_spec(tensor.spec)
    shape = tuple(int(value) for value in tensor.shape)
    if len(shape) != 3:
        raise ValueError("NEPQ shape must be [experts,out,K]")
    tables = np.asarray(tensor.table_payloads)
    bank_count = int(tables.shape[0]) if tables.ndim == 2 else 0
    _validate_dimensions(*shape, bank_count)
    if tables.shape != (bank_count, spec.table_bytes):
        raise ValueError(
            f"bad {spec.label} table-pool shape: {tables.shape}, expected "
            f"{(bank_count, spec.table_bytes)}"
        )
    for table in np.ascontiguousarray(tables, dtype=np.uint8):
        _validate_table_payload(spec, table.tobytes())

    n_experts, out_per_expert, neuron_len = shape
    ng = math.ceil(neuron_len / spec.groupsize)
    nvec = neuron_len // spec.vector_size
    nsuper = math.ceil(ng / spec.groups_per_supergroup)
    expected = {
        "neuron_scale": (n_experts, out_per_expert),
        "state": (n_experts, out_per_expert, ng),
        "indices": (n_experts, out_per_expert, nvec),
        "bank_ids": (n_experts, out_per_expert, nsuper),
    }
    for name, expected_shape in expected.items():
        value = np.asarray(getattr(tensor, name))
        if value.shape != expected_shape:
            raise ValueError(
                f"bad {spec.label} {name} shape: {value.shape}, expected {expected_shape}"
            )
    anchor = np.asarray(tensor.neuron_scale, dtype=np.float32)
    if not np.isfinite(anchor).all() or np.any(anchor < 0):
        raise ValueError(f"{spec.label} anchors must be finite and non-negative")
    state = np.asarray(tensor.state)
    indices = np.asarray(tensor.indices)
    bank_ids = np.asarray(tensor.bank_ids)
    for name, value, bits in (
        ("state", state, spec.state_bits),
        ("indices", indices, spec.index_bits),
    ):
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{spec.label} {name} must contain integers")
        if np.any(value < 0) or np.any(value >= (1 << bits)):
            raise ValueError(f"{spec.label} {name} exceeds {bits} bits")
    if not np.issubdtype(bank_ids.dtype, np.integer):
        raise ValueError(f"{spec.label} bank IDs must contain integers")
    if np.any(bank_ids < 0) or np.any(bank_ids >= bank_count):
        raise ValueError(f"{spec.label} bank ID exceeds the table pool")

    aux = np.empty(0, dtype=np.uint8) if tensor.aux is None else np.asarray(tensor.aux)
    expected_aux = (n_experts, out_per_expert, ng) if spec.aux_bits else (0,)
    if aux.shape != expected_aux:
        raise ValueError(
            f"bad {spec.label} aux shape: {aux.shape}, expected {expected_aux}"
        )
    if spec.aux_bits:
        if not np.issubdtype(aux.dtype, np.integer):
            raise ValueError(f"{spec.label} aux must contain integers")
        if np.any(aux < 0) or np.any(aux >= (1 << spec.aux_bits)):
            raise ValueError(f"{spec.label} aux exceeds {spec.aux_bits} bits")

    block = int(tensor.rotation_block)
    seed = int(tensor.rotation_seed)
    if block:
        if block & (block - 1) or neuron_len % block:
            raise ValueError("NEPQ rotation block must be a power of two dividing K")
        if seed < 0 or seed > (1 << 64) - 1:
            raise ValueError("NEPQ rotation seed must fit uint64")
    elif seed:
        raise ValueError("NEPQ rotation seed requires a nonzero block")

    codebook = tensor.residual_codebook
    first = tensor.residual_first
    second_mask = tensor.residual_second_mask
    second_records = tensor.residual_second_records
    padding = int(tensor.residual_padding_nbytes)
    if not spec.is_residual:
        if any(value is not None for value in (codebook, first, second_mask, second_records)):
            raise ValueError(f"{spec.label} cannot carry sparse residual streams")
        if padding:
            raise ValueError(f"{spec.label} cannot carry residual padding")
        return ng, nvec, nsuper, bank_count, n_experts * out_per_expert
    if not block:
        raise ValueError(f"{spec.label} requires a Hadamard rotation")
    if padding < 0 or padding > (1 << 32) - 1:
        raise ValueError(f"{spec.label} residual padding must fit uint32")
    dictionary = np.asarray(codebook)
    expected_dictionary = (_RESIDUAL_DICTIONARY_ENTRIES, spec.vector_size)
    if dictionary.shape != expected_dictionary:
        raise ValueError(
            f"bad {spec.label} residual dictionary shape: {dictionary.shape}, "
            f"expected {expected_dictionary}"
        )
    if not np.isfinite(dictionary.astype(np.float32)).all():
        raise ValueError(f"{spec.label} residual dictionary must be finite")
    blocks_per_row = math.ceil(nvec / spec.residual_block_vectors)
    expected_first = (n_experts, out_per_expert, blocks_per_row)
    first_values = np.asarray(first)
    if first_values.shape != expected_first:
        raise ValueError(
            f"bad {spec.label} first residual shape: {first_values.shape}, "
            f"expected {expected_first}"
        )
    _validate_residual_records(spec, first_values, nvec, blocks_per_row, "first")
    if spec.residual_second:
        mask = np.asarray(second_mask)
        if mask.shape != expected_first:
            raise ValueError(
                f"bad {spec.label} second residual mask shape: {mask.shape}, "
                f"expected {expected_first}"
            )
        if not np.issubdtype(mask.dtype, np.integer) and mask.dtype != np.bool_:
            raise ValueError(f"{spec.label} second residual mask must be binary")
        if np.any((mask != 0) & (mask != 1)):
            raise ValueError(f"{spec.label} second residual mask must be binary")
        records = np.asarray(second_records)
        if records.ndim != 1 or records.size != int(np.count_nonzero(mask)):
            raise ValueError(
                f"{spec.label} compact second residual count does not match its mask"
            )
        _validate_residual_records(
            spec,
            records,
            nvec,
            blocks_per_row,
            "second",
            selected_blocks=np.flatnonzero(mask.reshape(-1)),
        )
    else:
        if second_mask is not None or second_records is not None:
            raise ValueError(f"{spec.label} has no second residual stream")
    return ng, nvec, nsuper, bank_count, n_experts * out_per_expert


def _validate_residual_records(
    spec: NepqSpec,
    records: np.ndarray,
    nvec: int,
    blocks_per_row: int,
    name: str,
    *,
    selected_blocks: np.ndarray | None = None,
) -> None:
    values = np.asarray(records)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{spec.label} {name} residual records must contain integers")
    flat = values.reshape(-1).astype(np.uint32, copy=False)
    if np.any(flat >= (1 << spec.residual_record_bits)):
        raise ValueError(
            f"{spec.label} {name} residual record exceeds "
            f"{spec.residual_record_bits} bits"
        )
    position_mask = (1 << spec.residual_position_bits) - 1
    positions = flat & position_mask
    dictionary_ids = flat >> spec.residual_position_bits
    if np.any(dictionary_ids >= _RESIDUAL_DICTIONARY_ENTRIES):
        raise ValueError(f"{spec.label} {name} residual dictionary index is invalid")
    if selected_blocks is None:
        block_ids = np.arange(flat.size, dtype=np.int64)
    else:
        block_ids = np.asarray(selected_blocks, dtype=np.int64)
        if block_ids.size != flat.size:
            raise ValueError(f"{spec.label} {name} residual block mapping is invalid")
    block_in_row = block_ids % blocks_per_row
    available = np.minimum(
        spec.residual_block_vectors,
        nvec - block_in_row * spec.residual_block_vectors,
    )
    if np.any(positions >= available):
        raise ValueError(f"{spec.label} {name} residual position exceeds its block")


def pack_nepq(tensor: NepqTensor) -> bytes:
    ng, nvec, nsuper, bank_count, rows = validate_nepq(tensor)
    spec = nepq_spec(tensor.spec)
    flags = _FLAG_ROTATED if tensor.rotation_block else 0
    n_experts, out_per_expert, neuron_len = (int(value) for value in tensor.shape)
    aux = np.empty(0, dtype=np.uint8) if tensor.aux is None else np.asarray(tensor.aux)
    parts = [
            _HEADER.pack(
                _MAGIC,
                _VERSION,
                spec.profile_id,
                spec.groups_per_supergroup,
                flags,
                n_experts,
                out_per_expert,
                neuron_len,
                bank_count,
                int(tensor.rotation_block),
                int(tensor.rotation_seed),
            ),
            np.ascontiguousarray(tensor.table_payloads, dtype=np.uint8).tobytes(),
            np.ascontiguousarray(tensor.neuron_scale, dtype="<f2").tobytes(),
            _pack_bits(np.asarray(tensor.state), spec.state_bits),
            _pack_bits(np.asarray(tensor.indices), spec.index_bits),
            _pack_bits(aux, spec.aux_bits),
            np.ascontiguousarray(tensor.bank_ids, dtype=np.uint8).tobytes(),
        ]
    if spec.is_residual:
        first = np.asarray(tensor.residual_first)
        second = np.asarray(
            tensor.residual_second_records
            if tensor.residual_second_records is not None
            else np.empty(0, dtype=np.uint16)
        )
        mask = np.asarray(
            tensor.residual_second_mask
            if tensor.residual_second_mask is not None
            else np.empty(0, dtype=np.uint8)
        )
        flags = _RESIDUAL_FLAG_SECOND if spec.residual_second else 0
        parts.extend(
            [
                _RESIDUAL_HEADER.pack(
                    _RESIDUAL_MAGIC,
                    _RESIDUAL_VERSION,
                    spec.residual_record_bits,
                    spec.residual_position_bits,
                    flags,
                    _RESIDUAL_DICTIONARY_ENTRIES,
                    tensor.residual_block_count,
                    second.size,
                    int(tensor.residual_padding_nbytes),
                    0,
                ),
                np.ascontiguousarray(tensor.residual_codebook, dtype="<f2").tobytes(),
                _pack_bits(first, spec.residual_record_bits),
            ]
        )
        if spec.residual_second:
            parts.extend(
                [
                    _pack_bits(mask, 1),
                    _pack_bits(second, spec.residual_record_bits),
                ]
            )
        if tensor.residual_padding_nbytes:
            parts.append(bytes(int(tensor.residual_padding_nbytes)))
    return b"".join(parts)


def unpack_nepq(blob: bytes | memoryview) -> NepqTensor:
    if len(blob) < _HEADER.size:
        raise ValueError("truncated NEPQ header")
    (
        magic,
        version,
        profile_id,
        groups_per_supergroup,
        flags,
        n_experts,
        out_per_expert,
        neuron_len,
        bank_count,
        rotation_block,
        rotation_seed,
    ) = _HEADER.unpack_from(blob)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("invalid or unsupported NEPQ header")
    spec = nepq_spec(profile_id)
    if groups_per_supergroup != spec.groups_per_supergroup:
        raise ValueError("unsupported NEPQ super-group profile")
    if flags & ~_FLAG_ROTATED:
        raise ValueError("unsupported NEPQ flags")
    if bool(flags & _FLAG_ROTATED) != bool(rotation_block):
        raise ValueError("inconsistent NEPQ rotation flag")
    _validate_dimensions(n_experts, out_per_expert, neuron_len, bank_count)
    rows = int(n_experts) * int(out_per_expert)
    ng = math.ceil(int(neuron_len) / spec.groupsize)
    nvec = int(neuron_len) // spec.vector_size
    nsuper = math.ceil(ng / spec.groups_per_supergroup)
    offset = _HEADER.size
    table_nbytes = int(bank_count) * spec.table_bytes
    if offset + table_nbytes > len(blob):
        raise ValueError("truncated NEPQ table pool")
    table_payloads = np.frombuffer(
        blob, dtype=np.uint8, count=table_nbytes, offset=offset
    ).copy().reshape(int(bank_count), spec.table_bytes)
    offset += table_nbytes
    anchor_nbytes = 2 * rows
    if offset + anchor_nbytes > len(blob):
        raise ValueError("truncated NEPQ neuron anchors")
    neuron_scale = np.frombuffer(
        blob, dtype="<f2", count=rows, offset=offset
    ).astype(np.float32).reshape(int(n_experts), int(out_per_expert))
    offset += anchor_nbytes
    state, offset = _unpack_bits(blob, offset, rows * ng, spec.state_bits)
    indices, offset = _unpack_bits(blob, offset, rows * nvec, spec.index_bits)
    aux, offset = _unpack_bits(blob, offset, rows * ng, spec.aux_bits)
    selector_count = rows * nsuper
    selector_end = offset + selector_count
    if selector_end > len(blob):
        raise ValueError("truncated NEPQ bank selectors")
    bank_ids = np.frombuffer(
        blob, dtype=np.uint8, count=selector_count, offset=offset
    ).copy().reshape(int(n_experts), int(out_per_expert), nsuper)
    offset = selector_end
    residual_codebook = None
    residual_first = None
    residual_second_mask = None
    residual_second_records = None
    residual_padding_nbytes = 0
    if spec.is_residual:
        if offset + _RESIDUAL_HEADER.size > len(blob):
            raise ValueError("truncated NEPQ sparse residual header")
        (
            residual_magic,
            residual_version,
            record_bits,
            position_bits,
            residual_flags,
            dictionary_entries,
            block_count,
            second_count,
            residual_padding_nbytes,
            reserved,
        ) = _RESIDUAL_HEADER.unpack_from(blob, offset)
        offset += _RESIDUAL_HEADER.size
        expected_flags = _RESIDUAL_FLAG_SECOND if spec.residual_second else 0
        expected_blocks_per_row = math.ceil(nvec / spec.residual_block_vectors)
        expected_blocks = rows * expected_blocks_per_row
        if (
            residual_magic != _RESIDUAL_MAGIC
            or residual_version != _RESIDUAL_VERSION
            or record_bits != spec.residual_record_bits
            or position_bits != spec.residual_position_bits
            or residual_flags != expected_flags
            or dictionary_entries != _RESIDUAL_DICTIONARY_ENTRIES
            or block_count != expected_blocks
            or reserved != 0
        ):
            raise ValueError("invalid or unsupported NEPQ sparse residual header")
        dictionary_values = _RESIDUAL_DICTIONARY_ENTRIES * spec.vector_size
        dictionary_nbytes = dictionary_values * 2
        if offset + dictionary_nbytes > len(blob):
            raise ValueError("truncated NEPQ sparse residual dictionary")
        residual_codebook = np.frombuffer(
            blob, dtype="<f2", count=dictionary_values, offset=offset
        ).copy().reshape(_RESIDUAL_DICTIONARY_ENTRIES, spec.vector_size)
        offset += dictionary_nbytes
        residual_first, offset = _unpack_bits(
            blob, offset, expected_blocks, spec.residual_record_bits
        )
        residual_first = residual_first.reshape(
            int(n_experts), int(out_per_expert), expected_blocks_per_row
        )
        if spec.residual_second:
            residual_second_mask, offset = _unpack_bits(
                blob, offset, expected_blocks, 1
            )
            residual_second_mask = residual_second_mask.reshape(
                int(n_experts), int(out_per_expert), expected_blocks_per_row
            )
            if int(np.count_nonzero(residual_second_mask)) != int(second_count):
                raise ValueError("NEPQ second residual count does not match its mask")
            residual_second_records, offset = _unpack_bits(
                blob, offset, int(second_count), spec.residual_record_bits
            )
        elif second_count:
            raise ValueError(f"{spec.label} cannot contain second residual records")
        tail_end = offset + int(residual_padding_nbytes)
        if tail_end > len(blob):
            raise ValueError("truncated NEPQ sparse residual padding")
        if any(blob[offset:tail_end]):
            raise ValueError("NEPQ sparse residual padding must be zero")
        offset = tail_end
    if offset != len(blob):
        raise ValueError(f"invalid NEPQ tail: consumed={offset}, size={len(blob)}")
    tensor = NepqTensor(
        spec=spec,
        shape=(int(n_experts), int(out_per_expert), int(neuron_len)),
        neuron_scale=neuron_scale,
        state=state.reshape(int(n_experts), int(out_per_expert), ng),
        indices=indices.reshape(int(n_experts), int(out_per_expert), nvec),
        aux=(
            aux.reshape(int(n_experts), int(out_per_expert), ng)
            if spec.aux_bits
            else None
        ),
        bank_ids=bank_ids,
        table_payloads=table_payloads,
        rotation_block=int(rotation_block),
        rotation_seed=int(rotation_seed),
        residual_codebook=residual_codebook,
        residual_first=residual_first,
        residual_second_mask=residual_second_mask,
        residual_second_records=residual_second_records,
        residual_padding_nbytes=int(residual_padding_nbytes),
    )
    validate_nepq(tensor)
    return tensor


def _decoded_tables(tensor: NepqTensor) -> object:
    tables = [row.tobytes() for row in np.asarray(tensor.table_payloads, dtype=np.uint8)]
    base = nepq_base_spec(tensor.spec)
    if base is NEPQ0_S:
        unpacked = [unpack_npq0_s_tables(table)[:3] for table in tables]
        return tuple(np.stack(items) for items in zip(*unpacked, strict=True))
    if base is NEPQ0_L:
        unpacked = [unpack_npq0_l_tables(table)[:3] for table in tables]
        return tuple(np.stack(items) for items in zip(*unpacked, strict=True))
    if base is NEPQ1_S:
        return np.stack([unpack_nvq1_s_banked_codebook(table) for table in tables])
    if base is NEPQ1_L:
        return np.stack([unpack_ternary_codebook(table) for table in tables])
    raise ValueError(f"unsupported NEPQ spec: {tensor.spec.label}")


def dequantize_nepq(tensor: NepqTensor) -> np.ndarray:
    """Decode the stored (possibly rotated) expert matrices to float32."""

    validate_nepq(tensor)
    result = dequantize_nepq_rows(
        tensor,
        0,
        tensor.n_experts * tensor.out_per_expert,
        validate=False,
    )
    return result.reshape(tensor.shape)


def dequantize_nepq_rows(
    tensor: NepqTensor,
    start: int,
    stop: int,
    *,
    validate: bool = True,
    decoded_tables: object | None = None,
) -> np.ndarray:
    """Decode a contiguous flattened row range in stored coordinates."""

    if validate:
        ng, nvec, _, _, rows = validate_nepq(tensor)
    else:
        rows = tensor.n_experts * tensor.out_per_expert
        ng = math.ceil(tensor.neuron_len / tensor.spec.groupsize)
        nvec = tensor.neuron_len // tensor.spec.vector_size
    start = int(start)
    stop = int(stop)
    if start < 0 or stop < start or stop > rows:
        raise ValueError(f"invalid NEPQ row range [{start},{stop}) for {rows} rows")
    spec = tensor.spec
    base = nepq_base_spec(spec)
    k = tensor.neuron_len
    count = stop - start
    state = np.asarray(tensor.state, dtype=np.uint8).reshape(rows, ng)[start:stop]
    indices = np.asarray(tensor.indices, dtype=np.uint16).reshape(rows, nvec)[start:stop]
    anchors = np.asarray(tensor.neuron_scale, dtype=np.float32).reshape(rows)[start:stop]
    bank_ids = np.asarray(tensor.bank_ids, dtype=np.uint8).reshape(rows, -1)[start:stop]
    group_bank = np.repeat(
        bank_ids, spec.groups_per_supergroup, axis=1
    )[:, :ng]
    vector_bank = np.repeat(group_bank, spec.groupsize // spec.vector_size, axis=1)[
        :, :nvec
    ]
    vector_state = np.repeat(
        state, spec.groupsize // spec.vector_size, axis=1
    )[:, :nvec]
    tables = _decoded_tables(tensor) if decoded_tables is None else decoded_tables
    if base is NEPQ0_S or base is NEPQ0_L:
        scale_lut, first, second = tables
        composite = indices.astype(np.uint8)
        first_index = composite & 7
        second_index = composite >> 3
        first_code = first[vector_bank, vector_state, first_index]
        second_code = second[vector_bank, vector_state, second_index]
        code = np.concatenate((first_code, second_code), axis=-1).reshape(count, k)
        group_scale = anchors[:, None] * scale_lut[group_bank, state]
        scale = np.repeat(group_scale, spec.groupsize, axis=1)[:, :k]
        result = code.astype(np.float32) * scale
    elif base is NEPQ1_S:
        aux = np.asarray(tensor.aux, dtype=np.uint8).reshape(rows, ng)[start:stop]
        vector_aux = np.repeat(aux, spec.groupsize // spec.vector_size, axis=1)[
            :, :nvec
        ]
        code = tables[vector_bank, vector_aux, indices]
        delta = np.where(aux != 0, -0.15625, 0.15625).astype(np.float32)
        group_scale = anchors[:, None] * state.astype(np.float32)
        scale = np.repeat(group_scale, spec.groupsize, axis=1)[:, :k]
        bias = np.repeat(delta, spec.groupsize, axis=1)[:, :k]
        result = scale * (code.reshape(count, k).astype(np.float32) + bias)
    elif base is NEPQ1_L:
        aux = np.asarray(tensor.aux, dtype=np.uint8).reshape(rows, ng)[start:stop]
        code = tables[vector_bank, indices]
        delta = np.where(aux != 0, -0.125, 0.125).astype(np.float32)
        group_scale = anchors[:, None] * state.astype(np.float32)
        scale = np.repeat(group_scale, spec.groupsize, axis=1)[:, :k]
        bias = np.repeat(delta, spec.groupsize, axis=1)[:, :k]
        result = scale * (code.reshape(count, k).astype(np.float32) + bias)
    else:
        raise ValueError(f"unsupported NEPQ spec: {spec.label}")
    if spec.is_residual and count:
        _add_sparse_residual_rows(tensor, result, start, stop)
    return result


def _add_sparse_residual_rows(
    tensor: NepqTensor,
    result: np.ndarray,
    start: int,
    stop: int,
) -> None:
    spec = tensor.spec
    blocks_per_row = tensor.residual_blocks_per_row
    block_vectors = spec.residual_block_vectors
    position_bits = spec.residual_position_bits
    position_mask = (1 << position_bits) - 1
    dictionary = np.asarray(tensor.residual_codebook, dtype=np.float32)
    first = np.asarray(tensor.residual_first).reshape(-1, blocks_per_row)[start:stop]
    second_dense = None
    if spec.residual_second:
        mask = np.asarray(tensor.residual_second_mask, dtype=np.uint8).reshape(-1)
        records = np.asarray(tensor.residual_second_records, dtype=np.uint16)
        dense = np.full(mask.size, -1, dtype=np.int32)
        dense[np.flatnonzero(mask)] = records.astype(np.int32)
        second_dense = dense.reshape(-1, blocks_per_row)[start:stop]
    for local_row in range(stop - start):
        for block in range(blocks_per_row):
            record = int(first[local_row, block])
            position = record & position_mask
            dictionary_id = record >> position_bits
            vector = block * block_vectors + position
            offset = vector * spec.vector_size
            result[local_row, offset : offset + spec.vector_size] += dictionary[
                dictionary_id
            ]
            if second_dense is not None:
                record = int(second_dense[local_row, block])
                if record >= 0:
                    position = record & position_mask
                    dictionary_id = record >> position_bits
                    vector = block * block_vectors + position
                    offset = vector * spec.vector_size
                    result[local_row, offset : offset + spec.vector_size] += dictionary[
                        dictionary_id
                    ]


__all__ = [
    "NEPQ0_A",
    "NEPQ0_L",
    "NEPQ0_S",
    "NEPQ1_A",
    "NEPQ1_L",
    "NEPQ1_S",
    "NEPQ_SPECS",
    "NepqSpec",
    "NepqTensor",
    "dequantize_nepq",
    "dequantize_nepq_rows",
    "nepq_base_spec",
    "nepq_spec",
    "pack_nepq",
    "unpack_nepq",
    "validate_nepq",
]
