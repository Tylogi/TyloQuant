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

    @property
    def groupsize(self) -> int:
        return _GROUP_SIZE

    @property
    def vector_size(self) -> int:
        return _VECTOR_SIZE

    @property
    def groups_per_supergroup(self) -> int:
        return _GROUPS_PER_SUPERGROUP

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
NEPQ_SPECS = (NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L)
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
        return self.spec.payload_nbytes(
            self.n_experts,
            self.out_per_expert,
            self.neuron_len,
            bank_count=self.bank_count,
        )

    @property
    def payload_bpw(self) -> float:
        return 8.0 * self.payload_nbytes / int(np.prod(self.shape))


_MAGIC = b"NEP1"
_VERSION = 1
_FLAG_ROTATED = 1
_HEADER = struct.Struct("<4sBBBBIIIIIQ")


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
    if len(payload) != spec.table_bytes:
        raise ValueError(
            f"{spec.label} bank must contain {spec.table_bytes} bytes"
        )
    if spec is NEPQ0_S:
        _, _, _, consumed = unpack_npq0_s_tables(payload)
    elif spec is NEPQ0_L:
        _, _, _, consumed = unpack_npq0_l_tables(payload)
    elif spec is NEPQ1_S:
        unpack_nvq1_s_banked_codebook(payload)
        consumed = len(payload)
    elif spec is NEPQ1_L:
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
    return ng, nvec, nsuper, bank_count, n_experts * out_per_expert


def pack_nepq(tensor: NepqTensor) -> bytes:
    ng, nvec, nsuper, bank_count, rows = validate_nepq(tensor)
    spec = nepq_spec(tensor.spec)
    flags = _FLAG_ROTATED if tensor.rotation_block else 0
    n_experts, out_per_expert, neuron_len = (int(value) for value in tensor.shape)
    aux = np.empty(0, dtype=np.uint8) if tensor.aux is None else np.asarray(tensor.aux)
    return b"".join(
        [
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
    )


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
    )
    validate_nepq(tensor)
    return tensor


def _decoded_tables(tensor: NepqTensor) -> object:
    tables = [row.tobytes() for row in np.asarray(tensor.table_payloads, dtype=np.uint8)]
    if tensor.spec is NEPQ0_S:
        unpacked = [unpack_npq0_s_tables(table)[:3] for table in tables]
        return tuple(np.stack(items) for items in zip(*unpacked, strict=True))
    if tensor.spec is NEPQ0_L:
        unpacked = [unpack_npq0_l_tables(table)[:3] for table in tables]
        return tuple(np.stack(items) for items in zip(*unpacked, strict=True))
    if tensor.spec is NEPQ1_S:
        return np.stack([unpack_nvq1_s_banked_codebook(table) for table in tables])
    if tensor.spec is NEPQ1_L:
        return np.stack([unpack_ternary_codebook(table) for table in tables])
    raise ValueError(f"unsupported NEPQ spec: {tensor.spec.label}")


def dequantize_nepq(tensor: NepqTensor) -> np.ndarray:
    """Decode the stored (possibly rotated) expert matrices to float32."""

    ng, nvec, _, _, rows = validate_nepq(tensor)
    spec = tensor.spec
    k = tensor.neuron_len
    state = np.asarray(tensor.state, dtype=np.uint8).reshape(rows, ng)
    indices = np.asarray(tensor.indices, dtype=np.uint16).reshape(rows, nvec)
    anchors = np.asarray(tensor.neuron_scale, dtype=np.float32).reshape(rows)
    bank_ids = np.asarray(tensor.bank_ids, dtype=np.uint8).reshape(rows, -1)
    group_bank = np.repeat(
        bank_ids, spec.groups_per_supergroup, axis=1
    )[:, :ng]
    vector_bank = np.repeat(group_bank, spec.groupsize // spec.vector_size, axis=1)[
        :, :nvec
    ]
    vector_state = np.repeat(
        state, spec.groupsize // spec.vector_size, axis=1
    )[:, :nvec]
    tables = _decoded_tables(tensor)
    if spec is NEPQ0_S or spec is NEPQ0_L:
        scale_lut, first, second = tables
        composite = indices.astype(np.uint8)
        first_index = composite & 7
        second_index = composite >> 3
        first_code = first[vector_bank, vector_state, first_index]
        second_code = second[vector_bank, vector_state, second_index]
        code = np.concatenate((first_code, second_code), axis=-1).reshape(rows, k)
        group_scale = anchors[:, None] * scale_lut[group_bank, state]
        scale = np.repeat(group_scale, spec.groupsize, axis=1)[:, :k]
        result = code.astype(np.float32) * scale
    elif spec is NEPQ1_S:
        aux = np.asarray(tensor.aux, dtype=np.uint8).reshape(rows, ng)
        vector_aux = np.repeat(aux, spec.groupsize // spec.vector_size, axis=1)[
            :, :nvec
        ]
        code = tables[vector_bank, vector_aux, indices]
        delta = np.where(aux != 0, -0.15625, 0.15625).astype(np.float32)
        group_scale = anchors[:, None] * state.astype(np.float32)
        scale = np.repeat(group_scale, spec.groupsize, axis=1)[:, :k]
        bias = np.repeat(delta, spec.groupsize, axis=1)[:, :k]
        result = scale * (code.reshape(rows, k).astype(np.float32) + bias)
    elif spec is NEPQ1_L:
        aux = np.asarray(tensor.aux, dtype=np.uint8).reshape(rows, ng)
        code = tables[vector_bank, indices]
        delta = np.where(aux != 0, -0.125, 0.125).astype(np.float32)
        group_scale = anchors[:, None] * state.astype(np.float32)
        scale = np.repeat(group_scale, spec.groupsize, axis=1)[:, :k]
        bias = np.repeat(delta, spec.groupsize, axis=1)[:, :k]
        result = scale * (code.reshape(rows, k).astype(np.float32) + bias)
    else:
        raise ValueError(f"unsupported NEPQ spec: {spec.label}")
    return result.reshape(tensor.shape)


__all__ = [
    "NEPQ0_L",
    "NEPQ0_S",
    "NEPQ1_L",
    "NEPQ1_S",
    "NEPQ_SPECS",
    "NepqSpec",
    "NepqTensor",
    "dequantize_nepq",
    "nepq_spec",
    "pack_nepq",
    "unpack_nepq",
    "validate_nepq",
]
